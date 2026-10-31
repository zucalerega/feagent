"""Turn the mesh harness CFD output (spec ``analysis: cfd``, package ``feagent``) into a feacore flow case.

The mesher writes ``<out_dir>/mesh.msh`` (a named dim-2 physical group per spec group, the
``default_group`` with every remaining boundary face, ``walls`` by default, and a dim-3 group
``domain``), ``manifest.json`` and ``report.json``, whose ``bcs`` section declares

    bcs: {cfd: {inlet: [group, ...], outlet: [group, ...], velocity: 1.0}}

Mapping (``feacore.case`` flow BCs, solved by ``backend="openfoam"``):

* every ``inlet`` group: ``velocity_inlet`` with ``velocity`` as the speed along the inward normal;
* every ``outlet`` group: ``pressure_outlet`` with static pressure 0;
* every other surface group of the mesh (the default group ``walls`` and any named group that is
  neither inlet nor outlet): no-slip ``wall``.

Units follow the harness: lengths in mm, so the consistent set is mm, s, t (tonne), N, MPa:
``velocity`` in mm/s, density in t/mm^3, dynamic viscosity in MPa s, pressure in MPa, forces in N
and flow rates in mm^3/s. The default fluid is water at 20 C in those units (rho = 1e-9 t/mm^3,
mu = 1.0e-9 MPa s, i.e. 1000 kg/m^3 and 1 mPa s, kinematic viscosity 1 mm^2/s); override with
``rho`` / ``mu`` (``--rho`` / ``--mu``).

Default QoIs: the pressure drop inlet -> outlet (mean static pressure over each inlet; the outlets
are at p = 0), the flow rate through every inlet and outlet (positive out of the domain; they sum
to ~0), the fluid force on each wall group (x, y, z) and the peak speed.

    python -m feacore.adapters.mesh_harness_cfd path/to/out_dir --out case.json
    python -m feacore.adapters.mesh_harness_cfd path/to/mesh.yaml --out case.json [--rho R --mu M]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI

WATER_MM = {"rho": 1.0e-9, "mu": 1.0e-9}     # t/mm^3, MPa s (water, 20 C) in mm-t-s-N-MPa units
DOMAIN_GROUP = "domain"


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def surface_groups(mesh: str | Path) -> list[str]:
    """Names of the dim-2 physical groups of a gmsh file (2.2 or 4.1, ASCII or binary: the
    ``$PhysicalNames`` section is text in both)."""
    names = []
    with open(mesh, "rb") as f:
        inside = False
        for raw in f:
            line = raw.decode("latin-1").strip()
            if line == "$PhysicalNames":
                inside = True
                next(f)
                continue
            if line == "$EndPhysicalNames" or (not inside and line.startswith("$Nodes")):
                break
            if inside:
                m = re.match(r'(\d+)\s+\d+\s+"(.*)"', line)
                if m and int(m.group(1)) == 2:
                    names.append(m.group(2))
    return names


def default_qois(inlets: list[str], outlets: list[str], walls: list[str]) -> list[QoI]:
    q = []
    if len(inlets) == 1:
        q.append(QoI("pressure_drop", "mean", field="p", group=inlets[0]))
    else:
        q += [QoI(f"mean_p_{g}", "mean", field="p", group=g) for g in inlets]
    q += [QoI(f"flow_rate_{g}", "flow_rate", group=g) for g in inlets + outlets]
    for g in walls:
        q += [QoI(f"force_{g}_{ax}", "force", group=g, component=c) for c, ax in enumerate("xyz")]
    q.append(QoI("max_u_mag", "max", field="u_mag"))
    return q


def build_case(mesh: str | Path, cfd: dict, walls: list[str] | None = None, rho: float | None = None,
               mu: float | None = None, order: int = 2, options: dict | None = None,
               meta: dict | None = None) -> Case:
    """Flow case from a ``bcs.cfd`` mapping and the mesh (wall groups default to every surface
    group of the mesh that is neither an inlet nor an outlet)."""
    inlets = list(cfd.get("inlet", []))
    outlets = list(cfd.get("outlet", []))
    if not inlets or not outlets:
        raise ValueError("bcs.cfd needs at least one inlet and one outlet group")
    velocity = float(cfd.get("velocity", 1.0))
    if walls is None:
        walls = [g for g in surface_groups(mesh) if g not in inlets + outlets and g != DOMAIN_GROUP]
    bcs = ([BC("velocity_inlet", group=g, value=velocity) for g in inlets]
           + [BC("pressure_outlet", group=g, value=0.0) for g in outlets]
           + [BC("wall", group=g) for g in walls])
    return Case(
        mesh=str(Path(mesh).resolve()),
        analysis=Analysis(physics="flow", formulation="3d", order=int(order), options=dict(options or {})),
        materials={"fluid": Material(rho=float(rho if rho is not None else WATER_MM["rho"]),
                                     mu=float(mu if mu is not None else WATER_MM["mu"]))},
        bcs=bcs,
        qois=default_qois(inlets, outlets, walls),
        meta={"source": "mesh-harness", "units": "mm, s, t, N, MPa", **(meta or {})},
    )


def case_from_mesh_harness_cfd(out_dir: str | Path, cfd: dict | None = None, mesh: str | Path | None = None,
                               **kw) -> Case:
    """Build a 3D flow case from a mesh-harness output directory (``report.json`` bcs.cfd)."""
    out_dir = Path(out_dir)
    manifest = _read_json(out_dir / "manifest.json") or {}
    report = _read_json(out_dir / "report.json") or {}
    if cfd is None:
        decl = report.get("bcs") or report.get("solve") or {}
        cfd = decl.get("cfd") if isinstance(decl, dict) else None
    if not cfd:
        raise ValueError(f"no CFD boundary-condition declaration (bcs.cfd) found in {out_dir}")
    mesh_path = Path(mesh or manifest.get("mesh_path") or out_dir / "mesh.msh")
    if not mesh_path.is_absolute() and not mesh_path.is_file():
        mesh_path = out_dir / mesh_path
    meta = {"out_dir": str(out_dir.resolve()), "spec_path": manifest.get("spec_path")}
    return build_case(mesh_path, cfd, meta=meta, **kw)


def case_from_spec(spec_path: str | Path, mesh: str | Path | None = None, **kw) -> Case:
    """Case from a mesh-harness YAML spec (``bcs.cfd``) and the mesh it writes to ``out/mesh.msh``."""
    from dataclasses import asdict

    from feagent.spec import load_spec

    spec = load_spec(spec_path)
    if getattr(spec, "bcs_cfd", None) is None:
        raise ValueError(f"{spec_path}: no bcs.cfd section")
    out = spec.out_dir()
    mesh = Path(mesh or out / "mesh.msh")
    cfd = asdict(spec.bcs_cfd)
    if kw.get("walls") is None:
        named = [g for g in spec.groups if g not in cfd["inlet"] + cfd["outlet"]]
        dflt = [spec.default_group] if spec.default_group else []
        present = set(surface_groups(mesh)) if mesh.is_file() else set(named + dflt)
        kw["walls"] = [g for g in named + dflt if g in present]
    case = build_case(mesh, cfd, meta={"out_dir": str(out), "spec_path": str(Path(spec_path).resolve())}, **kw)
    return case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", help="harness output directory, or its YAML spec")
    ap.add_argument("--out", default=None, help="case JSON path (default: <out_dir>/case_cfd.json)")
    ap.add_argument("--rho", type=float, default=None, help="density (default water, t/mm^3)")
    ap.add_argument("--mu", type=float, default=None, help="dynamic viscosity (default water, MPa s)")
    ap.add_argument("--order", type=int, default=2, help="convection scheme order (1 upwind, 2 linearUpwind)")
    args = ap.parse_args(argv)
    src = Path(args.out_dir)
    kw = {"rho": args.rho, "mu": args.mu, "order": args.order}
    if src.suffix in (".yaml", ".yml"):
        case = case_from_spec(src, **kw).check()
        default = Path(case.mesh).parent / "case_cfd.json"
    else:
        case = case_from_mesh_harness_cfd(src, **kw).check()
        default = src / "case_cfd.json"
    print(case.save(args.out or default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
