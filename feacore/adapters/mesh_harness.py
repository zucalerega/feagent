"""Turn the mesh harness output (branch ``mesh-harness``, package ``feagent``) into a feacore case.

The mesher writes ``<out_dir>/mesh.msh`` (MSH 4.1, a named dim-2 physical group per spec group plus
a dim-3 group ``domain``), ``manifest.json`` (its MeshResult, incl. ``mesh_path`` and ``order``) and
``report.json`` whose ``bcs`` (formerly ``solve``) section declares the FEA boundary conditions:

    bcs: {fea: {fixed: [group, ...], loads: [{group, force: [fx, fy, fz]}, ...], E: 210000.0, nu: 0.3}}

``force`` is a total force spread uniformly over the group's area, which is exactly feacore's
``force`` BC. Units are whatever the part uses (the harness parts are mm / N / MPa).

Optional richer keys (feacore BC types; the harness spec can add them when it needs them):

    symmetry: [group, ...]                                  # zero normal displacement (axis-aligned planes)
    pressure: [{group, value}]                              # traction = -value * outward normal
    traction: [{group, value: [tx, ty, tz]}]                # force per area
    displacement: [{group, components: [0, 2], value: 0.0, where: "abs(z) < 1e-6"}]  # where optional

``case_from_spec`` reads the harness YAML spec directly (needs the ``feagent`` package):

    python -m feacore.adapters.mesh_harness path/to/mesh.yaml --out case.json

    python -m feacore.adapters.mesh_harness path/to/out_dir --out case.json [--order 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def default_qois(fixed: list[str], loads: list[dict]) -> list[QoI]:
    """Quantities every harness case reports: two that converge under refinement (strain energy,
    mean displacement of each loaded face), equilibrium checks (reactions) and the peak stress,
    which diverges at sharp re-entrant corners and point-like supports."""
    q = [QoI("strain_energy", "strain_energy"),
         QoI("max_von_mises", "max", field="von_mises"),
         QoI("max_displacement", "max", field="u_mag")]
    for ld in loads:
        q.append(QoI(f"mean_u_mag_{ld['group']}", "mean", field="u_mag", group=ld["group"]))
    for g in fixed:
        for c, ax in enumerate("xyz"):
            q.append(QoI(f"reaction_{g}_{ax}", "reaction", group=g, component=c))
    return q


def case_from_mesh_harness(out_dir: str | Path, order: int | None = None,
                           fea: dict | None = None, mesh: str | Path | None = None) -> Case:
    """Build a 3D elasticity case from a mesh-harness output directory.

    ``fea`` overrides the BC declaration (the ``bcs.fea`` / ``solve.fea`` mapping); ``mesh``
    overrides the mesh file; ``order`` overrides the FE order (default: the mesh order)."""
    out_dir = Path(out_dir)
    manifest = _read_json(out_dir / "manifest.json") or {}
    report = _read_json(out_dir / "report.json") or {}
    if fea is None:
        decl = report.get("bcs") or report.get("solve") or {}
        fea = decl.get("fea") if isinstance(decl, dict) and "fea" in decl else decl
    if not fea or not any(k in fea for k in ("fixed", "symmetry", "displacement")):
        raise ValueError(f"no FEA boundary-condition declaration (bcs.fea) found in {out_dir}")
    mesh_path = Path(mesh or manifest.get("mesh_path") or out_dir / "mesh.msh")
    if not mesh_path.is_absolute() and not mesh_path.is_file():
        mesh_path = out_dir / mesh_path
    order = int(order or manifest.get("order") or 2)
    fixed = list(fea.get("fixed", []))
    loads = [{"group": ld["group"], "force": [float(f) for f in ld["force"]]} for ld in fea.get("loads", [])]
    extra = [BC("symmetry", group=g) for g in fea.get("symmetry", [])]
    extra += [BC("pressure", group=d["group"], value=d["value"]) for d in fea.get("pressure", [])]
    extra += [BC("traction", group=d["group"], value=list(d["value"])) for d in fea.get("traction", [])]
    extra += [BC("displacement", group=d["group"], components=list(d["components"]),
                 value=d.get("value", 0.0), where=d.get("where")) for d in fea.get("displacement", [])]
    return Case(
        mesh=str(mesh_path.resolve()),
        analysis=Analysis(physics="elasticity", formulation="3d", order=order),
        materials={"material": Material(E=float(fea.get("E", 210000.0)), nu=float(fea.get("nu", 0.3)))},
        bcs=[BC("fixed", group=g) for g in fixed]
        + [BC("force", group=ld["group"], value=ld["force"]) for ld in loads] + extra,
        qois=default_qois(fixed, loads),
        meta={"source": "mesh-harness", "out_dir": str(out_dir.resolve()),
              "spec_path": manifest.get("spec_path")},
    )


def case_from_spec(spec_path: str | Path, order: int | None = None, mesh: str | Path | None = None) -> Case:
    """Case from a mesh-harness YAML spec (``solve.fea``) and the mesh it writes to ``out/mesh.msh``."""
    from dataclasses import asdict

    from feagent.spec import load_spec

    spec = load_spec(spec_path)
    if spec.fea is None:
        raise ValueError(f"{spec_path}: no solve.fea section")
    out = spec.out_dir()
    case = case_from_mesh_harness(out, order=order or spec.order, fea=asdict(spec.fea),
                                  mesh=mesh or out / "mesh.msh")
    case.meta["spec_path"] = str(Path(spec_path).resolve())
    return case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", help="harness output directory, or its YAML spec")
    ap.add_argument("--out", default=None, help="case JSON path (default: <out_dir>/case.json)")
    ap.add_argument("--order", type=int, default=None)
    args = ap.parse_args(argv)
    src = Path(args.out_dir)
    if src.suffix in (".yaml", ".yml"):
        case = case_from_spec(src, order=args.order).check()
        default = Path(case.mesh).parent / "case.json"
    else:
        case = case_from_mesh_harness(src, order=args.order).check()
        default = src / "case.json"
    path = case.save(args.out or default)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
