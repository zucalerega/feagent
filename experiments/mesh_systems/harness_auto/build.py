"""harness_auto: the mesh harness's zero-config start (`feagent automesh`'s default spec, `feagent.mesher.auto_spec`)
plus the two contract groups selected by geometric query (plane normal + a thin box at the platen position), meshed
and gated by `feagent check`. The baseline system (the meshes the first crush runs used).

    python experiments/mesh_systems/harness_auto/build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear

linear:    the automesh spec as generated (global = diag / 40, min = global / 20, curvature 12, 2 thin-wall layers,
           order 2, netgen optimisation).
nonlinear: the coarse spec of the earlier crush runs (fewer, larger cells so a crush to separation takes hours):
           003 global 8 mm / min 1 mm / curvature 4, order 2; 065 global 6 mm / min 0.5 mm / curvature 8, order 1,
           then midside nodes at straight edge midpoints (gmsh SecondOrderLinear) -> C3D10 with straight edges;
           1 thin-wall layer. Both <= 40k nodes.
Writes OUT_DIR/{mesh.msh, mesh.inp, report.json, mesh.png} (contract) and OUT_DIR/harness/ (spec + feagent output).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# contract table: axis, bottom / top plane position along it [m], face counts
PARTS = {"abc_00000003": {"axis": "z", "bottom": 0.0, "top": 0.06125, "n_bottom": 1, "n_top": 1},
         "abc_00000065": {"axis": "y", "bottom": -0.04396, "top": 0.03384, "n_bottom": 1, "n_top": 2}}
BOX_TOL = 5e-4             # half-thickness of the selection box along the axis [m]
NONLINEAR = {"abc_00000003": ({"global": 0.008, "min": 0.001, "curvature": 4, "thin_wall_layers": 1}, 2),
             "abc_00000065": ({"global": 0.006, "min": 0.0005, "curvature": 8, "thin_wall_layers": 1}, 1)}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
           "__GLX_VENDOR_LIBRARY_NAME": "mesa"}


def group_query(axis: str, pos: float, sign: int, expect: int) -> dict:
    k = "xyz".index(axis)
    normal = [0, 0, 0]
    normal[k] = sign
    lo, hi = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]       # parts are < 0.2 m: the box is a slab across the part
    lo[k], hi[k] = pos - BOX_TOL, pos + BOX_TOL
    return {"select": {"kind": "plane", "normal": normal, "box": [lo, hi]}, "expect": expect}


def write_inp(msh: Path, inp: Path, order: int) -> dict:
    """CalculiX mesh: *NODE, the volume cells only (C3D10 / C3D4, ELSET domain) and *NSET bottom / top / domain.
    No surface elements (a CalculiX 3D deck rejects CPS3 / CPS6 in the model)."""
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    m = build_model(read_gmsh(msh), 3, order)
    et = {1: "C3D4", 2: "C3D10"}[order]

    def ids(a):
        a = list(map(int, a))
        return "\n".join(", ".join(str(x + 1) for x in a[i:i + 16]) for i in range(0, len(a), 16))

    lines = ["*NODE"] + [f"{i + 1}, {x:.17g}, {y:.17g}, {z:.17g}" for i, (x, y, z) in enumerate(m.X)]
    lines += [f"*ELEMENT, TYPE={et}, ELSET=domain"]
    lines += [f"{e + 1}, " + ", ".join(str(int(n) + 1) for n in c) for e, c in enumerate(m.conn)]
    for g in ("bottom", "top"):
        lines += [f"*NSET, NSET={g}", ids(m.nodes_of(g))]
    lines += ["*NSET, NSET=domain", ids(range(m.n_nodes))]
    inp.write_text("\n".join(lines) + "\n")
    return {"n_nodes": int(m.n_nodes), "n_elements": int(m.n_elems), "element": et,
            "nset_sizes": {g: int(len(m.nodes_of(g))) for g in ("bottom", "top")}}


def promote_straight(src: Path, dst: Path) -> None:
    """Linear tets -> 10-node tets with midside nodes at the straight edge midpoints (gmsh SecondOrderLinear)."""
    import gmsh
    gmsh.initialize(["-nt", "1"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(src))
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(dst))
    finally:
        gmsh.finalize()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step"), ap.add_argument("axis", choices="xyz"), ap.add_argument("out")
    ap.add_argument("--purpose", choices=["linear", "nonlinear"], required=True)
    ap.add_argument("--part", default=None, help="part id (default: from the STEP path, <part>_bc0/public/...)")
    a = ap.parse_args(argv)
    t0 = time.time()
    os.environ.update(CPU_ENV)
    step = Path(a.step).resolve()
    part = a.part or step.parents[1].name.removesuffix("_bc0")
    P = PARTS[part]
    if a.axis != P["axis"]:
        raise SystemExit(f"{part}: contract axis is {P['axis']}, got {a.axis}")
    out = Path(a.out).resolve()
    work = out / "harness"
    if work.exists():
        shutil.rmtree(work)           # this system's own scratch output, rebuilt from the STEP every time
    work.mkdir(parents=True)

    from feagent.mesher import auto_spec
    spec = auto_spec(step, "fea", work / "mesh.yaml")        # what `feagent automesh` writes and meshes
    spec["groups"] = {"bottom": group_query(a.axis, P["bottom"], -1, P["n_bottom"]),
                      "top": group_query(a.axis, P["top"], +1, P["n_top"])}
    spec["export"] = ["msh"]                                  # mesh.inp is written below (volume cells only)
    order = 2
    if a.purpose == "nonlinear":
        spec["size"], order0 = NONLINEAR[part]
        spec["element"]["order"] = order0
    spec["geometry"] = os.path.relpath(step, work)
    (work / "mesh.yaml").write_text("# harness_auto: feagent auto_spec + contract groups by query\n"
                                    + yaml.safe_dump(spec, sort_keys=False))
    feagent = [sys.executable, "-m", "feagent.cli"]
    p = subprocess.run([*feagent, "check", str(work / "mesh.yaml"), "--json", "--no-render"], capture_output=True, text=True,
                       env={**os.environ, **CPU_ENV})
    hrep_path = work / "out" / "report.json"
    if not hrep_path.exists():
        raise SystemExit(f"feagent check failed ({p.returncode}): {p.stderr[-2000:]}")
    hrep = json.loads(hrep_path.read_text())
    faces = {g: len(r["tags"]) for g, r in hrep["mesh"]["groups"].items() if r["dim"] == 2}
    if faces != {"bottom": P["n_bottom"], "top": P["n_top"]}:
        raise SystemExit(f"{part}: face counts {faces} differ from the contract")
    msh = out / "mesh.msh"
    if a.purpose == "nonlinear" and hrep["mesh"]["order"] == 1:
        promote_straight(work / "out" / "mesh.msh", msh)
    else:
        shutil.copyfile(work / "out" / "mesh.msh", msh)
    inp = write_inp(msh, out / "mesh.inp", order)

    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    from feagent.types import MeshResult
    from feagent.geometry import analyze
    mr = MeshResult.from_dict({**hrep["mesh"], "mesh_path": str(msh), "n_nodes": inp["n_nodes"], "order": order})
    mr.geometry = analyze(step, ref_size=spec["size"]["global"])   # as the mesher does (CAD volume, thin walls)
    q = assess_mesh(msh, mr, QualitySpec(), "fea", "netgen")
    r = subprocess.run([*feagent, "render", str(msh), "--view", "quality", "-o", str(out / "mesh.png")],
                       capture_output=True, text=True, env={**os.environ, **CPU_ENV})
    d = q.to_dict()
    d["gates"] = [g if isinstance(g, dict) else g.to_dict() for g in d["gates"]]
    d["mesh_gates"] = hrep["mesh"]["gates"]
    d.update(system="harness_auto", part=part, purpose=a.purpose, n_nodes=inp["n_nodes"],
             n_elements=inp["n_elements"], order=order, dofs=3 * inp["n_nodes"], build_s=round(time.time() - t0, 2),
             element=inp["element"], face_counts=faces, nset_sizes=inp["nset_sizes"],
             harness_status=hrep["status"], harness_warnings=hrep["mesh"]["warnings"],
             straight_midside=bool(a.purpose == "nonlinear" and hrep["mesh"]["order"] == 1),
             knobs={"size": spec["size"], "element": spec["element"], "box_tol_m": BOX_TOL},
             render_ok=r.returncode == 0 and (out / "mesh.png").exists())
    (out / "report.json").write_text(json.dumps(d, indent=1, default=float))
    worst = {g["id"]: g["status"] for g in d["gates"]}
    print(json.dumps({k: d[k] for k in ("part", "purpose", "n_nodes", "n_elements", "dofs", "build_s",
                                        "face_counts")} | {"gates": worst}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
