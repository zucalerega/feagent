#!/usr/bin/env python
"""harness_neck: harness_expert plus local refinement at the necks of a ring-shaped part, found by geometry.

usage: build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear [--neck-size M] [--neck-dist-factor F]

The harness_expert spec (digest-driven groups, thin-wall layers, refinement at the platens / fillets /
holes; experiments/mesh_systems/harness_expert/build.py, imported unchanged) plus one or two refinements
that the expert spec does not have, both found from the `feagent` geometry digest:

- the bore: the largest concave cylinder whose axis is normal to the load axis;
- the ring around it: the convex cylinders coaxial with the bore and larger than it. The wall between them
  (thickness t = r_outer - r_bore) is the part's narrowest load path in bending (for abc_00000065 the two
  arms' necks, 8.1 mm, from about 33 degrees below to 25 degrees above the bore's centre), so the mesh is
  refined there: size `--neck-size` at the ring's outer faces (where bending puts the outer fibre in
  tension) and at the bore, growing to the global size over `--neck-dist-factor` x t, so the whole
  wall is fine.

Writes the contract outputs (experiments/mesh_systems/CONTRACT.md) to OUT_DIR: mesh.msh, mesh.inp, report.json
(gates + contract block + the necks found), mesh.png (harness quality render) and mesh.yaml.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("harness_expert_build", HERE.parent / "harness_expert" / "build.py")
expert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(expert)

AX = {"x": 0, "y": 1, "z": 2}
# neck refinement per purpose: element size at the ring (m) and reach in wall thicknesses
NECK = {"nonlinear": {"size": 0.0016, "dist_factor": 1.2}, "linear": {"size": 0.0010, "dist_factor": 1.2}}


def _line_dist(p, a, q):
    """Distance of point q from the line through p with unit direction a."""
    d = np.subtract(q, p)
    return float(np.linalg.norm(d - np.dot(d, a) * a))


def find_necks(step: Path, axis: str, ref_size: float) -> dict | None:
    """The bore, its coaxial outer ring faces and the wall thickness, from the harness geometry digest."""
    from feagent.geometry import analyze

    model = analyze(step, ref_size=ref_size)
    load = np.eye(3)[AX[axis]]
    cyl = [f for f in model.faces if f.kind == "cylinder" and f.axis is not None and f.radius]
    bores = [f for f in cyl if f.concave and abs(np.dot(f.axis, load)) < 0.05]
    if not bores:
        return None
    bore = max(bores, key=lambda f: f.radius)
    a = np.asarray(bore.axis, float) / np.linalg.norm(bore.axis)
    tol = 1e-3 * bore.radius
    ring = [f for f in cyl if f.concave is False and f.radius > bore.radius
            and abs(abs(np.dot(f.axis, a)) - 1) < 1e-3 and _line_dist(bore.axis_point, a, f.axis_point) < tol]
    if not ring:
        return None
    r_out = min(f.radius for f in ring)
    ring = [f for f in ring if abs(f.radius - r_out) < tol]
    t = r_out - bore.radius
    necks = []
    for f in ring:                       # mid-angle point of each ring face on the outer surface
        c = np.asarray(f.center, float)
        p = np.asarray(bore.axis_point, float)
        rvec = c - p - np.dot(c - p, a) * a
        mid = p + np.dot(c - p, a) * a + rvec / np.linalg.norm(rvec) * r_out
        necks.append({"face": f.tag, "outer_mid_point_m": mid.round(6).tolist(),
                      "load_axis_span_m": [f.bbox[0][AX[axis]], f.bbox[1][AX[axis]]]})
    return {"bore_radius_m": bore.radius, "bore_axis": a.tolist(), "bore_axis_point_m": list(bore.axis_point),
            "ring_outer_radius_m": r_out, "wall_thickness_m": t, "n_ring_faces": len(ring), "necks": necks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step"), ap.add_argument("axis"), ap.add_argument("out")
    ap.add_argument("--purpose", choices=["linear", "nonlinear"], required=True)
    ap.add_argument("--neck-size", type=float, default=None, help="element size at the necks, m")
    ap.add_argument("--neck-dist-factor", type=float, default=None, help="refinement reach / wall thickness")
    ap.add_argument("--global", dest="global_size", type=float, default=None,
                    help="override harness_expert's global size [m] (node-count matching)")
    a = ap.parse_args()
    step, out = Path(a.step), Path(a.out)
    part = step.parent.parent.name.replace("_bc0", "")
    out.mkdir(parents=True, exist_ok=True)
    spec = expert.make_spec(step, part, a.purpose, out, a.global_size)
    knobs = NECK[a.purpose]
    size = a.neck_size or knobs["size"]
    fac = a.neck_dist_factor or knobs["dist_factor"]
    t0 = time.time()
    neck = find_necks(step, a.axis, spec["size"]["global"])
    if neck is None:
        print("no bore with a coaxial outer ring found: harness_expert spec unchanged", flush=True)
    else:
        t = neck["wall_thickness_m"]
        ring_q = {"kind": "cylinder", "radius": neck["ring_outer_radius_m"], "concave": False,
                  "axis": neck["bore_axis"]}
        bore_q = {"kind": "cylinder", "radius": neck["bore_radius_m"], "concave": True, "axis": neck["bore_axis"]}
        spec["refine"] += [{"select": ring_q, "size": size, "distance": fac * t},
                           {"select": bore_q, "size": size, "distance": 0.5 * t}]
        neck.update({"size_m": size, "distance_m": fac * t})
        print(f"necks: wall {t * 1e3:.2f} mm between bore r {neck['bore_radius_m'] * 1e3:.2f} mm and "
              f"{neck['n_ring_faces']} ring faces r {neck['ring_outer_radius_m'] * 1e3:.2f} mm; size {size * 1e3:.2f} mm",
              flush=True)
    vec = [0.0, 0.0, 0.0]
    vec[AX[a.axis]] = -100.0                       # compressive on the top platen (gates only)
    spec["bcs"]["fea"]["loads"][0]["force"] = vec
    spec["name"] = f"{part}_{a.purpose}_neck"
    (out / "mesh.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    from feagent.check import run_check
    rep = run_check(out / "mesh.yaml")
    build_s = time.time() - t0
    expert.strip_surface_elements(out / "mesh.inp")
    rj = json.loads((out / "report.json").read_text())
    m = rj.get("mesh", {})
    n_nodes = m.get("n_nodes")
    order = spec["element"]["order"]
    rj["contract"] = {"system": "harness_neck", "part": part, "purpose": a.purpose,
                      "n_nodes": n_nodes, "n_elements": m.get("n_elements"), "order": order,
                      "dofs": 3 * n_nodes if n_nodes else None, "build_s": round(build_s, 1), "necks": neck}
    (out / "report.json").write_text(json.dumps(rj, indent=1))
    (out / "mesh.png").write_bytes((out / "quality.png").read_bytes())
    print(json.dumps({k: v for k, v in rj["contract"].items() if k != "necks"}), rep.status)
    for g in rep.gates:
        print(f"  {g.id:18s} {g.status:5s} {g.message}")
    return 0 if rep.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
