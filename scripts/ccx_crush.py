"""Crush a part between bonded platens until it snaps: feacore's nonlinear ccx path with NLGEOM, J2 plasticity
from a calibrated material JSON and progressive failure by element erosion (analysis.options.failure).

    python scripts/ccx_crush.py MESH.msh AXIS(x|y|z) MATERIAL.json OUT_DIR [--max-crush 0.5] [--threads 8]
           [--coarse 0.005] [--near 0.002] [--fine 0.001] [--workdir DIR]

Bottom group fixed, top group moved along -AXIS (lateral 0) up to --max-crush x height; the run stops when the
part has separated (no fragment joins top and bottom) or the platen force is < 1 % of its peak. Increment sizes
(mm of platen travel): ccx increments <= --inc through yield and <= --inc-near around failure (every increment
is output); erosion blocks <= --block, <= --block-near near failure, >= --block-min. MATERIAL.json (MPa): E, nu, plastic_table_true_stress_vs_plastic_strain, failure_peeq;
the mesh is in metres, so stresses are converted to Pa.
OUT_DIR/<run-id>/ (never overwritten): result_NNNN.vtu per increment (point: displacement, von_mises, PEEQ; cell: PEEQ (integration-point
mean), failed, state 0 intact / 1 failed / 2 detached fragment), force_displacement.csv, summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from feacore.case import BC, Analysis, Case, Material
from feacore.solver import solve

AX = {"x": 0, "y": 1, "z": 2}


def material(path) -> tuple[Material, float, dict]:
    d = json.loads(Path(path).read_text())
    tab = [(float(e), float(s) * 1e6) for s, e in d["plastic_table_true_stress_vs_plastic_strain"]]
    plast = {"model": "j2_isotropic", "yield_stress": tab[0][1], "hardening": [list(p) for p in tab]}
    return Material(E=float(d["E"]) * 1e6, nu=float(d["nu"]), plasticity=plast), float(d["failure_peeq"]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"), ap.add_argument("axis"), ap.add_argument("material"), ap.add_argument("out")
    ap.add_argument("--max-crush", type=float, default=0.5)
    ap.add_argument("--block", type=float, default=0.25, help="max erosion block, mm of travel")
    ap.add_argument("--block-near", type=float, default=0.06, help="max block once failure is near, mm")
    ap.add_argument("--block-min", type=float, default=0.02, help="min block, mm")
    ap.add_argument("--inc", type=float, default=0.05, help="max ccx increment through yield, mm")
    ap.add_argument("--inc-near", type=float, default=0.02, help="max ccx increment around failure, mm")
    ap.add_argument("--run-id", default=None, help="output subdirectory (default run-YYYYmmdd-HHMM)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    k = AX[a.axis]
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    m = build_model(read_gmsh(a.mesh), 3, 2)
    H = float(m.X[m.nodes_of("top"), k].mean() - m.X[m.nodes_of("bottom"), k].mean())
    D = a.max_crush * H
    mm = lambda v: v * 1e-3 / D                     # noqa: E731  mm of travel -> ramp fraction
    mat, limit, mdoc = material(a.material)
    val = [0.0, 0.0, 0.0]
    val[k] = -D
    case = Case(mesh=str(Path(a.mesh).resolve()),
                analysis=Analysis("elasticity", "3d", order=2, options={
                    "nonlinear": True, "nlgeom": True, "load_factor_max": 1.0,
                    "increments": int(round(1.0 / mm(a.block))),
                    "failure": {"criterion": "peeq", "limit": limit, "aggregate": "mean",
                                "near_increments": int(round(1.0 / mm(a.block_near))),
                                "fine_increments": int(round(1.0 / mm(a.block_min))),
                                "max_increment": mm(a.inc), "max_increment_near": mm(a.inc_near),
                                "path_groups": ["top", "bottom"], "stop_force_ratio": 0.01}}),
                materials={"resin": mat},
                bcs=[BC("fixed", group="bottom"), BC("displacement", group="top", components=[0, 1, 2], value=val)])
    run_id = a.run_id or time.strftime("run-%Y%m%d-%H%M")
    out = Path(a.out) / run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: never overwrite a previous run")
    out.mkdir(parents=True, exist_ok=True)
    wd = Path(a.workdir) / run_id if a.workdir else None
    case.save(out / "case.json")
    f = open(out / "force_displacement.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "platen_travel_mm", "force_N", "n_failed", "n_detached", "max_peeq", "iterations", "block"])
    rows = []

    def on_inc(rec):
        row = [rec["step"], rec["lambda"] * D * 1e3, -rec["reaction"][k], rec["n_failed"], rec["n_detached"],
               rec["max_peeq_alive"], rec["iterations"], rec["block"]]
        rows.append(row)
        w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        f.flush()

    t0 = time.time()
    r = solve(case, backend="ccx", threads=a.threads, workdir=wd, series_dir=str(out), on_increment=on_inc)
    wall = time.time() - t0
    f.close()
    F = np.array([x[2] for x in rows])
    ip = int(np.argmax(F))
    cen = m.X[m.conn[:, :4]].mean(axis=1)
    order = np.argsort(np.where(r.removed_at >= 0, r.removed_at, np.iinfo(np.int64).max))
    first = [i for i in order if r.state[i] == 1][:20]
    failed = r.state == 1
    summ = {"mesh": a.mesh, "axis": a.axis, "height_m": H, "max_crush_m": D, "material": mdoc, "failure_peeq": limit,
            "stop_reason": r.stop_reason, "separated": r.separated, "converged": r.converged, "wall_s": wall,
            "n_increments": len(rows), "n_blocks": r.history[-1]["block"] + 1 if r.history else 0,
            "n_cells": m.n_elems, "n_failed": int(failed.sum()), "n_detached": int((r.state == 2).sum()),
            "peak_force_N": float(F[ip]), "travel_at_peak_mm": rows[ip][1],
            "travel_at_end_mm": rows[-1][1], "force_at_end_N": float(F[-1]),
            "first_failure": {"travel_mm": next((x[1] for x in rows if x[3] > 0), None),
                              "cells_centroid_m": cen[first].mean(axis=0).tolist() if first else None},
            "failed_centroid_m": cen[failed].mean(axis=0).tolist() if failed.any() else None,
            "failed_bbox_m": [cen[failed].min(axis=0).tolist(), cen[failed].max(axis=0).tolist()] if failed.any()
            else None}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k2: v for k2, v in summ.items() if k2 != "material"}, indent=1, default=float))


if __name__ == "__main__":
    main()
