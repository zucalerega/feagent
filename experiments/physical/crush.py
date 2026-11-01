"""Crush one contract mesh between bonded platens until it snaps (nonlinear CalculiX, NLGEOM, J2 plasticity from
the dogbone, element erosion at PEEQ >= failure_peeq), with the user-required step policy. Built on
scripts/ccx_crush.py (material) and feacore/solver/_ccx_failure.py (erosion driver).

    python experiments/physical/crush.py --system harness_auto --part abc_00000003 --mesh-dir DIR
        [--out DIR] [--threads 8] [--run-id ID] [--resume RUN_DIR [--restart-file B.rout]]
        [--platen bonded|sliding|sliding_top] [--failure peeq|peeq_tension|principal] [--compression-factor 5]

--mesh-dir: a contract mesh dir (experiments/mesh_systems/CONTRACT.md): mesh.msh (+ mesh.inp) with groups bottom,
top (dim 2) and domain (dim 3), metres. Default $MESHSYS_OUT/<system>/<part>/nonlinear. The solver reads
mesh.msh (same mesh as mesh.inp). Linear tets (C3D4) run as C3D4, quadratic ones as C3D10 (straight interior edges).
--out: default runs/nonlinear/<system>/<part>; every run writes a new <out>/<run-id>/ (never overwritten).

Loading: bottom fixed; top moved along -axis (lateral 0), axis per part from the contract.
--platen (default bonded): bonded = every bottom / top node fully constrained (top: lateral 0);
  sliding_top = bottom bonded, top nodes constrained only along the axis (frictionless moving platen, the tips
  may slide sideways); sliding = both platens frictionless (only the axis component on bottom and top nodes)
  plus the minimum against rigid-body motion, both on bottom nodes: the one nearest the bottom face's
  centroid holds both lateral components, the bottom node farthest from it along the first lateral axis (in
  the band within 10 % of the face's extent of its mid-plane in the second) holds the second lateral component (no spin about the axis). Their
  coordinates are in run.json ("rigid_body_nodes").
--failure (default peeq): erosion criterion (feacore/solver/_ccx_failure.py): peeq = PEEQ >= failure_peeq in
  any stress state; peeq_tension = PEEQ >= failure_peeq where the cell is in tension (triaxiality > 0), PEEQ
  >= --compression-factor x failure_peeq elsewhere; principal = cell max principal stress >= the dogbone UTS
  (or PEEQ >= --compression-factor x failure_peeq). Alpha Fast's compressive strength is not on the TDS: the
  factor (default 5, PEEQ 0.215) is an assumption that a brittle resin crushes far later than it breaks in
  tension.

Step policy (platen travel per output increment; every converged ccx increment is written):
  * <= 0.05 mm through yield (until a cell reaches half the failure PEEQ);
  * <= 0.02 mm from then on, i.e. from before the first failed cell, until the force at a block end is below
    60 % of its peak (crack established);
  * <= 0.05 mm after that (tail).
  Erosion blocks (one ccx restart step each, failed cells are removed between blocks): <= 0.25 mm, <= 0.06 mm
  near failure, >= 0.02 mm, <= 0.10 mm in the tail; blocks that only release deferred failed cells advance
  0.002 mm. Release (--release adaptive, default): the failed cells over the limit are removed highest PEEQ first,
  up to a cap that starts at 64, doubles (max 64) after every converged block and halves on non-convergence down to
  1; --release fixed: at most 8 per block (the runs before 2026-10-31 21:10 ET).
Stop: full separation (no face-connected, supported chain of intact cells joins bottom and top), or force
< 1 % of peak over 3 consecutive increments after the first failure, or 30 % of the height of travel.

Output per run dir: result_NNNN.vtu (point: displacement [m], von_mises [Pa], PEEQ; cell: PEEQ (mean over the
cell's integration points), failed, state 0 intact / 1 failed / 2 detached fragment), force_displacement.csv,
summary.json, progress.log, case.json, checkpoint.json/.npz + ccx_work/ (restart file of the last block).

Resume: --resume RUN_DIR continues RUN_DIR into a new run dir from its checkpoint (same case, steps continue;
RUN_DIR is only read). A run without a checkpoint (driver before checkpoints) is continued from its last
result_NNNN.vtu + force_displacement.csv + case.json and the restart file of its last block (--restart-file).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from feacore.case import BC, Analysis, Case  # noqa: E402
from feacore.solver import solve  # noqa: E402
from feacore.solver._ccx_failure import load_checkpoint  # noqa: E402

PARTS = {"abc_00000003": "z", "abc_00000065": "y"}         # contract: axis per part
AX = {"x": 0, "y": 1, "z": 2}
POLICY_MM = {"inc": 0.05, "inc_near": 0.02, "inc_tail": 0.05, "block": 0.25, "block_near": 0.06,
             "block_min": 0.02, "block_tail": 0.10}
TAIL_RATIO, STOP_RATIO, STOP_COUNT, MAX_CRUSH = 0.6, 0.01, 3, 0.30
RELEASE = {"adaptive": {"adaptive_release": True, "max_remove": 64}, "fixed": {"max_remove": 8},
           "fixed-held": {"max_remove": 8, "release_travel": "held"}}   # fixed cap, platen held (comparison)
COLS = ["step", "platen_travel_mm", "force_N", "n_failed", "n_detached", "max_peeq", "iterations", "block",
        "retry", "tail", "removed_in_block", "deferred", "stuck", "imbalance_rel", "wall_s", "release_cap"]


def _material(path):
    spec = importlib.util.spec_from_file_location("ccx_crush", ROOT / "scripts" / "ccx_crush.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.material(path)


def failure_options(D: float, limit: float, end_fraction: float, release: str = "adaptive",
                    criterion: dict | None = None) -> dict:
    """criterion: {"criterion": ..., + its parameters} (default {"criterion": "peeq"})."""
    mm = lambda v: v * 1e-3 / D                     # noqa: E731  mm of travel -> ramp fraction
    p = POLICY_MM
    return {"criterion": "peeq", **(criterion or {}), "limit": limit, "aggregate": "mean",
            "near_increments": int(round(1.0 / mm(p["block_near"]))),
            "fine_increments": int(round(1.0 / mm(p["block_min"]))),
            "max_increment": mm(p["inc"]), "max_increment_near": mm(p["inc_near"]),
            "tail_force_ratio": TAIL_RATIO, "max_increment_tail": mm(p["inc_tail"]),
            "tail_increments": int(round(1.0 / mm(p["block_tail"]))),
            "path_groups": ["top", "bottom"], "stop_force_ratio": STOP_RATIO, "stop_force_count": STOP_COUNT,
            "end_fraction": end_fraction, **RELEASE[release]} | (
        {"release_travel": 1e-3 / int(round(1.0 / mm(p["block_min"])))} if release == "fixed-held" else {})


CRIT_KEYS = ("criterion", "compression_factor", "triaxiality_min", "stress_limit")


def criterion_options(failure: str, compression_factor: float | None, uts_pa: float) -> dict:
    if failure == "peeq":
        return {"criterion": "peeq"}
    out = {"criterion": failure, "compression_factor": compression_factor}
    if failure == "peeq_tension":
        out["triaxiality_min"] = 0.0
    if failure == "principal":
        out["stress_limit"] = uts_pa
    return out


def platen_bcs(model, k: int, D: float, platen: str) -> tuple[list[BC], list[list[float]]]:
    """Dirichlet BCs of the platens (see the module doc) and the rigid-body nodes' coordinates (sliding)."""
    val = [0.0, 0.0, 0.0]
    val[k] = -D
    lat = [c for c in range(3) if c != k]
    top = BC("displacement", group="top", components=[0, 1, 2], value=val) if platen == "bonded" else \
        BC("displacement", group="top", components=[k], value=[-D])
    if platen in ("bonded", "sliding_top"):
        return [BC("fixed", group="bottom"), top], []
    if platen != "sliding":
        raise SystemExit(f"--platen {platen!r}")
    X = model.X
    bot = model.nodes_of("bottom")
    i0 = bot[np.argmin(np.linalg.norm(X[bot] - X[bot].mean(axis=0), axis=1))]
    dz = np.abs(X[bot, lat[1]] - X[i0, lat[1]])
    cand = bot[dz <= 0.1 * np.ptp(X[bot, lat[1]]) + 1e-12]       # near the mid-plane of the second lateral axis
    i1 = cand[np.argmax(np.abs(X[cand, lat[0]] - X[i0, lat[0]]))]
    p0, p1 = X[i0].tolist(), X[i1].tolist()
    return [BC("displacement", group="bottom", components=[k], value=[0.0]), top,
            BC("displacement", point=p0, components=lat, value=[0.0, 0.0]),
            BC("displacement", point=p1, components=[lat[1]], value=[0.0])], [p0, p1]


def _platen_of(case: Case) -> str:
    top = next(b for b in case.bcs if b.group == "top")
    bot = next(b for b in case.bcs if b.group == "bottom")
    return "bonded" if len(top.components) == 3 else ("sliding_top" if bot.type == "fixed" else "sliding")


def _driven_value(case: Case) -> np.ndarray:
    for b in case.bcs:
        if b.type == "displacement" and b.group == "top":
            v = np.zeros(3)
            v[b.components] = b.value
            return v
    raise SystemExit("case has no displacement BC")


def _block_end_time(deck: Path, D: float) -> float:
    """Ramp time at the end of a block: its driven *BOUNDARY total (14 digits) / the full ramp travel D."""
    vals, in_bc = [], False
    for ln in deck.read_text().splitlines():
        if ln.startswith("*"):
            in_bc = ln.upper().startswith("*BOUNDARY")
        elif in_bc and len(p := ln.split(",")) >= 4 and float(p[3]) != 0.0:
            vals.append(abs(float(p[3])))
    if not vals or max(vals) - min(vals) > 1e-12 * max(vals):
        raise SystemExit(f"{deck}: cannot read the block-end platen travel")
    return vals[0] / D


def _legacy_checkpoint(parent: Path, rout: Path | None, model, D: float) -> tuple[dict, list[list]]:
    """Checkpoint of a run written before the driver had checkpoints: state / PEEQ / displacement from its last
    .vtu, step / block / peak from its CSV, removal steps from the first .vtu that flags each cell."""
    import meshio
    rows = list(csv.DictReader(open(parent / "force_displacement.csv")))
    vtus = sorted(parent.glob("result_*.vtu"))
    last = meshio.read(vtus[-1])
    step = int(rows[-1]["step"])
    if int(str(vtus[-1].stem).split("_")[1]) != step:
        raise SystemExit(f"{parent}: last .vtu is not the CSV's last step {step}")
    state = np.asarray(last.cell_data["state"][0], np.int8)
    removed_at = np.full(len(state), -1)
    for f in vtus:                              # first step at which each cell is flagged
        s = np.asarray(meshio.read(f).cell_data["state"][0])
        new = (s > 0) & (removed_at < 0)
        removed_at[new] = int(f.stem.split("_")[1])
    block = int(rows[-1]["block"])
    if rout is None:
        raise SystemExit("--restart-file: the .rout of the parent's last block is needed (no checkpoint in "
                         f"{parent}); it is b{block:04d}[rN].rout in the parent's ccx workdir")
    if not rout.name.startswith(f"b{block:04d}"):
        raise SystemExit(f"{rout} is not the restart file of block {block}")
    F = [float(r["force_N"]) for r in rows]
    peak = max(F)
    tail = bool(state.any() and F[-1] < TAIL_RATIO * peak)
    if model.n_elems != len(state) or model.n_nodes != len(last.points):
        raise SystemExit("parent .vtu does not match the mesh")
    ck = {"t": _block_end_time(rout.with_suffix(".inp"), D), "block": block + 1, "step": step, "peak": peak, "rate": 0.0,
          "tail": tail, "low": 0, "stuck": {}, "rout": str(rout.resolve()), "state": state, "removed_at": removed_at,
          "agg_peeq": np.asarray(last.cell_data["PEEQ"][0], float),
          "frozen_U": np.asarray(last.point_data["displacement"], float),
          "legacy_from": str(vtus[-1]), "parent_wall_s_approx": vtus[-1].stat().st_mtime - vtus[0].stat().st_mtime}
    return ck, [[r[c] for c in COLS[:8]] for r in rows]


def _history(run: Path) -> list[list]:
    """Rows (COLS[:8]) of a run and of every run it continued, up to its last written increment."""
    rows = [[r[c] for c in COLS[:8]] for r in csv.DictReader(open(run / "force_displacement.csv"))]
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    if meta.get("continued_from"):
        p = Path(meta["continued_from"])
        older = _history(p) if (p / "run.json").exists() else \
            [[r[c] for c in COLS[:8]] for r in csv.DictReader(open(p / "force_displacement.csv"))]
        rows = [x for x in older if int(float(x[0])) <= int(meta["continued_at_step"])] + rows
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--mesh-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--material", default=str(ROOT / "experiments" / "material" / "alpha_fast_dogbone1.json"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--release", choices=sorted(RELEASE), default="adaptive",
                    help="failed-cell release per block: adaptive cap 64 -> x2 / /2 (default) or fixed 8")
    ap.add_argument("--platen", choices=["bonded", "sliding", "sliding_top"], default="bonded",
                    help="platen contact: bonded (default), frictionless top, frictionless both")
    ap.add_argument("--failure", choices=["peeq", "peeq_tension", "principal"], default="peeq",
                    help="erosion criterion (default peeq: PEEQ >= failure_peeq in any stress state)")
    ap.add_argument("--compression-factor", type=float, default=5.0,
                    help="peeq_tension / principal: cells not in tension erode at this x failure_peeq (0: never)")
    ap.add_argument("--resume", default=None, help="run dir to continue (a new run dir is written)")
    ap.add_argument("--restart-file", default=None, help="parent's last-block .rout (runs without checkpoint)")
    a = ap.parse_args()
    from feacore.solver._ccx_mesh import build_model, read_gmsh

    axis = PARTS[a.part]
    k = AX[axis]
    out_root = Path(a.out) if a.out else Path(os.environ.get("FEACORE_RUNS_DIR", ROOT / "runs")) / "nonlinear" / \
        a.system / a.part
    run_id = a.run_id or time.strftime("run-%Y%m%d-%H%M")
    out = out_root / run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: never overwrite a previous run")
    mat, limit, mdoc = _material(a.material)
    crit = criterion_options(a.failure, a.compression_factor or None, float(mdoc.get("uts_eng", 0.0)) * 1e6)
    parent, resume, parent_rows, parent_summary = None, None, [], {}
    if a.resume:
        parent = Path(a.resume).resolve()
        case = Case.load(parent / "case.json")
        mesh_path = Path(case.mesh_path())
    else:
        md = Path(a.mesh_dir) if a.mesh_dir else Path(os.environ["MESHSYS_OUT"]) / a.system / a.part / "nonlinear"
        mesh_path = (md / "mesh.msh").resolve()
        if not (md / "mesh.inp").exists():
            print(f"warning: {md}/mesh.inp missing (the solver reads mesh.msh)", flush=True)
    gm = read_gmsh(mesh_path)
    for g, d in (("bottom", 2), ("top", 2), ("domain", 3)):
        if gm.group_dim.get(g) != d:
            raise SystemExit(f"{mesh_path}: contract group {g!r} (dim {d}) missing")
    quad = any(ct == "tetra10" for ct, _ in gm.groups["domain"])
    order = case.analysis.order if a.resume else (2 if quad else 1)
    m = build_model(gm, 3, order)
    H = float(m.X[m.nodes_of("top"), k].mean() - m.X[m.nodes_of("bottom"), k].mean())
    if a.resume:
        D = float(-_driven_value(case)[k])
        crit = {c: case.analysis.options["failure"][c] for c in CRIT_KEYS if c in case.analysis.options["failure"]}
        case.analysis.options["failure"] = failure_options(D, limit, min(1.0, MAX_CRUSH * H / D), a.release, crit)
        ck_dir = parent if (parent / "checkpoint.json").exists() else None
        if ck_dir is not None:
            resume = load_checkpoint(ck_dir)
            if (parent / "summary.json").exists():
                parent_summary = json.loads((parent / "summary.json").read_text())
            parent_rows = _history(parent)
        else:
            resume, parent_rows = _legacy_checkpoint(parent, Path(a.restart_file) if a.restart_file else None, m, D)
    else:
        D = MAX_CRUSH * H
        bcs = platen_bcs(m, k, D, a.platen)[0]
        case = Case(mesh=str(mesh_path),
                    analysis=Analysis("elasticity", "3d", order=order, options={
                        "nonlinear": True, "nlgeom": True, "load_factor_max": 1.0,
                        "increments": int(round(1.0 / (POLICY_MM["block"] * 1e-3 / D))),
                        "failure": failure_options(D, limit, 1.0, a.release, crit)}),
                    materials={"resin": mat}, bcs=bcs)
    out.mkdir(parents=True, exist_ok=True)
    case.save(out / "case.json")
    meta = {"system": a.system, "part": a.part, "axis": axis, "mesh": str(mesh_path), "order": order,
            "height_m": H, "ramp_travel_m": D, "max_travel_m": MAX_CRUSH * H, "threads": a.threads,
            "policy_mm": POLICY_MM, "tail_force_ratio": TAIL_RATIO, "stop_force_ratio": STOP_RATIO,
            "stop_force_count": STOP_COUNT, "release": a.release,
            "platen": _platen_of(case), "failure": {c: case.analysis.options["failure"].get(c) for c in CRIT_KEYS},
            "rigid_body_nodes": [b.point for b in case.bcs if b.point is not None], "continued_from": str(parent) if parent else None,
            "continued_at_step": int(resume["step"]) if resume else None,
            "resume_kind": None if not resume else ("legacy (vtu + csv + rout)" if "legacy_from" in resume
                                                    else "checkpoint"),
            "restart_file": resume["rout"] if resume else None, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    log = open(out / "progress.log", "a", buffering=1)

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        log.write(line + "\n")
        print(line, flush=True)

    say(f"start {a.system}/{a.part} -> {out}  ({json.dumps({x: meta[x] for x in ('mesh', 'order', 'height_m', 'ramp_travel_m', 'continued_from', 'continued_at_step')})})")
    f = open(out / "force_displacement.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(COLS)
    rows = []
    t0 = time.time()

    def on_inc(rec):
        row = [rec["step"], rec["lambda"] * D * 1e3, -rec["reaction"][k], rec["n_failed"], rec["n_detached"],
               rec["max_peeq_alive"], rec["iterations"], rec["block"], rec["retry"], int(rec.get("tail", False)),
               rec.get("removed_in_block", 0), rec.get("deferred", 0), rec.get("stuck", 0),
               rec.get("imbalance_rel"), time.time() - t0, rec.get("release_cap")]
        rows.append(row)
        w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        f.flush()
        say(f"step {row[0]} block {row[7]} travel {row[1]:.4f} mm force {row[2]:.1f} N failed {row[3]} "
            f"detached {row[4]} maxPEEQ {row[5]:.4g} it {row[6]} tail {row[9]}")

    wd = out / "ccx_work"
    try:
        r = solve(case, backend="ccx", threads=a.threads, workdir=str(wd), series_dir=str(out),
                  on_increment=on_inc, checkpoint_dir=str(out), resume=resume)
    except Exception as e:  # noqa: BLE001  record why, keep what was written
        say(f"ERROR {type(e).__name__}: {e}")
        raise
    wall = time.time() - t0
    f.close()
    full = [[float(x) for x in rr[:3]] + [int(float(x)) for x in rr[3:5]] + [float(rr[5]), int(float(rr[6])),
            int(float(rr[7]))] for rr in parent_rows] + [rr[:8] for rr in rows]
    if parent is not None:
        with open(out / "force_displacement_full.csv", "w", newline="") as g:
            cw = csv.writer(g)
            cw.writerow(COLS[:8])
            for rr in full:
                cw.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in rr])
    F = np.array([x[2] for x in full])
    ip = int(np.argmax(F))
    cen = m.X[m.conn[:, :m.n_corners]].mean(axis=1)
    order_rm = np.argsort(np.where(r.removed_at >= 0, r.removed_at, np.iinfo(np.int64).max), kind="stable")
    first = [i for i in order_rm if r.state[i] == 1][:20]
    failed = r.state == 1
    zb, zt = m.X[m.nodes_of("bottom"), k].mean(), m.X[m.nodes_of("top"), k].mean()
    ff = cen[first].mean(axis=0) if first else None
    sep_travel = full[-1][1] if r.separated else None
    summ = {**meta, "material": mdoc, "failure_peeq": limit, "stop_reason": r.stop_reason,
            "separated": r.separated, "converged": r.converged, "wall_s": wall,
            "wall_s_total_approx": wall + float(resume.get("parent_wall_s_approx", 0.0) if resume else 0.0)
            + float(parent_summary.get("wall_s_total_approx", 0.0)),
            "n_increments": len(full), "n_increments_this_run": len(rows), "n_cells": m.n_elems,
            "n_nodes": m.n_nodes, "n_failed": int(failed.sum()), "n_detached": int((r.state == 2).sum()),
            "peak_force_N": float(F[ip]), "travel_at_peak_mm": full[ip][1],
            "travel_at_separation_mm": sep_travel, "travel_at_end_mm": full[-1][1], "force_at_end_N": float(F[-1]),
            "first_failure": {"travel_mm": next((x[1] for x in full if x[3] > 0), None),
                              "step": next((x[0] for x in full if x[3] > 0), None),
                              "cells_centroid_m": ff.tolist() if ff is not None else None,
                              "height_fraction": float((ff[k] - zb) / (zt - zb)) if ff is not None else None},
            "failed_centroid_m": cen[failed].mean(axis=0).tolist() if failed.any() else None,
            "failed_bbox_m": [cen[failed].min(axis=0).tolist(), cen[failed].max(axis=0).tolist()]
            if failed.any() else None,
            "max_travel_per_increment_mm": float(np.max(np.diff([0.0] + [x[1] for x in full])))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    say(f"done: {r.stop_reason}; peak {F[ip]:.1f} N at {full[ip][1]:.4f} mm; end {full[-1][1]:.4f} mm; "
        f"{len(rows)} increments in {wall:.0f} s")


if __name__ == "__main__":
    main()
