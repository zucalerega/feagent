"""The --deep run report of a flow run (the counterpart of ace.run_report.build_run_report for stresses).

    build_flow_report(run, out_dir)   one re-solve of the run's scored setup on its own mesh + images
    flow_report_text(report, ...)     the report as compact text (run_report.report_text dispatches here)

Same shape as the solid report where the readers (deep teacher, self-reflection, disagreement map)
need it, with the velocity in place of the stress:

  error_map     field error (velocity, relative L2) and check-point error; hotspots: clusters of the
                quadrature points carrying 80 % of the squared velocity error, each with its nearest
                group, the run's local cell size there, the reference speed and the speed error.
  region_grid   the domain's bounding box cut into cells (10 x 10 in 2D, 6 x 6 x 4 in 3D): per cell
                the run's cells in it, their mean size, the cell's share of the velocity error and the
                worst local speed error.
  check_points  every check point (disclosed and hidden) with reference and run speed.
  vm_at_grid    the run's speed at the task's fixed quadrature points (a subsample) with their grid
                cells, for run_report.disagreement_map (grid_quantity = "speed").
  feedback      the reference-free flow feedback the agent saw (ace.flow_feedback): convergence, mass
                balance, reversed flow, the face-jump error indicator.
  images        mesh (DISCLOSED check points marked), error (the velocity error density, hotspots
                numbered) and speed (the run's next to the reference's).

Never raises: on failure the report is {"run_id", "error"}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ace import config, fea
from ace import field_error as fe
from ace import flow_feedback as ff
from ace import flow_field
from ace import inspect_tools as it
from ace.memory import agent_of

GRID = {2: (10, 10), 3: (6, 6, 4)}
SAMPLE = 3000              # quadrature points kept for the disagreement map
DPI = 100


def _sig(x, n: int = 3):
    return ff._sig(x, n)


def _hotspots(p, fl, pts: np.ndarray, e: np.ndarray, rel: np.ndarray, sp_ref: np.ndarray) -> tuple[list, float]:
    r = it.RADIUS * fl.diag
    total = float(e.sum())
    out = []
    if total <= 0:
        return out, r
    for m in sorted(it._hotspots(pts[:, : p.dim], e, r), key=lambda m: -e[m].sum())[: it.MAX_HOTSPOTS]:
        c = np.average(pts[m], axis=0, weights=e[m])
        g, d = fl.nearest_group(c)
        out.append({"centroid": [_sig(v, 4) for v in c[: p.dim]], "share": _sig(e[m].sum() / total),
                    "nearest_group": g, "distance": _sig(d), "local_h": _sig(fl.local_size(c, r)),
                    "speed_ref": _sig(float(np.mean(sp_ref[m]))),
                    "speed_rel_error": _sig(float(np.sqrt(np.mean(rel[m] ** 2))))})
    return out, r


def _grid(p, fl, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid cell index of each point, cell shape, cell size) over the fluid's bounding box; in 3D the
    4 cells go on the shortest axis."""
    lo, hi = fl.lo[: p.dim], fl.hi[: p.dim]
    shape = list(GRID[p.dim])
    if p.dim == 3:
        order = np.argsort(hi - lo)            # shortest axis first
        shape = [0, 0, 0]
        for ax, n in zip(order, (4, 6, 6)):
            shape[ax] = n
    shape = np.array(shape)
    size = (hi - lo) / shape
    idx = np.clip(((pts[:, : p.dim] - lo) / np.maximum(size, 1e-300)).astype(int), 0, shape - 1)
    flat = np.ravel_multi_index(idx.T, shape)
    return flat, shape, size


def _region_grid(p, fl, pts, e, rel) -> dict:
    flat, shape, size = _grid(p, fl, pts)
    cflat, _, _ = _grid(p, fl, fl.C)
    total = max(float(e.sum()), 1e-300)
    cells = []
    for k in np.unique(np.concatenate([flat, cflat])):
        inq, inc = flat == k, cflat == k
        centre = (fl.lo[: p.dim] + (np.array(np.unravel_index(k, shape)) + 0.5) * size).tolist()
        cells.append({"centre": [_sig(v, 4) for v in centre], "n_cells": int(inc.sum()),
                      "mean_h": _sig(float(fl.h[inc].mean())) if inc.any() else None,
                      "error_share": _sig(float(e[inq].sum()) / total), "max_rel_err": _sig(float(np.abs(rel[inq]).max()))
                      if inq.any() else None})
    return {"shape": shape.tolist(), "cell_size": [_sig(v, 4) for v in size], "cells": cells}


def _speed_at_grid(p, fl, pts, sp_run, task_id: str) -> dict:
    """The run's speed at a fixed per-task subsample of the quadrature points (the same for every run
    of the task), with their grid cells: run_report.disagreement_map compares runs on it."""
    step = max(1, len(pts) // SAMPLE)
    sel = np.arange(0, len(pts), step)
    flat, shape, size = _grid(p, fl, pts[sel])
    cells = {}
    for k in np.unique(flat):
        centre = fl.lo[: p.dim] + (np.array(np.unravel_index(k, shape)) + 0.5) * size
        cells[int(k)] = {"centre": [_sig(v, 4) for v in centre], "near": fl.nearest_group(centre)[0]}
    return {"cell": flat.tolist(), "vm": [float(v) for v in sp_run[sel]], "cells": cells}


def _images(problem, mesh: dict, res, fl, pts, e, em: dict, shown: list, ref_speed: np.ndarray,
            out_dir: Path, rid: str) -> dict:
    """mesh / error / speed pictures; e = squared velocity error at each quadrature point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    out["mesh"] = ff.render(problem, mesh, res, "mesh", "z", out_dir / f"{rid}_mesh.png")
    a, b = (0, 1)
    if problem.dim == 3:                                   # the section normal to the shortest axis
        ax3 = int(np.argmin(fl.hi - fl.lo))
        a, b = [k for k in range(3) if k != ax3]
        mid = 0.5 * (fl.lo[ax3] + fl.hi[ax3])
        band = np.abs(pts[:, ax3] - mid) <= 0.5 * max(float(np.median(fl.reach)) * 2, 1e-12)
    else:
        band = np.ones(len(pts), dtype=bool)
    # error density at the quadrature points (log), hotspots numbered, disclosed check points marked
    fig, ax = plt.subplots(figsize=(9, 5))
    dens = e                                               # squared velocity error per point (not per volume)
    v = np.log10(np.maximum(dens[band], max(float(dens.max()), 1e-300) * 1e-3))
    sc = ax.scatter(pts[band, a], pts[band, b], c=v, s=4, cmap="magma")
    fig.colorbar(sc, ax=ax, label="log10 |u_run - u_ref|^2", shrink=0.8)
    for k, h in enumerate(em["hotspots"], 1):
        c = h["centroid"]
        ax.annotate(str(k), (c[a], c[b]), color="cyan", fontsize=11, weight="bold", ha="center", va="center")
    for c in shown:
        ax.plot(c["xyz"][a], c["xyz"][b], "o", mfc="none", mec="lime", ms=8)
    ax.set_aspect("equal")
    ax.set_title(f"{rid}: where the velocity error is (field error {em['field_error']:.3g})", fontsize=10)
    fig.savefig(out_dir / f"{rid}_error.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    out["error"] = str(out_dir / f"{rid}_error.png")
    # the run's speed next to the reference's, same colour scale
    sp_run = np.linalg.norm(flow_field.velocity_at(res, pts), axis=1)
    vmax = float(max(sp_run[band].max(), ref_speed[band].max()))
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    for axx, val, name in ((axs[0], sp_run, "this run"), (axs[1], ref_speed, "reference")):
        s = axx.scatter(pts[band, a], pts[band, b], c=val[band], s=4, cmap="viridis", vmin=0, vmax=vmax)
        axx.set_aspect("equal")
        axx.set_title(f"speed |u|: {name}", fontsize=9)
    fig.colorbar(s, ax=axs, label="|u| [m/s]", shrink=0.8)
    fig.savefig(out_dir / f"{rid}_speed.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    out["speed"] = str(out_dir / f"{rid}_speed.png")
    return out


def build_flow_report(run: dict, out_dir: Path) -> dict:
    """The full report of one flow run (see the module docstring), also written to out_dir/<run_id>.json."""
    t0 = time.perf_counter()
    rid = run.get("run_id") or "run"
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pid = run["problem_id"]
        p = fea.get_problem(pid)
        cp, mesh, notes = it._scored_mesh(run, out_dir / f"{rid}_work")
        res = fe._solve(p, mesh, cp["setup"], config.TASKS[pid]["qoi"])
        fl = ff._fluid(res)
        ref = flow_field.reference(pid)
        pts = ref["pts"]
        u_run = flow_field.velocity_at(res, pts)
        e = ref["w"] * np.sum((u_run - ref["u"]) ** 2, axis=1)
        sp_ref, sp_run = np.linalg.norm(ref["u"], axis=1), np.linalg.norm(u_run, axis=1)
        rel = (sp_run - sp_ref) / np.maximum(sp_ref, 0.2 * float(np.median(sp_ref)))
        errs = flow_field.errors_of_result(pid, res)
        hs, radius = _hotspots(p, fl, pts, e, rel, sp_ref)
        em = {"field_error": errs["field"], "probe_error": errs["points"], "hotspots": hs, "n_hotspots": len(hs),
              "hotspot_radius": _sig(radius), "median_h": _sig(float(np.median(fl.h))), "notes": notes}
        pp = flow_field.probe_points(pid)
        sp_chk = flow_field.check_speeds(res, np.array(pp["points"]))
        checks = [{"i": i, "xyz": [_sig(v, 4) for v in x], "disclosed": i < config.PROBE_DISCLOSED,
                   "speed_ref": _sig(sr), "speed_run": _sig(float(s)), "rel_err": float((s - sr) / sr)}
                  for i, (x, sr, s) in enumerate(zip(pp["points"], pp["speed_ref"], sp_chk))]
        shown = [c for c in checks if c["disclosed"]]
        rg = _region_grid(p, fl, pts, e, rel)
        d2 = np.sum((u_run - ref["u"]) ** 2, axis=1)
        images = _images(p, mesh, res, fl, pts, d2, em, shown, sp_ref, out_dir, rid)
        solve = run["solves"][cp["solve_index"]]
        k = agent_of(rid)
        hid = [c for c in checks if not c["disclosed"]]
        hidden = None
        if hid:
            w = max(hid, key=lambda c: abs(c["rel_err"]))
            hidden = {"n": len(hid), "rms": float(np.sqrt(np.mean([c["rel_err"] ** 2 for c in hid]))),
                      "worst": float(abs(w["rel_err"])), "worst_near_group": fl.nearest_group(np.array(w["xyz"]))[0]}
        report = {
            "run_id": rid, "agent": None if k is None else f"a{k}", "task": pid, "dim": p.dim, "flow": True,
            "mesh_request": cp["request"],
            "mesh_stats": {"nodes": mesh.get("nodes"), "dofs": (solve["result"].get("info") or {}).get("n_dofs"),
                           "n_cells": int(len(fl.C)), "cells": mesh.get("cells"),
                           "fv_quality": {k2: (mesh.get("fv_quality") or {}).get(k2)
                                          for k2 in ("non_orthogonality", "skewness", "near_wall")}},
            "setup": cp["setup"], "error_map": em, "region_grid": rg, "check_points": checks,
            "hidden_summary": hidden, "vm_at_grid": _speed_at_grid(p, fl, pts, sp_run, pid), "grid_quantity": "speed",
            "feedback": ff.feedback(p, mesh, cp["setup"], res), "images": images,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001  a report is a bonus: never break the loop over one
        report = {"run_id": rid, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 2)}
    try:
        (out_dir / f"{rid}.json").write_text(json.dumps(report, indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        report.setdefault("error", f"could not write the report file: {e}")
    return report


def _g(x, n: int = 3) -> str:
    return "-" if x is None else (f"{x:.{n}g}" if isinstance(x, (int, float)) else str(x))


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.3g}%"


def _xyz(v) -> str:
    return "(" + ", ".join(_g(x) for x in v) + ")"


def flow_report_text(report: dict, max_chars: int = 6000, hide_hidden: bool = False) -> str:
    """A flow report as compact text: numbers, mesh request and quality, setup, velocity-error
    hotspots, check points, the reference-free feedback, then the region grid (worst first)."""
    L = [f"RUN {report['run_id']} (agent {report.get('agent')}, task {report.get('task')}, flow)"]
    n = report.get("numbers")
    if n:
        rank = f", rank {n['rank_in_group']}/{n['group_size']}" if n.get("rank_in_group") is not None else ""
        L.append(f"score {_g(n['score'])} (base {_g(n['base_score'])} + be-the-best bonus {_g(n['bonus'])}"
                 + (f" + diversity bonus {_g(n['diversity_bonus'])}" if n.get("diversity_bonus") else "")
                 + f"){rank}; valid {n.get('valid')}" + (f" ({n['invalid_because']})" if n.get("invalid_because") else ""))
        L.append(f"velocity field error {_pct(n['field_error'])}, check-point speed error {_pct(n['probe_error'])}; "
                 f"gain {_g(n['gain'])}x the automesher (target {config.AUTO_GAIN_TARGET:g}x: "
                 f"{'met' if n['meets_target'] else 'NOT met'}; target error {_pct(n['target_error'])}); "
                 f"accuracy term {_g(n['accuracy'])}, cost term {_g(n['efficiency'])}; "
                 f"DOFs {n['dofs']} vs automesher {n['automesher_dofs']} (error {_pct(n['automesher_error'])})")
    req = report.get("mesh_request") or {}
    refs = "; ".join(f"{r.get('group') or r.get('edge') or _xyz(r.get('point') or r.get('box') or [])} size "
                     f"{_g(r.get('size'))} radius {_g(r.get('radius'))}" for r in req.get("refinements") or []) or "none"
    extra = "".join(f", {k} {json.dumps(req[k], separators=(',', ':'))}" for k in ("quads", "boundary_layer", "sweep")
                    if req.get(k))
    L.append(f"mesh request: h {_g(req.get('h'))}{extra}, refinements: {refs}")
    ms, em = report["mesh_stats"], report["error_map"]
    fq = ms.get("fv_quality") or {}
    near = "; ".join(f"{g} {_g(v.get('wall_distance'))}" for g, v in (fq.get("near_wall") or {}).items())
    L.append(f"mesh: {ms['nodes']} nodes, {ms['dofs']} DOFs, cells {ms.get('cells')}, max non-orthogonality "
             f"{_g((fq.get('non_orthogonality') or {}).get('max'))} deg, max skewness {_g((fq.get('skewness') or {}).get('max'))}, "
             f"median h {_g(em['median_h'])}; first-cell wall distance: {near or '-'}")
    L.append("setup: " + json.dumps(report.get("setup") or {}, separators=(",", ":"), default=str))
    L.append(f"error map: velocity field error {_pct(em['field_error'])}, check-point RMS {_pct(em['probe_error'])}; "
             f"{em['n_hotspots']} hotspots hold 80% of the squared velocity error (radius {_g(em['hotspot_radius'])}); "
             "numbered as on the error image:")
    for k, h in enumerate(em["hotspots"], 1):
        L.append(f"  #{k} at {_xyz(h['centroid'])}: {_pct(h['share'])} of the error, nearest group "
                 f"{h['nearest_group']} ({_g(h['distance'])} away), local h {_g(h['local_h'])}, "
                 f"reference speed {_g(h['speed_ref'])}, speed error {_pct(h['speed_rel_error'])}")
    fb = report.get("feedback") or {}
    conv, mb, rev = fb.get("convergence") or {}, fb.get("mass_balance") or {}, fb.get("reversed_flow") or {}
    L.append(f"solve: {conv.get('iterations')} iterations, converged {conv.get('converged')}; mass imbalance "
             f"{_g(mb.get('imbalance'))}; reversed flow {_pct(rev.get('volume_share'))} of the domain"
             + "".join(f"; zone {_xyz(z['from'])}..{_xyz(z['to'])} near {z['nearest_group']}" for z in rev.get("zones", [])))
    ei = fb.get("error_indicator") or {}
    if ei:
        L.append(f"reference-free indicator (what the agent saw): {_g(ei.get('rel_indicator'))}; hotspots: "
                 + "; ".join(f"{_xyz(h['at'])} near {h['nearest_group']} ({_pct(h['share'])})" for h in ei.get("hotspots", [])))
    cps = report.get("check_points") or []
    L.append("check points (P = disclosed to the agent" + ("" if hide_hidden else ", H = hidden")
             + "; error = run vs reference speed):")
    for c in cps:
        if c["disclosed"] or not hide_hidden:
            L.append(f"  {'P' if c['disclosed'] else 'H'}{c['i']} {_xyz(c['xyz'])}: ref {_g(c['speed_ref'])}, "
                     f"run {_g(c['speed_run'])}, error {c['rel_err'] * 100:+.2f}%")
    hid = report.get("hidden_summary")
    if hide_hidden and hid:
        L.append(f"  {hid['n']} hidden check points (locations withheld): RMS error {_pct(hid['rms'])}, worst "
                 f"{_pct(hid['worst'])}, the worst one near group {hid['worst_near_group']}")
    rg = report["region_grid"]
    cells = sorted(rg["cells"], key=lambda c: -(c["error_share"] or 0))
    L.append(f"region grid ({'x'.join(map(str, rg['shape']))} cells of size {_xyz(rg['cell_size'])}; worst first): "
             "centre | run cells | mean h | share of the velocity error | worst local speed error")
    for c in cells[:20]:
        L.append(f"  {_xyz(c['centre'])} | {c['n_cells']} | {_g(c['mean_h'])} | {_pct(c['error_share'])} | "
                 f"{_pct(c['max_rel_err'])}")
    text = "\n".join(L)
    if len(text) > max_chars:
        cut = "\n... (cut: lower-error grid cells omitted)"
        text = text[: max_chars - len(cut)].rsplit("\n", 1)[0] + cut
    return text
