#!/usr/bin/env python3
"""Reference meshes of the real-data fine-tuning (problem abc_00000065_compression) through the frozen
nonlinear scoring (ace.nonlinear): their force-displacement curves against the Instron test, their errors,
the error-vs-nodes plot, and (--calibrate) the crosshead mapping u_cross = u_part + delta + c * F.

Reference meshes (all P2):
  auto       the automesher: gmsh default sizes, curved P2 (the score's baseline)
  swarm_bc0  the trained swarm's best bc0 mesh (div_s8 g7 a2; request in baselines/swarm_bc0_request.json),
             i.e. the mesh before fine-tuning on the test
  h4, h25, h20   uniform h = 4, 2.5 and 2 mm

    python scripts/nl_baselines.py                       # solve the missing ones (needs ccx), table + plot
    python scripts/nl_baselines.py --only auto h4 --force
    python scripts/nl_baselines.py --calibrate           # refit delta, c on the 2 mm curve -> nonlinear_spec.json
    python scripts/nl_baselines.py --plot-only --add "fine-tuned g3 a1=ace_runs/rd1/gen3/work/.../nonlinear/summary.json"

Writes results/real_data_finetune/baselines/<name>/ (run_curve's curve.csv, summary.json, series/, pngs, mesh.msh),
baselines/baselines.json and baselines/error_vs_nodes.png. Points added with --add are kept in baselines.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PID = "abc_00000065_compression"
OUT = ROOT / "results" / "real_data_finetune" / "baselines"
SWARM_REQUEST = OUT / "swarm_bc0_request.json"
# name -> how the mesh is made: "auto", a mesh request file (fea.make_mesh kwargs) or a uniform h in metres
BASELINES = {"auto": "auto", "swarm_bc0": SWARM_REQUEST, "h4": 0.004, "h25": 0.0025, "h20": 0.002}
LABELS = {"auto": "automesher", "swarm_bc0": "swarm bc0 mesh", "h4": "uniform 4 mm",
          "h25": "uniform 2.5 mm", "h20": "uniform 2 mm"}
CALIBRATION_MESH = "h20"
ERR_FLOOR = 0.005     # errors below 0.5 % count as 0.5 % in the score (as in field mode)


def need_ccx():
    """Solving needs CalculiX (only on the Brev box fe-ref-1, not on the laptop)."""
    from feacore.solver._ccx_io import CcxError, find_ccx
    try:
        find_ccx()
    except CcxError as e:
        sys.exit(f"{e}\nSolving the baselines needs CalculiX: run this on fe-ref-1 (or use --plot-only).")


# --------------------------------------------------------------------------- meshes and solves

def make_mesh(name: str, out_dir: Path) -> dict:
    """The baseline's mesh as ace.fea mesh stats ({"path", "order", ...})."""
    from ace import fea
    p = fea.get_problem(PID)
    path = out_dir / "mesh.msh"
    how = BASELINES[name]
    if how == "auto":
        import gmsh
        from ace.field_error import _unfold_tet10
        from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
        with gmsh_session(p.id):
            p.build_geometry()
            ensure_domain_group(p.dim)
            gmsh.model.mesh.generate(p.dim)                    # gmsh's default sizes
            gmsh.model.mesh.setOrder(2)                        # curved P2, as ace.nonlinear.automesher_baseline;
            _unfold_tet10()                                    # run_curve straightens only rejected edges
            gmsh.write(str(path))
        return {"ok": True, "path": str(path), "order": 2}
    if isinstance(how, Path):
        return fea.make_mesh(p, path, **json.loads(how.read_text()))
    return fea.make_mesh(p, path, h=how, order=2)


def solve_baseline(name: str, threads: int) -> dict:
    """Mesh + nonlinear scoring run of one baseline into OUT/<name>/; returns run_curve's dict."""
    from ace import nonlinear
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    mesh = make_mesh(name, d)
    if not mesh.get("ok"):
        return {"ok": False, "error": f"meshing failed: {mesh.get('error')}"}
    res = nonlinear.run_curve(PID, mesh, d, threads=threads, series=True)
    print(f"  {name}: ok={res['ok']} nodes={res.get('nodes')} error={_pct(res.get('mean_rel_err'))} "
          f"wall={res.get('wall_s', 0):.0f}s {res.get('error') or ''}", flush=True)
    return res


def load(name: str) -> dict | None:
    f = OUT / name / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


# --------------------------------------------------------------------------- crosshead mapping

def mapped(u_part, force, delta, c) -> np.ndarray:
    """Crosshead travel of a run_curve curve (origin first, kept at 0): u_cross = u_part + delta + c F."""
    u_part, force = np.asarray(u_part, float), np.asarray(force, float)
    return np.concatenate([[0.0], u_part[1:] + delta + c * force[1:]])


def mean_error(summary: dict, delta: float, c: float, meas: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """(mean relative error, per-checkpoint errors, simulated checkpoint forces) of one curve for given
    delta, c: the frozen definition of ace.nonlinear.run_curve (a checkpoint not reached counts 100 %)."""
    ck, Fm = np.asarray(meas["checkpoints_mm"]), np.asarray(meas["force_N"])
    u_cross = mapped(summary["u_part_mm"], summary["force_N"], delta, c)
    Fs = np.interp(ck, u_cross, summary["force_N"])
    rel = np.where(ck <= u_cross[-1], np.abs(Fs - Fm) / Fm, 1.0)
    return float(rel.mean()), rel, Fs


def calibrate(meas: dict, start: tuple[float, float]) -> tuple[float, float, float]:
    """Least-squares-style fit of (delta mm, c mm/N) on the finest baseline: Nelder-Mead on the mean
    relative error at the checkpoints. c is optimised in units of 1e-5 mm/N so both unknowns are O(0.1-1)."""
    from scipy.optimize import minimize
    fine = load(CALIBRATION_MESH)
    if not fine or not fine.get("ok"):
        sys.exit(f"--calibrate needs the {CALIBRATION_MESH} curve: run the baselines on fe-ref-1 first")
    f = lambda x: mean_error(fine, x[0], x[1] * 1e-5, meas)[0]
    r = minimize(f, [start[0], start[1] * 1e5], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
    delta, c = float(r.x[0]), float(r.x[1]) * 1e-5
    print(f"calibrated on {CALIBRATION_MESH}: delta = {delta:.4f} mm, c = {c:.3e} mm/N "
          f"({1 / c / 1e3 if c > 0 else float('inf'):.1f} kN/mm), mean error {r.fun * 100:.2f} %")
    return delta, c, float(r.fun)


def remap(s: dict, delta: float, c: float, meas: dict) -> dict:
    """A run_curve dict with its mapped fields (u_cross, checkpoint forces, errors) redone for new delta, c;
    the solve itself (u_part, force) does not change."""
    e, rel, Fs = mean_error(s, delta, c, meas)
    u_cross = mapped(s["u_part_mm"], s["force_N"], delta, c)
    reached = np.asarray(meas["checkpoints_mm"]) <= u_cross[-1]
    return s | {"u_cross_mm": u_cross.round(5).tolist(), "checkpoints_mm": list(meas["checkpoints_mm"]),
                "force_meas_N": list(meas["force_N"]), "rel_err": rel.round(5).tolist(), "mean_rel_err": e,
                "force_sim_N": [round(float(f), 2) if r else None for f, r in zip(Fs, reached)]}


def remap_baseline(name: str, delta: float, c: float, meas: dict) -> None:
    """remap() a saved baseline in place, and redraw its curve plot."""
    from ace import nonlinear
    s = load(name)
    if not s or not s.get("ok"):
        return
    s = remap(s, delta, c, meas)
    (OUT / name / "summary.json").write_text(json.dumps(s, indent=1))
    nonlinear.curve_plot([{"label": f"{LABELS[name]} ({s['nodes']} nodes): {s['mean_rel_err']:.1%}",
                           "u_cross_mm": s["u_cross_mm"], "force_N": s["force_N"]}],
                         OUT / name / "curve.png", title=LABELS[name])


# --------------------------------------------------------------------------- table and plot

def score(e: float, dofs: int, auto: dict) -> float:
    """base_score of the nonlinear mode: accuracy gain over the automesher per DOF ratio."""
    g = auto["mean_rel_err"] / max(e, ERR_FLOOR)
    acc = (g / 2) ** 2 if g < 2 else g / 2
    return acc / (dofs / auto["dofs"])


def table(auto_ref: dict | None) -> dict:
    """baselines.json rows from the saved summaries; scored against the automesher row."""
    rows = {}
    for name in BASELINES:
        s = load(name)
        if s and s.get("ok"):
            rows[name] = {"label": LABELS[name], "nodes": s["nodes"], "dofs": s["dofs"],
                          "mean_rel_err": s["mean_rel_err"], "rel_err": s["rel_err"],
                          "force_sim_N": s["force_sim_N"], "wall_s": s.get("wall_s"),
                          "straightened": s.get("straightened", False), "dir": str((OUT / name).relative_to(ROOT))}
    auto = rows.get("auto") or auto_ref
    for r in rows.values():
        if auto:
            r["score"] = score(r["mean_rel_err"], r["dofs"], auto)
    return rows


def plot(rows: dict, added: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ref = sorted(rows.values(), key=lambda r: r["nodes"])
    uni = [r for r in ref if r["label"].startswith("uniform")]
    if uni:     # the uniform refinements as a line: the error of the cheapest mesh at each node count
        ax.plot([r["nodes"] for r in uni], [r["mean_rel_err"] * 100 for r in uni], color="#9aa0a6", lw=1.2, zorder=1)
    raised = False
    for i, r in enumerate(ref):     # a label is raised when the previous (not raised) label is close by
        raised = i > 0 and not raised and r["nodes"] < 2 * ref[i - 1]["nodes"]
        ax.scatter(r["nodes"], r["mean_rel_err"] * 100, s=46, color="#3b6ea8", zorder=3)
        ax.annotate(f"{r['label']}\n{r['mean_rel_err'] * 100:.1f} %", (r["nodes"], r["mean_rel_err"] * 100),
                    textcoords="offset points", xytext=(14, 24) if raised else (7, 4), fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": "#9aa0a6", "lw": 0.8} if raised else None)
    for a in added:
        ax.scatter(a["nodes"], a["mean_rel_err"] * 100, s=60, marker="D", color="#d9822b", zorder=4)
        ax.annotate(f"{a['label']}\n{a['mean_rel_err'] * 100:.1f} %", (a["nodes"], a["mean_rel_err"] * 100),
                    textcoords="offset points", xytext=(7, -16), fontsize=8.5, color="#a85a12")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    pts = [r["nodes"] for r in ref] + [x["nodes"] for x in added]
    errs = [r["mean_rel_err"] * 100 for r in ref] + [x["mean_rel_err"] * 100 for x in added]
    if pts:     # room for the labels: right of the last point, above the highest
        ax.set_xlim(min(pts) / 1.25, max(pts) * 2.2)
        ax.set_ylim(0, max(errs) * 1.2)
    ax.set_xlabel("nodes (log scale)")
    ax.set_ylabel("mean error vs the test at the checkpoints (%)")
    ax.set_title("C-bracket compression: nonlinear simulation vs the Instron test", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _pct(e) -> str:
    return "-" if e is None else f"{e * 100:.2f} %"


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(BASELINES), help="baselines to (re)solve")
    ap.add_argument("--force", action="store_true", help="re-solve even if a summary exists")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--calibrate", action="store_true",
                    help=f"refit delta, c on the {CALIBRATION_MESH} curve and write them to nonlinear_spec.json")
    ap.add_argument("--plot-only", action="store_true", help="no solves: table and plot from the saved summaries")
    ap.add_argument("--add", action="append", default=[], metavar="LABEL=SUMMARY_JSON",
                    help="add a run's point (a run_curve summary.json) to the plot; kept in baselines.json")
    a = ap.parse_args()
    from ace import nonlinear

    OUT.mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        todo = [n for n in (a.only or BASELINES) if a.force or not (load(n) or {}).get("ok")]
        if todo:
            need_ccx()
            print("solving:", " ".join(todo), flush=True)
            for name in todo:
                if a.force:     # run_curve reuses a summary of the same mesh: drop it
                    (OUT / name / "summary.json").unlink(missing_ok=True)
                    shutil.rmtree(OUT / name / "series", ignore_errors=True)
                solve_baseline(name, a.threads)

    meas = nonlinear.measured()
    spec = dict(nonlinear.spec())
    if a.calibrate:
        delta, c, e = calibrate(meas, (spec.get("delta_mm", 0.233), spec.get("compliance_mm_per_N", 2.9e-5)))
        for name in BASELINES:
            remap_baseline(name, delta, c, meas)
        # the score's cached automesher baseline was mapped with the old delta, c: redo its errors too
        cache = nonlinear.CACHE / f"{PID}_nl_auto.json"
        if cache.exists():
            cache.write_text(json.dumps(remap(json.loads(cache.read_text()), delta, c, meas), indent=1))
            print("remapped", cache.relative_to(ROOT))
        spec.update(delta_mm=round(delta, 4), compliance_mm_per_N=float(f"{c:.4g}"))
        spec.setdefault("calibration", {})["fitted_error"] = round(e, 5)

    # the score's own baseline (cached by ace.nonlinear) for the table when "auto" is not solved here
    auto_ref = None
    if "auto" not in [n for n in BASELINES if (load(n) or {}).get("ok")]:
        try:
            auto_ref = nonlinear.automesher_baseline(PID)
        except Exception as e:      # no ccx and no cache: the table goes without scores
            print("no automesher baseline:", e)

    rows = table(auto_ref)
    old = json.loads((OUT / "baselines.json").read_text()) if (OUT / "baselines.json").exists() else {}
    added = {x["label"]: x for x in old.get("added", [])}
    for item in a.add:
        label, path = item.split("=", 1)
        s = json.loads(Path(path).read_text())
        added[label] = {"label": label, "nodes": s["nodes"], "dofs": s["dofs"],
                        "mean_rel_err": s["mean_rel_err"], "summary": path}
    out = {"problem_id": PID, "delta_mm": spec.get("delta_mm"), "compliance_mm_per_N": spec.get("compliance_mm_per_N"),
           "checkpoints_mm": meas["checkpoints_mm"], "force_meas_N": meas["force_N"],
           "baselines": rows, "added": list(added.values())}
    (OUT / "baselines.json").write_text(json.dumps(out, indent=1))
    plot(rows, out["added"], OUT / "error_vs_nodes.png")

    if a.calibrate:     # the reference numbers the teacher brief and the agents quote
        spec["reference_errors"] = {n: {"label": r["label"], "nodes": r["nodes"], "dofs": r["dofs"],
                                        "mean_rel_err": round(r["mean_rel_err"], 5)} for n, r in rows.items()}
        nonlinear.SPEC_FILE.write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", nonlinear.SPEC_FILE.relative_to(ROOT))

    print(f"\ndelta = {spec.get('delta_mm')} mm, c = {spec.get('compliance_mm_per_N')} mm/N")
    print(f"{'mesh':16} {'nodes':>7} {'dofs':>7} {'error':>8} {'score':>6}  per checkpoint")
    for r in rows.values():
        print(f"{r['label']:16} {r['nodes']:7d} {r['dofs']:7d} {_pct(r['mean_rel_err']):>8} "
              f"{r.get('score', float('nan')):6.2f}  " + " ".join(f"{x * 100:5.1f}" for x in r["rel_err"]))
    print("wrote", (OUT / "baselines.json").relative_to(ROOT), "and error_vs_nodes.png")


if __name__ == "__main__":
    main()
