#!/usr/bin/env python3
"""Plot a real-data fine-tuning run (problem abc_00000065_compression) for Act 3 of the talk.

    .venv/bin/python talk_plots/act3_real_data/plot_finetune_run.py ace_runs/<run> [--out DIR] [--max-nodes N]

Reads every run record <run>/gen*/runs/*.json of the problem. Per run it takes checks["nonlinear"]
("mean_rel_err", "nodes", "dofs", "ok", and the curve: "u_cross_mm"/"force_N" in the record, else the
nonlinear/curve.csv next to the scored mesh, columns u_cross_mm or u_part_mm and force_N or F_N; a
u_part-only curve is mapped with the frozen delta and compliance). A run with a curve but no error gets its
error recomputed from the curve with the frozen definition. Runs without either are counted as failed.

Writes (to --out, default this folder):
  fig7_finetune_progress.png/.pdf          best and mean error per generation vs the automesher and target
  fig4b_error_vs_nodes_finetuned.png/.pdf  figure 4 with each generation's best mesh added
  fig5b_curves_finetuned.png/.pdf          figure 5 with the overall best mesh's curve added
  finetune_data.json                       every run's numbers and the per-generation summary
Needs act3_data.json from make_figures.py (run that first).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PID = "abc_00000065_compression"


def read_curve_csv(path: Path, delta: float, c: float):
    """(u_cross mm, F N) from a curve.csv with u_cross_mm or u_part_mm and force_N or F_N; None if unreadable."""
    try:
        rows = list(csv.DictReader(path.read_text().splitlines()))
        cols = rows[0].keys()
        fk = "force_N" if "force_N" in cols else "F_N"
        F = np.array([float(r[fk]) for r in rows])
        if "u_cross_mm" in cols:
            u = np.array([float(r["u_cross_mm"]) for r in rows])
        else:
            u = mf.mapped(np.array([float(r["u_part_mm"]) for r in rows]), F, delta, c)
        return u, F
    except (OSError, IndexError, KeyError, ValueError):
        return None


def find_curve(rec: dict, nl: dict, rec_path: Path, run_dir: Path, delta: float, c: float):
    if nl.get("u_cross_mm") and nl.get("force_N"):
        return np.array(nl["u_cross_mm"], float), np.array(nl["force_N"], float)
    cands = []
    if nl.get("curve_csv"):
        p = Path(nl["curve_csv"])
        cands.append(p)
        parts = p.parts       # recorded on another machine: re-root the part after ".../<run name>/"
        if run_dir.name in parts:
            cands.append(run_dir.joinpath(*parts[parts.index(run_dir.name) + 1:]))
    m = re.match(r"g(\d+)-(.+)-a(\d+)$", rec.get("run_id", ""))
    if m:
        cands += sorted((rec_path.parent.parent / "work" / f"{m.group(2)}-a{m.group(3)}").glob("**/nonlinear/curve.csv"))
    for p in cands:
        if p.is_file():
            got = read_curve_csv(p, delta, c)
            if got is not None:
                return got
    return None


def load_runs(run_dir: Path, D: dict, max_nodes: int | None) -> list[dict]:
    ck = np.array(D["test"]["checkpoints_mm"])
    Fm = np.array(D["test"]["force_meas_N"])
    delta, c = D["spec"]["delta_mm"], D["spec"]["compliance_mm_per_N"]
    out = []
    for f in sorted(run_dir.glob("gen*/runs/*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("problem_id", PID) != PID:
            continue
        gen = rec.get("generation")
        if gen is None:
            m = re.match(r"gen(\d+)", f.parent.parent.name)
            gen = int(m.group(1)) if m else -1
        checks = rec.get("checks") or {}
        nl = checks.get("nonlinear") or {}
        curve = find_curve(rec, nl, f, run_dir, delta, c)
        err = nl.get("mean_rel_err")
        if err is None and curve is not None:            # recompute with the frozen definition
            u, F = curve
            Fs = np.interp(ck, u, F)
            err = float(np.where(ck <= u[-1], np.abs(Fs - Fm) / Fm, 1.0).mean())
        nodes = nl.get("nodes") or checks.get("nodes")
        ok = err is not None and nl.get("ok", True) is not False and nodes is not None
        if ok and max_nodes and nodes > max_nodes:
            ok = False
        out.append({"run_id": rec.get("run_id", f.stem), "generation": int(gen), "ok": bool(ok),
                    "mean_rel_err": err, "nodes": nodes, "dofs": nl.get("dofs") or checks.get("dofs"),
                    "error": nl.get("error") or checks.get("failed_gate"), "file": str(f),
                    "curve": None if curve is None else {"u_cross_mm": curve[0].tolist(), "F_N": curve[1].tolist()}})
    return out


def per_generation(runs: list[dict]) -> list[dict]:
    gens = []
    for g in sorted({r["generation"] for r in runs}):
        rs = [r for r in runs if r["generation"] == g]
        good = [r for r in rs if r["ok"]]
        row = {"generation": g, "n_runs": len(rs), "n_ok": len(good)}
        if good:
            b = min(good, key=lambda r: r["mean_rel_err"])
            row |= {"best_err": b["mean_rel_err"], "best_run": b["run_id"], "best_nodes": b["nodes"],
                    "mean_err": float(np.mean([r["mean_rel_err"] for r in good])),
                    "median_nodes": float(np.median([r["nodes"] for r in good]))}
        gens.append(row)
    return gens


def fig_progress(D, gens, out: Path, stem: str = "fig7_finetune_progress"):
    M = D["meshes"]
    fig, ax = plt.subplots()
    floor = M["h20"]["mean_rel_err"] * 100
    ax.axhspan(0, floor, color="#ececec", zorder=0)
    ax.axhline(M["auto"]["mean_rel_err"] * 100, color=mf.C_AUTO, ls="--", lw=2)
    ax.axhline(2.0, color=mf.C_SWARM, ls=":", lw=1.5)
    g = [r for r in gens if "best_err" in r]
    x = [r["generation"] for r in g]
    if g:
        ax.plot(x, [r["best_err"] * 100 for r in g], "-o", color=mf.C_SWARM, lw=3, ms=9, label="best agent")
        ax.plot(x, [r["mean_err"] * 100 for r in g], "--s", color=mf.C_SWARM, lw=2, ms=7, alpha=0.55,
                label="average agent")
        ax.legend(loc="center right")
    xmin = min(x) if x else 0
    xmax = max(x) if x else 1
    ax.plot([xmin - 0.6], [M["swarm"]["mean_rel_err"] * 100], "D", mfc="white", mec=mf.C_SWARM, mew=2.5, ms=11)
    ax.annotate(f"before fine-tuning\n(bc0 mesh) {M['swarm']['mean_rel_err']*100:.1f} %",
                (xmin - 0.6, M["swarm"]["mean_rel_err"] * 100), textcoords="offset points", xytext=(8, 8),
                fontsize=13, color=mf.C_SWARM)
    ax.text(xmax + 0.4, M["auto"]["mean_rel_err"] * 100, f"automesher {M['auto']['mean_rel_err']*100:.1f} %",
            ha="right", va="bottom", fontsize=13, color="#555555")
    ax.text(xmax + 0.4, 2.0, "target 2 %", ha="right", va="bottom", fontsize=13, color=mf.C_SWARM)
    ax.text(xmax + 0.4, floor / 2, f"model-error floor {floor:.1f} %", ha="right", va="center", fontsize=12,
            color="#555555")
    for r in gens:
        if r["n_ok"] < r["n_runs"]:
            ax.text(r["generation"], 0.15, f"{r['n_runs'] - r['n_ok']} failed", ha="center", fontsize=11,
                    color="#888888")
    ax.set_xlim(xmin - 1, xmax + 0.5)
    ax.set_xticks(sorted({r["generation"] for r in gens}) or [0])
    ymax = max([M["auto"]["mean_rel_err"] * 100] + [r["mean_err"] * 100 for r in g]) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xlabel("generation of fine-tuning on the real test")
    ax.set_ylabel("mean error vs the real test (%)")
    if not g:
        ax.text(0.5, 0.55, "TO BE FILLED BY THE FINE-TUNING RUN\n(plot_finetune_run.py ace_runs/<run>)",
                transform=ax.transAxes, ha="center", va="center", fontsize=20, color=mf.C_SWARM,
                fontweight="bold", alpha=0.8)
    if g:
        last = g[-1]
        title = (f"Fine-tuning on the real test: best mesh {M['swarm']['mean_rel_err']*100:.1f} % → "
                 f"{last['best_err']*100:.1f} % by generation {last['generation']}")
    else:
        title = "SLOT: fine-tuning the swarm on the real test (run pending)"
    mf.titled(fig, title, "error = mean |F_sim - F_test| / F_test at the 5 checkpoints, frozen setup; best and "
                          "average agent per generation (valid runs only)")
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--max-nodes", type=int, default=None, help="runs above this node count do not count")
    a = ap.parse_args()
    data_file = HERE / "act3_data.json"
    if not data_file.exists():
        sys.exit("act3_data.json missing: run make_figures.py first")
    D = json.loads(data_file.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    mf.style()
    runs = load_runs(a.run_dir, D, a.max_nodes)
    gens = per_generation(runs)
    print(f"{len(runs)} runs of {PID} in {a.run_dir}, {sum(r['ok'] for r in runs)} scored")
    for r in gens:
        print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    if not any(r["ok"] for r in runs):      # nothing scored yet: only the marked slot figure
        fig_progress(D, gens, a.out, stem="fig7_finetune_progress_STUB")
        print("no scored runs: wrote fig7_finetune_progress_STUB only")
        return
    fig_progress(D, gens, a.out)

    # figure 4 + each generation's best mesh
    extra = [{"nodes": r["best_nodes"], "mean_rel_err": r["best_err"], "label": f"g{r['generation']}"}
             for r in gens if "best_err" in r]
    fig, ax = plt.subplots()
    mf.draw_error_vs_nodes(ax, D, extra=extra)
    if extra:
        b = min(extra, key=lambda p: p["mean_rel_err"])
        title = f"Fine-tuned swarm: {b['mean_rel_err']*100:.1f} % at {b['nodes']:,} nodes ({b['label']})"
    else:
        title = "Fine-tuned swarm: no scored runs yet"
    mf.titled(fig, title, "filled diamonds: best mesh of each fine-tuning generation; frozen setup, error vs the "
                          "Instron curve at the 5 checkpoints")
    fig.savefig(a.out / "fig4b_error_vs_nodes_finetuned.png", dpi=200)
    fig.savefig(a.out / "fig4b_error_vs_nodes_finetuned.pdf")
    plt.close(fig)

    # figure 5 + the overall best run's curve
    with_curve = [r for r in runs if r["ok"] and r["curve"]]
    fig, ax = plt.subplots()
    best = min(with_curve, key=lambda r: r["mean_rel_err"]) if with_curve else None
    mf.draw_curves(ax, D, extra=[{"u_cross_mm": best["curve"]["u_cross_mm"], "F_N": best["curve"]["F_N"],
                                  "label": f"fine-tuned g{best['generation']}, {best['nodes']:,} nodes: "
                                           f"{best['mean_rel_err']*100:.1f} %"}] if best else None)
    mf.titled(fig, f"The fine-tuned mesh tracks the real test: {best['mean_rel_err']*100:.1f} % at "
                   f"{best['nodes']:,} nodes" if best else "Fine-tuned curve: no run with a curve yet",
              "frozen setup; dash-dot = the best fine-tuned run that has a saved curve; dots = the 5 scoring "
              "checkpoints")
    fig.savefig(a.out / "fig5b_curves_finetuned.png", dpi=200)
    fig.savefig(a.out / "fig5b_curves_finetuned.pdf")
    plt.close(fig)

    (a.out / "finetune_data.json").write_text(json.dumps(
        {"run_dir": str(a.run_dir), "max_nodes": a.max_nodes, "generations": gens,
         "runs": [{k: v for k, v in r.items() if k != "curve"} | {"has_curve": r["curve"] is not None} for r in runs]},
        indent=1))
    print("wrote fig7_finetune_progress, fig4b_error_vs_nodes_finetuned, fig5b_curves_finetuned, finetune_data.json"
          f" to {a.out}")


if __name__ == "__main__":
    main()
