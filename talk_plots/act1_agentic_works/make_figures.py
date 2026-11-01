"""Act 1 figures: agentic meshing beats the gmsh automesher, and training improves it.

Reads only raw run files (ace_runs/<run>/gen*/runs/*.json "checks"; ace_runs/<run>/cost.jsonl) and
automesher_sweep.json (made by automesher_sweep.py in this folder). Writes, in this folder only:
fig*.png / fig*.pdf, fig*.md (caption, numbers, sources, caveats), act1_runs.csv (every agent run
used), act1_numbers.json (every number quoted).

    .venv/bin/python talk_plots/act1_agentic_works/make_figures.py
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS_DIR = ROOT / "ace_runs"

# Field-mode runs (scored against the automesher). train1 / cook_field are point-mode (scored against
# the cheapest in-tolerance run, not the automesher), so they cannot answer "vs the automesher".
RUNS = {
    "deep1": "deep loop (Opus teacher), 4 agents/task",
    "cook_auto": "basic loop, 6 agents",
    "split1": "basic loop, 3 agents/task",
    "le11": "basic loop, 3 agents/task",
}
PROBLEMS = ["cooks_membrane", "nafems_le1", "nafems_le11_axisym", "nafems_le11_3d"]
PNAME = {"cooks_membrane": "Cook's membrane", "nafems_le1": "NAFEMS LE1",
         "nafems_le11_axisym": "NAFEMS LE11 (axisym.)", "nafems_le11_3d": "NAFEMS LE11 (3D)"}
RUN_COLOR = {"deep1": "#6a3fb5", "cook_auto": "#1f8fb0", "split1": "#b08a1e", "le11": "#b08a1e"}
AUTO_C, UNIF_C = "#888888", "#6b6b6b"
AUTO_SCORE = 0.25      # the automesher's own score, 1 / AUTO_GAIN_TARGET^2 (ace/config.py)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 13, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------- data

def load_runs() -> list[dict]:
    rows = []
    for run in RUNS:
        for f in sorted(glob.glob(str(RUNS_DIR / run / "gen*" / "runs" / "*.json"))):
            d = json.load(open(f))
            ch = d["checks"]
            base = ch["base_score"] if "base_score" in ch else ch["score"] - ch.get("best_bonus", 0.0)
            rows.append({"run": run, "problem": d["problem_id"], "gen": d["generation"], "run_id": d["run_id"],
                         "agent": d["run_id"].rsplit("-", 1)[-1], "valid": bool(ch["valid"]),
                         "base_score": base, "score": ch["score"], "gain": ch["gain"],
                         "error": ch["rel_error"], "field_error": ch["field_error"],
                         "probe_error": ch["probe_error"], "dofs": ch["dofs"],
                         "auto_error": ch["auto_error"], "auto_dofs": ch["auto_dofs"],
                         "dof_ratio": ch["dofs"] / ch["auto_dofs"],
                         "source": str(Path(f).relative_to(ROOT))})
    return rows


def per_gen(rows, run, prob):
    g = defaultdict(list)
    for r in rows:
        if r["run"] == run and r["problem"] == prob:
            g[r["gen"]].append(r)
    gens = sorted(g)
    return gens, [max(x["base_score"] for x in g[k]) for k in gens], [float(np.mean([x["base_score"] for x in g[k]])) for k in gens], g


def run_cost(run):
    tot = 0.0
    for line in open(RUNS_DIR / run / "cost.jsonl"):
        try:
            tot += float(json.loads(line).get("usd", 0.0))
        except (ValueError, AttributeError):
            pass
    return tot


def sweep_fns(sweep, prob):
    rows = sorted(sweep[prob]["rows"], key=lambda r: r["dofs"])
    ld = np.log([r["dofs"] for r in rows])
    le = np.log([r["error"] for r in rows])

    def err_at(dofs):         # uniform automesher error at these DOFs (log-log interpolation)
        x = np.log(dofs)
        return float(np.exp(np.interp(x, ld, le))), bool(x < ld[0] or x > ld[-1])

    def dofs_for(err):        # uniform automesher DOFs to reach this error
        y = np.log(err)
        return float(np.exp(np.interp(y, le[::-1], ld[::-1]))), bool(y > le[0] or y < le[-1])
    return rows, err_at, dofs_for


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)


def titles(fig, title, sub):
    fig.suptitle(title, fontsize=19, x=0.01, ha="left", y=0.985, fontweight="bold")
    fig.text(0.01, 0.915, sub, fontsize=14, color="#444444", ha="left")


def fmt(x, n=2):
    return f"{x:.{n}f}"


# --------------------------------------------------------------------------- numbers

def compute(rows, sweep):
    out = {"problems": {}, "runs": {}}
    for run in RUNS:
        out["runs"][run] = {"label": RUNS[run], "usd": round(run_cost(run), 2),
                            "n_runs": sum(r["run"] == run for r in rows),
                            "n_invalid": sum(r["run"] == run and not r["valid"] for r in rows)}
    for prob in PROBLEMS:
        pr = [r for r in rows if r["problem"] == prob and r["valid"]]
        best = max(pr, key=lambda r: (r["base_score"], -r["gen"]))
        srows, err_at, dofs_for = sweep_fns(sweep, prob)
        e_u, ex1 = err_at(best["dofs"])
        n_u, ex2 = dofs_for(best["error"])
        maxgain = max(pr, key=lambda r: r["gain"])
        per_run = {}
        for run in RUNS:
            gens, bests, means, g = per_gen(rows, run, prob)
            if not gens:
                continue
            last = gens[-1]
            per_run[run] = {
                "generations": len(gens), "agents_per_gen": len(g[0]),
                "gen0_best": bests[0], "gen0_mean": means[0],
                "last_gen": last, "last_best": bests[-1], "last_mean": means[-1],
                "best_ever": max(bests), "best_ever_gen": gens[int(np.argmax(bests))],
                "last_mean_gain": float(np.mean([x["gain"] for x in g[last]])),
                "last_mean_dof_ratio": float(np.mean([x["dof_ratio"] for x in g[last]])),
                "gen0_mean_gain": float(np.mean([x["gain"] for x in g[0]])),
                "best_per_gen": bests, "mean_per_gen": means, "gens": gens,
            }
        out["problems"][prob] = {
            "automesher": {"error": best["auto_error"], "dofs": best["auto_dofs"], "score": AUTO_SCORE},
            "best_run": {k: best[k] for k in ("run", "run_id", "gen", "base_score", "gain", "error", "field_error",
                                              "probe_error", "dofs", "dof_ratio", "source")},
            "best_score_over_automesher": best["base_score"] / AUTO_SCORE,
            "uniform_automesher_error_at_best_dofs": e_u, "uniform_error_extrapolated": ex1,
            "error_reduction_at_equal_dofs": e_u / best["error"],
            "uniform_automesher_dofs_for_best_error": n_u, "uniform_dofs_extrapolated": ex2,
            "dof_saving_at_equal_error": n_u / best["dofs"],
            "max_gain_run": {k: maxgain[k] for k in ("run", "run_id", "gen", "gain", "error", "dofs", "dof_ratio", "base_score")},
            "per_run": per_run,
        }
    return out


# --------------------------------------------------------------------------- figures

def gen_axis(ax, gens):
    ax.set_xlabel("generation (training step)")
    ax.set_xticks(gens if len(gens) <= 12 else gens[::2])


def fig1(rows, N):
    """deep1: best and mean base score per generation, Cook and LE11 axisym."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), sharey=True)
    c = RUN_COLOR["deep1"]
    data = {}
    for ax, prob in zip(axes, ["cooks_membrane", "nafems_le11_axisym"]):
        gens, bests, means, _ = per_gen(rows, "deep1", prob)
        data[prob] = {"gen": gens, "best": bests, "mean": means}
        ax.axhline(AUTO_SCORE, color=AUTO_C, ls="--", lw=2)
        ax.text(gens[-1], AUTO_SCORE - 0.03, "automesher 0.25", color="#555555", ha="right", va="top", fontsize=13)
        ax.plot(gens, bests, "-o", color=c, lw=2.5, ms=8, label="best agent")
        ax.plot(gens, means, "--", color=c, lw=2, alpha=0.55, label="mean agent")
        ax.text(gens[-1] + 0.2, bests[-1] + 0.02, f"best {bests[-1]:.2f}", color=c, fontsize=13, va="bottom", ha="right")
        ax.text(gens[-1] + 0.2, means[-1] - 0.07, f"mean {means[-1]:.2f}", color=c, alpha=0.8, fontsize=13, va="top", ha="right")
        ax.set_title(PNAME[prob], loc="left", fontsize=16)
        gen_axis(ax, gens)
        ax.set_ylim(-0.03, 1.25)
    axes[0].set_ylabel("base score (automesher = 0.25)")
    axes[0].legend(loc="center right", frameon=False)
    p = N["problems"]
    ck, le = p["cooks_membrane"]["per_run"]["deep1"], p["nafems_le11_axisym"]["per_run"]["deep1"]
    axes[1].annotate("gen 0: every agent had the\nsame temperature-field bug\n(setup, not mesh)",
                     xy=(0, le["gen0_mean"]), xytext=(1.6, 0.28), fontsize=12, color="#444444",
                     arrowprops=dict(arrowstyle="->", color="#777777"))
    titles(fig, f"Training works: the mean agent climbs {ck['gen0_mean'] / AUTO_SCORE:.1f}x \u2192 {ck['last_mean'] / AUTO_SCORE:.1f}x the automesher (Cook)",
           f"deep1: 8 Sonnet agents (4 per task) + Opus teacher, field-error score; best and mean agent per generation")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98, wspace=0.08)
    save(fig, "fig1_learning_curves")
    return data


def fig1b(rows, N):
    """Every field-mode run on every problem: best (solid) and mean (dashed)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.75), sharex=False)
    data = {}
    for ax, prob in zip(axes.ravel(), PROBLEMS):
        ax.axhline(AUTO_SCORE, color=AUTO_C, ls="--", lw=1.8)
        for run in RUNS:
            gens, bests, means, _ = per_gen(rows, run, prob)
            if not gens:
                continue
            data[f"{run}/{prob}"] = {"gen": gens, "best": bests, "mean": means}
            c = RUN_COLOR[run]
            ax.plot(gens, bests, "-o", color=c, lw=2, ms=5)
            ax.plot(gens, means, "--", color=c, lw=1.5, alpha=0.5)
            dy = {"cook_auto": -0.09, "split1": 0.05}.get(run, 0) if prob == "cooks_membrane" else 0
            ax.text(gens[-1] + 0.4, bests[-1] + dy, run, color=c, fontsize=12, va="center")
        ax.set_title(PNAME[prob], loc="left", fontsize=15)
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(-0.03, 1.2)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    for ax in axes[:, 0]:
        ax.set_ylabel("base score")
    titles(fig, "The deep loop learns steadily; the basic loop plateaus or drifts",
           "every field-scored run: best agent (solid, dots) and mean agent (dashed); grey dashed = automesher (0.25)")
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.07, right=0.98, hspace=0.42, wspace=0.12)
    save(fig, "fig1b_learning_curves_all_runs")
    return data


def fig2(rows, N, sweep):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.75))
    data = {}
    for ax, prob in zip(axes.ravel(), PROBLEMS):
        P = N["problems"][prob]
        srows, _, _ = sweep_fns(sweep, prob)
        ax.grid(True, which="major", axis="both", color="#eeeeee")
        ax.plot([r["dofs"] for r in srows], [r["error"] for r in srows], "-s", color=UNIF_C, lw=1.8, ms=4)
        pr = [r for r in rows if r["problem"] == prob and r["valid"]]
        for run in RUNS:
            rr = [r for r in pr if r["run"] == run]
            c = RUN_COLOR[run]
            late = [r for r in rr if r["gen"] > 0]
            early = [r for r in rr if r["gen"] == 0]
            ax.scatter([r["dofs"] for r in late], [r["error"] for r in late], s=22, color=c, alpha=0.45, lw=0)
            ax.scatter([r["dofs"] for r in early], [r["error"] for r in early], s=30, facecolor="white",
                       edgecolor=c, lw=1.3)
        a = P["automesher"]
        ax.scatter([a["dofs"]], [a["error"]], marker="X", s=160, color=AUTO_C, edgecolor="black", lw=0.8, zorder=5)
        b = P["best_run"]
        ax.scatter([b["dofs"]], [b["error"]], marker="*", s=380, color=RUN_COLOR[b["run"]], edgecolor="black",
                   lw=0.8, zorder=6)
        bad = [r for r in pr if r["error"] > 0.2]
        if bad:
            why = "wrong temperature field" if prob == "nafems_le11_axisym" else "wrong supports"
            ax.text(0.02, 0.8, f"top: {len(bad)} runs with {why}\n(setup error, not mesh)", transform=ax.transAxes,
                    fontsize=10, color="#555555", va="top", ha="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10))
            axis.set_minor_formatter(NullFormatter())
        fx = lambda v: f"{v:.2f}" if v < 1.5 else f"{v:.1f}"
        ax.set_title(f"{PNAME[prob]}\nbest run: {fx(P['error_reduction_at_equal_dofs'])}x less error, "
                     f"{fx(P['dof_saving_at_equal_error'])}x fewer DOFs", loc="left", fontsize=12)
        data[prob] = {"uniform": srows, "automesher": a, "best": b,
                      "agents": [{k: r[k] for k in ("run", "run_id", "gen", "dofs", "error")} for r in pr]}
    for ax in axes[1]:
        ax.set_xlabel("DOFs")
    for ax in axes[:, 0]:
        ax.set_ylabel("scored error")
    h = [Line2D([], [], color=UNIF_C, marker="s", ms=4, lw=1.8),
         Line2D([], [], marker="X", ls="", ms=11, color=AUTO_C, markeredgecolor="black"),
         Line2D([], [], marker="o", ls="", ms=6, color=RUN_COLOR["deep1"], alpha=0.6),
         Line2D([], [], marker="o", ls="", ms=6, color=RUN_COLOR["cook_auto"], alpha=0.6),
         Line2D([], [], marker="o", ls="", ms=6, color=RUN_COLOR["split1"], alpha=0.6),
         Line2D([], [], marker="o", ls="", ms=6, markerfacecolor="white", markeredgecolor="#444444"),
         Line2D([], [], marker="*", ls="", ms=16, color="#bbbbbb", markeredgecolor="black")]
    lab = ["automesher refined\nuniformly", "automesher\n(gmsh defaults)", "deep1 agents", "cook_auto agents",
           "split1 / le11 agents", "open = generation 0", "best-scoring run"]
    fig.legend(h, lab, loc="center right", frameon=False, fontsize=12, bbox_to_anchor=(1.0, 0.5))
    titles(fig, "2D: the best agents beat the automesher's whole cost curve; 3D: a tie",
           "every valid agent run in the field-scored runs; error = max(energy-norm field error, probe-point RMS)")
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.07, right=0.77, hspace=0.6, wspace=0.2)
    save(fig, "fig2_error_vs_dofs")
    return data


def fig3(N):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    cats = [("automesher", AUTO_C), ("gen-0 agents (mean)", "#cdbfe6"), ("last gen (mean)", "#9b7fd1"),
            ("best agent", "#6a3fb5")]
    x = np.arange(len(PROBLEMS))
    w = 0.2
    data = {}
    for i, prob in enumerate(PROBLEMS):
        P = N["problems"][prob]
        R = P["per_run"][P["best_run"]["run"]]
        vals = [AUTO_SCORE, R["gen0_mean"], R["last_mean"], P["best_run"]["base_score"]]
        data[prob] = dict(zip([c for c, _ in cats], vals), run=P["best_run"]["run"])
        for j, ((name, col), v) in enumerate(zip(cats, vals)):
            ax.bar(x[i] + (j - 1.5) * w, v, w * 0.92, color=col, label=name if i == 0 else None,
                   hatch="//" if j == 0 else None, edgecolor="white" if j else "#666666", lw=0)
            ax.text(x[i] + (j - 1.5) * w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
        ax.text(x[i] + 1.5 * w, P["best_run"]["base_score"] + 0.08, f"{P['best_score_over_automesher']:.1f}x",
                ha="center", fontsize=14, fontweight="bold", color="#6a3fb5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PNAME[p]}\n({N['problems'][p]['best_run']['run']})" for p in PROBLEMS])
    ax.set_ylabel("base score (automesher = 0.25)")
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    lo = min(N["problems"][p]["best_score_over_automesher"] for p in PROBLEMS)
    hi = max(N["problems"][p]["best_score_over_automesher"] for p in PROBLEMS)
    titles(fig, f"The best agent scores {lo:.1f}-{hi:.1f}x the automesher on every benchmark",
           "base score (no bonus): automesher, gen-0 mean, last-gen mean, best agent of the run named below")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.98)
    save(fig, "fig3_headline_scores")
    return data


def fig3b(N):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    x = np.arange(len(PROBLEMS))
    w = 0.36
    data = {}
    for i, prob in enumerate(PROBLEMS):
        P = N["problems"][prob]
        v1, v2 = P["error_reduction_at_equal_dofs"], P["dof_saving_at_equal_error"]
        data[prob] = {"error_reduction_at_equal_dofs": v1, "dof_saving_at_equal_error": v2,
                      "best_run": P["best_run"]["run_id"], "run": P["best_run"]["run"]}
        ax.bar(x[i] - w / 2, v1, w * 0.94, color="#9b7fd1", label="lower error at the same DOFs" if i == 0 else None)
        ax.bar(x[i] + w / 2, v2, w * 0.94, color="#6a3fb5", label="fewer DOFs for the same error" if i == 0 else None)
        ax.text(x[i] - w / 2, v1 + 0.15, f"{v1:.2f}x" if v1 < 1.5 else f"{v1:.1f}x", ha="center", fontsize=14)
        ax.text(x[i] + w / 2, v2 + 0.15, (f"{v2:.2f}x" if v2 < 1.5 else f"{v2:.1f}x") + ("*" if P["uniform_dofs_extrapolated"] else ""), ha="center",
                fontsize=14, fontweight="bold")
    ax.axhline(1, color=AUTO_C, ls="--", lw=2, label="automesher = 1x")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PNAME[p]}\n{N['problems'][p]['best_run']['run_id']}" for p in PROBLEMS], fontsize=12)
    ax.set_ylabel("factor vs automesher refined uniformly")
    ax.legend(loc="upper right", frameon=False)
    ck = N["problems"]["cooks_membrane"]
    titles(fig, f"Matching the best agent costs the automesher up to {max(N['problems'][p]['dof_saving_at_equal_error'] for p in PROBLEMS):.0f}x the DOFs",
           "best-scoring agent run vs gmsh's default mesh refined uniformly (size-factor sweep, log-log interpolated)")
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.08, right=0.98)
    save(fig, "fig3b_equal_cost_equal_accuracy")
    return data


def fig4(rows, N):
    """deep1 Cook: a gen-0 mesh (automesher-level accuracy) next to the best gen-10 mesh (existing PNGs)."""
    by_id = {r["run_id"]: r for r in rows if r["run"] == "deep1"}
    left, right = by_id["g0-cooks_membrane-a1"], by_id["g10-cooks_membrane-a7"]
    imgs = [RUNS_DIR / "deep1/gen0/reports/g0-cooks_membrane-a1_mesh.png",
            RUNS_DIR / "deep1/gen10/reports/g10-cooks_membrane-a7_mesh.png"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    a = N["problems"]["cooks_membrane"]["automesher"]
    for ax, img, r, head in zip(axes, imgs, [left, right], ["Generation 0 agent", "Generation 10 agent (best)"]):
        im = mpimg.imread(img)
        ax.imshow(im[40:, :584])           # drop the PNG's own title and legend; keep the mesh
        ax.axis("off")
        ax.set_title(f"{head}\n{r['dofs']} DOFs, error {100 * r['error']:.2f}%: {r['gain']:.2f}x automesher accuracy",
                     fontsize=14, loc="left")
    titles(fig, "Same part, same solver: the trained agent grades the mesh into the corners",
           f"Cook's membrane, deep1 meshes (report PNGs); automesher: {a['dofs']} DOFs, error {100 * a['error']:.2f}%")
    fig.subplots_adjust(top=0.82, bottom=0.02, left=0.02, right=0.98, wspace=0.05)
    save(fig, "fig4_mesh_before_after")
    return {"left": {k: left[k] for k in ("run_id", "dofs", "error", "gain", "base_score", "source")},
            "right": {k: right[k] for k in ("run_id", "dofs", "error", "gain", "base_score", "source")},
            "images": [str(p.relative_to(ROOT)) for p in imgs], "automesher": a}


# --------------------------------------------------------------------------- main

def main():
    rows = load_runs()
    sweep = json.loads((HERE / "automesher_sweep.json").read_text())
    assert all(sweep[p]["factor1_matches_cache"] for p in PROBLEMS)
    N = compute(rows, sweep)
    plotted = {"fig1_learning_curves": fig1(rows, N), "fig1b_learning_curves_all_runs": fig1b(rows, N),
               "fig2_error_vs_dofs": fig2(rows, N, sweep), "fig3_headline_scores": fig3(N),
               "fig3b_equal_cost_equal_accuracy": fig3b(N), "fig4_mesh_before_after": fig4(rows, N)}
    with open(HERE / "act1_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "act1_numbers.json").write_text(json.dumps({"numbers": N, "plotted": plotted}, indent=1, default=float))
    print(json.dumps({p: {k: N["problems"][p][k] for k in ("best_score_over_automesher", "error_reduction_at_equal_dofs",
                                                         "dof_saving_at_equal_error", "uniform_dofs_extrapolated")}
                      | {"best": N["problems"][p]["best_run"]["run_id"], "gain": N["problems"][p]["best_run"]["gain"],
                         "dof_ratio": N["problems"][p]["best_run"]["dof_ratio"]} for p in PROBLEMS}, indent=1))
    print({r: v for r, v in N["runs"].items()})


if __name__ == "__main__":
    main()
