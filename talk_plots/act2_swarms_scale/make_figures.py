"""Act 2 figures: swarms train better and more stably, and this scales.

Reads only the raw run files (never the precomputed results/ tables, which are used as a cross-check):
  ace_runs/{div_s1,div_s2,div_s4,div_s8}/gen*/runs/g{g}-{task}-*.json   (afternoon, diversity mechanisms on)
  ace_runs/{scale_s1,scale_s1_opus,scale_s2,scale_s4,scale_s8}/gen*/runs/...   (morning)
  ace_runs/<run>/cost.jsonl                                             ("usd" per API call)

Score = the run's checks["base_score"] (field score without the be-the-best / diversity bonuses;
ace/config.py ~line 85: g = e_auto / e, c = DOFs / DOFs_auto, A = (g/2)^2 below g = 2 else g/2,
score = A / c; the automesher scores 0.25). It is recomputed here from rel_error, auto_error, dofs and
auto_dofs and checked against the stored value. An invalid run scores 0 (score = valid * ...).

Run:  .venv/bin/python talk_plots/act2_swarms_scale/make_figures.py
Writes PNG (200 dpi) + PDF + <stem>.md per figure, data.json and runs.csv, all in this folder.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "ace_runs"

TASKS = ["abc_00000065_bc0", "abc_00000003_bc7"]          # left panel = the task with room to learn
TLAB = {"abc_00000065_bc0": "abc_00000065_bc0", "abc_00000003_bc7": "abc_00000003_bc7"}
COL = {1: "#2a78d6", 2: "#e0882a", 4: "#1e9e6a", 8: "#c23d6e", 16: "#5b3fa0"}
MARK = {1: "D", 2: "s", 4: "^", 8: "o", 16: "v"}                 # secondary encoding (green/pink are close for deutans)
AUTO_COL = "#888888"
AUTO_SCORE = 0.25
DIV = {1: "div_s1", 2: "div_s2", 4: "div_s4", 8: "div_s8"}
SCALE = {1: "scale_s1", 2: "scale_s2", 4: "scale_s4", 8: "scale_s8"}
ALL_RUNS = list(DIV.values()) + list(SCALE.values()) + ["scale_s1_opus"]
TOL_FLAT = 0.005          # a generation-to-generation change smaller than this counts as "flat"
DIV_WINDOW = range(1, 8)  # generations after learning starts; all four afternoon runs have gens 0-7
SCALE_WINDOW = range(1, 5)  # gens 1-4: the generations every morning swarm (2/4/8) completed
# 16 agents (afternoon setup, Jaime, 2026-11-01): its run folder stays on his laptop, so fig2 reads the
# per-generation best/mean he exported from it (same base score, recomputed from the run files there)
S16 = ROOT / "results" / "diversity_s16_2026-11-01" / "scores.json"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 13, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.facecolor": "white",
})


# ----------------------------------------------------------------------------------------------- data
def recompute_base(c):
    g = c["auto_error"] / c["rel_error"]
    cost = c["dofs"] / c["auto_dofs"]
    acc = (g / 2) ** 2 if g < 2 else g / 2
    return acc / cost


def load_runs(run):
    rows = []
    for f in sorted(glob.glob(str(RUNS / run / "gen*" / "runs" / "g*.json"))):
        d = json.load(open(f))
        c = d["checks"]
        m = re.match(r"g(\d+)-(abc_\d+_bc\d+)-(\w+)\.json", os.path.basename(f))
        valid = bool(c.get("valid"))
        stored = c.get("base_score")
        rows.append({
            "run": run, "gen": int(m.group(1)), "task": m.group(2), "agent": m.group(3), "valid": valid,
            "base_score": float(stored) if valid else 0.0,
            "recomputed_base": recompute_base(c) if valid else 0.0,
            "field_error": c.get("field_error"), "rel_error": c.get("rel_error"), "dofs": c.get("dofs"),
            "auto_error": c.get("auto_error"), "auto_dofs": c.get("auto_dofs"),
            "score_with_bonuses": c.get("score"), "invalid_because": c.get("invalid_because"),
            "file": str(Path(f).relative_to(ROOT)),
        })
    return rows


def per_gen(rows, task):
    by = defaultdict(list)
    for r in rows:
        if r["task"] == task:
            by[r["gen"]].append(r)
    out = []
    for g in sorted(by):
        rs = by[g]
        best = max(rs, key=lambda r: r["base_score"])
        out.append({"gen": g, "n": len(rs), "n_invalid": sum(not r["valid"] for r in rs),
                    "best": best["base_score"], "best_agent": best["agent"], "best_dofs": best["dofs"],
                    "best_rel_error": best["rel_error"],
                    "mean": float(np.mean([r["base_score"] for r in rs]))})
    return out


def costs(run):
    """$ per generation: a generator call carries its generation in run_id ('g3-...'); the reflection,
    teacher and organizer calls that follow (empty run_id) belong to the latest generation started."""
    rows = sorted((json.loads(l) for l in open(RUNS / run / "cost.jsonl")), key=lambda d: d["t"])
    cur, per, role = 0, defaultdict(float), defaultdict(float)
    for d in rows:
        m = re.match(r"g(\d+)-", d.get("run_id") or "")
        if m:
            cur = max(cur, int(m.group(1)))
        per[int(m.group(1)) if m else cur] += d["usd"]
        role[d["role"]] += d["usd"]
    return {"total": sum(d["usd"] for d in rows), "per_gen": dict(sorted(per.items())), "by_role": dict(role)}


def step_stats(series):
    best = np.array([p["best"] for p in series])
    d = np.diff(best)
    return {
        "deltas": [float(x) for x in d],
        "up": int((d > TOL_FLAT).sum()), "down": int((d < -TOL_FLAT).sum()),
        "flat": int((np.abs(d) <= TOL_FLAT).sum()),
        "largest_drop": float(d.min()) if (d < 0).any() else 0.0,
        "largest_drop_step": f"{int(np.argmin(d))}->{int(np.argmin(d)) + 1}" if (d < 0).any() else "-",
        "std_best": float(np.std(best, ddof=1)),
        "net_gain": float(best[-1] - best[0]),
        "first": float(best[0]), "last": float(best[-1]), "max": float(best.max()),
        "gens_below_gen0": int((best[1:] < best[0]).sum()),
    }


def window_mean(series, key, window):
    vals = [p[key] for p in series if p["gen"] in window]
    return float(np.mean(vals)), len(vals)


DATA = {}
RAW = {}
for run in ALL_RUNS:
    rows = load_runs(run)
    RAW[run] = rows
    DATA[run] = {"cost": costs(run), "tasks": {t: per_gen(rows, t) for t in TASKS}}

max_mismatch = max(abs(r["base_score"] - r["recomputed_base"]) for rs in RAW.values() for r in rs)
n_invalid = [(r["run"], r["gen"], r["task"]) for rs in RAW.values() for r in rs if not r["valid"]]


# ------------------------------------------------------------------------------------------- helpers
def takeaway(fig, title, subtitle):
    fig.text(0.015, 0.975, title, ha="left", va="top", fontsize=19, fontweight="bold")
    fig.text(0.015, 0.905, subtitle, ha="left", va="top", fontsize=14, color="#555555")


def automesher(ax, label=True, x=None):
    ax.axhline(AUTO_SCORE, color=AUTO_COL, ls="--", lw=1.8, zorder=1)
    if label:
        xl = ax.get_xlim()
        ax.text(xl[0] + 0.02 * (xl[1] - xl[0]) if x is None else x, AUTO_SCORE, "automesher (0.25)",
                color="#666666", fontsize=13, va="bottom", ha="left")


def save(fig, stem, md):
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    (HERE / f"{stem}.md").write_text(md.strip() + "\n")


def f2(x):
    return f"{x:.3f}"


def series(run, task, key="best"):
    s = DATA[run]["tasks"][task]
    return [p["gen"] for p in s], [p[key] for p in s]


def end_labels(ax, items, min_gap):
    """Direct labels at the right end of lines; nudged apart vertically so they do not collide."""
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y, _, _ in items:
        y = max(y, ys[-1] + min_gap) if ys else y
        ys.append(y)
    for (x, _, text, col), y in zip(items, ys):
        ax.text(x + 0.25, y, text, color="#222222", fontsize=14, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1.5))


SRC_DIV = ("`ace_runs/div_s{1,2,4,8}/gen*/runs/g*-{task}-*.json` (`checks.base_score`, recomputed from "
           "`rel_error`, `auto_error`, `dofs`, `auto_dofs`)")
SRC_SCALE = "`ace_runs/scale_s{1,1_opus,2,4,8}/gen*/runs/g*-{task}-*.json` (same fields)"
COMMON_CAVEATS = (
    "- One run per configuration; the afternoon README estimates run-to-run noise at 0.03-0.1 per generation.\n"
    "- Team size n splits 50/50 over the two tasks, so the per-task best is the max of n/2 runs "
    "(1, 2, 4 for n = 2, 4, 8). Best-of-n includes more draws; the framing used here is that the swarm is "
    "trained and deployed together, so its best mesh is the one used.\n")


# --------------------------------------------------------------------------- fig 1: best per generation
def fig1(include_morning=False):
    stem = "fig1b_best_per_gen_with_morning" if include_morning else "fig1_best_per_gen"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.95, top=0.79, bottom=0.12, wspace=0.22)
    for ax, task in zip(axes, TASKS):
        labels = []
        for n in (2, 4, 8):
            if include_morning:
                x, y = series(SCALE[n], task)
                ax.plot(x, y, color=COL[n], ls="--", lw=1.8, alpha=0.45, marker=MARK[n], ms=6, mfc="white")
            x, y = series(DIV[n], task)
            ax.plot(x, y, color=COL[n], lw=2.6, marker=MARK[n], ms=8, zorder=3)
            labels.append((x[-1], y[-1], f"{n} agents", COL[n]))
        ax.set_xlim(-0.4, 9.6)
        ax.set_xticks(range(8))
        if task == "abc_00000065_bc0":
            ax.set_ylim(0.2, 0.8)
            end_labels(ax, labels, 0.045)
        else:
            ax.set_ylim(0.12, 0.27)
            end_labels(ax, labels, 0.011)
        automesher(ax)
        ax.set_title(TLAB[task], loc="left", fontsize=16)
        ax.set_xlabel("generation")
    axes[0].set_ylabel("best agent's base score")
    if include_morning:
        axes[1].text(0.98, 0.04, "solid: afternoon (diversity on, 8 gens)\ndashed, faded: morning (stopped at 3-6 gens)",
                     transform=axes[1].transAxes, ha="right", va="bottom", fontsize=12, color="#555555")
    s8 = series(DIV[8], TASKS[0])[1]
    s4 = series(DIV[4], TASKS[0])[1]
    s2 = series(DIV[2], TASKS[0])[1]
    takeaway(fig, "On 65_bc0, 8 agents improve almost every generation; 2 agents stall",
             "Best agent's base score per generation, afternoon runs (diversity on); higher is better, automesher = 0.25"
             )
    st = {n: step_stats(DATA[DIV[n]]["tasks"][TASKS[0]]) for n in (2, 4, 8)}
    sb = {n: step_stats(DATA[DIV[n]]["tasks"][TASKS[1]]) for n in (2, 4, 8)}
    bc7max = max(p["best"] for run in ALL_RUNS for p in DATA[run]["tasks"]["abc_00000003_bc7"])
    md = f"""
# {stem}: on 65_bc0, 8 agents improve almost every generation; 2 agents stall

**Caption.** Best agent's base score in each generation for swarms of 2, 4 and 8 Sonnet agents (afternoon
runs, diversity mechanisms on){'; morning runs dashed and faded' if include_morning else ''}. On abc_00000065_bc0
the 8-agent swarm starts highest and climbs to about 3x the automesher; on abc_00000003_bc7 no team beats the automesher.

**Key numbers (afternoon, best per generation, gens 0 -> 7).**
- abc_00000065_bc0: 8 agents {f2(s8[0])} -> {f2(s8[-1])} (max {f2(st[8]['max'])}; steps up/down/flat
  {st[8]['up']}/{st[8]['down']}/{st[8]['flat']}, largest drop {f2(st[8]['largest_drop'])});
  4 agents {f2(s4[0])} -> {f2(s4[-1])} (up/down/flat {st[4]['up']}/{st[4]['down']}/{st[4]['flat']}, largest drop {f2(st[4]['largest_drop'])});
  2 agents {f2(s2[0])} -> {f2(s2[-1])} (up/down/flat {st[2]['up']}/{st[2]['down']}/{st[2]['flat']}, largest drop
  {f2(st[2]['largest_drop'])}, low point {f2(min(s2))} in gen {s2.index(min(s2))}).
- abc_00000003_bc7: 8 agents {f2(sb[8]['first'])} -> {f2(sb[8]['last'])} (max {f2(sb[8]['max'])}), 4 agents
  {f2(sb[4]['first'])} -> {f2(sb[4]['last'])}, 2 agents {f2(sb[2]['first'])} -> {f2(sb[2]['last'])}.
  Highest bc7 base score in any of the nine runs: {f2(bc7max)} (< 0.25).
- Step counts use a flat band of +-{TOL_FLAT}.

**Data.** {SRC_DIV}{'; ' + SRC_SCALE if include_morning else ''}. Plotted values: `data.json` -> `runs.<run>.tasks.<task>`.

**Caveats.**
{COMMON_CAVEATS}- abc_00000003_bc7: every team stays below the automesher (best {f2(bc7max)}); the differences between team sizes'
  mean-of-best (gens 1-7) are about 0.02-0.04 and the 8 > 4 > 2 order is already there in gen 0, before any learning.
- "Almost every generation" means {st[8]['up']} of 7 steps up on abc_00000065_bc0 for 8 agents; the 4-agent swarm has
  nearly as many up-steps ({st[4]['up']} vs {st[8]['up']}) and a larger net gain ({f2(st[4]['net_gain'])} vs {f2(st[8]['net_gain'])}), but it
  sits around 0.56-0.64 for gens 0-5 and only catches up in gens 6-7.
"""
    save(fig, stem, md)


# ------------------------------------------------------------ fig 2: mean of per-gen best, grouped bars
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.12, wspace=0.08)
    vals = {}
    w = 0.2
    s16 = json.load(open(S16))["div_s16"]
    for ax, task in zip(axes, TASKS):
        for gi, (study, runs, window, sizes) in enumerate(
                [("div", DIV, DIV_WINDOW, (2, 4, 8, 16)), ("scale", SCALE, SCALE_WINDOW, (2, 4, 8))]):
            for k, n in enumerate(sizes):
                if n == 16:
                    v = float(np.mean([b for g, b in enumerate(s16[task]["best"]) if g in window]))
                else:
                    v, cnt = window_mean(DATA[runs[n]]["tasks"][task], "best", window)
                vals[(study, task, n)] = v
                x = gi + 0.18 * (study == "div") + (k - (len(sizes) - 1) / 2) * (w + 0.02)
                ax.bar(x, v, width=w, color=COL[n], alpha=1.0 if study == "div" else 0.55,
                       hatch=None if study == "div" else "//", edgecolor="white", lw=0)
                ax.text(x, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=11.5)
                ax.text(x, 0.012, str(n), ha="center", va="bottom", fontsize=13, color="white", fontweight="bold")
        ax.set_xticks([0.18, 1])
        ax.set_xticklabels(["afternoon\n(diversity on, gens 1-7)", "morning\n(gens 1-4)"])
        ax.set_xlim(-0.45, 1.5)
        ax.set_ylim(0, 0.85)
        ax.axhline(AUTO_SCORE, color=AUTO_COL, ls="--", lw=1.8, zorder=1)
        ax.set_title(TLAB[task], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean over generations of\nthe best agent's base score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[n]) for n in (2, 4, 8, 16)]
    handles.append(plt.Line2D([], [], color=AUTO_COL, ls="--", lw=1.8))
    axes[1].legend(handles, ["2 agents", "4 agents", "8 agents", "16 agents", "automesher (0.25)"], loc="upper right",
                   frameon=False)
    order_ok = {(s, t): vals[(s, t, 8)] > vals[(s, t, 4)] > vals[(s, t, 2)] for s in ("div", "scale") for t in TASKS}
    order16 = {t: vals[("div", t, 16)] > vals[("div", t, 8)] for t in TASKS}
    takeaway(fig, "More agents, better swarm: 16 > 8 > 4 > 2" if all(order_ok.values()) and all(order16.values())
             else "Team-size ordering is not monotone in every study",
             "Mean over generations (after gen 0) of the best agent's base score; numbers in bars = team size")
    # the morning study's own windows (as in its README), for the cross-check
    own = {n: {t: window_mean(DATA[SCALE[n]]["tasks"][t], "best", range(1, 99))[0] for t in TASKS} for n in (2, 4, 8)}
    lines = "\n".join(
        f"| {s} | {TLAB[t]} | {f2(vals[(s, t, 2)])} | {f2(vals[(s, t, 4)])} | {f2(vals[(s, t, 8)])} | "
        f"{'yes' if order_ok[(s, t)] else 'NO'} |" for s in ("div", "scale") for t in TASKS)
    md = f"""
# fig2_mean_best_by_team_size: 16 > 8 > 4 > 2

**Caption.** For each study and task, the best agent's base score averaged over the generations after gen 0.
The ordering 8 > 4 > 2 agents holds in {sum(order_ok.values())} of 4 cases. The 16-agent swarm (afternoon setup only)
beats 8 agents on {sum(order16.values())} of 2 tasks: {' and '.join(f"{TLAB[t]} {f2(vals[('div', t, 16)])} vs {f2(vals[('div', t, 8)])}" for t in TASKS)}.

**Key numbers.**

| study | task | 2 agents | 4 agents | 8 agents | 8 > 4 > 2 |
|---|---|---|---|---|---|
{lines}

Windows: afternoon gens 1-7 (all runs have gens 0-7); morning gens 1-4, the window all three morning swarms
completed. With each morning run's own window (2 agents gens 1-5, 4 and 8 agents gens 1-4, as in
`results/scaling_gold_2026-10-31/README.md`) the morning values are: 2 agents {f2(own[2]['abc_00000003_bc7'])} / {f2(own[2]['abc_00000065_bc0'])},
4 agents {f2(own[4]['abc_00000003_bc7'])} / {f2(own[4]['abc_00000065_bc0'])}, 8 agents {f2(own[8]['abc_00000003_bc7'])} / {f2(own[8]['abc_00000065_bc0'])} (bc7 / 65_bc0);
the order is the same.

**Data.** {SRC_DIV}; {SRC_SCALE}; 16 agents: `results/diversity_s16_2026-11-01/scores.json` (per-generation best, exported
by Jaime from `ace_runs/div_s16`, which stays on his laptop). Plotted values: `data.json` -> `fig2`.

**Caveats.**
{COMMON_CAVEATS}- 16 agents: one run, read from its exported per-generation scores rather than the raw run files. Its field
  references were rebuilt for that run with `ace.gold_reference` (same study meshes, last level 3% under the DOF cap,
  commit 603f3e7), so its scores may differ slightly in scale from the 2-8 agent runs. There is no morning 16-agent run.
- The returns diminish: on 65_bc0, 4 -> 8 agents gains {f2(vals[('div', 'abc_00000065_bc0', 8)] - vals[('div', 'abc_00000065_bc0', 4)])}
  and 8 -> 16 gains {f2(vals[('div', 'abc_00000065_bc0', 16)] - vals[('div', 'abc_00000065_bc0', 8)])}, for $79.54 vs $41.45 of API calls.
- Morning 65_bc0 gaps are small (2 / 4 / 8: {f2(vals[('scale', 'abc_00000065_bc0', 2)])} / {f2(vals[('scale', 'abc_00000065_bc0', 4)])} /
  {f2(vals[('scale', 'abc_00000065_bc0', 8)])}), well inside the 0.03-0.1 per-generation noise; and the single highest morning
  65_bc0 score was the 2-agent run's 0.717 (gen 4). The morning bc7 order is consistent, the morning 65_bc0 order is weak.
- On bc7 the bars are all below the automesher's 0.25.
"""
    DATA["fig2"] = {f"{s}|{t}|{n}": vals[(s, t, n)] for (s, t, n) in vals}
    save(fig, "fig2_mean_best_by_team_size", md)
    return order_ok, vals


# --------------------------------------- fig 3: talk version of fig 2, abc_00000065_bc0 only, extrapolated
BC0 = "abc_00000065_bc0"
SIZES = (2, 4, 8, 16)


def bc0_best_per_gen(n):
    """Best agent's base score per generation on bc0, divided by the automesher's (automesher = 1)."""
    if n == 16:
        return np.array(json.load(open(S16))["div_s16"][BC0]["best"]) / AUTO_SCORE
    return np.array([p["best"] for p in DATA[DIV[n]]["tasks"][BC0]]) / AUTO_SCORE


def fig3(stat):
    """stat="max": the best score any agent reached in gens 0-7; stat="avg": the fig2 bar (mean over gens 1-7 of the
    per-generation best). Projection: from the 16-agent bar, y16 + f(n) - f(16), where f = c - d n^-1/2 is fitted to
    the four bars (it flattens; the exponent 1/2 is chosen, as in fig7; a free exponent flattens faster)."""
    a = 0.5
    sat = lambda n, c, d, a: c - d * n ** (-a)
    y = np.array([bc0_best_per_gen(n).max() if stat == "max" else bc0_best_per_gen(n)[1:].mean() for n in SIZES])
    slope, c = np.polyfit(np.array(SIZES, float) ** -a, y, 1)
    d = -slope
    resid = y - sat(np.array(SIZES, float), c, d, a)
    stem = f"fig3_scaling_bc0_{stat}"
    xp = np.array([5.0, 6.0])                                                  # log2 of 32 and 64 agents
    proj = lambda n: y[-1] + sat(n, c, d, a) - sat(16.0, c, d, a)
    yp = proj(2 ** xp)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.13)
    ax.axvspan(4.5, 6.5, color="#f2f2f4", zorder=0, lw=0)
    ax.text(5.5, 4.25, "projected", ha="center", va="top", fontsize=13, color="#666666")
    for xi, yi, n in zip(np.log2(SIZES), y, SIZES):
        ax.bar(xi, yi, width=0.7, color=COL[n], edgecolor="white", lw=0, zorder=2)
        ax.text(xi, yi - 0.15, f"{yi:.2f}x", ha="center", va="top", fontsize=17, color="white", fontweight="bold")
    xf = np.linspace(4, 6, 100)
    ax.plot(xf, proj(2 ** xf), color="#333333", ls="--", lw=2, zorder=3)
    ax.plot(xp, yp, ls="none", marker="o", ms=11, mfc="white", mec="#333333", mew=2, zorder=4)
    for xi, yi in zip(xp, yp):
        ax.text(xi, yi + 0.12, f"~{yi:.2f}x", ha="center", va="bottom", fontsize=15, color="#333333")
    ax.axhline(1, color=AUTO_COL, ls="--", lw=1.8, zorder=1)
    ax.text(4.6, 1.03, "automesher = 1", color="#666666", fontsize=13, va="bottom", ha="left")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.4, 6.6)
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("agents in the swarm")
    ax.set_ylabel("score relative to the automesher")
    ax.grid(axis="x", visible=False)
    fig.text(0.015, 0.975, "More agents, better swarm", ha="left", va="top", fontsize=21, fontweight="bold")
    what = ("the best score any agent reached in gens 0-7" if stat == "max"
            else "the best agent's base score averaged over gens 1-7")
    md = f"""
# {stem}: 16 > 8 > 4 > 2, extrapolated

**Caption.** Talk version of fig2 for {BC0} only, afternoon setup (diversity on), scores divided by the
automesher's 0.25 so that the automesher is 1. Bars: {what}.
Dashed: projection from the 16-agent bar, y16 + f(n) - f(16), with f = c - d n^-1/2 (c = {c:.2f}, d = {d:.2f})
a least-squares fit to the four bars: 32 agents {yp[0]:.2f}x, 64 agents {yp[1]:.2f}x (no swarm of that size was run).

**Key numbers.** 2 / 4 / 8 / 16 agents: {' / '.join(f'{v:.2f}x' for v in y)} the automesher.
Fit residuals: {' / '.join(f'{r:+.2f}' for r in resid)} (the projection is anchored on the measured 16-agent value).

**Data.** As fig2 (16 agents from `results/diversity_s16_2026-11-01/scores.json`). Plotted values: `data.json` -> `{stem}`.

**Caveats.**
- A 2-parameter fit to 4 points, one run each (noise 0.03-0.1 in score, 0.1-0.4 on this axis). The exponent 1/2 is
  the optimistic choice: a free exponent fits these bars with a faster plateau (projections near the 16-agent value).
- The 16-agent run's references were rebuilt (see fig2), and team size n means n/2 agents per task, so a bigger
  team also takes the best of more meshes.
"""
    DATA[stem] = {"sizes": list(SIZES), "rel_to_auto": [float(v) for v in y],
                  "fit": {"c": float(c), "d": float(d), "alpha": float(a)},
                  "projected": {"32": float(yp[0]), "64": float(yp[1])}}
    save(fig, stem, md)


# ------------------------------------------- fig 7: bc0 training curves per team size, extended in generations
def fig7(stat):
    """stat="max": best score so far (running max of the best agent per generation; gen 7 = the fig3 max bar);
    stat="avg": running mean over gens 1..g of the best agent per generation (gen 7 = the fig3 avg bar).
    Projection to gen 15: y_n(g) = a_n + b (g + 1)^-1/2 with one rate b shared by all team sizes (fitted to gens 1-7 of
    the four curves; team size sets only the level a_n), anchored at each team's gen-7 value, so the projected order
    is the measured one. The exponent 1/2 is chosen: a free exponent runs to ~0 (no plateau visible in 8 gens)."""
    G, GL = np.arange(8), np.arange(1, 8)
    stem = f"fig7_training_bc0_{stat}"
    curves = {}
    for n in SIZES:
        raw = bc0_best_per_gen(n)
        curves[n] = (np.maximum.accumulate(raw) if stat == "max"
                     else np.concatenate([raw[:1], np.cumsum(raw[1:]) / np.arange(1, 8)]))
    X = (GL + 1) ** -0.5                                         # shared rate b: within-team least squares
    b = (sum(((X - X.mean()) * (curves[n][1:] - curves[n][1:].mean())).sum() for n in SIZES)
         / (len(SIZES) * ((X - X.mean()) ** 2).sum()))
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.84, top=0.87, bottom=0.13)
    ax.axvspan(7.5, 15.5, color="#f2f2f4", zorder=0, lw=0)
    ax.text(11.5, 3.75, "projected", ha="center", va="top", fontsize=13, color="#666666")
    labels, out = [], {}
    gf = np.linspace(7, 15, 60)
    for n in SIZES:
        y = curves[n]
        fit = y[-1] + b * ((gf + 1) ** -0.5 - 8 ** -0.5)
        ax.plot(gf, fit, color=COL[n], ls="--", lw=2.2, zorder=2)
        ax.plot(G, y, color=COL[n], lw=2.8, marker=MARK[n], ms=8, zorder=3 + n / 100)
        labels.append((15, fit[-1], f"{n} agents", COL[n]))
        out[n] = {"measured": [float(v) for v in y], "gen15": float(fit[-1])}
    end_labels(ax, labels, 0.15)
    ax.axhline(1, color=AUTO_COL, ls="--", lw=1.8, zorder=1)
    ax.text(0.1, 1.03, "automesher = 1", color="#666666", fontsize=13, va="bottom", ha="left")
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0.9, 3.8)
    ax.set_xticks(range(0, 16, 2))
    ax.set_xlabel("generation")
    ax.set_ylabel("score relative to the automesher")
    ax.grid(axis="x", visible=False)
    fig.text(0.015, 0.975, "Swarms keep improving with training", ha="left", va="top", fontsize=21, fontweight="bold")
    what = ("the best score reached so far (running max of the best agent per generation); gen 7 equals the fig3 max bar"
            if stat == "max" else "the running mean over gens 1..g of the best agent per generation (gen 0: the gen-0 "
            "best); gen 7 equals the fig3 avg bar")
    rows = "\n".join(f"| {n} | {' / '.join(f'{v:.2f}' for v in out[n]['measured'])} | {out[n]['gen15']:.2f} |"
                     for n in SIZES)
    md = f"""
# {stem}: training curves on {BC0}, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: {what}. Dashed: y = a_n + b (g + 1)^-1/2, one rate
b = {b:.2f} shared by all team sizes (fitted to gens 1-7 of the four curves), started from each team's gen-7 value and
extended to gen 15 (no run went past gen 7). With a shared rate the projected order is the measured order.

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
{rows}

**Caveats.**
- One run per team size. The shared rate pools the four runs; per-team rates differ (the 4-agent run jumped in gens 6-7,
  the 8-agent run is flat from gen 5). The exponent 1/2 is chosen, not fitted: a free exponent runs to ~0, i.e. the
  8 generations do not show a plateau yet.
"""
    DATA[stem] = {"shared_rate_b": float(b), "curves": out}
    save(fig, stem, md)


# --------------------------------------------------------------------- fig 4: average agent per gen
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.95, top=0.79, bottom=0.12, wspace=0.22)
    for ax, task in zip(axes, TASKS):
        labels = []
        for n in (2, 4, 8):
            x, y = series(DIV[n], task, "mean")
            ax.plot(x, y, color=COL[n], lw=2.6, marker=MARK[n], ms=8, zorder=3)
            labels.append((x[-1], y[-1], f"{n} agents", COL[n]))
        ax.set_xlim(-0.4, 9.9)
        ax.set_xticks(range(8))
        if task == "abc_00000065_bc0":
            ax.set_ylim(0.2, 0.8)
            end_labels(ax, labels, 0.045)
        else:
            ax.set_ylim(0.12, 0.27)
            end_labels(ax, labels, 0.011)
        automesher(ax)
        ax.set_title(TLAB[task], loc="left")
        ax.set_xlabel("generation")
    axes[0].set_ylabel("mean base score over the team's agents")
    m8 = series(DIV[8], TASKS[0], "mean")[1]
    m4 = series(DIV[4], TASKS[0], "mean")[1]
    m2 = series(DIV[2], TASKS[0], "mean")[1]
    takeaway(fig, f"Not just more draws: the average agent learns too ({m8[0]:.2f} → {m8[-1]:.2f} with 8 agents)",
             "Mean base score of all agents on the task in each generation (afternoon runs); automesher = 0.25")
    mb = {n: series(DIV[n], TASKS[1], "mean")[1] for n in (2, 4, 8)}
    w = {n: window_mean(DATA[DIV[n]]["tasks"][TASKS[0]], "mean", DIV_WINDOW)[0] for n in (2, 4, 8)}
    md = f"""
# fig4_mean_agent_per_gen: the average agent learns too

**Caption.** The mean base score over all agents working on the task in each generation. With 8 agents (4 per task)
the average agent on abc_00000065_bc0 rises from {m8[0]:.3f} to {m8[-1]:.3f}, so the swarm's gain is not only a best-of-more-draws effect.

**Key numbers (abc_00000065_bc0, mean agent per generation, gens 0-7).**
- 8 agents: {', '.join(f'{v:.3f}' for v in m8)}
- 4 agents: {', '.join(f'{v:.3f}' for v in m4)}
- 2 agents (one run per task, so mean = best): {', '.join(f'{v:.3f}' for v in m2)}
- mean over gens 1-7: 8 agents {w[8]:.3f}, 4 agents {w[4]:.3f}, 2 agents {w[2]:.3f}.
- abc_00000003_bc7, gen 0 -> gen 7: 8 agents {mb[8][0]:.3f} -> {mb[8][-1]:.3f}, 4 agents {mb[4][0]:.3f} -> {mb[4][-1]:.3f},
  2 agents {mb[2][0]:.3f} -> {mb[2][-1]:.3f} (range of the 8-agent mean: {min(mb[8]):.3f}-{max(mb[8]):.3f}).

**Data.** {SRC_DIV}. Plotted values: `data.json` -> `runs.<run>.tasks.<task>[].mean`.

**Caveats.**
- The 4-agent swarm's mean agent ends at the same level ({m4[-1]:.3f} vs {m8[-1]:.3f} in gen 7); 8 agents get there sooner
  (8-agent mean >= 0.68 from gen 3, the 4-agent mean first above 0.65 in gen 6). The 8-agent mean dips in gen 6 ({m8[6]:.3f}).
- On bc7 the mean agent barely moves for any team size.
- With roles on, part of each team is told to explore (lowest-ranked third) or to vary one factor, which pulls the mean
  down by design; the mean is a conservative measure.
- One run per configuration.
"""
    save(fig, "fig4_mean_agent_per_gen", md)


# ------------------------------------------------------------ fig 5: 8-agent swarm vs single agent
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.25, wspace=0.22)
    c = {r: DATA[r]["cost"]["total"] for r in ALL_RUNS}
    spec = [
        ("div_s8", COL[8], "-", 2.6, 1.0, "o", f"8 Sonnet agents, afternoon (${c['div_s8']:.2f})"),
        ("div_s1", COL[1], "-", 2.6, 1.0, "D", f"1 Sonnet agent, budget-matched, afternoon (${c['div_s1']:.2f})"),
        ("scale_s8", COL[8], "--", 1.8, 0.45, "o", f"8 Sonnet agents, morning (${c['scale_s8']:.2f})"),
        ("scale_s1", COL[1], "--", 1.8, 0.45, "D", f"1 Sonnet agent, morning (${c['scale_s1']:.2f})"),
        ("scale_s1_opus", COL[1], ":", 2.2, 0.6, "P", f"1 Opus agent, morning (${c['scale_s1_opus']:.2f})"),
    ]
    handles = []
    for ax, task in zip(axes, TASKS):
        for run, col, ls, lw, a, mk, lab in spec:
            s = DATA[run]["tasks"][task]
            x = [p["gen"] for p in s if p["n_invalid"] < p["n"]]
            y = [p["best"] for p in s if p["n_invalid"] < p["n"]]
            h, = ax.plot(x, y, color=col, ls=ls, lw=lw, alpha=a, marker=mk, ms=8 if a == 1 else 7,
                         mfc=col if a == 1 else "white", zorder=3 if a == 1 else 2, label=lab)
            if task == TASKS[0]:
                handles.append(h)
            for p in s:
                if p["n_invalid"] == p["n"]:
                    ax.annotate(f"1 Opus agent, gen {p['gen']}:\ninvalid mesh (score 0)", xy=(p["gen"], ax.get_ylim()[0]),
                                xytext=(p["gen"] + 0.2, 0.135), fontsize=11.5, color="#555555")
        ax.set_xticks(range(8))
        ax.set_xlim(-0.4, 7.4)
        ax.set_ylim((0.2, 0.8) if task == "abc_00000065_bc0" else (0.12, 0.27))
        automesher(ax)
        ax.set_title(TLAB[task], loc="left")
        ax.set_xlabel("generation")
    axes[0].set_ylabel("best agent's base score")
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.0))
    w8 = {t: window_mean(DATA["div_s8"]["tasks"][t], "best", DIV_WINDOW)[0] for t in TASKS}
    w1 = {t: window_mean(DATA["div_s1"]["tasks"][t], "best", DIV_WINDOW)[0] for t in TASKS}
    w8m = {t: window_mean(DATA["scale_s8"]["tasks"][t], "best", range(1, 3))[0] for t in TASKS}
    w1m = {t: window_mean(DATA["scale_s1"]["tasks"][t], "best", range(1, 3))[0] for t in TASKS}
    sb = step_stats(DATA["div_s1"]["tasks"]["abc_00000065_bc0"])
    s8 = step_stats(DATA["div_s8"]["tasks"]["abc_00000065_bc0"])
    takeaway(fig, "8 agents beat one budget-matched agent on 65_bc0 and tie on bc7",
             "Best base score per generation: 8-agent swarm vs a single Sonnet agent with the swarm's solves, turns and time")
    b1 = DATA["div_s1"]["tasks"]
    single_bc7_late = [p["best"] for p in b1["abc_00000003_bc7"] if p["gen"] >= 3]
    s8_bc7_late = [p["best"] for p in DATA["div_s8"]["tasks"]["abc_00000003_bc7"] if p["gen"] >= 3]
    opus = DATA["scale_s1_opus"]["tasks"]
    opus_line = "; ".join(
        f"gen {a['gen']}: {'invalid' if a['n_invalid'] == a['n'] else format(a['best'], '.3f')} / {b['best']:.3f}"
        for a, b in zip(opus["abc_00000003_bc7"], opus["abc_00000065_bc0"]))
    md = f"""
# fig5_swarm_vs_single: 8 agents beat one budget-matched agent on 65_bc0 and tie on bc7

**Caption.** The afternoon 8-agent swarm (solid pink) against a single Sonnet agent that gets the whole swarm's solve
and turn budget and its wall-clock time per generation (solid blue). Shorter morning runs dashed: the 8-agent swarm, the
single Sonnet agent and a single Opus agent. API cost in the legend.

**Key numbers.**
- Mean of per-generation best, gens 1-7 (afternoon): abc_00000065_bc0 8 agents {w8['abc_00000065_bc0']:.3f} vs single {w1['abc_00000065_bc0']:.3f};
  abc_00000003_bc7 8 agents {w8['abc_00000003_bc7']:.3f} vs single {w1['abc_00000003_bc7']:.3f}.
- abc_00000065_bc0 max: 8 agents {s8['max']:.3f}, single {sb['max']:.3f}.
- abc_00000003_bc7 from gen 3 on: single {min(single_bc7_late):.3f}-{max(single_bc7_late):.3f}, 8 agents {min(s8_bc7_late):.3f}-{max(s8_bc7_late):.3f}: the single agent matches the swarm there.
- Morning, gens 1-2 (all morning runs have them): 65_bc0 8 agents {w8m['abc_00000065_bc0']:.3f} vs single Sonnet {w1m['abc_00000065_bc0']:.3f};
  bc7 {w8m['abc_00000003_bc7']:.3f} vs {w1m['abc_00000003_bc7']:.3f}. Single Opus per gen (bc7 / 65_bc0):
  {opus_line}.
- API cost: 8 agents afternoon ${c['div_s8']:.2f}, single afternoon ${c['div_s1']:.2f}; morning 8 agents ${c['scale_s8']:.2f},
  single Sonnet ${c['scale_s1']:.2f}, single Opus ${c['scale_s1_opus']:.2f}.

**Data.** `ace_runs/div_s8`, `ace_runs/div_s1`, `ace_runs/scale_s8`, `ace_runs/scale_s1`, `ace_runs/scale_s1_opus`
(`gen*/runs/*.json` checks; `cost.jsonl`). Budget matching: `ace_runs/run_div_s1.sh`, `ace/single_loop.py`
(per task MAX_SOLVES x swarm agents per task = 16 solves, MAX_TURNS x 8 turns per generation, time-matched to div_s8).

**Caveats.**
- "Budget-matched" means solves, turns and wall-clock time, not dollars: the single agent cost about 1/3 of the swarm
  (${c['div_s1']:.2f} vs ${c['div_s8']:.2f}); the swarm's cost is dominated by the Opus teacher (see fig 6).
- The single agent is not stronger per call: div_s1 and scale_s1 use Sonnet like the swarm agents (high effort);
  only scale_s1_opus uses Opus, and it ran 3 generations, with an invalid bc7 mesh in gen 1.
- div_s1 gen 7 resubmitted exactly the gen-6 meshes on both tasks (same DOFs and error).
- On bc7 the single agent catches up with the swarm from gen 3; the swarm's advantage is clear only on 65_bc0.
- The morning runs point the other way on 65_bc0: over gens 1-2 the single Sonnet agent ({w1m['abc_00000065_bc0']:.3f})
  is above the morning 8-agent swarm ({w8m['abc_00000065_bc0']:.3f}), while the swarm leads on bc7 ({w8m['abc_00000003_bc7']:.3f} vs
  {w1m['abc_00000003_bc7']:.3f}). Only 2 learning generations, so this is weak evidence either way, but the
  afternoon result is not replicated by the morning pair.
- One run per configuration; the morning runs are short (3 generations for the single agents).
"""
    save(fig, "fig5_swarm_vs_single", md)


# ------------------------------------------------------------------------ fig 6: cost per generation
def fig6():
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.79, bottom=0.12)
    labels = []
    for n in (1, 2, 4, 8):
        pg = DATA[DIV[n]]["cost"]["per_gen"]
        x, y = list(pg.keys()), list(pg.values())
        ax.plot(x[:7], y[:7], color=COL[n], lw=2.6, marker=MARK[n], ms=8)
        ax.plot(x[7:], y[7:], ls="none", marker=MARK[n], ms=8, mfc="white", mec=COL[n], mew=1.8)
        labels.append((7, y[6], f"{n} agent{'s' if n > 1 else ''}: ${DATA[DIV[n]]['cost']['total']:.2f} total", COL[n]))
    ax.set_xticks(range(8))
    ax.set_xlim(-0.3, 10.2)
    ax.set_ylim(0, 7.2)
    end_labels(ax, labels, 0.42)
    ax.set_xlabel("generation")
    ax.set_ylabel("API cost per generation ($)")
    ax.text(8.6, 0.25, "hollow: gen 7, the last generation,\nhas no teacher/organizer step", fontsize=12, color="#555555",
            ha="center", va="bottom")
    mean_pg = {n: float(np.mean([DATA[DIV[n]]["cost"]["per_gen"][g] for g in range(7)])) for n in (1, 2, 4, 8)}
    roles = {n: DATA[DIV[n]]["cost"]["by_role"] for n in (1, 2, 4, 8)}
    takeaway(fig, f"Cost grows sub-linearly: 4x the agents cost {mean_pg[8] / mean_pg[2]:.1f}x per generation",
             "API cost per generation, afternoon runs (Sonnet agents, Opus teacher); labels give each run's total")

    def rl(n):
        r = roles[n]
        gen = r.get("generator", 0) + r.get("big_generator", 0)
        return (f"| {n} | ${DATA[DIV[n]]['cost']['total']:.2f} | ${mean_pg[n]:.2f} | ${gen:.2f} | ${r.get('teacher', 0):.2f} | "
                f"${r.get('reflect', 0):.2f} | ${r.get('organizer', 0):.2f} |")
    mor = {n: DATA[SCALE[n]]["cost"]["total"] for n in (1, 2, 4, 8)}
    md = f"""
# fig6_cost_per_gen: cost grows sub-linearly with team size

**Caption.** API dollars spent per generation by each afternoon run. Going from 2 to 8 agents multiplies the mean
cost per generation (gens 0-6) by {mean_pg[8] / mean_pg[2]:.2f}, not 4 (2 -> 4 agents: {mean_pg[4] / mean_pg[2]:.2f}x; 4 -> 8: {mean_pg[8] / mean_pg[4]:.2f}x).
Both the agents' spend and the Opus teacher's spend grow less than linearly (table).

**Key numbers (afternoon runs; mean $/gen over gens 0-6, gen 7 excluded because the run ends before its teacher step).**

| agents | total $ | mean $/gen (0-6) | agents (generator) $ | teacher (Opus) $ | reflect $ | organizer $ |
|---|---|---|---|---|---|---|
{chr(10).join(rl(n) for n in (1, 2, 4, 8))}

Morning totals: 1 agent ${mor[1]:.2f}, 2 agents ${mor[2]:.2f}, 4 agents ${mor[4]:.2f}, 8 agents ${mor[8]:.2f};
single Opus ${DATA['scale_s1_opus']['cost']['total']:.2f}. Afternoon total ${sum(DATA[DIV[n]]['cost']['total'] for n in (1, 2, 4, 8)):.2f}.

**Data.** `ace_runs/div_s{{1,2,4,8}}/cost.jsonl` (`usd` per call). A generator call carries its generation in `run_id`;
reflection, teacher and organizer calls (empty `run_id`) are assigned to the latest generation started. This attribution
reproduces `ace_runs/div_s1/generations.jsonl` and the cumulative $ in `ace_runs/div_s8/summary.md` exactly.

**Caveats.**
- The 1-agent row is the single agent (div_s1, one conversation for both tasks); it costs slightly more per generation
  than the 2-agent swarm.
- "Sub-linear" is from one run per size and 7 generations; it is not a cost model.
- The 8-agent swarm's teacher spend (${roles[8].get('teacher', 0):.2f}) is larger than its agents' spend
  (${roles[8].get('generator', 0):.2f}).
"""
    DATA["fig6"] = {"mean_usd_per_gen_0_6": mean_pg, "by_role": roles}
    save(fig, "fig6_cost_per_gen", md)


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    fig1(False)
    fig1(True)
    order_ok, vals2 = fig2()
    for stat in ("max", "avg"):
        fig3(stat)
        fig7(stat)
    fig4()
    fig5()
    fig6()
    out = {"note": "per-generation best/mean base score per run and task; costs; figure tables",
           "max_abs_stored_vs_recomputed_base": max_mismatch, "invalid_runs": n_invalid,
           "runs": {r: DATA[r] for r in ALL_RUNS},
           **{k: v for k, v in DATA.items() if k.startswith("fig")}}
    json.dump(out, open(HERE / "data.json", "w"), indent=1, default=str)
    keys = ["run", "gen", "task", "agent", "valid", "base_score", "recomputed_base", "field_error", "rel_error",
            "dofs", "auto_error", "auto_dofs", "score_with_bonuses", "invalid_because", "file"]
    with open(HERE / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for run in ALL_RUNS:
            w.writerows(RAW[run])
    print("max |stored - recomputed| base score:", max_mismatch)
    print("invalid runs:", n_invalid)
    print("8>4>2 ordering:", order_ok)
