"""Graphs: what multiple agents buy over one agent, from the team's swarm runs and the 6-system mesh study.

usage: make_figs.py COMPARISON_JSON SWARM_DATA_JSON README_NUMBERS_JSON OUT_DIR
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
                     "axes.facecolor": SURF})
SYS = ["harness_auto", "harness_expert", "ace_sizing", "gold_nested", "netgen", "harness_hex"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_swarm(sw, rn, out):
    """Team size and swarm vs a budget-matched single agent (the team's ACE runs on main)."""
    f2 = sw["fig2"]
    single = {r["task"]: r["best"] for r in rn["best_vs_mean"] if r["run"] == "div_s1"}
    tasks = [("abc_00000065_bc0", "abc_00000065 (the clamp), load case bc0"),
             ("abc_00000003_bc7", "abc_00000003 (the tube), load case bc7")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (t, name) in zip(axs, tasks):
        xs = ["1\n(budget-\nmatched)", "2", "4", "8", "16"]
        vals = [single[t]] + [f2[f"div|{t}|{n}"] for n in (2, 4, 8, 16)]
        cols = [MUTED] + [S1] * 4
        b = ax.bar(range(5), vals, 0.62, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(single[t], color=MUTED, ls="--", lw=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(xs, fontsize=9)
        ax.set_xlabel("agents training together")
        ax.set_ylim(0, max(vals) * 1.18); style(ax)
        gain = (vals[-2] / vals[0] - 1) * 100
        ax.set_title(f"{name}\n8 agents vs 1 budget-matched agent: {gain:+.0f} %", loc="left", fontsize=10.5)
    axs[0].set_ylabel("score (mean of each round's best mesh;\nhigher = more accurate per DOF)")
    fig.suptitle("More agents train better meshes than one agent given the same solves, turns and time",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.06, "Data: talk_plots/act2_swarms_scale/data.json and readme_numbers.json on main (rounds 1-7; "
             "16 agents: Jaime's run). One run per configuration. On the tube the single agent ties the swarm from round 3; "
             "2 agents score below 1 agent on the clamp.", fontsize=8, color=INK2, wrap=True)
    fig.tight_layout()
    fig.savefig(out / "fig_ma1_swarm_vs_single.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_crosscheck(cmp, out):
    """Six independent mesh agents: spread per quantity says which numbers can be trusted."""
    q = [("compliance_um_at_100N", "stiffness\n(compliance)"), ("robust_vm_MPa_at_Ffull", "stress, robust\n(99.9th pct)"),
         ("crush_peak_force_N", "crush peak force\n(nonlinear)"), ("peak_vm_MPa_at_Ffull", "stress, peak\n(single point)")]
    parts = [("abc_00000003", "abc_00000003 (tube)"), ("abc_00000065", "abc_00000065 (clamp)")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    marks = ["o", "s", "D", "^", "v", "P"]
    cols = [S1, S2, S3, S4, "#e87ba4", "#6250d6"]
    for ax, (p, name) in zip(axs, parts):
        sp = cmp["spread"][p]
        for i, (k, lab) in enumerate(q):
            dev = sp[k]["dev_pct"]
            for j, s in enumerate(SYS):
                if s in dev and dev[s] is not None:
                    ax.scatter(i + (j - 2.5) * 0.07, dev[s], s=42, marker=marks[j], color=cols[j], edgecolor=SURF,
                               lw=1.2, zorder=3, label=s if (i == 0 and p == "abc_00000003") else None)
            ax.text(i, max(v for v in dev.values() if v is not None) + 4, f"range {sp[k]['range_pct']:.1f} %",
                    ha="center", fontsize=9, color=INK)
        ax.axhline(0, color=INK2, lw=1)
        ax.axhspan(-2, 2, color="#cde2fb", alpha=0.5, lw=0)
        ax.set_xticks(range(len(q))); ax.set_xticklabels([l for _, l in q], fontsize=9)
        ax.set_title(name, loc="left", fontsize=11); style(ax)
    axs[0].set_ylabel("each agent's result vs the median of the six (%)")
    axs[1].text(3.45, -4.5, "band = within 2 %", fontsize=8, color="#184f95", ha="right")
    fig.legend(loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=9)
    fig.suptitle("Six independent mesh agents check each other: one agent can't tell which of its numbers are reliable",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.05, "Same part, loads and solver; only the mesh agent differs (6 systems, 40k-400k DOFs). Where the six "
             "agree (stiffness, within 1.3 %) a single answer is safe. Where they spread (crush force 20-31 %, peak stress at a singular "
             "platen edge 42-88 %), single agents land up to 60 % from the median, and no agent can see that from its own run.",
             fontsize=8, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_ma2_crosscheck.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_neck(cmp, neck, out):
    """Part 65: the first recipe (one agent's setup) vs the combined neck recipe on matched meshes."""
    old = {s: cmp["rows"][f"{s}/abc_00000065"]["crush"]["peak_force_N"] / 1e3 for s in SYS}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    x = np.arange(len(SYS) + 1)
    labels = SYS + ["harness_neck"]
    ov = [old[s] for s in SYS] + [np.nan]
    nv = [neck.get(s, (np.nan, np.nan))[0] for s in labels]
    ax.bar(x - 0.2, ov, 0.38, color=MUTED, label="first recipe: PEEQ erosion, each agent's own mesh size")
    ax.bar(x + 0.2, nv, 0.38, color=S1, label="combined recipe: tension-only erosion, buckle, bonded tips, 40k nodes each (hex: 54k)")
    for i, v in enumerate(nv):
        ax.text(i + 0.2, (v if np.isfinite(v) else 0) + 0.08, f"{v:.2f}" if np.isfinite(v) else "running",
                ha="center", fontsize=8, color=INK, rotation=0 if np.isfinite(v) else 90,
                va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=20)
    ax.set_ylabel("peak force = predicted snap load (kN)"); style(ax)
    ov_ = np.array([v for v in ov if np.isfinite(v)]); nv_ = np.array([v for v in nv if np.isfinite(v)])
    ax.set_title(f"Spread across meshes: {np.ptp(ov_) / np.median(ov_) * 100:.0f} % before, "
                 f"{np.ptp(nv_) / np.median(nv_) * 100:.1f} % after (tets only: {np.ptp(nv_[:5].tolist() + nv_[6:].tolist()) / np.median(nv_) * 100:.1f} %)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2))
    ax.set_ylim(0, 7)
    ax = axs[1]
    hf_old = [0.89] * 6
    hf_new = [neck[s][1] for s in labels if s in neck and np.isfinite(neck[s][1])]
    ax.axhspan(0.45, 0.60, color="#cde2fb", lw=0)
    ax.text(0.03, 0.525, "where the printed part broke\n(estimated from the photo)", fontsize=9, color="#184f95", va="center",
            transform=ax.get_yaxis_transform())
    ax.scatter(np.full(6, 0) + np.linspace(-0.12, 0.12, 6), hf_old, s=46, color=MUTED, edgecolor=SURF, lw=1.2, zorder=3)
    ax.scatter(np.full(len(hf_new), 1) + np.linspace(-0.1, 0.1, len(hf_new)), hf_new, s=46, color=S1,
               edgecolor=SURF, lw=1.2, zorder=3)
    ax.text(0, 0.92, "all 6: arm tips", ha="center", fontsize=9, color=INK)
    ax.text(1, max(hf_new) + 0.03, f"all {len(hf_new)}: arm necks", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["first recipe", "combined recipe"])
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.0)
    ax.set_ylabel("crack height (0 = base, 1 = top)"); style(ax)
    ax.set_title("Where the main crack starts", loc="left", fontsize=10.5)
    fig.suptitle("abc_00000065: with the neck-study recipe, all 7 mesh agents agree and crack at the necks, like the real part",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_ma3_neck_recipe.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    cmp, sw, rn = (json.loads(Path(p).read_text()) for p in sys.argv[1:4])
    out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
    neck = json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else {}
    neck = {k: (v[0], v[1]) for k, v in neck.items()}
    fig_swarm(sw, rn, out)
    fig_crosscheck(cmp, out)
    fig_neck(cmp, neck, out)


if __name__ == "__main__":
    main()
