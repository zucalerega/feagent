# fig3_scaling_bc0_max: 16 > 8 > 4 > 2, extrapolated

**Caption.** Talk version of fig2 for abc_00000065_bc0 only, afternoon setup (diversity on), scores divided by the
automesher's 0.25 so that the automesher is 1. Bars: the best score any agent reached in gens 0-7.
Dashed: least-squares fit y = 2.296 + 0.237 log2(n) to the four measured bars, extended to 32 and 64 agents
(projected 3.48x and 3.72x; no swarm of that size was run).

**Key numbers.** 2 / 4 / 8 / 16 agents: 2.40x / 2.97x / 3.01x / 3.18x the automesher.
Fit residuals: -0.13 / +0.20 / +0.01 / -0.07.

**Data.** As fig2 (16 agents from `results/diversity_s16_2026-11-01/scores.json`). Plotted values: `data.json` -> `fig3`.

**Caveats.**
- The extrapolation is a fit to 4 points, one run each (noise 0.03-0.1 in score, 0.1-0.4 on this axis).
- Measured gain per doubling: +0.57, +0.05, +0.16.
- The 16-agent run's references were rebuilt (see fig2), and team size n means n/2 agents per task, so a bigger
  team also takes the best of more meshes.
