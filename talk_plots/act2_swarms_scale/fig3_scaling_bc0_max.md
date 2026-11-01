# fig3_scaling_bc0_max: 16 > 8 > 4 > 2, extrapolated

**Caption.** Talk version of fig2 for abc_00000065_bc0 only, afternoon setup (diversity on), scores divided by the
automesher's 0.25 so that the automesher is 1. Bars: the best score any agent reached in gens 0-7.
Dashed: projection from the 16-agent bar, y16 + f(n) - f(16), with f = c - d n^-alpha (c = 3.15, d = 2.62,
alpha = 1.82) a least-squares saturating fit to the four bars: 32 agents 3.19x, 64 agents 3.19x
(no swarm of that size was run).

**Key numbers.** 2 / 4 / 8 / 16 agents: 2.40x / 2.97x / 3.01x / 3.18x the automesher.
Fit residuals: -0.00 / +0.03 / -0.07 / +0.05 (the projection is anchored on the measured 16-agent value).

**Data.** As fig2 (16 agents from `results/diversity_s16_2026-11-01/scores.json`). Plotted values: `data.json` -> `fig3_scaling_bc0_max`.

**Caveats.**
- A 3-parameter fit to 4 points, one run each (noise 0.03-0.1 in score, 0.1-0.4 on this axis); the plateau c is
  not pinned down by the data.
- The 16-agent run's references were rebuilt (see fig2), and team size n means n/2 agents per task, so a bigger
  team also takes the best of more meshes.
