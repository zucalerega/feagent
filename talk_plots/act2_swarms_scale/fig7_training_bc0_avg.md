# fig7_training_bc0_avg: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: the best agent of each generation; its gens 1-7 mean equals the fig3 avg bar. Dashed: continues from each team's gen-7 value along a fit
y = a + b (g + 1)^-1/2 to its gens 0-7, a curve that flattens, out to gen 15 (no run went past gen 7).

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 1.89 / 1.32 / 1.92 / 2.00 / 2.36 | 2.31 |
| 4 | 2.30 / 2.24 / 2.39 / 2.47 / 2.47 / 2.56 / 2.85 / 2.97 | 3.06 |
| 8 | 2.60 / 2.81 / 2.85 / 2.92 / 2.96 / 3.01 / 2.95 / 2.98 | 3.04 |
| 16 | 2.56 / 2.86 / 2.84 / 3.00 / 2.99 / 3.10 / 3.15 / 3.18 | 3.27 |

**Caveats.**
- One run per team size; the projection is a 2-parameter fit to 8 points, and its exponent 1/2 is chosen, not
  fitted: with a free exponent shared by the four teams the fit runs to ~0 (no plateau visible in 8 generations).
- 2 agents (1 per task) stall: their per-generation best falls to 1.32x in gen 4, so their projection points down.
- The 4-agent projection ends above the 8-agent one, from the 4-agent run's jump in gens 6-7.
