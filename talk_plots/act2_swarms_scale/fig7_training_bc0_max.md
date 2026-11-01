# fig7_training_bc0_max: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: the best score reached so far (running max of the best agent per generation); gen 7 equals the fig3 max bar. Dashed: continues from each team's gen-7 value along a fit
y = a + b (g + 1)^-1/2 to its gens 0-7, a curve that flattens, out to gen 15 (no run went past gen 7).

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 2.40 / 2.40 / 2.40 / 2.40 / 2.40 | 2.44 |
| 4 | 2.30 / 2.30 / 2.39 / 2.47 / 2.47 / 2.56 / 2.85 / 2.97 | 3.05 |
| 8 | 2.60 / 2.81 / 2.85 / 2.92 / 2.96 / 3.01 / 3.01 / 3.01 | 3.08 |
| 16 | 2.56 / 2.86 / 2.86 / 3.00 / 3.00 / 3.10 / 3.15 / 3.18 | 3.27 |

**Caveats.**
- One run per team size; the projection is a 2-parameter fit to 8 points, and its exponent 1/2 is chosen, not
  fitted: with a free exponent shared by the four teams the fit runs to ~0 (no plateau visible in 8 generations).
- 2 agents (1 per task): the running max is flat at 2.40x from gen 2; 8 agents: flat at 3.01x from gen 5.
