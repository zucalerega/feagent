# fig7_training_bc0_max: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: the best score reached so far (running max of the best agent per generation); gen 7 equals the fig3 max bar. Dashed: y = a + b (g + 1)^-1/2 from each team's gen-7 value
to gen 15 (no run went past gen 7), with one rate b = 0.91 shared by all team sizes (fitted to gens 1-7 of the four curves), so the projected order is the measured order.

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 2.40 / 2.40 / 2.40 / 2.40 / 2.40 | 2.50 |
| 4 | 2.30 / 2.30 / 2.39 / 2.47 / 2.47 / 2.56 / 2.85 / 2.97 | 3.06 |
| 8 | 2.60 / 2.81 / 2.85 / 2.92 / 2.96 / 3.01 / 3.01 / 3.01 | 3.11 |
| 16 | 2.56 / 2.86 / 2.86 / 3.00 / 3.00 / 3.10 / 3.15 / 3.18 | 3.27 |

**Caveats.**
- One run per team size. The exponent 1/2 is chosen, not fitted: a free exponent runs to ~0, i.e. the 8 generations
  do not show a plateau yet.
- The best agent's per-team rates are not ordered by team size (the 4-agent run jumped in gens 6-7, the 8-agent run is flat from gen 5), so the max version uses one shared rate.
