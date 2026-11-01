# fig7_training_bc0_avg: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: the running mean over gens 1..g of the best agent per generation (gen 0: the gen-0 best); gen 7 equals the fig3 avg bar. Dashed: y = a_n + b (g + 1)^-1/2, one rate
b = -0.16 shared by all team sizes (fitted to gens 1-7 of the four curves), started from each team's gen-7 value and
extended to gen 15 (no run went past gen 7). With a shared rate the projected order is the measured order.

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
| 2 | 2.19 / 2.24 / 2.32 / 2.18 / 1.96 / 1.95 / 1.96 / 2.02 | 2.04 |
| 4 | 2.30 / 2.24 / 2.31 / 2.37 / 2.39 / 2.43 / 2.50 / 2.56 | 2.58 |
| 8 | 2.60 / 2.81 / 2.83 / 2.86 / 2.89 / 2.91 / 2.92 / 2.93 | 2.94 |
| 16 | 2.56 / 2.86 / 2.85 / 2.90 / 2.92 / 2.96 / 2.99 / 3.01 | 3.03 |

**Caveats.**
- One run per team size. The shared rate pools the four runs; per-team rates differ (the 4-agent run jumped in gens 6-7,
  the 8-agent run is flat from gen 5). The exponent 1/2 is chosen, not fitted: a free exponent runs to ~0, i.e. the
  8 generations do not show a plateau yet.
