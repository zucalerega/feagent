# fig7_training_bc0_bestmean: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: running mean of the best agent per generation (gens 1..g); gen 7 is the fig3_scaling_bc0_bestmean bar.
Dashed: the best agent's per-generation score is fitted with y = a + b ln(g + 1) on all 8 measured
generations used and continued from its last measured value at that rate to gen 15; the same measure is
applied to measured + projected values. Measured: gens 0-7 (no run went past gen 7).

| agents | gens 0-7 | rate b | projected gen 15 |
|---|---|---|---|
| 2 | 2.19 / 2.24 / 2.32 / 2.18 / 1.96 / 1.95 / 1.96 / 2.02 | -0.13 | 2.17 |
| 4 | 2.30 / 2.24 / 2.31 / 2.37 / 2.39 / 2.43 / 2.50 / 2.56 | 0.30 | 2.85 |
| 8 | 2.60 / 2.81 / 2.83 / 2.86 / 2.89 / 2.91 / 2.92 / 2.93 | 0.18 | 3.00 |
| 16 | 2.56 / 2.86 / 2.85 / 2.90 / 2.92 / 2.96 / 2.99 / 3.01 | 0.28 | 3.17 |

**Data.** 2/4/8 agents: `ace_runs/div_s{2,4,8}/gen*/runs/g*-abc_00000065_bc0-*.json` (`checks.base_score`). 16 agents:
`results/diversity_s16_2026-11-01/scores.json` (per-generation best and mean exported by Jaime from his run).

**Caveats.**
- One run per team size; the rate is a 2-parameter fit to 8 points. The log curve is the shape the data prefer (a free
  saturating exponent runs to ~0): the 8 generations do not show a plateau yet, which is not evidence that there is none.
- 2 agents = 1 agent per task, i.e. no swarm; its per-generation score falls to 1.32x in gen 4.
