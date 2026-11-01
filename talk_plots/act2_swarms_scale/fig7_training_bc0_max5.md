# fig7_training_bc0_max5: training curves on abc_00000065_bc0, extended to gen 9

**Caption.** Afternoon setup, automesher = 1. Solid: best score reached so far, gens 0-4 only; gen 4 is the fig3_scaling_bc0_max5 bar.
Dashed: the best agent's per-generation score is fitted with y = a + b ln(g + 1) on all 8 measured
generations used and continued from its last measured value at that rate to gen 9; the same measure is
applied to measured + projected values. Measured: gens 0-4 (the runs went to gen 7; gens 5-7 are left out of this version).

| agents | gens 0-4 | rate b | projected gen 9 |
|---|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 2.40 / 2.40 | -0.42 | 2.40 |
| 4 | 2.30 / 2.30 / 2.39 / 2.47 / 2.47 | 0.13 | 2.57 |
| 8 | 2.60 / 2.81 / 2.85 / 2.92 / 2.96 | 0.22 | 3.11 |
| 16 | 2.56 / 2.86 / 2.86 / 3.00 / 3.00 | 0.27 | 3.17 |

**Data.** 2/4/8 agents: `ace_runs/div_s{2,4,8}/gen*/runs/g*-abc_00000065_bc0-*.json` (`checks.base_score`). 16 agents:
`results/diversity_s16_2026-11-01/scores.json` (per-generation best and mean exported by Jaime from his run).

**Caveats.**
- One run per team size; the rate is a 2-parameter fit to 8 points. The log curve is the shape the data prefer (a free
  saturating exponent runs to ~0): the 8 generations do not show a plateau yet, which is not evidence that there is none.
- 2 agents = 1 agent per task, i.e. no swarm; its per-generation score falls to 1.32x in gen 4.
