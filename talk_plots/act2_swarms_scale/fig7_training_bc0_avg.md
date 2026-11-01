# fig7_training_bc0_avg: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: average agent of each generation; gen 7 is the fig3_scaling_bc0_avg bar.
Dashed: the mean agent's per-generation score is fitted with y = a + b ln(g + 1) on all 8 measured
generations and continued from its gen-7 value at that rate to gen 15; the same measure is applied to measured +
projected values (no run went past gen 7).

| agents | gens 0-7 | rate b | projected gen 15 |
|---|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 1.89 / 1.32 / 1.92 / 2.00 / 2.36 | -0.13 | 2.27 |
| 4 | 2.20 / 2.18 / 2.39 / 2.45 / 2.43 / 2.48 / 2.71 / 2.91 | 0.30 | 3.11 |
| 8 | 2.20 / 2.45 / 2.53 / 2.72 / 2.89 / 2.92 / 2.73 / 2.93 | 0.35 | 3.17 |
| 16 | 2.25 / 2.24 / 2.76 / 2.73 / 2.83 / 2.87 / 2.90 / 3.03 | 0.39 | 3.30 |

**Data.** 2/4/8 agents: `ace_runs/div_s{2,4,8}/gen*/runs/g*-abc_00000065_bc0-*.json` (`checks.base_score`). 16 agents:
`results/diversity_s16_2026-11-01/scores.json` (per-generation best and mean exported by Jaime from his run).

**Caveats.**
- One run per team size; the rate is a 2-parameter fit to 8 points. The log curve is the shape the data prefer (a free
  saturating exponent runs to ~0): the 8 generations do not show a plateau yet, which is not evidence that there is none.
- 2 agents = 1 agent per task, i.e. no swarm; its per-generation score falls to 1.32x in gen 4.
