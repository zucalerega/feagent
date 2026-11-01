# fig7_training_bc0_avg: training curves on abc_00000065_bc0, extended to gen 15

**Caption.** Afternoon setup, automesher = 1. Solid: the average agent of each generation; gen 7 equals the fig3 avg bar. Dashed: y = a + b (g + 1)^-1/2 from each team's gen-7 value
to gen 15 (no run went past gen 7), with each team's own rate fitted to its gens 0-7: 2: -0.46 / 4: 0.89 / 8: 1.13 / 16: 1.24.

| agents | gens 0-7 | projected gen 15 |
|---|---|---|
| 2 | 2.19 / 2.24 / 2.40 / 1.89 / 1.32 / 1.92 / 2.00 / 2.36 | 2.31 |
| 4 | 2.20 / 2.18 / 2.39 / 2.45 / 2.43 / 2.48 / 2.71 / 2.91 | 3.00 |
| 8 | 2.20 / 2.45 / 2.53 / 2.72 / 2.89 / 2.92 / 2.73 / 2.93 | 3.05 |
| 16 | 2.25 / 2.24 / 2.76 / 2.73 / 2.83 / 2.87 / 2.90 / 3.03 | 3.16 |

**Caveats.**
- One run per team size. The exponent 1/2 is chosen, not fitted: a free exponent runs to ~0, i.e. the 8 generations
  do not show a plateau yet.
- 2 agents = 1 agent per task, i.e. no swarm: its average agent does not learn (rate below 0) and is the noisiest (residual sd 0.37 vs 0.09-0.14 for 4-16 agents). Among 4, 8 and 16 agents the noise is similar.
- No ablation of the competition/collaboration mechanics was run; bigger swarms also put more examples into the shared playbook per generation.
