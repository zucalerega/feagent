# fig1_best_per_gen: on 65_bc0, 8 agents improve almost every generation; 2 agents stall

**Caption.** Best agent's base score in each generation for swarms of 2, 4 and 8 Sonnet agents (afternoon
runs, diversity mechanisms on). On abc_00000065_bc0
the 8-agent swarm starts highest and climbs to about 3x the automesher; on abc_00000003_bc7 no team beats the automesher.

**Key numbers (afternoon, best per generation, gens 0 -> 7).**
- abc_00000065_bc0: 8 agents 0.651 -> 0.745 (max 0.753; steps up/down/flat
  6/1/0, largest drop -0.017);
  4 agents 0.575 -> 0.742 (up/down/flat 5/1/1, largest drop -0.016);
  2 agents 0.548 -> 0.590 (up/down/flat 5/2/0, largest drop
  -0.141, low point 0.331 in gen 4).
- abc_00000003_bc7: 8 agents 0.230 -> 0.233 (max 0.237), 4 agents
  0.205 -> 0.205, 2 agents 0.175 -> 0.206.
  Highest bc7 base score in any of the nine runs: 0.237 (< 0.25).
- Step counts use a flat band of +-0.005.

**Data.** `ace_runs/div_s{1,2,4,8}/gen*/runs/g*-{task}-*.json` (`checks.base_score`, recomputed from `rel_error`, `auto_error`, `dofs`, `auto_dofs`). Plotted values: `data.json` -> `runs.<run>.tasks.<task>`.

**Caveats.**
- One run per configuration; the afternoon README estimates run-to-run noise at 0.03-0.1 per generation.
- Team size n splits 50/50 over the two tasks, so the per-task best is the max of n/2 runs (1, 2, 4 for n = 2, 4, 8). Best-of-n includes more draws; the framing used here is that the swarm is trained and deployed together, so its best mesh is the one used.
- abc_00000003_bc7: every team stays below the automesher (best 0.237); the differences between team sizes'
  mean-of-best (gens 1-7) are about 0.02-0.04 and the 8 > 4 > 2 order is already there in gen 0, before any learning.
- "Almost every generation" means 6 of 7 steps up on abc_00000065_bc0 for 8 agents; the 4-agent swarm has
  nearly as many up-steps (5 vs 6) and a larger net gain (0.166 vs 0.094), but it
  sits around 0.56-0.64 for gens 0-5 and only catches up in gens 6-7.
