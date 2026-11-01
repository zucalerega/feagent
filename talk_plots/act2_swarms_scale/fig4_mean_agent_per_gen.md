# fig4_mean_agent_per_gen: the average agent learns too

**Caption.** The mean base score over all agents working on the task in each generation. With 8 agents (4 per task)
the average agent on abc_00000065_bc0 rises from 0.550 to 0.733, so the swarm's gain is not only a best-of-more-draws effect.

**Key numbers (abc_00000065_bc0, mean agent per generation, gens 0-7).**
- 8 agents: 0.550, 0.611, 0.633, 0.680, 0.723, 0.729, 0.683, 0.733
- 4 agents: 0.550, 0.544, 0.596, 0.613, 0.607, 0.619, 0.678, 0.727
- 2 agents (one run per task, so mean = best): 0.548, 0.560, 0.600, 0.472, 0.331, 0.481, 0.501, 0.590
- mean over gens 1-7: 8 agents 0.685, 4 agents 0.626, 2 agents 0.505.
- abc_00000003_bc7, gen 0 -> gen 7: 8 agents 0.221 -> 0.229, 4 agents 0.203 -> 0.204,
  2 agents 0.175 -> 0.206 (range of the 8-agent mean: 0.206-0.234).

**Data.** `ace_runs/div_s{1,2,4,8}/gen*/runs/g*-{task}-*.json` (`checks.base_score`, recomputed from `rel_error`, `auto_error`, `dofs`, `auto_dofs`). Plotted values: `data.json` -> `runs.<run>.tasks.<task>[].mean`.

**Caveats.**
- The 4-agent swarm's mean agent ends at the same level (0.727 vs 0.733 in gen 7); 8 agents get there sooner
  (8-agent mean >= 0.68 from gen 3, the 4-agent mean first above 0.65 in gen 6). The 8-agent mean dips in gen 6 (0.683).
- On bc7 the mean agent barely moves for any team size.
- With roles on, part of each team is told to explore (lowest-ranked third) or to vary one factor, which pulls the mean
  down by design; the mean is a conservative measure.
- One run per configuration.
