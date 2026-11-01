# fig2_mean_best_by_team_size: 8 > 4 > 2 in all four comparisons

**Caption.** For each study and task, the best agent's base score averaged over the generations after gen 0.
The ordering 8 > 4 > 2 agents holds in 4 of 4 cases.

**Key numbers.**

| study | task | 2 agents | 4 agents | 8 agents | 8 > 4 > 2 |
|---|---|---|---|---|---|
| div | abc_00000065_bc0 | 0.505 | 0.641 | 0.731 | yes |
| div | abc_00000003_bc7 | 0.194 | 0.213 | 0.230 | yes |
| scale | abc_00000065_bc0 | 0.642 | 0.650 | 0.668 | yes |
| scale | abc_00000003_bc7 | 0.198 | 0.218 | 0.231 | yes |

Windows: afternoon gens 1-7 (all runs have gens 0-7); morning gens 1-4, the window all three morning swarms
completed. With each morning run's own window (2 agents gens 1-5, 4 and 8 agents gens 1-4, as in
`results/scaling_gold_2026-10-31/README.md`) the morning values are: 2 agents 0.199 / 0.631,
4 agents 0.218 / 0.650, 8 agents 0.231 / 0.668 (bc7 / 65_bc0);
the order is the same.

**Data.** `ace_runs/div_s{1,2,4,8}/gen*/runs/g*-{task}-*.json` (`checks.base_score`, recomputed from `rel_error`, `auto_error`, `dofs`, `auto_dofs`); `ace_runs/scale_s{1,1_opus,2,4,8}/gen*/runs/g*-{task}-*.json` (same fields). Plotted values: `data.json` -> `fig2`.

**Caveats.**
- One run per configuration; the afternoon README estimates run-to-run noise at 0.03-0.1 per generation.
- Team size n splits 50/50 over the two tasks, so the per-task best is the max of n/2 runs (1, 2, 4 for n = 2, 4, 8). Best-of-n includes more draws; the framing used here is that the swarm is trained and deployed together, so its best mesh is the one used.
- Morning 65_bc0 gaps are small (2 / 4 / 8: 0.642 / 0.650 /
  0.668), well inside the 0.03-0.1 per-generation noise; and the single highest morning
  65_bc0 score was the 2-agent run's 0.717 (gen 4). The morning bc7 order is consistent, the morning 65_bc0 order is weak.
- On bc7 the bars are all below the automesher's 0.25.
