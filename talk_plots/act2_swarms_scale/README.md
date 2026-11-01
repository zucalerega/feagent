# Act 2: swarms train better and more stably, and this scales

Regenerate everything: `.venv/bin/python talk_plots/act2_swarms_scale/make_figures.py` (reads `ace_runs/` only, makes no API calls).
Each figure has a PNG (200 dpi), a PDF and a `<stem>.md` with its caption, key numbers, data sources and caveats.
`data.json` holds every plotted value (per run, task and generation: best, mean, n, best agent), and `runs.csv` holds
every individual run (base score stored and recomputed, field error, DOFs, source file).

Score = `checks.base_score`, the field score without bonuses (1.0 = 2x the automesher's accuracy at its DOFs; the automesher
scores 0.25). It is recomputed from `rel_error`, `auto_error`, `dofs` and `auto_dofs` and matches the stored value to 1e-16 in all
runs of the 9 run folders. Only one run is invalid: `scale_s1_opus` gen 1 bc7 (score 0).

| figure | takeaway title |
|---|---|
| `fig1_best_per_gen` (HEADLINE) | On 65_bc0, 8 agents improve almost every generation; 2 agents stall |
| `fig1b_best_per_gen_with_morning` | the same, with the morning runs dashed |
| `fig2_mean_best_by_team_size` | More agents, better swarm: 16 > 8 > 4 > 2 (16 agents: afternoon setup only) |
| `fig4_mean_agent_per_gen` | Not just more draws: the average agent learns too (0.55 → 0.73 with 8 agents) |
| `fig5_swarm_vs_single` | 8 agents beat one budget-matched agent on 65_bc0 and tie on bc7 |
| `fig6_cost_per_gen` | Cost grows sub-linearly: 4x the agents cost 3.3x per generation |

## Best agent vs average agent (the question behind this act)

Per task and generation, "best" = the highest base score among that team's runs on the task, and "mean" = the average over them.
A team of n agents puts n/2 runs on each task, so for 2 agents best = mean.

| afternoon, gens 1-7 | 65_bc0 best | 65_bc0 mean | bc7 best | bc7 mean |
|---|---|---|---|---|
| 2 agents | 0.505 | 0.505 | 0.194 | 0.194 |
| 4 agents | 0.641 | 0.626 | 0.213 | 0.203 |
| 8 agents | 0.731 | 0.685 | 0.230 | 0.223 |

The 8 > 4 > 2 order holds for both measures, on both tasks. The best widens the gap: 8 vs 2 agents on 65_bc0 is +0.226 for the
best and +0.180 for the mean.

## Against the automesher: what "3x" means

On 65_bc0 the best run (div_s8 gen 5, a1) scores 0.753 = 3.0x the automesher's 0.25. In physical terms it reaches 8.4% field
error at 16,713 DOFs, against 24.1% at 8,811 DOFs for the automesher: 2.86x more accurate for 1.90x the DOFs. On bc7 the best is
0.237 (div_s8 gen 5, a2; also gen 6): 16.3% vs 31.9% error (1.95x more accurate) for 4.02x the DOFs. That is below the automesher,
because the score charges for the extra DOFs.

## What the data does not support, and what surprised me

- **bc7:** no run in any of the nine folders beats the automesher (best 0.237). The 8 > 4 > 2 order on bc7 is already there in gen 0,
  before any learning, and the per-generation changes are within +-0.06 (largest +0.058, 2 agents, gen 4 -> 5).
- **Average agent:** the 4-agent swarm's mean agent also ends at 0.727 in gen 7 (8 agents: 0.733). 8 agents get there sooner.
- **Morning single agent vs morning swarm:** on 65_bc0 over gens 1-2 the single Sonnet agent (0.652) is above the morning 8-agent
  swarm (0.633), the opposite of the afternoon result. The swarm leads on bc7 (0.229 vs 0.181). There are only 2 learning
  generations, so this is weak evidence.
- **Morning 65_bc0 ordering:** 0.642 / 0.650 / 0.668 for 2 / 4 / 8 agents, well within noise. The highest single morning 65_bc0
  score was the 2-agent run's 0.717.
- **div_s1 gen 7** resubmitted exactly the gen-6 meshes on both tasks.
- **Cost:** the Opus teacher is the largest cost item in every afternoon run (8 agents: $23.09 teacher vs $12.51 agents). The
  "budget-matched" single agent is matched on solves, turns and time, not on dollars: it cost $13.94 vs $41.45.
- **Colour:** the fixed green (4 agents) and pink (8 agents) are hard to tell apart for deuteranopes (validator dE 4.9). Every
  team size therefore also has its own marker (2 square, 4 triangle, 8 circle, 1 diamond) and a direct label.

One run per configuration throughout. The afternoon README estimates run-to-run noise at 0.03-0.1 per generation.
