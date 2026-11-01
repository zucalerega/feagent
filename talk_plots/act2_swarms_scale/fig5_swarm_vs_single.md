# fig5_swarm_vs_single: 8 agents beat one budget-matched agent on 65_bc0 and tie on bc7

**Caption.** The afternoon 8-agent swarm (solid pink) against a single Sonnet agent that gets the whole swarm's solve
and turn budget and its wall-clock time per generation (solid blue). Shorter morning runs dashed: the 8-agent swarm, the
single Sonnet agent and a single Opus agent. API cost in the legend.

**Key numbers.**
- Mean of per-generation best, gens 1-7 (afternoon): abc_00000065_bc0 8 agents 0.731 vs single 0.592;
  abc_00000003_bc7 8 agents 0.230 vs single 0.218.
- abc_00000065_bc0 max: 8 agents 0.753, single 0.686.
- abc_00000003_bc7 from gen 3 on: single 0.229-0.233, 8 agents 0.224-0.237: the single agent matches the swarm there.
- Morning, gens 1-2 (all morning runs have them): 65_bc0 8 agents 0.633 vs single Sonnet 0.652;
  bc7 0.229 vs 0.181. Single Opus per gen (bc7 / 65_bc0):
  gen 0: 0.203 / 0.598; gen 1: invalid / 0.641; gen 2: 0.196 / 0.539.
- API cost: 8 agents afternoon $41.45, single afternoon $13.94; morning 8 agents $27.41,
  single Sonnet $3.72, single Opus $3.60.

**Data.** `ace_runs/div_s8`, `ace_runs/div_s1`, `ace_runs/scale_s8`, `ace_runs/scale_s1`, `ace_runs/scale_s1_opus`
(`gen*/runs/*.json` checks; `cost.jsonl`). Budget matching: `ace_runs/run_div_s1.sh`, `ace/single_loop.py`
(per task MAX_SOLVES x swarm agents per task = 16 solves, MAX_TURNS x 8 turns per generation, time-matched to div_s8).

**Caveats.**
- "Budget-matched" means solves, turns and wall-clock time, not dollars: the single agent cost about 1/3 of the swarm
  ($13.94 vs $41.45); the swarm's cost is dominated by the Opus teacher (see fig 6).
- The single agent is not stronger per call: div_s1 and scale_s1 use Sonnet like the swarm agents (high effort);
  only scale_s1_opus uses Opus, and it ran 3 generations, with an invalid bc7 mesh in gen 1.
- div_s1 gen 7 resubmitted exactly the gen-6 meshes on both tasks (same DOFs and error).
- On bc7 the single agent catches up with the swarm from gen 3; the swarm's advantage is clear only on 65_bc0.
- The morning runs point the other way on 65_bc0: over gens 1-2 the single Sonnet agent (0.652)
  is above the morning 8-agent swarm (0.633), while the swarm leads on bc7 (0.229 vs
  0.181). Only 2 learning generations, so this is weak evidence either way, but the
  afternoon result is not replicated by the morning pair.
- One run per configuration; the morning runs are short (3 generations for the single agents).
