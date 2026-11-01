# fig6_cost_per_gen: cost grows sub-linearly with team size

**Caption.** API dollars spent per generation by each afternoon run. Going from 2 to 8 agents multiplies the mean
cost per generation (gens 0-6) by 3.31, not 4 (2 -> 4 agents: 1.71x; 4 -> 8: 1.94x).
Both the agents' spend and the Opus teacher's spend grow less than linearly (table).

**Key numbers (afternoon runs; mean $/gen over gens 0-6, gen 7 excluded because the run ends before its teacher step).**

| agents | total $ | mean $/gen (0-6) | agents (generator) $ | teacher (Opus) $ | reflect $ | organizer $ |
|---|---|---|---|---|---|---|
| 1 | $13.94 | $1.91 | $3.96 | $8.33 | $0.56 | $1.09 |
| 2 | $12.76 | $1.71 | $3.47 | $7.49 | $0.54 | $1.26 |
| 4 | $21.34 | $2.92 | $6.05 | $12.21 | $1.10 | $1.98 |
| 8 | $41.45 | $5.67 | $12.51 | $23.09 | $2.23 | $3.61 |

Morning totals: 1 agent $3.72, 2 agents $8.69, 4 agents $15.52, 8 agents $27.41;
single Opus $3.60. Afternoon total $89.49.

**Data.** `ace_runs/div_s{1,2,4,8}/cost.jsonl` (`usd` per call). A generator call carries its generation in `run_id`;
reflection, teacher and organizer calls (empty `run_id`) are assigned to the latest generation started. This attribution
reproduces `ace_runs/div_s1/generations.jsonl` and the cumulative $ in `ace_runs/div_s8/summary.md` exactly.

**Caveats.**
- The 1-agent row is the single agent (div_s1, one conversation for both tasks); it costs slightly more per generation
  than the 2-agent swarm.
- "Sub-linear" is from one run per size and 7 generations; it is not a cost model.
- The 8-agent swarm's teacher spend ($23.09) is larger than its agents' spend
  ($12.51).
