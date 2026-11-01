# fig3_scaling_bc0: 16 > 8 > 4 > 2, extrapolated

**Caption.** Talk version of fig2 for abc_00000065_bc0 only, afternoon setup (diversity on), scores divided by the
automesher's 0.25 so that the automesher is 1. Bars: the best agent's base score averaged over gens 1-7.
Dashed: least-squares fit y = 1.794 + 0.335 log2(n) to the four measured bars, extended to 32 and 64 agents
(projected 3.47x and 3.80x; no swarm of that size was run).

**Key numbers.** 2 / 4 / 8 / 16 agents: 2.02x / 2.56x / 2.93x / 3.01x the automesher.
Fit residuals: -0.11 / +0.10 / +0.13 / -0.12.

**Data.** As fig2 (16 agents from `results/diversity_s16_2026-11-01/scores.json`). Plotted values: `data.json` -> `fig3`.

**Caveats.**
- The extrapolation is a fit to 4 points, one run each (noise 0.03-0.1 in score, 0.1-0.4 on this axis).
- The measured gains shrink: +0.54, +0.36, +0.09 per doubling. The residuals
  run -, +, +, -, i.e. the data bend below a log curve; a saturating fit (c - d n^-alpha, 3 parameters) fits them as well and
  levels off near 3.2x. The log fit is the optimistic reading.
- The 16-agent run's references were rebuilt (see fig2), and team size n means n/2 agents per task, so a bigger
  team also takes the best of more meshes.
