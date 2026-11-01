# fig3_scaling_bc0_bestmean: 16 > 8 > 4 > 2, extrapolated

**Caption.** abc_00000065_bc0, afternoon setup (diversity on), automesher = 1 (scores divided by 0.25). Bars: the gen-7 value of
`bestmean` = running mean of the best agent per generation (gens 1..g) (fig7_training_bc0_bestmean ends on these bars). Dashed: y = c - d n^-1/2 fitted to the four bars
(c = 3.65, d = 2.25); 32 agents 3.25x, 64 agents 3.37x (no swarm of that size was run).

**Key numbers.** 2 / 4 / 8 / 16 agents: 2.02x / 2.56x / 2.93x / 3.01x. Fit residuals: -0.04 / +0.04 / +0.07 / -0.07.

**Data.** 2/4/8 agents: `ace_runs/div_s{2,4,8}/gen*/runs/g*-abc_00000065_bc0-*.json` (`checks.base_score`). 16 agents:
`results/diversity_s16_2026-11-01/scores.json` (per-generation best and mean exported by Jaime from his run).

**Caveats.**
- One run per team size (noise 0.03-0.1 in score, 0.1-0.4 on this axis); a 2-parameter fit to 4 points. The exponent
  1/2 is chosen; a free exponent fits a faster plateau.
- The 16-agent run's references were rebuilt (see fig2).
