# fig7_post_training_progress: "Post-training: best score 2.7x → 3.6x the automesher, average 2.3x → 3.5x"

**Caption.** Post-training the swarm on the real test, 10 generations of 4 agents. Left: the best agent's score
rises from 2.67x to 3.57x the automesher (peak in generation 7) and the
average agent's from 2.29x to 3.52x: every agent, not just the best, ends well above the
uniform 4 mm mesh (2.44x) and the pre-trained swarm mesh
(1.34x). Right: the most accurate mesh is already at 1.97 % in
generation 0; later generations keep the error near 2.2-2.6 % while cutting the nodes from 8.8k to about 6k,
which is what the score rewards.

**Score** (checks.base_score, field-mode formula against the automesher's run of the same frozen ramp): g =
e_auto / e, accuracy = (g/2)^2 below g = 2 and g/2 above, score = accuracy x automesher DOFs / own DOFs. The
automesher scores 0.25 (= 1x here).

| gen | best base score | mean base score | most accurate mesh |
|---|---|---|---|
| 0 | 0.668 (2.67x, g0-a2) | 0.572 (2.29x) | 1.97 % at 8,753 |
| 1 | 0.714 (2.85x, g1-a0) | 0.684 (2.74x) | 1.97 % at 8,753 |
| 2 | 0.729 (2.92x, g2-a2) | 0.691 (2.76x) | 2.60 % at 6,132 |
| 3 | 0.797 (3.19x, g3-a2) | 0.753 (3.01x) | 2.51 % at 5,743 |
| 4 | 0.816 (3.26x, g4-a3) | 0.802 (3.21x) | 2.40 % at 6,017 |
| 5 | 0.827 (3.31x, g5-a1) | 0.809 (3.24x) | 2.40 % at 5,814 |
| 6 | 0.846 (3.38x, g6-a2) | 0.831 (3.32x) | 2.42 % at 5,697 |
| 7 | 0.892 (3.57x, g7-a1) | 0.850 (3.40x) | 2.30 % at 5,612 |
| 8 | 0.890 (3.56x, g8-a3) | 0.880 (3.52x) | 2.21 % at 5,981 |
| 9 | 0.890 (3.56x, g9-a2) | 0.880 (3.52x) | 2.21 % at 5,981 |

**Data.** ace_runs/nl_ft/gen*/runs/*.json (checks.base_score, checks.nonlinear.mean_rel_err and
nodes); references from results/real_data_finetune/baselines/baselines.json (score, mean_rel_err).

**Caveats.**
- One run, no repeats. The gain is mostly in node cost: the error of the most accurate mesh does not fall after
  generation 0-1 (1.97 %), it stays near 2.2-2.6 % with about 30 % fewer nodes.
- Generations 0 and 1 contain the same 8,753-node mesh (g0 a2, g1 a1, g1 a2).
