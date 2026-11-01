# fig4_error_vs_nodes: "With the learned setup, the automesher is 17 % off the real test"

**Caption.** With the setup frozen, the mesh decides the error against the real test: gmsh's default mesh is
16.6 % off; uniform refinement levels off at 1.1 % on the finest mesh, roughly the model-error floor (labelled as such in the figure). The swarm's best mesh from the gold
training (bc0, a different load case) is already at 4.5 % with 6.4k nodes. The dashed box is the fine-tuning
target, to be filled by the fine-tuning run.

**Key numbers** (frozen spec, mean error at the 5 checkpoints):

| mesh | nodes | error vs test | per checkpoint (%) |
|---|---|---|---|
| gmsh default (automesher) | 2937 | 16.6 % (16.58) | 14.3 / 19.4 / 16.0 / 15.8 / 17.5 |
| swarm's best bc0 mesh (div_s8 g7 a2) | 6430 | 4.5 % (4.55) | 3.2 / 7.4 / 4.0 / 3.5 / 4.7 |
| uniform 4 mm | 9506 | 2.0 % (1.97) | 1.2 / 5.1 / 1.5 / 0.6 / 1.4 |
| uniform 2.5 mm | 24194 | 1.2 % (1.17) | 0.3 / 4.2 / 0.5 / 0.4 / 0.3 |
| uniform 2 mm | 41186 | 1.1 % (1.07) | 0.0 / 3.8 / 0.0 / 1.1 / 0.4 |

Accuracy gain over the automesher: swarm bc0 mesh 16.58 / 4.55 = 3.6x at 2.2x the nodes; uniform 4 mm 8.4x at
3.2x the nodes.

**Data.** /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_mesh/<auto|swarm|h4|h25|h20>/curve.csv and /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_mesh/nodes.txt;
delta, c, checkpoints from physical_tests/abc_00000065_compression/nonlinear_spec.json.
Cross-check: the committed results/real_data_finetune/baselines/baselines.json (re-solved to u_part 2.4 mm, the
spec's ramp) gave 16.60 / 4.57 / 1.99 / 1.20 / 1.08 % when checked (that file is being rewritten by another workflow; its h20 value moved from 1.076 % to 1.0755 % during this check), within 0.03 points of mine.

**Caveats.**
- The target box (<= 2 % at <= 8k nodes) is a goal, not a result. Nothing in it is measured yet.
- The swarm's bc0 mesh was trained for a one-sided load on face f7 (bc0), not for this symmetric crush; it is a
  transfer result, not a fine-tuned one.
- The "floor" is the finest mesh's error (1.07 %); a still finer mesh could move it slightly. Below it, error
  differences are model error, not mesh error.
- The task brief quoted auto 16.8 %, swarm 4.8 %, h4 2.2 %, h25 1.4 %, h20 1.1 %. I could not reproduce those:
  the frozen spec gives 16.6 / 4.5 / 2.0 / 1.2 / 1.1 %, and the alternative definition (checkpoint 2.556 mm,
  the other digitization, delta 0.233, c 0.29e-4) gives 16.6 / 4.5 / 1.9 / 1.1 / 1.1 %. The figure uses the
  frozen spec.
- The teacher brief says every mesh sits above the test at every checkpoint; the 2 mm mesh is 1.1 % BELOW it at 2.0 mm.
