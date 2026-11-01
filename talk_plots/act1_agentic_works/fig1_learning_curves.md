# fig1_learning_curves: "Training works: the mean agent climbs 1.3x → 4.4x the automesher (Cook)"

**Title numbers.** 1.3x = gen-0 mean 0.318 / 0.25; 4.4x = gen-10 mean 1.095 / 0.25 (the best agent's 1.126 = 4.5x is labelled on the curve).

**Caption.** Best and mean base score per generation in the deep loop (deep1: 8 Sonnet agents split 4 per task, Opus teacher, self-reflection, organizer). The grey dashed line is the gmsh automesher's own score, 0.25. Each generation is one training step.

**Key numbers** (all in `act1_numbers.json`)
- Cook's membrane: the mean agent goes 0.318 -> 1.095 (gen 0 -> gen 10), and the best agent goes 0.347 -> 1.126. 1.126 / 0.25 = 4.50x the automesher's score.
- Cook's accuracy gain g = e_auto / e: the gen-0 mean is 1.10; the gen-10 mean is 4.68, at a mean 2.14x the automesher's DOFs.
- Cook's best run, g10-cooks_membrane-a7: error 1.07% vs the automesher's 5.11%, so g = 4.76 at 710 / 336 = 2.11x its DOFs.
- LE11 axisymmetric: the best agent goes 0.001 -> 0.880 and the mean goes 0.001 -> 0.845. The best is 3.52x the automesher.
- On Cook the best agent dips twice (gen 2->3: 0.983 -> 0.937; gen 4->5: 0.970 -> 0.949) and otherwise rises. The mean rises in 8 of 10 steps on Cook (two small dips: gen 3->4 by 0.001, gen 4->5 by 0.020).
- Cost of the whole run: $46.37. The teacher took $30.90 of it (`ace_runs/deep1/summary.md`).

**Sources.** `ace_runs/deep1/gen*/runs/*.json` -> `checks.base_score`, `gain`, `rel_error`, `dofs`, `auto_error`, `auto_dofs` (44 runs per task, none invalid).

**Caveats**
- The LE11 axisymmetric jump from gen 0 to gen 1 is a **setup fix, not meshing**. All 4 gen-0 agents wrote the temperature field as sqrt(x^2+y^2)+y instead of x+y (r+z). That gave about 30% error whatever the mesh (`gen0/teacher_nafems_le11_axisym.json`). The mesh-only learning on that task is the gen 1 -> 10 rise: best agent 0.706 -> 0.880, mean 0.678 -> 0.845.
- Cook is mesh-only learning: the gen-0 setups were already valid (g ≈ 1).
- 4 agents per task per generation, and one run (no seeds or repeats).
- deep1 was stopped by hand (STOP file) after generation 10.
- "Base score" leaves out the be-the-best bonus.
