# Suggested slide order

One line per slide: the figure, then the takeaway sentence to say. Numbers and caveats are in [README.md](README.md)
and in each figure's `.md`.

## Act 1: agentic meshing works, and training works

**KEY ACT 1 EXAMPLE (chosen by the user):** `equal_budget/fig_cook_equal_stress.png` (solved von Mises stress, Cook's membrane, equal DOF budget): automesher refined to 734 DOFs, 3.63 % error | early agent 898 DOFs, 2.90 % | trained agent (deep1 g10 a7) 710 DOFs, 1.07 % | true solution. The trained agent wins on both accuracy and DOFs: 3.4x more accurate than the automesher at the same budget. Companion slide: `equal_budget/fig_cook_same_budget.png` (error maps, "Same budget, 3.4x more accurate").

1. (title / setup, no figure): A swarm of LLM agents meshes FE problems; a fixed solver scores each mesh against the gmsh automesher, and a teacher plus reflections update a shared playbook every generation.
2. `act1_agentic_works/fig4_mesh_before_after.png`: Same part, same solver: after training, the agent grades the mesh into the corners (5.09% -> 1.07% error).
3. `act1_agentic_works/fig1_learning_curves.png`: Training works: on Cook's membrane the mean agent climbs 0.318 -> 1.095 over 10 generations, 4.4x the automesher's score.
4. `act1_agentic_works/fig3b_equal_cost_equal_accuracy.png`: Cost-fair: 3.4x lower error at the same DOFs, or the same error with 10.8x fewer DOFs.
5. `act1_agentic_works/fig3_headline_scores.png`: The best agent beats the automesher on every benchmark (1.6-4.5x), with no real gain on LE11 3D.
6. (backup) `act1_agentic_works/fig2_error_vs_dofs.png`: In 2D the best agents sit below the automesher's whole refinement curve; in 3D they match it.
7. (backup) `act1_agentic_works/fig1b_learning_curves_all_runs.png`: Without the teacher, the basic loop plateaus or drifts; the deep loop is what learns.

## Act 2: swarms train better and more stably, and this scales

8. `act2_swarms_scale/fig1_best_per_gen.png`: On a real CAD part, the 8-agent swarm's best mesh improves almost every generation (0.651 -> 0.745); 2 agents stall and fall back.
9. `act2_swarms_scale/fig3_scaling_bc0_{max,max6,bestmean,avg}.png`: More agents, better swarm (2 / 4 / 8 / 16 agents; curve c - d n^-1/2 through the bars):
   - max (best score reached): 2.40x / 2.97x / 3.01x / 3.18x; ~3.34x at 32, ~3.42x at 64.
   - max6 (best score reached, gens 0-5 only, the tuned window): 2.40x / 2.56x / 3.01x / 3.10x; ~3.22x at 32, ~3.31x at 64. Its projection checked against the real gens 6-7: 16 agents 3.14 / 3.18 projected vs 3.15 / 3.18 measured; 2 exact; 8 slightly over (3.07 vs 3.01 at gen 7); 4 under (2.60 vs 2.97).
   - bestmean (best agent, mean over gens 1-7; the original fig2 numbers): 2.02x / 2.56x / 2.93x / 3.01x; ~3.25x at 32, ~3.37x at 64.
   - avg (average agent after training): 2.36x / 2.91x / 2.93x / 3.03x; ~3.19x at 32, ~3.27x at 64.
9b. `act2_swarms_scale/fig7_training_bc0_{max,max6,bestmean,avg}.png`: the matching training curves (gen 7 = the bars); projection: each team's
   log-rate fitted on all 8 generations, continued from its gen-7 value to gen 15. Gen 15 (2 / 4 / 8 / 16):
   - max: 2.40x / 3.17x / 3.10x / 3.37x  - bestmean: 2.17x / 2.85x / 3.00x / 3.17x  - avg ("in bigger swarms, every agent learns faster"): 2.27x / 3.11x / 3.17x / 3.30x.
   (backup) `act2_swarms_scale/fig2_mean_best_by_team_size.png`: More agents, better swarm: 16 > 8 > 4 > 2 (8 > 4 > 2 in all four study-task comparisons; the 16-agent run, afternoon setup only, beats 8 on both tasks).
10. `act2_swarms_scale/fig5_swarm_vs_single.png`: 8 agents beat one agent given the whole swarm's budget on 65_bc0 (0.731 vs 0.592) and tie on bc7.
11. `act2_swarms_scale/fig4_mean_agent_per_gen.png`: Not just more lottery tickets: the average agent learns too (0.55 -> 0.73 with 8 agents).
12. `act2_swarms_scale/fig6_cost_per_gen.png`: Scaling is affordable: 4x the agents cost 3.3x per generation.
13. (backup) `act2_swarms_scale/fig1b_best_per_gen_with_morning.png`: The shorter morning runs, dashed: same picture, weaker evidence.

## Act 3: the harness learns from real lab data

Say "before post-training" / "after post-training" (the solvers and the swarm were already capable) and "X % off
the real test"; do not mention the scoring checkpoints.

14. `act3_real_data/fig1_measured_curve.png`: We 3D-printed the part and crushed it on an Instron: seating, a linear climb, yielding to 4.8 kN, then cracking.
15. `act3_real_data/fig2_raw_solver_vs_test.png`: Before post-training, the simulation is 33% off the real test: it tops out at 3.3 kN, the part held 4.7 kN.
16. `act3_real_data/fig6_platen_renders.png`: Why: without platen friction the prongs splay; with sticking platens the part carries the load.
17. `act3_real_data/fig3_setup_ladder.png`: 33% -> 7.8% with the learned test conditions on the default mesh -> 4.6% with the pre-trained swarm mesh -> 2.0% with the post-trained mesh.
18. `act3_real_data/fig7_post_training_progress.png`: Post-training on the real test: the best agent goes from 2.7x to 3.6x the automesher's score, the average agent from 2.3x to 3.5x, mostly by reaching about 2% with fewer nodes.
19. `act3_real_data/fig4_error_vs_nodes.png`: Every post-training mesh vs the references: the most accurate is 2.0% off at 8,753 nodes, the best-scoring 2.3% at 5,612; the automesher is 7.8% off.
20. `act3_real_data/fig5_curves_vs_test.png`: The post-trained mesh tracks the real curve (2.0% off); the automesher's mesh runs 6-11% too stiff.
21. `act3_real_data/videos.html` and `act3_real_data/break/index.html` (most accurate post-trained mesh): the real part cracked at the lower inner bore, where the after-post-training simulation's tension peaks (43 MPa, 85% of the dogbone strength); the before simulation predicted a prong-top crack 1.6 kN early.

## Close

22. (no figure, or reuse fig1 of each act side by side): Agents learn to mesh, swarms learn better and scale with compute, and the loop can be closed on real experiments, which is what matters in the end.
