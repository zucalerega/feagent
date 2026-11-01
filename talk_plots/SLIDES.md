# Suggested slide order

One line per slide: the figure, then the takeaway sentence to say. Numbers and caveats are in [README.md](README.md)
and in each figure's `.md`.

## Act 1: agentic meshing works, and training works

1. (title / setup, no figure): A swarm of LLM agents meshes FE problems; a fixed solver scores each mesh against the gmsh automesher, and a teacher plus reflections update a shared playbook every generation.
2. `act1_agentic_works/fig4_mesh_before_after.png`: Same part, same solver: after training, the agent grades the mesh into the corners (5.09% -> 1.07% error).
3. `act1_agentic_works/fig1_learning_curves.png`: Training works: on Cook's membrane the mean agent climbs 0.318 -> 1.095 over 10 generations, 4.4x the automesher's score.
4. `act1_agentic_works/fig3b_equal_cost_equal_accuracy.png`: Cost-fair: 3.4x lower error at the same DOFs, or the same error with 10.8x fewer DOFs.
5. `act1_agentic_works/fig3_headline_scores.png`: The best agent beats the automesher on every benchmark (1.6-4.5x), with no real gain on LE11 3D.
6. (backup) `act1_agentic_works/fig2_error_vs_dofs.png`: In 2D the best agents sit below the automesher's whole refinement curve; in 3D they match it.
7. (backup) `act1_agentic_works/fig1b_learning_curves_all_runs.png`: Without the teacher, the basic loop plateaus or drifts; the deep loop is what learns.

## Act 2: swarms train better and more stably, and this scales

8. `act2_swarms_scale/fig1_best_per_gen.png`: On a real CAD part, the 8-agent swarm's best mesh improves almost every generation (0.651 -> 0.745); 2 agents stall and fall back.
9. `act2_swarms_scale/fig2_mean_best_by_team_size.png`: More agents, better swarm: 8 > 4 > 2 in all four study-task comparisons.
10. `act2_swarms_scale/fig5_swarm_vs_single.png`: 8 agents beat one agent given the whole swarm's budget on 65_bc0 (0.731 vs 0.592) and tie on bc7.
11. `act2_swarms_scale/fig4_mean_agent_per_gen.png`: Not just more lottery tickets: the average agent learns too (0.55 -> 0.73 with 8 agents).
12. `act2_swarms_scale/fig6_cost_per_gen.png`: Scaling is affordable: 4x the agents cost 3.3x per generation.
13. (backup) `act2_swarms_scale/fig1b_best_per_gen_with_morning.png`: The shorter morning runs, dashed: same picture, weaker evidence.

## Act 3: the harness learns from real lab data

14. `act3_real_data/fig1_measured_curve.png`: We 3D-printed the part and crushed it on an Instron: seating, a linear climb, yielding to 4.8 kN, then cracking.
15. `act3_real_data/fig2_raw_solver_vs_test.png`: The raw simulation misses reality: frictionless platens give out at 3.0 kN, against 4.74 kN measured at 2.5 mm.
16. `act3_real_data/fig6_platen_renders.png`: Why: without platen friction the prongs splay; with sticking platens the part carries the load.
17. `act3_real_data/fig3_setup_ladder.png`: Learning the setup from the experiment cuts the error from 33% to 2.0% on the same mesh; a fine mesh then reaches 1.1%.
18. `act3_real_data/fig4_error_vs_nodes.png`: With the setup frozen, the mesh is what's left: the automesher is 16.6% off the real test, and a swarm mesh is already at 4.5%.
19. `act3_real_data/fig5_curves_vs_test.png`: The automesher's mesh is too stiff, 14-19% above the test at every checkpoint.
20. `act3_real_data/fig7_finetune_progress_STUB.png`: PENDING: fine-tuning the swarm against the real test (replace with `fig7_finetune_progress.png` once the run exists).

## Close

21. (no figure, or reuse fig1 of each act side by side): Agents learn to mesh, swarms learn better and scale with compute, and the loop can be closed on real experiments, which is what matters in the end.
