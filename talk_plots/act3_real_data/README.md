# Act 3: the harness learns from real lab data

A 3D-printed abc_00000065 C-bracket was crushed on an Instron. Before post-training the simulation is 33 % off the
real test. Learning the test conditions from the experiment (sticking platens, seating gap, load-train compliance)
and then post-training the swarm on the test (`ace_runs/nl_ft`, 10 generations x 4 agents) bring it to 2.0 %.

Regenerate everything (read-only on inputs, no solver, no API calls; writes PNG 200 dpi + PDF + .md per figure,
act3_data.json and act3_errors.csv):

    .venv/bin/python talk_plots/act3_real_data/make_figures.py

| figure | takeaway title | key numbers |
|---|---|---|
| fig1_measured_curve | Real test: seating, linear climb, yielding to 4.8 kN, then cracking | peak 4755 N at 2.51 mm (4.80 kN at 2.56 mm per the report), linear 2.70 kN/mm, break 8.68 mm |
| fig2_raw_solver_vs_test | Before post-training, the simulation is 33 % off the real test | frictionless platens, automesher mesh, no seating/compliance: 32.65 %, 3.3 kN vs 4.7 kN at 2.5 mm |
| fig3_setup_ladder | From 33 % to 2.0 % off the real test: test conditions, then post-training | 32.7 -> 7.84 (learned conditions, default mesh) -> 4.57 (pre-trained mesh) -> 1.97 % (post-trained mesh) |
| fig4_error_vs_nodes | Post-trained meshes: 2.0-2.3 % off the real test, automesher 7.8 % | all 40 runs by generation; most accurate 1.97 % at 8,753 nodes; best-scoring 2.30 % at 5,612; uniform 4 / 2.5 / 2 mm 1.99 / 1.20 / 1.08 % |
| fig5_curves_vs_test | Post-trained mesh tracks the real test: 2.0 % off, automesher 7.8 % | automesher 6-11 % above the test; post-trained most accurate mesh (g1 a2) |
| fig6_platen_renders | Platen friction carries the strength: without it the prongs splay | team_slide F_max 2.92 kN; team_stick 4.90 kN at the test's peak travel |
| fig7_post_training_progress | Post-training: best score 2.7x -> 3.6x the automesher, average 2.3x -> 3.5x | best base score 0.668 -> 0.892 (g7), mean 0.572 -> 0.880; most accurate mesh per generation 1.97-2.60 % |

## Inputs (all local)

- test: physical_tests/abc_00000065_compression/cload_compression.csv; spec nonlinear_spec.json (delta 0.2335 mm,
  compliance 2.898e-5 mm/N)
- before post-training: videos/before_run/summary.json and curve.csv
- fair baselines: results/real_data_finetune/baselines/baselines.json and <auto|swarm_bc0|h4|h25|h20>/curve.csv
  (automesher = curved P2, only the midside nodes CalculiX rejects straightened)
- post-training run: ace_runs/nl_ft/gen*/runs/*.json and gen*/work/<pid>-a<k>/nonlinear/curve.csv
- renders: physical_tests/abc_00000065_compression/ccx_ramp/

`_superseded/` holds the old fig7 placeholder and plot_finetune_run.py, which the run-based fig7 replaces.

## Things to be careful about in the talk

- Wording: "before / after post-training" (the solvers and the swarm were already capable); "X % off the real
  test"; the scoring checkpoints are not shown or mentioned.
- The automesher is 7.8 % off; an earlier, higher automesher figure straightened every midside node and was wrong.
- Post-training gains nodes more than accuracy: 1.97 % was reached in generation 0; later meshes are 2.2-2.6 %
  with about 30 % fewer nodes. Uniform 4 mm is 1.99 % at 9,506 nodes.
- delta and compliance were fitted on this specimen; one specimen, one run.
- The real crack was at the lower inner bore (team observation), where the post-trained simulation's tension
  peaks (43 MPa, 85 % of the dogbone strength); the before simulation predicted a prong-top crack 1.6 kN early
  (break/break.json).
