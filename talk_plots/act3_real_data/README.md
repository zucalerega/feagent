# Act 3: the harness learns from real lab data

A 3D-printed abc_00000065 C-bracket was crushed on an Instron. The raw nonlinear simulation is far off. Analysing
the experiment finds the missing physics (sticking platens, a seating gap, load-train compliance). With that setup
frozen, the error against the REAL test is set by the mesh, and that is what the swarm fine-tunes.

Regenerate everything (read-only on inputs, no solver, no API calls):

    .venv/bin/python talk_plots/act3_real_data/make_figures.py
    .venv/bin/python talk_plots/act3_real_data/plot_finetune_run.py ace_runs/<run>    # after the fine-tuning run

| figure | takeaway title | key numbers |
|---|---|---|
| fig1_measured_curve | Real test: seating, linear climb, yielding to 4.8 kN, then cracking | toe 1.66 kN/mm, linear 2.70 kN/mm, peak 4755 N at 2.51 mm (4.80 kN at 2.56 mm per the report), break 8.68 mm |
| fig2_raw_solver_vs_test | The raw solver misses reality: frictionless platens give out at 3.0 kN | test 4.74 kN at 2.5 mm; frictionless 3.0 kN, 33.2 %; sticking 5.1 kN, 35.6 % raw |
| fig3_setup_ladder | Learning the setup: error 33 % → 2.0 %; a fine mesh: 1.1 % | 33.2 -> 20.2 (seating) -> 5.0 (sticking) -> 2.0 (compliance) -> 1.07 % (2 mm mesh) |
| fig4_error_vs_nodes | With the learned setup, the automesher is 17 % off the real test | auto 2937 nodes 16.6 %; swarm bc0 6430 nodes 4.5 %; uniform 4 / 2.5 / 2 mm 2.0 / 1.2 / 1.1 %; target box <= 2 % at <= 8k nodes (empty); the grey band is the finest mesh's 1.1 %, roughly the model-error floor |
| fig5_curves_vs_test | The automesher's mesh is too stiff: 14-19 % above the real test | per-checkpoint forces in the md |
| fig6_platen_renders | Platen friction carries the strength: without it the prongs splay | team_slide F_max 2.92 kN; team_stick 4.90 kN at the test's peak travel |
| fig7_finetune_progress_STUB | SLOT: fine-tuning the swarm on the real test (run pending) | placeholder; filled by plot_finetune_run.py |

Each figure has <stem>.png (200 dpi), <stem>.pdf and <stem>.md (caption, numbers, sources, caveats).
act3_data.json holds every plotted number and curve; act3_errors.csv the error table.
render_team_slide.png / render_team_stick.png are unchanged copies of the teammates' renders.

## Error definition (= ace.nonlinear.run_curve, frozen in physical_tests/abc_00000065_compression/nonlinear_spec.json)

u_cross = u_part + delta + c F (delta 0.2335 mm, c 2.898e-5 mm/N, fitted once on the 2 mm mesh), (0, 0) prepended;
error = mean over the checkpoints 0.5, 1.0, 1.5, 2.0, 2.5 mm of |F_sim - F_test| / F_test, F_test on the loading
branch of physical_tests/abc_00000065_compression/cload_compression.csv up to its force maximum.

## Inputs

- test: physical_tests/abc_00000065_compression/cload_compression.csv (+ cload_compression.json notes)
- BC variants, 4 mm P2 mesh: /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_bc/<variant>/curve.csv
- mesh sweep, sticking setup: /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_mesh/<auto|swarm|h4|h25|h20>/curve.csv, nodes.txt
- renders and teammates' ramps: physical_tests/abc_00000065_compression/ccx_ramp/
- cross-check only: results/real_data_finetune/baselines/baselines.json (agrees within 0.03 points)

The scratchpad inputs are temporary: if they are gone, make_figures.py cannot regenerate; act3_data.json keeps
every number and curve that was plotted.

## Things to be careful about in the talk

- Order of the ladder: sticking alone, without the seating gap, is WORSE than frictionless (35.6 % vs 33.2 %).
  The monotone ladder needs seating first.
- delta and c are calibrated on the same test the mesh is scored against: the 1.1 % floor is a fit, not a blind
  prediction, and there is one specimen.
- The swarm bc0 mesh (4.5 %) was trained for a different load case; it is a transfer result, not fine-tuning.
- The fine-tuning run has not happened: fig4's target box and fig7 are placeholders.
- The numbers quoted in the brief (16.8 / 4.8 / 2.2 / 1.4 / 1.1 %) do not reproduce; the frozen spec gives
  16.6 / 4.5 / 2.0 / 1.2 / 1.1 %.
