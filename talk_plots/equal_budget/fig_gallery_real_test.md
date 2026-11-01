# Real test: force curves vs the measurement

The Instron compression test of the printed C-bracket (`physical_tests/abc_00000065_compression/cload_compression.csv`, loading branch to the peak, via `ace/nonlinear.py measured`) against the frozen elastoplastic CalculiX ramp (`ace/nonlinear.py`) on two meshes, from `results/real_data_finetune/baselines/<name>/curve.csv` and `baselines.json` (computed on the Brev box by `scripts/nl_baselines.py`; nothing re-solved here, CalculiX is not installed locally). Error = mean over the 5 checkpoints [0.5, 1.0, 1.5, 2.0, 2.5] mm of |F_sim - F_meas| / F_meas.

- automesher: 2,937 nodes (8,811 DOFs), 7.84 %; per checkpoint ['5.89 %', '10.51 %', '7.17 %', '6.99 %', '8.63 %']
- swarm bc0 mesh (div_s8 g7 a2, trained on simulation only; request `results/real_data_finetune/baselines/swarm_bc0_request.json`): 6,430 nodes (19,290 DOFs), 4.57 %; per checkpoint ['3.27 %', '7.51 %', '3.94 %', '3.45 %', '4.70 %']
- fine-tuned mesh (ace_runs/nl_ft, running on the Brev box): pending; set `FINETUNED_SUMMARY` in `make_gallery.py` to its nonlinear `summary.json` to draw it.

Caveat: this figure does NOT meet the equal-budget rule. The swarm's mesh has 2.19x the automesher's nodes, and the only local mesh with fewer nodes than it is the automesher (uniform 4 mm: 9,506 nodes, 1.99 %, has more). An equal-budget version needs either the fine-tuned mesh at <= the automesher's nodes or the automesher refined to 6,430 nodes through the same ramp (CalculiX, Brev box). The two curves differ by only ~-186 N of ~4966 N at 2.5 mm, so the gap is modest on screen.
