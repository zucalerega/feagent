# fig5_curves_vs_test: "Post-trained mesh tracks the real test: 2.0 % off, automesher 7.8 %"

**Caption.** Same conditions, only the mesh differs. gmsh's default mesh is too stiff: its curve runs
6-11 % above the real test (7.8 % off on average). The
most accurate mesh from post-training (g1-abc_00000065_compression-a2, 8,753 nodes) follows the measured curve,
2.0 % off.

**Key numbers** (force at 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm crosshead):
- real test: 860 / 2164 / 3308 / 4171 / 4743 N
- automesher: 910 / 2392 / 3546 / 4462 / 5152 N (+5.9 / +10.5 / +7.2 / +7.0 / +8.6 %)
- post-trained: 870 / 2277 / 3357 / 4195 / 4809 N (+1.2 / +5.2 / +1.5 / +0.6 / +1.4 %)

**Data.** results/real_data_finetune/baselines/auto/curve.csv; ace_runs/nl_ft/gen1/work/abc_00000065_compression-a2/nonlinear/curve.csv.

**Caveats.** The largest remaining gap is at 1.0 mm (+5.2 %), inside the seating toe, which the
simple seating-gap model only approximates; every mesh shares it (uniform 2 mm: +3.9 %). This mesh is the one used
for the Act 3 crack and video pages.
