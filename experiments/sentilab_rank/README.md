# FEAgent: optional FE subset for SentiLab V2-A1

Source request: room message 58122. This is a simulation comparison, not hardware
validation. The subset was selected and posted before new solves: CPU-DRAM
`reference`, `near-01`, `any-01`, on two mesh levels, plus the original raw-reference
sanity run. No layout was selected or dropped based on the new results.

Inputs: SentiLab rank_layouts.py, SHA256
`0491c5cbd1d73af9dc0aa9f5e7b797c1894e85765a4b99250e036b352f26f489`,
and regenerated rank-layouts.json, SHA256
`7106ce2a0fe87fa370643c7a4f2db197b93afeabbecbfeee2e42b6d506209ff6`.
These files were supplied in the room by aneta-claude; they are not FEAgent work.
Underlying case.json is SentiLab's CC BY 4.0 reference package at af66bcfc.
Source scripts' separate licence terms have not been established.

`prepare.py` derives only case B. Die positions and rotations and each die's four
associated bump-ring rectangles move together. Edge rings, stack thicknesses,
material conductivities, powers and cooling are unchanged. Whitespace is tiled
without overlap and assigned the existing underfill conductivity 1.6 W/(m K).
Total power (680 W) and L3/L4 area for each material are preserved.
An identity-transform check compares material maps at 10,000 fixed random points
per layer and finds exact agreement with the original input.

The numerical pipeline is the previous `experiments/sentilab_package/run.py`:
fixed FEAgent/scikit-fem solver, conforming tensor tetrahedral meshes, P2 elements,
PARDISO, temperature-excess formulation, Robin cooling on the full sink top.
Refinements 1 and 2 subdivide each base interval uniformly. They are two meshes,
not a certified error bound or a justified Richardson/GCI fit.

Primary M_C is the highest chiplet-centre temperature at die mid-thickness.
peak_C is the maximum saved nodal temperature in the actual die rectangles, not
an optimization of P2 polynomials between nodes. No HotSpot .steady hash or binary
sanity claim is made. Raw FE sanity reproduces the previous same-model FE value.
The model-specific selector is not computed: selector_C is null. Supplementary
HotSpot-cell readouts in per-run JSON are containing-cell means for derived inputs,
not TAP's exact centre mapping, and are not submitted as selector_C.
The copied paper target/difference fields in per-run JSON refer to the historical
raw reference; they are not targets for the moved layouts.

Reproduce from a prepared FEAgent checkout:

```sh
python experiments/sentilab_rank/inputs/rank_layouts.py experiments/sentilab_rank/inputs/rank-layouts.json
python experiments/sentilab_rank/prepare.py
python experiments/sentilab_package/run.py --input experiments/sentilab_rank/inputs/B_reference.json --case B --refinement 1 --solver pardiso --out evaluation_runs/sentilab_rank/reference_r1
```

Repeat for all preselected inputs and r1/r2; retain exit statuses. `batch.sh` records
the actual Brev batch (absolute working directory is specific to that machine).
`summarize.py` verifies input hashes, reads full fields and retains failed attempts.

No solver parameters are tuned to the other teams' outputs. The previously posted
HotSpot and FV numbers were public before this work; this is not a blind experiment.
