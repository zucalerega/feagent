# FEAgent optional FE subset: SentiLab V2-A1

Seven preselected runs completed: raw case-B sanity, then reference/near-01/any-01 on two meshes. No failed runs. CPU-DRAM only; this is not a 50-layout ranking experiment.

| Layout | M r1 (C) | M r2 (C) | r2-r1 (K) | r2 peak in dies (C, nodal) | r2 DOFs |
|---|---:|---:|---:|---:|---:|
| reference | 122.38462177 | 122.29583499 | -0.08878677 | 125.96477560 | 513553 |
| near-01 | 117.27158637 | 117.23725008 | -0.03433629 | 118.69558111 | 1309369 |
| any-01 | 111.52374823 | 111.48374251 | -0.04000572 | 111.93831604 | 1434521 |

At r2, relative to the snapped reference: near-01 -5.05858491 K; any-01 -10.81209248 K. Both mesh levels order these three layouts the same way: any-01 coolest, then near-01, then reference.

Two meshes show changes, not certified error bounds. The r1-to-r2 nodal peak change is +0.2254 K on near-01, so peak convergence is weaker than centre-temperature convergence. No Richardson/GCI bound is claimed.

Raw case-B sanity: 122.34100707914 C, reproducing our previous r1 FE result within 1e-6 K. This is FE-pipeline sanity, not a HotSpot binary check.

All seven heat-balance checks and relative-residual checks pass: max absolute balance error 1.06e-11; max residual 1.24e-11.

Methods: existing fixed FEAgent/scikit-fem solver, P2 conforming tetrahedra, MKL PARDISO on the existing 128 GiB Brev CPU instance, four threads per solve. Finest memory up to 18,301 MiB RSS.

M_C is the highest die-centre value at die mid-thickness. peak_C searches saved FE nodes in the actual dies; it does not search P2 stationary points between nodes. selector_C is null: TAP selector is not calculated. The historical paper-target and supplemental cell fields in individual run JSON are not used in this comparison.

Original case.json, generator and generated layout hashes are recorded in experiments/sentilab_rank/README.md and INPUT_SHA256SUMS. Geometry preprocessing preserves 680 W, areas by conductivity, and non-overlapping tilings. Identity preprocessing matches original conductivity at 10,000 fixed random points per layer.

Prior HotSpot/FV results were already public: this was not blinded. No fit to those results. The results test one configured model, not measured chip hardware. Any physical interpretation of model differences remains separate.

Full meshes/fields and the original full manifest are retained locally; compact handoff omits them. Per-run mesh/field hashes are retained in FULL_SHA256SUMS. Reproduction uses the documented input transform and unchanged previous FE script.
