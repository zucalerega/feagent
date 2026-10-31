# Verification suite: CAD tasks with trustworthy references

Ten agent tasks. Each gives a STEP file (mm), a structured spec (`task/task.json`) and a brief
(`task/task.md`). The hidden reference in `reference/` holds QoI specs and reference values,
plus `convergence.json`: the evidence that each tolerance is attainable.

| Case | What it tests |
|---|---|
| `lame_cylinder` | pressure on a curved face, symmetry planes, plane strain (exact Lame solution) |
| `lame_sphere` | 3 symmetry planes, curved-surface stress with a steep 1/r^3 gradient (exact) |
| `torsion_shaft` | torque application (traction or distributing coupling), surface shear (exact Saint-Venant) |
| `gravity_column` | body force, density/acceleration units, rigid-body modes without over-constraint (exact) |
| `cantilever_tip_load` | distributed end load, root clamp; bending stress exact, deflection converged |
| `nafems_le10` | NAFEMS thick plate: mixed displacement constraints incl. a line constraint, 3D stress at a corner point (published) |
| `nafems_le1` | NAFEMS elliptic membrane as a 3D plate: outward traction, stress concentration at a notch (published, plane-stress definition) |
| `cantilever_modal` | natural frequencies, mass/density units (converged; Euler-Bernoulli shown) |
| `euler_buckling` | linear buckling with a reference load (converged; Euler shown) |
| `angle_bracket` | filleted part, bolt-hole fixation, pin-hole load, fillet peak stress (converged) |

**How references were established.** Closed-form solutions where an exact 3D solution of
exactly the stated problem exists. NAFEMS published targets for LE1/LE10. Otherwise, a
mesh-convergence study of the reference pipeline: gmsh quadratic tets on the task's own STEP,
fed to ccx 2.21 and scored by the same evaluator agents face, at five mesh sizes, with
Richardson extrapolation and Roache GCI (see `richardson()` in `feabench/suites/verification.py`).
Analytical and published references are *also* run through the convergence study, which proves
the tolerance is attainable and shows the cost of attaining it.

Regenerate: `feabench build-verification` (case files), `feabench convergence <case>`
(evidence and numerical references). Pipeline runs use 2 threads; CPU times are user+sys.

## References, tolerances and convergence evidence

| Case | QoI | Reference | Type | Tol. | Pipeline error at coarsest / finest mesh |
|---|---|---|---|---|---|
| lame_cylinder | `ur_inner` | 0.0453968 | analytical | 0.5 % | 0.08 % / 0.001 % |
| lame_cylinder | `ur_outer` | 0.0288889 | analytical | 0.5 % | 0.03 % / 0.000 % |
| lame_cylinder | `hoop_inner` | 166.667 | analytical | 1.5 % | 0.03 % / 0.011 % |
| lame_cylinder | `hoop_outer` | 66.6667 | analytical | 1.5 % | 0.22 % / 0.008 % |
| lame_cylinder | `axial_mid` | 20 | analytical | 1.5 % | 0.87 % / 0.017 % |
| lame_cylinder | `reaction_x` (gate) | -100000 | analytical | 0.5 % | 0.04 % / 0.000 % |
| lame_sphere | `ur_inner` | 0.0190476 | analytical | 0.5 % | 0.43 % / 0.011 % |
| lame_sphere | `ur_outer` | 0.00714286 | analytical | 0.5 % | 0.13 % / 0.000 % |
| lame_sphere | `hoop_inner` | 71.4286 | analytical | 2.5 % | 4.08 % / 0.182 % |
| lame_sphere | `hoop_outer` | 21.4286 | analytical | 1.5 % | 1.17 % / 0.037 % |
| lame_sphere | `reaction_x` (gate) | -196350 | analytical | 0.5 % | 0.02 % / 0.000 % |
| torsion_shaft | `ut_mid` | 0.0394098 | analytical | 1 % | 0.01 % / 0.000 % |
| torsion_shaft | `ut_3q` | 0.0591147 | analytical | 1 % | 0.01 % / 0.000 % |
| torsion_shaft | `mises_mid` | 110.266 | analytical | 1.5 % | 0.27 % / 0.004 % |
| torsion_shaft | `reaction_torque` (gate) | -100000 | analytical | 0.5 % | 0.00 % / 0.000 % |
| gravity_column | `uz_top` | -0.00467262 | analytical | 0.5 % | 0.05 % / 0.050 % |
| gravity_column | `szz_mid` | -1.9625 | analytical | 1 % | 0.00 % / 0.001 % |
| gravity_column | `reaction_z` (gate) | 9812.5 | analytical | 2 % | 0.79 % / 0.218 % |
| cantilever_tip_load | `tip_deflection` | -1.91129 | numerical | 1 % | 0.25 % / 0.032 % |
| cantilever_tip_load | `bending_stress_mid` | 150 | analytical | 1 % | 0.00 % / 0.007 % |
| cantilever_tip_load | `reaction_z` (gate) | 1000 | analytical | 0.5 % | 0.00 % / 0.000 % |
| nafems_le10 | `syy_D` | -5.38 | published | 2 % | 4.14 % / 0.243 % |
| nafems_le10 | `reaction_z` (gate) | 5.4487e+06 | analytical | 0.5 % | 0.00 % / 0.000 % |
| nafems_le1 | `syy_D` | 92.7 | published | 2 % | 0.28 % / 0.139 % |
| nafems_le1 | `reaction_x` (gate) | -1.375e+06 | analytical | 0.5 % | 0.00 % / 0.001 % |
| cantilever_modal | `f1` | 209.369 | numerical | 1 % | 0.25 % / 0.027 % |
| cantilever_modal | `f2` | 415.672 | numerical | 1 % | 0.17 % / 0.021 % |
| euler_buckling | `lambda1` | 2.70059 | numerical | 1 % | 0.09 % / 0.005 % |
| angle_bracket | `pin_disp_y` | -0.35885 | numerical | 1 % | 1.38 % / 0.079 % |
| angle_bracket | `fillet_mises_max` | 135.562 | numerical | 3.5 % | 9.56 % / 0.000 % |
| angle_bracket | `reaction_y` (gate) | 3000 | analytical | 0.5 % | 0.00 % / 0.000 % |

| Case | Cheapest passing mesh (DOF) | Solver CPU there | Finest mesh (DOF) | CPU |
|---|---|---|---|---|
| lame_cylinder | 2208 | 0.07 s | 165861 | 54.2 s |
| lame_sphere | 13227 | 0.90 s | 111639 | 44.0 s |
| torsion_shaft | 4905 | 0.20 s | 197451 | 85.5 s |
| gravity_column | 1392 | 0.06 s | 87270 | 11.4 s |
| cantilever_tip_load | 2865 | 0.10 s | 97299 | 10.6 s |
| nafems_le10 | 7551 | 0.36 s | 137124 | 60.4 s |
| nafems_le1 | 2973 | 0.09 s | 104214 | 7.6 s |
| cantilever_modal | 2865 | 0.19 s | 97299 | 21.0 s |
| euler_buckling | 5412 | 0.29 s | 111951 | 16.4 s |
| angle_bracket | 40983 | 3.54 s | 563730 | 475.5 s |
