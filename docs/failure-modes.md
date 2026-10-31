# Failure modes and the speed / accuracy metric

Code: `feacore/diagnostics/` (detectors, thresholds in `flags.py`), tests: `tests/test_diagnostics.py`.
Every detector is deterministic: same case + mesh → same flags. Severity: **fail** = the solution
must not be used, **warn** = usable but a number in it is suspect, **info** = advice.

```python
from feacore.diagnostics import diagnose, precheck
rep = precheck(case)                      # no solve: catches unconstrained / unloaded setups
rep, res = diagnose(case, design={"yield": 275e6})   # precheck -> solve -> post-solve checks
from feacore.diagnostics.verify import refinement_study, p_sensitivity   # extra solves
from feacore.diagnostics.score import score_population
```

## Three tiers of checks

| Tier | Cost | When |
|---|---|---|
| Precheck | assemble only | before solving; an agent can call it to justify `INVALID_CONFIG` |
| Post-solve | ~free (one solve) | every solution |
| Verification | 2 extra solves (refinement) / 1 (P2) | evaluator, to estimate accuracy without a reference |

## Catalogue

✅ implemented and tested · ⏳ planned

### Setup (precheck)
| id | sev | detection | |
|---|---|---|---|
| `rigid_body_mode` | fail | Rigid-body modes evaluated at the constrained DOFs; rank deficiency (SVD) → which of tx/ty/tz/rx/ry/rz are free. Per connected body. Exact, no solve. | ✅ |
| `disconnected_bodies` | info | Connected components of the cell graph. | ✅ |
| `no_load` | fail | f = 0 and no prescribed displacement. | ✅ |
| `point_constraint` | warn | Dirichlet BC on a point / vertex group / 3D edge: stress there is infinite in the continuum. | ✅ |
| `volumetric_locking` | warn | P1 with ν > 0.45. | ✅ |
| `linear_elements` | info | P1 simplices: over-stiff in bending (confirmed by `p_sensitive`). | ✅ |
| BC on a missing / wrong-dimension group, non-planar symmetry | fail | raised by the solver (`MeshError` / `CaseError`) with the available groups. | ✅ |
| `over_constraint` | info | Σ over support groups and components of \|R\| / Σ \|F\|; 1.0 when no supports fight. Example: a block clamped between two walls gives 1.22. | ✅ |
| contact / interference between parts | | not representable in the linear solver; harness should reject | ⏳ |
| thermal over-constraint | | needs thermal loads in the solver | ⏳ |

### Mesh
| id | sev | detection | |
|---|---|---|---|
| `degenerate_element` / `poor_element_quality` | fail / warn | mean-ratio quality of the corner simplex < 0.01 / 0.1 | ✅ |
| `inverted_element` | fail | det J changes sign inside a curved cell | ✅ |
| too few elements through a thin wall, CAD fidelity | | already gated in the mesh harness (`thin_wall_min_elements`, `volume_fidelity`) | (harness) |

### Solve integrity
| id | sev | detection | |
|---|---|---|---|
| `solver_failure`, `non_finite` | fail | the linear solver raises / NaN or Inf in u or σ | ✅ |
| `linear_residual` | fail | ‖Ku − f‖ / ‖f‖ on the free DOFs > 1e-6 (iterative tolerance too loose, ill-conditioning) | ✅ |
| `equilibrium` | fail | Σ applied + Σ reactions per component ≠ 0 (relative 1e-6) | ✅ |
| near-mechanism, ill-conditioning | | condition-number estimate, smallest eigenvalue of K. Today a near-mechanism shows up as `large_displacement`. | ⏳ |

### Validity of linear static analysis ("numerical blow-up")
| id | sev | detection | |
|---|---|---|---|
| `large_displacement` | fail | max \|u\| > 5% of the bounding-box diagonal (near-mechanism, missing support, unit mismatch) | ✅ |
| `large_strain` / `high_strain` | fail / warn | max equivalent strain > 5% / 1% (E in MPa with loads in N/m², …) | ✅ |
| buckling of slender compressed members | | linear static cannot see it; needs a linear buckling eigen-solve | ⏳ |

### Discretization (accuracy)
| id | sev | detection | |
|---|---|---|---|
| `under_resolved` | warn / fail | Zienkiewicz–Zhu energy-norm estimate (raw vs recovered stress) > 10% / 30%; reports the worst cell | ✅ |
| `p_sensitive` | warn | P1 vs P2 on the same mesh: strain energy changes > 5% (locking) | ✅ |
| `reduced_order` | info | QoI converges at p < ½ theory: a singularity elsewhere slows it; refine locally | ✅ |
| `oscillatory_qoi` | info | GCI convergence ratio −1 < R < 0 | ✅ |
| `not_accurate_enough` / `over_resolved` | warn / info | estimated error > tolerance / every QoI 10× inside tolerance (wasted DOFs) | ✅ |

### Singularities and stress concentrations
| id | sev | detection | |
|---|---|---|---|
| `peak_at_support` | warn | von Mises peak within 2h of the rim of a clamped support (rims on symmetry / roller planes ignored) | ✅ |
| `peak_at_load_edge` | warn | peak within 2h of where a load stops part-way across a smooth surface | ✅ |
| `unresolved_peak` | warn | peak cell's local ZZ error > 30%: a notch or re-entrant corner that is under-resolved or singular | ✅ |
| `singular_qoi` | warn | refinement study: successive changes grow (GCI divergent). The QoI has no mesh-independent value and is excluded from scoring. | ✅ |
| crack tip: extract K_I instead of peak stress | | | ⏳ |

`vm_trusted` (metric) is the max von Mises excluding nodes near support or load rims. The design checks use it, not the raw peak.

### Design (the part fails, not the analysis); needs `design={...}`
| id | sev | detection | |
|---|---|---|---|
| `yield` | warn | yield / `vm_trusted` < `min_safety_factor` | ✅ |
| `deflection_limit` | warn | max \|u\| > `max_disp` | ✅ |
| fatigue, buckling | | | ⏳ |

## The accuracy metric

Comparing fields point by point doesn't work: agents' meshes differ, and peak stresses at singularities never converge. Accuracy is therefore defined on **quantities of interest (QoIs)**, functionals that have a mesh-independent limit:

- **strain energy** (= ½ compliance): always added; converges fastest; global
- reaction forces, displacements at points
- stresses at points or regions away from singularities (`vm_trusted`)

Each QoI gets an **estimated relative error**, from the strongest source available:

1. **Reference** (benchmarks only): |q − q_ref| / |q_ref|. Used to *calibrate* sources 2 and 3, not to score agents on real CAD.
2. **Refinement study of the agent's own setup** (`verify.refinement_study`). The evaluator re-meshes the agent's setup with every size × r and ÷ r. The agent's mesh is the middle of the three. Richardson extrapolation gives q_ext and an error of |q_agent − q_ext| / |q_ext|. A QoI that diverges is classified `singular` and dropped. Each agent's BCs and loads differ, so each agent is its own reference, and no shared "true answer" is needed.
3. **Single mesh, ZZ** (no extra solves). The relative strain-energy error is about η_rel² for a Galerkin solution. P2 with L2-projection recovery underestimates it by a stable factor: measured effectivity 0.39 on Lamé (two meshes) and 0.44 on the 3D Lamé sphere. `score.ZZ_ENERGY_EFFECTIVITY` holds the calibration, and a test re-measures it.

### Score (`score.score_population`)

```
valid       = no fail flag
accuracy    = min over QoIs of min(1, tol_q / err_q)      worst QoI governs
meets_tol   = every judged QoI within tolerance
cost        = DOFs   (deterministic; wall time is reported but too noisy to score)
efficiency  = cost of the cheapest valid agent that meets tolerance / own cost
score       = valid × accuracy × efficiency  ∈ [0, 1]
```

1.0 is "the cheapest setup that is accurate enough". Fewer DOFs that miss tolerance lose on accuracy; more DOFs than needed lose on efficiency. Each agent also gets a Pareto rank on (max error, DOFs), so the evaluator can show the whole trade-off curve.

## Benchmarks → what they test

| Benchmark | Role | Status |
|---|---|---|
| Lamé cylinder, plane strain (Tier B, smooth) | must come out clean; calibrates ZZ effectivity | ✅ test |
| Lamé sphere 3D | clean in 3D; symmetry planes are not over-constraint | ✅ test |
| NAFEMS LE1 (Tier A) | refinement-study error within 3× of the true error; q_ext within 1% of 92.7 MPa | ✅ test |
| L-shape, re-entrant corner (Tier B, singular) | peak located at the corner; `vm_max` singular; energy `reduced_order` at p ≈ 2λ ≈ 1.09 | ✅ test (elastic L instead of Poisson) |
| Cantilever, P1 (Tier B, locking) | `p_sensitive`; clamped root gives `peak_at_support` | ✅ test |
| NAFEMS LE10 (3D) | | ⏳ teammate's geometry builder currently raises |
| LE11 (thermal) | | ⏳ solver has no thermal loads |
| LE3 / LE5, MacNeal–Harder (shells) | | ⏳ no shell elements; out of scope unless shells are added |
| Kirsch plate (Kt = 3, Howland for finite width) | concentration resolved vs `unresolved_peak` | ⏳ needs a Problem class |
| Cracked plate (K_I known) | second singularity case; K_I extraction | ⏳ |
| Manufactured solutions | exact error everywhere; calibrates ZZ on any geometry | ⏳ needs body-force expressions in the solver |
| SimJEB (Tier D) | run `refinement_study` on single-mesh GE bracket results to show how under-converged they are | ⏳ check licence / download first |
