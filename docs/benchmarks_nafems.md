# NAFEMS and published-reference benchmarks (`feacore/problems/nafems/`)

Benchmark problems with published target values, used to validate the fixed solver and as agent
tasks. Every problem is a `feacore.problems.base.Problem` registered at import, tagged `nafems`.
Cook's membrane is not a NAFEMS test but lives in this suite, so it carries the tag too.

| id | analysis | QoI (agent reports) | reference | tol | kind |
|---|---|---|---|---|---|
| `nafems_le1` | elliptic membrane, plane stress, quarter model | `syy_D`: sigma_yy at D = (2, 0) [Pa] | 92.7 MPa | 2 % | published |
| `nafems_le10` | thick elliptic plate under pressure, 3D quarter model | `syy_D`: sigma_yy at D = (2, 0, 0.3) [Pa] | -5.38 MPa | 2 % | published |
| `nafems_le11_axisym` | cylinder/taper/sphere, thermal load T = r + z, axisymmetric | `szz_A`: axial stress at A = (r 1, z 0) [Pa] (field `syy`) | -105 MPa | 2 % | published |
| `nafems_le11_3d` | same, 3D quarter model | `szz_A`: sigma_zz at A = (0, 1, 0) [Pa] | -105 MPa | 2 % | published |
| `cooks_membrane` | tapered panel, plane stress, end shear (E = 1, nu = 1/3) | `uy_C`: uy at C = (48, 52), the mid-point of the loaded edge | 23.96 | 1 % | converged_numerical |
| `nafems_t4` | 2D steady conduction with convection | `T_E`: T at E = (0.6, 0.2) [degC] | 18.25 degC | 1 % | published |

Files: `_common.py` (shared geometry and constants), `le1.py`, `le10.py`, `le11.py`, `cooks.py`,
`t4.py`. Tests: `tests/test_problems_nafems.py`.

## Sources and verification

The NAFEMS booklet (*The Standard NAFEMS Benchmarks*, TNSB Rev. 3, 1990) is not public. Each
definition below was checked against public reproductions. The best of these are the scanned NAFEMS
data sheets for LE10 and LE11 in the FeenoX repository (https://github.com/seamplex/feenox,
`doc/nafems-le10-problem-input.svg`, `doc/nafems-le11-problem.png`; read from a local clone).
`abaqus-docs.mit.edu` and `docs.software.vt.edu` (the Abaqus Benchmarks Manual) are not reachable from
the build session, so for those pages and for the COMSOL and Altair pages I used search-engine
extracts. The full citation for each problem is in its `Reference.source`.

**LE1 elliptic membrane.** The membrane lies between the inner ellipse (x/2)^2 + y^2 = 1 and the
outer ellipse (x/3.25)^2 + (y/2.75)^2 = 1, with A = (0, 1), B = (0, 2.75), C = (3.25, 0) and
D = (2, 0). It is 0.1 m thick, with E = 210 GPa and nu = 0.3. Supports are ux = 0 on AB and uy = 0
on CD. The load is a 10 MPa outward normal traction on BC. In the case JSON this is a
`pressure` of **-10 MPa**, because the schema treats positive pressure as compressive. The target is
sigma_yy(D) = 92.7 MPa.
- Sources: FeenoX `examples/nafems-le1.geo/.fee`; COMSOL "Elliptic Membrane"; Altair OS-V 0010.
- COMSOL's mesh-converged value is 92.658 MPa, and ours extrapolates to 92.66 MPa. The published
  92.7 is a 3-digit rounding, about 0.04 % high.

**LE10 thick plate.** The plan is the same as LE1, with the thickness running from z = -0.3 to
z = 0.3. A 1 MPa pressure acts on the upper face. Supports are uy = 0 on face DCD'C' (y = 0) and
ux = 0 on face ABA'B' (x = 0). On the outer face BCB'C', ux = uy = 0 over the whole face, but
uz = 0 only along the face's mid-plane line (z = 0).
- The CAD imprints this line on the outer face: OCC `fragment` of the volume with the mid-plane
  ellipse arc, as in FeenoX `nafems-le10-cad.geo`. The result is one solid whose outer face is split
  in two. Physical curve `line_outer_midplane` carries the uz BC.
- The NAFEMS sheet asks for "direct stress sigma_yy at point D" with target 5.38 MPa, printed as a
  magnitude. The stress at D = (2, 0, 0.3) on the upper surface is compressive, -5.38 MPa (FeenoX
  -5.380 MPa; Fino -5.38 MPa).
- **Naming discrepancy:** the task brief called this point "E". On the NAFEMS sheet it is D. E/E'
  is the Abaqus manual's name for the mid-plane support points.

**LE11 cylinder/taper/sphere.** These are the sheet dimensions in (r, z):
- A = (1, 0), B = (1.4, 0).
- Outer sphere of radius 1.4 about the origin, from B up to D at z = 0.7.
- Straight taper from D to G = (1.0, 1.39).
- Cylinder r = 1.0 up to the top face at z = 1.79.
- Inner cylinder r = 0.7071 from H = (0.7071, 1.79) down to E = (0.7071, 0.7071), the 45-degree
  point of the unit circle.
- Inner sphere of radius 1 from E back to A.

Material is E = 210 GPa, nu = 0.3, alpha = 2.3e-4 /degC, with T = sqrt(x^2 + y^2) + z and
T_ref = 0. The axial displacement is zero on the bottom face (z = 0) and the top face (HIH'I'); 3D
adds symmetry on x = 0 and y = 0. The output is sigma_zz at A, target -105 MPa ("refined
axisymmetric").
- FeenoX's `.geo` puts D at z = 0.7071, giving a total height of 1.7971. The sheet's dimension
  chain (0.7 + 0.345 + 0.345 + 0.4) and FEATool (height 1.79) agree on D at z = 0.7, so we use 0.7.
- In axisymmetric runs the temperature expression is `x + y` (x = r, y = z) and the QoI field is
  `syy`.

**Cook's membrane.** The panel has corners (0,0), (48,44), (48,60) and (0,44), thickness 1,
E = 1 and nu = 1/3, in plane stress. The left edge is clamped, and a total shear of 1 is spread
uniformly over the right edge. The source is Cook (1974); Bergan & Felippa (1985) give 23.9, and the
ENERCALC C-04 verification reproduces 23.92 for a compatible 64x64 Q4 mesh and 23.96 for an
incompatible one.
- **Finding: the reference value is the vertical displacement at the mid-point of the loaded edge,
  (48, 52), not at the top corner (48, 60).** Structured compatible Q4 meshes reproduce the classic
  published table exactly at (48, 52): 18.29 / 22.08 / 23.43 / 23.92 for 4x4 / 8x8 / 16x16 / 64x64.
  We checked this with plain scikit-fem, mapping the unit square to the panel with x = 48 xi,
  y = 44 xi + eta (44 - 28 xi). At (48, 60) the same meshes give 18.61 / 22.67 / 24.27 / 25.04, and
  that value converges to about 25.2. An agent reporting the corner value will be about 5 %
  off.
- The converged mid-edge value is 23.967 (P2 extrapolation; Q9 32x32 gives 23.961), so the published
  23.96 is good to 0.03 %.
- Stresses are singular at the clamped corner (0, 44). It is a clamped/free wedge of 108.4 deg with
  Williams exponent lambda = 0.677, the smallest root of the 4x4 Muskhelishvili eigen-determinant
  with plane stress kappa = 2. So on quasi-uniform meshes point displacements converge like
  h^(2 lambda) = h^1.35. `expected_order` returns this value.

**T4.** The domain is the rectangle 0 <= x <= 0.6, 0 <= y <= 1.0 with k = 52 W/(m degC).
Boundary conditions are T = 100 degC on y = 0, an insulated edge x = 0, and convection
(h = 750 W/(m^2 degC), T_inf = 0) on x = 0.6 and y = 1.0. The target is T(0.6, 0.2) = 18.25 degC.
- Sources: COMSOL "Steady-State 2D Heat Transfer with Conduction" (boundary conditions, point and
  18.25; their 9x15 quadratic mesh gives 18.265); Abaqus Benchmarks 4.3.4 (k, h, target printed as
  18.3); Altair OS-V 0120.
- Our converged value is 18.2538 degC.

## Validation (fixed solver `skfem`)

Run with `convergence_study(problem, order, backend="skfem")` at the problem's default
`study_sizes`. 2D problems were run at P2 and P1, 3D at P2. This build saved the studies in
`/tmp/nafems/studies/*.json`; `scripts/validate.py` regenerates them. Error % is relative to the reference: the extrapolated value when
the convergence is monotone, otherwise the finest value.

| problem | order | QoI | reference | values (coarse -> fine) | extrapolated | p (theory) | GCI % | error % | convergence | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| `nafems_le1` | P2 | syy_D | 92.7 MPa | 92.151, 92.516, 92.621 | 92.662 | 1.85 (2.00) | 0.055 | -0.041 | monotone | pass |
| `nafems_le1` | P1 | syy_D | 92.7 MPa | 90.953, 92.018, 92.447 | 92.725 | 1.36 (1.00) | 0.375 | +0.027 | monotone | pass |
| `nafems_le10` | P2 | syy_D | -5.38 MPa | -5.3943, -5.4275, -5.3832 | - | - (2.00) | - | +0.059 (finest) | oscillatory_divergent | pass |
| `nafems_le11_axisym` | P2 | szz_A | -105 MPa | -105.02, -105.33, -105.5 | -105.67 | 1.00 (1.00) | 0.203 | +0.637 | monotone | pass |
| `nafems_le11_axisym` | P1 | szz_A | -105 MPa | -104.19, -104.78, -105.14 | -105.63 | 0.79 (1.00) | 0.581 | +0.596 | monotone (p_below_theory) | pass |
| `nafems_le11_3d` | P2 | szz_A | -105 MPa | -105.08, -105.19, -105.31 | - | - (1.00) | - | +0.291 (finest) | divergent | pass |
| `cooks_membrane` | P2 | uy_C | 23.96 | 23.947, 23.96, 23.964 | 23.966 | 1.50 (1.35) | 0.013 | +0.027 | monotone | pass |
| `cooks_membrane` | P1 | uy_C | 23.96 | 23.756, 23.904, 23.949 | 23.968 | 1.75 (1.35) | 0.100 | +0.033 | monotone | pass |
| `nafems_t4` | P2 | T_E | 18.25 | 18.263, 18.255, 18.254 | 18.254 | 3.09 (3.00) | 0.001 | +0.021 | monotone | pass |
| `nafems_t4` | P1 | T_E | 18.25 | 18.205, 18.242, 18.251 | 18.254 | 1.96 (2.00) | 0.022 | +0.022 | monotone | pass |

The CalculiX backend gives an independent cross-check, all within tolerance:

| problem | mesh h | skfem (P2) | ccx (P2) | reference |
|---|---|---|---|---|
| `nafems_le1` | 0.1 | 92.516 MPa | 92.887 MPa | 92.7 MPa |
| `nafems_le10` | 0.2 | -5.4275 MPa | -5.4550 MPa | -5.38 MPa |
| `nafems_le11_axisym` | 0.05 | -105.33 MPa | -105.33 MPa | -105 MPa |
| `nafems_le11_3d` | 0.1333 | -105.21 MPa | -105.29 MPa | -105 MPa |
| `cooks_membrane` | 2 | 23.960 | 23.958 | 23.96 |
| `nafems_t4` | 0.025 | 18.2549 degC | 18.2557 degC | 18.25 degC |

The LE10 mesh at h = 0.2 is the one that comes out high in the study. CalculiX, whose stress
recovery is different, sees the same high value on it, so the scatter comes from the mesh and not from the
stress recovery.

Notes per problem:

- **LE1.** P2 has p = 1.85 (theory 2) and extrapolates to 92.66 MPa. P1 on straight-sided meshes
  converges irregularly at the curved-edge vertex D. The default sequence (h = 0.2 / 0.1 / 0.05) is
  monotone, but finer P1 sequences give 92.45, 92.48, 92.59, 92.61.
- **Cook.** The reference study uses quasi-uniform meshes. Meshes graded towards the clamped corners
  are far more accurate (about 1e-5 at h0), but then the mesh-to-mesh changes are noise and the GCI
  reports "divergent".
- **T4.** The reference study uses quasi-uniform meshes. P2 converges at p = 3.1 (theory 3). P1
  starts at h0/2 (`FinerP1Studies`), because at h0 it is still pre-asymptotic (apparent p = 1.0).
- **LE11.**
  - The surface value at A converges at only p of about 1 for P2, on graded and uniform meshes and
    in 2D and 3D. A sits on the corner where the axially supported bottom face meets the free inner
    sphere, and sigma_zz varies steeply and non-smoothly into that corner. Points 0.02 m above A
    converge at the full rate. `expected_order` returns 1 for `szz_A`.
  - The converged value is about -105.6 to -105.7 MPa, 0.6 % beyond the published -105 MPa. Coarse
    meshes give -104 to -105 MPa.
  - P1 axisymmetric starts at h0/2.
- **3D point stresses at corner vertices** (LE10 D, LE11 A) fluctuate by about 0.1-0.5 % between
  unstructured tet meshes of similar size. This noise is about as large as the discretisation error
  of the finer meshes, so the three-mesh GCI reads it as oscillatory or divergent. Use the reference
  tolerance for these QoIs, and treat a non-monotone GCI as inconclusive rather than as a setup
  error. Our 20 LE10 runs on locally refined meshes (graded towards D or along DD', Netgen-optimised or not, h = 0.1-0.3) gave -5.37 to -5.43 MPa, with a mean of -5.40 MPa. All 13 LE11-3D runs with the final point refinement at A (h/6 within 0.05 m) gave -105.08 to -105.46 MPa. A coarser refinement (h/3) gives -103.9 to -105.0 MPa. Uniform LE10 meshes without local refinement at D come out 1.0-1.6 % low (-5.29 to -5.33 MPa).

## NAFEMS tests not included

- **Shell and plate tests:** LE2 (cylindrical shell patch test), LE3 (hemispherical shell with point
  loads), LE5 (Z-section cantilever) and LE6 (skew plate, normal pressure). They are intended for
  shell or plate elements, their targets are shell-theory values, and solid models of them need many
  elements through a thin wall.
- **Axisymmetric shell tests:** LE4 (hyperbolic shell), LE8 (shell with pressure) and LE9 (branched
  shell). These are axisymmetric shell-element tests.
- **No verifiable public definition reachable from this session:** LE7 (axisymmetric
  cylinder/sphere, pressure) and T1 (membrane with hot-spot, a thermal-stress membrane). Both could be
  posed with the existing solid formulations once their data sheets are available.
- **Outside the solver's scope:** T2 needs a radiation BC, which the case schema lacks. T3 and
  the other transient tests need transient analysis. The FV (free vibration) tests need modal
  analysis.

## Reproduce

```bash
export OMP_NUM_THREADS=1
python -m pytest tests/test_problems_nafems.py -q            # ~10 s; FEACORE_SLOW=1 adds the 3D studies
python scripts/validate.py --tags nafems --order 2 --jobs 2  # writes benchmarks/<id>/private/validation_*.json
```
