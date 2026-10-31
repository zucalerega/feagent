# Analytical benchmark problems

Package `feacore/problems/analytical/` (registered automatically by `feacore.problems.registry`).
These are the problems with *known* answers: closed-form elasticity / potential solutions, one
handbook family, and two singular problems whose "answer" is the convergence behaviour itself. They
serve three purposes:

1. **Verify the fixed solver** (skfem) and the verification layer (Richardson / GCI) on problems where
   the exact value is known, including curved boundaries, axisymmetry, 3D, body forces and expression BCs.
2. **Benchmark the agents**: each problem has an agent-facing statement + STEP (no answers) and a private
   reference setup + reference values.
3. **Exercise singularity detection**: QoIs that must *diverge* under refinement (re-entrant corners).

All units SI (m, N, Pa, K). Steel E = 200 GPa, nu = 0.3 unless stated (bracket: aluminium 70 GPa, 0.33).

## Problems

| id | family | physics / formulation | model | what it tests | QoIs (reference kind) |
|---|---|---|---|---|---|
| `lame_cylinder_plane_strain` | lame_cylinder | elasticity / plane strain | quarter annulus a = 0.1, b = 0.2 m, p = 100 MPa, symmetry x = 0, y = 0 | curved boundary, pressure, plane-strain szz | ur_bore, hoop_bore, szz_bore, vm_max, strain_energy (analytical) |
| `lame_cylinder_axisym` | lame_cylinder | elasticity / axisymmetric | r-z slice a..b, H = 0.05 m, **u_z = 0 on both ends** (= plane strain) | axisymmetric formulation, hoop stress | ur_bore, hoop_bore, szz_bore, vm_max, strain_energy (analytical) |
| `lame_cylinder_axisym_open` | lame_cylinder | elasticity / axisymmetric | same slice, **open ends**: u_z = 0 at z = 0 only, top free (s_zz = 0) | axial contraction, minimal supports | ur_bore, hoop_bore, uz_top, vm_max, strain_energy (analytical) |
| `lame_sphere_axisym` | lame_sphere | elasticity / axisymmetric | quarter meridian annulus, axis x = 0, equator y = 0 | axis treatment (u_r = 0), spherical stress state | ur_bore, uz_pole, hoop_bore, vm_max, strain_energy (analytical) |
| `lame_sphere_3d` | lame_sphere | elasticity / 3d | octant of the shell, 3 symmetry planes | 3D tet10, curved faces | ur_bore, hoop_bore, vm_max, strain_energy (analytical) |
| `kirsch_exact_traction` | kirsch | elasticity / plane stress | [0, L]^2 minus quarter hole, a = 0.05, L = 0.25 m; **exact Kirsch tractions on x = L and y = L** | expression tractions, curved boundary, peak-stress recovery; exact everywhere | sxx_hole_top (= 3 sigma0), syy_hole_side (= -sigma0), ux_hole_side, uy_hole_top, ux_right, vm_max, strain_energy (analytical) |
| `plate_hole_finite_width_d01`, `_d03`, `_d05` | plate_hole_finite_width | elasticity / plane stress | quarter of a strip W = 0.2 m, d/W = 0.1 / 0.3 / 0.5, half-length 2W, end traction 100 MPa | stress concentration in a finite strip | s_peak, vm_max (handbook: Peterson chart 4.1) |
| `timoshenko_cantilever` | timoshenko_cantilever | elasticity / plane stress | L = 1, D = 0.2 m, P = 10 kN parabolic end shear, **exact root displacement prescribed** | expression displacement + traction BCs, bending, exact everywhere | tip_deflection, sxx_top_mid, ux_tip_top, vm_max, strain_energy (analytical) |
| `lshape_poisson` | lshape_poisson | heat / 2d (k = 1, Laplace) | (-1,1)^2 minus [0,1]x[-1,0]; T = r^(2/3) sin(2 theta/3) on the boundary | re-entrant singularity, reduced convergence order, Dirichlet expressions | T_point, T_integral (analytical); q_max (**diverge**) |
| `lbracket_reentrant` | lbracket_reentrant | elasticity / plane stress | L-bracket, legs 0.04 m wide, 0.12 m long, clamped top, 200 N on the free end | singularity detection; no reference | vm_max (**diverge**), tip_uy, strain_energy (converge, p ~ 1.09) |
| `rotating_disk` | rotating_disk | elasticity / plane stress | quarter disk b = 0.25 m, rho = 7850, omega = 1000 rad/s | body force from an expression | ur_rim, s_center, hoop_rim, vm_max, strain_energy (analytical) |

Plane-stress models are thin (t = 2 mm; disk 20 mm, cantilever 10 mm) so that the 2D plane-stress
idealisation is physically valid (see the CalculiX cross-check below).

## Exact solutions and references

All closed forms below were re-derived symbolically (sympy: strain-displacement, Hooke, equilibrium,
traction-free surfaces), and every strain-energy reference was checked against an independent domain
quadrature of the exact energy density (Gauss-Legendre in polar / spherical / r-z coordinates;
agreement <= 1e-14). The tests repeat the Hooke / equilibrium /
Laplace checks by finite differences on every exact field.

* **Lamé cylinder** (Lamé 1852; Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. 1970, Ch. 4
  "Stress distribution symmetrical about an axis" / "Displacements for symmetrical stress distributions";
  Barber, *Elasticity*, Ch. 8). K = p a^2/(b^2 - a^2):
  s_rr = K(1 - b^2/r^2), s_tt = K(1 + b^2/r^2); plane strain s_zz = 2 nu K,
  u_r = (1+nu)K/E [(1-2nu) r + b^2/r]; open ends s_zz = 0, u_r = K/E [(1-nu) r + (1+nu) b^2/r],
  u_z = -2 nu K z/E. von Mises max at r = a: K sqrt(3 b^4/a^4 + (1-2nu)^2) (plane strain) or
  K sqrt(3 b^4/a^4 + 1) (open).
* **Lamé sphere** (T&G "symmetrical stress distribution in a sphere"; Barber). K = p a^3/(b^3 - a^3):
  s_rr = K(1 - b^3/r^3), s_tt = K(1 + b^3/(2r^3)), u_r = K/E [(1-2nu) r + (1+nu) b^3/(2r^2)];
  von Mises = 1.5 K b^3/r^3.
* **Strain energies** (Lamé, disk, Kirsch): Clapeyron, U = 1/2 (work of the external loads); symmetry
  planes and constrained ends do no work. Kirsch: Gauss-Legendre (200 points per edge) of t.u on the two
  loaded edges.
* **Kirsch** (Kirsch 1898, Z. VDI 42; T&G Ch. 4 "The effect of circular holes on stress distributions in
  plates"): Cartesian stresses in the module docstring; plane-stress displacements from the
  Kolosov-Muskhelishvili potentials (kappa = (3-nu)/(1+nu)). Checks: u_x(a,0) = 3 sigma0 a/E,
  u_y(0,a) = -sigma0 a/E, far field u = (sigma0 x/E, -nu sigma0 y/E).
* **Peterson / Howland** (Pilkey & Pilkey, *Peterson's Stress Concentration Factors*, 3rd ed. 2008,
  Chart 4.1; Howland 1930, Phil. Trans. R. Soc. A 229:49): K_tn = 2 + 0.284(1-d/W) - 0.600(1-d/W)^2 +
  1.32(1-d/W)^3, sigma_max = K_tn sigma0 W/(W-d). Kind `handbook`, tolerance 2 %: I could not check an
  error bound for the fit here (no network access to the book), so 2 % is a judgement covering
  fit + chart accuracy. See the validation: our converged FE peak is above the fit by 0.05 / 0.22 /
  0.79 % for d/W = 0.1 / 0.3 / 0.5.
* **Timoshenko cantilever** (T&G Ch. 3 "Bending of a cantilever loaded at the end"; displacement field as
  in Augarde & Deeks, Finite Elem. Anal. Des. 44 (2008) 595): fields in the module docstring;
  tip deflection -P L (8L^2 + (4+5nu) D^2)/(24 E I); U = P^2 L^3/(6EI) + 3P^2 L/(5 G t D).
* **L-shape** (exact harmonic function; corner singularity: Grisvard 1985): T_integral = 1.5839289449053858
  by mpmath tanh-sinh quadrature over the three unit squares (30 digits), confirmed by an independent
  polar quadrature (int_0^R r^(5/3) dr = 3/8 R^(8/3)) to all 30 digits. T(-0.5, 0.5) = 2^(-1/3).
* **Rotating disk** (T&G Ch. 4 "Rotating disks"): s_rr = (3+nu)/8 c (b^2 - r^2),
  s_tt = (3+nu)/8 c b^2 - (1+3nu)/8 c r^2, u_r = c r (1-nu)/(8E) [(3+nu) b^2 - (1+nu) r^2], c = rho omega^2.
  von Mises is maximal at the centre (checked on 1001 radii).
* **Williams exponent** for the 270 degree free corner: lambda = 0.5444837 (root of
  sin(2 lambda alpha) + lambda sin(2 alpha) = 0, alpha = 3 pi/4; Williams 1952), so sigma ~ r^(lambda-1).

Citations name sections, not article numbers (I could not verify the 3rd-edition article numbering
offline).

## Design notes

* **Exact everywhere**: Kirsch (exact tractions on the cut edges) and the cantilever (exact root
  displacement) have no modelling error, so any deviation is discretisation or a setup error.
* **No answers in the statement**: expression BCs are given *restricted to their edge* (the constant
  coordinate substituted), e.g. the Kirsch traction on x = L as a function of y only, and the L-shape data
  per outer edge. Otherwise an agent could evaluate the full field at the QoI point. For the same reason
  the cantilever has no shear-stress QoI: the prescribed tip traction already is s_xy(y), which does not
  depend on x. `tests/test_problems_analytical.py` checks that no reference value (several formats, SI /
  MPa / mm) appears in `statement()`.
* **Meshes**: uniform for the Lamé 2D problems, the L-shape and the bracket (the singular problems must
  be refined uniformly so that the expected orders apply); distance-threshold refinement around the hole
  (Kirsch, plate family) and a bore-graded mesh for the 3D sphere (0.4 h on the bore -> 1.2 h on the
  outer face). All sizes scale with h, so a study is geometrically self-similar.
* **Study sizes**: P2: h0, h0/2, h0/4 (2D) and h0, h0/1.5, h0/2.25 (3D); P1 starts one level finer
  (h0/2 .. h0/8), so that first-order stresses are closer to the asymptotic range.
* **Expected orders**: the defaults of `feacore.verification.study` (displacement p+1, stress p, energy 2p,
  integral p+1) except `lshape_poisson` (T_point, T_integral: 4/3 = 2 x 2/3 from pollution by the corner)
  and `lbracket_reentrant` (tip_uy, strain_energy: 2 lambda = 1.089). Diverging QoIs: `q_max`, `vm_max`
  of the bracket.
* **Tolerances** (relative, for judging agents and the reference runs): displacements and energies 1 %,
  stresses 2 %, L-shape values 0.5 %, handbook 2 %.
* **Axisymmetric field names**: hoop = `stt`, axial = `syy`. Note that the skfem backend also exposes the
  hoop stress as `szz` in axisymmetric runs; `exact()` follows the case.py contract (szz = 0 in 2D
  except plane strain). The QoIs use `stt` / `syy` only.

## Validation

`feacore.verification.study.convergence_study(problem, order, backend="skfem")` for every problem, order 2
(all) and order 1 (2D), 1 core (`OMP_NUM_THREADS=1`), 2026-10-31. Studies saved as JSON in
`/tmp/analytical/studies/<id>_order<k>.json` (scratch; re-create with `python scripts/validate.py
--problems <ids> --order <k>`). Columns: values on the three meshes (coarse -> fine), reference,
Richardson-extrapolated value, observed / theoretical order, Celik convergence class, GCI of the fine
solution, error vs reference (of the extrapolated value when the convergence is monotone, else of the
finest value, as `study._judge` does), and the verdict.

**Summary: order 2: 13/13 problems pass, every judged QoI passes. Order 1: 11/12 pass; the only failing
QoI (`lame_cylinder_plane_strain` vm_max) is a pre-asymptotic sequence judged on its extrapolation (see
below). Total cost: about 27 s for all P2 studies (16 s of it the 3D sphere) and 16 s for all P1 studies.**

### Order 2 (P2, isoparametric P2 geometry)

| problem | QoI | phi(h1) | phi(h2) | phi(h3) | reference | phi_ext | p obs / th | convergence | GCI_fine % | err % | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kirsch_exact_traction | sxx_hole_top | 2.92855e+08 | 2.98066e+08 | 2.99450e+08 | 3.00000e+08 | 2.99905e+08 | 2.09 / 2.00 | monotone | 0.190 | -0.032 (ext) | yes |
| kirsch_exact_traction | syy_hole_side | -9.44820e+07 | -9.84115e+07 | -9.95997e+07 | -1.00000e+08 | -1.00067e+08 | 1.89 / 2.00 | monotone | 0.586 | +0.067 (ext) | yes |
| kirsch_exact_traction | ux_hole_side | 7.50037e-05 | 7.50020e-05 | 7.50003e-05 | 7.50000e-05 | - | - / 3.00 | divergent | - | +0.000 (fine) | yes |
| kirsch_exact_traction | uy_hole_top | -2.50153e-05 | -2.50028e-05 | -2.50004e-05 | -2.50000e-05 | -2.50000e-05 | 2.65 / 3.00 | monotone | 0.002 | -0.000 (ext) | yes |
| kirsch_exact_traction | ux_right | 1.38109e-04 | 1.38119e-04 | 1.38120e-04 | 1.38120e-04 | 1.38120e-04 | 4.33 / 3.00 | monotone | 0.000 | -0.000 (ext) | yes |
| kirsch_exact_traction | vm_max | 2.99940e+08 | 2.99799e+08 | 3.00007e+08 | 3.00000e+08 | - | - / 2.00 | oscillatory_divergent | - | +0.002 (fine) | yes |
| kirsch_exact_traction | strain_energy | 3.24375 | 3.24385 | 3.24385 | 3.24385 | 3.24385 | 4.07 / 4.00 | monotone | 0.000 | -0.000 (ext) | yes |
| lame_cylinder_axisym_open | ur_bore | 9.83281e-05 | 9.83327e-05 | 9.83332e-05 | 9.83333e-05 | 9.83333e-05 | 3.17 / 3.00 | monotone | 0.000 | -0.000 (ext) | yes |
| lame_cylinder_axisym_open | hoop_bore | 1.67160e+08 | 1.66789e+08 | 1.66704e+08 | 1.66667e+08 | 1.66677e+08 | 2.16 / 2.00 | monotone | 0.020 | +0.006 (ext) | yes |
| lame_cylinder_axisym_open | uz_top | -4.99986e-06 | -4.99998e-06 | -5.00000e-06 | -5.00000e-06 | -5.00000e-06 | 2.97 / 3.00 | monotone | 0.000 | +0.000 (ext) | yes |
| lame_cylinder_axisym_open | vm_max | 2.33053e+08 | 2.33259e+08 | 2.33318e+08 | 2.33333e+08 | 2.33344e+08 | 1.82 / 2.00 | monotone | 0.014 | +0.005 (ext) | yes |
| lame_cylinder_axisym_open | strain_energy | 154.461 | 154.462 | 154.462 | 154.462 | 154.462 | 4.47 / 4.00 | monotone | 0.000 | -0.000 (ext) | yes |
| lame_cylinder_axisym | ur_bore | 9.53281e-05 | 9.53327e-05 | 9.53332e-05 | 9.53333e-05 | 9.53333e-05 | 3.17 / 3.00 | monotone | 0.000 | -0.000 (ext) | yes |
| lame_cylinder_axisym | hoop_bore | 1.67160e+08 | 1.66789e+08 | 1.66704e+08 | 1.66667e+08 | 1.66677e+08 | 2.17 / 2.00 | monotone | 0.020 | +0.006 (ext) | yes |
| lame_cylinder_axisym | szz_bore | 2.05084e+07 | 2.01178e+07 | 2.00345e+07 | 2.00000e+07 | 2.00104e+07 | 2.27 / 2.00 | monotone | 0.150 | +0.052 (ext) | yes |
| lame_cylinder_axisym | vm_max | 2.31029e+08 | 2.31246e+08 | 2.31309e+08 | 2.31325e+08 | 2.31337e+08 | 1.81 / 2.00 | monotone | 0.015 | +0.005 (ext) | yes |
| lame_cylinder_axisym | strain_energy | 149.749 | 149.749 | 149.749 | 149.749 | 149.749 | 4.47 / 4.00 | monotone | 0.000 | -0.000 (ext) | yes |
| lame_cylinder_plane_strain | ur_bore | 9.53292e-05 | 9.53327e-05 | 9.53333e-05 | 9.53333e-05 | 9.53334e-05 | 2.64 / 3.00 | monotone | 0.000 | +0.000 (ext) | yes |
| lame_cylinder_plane_strain | hoop_bore | 1.65782e+08 | 1.66417e+08 | 1.66599e+08 | 1.66667e+08 | 1.66667e+08 | 1.89 / 2.00 | monotone | 0.051 | +0.000 (ext) | yes |
| lame_cylinder_plane_strain | szz_bore | 1.96088e+07 | 1.98968e+07 | 1.99693e+07 | 2.00000e+07 | 1.99924e+07 | 2.07 / 2.00 | monotone | 0.145 | -0.038 (ext) | yes |
| lame_cylinder_plane_strain | vm_max | 2.31397e+08 | 2.31367e+08 | 2.31348e+08 | 2.31325e+08 | 2.31314e+08 | 0.66 / 2.00 | monotone | 0.018 | -0.004 (ext) | yes |
| lame_cylinder_plane_strain | strain_energy | 748.738 | 748.746 | 748.746 | 748.746 | 748.746 | 3.90 / 4.00 | monotone | 0.000 | +0.000 (ext) | yes |
| lame_sphere_3d | ur_bore | 3.99812e-05 | 3.99899e-05 | 3.99985e-05 | 4.00000e-05 | 4.01232e-05 | 0.17 / 3.00 | monotone | 0.390 | +0.308 (ext) | yes |
| lame_sphere_3d | hoop_bore | 7.10180e+07 | 7.17949e+07 | 7.14441e+07 | 7.14286e+07 | 7.11688e+07 | 2.15 / 2.00 | oscillatory | 0.482 | +0.022 (fine) | yes |
| lame_sphere_3d | vm_max | 1.73176e+08 | 1.72210e+08 | 1.71768e+08 | 1.71429e+08 | 1.71442e+08 | 2.23 / 2.00 | monotone | 0.238 | +0.008 (ext) | yes |
| lame_sphere_3d | strain_energy | 31.4071 | 31.414 | 31.4155 | 31.4159 | 31.4159 | 4.16 / 4.00 | monotone | 0.002 | -0.000 (ext) | yes |
| lame_sphere_axisym | ur_bore | 3.99930e-05 | 3.99987e-05 | 3.99999e-05 | 4.00000e-05 | 4.00001e-05 | 2.46 / 3.00 | monotone | 0.001 | +0.000 (ext) | yes |
| lame_sphere_axisym | uz_pole | 3.99951e-05 | 3.99990e-05 | 4.00000e-05 | 4.00000e-05 | 4.00002e-05 | 2.15 / 3.00 | monotone | 0.001 | +0.001 (ext) | yes |
| lame_sphere_axisym | hoop_bore | 7.14379e+07 | 7.14516e+07 | 7.14317e+07 | 7.14286e+07 | - | - / 2.00 | oscillatory_divergent | - | +0.004 (fine) | yes |
| lame_sphere_axisym | vm_max | 1.71486e+08 | 1.71412e+08 | 1.71425e+08 | 1.71429e+08 | 1.71428e+08 | 2.58 / 2.00 | oscillatory | 0.002 | -0.002 (fine) | yes |
| lame_sphere_axisym | strain_energy | 125.661 | 125.663 | 125.664 | 125.664 | 125.664 | 3.75 / 4.00 | monotone | 0.000 | +0.000 (ext) | yes |
| lbracket_reentrant | vm_max | 5.98107e+07 | 7.51914e+07 | 9.98757e+07 | - | - | - / 2.00 | divergent (diverge expected) | - | - | yes |
| lbracket_reentrant | tip_uy | -3.30129e-04 | -3.31844e-04 | -3.32721e-04 | - | -3.33645e-04 | 1.00 / 1.09 | monotone | 0.347 | - | n/a |
| lbracket_reentrant | strain_energy | 0.033041 | 0.0332135 | 0.0333014 | - | 0.0333933 | 1.01 / 1.09 | monotone | 0.345 | - | n/a |
| lshape_poisson | T_point | 0.792931 | 0.793393 | 0.793578 | 0.793701 | 0.793699 | 1.37 / 1.33 | monotone | 0.019 | -0.000 (ext) | yes |
| lshape_poisson | q_max | 1.84526 | 2.33855 | 2.94492 | - | - | - / 2.00 | divergent (diverge expected) | - | - | yes |
| lshape_poisson | T_integral | 1.58219 | 1.58324 | 1.58365 | 1.58393 | 1.58392 | 1.39 / 1.33 | monotone | 0.021 | -0.000 (ext) | yes |
| plate_hole_finite_width_d01 | s_peak | 2.98861e+08 | 3.02258e+08 | 3.03272e+08 | 3.03542e+08 | 3.03683e+08 | 1.83 / 2.00 | monotone | 0.169 | +0.046 (ext) | yes |
| plate_hole_finite_width_d01 | vm_max | 3.03625e+08 | 3.03686e+08 | 3.03634e+08 | 3.03542e+08 | 3.03302e+08 | 0.22 / 2.00 | oscillatory | 0.137 | +0.030 (fine) | yes |
| plate_hole_finite_width_d03 | s_peak | 3.33665e+08 | 3.36332e+08 | 3.37169e+08 | 3.36794e+08 | 3.37526e+08 | 1.74 / 2.00 | monotone | 0.132 | +0.217 (ext) | yes |
| plate_hole_finite_width_d03 | vm_max | 3.36961e+08 | 3.37425e+08 | 3.37417e+08 | 3.36794e+08 | 3.37417e+08 | 5.90 / 2.00 | oscillatory | 0.000 | +0.185 (fine) | yes |
| plate_hole_finite_width_d05 | s_peak | 4.29718e+08 | 4.33308e+08 | 4.34393e+08 | 4.31400e+08 | 4.34808e+08 | 1.87 / 2.00 | monotone | 0.119 | +0.790 (ext) | yes |
| plate_hole_finite_width_d05 | vm_max | 4.33726e+08 | 4.34779e+08 | 4.34723e+08 | 4.31400e+08 | 4.34720e+08 | 4.47 / 2.00 | oscillatory | 0.001 | +0.770 (fine) | yes |
| rotating_disk | ur_rim | 1.07277e-04 | 1.07326e-04 | 1.07324e-04 | 1.07324e-04 | 1.07324e-04 | 4.97 / 3.00 | oscillatory | 0.000 | +0.000 (fine) | yes |
| rotating_disk | s_center | 2.04118e+08 | 2.02819e+08 | 2.02495e+08 | 2.02383e+08 | 2.02390e+08 | 2.06 / 2.00 | monotone | 0.065 | +0.004 (ext) | yes |
| rotating_disk | hoop_rim | 8.56118e+07 | 8.61717e+07 | 8.59161e+07 | 8.58594e+07 | 8.57027e+07 | 1.15 / 2.00 | oscillatory | 0.310 | +0.066 (fine) | yes |
| rotating_disk | vm_max | 2.04122e+08 | 2.02817e+08 | 2.02495e+08 | 2.02383e+08 | 2.02391e+08 | 2.07 / 2.00 | monotone | 0.064 | +0.004 (ext) | yes |
| rotating_disk | strain_energy | 62.8937 | 62.8953 | 62.8954 | 62.8954 | 62.8954 | 4.25 / 4.00 | monotone | 0.000 | -0.000 (ext) | yes |
| timoshenko_cantilever | tip_deflection | -0.00256877 | -0.00256875 | -0.00256875 | -0.00256875 | -0.00256875 | 3.57 / 3.00 | monotone | 0.000 | +0.000 (ext) | yes |
| timoshenko_cantilever | sxx_top_mid | 7.49672e+07 | 7.50000e+07 | 7.50000e+07 | 7.50000e+07 | 7.50000e+07 | 14.58 / 2.00 | monotone | 0.000 | +0.000 (ext) | yes |
| timoshenko_cantilever | ux_tip_top | 3.75004e-04 | 3.75001e-04 | 3.75000e-04 | 3.75000e-04 | 3.75000e-04 | 2.76 / 3.00 | monotone | 0.000 | -0.000 (ext) | yes |
| timoshenko_cantilever | vm_max | 1.49602e+08 | 1.49907e+08 | 1.49976e+08 | 1.50000e+08 | 1.49996e+08 | 2.23 / 2.00 | monotone | 0.017 | -0.003 (ext) | yes |
| timoshenko_cantilever | strain_energy | 12.8898 | 12.89 | 12.89 | 12.89 | 12.89 | 3.84 / 4.00 | monotone | 0.000 | +0.000 (ext) | yes |

| problem | h nominal | cells (finest) | dofs (finest) | solve time total [s] | study passed |
|---|---|---|---|---|---|
| kirsch_exact_traction | 0.04, 0.02, 0.01 | 2864 | 11754 | 1.1 | yes |
| lame_cylinder_axisym_open | 0.0125, 0.00625, 0.003125 | 1208 | 5026 | 0.2 | yes |
| lame_cylinder_axisym | 0.0125, 0.00625, 0.003125 | 1208 | 5026 | 0.2 | yes |
| lame_cylinder_plane_strain | 0.0125, 0.00625, 0.003125 | 5660 | 23074 | 0.9 | yes |
| lame_sphere_3d | 0.03, 0.02, 0.01333 | 18306 | 83805 | 16.3 | yes |
| lame_sphere_axisym | 0.0125, 0.00625, 0.003125 | 5660 | 23074 | 0.9 | yes |
| lbracket_reentrant | 0.01, 0.005, 0.0025 | 3006 | 12410 | 0.7 | yes |
| lshape_poisson | 0.2, 0.1, 0.05 | 2810 | 5781 | 0.2 | yes |
| plate_hole_finite_width_d01 | 0.0015, 0.00075, 0.000375 | 8214 | 33534 | 1.9 | yes |
| plate_hole_finite_width_d03 | 0.004, 0.002, 0.001 | 9437 | 38408 | 2.0 | yes |
| plate_hole_finite_width_d05 | 0.006, 0.003, 0.0015 | 7062 | 28830 | 1.7 | yes |
| rotating_disk | 0.05, 0.025, 0.0125 | 760 | 3186 | 0.2 | yes |
| timoshenko_cantilever | 0.05, 0.025, 0.0125 | 2990 | 12346 | 0.5 | yes |

### Order 1 (P1, straight-sided P1 geometry)

| problem | QoI | phi(h1) | phi(h2) | phi(h3) | reference | phi_ext | p obs / th | convergence | GCI_fine % | err % | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kirsch_exact_traction | sxx_hole_top | 2.89960e+08 | 2.96384e+08 | 2.98545e+08 | 3.00000e+08 | 2.99575e+08 | 1.65 / 1.00 | monotone | 0.431 | -0.142 (ext) | yes |
| kirsch_exact_traction | syy_hole_side | -9.11929e+07 | -9.65020e+07 | -9.85032e+07 | -1.00000e+08 | -9.96390e+07 | 1.48 / 1.00 | monotone | 1.441 | -0.361 (ext) | yes |
| kirsch_exact_traction | ux_hole_side | 7.44127e-05 | 7.48363e-05 | 7.49555e-05 | 7.50000e-05 | 7.49993e-05 | 1.92 / 2.00 | monotone | 0.073 | -0.001 (ext) | yes |
| kirsch_exact_traction | uy_hole_top | -2.46075e-05 | -2.48886e-05 | -2.49690e-05 | -2.50000e-05 | -2.49993e-05 | 1.89 / 2.00 | monotone | 0.152 | -0.003 (ext) | yes |
| kirsch_exact_traction | ux_right | 1.37860e-04 | 1.38053e-04 | 1.38103e-04 | 1.38120e-04 | 1.38119e-04 | 2.04 / 2.00 | monotone | 0.015 | -0.001 (ext) | yes |
| kirsch_exact_traction | vm_max | 2.94799e+08 | 3.00128e+08 | 3.00422e+08 | 3.00000e+08 | 3.00438e+08 | 4.34 / 1.00 | monotone | 0.007 | +0.146 (ext) | yes |
| kirsch_exact_traction | strain_energy | 3.24086 | 3.24306 | 3.24365 | 3.24385 | 3.24385 | 1.98 / 2.00 | monotone | 0.008 | -0.000 (ext) | yes |
| lame_cylinder_axisym_open | ur_bore | 9.82907e-05 | 9.83205e-05 | 9.83310e-05 | 9.83333e-05 | 9.83362e-05 | 1.61 / 2.00 | monotone | 0.007 | +0.003 (ext) | yes |
| lame_cylinder_axisym_open | hoop_bore | 1.69350e+08 | 1.68215e+08 | 1.67386e+08 | 1.66667e+08 | 1.65514e+08 | 0.53 / 1.00 | monotone | 1.398 | -0.692 (ext) | yes |
| lame_cylinder_axisym_open | uz_top | -5.00132e-06 | -5.00041e-06 | -5.00023e-06 | -5.00000e-06 | -5.00019e-06 | 2.50 / 2.00 | monotone | 0.001 | +0.004 (ext) | yes |
| lame_cylinder_axisym_open | vm_max | 2.30029e+08 | 2.31865e+08 | 2.32613e+08 | 2.33333e+08 | 2.33073e+08 | 1.40 / 1.00 | monotone | 0.247 | -0.112 (ext) | yes |
| lame_cylinder_axisym_open | strain_energy | 154.405 | 154.447 | 154.458 | 154.462 | 154.462 | 2.07 / 2.00 | monotone | 0.003 | -0.000 (ext) | yes |
| lame_cylinder_axisym | ur_bore | 9.52898e-05 | 9.53209e-05 | 9.53312e-05 | 9.53333e-05 | 9.53358e-05 | 1.71 / 2.00 | monotone | 0.006 | +0.003 (ext) | yes |
| lame_cylinder_axisym | hoop_bore | 1.69345e+08 | 1.68207e+08 | 1.67383e+08 | 1.66667e+08 | 1.65579e+08 | 0.55 / 1.00 | monotone | 1.348 | -0.653 (ext) | yes |
| lame_cylinder_axisym | szz_bore | 2.26843e+07 | 2.14917e+07 | 2.07598e+07 | 2.00000e+07 | 1.97487e+07 | 0.79 / 1.00 | monotone | 6.088 | -1.256 (ext) | yes |
| lame_cylinder_axisym | vm_max | 2.27832e+08 | 2.29764e+08 | 2.30561e+08 | 2.31325e+08 | 2.31059e+08 | 1.39 / 1.00 | monotone | 0.271 | -0.115 (ext) | yes |
| lame_cylinder_axisym | strain_energy | 149.692 | 149.735 | 149.746 | 149.749 | 149.749 | 2.07 / 2.00 | monotone | 0.003 | -0.000 (ext) | yes |
| lame_cylinder_plane_strain | ur_bore | 9.51047e-05 | 9.52866e-05 | 9.53228e-05 | 9.53333e-05 | 9.53316e-05 | 2.36 / 2.00 | monotone | 0.012 | -0.002 (ext) | yes |
| lame_cylinder_plane_strain | hoop_bore | 1.68533e+08 | 1.67516e+08 | 1.67042e+08 | 1.66667e+08 | 1.66636e+08 | 1.12 / 1.00 | monotone | 0.304 | -0.018 (ext) | yes |
| lame_cylinder_plane_strain | szz_bore | 2.39475e+07 | 2.18178e+07 | 2.09181e+07 | 2.00000e+07 | 2.02716e+07 | 1.26 / 1.00 | monotone | 3.863 | +1.358 (ext) | yes |
| lame_cylinder_plane_strain | vm_max | 2.30590e+08 | 2.30969e+08 | 2.31339e+08 | 2.31325e+08 | 2.43383e+08 | 0.04 / 1.00 | monotone | 6.508 | +5.213 (ext) | **no** |
| lame_cylinder_plane_strain | strain_energy | 746.634 | 748.251 | 748.623 | 748.746 | 748.732 | 2.15 / 2.00 | monotone | 0.018 | -0.002 (ext) | yes |
| lame_sphere_axisym | ur_bore | 3.98823e-05 | 3.99756e-05 | 3.99945e-05 | 4.00000e-05 | 3.99993e-05 | 2.33 / 2.00 | monotone | 0.015 | -0.002 (ext) | yes |
| lame_sphere_axisym | uz_pole | 3.99468e-05 | 4.00411e-05 | 4.00137e-05 | 4.00000e-05 | 4.00026e-05 | 1.80 / 2.00 | oscillatory | 0.035 | +0.034 (fine) | yes |
| lame_sphere_axisym | hoop_bore | 7.60007e+07 | 7.37283e+07 | 7.25815e+07 | 7.14286e+07 | 7.14354e+07 | 1.01 / 1.00 | monotone | 1.974 | +0.010 (ext) | yes |
| lame_sphere_axisym | vm_max | 1.68690e+08 | 1.69650e+08 | 1.70685e+08 | 1.71429e+08 | - | - / 1.00 | divergent | - | -0.434 (fine) | yes |
| lame_sphere_axisym | strain_energy | 125.172 | 125.547 | 125.635 | 125.664 | 125.661 | 2.13 / 2.00 | monotone | 0.026 | -0.002 (ext) | yes |
| lbracket_reentrant | vm_max | 5.77352e+07 | 7.44723e+07 | 9.60808e+07 | - | - | - / 1.00 | divergent (diverge expected) | - | - | yes |
| lbracket_reentrant | tip_uy | -3.21019e-04 | -3.28745e-04 | -3.31736e-04 | - | -3.33463e-04 | 1.46 / 1.09 | monotone | 0.651 | - | n/a |
| lbracket_reentrant | strain_energy | 0.032125 | 0.0329016 | 0.0332022 | - | 0.0333756 | 1.46 / 1.09 | monotone | 0.653 | - | n/a |
| lshape_poisson | T_point | 0.792013 | 0.793031 | 0.793435 | 0.793701 | 0.793691 | 1.38 / 1.33 | monotone | 0.040 | -0.001 (ext) | yes |
| lshape_poisson | q_max | 1.73962 | 2.18981 | 2.83392 | - | - | - / 1.00 | divergent (diverge expected) | - | - | yes |
| lshape_poisson | T_integral | 1.58016 | 1.58243 | 1.58334 | 1.58393 | 1.58391 | 1.38 / 1.33 | monotone | 0.045 | -0.001 (ext) | yes |
| plate_hole_finite_width_d01 | s_peak | 2.97149e+08 | 3.01655e+08 | 3.02802e+08 | 3.03542e+08 | 3.03178e+08 | 2.03 / 1.00 | monotone | 0.155 | -0.120 (ext) | yes |
| plate_hole_finite_width_d01 | vm_max | 3.02116e+08 | 3.05011e+08 | 3.03901e+08 | 3.03542e+08 | 3.03224e+08 | 1.41 / 1.00 | oscillatory | 0.279 | +0.118 (fine) | yes |
| plate_hole_finite_width_d03 | s_peak | 3.32857e+08 | 3.36595e+08 | 3.37860e+08 | 3.36794e+08 | 3.38518e+08 | 1.56 / 1.00 | monotone | 0.243 | +0.512 (ext) | yes |
| plate_hole_finite_width_d03 | vm_max | 3.36127e+08 | 3.37138e+08 | 3.38618e+08 | 3.36794e+08 | - | - / 1.00 | divergent | - | +0.542 (fine) | yes |
| plate_hole_finite_width_d05 | s_peak | 4.27761e+08 | 4.32212e+08 | 4.32570e+08 | 4.31400e+08 | 4.32601e+08 | 3.67 / 1.00 | monotone | 0.009 | +0.279 (ext) | yes |
| plate_hole_finite_width_d05 | vm_max | 4.30866e+08 | 4.36106e+08 | 4.35184e+08 | 4.31400e+08 | 4.34988e+08 | 2.53 / 1.00 | oscillatory | 0.056 | +0.877 (fine) | yes |
| rotating_disk | ur_rim | 1.07261e-04 | 1.07283e-04 | 1.07319e-04 | 1.07324e-04 | - | - / 2.00 | divergent | - | -0.004 (fine) | yes |
| rotating_disk | s_center | 2.02000e+08 | 2.02307e+08 | 2.02360e+08 | 2.02383e+08 | 2.02371e+08 | 2.57 / 1.00 | monotone | 0.007 | -0.006 (ext) | yes |
| rotating_disk | hoop_rim | 8.91363e+07 | 8.70622e+07 | 8.69349e+07 | 8.58594e+07 | 8.69264e+07 | 4.09 / 1.00 | monotone | 0.012 | +1.243 (ext) | yes |
| rotating_disk | vm_max | 2.01999e+08 | 2.02297e+08 | 2.02360e+08 | 2.02383e+08 | 2.02376e+08 | 2.29 / 1.00 | monotone | 0.010 | -0.003 (ext) | yes |
| rotating_disk | strain_energy | 62.4875 | 62.793 | 62.8689 | 62.8954 | 62.8946 | 2.04 / 2.00 | monotone | 0.051 | -0.001 (ext) | yes |
| timoshenko_cantilever | tip_deflection | -0.00251578 | -0.00255444 | -0.00256511 | -0.00256875 | -0.00256871 | 1.99 / 2.00 | monotone | 0.175 | -0.002 (ext) | yes |
| timoshenko_cantilever | sxx_top_mid | 7.33234e+07 | 7.45383e+07 | 7.48775e+07 | 7.50000e+07 | 7.49940e+07 | 1.97 / 1.00 | monotone | 0.194 | -0.008 (ext) | yes |
| timoshenko_cantilever | ux_tip_top | 3.67004e-04 | 3.72845e-04 | 3.74451e-04 | 3.75000e-04 | 3.74992e-04 | 2.00 / 2.00 | monotone | 0.180 | -0.002 (ext) | yes |
| timoshenko_cantilever | vm_max | 1.46133e+08 | 1.48704e+08 | 1.49610e+08 | 1.50000e+08 | 1.50046e+08 | 1.63 / 1.00 | monotone | 0.364 | +0.030 (ext) | yes |
| timoshenko_cantilever | strain_energy | 12.6225 | 12.8172 | 12.8715 | 12.89 | 12.89 | 1.98 / 2.00 | monotone | 0.180 | +0.000 (ext) | yes |

| problem | h nominal | cells (finest) | dofs (finest) | solve time total [s] | study passed |
|---|---|---|---|---|---|
| kirsch_exact_traction | 0.02, 0.01, 0.005 | 11289 | 11586 | 0.9 | yes |
| lame_cylinder_axisym_open | 0.00625, 0.003125, 0.001563 | 4776 | 4970 | 0.3 | yes |
| lame_cylinder_axisym | 0.00625, 0.003125, 0.001563 | 4776 | 4970 | 0.3 | yes |
| lame_cylinder_plane_strain | 0.00625, 0.003125, 0.001563 | 22473 | 22906 | 1.4 | **no** |
| lame_sphere_axisym | 0.00625, 0.003125, 0.001563 | 22473 | 22906 | 1.4 | yes |
| lbracket_reentrant | 0.005, 0.0025, 0.00125 | 11912 | 12298 | 0.7 | yes |
| lshape_poisson | 0.1, 0.05, 0.025 | 11098 | 5710 | 0.4 | yes |
| plate_hole_finite_width_d01 | 0.00075, 0.000375, 0.0001875 | 32596 | 33274 | 2.7 | yes |
| plate_hole_finite_width_d03 | 0.002, 0.001, 0.0005 | 37311 | 37970 | 3.5 | yes |
| plate_hole_finite_width_d05 | 0.003, 0.0015, 0.00075 | 27999 | 28576 | 2.9 | yes |
| rotating_disk | 0.025, 0.0125, 0.00625 | 2939 | 3084 | 0.2 | yes |
| timoshenko_cantilever | 0.025, 0.0125, 0.00625 | 11904 | 12290 | 0.9 | yes |

### Discussion

* **Lamé, Kirsch, cantilever, disk, sphere (P2)**: observed orders match theory (stress ~2, displacement
  ~3, energy ~4) and the extrapolated values agree with the exact ones to <= 0.05 % (3D sphere: 0.3 % on a
  displacement whose three values all lie within 0.05 % of exact). The axisymmetric and plane-strain
  cylinders agree with each other and with the exact u_r to 1e-5, which verifies the axisymmetric
  formulation, the axis constraint and the 2*pi*r energy weighting.
* **Noise floor of the GCI**: for QoIs converged to ~1e-5 on the coarsest mesh (Kirsch ux_hole_side,
  vm_max; sphere-axisym hoop_bore) mesh-to-mesh changes are noise, and the three-mesh analysis reports
  `divergent` / `oscillatory_divergent`. These are not failures (the finest value is within 0.005 % of
  exact). The evaluator should not read such flags as setup errors when |e_a21| < ~1e-4.
* **Point stresses on boundary nodes converge pre-asymptotically on coarse meshes.** With the first
  choice h0 = 0.025 (4 elements through the wall) the axisymmetric hoop / axial stresses at the bore had
  p = 0.12 / 0.39. Richardson then extrapolated to -2.5 % / -5.8 % although the finest values were
  +0.07 % / +0.6 % off, and the study failed. With h0 = 0.0125 all are asymptotic (p = 2.2 / 2.3). Same
  story in 3D: on uniform tet10 meshes (<= 7.5 elements through the wall) the L2-recovered stress at bore
  nodes scatters by -3.7 % ... +2.3 % node to node, and the vertex (a, 0, 0), where three boundary surfaces
  meet, is the worst. Hence the hoop QoI sits on the bore / y = 0 edge at 45 degrees and the mesh is
  graded to the bore. Displacements and energy were always fine.
* **Peterson chart 4.1 vs FE**: converged FE peak stresses (GCI 0.12-0.17 %) exceed the curve fit by
  +0.05 % (d/W = 0.1), +0.22 % (0.3) and +0.79 % (0.5). This is within the fit's 2 % tolerance and
  consistent with a fit that loses accuracy at large d/W. End effects are negligible: lengthening the
  plate from 4 W to 6 W changed the d/W = 0.5 peak by < 1e-5 relative.
* **Singular problems**: `lbracket_reentrant` vm_max grows 59.8 -> 75.2 -> 99.9 MPa (P2) and `lshape_poisson`
  q_max 1.85 -> 2.34 -> 2.94. Both are classified `divergent`, which is the expected verdict. The
  converging QoIs show the reduced orders predicted by the singularity: L-shape p = 1.37-1.39 vs 4/3 (P1
  and P2 alike: higher order does not help on uniform meshes); bracket p = 1.00 (P2) and 1.46 (P1) vs
  2 lambda = 1.09. The P1 value is pre-asymptotic, and the extrapolated tip_uy agrees between P1 and P2
  to 0.06 %.
* **P1 failure** `lame_cylinder_plane_strain` vm_max: the values -0.32 %, -0.15 %, +0.006 % cross the exact
  value, so p = 0.04 and Richardson extrapolates to +5.2 %. One more level (h = 0.003125 -> 0.00078) gives
  p = 1.63, extrapolated error +0.08 %, pass. Cause: pre-asymptotic P1 boundary-node stresses (polygonal
  bore). This is neither the problem definition, the reference nor the solver. `_judge` uses the
  extrapolated value whenever the convergence is monotone, even with p = 0.04 against a theoretical 1.
  Similar irregular P1 boundary stresses: `rotating_disk` hoop_rim (+3.8, +1.4, +1.25, +0.66, +0.29 % on
  five meshes: it does converge) and the P1 Lamé bore stresses (p 0.5-1.3).

### Cross-check with CalculiX (`backend="ccx"`)

Second (medium) study mesh, P2, same case JSON. Largest |ccx - skfem| / |skfem|:

| problem | displacements / energies | recovered stresses |
|---|---|---|
| kirsch_exact_traction | 0.001 % | 0.60 % (syy_hole_side) |
| lame_cylinder_axisym (+ open) | 0.000 % | 0.61 % (szz_bore) |
| lame_cylinder_plane_strain | 0.000 % | 0.53 % (szz_bore) |
| lame_sphere_axisym | 0.003 % | 0.26 % (hoop_bore) |
| lame_sphere_3d | 0.001 % | 0.40 % (vm_max) |
| lshape_poisson | 0.000 % | 4.0 % (q_max, singular) |
| lbracket_reentrant | 0.024 % | 8.6 % (vm_max, singular) |
| plate_hole_finite_width_d01/03/05 | - | 0.56 / 0.25 / 0.19 % (s_peak) |
| rotating_disk | 0.014 % | 0.42 % (hoop_rim) |
| timoshenko_cantilever | 0.009 % | 0.05 % |

Stress differences come from the recovery (skfem: global L2 projection; ccx: extrapolated and averaged
integration-point values). Both converge to the exact values. **Finding**: with the original thicknesses
ccx disagreed for thick plane-stress models: cantilever t/D = 0.5 gave 0.71 % less deflection and
energy, bracket t/b = 0.25 gave 0.33 %, Kirsch t/a = 0.2 gave 0.09 %. The difference vanishes as t -> 0
(cantilever: 0.71 / 0.008 / 0.0000 % for t = 0.1 / 0.01 / 0.001 m). CalculiX solves CPS elements as an
expanded 3D layer of the given thickness, i.e. a thick slab rather than 2D plane stress. The problems now
use thin plates. Reproducer: `/tmp/analytical/repro_ccx_plane_stress_thickness/` (case.json, mesh.msh,
README.txt).

## Tests

`tests/test_problems_analytical.py` (87 tests, about 7 s; `-m "not slow"` skips the 3D solve and the full
studies):

* registry: all 13 ids registered, no import errors;
* per problem: `case(...).check()` for P1 / P2, JSON round trip, QoIs = case QoIs, references refer to
  QoIs, diverging QoIs have no reference, `private()` / `statement()` serialisable;
* per problem: the reference mesh builds at h0, contains every group the case references (read back with
  meshio), point QoIs lie on the mesh; STEP export writes a valid file that re-imports;
* per problem: no reference value appears in `statement()`;
* exact fields: point references = `exact()`; Hooke + equilibrium (incl. body force) by finite
  differences for every elasticity problem with an exact field; Laplace + flux + per-edge Dirichlet data
  for the L-shape; expression BCs = exact fields on their edges; Peterson / Williams constants;
* solver (skipped without the skfem backend): one coarse P2 solve per family against the references,
  the bracket singularity (vm_max grows, tip_uy converges); `slow`: the 3D sphere, and full convergence
  studies of Kirsch, L-shape, bracket and plate d/W = 0.5 that must pass (including the divergence verdicts).

## Recommendations for the verification layer (not implemented here)

1. `study._judge`: use the extrapolated value only when the observed order is plausible (e.g.
   0.5 p_th <= p <= 2 p_th and no `not_asymptotic` flag); otherwise judge the finest value.
2. `gci3`: add a noise floor, so that |e21|, |e32| < ~1e-5 |phi| counts as converged rather than
   divergent / oscillatory.
3. Register the `slow` pytest marker in `pyproject.toml`.
4. ccx backend: run plane-stress cases with a small nominal thickness and rescale (or document that ccx
   plane stress is a 3D slab).
5. case.py contract: say what `szz` means in axisymmetric runs (skfem returns the hoop stress there).
