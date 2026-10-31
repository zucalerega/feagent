# Wingbox LC1 verification verdict (2026-10-31)

**Overall: PASS.** Equilibrium, beam theory and convergence pass. The peak von Mises stress is a
clamped-root corner singularity, so it diverges under refinement and is not a design value. The
stresses away from the root converge and agree with M c / I.

Units: mm, N, MPa. 20 kN +z on the tip rib face, root faces fully fixed, 7075-T6 (E 71700, nu 0.33).
Numbers come from `verify/verification.json`, written by `verify/verify_all.py` (CPU, laptop) from
`<case>/gpu/result.json` and `result.vtu`. Re-running `verify_all.py` resets `summary.json`'s
`verdict` to null, so set it again afterwards.

## Solves (cuDSS on the NVIDIA GB10, DGX Spark)

Each case ran as `python -m feacore.cli case.json --out result.json --vtu result.vtu --linear-solver cudss`.
The command was wrapped in `systemd-run --user --scope -p MemoryMax=60G`, with a 1 s watchdog that
kills the job if MemAvailable < 12 GiB (`verify/run_case.sh`). Code: `~/feagent-mh`, a detached
worktree at origin/mesh-harness dd7292c. The Spark's `~/feagent` (spark-local-llm) cannot load these
cases (`Material.__init__() got an unexpected keyword argument 'rho'`). It also has the old full-SVD
rigid-body check: with ~1e5 constrained DOFs that check builds an n x n matrix of tens of GB.
`~/feagent` was not modified.

| mesh | DOFs | solve() s | cuDSS factor+solve s | process wall s | rel. residual | cgroup peak GiB | MemAvailable drop GiB | min MemAvailable GiB | watchdog |
|---|---|---|---|---|---|---|---|---|---|
| c2 (x1.69) | 150,213 | 3.20 | 1.31 | 5.0 | 1.6e-9 | 1.11 | 1.57 | 116.4 | not triggered |
| c1 (x1.3) | 237,273 | 4.80 | 2.03 | 6.0 | 1.6e-9 | 1.45 | 2.82 | 114.8 | not triggered |
| fine (x1) | 370,170 | 7.65 | 3.27 | 10.1 | 1.9e-9 | 1.93 | 3.89 | 113.2 | not triggered |
| **xf (x0.56)** | **1,150,527** | **24.24** | **9.79** | 29.1 | 4.0e-9 | 4.96 | 11.33 | 105.6 | not triggered |

Every run reported `linear_solver=cudss (cuDSS via nvmath-python, SPD)` and fell back to nothing
else. On c2 the cuDSS result matches the earlier laptop SciPy sanity solve to 1e-10 (tip 7.671101,
strain energy 76711.011).

## xf mesh (`assembly_xf/`)

The xf spec is `assembly/mesh.yaml` with every size (global, min, refine) multiplied by 0.56, giving
19.04 / 10.64 / 7.84 mm. The refine distances are unchanged. The mesh has 208,781 tet10 elements and
383,509 nodes, or 1,150,527 DOFs.

`feagent check` results:
- Pass: geometry (10 volumes), groups (root 5, tip 1, tip_ends 4, lower_outer 1), inverted 0, SICN 0.235,
  gamma 0.153, volume fidelity 3.4e-12, area fidelity 3.9e-15, connectivity 1.
- **thin_walls warns: min 0.96 elements through the wall (threshold 2).** Per family: ribs 1.08,
  spar webs 0.96 to 1.39, skins 1.06 to 1.98.
  - Both minima are on the long, single outer faces (spar outer webs, skin outer faces). On those faces
    the refine field under-delivers. This is the same Distance-field sampling issue that README.md
    reports.
  - Two elements through every wall would need about 3 M DOFs, so it does not fit the 1.0 to 1.3 M
    budget.
  - A non-uniform trial (15.5 / 11.6 / 7.8, 1.20 M DOFs) reached only 1.04. I kept the uniform scale
    so the four meshes form one systematic family.

## (a) Equilibrium: PASS

The root reaction sums balance the applied 20 kN to about 1e-11 (tolerance 1e-6).

| mesh | R_z (N) | rel. error R_z vs 20 kN | abs(R_x)/P | abs(R_y)/P | abs(U - P ū/2)/U |
|---|---|---|---|---|---|
| c2 | -20000.0000009 | 4.7e-11 | 7.7e-12 | 4.4e-12 | 2.2e-11 |
| c1 | -20000.0000016 | 8.2e-11 | 1.5e-11 | 1.8e-11 | 4.3e-11 |
| fine | -19999.9999989 | 5.3e-11 | 2.6e-12 | 1.4e-11 | 5.8e-12 |
| xf | -19999.99999986 | 6.8e-12 | 1.5e-12 | 2.6e-11 | 5.7e-11 |

## (b) Box-beam theory: PASS (FE is 2.5 % above the elementary estimate; limit 10 %)

`verify/beam_theory.py` uses the parallel-axis theorem about the centroid at z = 75 and ignores the ribs.
- Skins: 2 x (600 x 20^3 / 12 + 600 x 20 x 65^2) = 102.20e6 mm^4.
- Webs: 2 x 15 x 110^3 / 12 = 3.33e6 mm^4.
- I = 105.53e6 mm^4.

δ = P L^3 / (3 E I) + P L / (G A_webs), with G = 26,955 MPa and A_webs = 2 x 15 x 110 = 3300 mm^2:
- bending term 7.049 mm;
- shear term 0.450 mm;
- **δ_beam = 7.498 mm.**

| mesh | mean u_z tip rib | vs δ_beam | mean u_z tip_ends (skin and spar ends) |
|---|---|---|---|
| c2 | 7.6711 | +2.30 % | 7.6606 |
| c1 | 7.6756 | +2.36 % | 7.6651 |
| fine | 7.6788 | +2.40 % | 7.6683 |
| xf | 7.6833 | +2.47 % | 7.6728 |
| extrapolated (h → 0) | 7.6866 | +2.51 % | 7.6760 |

The converged FE is 0.188 mm (2.5 %) more flexible than the elementary formula. In order of size:
1. **The shear term is the weak part of the formula. Shear lag and warping restraint at the clamped
   root decide where the FE lands.**
   - The term P L / (G A_webs) puts all of the shear uniformly into the webs. In the closed box the
     vertical shear also drives an in-plane shear flow in the skins: q = V/I t 65 s, about 72 N/mm at
     the corners. That is the shear flow behind shear lag.
   - A thin-walled shear-flow estimate with free warping gives 0.372 mm (webs) + 0.376 mm (skins). That
     totals δ = 7.797 mm, which is an upper bound.
   - The fully fixed root face suppresses this warping, and the shear-lag zone, near y = 0. That
     removes part of the skin shear flexibility.
   - The FE (7.687) sits between the two bounds, at about 63 % of the way from the warping-restrained
     to the free-warping estimate.
   - The FE web shear at the neutral axis (y = 1000) is 5.17 MPa. Jourawski's V Q / (I 2t) gives
     5.21 MPa, and the uniform P / A_webs gives 6.06 MPa. So the shear is not uniform in the webs, as
     the elementary term assumes.
2. **Shear lag in the bending stresses is small here.** At y = 200 (root bay), σ_yy at the upper-skin
   mid-plane across the width runs -21.7 / -22.6 / -21.7 MPa (x = 7.5 over the spar / x = 100 / x = 300 centre). Beam theory
   gives -22.2. At y = 1000 σ_yy is -12.23 to -12.34, and beam theory gives -12.32. The skins work at
   about full width (the variation is under 4 %). The skins are thick (20 mm) and the span-to-width
   ratio is 3.3.
3. **Load introduction through the tip rib** adds only about 0.010 mm locally. The loaded rib face
   deflects 7.6866 mm, against 7.6760 for the skin and spar ends at y = 2000. The rib carries the load
   in-plane to the spar webs without much local deformation.
4. Ribs (ignored in the theory) and the clamp's restraint of Poisson contraction both stiffen the
   model slightly, so they partly offset item 1.

## (c) Convergence: PASS

h = DOFs^(-1/3). The refinement ratios are r(xf/fine) = 1.459, r(fine/c1) = 1.160 and r(c1/c2) = 1.165.

| QoI | c2 | c1 | fine | xf | p (3 finest) | φ_ext | GCI_fine (Fs 1.25) | GCI_fine conservative (Fs 3) | asymptotic ratio | 4-mesh fit p / φ0 (rms resid.) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean_uz_tip (mm) | 7.67110 | 7.67564 | 7.67879 | 7.68331 | **2.29** | **7.68659** | **0.054 %** | 0.13 % | 1.0006 | 2.28 / 7.68662 (2.1e-6) |
| strain_energy (N mm) | 76711.0 | 76756.4 | 76787.9 | 76833.1 | **2.29** | **76865.9** | **0.054 %** | 0.13 % | 1.0006 | 2.28 / 76866.2 (0.021) |
| mean_uz_tip_ends (mm) | 7.66062 | 7.66514 | 7.66827 | 7.67276 | 2.29 | 7.67601 | 0.053 % | 0.13 % | 1.0006 | 2.28 / 7.67604 |

Convergence is monotone. The three-mesh and four-mesh estimates agree: p 2.29 vs 2.28, φ_ext within
4e-6 relative. The coarse triple (c2/c1/fine) gives p = 2.26 and φ_ext = 7.68671.

Strain energy equals P ū / 2 exactly (uniform traction), so its order equals the tip order.

The observed order of 2.3 is below the smooth tet10 rates (3 for mean displacement, 4 for energy). The
cause is the root-corner singularity, which pollutes the global QoIs. The pollution is small: the xf
value is within 0.04 % of the extrapolated value.

**Harness finding:** `feacore.verification.gci3` (and so `analyze()['gci']`) labels both sequences
**"divergent"**, with R = e21/e32 = 1.43.
- The cause: it classifies by R < 1 before computing p. With unequal ratios (1.46 vs 1.16), a
  monotone sequence at p = 2.29 has R = (r21^p - 1) / (r21^p (r32^p - 1)) = 1.43.
- The p, φ_ext and GCI above use Celik's procedure with feacore's own `_observed_order`, implemented
  in `verify_all.py: celik`. They check monotonicity on the signs of e21 and e32.
- The R test in gci3 should account for r21 ≠ r32.

## (d) Peak von Mises: root-corner singularity, reported separately. Stresses away from the root PASS

| mesh | max von Mises (MPa) | at (x, y, z) | max vM, 300 ≤ y ≤ 1700 | max vM, 100 ≤ y ≤ 1900 | σ_yy upper skin (300, 1000, 140) |
|---|---|---|---|---|---|
| c2 | 43.77 | (600, 0, 150) | 23.94 | 29.22 | -12.340 |
| c1 | 48.18 | (600, 0, 150) | 24.19 | 28.86 | -12.346 |
| fine | 50.75 | (600, 0, 150) | 24.02 | 29.54 | -12.354 |
| xf | **60.61** | **(600, 0, 150)** | 24.19 | 29.25 | -12.333 |
| beam M c / I | 28.43 (root, outer fibre) | | ≈ 24.2 (y = 300, c = 75) | | -12.32 |

- **Where the peak is:** in all four meshes the peak is the same node, the outer corner of the clamped
  root at (600, 0, 150): x = 600 (rear-spar outer face), y = 0 (root), z = 150 (upper-skin outer
  surface). The 20 highest nodes all lie within y < 10 to 29 mm of the root, and that band narrows
  with h.
- **The peak is singular.** It grows monotonically as 43.8 → 48.2 → 50.7 → 60.6 MPa, and there is no
  converging fit (the fit p hits its 0.05 lower bound). A fully fixed face meeting free surfaces at
  ν ≠ 0 gives a stress singularity at the clamp edge and corner. **60.6 MPa is a mesh-dependent value,
  not a stress to design to.**
- **Away from the root the stresses converge:**
  - max vM over 300 ≤ y ≤ 1700 is 24.0 to 24.2 MPa (±0.5 %, scattered by stress recovery, no trend);
  - over 100 ≤ y ≤ 1900 it is 28.9 to 29.5 MPa;
  - mid-span skin σ_yy = -12.33 to -12.35 MPa, against -12.32 from beam theory (0.1 %).
- **For assessment:** use the converged fibre stress, about 28.4 MPa at the root by M c / I, rather
  than the singular corner peak. For a root stress, model the real attachment (fitting or bolts)
  instead of a fully fixed face. Either value is far below 7075-T6 yield (about 500 MPa).

## Files

- `verify/beam_theory.py`, `verify/verify_all.py`, `verify/run_case.sh` (the Spark watchdog
  runner), `verify/watchdog.log`, `verify/verification.json`
- `summary.json`
- `<case>/gpu/{result.json,result.vtu,solve.log,watchdog.txt}` for c2, c1, fine and xf
- `assembly_xf/{mesh.yaml,case.json,check.log,out/}`
