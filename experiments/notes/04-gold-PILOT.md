# Pilot: mesh harness vs gold certified answers (6 parts, 1 BC set each)

Date 2026-10-31, DGX Spark (GB10). Worktree `~/feagent-mh` (origin/mesh-harness, dd7292c), run with
`PYTHONPATH=~/feagent-mh/meshing-harness/src:~/feagent-mh` (harness and feacore from the worktree), every Python
job under `systemd-run --user --scope -p MemoryMax=40G` + a 12 GiB MemAvailable watchdog (`~/gold-catalog/scripts/wd.sh`).
Driver: `~/gold-catalog/scripts/gold_case.py` (`mesh` / `case` / `solve`), `pilot.sh`; outputs in `~/gold-catalog/pilot/<part>/`.

## Result

**5 of 6 parts solved; all 39 certified QoIs of the 5 solved sets lie inside their certified (GCI) band** (worst |err| 1.93 %,
u_mean_f10 of abc_00000974_bc4, band 2.93 %). Errors are of the size of the generator's own uniform L1-L2 meshes,
at 0.23-1.19 x the gold L2 DOFs. Gate failures: 2 of 6 (`sicn`), one of which (abc_00001748) still solved accurately.
Harness failure: abc_00000223 (257 faces) - automesh produced a **3.84M-node** tet10 mesh (11.5M DOFs, ~10 min per mesh,
sicn fail) vs gold L2 of 125k DOFs; not solvable within the memory budget, so not solved.
The 5th part was originally abc_00000223; abc_00000264_bc9 (260 faces) was added so that 5 parts have answers.

| set | faces | gates | failed | harness DOFs | gold L1 / L2 DOFs | in band | worst abs err | automesh + check s | solve s |
|---|---|---|---|---|---|---|---|---|---|
| abc_00000627_bc4 | 6 | pass | - | 79,266 | 36,156 / 258,366 | 6/6 | 0.16% | 3 | 3 |
| abc_00000974_bc4 | 14 | pass | - | 214,464 | 75,450 / 561,042 | 11/11 | 1.93% | 6 | 9 |
| abc_00001748_bc0 | 25 | fail | sicn | 106,701 | 64,839 / 463,995 | 7/7 | 0.99% | 4 | 3 |
| abc_00000330_bc1 | 54 | pass | - | 528,729 | 62,526 / 443,418 | 6/6 | 0.64% | 46 | 16 |
| abc_00000264_bc9 | 260 | pass | - | 712,938 | 143,115 / 1,023,372 | 9/9 | 1.35% | 104 | 22 |
| abc_00000223_bc8 | 257 | fail | sicn | not solved | 46,782 / 124,566 | 0/0 | - | 1282 | 0 |

(automesh and check each mesh the part once; `solve s` = whole solve process incl. assembly, cuDSS, stress recovery, QoIs.)

## Method (per part, one BC set)

1. `feagent automesh part.step -o auto --no-render` (the gold STEP, in metres; automesh sizes from the 100 mm diagonal:
   global 2.5 mm, min 0.13 mm, curvature 12, thin_wall_layers 2, tet10, netgen optimise).
2. Spec = the automesh spec + one group per gold face the set uses (supports, loads, QoI faces), named as the gold
   (`f<tag>`), selected with `{select: {ids: [<tag>]}, expect: 1}` (the faithful mapping: gold names faces by OCC tag
   of this STEP; the harness warns "selects faces by ids" on every group, as expected) + `bcs.fea` with the gold BCs in
   SI: `fixed`, `loads` (total force vector), `pressure`, `displacement` (rollers: components, value 0), E 210 GPa, nu 0.3.
3. `feagent check mesh.yaml --no-render` -> gates (`out/report.json`).
4. Adapter `feacore.adapters.mesh_harness.case_from_mesh_harness(out_dir, fea=<bcs.fea>, mesh=out/mesh.msh)` -> `case.json`;
   its default QoIs are replaced by the gold `reference_case` QoI definitions (same names/types/groups), so every
   certified QoI is evaluated identically. BC types map 1:1 (gold fixed/force/pressure/displacement = feacore types).
5. Mapping check: per group, harness mesh facet area and centroid vs the gold part's face record
   (all groups of all 6 parts: area within 1e-3, centroid within 1e-4 of the diagonal -> the ids map to the right faces).
6. `feacore.solver.solve(case, backend="skfem", linear_solver="cudss")`; compare each certified reference:
   rel err = harness/gold - 1, inside band if |rel err| <= the reference's tolerance (max(limit, 2 x GCI)).

## Gold data sanity check (`scripts/gold.py verify`, FEACORE_LINEAR_SOLVER=cudss)

| set | level | DOFs | result |
|---|---|---|---|
| abc_00000627_bc4 | L1 | 36,156 | OK, 9/9 QoIs reproduce the study (1e-6) |
| abc_00000974_bc4 | L1 | 75,450 | OK, 12/12 |
| abc_00001748_bc0 | L1 | 64,839 | OK, 10/10 |
| abc_00000065_bc0 | L2 | 804,333 | OK, 5/5 (28.6 s with cuDSS) |

## Observations

- Mapping by `ids` is exact: the harness's gmsh STEP import keeps the OCC face tags the generator used.
- The harness default mesh is a good match for this benchmark when it works: 4 of 5 solved sets sit between gold L1 and
  L2 in DOFs and land at L1-to-L2 accuracy; abc_00000330 (54 faces, thin walls auto-refined) went to 1.19 x L2 DOFs.
- `vm_max` is certified (converged) for abc_00000627_bc4 and the harness matches it to 0.05 %.
- `sicn` gate: abc_00001748 fails at min SICN 0.063 (a sliver touching CAD face 2); the solution is still within band.
  Following the gate hints mechanically (hint's `refine` snippet, then size.min x3, then size.global x0.7; tested on this
  part) did not clear it (min SICN 0.038-0.044), so it stays a gate failure.
- Thin-wall auto-refinement can explode the mesh on complex parts (abc_00000223: 3.84M nodes, 10+ min per mesh, while
  the generator certified the part at 125k DOFs). This is the main harness risk for the full run.
- Gold-data caveat found while testing the full-run driver on abc_00000627 (all 10 sets): two certified reactions
  (abc_00000627_bc7 Ry_f4 / Ry_f6 = -1.2 N next to ~9 kN supports; Rx at the same faces is "oscillatory") are
  numerical noise certified as "converged"; the harness gives +0.29 N (-124 %). The full-run report treats reaction
  components below 0.1 % of the set's largest reaction separately.

## Per case
## Per case

### abc_00000627_bc4 (6 faces, STEP 18 kB)

BCs: Clamped (all displacements zero): face f2 (centroid (0.00, 0.00, 5.54) mm, area 1721.7 mm^2).; Roller (zero z-displacement only): planar face f4 (outward normal (0.000, -0.000, 1.000), centroid (0.00, -0.00, 71.44) mm, area 299.8 mm^2).; Uniform pressure 46.7 MPa pushing into the part on face f3 (centroid (0.00, -0.00, 35.72) mm, area 2983.4 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f2 area -6.9e-05, centroid 2.5e-07 diag, f4 area -4.2e-05, centroid 2.1e-14 diag, f3 area -1.0e-03, centroid 5.1e-06 diag

`feagent automesh`: status pass, 26422 nodes / 15758 tets, 1.5 s.
`feagent check` (spec = automesh sizes + groups + bcs): **pass**, 26422 nodes, 15758 tets, 1.5 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn pass, gamma pass, volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls skip.
Solve: 79,266 DOFs (gold L0 5,505, L1 36,156, L2 258,366; harness/L2 = 0.31), linear_solver=cudss (cuDSS via nvmath-python, SPD), n=74147, nnz=5503447, t=0.94s, rel_residual=3.85e-14, solve() 2.2 s, process peak RSS 1.2 GB.

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|
| strain_energy | 0.18103 | 0.18131 | -0.16% | ±1.00% | yes | -0.30% | -0.04% |
| vm_mean | 3.4408e+07 | 3.4444e+07 | -0.10% | ±3.00% | yes | -0.26% | -0.03% |
| vm_max | 1.1921e+08 | 1.1927e+08 | -0.05% | ±3.27% | yes | +3.16% | +1.32% |
| u_mean_f3 | 2.746e-06 | 2.7496e-06 | -0.13% | ±2.00% | yes | -0.30% | -0.04% |
| Rz_f2 | -2313.6 | -2316 | -0.10% | ±2.00% | yes | +0.19% | +0.02% |
| Rz_f4 | 2313.6 | 2316 | -0.10% | ±2.00% | yes | +0.19% | +0.02% |

Singular (no reference, not scored): u_max, Ry_f2.

### abc_00000974_bc4 (14 faces, STEP 98 kB)

BCs: Clamped (all displacements zero): cylindrical face f3 (radius 8.98 mm, axis (-0.000, -1.000, -0.000), centroid (33.38, -19.73, 25.56) mm, area 591.4 mm^2).; Clamped (all displacements zero): planar face f13 (outward normal (-0.000, 1.000, -0.000), centroid (3.80, 4.49, -4.07) mm, area 3150.4 mm^2).; Total force (8661.0, -18138.3, 0.0) N spread uniformly over cylindrical face f8 (radius 12.06 mm, axis (-0.000, -0.000, 1.000), centroid (0.00, -36.41, 10.09) mm, area 755.8 mm^2).; Total force (0.0, 0.0, -4500.0) N spread uniformly over planar face f10 (outward normal (0.000, -0.000, 1.000), centroid (4.90, -16.74, 29.16) mm, area 1111.7 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f3 area -8.8e-04, centroid 9.1e-05 diag, f13 area +8.9e-14, centroid 2.0e-14 diag, f8 area -3.0e-04, centroid 5.1e-05 diag, f10 area +2.6e-04, centroid 3.0e-05 diag

`feagent automesh`: status pass, 71488 nodes / 47679 tets, 3.34 s.
`feagent check` (spec = automesh sizes + groups + bcs): **pass**, 71488 nodes, 47679 tets, 3.09 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn pass, gamma pass, volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls skip.
Solve: 214,464 DOFs (gold L0 10,803, L1 75,450, L2 561,042; harness/L2 = 0.38), linear_solver=cudss (cuDSS via nvmath-python, SPD), n=205098, nnz=16359682, t=4.37s, rel_residual=2.85e-13, solve() 7.9 s, process peak RSS 1.9 GB.

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|
| strain_energy | 0.038164 | 0.03844 | -0.72% | ±1.05% | yes | -1.23% | -0.42% |
| u_max | 5.7037e-06 | 5.8064e-06 | -1.77% | ±3.19% | yes | -3.33% | -1.26% |
| vm_mean | 7.7601e+06 | 7.7924e+06 | -0.41% | ±3.00% | yes | -0.62% | -0.24% |
| u_mean_f8 | 3.9205e-06 | 3.9526e-06 | -0.81% | ±2.00% | yes | -1.33% | -0.48% |
| u_mean_f10 | 1.7943e-06 | 1.8296e-06 | -1.93% | ±2.93% | yes | -2.54% | -1.16% |
| Rx_f3 | -2822.7 | -2783.1 | +1.42% | ±2.18% | yes | +1.63% | +0.88% |
| Ry_f3 | 1324 | 1324.8 | -0.07% | ±2.00% | yes | +0.46% | +0.11% |
| Rz_f3 | 2130.6 | 2113.7 | +0.80% | ±2.00% | yes | +0.96% | +0.34% |
| Rx_f13 | -5838.8 | -5878.8 | -0.68% | ±2.00% | yes | -0.77% | -0.43% |
| Ry_f13 | 16814 | 16813 | +0.01% | ±2.00% | yes | -0.04% | -0.01% |
| Rz_f13 | 2369.7 | 2386.2 | -0.69% | ±2.00% | yes | -0.82% | -0.29% |

Singular (no reference, not scored): vm_max.

### abc_00001748_bc0 (25 faces, STEP 98 kB)

BCs: Clamped (all displacements zero): planar face f3 (outward normal (-0.000, 0.000, -1.000), centroid (0.00, -15.09, 0.00) mm, area 1853.7 mm^2).; Roller (zero y-displacement only): planar face f9 (outward normal (0.000, -1.000, -0.000), centroid (0.00, -45.07, 16.60) mm, area 2248.7 mm^2).; Uniform pressure 48.9 MPa pushing into the part on cylindrical face f1 (radius 20.58 mm, axis (-0.000, -0.000, 1.000), centroid (0.00, 13.10, 8.30) mm, area 1073.5 mm^2).; Total force (-9038.2, -15334.6, 0.0) N spread uniformly over cylindrical face f13 (radius 9.96 mm, axis (-0.000, -0.000, -1.000), centroid (24.20, -34.81, 16.60) mm, area 519.5 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f3 area +1.8e-04, centroid 2.0e-05 diag, f9 area -5.8e-14, centroid 2.9e-14 diag, f1 area -1.1e-04, centroid 3.0e-05 diag, f13 area -4.0e-04, centroid 7.2e-05 diag

`feagent automesh`: status fail (failed: sicn), 35567 nodes / 21099 tets, 2.15 s.
`feagent check` (spec = automesh sizes + groups + bcs): **fail**, 35567 nodes, 21099 tets, 1.85 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn fail (0.0628 vs 0.1), gamma pass, volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls skip.
Solve: 106,701 DOFs (gold L0 9,870, L1 64,839, L2 463,995; harness/L2 = 0.23), linear_solver=cudss (cuDSS via nvmath-python, SPD), n=99402, nnz=7291944, t=1.12s, rel_residual=1.18e-13, solve() 2.7 s, process peak RSS 1.3 GB.

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|
| strain_energy | 0.1472 | 0.14807 | -0.58% | ±1.80% | yes | -2.02% | -0.71% |
| u_max | 1.0125e-05 | 1.014e-05 | -0.14% | ±2.00% | yes | -0.88% | -0.26% |
| vm_mean | 2.655e+07 | 2.6669e+07 | -0.45% | ±3.00% | yes | -1.33% | -0.48% |
| u_mean_f1 | 5.3014e-06 | 5.332e-06 | -0.57% | ±2.20% | yes | -2.42% | -0.87% |
| u_mean_f13 | 4.1935e-06 | 4.2353e-06 | -0.99% | ±2.00% | yes | -1.83% | -0.64% |
| Rx_f3 | 9038.2 | 9038.2 | -0.00% | ±2.00% | yes | -0.00% | +0.00% |
| Ry_f3 | 33519 | 33540 | -0.06% | ±2.00% | yes | +0.02% | +0.00% |

Singular (no reference, not scored): vm_max, Rz_f3.

### abc_00000330_bc1 (54 faces, STEP 213 kB)

BCs: Clamped (all displacements zero): planar face f46 (outward normal (-0.000, 0.000, 1.000), centroid (-12.46, 3.45, 18.63) mm, area 132.3 mm^2).; Roller (zero y-displacement only): planar face f22 (outward normal (-0.000, -1.000, -0.000), centroid (16.56, 0.00, 8.80) mm, area 582.3 mm^2).; Uniform pressure 41.1 MPa pushing into the part on planar face f14 (outward normal (0.087, -0.996, -0.000), centroid (50.53, 1.51, 9.32) mm, area 372.4 mm^2).; Total force (0.0, -1800.0, 0.0) N spread uniformly over planar face f1 (outward normal (0.000, -1.000, 0.000), centroid (46.22, 2.76, 9.32) mm, area 128.0 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f46 area -5.4e-05, centroid 1.1e-05 diag, f22 area +3.4e-05, centroid 2.7e-06 diag, f14 area +4.2e-05, centroid 3.5e-06 diag, f1 area +4.5e-04, centroid 2.0e-05 diag

`feagent automesh`: status pass, 176243 nodes / 106910 tets, 23.84 s.
`feagent check` (spec = automesh sizes + groups + bcs): **pass**, 176243 nodes, 106910 tets, 21.82 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn pass, gamma pass, volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls warn (1.78 vs 2.0).
Solve: 528,729 DOFs (gold L0 9,675, L1 62,526, L2 443,418; harness/L2 = 1.19), linear_solver=cudss (cuDSS via nvmath-python, SPD), n=522179, nnz=39610411, t=5.97s, rel_residual=1.59e-10, solve() 13.7 s, process peak RSS 3.1 GB.

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|
| strain_energy | 9.2188 | 9.2465 | -0.30% | ±1.97% | yes | -2.18% | -0.78% |
| u_max | 0.0032625 | 0.0032679 | -0.17% | ±2.00% | yes | -1.57% | -0.51% |
| vm_mean | 4.0406e+08 | 4.0455e+08 | -0.12% | ±3.00% | yes | -1.61% | -0.48% |
| u_mean_f14 | 0.0013432 | 0.0013475 | -0.32% | ±2.02% | yes | -2.22% | -0.80% |
| u_mean_f1 | 0.00086921 | 0.00087485 | -0.64% | ±3.03% | yes | -3.02% | -1.20% |
| Rx_f46 | 1324.5 | 1324.5 | -0.00% | ±2.00% | yes | +0.00% | +0.00% |

Singular (no reference, not scored): vm_max.

### abc_00000264_bc9 (260 faces, STEP 970 kB)

BCs: Clamped (all displacements zero): planar face f55 (outward normal (-1.000, -0.000, -0.000), centroid (-42.56, 0.17, -0.18) mm, area 674.7 mm^2).; Clamped (all displacements zero): planar face f236 (outward normal (-0.000, 1.000, 0.000), centroid (0.00, 11.59, 0.97) mm, area 2281.8 mm^2).; Total force (-30500.0, 0.0, 0.0) N spread uniformly over planar face f206 (outward normal (1.000, 0.000, 0.000), centroid (42.56, 0.17, -0.18) mm, area 674.7 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f55 area +1.1e-13, centroid 2.6e-14 diag, f236 area +9.3e-14, centroid 1.9e-14 diag, f206 area +1.1e-13, centroid 2.6e-14 diag

`feagent automesh`: status pass, 237646 nodes / 139967 tets, 54.68 s.
`feagent check` (spec = automesh sizes + groups + bcs): **pass**, 237646 nodes, 139967 tets, 49.07 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn pass, gamma pass, volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls pass.
Solve: 712,938 DOFs (gold L0 21,858, L1 143,115, L2 1,023,372; harness/L2 = 0.70), linear_solver=cudss (cuDSS via nvmath-python, SPD), n=676911, nnz=50013337, t=8.39s, rel_residual=3.07e-12, solve() 19.2 s, process peak RSS 3.9 GB.

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|
| u_max | 0.0001746 | 0.00017681 | -1.25% | ±3.44% | yes | -4.22% | -1.36% |
| vm_mean | 5.0345e+07 | 5.0681e+07 | -0.66% | ±3.00% | yes | -1.90% | -0.64% |
| u_mean_f206 | 3.9708e-05 | 4.0253e-05 | -1.35% | ±3.48% | yes | -4.17% | -1.37% |
| Rx_f55 | 2081.6 | 2096.8 | -0.72% | ±2.00% | yes | -1.63% | -0.67% |
| Ry_f55 | -203.3 | -204.13 | -0.41% | ±2.00% | yes | -1.29% | -0.51% |
| Rz_f55 | 384.01 | 387.43 | -0.88% | ±2.08% | yes | -2.03% | -0.83% |
| Rx_f236 | 28419 | 28403 | +0.06% | ±2.00% | yes | +0.13% | +0.05% |
| Ry_f236 | 202.76 | 204.52 | -0.86% | ±2.37% | yes | -2.44% | -0.94% |
| Rz_f236 | -383.98 | -387.62 | -0.94% | ±2.24% | yes | -2.13% | -0.89% |

Singular (no reference, not scored): vm_max.

### abc_00000223_bc8 (257 faces, STEP 3812 kB)

BCs: Clamped (all displacements zero): face f154 (centroid (0.00, 0.00, 2.00) mm, area 408.4 mm^2).; Roller (zero z-displacement only): planar face f54 (outward normal (0.000, -0.000, 1.000), centroid (-19.30, 16.83, 2.34) mm, area 147.8 mm^2).; Total force (-6487.6, -14953.3, 0.0) N spread uniformly over cylindrical face f152 (radius 33.41 mm, axis (-0.000, -0.000, -1.000), centroid (0.02, -31.37, 2.00) mm, area 139.0 mm^2).

Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): f154 area -5.1e-07, centroid 4.4e-08 diag, f54 area +6.3e-06, centroid 2.7e-06 diag, f152 area +1.5e-04, centroid 1.3e-06 diag

`feagent automesh`: status fail (failed: sicn), 3842243 nodes / 2533342 tets, 672.74 s.
`feagent check` (spec = automesh sizes + groups + bcs): **fail**, 3842243 nodes, 2533342 tets, 609.7 s. Gates: geometry pass, groups pass, mesh_generated pass, inverted pass, sicn fail (0.00571 vs 0.1), gamma warn (0.000567 vs 0.05), volume_fidelity pass, area_fidelity pass, connectivity pass, thin_walls pass.
Solve: **not run**. The harness mesh has 3,842,243 nodes (11.5M DOFs; est. 28 KB/DOF -> ~325 GB), far over the GB10 budget, and gold L2 for this part is 124,566 DOFs. Cause: automesh thin-wall auto-refinement (faces 2, 8 at t = 1.64 mm -> size 0.66 mm) applied to large faces of a 257-face part, plus an sicn failure (min 0.0057).

| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |
|---|---|---|---|---|---|---|---|

Singular (no reference, not scored): -.

