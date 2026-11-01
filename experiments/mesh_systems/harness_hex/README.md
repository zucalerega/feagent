# harness_hex: the harness's gmsh, hexahedra where they work

Same two parts and contract as the other systems (`../CONTRACT.md`). One command per mesh:

    uv run --extra pardiso python experiments/mesh_systems/harness_hex/build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear

(`CUDA_VISIBLE_DEVICES=""`, 4 threads; `OUT_DIR/{mesh.msh, mesh.inp, report.json, mesh.png, mesh.yaml, out/}`.)

## Approach

- **Linear = tet10, not hex.** `feacore/solver/_skfem_mesh.py` lists `hexahedron*` in `UNSUPPORTED_DOMAIN`: feacore/skfem
  cannot solve hexes. The linear mesh therefore goes through the stock harness (`feagent.mesher.run_mesh`: curvature
  sizing, thin-wall refinement, Netgen optimisation, order 2 + HighOrder optimisation) from a generated `mesh.yaml`
  (faces by geometry: `kind: plane`, normal along the axis, `box` slab at the bbox extreme; counts checked by `expect`).
  Gates: the harness's own `assess_mesh`, unchanged.
- **Nonlinear = all-hex C3D8** (ccx accepts it; verified). Tried in the order of the brief:
  (a) *all-hex by subdivision*: tet mesh (Delaunay, Netgen-optimised) then `Mesh.SubdivisionAlgorithm=2` +
  `mesh.refine()`: each tet becomes 4 hexes. Works, but raw hexes are poor (min corner scaled Jacobian -0.7..+0.1, several
  inverted), so build.py runs a guarded Laplacian smoothing of interior nodes (a sweep is kept only if the worst corner
  SJ does not drop) and walks a list of base sizes (10 mm down to 3 mm), keeping the first mesh with min corner SJ >= 0.10.
  Boundary nodes are re-projected onto the CAD (gmsh `refine` already puts them on it: projection moved nothing).
  (b) *hex-dominant* (`Mesh.RecombineAll`, `Recombine3DAll`, `Algorithm3D` 1/4/10): does NOT work on these parts. Algorithm 4
  stops with "Cannot use frontal 3D algorithm with quadrangles on boundary", 1 and 10 silently return tets (type 4), and
  `RecombinationAlgorithm=3` fails with "Full-quad recombination not ready yet for periodic surfaces"; 10 once crashed
  ("corrupted size vs. prev_size"). No pure or dominant hex mesh came out of recombination, so only (a) is used.
  (c) fallback if no candidate reaches SJ >= 0.10: straight-sided tet10 + Netgen + HighOrder, labelled
  `mesh_kind: tet10-fallback` in report.json. Not needed for either part.
- Pieces used: `feagent.mesher` (stages, groups, exports; `_generate` is swapped for the hex generator), `feagent.geometry`,
  `feagent.quality.assess_mesh` (tets), `feagent.gmsh_util`, `feagent.render.render_quality` (with a hexahedron grid
  substituted), `feagent.gates`. No shared code edited.

## Knobs

`--h` fixes the (tet) global size; `--thin-factor` (harness `THIN_WALL_SIZE_FACTOR`; default 0.8 linear, 1.4 hex: the hex
mesh halves every tet edge so thin walls need only 1 tet layer; 2.0 made Netgen hang on part 3), `--curvature` (12 linear,
8 hex), `--no-hex`. Constants in build.py: `HEX_H_CANDIDATES`, `HEX_MIN_SJ`, `SMOOTH_ITERS`.

## Gates on hexes (harness gates are tet-oriented)

Applicable and reported (same thresholds): inverted (gmsh minSJ <= 0), SICN >= 0.1 (gmsh hex SICN), volume fidelity <= 1 %,
group area fidelity <= 2 %, connectivity (node-sharing components), plus the harness's geometry/groups/mesh_generated.
Not applicable, reported as `skip`: gamma (inscribed/circumscribed radius, tet-only), thin-wall element count (the tet-only
face-edge measure; thin walls are resolved at the tet stage, >= 2 hexes through every wall seen), non-orthogonality/skewness
(CFD only). Extra number in report.json: `quality.corner_sj_min` (worst Jacobian at any of the 8 corners: 0.106 and 0.102 here).

## Numbers (CPU, 4 threads)

| Part | purpose | element | nodes | elements | DOFs | gates | build s |
|---|---|---|---|---|---|---|---|
| abc_00000003 | linear | tet10 | 120 387 | 70 215 | 361 161 | pass (thin_walls warn: 1.76 el/wall) | 33 |
| abc_00000003 | nonlinear | hex8 C3D8 | 38 417 | 29 340 | 115 251 | pass (SICN min 0.174) | 11 |
| abc_00000065 | linear | tet10 | 129 039 | 78 582 | 387 117 | pass | 44 |
| abc_00000065 | nonlinear | hex8 C3D8 | 56 656 | 44 776 | 169 968 | pass (SICN min 0.133) | 24 |

Sanity solve (feacore/skfem, pardiso, `sanity_solve.py`; 100 N compressive on `top`, bottom fixed, E = 1879.8 MPa, nu 0.35):

| Part | reaction N | top mean u_mag (m) | top mean u_axis (m) | peak von Mises (Pa) | strain energy J | solve s |
|---|---|---|---|---|---|
| abc_00000003 | 100.0000 | 1.842e-05 | -1.8229e-05 | 2.148e6 | 9.114e-4 | 36 |
| abc_00000065 | 100.0000 | 1.0690e-04 | -3.6990e-05 | 7.143e6 | 1.850e-3 | 37 |

Note: the cross-system 'top mean u' is the mean of u_mag (as ace_sizing/netgen/gold_nested report it). Part 65: 1.069e-4 m, vs 1.072e-4 (ace_sizing); an earlier version of this table gave the signed axis component (-3.70e-5, consistent with 2U/F), which is not the same quantity.

CalculiX smoke (`ccx_smoke.py`: linear-elastic static, bottom fixed, top moved -1e-5 m along the axis): ccx finishes without
errors on both hex meshes. Reaction for 1e-5 m: part 3 C3D8 80.5 N (C3D8I 79.7, C3D8R 78.9) vs skfem tet10 flat-platen 78.7 N;
part 65 C3D8 34.6 N vs skfem tet10 33.4 N. So C3D8 is 2-4 % stiffer than tet10 at this resolution.

## Known limits

- The hex mesh does not meet the 40k-node budget on part 65 (56.7k nodes): its 50+ small features set the size, base `h`
  barely changes the count. Part 3 meets it (38.4k).
- Minimum corner SJ of the accepted hex meshes is just over the 0.10 acceptance (0.106 and 0.102); SICN min 0.17 / 0.13.
  Hex quality is bounded by the tet mesh it came from; this is not a mapped/swept hex mesh. 2 hexes across a 2 mm wall.
- `mesh.inp` from gmsh contains the quad surface elements as `CPS4` blocks: ccx treats them as 2D elements and aborts
  (`*ERROR in gen3delem`). `ccx_smoke.py:strip_surface` removes those blocks and the `ELSET bottom/top`; do the same before
  `*SOLID SECTION, ELSET=domain` (the same holds for the tets' `CPS3`/`CPS6` blocks).
- The ccx backend of feacore (`_ccx_mesh.py`) handles simplices; a hex mesh is for hand-written ccx decks, not for `ccx_crush.py`
  without extending it. Curved (hex20) elements were not built.
- The linear tet counts are near the top of the 50k-400k DOF window (thin walls force the size); a larger `--h` plus `--thin-factor` lowers them (not tuned).
- Gate thresholds are tet-calibrated; the hex SICN/SJ numbers are not directly comparable to tet SICN.
