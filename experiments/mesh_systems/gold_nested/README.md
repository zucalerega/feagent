# gold_nested

The gold-benchmark generator's meshing (`feacore/goldgen/mesh.py`, branch `gold-benchmarks`) re-implemented in `build.py`
(feacore.goldgen is not on this branch). One base tet mesh (gmsh Delaunay, global size F x bbox diagonal, curvature
sizing, 12 elements per 2 pi) is split uniformly (`gmsh.model.mesh.refine()`: 8x cells per level, new boundary nodes
snapped to the CAD) into nested levels L0, L1, ..., each made quadratic (tet10, boundary midside nodes on the CAD).
Faces are picked by geometry (planar, normal along the axis, at the extreme position) and carried through the splits as
physical groups `bottom`, `top`, `domain`. Quality gates: `feagent.quality.assess_mesh`; picture: `feagent render --view quality`.

    uv run --extra pardiso python experiments/mesh_systems/gold_nested/build.py STEP AXIS OUT --purpose linear|nonlinear
    python experiments/mesh_systems/gold_nested/sanity.py OUT/mesh.msh AXIS        # skfem sanity solve
    python experiments/mesh_systems/gold_nested/sanity.py OUT/mesh.inp AXIS --ccx  # CalculiX smoke check

Every level is written to `OUT/levels/L<k>/{mesh.msh,mesh.inp,report.json}` (L2+ only if it fits `--max-export-nodes`);
the selected level (L1) is copied to `OUT/`. The base size F is fitted by bisection so that L1 meets the budget:
linear = L1 <= 400k DOFs (tet10, 133k nodes), nonlinear = L1 <= 40k nodes. Knobs: `--h0-frac`, `--curv`, `--levels`,
`--select`, `--fit-level`, `--no-ho`, `--max-export-nodes`. Deterministic (1 thread, Delaunay).

Pipeline per level: base -> repair to corner quality 0.3 (best effort) -> split k times (tet optimizer only when a split
leaves quality < 0.05, which breaks strict nesting for that level) -> setOrder(2) -> linear: gmsh high-order optimizer
(`curved-opt`); nonlinear: none; cells with scaled Jacobian < 0.05 are straightened (they lose boundary curvature, interior
edges are always straight; the optimizer is skipped for nonlinear so interior edges stay straight and because it aborted
gmsh on abc_00000065). `mesh.inp` has the surface triangles removed (CalculiX rejects gmsh's CPS6): C3D10 elements,
ELSET `domain`, NSET `bottom`, `top`, `domain`.

## Numbers (selected level L1; sanity: fixed bottom, 100 N compressive on top, Alpha Fast E = 1879.8 MPa, nu = 0.35, skfem P2, PARDISO)

| | abc_00000003 linear | abc_00000003 nonlinear | abc_00000065 linear | abc_00000065 nonlinear |
|---|---|---|---|---|
| nodes / tet10 / DOFs | 131505 / 79672 / 394515 | 39841 / 23774 / 119523 | 130096 / 80736 / 390288 | 39234 / 23560 / 117702 |
| base size F | 0.0543 | 0.3721 | 0.0773 | 0.324 |
| gates | sicn fail (min 0.011), gamma warn, rest pass | sicn fail (0.002), gamma warn, area fail (13.8 % on the 1 mm-wide bottom ring) | sicn fail (0.057), gamma warn | sicn fail (0.016), gamma warn |
| min scaled Jacobian | 0.10 | 0.16 | 0.30 | 0.08 |
| build s (fit + all levels) | 62 (51 fit) | 27 (23 fit) | 32 (26 fit) | 15.5 (14 fit) |
| reaction | 100.000 N | CalculiX smoke ok | 100.000 N | CalculiX smoke ok |
| mean abs. u top, L0 -> L1 [m] | 1.7956e-5 -> 1.8296e-5 | | 1.0543e-4 -> 1.0685e-4 | |
| peak von Mises L0 -> L1 [Pa] | 2.23e6 -> 3.38e6 | | 2.99e6 -> 4.51e6 | |
| strain energy L0 -> L1 [J] | 8.885e-4 -> 9.053e-4 | | 1.7888e-3 -> 1.8384e-3 | |

Sanity solve of the selected level: 32 s (abc3), 25 s (abc65) on 4 threads.

## Known limits

- Three nested levels within 400k DOFs are not possible: L1 is ~7x L0 and L2 would be ~2.8M DOFs; a coarse base (F = 0.5)
  fails the Jacobian check at L2. The nested convergence check is therefore L0 -> L1 only; mean top displacement moves
  1.9 % (abc3) and 1.3 % (abc65) between them, peak von Mises (a point value) moves 50 %, so judge it against L2 offline.
  L2 can be exported (`--levels 3 --max-export-nodes 1000000`, ~950k nodes for abc3) but not solved here.
- The SICN gate fails for all four meshes: 10-30 curved or sliver cells in 10^4-10^5 (SICN 0.002-0.06, p01 0.25-0.4, median 0.65-0.75).
  Scaled Jacobian is above 0.05 everywhere, so the solver accepts them; a crush simulation should watch them.
- abc_00000003 has a 1 mm-wide bottom ring at the cone: the nonlinear mesh straightens ~50 cells there and misses the CAD
  area by 14 %.
- Refinement of a base meshed with curvature sizing only: no local refinement at thin walls (1-2 elements through 2 mm walls).
