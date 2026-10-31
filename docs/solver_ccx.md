# CalculiX backend (`backend="ccx"`)

An independent second solver for cross-checking the fixed scikit-fem solver.
`feacore.solver.solve(case, backend="ccx")` writes a CalculiX deck, runs `ccx` (2.21) in a
scratch directory, parses the ASCII `.frd` / `.dat` and returns a `CcxResult`
(`probe` / `nodal` / `evaluate` / `summary`). The FE solve, element integration, stress
recovery, reactions and energy come from CalculiX. Python handles the parts CalculiX has no
input for: mesh reading and conversion, consistent nodal loads for expression and traction
loads, interpolation at probe points, and QoI integrals.

Files: `feacore/solver/ccx_backend.py` (deck + result), `_ccx_mesh.py` (gmsh → CalculiX
model), `_ccx_fem.py` (shape functions, quadrature, point location), `_ccx_io.py` (run,
`.frd` / `.dat` parsers). Tests: `tests/test_solver_ccx.py` (38 tests, about 3 s).

Options: `workdir` (keep the run there), `keep=True`, `ccx` (executable; also `$FEACORE_CCX`),
`threads` (default `$OMP_NUM_THREADS` or 1), `timeout`. Scratch runs go under
`$FEACORE_CCX_TMP` (default `/tmp`) `/ccx/`. A run directory is deleted after success and kept
on failure; the `CcxError` message names it.

## Mapping

| case | CalculiX |
|---|---|
| plane_stress / plane_strain / axisymmetric / 3d, order 1 / 2 | CPS3/CPS6 (thin internal section, see below), CPE3/CPE6, CAX6 (CAX3 refused), C3D4/C3D10; thickness on `*SOLID SECTION` |
| heat 2d / axisymmetric / 3d | same element types, `*HEAT TRANSFER, STEADY STATE`, `*CONDUCTIVITY` |
| materials + regions | one `*MATERIAL` + `*SOLID SECTION` per material; each cell takes the first matching region (as `Case.material_for`) |
| fixed / displacement / symmetry / temperature | `*BOUNDARY` per node and dof (expressions evaluated at nodes). Later BCs override earlier ones on the same node/dof. `point` = nearest node. Symmetry: the normal is the axis along which the group's coordinates do not vary (otherwise `CaseError`) |
| pressure (constant) | `*DLOAD elem, Pk` on the element faces of the group |
| pressure (expression; also any CAX3 pressure), traction, force | consistent nodal loads ∫N·t dA (exact quadrature on the isoparametric faces, ×thickness or ×2πr) → `*CLOAD` |
| body_force (constant) | `*DLOAD elset, BX/BY/BZ` (force per volume) |
| body_force (expression, or axisymmetric) | consistent nodal loads → `*CLOAD` |
| temperature load | `*INITIAL CONDITIONS, TYPE=TEMPERATURE` = T_ref, `*TEMPERATURE` nodal values, `*EXPANSION, ZERO=T_ref`. Nodes outside the loaded groups stay at T_ref. Nodes shared by two loaded groups take the later load |
| flux (constant) / convection / source (constant) | `*DFLUX Sk` / `*FILM Fk` / `*DFLUX BF`; expression fluxes and sources, and CAX3 fluxes, → `*CFLUX` |

**Element order.** `analysis.order` sets the element order. A mesh of the other order is
converted, with a warning in `SolveInfo.warnings`. On a quadratic mesh, order 1 drops the
midside nodes (straight-sided linear cells). On a linear mesh, order 2 inserts midside nodes at
the edge midpoints. The skfem backend does the same. Mixed linear/quadratic cells, quads and
hexes raise `MeshError`.

**Node ordering.** meshio returns VTK order. tri6 matches gmsh and CalculiX (corners, then the
(1,2), (2,3), (3,1) midsides). For tet10, gmsh stores edge (3,4) before edge (2,4); meshio swaps
them, which gives CalculiX C3D10 order (midsides 12, 23, 31, 14, 24, 34). `FEModel.check_midside`
rejects a mesh whose midside nodes are not near their edge midpoints. Cells with negative
orientation are flipped (CalculiX rejects clockwise 2D cells). Domain cells that msh2 lists once
per physical group are deduplicated.

**Faces.** Triangle: face 1 = 1-2, face 2 = 2-3, face 3 = 3-1. Tetrahedron: face 1 = 1-2-3,
face 2 = 1-4-2, face 3 = 2-4-3, face 4 = 3-4-1. Facets are matched to element faces by their
sorted corner nodes. Verified on single-element tests (every face of CPS3/CPS6/CAX3/CAX6) and
by exact pressure resultants on curved 2D and 3D faces.

## Results

* `.frd`: U (DISP), RF (FORC), S (STRESS), NT, HFL (FLUX), RFL. Values have **6 significant
  digits** (E12.5), which limits nodal fields to about 1e-6 relative.
* `.dat` totals (`*NODE PRINT, TOTALS=ONLY` / `*EL PRINT, TOTALS=ONLY`, 7 digits, summed in
  double precision): RF / RFL over every Dirichlet group and every reaction-QoI group, and
  ELSE (internal energy).
* **Stress / heat flux recovery (CalculiX):** stresses are computed at the integration points,
  extrapolated to the nodes of each element, then averaged without weights over all elements
  that share a node, across material boundaries too. Nodal values from quadratic elements carry
  a ~1e-5 relative artifact: in a constant-stress patch the Gauss-point values are exact, but the
  nodal values are 0.99999·σ, apparently from ccx's truncated extrapolation coefficients. Linear
  elements are exact. Point stresses interpolate the nodal field with the element shape
  functions; von Mises, Tresca and principal stresses are computed from the interpolated
  components. The skfem backend uses an L2 projection instead, so point stresses at boundaries
  differ by about 0.2-0.3% at h = 0.1 (see the cross-check below).
* Stress components: plane stress szz = 0; plane strain szz = SZZ; axisymmetric sxx = s_rr,
  syy = s_zz (axial), sxy = s_rz, stt = SZZ (hoop), and szz = hoop, as in the skfem backend.
  stt = 0 outside axisymmetric.
* **Reaction** = Σ over the group's nodes of (RF − applied nodal load). CalculiX's RF is the
  internal nodal force K·u, which at a free node equals the applied load, so loads acting on
  constrained nodes (pressure next to a support) are subtracted. This is the skfem definition
  (K·u − f). Heat: net heat flow into the body through the group (negative where heat leaves).
* `strain_energy` = ELSE (0.5 u·K·u). With thermal loads ccx's ELSE counts only the mechanical
  strain, so 0.5 u·K·u is integrated from the ccx displacements instead. That matches ELSE to
  1e-7 in plane strain and 3D.
* `integral` / `mean` integrate over the physical body (×thickness, ×2πr), on the isoparametric
  cells or faces, like the skfem backend. `max` / `min` / `max_abs` use nodal values, midside
  nodes included.

## Axisymmetric (CAX) conventions, measured

CalculiX expands CAX elements into a 3D wedge from −1° to +1°. Measured against closed form
(`test_axisymmetric_units`, Lamé, heat):

* `*CLOAD` / `*CFLUX` values are **full-360° totals** (no conversion). With per-radian input the
  displacements came out 2π too small.
* RF, RFL and ELSE are reported **for the 2° wedge**: full 360° = 180 × ccx value
  (`AXI_OUTPUT_FACTOR`). Checked to 1e-6 with `*DLOAD` and with `*CLOAD` loading.
* Radial displacement output is the x-component of expanded nodes. Nodes that exist only on the
  ±1° faces (CAX6 midside nodes, all CAX3 nodes) report u_r·cos 1°. CAX6 corner nodes report
  u_r·(4 + 2 cos 1°)/6. Measured to 2e-6 on exact linear fields; the backend divides these out.
  Without the correction, Lamé u_r stalls at −1.5e-4 under refinement; with it, u_r converges
  (2.8e-5 → 7.2e-6, h = 0.2 → 0.1).
* **CAX3 caveats:** ccx's `*DLOAD` on CAX3 faces has the right total but splits it wrongly over
  the nodes (up to 5% displacement error next to the loaded edge in a uniform-stress patch), so
  CAX3 pressure and flux go through consistent nodal loads. `*FILM` (convection) on CAX3 still
  uses ccx's face integration and is untested. CAX3 is also about 2e-4 stiffer in the axial
  direction. CAX stresses are not corrected for the ±1° projection; that artifact is below
  sin²(1°)·|σθθ − σrr| ≈ 3e-4·|σθθ − σrr|.

## Refused configurations (raise `NotImplementedError`, so verification skips them)

Found by the independent MMS / patch tests (`tests/test_code_verification.py`). Refused because
CalculiX cannot do them exactly:

* **Axisymmetric order 1 (CAX3)**, elasticity and heat. The 2-degree wedge expansion is about
  2e-4 too stiff, and its face loads and films are split inconsistently over the nodes (heat with
  convection: 1e-2). Use `order = 2`.
* **C3D10 or CAX6 cells with curved interior edges** (midside offset > 1e-3 × edge length,
  `CURVED_INTERIOR_TOL`). CalculiX's quadrature (C3D10: 4 points) is not exact on curved
  isoparametric cells, and the patch test fails: 6e-5 for C3D10, up to 4e-3 for CAX6. CPS6 / CPE6
  pass. gmsh keeps interior edges straight and curves only boundary edges, by O(h²/R); those
  meshes are accepted with a warning, and the remaining difference from skfem is at
  discretization level and vanishes under refinement (quarter cylinder, h/R = 0.3: u_r 8e-5).

## Fixes after code verification

* **Plane stress** is no longer the thick slab. CPS elements are 3D slabs with free faces, so a
  thick slab carries transverse shear between cells and converges to a 3D plate, not plane
  stress. The MMS order was about 0 with body forces, and the result was 1-3% off skfem. ccx now
  runs with an internal section thickness `t' = 1e-3 × shortest cell edge`
  (`analysis.options["ccx_plane_stress_ratio"]`). Loads are scaled by t'/t, and RF / ELSE are
  scaled back by t/t'. Displacements and stresses of plane stress don't depend on the thickness.
  The MMS convergence studies (E-ps-square-dirichlet / -mixed, E-ps-annulus-lame, P1 and P2) now
  reach the theoretical orders. A residual difference from skfem on a given mesh remains,
  2e-4 to 1e-3 at P1 on coarse meshes: in the thin-slab limit ε_zz is continuous between cells,
  whereas discrete plane stress makes it cell-wise. It converges at the optimal rate.
* **probe in curved P2 cells**: a point inside a cell's straight-sided triangle can lie in a
  neighbour's curved cell. The map is now inverted (damped Newton) in up to 9 candidate cells,
  and a converged inverse inside the reference cell is kept. Error went from 4e-4 to 1e-7.
* **Remaining CAX6 limit**: on straight-sided cells with radial surface loads (`*CLOAD` radial
  components) the axisymmetric patch test shows u 4e-5 and σ 9e-5 errors; the strain energy is
  exact. Thermal and axial-load patches are exact, and Lamé with `*DLOAD` pressure converges.
  Treat axisymmetric ccx results as accurate to about 1e-4.

## Verification (tests/test_solver_ccx.py)

| test | result |
|---|---|
| uniaxial tet bar (C3D4, C3D10): u = PL/AE, R = −P, W | ≤ 1e-6 (σ: 1e-5 from the extrapolation artifact) |
| plane stress / plane strain bars: pressure, traction, force; msh 4.1 and 2.2; order conversion both ways | ≤ 1e-6 |
| Lamé, plane-strain quarter model with symmetry (P2, h = 0.1) | u 2e-5, σθθ(a) 6e-4, R = −p·a exact to 1e-8, W 4e-6 |
| Lamé, axisymmetric (P2, h = 0.1) | u 7e-6, σθθ(a) 9e-4, Rz 4e-7, W 3e-7 |
| CAX units: solid cylinder in tension (DLOAD, force, traction) | uz, R, W ≤ 1e-6; u_r 3e-6 (5e-5 before the ±1° correction) |
| pressure on a curved 3D face (quarter cylinder) | Σ nodal loads exact (2e-15); reactions 3e-8 (C3D4), 1.5e-5 (C3D10: ccx's own face quadrature is not exact on curved tri6 faces) |
| free thermal expansion (3D, plane stress, axisymmetric) | σ = 0 to round-off; plane strain σzz = −EαΔT |
| body force (`*DLOAD BX` and expression), two materials in series | ≤ 1e-6 |
| heat: 1D slab (Dirichlet, flux, convection, source), axisymmetric radial conduction | ≤ 1e-5 |

**Cross-check against skfem** (same case and P2 mesh; `pytest -s -k cross_check`):
displacement, reaction, energy, temperature and integral QoIs agree to 1e-8 to 4e-6 in 2D and
axisymmetric. In 3D: Ry 1.5e-5, a curved-surface point u_r 8e-5. Point stresses on the boundary
differ by 2-3e-3 and max von Mises by 7e-5 (stress recovery).
