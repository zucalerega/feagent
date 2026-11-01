# OpenFOAM backend (`backend="openfoam"`, physics `flow`)

Steady incompressible flow for the same agent loop as the FEA problems: the agent produces a case
JSON plus a gmsh `.msh` with named groups, and `feacore.solver.solve(case, backend="openfoam")`
runs OpenFOAM's `simpleFoam` on it and returns a `Result` like the other backends.

Files: `feacore/solver/openfoam_backend.py` (setup, run, `OpenFOAMResult`),
`_foam_env.py` (finding OpenFOAM, running its applications), `_foam_mesh.py` (gmsh to `gmshToFoam`
input, 2D extrusion, patches), `_foam_io.py` (dictionaries via `foamlib`, readers, log parsing),
`_foam_fv.py` (finite-volume post-processing), `feacore/adapters/mesh_harness_cfd.py` (mesh harness
`bcs.cfd` to case). Tests: `tests/test_solver_openfoam.py` (about 15 s; skipped without OpenFOAM
or foamlib).

## What it does

| step | how |
|---|---|
| mesh | meshio reads the `.msh` (2.2 / 4.1, ASCII / binary; tri, quad, tet, hex, prism, pyramid; quadratic cells are linearised). 2D: one cell layer of `analysis.thickness` is extruded (z from 0 to the thickness; prisms / hexes) and written as gmsh 2.2 for `gmshToFoam` |
| patches | one patch per boundary group. Overlapping groups are split into "atoms" (`a__b`), and boundary faces in no group go to `unassigned`. 2D front and back form `frontAndBack` (`empty`). Patch types: `wall`, `patch`, `symmetry`, `empty` |
| dictionaries | written with `foamlib`, following the shipped `simpleFoam/pitzDaily` tutorial and the `etc/templates/inflowOutflow` template: SIMPLEC (`consistent yes`, U relaxation 0.9), GAMG for p and smoothSolver for U, `Gauss linear` gradients, `Gauss linear corrected` laplacian. In 3D (`skew_correction`): `pointCellsLeastSquares` gradients except `grad(p)`, and `flux(HbyA) skewCorrected linear` (see "3D meshes") |
| convection | `analysis.order` 1 gives `bounded Gauss upwind`; 2 gives `bounded Gauss linearUpwind grad(U)`. Turbulence (when on) uses `limitedLinear 1` or upwind |
| run | `simpleFoam` runs until `residualControl` (`residual_tol` on the initial residuals of p, U and turbulence) or `max_iterations` |
| results | the U, p, phi (and nut) fields of the last time are read with foamlib. Cell centres and volumes, gradients and forces are computed in numpy with OpenFOAM's own formulas (see "Version notes") |

BC mapping (pressure in the case is static pressure in Pa; OpenFOAM gets p / rho):

| case BC | U | p |
|---|---|---|
| `velocity_inlet`, vector (entries may be expressions) | `fixedValue` (expressions evaluated at face centres, `nonuniform List`) | `zeroGradient` |
| `velocity_inlet`, scalar speed (or expression) | `surfaceNormalFixedValue`, `refValue = -speed` (inward) | `zeroGradient` |
| `pressure_outlet` (value in Pa, or an expression) | `pressureInletOutletVelocity` | `fixedValue` value / rho |
| `wall` (optional moving-wall `value`) | `noSlip`, or `fixedValue` | `zeroGradient` |
| `slip` | `slip` | `zeroGradient` |
| `symmetry` | `symmetry` | `symmetry` |
| boundary group without a BC, `unassigned` | `noSlip` (with an info warning) | `zeroGradient` |

RANS (`analysis.options.turbulence = "kOmegaSST"` or `"kOmega"`): the inlet k and omega come from
`turbulence_intensity` (default 0.05) and `turbulence_length` (default 0.07 times the smallest bbox
extent). Walls use `kqRWallFunction` / `omegaWallFunction` / `nutkWallFunction`; outlets use
`inletOutlet`. This path is wired and sanity-checked: it runs, conserves mass and drags the walls.
It is not validated, and SIMPLE + SST may stall around 1e-5 instead of reaching a tight
`residual_tol`.

## Results and QoIs

* `nodal(field)`: cell centres (x, y in 2D) and cell values.
* `probe(field, points)`: continuous, cellPoint-style interpolation (`CellPointInterpolator` in
  `_foam_fv.py`).
  * Vertex values come from a weighted least-squares linear fit to the surrounding cell centres.
    Where the 1-ring of cells is too small (walls, corners) the fit uses the 2-ring. On Dirichlet
    patches (walls, velocity inlets for U; a fixed-pressure outlet for p) the vertex value is the
    mean of the adjacent face values.
  * Inside a cell the value is linear on its decomposition into tets (cell centre, face centre,
    face edge).
  * It is exact for linear fields and continuous across faces, so a probe on a face or vertex
    (Ghia points, axis points, wall points) does not depend on the cell numbering. A reversed
    numbering agrees to the iteration error, and points 2e-9 apart across a face agree to 1e-9.
  * A point outside every cell and farther than 2 local cell sizes from every centre raises
    `CaseError`. p is in Pa.
* `max` / `min` / `max_abs`: over cell values (group `None` / `*` or a cell group) or over the face
  values of a boundary group.
* `integral` / `mean`: cell values times volumes, or face values times face areas. Volumes and
  areas include the 2D thickness.
* `flow_rate`: the sum of the face flux `phi` over the group, positive out of the domain.
* `force`: the fluid force on the body through the group (pressure + viscous), one component.
  This is OpenFOAM's `forces` formula: `rho (p - pRef) Sf + Sf . devRhoReff`, with
  `devRhoReff = -rho nuEff dev(twoSymm(grad U))`. The wall gradient takes its normal part from
  `snGrad`, over the wall-normal distance of the cell centre `n . (Cf - C)` (OpenFOAM's
  `fvPatch::delta`, which the solver's own wall flux uses). Before 2026-10-31 feacore divided by
  `|Cf - C|`, which reads about 10 % low on tet walls and 0.2 % low on the DFG cylinder.
* `point_difference`: from `probe` (`Result.evaluate_qoi`), e.g. a pressure drop between two probes.
* Closed domains (no `pressure_outlet`) fix p at cell 0 while solving. The result is shifted to a
  zero volume mean, which is mesh-independent. Forces use the same shift.

`info` is a `FlowSolveInfo` (a `SolveInfo` subclass, so `asdict` / `summary()` include the extra
fields): `iterations`, `converged`, `residuals` (the last initial residual per equation, with
U = max over components), `residual_tol`, `openfoam` (version), `case_dir` (only if kept).
`n_cells` is the 2D cell count for extruded cases, and `measure` is the area (2D) or volume (3D).
`h_rep = (measure / n_cells)^(1/dim)` and `n_dofs = n_cells * (dim + 1)`.

Options (`analysis.options`, or keywords to `solve`): `turbulence` ("laminar"), `max_iterations`
(5000), `residual_tol` (1e-8), `relaxation_U` (0.9), `relaxation_p` (none: SIMPLEC),
`n_non_orth_correctors` (0), `timeout` (seconds; the solver writes and stops at the limit),
`case_dir` (run there and keep it; it must be new or empty, never in the repo), `keep_case`,
`turbulence_intensity`, `turbulence_length`, `skew_correction` ("auto": on in 3D, off in 2D; or
true / false).

## 3D meshes: skew-consistent schemes (`skew_correction`)

With OpenFOAM's plain schemes, 3D flow on unstructured tets converges to a wrong solution. On the
`pipe_poiseuille_3d` tets an agent makes (h = 1 mm, 22k cells), the axial pressure gradient was
+15.3 % and the wall shear +17 %. A flat-walled 3D channel on tets gave +12.2, +13.7, +14.2 and
+14.1 % at 9k, 24k, 70k and 183k cells, so the error does not shrink with h. Structured hexes
converge at second order, and 2D triangles are accurate.

**Cause.** A tet with a face on a wall has its centroid at 1/4 of its height above the wall, but
its faces that contain a wall edge have their centroids at 1/3. Linear interpolation between cell
centres therefore misses those face values in the same direction all along the wall, so the
interpolation error is systematic there rather than random. Two terms inherit it:

* `Gauss linear` gradients are built from those face values, so grad(U) is wrong by O(1) in the
  wall cells. grad(U) feeds the non-orthogonal correction of `laplacian(nuEff,U)` and
  `div(nuEff*dev2(T(grad(U))))`, so the viscous flux near the wall is off by an amount that does
  not shrink with h.
* The momentum-interpolation flux `flux(HbyA)` is interpolated linearly to the same faces, so the
  near-wall face fluxes come out low.

The discrete solution compensates with a flattened profile: near-wall cells about 13 % fast, the
axis about 1.4 % slow. The higher wall shear then needs a higher pressure gradient.

**Evidence** (pipe and channel above, OpenFOAM v1912):

* It is not the probes or the kinematic-pressure conversion. A linear fit of the cell pressures
  gives +15.4 % at every radius, the inlet patch pressure is +15 %, and the solver's own discrete
  x-momentum balance closes (inlet pressure force = wall flux with the wall-normal distance).
* It is not convection: Stokes flow (Re 0.1) gives +15.2 %, and upwind +19.6 %.
* It is not the faceted wall: the flat-walled channel on tets gives +13.7 %, against −1.5 % (second
  order) on structured hexes at the same h.
* It is not mesh quality: `checkMesh` passes (non-orthogonality max 62°, mean 21°).
* It is not specific to tets: the same pipe split into hexes (gmsh `SubdivisionAlgorithm` 2) gave
  +20 %.
* 2D triangles are fine (−0.2 % and +0.6 % at H/10 and H/20). gmsh's near-wall triangles alternate
  up and down, so the interpolation offsets cancel.

Scheme scan on the channel (h = 1 mm, Stokes), error in -dp/dx:

| gradients | `flux(HbyA)` | error |
|---|---|---|
| `Gauss linear` (default) | `linear` | +13.2 % |
| `leastSquares` | `linear` | +14.8 % |
| `Gauss skewCorrected linear` | `linear` | +6.8 % |
| `Gauss pointLinear` | `linear` | +3.8 % |
| `pointCellsLeastSquares` | `linear` | −1.7 % (−1.3 to −1.7 % for any h: still inconsistent) |
| `Gauss linear` | `skewCorrected linear` | +6.6 % |
| `pointCellsLeastSquares`, `grad(p)` `Gauss linear` | `skewCorrected linear` | +0.4 % (h = 1.4, 1.0, 0.7 mm: +0.16, +0.38, +0.37 %) |

`skew_correction` (the default in 3D) uses the last row. `pointCellsLeastSquares` is exact for
linear fields on any mesh. `grad(p)` stays `Gauss linear` because it is conservative (the cell
pressure forces sum to the boundary pressure force). A least-squares `grad(p)` takes the
zero-gradient wall pressures at face centres that are shifted along the flow. On the split-tet
hexes that alone left +13.6 % with an exact velocity field, against +0.7 % with `Gauss linear`.
Changing the laplacian (`limited`, `uncorrected`, `orthogonal`) or adding non-orthogonal correctors
does not help.

Results with `skew_correction` (error against the exact values):

| case | mesh | cells | dp_AB before → after | wall force after | other |
|---|---|---|---|---|---|
| `pipe_poiseuille_3d` | tets h = 1 mm | 22k | +15.3 → +1.3 % | +1.6 % | flow rate +0.6 % |
| `pipe_poiseuille_3d` | tets h = 0.7 mm | 59k | → +1.2 % | +1.5 % | |
| `pipe_poiseuille_3d` | split-tet hexes h = 1.4 mm | 36k | +20.1 → +0.7 % | +1.1 % | |
| `pipe_poiseuille_3d` | reference hexes h0, h0/2 | 640, 5k | +7.0, +2.2 → +6.4, +1.7 % | +3.1, +0.9 % | |
| `square_duct_3d` | tets h = 1.4 / 1.0 / 0.8 mm | 18k / 48k / 92k | +11.7 → −0.8 / +0.0 / +0.1 % | | |
| `square_duct_3d` | reference hexes h0 | 5k | −5.55 → −5.55 % | | identical |

What remains on the pipe is mostly the faceted wall: a 31-gon at h = 1 mm is 0.6 % short in area,
which is worth about +1.3 % in pressure gradient at a given flow rate. The regression test
(`test_pipe_3d_tets_pressure_drop`) compares with the exact values for the meshed section.

Costs and limits:

* The point stencils take memory. At 70k tets simpleFoam peaks at 254 MB instead of 100 MB; at 183k
  tets the peak is 574 MB. On a Mac whose Docker VM has 2 GB shared with other containers, that
  run was OOM-killed.
* Point probes at a smooth velocity maximum read low on coarse tets. This is the vertex fit of
  `probe` flattening the peak, not the solution. On the duct at h = 1 mm, a quadratic fit of the cell
  values gives u_axis −0.74 %, while the probe reads −2.6 to −4.8 % depending on x. It converges
  with h (pipe u_centreline −2.4 % at 1 mm, −0.6 % at 0.7 mm). Pressure drops, forces and flow rates
  are not affected.
* 2D keeps the plain schemes; `skew_correction: true` also works in 2D.

## residual_tol and the GCI

The GCI measures discretisation error. Iteration error must stay well below it, hence the tight
default (1e-8 on OpenFOAM's normalised initial residuals). Both outcomes are reported:

* If `residual_tol` is not reached within `max_iterations` (or `timeout`), the result carries a
  `NOT CONVERGED: ...` warning and `info.converged = False`. Don't use such a result in a GCI study.
* A velocity component that is nearly zero (Uy of Poiseuille flow, the cross-stream components in
  a straight pipe) has its residual normalised by its own tiny scale, so it stalls near 1e-7 at
  round-off. The backend watches the log while the solver runs. If only such components stay above
  `residual_tol` for 100 iterations while everything else is converged, it stops the solver with
  `stopAtWriteNowSignal`, which writes the current state cleanly. The run counts as converged if
  those residuals, rescaled by max|U|, are below `residual_tol` (an info warning says so).
  Workarounds such as capping `max_iterations` for this case are no longer needed.

Measured (`tests/test_solver_openfoam.py`; 2D channel, H = 0.1, L = 0.6, Re = 10, parabolic
inlet, quads):

| cells across H | -dp/dx error | wall force error | mass balance | iterations | wall time |
|---|---|---|---|---|---|
| 8 | -2.27e-2 | -2.48e-2 | 8e-11 | 152 | 0.3 s |
| 16 | -5.81e-3 | -6.59e-3 | 1e-10 | 144 | 0.5 s |
| 32 | -1.46e-3 | -1.72e-3 | 5e-13 | 253 | 2.1 s |

The error ratio is 3.9, i.e. second order. The n = 8 value agrees exactly with the discrete
analysis: the 2-point wall gradient plus midpoint sampling of the inlet parabola give
dp/dx_h = dp/dx (1 + h^2/2H^2) / (1 + 2h^2/H^2). Unstructured triangles (h = H/8) give -2.6e-3.

## Accuracy notes

* Everything is second order (the FV discretisation, the probes, the integrals). There are two
  exceptions:
  * p on a `zeroGradient` patch (velocity inlets, walls) is the adjacent cell value, so a mean
    pressure over an inlet is first-order accurate (error ~ dp/dn h/2). For pressure drops prefer
    `point_difference` between interior points, or check the observed order.
  * Wall shear uses OpenFOAM's 2-point `snGrad`. It is consistent with the discrete momentum
    balance and second order, but with a large constant on coarse wall-normal spacing.
* Inlet flow rates come from face-centre samples of the profile: a parabola gives
  Q (1 + h^2 / 2H^2), not Q. Faceted curved inlets lose area at O(h^2).
* Unstructured triangles (prisms) converge poorly with the default `linear` interpolation, because
  OpenFOAM's skewness error does not shrink with h. Poiseuille dp/dx errors at H/8, H/16, H/32:
  * default: -6.9e-4, 2.5e-3, 3.8e-3; cell-value u rms 3.5e-3, 1.9e-3, 1.9e-3; p noise about 2e-2;
  * `skewCorrected linear` (tried, not the default): u rms 3.1e-3, 9.3e-4, 4.9e-4; p noise halving
    with h; dp/dx -1.5e-3, -3.4e-4, 8.3e-4.

  Prefer quad / hex meshes for GCI studies. Quad results converge cleanly at second order.
* Linear interpolation clips smooth extrema at O(h^2). A probe on the Poiseuille centreline
  (a grid line) reads about 1.5 (h/H)^2 low. Probe errors on quads: 3.4e-2, 8.6e-3, 2.2e-3 at
  H/8, H/16, H/32 (second order).
* A flow case may name its fluid region (`regions=[Region("fluid", "water")]`). Only one fluid
  material is allowed.

## Version notes

* Supported: the ESI / OpenCFD line, v1912 to current (`simpleFoam`, `turbulenceProperties`,
  `-opt-switch`). The openfoam.org line (versions 11+, `foamRun`) is detected and refused.
* Discovery order: `FEACORE_OPENFOAM_BASHRC` (path to `etc/bashrc`); then an already sourced
  environment; then `/usr/lib/openfoam/openfoam*/etc/bashrc` (openfoam.com packages, newest
  first), `/opt/...`, `~/OpenFOAM/...`; then `/usr/share/openfoam/etc/bashrc` (Ubuntu's package).
  The environment is captured once and passed to each subprocess; `os.environ` is never changed.
* Function objects are not used. In the Ubuntu 24.04 `openfoam` package (v1912) every function
  object aborts (`error in IOstream "sha1"`, including `postProcess -func`). Computing centres,
  gradients and forces in numpy with OpenFOAM's formulas also keeps results identical across
  versions, whose `forces` output formats differ.
* Tested here: OpenFOAM v1912 (Ubuntu 24.04 `openfoam` 1912.200626) with foamlib 1.10.

## Install (Brev / Ubuntu)

```bash
# ESI packages (recommended; x86_64): adds the openfoam.com apt repo
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install -y openfoam2406-default      # or a newer openfoam24xx/25xx-default
# or Ubuntu's own package (v1912 on 24.04; also built for arm64 / DGX Spark):
sudo apt-get install -y openfoam
pip install foamlib                                 # Python file I/O used by the backend
python -c "from feacore.solver._foam_env import require_openfoam; print(require_openfoam().describe())"
```

No sourcing is needed: the backend finds `/usr/lib/openfoam/openfoam*/etc/bashrc` or
`/usr/share/openfoam/etc/bashrc` itself. For a custom build, set
`FEACORE_OPENFOAM_BASHRC=/path/to/OpenFOAM-vXXXX/etc/bashrc`. Runs are serial (one core per case).
`OMP_NUM_THREADS=1` is set for the OpenFOAM processes.

## Mesh harness (`analysis: cfd`)

`python -m feacore.adapters.mesh_harness_cfd <out_dir | mesh.yaml> [--rho R --mu M --order 2]`.
The mapping:

* `bcs.cfd.inlet` groups become `velocity_inlet` with `velocity` as the inward normal speed;
* `outlet` groups become `pressure_outlet` 0;
* every other surface group (the `default_group`, `walls` by default) becomes a no-slip `wall`.

Harness units are mm, so the case uses mm, s, t, N, MPa. The default fluid is water: rho = 1e-9
t/mm^3, mu = 1e-9 MPa s, which is 1 mm^2/s. Velocity is in mm/s, pressure in MPa, force in N.

Default QoIs:

* `pressure_drop` (the mean p over the inlet; the outlets are at 0; with several inlets,
  `mean_p_<g>` for each);
* `flow_rate_<g>` for every inlet and outlet;
* `force_<wall>_{x,y,z}`;
* `max_u_mag`.

Tested on the harness `tee` part (tets, Re = 200): mass balance 1e-9, converged in 200 iterations,
about 2 s.

## Limitations

* Steady, incompressible, single fluid, serial. No unsteady (pimpleFoam) and no parallel runs.
* BCs and the `flow_rate` / `force` QoIs need boundary groups. Facet groups inside the domain
  (interior surfaces) are detected but not supported yet.
* `write_vtu` writes cell centres with values as a point cloud. Use `case_dir` and ParaView for
  the full mesh.
* RANS: only `kOmegaSST` / `kOmega` with wall functions, and not validated (see above).
* A closed domain's pressure is defined up to a constant; it is reported with zero mean.
