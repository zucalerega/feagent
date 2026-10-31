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
| dictionaries | written with `foamlib`, following the shipped `simpleFoam/pitzDaily` tutorial and the `etc/templates/inflowOutflow` template: SIMPLEC (`consistent yes`, U relaxation 0.9), GAMG for p and smoothSolver for U, `Gauss linear` gradients, `Gauss linear corrected` laplacian |
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
* `probe(field, points)`: linear reconstruction from the nearest cell centre (cell value plus Gauss
  gradient times the offset). This is second-order accurate and works for any cell shape. A point
  farther than 2 local cell sizes from every centre raises `CaseError`. p is in Pa.
* `max` / `min` / `max_abs`: over cell values (group `None` / `*` or a cell group) or over the face
  values of a boundary group.
* `integral` / `mean`: cell values times volumes, or face values times face areas. Volumes and
  areas include the 2D thickness.
* `flow_rate`: the sum of the face flux `phi` over the group, positive out of the domain.
* `force`: the fluid force on the body through the group (pressure + viscous), one component.
  This is OpenFOAM's `forces` formula: `rho (p - pRef) Sf + Sf . devRhoReff`, with
  `devRhoReff = -rho nuEff dev(twoSymm(grad U))`. The wall gradient takes its normal part from
  `snGrad`.
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
`turbulence_intensity`, `turbulence_length`.

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
