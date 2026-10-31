# Fixed solver: scikit-fem backend (`backend="skfem"`)

The fixed solver that runs every agent's setup. Input is a case JSON (`feacore/case.py`) plus a gmsh
`.msh` file with named physical groups. Output is a `Result` (`feacore/solver/base.py`).

```python
from feacore.solver import solve
res = solve("case.json", backend="skfem")   # or a Case / dict
res.evaluate_all()                          # {qoi name: value}
res.probe("von_mises", [[0.1, 0.2]])        # recovered field at points
res.nodal("ux")                             # (coords, values) at the solution nodes
res.write_vtu("out.vtu")                    # every field, for ParaView
res.info                                    # SolveInfo (sizes, h_rep, timing, warnings)
```

CLI: `python -m feacore.cli case.json [--backend skfem] [--out result.json] [--vtu out.vtu]
[--linear-solver auto|scipy|pardiso|cudss]`. It writes `{"info", "qois", "meta"}` as standard JSON.
Exit code 0 means success. Exit code 2 means an invalid case or mesh: unknown group, wrong group
kind, insufficient constraints, a non-axis-aligned symmetry group, and so on. Exit code 1 means any
other failure. On failure, `--out` holds `{"error": message}`. Error messages name the problem and,
for groups, list the available names.

Files: `feacore/solver/skfem_backend.py` (driver, physics, `SkfemResult`), `_skfem_mesh.py`
(reads the .msh and maps the groups), `_skfem_assembly.py` (vectorised stiffness and mass),
`_skfem_locate.py` (point location for probes) and `linsolve.py` (the pluggable linear solver).

## Supported features

| | |
|---|---|
| Physics / formulation | elasticity: `plane_stress`, `plane_strain`, `axisymmetric`, `3d`; heat (also Poisson): `2d`, `axisymmetric`, `3d` |
| Interpolation | Lagrange P1 / P2 (`analysis.order`), independent of the mesh order |
| Mesh | gmsh 4.1 or 2.2, ASCII or binary; tri3 / tri6 / tet4 / tet10. 2D meshes in the xy-plane (z dropped) |
| Geometry | quadratic mesh + order 2: isoparametric (curved cells). Order 1 always runs on the straight-sided corner mesh, so P1 on a tri6/tet10 file ignores the midside nodes (`info:` warning). Order 2 on a linear mesh gives straight-sided P2 |
| Groups | by dimension: cells (mesh dim), facets (dim-1), edges (curves in 3D), vertices (points). An unknown name raises an error that lists every group with its kind and size |
| Elasticity BCs | `fixed`, `displacement` (any subset of components; scalar, per-component list or expressions), `symmetry`, `pressure`, `traction`, `force`. Dirichlet types also accept point groups, curve groups (3D), or `point` = nearest solution node |
| Heat BCs | `temperature` (group or point; expression allowed), `flux` (into the body; expression allowed), `convection` (`h`, `T_inf`) |
| Loads | elasticity: `body_force` (vector, expressions), `temperature` (thermal strain `alpha (T - T_ref)`, expression); heat: `source` (expression). Applied on a cell group or `"*"` |
| Materials | multi-material `regions` (first matching region wins, as in `Case.material_for`; uncovered cells are an error) |
| QoIs | `point`, `max`, `min`, `max_abs`, `reaction`, `integral`, `mean`, `strain_energy` |
| Fields | elasticity: `ux uy uz u_mag sxx syy szz sxy syz sxz stt von_mises tresca s1 s2 s3`; heat: `T qx qy qz q_mag` |

## Conventions (what the numbers mean)

- **Plane formulations**: stiffness and every load are multiplied by `analysis.thickness`, so forces,
  reactions, energies and integrals are per given thickness. Plane stress uses
  `lambda* = 2 lambda mu / (lambda + 2 mu)`. Plane strain reports `szz = nu (sxx + syy) - E alpha dT`.
  Plane stress has `szz = 0`.
- **Axisymmetric**: x = r >= 0, y = z, and all integrals carry `2 pi r`. Forces, reactions, energies
  and integrals are full 360-degree totals. The hoop strain is `u_r / r`. Field names: `sxx = s_rr`,
  `syy = s_zz`, `sxy = s_rz`, `stt` = hoop stress. `szz` returns `stt`, the out-of-plane component,
  by analogy with plane strain. `u_r = 0` is imposed automatically on nodes at r = 0 (an `info:`
  warning is recorded; `analysis.options["axis_constraint"] = false` disables it). `stt` is 0 in
  the other formulations.
- **pressure**: traction `-p n`, where n is the outward normal of the cell adjacent to the boundary
  facet, so positive pressure pushes into the body. Pressure on interior facets is an error.
- **force**: `traction = F / measure(group)`, where the measure includes the thickness (plane) or
  `2 pi r` (axisymmetric).
- **symmetry**: zero normal displacement. The group must be a planar facet group normal to x, y or z;
  anything else is an error.
- **reaction**: `sum over the group's DOFs of (K u - f)` for the component. This is the force the
  support exerts on the body, in the same units as `force` (thickness or 360° included). If the group
  has no Dirichlet constraint in that component, the value is ~0 and a warning is added. Heat
  (`component` 0): the heat flow into the body through the Dirichlet group.
- **strain_energy**: `0.5 u.K.u` with the full K (thickness or `2 pi r` included). With thermal
  loads this is the energy of the total strain, not the elastic strain, as the case schema states.
- **max / min / max_abs**: over the nodal values of the recovered field at the nodes of the solution
  space (vertices, plus midside nodes for P2) of the group (`None` = whole mesh). The group may be of
  any kind.
- **integral / mean**: Gauss quadrature over a cell or facet group (`None`/`"*"` = all cells) with
  the thickness or `2 pi r` weight. `mean = integral / weighted measure`. Displacements and T use the
  FE solution. Stresses and fluxes use the recovered continuous field.
- **point**: probes the field (stresses: the recovered field). Points slightly outside the discrete
  domain (curved boundaries approximated by chords, round-off on edges and vertices) are evaluated
  in the nearest cell by polynomial extension. Points more than 1% of the bounding-box diagonal away
  raise `ValueError`.
- **point BCs**: the nearest node of the solution space. A warning is added if that node is more than
  1% of the diagonal away.
- **Several Dirichlet BCs on one DOF**: the last one in the list wins, with a warning if the values
  differ.
- **Loads**: body forces add up. Temperature loads assign `T` on their cell group; a later load
  overrides an earlier one on overlapping cells. Cells without a temperature load are at `T_ref`.
- **Heat `2d`**: multiplied by the thickness too, so reactions (heat flows) and integrals are per
  thickness. Convection adds `∫ h T v` to the matrix and `∫ h T_inf v` to the load.
- **QoI failures**: `evaluate()` raises. `evaluate_all()` / `summary()` give NaN for that QoI and add
  a warning, so one bad probe point does not lose the other QoIs. The CLI writes it as `null`.
- **Expressions**: `feacore.case.eval_expr` in x, y, z, where axisymmetric x = r and y = z. Dirichlet
  values are evaluated at the node locations; load and flux values at the quadrature points.
- `SolveInfo`: `n_nodes` = nodes of the mesh actually used (the corner mesh for P1 on a quadratic
  file); `n_dofs` = unknowns before Dirichlet elimination; `measure` = area/volume of the discrete
  mesh (no thickness or `2 pi`); `h_rep = (measure / n_cells)^(1/dim)`; `wall_time` = mesh read +
  assembly + solve + recovery. `warnings` holds real warnings and `info:` lines (linear solver used,
  axis constraint, P1 on a quadratic mesh).

## Stress (and heat flux) recovery

The default `analysis.options["stress_recovery"] = "l2"` is a global L2 projection. Each stress
component is computed at the quadrature points from the FE displacement (thermal strain removed;
axisymmetric hoop term included). It is then projected onto the continuous scalar Lagrange space of
the solution order on the same mesh: solve `M s = ∫ σ_h φ dx` with the consistent, unweighted mass
matrix, factorised once and solved for all components with the configured direct solver. von Mises,
Tresca and the principal stresses (eigenvalues of the full 3x3 tensor: the hoop stress is the third
normal component in axisymmetric, 0 / `szz` in plane stress / plane strain) are computed pointwise
from the recovered components. Heat flux `q = -k grad T` is recovered the same way. The projection
reproduces constant fields exactly (patch tests) and converges at O(h^p) in general, with the largest
errors at boundary nodes and corners.

## Linear solvers (`feacore/solver/linsolve.py`)

`solve_linear(K, f, method="auto")`. All methods are direct by default, so the answer doesn't depend on
the hardware or on an iterative tolerance that would contaminate a GCI study.

| method | what |
|---|---|
| `scipy` | SuperLU with MMD(AᵀA+A) ordering, symmetric mode, diagonal pivoting (Cholesky-like for SPD). Always available, single-threaded |
| `pardiso` | MKL PARDISO via `pypardiso` (`pip install pypardiso`, x86-64 only), real SPD (mtype 2) on the upper triangle. Multithreaded (`MKL_NUM_THREADS`) |
| `cudss` | NVIDIA cuDSS via nvmath-python (`pip install nvmath-python[cu12]` or `[cu13]`): `nvmath.sparse.advanced.direct_solver(A, b, options=DirectSolverOptions(sparse_system_type=SPD), execution="cuda")` with host SciPy/NumPy operands (int32 indices; int64 above 2^31 nonzeros) |
| `auto` | `cudss` if a CUDA GPU and nvmath are present and n >= `FEACORE_CUDSS_MIN_N` (default 20000; smaller systems are faster on the CPU), else `pardiso` if installed, else `scipy` |
| `cg` | opt-in iterative (pyamg smoothed aggregation if installed, else Jacobi), rtol 1e-10, flagged `ITERATIVE` in the info line |

Selection order: an explicit `analysis.options["linear_solver"]` or `solve(case, linear_solver=...)`
(not `auto`) wins. Otherwise the env var `FEACORE_LINEAR_SOLVER` is used, otherwise `auto`. If the
requested method is unusable on this machine, the solver falls back to the next direct method and
notes it (`strict=True` raises instead). After every solve the relative residual is checked: above
1e-6 the case fails as singular. The method, ordering or matrix type, size, time and residual go into
`SolveInfo.warnings` as an `info: linear_solver=...` line.

Before solving, the backend checks that the Dirichlet set removes all rigid-body modes: 2D 3,
3D 6, axisymmetric the z translation, heat the constant mode (unless there is convection). If one is
left free it raises `InsufficientConstraints` (a `CaseError`): `insufficient constraints: the model
can move as a rigid body (rotation about z)`. That tells the agent exactly what is missing.
Mechanisms that are not rigid-body modes (e.g. two parts joined at a single node) are caught by the
residual check. MKL threads follow `MKL_NUM_THREADS` / `OMP_NUM_THREADS`: set them to the cores
you can spare per solve when running many solves in parallel.

## Assembly

The stiffness and mass matrices come from batched dense products over the quadrature points
(`_skfem_assembly.py`, cells in batches sized to keep temporaries around 20 MB). scikit-fem's generic `BilinearForm` loops in Python over
the 900 local basis pairs of a P2 tetrahedron. The vectorised assembly matches it to 1e-16
(`analysis.options["assembly"] = "skfem"` selects the reference path; there is a test for it) and is
about 15x faster for 3D P2. Loads use scikit-fem linear forms per component. Quadrature degree:
2·order (+1 in axisymmetric); override it with `analysis.options["intorder"]`. Vector unknowns are
interleaved per node.

## Verification (`tests/test_solver_skfem.py`, `tests/test_linsolve.py`; about 5 s)

- Patch tests on distorted unstructured meshes reproduce a homogeneous stress to < 1e-9 (observed
  1e-12) for every formulation (plane stress, plane strain, axisymmetric with and without the axis,
  3D), P1 and P2, on linear and quadratic meshes. Both a traction-driven version and a Dirichlet
  version (expression BCs) are tested.
- Lamé thick cylinder (a=1, b=2, internal p), plane-strain quarter with symmetry BCs, h = 0.1 against
  the closed form. P2: u_r(a) -2.3e-5, hoop stress at a -3.7e-3, at b -7.2e-4. P1: u_r -5.1e-3. The
  x-reaction on the symmetry plane matches the hoop resultant to 1e-9. Axisymmetric (h = 0.1) P2:
  u_r 1.1e-5, hoop 1.4e-3, axial stress 1.8e-3; the full-360° axial reaction is exact to 1e-13.
- Convergence (Lamé strain energy, h 0.2 → 0.1): observed order ~2 for P1 and ~4 for P2.
- Equilibrium: reaction = -applied force (2D, 3D), pressure expression resultant, body-force weight.
  Two materials in series match the analytic elongation.
- Free thermal expansion gives zero stress (< 1e-9 E alpha dT) in all four formulations with
  u = alpha dT x (plane strain: (1+nu) alpha dT x and szz = -E alpha dT). A constrained strip gives
  `-E alpha dT`, and a linear temperature field in a free plate gives no stress.
- Heat: a slab with a source is exact with P2 (quadratic profile); flux plus convection is exact; the
  axisymmetric hollow cylinder heat flow matches 2πkH ΔT/ln(b/a) to 2.4e-7 (P2); 3D slab.
- Linear solvers: PARDISO = SciPy to 1e-10 on a 3D P2 elasticity system. The cuDSS test runs only
  with a CUDA GPU and nvmath (it compares against SciPy to 1e-10), so it is skipped here.

## Performance

Measured in the dev container: 4 shared cores, load average 5-8 from other agents, so expect
±30% noise. `OMP_NUM_THREADS=1`, so PARDISO runs single-threaded. Model: cantilever block, clamped
face plus end force, P2 unless noted, quadratic mesh for P2. "wall" is `SolveInfo.wall_time` (mesh
read, assembly, solve, stress recovery); "solve" is the stiffness factorisation and solve alone.
To reproduce: mesh a box (or a 2x1 rectangle) with gmsh at size h, then run `solve(case)` and read
`info.wall_time` and the `info: linear_solver=... t=` line.

| model | DOFs | cells | solver | wall [s] | solve [s] |
|---|---:|---:|---|---:|---:|
| 2D plane stress P2 | 7.7k | 1.9k | pardiso | 0.14 | 0.05 |
| 2D plane stress P2 | 30k | 7.4k | pardiso | 0.66 | 0.33 |
| 2D plane stress P2 | 119k | 29.5k | pardiso | 2.4 | 1.0 |
| 2D plane stress P2 | 476k | 119k | pardiso | 14.9 | 4.8 |
| 2D plane stress P2 | 119k | 29.5k | scipy | 4.1 | 2.3 |
| 2D plane stress P1 | 120k | 119k | pardiso | 2.9-4.2 | 0.7-1.2 |
| 3D P2 (tet10) | 8.6k | 1.6k | pardiso | 0.56 | 0.23 |
| 3D P2 (tet10) | 24k | 5.0k | pardiso | 2.3 | 0.79 |
| 3D P2 (tet10) | 72k | 15.7k | pardiso | 7.4-11.4 | 4.4-4.8 |
| 3D P2 (tet10) | 141k | 31.6k | pardiso | 21.5-24.7 | 13.3-14.4 |
| 3D P2 (tet10) | 24k | 5.0k | scipy | 8.2 | 6.7 |
| 3D P2 (tet10) | 72k | 15.7k | scipy | 100 | 72 |
| 3D P1 (tet4) | 62k | 111k | pardiso | 8.3-10.5 | 3.0 |
| 3D P1 (tet4) | 156k | 290k | pardiso | 33.6 | 15.4 |

Takeaways. In 3D the direct solve dominates and SciPy SuperLU is 10-15x slower than PARDISO, so
install `pypardiso` (x86) or use cuDSS (GPU). Multithreaded PARDISO or cuDSS on a Brev box should
cut the "solve" column several-fold. What is left is mostly sparse-matrix construction (scipy
COO→CSR), roughly 50 µs per P1 tet and 200 µs per P2 tet (everything except the solve,
single-threaded). For every registered
benchmark problem at its coarsest study size, the solve takes 0.05-4 s.

## Limitations / not done

- Linear elastostatics and steady heat only: no contact, plasticity, large deformation, dynamics,
  modal or buckling analysis. No coupled thermo-mechanics: thermal loads are prescribed temperature
  fields (scalar or expression), not a heat solution.
- Simplices only (tri3/tri6/tet4/tet10). Quads, hexes, wedges and 3rd-order cells raise a clear error.
- One material per cell (regions by cell groups). Isotropic linear elasticity only.
- max/min are nodal and do not search inside the cells for P2. Recovered stresses at re-entrant
  corners and singular points diverge with refinement, as they should; the verification layer flags
  this.
- Stress recovery is a global L2 projection only (no SPR/ZZ alternative yet). The projection solve is
  a direct factorisation of the mass matrix (cheap compared with the stiffness).
- cuDSS is written against the nvmath-python 1.0 API (checked against the 1.0.0 wheel source) but has
  not been run here (no GPU). The test runs automatically on a CUDA box.
- Inverted or tangled curved cells (Jacobian changing sign) and degenerate cells are only reported
  as warnings.
