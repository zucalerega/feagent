# Time-dependent and modal analyses in the fixed solver

The fixed solver (`feacore`) was steady and linear. This document is the shared definition of the analyses
that go beyond that: what a case looks like, what a result looks like, and how each one is verified. The
case format is in the module docstring of `feacore/case.py` ("Time-dependent and modal analyses"); this file
adds the conventions that hold across physics and one section per capability.

| capability | physics, backend | module | status |
|---|---|---|---|
| transient heat conduction | heat, skfem | `feacore/solver/_skfem_transient.py` | done; verified (closed forms, orders, NAFEMS T3), `tests/test_transient_heat.py` |
| modal analysis | elasticity, skfem | `feacore/solver/_skfem_modal.py` | done, verified (NAFEMS FV32 / FV41 / FV52, exact Rayleigh-Lamb, beam theory) |
| transient dynamics | elasticity, skfem | `feacore/solver/_skfem_dynamics.py` | done, verified (exact discrete and modal solutions, order 2, energy, waves) |
| transient incompressible flow | flow, openfoam | `feacore/solver/_foam_transient.py` | see section |
| turbulent flow (RANS) | flow, openfoam | `feacore/solver/openfoam_backend.py` | see section |
| compressible internal flow | compressible, openfoam | `feacore/solver/_foam_compressible.py`, `_foam_internal.py` | done, verified (exact quasi-1D nozzle in three regimes, NACA 1135), `tests/test_compressible_internal.py` |

A capability whose module is not there yet raises `NotImplementedError` with its name; nothing falls back
to a steady solve.

## What is shared

**The case.** `analysis.options["transient"] = {"t_end", "dt" | "n_steps", "scheme", "initial",
"output_times" | "output_every"}` makes a case time-dependent; `analysis.options["modal"] = {"n_modes",
"shift"}` asks for natural frequencies. `Case.validate` checks both (`_validate_time` in `feacore/case.py`):
unknown keys are errors, so a misspelt option cannot pass in silence.

**Values in time.** In a transient case a value may use the time `t` in an expression, or be a table
`{"table": [[t0, v0], [t1, v1], ...]}` (linear between points, constant outside). `feacore.case.as_field(value,
x, y, z, t)` and `eval_expr(expr, x, y, z, t)` evaluate both; `uses_time(value)` says whether a value depends
on time, so a solver can assemble once what does not. A steady case refuses a value that depends on time.

**Results.** A transient solve returns a `feacore.solver.base.TransientResult`: `times` (every step), `kept`
(the steps whose full state is stored), `series(qoi)` (the QoI at every step), `state(t)` (the stored state
nearest to t, a `Result`), and `history(qoi)`. `evaluate(qoi)` reduces the series as `QoI.time` asks (the value
at `t_end` by default; a time; "max", "min", "max_abs", "mean", "rms", "frequency") over `QoI.window`. The
reductions are in `reduce_series` and `dominant_frequency` of the same module, so every backend reduces the
same way. A modal solve returns a `ModalResult`: `frequencies` (Hz, elastic modes, ascending), `rigid_modes`,
`effective_mass`, `shape(mode)`; QoI types `natural_frequency` and `effective_mass` with `QoI.mode`.

**SolveInfo.** A transient result's `info` carries the scheme, the number of steps and the step in its
`warnings` as `info:` lines, like the steady backends do for the linear solver.

## How each capability is verified

The rules of the existing solver hold (`docs/code_verification.md`, `CLAUDE.md` "FEA core decisions"):

1. **A closed form or a published benchmark, with its source.** Every capability is checked against an exact
   solution or a benchmark with published reference values, in a test, with the tolerance stated and
   explained.
2. **The order of accuracy is observed, in space and in time.** Refine the mesh at a small fixed step, and
   the step on a fine fixed mesh; the observed order must match the scheme (2 for Crank-Nicolson and for the
   average-acceleration Newmark scheme, 1 for backward Euler). `feacore.verification.gci.gci3` takes the step
   size as the refinement parameter just as it takes h. A time step is a discretization like a mesh: a
   transient answer without a step-refinement check is not a converged answer.
3. **Existing cases do not change.** A steady case gives the same numbers as before, to round-off.
4. **Limits are stated.** What the scheme cannot do (a step too large for a load that changes fast, a mode the
   mesh cannot resolve, a turbulence model outside its validated range) is in this document and, where the
   solver can detect it, in `info.warnings`.
5. **Benchmark problems are registered** in `feacore/problems/` like the steady ones (statement, reference
   values with their source, tolerances), so the dashboard and the agents can run them.

## Transient heat conduction

Module `feacore/solver/_skfem_transient.py` (its docstring has the details), tests
`tests/test_transient_heat.py`, benchmarks `feacore/problems/transient_heat/`.

**Equation.** rho cp dT/dt - div(k grad T) = q, T(x, 0) = T0(x), with the BCs of the steady heat case
(`temperature`, `flux`, `convection`) and `source` loads. Any of them may depend on time (an expression with
`t`, or a table); for a convection BC also `h` and `T_inf` (in t only, not in x, y, z). The space is the steady
one (P1 / P2, `2d` with thickness, `axisymmetric` with 2 pi r, `3d`). The semi-discrete system is
M dT/dt + K(t) T = f(t) with the consistent capacity matrix M = int rho cp N N (rho cp per material region).

**Schemes** (fixed step dt = t_end / n from `Analysis.time_steps()`), A(a, t) = a M + K(t) on the free unknowns:

| scheme | step | order |
|---|---|---|
| `crank_nicolson` (default) | A(2/dt, t1) T1 = (2/dt M - K(t0)) T0 + f(t0) + f(t1) | 2 |
| `backward_euler` | A(1/dt, t1) T1 = M T0 / dt + f(t1) | 1 |
| `bdf2` | A(3/(2dt), t1) T1 = M (2 T0 - T_-1 / 2) / dt + f(t1); first step backward Euler | 2 |

A prescribed temperature g(t) enters with its value at the new time and, through the old states (which hold
g at the old times), at the old times.

**What is assembled once.** K, f and the prescribed values come from the steady assembler `_assemble_heat`
(no second implementation), called on sub-cases: once for everything that does not depend on time; once
with a unit value for each term whose value depends on t only (scaled per step: h(t), T_inf(t), a flux, a
source or a temperature given by a table or an expression in t); and at every step only for a value in x
and t together. M is assembled once. The matrix A is factored once and its factorization solves every
step; it is factored again only when it changes (a time-dependent `h`, BDF2's second step, a start-up
half-step at a new h). `linsolve` has no reusable factorization, so the module keeps its own: MKL PARDISO
(SPD) through `pypardiso` when installed, else SciPy SuperLU in symmetric mode. cuDSS / CG are not used for
time stepping (noted in `info.warnings`).

**Options** (`analysis.options`): `transient` as in the shared schema (`scheme`, `initial` = `{"T": number |
expression in x, y, z | {"field": "<.npz with x and T>"}}`, required; a field file goes to the nodes by the
rule of the steady thermal load, nearest point within 1e-6 of the bounding-box diagonal), `cn_startup` =
`"auto"` (default) | `"always"` | `"off"`, and the steady ones (`linear_solver`, `intorder`, `assembly`).
Unknown options warn, as in a steady solve.

**Start-up rule (Crank-Nicolson).** Crank-Nicolson multiplies the modes a step cannot resolve by nearly -1
each step. A jump of a face temperature at t = 0 (quench) excites them fully, and so does, less visibly, a
step of a flux or of a convection BC's fluid temperature (the initial state then does not satisfy the
boundary flux): measured on the convective slab at Bi = 10, the surface temperature alternated by 25 % of the
step for many steps without a start-up. Rule: with `cn_startup = "auto"`, whenever the initial state is not
a steady state of the data at t = 0 (residual f(0) - K T0 of the free unknowns above 1e-9 of its terms),
the first two steps are taken as four backward-Euler half-steps (Rannacher 1984; Luskin & Rannacher 1982).
They use the Crank-Nicolson matrix (no extra factorization) and keep second order (the manufactured
solution's observed order is 2.0 with them, 1.6 without: projecting a smooth T0 on the mesh is itself a
slightly rough start). An incompatible temperature BC is also named in `info.warnings`; with `"off"` it is a
warning. Backward Euler and BDF2 damp these modes and need no start-up.

**Results.** `TransientHeatResult`: `times`, `kept` (default 10 states evenly spaced and the last;
`output_times` / `output_every`), `series(qoi)` (every QoI of the case at every step, evaluated during the
run: point, point_difference, max, min, max_abs, mean, integral, and reaction = heat flow into the body
through a temperature BC, M dT/dt + K T - f on its nodes with dT/dt the backward difference of the scheme's
order), `state(t)` (a `SkfemResult` of a kept step: `probe`, `nodal("T")`, `nodal("qx")`, `evaluate`,
`write_vtu`), `evaluate(qoi)` (reduced as `QoI.time` asks), `info` (scheme, steps, dt, factorizations, solve
times as `info:` lines; wall time).

**Verification** (`tests/test_transient_heat.py`; errors relative to the temperature step):

| check | setup | measured |
|---|---|---|
| quench slab, Fourier series | P2 40 cells, CN 80 steps to Fo = 0.2; `2d` / `axisymmetric` / `3d` (P1) | 1.9e-4 / 1.9e-4 / 4.7e-4 (t >= t_end / 4) |
| convective slab, roots of z tan z = Bi | P2 20 cells, CN 100 steps to Fo = 1, Bi = 0.1 / 1 / 10 | 7e-5 / 6.6e-4 / 1.1e-3 after step 5 (step 1: 0.07 % / 0.7 % / 3.7 %, the sqrt(t) start) |
| convective slab in 3D | P2 tets, 60 steps, Bi = 1 | 8e-4 after step 5 |
| lumped body, Bi = 1e-3 | P1, CN 40 steps to 2 tau | 1.3e-3 (O(Bi) model error + start-up) |
| solid cylinder, Bessel series | axisymmetric P2 20 cells, 100 steps, Bi = 1 | 5e-4 after step 5 |
| order in time (manufactured T(x, y, t), P2 12x12, 8 .. 64 steps) | CN / BDF2 / BE | 1.97 / 2.31 (2.19 on 16 .. 128) / 0.97 |
| order in space (manufactured, small step, max error at 30 probes) | P1 8 .. 32 / P2 4 .. 16 cells | 1.92 / 3.25 |
| NAFEMS T3, 36.60 degC at x = 0.08 m, t = 32 s (exact series 36.603) | registered `nafems_t3`, h0 = 5 mm, CN 64 steps | P2 36.599, P1 36.83 |
| long-time limit | BE to Fo = 60 | equals the steady solve to 1e-6 |
| nothing changes | all schemes | stays at T0 to 1e-12 |
| energy balance (stored heat vs time integral of the boundary heat flows) | smooth data, 200 steps | BE 4e-13, CN 2e-5, BDF2 2.6e-3 |

Also tested: h(t) against the lumped exponential of int h dt, a table h that steps once (two factorizations),
a Dirichlet table against the same expression, an initial field from a file, the start-up against ringing,
`QoI.time` = a time / "max" / "mean" with a window, `output_times`, unknown options, a fast-changing value.
Typical 3D run (box, P2 tets, 13 260 nodes, 100 steps, CN, two time-dependent BCs, three QoIs every step):
17 s on 2 threads of a shared machine (one factorization 0.55 s, 102 solves 12.4 s, the rest QoIs).

**Benchmarks registered** (`feacore/problems/transient_heat/`, tag `transient`): `nafems_t3` (published
target 36.60 degC, 1 %; source: the NAFEMS TNSB T3 data as reproduced in the Abaqus Benchmarks Manual "T3:
One-dimensional transient heat transfer" and Altair OptiStruct OS-V: 0110) and `convective_slab_chilldown`
(a 10 mm stainless wall wetted by liquid nitrogen, h = 2000 W/(m^2 K), Bi = 1.25; analytical references from
the series, 0.2 %).

**Limits.**
- The step must resolve the fastest change of the data: a value in time that leaves the chord between two
  steps by more than 2 % of its range is warned about (`info.warnings`). The schemes see step values only.
- A sudden change at a face (a quench, a fluid step) is a singularity in time: the first steps carry an
  error of a few % at the face whatever the scheme (sqrt(t) start). When the cells at a face whose
  temperature jumps are larger than the penetration depth sqrt(alpha dt) of one step, the consistent
  capacity matrix over- and undershoots in the first cells: warned; refine the mesh towards the face.
- Unresolved modes: Crank-Nicolson keeps them (hence the start-up); backward Euler and BDF2 damp them, at the
  price of first order (BE) or of damping physical fast modes (both).
- Linear only: k, rho, cp constant in time and temperature; no radiation; h and T_inf may depend on t only.
  The step is fixed (no adaptivity).

**Example.**
```python
from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.solver import solve

case = Case(mesh="wall.msh",
            analysis=Analysis(physics="heat", formulation="2d", order=2, options={"transient": {
                "t_end": 30.0, "n_steps": 120, "scheme": "crank_nicolson", "initial": {"T": 293.15},
                "output_times": [10.0, 30.0]}}),
            materials={"steel": Material(k=16.0, rho=8000.0, cp=500.0)},
            bcs=[BC(type="convection", group="wetted", h=2000.0, T_inf={"table": [[0, 293.15], [1, 77.4]]})],
            qois=[QoI(name="T_outer_min", type="point", field="T", point=[0.01, 0.001], time="min")])
r = solve(case)
r.evaluate_all(); r.series(case.qois[0]); r.state(10.0).write_vtu("t10.vtu")
```

## Modal analysis

`feacore/solver/_skfem_modal.py`; tests `tests/test_modal.py`; problems `feacore/problems/dynamics/`
(`nafems_fv32`, `nafems_fv41`, `nafems_fv52`, `ss_strip_modes`, `ss_thick_plate_modes`).

**Equations.** K phi = omega^2 M phi. K is the stiffness of the steady elasticity case
(`_assemble_elasticity`: the same formulations, materials, Dirichlet sets and options), M the consistent
mass matrix int rho c u.v with the density of each material region and c = thickness (plane stress / strain),
2 pi r (axisymmetric) or 1 (3D). Loads and traction / pressure / force BCs are ignored, and every Dirichlet BC
(`fixed`, `displacement`, `symmetry`, the axis constraint) acts as zero, a prescribed non-zero displacement
included; both are said in `info.warnings`.

**Solver.** `{"modal": {"n_modes": n, "shift": f}}`: the n lowest elastic modes above f Hz (default 0) by
shift-invert Lanczos (`scipy.sparse.linalg.eigsh`) with ONE sparse factorization of K - sigma M (MKL PARDISO when
installed, else SuperLU; `linear_solver` chooses) kept for all iterations. Restrained model: sigma = 0. Free
body: sigma is a small negative number (minus a geometric estimate of the first elastic eigenvalue), so
K - sigma M is definite. Shift f > 0: sigma = (2 pi f)^2 (an indefinite factorization). Below 400 free
unknowns the dense LAPACK solver is used. Unknown options warn (`unknown analysis options ignored`).

**Rigid-body modes.** A model that is not restrained is legal. The rigid motions the Dirichlet set leaves free
(2D: two translations and the rotation; 3D: three and three; axisymmetric: the axial translation) are built in
closed form, as `skfem_backend._check_rigid_modes` builds them, and the Lanczos operator is deflated by them
(M-orthogonal projection). So `ModalResult.rigid_modes` is an exact count, not a threshold on small
eigenvalues; it is in `info.warnings` with the names of the motions, and the rigid modes are not counted
among the `n_modes`. A near-zero elastic eigenvalue that is not a rigid motion of the whole model (a mechanism,
a part that is not connected) is flagged.

**Result.** `SkfemModalResult`: `frequencies` (Hz, ascending), `eigenvalues` (omega^2), `modes` (N, n:
mass-normalised, phi^T M phi = 1, sign: largest component positive), `rigid_modes`, `rigid_names`,
`rigid_shapes`, `effective_mass` (n, dim): (phi_k^T M 1_j)^2 / m_j, the share of the total mass that mode k moves
along axis j (for a restrained model the shares of all modes sum toward 1; the rest sits on the supports;
axisymmetric: only the axial column, the radial one is NaN), `participation`, `total_mass`, `residuals`,
`K`, `M`; `shape(mode)`, `nodal(field, mode=1)`, `probe(field, points, mode=1)` for ux, uy, uz, u_mag (a mode has
no amplitude, so stresses are refused), `write_vtu` (one file, a vector `mode_<k>` per mode, the frequencies in
the field data). QoIs: `natural_frequency` and `effective_mass` with `mode` (and `component`).

**Verification** (`tests/test_modal.py`; P2, all errors measured):

| check | reference (source) | result |
|---|---|---|
| plane-stress cantilever, 4 cells through the depth, L/H = 10, 20, 40 | Euler-Bernoulli (beta L = 1.8751, 4.6941) | f1 -0.65 %, -0.13 %, -0.017 %; f2 -4.9 %, -1.27 %, -0.31 %: always below (shear and rotary inertia), the gap / 4 per doubling of L/H; at L/H = 80 f1 is +0.004 % (the mesh error takes over) |
| simply supported strip (sliding supports), plane strain, L/t = 10, 20 | Rayleigh-Lamb A0 standing wave (exact, Graff 8.1); pinned Euler-Bernoulli | Lamb within 2e-4; the gap to the beam shrinks as (t/L)^2 |
| free-free bar 2D (nu = 0) / 3D box | c / (2 L) exact; free-free Euler-Bernoulli | 3 / 6 rigid-body modes; axial mode within 1e-5 (2D) / 1e-4 (3D); first bending within 1 % |
| `ss_strip_modes` (L = 10, t = 1, plane strain): six modes (A1..A5, S1, one rigid mode) | Rayleigh-Lamb, k = n pi / L (exact for 3D elasticity) | h = 0.5: +1.9e-4 .. +4.5e-3; h = 0.25: +7.8e-6 .. +2.0e-4; h = 0.125: +5.6e-7 .. +1.4e-5; observed order 4.4-4.6 then 3.8-3.9 (2p = 4); from above (consistent mass) |
| NAFEMS FV32 tapered membrane, plane stress | 44.623, 130.03, 162.70, 246.05, 379.90, 391.44 Hz | h = 1 (h0): +0.007 .. +0.34 %; h = 0.5: -0.013 .. 0 %; h = 0.25: -0.044 .. -0.005 % |
| NAFEMS FV41 free cylinder, axisymmetric | 243.53, 377.41, 394.11, 397.72, 405.28 Hz | 1 rigid mode (axial); h = 0.2 (h0): -0.014 .. +0.030 %; h = 0.1: -0.014 .. +0.029 % |
| NAFEMS FV52 'solid' square plate, 3D, uz = 0 on the bottom edges | 45.897, 109.44, 109.44, 167.89, 193.59, 206.19, 206.19 Hz | 3 rigid modes; h = 2: f1 -2.5 %, the rest within 0.22 %; f1 drifts down with refinement (-4.2 % at h = 1, -5.6 % at 0.5): a line support of a solid has no converged answer; only the in-plane mode f5 converges (problem module) |
| `ss_thick_plate_modes` (3D, 10 x 10 x 1, sliding supports) | Rayleigh-Lamb A(1,1), A(1,2)x2, shear-horizontal c_t / (2 a) x2, A(2,2) | h = 1: +0.5, +1.1, +1.2, 0.000, 0.000, +1.8 % (slow test; the flexural modes converge slowly with one or two cells through the thickness) |
| mass matrix | rho V t (plane), rho pi (ro^2 - ri^2) L (axisymmetric), rho V (3D); 1^T M 1 = m per axis | to 1e-12 |
| mode shapes | Phi^T M Phi = I, Phi^T K Phi = diag(omega^2) | 1e-9; residuals below 1e-8 |
| effective mass | cantilever mode 1: 0.613 (Euler-Bernoulli) | 0.613 +- 0.005; 30 modes sum to > 0.9 in y; free body: 0 |
| shift, dense path | the modes above the shift equal modes 4-6 of a shift-0 run; dense LAPACK = Lanczos | 1e-9 |

Sources of the NAFEMS values: NAFEMS TNSB Rev. 3 (1990) and R0015 (1987), as reproduced in the Abaqus Benchmarks
Guide (FV32, FV41, FV52), the Altair OptiStruct verification manual (OS-V 0440, 0450, 0455) and the Ansys
Verification Manual (benchmark C5, FV52). The booklets and those pages were not opened from this machine (their
hosts are blocked); the values come from search extracts of them, which confirm the FV32 frequencies, plate
thickness 0.05 m and the 5 m to 1 m taper. The FV32 length (10 m) and the FV41 radii and length (1.8 / 2.2 m,
10 m) are not confirmed by an extract; with them the solver converges to every published frequency within
0.05 %, which wrong dimensions would not do.

**Limits.** Linear elastic, small displacements, no prestress (no geometric stiffness), no damping, no
complex modes. Modes the mesh cannot resolve: a frequency is good when the mesh has several P2 cells per half
wave of the mode (the strip: 8 cells per half wave give 2e-4); the highest of `n_modes` is the least
accurate. Point or line supports of a solid soften with refinement (FV52). Clustered or repeated
frequencies are found (Lanczos with two spare vectors), but the shapes within a cluster are any orthonormal
basis of it. Cost: 13 000 unknowns (3D P2 cantilever, 4436 nodes), 10 modes: 5-11 s on 2 shared cores.

**Example.**

```python
Case(mesh="plate.msh",
     analysis=Analysis(formulation="plane_stress", order=2, thickness=0.05,
                       options={"modal": {"n_modes": 6}}),
     materials={"steel": Material(E=200e9, nu=0.3, rho=8000)},
     bcs=[BC(type="fixed", group="root")],
     qois=[QoI("f1", "natural_frequency", mode=1), QoI("m1y", "effective_mass", mode=1, component=1)])
```

## Transient dynamics

`feacore/solver/_skfem_dynamics.py`; tests `tests/test_dynamics.py`; problem `bar_wave_step`
(`feacore/problems/dynamics/bar_wave.py`).

**Equations.** M a + C v + K u = f(t). K, the loads and the Dirichlet set are the steady elasticity case's;
M is the consistent mass of the modal analysis; C = a M + b K from `analysis.options["damping"] =
{"rayleigh": [a, b]}` (default none), i.e. a modal damping ratio zeta(omega) = a / (2 omega) + b omega / 2. Any
load or BC value may depend on time (an expression with `t`, or a table); what does not is assembled once. A
timed value without x, y, z is one vector times a number of t (assembled once); one that varies in space and
time is reassembled each step on a kept basis. Body forces, pressures, tractions, forces and Dirichlet values
may depend on time; a temperature load that does is refused (`CaseError`).

**Schemes** (`transient.scheme`), both in the Hilber-Hughes-Taylor form

    M a_{n+1} + (1 + alpha) (C v_{n+1} + K u_{n+1}) - alpha (C v_n + K u_n) = (1 + alpha) f_{n+1} - alpha f_n,
    beta = (1 - alpha)^2 / 4,  gamma = (1 - 2 alpha) / 2

- `newmark` (default): alpha = 0, the average-acceleration (trapezoidal) rule: unconditionally stable, second
  order, no numerical damping; each mode turns by omega_h dt = 2 arctan(omega dt / 2) per step, a period
  elongation of (omega dt)^2 / 12 + O(dt^4).
- `hht`: `transient.alpha` in [-1/3, 0], default -0.05: second order, damps the modes the step cannot resolve
  (spectral radius (1 + alpha) / (1 - alpha) as omega dt -> infinity).

The fixed step comes from `Analysis.time_steps()`. The unknown of a step is u_{n+1}; a and v follow from the
Newmark relations. **Factored once:** the effective matrix c0 M + (1 + alpha) (c1 C + K) (c0 = 1 / (beta dt^2),
c1 = gamma / (beta dt)) is constant for the fixed step; its free-free block is factored once and reused at every
step (`_skfem_modal.Factorization`: SuperLU below 30 000 unknowns, where PARDISO's per-solve overhead
dominates, else the selected solver), as are M_II (for the initial acceleration) and the scalar mass of the
stress recovery.

**Initial state.** `transient.initial = {"u": [...], "v": [...]}` (components as numbers or expressions in x, y,
z; default at rest). The initial acceleration is the equilibrium one: M_II a_I = (f - C v - K u)_I - M_ID a_D at
t = 0.

**Dirichlet BCs.** The constrained dofs are those of the steady case (same rules: the last BC wins, u_r = 0 on
the axis). A prescribed value in time is imposed on u_{n+1}; its velocity and acceleration are the ones the
Newmark relations give for that displacement history, so the prescribed motion is consistent with the scheme
(at t = 0 they come from one-sided differences of the prescribed value with a step of dt / 100). A prescribed
initial u or v on a constrained dof is overridden by the BC (warned).

**Rigid-body modes.** A free or partly free body is legal (M makes the effective matrix definite); the number of
rigid modes is in `info.warnings` and `result.rigid_modes`. A net load accelerates the body: the centre of mass
moves as F t^2 / (2 m), exactly.

**Result.** `SkfemDynamicsResult` (a `TransientResult`): `times`, `kept` (default 10 states evenly spaced and the
last; `output_times` / `output_every`), `series(qoi)` for every QoI of the case at every step, `evaluate(qoi)`
reducing it by `QoI.time` / `QoI.window`. Supported QoI types at each step: point, point_difference, max, min,
max_abs (displacements and recovered stresses), reaction (the DYNAMIC support reaction M a + C v + K u - f on the
constrained dofs), integral, mean, strain_energy. A stress QoI costs one stress recovery per step (a solve with
the kept factorization). `state(t)`: the kept state nearest to t as a steady `SkfemResult` (`DynamicState`:
probe, nodal, evaluate, stresses, write_vtu with velocity and acceleration). `energy`: per-step arrays
`kinetic`, `strain`, `external_work` (loads and support reactions, trapezoidal), `damping` (dissipated) and
`balance` = kinetic + strain + damping - work - initial energy, zero to round-off for `newmark`.
`info.warnings` has an `info: dynamics:` line with the scheme, alpha, beta, gamma, the steps, the damping and the
factorization.

**Verification** (`tests/test_dynamics.py`; P2; measured):

| check | reference | result |
|---|---|---|
| free vibration of a bar (nu = 0, near its first axial mode) | exact DISCRETE solution: each mode at constant amplitude turning by 2 arctan(omega dt / 2) (dense eigen-decomposition of the solver's K, M) | 2e-12 relative at 10 and 40 steps per period: the scheme is implemented exactly |
| period error | omega_h / omega = 2 arctan(omega dt / 2) / (omega dt) | measured frequency -3.1 %, -0.81 %, -0.20 % at 10, 20, 40 steps per period, matching the closed form within 3e-4 / 5e-5 |
| order in time, bar | exact semi-discrete solution (modal superposition) | newmark: errors 0.35, 0.089, 0.023, 0.0057 at 10..80 steps per period (orders 1.96, 1.98, 1.99); Rayleigh [200, 1e-7]: 0.30, 0.077, 0.0195, 0.0049 (1.96, 1.99, 1.99); HHT -0.05: 1.94, 1.97; HHT -0.3: 1.89, 1.97 |
| Rayleigh damping | zeta_1 = a / (2 omega_1) + b omega_1 / 2 = 0.0131 | the logarithmic decrement of the free decay within 1 % |
| energy, undamped newmark free vibration | conserved | kinetic + strain constant to 1e-13 relative over 5 periods |
| energy balance with damping and a load in time | kinetic + strain + dissipated = work | below 1e-10 of the work (measured 1e-13 .. 2e-11); HHT dissipates |
| prescribed base motion u_D = U (1 - cos W t) | the same problem in the moving frame (body force -rho a_D); exact modal solution | the two formulations differ by O(dt^2) (orders 1.99, 2.02); both 2.2 %, 0.84 %, 0.35 % from the exact at 40, 80, 160 steps (the step excites modes it cannot resolve at first) |
| free body under a constant force | u_cm = F t^2 / (2 m) | 1e-8 |
| time-only vs space-time load; table vs expression | the same load | 1e-12; 1e-9 |
| cantilever (plane stress, 1 x 0.1 m) under a step tip load | static deflection; f1 from the modal solver | peak / static 1.984, 1.985, 1.989 at 20, 40, 80 steps per period (HHT -0.05); dominant frequency -0.81 %, -0.22 %, +0.19 % from f1; mean over 4 periods +0.88 %, +0.22 %, +0.06 % from the static value |
| quasi-static ramp with damping | static reaction | dynamic reaction within 1 % of the load |
| `bar_wave_step`: step end load on a fixed-free bar (nu = 0), HHT -0.1, 200 steps per period | d'Alembert solution (Graff 2.1): u_tip a triangle wave, peak 2 sigma L / E, mean sigma L / E, frequency c / (4 L) | h = 0.05: peak -1.4 %, mean +0.055 %, frequency -0.009 %; peak -0.9 % at h = 0.025, -0.56 % at 0.0125 (a corner of the exact history: reached from below, slowly); the stress at mid-length is 0.6 % of sigma before the front arrives (0.8 L / (2 c)) and sigma within 15 % after it |
| QoI.time reductions | `reduce_series` | max / min / mean with window / a time / frequency through `result.evaluate` |

**Limits.** The step against the highest frequency of interest: the period error is 3.1 % at 10 steps per
period, 0.8 % at 20, 0.2 % at 40, 0.03 % at 100; a mode with fewer than ~5 steps per period is not followed at
all (newmark keeps its energy and turns it at the wrong frequency; HHT damps it). A step load or an impact
excites every mode of the mesh, so point histories converge slower than dt^2 until the step resolves the modes
that carry the response; use HHT or Rayleigh damping to quiet the unresolved ones, and check the step by
halving it. Modes the mesh cannot resolve (fewer than a few P2 cells per half wavelength) are wrong whatever the
step: a wave front is smeared over a few cells. A table whose points are closer than the step is warned about.
Linear elastic material, small displacements and rotations, no contact, no time-dependent temperature load,
no variable step. Cost: 217 nodes, 160 steps: 0.2 s; 3D P2 cantilever, 4436 nodes (13 000 unknowns), 200 steps
with a von Mises QoI: 51 s on 2 shared cores (about 0.13 s of each step is the stress recovery).

**Example.**

```python
Case(mesh="beam.msh",
     analysis=Analysis(formulation="plane_stress", order=2, thickness=0.01,
                       options={"transient": {"t_end": 0.05, "dt": 2.5e-4, "scheme": "hht", "alpha": -0.05},
                                "damping": {"rayleigh": [5.0, 0.0]}}),
     materials={"steel": Material(E=200e9, nu=0.3, rho=8000)},
     bcs=[BC(type="fixed", group="root"),
          BC(type="force", group="tip", value=[0.0, {"table": [[0.0, 0.0], [1e-3, -1000.0]]}])],
     qois=[QoI("peak", "point", field="uy", point=[1.0, 0.05], time="min"),
           QoI("f", "point", field="uy", point=[1.0, 0.05], time="frequency"),
           QoI("r_max", "reaction", group="root", component=1, time="max_abs")])
```

## Transient incompressible flow

(section written by the capability's author)

## Turbulent flow (RANS)

Steady RANS in the OpenFOAM backend (`feacore/solver/openfoam_backend.py`, `simpleFoam`):
`analysis.options["turbulence"] = "kOmegaSST"` (also wired: `"kOmega"`). Laminar cases are unchanged
(no k, omega or nut files are written).

**Wall treatment.** Every `wall` BC gets wall functions: `nutkWallFunction`, `kqRWallFunction`,
`omegaWallFunction` (smooth wall). The first cell centre must lie in the log layer. The result gives
`res.y_plus(group)` (per wall face, from k as the wall function computes it: y+ = Cmu^0.25 sqrt(k) y / nu;
from the wall shear when there is no k), and a turbulent solve adds an `info: y+ on wall ...` line (5 %,
median, 95 % of the wall area) and a warning when more than 5 % of a wall is under y+ = 30.

**Inlet turbulence.** k = 1.5 (I U_ref)^2 and omega = sqrt(k) / (Cmu^0.25 L) from `turbulence_intensity`
(I, default 0.05), `turbulence_length` (L, default 0.07 x the smallest bounding-box extent) and the new
option `turbulence_velocity` (U_ref; default the largest constant inlet speed, and 1 m/s when every inlet
is an expression, as before). The validation uses I = 0.16 Re^-1/8 and L = 0.07 D.

**Validated** (OpenFOAM v1912, structured hexahedra, `skew_correction` off, order 2, residual_tol 1e-5).
Developed flow in a smooth pipe of 60 D with a power-law inlet profile; friction factor from the pressure
gradient between 42 and 57 D against the Colebrook formula for a smooth wall (Prandtl's law). Meshes: the
O-grid of `dossier.analysis.flow.build_mesh`, m cells per quarter of the wall
(`tests/dossier/test_dossier_flow.py`, slow):

| Re | m = 4 | m = 5 | m = 7 | m = 10 | y+ of the wall cells |
|---|---|---|---|---|---|
| 1.0e5 | -3.0 % | -4.1 % | -4.6 % | | 169, 114, 85 |
| 1.384e6 | | -2.6 % | -2.9 % | -3.2 % | 1240, 940, 630 |

The benchmark problem `pipe_turbulent_3d` (`feacore/problems/cfd/pipe_turbulent.py`, Re 1e5, the O-grid of
`pipe_poiseuille_3d`, reference Darcy-Weisbach with Colebrook, tolerance 10 %) gives dp_AB +1.1 %, -1.5 %,
-3.6 % on its three study levels (median y+ 139, 126, 74). Sudden expansion d1/d2 = 0.7 at Re1 = 1.384e6
(42 D upstream, 45 D2 downstream, loss from the extrapolated mass-flow-mean total pressure lines): loss
coefficient +7.0 % (m 5) and +6.8 % (m 7) above Borda-Carnot (1 - beta^2)^2 = 0.260; the residuals stall at
about 3e-3 there (separated shear layer), while the loss changes by 0.2 % between the meshes.

Summary: the friction factor is 3 to 5 % low, a separation loss about 7 % high; within 10 % on the
friction factor, so usable with a stated model uncertainty of 10 % (the dossier flow engine adds it to the
mesh uncertainty).

**Limits (not validated, do not rely on them):**
- Only y+ 60 to 1600 on the wall cells. Finer wall meshes give a lower friction factor with this wall
  treatment (the error grows from -3 % to -4.6 % as y+ drops from 169 to 85 at Re 1e5); y+ under 30 is
  outside the wall functions' range, and a wall-resolved (low-Re) setup is not validated at all.
- Only structured hexahedra. On tetrahedra the same case gave -15 % (plain schemes) and -26 % (skew-corrected
  schemes) and did not converge; triangular prisms with the skew-corrected schemes gave +13 to +17 %.
- Mesh refinement does not converge the friction factor in the usual sense: with wall functions the result
  depends on y+, so a GCI over wall-function meshes estimates the discretisation error only while y+ stays in
  the range above.
- Smooth walls only (no roughness in the wall functions), Re 1e5 to 1.4e6, internal flow without swirl,
  curvature, heat transfer or other models (`kEpsilon` etc. raise "not wired up yet").
- Separated flow: steady SIMPLE does not reach residual_tol 1e-5 behind a step (the residuals stall near
  1e-4 to 3e-3). Judge such runs by the stability of the QoI: the option `history_interval` makes the solver
  write the fields every N iterations, and `res.history_times()` / `res.at_time(t)` give the earlier
  solutions (the dossier flow engine accepts a stalled run when the residuals are under 1e-2 and the pressure
  drop changed less than 0.1 % in the last 200 iterations).

## Compressible internal flow

Physics `compressible`, OpenFOAM backend: subsonic inflow from a reservoir, choking, normal shocks in a
duct and supersonic or subsonic outflow against a back pressure. Code: `feacore/solver/_foam_compressible.py`
(the run, the stopping rule, the `mass_flow` QoI) and its helper `feacore/solver/_foam_internal.py` (the
boundary conditions, the outlet switch, the mass balance). Benchmarks: `cd_nozzle_subsonic_2d`,
`cd_nozzle_shock_2d`, `cd_nozzle_supersonic_2d` (`feacore/problems/compressible/nozzle.py`; exact solution in
`feacore/problems/compressible/quasi1d.py`). Tests: `tests/test_compressible_internal.py`.

**Case.** `BC("total_inlet", group, p0=..., T0=...)` (Pa, K): the reservoir; the gas enters along the inward
normal. `BC("pressure_outlet", group, value=pb)`: the static back pressure (Pa, absolute; a number or an
expression in x, y, z). `QoI(name, "mass_flow", group=...)`: kg/s through the group, positive out of the
domain (2D: through the extruded depth `analysis.thickness`). Walls are `slip` (inviscid) or `wall` (laminar,
`mu > 0`), plus `symmetry`. A case with a `total_inlet` takes no `freestream` and needs a `pressure_outlet`
(`CaseError` otherwise). Forces use the gauge pressure p - pb of the first pressure outlet. A `freestream`
case may also use a `pressure_outlet` (supersonic inflow against a back pressure).

**Solver: `rhoCentralFoam`, as for the external flows.** Density based, Kurganov-Tadmor central-upwind
fluxes, explicit, conservative. Why not the pressure-based transonic solvers: a time-boxed trial of
`rhoPimpleFoam` (`transonic yes`, v1912, the same mesh and boundary conditions, cd_nozzle_shock_2d at h0)
blew up within a few iterations with local time stepping (`localEuler`) started from rest; time accurate at
Courant 0.5 it ran (5500 steps for 32 flow times, three times fewer than rhoCentralFoam) to a steady field
whose cell mass fluxes at inlet and outlet differed by 0.8 % and whose shock sat 1.5 % of L downstream of the
exact position, where rhoCentralFoam's mass balance closes to 1e-7 and its shock is within 0.2 % of L. The
density-based scheme conserves mass, momentum and energy across the shock by construction and starts from
rest without tuning. Its cost: the time step follows the acoustic speed, so a flow time L / a* takes ~600
steps per 40 cells along the duct, whatever the Mach number; below Mach ~0.1 the steps grow without bound
relative to the flow and the central scheme's dissipation swamps the small pressure differences
(`info.warnings` says so when the largest Mach number of the result is below 0.1; solve such a flow as
incompressible, physics `flow`).

**Boundary conditions as written to OpenFOAM** (justification in the docstring of `_foam_internal.py`).
rhoCentralFoam builds a boundary face's flux from the boundary values of p, T and U alone, so each must be a
state the gas can have.
- `total_inlet`: U `pressureInletOutletVelocity` (the normal velocity from the interior, no tangential
  velocity), p `totalPressure` and T `totalTemperature` with `psi thermo:psi` and `gamma`, i.e. the
  isentropic p = p0 (1 + (gamma - 1) / 2 M^2)^(-gamma / (gamma - 1)), T = T0 / (1 + (gamma - 1) / 2 M^2) at
  the boundary's own Mach number. A subsonic inflow has one outgoing characteristic: the velocity is that one
  extrapolated quantity. (`pressureInletVelocity` would be circular: the flux it reads is made from the
  boundary velocity.)
- `pressure_outlet`: p `mixed` with `refValue` = pb and a per-face `valueFraction` of 1 (impose pb) or 0
  (zero gradient), U `pressureInletOutletVelocity`, T `inletOutlet` (the reservoir's T0 for gas that
  re-enters). v1912 has no stock condition that switches on the Mach number, and `waveTransmissive` relaxes
  p towards pb in both regimes (a subsonic outlet does not sit at pb, a supersonic one still feels it), so
  the solver sets the value fraction itself (`OutletSwitch`): from the state next to the face that the
  boundary has not disturbed (the owner cell of an extrapolating face, the cell behind it of an imposing
  one), with M_n its Mach number normal to the face: impose where M_n < 1, or where M_n >= 1 and pb is at
  least the pressure behind a normal shock at M_n (the shock then moves into the duct); extrapolate
  otherwise (an over- or under-expanded jet adjusts outside the domain).

**Start-up.** The initial field is the gas at rest at the reservoir state (p0, T0), and the back pressure
is applied at once, as when a diaphragm at the exit bursts. The gas already has the reservoir's entropy, so
only pressure waves have to settle. For the first 5 flow times the run stops every 0.25 flow time to update
the outlet switch, so the starting waves leaving the duct supersonically are not held at pb (that drove the
boundary state out of range, a square root of a negative temperature, in the first trials at pb / p0 = 0.05
and 0.4); afterwards it updates after every chunk. The nozzle starts from rest at every back pressure tried
(pb / p0 = 0.05 .. 0.92), so no quasi-1D initial state or back-pressure ramp is needed. `init_from` maps a
previous solution instead (the switch then starts from that field).

**Stopping rule.** As for external flow, with t_ref = L / a* (L: the extent from the inlets to the outlets,
a* = sqrt(2 gamma R T0 / (gamma + 1))) and chunks of 5 flow times. Steady means the field stopped changing
(`steady_tol`), or every QoI held to `qoi_tol` of its scale over the last 3 chunks (p0 for pressures, a* for
velocities, 1 for the Mach number, rho* a* times the smaller of the inlet and outlet areas for mass flows),
and in addition no outlet face switched and the mass flows in and out agree to `mass_tol` (1e-4; to
`qoi_tol` when the QoIs decide, since a limit cycle makes the mass in the duct breathe at that level). The
balance is reported in `info.warnings` ("info: mass balance: ... in, ... out, imbalance ...") and in
`info.mass_balance`; the outlets' state in `info.outlets` and a warning line each. Gas re-entering through a
pressure outlet, gas leaving through a total inlet and an inflow that reaches Mach 1 are warned about.

The limiter is Minmod for internal flow (vanAlbada for external flow). On cd_nozzle_shock_2d at h0 / 2,
vanAlbada keeps a transverse acoustic oscillation in the converging part that never decays (the field
changes by 4e-3 per flow time after 300 flow times, the mass flow by 2e-3 of its value); Minmod leaves a
limit cycle ten times smaller (5e-4 per flow time, mass flow 3e-4) at the same error at h0 (table below,
and the vanAlbada errors at h0: mass flow -0.044 % choked / -0.62 % subsonic, shock +0.16 % of L). The nozzle
benchmarks therefore use `qoi_tol` 1e-3, as the blunted wedge does for its limit cycle.

**Verification: the converging-diverging nozzle** (`cd_nozzle_*_2d`). Upper half of a planar nozzle,
-1 <= x <= 1 m, 0 <= y <= h(x) = 0.1 (1 + 0.5 x^2) m (A / A_t = h / h_t, inlet and exit area ratio 1.5),
symmetry at y = 0, slip wall, inviscid air (gamma 1.4, R 287), p0 = 1e5 Pa, T0 = 300 K. Wall slope at most
0.1, throat radius of curvature 10 h_t: the departure from quasi-1D flow is O(h'^2) ~ 1e-2 locally and the
planar throat's discharge deficit ~2.5e-4 (Kliegel and Levine 1969: (gamma + 1) / 96 (h_t / R_c)^2). Quad
meshes, 80 x 6 cells at h0 (cells nearly square at the throat). The exact solution (area-Mach relation,
isentropic and normal-shock relations, scipy `brentq`) is itself checked against NACA Report 1135's tables
(isentropic M = 0.5, 1, 2, 3; normal shock M = 1.5, 2, 3; choked flux 0.040418 sqrt(T0) / p0) to 1e-4.
Regimes: critical back pressures p1 / p0 = 0.8805, p2 / p0 = 0.6157, p3 / p0 = 0.1602.

1. subsonic, pb / p0 = 0.92 (not choked: mdot = 19.5521 kg/s, Me = 0.3472);
2. shock, pb / p0 = 0.80 (choked: mdot = rho* a* A_t = 23.3356 kg/s, normal shock at x_s = 0.5864 m, M1 =
   1.494);
3. supersonic, pb / p0 = 0.10 (choked, Me = 1.8541, pe / p0 = 0.1602; the same at pb / p0 = 0.05 and 0.4).

Measured (Minmod, one core, the shared 4-core machine under load; errors relative; x_s from
`nozzle.shock_position`, the section-averaged pressure's mid-jump point; profile = the largest deviation of
the section-averaged p (/ p0) and Mach number at 39 stations, 0.1 away from the shock):

COMPRESSIBLE_TABLE

Reading it:
- The choked mass flow is within 0.06 % at every mesh from h0 on; its remaining error is the planar throat's
  discharge deficit (~ -0.03 %) plus the limit cycle (+-0.02 % at h0 / 2), so it does not converge further
  towards the 1D value, as expected.
- The shock position agrees to 0.2 % of L at every mesh (the shock is captured in about two cells; the mid-jump
  point sits within a tenth of a cell of the 1D position, the remainder being the 2D shock's slight curvature).
- The unchoked (subsonic) mass flow is the sensitive one: it follows from p0 - pb = 8 % of p0 and carries the
  scheme's total-pressure loss: SUBSONIC_ORDER.
- The exit pressure of the supersonic regime converges towards a 2D offset of ~0.3 % (the exit flow is not
  parallel: the wall turns it by 5.7 deg), and the exit Mach number with it.
- The result does not depend on the back pressure below p2: pb / p0 = 0.05, 0.10 and 0.40 give the same
  mass flow and exit state to 2e-5 (the outlet extrapolates on every face in all three).

Tolerances of the benchmark (`nozzle.TOL`, relative): mass flow 0.5 % (1 % not choked), exit pressure and
Mach number 1 %, shock position 2 % of L: at least three times the error at h0 / 2 and well above the 2D
effects.

**Limits.**
- Perfect gas only (constant gamma, cp; no real-gas properties, no condensation or two-phase flow).
- Inviscid (`slip`) or laminar walls (`wall`, constant mu); no turbulence, so no shock-boundary-layer
  interaction or separation behind the shock as in a real over-expanded nozzle.
- Mach range: about 0.1 (largest Mach number of the flow) up to the hypersonic range of the external path.
  Inflow through a `total_inlet` must stay subsonic (warned).
- Steady flow only: the run is time-accurate, but only the end state is returned; limit cycles of the
  scheme (above) are averaged by nothing: the QoIs are the last snapshot, within `qoi_tol`.
- Serial, explicit: about 1.4 us per cell per step on one core; a flow time at h0 (480 cells) takes
  ~700 steps, at h0 / 2 ~1500.

Run times (one core, shared machine): RUNTIME_LINE.

**Example.**

```python
from feacore.case import BC, QoI
from feacore.problems.registry import get
from feacore.solver import solve

p = get("cd_nozzle_shock_2d")
case = p.case(p.mesh("nozzle.msh", p.params["h0"]))      # total_inlet, pressure_outlet, slip, symmetry
# case.bcs: [BC("total_inlet", group="inlet", p0=1e5, T0=300.0),
#            BC("pressure_outlet", group="outlet", value=8e4), BC("slip", group="wall"),
#            BC("symmetry", group="axis")]; case.qois: mass_flow (outlet), p_exit, mach_exit
res = solve(case, backend="openfoam")
res.evaluate_all()          # {"mass_flow": 23.32, "p_exit": 8e4, "mach_exit": 0.472}
res.info.mass_balance       # {"in": ..., "out": ..., "imbalance": ...}
from feacore.problems.compressible.nozzle import shock_position
shock_position(res, p)      # 0.588 (exact 0.5864)
```
