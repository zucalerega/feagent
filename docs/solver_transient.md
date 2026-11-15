# Time-dependent and modal analyses in the fixed solver

The fixed solver (`feacore`) was steady and linear. This document is the shared definition of the analyses
that go beyond that: what a case looks like, what a result looks like, and how each one is verified. The
case format is in the module docstring of `feacore/case.py` ("Time-dependent and modal analyses"); this file
adds the conventions that hold across physics and one section per capability.

| capability | physics, backend | module | status |
|---|---|---|---|
| transient heat conduction | heat, skfem | `feacore/solver/_skfem_transient.py` | done; verified (closed forms, orders, NAFEMS T3), `tests/test_transient_heat.py` |
| modal analysis | elasticity, skfem | `feacore/solver/_skfem_modal.py` | see section |
| transient dynamics | elasticity, skfem | `feacore/solver/_skfem_dynamics.py` | see section |
| transient incompressible flow | flow, openfoam | `feacore/solver/_foam_transient.py` | see section |
| turbulent flow (RANS) | flow, openfoam | `feacore/solver/openfoam_backend.py` | see section |
| compressible internal flow | compressible, openfoam | `feacore/solver/_foam_compressible.py` | see section |

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

(section written by the capability's author)

## Transient dynamics

(section written by the capability's author)

## Transient incompressible flow

(section written by the capability's author)

## Turbulent flow (RANS)

(section written by the capability's author)

## Compressible internal flow

(section written by the capability's author)
