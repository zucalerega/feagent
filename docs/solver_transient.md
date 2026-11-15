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

(section written by the capability's author)

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
