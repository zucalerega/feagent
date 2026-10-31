# CLAUDE.md

Shared context for anyone (human or agent) working in this repo.

## What this is

ScienceClaw hackathon project. A population of (agent, improver) pairs that get better over time at
setting up FEA problems from CAD designs.

## Architecture

- **Agent**: assigned at random to one CAD design problem (many agents per problem). It uses an MCP to
  set up geometry, mesh it, and apply boundary conditions. The solver is fixed, so agents don't touch it.
- **Evaluator**: a separate agent that reviews every solution. It ranks each agent across key areas,
  scores the population as a whole, and writes a summary of what worked, what failed, and the good ideas.
- **Improver**: paired 1:1 with an agent. It reads the evaluator's summary and updates its agent's
  context + harness so the agent does better on future problems.

One train iteration: agents solve → evaluator ranks and summarizes → improvers update their agents.

## Status

- `improver-loop` branch: improver side of the loop.
- `mesh-harness` branch: the meshing harness (package `feagent` in `src/`): STEP → YAML mesh spec →
  gmsh mesh with named groups and quality gates. It declares BCs for the solver but does not solve.
- `fea-solver-benchmarks` branch: the fixed solver, solution verification, benchmark problems, test
  parts and a UI (package `feacore/`, see below). In progress: interfaces are in place, the solver
  backends and problems are being built.

## FEA core (`feacore/`, branch `fea-solver-benchmarks`)

Decisions:

- **Fixed solver** = scikit-fem backend (`backend="skfem"`, CPU). The linear solve is pluggable: SciPy /
  MKL PARDISO on CPU, NVIDIA cuDSS (via nvmath-python) on a GPU when one is present. Both are direct
  solves, so the answer doesn't depend on the hardware.
- **Cross-check solver** = CalculiX (`backend="ccx"`), independent of the fixed solver.
- **Neural surrogate** (`backend="nn"`, optional, GPU): fast but unverified. It is never the reference;
  the verification layer judges it like any other submission.
- **Solver input contract** = a case JSON (`feacore/case.py`: analysis, materials, BCs, loads, QoIs)
  plus a gmsh `.msh` file whose physical groups are named. This is what the agent's MCP / mesh harness
  must produce. Run it with `feacore.solver.solve(case, backend)` or the CLI
  `python -m feacore.cli case.json --out result.json`.
- **Mesh harness → solver:** `feacore/adapters/mesh_harness.py` turns a harness output directory
  (`mesh.msh` + `manifest.json` + `report.json` with `bcs.fea: {fixed, loads: [{group, force}], E, nu}`)
  into a case: `python -m feacore.adapters.mesh_harness <out_dir>`. `fixed` → `fixed` and
  `loads[].force` → `force` (total force spread per area) map 1:1. The case schema also has pressure,
  symmetry, prescribed displacement, thermal loads and explicit QoIs if the harness wants to declare them.
- **Problems** (the "CAD design problems"): `feacore/problems/` (the `Problem` API + registry),
  exported to `benchmarks/<id>/public/` (statement + STEP: give these to agents) and
  `benchmarks/<id>/private/` (reference values, reference setup, validation studies: evaluator only,
  never give these to agents).
- **Verification** = Richardson extrapolation / Grid Convergence Index (Celik et al. 2008),
  `feacore/verification/`. Proposed scoring input for the evaluator: QoI error vs the reference, plus
  self-consistency from mesh refinement (observed order, GCI band, and divergence flags that point to
  singularities or setup errors). The evaluator owns the final rubric.
- Setup on a fresh Ubuntu / Brev box: `apt-get install -y libglu1-mesa libxcursor1 libxft2
  libxinerama1 libxrender1 libxi6 libgl1 calculix-ccx` and `pip install -e ".[dev]"`.

## Open questions

- How exactly is a solution scored? (proposal above; the evaluator decides)
- Which MCP tools does the agent get (geometry / mesh / BCs)? It must emit the case JSON + `.msh` above.
- Merging into `main`: `mesh-harness` and `fea-solver-benchmarks` both add a root `pyproject.toml`
  (`feagent` with uv_build, `feacore` with setuptools). Pick one root project that ships both
  packages, or make `feacore` a sub-project.
- What exactly can an improver change: system prompt, tool descriptions, few-shot examples, harness code?
- What does the evaluator output look like (schema for rankings + summary)?

## Conventions

- Branch off `main` per workstream; don't push directly to someone else's branch.
- Update this file when a decision above gets made.
