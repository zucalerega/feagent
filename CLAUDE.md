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

- `main` has the mesh harness (package `feagent` in `src/`: STEP → YAML mesh spec → gmsh mesh with named
  groups, quality gates, `report.json` with the spec's `bcs:`), the fixed solver, verification,
  benchmarks and UI (`feacore/`), and the diagnostics (`feacore/diagnostics/`). One root
  `pyproject.toml` ships both packages.
- `mesh-harness` branch: Alexander's ongoing harness work; merged into `main` as it lands.
- `improver-loop` branch: improver side of the loop.
- `ace-loop` branch: minimal ACE improver loop (package `ace/`, see `ace/README.md`) on top of
  `feacore`: Sonnet agent → node-capped accuracy score → Opus reflector → curator → playbook.
- `ace-cfd` branch (off `ace-loop`, with `main` merged in): the same ACE loop on steady laminar CFD.
  - Run it with `python -m ace.run_loop --suite cfd`.
  - Flow tools: finite-volume mesh with optional quads; solve with fluid, convection scheme and flow BCs on the openfoam backend.
  - Oracles: spec compliance, gaming, known answer, and a refinement study on the openfoam backend.
  - Two new exact benchmarks: `taylor_couette_2d` (rotating cylinder) and `square_duct_3d` (developing duct flow).
  - The tasks are 2D only, because of the 3D unstructured-mesh bias listed under the known solver limits.
  - Without a local OpenFOAM: `scripts/openfoam_docker.sh` (OpenFOAM v1912 in Docker, plus shims) and `FEACORE_OPENFOAM_BASHRC=~/.feacore_foam/bashrc TMPDIR=~/.feacore_foam/tmp`.
- `diagnostics` branch (off `main`): `feacore/diagnostics/` on top of the fixed skfem solver:
  deterministic failure-mode flags (a precheck before solving plus post-solve checks), reference-free
  accuracy (ZZ estimate, a refinement study of the agent's own setup, P1 vs P2) and the speed / accuracy
  score (`docs/failure-modes.md`). `scripts/nafems_task.py` / `scripts/compare_nafems.py` give the agent
  prompt for a NAFEMS benchmark and score the reply against the reference. LE10 comes first, with
  `benchmarks/nafems_le10/public/mesh.msh`.
- `cfd-openfoam` branch (PR #3): OpenFOAM backend and CFD benchmarks, in progress.
- `benchmarks` branch (PR #13): `feabench/` plus `benchmarks/{verification,code_aster,calculix,simjeb}`
  and `docs/eval-design.md`. It was built before `feacore` landed, and its ccx-deck evaluator and
  `model.inp` contract are superseded by the decisions below (skfem, case JSON + `.msh`).
  Still useful:
  - code_aster reference values with locations;
  - the CalculiX deck corpus;
  - the SimJEB loader (needs `dataverse.harvard.edu` allowed);
  - the scoring proposal in `docs/eval-design.md`.
  Next: port the verification cases and QoIs to `feacore` Problems.
- `gold-benchmarks` branch (Jaime): the gold-benchmark generator `feacore.goldgen`. It takes ABC
  CAD parts, generates BC sets, meshes nested tet10 levels L0-L2 and keeps the QoIs that the GCI
  certifies; it runs on an H100 and packs its output into numbered archives
  (`goldgen-l2_0000.tgz`, ...). `benchmarks/gold/` holds the imported sets (see FEA core). Its
  solver fixes for curved tet10 are ported to main byte-identical: a robust inverse isoparametric
  map, curved-cell validity judged by the Jacobian, and a reduced SVD in the rigid-body checks.
  Without them skfem rejects some of these meshes.

## FEA core (`feacore/`, branch `fea-solver-benchmarks`)

Decisions:

- **Fixed solver** = scikit-fem backend (`backend="skfem"`, CPU). The linear solve is pluggable: SciPy /
  MKL PARDISO on CPU, NVIDIA cuDSS (via nvmath-python) on a GPU when one is present. Both are direct
  solves, so the answer doesn't depend on the hardware.
- **Cross-check solver** = CalculiX (`backend="ccx"`), independent of the fixed solver. On the same
  case and mesh the two agree to about 1e-5 (displacements, reactions, energies) and about 1e-3 for
  boundary point stresses (different stress recovery). A bigger disagreement means a setup or solver bug.
  The ccx backend runs plane stress on an internal thin section, so CalculiX's 3D-slab expansion no
  longer distorts thick plates. It refuses (NotImplementedError) axisymmetric order 1 (CAX3) and
  quadratic tets / CAX6 with curved interior edges, where CalculiX's own elements aren't exact. gmsh
  meshes only curve boundary edges, so they're accepted with a warning. Axisymmetric ccx results are
  good to about 1e-4 relative.
- **Mesh sequences for GCI:** independently remeshed unstructured sequences (what agents produce)
  scatter the observed order by a few tenths, so treat a 3-mesh GCI from them as approximate and use
  4+ meshes (the least-squares fit in `analyze`) when it matters. Nested refinement
  (`convergence_study(..., nested=True)`, i.e. gmsh `mesh.refine()` splitting) gives clean orders for
  global norms. Point QoIs can still be irregular on nested meshes.
- **Neural surrogate** (`backend="nn"`, optional, GPU): fast but unverified. It is never the reference;
  the verification layer judges it like any other submission.
- **Solver input contract** = a case JSON (`feacore/case.py`: analysis, materials, BCs, loads, QoIs)
  plus a gmsh `.msh` file whose physical groups are named. This is what the agent's MCP / mesh harness
  must produce. Run it with `feacore.solver.solve(case, backend)` or the CLI
  `python -m feacore.cli case.json --out result.json`.
- **Mesh harness → solver:** `feacore/adapters/mesh_harness.py` turns a harness output directory or its
  YAML spec into a case: `python -m feacore.adapters.mesh_harness <out_dir | mesh.yaml>`. The spec's
  `bcs.fea` maps 1:1:
  - `fixed`, `loads: [{group, force}]` (total force spread per area);
  - `symmetry: [group]`, `pressure: [{group, value}]`;
  - `displacement: [{group, components, value, where}]`, where `where` filters nodes, e.g. a line on a
    face for LE10's mid-thickness support.

  `E` and `nu` give the material.
- **Problems** (the "CAD design problems"): `feacore/problems/` (the `Problem` API + registry),
  exported to `benchmarks/<id>/public/` (statement + STEP: give these to agents) and
  `benchmarks/<id>/private/` (reference values, reference setup, validation studies: evaluator only,
  never give these to agents).
- **Gold benchmarks** (`benchmarks/gold/`, `feacore/problems/gold.py`, `benchmarks/gold/README.md`):
  - Import generator archives with `python scripts/gold.py import ARCHIVE`. It checks every set,
    stores one STEP blob per part, and lists the meshes with their sha256 in `index.json`.
  - Each set registers as a `GoldProblem` (tag `gold`, family = part id) read straight from its
    files. The UI and the ACE loop see it with no export step, and `export_all` skips it.
  - Groups are `f<tag>` (the STEP's OCC face tags) and `solid`.
  - The study meshes (`benchmarks/gold/_meshes/`) are not in git.
  - `scripts/gold.py verify` re-solves a stored level against the study. `scripts/gold.py render`
    records runs for the UI viewer, and `--png` writes images.
- **Verification** = Richardson extrapolation / Grid Convergence Index (Celik et al. 2008),
  `feacore/verification/`. Proposed scoring input for the evaluator: QoI error vs the reference, plus
  self-consistency from mesh refinement (observed order, GCI band, and divergence flags that point to
  singularities or setup errors). The evaluator owns the final rubric.
- Setup on a fresh Ubuntu / Brev box / DGX Spark: `bash scripts/setup_gpu_box.sh` (system libs,
  CalculiX, a Python >= 3.12 venv, and GPU extras when `nvidia-smi` works). Brev walkthrough (shell,
  GPU check, validation, UI tunnel on port 8000): `docs/brev.md`. Check the GPU solve with
  `python scripts/gpu_check.py` before relying on cuDSS.

- **UI + solver service:** `python -m feacore.ui --host 0.0.0.0 --port 8000` (`docs/ui.md`). The
  harness can call `POST /api/solve` (multipart case JSON + .msh) or write run, study and submission
  records with `feacore.runstore.RunStore` (`record_result`, `create_submission`). Records live under
  `runs/` (`FEACORE_RUNS_DIR`) and are never committed. A submission is one agent's runs on 3+ of its own
  meshes; the consensus view uses the median and MAD of each submission's best estimate. For display
  only, the UI calls a QoI converged when both mesh-to-mesh changes are under 0.1%, even if `gci3`
  can't assign an order. Verdicts still come from `feacore.verification`.
- **Running it for the team (Brev):** `bash scripts/serve.sh` installs the UI/solver service (systemd,
  starts on boot) plus an auto-update timer. Every minute it fast-forwards the checked-out branch and
  restarts the service, re-runs setup when dependencies change, and waits while jobs are running.
  Set `FEACORE_UI_TOKEN` (or start the UI with `--token auto`) before sharing a tunnel URL. Every
  `/api/*` call then needs the token. Teammates' scripts and agents call `POST <url>/api/solve`;
  the UI's Connect page shows copy-paste examples. Details are in `docs/brev.md`; any account's box is set up
  with one command (`curl -fsSL https://raw.githubusercontent.com/zucalerega/feagent/main/scripts/bootstrap.sh | bash`),
  which a Brev Launchable can also run as its setup script. Expose port 8000 via the instance's HTTP Ports.
- The UI shows small trendless mesh-to-mesh changes (under 2%) as "scatter between meshes", not as
  divergence. Real singularities still show as diverging.
- **CFD (`backend="openfoam"`, physics "flow"):** OpenFOAM ESI (v1912 from Ubuntu's `openfoam`
  package works) plus `foamlib` (the `cfd` extra). It runs 2D/3D laminar steady SIMPLEC with tight
  residuals; forces and the probes are computed in numpy, because v1912's function objects crash.
  Benchmarks: Poiseuille channel, lid-driven cavity Re 100, DFG 2D-1 cylinder, 3D pipe
  (`docs/benchmarks_cfd.md`). Use quad/hex meshes for GCI: on unstructured triangles OpenFOAM's cell
  values stop converging. Ghia's cavity table is itself 1.4-3.5% off converged solutions, so its
  tolerances are 3-5%. Pressure drops use the `point_difference` QoI. To enable it on a box:
  `sudo apt-get install -y openfoam && uv pip install --python .venv/bin/python foamlib`.
- Known solver/verification limits (from the benchmark runs):
  - 3D point stresses at mesh vertices scatter 0.1-0.5% between unstructured tet meshes, in both
    backends. GCI on such QoIs reads as oscillatory or divergent, so judge them against the reference
    tolerance, or use averaged / integrated QoIs.
  - The stress recovery is a global continuous L2 projection, so it smears stress jumps at material
    or thermal-load interfaces. Don't use point stresses right at an interface.
  - Cook's membrane: the literature 23.96 is uy at the loaded-edge midpoint (48, 52), not the corner.
  - 3D flow on unstructured meshes looks biased in the openfoam backend (found on `ace-cfd`). On tets of the
    3D pipe (4.6k nodes) the axial pressure gradient is +15 %, while the wall force is +3 % and the flow rate
    +0.6 %. Tets split into hexes are still +12 % at 33k nodes, and the square duct on tets is +12 %. So the
    pressure force and the wall shear do not balance. 2D triangles are as accurate as quads, and structured
    hexes converge. Not investigated yet; until then, judge 3D flow only on structured hex meshes.

## NVIDIA compute (hackathon)

- Provided (per the ScienceClaw NVIDIA docs index): Brev GPU instances (credits), hosted model
  endpoints on build.nvidia.com (e.g. Nemotron, API key), and Dell GB10 "DGX Spark" boxes for local
  LLMs / NIMs. The doc contents themselves are not yet in this repo.
- **DGX Spark is aarch64 (Grace CPU).** PyPI has no Linux ARM wheel for `gmsh` (only macOS arm64), so
  `pip install gmsh` fails there. That affects the mesh harness too. Use conda-forge (`gmsh`,
  `python-gmsh`), as `scripts/setup_gpu_box.sh` does. MKL PARDISO is x86-only. CuPy, nvmath-python
  (cuDSS) and torch do have aarch64 wheels.

## Open questions

- How exactly is a solution scored? (proposal above; the evaluator decides)
- Which MCP tools does the agent get (geometry / mesh / BCs)? It must emit the case JSON + `.msh` above.
- What exactly can an improver change: system prompt, tool descriptions, few-shot examples, harness code?
- What does the evaluator output look like (schema for rankings + summary)?

## Conventions

- Branch off `main` per workstream; don't push directly to someone else's branch.
- Update this file when a decision above gets made.
