# CLAUDE.md

Shared context for anyone (human or agent) working in this repo.

## What this is

ScienceClaw hackathon project, consolidated afterwards into one branch. A swarm of Claude agents learns
to set up finite-element (and flow) simulations from CAD: each agent meshes real parts, a fixed solver
scores the result against answers the agents never see, and every round a teacher, the agents' own
reflections and an organizer rewrite each agent's notebook of lessons. No model weights change. The
README tells the story and the results; this file records where things are and the decisions behind them.

Everything is on `main`. The dashboard, `python -m feacore.ui`, is the single entry point for running it.

## Architecture (as built)

The original plan had three roles: agent, evaluator and improver. In `ace/` they became:

- **Agent** (`ace/agent.py`, Claude Sonnet): one persistent agent per notebook, assigned to a task. Its tools
  are `inspect_geometry`, `mesh(h, order, refinements)`, `solve`, `view`, `compare`, `submit`, plus
  `nonlinear_check` on the nonlinear task and the flow tools on CFD tasks. The solver is fixed; agents never
  touch it. Budgets of solves and turns per task (`ace/config.py` `TASKS`).
- **Evaluator**: deterministic scoring with no LLM (`ace/checks.py`): field mode against a fine reference,
  hidden probe points, x the gmsh automesher, plus the "be the best" and "be different" bonuses. Then a
  teacher (`ace/deep_teacher.py`, Claude Opus) reviews every run and writes a critique and the next
  experiment per agent (control / variant / explorer roles).
- **Improver**: each agent's self-reflection (`ace/self_reflect.py`) and an organizer (`ace/organizer.py`)
  that merges and rewrites the private notes and the shared playbook; `ace/process_brief.py` builds each
  agent's next context.

One generation: agents solve → score → teacher + self-reflection → organizer rewrites the notebooks.
`python -m ace.run_loop --mock` runs the whole loop for free with canned LLM responses.

The mesh harness in `meshing-harness/` is the other agent interface: an MCP server, skill and Stop hook
that let a Claude Code agent mesh raw STEP through a YAML spec with quality gates. The ACE agents use
`ace/fea.py`'s gmsh tools instead; `ACE_HARNESS=1` adds the harness's geometry digest and mesh gates to
their 3D tool results (`ace/harness.py`).

## Repository layout

| Path | What |
|---|---|
| `ace/` | The learning loop (swarm and single agent), scoring, teacher, reflection, organizer, memory, export / inference, live tracker, CFD and nonlinear tasks. `ace/README.md` |
| `feacore/` | Fixed solver (scikit-fem), CalculiX cross-check and nonlinear path, OpenFOAM flow, verification (GCI, MMS, ZZ), diagnostics, problem registry, gold generator (`goldgen/`), run store, and the dashboard (`ui/`) |
| `meshing-harness/` | Package `feagent`: STEP → YAML mesh spec → named, quality-gated gmsh mesh; MCP server `feagent-mcp`, `fe-mesh` skill, Stop hook, own tests. `meshing-harness/README.md` |
| `benchmarks/` | Problem exports (`public/` for agents, `private/` for the evaluator), the gold sets (`benchmarks/gold/`), and the verification / code_aster / CalculiX / SimJEB suites |
| `feabench/` | Benchmark-suite runner (`feabench` CLI) for the verification / code_aster / CalculiX suites |
| `physical_tests/` | Real test data: the Instron compression test of the printed C-bracket, and the printable pull-test parts with their test plan |
| `experiments/` | Reproducible studies: mesh systems, crush to fracture, neck break, material fit, wingbox, DGX Spark scripts. `experiments/README.md` |
| `results/`, `talk_plots/`, `docs/gallery/` | Run summaries and exported trained swarms (`results/trained_swarms/`), every presentation figure with its script, the image/video gallery |
| `wsp/` | Teaching a small open model to mesh (weight-space programming, LoRA distillation). `wsp/README.md` |
| `single_agent/` | One analyst agent per arm (Opus or Sonnet) from component design data to a verified analysis, with a deterministic scorecard and a learned rulebook. `single_agent/README.md` |
| `teams/` | Sculptured team definitions (wingbox; parallel mesh systems → synthesis) |
| `scripts/` | Setup (`setup_gpu_box.sh`, `bootstrap.sh`, `serve.sh`), the gold pipeline, nonlinear runs and videos, figures |
| `docs/` | Solver, benchmark, UI and Brev docs |
| `tests/` | The root test suite (`meshing-harness/tests/` has the harness's) |

Never committed: `runs/` (the run store), `ace_runs/`, `swarms/`, `benchmarks/gold/_meshes/`, `wsp_runs/`,
`sa_runs/`, `sa_workspaces/`.

## The dashboard (`feacore/ui/`, `docs/ui.md`)

`python -m feacore.ui --port 8000` (add `--host 0.0.0.0 --token auto` to share it). Sections:

- **Benchmarks**: every registered problem (statement, 3D CAD preview, evaluator view), convergence studies, batches.
- **Gold**: the gold sets and their nested study meshes: local status against `index.json`, regeneration with
  `feacore.goldgen` (gmsh is not bitwise reproducible, so a regenerated mesh never matches the stored sha256; it
  is fine to view and solve), verify / render / import, *Solve L<k>* into the viewer.
- **Your CAD**: upload a STEP (`feacore/problems/custom.py`, `feacore/ui/cad_api.py`). Each upload is a mini gold
  root under `<runs>/cad/<upload>/`; its BC sets (picked on the 3D view, or generated with the gold rules) are
  registered problems `<upload>_bc<k>` / `<upload>_m<k>`, tag `custom`, found by every process whose
  `FEACORE_RUNS_DIR` (or `FEACORE_CAD_DIR`) points there, with an `ace.config.TASKS` entry. `ace.infer` works on
  them reference-free (ZZ, agreement between agents); the training loops need a certified reference first (the
  page's reference study, or adopting a finished P2 study).
- **Agents**: train a swarm or a single agent, watch it live (the `ace.watch` tracker), stop it, browse runs,
  compare swarm vs single, export a trained swarm and run it on any problem (`ace.infer`). `#/swarm` is the live
  per-solve view fed by the run-store bridge (`ace/ui_bridge.py`). The Train tab also runs the single analyst
  agent (`single_agent`: loop, oracle, comparison; tools in `feacore/ui/tools/single_agent.py`).
- **Nonlinear**: the agents' scoring ramp, the elastoplastic ramp and crush to fracture (with *Resume* from a
  checkpoint), each run's force-travel curve over the Instron test (or an uploaded one) with the scoring error,
  renders to stills and videos, and the committed gallery. Outputs: `<runs>/nonlinear/<kind>/<tool>-<task>/`.
- **Test data**: the physical tests and printable parts; a machine export (CSV / TXT, read by
  `ace.evidence.read_table`) is checked and saved as `<runs>/testdata/<id>/`, a ready `--evidence` folder.
- **Runs**, **Tasks**, **Solve**, **Connect** (API examples), **Health**.

Long jobs run on the **task lane** (`feacore/ui/tasks.py`): a tool is a whitelisted command with typed
parameters, registered by a module in `feacore/ui/tools/`; the browser only names a tool and fills its form.
To expose a new command, add a `Tool` there (and a page section if it needs one). Tasks get
`FEACORE_RUNS_DIR` = the server's run store, and `/api/health` counts them as active jobs. A tool that renders
with pyvista / VTK sets `display=True`: VTK segfaults (it does not raise) without an OpenGL context, so such tools
start through `feacore/ui/offscreen.py` (`xvfb-run` when the server has no display). Run renders by hand on a
headless box the same way: `xvfb-run -a python ...`.

## What's where (features and how to run them)

- **Swarm training** (`ace/run_loop.py`): `--deep` (run reports, teacher, self-reflection, organizer),
  `--memory shared+private --agents N --split`, `--personas` / `--persona-count`, `--suite fea|cfd|cfd3d`,
  `--statement requirements` (engineering requirements instead of BCs; `ace/requirements.py`,
  `ace/requirements_data/`), `--evidence` / `--evidence-score` (physical test data; `ace/evidence.py`),
  `--init-from <exported swarm> --init-agents ... --pre-teach` (start from a trained swarm). Stop cleanly with
  `<out>/STOP`. `ACE_SCORE=field` selects field scoring.
- **Single-agent baseline** (`ace/single_loop.py`) and the comparison `ace/compare.py`.
- **Trained swarms**: `ace.export_swarm` saves one; `ace.infer` runs it on any problem without learning.
  Committed swarms: `results/trained_swarms/` (`div_s1` … `div_s16`, `nl_ft`).
- **Watch / report**: `ace.watch <run>` (live tracker, also inside the dashboard), `ace.report <run>` (one HTML page).
- **Physical tests**: `abc_00000065_compression` (the printed C-bracket crushed on an Instron, no reference:
  always scored against the test; `physical_tests/abc_00000065_compression/`, `ace/nonlinear.py`) and
  `abc_00000003_pull` (a planned pull test, defined as a problem, data not in the repo;
  `feacore/problems/physical.py`). Printable parts: `physical_tests/parts/`, `scripts/physical_parts.py`.
- **Nonlinear analysis**: feacore's elastoplastic CalculiX path (J2 hardening, NLGEOM) and opt-in progressive
  failure by element erosion (`feacore/solver/_ccx_nl.py`, `_ccx_failure.py`; `docs/solver_ccx_nonlinear.md`).
  Drivers: `scripts/ccx_compression_ramp.py` (the frozen ramp the agents are scored with),
  `scripts/ccx_crush.py`, `experiments/physical/crush.py` (platen, contact, failure criteria, resume),
  the six-mesh-system study (`experiments/mesh_systems/`, `experiments/physical/CRUSH_RUNS.md`), videos
  (`scripts/nl_video.py`, `nl_story_video.py`, `nl_break.py`, `experiments/physical/render_*.py`).
- **CFD on the ACE loop** (`--suite cfd` / `cfd3d`): flow mesher (`ace/flow_mesh.py`: agents mesh the plain CAD,
  wall `boundary_layer`, 3D `sweep`), finite-volume quality gate (`feacore/geometry/fv_quality.py`), flow feedback
  and field mode (`ace/flow_feedback.py`, `ace/flow_field.py`, `ace/flow_report.py`), oracles, the exact benchmarks
  `taylor_couette_2d` and `square_duct_3d`, and `backward_facing_step_2d` (scored on `u_bubble`). Real runs:
  `bash scripts/setup_ace_cfd.sh`, then `python -m ace.preflight --suite cfd --ping`. Without a local OpenFOAM:
  `scripts/openfoam_docker.sh`. `--teacher inspect|both`, `--statement requirements` and `--pre-teach` are
  elasticity-only.
- **Supersonic flow**: compressible OpenFOAM (`rhoCentralFoam`) and the blunted-wedge ground-truth case
  (FEA core, compressible bullet).
- **Gold benchmarks**: generator `feacore.goldgen` (H100), imported sets in `benchmarks/gold/`, J2-plastic gold
  sets from the linear meshes (`scripts/gold_nonlinear.py`). FEA core, gold bullet.
- **Diagnostics** (`feacore/diagnostics/`): failure-mode flags before and after solving, reference-free accuracy
  (ZZ, a refinement study of the agent's own setup, P1 vs P2), the speed / accuracy score
  (`docs/failure-modes.md`). `scripts/nafems_task.py` / `scripts/compare_nafems.py` score a single reply.
- **Benchmark suites**: `feabench/` with `benchmarks/{verification,code_aster,calculix,simjeb}` and
  `docs/eval-design.md` (built before `feacore`; its ccx-deck contract is superseded by the case JSON + `.msh`).
  The SimJEB loader needs `dataverse.harvard.edu` allowed.
- **Open-model study** (`wsp/`): a teacher (Claude Opus) teaches `Qwen/Qwen3.5-0.8B` (CPU fp32) one-shot mesh
  recipes, scored like the swarm; weight-space programming, LoRA distillation, RFT, best-of-k by ZZ, the
  playbook. `bash scripts/wsp_study.sh`; huggingface.co must be reachable for the model.
- **Single analyst agent** (`single_agent/`, `single_agent/README.md`): ONE agent per arm (Opus or Sonnet) does
  the whole analysis layer from component design data. Design packages (component, system context, review notes,
  questions; faces described by feature, never by tag) for the 9 gold BC sets; a blind harness (STEP + package
  only: the ace geometry/feedback helpers would leak the gold statement's supports); a deterministic scorecard
  (setup equivalence on the agent's mesh, answers vs gold, calibration of stated uncertainty, singular peaks,
  study); a rulebook = a general handbook + rules the arm learns from its training scorecards (holdout never
  reflected on). `python -m single_agent.loop --mock` is free; `single_agent.oracle` is the non-LLM baseline and
  `single_agent.compare` compares arms. Pilot (Claude Code subagents through `single_agent.cli`, holdout
  `abc_00000065_bc1`, handbook only): Opus and Sonnet both scored 1.00; Opus was more precise and also found a
  converged ring-root stress above yield that fails the part (`single_agent/pilots/`). The API loop has not run
  on real models yet. Also in the dashboard (Agents section).
- **Sculptured teams** (`teams/`): the wingbox team and the parallel-mesh-systems → synthesis team.
- `make_problem.py`: prints a unique, seeded problem statement per (agent number, call number).

## FEA core decisions (`feacore/`)

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
  - `abc_00000003_bc3` is retired: its references were 12 % under the FE energy bound. Cause: the part's nested
    L1/L2 meshes fold into the bore, because splitting snapped vertices of the coarse base's 1 mm ring f24 onto
    its plane but outside the ring. goldgen now refines such faces before splitting (`mesh._base_mesh`), rejects
    levels with nodes off their faces (`mesh._face_leaks`), and certifies nothing when the strain energy falls
    between levels (`run.energy_drop`, also an import check). The sibling sets stay within tolerance, but bc0's
    and bc7's references lean up to ~1.4 % low. Regenerating abc_00000003 with the fixed generator (GPU box for
    L2) fixes them and can restore bc3 (`benchmarks/gold/README.md`, "Retired").
  - Plane outward normals are probed at points on the face (`goldgen.parts.face_points`, shared with uploaded
    CAD), not at the centroid, which for a ring lies in its hole and had stated abc_00000003's f18/f19 inward.
    `scripts/gold.py restate` rewrote those words (forces carry explicit vectors); `restate --check` finds none.
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
  3D flow uses skew-consistent schemes (option `skew_correction`, on by default in 3D):
  `pointCellsLeastSquares` gradients (but `grad(p)` stays Gauss linear) and a `skewCorrected linear`
  `flux(HbyA)`. With the plain schemes, tets converged to a pressure gradient 12-15% too high at
  every h: next to a wall, linear interpolation misses the face values in the same direction
  everywhere. Wall forces use the wall-normal distance, as the solver does.
  `docs/solver_openfoam.md`, "3D meshes".
- **Compressible flow (`backend="openfoam"`, physics "compressible"):** a perfect gas (`Material.gamma`,
  `Material.gas_constant`; `mu = 0` is inviscid), solved by `rhoCentralFoam`
  (`feacore/solver/_foam_compressible.py`, `docs/solver_openfoam.md`).
  - BCs: `freestream` (velocity, `p_inf`, `T_inf`), `outlet`, `wall`, `slip`, `symmetry`; supersonic
    flow only.
  - Forces use the gauge pressure p - p_inf.
  - It runs from the freestream until the field, or every QoI, stops changing. The QoI rule exists
    because limited central schemes keep small limit cycles at a blunt body's sonic point.
  - Defaults: vanAlbada at Courant 0.2. vanLeer cycled at every Courant number.
  - Ground-truth case: `supersonic_blunt_wedge_2d` (`feacore/problems/supersonic/`), a cylinder-wedge at
    Mach 3 and 10 deg incidence. Its references are mesh-converged (`scripts/supersonic_blunt_study.py`).
    At alpha = 0 it is validated against the exact pitot pressure, and its standoff against Billig
    (`docs/benchmarks_cfd.md`).
- Known solver/verification limits (from the benchmark runs):
  - 3D point stresses at mesh vertices scatter 0.1-0.5% between unstructured tet meshes, in both
    backends. GCI on such QoIs reads as oscillatory or divergent, so judge them against the reference
    tolerance, or use averaged / integrated QoIs.
  - The stress recovery is a global continuous L2 projection, so it smears stress jumps at material
    or thermal-load interfaces. Don't use point stresses right at an interface.
  - Cook's membrane: the literature 23.96 is uy at the loaded-edge midpoint (48, 52), not the corner.
  - `feacore.verification.gci.gci3` classifies convergence by R = e21 / e32 alone, which is exact only for equal
    refinement ratios: with unequal ratios a converging QoI can read divergent, or a stalling one monotone.
    `single_agent/harness.py` (`gci_unequal`) has a ratio-corrected test; `gci3` itself is unchanged.
  - 3D flow on tets: fixed by the skew-consistent schemes (CFD bullet above). The old +15 % pressure gradient
    on tets was a solver bias. The apparent force imbalance came from the wall force dividing by |Cf - C|
    instead of the wall-normal distance. Point velocities at a smooth maximum still read 2-5 % low on coarse
    tets (the probe's vertex fit); pressure drops, forces and flow rates are fine.

## NVIDIA compute (hackathon)

- Provided (per the ScienceClaw NVIDIA docs index): Brev GPU instances (credits), hosted model
  endpoints on build.nvidia.com (e.g. Nemotron, API key), and Dell GB10 "DGX Spark" boxes for local
  LLMs / NIMs. The doc contents themselves are not yet in this repo.
- **DGX Spark is aarch64 (Grace CPU).** PyPI has no Linux ARM wheel for `gmsh` (only macOS arm64), so
  `pip install gmsh` fails there. That affects the mesh harness too. Use conda-forge (`gmsh`,
  `python-gmsh`), as `scripts/setup_gpu_box.sh` does. MKL PARDISO is x86-only. CuPy, nvmath-python
  (cuDSS) and torch do have aarch64 wheels.

## Mesh harness decisions (`meshing-harness/`)

- **Agent MCP tools** (`meshing-harness/`, server `uv run feagent-mcp`): `automesh` (zero-config start),
  `inspect_geometry` (digest: features, faces, hazards), `find_faces` (geometric face query with
  expected count), `render_view`, `mesh`, `check_mesh`; prompt `meshing_workflow`. Geometry, mesh and BCs
  are all declared in `mesh.yaml` (`groups:` by query, `bcs:` for fixed/loads or inlet/outlet).
- **Solver hand-off:** `out/mesh.msh` (named physical groups) + `out/report.json` → `bcs`, turned into
  the fixed solver's case JSON by `python -m feacore.adapters.mesh_harness <out_dir | mesh.yaml>` (CFD:
  `feacore.adapters.mesh_harness_cfd`). Also `out/mesh.inp` (every group as NSET/ELSET; surface elements
  for groups must not get a section) and `.su2`.
- **Evaluator input:** `out/report.json` (`schema_version: 1`, per-gate pass/warn/fail with value and
  threshold) — schema in `meshing-harness/docs/report-schema.md`.
- **Improver-editable surface:** `meshing-harness/agent_context/` — `SKILL.md` (workflow prompt),
  `tools.yaml` (MCP tool descriptions), `examples/` (few-shot specs). Give each agent its own copy via
  `FEAGENT_CONTEXT_DIR`. Harness code changes are possible but not required for prompt-level improvement.

## Open questions

- Scoring is settled for the ACE loop (`ace/checks.py`: field mode x the automesher, hidden probe points,
  bonuses; nonlinear tasks against the measured curve). Still open: one rubric across FEA, CFD and the
  nonlinear task, and how much a reference-free estimate (ZZ) may stand in for a reference on custom CAD.
- The improver edits notebooks (private notes + shared playbook) only. Whether it should also edit
  `meshing-harness/agent_context/` (skill, tool descriptions, examples) or harness defaults is untested.
- The mesh-harness MCP is not yet the ACE agents' tool set: `ACE_HARNESS=1` only adds its digest and gates.
- A pull-test data set for `abc_00000003_pull` (parts and plan in `physical_tests/`) has not been measured.

## Conventions

- `main` is the only long-lived branch and the source of truth. Branch off `main` per workstream, merge back
  through a pull request, and delete the branch once it is merged. Don't push directly to someone else's branch.
- Dropped or superseded work is kept as a tag `archive/<branch name>` before its branch is deleted
  (`git tag -l 'archive/*'`; restore with `git checkout -b <name> archive/<name>`).
- Never commit run outputs (`runs/`, `ace_runs/`, `swarms/`, `wsp_runs/`), study meshes
  (`benchmarks/gold/_meshes/`), API keys or tokens. Agents get `public/` problem files only, never `private/`.
- New long-running commands that people should be able to start go into the dashboard as a tool
  (`feacore/ui/tools/`), not only into a script.
- Tests: `pytest -m "not slow"` (root `tests/` and `meshing-harness/tests/`); `-m slow` for the long convergence
  checks.
- Update this file when a decision above gets made.
