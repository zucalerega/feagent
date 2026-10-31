# feacore UI and solver service

A web UI and HTTP API on top of `feacore`. It serves:

- the **solver service**: the fixed solver behind `POST /api/solve` (case JSON + gmsh `.msh`). The agents' MCP and the mesh harness call it.
- **convergence studies** of the benchmark problems (Richardson / GCI, `feacore.verification`).
- the **run store** (`feacore/runstore.py`): solves, studies and agent submissions as files, readable by the evaluator and the improvers.
- the **multi-agent consensus view**: every agent's answer with its own error band, against the population and the reference.

Pages (single-page app, no build step, `feacore/ui/static/`):

| page | what it shows |
|---|---|
| Benchmarks `#/benchmarks` | problem library, tag filter, detail with statement, 3D CAD preview with named groups, **evaluator view** (references, expectations, reference setup, studies) |
| Run study `#/study` | pick problem / backend / element order / refinement; live progress; per QoI: QoI vs log h with the Richardson extrapolation, the GCI band and the reference; error vs h (log-log, observed vs theoretical slope); verdict and flags in plain words |
| Consensus `#/consensus/<problem>` | forest plot per QoI: each submission's finest-mesh value ± GCI, extrapolated value, consensus median, reference ± tolerance; per-submission verdicts and flags |
| Field viewer `#/viewer/<run>` | 2D mesh / 3D boundary surface coloured by any field, colour bar, mesh edges, deformed shape with scale, study-level switcher. Tested smooth at 230k display triangles |
| Solve `#/solve` | upload case + mesh, run, see QoIs / SolveInfo / errors, open in the viewer |
| Health `#/health` | backends (import check), CPU / GPU (`nvidia-smi`), versions, jobs |

Screenshots: `docs/ui_screenshots/`.

## Run locally

```bash
pip install -e ".[dev]"
pip install fastapi uvicorn python-multipart httpx   # UI deps (httpx: tests / clients only)
export OMP_NUM_THREADS=1                              # on a shared box
python -m feacore.ui --host 127.0.0.1 --port 8000     # open http://localhost:8000
```

Options: `--runs-dir DIR` (default `$FEACORE_RUNS_DIR`, else `<repo>/runs`), `--quick-workers N` (parallel solves / geometry previews, default 1), `--study-workers N` (parallel studies, default 1), `--threads N` (sets OMP/MKL/OPENBLAS threads for the worker processes), `--extra-problems a.py,b.py` (register scratch problems that are not in the repo), `--fetch-vendor` (see below).

Environment: `FEACORE_RUNS_DIR`, `FEACORE_UI_QUICK_WORKERS`, `FEACORE_UI_STUDY_WORKERS`, `FEACORE_UI_SOLVE_TIMEOUT` (s, default 1800), `FEACORE_UI_STUDY_TIMEOUT` (default 4 h), `FEACORE_EXTRA_PROBLEMS`, `FEACORE_UI_VENDOR_DIR`, `FEACORE_BENCHMARKS_DIR`.

The browser loads three.js 0.170.0 from jsDelivr (pinned). For an offline demo (projector, flaky Wi-Fi), vendor it once: `python -m feacore.ui --fetch-vendor`. This writes `feacore/ui/static/vendor/three-0.170.0/`, about 720 kB, and the server then serves the local copy. Everything else is in the repo; no Node is needed.

## Run on Brev

On the instance (setup per CLAUDE.md, then the UI deps above):

```bash
cd feagent && tmux new -s ui          # or nohup ... &
python -m feacore.ui --host 0.0.0.0 --port 8000 --threads 4 --quick-workers 2
```

Then reach port 8000 in one of these ways:

1. **Port-forward with the CLI** (best for scripts, the agent harness and API calls): `brev port-forward <instance-name> --port 8000:8000`, then open `http://localhost:8000`. Use `--port 9000:8000` if local 8000 is taken.
2. **Shareable URL** (for the demo audience): in the Brev web console, open the instance's *Access* section → *Using Tunnels*, and expose port 8000. The tunnel runs through Cloudflare and asks for browser authentication on first access. So scripted API clients should use (1) instead.
3. Plain SSH: `ssh -L 8000:localhost:8000 <instance-name>`, using the SSH config that `brev` writes.

The steps come from the Brev connectivity docs (docs.nvidia.com/brev, *CLI → connectivity*). Check the console menu names there; they could not be fetched from the dev container. The server has **no authentication**. Share the tunnel URL only with the team, and do not bind 0.0.0.0 on a machine with a public IP unless a firewall is in front of it.

## Architecture

- `feacore/ui/server.py`: FastAPI app, `create_app(runs_dir=...)`. Handlers only read and write the run store.
- `feacore/ui/jobs.py`: job runner. **Every mesh / solve / geometry job runs in its own child process** (`python -m feacore.ui.worker '<json>'`), because gmsh is process-global and not thread-safe, and a crashing or hanging solve must not take the server down. There are two lanes: `quick` for solves and geometry previews, and `study` for convergence studies. Timeouts kill the child. A job that dies without reporting is marked `failed` with the exit code. Records left `queued` / `running` by a previous server process are marked `interrupted` at startup.
- `feacore/ui/worker.py`: job bodies. A study calls the contract `feacore.verification.study.convergence_study` and wraps the solver it uses, so that each level's `Result` is stored (fields for the viewer) and progress is reported while the study runs.
- `feacore/ui/viewdata.py`: viewer data. For 3D, the boundary faces of the volume mesh (oriented outward); for 2D, every cell. Quadratic faces are split into 4 display triangles. Nodal values are matched to the solution-space nodes with a KD-tree; `probe` fills in the rest.
- `feacore/ui/analysis.py`: verdicts, flag explanations, submission analysis, consensus (pure numpy).

## API

All responses are strict JSON (NaN/inf become `null`). Large arrays are typed arrays: `{"dtype": "float32", "shape": [n, 3], "b64": "<little-endian base64>"}`. Pass `encoding=list` for plain lists. `feacore.ui.viewdata.decode_array` decodes either form. Interactive docs: `http://HOST:8000/docs`.

```bash
S=http://localhost:8000
curl $S/api/health                         # backends (import check), CPU/GPU, versions, jobs
curl "$S/api/problems?tag=nafems&tag=2d"   # list (tags AND-ed), tag counts, import errors
curl $S/api/problems/nafems_le1            # statement + evaluator view (references, expectations, reference case, studies)
curl $S/api/problems/nafems_le1/geometry   # coarse CAD triangulation + named groups (gmsh in a worker, cached)
```

### Solve service

`POST /api/solve` takes `multipart/form-data` with these fields:

| field | |
|---|---|
| `case` | case JSON (file or text field), schema in `feacore/case.py`. Its `mesh` entry is replaced by the uploaded mesh |
| `mesh` | gmsh `.msh` (v2.2 / v4.1) with named physical groups |
| `backend` | `skfem` (default, the fixed solver), `ccx`, `nn` |
| `agent_id`, `problem_id`, `submission_id` | optional labels stored with the run |
| `wait` | `true` (default): block until solved; `false`: return 202 at once and poll `GET /api/runs/{id}` |
| `fields` | `true` (default): store viewer data |
| `meta` | optional JSON object stored with the run |

Responses:

- **200**: the run record, `status: "succeeded"`.
- **422**: either `{"detail": "invalid case", "errors": [...]}` (schema check, nothing stored), or the run record with `status: "failed"` and `error` / `traceback` (the solver rejected the setup, e.g. an unknown group name).
- **202**: with `wait=false`.

```bash
curl -F case=@case.json -F mesh=@part.msh -F backend=skfem \
     -F agent_id=agent-07 -F problem_id=nafems_le1 $S/api/solve
```

```python
import httpx, json
r = httpx.post(f"{S}/api/solve", timeout=None,
               files={"case": ("case.json", json.dumps(case)), "mesh": open("part.msh", "rb")},
               data={"agent_id": "agent-07", "problem_id": "nafems_le1"})
run = r.json()          # run["status"], run["qois"], run["info"], run["error"]
```

The mesh harness's output directory becomes a case with `python -m feacore.adapters.mesh_harness <out_dir>` (see CLAUDE.md). Post that case with the harness's `mesh.msh`.

### Runs

```bash
curl "$S/api/runs?problem_id=nafems_le1&agent_id=agent-07&limit=50"   # summaries, newest first
curl $S/api/runs/RUN_ID                                                 # full record
curl "$S/api/runs/RUN_ID/field?name=von_mises"            # points, triangles, edges, values, displacement, range
curl "$S/api/runs/RUN_ID/field?name=sxx&geometry=false"   # values only (field switching)
curl -O $S/api/runs/RUN_ID/mesh                           # the stored mesh; /log = worker log
curl -X POST $S/api/runs -H 'Content-Type: application/json' \
     -d '{"problem_id": "nafems_le1", "agent_id": "agent-07", "backend": "skfem",
          "qois": {"syy_D": 9.26e7}, "info": {"h_rep": 0.05, "order": 2, "n_dofs": 31244}}'  # register a run solved elsewhere
```

### Studies

```bash
curl -X POST $S/api/studies -H 'Content-Type: application/json' \
     -d '{"problem_id": "nafems_le1", "backend": "skfem", "order": 2}'          # default sizes
#    "sizes": [0.2, 0.1, 0.05]   or   "levels": 4, "ratio": 2   (from the coarsest default size)
curl $S/api/studies/STUDY_ID     # status, progress {done, total, stage, message, levels[]}, result, interpretation, runs
curl "$S/api/studies?problem_id=nafems_le1"
```

`result` is exactly the `convergence_study(...)` dict from `feacore.verification.study`. `interpretation` gives, per QoI, `{verdict, status, label, flags, explanations[{flag, severity, title, text}], fine, band_abs, phi_ext, extrapolation_reliable, p, p_theory, summary}`.

### Submissions and consensus

A **submission** is one agent's answer to one problem: its runs on a sequence of its own meshes (three or more meshes for a GCI estimate).

```bash
curl -X POST $S/api/submissions -H 'Content-Type: application/json' \
     -d '{"problem_id": "nafems_le1", "agent_id": "agent-07", "run_ids": ["run-...", "run-...", "run-..."]}'
# or, for runs solved elsewhere: "levels": [{"h": 0.1, "qois": {"syy_D": 9.25e7}, "order": 2, "n_dofs": 9000}, ...]
curl "$S/api/submissions?problem_id=nafems_le1"
curl $S/api/consensus/nafems_le1
```

`GET /api/consensus/{problem}` returns, per QoI:

- the `reference` (evaluator-only) and its `tolerance`;
- one row per submission: `fine`, `band_abs` (GCI × |fine|), `phi_ext`, `extrapolation_reliable`, `p`, `p_theory`, `verdict`, `flags`, `explanations`, `err_ref_fine`, `err_ref_best`, `within_tolerance`;
- `consensus`: median and 1.4826·MAD of the per-submission best estimates, min/max, `n_outliers`;
- `reference_studies`: the latest fixed-solver studies and the exported `validation_*.json` files.

The best estimate is the extrapolated value when the convergence and the extrapolation are reliable, otherwise the finest-mesh value. With 3 or more submissions, a submission is an `outlier_vs_consensus` when |best − median| > max(3·σ_robust, its own band, tolerance·|median|).

## Run-record format (for the harness and the evaluator)

Root: `$FEACORE_RUNS_DIR`, else `<repo>/runs`. The store writes `runs/.gitignore` containing `*`, so nothing in it is ever committed. Every write is atomic (temp file + rename).

```
runs/
  runs/<run_id>/run.json        one solve
  runs/<run_id>/mesh.msh        uploaded / copied mesh (study runs point into the study instead)
  runs/<run_id>/fields.npz      viewer data
  runs/<run_id>/worker.log      stdout/stderr of the worker process
  studies/<study_id>/study.json
  studies/<study_id>/meshes/    the study's meshes
  submissions/<submission_id>.json
  cache/geometry/               CAD previews
```

IDs look like `run-20261031-024430-4b88dc` (UTC time + random suffix, so they sort by creation time). Custom ids are allowed: `[A-Za-z0-9][A-Za-z0-9._-]*`.

**`run.json`** (`schema: "feacore.run/1"`):

| key | |
|---|---|
| `run_id`, `created_at`, `started_at`, `finished_at` | ISO 8601 UTC |
| `status` | `queued` / `running` / `succeeded` / `failed` / `interrupted` |
| `source` | `api` (POST /api/solve), `ui` (study level), `import` (POST /api/runs), `python` (RunStore API) |
| `problem_id`, `agent_id`, `submission_id`, `study_id`, `level`, `h_nominal` | labels (any may be null) |
| `backend` | `skfem` / `ccx` / `nn` |
| `case` | the case dict (`feacore.case.Case.to_dict()`) |
| `mesh_file` | mesh path, relative to the store root |
| `info` | `SolveInfo` dict: `backend, physics, formulation, order, dim, cell_type, n_nodes, n_cells, n_dofs, measure, h_rep, wall_time, warnings` |
| `qois` | `{name: value}` |
| `fields` | `{file, available[], stats{field: {min, max, boundary_min, boundary_max, n_nan}}, dim, n_points, n_triangles, has_displacement, cell_type}` or null |
| `error`, `traceback`, `warnings` | on failure / partial success |
| `meta` | free-form |

**`fields.npz`** holds: `points` float32 (n, 3) (2D: z = 0); `triangles` uint32 (m, 3); `edges` uint32 (k, 2) (element edges, curved for P2); `disp` float32 (n, 3) (elasticity only); `f_<field>` float32 (n,); `meta` (JSON string).

**`study.json`** (`feacore.study/1`) holds: `study_id, created_at, started_at, finished_at, status, source, problem_id, backend, order, geom_order, sizes, progress{done, total, stage, message, levels[{index, h, status, run_id, n_cells, n_dofs, h_rep, wall_time, qois}]}, run_ids, result` (the `convergence_study` dict), plus `interpretation, error, traceback, meta`.

**`submissions/<id>.json`** (`feacore.submission/1`) holds: `submission_id, created_at, problem_id, agent_id, run_ids, levels[{run_id, h, qois, n_dofs, n_cells, order, backend}], order, notes, meta, analysis{order, n_levels, qois[{name, type, field, h[], values[], run_ids[], p_theory, expectation, gci, fit, interpretation}]}`. Here `h` is `info.h_rep` of each run. `gci` / `fit` come from `feacore.verification.gci.analyze`, and `p_theory` from `Problem.expected_order` or `default_expected_order`.

Python API (writes the same files; no server needed):

```python
from feacore.runstore import RunStore
from feacore.solver import solve

store = RunStore()                                   # or RunStore("/path/to/runs")
ids = []
for h, mesh in meshes:                               # the agent's own mesh sequence
    res = solve(case_for(mesh), backend="skfem")
    ids.append(store.record_result(res, mesh_path=mesh, problem_id="nafems_le1",
                                   agent_id="agent-07"))      # case, SolveInfo, QoIs, viewer fields
sub = store.create_submission("nafems_le1", "agent-07", run_ids=ids)
sub["analysis"]["qois"][0]["interpretation"]["verdict"]       # e.g. "converged"
store.save_run({"qois": {...}, "info": {...}, "agent_id": "x"})  # import without a Result
```

## Verdicts and flags (plain words in the UI)

The GCI classification comes from `gci3` (Celik et al. 2008). The UI adds verdicts on top of it:

| verdict | when |
|---|---|
| `converged` | monotone convergence, order defined, asymptotic |
| `converged_noise` | `gci3` convergence `converged`: changes below 1e-6 (round-off) |
| `plateau` | **UI rule**: both mesh-to-mesh changes below 0.1 % although `gci3` reports oscillatory or divergent. These are noise-level wiggles (e.g. a max over a recovered field), not a singularity. The band shown is \|fine − medium\| |
| `converging` | monotone but `not_asymptotic` |
| `slow` | `p_far_below_theory` |
| `oscillating` | `gci3` oscillatory |
| `diverging` | divergent / oscillatory-divergent |
| `expected_divergence` / `unexpected_convergence` | the problem marks the QoI as singular (`expectations()`) |
| `insufficient` | fewer than three meshes |

Derived flags:

- `singular_suspected`: a stress-type QoI that diverges, or converges with p < ½ p_theory.
- `extrapolation_unreliable`: the same rule as the contract's `extrapolation_reliable` (`study._judge`). The finest-mesh value is used instead.
- `outlier_vs_consensus`: see the consensus rule above.

Each flag carries a one-paragraph explanation (`feacore/ui/analysis.py`, `FLAG_INFO`).

## Tests and screenshots

```bash
python -m pytest tests/test_ui_api.py -q
```

The tests use FastAPI's TestClient on a temporary store and cover every endpoint, including real solves through `POST /api/solve` and a real three-level study, submissions and consensus. They skip if no 2D elasticity problem is registered or `skfem` does not import.

The screenshots were made with headless Chromium (Playwright) against a scratch store. The consensus screenshot shows **scripted test agents**: deterministic scripts (`agent_id` `scripted-*`) that mesh the CAD themselves, change the reference setup (P1 vs P2, other mesh sizes, two meshes only, a symmetry edge clamped by mistake) and solve through `POST /api/solve`. Every number is a real solve. None of them is an LLM agent.

## Known gaps

- No authentication or rate limiting. Uploads are trusted; the size limit is whatever the proxy allows.
- Each job starts a fresh Python process (about 1 s overhead). That is fine for meshes that take seconds to solve, but wasteful for thousands of tiny solves.
- List endpoints scan the store directory on each call. That is fine up to a few thousand runs; past that, add an index.
- The viewer shows the boundary surface only (no cut planes or interior clipping in 3D). Axisymmetric results are shown as the 2D meridian section.
- The `nn` backend shows whatever its module provides. The UI treats it like any other backend.
