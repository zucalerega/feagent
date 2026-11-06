# feacore UI and solver service

A web UI and HTTP API on top of `feacore`. It serves:

- the **solver service**: the fixed solver behind `POST /api/solve` (case JSON + gmsh `.msh`). The agents' MCP and the mesh harness call it.
- **convergence studies** of the benchmark problems (Richardson / GCI, `feacore.verification`).
- the **run store** (`feacore/runstore.py`): solves, studies and agent submissions as files, readable by the evaluator and the improvers.
- the **multi-agent consensus view**: every agent's answer with its own error band, against the population and the reference.

Navigation has five sections: **Benchmarks** (library, problem pages, batch runs), **Runs** (studies, batches, solves, consensus), **Swarm** (the ACE agent loop), **Solve** and **Connect** (plus Health). Pages (single-page app, no build step, `feacore/ui/static/`):

| page | what it shows |
|---|---|
| Benchmarks `#/benchmarks` | problem library, tag filter, detail with statement, 3D CAD preview with named groups, **evaluator view** (references, expectations, reference setup, studies); **Run with defaults** starts the reference study (skfem, P2, the problem's sizes) and opens its live view |
| Run benchmarks `#/batch` | select many problems (by tag, family, ticked rows, "all with references"), choose backend / order / nested / meshes, queue one convergence study each; live per-problem progress; results table (reference, finest, extrapolated, p vs theory, GCI, error, pass / fail / divergence detected as expected, runtime) with links to each study's plot and finest-mesh field; CSV / JSON download. Batches persist in the run store |
| Runs `#/runs` | tabs for convergence studies, benchmark batches, solves (opens the field viewer) and consensus views; "New study" and "Run benchmarks" start new work |
| Agents `#/agents` | the ACE loops from the dashboard: train swarms and the single-agent baseline, runs with their training curve, per-agent views and the live tracker, then compare, export and run a trained swarm on any problem. See "Agents section" below |
| Swarm `#/swarm` | the ACE agent loop, live: loop and generation selectors, score distribution per generation, per-problem leaderboards (score, QoI vs reference, mesh, DOFs, solves, gates) linking to the field viewer and the problem's Consensus view. Empty state until the ACE bridge writes records |
| Run study `#/study` | pick problem / backend / element order / refinement; live progress; per QoI: QoI vs log h with the Richardson extrapolation, the GCI band and the reference; error vs h (log-log, observed vs theoretical slope); verdict and flags in plain words |
| Consensus `#/consensus/<problem>` | forest plot per QoI: each submission's finest-mesh value ± GCI, extrapolated value, consensus median, reference ± tolerance; per-submission verdicts and flags |
| Field viewer `#/viewer/<run>` | 2D mesh / 3D boundary surface coloured by any field, colour bar, mesh edges, deformed shape with scale, study-level switcher. Tested smooth at 230k display triangles |
| Solve `#/solve` | upload case + mesh, run, see QoIs / SolveInfo / errors, open in the viewer |
| Connect `#/connect` | API base URL as the browser sees it, the (masked) token, copy-paste curl / httpx / requests examples for solving and submitting runs |
| Health `#/health` | backends (import check), CPU / GPU (`nvidia-smi`), versions, jobs |

Screenshots: `docs/ui_screenshots/`.

## Access the UI

Locally:

```bash
pip install -e ".[ui]"                 # fastapi, uvicorn, python-multipart (httpx for the tests: dev group)
export OMP_NUM_THREADS=1               # on a shared box
python -m feacore.ui --port 8000       # then open http://localhost:8000
```

Options: `--host` (default 127.0.0.1; 0.0.0.0 to expose), `--runs-dir DIR` (default `$FEACORE_RUNS_DIR`, else `<repo>/runs`), `--quick-workers N` (parallel solves / geometry previews, default 1), `--study-workers N` (parallel studies; default one per 4 cores, at most 4), `--token T` (shared-token auth, `auto` generates one; see below), `--threads N` (sets OMP/MKL/OPENBLAS threads for the worker processes), `--extra-problems a.py,b.py` (register scratch problems that are not in the repo), `--fetch-vendor` (see below).

Environment: `FEACORE_RUNS_DIR`, `FEACORE_UI_TOKEN`, `FEACORE_UI_QUICK_WORKERS`, `FEACORE_UI_STUDY_WORKERS`, `FEACORE_UI_SOLVE_TIMEOUT` (s, default 1800), `FEACORE_UI_STUDY_TIMEOUT` (default 4 h), `FEACORE_EXTRA_PROBLEMS`, `FEACORE_UI_VENDOR_DIR`, `FEACORE_BENCHMARKS_DIR`.

three.js (pinned 0.170.0, the only external JS) is served **same-origin** at `/vendor/...`, so the viewer's browser never needs to reach a CDN. The server serves the first copy it finds:

1. a vendored copy (`python -m feacore.ui --fetch-vendor` writes `feacore/ui/static/vendor/three-0.170.0/`, about 720 kB);
2. a copy it downloaded once into the run-store cache (`runs/cache/vendor/`), from jsDelivr or, if jsDelivr is blocked, the npm registry tarball;
3. a redirect to jsDelivr, if the server itself is offline.

No Node and no build step are needed.

## Run on Brev

Setup on any account is one command (details in `docs/brev.md`):

```bash
curl -fsSL https://raw.githubusercontent.com/zucalerega/feagent/main/scripts/bootstrap.sh | bash
```

`scripts/serve.sh` keeps the UI running as a service (`python -m feacore.ui --host 0.0.0.0 --port 8000`). It auto-updates and waits while `active_jobs` (from `/api/health`) is non-zero. To reach port 8000:

- **Brev HTTP Ports** (the team link): on the instance page, open **HTTP Ports**, add port **8000** and open the https link. Brev's sign-in sits in front of it, so no feacore token is needed there.
- **Port-forward** (your laptop, scripts, the agent harness): `brev port-forward <instance> --port 8000:8000`, then use http://localhost:8000. Scripts can't pass Brev's browser sign-in, so they use this route.
- Don't use the console's "TCP/UDP Open Port" dialog for this service. It opens the port unauthenticated to 0.0.0.0/0.

### Shared-token auth (only for links without Brev's sign-in)

If `FEACORE_UI_TOKEN` or `--token` is set, every `/api/*` request must carry the token in one of three forms:

- `Authorization: Bearer <token>`
- an `X-Feacore-Token: <token>` header
- `?token=<token>`

A missing or wrong token gets **401** with an explanation. The comparison is constant-time.

The page shell and static files stay open, so the UI can load. Share the UI as `https://<tunnel>/?token=<token>`: the browser stores the token in localStorage, removes it from the address bar, and sends it on every API call. Download links get `?token=`. Without a token the UI shows a prompt.

`GET /api/auth` (always open) reports `{required, valid}`. `GET /api/health` is open too, so deploy scripts can poll it locally. Without a valid token it returns only `{"ok": true, "active_jobs": N, "auth_required": true}`; with one, the full payload. With no token configured, nothing changes. The **Connect** page (`#/connect`) shows the base URL, the masked token (click to reveal) and copy-paste examples for teammates.

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
curl $S/api/health                         # active_jobs (queued + running solves/studies), backends, CPU/GPU, versions
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

### Benchmark batches

```bash
curl -X POST $S/api/batches -H 'Content-Type: application/json' -d '{
  "problem_ids": ["nafems_le1", "nafems_le10", "kirsch_exact_traction"],
  "backend": "skfem", "order": 2, "nested": false, "levels": null, "name": "nightly"}'
#   levels: meshes per study (null = the problem's own sizes), ratio: refinement ratio with levels
curl $S/api/batches                    # recent batches (status, counts, outcomes)
curl $S/api/batches/BATCH_ID           # items[] (live progress per problem) + rows[] (results table)
curl -O $S/api/batches/BATCH_ID/results.csv     # or results.json
curl -X POST $S/api/batches/BATCH_ID/cancel     # cancel queued / running studies
```

A batch queues one convergence study per problem in the study lane, which runs `--study-workers` at a time. Each `rows[]` entry is one QoI of one problem, with these fields:

- `reference`, `tolerance_pct`, `finest`, `extrapolated`, `extrapolation_reliable`
- `p_observed`, `p_theory`, `gci_pct`
- `error_pct` and `error_basis`: the error uses the extrapolated value when the extrapolation is reliable, otherwise the finest mesh, as in `study._judge`
- `outcome`: `pass` / `fail` / `divergence_detected` / `divergence_missed` / `no_reference`
- `verdict`, `runtime_s`, `study_id`, `finest_run_id`

Single studies take `"nested": true` too (`POST /api/studies`). That meshes once at the coarsest size and splits it with ratio 2 per level, which gives cleaner observed orders. Each 3D split multiplies the cells by 8.

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

## Agents section

`#/agents` runs the ACE loops (`ace/`) from the dashboard. It replaces the Streamlit agent tracker
(`ace/agent_dashboard.py`, removed): its per-agent views are in each run's page. Three tabs:

- **Train** (`#/agents?tab=train`): start a swarm (`ace.run_loop`) or the single-agent baseline
  (`ace.single_loop`) as a task. One click fills the form with a recipe: a free mock try, the deep
  swarm of Result 2, the post-training of Result 3 (from `results/trained_swarms/div_s8`), or the
  basic loop. Problems come from the loop's task catalogue (`ace/config.py` `TASKS`): click a problem
  once to train on it, twice to hold it out. Every flag of the two loops is in the form (the rarer
  ones under "More options"). The table below lists the Agents tasks, with Stop (writes
  `<run>/STOP`: the loop ends after the current generation and still writes its summary) and Kill.
- **Runs** (`#/agents?tab=runs`): every run folder in `ace_runs/` and every folder an Agents task
  wrote, with status, generations, cost and scores. A run's page (`#/agents/run/<key>`) refreshes
  while the run goes. It shows:
  - the training curve (mean or best score per problem and generation) and its table;
  - an agents × generations score matrix. A cell opens that agent run: its score parts, the run
    report's pictures (what the teacher and the agent's self-reflection looked at), the context it
    saw with the changes since its previous one, the notes as stored, the teacher's feedback, its
    self-reflection, the organizer's changes, its tool calls, and its solves with links to the
    field viewer;
  - the live tracker of `ace.watch`, `report.html` (built with the Run report tool), the summary,
    `scores.png` and the task's console output.
- **Evaluate** (`#/agents?tab=evaluate`):
  - run an exported swarm (`results/trained_swarms/`, `swarms/`, or an export made here) on any
    registered problem, your CAD and the gold parts included (`ace.infer`);
  - export a run (`ace.export_swarm`, default `swarms/<run>`);
  - compare a deep swarm with a single-agent run (`ace.compare`);
  - the results: `#/agents/infer/<key>` (the consensus, each agent's prediction and pictures),
    `#/agents/swarm/<key>` and `#/agents/compare/<key>`.

Real runs need `ANTHROPIC_API_KEY` in the server's environment. Without it only mock runs start,
and the forms start with "mock LLM" ticked. Field scoring (`ACE_SCORE=field`) and the mesh harness
(`ACE_HARNESS=1`) are environment settings of ACE: the tools set them per task (`Tool.build_env`).
Deep runs and the single agent always use field scoring. Training runs record every solve and
agent run in this server's run store (the task runner sets `FEACORE_RUNS_DIR`), so they also fill
the Swarm page, the Consensus views and the field viewer.

API (`feacore/ui/ace_runs.py`, read-only; the work itself runs through `POST /api/tasks` with the
tools `ace.swarm`, `ace.single`, `ace.compare`, `ace.report`, `ace.export_swarm`, `ace.infer`):

| endpoint | |
|---|---|
| `GET /api/ace/catalog` | the loop's task catalogue and suites (for the forms) |
| `GET /api/ace/runs` | training runs, newest first, each with a summary |
| `GET /api/ace/run?dir=KEY` | one run: per generation and problem, the agents, files, tasks |
| `GET /api/ace/agent-run?dir=KEY&run_id=ID` | one agent run (see the run page above) |
| `GET /api/ace/file?dir=KEY&path=REL[&swarm=true]` | a picture or text file inside a known folder |
| `GET /api/ace/report?dir=KEY` | `<run>/report.html` |
| `GET /api/ace/tracker?dir=KEY`, `/api/ace/tracker/data?dir=KEY` | the live tracker page and its data (`ace.watch.collect`) |
| `GET /api/ace/swarms`, `/api/ace/swarm?dir=KEY` | exported swarms |
| `GET /api/ace/infers`, `/api/ace/infer?dir=KEY` | `ace.infer` results |
| `GET /api/ace/compares`, `/api/ace/compare?dir=KEY` | `ace.compare` outputs |

A folder's key is its path relative to the repository (`ace_runs/div_s8`), or `@runs/<path>` for a
folder inside the run store. A key is refused (404) unless it names one of the folders above; a
symlink that leaves its root is refused too. The report and the tracker pages are served with a
Content-Security-Policy that lets them run only their own inline code and talk only to this
server. The tracker polls its data without headers, so when the server has a token the page is
served with its data URL carrying the token it was opened with (`?token=`).

## ACE swarm records (Swarm page)

The ACE loop's bridge writes into the same run store, through `RunStore` or the HTTP API. `GET /api/swarm?loop_id=&generation=` reads it back. The response holds:

- `loops`: each with its generations, counts and an `active` flag (new records in the last 3 min, which makes the page poll every 4 s);
- `generations[]`: `n_submissions`, `n_running`, `n_gated`, median and best score, and one point per agent run;
- `problems[].rows[]`: the selected generation's leaderboard.

**Each ACE solve** is a run record:

```json
{"source": "ace", "agent_id": "ace:<ace_run_id>", "problem_id": "nafems_le1",
 "meta": {"source": "ace", "loop_id": "L1", "generation": 3, "ace_run_id": "<id>", "solve_index": 2,
          "mesh": {"h": 0.05, "order": 2, "refinements": 1}},
 "qois": {...}, "info": {...SolveInfo...}, "status": "succeeded"}
```

**Each finished ACE agent run** is a submission. `run_ids` are its successful solves:

```json
{"problem_id": "nafems_le1", "agent_id": "ace:<ace_run_id>", "run_ids": ["run-...", "..."],
 "meta": {"source": "ace", "loop_id": "L1", "generation": 3, "ace_run_id": "<id>",
          "submitted_run_id": "run-...", "score": 0.42, "gates": [], "setup_diagnostics": ["..."],
          "cited_bullets": ["..."], "tokens": 12345}}
```

Python: `store.create_run(..., source="ace", meta=...)` then `store.update_run(...)`, or `store.record_result(result, ..., source="ace", meta=...)`. Finish with `store.create_submission(problem_id, "ace:<id>", run_ids=..., meta=...)`.

**Score** = log10(e_uniform / e_agent): 0 means as good as the best uniform mesh with the same node budget, +1 means ten times better, and a failed gate scores −2. The bridge computes it; the UI only displays it.

**Speed / accuracy score (current ACE).** Submissions with `meta.score_parts` use a different score: `score` in [0, 1] = valid × accuracy × efficiency.

- 1 is the cheapest run on that problem within tolerance; invalid runs score 0, and their `gates` say why (e.g. `"invalid: <flags>"`).
- The score is relative to every run on the problem so far, so the bridge rewrites each submission's meta in place every generation. The page re-reads the records on every refresh.
- `score: null` means the generation isn't scored yet (shown as *pending*), or, with `meta.error` set, a crash.
- `score_parts = {accuracy, efficiency, valid, rel_error, tolerance, meets_tol, dofs, pareto_rank, population}` appears under the score bar and in its tooltip.
- `diagnosis` (the diagnoser's verdict) sits in a collapsed column.
- `agent` is a persistent id like `a3` (run ids end in `-a<k>`). It feeds the *Agents across generations* table, so one agent can be followed as its playbook changes.

The page detects the scheme per loop (`score_scheme` = `speed_accuracy` if any submission has `score_parts`, else `log`) and draws a 0–1 axis and bar or the older log-score view.

Details the UI relies on (as written by the bridge, `ace/ui_bridge.py` on `ace-ui-bridge`):

- `meta.mesh.refinements` is the agent's list of refinement dicts. The leaderboard shows their count, with the details in a tooltip.
- `solve_index` counts from 0, and the "scored #k" note is 1-based.
- `submitted_run_id` is the solve ACE scored, and the QoI value comes from that run. The reference is matched by the QoI name in that run's case (or `meta.qoi`), not by the problem's first reference.
- `tokens` = `{input, output, cache_write, cache_read, total}`.
- An agent run with **no successful solve** is a submission with `run_ids: []`, score −2 and the gate "no successful solve". A **crashed** run has score `null` and `meta.error`. Both show as finished (crashed in red), never as running.
- ACE ids are unique only within a loop, so the page keys agents by `(loop_id, ace_run_id)` and the Consensus view labels ACE agents with their loop.

Per agent run, the leaderboard shows:

- the score;
- the submitted run's value of the problem's primary QoI (the first reference) and its error;
- the mesh (`meta.mesh`) and DOFs;
- solves, as successful / attempted;
- gates and diagnostics counts.

Agents without a submission yet are listed as running. ACE submissions also appear in the regular Consensus view of each problem.

## Nonlinear and Test data pages

**Nonlinear** `#/nonlinear` (`feacore/ui/nonlinear_api.py`, tools in `feacore/ui/tools/nonlinear.py`) has three tabs:

- **Runs**: every nonlinear run found in the run store (`<runs>/nonlinear/`, `<runs>/ccx_ramp/`) and in the committed results, with its force-travel curves over the Instron test of the printed C-bracket (or an uploaded test). The simulated travel is mapped to the crosshead with the frozen seating gap and compliance of `nonlinear_spec.json` (editable), and each curve gets the mean force error at the test's checkpoints, as the agents are scored. `#/nonlinear/run/<ref>` shows one run: facts, curves, its pictures and videos, render buttons, files and the live `progress.log`.
- **Run an analysis**: the tools, with small defaults timed on a 4-core box: the agents' frozen scoring ramp (`scripts/nl_baselines.py --out`, automesher + 4 mm in about 1 min), the elastoplastic ramp of the test (`scripts/ccx_compression_ramp.py --straight`, about 30 s at 5 mm), and crush to fracture in three steps (a straight-edged crush mesh from `mesh_systems/ace_sizing --straight`, an optional buckling imperfection, `experiments/physical/crush.py`: about 6 min at 8 mm, hours for the 40k-node study meshes). Renderers turn a finished run into stills (`scripts/render_ccx_ramp.py`) or videos (`experiments/physical/render_crush.py`, `scripts/nl_video.py ramp`); the pyvista ones run through `python -m feacore.ui.offscreen`, which uses `xvfb-run` when there is no display and imageio-ffmpeg when there is no system ffmpeg.
- **Gallery**: the committed figures and videos (`talk_plots/act3_real_data/`, `docs/gallery/`, `physical_tests/.../ccx_ramp/`, the scoring baselines) with the captions of their READMEs and HTML pages. The Omniverse RTX renders need the DGX Spark: committed media only.

Dashboard outputs: `<runs>/nonlinear/{score,ramp,crush,meshes,renders}/<tool>-<task id>/`. A crush run keeps a checkpoint when it is killed; *Resume* continues it into a new run directory.

**Test data** `#/testdata` (`feacore/ui/testdata_api.py`) lists the physical tests (committed evidence specs under `physical_tests/`, uploads under `<runs>/testdata/<id>/`) with their curves, and the printable test parts (STL / STEP). An upload is a machine export (CSV / TXT as Bluehill writes it) read with the loop's own `ace.evidence.read_table`: the columns and units are detected and can be changed, the curve and the loop's processing (`process_test`: loadings, fitted compliance, uncertainty budget) are shown before it is saved. A saved record is `test.json` (an evidence spec plus a `lab` block) next to the export, a ready `--evidence` folder: `python -m ace.run_loop --problems <problem> --evidence <runs>/testdata/<id>`.

API: `GET /api/nonlinear/{runs,run?ref=,test,meshes,gallery,capabilities,file?ref=}`; `GET /api/testdata`, `GET|DELETE /api/testdata/item?id=`, `POST /api/testdata/parse` and `POST /api/testdata` (multipart `file` + fields), `GET /api/testdata/file?ref=`. Files are served only from the run store's `nonlinear/`, `ccx_ramp/` and `testdata/` and from the committed media folders, with a known media type (never HTML); links carry `?token=` when the server has one.

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
  batches/<batch_id>.json       benchmark batch: settings + [{problem_id, study_id}]
  cache/geometry/               CAD previews
  cache/vendor/                 three.js fetched once by the server
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

**`batches/<id>.json`** (`feacore.batch/1`) holds `batch_id, created_at, name, settings{backend, order, nested, levels, ratio, study_workers}, items[{problem_id, study_id}], meta`. Progress and results are derived from the studies when the batch is read. Each study also records `batch_id`, `nested` and `elapsed` (s), and its `progress` has `stage` (meshing / solving / storing / solved), `stage_started_at` and per-level `started_at` (epoch s), so long 3D levels show a running clock.

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
| `scatter` | **UI rule**: divergent per `gci3`, but the changes are below 2% and \|value\| does not grow monotonically. This is typical of point stresses on independently remeshed 3D tet meshes (e.g. LE10 `syy_D`: 0.6-0.9%), not a singularity. Judge against the reference tolerance |
| `converging` | monotone but `not_asymptotic` |
| `slow` | `p_far_below_theory` |
| `oscillating` | `gci3` oscillatory |
| `diverging` | divergent / oscillatory-divergent |
| `expected_divergence` / `unexpected_convergence` | the problem marks the QoI as singular (`expectations()`) |
| `insufficient` | fewer than three meshes |

Derived flags:

- `singular_suspected`: a stress-type QoI that diverges (beyond the scatter rule), or converges with p < ½ p_theory.
- `extrapolation_unreliable`: the same rule as the contract's `extrapolation_reliable` (`study._judge`). The finest-mesh value is used instead.
- `outlier_vs_consensus`: see the consensus rule above.

Each flag carries a one-paragraph explanation (`feacore/ui/analysis.py`, `FLAG_INFO`).

## Gold and Your CAD pages

Routes `#/gold` and `#/cad` (`#/cad/<upload>`). Server side: `feacore/ui/gold_api.py` (mounted once by
`create_app`; it also mounts `feacore/ui/cad_api.py`), tools in `feacore/ui/tools/gold.py` and `cad.py`.

```bash
curl $S/api/gold                                   # parts -> BC sets, study meshes with their status here
curl -OJ $S/api/gold/meshes/abc_00000003/L0         # a stored study mesh (404 when not on this machine)
curl -X POST $S/api/gold/solve -H 'Content-Type: application/json' \
     -d '{"set_id": "abc_00000003_bc0", "level": 0, "source": "regenerated"}'    # a run for the viewer
curl -T goldgen-l2_0001.tgz $S/api/gold/archives/goldgen-l2_0001.tgz              # then the gold.import tool
curl -F file=@part.step -F units=si $S/api/cad/uploads                            # a STEP upload -> its faces
curl -X POST $S/api/cad/uploads/<upload>/sets -H 'Content-Type: application/json' \
     -d '{"mode": "manual", "fixed": ["f12"], "loads": [{"group": "f7", "type": "pressure", "value": 2e6}]}'
```

- **Mesh status** on the Gold page: `ok` (the sha256 of index.json), `mismatch`, `missing` (the meshes are
  not in git), `unlisted`. Tools: `gold.regenerate` (`python -m feacore.goldgen.local regenerate PART
  --levels 0,1`: the generator's settings into `<runs>/gold_regen/<part>/`, sha256 and cells compared with
  the index; `--install` only for a byte-identical level), `gold.verify` / `gold.render`
  (`scripts/gold.py`), `gold.import`, `gold.nonlinear` (`scripts/gold_nonlinear.py` on a view of the gold
  root). Regenerated meshes do not reproduce the stored bytes: gmsh is not even bitwise reproducible from
  run to run, and the input is the rescaled public STEP (cells within ~0.5 % at L0).
- **Uploads** (`feacore/problems/custom.py`): checked (size limit `FEACORE_CAD_MAX_BYTES`, default 30 MB;
  STEP header and end; exactly one solid; a coarse tet mesh), read by the CAD kernel in its own process,
  stored in the gold layout under `<runs>/cad/<upload>/`. Each BC set (generated with the gold rules, or
  picked on the 3D view) is a registered problem `<upload>_bc<k>` / `<upload>_m<k>`, tag `custom`, found by
  every process whose `FEACORE_RUNS_DIR` (or `FEACORE_CAD_DIR`) points there, and by ACE
  (`ace.config.TASKS` entry: scored QoI and node cap). Without a reference its runs are judged
  reference-free (`ace.infer`: ZZ estimate, agreement); the `cad.reference` tool or a finished P2 study
  (`POST /api/cad/problems/<id>/adopt`) certifies references with the gold rules.

## Tests and screenshots

```bash
python -m pytest tests/test_ui_api.py -q
python -m pytest tests/test_ui_tasks.py -q              # the task lane: tools, arguments, stop / kill, the offscreen launcher
python -m pytest tests/test_ui_agents.py -q             # Agents section: tools, folders API, one mock train/export/infer run
python -m pytest tests/test_ui_gold_cad.py -q           # Gold + Your CAD (temporary gold root, a STEP upload)
python -m pytest tests/test_ui_nonlinear_testdata.py -q # Nonlinear + Test data (fake run folders, one short ccx scoring run)
```

The tests use FastAPI's TestClient on a temporary store and cover every endpoint. That includes real solves through `POST /api/solve`, a real three-level study, a benchmark batch (run, CSV / JSON, cancel), submissions, consensus and the token auth. They skip if no 2D elasticity problem is registered or `skfem` does not import.

The screenshots were made with headless Chromium (Playwright) against a scratch store. `swarm_empty.png` shows the Swarm page before any ACE records exist. `swarm.png` and `consensus_ace.png` show real bridge records from the ACE loop's **mock-LLM** path: 3 generations × 3 agent runs on Lamé / LE10 (train) and LE1 (held out), written by `python -m ace.run_loop --mock --ui-runs-dir <dir>`. The loop and its solves are real; the LLM decisions are the mock's. `batch_*.png` show a real batch (8 fast 2D benchmarks + NAFEMS LE10); `study_live_le10.png` / `study_le10.png` show LE10 started with *Run with defaults*. The consensus screenshot shows **scripted test agents**: deterministic scripts (`agent_id` `scripted-*`) that mesh the CAD themselves, change the reference setup (P1 vs P2, other mesh sizes, two meshes only, a symmetry edge clamped by mistake) and solve through `POST /api/solve`. Every number is a real solve. None of them is an LLM agent.

## Known gaps

- Auth is a single shared token (optional), with no users, roles or rate limiting. Uploads are trusted; the size limit is whatever the proxy allows.
- A batch has no priority over single studies: everything shares the study lane in FIFO order. Cancel stops queued and running studies but keeps finished ones.
- Each job starts a fresh Python process (about 1 s overhead). That is fine for meshes that take seconds to solve, but wasteful for thousands of tiny solves.
- List endpoints scan the store directory on each call. That is fine up to a few thousand runs; past that, add an index.
- The viewer shows the boundary surface only (no cut planes or interior clipping in 3D). Axisymmetric results are shown as the 2D meridian section.
- The `nn` backend shows whatever its module provides. The UI treats it like any other backend.
