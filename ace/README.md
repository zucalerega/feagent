# ace: a minimal ACE loop for the FEA agent

A small, readable implementation of ACE (Agentic Context Engineering, Zhang et al., ICLR 2026,
arXiv 2510.04618) on top of `feacore`. The agent never changes; what improves is a **playbook**
of short lessons pasted into its system prompt.

## The loop in plain words

Each generation:

1. **Generator** (`agent.py`, Sonnet). Several copies of a small tool-use agent each attempt an
   FEA problem: `inspect_geometry` → `mesh(h, order, refinements)` → `solve(setup)` (repeat
   within the limits) → `submit(solve)`, which picks the solve that gets scored. The system
   prompt is a short, deliberately non-expert base prompt plus the current playbook. Every tool
   call has a `using` argument where the agent lists the bullet IDs it applied (citations as
   free text were never written in practice). Its reasoning summaries are kept in the trace.
2. **Checks** (`checks.py`, no LLM). The submitted solve is scored:
   `score = log10(e_uniform / e_agent)`. `e_agent` is the relative error of one scored quantity
   (QoI), computed by our solver from the agent's mesh and setup. `e_uniform` is the error of the
   best uniform mesh (linear or quadratic) with the same node budget. 0 = that, +1 = 10x more
   accurate. Gates score -2: no successful solve, mesh over the node cap, element quality < 0.1.
   A setup diagnostic (BCs that differ from the reference, sign flips, wrong formulation) is
   recorded for the Reflector; it can flag equivalent setups, so it is not a gate.
3. **Reflector** (`reflector.py`, Opus). One call per training problem: all runs ranked best to
   worst, with checks and condensed traces. Returns validated JSON: where best and worst
   diverged, root causes, 1-3 lessons ("When X, do Y, because Z.") with a scope, and a
   helpful/harmful/neutral vote on every cited bullet.
4. **Curator** (`curator.py`, Sonnet). Turns that into delta ops: `ADD` a bullet, or `VOTE` on
   one. It drops `one_off` lessons and checks each new lesson against the existing bullets in
   its section, so a near-duplicate becomes a vote instead of a new bullet. It only proposes.
5. **Merge** (`merge.py`, no LLM). Applies the ops, deduplicates again with rapidfuzz, prunes
   bullets where harmful ≥ helpful + 2, and saves the new playbook version plus a diff.

Generation 0 runs with an empty playbook as the baseline. The held-out problems run every
generation but are never reflected on, so their scores show whether lessons transfer:
`notched_bar_t10_r4` (same family as a training problem, different notch) and
`plate_hole_finite_width_d03` (a different shape).

The problems are stress concentrations whose references come from the team's own converged FE
studies (the plate with a hole uses a handbook value). We moved off the NAFEMS and Lamé problems
because the agent recalled their published answers and used them to decide when to stop.

Why the node cap: without it, a dense mesh is always more accurate, and nothing is learned.
With a cap, accuracy depends on skill: element order, where to refine, getting the BCs right.

## Run it

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev,ace]"

# free: canned LLM responses, exercises the whole loop
.venv/bin/python -m ace.run_loop --mock
.venv/bin/pytest tests/test_ace.py

# real: reads ANTHROPIC_API_KEY from the environment (never put it in the repo)
.venv/bin/python -m ace.run_loop                     # defaults: 6 runs/problem, 4 generations, $10 cap
.venv/bin/python -m ace.report ace_runs/<dir>        # the whole run as one HTML page
```

Useful flags: `--problems a,b` (train), `--holdout c`, `--concurrency 4`, `--top-n 30`,
`--playbook path.json` (start from an earlier playbook), `--out dir`, `--ui-runs-dir dir` (see
"See runs in the feacore UI" below).

The budget is checked before each agent run is started and after each Reflector call. It can
overshoot by the runs already in flight (at most `--concurrency` runs). Mock mode logs fake token
counts so the cost code is exercised; those dollars are not real.

## Watch it evolve

Live, while a run is going (refreshes itself every few seconds):

```bash
.venv/bin/python -m ace.watch ace_runs/<dir>                       # then open http://localhost:8765
.venv/bin/python -m ace.watch ace_runs/<dir> --export tracker.html  # a static snapshot to share
```

The tracker follows the team's whiteboard sketch: the shared playbook and what each generation
taught the next on the left; one lane per task with every agent's card in the middle (its score
and placing, error over its solves against the uniform-mesh baseline, a strengths/weaknesses
profile, which mesh attempts fit the node cap, which bullets it cited); and on the right where
each agent placed and how everyone did, generation by generation. Agents still running show
their mesh attempts so far.

Everything goes to `ace_runs/<timestamp>/` (gitignored):

| file | what |
|---|---|
| `summary.md` | per-generation score table (train vs held-out) and bullets added/voted/pruned |
| `scores.png` | mean score per generation, one line per problem |
| `playbook_gen{N}.json` | the playbook generation N's agents used |
| `diff_gen{N}.md` | what changed from N-1 to N, plus the rendered playbook |
| `gen{N}/reflection_<problem>.json` | the Reflector's analysis |
| `gen{N}/runs/<run_id>.json` | full run record: trace, meshes, solves, cited bullets, checks |
| `gen{N}/reflection_prompt_<problem>.md` | exactly what the Reflector was shown |
| `gen{N}/curator_ops.json`, `gen{N}/merge_changes.json` | the proposed ops and what merge did with them |
| `cost.jsonl` | one line per LLM call: tokens and estimated $ |
| `report.html` | after `python -m ace.report ace_runs/<dir>`: the whole run as one page |

## See runs in the feacore UI

The loop can also write every solve and every finished agent run to the feacore run store, so
they show up live in the feacore web UI (Swarm page). It is off unless you turn it on. On the
box that serves the UI, point the loop at the run store the UI reads (the UI service's default
is `~/feagent/runs`):

```bash
FEACORE_RUNS_DIR=~/feagent/runs .venv/bin/python -m ace.run_loop ...   # or --ui-runs-dir ~/feagent/runs
```

- Each solve becomes a run record (agent `ace:<run_id>`, source `ace`) with the case, the scored
  quantity and the fields for the viewer, or the error if the solve failed.
- Each agent run becomes a submission of its successful solves as soon as it finishes, with its
  failed gates, setup diagnostics, cited bullets and token counts. Its score (and the score's
  parts: accuracy, efficiency, ...) is filled in when the loop scores the generation, and updated
  each later generation, because the score is relative to every run on the problem so far. The
  diagnoser's verdict on a run is added to its diagnostics.
- The UI's loop id is the name of the output directory. With the bridge on, each run record in
  `gen{N}/runs/<run_id>.json` also gets `ui_submission_id`, and each of its solves `ui_run_id`.
- If the store can't be written (permissions, full disk), the loop only prints a warning and
  carries on; its own results don't change. Code: `ace/ui_bridge.py`.

## The CFD suite (steady laminar flow)

The same loop also trains agents on flow problems. Select them with `--suite cfd`, or name the
problems directly:

```bash
python -m ace.run_loop --suite cfd --mock                  # free, needs OpenFOAM (see below)
python -m ace.run_loop --suite cfd                         # real agents
```

| split | problem | scored QoI | node cap | what makes it hard |
|---|---|---|---|---|
| train | `poiseuille_channel` | `dp_AB` (exact, 1 %) | 1200 | easy: the scheme choice matters more than the mesh |
| train | `lid_driven_cavity_re100` | `u_vcl_min` (Ghia 1982, 3 %) | 2500 | closed cavity, lid-corner singularities |
| train | `taylor_couette_2d` | `u_theta_mid` (exact, 1 %) | 2500 | curved walls, moving-wall expression BC, closed domain |
| held out | `cylinder_channel_dfg_2d1` | `drag` (Schaefer & Turek 1996, 1 %) | 12000 | external flow: refine at the cylinder, not everywhere |

`taylor_couette_2d` is new in this branch. So is `square_duct_3d`, which is a benchmark but not a
task. Both are checked against the fixed solver in `docs/benchmarks_cfd.md`.

Every task is achievable under its cap: `planted.GOOD_MESH` holds a sensible mesh for each, and
`tests/test_diagnosis.py` checks that it passes every oracle. For example, the cylinder drag is
0.28 % off at 5440 nodes with refinement at the cylinder. With lighter refinement (3277 nodes) it
is 1.71 % off and fails the 1 % tolerance, so local refinement is what the agents have to learn.
The caps are safety limits, as in the elasticity suite. A uniform mesh at the cap with
linearUpwind meets the tolerance (`calibration.json`: Poiseuille 0.08 %, Taylor-Couette 0.36 %,
cavity 1.33 %), so the score rewards the cheapest accurate setup. Upwind is far worse on
Poiseuille: 3.6 % against 0.08 %.

**3D flow is not a task yet.** The agent's mesh tool makes unstructured tets in 3D. On those, the
backend's axial pressure gradient comes out 12-19 % high:

| case | mesh | pressure gradient |
|---|---|---|
| `pipe_poiseuille_3d` | tets, 4.6k nodes | +15 % |
| `pipe_poiseuille_3d` | tets split into hexes, 33k nodes | still +12 % |
| `square_duct_3d` | tets, 6.5-8k nodes | +12 % |

On the same tet mesh of the pipe, the flow rate is right (+0.6 %) and the wall force +3 %, so the
probed pressure and the wall shear no longer balance. 2D triangles are as accurate as quads, and
structured hexes converge (`docs/benchmarks_cfd.md`). Until that is resolved, or the mesh tool can
make structured or extruded hex meshes, a 2 % tolerance is out of reach for any agent in 3D.

What changes for flow (it is chosen from the problem's physics, so the two suites can be mixed):

- **Tools** (`TOOLS_FLOW` in `agent.py`):
  - `mesh` has no element order: cells are straight-sided finite volumes, and `quads=true` asks
    for quad-dominant 2D cells.
  - `solve` takes a fluid (`rho`, `mu`), a `convection` scheme (`upwind` or `linearUpwind`) and
    the flow BC types (`velocity_inlet`, `pressure_outlet`, `wall`, `slip`, `symmetry`). Values
    may be expressions in x, y, z.
  - The solver controls (laminar, iteration cap, residual target) come from the problem, not
    the agent.
- **Score**: unchanged. DOFs are cells × unknowns. The uniform-mesh baseline compares the two
  convection schemes, where elasticity compares the two element orders.
- **Oracles**: feacore's diagnostics run the elasticity solver, so equilibrium, singularity,
  sanity bounds and symmetry are reported as unavailable for flow.
  - These still run: spec compliance, gaming, the known answer, and the refinement study of the
    agent's own setup (three OpenFOAM solves).
  - The refinement estimate is pessimistic on independently remeshed unstructured meshes.
- **Diagnoser interventions** (`counterfactual.py`):
  - inlets and outlets count as loads, and walls, slip and symmetry as supports;
  - `convert_units` also takes `mu` or `rho`;
  - `set_convection` switches the scheme.

OpenFOAM is required: ESI line, v1912 or newer, as on the Brev box. On a Mac, run
`bash scripts/openfoam_docker.sh` once. It runs OpenFOAM v1912 in a Docker container and writes
shims. Then:

```bash
export FEACORE_OPENFOAM_BASHRC=~/.feacore_foam/bashrc TMPDIR=~/.feacore_foam/tmp
```

A two-generation mock run on the three 2D problems takes about 16 s that way. The solver tests in
`tests/test_ace_cfd.py` skip without OpenFOAM.

## Is the collective actually useful? Single-agent baselines

`ace/baselines.py` runs the same loop *without* the collective, and `ace/compare.py` puts every
run on one scale. Each baseline is `run_loop --memory shared+private --agents 1`: the swarm
member's own prompt, diagnoser, memory and scoring code, with no peers and nothing promoted.

| arm | command | what it isolates |
|---|---|---|
| swarm | `python -m ace.run_loop` (6 agents) | the system under test |
| matched | `python -m ace.baselines matched --swarm-agents 6` | one agent, 6 attempts per problem per generation: same runs as the swarm, no collective |
| isolated | `python -m ace.baselines isolated --swarm-agents 6 --jobs 3` | 6 independent agents, no channel between them: the value of sharing; also the spread between agents |
| single | `python -m ace.baselines single` | one agent, one attempt: the cheap baseline (1/6 of the runs) |

Give every arm the same `--problems`, `--holdout`, `--generations` and `--top-n`. Give matched and
isolated the swarm's `--budget-usd` (isolated splits it across its agents). Then:

```bash
python -m ace.compare --exp swarm=ace_runs/<swarm> --exp matched=ace_runs/<matched> \
    --exp isolated=ace_runs/<isolated> --exp single=ace_runs/<single>
```

How it works:
- **Setup check.** `run_loop` now writes `run_config.json`: problems, models, limits, diagnoser
  settings and a hash of the loop's code. `compare` refuses experiments whose setups differ in
  anything but the treatment (memory, agents, attempts), the run size and the budget.
- **Joint re-scoring.** The score is relative (efficiency compares a run with the cheapest
  in-tolerance run in the same output dir), so `compare` pools all arms' runs per problem and
  re-scores them together. Read differences, not absolute values.
- **Outputs.** `compare.md`, `compare.json` and `compare.png`. They contain:
  - the last generation's held-out score (each problem weighted equally) with a bootstrap 90%
    interval;
  - swarm minus each baseline, with its interval and P(swarm better);
  - the held-out score at matched spend;
  - learning curves against cumulative $ (`progress.json` now records spend per generation).
- **Run ids.** With `--attempts N` they look like `g0-<problem>-r1-a0`. The `-a<k>` suffix still
  routes lessons to the agent's notes.
- **Cost of isolated.** Equal agent runs is not equal spend: each isolated agent pays for its own
  diagnoser sessions. The report shows both runs and $, so compare at matched spend too.

## Knobs that matter (`ace/config.py`)

- `MODELS`: one line per role. The Reflector is the step most worth a strong model.
- `PRICES`: per-model $/MTok used for the cost log. **Verify before a real run.**
- `TASKS`: problem → scored QoI and node cap. `ace/calibration.json` caches `e_uniform`; it is
  recomputed automatically when a cap changes.
- `RUNS_PER_PROBLEM`: more runs = more contrast for the Reflector, less noisy scores.
- `GENERATIONS`: how many playbook updates you get to see.
- `TOP_N_BULLETS`: how much of the playbook the agent sees.
- `MAX_SOLVES`, `MAX_TURNS`: per-run limits (a stand-in for a wall-clock budget later).
- `EFFORT`, `MAX_TOKENS`: per-role thinking effort and output cap.

## Not done yet

- Mesh-refinement convergence check (GCI) as a diagnostic: TODO in `checks.py`.
- A rediscovered lesson counts as a helpful vote on the existing bullet. That rewards ideas the
  Reflector keeps finding, not ideas shown to help; cited-bullet votes are the stronger signal.
- The agent works on the problem's own named geometry, not raw STEP. Swapping its tools for
  the `feagent` mesh-harness MCP is the natural next step.
