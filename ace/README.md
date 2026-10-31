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

### Swarm vs a single agent (the baseline)

`ace/single_loop.py` is the swarm's moving baseline: ONE big agent (config role `big_generator`,
Opus at max effort) does every task in one conversation each generation, with the whole swarm's
solve and turn budget, and learns through the same teacher, self-reflection and organizer (see the
module docstring for what is the same and what differs). Run it next to a `--deep` swarm on the
same tasks, then compare (base score against generations, cumulative $ and wall time):

```bash
ACE_SCORE=field .venv/bin/python -m ace.single_loop --mock --generations 2          # free
ACE_SCORE=field .venv/bin/python -m ace.single_loop --problems cooks_membrane,nafems_le11_axisym \
  --generations 20 --budget-usd 60 --out ace_runs/single1 --swarm-dir ace_runs/deep2
.venv/bin/python -m ace.compare ace_runs/deep2 ace_runs/single1                     # compare.md, compare.png
```

`run_loop --personas` gives agents a fixed exploration style: the flag alone = every agent,
`--personas a0,a3,a5` or `--persona-count 3` = only some (default: none). Who has which style is
written to `<out>/personas.json`.

### Use a trained swarm on new problems (export, then inference)

`ace.export_swarm` copies what a finished `--deep` run (or `single_loop` run) has learned into a
small folder: the final shared playbook, every agent's private notes, the personas, each agent's
last teacher feedback / reflection / run summary (`state.json`), a `manifest.json` (run, generation,
tasks, agents, models, git commit) and `context/<agent>.md`, the context each agent gets on a new
problem. No meshes, no traces. The final generation is the last one whose notes were saved, so a
run stopped mid-generation still exports its most trained state.

`ace.infer` runs the trained agents on any feacore problem (trained on or not; e.g. a NAFEMS case
or a gold part), one process per agent, with that context and the training tools and agent loop.
Nothing learns: no teacher, reflection or organizer. It writes `summary.md` and `results.json`: per
agent the submitted solve's statement quantities, mesh, ZZ error estimate and mesh/stress images;
a consensus (median and spread of each quantity across agents, and the agent with the smallest own
error estimate). When the problem has references and field caches (`ace/field_refs/`), each run is
also scored in field mode.

```bash
.venv/bin/python -m ace.export_swarm ace_runs/scale_s8 swarms/scale_s8
.venv/bin/python -m ace.infer --swarm swarms/scale_s8 --problem abc_00000065_bc0 --agents a0,a3 --mock  # free
.venv/bin/python -m ace.infer --swarm swarms/scale_s8 --problem nafems_le1 --out ace_runs/infer_le1      # real
```

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
python -m ace.run_loop --suite cfd3d                       # real agents, 3D (slower solves)
```

| suite | split | problem | scored QoI | node cap | what makes it hard |
|---|---|---|---|---|---|
| `cfd` | train | `poiseuille_channel` | `dp_AB` (exact, 1 %) | 1200 | easy: the scheme choice matters more than the mesh |
| `cfd` | train | `lid_driven_cavity_re100` | `u_vcl_min` (Ghia 1982, 3 %) | 2500 | closed cavity, lid-corner singularities |
| `cfd` | train | `taylor_couette_2d` | `u_theta_mid` (exact, 1 %) | 2500 | curved walls, moving-wall expression BC, closed domain |
| `cfd` | held out | `cylinder_channel_dfg_2d1` | `drag` (Schaefer & Turek 1996, 1 %) | 12000 | external flow: refine at the cylinder, not everywhere |
| `cfd3d` | train | `pipe_poiseuille_3d` | `dp_AB` (exact, 2 %) | 6000 | 3D tets; the faceted wall loses area at O(h^2) |
| `cfd3d` | train | `square_duct_3d` | `dp_AB` (exact developed, 2 %) | 6000 | uniform inflow develops first; 3D tets |

`taylor_couette_2d` and `square_duct_3d` are new in this branch. Both are checked against the fixed
solver in `docs/benchmarks_cfd.md`. The suites can be mixed with `--problems` / `--holdout`.

Every task is achievable under its cap: `planted.GOOD_MESH` holds a sensible mesh for each, and
`tests/test_diagnosis.py` checks that it passes every oracle. For example, the cylinder drag is
0.37 % off at 5440 nodes with refinement at the cylinder. With lighter refinement (3277 nodes) it
was 1.71 % off (measured before the wall-distance fix in the force evaluation) and fails the 1 %
tolerance, so local refinement is what the agents have to learn.
The caps are safety limits, as in the elasticity suite. A uniform mesh at the cap with
linearUpwind meets the tolerance (`calibration.json`: Poiseuille 0.08 %, Taylor-Couette 0.36 %,
cavity 1.33 %, 3D pipe 1.50 %, 3D duct 0.84 %), so the score rewards the cheapest accurate setup.
Upwind is far worse: 3.6 % against 0.08 % on Poiseuille, and 6.2 % and 6.7 % on the pipe and the duct.

**3D flow (`cfd3d`).** The agent's mesh tool makes unstructured tets in 3D. With OpenFOAM's plain
schemes, those converged to an axial pressure gradient 12-19 % high at any mesh size, so 3D was left
out at first. Next to a wall, linear interpolation between cell centres misses the face values in
the same direction everywhere, and that biased the viscous flux.

The backend now uses skew-consistent schemes in 3D: `pointCellsLeastSquares` gradients and a
skew-corrected momentum-interpolation flux. Wall forces now use the wall-normal distance; before,
the force read 10 % low, which made the old results look like a force imbalance. Details are in
`docs/solver_openfoam.md`, "3D meshes".

| case | mesh | dp_AB before → after |
|---|---|---|
| `pipe_poiseuille_3d` | uniform tets, 4.6k nodes | +15.3 % → +1.3 % |
| `pipe_poiseuille_3d` | tets split into hexes | +20 % → +0.7 % |
| `square_duct_3d` | uniform tets, 4.3k / 10k nodes | +11.7 % → −0.8 % / +0.0 % |

A uniform tet mesh below the 6000-node cap meets the 2 % tolerance (`planted.GOOD_MESH`), so the
score is again about the cheapest accurate mesh. For the pipe that means fewer nodes away from the
wall; for the duct, coarser cells in the developed flow.

- **What is still weak in 3D:** point *velocities* at a smooth maximum (`u_centreline`, `u_axis`)
  read 2-5 % low on coarse tets. The probe's vertex fit flattens the peak; the solution itself is
  within 1 %. That is why both 3D tasks score `dp_AB`.
- **Cost:** a 3D solve takes 5-30 s and up to about 250 MB per 70k tets. The refinement-study oracle
  refines 3D flow meshes by 1.5 rather than 2, so its fine level stays near 3.4x the agent's nodes.

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

### Running the CFD suites with the Claude API (real credits)

On a Brev box (Ubuntu, after `bash scripts/setup_gpu_box.sh` or `bootstrap.sh`):

```bash
bash scripts/setup_ace_cfd.sh            # OpenFOAM (apt), the ace + cfd extras, pre-flight solves
export ANTHROPIC_API_KEY=...             # never commit it
.venv/bin/python -m ace.preflight --suite cfd --ping     # one real call (~$0.01): key, model, tool schemas

# a small first run to measure the cost per run on this suite
.venv/bin/python -m ace.run_loop --suite cfd --generations 1 --agents 2 --budget-usd 3 --out ace_runs/cfd_probe

# the experiment (defaults: 6 persistent agents, 4 generations, shared + private memory)
tmux new -s ace
.venv/bin/python -m ace.run_loop --suite cfd --budget-usd 40 --out ace_runs/cfd_$(date +%m%d)
.venv/bin/python -m ace.run_loop --suite cfd3d --budget-usd 25 --out ace_runs/cfd3d_$(date +%m%d)
```

- **Pre-flight** (`ace/preflight.py`) solves every task's `planted.GOOD_MESH` with the reference
  setup and checks it is within tolerance and under the cap. It needs no API key unless `--ping`.
  `run_loop` itself also stops before spending anything if a flow task cannot find OpenFOAM or
  foamlib.
- **Budget:** `--budget-usd` defaults to $10, which stops the CFD suite after about one generation.
  The loop checks it before each agent run, so the last generation can overshoot by
  `--concurrency` runs.
  - Rough estimate, not yet measured on these suites: $0.2-0.4 per agent run with the Sonnet
    generator. A `cfd` generation (6 agents x 4 tasks) is then about $5-10, and `cfd3d`
    (6 x 2) about half that.
  - The probe run's `Spend by role` line and `cost.jsonl` give the real figure; verify `PRICES` in
    `config.py` first.
- **Modes that work for flow:** the default point scoring (leave `ACE_SCORE` unset), `--teacher rerun`
  (the default), `--memory shared|shared+private`, `--split`, `--peers`, `--personas`.
- **Modes that don't:** `--deep`, `--teacher inspect|both` and `ACE_SCORE=field` score the elastic
  stress field, and `--statement requirements` writes elasticity setups. `run_loop` refuses them
  for flow tasks.
- **Concurrency:** `--concurrency` (default 4) agent runs solve in parallel, and as many scoring
  workers start on finished runs while the others still work. 2D solves are light. A 3D solve and
  its refinement study can take 0.3-0.5 GB each, so keep `--concurrency` at most (RAM in GB) / 4
  for `cfd3d`.
- **Watching and stopping:** `.venv/bin/python -m ace.watch ace_runs/<dir>` (port 8765; tunnel it from
  the Brev box) and `.venv/bin/python -m ace.report ace_runs/<dir>` (see "Watch it evolve").
  `touch ace_runs/<dir>/STOP` ends the run cleanly before the next generation.

## Requirements instead of boundary conditions (`--statement requirements`)

By default an agent gets the problem statement, which spells out the supports, loads and analysis
type. With `--statement requirements` it gets the problem's **engineering requirements** instead,
and has to set the boundary conditions and the analysis type itself. The requirements say:
- what the part is and what the CAD holds (a quarter, a meridian section, a thin plate);
- its material;
- how it is loaded in service;
- how it is held: welded, bolted, between frictionless platens, on a knife-edge;
- what is symmetric;
- the result wanted.

```bash
python -m ace.run_loop --statement requirements ...          # agents set the BCs themselves
python -m ace.requirements show cooks_membrane               # the task prompt an agent gets
python -m ace.requirements check                             # lint + staleness of every stored file
python -m ace.requirements check --round-trip                # + an agent-like model reads them back (API)
python -m ace.requirements generate --problems a,b --force   # (re)write with the translator (API)
```

- **What changes.** The task prompt (`agent.task_prompt`) shows the requirements and a neutral
  task label instead of the problem id; the id names the benchmark, and a model could recall its
  setup. Groups whose names give a support or load away get neutral names everywhere, through
  `fea.get_problem`: Cook's `clamped`/`loaded` become `left_edge`/`right_edge`, and the `sym_*`
  groups become `face_x0`/`face_y0` (LE11 3D) and `edge_x0`/`edge_y0` (Lamé). The reflector, diagnoser and deep teacher read the same task
  prompt, so their lessons are about turning requirements into a setup.
- **What does not change.** Scoring, gates and `checks.bc_report`, which compares the agent's setup
  with the reference setup and feeds the teachers.
- **Where the requirements live.** One JSON per problem in `ace/requirements_data/`:
  - the agent-facing text;
  - `group_aliases`;
  - `mapping`: which requirement implies each reference item, for reviewers only;
  - `source`, with a fingerprint of the statement and reference setup they were written from;
  - `validation`.
- **How `generate` writes them** (`ace/requirements.py`):
  1. The translator model rewrites the statement and reference setup under rules that forbid
     boundary-condition language.
  2. A deterministic lint rejects BC or solver words (`fixed`, `clamped`, `traction`, `zero
     x-displacement`, `plane stress`, ...), benchmark names, CAD group identifiers, a missing
     number of the reference setup, telltale or colliding aliases, and a stale fingerprint.
  3. A round trip: the agents' model reads only what an agent sees and proposes a setup, three
     times. Each proposal is solved next to the reference setup on the same mesh, so equivalent
     setups pass (force vs force-per-area, fixed vs zero displacement).
  4. Failures go back to the translator for up to two re-drafts.
- **The loop refuses to start** in requirements mode when a problem has no file, or its file fails
  the lint.
- **Current files.** The six `TASKS` problems are written and lint clean. They were written in a
  session without an API key, so their round trip has not been run yet (`validation.round_trip:
  null`): run `python -m ace.requirements check --round-trip` once a key is available.

## Physical test data (`--evidence`)

Raw data from a physical test (e.g. an Instron pull of a printed gold part) can be given to the loop at
any time, including after the population has settled on a consensus. The loop compares every run with
it and shows the comparison to the agents and the teachers, so they learn from it. Code:
`ace/evidence.py`.

```bash
python -m ace.evidence summarize lab/abc3_pull_01.json            # check the export reads and fits
python -m ace.evidence summarize lab/abc3_pull_01.json --predict  # + the problem's own setup vs the test
python -m ace.run_loop --problems abc_00000003_pull --evidence lab/ ...      # data from the start
cp -r lab/ <out>/evidence/                                        # or drop it in while the loop runs
python -m ace.run_loop ... --evidence lab/ --evidence-score       # score against the test, not the reference
```

**The test spec.** One small JSON per test, next to the machine's export:

```json
{"test_id": "abc3-pull-01", "problem_id": "abc_00000003_pull",
 "description": "Rigid 10K, UV cure 60 min at 70 C; V-jaw wedge grips over the added 40 mm; 5 kN cell; video extensometer dots at x = -27 and +27 mm on the stub fronts; 3 cycles to 300 N",
 "data": {"file": "abc3_pull_01.csv", "load": "Load", "displacement": "Extension (Video)"},
 "measurement": {"qoi": "ext_marks"},
 "uncertainty": {"geometry": 0.05, "instrument": 0.01},
 "material": {"E": 9.7e9, "source": "3 D638 dogbones, same batch and cure"}}
```

- **Columns:** named as in the export's header, with units from the header or the units row below
  it. Comma, semicolon or tab delimiters and decimal commas all work. A strain channel plus
  `gauge_length_mm` works too.
- **Measurement:** a QoI of the problem (default: its scored QoI), or an explicit `point_difference`
  / `point` / `mean` QoI.
- **Material:** a measured modulus rescales every prediction to it, so the comparison tests the
  modelling and not the datasheet. The material counts as 2% uncertainty with a measured modulus,
  5% without one.

**What the loop does with it**, at the start of every generation:
1. It reads the export, splits it into load cycles and fits the slope of measurement vs load (the
   compliance) between 10% and 90% of each loading's peak. A first loading that differs from the
   others by more than 2% is left out as grip seating. It reports how much the curve bends (and
   whether that is more than noise) and what is left after unloading.
2. It builds the uncertainty budget: scatter between cycles, fit, instrument, material and
   geometry.
3. It compares each run's scored solve with the test, in percent and in standard deviations. When
   the measurement is not the scored quantity, it re-solves the run's own mesh and setup.
4. It computes the population's consensus (the median of the latest generation) and how far it is
   from the test.
5. Agents see the test and the consensus gap in their task prompt, plus a `test_check` in every
   solve result.
6. The diagnoser, reflector and deep teacher get the cycles, the curve, the budget and every run's
   gap, and are asked to separate modelling causes from the test's own uncertainty.
7. With `--evidence-score`, problems with a test are scored against it. The tolerance is at least
   2 sigma of the test. Earlier runs are re-checked and re-scored when the data arrives.

Processed evidence is kept in `<out>/evidence_active/`; each run's comparison is in
`checks["evidence"]`.

### The pull test of the printed gold part: `abc_00000003_pull`

`feacore/problems/physical.py` defines this problem: the gold part `abc_00000003` with both stubs
lengthened by 40 mm for wedge grips (the print files of `physical_tests/` on the
`physical-validation` branch).
- **Setup:** the −x stub is held, the +x stub is pulled with 300 N along x and held sideways,
  Rigid 10K (E = 10 GPa, ν = 0.35).
- **Scored quantity:** `ext_marks`, the x-elongation between video-extensometer marks at x = ±27 mm
  on the front of the original stubs (y = 7.656 mm, z = 30.588 mm), a 54 mm gauge length.
- **Reference:** 4.733e-5 m at 300 N (0.158 µm/N), from a P2 mesh study (h = 5 to 1.8 mm,
  Richardson; a 4-mesh fit agrees to 0.08%); tolerance 3%.
- **Meshing it:** the 2.5 mm tube wall bends, so it needs small elements all around the tube,
  not at one point. Uniform 2.2 mm gives −2.3% at 40k nodes (the cap is 60k). Grading from 4 mm
  to the 1.65 mm size floor over the tube gives −2.75% at 23k nodes.

Two consequences for the lab:
- **Use the video extensometer.** The signal is about 47 µm at 300 N, which the crosshead cannot
  resolve.
- **Stay at or below 300 N for the elastic runs.** The highest stress away from the grips is at
  the sharp edge where each boss shoulder meets its stub: 21.7 MPa at 500 N on the finest mesh,
  and still rising with refinement. Marks on the bosses were rejected because they sit next to
  that edge and converge erratically.

### The compression test of the printed C-bracket: `abc_00000065_compression` (no reference)

The gold part `abc_00000065` printed in Formlabs Alpha Fast (2 min post-cure) and crushed between the
Instron's platens: the bottom face (f33) sits on the lower platen, the upper platen pushes the two prong
tops (f17, f34). `feacore/problems/physical.py` states it as the bottom held and 1000 N spread over the
prong tops, E = 2230 MPa (datasheet), ν = 0.35 (assumed). Scored quantity: `u_top`, the mean
y-displacement of the prong tops.
- **There is no reference solution.** The test is the only check: every run is scored against it, with or
  without `--evidence-score`, and the loop refuses to start on a problem that has neither a reference nor
  test data. In field mode the gain is measured the same way: the automesher's error against the test
  over the run's. There is no be-the-best bonus: the hidden probe points judge the stated setup's field,
  not the test.
- **The data** (`physical_tests/abc_00000065_compression/`): the Bluehill reports had no data export, so
  `scripts/digitize_bluehill_plot.py` traced the curve in the report's plot image (±0.01 mm, ±15 N; the
  peak and break, hidden by markers, read 1-3% low). The spec fits the straight part after a toe, up to
  the peak: 2696 N/mm (±12% at 1 sigma, mostly the material value and the crosshead).
- **Swarm against single agent, with a second task** (`--split` now also runs on one task):

```bash
ACE_SCORE=field python -m ace.run_loop --deep --split --memory shared+private --agents 8 \
    --problems abc_00000065_compression,abc_00000065_bc0 --holdout "" --generations 8 \
    --evidence physical_tests/abc_00000065_compression --budget-usd 50 --out ace_runs/cbracket_swarm
ACE_SCORE=field python -m ace.single_loop --problems abc_00000065_compression,abc_00000065_bc0 \
    --generations 8 --evidence physical_tests/abc_00000065_compression --budget-usd 20 \
    --out ace_runs/cbracket_single --swarm-dir ace_runs/cbracket_swarm
python -m ace.compare ace_runs/cbracket_swarm ace_runs/cbracket_single
```

**Past yield: `scripts/ccx_compression_ramp.py`.** The loop's solver is linear. This script ramps the part
on feacore's opt-in elastoplastic CalculiX path (J2 plasticity, NLGEOM; from the `nonlinear-setup`
branch): the prong tops pushed down by a rigid platen, the bottom held. It uses two calibrations of
the same dogbone (`dogbone1.pdf`, an ASTM D638 Type I, 13 x 3.2 mm, measured by crosshead):
- **team:** `experiments/material/alpha_fast_dogbone1.json`, strain = crosshead / 115 mm, so the
  modulus is the apparent 1880 MPa;
- **corrected:** the datasheet modulus, with the grip slack (the initial slope minus the printed
  geometry's own compliance) taken out, and the plastic stretch put in the narrow section.

It writes `physical_tests/abc_00000065_compression/ccx_ramp/` (curves, `summary.json`, `compare.png`)
against the test.

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
