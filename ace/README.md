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
