# feagent: a swarm of AI agents that learns to set up engineering simulations

![Three simulations of a real compression test](talk_plots/act3_real_data/break/three_way_break.gif)

*A 3D-printed bracket was crushed on an Instron, and three simulations replay that test. Left: our team's first
nonlinear simulation. Middle: before post-training, 33 % off the measured force curve, with an early crack
predicted on a prong. Right: after post-training, **2.0 % off**. Nobody told the simulation where the real part
broke. Its highest tension (85 % of the resin's strength) builds up on the lower inner bore at 2.51 mm, and that is
exactly where, and when, the real part first cracked (2.51 mm, 4.76 kN).*

A finite-element simulation is only as good as its mesh and its setup, and experts still make both by hand.
**feagent** trains a swarm of Claude agents to do this job. The agents mesh real CAD parts. A fixed solver checks
their work against answers they never see. Every round, they learn from a teacher, from their own mistakes and
from each other. No model weights change: what learns is each agent's notebook of lessons.

We report results as **"x the automesher"**: a score that rewards accuracy and penalizes mesh size, where the default
gmsh mesh scores 1x. Mesh size is counted in **DOFs**, the number of unknowns the solver computes.

**In three results:**

1. **Agentic meshing beats the gmsh automesher, and training works.** On Cook's membrane (a standard 2D test: a
   tapered plate clamped at one end and loaded at the other), a trained agent is **3.4x more accurate** than the
   automesher at the same number of DOFs, or as accurate with **10.8x fewer DOFs**. Over 10 rounds of training,
   the average agent goes from 1.3x to **4.4x** the automesher.
2. **Swarms train better, and scale.** On a real CAD part, more agents trained together reach better meshes:
   **16 > 8 > 4 > 2** agents. On that part, 8 agents also beat one agent given the same solves, turns and time.
   The cost per round grows more slowly than the team size.
3. **The harness learns from real lab data.** From one physical test, we fit the test conditions and post-train
   the swarm on it. The simulation of a printed part goes from **33 % to 2.0 %** off the measured force curve,
   and it finds the spot where the real part cracked.

---

## How it works

![How the swarm trains](talk_plots/schematic/training_system.png)

- **Each agent** is Claude Sonnet, with tools to inspect the geometry, mesh, solve, look at its result, compare
  attempts and submit. The model never changes. Before every attempt the agent reads its notebook: what to try
  this round, feedback on its last try, its own notes and the team's notes.
- **The swarm is decentralized.** Every agent keeps its own private notes and runs its own experiment each round.
  The agent with the best mesh so far is the *control* and re-runs the best meshing recipe. *Variants* each change
  one factor. The lowest-ranked agents are *explorers* and try a new strategy.
- **The agents compete.** Meshes are ranked against the automesher. The score is also checked at 8 hidden points
  the agents never see. At each one, only the most accurate mesh earns a small bonus, and a tie gives nobody the
  bonus, so copying the leader earns nothing. A second bonus goes to a structurally different recipe that scores
  within 90 % of the best.
- **The agents collaborate.** A shared team notebook collects the teacher's lessons and the best attempts'
  highlights, and every agent sees what worked for the others and how to copy it.
- **Every round**, a teacher (Claude Opus) reviews all attempts, each agent reflects on its own, and an organizer
  rewrites every notebook.
- **We built our own benchmark.** We took real third-party CAD from the ABC dataset, generated supports and loads,
  solved nested meshes on an H100 and certified the reference answers with convergence studies. The agents get
  only the task. The answers are hidden and used only for scoring. On top of that we ran our own lab test: a
  3D-printed part crushed on an Instron.

---

## Results

### 1. Agentic meshing works, and training works

![Cook's membrane at an equal DOF budget](talk_plots/equal_budget/fig_cook_equal_stress.png)

**Same budget, 3.4x more accurate.** This is Cook's membrane at about the same number of DOFs:
- the automesher, refined to 734 DOFs: 3.63 % off;
- an early agent from a separate run without the teacher (898 DOFs): 2.90 %;
- the trained agent (710 DOFs): 1.07 %.

| | |
|---|---|
| ![](talk_plots/equal_budget/fig_cook_same_budget.png) | ![](talk_plots/act1_agentic_works/fig4_mesh_before_after.png) |
| Error maps at the same budget. The automesher's error sits in the two corners. The trained agent shrinks it to a few tiny elements there: its worst 5 % of elements hold 17 % of its squared error, against 90 % for the automesher. | The trained agent grades the mesh into the corners: 5.09 % before training (282 DOFs), 1.07 % after (710 DOFs). |
| ![](talk_plots/act1_agentic_works/fig1_learning_curves.png) | ![](talk_plots/act1_agentic_works/fig3b_equal_cost_equal_accuracy.png) |
| Training works: the average agent climbs from 1.3x to 4.4x the automesher in 10 rounds. | To match the best agent, the automesher needs up to 10.8x the DOFs. |

![Headline scores on every benchmark](talk_plots/act1_agentic_works/fig3_headline_scores.png)

### 2. Swarms train better, and this scales

| | |
|---|---|
| ![](talk_plots/act2_swarms_scale/fig3_scaling_bc0_max6.png) | ![](talk_plots/act2_swarms_scale/fig7_training_bc0_max6.png) |
| More agents, better swarm: with 2 / 4 / 8 / 16 agents the best mesh reaches 2.40x / 2.56x / 3.01x / 3.10x the automesher (rounds 0-5). | Bigger swarms improve faster. The 8- and 16-agent swarms have not plateaued, while 2 agents stall at 2.40x from round 2. Dashed lines are projections, not runs. |
| ![](talk_plots/act2_swarms_scale/fig5_swarm_vs_single.png) | ![](talk_plots/act2_swarms_scale/fig4_mean_agent_per_gen.png) |
| 8 agents beat one agent given the same solves, turns and time: the best per round, averaged over rounds 1-7, is 2.92x against 2.37x the automesher. | It is not just more lottery tickets: the average agent learns too, from 2.2x to 2.9x with 8 agents. |

![Cost per generation](talk_plots/act2_swarms_scale/fig6_cost_per_gen.png)

Scaling is affordable: going from 2 to 8 agents multiplies the cost per round by 3.3, not 4.

### 3. The harness learns from real lab data

![Setup ladder](talk_plots/act3_real_data/fig3_setup_ladder.png)

**From 33 % to 2.0 % off the real test: two 4x gains.** The measured force curve of the printed C-bracket
peaks at 4,755 N.
1. Our original simulation is 32.7 % off it.
2. **Test conditions, 4.2x.** We fitted the real test conditions to the measured curve: the platens don't let the
   part slide, the part settles in before loading, and the machine itself flexes. With them, the automesher's mesh
   is 7.8 % off.
3. **The swarm's meshes, another 4.0x.** The swarm trained in Result 2 brings it to 4.6 %. Post-training on this
   test gets 2.0 % (10 rounds, $32): the teacher first rewrites the notebooks from the test brief. The most
   accurate mesh appears in the first round, and later rounds hold about 2.2 % with 30 % fewer nodes.
4. **The crack site.** Nobody told the simulation where the part broke. The post-trained simulation's highest
   tension builds up on the lower inner bore, where the real part cracked, at the same travel (2.51 mm).
   Before post-training, the simulation predicted a crack on a prong at 1.76 mm, far too early and in the wrong
   place.

| | |
|---|---|
| ![](talk_plots/act3_real_data/fig1_measured_curve.png) | ![](talk_plots/act3_real_data/fig2_raw_solver_vs_test.png) |
| The real test: seating, a linear climb, yielding to 4.8 kN, then cracking. | Before post-training, the simulation is 33 % off the real test. |
| ![](talk_plots/act3_real_data/fig5_curves_vs_test.png) | ![](talk_plots/act3_real_data/fig7_post_training_progress.png) |
| The post-trained mesh tracks the real test: 2.0 % off, against 7.8 % for the automesher. | Post-training: the best score goes from 2.7x to 3.6x the automesher, and the average from 2.3x to 3.5x. |
| ![](talk_plots/act3_real_data/fig4_error_vs_nodes.png) | ![](talk_plots/act3_real_data/fig6_platen_renders.png) |
| Post-trained meshes: the most accurate is 2.0 % off and the best-scoring 2.3 %. The last round's four meshes are 2.2-2.3 % off. | Platen friction carries the strength: without it the prongs splay. |

![Crack onset vs the real crack](talk_plots/act3_real_data/break/break_table.png)

More: the [videos](talk_plots/act3_real_data/videos.html) and the [crack-prediction page](talk_plots/act3_real_data/break/index.html).
Every figure has its caption and data in [`talk_plots/`](talk_plots/README.md), listed in [slide order](talk_plots/SLIDES.md).

### Limitations and what's next

- **One run per configuration.** This was a hackathon, so there are no repeats or confidence intervals. Run-to-run
  noise is about 0.1-0.4x on the "x the automesher" scale.
- **Result 1:** the gains are largest on 2D problems. On Cook's membrane the average agent goes from 1.3x to 4.4x,
  while 3D LE11 stays flat. Sustained learning needs the full loop, where Claude Opus reviews every attempt.
- **Result 2:** the 32- and 64-agent points and the curves past the last measured round are projections from
  fits, not runs.
- **Result 3:**
  - **A strong teacher helped.** The swarm learned real things about meshing this part, but in post-training it
    leans on the Claude Opus teacher, which reads the test brief and rewrites every notebook.
  - **The force at buckling is still well off.** The material model comes from a tensile dogbone test, and adapting
    it to how this part buckles and cracks under compression is an open question.
  - **Crack onset only.** The simulation predicts where and when a crack starts; it does not model the fracture
    itself.

---

## Technical description

### Repository layout

| Path | What it is |
|---|---|
| [`ace/`](ace/README.md) | The learning loop: agents, scoring, teacher, self-reflection, organizer, memory, export and inference. |
| `feacore/` | The fixed solver and everything around it: scikit-fem (the solver of record), CalculiX (cross-check and nonlinear), verification (GCI, MMS, ZZ), the problem registry, the UI and an OpenFOAM flow backend. |
| `feacore/goldgen/` | The gold-benchmark generator: ABC CAD → load cases → nested tet10 meshes → GCI-certified references, run on an H100. |
| [`benchmarks/`](benchmarks/README.md) | Benchmark problems, each with `public/` for agents (statement + STEP) and `private/` for the evaluator (references). The gold set is in [`benchmarks/gold/`](benchmarks/gold/README.md). It also holds the verification, SimJEB, code_aster and CalculiX suites. |
| `physical_tests/` | Real test data for the Instron compression test of the printed C-bracket (`abc_00000065_compression/`): the force curve, the dogbone curve of the same resin, the frozen nonlinear scoring spec and the teacher brief. |
| `results/trained_swarms/` | Exported trained swarms (notebooks, personas, roles), ready for inference: `div_s1` … `div_s16`, `nl_ft`. |
| [`talk_plots/`](talk_plots/README.md) | Every figure and video in the results, with captions, data files and the scripts that make them. |
| `src/feagent/` | The YAML-spec gmsh mesh harness: STEP → a named, quality-gated mesh. |
| `scripts/` | Setup, baselines, figure and video scripts, and the gold pipeline. |

### The learning loop (`ace/`)

The loop implements Agentic Context Engineering (ACE, Zhang et al., ICLR 2026) and extends it to a multi-agent
swarm. The Result 2 and 3 runs and the main Result 1 run (`deep1`) use the `--deep` mode. Some Result 1 numbers
come from the basic loop with no teacher: the best LE1 run (`split1`), the best 3D LE11 run (`le11`), and the early
agent in the Cook equal-budget figure (`cook_auto`).

1. **Generate.** Persistent agents (`a0`, `a1`, …, Claude Sonnet) each mesh their task, within a per-task budget
   of solves and turns. Their tools are `inspect_geometry`, `mesh(h, order, refinements)`, `solve`, `view`,
   `compare` and `submit`, plus `nonlinear_check` on nonlinear tasks.
2. **Score** (`ace/checks.py`, no LLM). In field mode, *e* is the larger of two errors, with errors below 0.5 %
   counted as 0.5 %:
   - the energy-norm error of the whole stress field against a fine reference solve;
   - the RMS von Mises error at 12 probe points.

   With g = e_auto / e and c = DOFs / DOFs_auto, the accuracy term is A = (g/2)² if g < 2 and g/2 otherwise, and
   the score is A / c. The gmsh default mesh scores 0.25, which the figures show as 1x the automesher.

   Two bonuses are added to the score:
   - **Be the best.** 4 of the 12 probe points are disclosed and 8 are hidden. Each hidden point pays 1/8 of a
     0.25 bonus to the single most accurate run, and a tie pays nobody. It is not paid on the nonlinear task.
   - **Be different.** +0.1 for a structurally different recipe within 90 % of the best base score. This bonus
     was added after `deep1`; the Result 2 and 3 runs use it.
3. **Teach** (`ace/deep_teacher.py`, Claude Opus). The roles (control / variant / explorer) and each variant's
   factor are fixed in code from the ranking. The teacher reviews every run from its numbers, error maps,
   pictures, reasoning and tool calls. It writes each agent a critique and its next experiment in its role, plus
   cross-run lessons and highlights.
4. **Reflect** (`ace/self_reflect.py`). Each agent critiques its own run. Hidden check points are shown only in
   aggregate, so agents cannot learn the test.
5. **Organize** (`ace/organizer.py`). An organizer merges, rewrites and orders each agent's private notes and the
   shared playbook. Then `ace/process_brief.py` builds every agent's next context.

`ace.export_swarm` saves a trained swarm, and `ace.infer` runs it on new problems without learning.

### Real-data post-training (Result 3)

The task `abc_00000065_compression` is scored against the measured Instron curve, not a reference solve
(`ace/nonlinear.py`, `physical_tests/abc_00000065_compression/nonlinear_spec.json`). Every mesh goes through one
frozen elastoplastic CalculiX ramp:
- J2 hardening fitted to a dogbone test of the same resin, with geometric nonlinearity (NLGEOM);
- sticking platens;
- the crosshead mapped as u = u_part + 0.2335 mm + 2.898e-5 mm/N · F. The seating gap and machine compliance in
  this mapping were fitted once on a fine mesh against the measured curve, then frozen.

The error is the mean force error at checkpoints from 0.5 to 2.5 mm. Post-training starts from 4 agents of the
trained 8-agent swarm (`--init-from`). `--pre-teach` lets the teacher adapt their notebooks to the test brief, and
10 rounds follow.

### Solver and verification (`feacore/`)

- **Solver input:** a case JSON (analysis, materials, BCs, loads, QoIs) plus a gmsh `.msh` with named groups.
  Run it with `python -m feacore.cli case.json --out result.json`.
- **Backends:** scikit-fem is the solver of record (CPU, with an optional NVIDIA cuDSS direct solve on GPU).
  CalculiX is an independent cross-check and runs the elastoplastic J2/NLGEOM path.
- **Verification:**
  - Richardson extrapolation and GCI (Celik et al. 2008);
  - method-of-manufactured-solutions code checks;
  - a Zienkiewicz-Zhu error estimate.

### Our benchmarks

- **Gold benchmarks** (`feacore/goldgen`, `benchmarks/gold/`). The pipeline:
  1. Screen ABC CAD parts and rescale them to a 100 mm diagonal.
  2. Generate seeded supports and loads (1-50 kN or 1-50 MPa).
  3. Solve nested tet10 meshes L0-L2, up to 1M DOFs, on an H100.
  4. Keep only the QoIs whose GCI certifies them: 1 % for energy, 2 % for displacement or reaction, 3 % for stress.

  For field scoring, the reference fields are refined adaptively beyond L2 (`ace/gold_reference.py`).
- **Classical benchmarks:** NAFEMS LE1, LE10 and LE11 (axisymmetric and 3D), Cook's membrane and Lamé. These are
  standard textbook FE test problems.
- **Physical tests:**
  - Instron compression of a printed C-bracket (Formlabs Alpha Fast) and a dogbone test of the same resin.
  - A pull test of a printed gold part is defined as a problem (`abc_00000003_pull`), but its data is not in the
    repo.

### Run it

```bash
# setup (laptop)
python3.13 -m venv .venv && .venv/bin/pip install -e ".[ace]" pytest
export ANTHROPIC_API_KEY=...          # environment only, never in the repo
# setup (fresh Ubuntu / Brev box, with CalculiX and GPU extras); then add the ACE dependencies
bash scripts/setup_gpu_box.sh && uv pip install --python .venv/bin/python -e ".[ace]"

# free dry run with canned LLM responses
.venv/bin/python -m ace.run_loop --mock

# build the gold field references once (needs the gold study meshes in benchmarks/gold/_meshes/,
# from the goldgen archive, and a GPU or plenty of RAM)
.venv/bin/python -m ace.gold_reference abc_00000065_bc0

# train an 8-agent swarm on the gold part abc_00000065_bc0 (the Result 2 setup)
ACE_SCORE=field .venv/bin/python -m ace.run_loop --deep --split \
  --problems abc_00000065_bc0 --holdout "" --memory shared+private \
  --agents 8 --concurrency 8 --persona-count 4 --generations 8 --budget-usd 45 --out ace_runs/div_s8

# post-train it on the real compression test (the Result 3 run; needs CalculiX)
ACE_SCORE=field .venv/bin/python -m ace.run_loop --deep --split --problems abc_00000065_compression \
  --holdout "" --memory shared+private --agents 4 --concurrency 4 \
  --init-from results/trained_swarms/div_s8 --init-agents a6,a3,a1,a7 --pre-teach \
  --generations 10 --budget-usd 45 --out ace_runs/nl_ft

# watch a run live, export it, run a trained swarm on a new problem
.venv/bin/python -m ace.watch ace_runs/div_s8                # http://localhost:8765
.venv/bin/python -m ace.export_swarm ace_runs/div_s8 results/trained_swarms/div_s8
.venv/bin/python -m ace.infer --swarm results/trained_swarms/div_s8 --problem nafems_le1 --out ace_runs/infer_le1
```

The figures are rebuilt from the raw run folders (`ace_runs/`, kept out of git because of their size) and from
`physical_tests/` by these scripts:
- `talk_plots/*/make_figures.py` and `talk_plots/equal_budget/make_cook_figures.py`;
- for the crack figures and videos, `scripts/nl_break.py`, `scripts/nl_video.py` and `scripts/nl_story_video.py`.

The schematic is `talk_plots/schematic/training_system.html`.

Built at the ScienceClaw hackathon. Models: Claude Sonnet 5.5 (agents, reflection, organizer) and Claude Opus 5.5 (teacher).
