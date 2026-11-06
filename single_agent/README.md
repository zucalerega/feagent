# single_agent: one analyst agent, from design data to a verified analysis

**The question.** Given a component's design data (what the part is, how it sits in its system, what the
design review wants to know) and its CAD, can a current model with a proper harness do the **analysis layer**
on its own: decide the boundary conditions, mesh a complex part, run a convergence study, and answer the review
with honest uncertainties? And can **a general set of rules plus learning** make it better across parts and
load cases, without a swarm? We run this as two arms, **Opus** (`claude-opus-5-5`) and **Sonnet**
(`claude-sonnet-5-5`), each one model doing everything, including its own learning.

This differs from the swarm (`ace/`) in three ways:
- **One agent per arm**, and no peers, competition or private notes. There is one rulebook: a fixed general
  handbook plus the rules the agent learns.
- **Design data instead of boundary conditions.** The agent gets a design package: the component, the system
  context, review notes with irrelevant details, and the questions. It must find the faces from functional
  descriptions ("the bore's flat end face, under the cap") and decide how to model each interface. The package
  never names a face or uses solver words.
- **The whole analysis layer is scored**, not only the mesh: the idealization, each answer, the honesty of each
  uncertainty, whether a singular peak stress is recognised, and the convergence study.

The loop is closed twice. Each analysis ends in answers to the design review: values with uncertainties, a
verdict against the allowable, and recommendations. A deterministic scorecard then grades every analysis
against the gold references, and the learning step turns the training scorecards into general rules for the
next generation.

```
design package + CAD ──► agent (rulebook in its prompt) ──► report: idealization, study, answers ± U
                               ▲                                        │
                               │                          scorecard (gold references, no LLM)
                    rulebook: handbook + learned rules ◄── learning step (same model; training problems only)
```

## Quick start

```bash
bash scripts/setup_gpu_box.sh && uv pip install --python .venv/bin/python -e ".[ace]"   # or a venv with .[dev,ace]
export ANTHROPIC_API_KEY=...                                   # environment only, never in the repo

.venv/bin/python -m single_agent.loop --mock --generations 2   # free: canned model, real harness and scoring
.venv/bin/python -m single_agent.oracle                        # non-LLM baseline: reference setup + scripted study

# the two arms (defaults: 4 training problems, 3 holdout, 4 generations; holdout in the first and last)
.venv/bin/python -m single_agent.loop --model sonnet --budget-usd 40 --out sa_runs/sonnet
.venv/bin/python -m single_agent.loop --model opus   --budget-usd 60 --out sa_runs/opus
.venv/bin/python -m single_agent.compare sa_runs/sonnet sa_runs/opus --oracle sa_runs/oracle --out sa_runs
```

Ablations: `--rules none` (no handbook: the model's own judgement plus what it learns), `--no-learn` (handbook
only), `--reflector opus` (a Sonnet analyst with an Opus teacher), `--rulebook RUN/rulebook_final.json` (start
from another arm's learned rules, e.g. to see whether Opus's rules help Sonnet).

The tools also work from a shell, for any agent or a person (`single_agent/cli.py`):

```bash
.venv/bin/python -m single_agent.cli new abc_00000065_bc1 --root sa_workspaces   # -> sa_workspaces/part-xxxxxxxx/
.venv/bin/python -m single_agent.cli sa_workspaces/part-xxxxxxxx tools           # schemas
.venv/bin/python -m single_agent.cli sa_workspaces/part-xxxxxxxx inspect_geometry
.venv/bin/python -m single_agent.cli sa_workspaces/part-xxxxxxxx solve @setup.json
```

## The problems: ten load cases on two complex parts

The geometry is the two gold ABC parts (`benchmarks/gold/`): a trunnion cap (a cup with two trunnion pins on
bosses, 25 faces) and a split clamp body (a jaw with prongs, cross holes, nut pockets and waist notches, 55
faces). Each gold BC set became one design package with a mechanism that implies its supports and loads. A face
"bonded to a rigid post" is held; a face that "bears on a preloaded guide plate that stops it along x only" is
held along x only; a shrink-fit is "a uniform contact pressure on the bore wall, not on its end face".

| problem | load case | interfaces | questions | oracle |
|---|---|---|---|---|
| `abc_00000003_bc0` | TC-35 / LC-07: side pull on the −x trunnion shoulder (rig test) | 3 | 2 | 1.00 |
| `abc_00000003_bc1` | TC-35 / LC-02: lift-out of the actuator by the cap | 3 | 3 (1 singular) | 1.00 |
| `abc_00000003_bc2` | TC-35 / LC-04: trunnion clamp rings during transport | 4 | 6 (1 singular) | 0.90 |
| `abc_00000003_bc3` | TC-35 / LC-11: shrink-fitting the bore bushing (**suspect reference**, see below) | 3 | 5 (1 singular) | 0.51 |
| `abc_00000003_bc4` | TC-35 / LC-12: grinding the +x trunnion pin | 2 | 4 (1 singular) | 1.00 |
| `abc_00000003_bc5` | TC-35 / LC-09: cantilevered test of the cap with an internal spring pad | 3 | 2 (1 singular) | 1.00 |
| `abc_00000003_bc6` | TC-35 / LC-14: sensor carrier side load during −x pin grinding | 3 | 3 (1 singular) | 1.00 |
| `abc_00000003_bc7` | TC-35 / LC-15: hydrostatic proof of the cap in the test vessel | 3 | 4 | 1.00 |
| `abc_00000065_bc0` | SC-40 / LC-03: side push on the jaw of a bonded clamp | 2 | 5 (1 singular) | 0.92 |
| `abc_00000065_bc1` | SC-40 / LC-06: pressing the clamp onto its locating pins | 3 | 6 (1 singular) | 0.97 |

Default split (`config.py`): train bc0, bc1, bc2, bc5. Holdout bc7 (the same part, a new installation) and both
clamp sets (a new part); the holdout is never reflected on. Spares: bc4, bc6.

- **Questions** ask for what a review needs: the mean deflection of an interface, the largest deflection, a
  support's reaction for bolt sizing, the stored energy, the volume-averaged von Mises stress, and nearly always
  the peak stress against an allowable, with "a value that does not depend on the mesh". Where the gold study
  found the peak diverging (at the rim of a held face or a re-entrant edge), the right answer is to say it is
  singular, not to give a number.
- **Files** (`design/`). `_parts/<part>.md` is the component description shared by a part's load cases.
  `<problem>/case.md` holds the system context, the review notes and the questions. `<problem>/key.json` is
  evaluator-only: which gold QoI each question asks for and which phrase implies each reference item.
  `python -m single_agent.workspace check` lints every package (no face tags, no solver vocabulary, no benchmark
  names) and checks every key against the gold data. `show <problem>` prints a package as the agent reads it.

## The harness (`harness.py`)

A **workspace** holds only `geometry.step` and `package.md`. The part is a `BlindPart` (`part.py`): the STEP
alone, registered under a neutral label (`part-xxxxxxxx`), with no statement, reference setup or reference
values. This matters because the swarm's geometry and feedback helpers read the problem statement. On a gold
problem they would hand the agent the supported and loaded faces ("NAMED in statement (support)", the clamp-rim
edges, QoIs named `u_mean_f18`). The tests check that nothing of that reaches the agent.

| tool | what it does |
|---|---|
| `inspect_geometry` | faces (type, area, centroid, normal / radius, neighbours), edges with material angles, re-entrant edges, smallest features |
| `find_faces` | geometric face queries (feagent's selector: kind, normal, axis, radius, `at: xmin`, near, adjacent_to, largest, ...) |
| `view_geometry` | a picture of the part with chosen faces coloured and labelled: checks "is this the boss shoulder?" before any solve |
| `mesh`, `refine_mesh` | gmsh tets (order 1/2), local sizes at faces, edges, points, boxes; named face groups; `refine_mesh` divides every size of a recipe by a factor (systematic studies) |
| `solve` | the fixed skfem solver with the agent's material, supports, loads and its own QoIs; feedback: equilibrium, extremes and where, ZZ error estimate, hotspots tagged with the nearest re-entrant edge and the rim of the agent's own supports |
| `study` | Richardson / GCI per QoI over 2+ solves of one setup: convergence type, observed vs theoretical order, extrapolated value, GCI and conservative GCI, a plain-words reading |
| `view`, `compare` | pictures of a solve (mesh, stress, error); the difference between two solves |
| `submit_report` | idealization, final solve, study solves, one answer per question (value, relative uncertainty, status converged / not_converged / singular, verdict), assumptions, rules used, recommendations |

Limits (per analysis): 50k nodes per solved mesh (P2: ~150k DOFs, about 6 s with PARDISO), 12 solves, 12
pictures, 60 turns. The agent loop (`agent.py`) uses `ace.llm` for the API calls: adaptive thinking with
summaries, effort `high` for both models (Opus 5.5 defaults to `medium`), prompt caching and the server-side
refusal fallback. It warns at 8 and 3 turns left and nudges once if the model stops without submitting.

## The scorecard (`evaluate.py`, no LLM)

| part | weight | how |
|---|---|---|
| setup | 0.30 | The agent's final setup and the reference setup, solved on the agent's own final mesh. Within 0.2% on every global and face QoI they are equivalent (1.0): a force or the same force per area, a bonded face as fixed or as all components zero, all count. Otherwise 0.5 × the share of reference items matched by face, kind and value. |
| answers | 0.40 | Per question, 1 within the gold tolerance, else tol / error (reactions by magnitude). A singular question earns 1 only when reported singular or not converged. |
| calibration | 0.15 | The share of converging answers whose stated uncertainty covers the truth, plus a quarter of the reference's own tolerance, while staying at or under 10%. |
| study | 0.15 | 3+ meshes with the final setup, analysed with the study tool (1.0); without the tool (0.7); 2 meshes (0.4). |

Also reported, outside the composite:
- **meshing**: the agent's final mesh with the reference setup, against the gold references and against uniform
  refinement at the same DOFs (from the gold study's nested meshes);
- whether each reported value comes from the agent's own solves;
- cost: DOFs, solves, turns, tokens and dollars.

The **oracle** (`oracle.py`) is the evaluator's sanity check and the non-LLM baseline. It takes the reference
setup, refines at the re-entrant edges, studies 3 levels fitted under the node cap, and reports by fixed rules.
It scores 1.00 on six sets and 0.90-0.97 on three, so the references are reachable under the cap. Its misses are
informative: twice it called a peak stress converged because 3 meshes under 40k nodes did not show the
divergence. Spotting those takes reasoning about where the peak is, which is what the agent is tested on.

## Learning (`rulebook.py`, `reflect.py`)

- **Handbook** (`handbook.md`, A1-F3). Hand-written, general rules on:
  - turning design-data phrases into supports and loads;
  - finding faces;
  - meshing;
  - convergence studies (GCI, extrapolation, what to do when the order is off);
  - singularities;
  - reporting.

  It is the "general set of rules"; `--rules none` removes it.
- **Learned rules** (`L001`...), at most 30. After each generation, one structured call per arm (the arm's own
  model by default) reads every training analysis: the package, a one-line-per-call summary of what the agent
  did, its report, and the scorecard with the intended idealization item by item and the true answers. It
  diagnoses each analysis and returns operations: add, edit, remove, or vote helpful / harmful.
- **Lint.** A rule must be general: no face tag, problem, part, load case, load magnitude or specific value, and
  at most 70 words. Near-duplicates become votes, and rules voted harmful (harmful ≥ helpful + 2) are pruned.
- **Holdout problems are never reflected on.** The holdout score is the test of whether the rules generalise.

## Findings so far (from building and validating the benchmark)

1. **The gold references of `abc_00000003_bc3` look wrong.** For a pressure-loaded part with rigid supports, the
   FE strain energy is a lower bound that rises with refinement.
   - Independent P2 meshes give 2.111, 2.163, 2.176, 2.184 J (uniform) and 2.190, 2.194, 2.195 J (edge-refined,
     up to 118k DOFs). The mesh volume is within 1e-4 of the CAD.
   - The gold nested sequence falls instead: 2.166, 1.983, 1.940 J. Its reference is 1.927 J, 12% below values
     that bound it from below.
   - u_max, vm_mean and Rx_f16 are off by 7-18% the same way.

   The likely cause is in the nested L1/L2 meshes (split, snapped, then tet and high-order optimisation). They are
   not in git, so they could not be checked here. bc3 stays packaged but is in neither default list.
2. **The shared three-mesh GCI (`feacore.verification.gci.gci3`) misreads unequal refinement ratios**, which
   independently remeshed sequences always have:
   - It classifies by R = e21/e32 alone. With h ratios 1.34 (fine) and 1.21 (coarse), a strain energy converging
     at p ≈ 1.7 has R ≈ 1.02 and is called "divergent". An agent would then call a converging quantity singular.
   - With the small step on the fine side, a stalled sequence can be called "monotone".
   - It raises a math domain error on some diverging unequal-ratio sequences.

   The study tool uses the ratio-corrected test: some p > 0 fits iff e32/e21 > ln r32 / ln r21. It is in
   `harness.gci_unequal`, with tests; feacore is not modified.
3. **Small ratios on coarse meshes make the GCI optimistic.** Observed orders of 6-9 against a theory of 3-4
   came with GCIs of about 2% when the true error was 8%. The study tool now reports the conservative GCI and says
   so in words, and handbook rule D3 tells the agent to use the largest estimate.

## Files

| file | role |
|---|---|
| `config.py` | arms, effort, limits, problem split, score weights, learning caps |
| `part.py`, `workspace.py` | the blind part; design packages, lint, workspaces |
| `harness.py`, `render.py`, `cli.py` | the tools; geometry pictures; the shell interface |
| `prompts.py`, `agent.py` | what the agent reads; one analysis as a tool-use conversation |
| `evaluate.py`, `oracle.py` | the scorecard; the scripted baseline |
| `handbook.md`, `rulebook.py`, `reflect.py` | the general rules; learned rules and their lint/merge; the learning step |
| `loop.py`, `compare.py`, `mock.py` | the generations; arms side by side; the free mock |
| `design/` | the ten design packages and their evaluator keys |

Run outputs go to `sa_runs/` and workspaces to `sa_workspaces/`; both are git-ignored.

## Cost and time (estimates, not yet measured on the API)

One analysis is about 30-50 model turns with a growing context: on the order of $1-2.5 with Opus 5.5 and
$0.5-1.5 with Sonnet 5.5, with caching. A generation with 4 training problems, plus 3 holdout problems in the
first and last generation, then costs about $5-10 (Sonnet) to $10-20 (Opus). Most of a turn's time is the
model; solves take seconds with PARDISO (x86) or cuDSS (GPU). The scipy fallback is about 7x slower at the cap.

## Limitations and next steps

- **Two parts, ten load cases.** That is enough to see whether rules transfer to a new installation of the same
  part (bc7) and to a new part (the clamp). It is not enough for statistics. More gold parts would need the
  generator (`feacore.goldgen`, an H100) and a design package each.
- **The design packages are hand-written** by the person who knew the reference setup; the lint and the oracle
  check them. A round trip (a model reads only the package and proposes a setup, as `ace.requirements` does for
  its requirements files) would catch remaining ambiguities.
- **Linear elasticity only.** The physical-test problems (`abc_00000003_pull`, `abc_00000065_compression`) are
  the natural next packages: there the loop closes on real test data instead of a gold reference.
