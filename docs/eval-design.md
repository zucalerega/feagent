# Evaluating the swarm: accuracy vs speed vs compute cost

Status: proposal, implemented as the default in `feabench/scoring.py` and `feabench/evaluate.py`.
Written overnight on 2026-10-31 without the team available; every choice below is a default to
argue with, and the decision log (`docs/decisions.md`) lists them one by one.

## TL;DR

1. **Numbers come from a deterministic harness, never from the LLM evaluator.** The harness runs the
   fixed solver on the agent's deck, measures quantities of interest (QoIs) on the agent's own mesh
   and compares them with hidden references. The evaluator agent *explains* the numbers (failure
   taxonomy, good ideas); it does not produce them.
2. **Per-case score**

   ```
   S = G x A x (1 - 0.2 (1 - E))
   ```

   * **G, gates (0/1).** The solver ran cleanly. The analysis type is right. Materials match the
     spec. Mesh volume matches the CAD within 2 %. The load path is right (support reactions equal
     the applied load). The run stayed within the hard budget. Any failure gives S = 0.
   * **A, accuracy (0..1).** Weighted mean of per-QoI credits. A credit is 1 inside the tolerance,
     falls linearly in log10(error/tolerance) and reaches 0 at 10x the tolerance.
   * **E, efficiency (0..1).** Geometric mean of cost efficiency (LLM tokens plus solver CPU, in $)
     and time efficiency (episode wall time). Each is 1 at or below the case's reference cost and 0
     at 10x it, log-linear in between.
   * **0.2**: efficiency can take at most 20 % off an accurate answer and can never rescue an
     inaccurate one.
3. **Rank agents with within-case comparisons** (Bradley-Terry ratings accumulated over iterations),
   not raw mean score. Agents are assigned to random problems of very different difficulty.
4. **Population health metrics.**
   * Pass rate.
   * Coverage: cases solved by at least one agent.
   * Median $ per solved case.
   * Pareto front and hypervolume in (accuracy, log cost).
   * Failure taxonomy rates.
   * The trend on held-out cases is the headline number.
5. **Improvers get the decomposed vector and the failure taxonomy, never reference values.** Some
   cases are held out and never summarised, so we can see whether improvements generalise.

The rest of this note argues for each choice and lists the alternatives.

---

## 1. What are we optimising?

The agent's product is an FE model of a CAD part whose answer an engineer can act on. That answer
has to be correct within an engineering tolerance, and it should be cheap and fast to get. In
practice the value is close to a step function of correctness:

* A **wrong answer has negative value**, because it creates false confidence. Typical wrong
  answers are a missing load, a unit error, a boundary condition on the wrong face, or a mesh so
  coarse the peak stress is off by 30 %. They are worse than no answer.
* A **correct answer is worth roughly the same** whether it is 0.5 % or 0.05 % accurate, once it
  is inside the tolerance the decision needs.
* **Cost and time matter, but second.** Among correct answers we prefer the cheaper and faster
  ones, and both effects have diminishing returns.

The score therefore multiplies accuracy by a bounded efficiency factor. It does not add them.

## 2. What the harness measures (per episode)

| Dimension | Measured by | Deterministic? |
|---|---|---|
| Validity | ccx exit status and `*ERROR` lines, results present, analysis type, materials vs spec, mesh volume vs CAD volume, support reactions vs applied load | yes |
| Accuracy | QoIs on the agent's mesh: point values (shape-function interpolation), region statistics, reactions, rotations, eigenfrequencies, buckling factors, field-level relative L2 (SimJEB) | yes |
| Solver cost | CPU seconds, wall time and peak memory of the ccx process (`os.wait4`), number of equations | yes |
| Agent cost | input / output / cache tokens, tool calls, solver runs during the episode, wall time, all reported by the agent harness as `agent_cost` | yes, once logged |
| Process quality | did the agent check convergence, check reactions, state assumptions, flag ambiguities | LLM evaluator, qualitative |
| Self-assessment | the agent's own value and error estimate (`submission.json`) | recorded; optional calibration score |

Process quality and self-assessment are deliberately **not** in S. They go into the evaluator's
summary for the improvers.

## 3. Accuracy

### 3.1 References and how much to trust them

Each QoI carries a `reference_type`. A tolerance is only meaningful if it is several times the
reference's own uncertainty.

| Tier | Example | Uncertainty | Typical tolerance |
|---|---|---|---|
| analytical | Lame cylinder and sphere, Saint-Venant torsion, self-weight column | 0 (exact for the stated problem) | 0.5 % displacement, 1-2.5 % stress |
| published | NAFEMS LE1 (92.7 MPa) and LE10 (-5.38 MPa) | 3 significant digits; our finest meshes give -5.393 (0.24 % from the target) and 92.57 (0.14 %) | 2 % |
| numerical | cantilever tip deflection, eigenfrequencies, buckling factor, bracket QoIs | Richardson GCI from the convergence study, e.g. 0.036 % on the cantilever tip | at least 3x the GCI, usually 1-3 % |
| code-to-code | SimJEB OptiStruct fields | unknown: one solver, one mesh, RBE modelling choices baked in | wide: 5 % field L2, 3 % max displacement, 10 % p99.9 von Mises |
| regression | ccx `.dat.ref`, code_aster `VALE_CALC` | solver self-consistency only | not used for agent scoring |

Every verification case also carries a **demonstrated attainability**: the scripted reference
pipeline (gmsh quadratic tets fed to ccx) is run at five mesh sizes on the task's own STEP file
and scored by the same evaluator. That shows the tolerance can be met and at what cost. It also
caught problems before any agent saw them:

* **A too-tight tolerance.** On the sphere's bore, hoop stress (1/r^3 gradient) scattered 0.5 %,
  1.7 % and 0.2 % on meshes of 13k, 43k and 112k DOF. A 1.5 % tolerance would have made pass/fail
  a coin flip for reasonable meshes, so it is now 2.5 %.
* **A point QoI at a trihedral corner.** NAFEMS LE1's D is one; its nodal stress depends on how
  few tets share the node. The QoI is now the thickness average, which is what the plane-stress
  target means.
* **A genuinely bad mesh, kept in the record.** One of the five LE1 pipeline meshes has sliver
  tets at the notch (radius ratio 0.001) and is 14 % low. That is real FE error, not evaluation
  noise. Point and line QoIs now report the quality of the elements they were evaluated in, so the
  evaluator can name "slivers at the critical location" as the cause.

### 3.2 Turning error into credit

`credit = 1` if `err <= tol`, else `max(0, 1 - log10(err / tol))`.

* **Why a log scale.** Model errors span orders of magnitude. Typical sizes are 0.1 % (fine mesh),
  3 % (coarse mesh), 30 % (wrong boundary condition), 1e3 (unit error). Linear credit spends all
  its resolution on the first two. Log credit still separates "a bit coarse" from "physically
  wrong".
* **Why not a hard pass/fail.** Improvers need a gradient. "7 % instead of 2 %" (credit 0.46) and
  "40 % instead of 2 %" (credit 0) call for different fixes. The binary `all_qois_within_tolerance`
  is still reported, as the engineering acceptance criterion and for pass rates.
* **Why bounded at 0.** One catastrophic QoI should not swamp the case: gates already zero out
  catastrophic models. Weights keep each QoI's influence explicit.

### 3.3 Which QoIs (and which not)

* **Prefer robust quantities.** Displacements, rotations, reaction sums, compliance and
  eigenvalues converge fast and do not depend on how a peak is post-processed.
* **Stresses only where they are well-defined.**
  * Stress at points away from re-entrant corners and constraint edges.
  * Peaks in smooth stress concentrations (the bracket fillet, with a `within` box that excludes
    the side faces).
  * Percentiles (p99.9) rather than maxima on complex parts.
  * A maximum next to a fixed edge or a point load is a mesh-size meter, not a QoI.
* **Field-level error** (relative L2 of the displacement field interpolated at reference nodes)
  for complex parts where we have a reference field (SimJEB). It is hard to game and summarises
  global correctness in one number.
* **Load-path gates.** A reaction QoI flagged `gate` catches the most common catastrophic mistakes
  cheaply: missing, doubled or misdirected loads, and loads on the wrong face. Note that ccx's RF
  output is the nodal internal force K u at *every* node. On a node that is also loaded it equals
  reaction plus load. So gate regions are chosen to be free of applied load (LE10 uses line EE',
  not the whole model), and the self-weight case has a 2 % gate because its support nodes carry an
  O(h) share of the body force.
* **Geometry and material gates.** Mesh volume against CAD volume (2 %) catches mm/m mix-ups and
  missing bodies. Integration is isoparametric, so coarse curved quadratic meshes are not punished
  for nothing. The material check (`*ELASTIC` / `*DENSITY` against the spec) reports a density off
  by 1e3 as exactly that.

### 3.4 Optional: calibration

An engineer who knows how accurate their answer is is worth more than one who doesn't. If agents
report an estimate with an uncertainty (e.g. from their own two-mesh check), score it with an
interval score. That rewards honest error bars and convergence checks, and discourages both
over-confidence and refining blindly. It is not in S yet. Add it once agents report uncertainties.

## 4. Cost and speed

### 4.1 Where the money and time actually go

The solver is three to six orders of magnitude cheaper than the LLM. Measured tonight (ccx 2.21,
SPOOLES, 2 threads) on the verification cases:

| Case | Cheapest mesh passing all QoIs | Solver CPU at that mesh | Finest mesh run | CPU |
|---|---|---|---|---|
| Lame cylinder | 2.2k DOF | 0.07 s | 166k DOF | 54 s |
| Saint-Venant torsion | 4.9k DOF | 0.2 s | 197k DOF | 85 s |
| Self-weight column | 1.4k DOF | 0.06 s | 87k DOF | 11 s |
| NAFEMS LE10 | 7.6k DOF | 0.36 s | 137k DOF | 60 s |
| Angle bracket (fillet peak) | 41k DOF | 3.5 s | 564k DOF | 475 s |

The full per-case table is in `benchmarks/verification/README.md`.

At about $0.05 per CPU-hour, a sub-second solve costs about $1e-5. An agent episode is roughly
30 tool calls, about 1.5M cumulative input tokens because history is re-sent each turn, and about
30k output tokens. On Claude Opus 5.5 ($4 / $20 per MTok, $0.20 cache reads) that is about $1.4
with prompt caching and about $7 without. These are illustrative assumptions: log the real
`usage` from every response.

Consequences:

* **"Compute cost" means LLM tokens.** Penalise tokens (in $) and wall time. Solver CPU is added
  to the $ cost for completeness, but it only matters for very large meshes.
* **Mesh size matters through wall time and memory, not dollars.** Solve time grows roughly as
  N^1.7 (12.8k -> 166k DOF took 0.77 s -> 54 s on the cylinder; 564k DOF took 8 minutes on the
  bracket). A 1M-DOF quadratic tet model
  would take tens of minutes and many GB. So: a **hard DOF/time cap** per case, i.e. a budget
  gate. Inside the cap, wall time is penalised softly.
* **Prompt caching is the biggest single cost lever**, about 4-5x on agent episodes. It is a free
  win for the agent harness, not something to select for.
* **Evaluation throughput is cheap.** Scoring a submission costs one solve plus milliseconds of
  post-processing. The evaluator LLM call per case, not the harness, dominates evaluation cost.

### 4.2 Normalising cost per case

Raw $ is not comparable across cases: a bracket legitimately needs more steps than a cube. Each
case has a **reference cost** (from its budget block, otherwise defaults of $0.50 and 300 s) and a
**cap** at 10x. Efficiency is log-linear in between. "Twice the reference" costs 0.3 of E, which
is 6 % of S at w_eff = 0.2. "10x" costs the full 20 %. Once the swarm has run, set reference costs
to the population median of passing episodes per case, so the scale tracks reality.

### 4.3 Wall time

Wall time is noisy: API latency, shared CPUs, retries. It enters only through a generous soft
term, combined geometrically with $ efficiency. The deterministic proxies (tokens, tool calls,
solver runs, DOFs) are logged for analysis and for improvers ("you re-meshed 9 times").

### 4.4 Alternatives considered for combining the axes

| Scheme | Pros | Cons | Verdict |
|---|---|---|---|
| Weighted sum, a A - b log C - c log T | simple | the weights carry units; a cheap wrong answer can beat an expensive right one | rejected |
| Lexicographic: pass first, then cost | matches engineering acceptance | no gradient below the threshold; tiny cost differences decide rank among passers | used only as a tie-break |
| Pure Pareto ranking (NSGA-style) | keeps diverse strategies alive; no weights | no scalar for improvers; hard to explain | used for population metrics and selection diversity, not for S |
| Budget-constrained accuracy (fixed cap, score accuracy only) | fair, simple | no pressure to be cheaper than the cap | kept as the hard cap inside our scheme |
| Time / cost to solution (cost until the first passing answer) | the real engineering metric | needs repeated attempts per agent-case | report it at population level when agents get multiple attempts |
| **Multiplicative with bounded efficiency (chosen)** | correctness dominates; efficiency still matters among correct answers; scale-free via log; one scalar plus a decomposition | one tunable weight (0.2) | default |

## 5. Ranking agents that solved different problems

Agents are assigned at random to one problem each, and the problems differ a lot in difficulty.
LE10's stress point is far harder than the Lame cylinder, which passes at 2.2k DOF. A mean of S
across agents would mostly measure assignment luck. Instead:

* **Compare only within a case.** For every case, each pair of agents that solved it gives a
  win, loss or tie on S. Fit **Bradley-Terry** strengths over all pairs (`bradley_terry` in
  `scoring.py`, with a weak prior so undefeated agents stay finite). Case difficulty cancels out.
* **Make it work.** Assign at least 3-5 agents to each case per iteration (fewer cases per
  iteration, more agents per case). Accumulate comparisons over iterations with forgetting
  (e.g. only the last k iterations, or decay weights), because agents change as improvers edit
  them.
* **Account for noise.** LLM agents are stochastic; one episode is one sample. Promote an
  improver's change only when the child beats its parent over several cases (racing / successive
  halving), or keep both in the population and let selection decide.
* **Later: item response theory.** Fit case difficulty and discrimination jointly with agent
  skill. This makes scores comparable when the case pool changes, and identifies cases that do
  not discriminate (everyone passes or everyone fails) so they can be retired.

## 6. Population-level metrics ("scores the population as a whole")

Reported by `population_summary`:

* **Pass rate**: the fraction of episodes where all gates and all QoIs pass. This is the
  engineering yield.
* **Coverage**: the fraction of cases solved by at least one agent, i.e. what the population can
  do collectively. Its gap to pass rate is what sharing ideas could recover.
* **Gate-failure rate**, by gate. Each gate maps to a distinct fix: units, load path, geometry
  import, solver errors.
* **Median $ per passing episode**, and later $ per solved case.
* **Pareto front and hypervolume** in (accuracy, log10 cost). This single number rises when the
  population gets more accurate *or* cheaper, and keeps diverse strategies valuable.
* **Diversity** (to add): the spread of mesh sizes, element types and BC idealisations. Without
  it a swarm collapses onto one strategy.
* **Trend on held-out cases**: the headline progress metric.

## 7. What the evaluator agent does (and must not do)

Inputs: harness reports (all numbers), decks, solver logs and the agents' notes.

Outputs:
* **Failure taxonomy per episode**: units, BC location, BC type (fixed vs symmetry), load
  magnitude or direction, load distribution, mesh too coarse at the QoI, element type, solver
  error, geometry import, plus "other".
* **Strategies that worked**, e.g. "local refinement at the bore with h <= R/8 gave the hoop stress
  within 1 %", "symmetry planes instead of a full model halved tokens".
* **Ideas worth spreading.**

It must not:
* **Produce or change numeric scores.** That keeps them reproducible, and an LLM judge drifts.
* **Read raw agent output without isolation.** Agent decks and notes are untrusted text, and an
  agent (or an improver chasing score) could embed instructions aimed at the evaluator. Present
  them as quoted data. The scores do not depend on the evaluator anyway.
* **Put reference values or case-specific numbers in improver summaries.** That is the main
  overfitting channel. Summaries should state transferable practice ("density must be in tonne/mm^3
  in mm-N-s units"), not answers ("the cylinder's u_r is 0.0454").

## 8. Anti-gaming / Goodhart checklist

* Agents get only `task/` (`feabench export-task`). Never mount `benchmarks/` or the repo into an
  agent's sandbox. References live in `reference/` and stay with the harness.
* QoIs are computed by the harness from the solver results on the agent's mesh. Self-reported
  values are recorded, never scored.
* The solver is pinned (ccx 2.21) and accepted against its upstream regression suite. Output
  requests are appended by the harness, so an agent cannot hide results.
* Gates on geometry, materials and load path block "tuning" a model to hit a number.
* There is a held-out split (cases never shown to improvers or summarised) and case rotation.
  Watch the gap between train and held-out scores.
* Penalties are bounded and scale-free, so no single metric can be farmed.

## 9. Budget planning (illustrative, Claude Opus 5.5)

* Agent episode: about $1-2 with caching.
* Evaluator: one call per case per iteration, about $0.5.
* Improver: one call per agent per iteration, about $0.5.

For 16 agents, 4 cases per iteration (4 agents each) and 10 iterations:
* 160 agent episodes: about $160-320
* plus about $40 evaluator
* plus about $80 improver
* **total: about $300-450.** Solver compute is negligible (minutes of CPU in total).

Wall time per iteration is the slowest episode in the iteration, typically minutes. Halve the
token cost by running improvers and the evaluator on Batches when latency doesn't matter. Measure
before trusting any of these numbers.

## 10. Worked example

[`population-demo.md`](population-demo.md) runs five scripted pseudo-agents on four cases through
the real harness. The pseudo-agents are the reference pipeline at different mesh levels, two of
them with planted mistakes, and their token counts are assumed.

* **"quick"** (coarsest mesh, cheapest) scores a perfect 1.0 where a coarse mesh already meets
  every tolerance (torsion, cantilever). On LE10 it drops to 0.68, where it doesn't.
* **"overkill"** (finest mesh, 3x the tokens, 25 minutes) is perfectly accurate but loses about
  15 % to efficiency.
* **"careful"** ranks first overall on within-case Bradley-Terry.
* **The planted unit error (E in Pa) and the 10x load error both score 0.** The gate that fires
  names the bug. On LE10 the E-in-Pa model still had *perfect* accuracy, because stresses under
  pressure loads do not depend on E. The material gate, not the QoIs, catches it.

[`ccx-recipes.md`](ccx-recipes.md) collects the deck-writing techniques the harness validated.
It is seed context for agents, for improvers to evolve.

## 11. Open items

* Define the MCP tool surface, then make the agent harness emit `agent_cost` (tokens from each
  response's `usage`, wall time, tool calls, solver runs).
* Write the evaluator prompt around the taxonomy above, with agent text quoted as data.
* Decide the held-out split (proposal: NAFEMS LE1 and the angle bracket held out from the
  verification suite; a random 20 % of SimJEB brackets once SimJEB is reachable).
* Calibrate `cost_ref_usd` and `time_ref_s` per case from the first population run.
* Consider the calibration score (3.4) and diversity metrics (6).
