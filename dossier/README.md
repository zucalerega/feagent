# dossier: verify a component by analysis, in the context of its system

**The question.** Given a component, its requirements and what its system does to it (loads, actuations,
temperatures, in every operating phase), does the component meet its specification, and is the
specification itself right? `dossier` answers with a set of linked records that an engineer can follow,
check and sign.

```
 system.yaml ─┐                                     ┌─ findings: is the specification right?
 component  ──┼─► context ─► geometry ─► validate ──┤
 requirements ┘   every      features    system and └─► plan ─► analyze ─► assess ─► report
                  number     to faces    requirement    load     one engine margins   linked,
                  evaluated  + meshes    checks         cases +  per type   verdicts  STE100
                                                        hand calcs
 every stage also writes: decisions (a choice, why, the alternatives) and assumptions (with their impact)
 a person reviews them: accept / reject / request changes
```

## Quick start

```bash
bash scripts/setup_gpu_box.sh                       # or a venv with: pip install -e ".[ui,ace]" --group dev
python -m dossier check  dossier/examples/mov_valve          # read the inputs, evaluate every number, check units
python -m dossier run    dossier/examples/mov_valve --quick  # every stage, coarse meshes (minutes)
python -m dossier run    dossier/examples/mov_valve          # full meshes
python -m dossier status dossier/examples/mov_valve          # stages, verdicts, open reviews
python -m dossier show   dossier/examples/mov_valve REQ-MOV-001
python -m dossier review dossier/examples/mov_valve DEC-... --accept --by "A. Engineer" --comment "..."
python -m dossier new    my_valve                            # a copy of the example to edit
```

The report is `<project>/out/report/index.html`. In the dashboard (`python -m feacore.ui`) the same thing is
the **Projects** section: edit the inputs, run the stages as tasks, read the report, give reviews.

## Run it on your machine

Install (needs Python 3.12 or newer and `uv`; run it in a checkout of the repository):

```bash
uv venv .venv --python 3.12
UV_NATIVE_TLS=1 uv pip install --python .venv/bin/python -e ".[ui]"     # Windows: .venv\Scripts\python.exe
.venv/bin/python -m dossier ui                                         # then open http://localhost:8000/#/projects
```

`python -m dossier ui [--port 8000] [--runs-dir DIR] [--host 127.0.0.1]` starts the dashboard (the same as
`python -m feacore.ui`), prints the address of the Projects page and stops with Ctrl+C. Projects and runs are
kept in `./runs` of the checkout, or in `--runs-dir` / `$FEACORE_RUNS_DIR`. `UV_NATIVE_TLS=1` is only needed
behind a proxy that re-signs certificates.

What the page offers:

- **New project**: a copy of the example `mov_valve` (the inputs are yours to change).
- **Check inputs**, **Run verification (quick or full)**, **Re-assess** (margins and report again from the
  analyses that exist, in seconds: use it after a review), **Build report**, and a panel to run some stages only.
- **Report**: the real report in the page (links and the "Why" panel work); accept, reject or ask for changes
  on any decision, assumption or finding, with your name and a comment (written to `reviews.yaml`).
- **Inputs**: an editor for the YAML files (a problem shows with file, line and message) and a CAD replace.
- **Numbers**, **Load cases** (each links to its record in the report) and **Reviews** (all records, filters).

Tested here on Ubuntu 24.04 (Python 3.12, a new venv made with the install command above, then
`python -m dossier ui --port 8124 --runs-dir ...` started from another folder, and the quick verification run
from the page's API): the install is enough for all seven stages. `matplotlib` (formulas in the report) and
`pyvista` (pictures) are already installed by the base dependencies of `pyproject.toml`; `pint` and `pydantic`
are base dependencies too. A quick run takes 3 to 5 minutes on 4 CPUs.

**Without OpenFOAM.** Everything runs except the flow load case (stepped bore, pressure drop). Tested on Ubuntu
with OpenFOAM hidden from the backend (no OpenFOAM program on `PATH` and none in the usual install folders, the
lookup in `feacore/solver/_foam_env.py` found nothing): the run completes; the analysis of `LC-FLOW-FIRE` is
"not supported" with the text "OpenFOAM is not installed ..."; the pressure-drop requirement `REQ-MOV-008` gets
the verdict inconclusive, "the value is an estimate from a hand calculation" (the hand calculation of the
pressure drop still runs); the page shows the text in the Load cases tab and in the report. With the extra
`cfd` (`pip install -e ".[ui,cfd]"`, the Python package `foamlib`) and OpenFOAM ESI v1912 or newer installed,
the flow case is solved. Without `foamlib` the message says that package is missing (tested). The lookup of
OpenFOAM uses `bash` and Linux install folders (read in the code), so OpenFOAM runs on Linux (or WSL) only.

**Pictures.** The geometry stage of a full run (not a quick run) makes pictures of the part with VTK, which needs
an OpenGL context. The page starts such a run through `feacore/ui/offscreen.py`. On Linux without a display it
uses `xvfb-run` (tested: `sudo apt-get install xvfb`; the pictures are in the report); with a display it runs
directly. On macOS and Windows there is no `xvfb-run` and the launcher runs the script directly with the
system's own window system (read in the code, with a unit test of the choice; not tested on those systems). VTK
does not raise an error when it has no OpenGL context, it can crash the process; if a full run stops in the
geometry stage, run again with "no pictures" (in "Run a part of the pipeline", or `--no-pictures` on the
command line), or use a quick run, which makes no pictures.

**Not tested: macOS and Windows.** `gmsh` has wheels for macOS arm64 on PyPI, but not for Linux ARM (see
`CLAUDE.md`). The task lane uses POSIX process groups (`feacore/ui/tasks.py`: `os.killpg`), so the buttons that
stop or kill a run are expected not to work on native Windows (read, not run); use WSL 2 there. The MKL PARDISO
solver is x86 only, the solver then uses SciPy (as stated in `CLAUDE.md`).

## What you write

One folder per project, plain YAML, made to be read and reviewed by people and kept in git.

| file | what it holds |
|---|---|
| `project.yaml` | the project, and which component is under verification |
| `system.yaml` | the system: elements, the connections between them, phases, duty cycles, named numbers with their sources, and the **effects** of each connection on an element in each phase |
| `components/<name>/component.yaml` | the component: CAD, material, named features (face sets selected by geometry), interfaces (which features realize which connection), stress classification lines, hand-calculation models |
| `components/<name>/requirements.yaml` | the requirements: source text, parameters (the numbers it states), criteria (response, limit, factor of safety), the phases it applies to |
| `materials.yaml`, `policy.yaml` | material properties with their sources; the factor policy of the governing standard |
| `reviews.yaml` | written by `dossier review` and the dashboard: who accepted or rejected which record |

`dossier/examples/mov_valve/` is a complete project: the body of a main oxidizer valve in a pump-fed
LOX / kerosene engine. Start from it (`python -m dossier new my_valve`).

### Numbers have units and sources

Every number is a quantity (`10.4 MPa`, `25 mm`, `90 K`), or an expression of named numbers whose result is
checked for its dimension:

```yaml
parameters:
  p_pump_max:  {value: 10.4 MPa, source: "ENG-ICD-002 rev B, table 4"}
  dp_surge:    {expr: "rho_lox * l_line * v_lox / t_close", unit: MPa, basis: derived}
  p_max_inlet: {expr: "p_pump_max + dp_surge", unit: MPa, basis: derived}
```

A pressure given in metres, or a sum of a pressure and a length, is an error message with the name of the
parameter, not a wrong margin. `basis` says how good a number is: specified, measured, reference, derived or
assumed. Every assumed number becomes an assumption record that a person reviews.

### The system context is a list of effects

```yaml
connections:
  - id: IF-PUMP-MOV
    from: EL-PUMP
    to: EL-MOV
    kind: fluid
    effects:
      - {id: EF-IN-P-SHUTDOWN, phase: PH-SHUTDOWN, kind: pressure, value: {expr: p_max_inlet}}
      - {id: EF-IN-T-FIRE, phase: PH-FIRE, kind: fluid_temperature, value: {expr: t_lox}, h: {expr: h_lox}}
      - {id: EF-IN-HOLD, phase: [PH-FIRE, PH-SHUTDOWN], kind: support}
```

Effect kinds: pressure, force, moment, acceleration, fluid_temperature (with a heat transfer coefficient),
temperature, heat_flux, mass_flow, displacement, support (the connection holds the element), signal. The
component's `interfaces` say which CAD features each connection acts on, by role: `wetted`, `seat`, `bolts`,
`external`.

## Rules of the model

1. **Loads come from the system context.** Requirements give criteria and factors. A number a requirement
   states (MEOP, design cycles, a temperature range) is a claim, and the validation compares it with the
   context.
2. **A load case is a phase at limit loads.** Factors of safety are applied to the response in the margin, so
   one solve serves every criterion of the phase.
3. **Nothing is silent.** A choice is a decision with its reason and the alternatives. A thing taken as true is
   an assumption with its impact. A problem in the inputs is a finding.
4. **Every record links to what it rests on**, down to a number with its source document
   (`dossier/trace.py`).
5. **Verdicts are honest**: pass, fail, inconclusive, not assessed. A result that is not converged, a singular
   peak, or a hand calculation that disagrees makes a verdict inconclusive, never a pass. So does a finding of
   severity error or major that questions a value the criterion uses, until a person resolves it.
6. **Rules first, judgement on top.** The engines are deterministic and reproducible. Language-model roles
   read the dossier and propose findings and decisions; their records are marked as proposed by an agent and go
   through the same review.
7. **Text is ASD-STE100** Simplified Technical English, checked by `dossier.ste`.

## The stages

`python -m dossier run` runs seven stages in order (`--from` and `--to` select a range, `--only LC-...`
selects load cases, `--force` analyses again). Only `pipeline.py` writes stage files, into `<project>/out/`.
A stage whose engine fails stops the run with the reason. A stage file that is older than the inputs is
*stale*: the verdict rules treat it as doubtful evidence, and the dashboard shows it.

**context.** Evaluates every named number (units, expressions, dimension check) and every effect in SI per
phase (`dossier/context.py`). Writes `context.json`. A number of basis
`assumed` becomes an assumption record.

**geometry** (`geometry.py`, `meshing.py`). Reads the STEP file (exactly one solid, one length unit), measures
it and turns every feature query into CAD faces, with area, centroid, radius and axis. Finds re-entrant edges
(sampled at 9 points per edge, so an angle that peaks between samples can be missed) and checks that each stress
classification line starts and ends on the surface and stays in the material. Then it makes one shared study of
quadratic tetrahedra, coarse to fine (successive levels about 1.4 apart in element size, local refinement at
re-entrant edges and small fillets, finest level under `node_cap` nodes). Every mesh carries physical groups
for each face, each feature and each interface role. The element size cannot go below 1 % of the bounding-box
diagonal (2.42 mm for the example), so a small fillet gets at most two quadratic elements along its arc. A
decision states how many elements the fillets and the thinnest wall get and asks for a review. Writes `geometry.json`, the
meshes with an index (cached by the hash of CAD, groups, sizes and gmsh version), and pictures.

**validate** (`system/check.py`, `requirements/validate.py`). Two sets of rules. The *system check* looks for
what is legal but wrong or weak: a load with no interface or no faces, a convection effect with no heat transfer
coefficient, a number with no source or an assumed number with no reason, a load on a face that a support
holds, a phase in which nothing holds the part, a duty cycle with no load. The *requirement rules* have
families: quality (verifiable criterion, "shall", vague words, vocabulary of the criterion, location,
allowable, limit, factor), consistency (MEOP against the largest pressure in the operation phases, fluid
temperature range, rated flow within 1 %, design cycles against the duty cycles, explicit parameter checks),
policy (a factor not below the policy), coverage (every phase with loads, every loading duty cycle),
conflict (parameters with one role and different values), and material (service temperature range, medium
compatibility, properties at the temperatures of the context). Output: findings with a severity, and
assumptions. A finding does not stop the run; it travels to the verdicts as a caveat, and a caveat of severity
error or major blocks a pass (see "Verdicts").

**plan** (`loadcases/select.py`). A requirement counts when it is verified by analysis and has a criterion. The
quantity of the criterion gives the analysis family (stress, displacement: structural; temperature, heat flow:
thermal; pressure drop: flow; damage: fatigue). One load case is one phase at limit loads (factor 1.0 on every
load; factors act on the response in the margin). Loads come from the system context only, never from a
requirement. A structural case is thermo-structural when a thermal effect differs from the component's
reference temperature by more than `policy.thermal_delta`; it then uses the steady thermal case of the phase
(`LC-TH-<phase>`). Fatigue has one case, `LC-FATIGUE`, with one cycle use per duty cycle. A structural case
whose loads are a multiple of another case's loads, with the same supports and no thermal effect, is not solved:
it is scaled from the base case (`scaled_from`, `scale`, status `scaled`). This is exact for linear statics.
The planner never drops a case because another looks more severe: loads can cancel and another load path can
govern. Every case it makes, scales or leaves out is a decision with its reason. A criterion that no case can
serve stays visible as a case with status `not_supported` and a reason, so its verdict is "not assessed". The
plan also instantiates the hand calculations of the component (`handcalc/`): closed-form models with their
sources (Lamé cylinder and sphere, thin cylinder, circular plate, beam and bolt-circle loads, Euler buckling,
fin, constrained thermal stress, cylinder-wall conduction, Dittus-Boelter, minor-loss pressure drop, column
surge, Basquin life), each with a `when` rule and a `band` in which the analysis result must fall.

**analyze** (`analysis/`). One engine per analysis type; each writes one analysis record with the model, the
convergence study, results with an uncertainty and a status, and checks. An engine that cannot do a case returns
status `not_supported` with the exact reason and never invents a result. After the engines, each hand
calculation is compared with the analysis result (`handcalc.compare`).

- *Structural* (`structural.py`, `idealize.py`, `scl.py`). Linear elastic, small displacement, with the fixed
  solver (scikit-fem, quadratic tetrahedra, a direct solve) on every mesh level. Pressure, force, moment (a
  traction that is statically exact), acceleration (body force) and supports become boundary conditions, each
  as a decision. A free part is balanced and held at three nodes with inertia relief for the residual. A
  thermo-structural case reads the temperature from the thermal engine on the same mesh. Results: peak von Mises
  and principal stress at the part, features and interfaces, linearized membrane and membrane-plus-bending
  stress on each stress classification line (all six components, not the ASME choice), displacements and out-of-roundness of
  bores. Convergence: two levels give an uncertainty of 3 times the change (scaled); three levels use the grid
  convergence index with safety factors; a peak near a re-entrant edge or the rim of a support is never
  converged and is marked `singular` when it grows at every refinement. Stress lines get a floor from moving
  the line by a quarter element. Checks: load balance, reactions, mesh quality, a Zienkiewicz-Zhu estimate.
  Verified in `tests/dossier/test_dossier_structural.py`. Not verified: the thermal support with a real
  temperature gradient (checked with a uniform temperature only), moment loads on curved faces (flat faces
  against beam theory only), the stress next to seats (the real load path through bolts and gaskets is not
  modelled).
- *Thermal* (`thermal.py`). Steady heat conduction, one conductivity, convection with a given heat transfer
  coefficient, prescribed temperature, heat flux. Faces with no thermal load are adiabatic (an assumption
  records which, and the direction of the error). The conductivity assumption is tested with two more solves at
  minus and plus 20 %. Convergence uses the temperature range of the case as the reference, because heat
  conduction depends on temperature differences. Verified in `tests/dossier/test_dossier_thermal.py` against a
  slab with convection, a hollow cylinder (logarithmic profile) and a hollow fin against a numerical solution.
  Not modelled: radiation, contact resistance, volume heat sources, temperature-dependent conductivity, and
  time dependence.
- *Fatigue* (`fatigue.py`). Elastic stress-life. The stress range of two states is the von Mises stress of the
  tensor difference (ASME VIII-2, 5.5.3). Mean stress correction by Goodman, Smith-Watson-Topper or none; cycles
  to failure by the Basquin relation; cycles counted by the max-min method of Annex 5-B.5 over all duty cycles
  together, because the order of the duty cycles is not known; damage by Miner's sum at every node. Singular
  zones (re-entrant edges, rims of held faces) are left out of the `smooth` result and are never converged.
  Two checks fail the result where the material is not elastic: a range above 2 Sy (shakedown) and a
  first-cycle von Mises stress above Sy. The engine then does not give a valid damage. There is no fatigue
  strength reduction (Kf = 1), no stress between phases (a thermal transient or a surge is not seen), no
  residual stress, and no sequence effect in Miner's rule. Verified in `tests/dossier/test_dossier_fatigue.py`.
- *Flow* (`flow.py`). The pressure drop of a stepped bore of revolution (planes across the axis, cylinders about
  the axis), read from the STEP file and checked against its volume. The engine builds a structured hexahedral
  mesh with a straight pipe upstream and downstream and solves with OpenFOAM (`simpleFoam`): laminar below Re
  2000, k-omega SST with wall functions at Re 1e5 and above. Between these, above 1.4e6 (the validated range
  ends there, see `docs/solver_transient.md`) and for any other shape the case is `not_supported`. The turbulent
  model was checked against Colebrook in a smooth pipe (friction factor 2.6 to 4.7 % low) and against Borda-Carnot
  for a sudden expansion (loss coefficient about 7 % high); the engine adds a model uncertainty of 10 % to the
  result. Tetrahedra (15 to 26 % low) and prisms (13 to 17 % high) did not validate and are not used. The
  convergence status needs a mesh part under `policy.max_uncertainty` and a solver that converged on every mesh,
  or residuals that stalled under 3e-3 with a steady pressure drop over the last 500 iterations. Rough walls
  are a correction by the Colebrook formula, not a solver result. No swirl, cavitation, bends, non-circular
  bores or compressible flow. Tests: `tests/dossier/test_dossier_flow.py` (the turbulent validation is marked
  slow).
- *Scaled* (`scaled.py`). Takes the base analysis and multiplies results by the scale.

**assess** (`compliance.py`). Margins and verdicts. See "Verdicts".

**report** (`report/`). Builds `out/report/index.html` and `report.md`. See "The report".

## Verdicts

Every requirement of the component gets one verdict: **pass**, **fail**, **inconclusive** or **not assessed**.
An engineer signs verdicts, so a pass must be earned and every other verdict gives its reason
(`dossier/compliance.py` is the source of truth).

*Margin.* For a limit that the response must not exceed: MS = limit / (factor x response) - 1. For a minimum:
MS = (response / factor) / limit - 1. The factor is a number, a requirement parameter or a factor of the policy.
The limit is a material property (`yield_strength`, `ultimate_strength`) or a stated value, with the same
dimension as the response. A response of zero or less has no finite margin.

*Uncertainty.* dU is the absolute uncertainty of the analysis result. The conservative value is value + dU
for a maximum and value - dU for a minimum. If the analysis gives no uncertainty, the verdict says so.

*Verdict of one criterion in one load case, in this order:*

1. The factor or limit has no valid value, or result and limit differ in dimension: **not assessed**.
2. No analysis, an analysis that is not "done", or no value for the result: **not assessed** with the
   message of the analysis. One exception: a hand calculation of the same case with a comparison on the same
   result key gives an estimate, and the verdict is **inconclusive** with that margin.
3. By the convergence status of the result:
   - *converged*: **pass** when the margin of the conservative value is not negative; **fail** when it is
     negative even with the uncertainty on the safe side; otherwise **inconclusive**. A converged result with
     no uncertainty is **inconclusive** unless it comes from a hand calculation or a derived number.
   - *singular*: **inconclusive**. The text says when the criterion fails on every mesh of the study.
   - *not converged*: if the analysis gives an error estimate (`error_estimated` true), the result can still
     decide, with `policy.unconverged_margin_factor` = k (default 2): **pass** when the margin is not
     negative with k x dU on the unsafe side; **fail** when it is negative with k x dU on the safe side;
     otherwise **inconclusive**. The margin shown is then the margin with k x dU. With no error estimate (the
     uncertainty is only the spread of values that do not converge), no uncertainty, or k = 0, the verdict is
     **inconclusive**. A decision records this rule.
   - *estimate*: **inconclusive**.
4. A pass or fail becomes **inconclusive** when the evidence is in doubt: a check with `passed` false in the
   analysis or in an analysis it rests on (`uses`, `scaled_from`, the states of a fatigue case); such an
   analysis that is missing or not "done"; a plan or analysis file older than the inputs. A check with `passed`
   none is a note only. A failed hand-calculation comparison (the result outside its band) is such a check.

Stress classification lines are assessed one by one. The aggregate key for all lines is never used. The worst
line governs, first by verdict, then by margin. A line with no result makes a pass inconclusive, and a fail on
another line stays a fail.

*Verdict of a requirement:* fail if any item fails; else inconclusive if any item is inconclusive; else not
assessed if an item or criterion has no verdict; else pass. A requirement that is not verified by analysis, or
has no criterion, is not assessed. The governing item is the one with the smallest margin among the items that
decide.

*Caveats.* Findings of severity `error` or `major` whose subject is a record that the item rests on (the
requirement and its parameters, the load case, its phase and effects, the numbers behind the limit and the
factor, the material, the location) appear in the statement. Such a caveat blocks a pass: the verdict becomes
**inconclusive**, the margin stays as computed, and the statement and the first reason name the finding. A pass
on a requirement whose own numbers are in question tells the engineer nothing about the real environment. The
block lifts when the finding no longer exists (the inputs were corrected) or when a person rejects it in
`reviews.yaml` (the finding is wrong). A review that accepts the finding (the requirement is wrong) keeps the
block, and a stale review does not count. A fail stays a fail. A decision records the rule.
`questioned_requirement_blocks_pass: false` in the policy restores the old rule: a caveat only.

## Decisions, assumptions and reviews

Three record kinds keep choices visible. A **decision** is a choice with its reason and the alternatives (a
selected or rejected load case, a boundary condition rule, the mesh sizes, the verdict rule for unconverged
results). An **assumption** is something taken as true with what happens if it is wrong; it can be marked as
not on the safe side. A **finding** is a problem in the inputs (category, severity). Every record has an id
that is the same on every run, an origin (the rule, the stage, or an agent), and links.

A person reviews records in `reviews.yaml`, from the command line or the dashboard: accept, reject, or request
changes, with a name and a comment. A review is tied to the content of the record. If the record changes, the
review turns *stale*: the record needs a new review. The report lists open reviews. A decision that asks for a
review, an assumption that is not conservative, a proposal of an agent, a rejected or stale review all appear
under "Needs attention". A review does not change a verdict by itself. Verdicts come from the rules above; the
one rule that reads reviews is the block of a pass by a finding, which a rejected review lifts at the next
assess stage.

## The report

`out/report/index.html` is one self-contained page (no external script, font or picture), and
`out/report/report.md` has the same content as text. It has eleven sections: Summary (verdicts, review
progress, stage status), Needs attention, System context (diagram, effects by phase, named numbers with their
sources), Component (CAD, features, interfaces, stress classification lines), Requirements and their
validation, Load cases (with a matrix of criteria against cases), Hand calculations (formulas, inputs,
sources), Analyses (meshes, convergence, checks, pictures), Compliance (the margin of each criterion, with the
governing item), Decisions and assumptions, and About this report (input files with hashes, the STE check of
the text, limits).

The page supports three ways to follow the records (`dossier/trace.py`):

- **Result flow**: requirement, load case, analysis, margin, verdict, and back.
- **Decision flow**: the decisions and assumptions of each stage, in order.
- **Justification flow**: for any record, what it rests on, down to a number with its source document; and in
  the other direction, which results change if a number changes.

Inside the dashboard the review buttons of the page work; opened as a file, the page shows the reviews of
the build time.

## Language-model roles

`dossier/agents/` has three reviewer roles: `requirements-reviewer` (the specification against the system
context), `plan-reviewer` (load cases and hand calculations) and `analysis-checker` (each analysis, as an
independent checker). A role reads a brief (the dossier as text, with instructions) and submits a JSON
document of findings, decisions and assumptions. What a role does: it notices what rules cannot, such as a
requirement that can be read two ways, a missing load path, an idealization that is wrong, an assumption
that is not conservative. What it does not do: it gives no review, changes no input file, changes no
record of a rule, and cannot change a verdict. Its records are marked "proposed by an agent", decisions
need review, and `submit` rejects a submission that names a record that does not exist, leaves a required
text empty, repeats an existing finding or uses a word outside the vocabulary. Numbers in the text that
are not numbers of the dossier produce a warning.

What was exercised: the mock path (`python -m dossier.agents run PROJECT ROLE --mock`, a canned submission,
no model call), the submission checks in `tests/dossier/test_dossier_agents.py`, and a hand-driven pilot in
which an agent read a brief and submitted through the shell. The API loop (`dossier.agents.loop`, one structured call
per role through `ace.llm`) has not run on a real model.

## The example and its results

The example is the body of a main oxidizer valve (invented data; `dossier/examples/mov_valve/README.md`). The
table below is from one full run (not `--quick`) of the final code: `out/assess.json` and the analysis records in
`out/analyses/` of that run. The run took about 15 minutes on one machine; the flow analysis took 7.4 minutes of
it and each structural or thermal case about 43 seconds. The verdicts are from the assess stage of the same run
again (`--from assess`), with the rule that a questioned requirement blocks a pass; no analysis ran again.

*What the run did.* 13 load cases: 11 solved, 2 scaled from `LC-PROOF` (`LC-BURST`, `LC-LEAK`). 12 hand-calculation
comparisons, all inside their bands. The validation stage wrote 12 findings: 7 major and 5 information. The 7 major
findings cover all six planted defects (the 009 requirement gives two: no measurable criterion and a vague word):
MEOP 10.5 MPa below the 10.72 MPa of the context, yield factor 1.1 below the policy value of 1.25, a fluid
temperature range that does not include 90 K, 200 design cycles against 267, "sufficiently stiff" with no criterion,
and the transport phase that no requirement covers.

| Requirement | Verdict | Deciding margin | What governs | Why |
|---|---|---|---|---|
| REQ-MOV-001 MEOP | not assessed | none | the verification method is review | Review is the method, not analysis. Finding `FND-REQ-MOV-001-MEOP-LOW` says the MEOP is too low. |
| REQ-MOV-002 proof pressure | inconclusive | +0.48 (not a verdict) | von Mises membrane plus bending on `SCL-FLOOR`, `LC-PROOF`, 76.2 MPa against 170 MPa yield | The result is not converged (uncertainty 25 %, limit 5 %), but it has an error estimate. The margin is +0.48 with 2 times the uncertainty on the unsafe side. Finding `FND-REQ-MOV-001-MEOP-LOW` questions the MEOP behind the proof pressure, so the pass is blocked until a person resolves it. |
| REQ-MOV-003 burst pressure | inconclusive | +2.67 (not a verdict) | von Mises membrane on `SCL-SHELL`, `LC-BURST` (scaled from the proof case), 116.3 MPa against 485 MPa ultimate | Not converged (6.9 %), with an error estimate. The margin holds with 2 times the uncertainty. Blocked by `FND-REQ-MOV-001-MEOP-LOW`, as for 002. |
| REQ-MOV-004 margins in all phases | inconclusive | -0.38 (not a verdict) | peak von Mises stress in `LC-FIRE`, 247.6 MPa against 170 MPa yield, factor 1.1 | All 8 items are inconclusive. The 3 ultimate items (on stress classification lines) have margins that hold, but finding `FND-REQ-MOV-004-YIELD-FACTOR` on the requirement blocks their pass. The other 5 are inconclusive by the result. The peak is at a sharp re-entrant edge, where linear elasticity gives no limit; the three meshes do not converge, so there is no error estimate. `LC-SHUTDOWN` reads 171.5 MPa (margin -0.099) in the same way, `LC-CHILL` is a singular peak. The criterion asks for the peak, so the result cannot say fail or pass. |
| REQ-MOV-005 temperature range | not assessed | none | the verification method is review | Finding `FND-REQ-MOV-005-MIN-TEMPERATURE`: the range 233 K to 323 K excludes the 90 K of the LOX. |
| REQ-MOV-006 fatigue | inconclusive | +375.6 (not a verdict) | damage at the part, `LC-FATIGUE`: 1.76e-5 against 1, scatter factor 4 | The analysis check on yield fails: 247.6 MPa at the peak is above the 170 MPa yield strength in the first cycle, so the elastic stress-life damage is not valid. The damage also changes by a factor of 37.7 between the two finest meshes. Finding `FND-REQ-MOV-006-CHECK-DESIGN-CYCLES`: 200 cycles against 267 in the context. |
| REQ-MOV-007 actuator face temperature | pass | +0.034 | minimum temperature on `INT-ACT`, `LC-TH-FIRE`: 242.15 K against 233 K | Converged (uncertainty 1.2 K). The hand calculation `HC-ACT-TEMP` agrees. The margin is small. |
| REQ-MOV-008 pressure drop | inconclusive | +7.04 (not a verdict) | pressure drop of the open valve, `LC-FLOW-FIRE`: 5.84 kPa against 60 kPa | The uncertainty is 28 % and it is not an error estimate. On the coarsest mesh the residual stalled at 6.2e-3, above the stall limit of 3e-3, and the three meshes (3.554, 3.552 and 3.58 kPa, smooth walls) do not converge in order; the roughness correction adds 2.26 kPa. The hand calculation agrees. The margin is large, but the rules do not let an unconverged result without an estimate count. |
| REQ-MOV-009 stiffness | not assessed | none | no criterion | "Sufficiently stiff" has no measure. Findings `FND-REQ-MOV-009-NOT-VERIFIABLE` and `-VAGUE`. |
| REQ-MOV-010 out-of-roundness | pass | +0.31 | range of radial displacement of `FT-SEAT-OUT-WALL`, `LC-PRESTART`: 13.7 µm against 20 µm | Not converged (6 %, limit 5 %), with an error estimate. The margin holds with 2 times the uncertainty. |

Count: 2 pass, 5 inconclusive, 3 not assessed, no fail. Read the table with these points.

- **No verdict is "fail", although REQ-MOV-004 shows a negative margin.** The margin is of a peak that has no limit
  in linear elasticity. The analysis cannot tell a real overstress from the mathematical singularity. A person
  must decide: model the real fillet radius, or classify the stress on a line.
- **Margins rest on the k = 2 rule** for REQ-MOV-002, -003 and -010. Their results are not converged by the 5 %
  policy but have an error estimate. The margin is shown with 2 times the uncertainty.
- **The five inconclusive verdicts have different causes**: a singular peak (004), a material that yields so the
  method does not apply (006), a flow solution without an error estimate (008), and a questioned requirement
  (002, 003).
- **Questioned requirements block a pass**: REQ-MOV-002 and -003 would pass, but the system file reads the MEOP by
  name from REQ-MOV-001 (`proof_factor * meop`), so the proof and burst pressures are 1.5 and 2.5 times 10.5 MPa,
  not times 10.72 MPa. Whether the margins still hold at the higher MEOP the run does not show. A correction of
  the MEOP (and the analyses again), or a review that rejects `FND-REQ-MOV-001-MEOP-LOW`, lifts the block.
- The inputs are example data. The numbers show what the tool does with them, not how a valve body performs.

## Limits

- **Linear elastic only.** Small displacements, no yield, no contact, no bolt preload, no gasket. The joints are
  fixed supports, so the stress next to a seat is approximate.
- **Thermal is steady state in the pipeline.** The solver now has transient heat conduction
  (`docs/solver_transient.md`), but the planner does not use it. A steady temperature does not bound the thermal
  stress of a chill-down. The load cases name that assumption.
- **One material per component**, one conductivity and one set of properties at all temperatures (an assumption
  tests +-20 % on the conductivity).
- **No buckling and no natural-frequency load case.** The solver now has modal analysis. The planner does not
  create such cases, and no requirement quantity maps to them. Euler buckling exists only as a hand calculation.
- **Fatigue is elastic stress-life only** and stops where the part yields: there it gives no valid damage and
  no pass. No strain-life method, no fatigue strength reduction, no stress between phases, no sequence effect.
- **Flow handles stepped bores of revolution** with a wall-function RANS model in the validated range only
  (Re 1e5 to 1.4e6, y+ 60 to 1600, hexahedral meshes, smooth walls plus a roughness correction). It carries a
  model uncertainty of 10 %. Other shapes, bends, swirl and cavitation are `not_supported`.
- **Peak stresses at sharp edges and support rims have no limit** in linear elasticity. They are marked
  singular and give inconclusive verdicts. The remedy is a stress classification line or a model of the
  real geometry, not a finer mesh.
- **The mesh size floor** is 1 % of the bounding-box diagonal. Small fillets are resolved to a few percent at
  best.
- **The STE check** covers the writing rules that a program can examine (sentence length, tenses, modal verbs,
  contractions, a small list of words that are not approved, heuristics for the passive voice and noun
  clusters). It does not hold the ASD dictionary. Without the user's dictionary file the vocabulary check is
  partial. Meaning, one topic per sentence and the article rules are not checked.
- **CAD**: one solid, STEP. Millimetre-unit files from a CAD system were tested with rewritten headers of the
  example only. Free-form faces have no radius or axis.
- **The example's data is invented.** Do not use a number from it for design.
- **Not verified in the structural engine**: the thermal support with a real gradient, moment loads on curved
  faces.
- **The language-model roles** have not run through the API on a real model (see above).

## What was checked independently

Two independent checks of the code were made, and the team fixed what they found. This list is the team's report
of them; the fixes are in the code, the checks themselves are not part of the repository.

- The verdict rules had three ways to a false pass. They are closed, and
  `tests/dossier/test_dossier_compliance.py` tests the rules above.
- A conversion error of a per-degree unit (a temperature difference treated as a temperature) was fixed.
- The convergence rule of the flow engine was weak: it compared two moments of the iteration, which an
  oscillation or a slow drift could pass. It now uses every snapshot over a window of 500 iterations
  (spread and trend both under 0.1 %).
- Transient heat conduction missed a start-up after a later jump in a boundary value. This was fixed in the solver.
- The frequency estimate of a transient result could be wrong without a warning. This was fixed in the solver.

Not independently checked: the transient incompressible flow capability. Its own tests exist (Taylor-Green,
DFG 2D-2, `docs/solver_transient.md`), but no one other than its author reviewed it. The rest of the new
solver capabilities (modal analysis, transient dynamics, compressible internal flow) are reported as verified
in `docs/solver_transient.md` by their authors' tests.

Test status: `pytest -m "not slow" tests meshing-harness/tests` on the final code gave 1938 passed, 71 skipped, 73 deselected
and 1 failed in 20 minutes. The one failure is `tests/test_code_aster.py::test_vendored_manifest_and_mesh_groups`: the
`h5py` module is not installed in this environment. It is outside `dossier`. The slow tests (`-m slow`, among them the
turbulent flow validation) were not part of this run.

## Files

| file | role |
|---|---|
| `model.py` | the inputs and the records (pydantic) |
| `units.py` | quantities with units (pint), expressions with a dimension check |
| `project.py` | the project on disk, stage files, reviews |
| `context.py` | every named number evaluated, every effect in SI per phase, effects mapped to features |
| `pipeline.py`, `cli.py` | the stages and the command line |
| `trace.py` | the trace graph: what each record rests on, what depends on it |
| `geometry.py`, `meshing.py` | features to CAD faces; the shared mesh study |
| `system/`, `requirements/` | the system check and diagram; the requirement rules |
| `loadcases/`, `handcalc/` | load-case selection and scaling; closed-form models |
| `analysis/` | engines: `structural`, `thermal`, `fatigue`, `flow`, `scaled`, with `idealize` and `scl` |
| `compliance.py` | margins and verdicts |
| `report/` | the HTML and Markdown report |
| `agents/` | language-model reviewer roles: brief, submit, mock, API loop |
| `ste/` | the Simplified Technical English checker and writer helpers |
| `examples/mov_valve/` | the worked example, with its own README |
| `records.py` | how a stage writes decisions, assumptions and findings |
| `CONTRACTS.md` | the contract of each engine, for people who add or change one |
