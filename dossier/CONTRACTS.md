# dossier: contracts for engine authors

`dossier` verifies one component by analysis, in the context of its system. This file is the contract
between the core (data model, project store, context, pipeline) and the engines (one module per stage).
Read `model.py` and `pipeline.py` with it; they are short and they are the source of truth.

## The idea in one picture

```
 system.yaml ─┐                                     ┌─ findings (is the specification right?)
 component  ──┼─► context ─► geometry ─► validate ──┤
 requirements ┘   (numbers)  (faces)                └─► plan ─► analyze ─► assess ─► report
                                                        load      one engine  margins   linked,
                                                        cases +   per type    verdicts  STE100
                                                        hand calcs
 every stage also writes: decisions (a choice + why + alternatives), assumptions (+ impact)
 a person reviews them: reviews.yaml (accept / reject / changes requested)
```

Rules of the model (do not work around them):

1. **Loads come from the system context** (effects in phases). Requirements give criteria and factors. A
   number a requirement states (MEOP, design cycles) is a *claim*; `validate` compares it with the context.
2. **A load case is a phase at limit loads** (factor 1.0). Factors of safety are applied to the response in
   the margin. One solve serves every criterion of the phase.
3. **Nothing is silent.** A choice is a `Decision` with its rationale and the alternatives. A thing taken as
   true is an `Assumption` with its impact. A problem in the inputs is a `Finding`. Use `records.Journal`.
4. **Every record links to what it comes from** (`links`, typed fields; `Record.refs()` lists them all).
5. **Verdicts are honest**: pass, fail, inconclusive, not_assessed. A result that is not converged, a
   singular peak, or a hand calculation that disagrees makes a verdict inconclusive, never a pass.
6. **SI inside records** (Pa, m, N, K, kg, s). The unit people read is next to it (`display`). Use
   `dossier.units` for every conversion; never multiply by 1e6 by hand.
7. **Text is ASD-STE100** (see "Text" below).

## The example project (the fixture every engine runs on)

`dossier/examples/mov_valve/`: the body of a main oxidizer valve in a pump-fed LOX / kerosene engine.
CAD frame: x = flow (inlet at -x), z = stem axis, origin at the ball centre, metres. 47 faces, 18 named
features, 5 interfaces, 5 stress classification lines, 6 hand-calculation specs, 8 phases, 3 duty cycles,
67 effects, 10 requirements. The specification holds six deliberate defects:

| # | requirement | defect | what `validate` must find |
|---|---|---|---|
| 1 | REQ-MOV-001 | MEOP 10.5 MPa, but the context gives 10.72 MPa at the inlet during shutdown (pump maximum + closure surge) | role `meop` < largest pressure of the operation phases |
| 2 | REQ-MOV-004 | yield factor 1.1, the policy asks for 1.25 | role `yield_factor` < policy |
| 3 | REQ-MOV-005 | fluid temperature 233 K to 323 K, but the fluid is LOX at 90 K | role `min_temperature` > lowest fluid temperature |
| 4 | REQ-MOV-006 | 200 design cycles, the context gives 267 pressure cycles | explicit `check` fails |
| 5 | REQ-MOV-009 | "sufficiently stiff", no criterion | not verifiable: vague word, no criterion |
| 6 | (none) | no requirement applies to PH-TRANSPORT (category handling) | a phase with loads on the part that no requirement covers |

```
.venv/bin/python -m dossier check  dossier/examples/mov_valve
.venv/bin/python -m dossier params dossier/examples/mov_valve dp_surge p_max_inlet
.venv/bin/python -m dossier run    dossier/examples/mov_valve --quick     # all stages; coarse meshes
```

Use the repo's virtual environment (`.venv/bin/python`). 3D pictures need a display: `xvfb-run -a ...`.

## Core API (do not edit these files; ask for a change in your final report)

| module | what it gives you |
|---|---|
| `dossier.model` | every input model and every record (pydantic, `extra="forbid"`) |
| `dossier.units` | `parse("10.4 MPa")`, `si(q)`, `si_unit(q)`, `display(q)`, `fmt(q, unit)`, `from_si(v, unit)`, `evaluate(expr, names, functions)` |
| `dossier.project.Project` | `Project.load(root)`; `.system`, `.subject`, `.components`, `.reqs()`, `.material_of()`, `.policy`, `.phases_of(req)`, `.inputs()` (every input record by id), `.cad_path()`, `.out`, `.analysis_dir(lc)`, `.collection("load_cases")`, `.records()`, `.reviews()` |
| `dossier.project.make_id` | `make_id("LC", "PH-FIRE")` -> `LC-PH-FIRE`: stable ids, the same on every run |
| `dossier.context` | `build(project) -> Context`; `ctx.params[name]` (`.si`, `.unit`, `.display`, `.expr`, `.depends_on`, `.source`, `.basis`, `.origin`), `ctx.names()` (pint quantities for expressions), `ctx.effects`, `ctx.effects_on(element, phase, kind)`, `ctx.loads(project, component, phase) -> [LoadItem]`, `ctx.envelope(project, component, kind, phases, how)`, `ctx.duty_cycles`, `ctx.functions(loads, geometry)` (the `load()` / `feat()` callables of hand-calculation inputs) |
| `dossier.records.Journal` | `j.decide(...)`, `j.assume(...)`, `j.find(...)` -> records with stable ids and an origin |
| `dossier.pipeline` | the stages, `Options` (`quick`, `levels`, `node_cap`, `pictures`, `only`, `log`), `Plan`, `Analysis`, `StageError` |

`Options.quick` means: a test or a first look. Coarse meshes (finest about 25 000 nodes), two levels, no
pictures. Every engine must finish its part of the example in well under a minute with `quick`.

## Engine contracts

Signatures are exact. Each engine is a pure function of its arguments and of the stage files written
before it; only `pipeline.py` writes stage files.

### geometry — `dossier/geometry.py`, `dossier/meshing.py`

```python
def build(project, opts) -> tuple[dict, Journal]
```
The dict (JSON) is what every later stage calls `geo`:

```
{"component": "CMP-...",
 "cad": {"file": "...", "sha256": "...", "volume": m3, "mass": kg, "bbox": [[x0,y0,z0],[x1,y1,z1]], "n_faces": 47},
 "features": {"FT-ID": {"faces": [44], "area": m2, "centroid": [x,y,z], "kind": "cylinder|plane|torus|mixed",
                         "radius": m|None, "diameter": m|None, "axis": [..]|None, "normal": [..]|None,
                         "length": m|None, "description": "..."}},
 "interfaces": {"INT-ID": {"wetted": [tags], "seat": [tags], "bolts": [tags], "external": [tags]}},   # "*" resolved
 "faces": {"44": {"kind": ..., "area": ..., "centroid": [...], "normal": ..., "radius": ..., "axis": ...,
                  "features": ["FT-..."], "roles": ["INT-INLET:wetted"]}},
 "edges": {"reentrant": [{"tag": 12, "faces": [37, 42], "angle_deg": 270.0, "length": m, "midpoint": [x,y,z]}]},
 "scls": {"SCL-ID": {"start": [...], "end": [...], "length": m, "ok": true, "message": ""}},
 "pictures": [{"path": "geometry/overview.png", "caption": "..."}],     # relative to project.out
 "problems": ["..."]}
```
`external: ["*"]` means every face that no other role of any interface uses. Findings (category `model`):
a feature whose selection gives a number of faces other than `expect`; a face that is `wetted` in two
interfaces; a stress classification line whose ends are not on the surface or that leaves the material.

```python
# dossier/meshing.py
def levels(project, geo, opts, extra_refinements=None) -> list[MeshLevel]   # coarse -> fine, cached in out/meshes/
def part_for(project, component_id=None)          # the feacore Problem of the component (for ace.fea helpers)
GROUP_SOLID = "solid"
```
Quadratic tetrahedra. Physical groups in every mesh: `f<tag>` for every face, **every feature id** (so
`FT-INLET-BORE` is a group name), and `solid`. Sizes are chosen so that the finest mesh stays under
`opts.node_cap`, with local refinement at re-entrant edges and small fillets; successive levels differ by a
factor of about 1.4 in size. `MeshLevel.path` is relative to `project.out`.

### validate — `dossier/system/check.py`, `dossier/system/diagram.py`, `dossier/requirements/validate.py`

```python
def check(project, ctx, geo) -> Journal        # dossier.system.check: is the system model sound and complete?
def render(project, ctx) -> str                # dossier.system.diagram: the system diagram, one SVG document
def validate(project, ctx, geo) -> Journal     # dossier.requirements.validate: are the requirements right?
```

### plan — `dossier/loadcases/`, `dossier/handcalc/`

```python
def plan(project, ctx, geo) -> pipeline.Plan            # dossier.loadcases (package __init__ exports plan)
# dossier.handcalc
def evaluate(spec: HandCalcSpec, project, ctx, geo, load_case: LoadCase | None) -> HandCalcResult
def applies(spec: HandCalcSpec, load_case: LoadCase) -> bool        # spec.when against the load case
def compare(hc: HandCalcResult, analysis: AnalysisRecord, project) -> list[Check]
```
Load case ids: `LC-<phase id without PH->` for a structural case of a phase (`LC-FIRE`), `LC-TH-<phase>` for
its thermal case, `LC-FATIGUE`, `LC-FLOW-<phase>`. `LoadCase.result_keys` lists the results the analysis
must give: the keys of its criteria and of its hand-calculation comparisons. A proportional case (every load
is the same multiple of another case's loads, same supports) is not solved: `scaled_from` + `scale`,
`status: "scaled"`. A thermo-structural case lists its thermal case in `uses`.

### analyze — `dossier/analysis/{structural,thermal,fatigue,flow,scaled}.py`

```python
def run(project, ctx, geo, lc: LoadCase, workdir: Path, opts) -> pipeline.Analysis     # (AnalysisRecord, Journal)
```
`AnalysisRecord.id` is `AN-` + the load case id without `LC-`. `workdir` is `project.analysis_dir(lc.id)`;
write meshes, fields and pictures there and list them in `artifacts` with paths relative to `project.out`.
If the engine cannot do the load case, return a record with `status="not_supported"` and a `message` that
says exactly what is missing. Never invent a result.

Result keys: `<quantity>.<measure>@<location>`.

| part | values |
|---|---|
| quantity | `von_mises`, `max_principal`, `sxx` `syy` `szz` (global components, on an SCL), `displacement` (magnitude), `radial_displacement` (about the axis of a cylindrical feature), `temperature`, `heat_flow`, `pressure_drop`, `damage` |
| measure | `peak` (the largest value in the location, with its convergence status and where it is), `max`, `min`, `mean`, `range` (max - min), `membrane`, `membrane_bending`, `total` |
| location | `part`, a feature id, an interface id (all its features), an SCL id, `scls` (the worst over every SCL) |

Section stresses on an SCL: sample the six stress components along the line, average them (membrane tensor),
fit the linear part (bending tensor), and take the invariant of the tensor: `von_mises.membrane@SCL-X` is
the von Mises stress of the membrane tensor; `membrane_bending` is the larger invariant of membrane +
bending at the two ends.

Every `ResultValue` has a `status`: `converged` (the discretization uncertainty from the mesh study is under
`policy.max_uncertainty`), `not_converged`, `singular` (the value grows without bound under refinement: a
peak at a sharp re-entrant edge or at the rim of a rigid support), `estimate` (a hand calculation or a
bound), `not_run`. `uncertainty` is relative (0.02 = 2 %). `uncertainty_abs` is the same in the unit of the
value; a temperature result gives `uncertainty_abs` (in K), and when it is given the compliance stage uses it
in place of the relative one. `per_level` holds the value on each mesh.

Fields for later engines (fatigue): for the two finest meshes write `fields_L<k>.npz` in the workdir with
`x` (n, 3) node coordinates, `stress` (n, 6) = sxx, syy, szz, sxy, syz, sxz in Pa, `u` (n, 3) in m, and for
a thermal analysis `T` (n,) in K. List them as `Artifact(kind="field")`.

Checks every analysis makes on itself (`AnalysisRecord.checks`): equilibrium (reactions against applied
loads), mesh quality, the error estimate, and one check per hand-calculation comparison (the pipeline adds
those with `handcalc.compare`).

### assess — `dossier/compliance.py`

```python
def assess(project, ctx) -> Assessment      # .items: [ComplianceItem], .verdicts: [RequirementVerdict], .journal
```
Margin of safety: `MS = allowable / (factor x response) - 1` for `<=`; `MS = (response / factor) / limit - 1`
for `>=`. The response enters with its uncertainty on the unsafe side (`conservative_value`).

### report — `dossier/report/`, `dossier/ste/`

```python
def build(project, opts) -> dict            # dossier.report: {"html": path, "md": path, "ste": {...}}
# dossier.ste
def check(text, kind="descriptive", terms=None) -> list[Issue]     # ASD-STE100 writing rules
```

## Text: ASD-STE100 Simplified Technical English

Every text field an engine writes (`statement`, `rationale`, `choice`, `impact`, `suggestion`, `title`,
check statements) is read by the engineer who signs the work. Write it in STE:

- One topic per sentence. A descriptive sentence has 25 words or fewer; an instruction has 20 or fewer.
- Active voice. Simple tenses (present, past, future). No "-ing" verb forms, no "has been", no "would".
- No "should", "may", "might", "could". Use "must" for an obligation and "can" for an ability.
- Use the same word for the same thing every time (always "load case", never "loading scenario").
- Give numbers with units, and say where a number comes from: "The requirement gives 10.5 MPa."
- No contractions, no "etc.", no "and/or", no Latin abbreviations ("e.g.", "i.e.").
- A paragraph has six sentences or fewer.
- Use the article: "the valve body", not "valve body".

Good: "The requirement gives an MEOP of 10.5 MPa. The system context gives 10.72 MPa at the inlet during
shutdown. The requirement value is too low." Bad: "MEOP may be insufficient considering surge effects."

Text an engineer wrote (requirement text, descriptions in the YAML files) is quoted as it is.

## Tests and working rules

- Tests go in `tests/dossier/test_dossier_<your area>.py` (the `dossier_` in the file name is needed: test
  file names must be unique in the repository, and the folder has no `__init__.py`). Fixtures in
  `tests/dossier/conftest.py`: `example` (the example project, read-only), `example_ctx`, `workproject`
  (a fresh writable copy). Run only your own file while you work
  (`.venv/bin/python -m pytest tests/dossier/test_dossier_<area>.py -q -p no:cacheprovider`). Mark anything
  over about 20 s `@pytest.mark.slow`.
- Other agents are working in this repository at the same time. Edit only the files your brief gives you.
  Do not run git commands that change the repository (no commit, stash, checkout, reset). Do not edit
  `dossier/model.py`, `project.py`, `context.py`, `pipeline.py`, `records.py`, `units.py`, `cli.py` or the
  example's YAML files unless your brief says so; if you need a change there, say exactly what in your final
  report.
- No new dependency without need. Available: numpy, scipy, pint, pydantic, pyyaml, gmsh, meshio,
  scikit-fem, pyvista, matplotlib, fastapi.
- Do not write planning documents. Put what a user must know in your module docstrings.

## Shared conventions between engines

- `dossier.context.proportional(loads_a, loads_b) -> float | None`: the scale between two phases whose
  mechanical loads are the same pattern (no thermal loads). Used by the planner (scaled load cases) and by
  the requirement validation (a phase whose loads are a smaller multiple of a covered phase is covered).
- Finding ids are `make_id("FND", subject, key)`. Keys other engines link to: a phase with loads that no
  requirement covers: subject = the phase id, key `uncovered` (`FND-PH-TRANSPORT-UNCOVERED`); a requirement
  that analysis cannot verify (no criterion): subject = the requirement id, key `not-verifiable`.
- When an analysis is not available for a load case, the compliance stage can use the hand calculation of
  that load case for the same result key as an *estimate*: the verdict is then inconclusive, with the margin
  of the estimate shown.
- Stress at a fixed support is not a result: the peak evaluation leaves out the zone next to a fixed face
  (the structural engine records the distance as a decision).

## Report, dashboard and agent roles

- `dossier.trace.graph(project) -> {"nodes": {...}, "edges": [...]}` is the one trace graph (inputs, context
  numbers and derived records, with `upstream` and `downstream`). The report and the dashboard read it; they do
  not build their own.
- The report is one self-contained page, `out/report/index.html` (pictures beside it in `out/report/assets/`).
  It is also the reading surface of the dashboard: the server serves the same file and injects
  `<script>window.DOSSIER_API = "/api/dossier/<project id>";</script>` before `</head>`. When `DOSSIER_API` is
  set, the page turns its review controls on:
  `GET  {DOSSIER_API}/reviews` -> `{"reviews": {record id: {"state", "by", "date", "comment", "stale"}}}`;
  `POST {DOSSIER_API}/reviews` with `{"record", "state", "by", "comment"}` -> the stored review
  (`state`: accepted | rejected | changes_requested | open). Without it the page shows the reviews that
  `reviews.yaml` held when the report was built, and the command to give one.
- Records an LLM role proposes go in `<project>/agent/<role>.json`, a stage file with the usual collections
  (`findings`, `decisions`, `assumptions`) whose records have `origin.by = "agent:<role>"`. `Project` reads
  them with the other stage files, so they appear in the report, marked as proposed by an agent, and a person
  reviews them like any other record.
