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
   peak, or a hand calculation that disagrees makes a verdict inconclusive, never a pass.
6. **Rules first, judgement on top.** The engines are deterministic and reproducible. Language-model roles
   read the dossier and propose findings and decisions; their records are marked as proposed by an agent and go
   through the same review.
7. **Text is ASD-STE100** Simplified Technical English, checked by `dossier.ste`.

## Files

| file | role |
|---|---|
| `model.py` | the inputs and the records (pydantic) |
| `units.py` | quantities with units (pint), expressions with a dimension check |
| `project.py` | the project on disk, stage files, reviews |
| `context.py` | every named number evaluated, every effect in SI per phase, effects mapped to features |
| `pipeline.py`, `cli.py` | the stages and the command line |
| `trace.py` | the trace graph: what each record rests on, what depends on it |
| `records.py` | how a stage writes decisions, assumptions and findings |
| `CONTRACTS.md` | the contract of each engine, for people who add or change one |
