"""The plan stage (dossier.loadcases): the load cases of the example, and each rule on a changed copy.

The tests of the planner's own rules use a stand-in for the hand-calculation engine (`StandIn`: `applies` reads
`when` as the contract says, `evaluate` gives a record without numbers), so that they test the planner and
not the formulas. `test_hand_calculations_of_the_example_with_the_real_engine` and the pipeline test run
`dossier.handcalc` itself.
"""

import re

import pytest
import yaml

from dossier import context, pipeline
from dossier.context import MECHANICAL, THERMAL
from dossier.loadcases import plan, select
from dossier.model import Comparison, HandCalcResult, LoadCase, Origin
from dossier.project import Project, make_id

SYSTEM = "system.yaml"
POLICY = "policy.yaml"
REQUIREMENTS = "components/mov_body/requirements.yaml"
COMPONENT = "components/mov_body/component.yaml"

SHELL = "max_principal.membrane@SCL-SHELL"
NOZZLE = "max_principal.membrane@SCL-NOZZLE-OUT"
FLOOR = "max_principal.membrane_bending@SCL-FLOOR"
PRESSURE_CHECKS = ["HC-SHELL-HOOP", "HC-NOZZLE-HOOP", "HC-FLOOR-PLATE"]

# the plan of the example: id -> (analysis type, status, scaled from, uses)
EXPECTED = {
    "LC-PROOF": ("static_structural", "selected", None, []),
    "LC-BURST": ("static_structural", "scaled", "LC-PROOF", []),
    "LC-LEAK": ("static_structural", "scaled", "LC-PROOF", []),
    "LC-TH-CHILL": ("thermal_steady", "selected", None, []),
    "LC-CHILL": ("thermo_structural", "selected", None, ["LC-TH-CHILL"]),
    "LC-TH-PRESTART": ("thermal_steady", "selected", None, []),
    "LC-PRESTART": ("thermo_structural", "selected", None, ["LC-TH-PRESTART"]),
    "LC-TH-FIRE": ("thermal_steady", "selected", None, []),
    "LC-FIRE": ("thermo_structural", "selected", None, ["LC-TH-FIRE"]),
    "LC-FLOW-FIRE": ("flow", "selected", None, []),
    "LC-TH-SHUTDOWN": ("thermal_steady", "selected", None, []),
    "LC-SHUTDOWN": ("thermo_structural", "selected", None, ["LC-TH-SHUTDOWN"]),
    "LC-FATIGUE": ("fatigue", "selected", None, ["LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN", "LC-PROOF", "LC-LEAK"]),
}
# the hand calculations of the example: load case -> specs
HAND_CALCS = {"LC-PROOF": PRESSURE_CHECKS, "LC-BURST": PRESSURE_CHECKS, "LC-LEAK": PRESSURE_CHECKS,
              "LC-FIRE": ["HC-NECK-BENDING"], "LC-TH-FIRE": ["HC-ACT-TEMP"], "LC-FLOW-FIRE": ["HC-PRESSURE-DROP"]}
RESULT_KEYS = {
    "LC-PROOF": {"von_mises.membrane_bending@scls", SHELL, NOZZLE, FLOOR, "von_mises.membrane@scls"},
    "LC-BURST": {"von_mises.membrane@scls", SHELL, NOZZLE, FLOOR},
    "LC-LEAK": {SHELL, NOZZLE, FLOOR},
    "LC-TH-CHILL": set(),
    "LC-CHILL": {"von_mises.peak@part", "von_mises.membrane_bending@scls"},
    "LC-TH-PRESTART": set(),
    "LC-PRESTART": {"von_mises.peak@part", "von_mises.membrane_bending@scls", "radial_displacement.range@FT-SEAT-OUT-WALL"},
    "LC-TH-FIRE": {"temperature.min@INT-ACT"},
    "LC-FIRE": {"von_mises.peak@part", "von_mises.membrane_bending@scls", "szz.membrane@SCL-NECK-ROOT"},
    "LC-FLOW-FIRE": {"pressure_drop.total@part"},
    "LC-TH-SHUTDOWN": set(),
    "LC-SHUTDOWN": {"von_mises.peak@part", "von_mises.membrane_bending@scls"},
    "LC-FATIGUE": {"damage.total@part"},
}


# --------------------------------------------------------------------------- helpers


class StandIn:
    """dossier.handcalc for the tests of the planner: `when` decides, `evaluate` gives no numbers."""

    def __init__(self, fail=(), fail_when=()):
        self.fail, self.fail_when = set(fail), set(fail_when)

    def applies(self, spec, lc):
        if spec.id in self.fail_when:
            raise KeyError("no such interface INT-NOWHERE")
        when = spec.when
        have = {f"{x.kind}@{x.interface}" for x in lc.loads}
        return ((not when.analysis or lc.analysis_type in when.analysis) and (not when.phases or lc.phase in when.phases)
                and all(need in have for need in when.needs))

    def evaluate(self, spec, project, ctx, geo, lc):
        if spec.id in self.fail:
            raise ValueError(f"load('INT-NOWHERE', 'pressure'): this load case has no such load ({spec.id})")
        return HandCalcResult(id=make_id("HCR", spec.id, lc.id), title=spec.title, spec=spec.id, model=spec.model,
                              load_case=lc.id, origin=Origin(by="engine:test", stage="plan"),
                              comparisons=[Comparison(output=c.output, result_key=c.with_, hand_value=1.0)
                                           for c in spec.compare])


def make(project, engine=None, ctx=None):
    """The plan with the stand-in hand-calculation engine."""
    return select._plan(project, ctx or context.build(project), None, engine or StandIn())


def edit(project, name, change) -> Project:
    """Change one YAML file of a work project (`change(data)` edits the parsed file) and load it again."""
    path = project.root / name
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return Project.load(project.root)


def effect(data, effect_id):
    return next(e for c in data["connections"] for e in c.get("effects", []) if e["id"] == effect_id)


def requirement(data, req_id):
    return next(r for r in data["requirements"] if r["id"] == req_id)


def cases(p):
    return {lc.id: lc for lc in p.load_cases}


def decisions(p):
    return {d.id: d for d in p.journal.decisions}


def assumptions(p):
    return {a.id: a for a in p.journal.assumptions}


def findings(p):
    return {f.id: f for f in p.journal.findings}


def linked(record):
    return {l.to for l in record.links}


# ---- a small check of the STE rules of CONTRACTS.md (the report stage has the full checker, dossier.ste)

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_BANNED = re.compile(r"\b(should|may|might|could|would|shall|etc|has been|have been|had been)\b|e\.g\.|i\.e\.|and/or|"
                     r"\w'(t|s|re|ll|ve|d)\b", re.I)
_ING = re.compile(r"\b[a-z]+ing\b")
_ING_WORDS = {"during", "thing", "nothing", "anything", "everything", "finding"}     # "finding": a kind of record


def ste_problems(p, project) -> list[str]:
    """Sentences that are too long, paragraphs with too many sentences, and words the STE rules forbid, in
    every text the planner wrote. Names an engineer wrote (phase names, titles) are quoted text."""
    quoted = sorted({ph.name for ph in project.system.phases} | {r.title for r in project.reqs()}
                    | {h.title for h in project.subject.hand_calcs}, key=len, reverse=True)
    out = []

    def check(rid, field, text, limit=25, sentences=True):
        plain = text
        for q in quoted:
            plain = plain.replace(q, "NAME")
        for m in _BANNED.finditer(plain):
            out.append(f"{rid}.{field}: '{m.group(0)}' in: {text}")
        for word in _ING.findall(plain):
            if word not in _ING_WORDS:
                out.append(f"{rid}.{field}: the -ing form '{word}' in: {text}")
        if not sentences:
            return
        parts = [s for s in _SPLIT.split(plain.strip()) if s]
        if len(parts) > 6:
            out.append(f"{rid}.{field}: {len(parts)} sentences in one paragraph: {text}")
        for s in parts:
            if len(s.split()) > limit:
                out.append(f"{rid}.{field}: {len(s.split())} words (limit {limit}): {s}")
            if not s.rstrip().endswith((".", "!", "?")):
                out.append(f"{rid}.{field}: no full stop: {s}")

    for lc in p.load_cases:
        check(lc.id, "title", lc.title, sentences=False)
        check(lc.id, "statement", lc.statement)
        if lc.note:
            check(lc.id, "note", lc.note)
    for d in p.journal.decisions:
        check(d.id, "title", d.title, sentences=False)
        check(d.id, "choice", d.choice, limit=20)
        check(d.id, "rationale", d.rationale)
        for a in d.alternatives:
            check(d.id, "option", a.option, limit=20)
            check(d.id, "why_not", a.why_not)
    for a in p.journal.assumptions:
        check(a.id, "title", a.title, sentences=False)
        check(a.id, "statement", a.statement)
        check(a.id, "rationale", a.rationale)
        check(a.id, "impact", a.impact)
        check(a.id, "verify_by", a.verify_by, limit=20)
    for f in p.journal.findings:
        check(f.id, "title", f.title, sentences=False)
        check(f.id, "statement", f.statement)
        if "error:" not in f.rationale:                 # an error text of the software is quoted as it is
            check(f.id, "rationale", f.rationale)
        check(f.id, "suggestion", f.suggestion, limit=20)
    return out


def sound(p, project):
    """What must hold for every plan: unique ids, uses that exist, an analysis order, records that read back,
    text in simple English, and one `selected` (or `not-supported`) decision with alternatives per load case."""
    ids = [lc.id for lc in p.load_cases]
    assert len(set(ids)) == len(ids)
    order = [lc.id for lc in pipeline.analysis_order(p.load_cases)]
    assert sorted(order) == sorted(ids)
    for lc in p.load_cases:
        for dep in [*lc.uses, *([lc.scaled_from] if lc.scaled_from else [])]:
            assert dep in ids and order.index(dep) < order.index(lc.id), (lc.id, dep)
        assert LoadCase.model_validate(lc.model_dump(mode="json", by_alias=True)) == lc
        assert lc.component == project.subject.id and lc.title and lc.statement and lc.origin.stage == "plan"
        assert all(x.factor == 1.0 for x in lc.loads)                   # limit loads
        assert lc.hand_calcs == [h.id for h in p.hand_calcs if h.load_case == lc.id]
    dec = decisions(p)
    assert len(dec) == len(p.journal.decisions)                         # ids are unique
    for lc in p.load_cases:
        key = "not-supported" if lc.status == "not_supported" else "selected"
        d = dec[make_id("DEC", lc.id, key)]
        assert d.subject == lc.id and d.stage == "plan" and d.alternatives and d.rationale
        assert (make_id("DEC", lc.id, "scaled") in dec) == (lc.status == "scaled")
    for d in p.journal.decisions:
        assert d.alternatives, d.id                                     # every choice names what it did not do
    assert ste_problems(p, project) == []
    return p


# --------------------------------------------------------------------------- the example


@pytest.fixture(scope="module")
def example_plan(example, example_ctx):
    return make(example, ctx=example_ctx)


def test_example_load_cases(example, example_ctx, example_plan):
    lcs = cases(sound(example_plan, example))
    assert list(lcs) == list(EXPECTED)
    for lc_id, (kind, status, base, uses) in EXPECTED.items():
        lc = lcs[lc_id]
        assert (lc.analysis_type, lc.status, lc.scaled_from, lc.uses) == (kind, status, base, uses), lc_id
        assert set(lc.result_keys) == RESULT_KEYS[lc_id] and len(lc.result_keys) == len(RESULT_KEYS[lc_id]), lc_id
    assert lcs["LC-BURST"].scale == pytest.approx(2.5 / 1.5) and lcs["LC-LEAK"].scale == pytest.approx(1 / 1.5)
    assert all(lc.scale == 1.0 for lc in lcs.values() if not lc.scaled_from)
    # no load case for the phase that no requirement covers
    assert all(lc.phase != "PH-TRANSPORT" for lc in lcs.values())
    assert {lc.phase for lc in lcs.values()} == {"PH-PROOF", "PH-BURST", "PH-LEAK", "PH-CHILL", "PH-PRESTART",
                                                 "PH-FIRE", "PH-SHUTDOWN", None}


def test_example_criteria(example_plan):
    lcs = cases(example_plan)
    crit = {i: [(c.requirement, c.criterion) for c in lc.criteria] for i, lc in lcs.items()}
    operation = [("REQ-MOV-004", 0), ("REQ-MOV-004", 1)]
    assert crit == {
        "LC-PROOF": [("REQ-MOV-002", 0)], "LC-BURST": [("REQ-MOV-003", 0)], "LC-LEAK": [],
        "LC-TH-CHILL": [], "LC-CHILL": operation, "LC-TH-PRESTART": [],
        "LC-PRESTART": [*operation, ("REQ-MOV-010", 0)], "LC-TH-FIRE": [("REQ-MOV-007", 0)], "LC-FIRE": operation,
        "LC-FLOW-FIRE": [("REQ-MOV-008", 0)], "LC-TH-SHUTDOWN": [], "LC-SHUTDOWN": operation,
        "LC-FATIGUE": [("REQ-MOV-006", 0)]}
    # every criterion of an analysis requirement is in a load case, once per phase it applies to
    served = [c for lc in lcs.values() for c in lc.criteria]
    assert len(served) == 1 + 1 + 4 * 2 + 1 + 1 + 1 + 1
    # the links a record gives: requirements, effects, the load cases it needs, its phase
    refs = {(l.rel, l.to) for l in lcs["LC-FIRE"].refs()}
    assert {("verifies", "REQ-MOV-004"), ("uses", "EF-IN-P-FIRE"), ("uses", "LC-TH-FIRE"), ("derives_from", "PH-FIRE"),
            ("uses", "EF-IN-HOLD"), ("derives_from", "DC-RUN")} <= refs
    assert ("derives_from", "LC-PROOF") in {(l.rel, l.to) for l in lcs["LC-BURST"].refs()}


def test_example_loads_and_supports(example, example_ctx, example_plan):
    lcs = cases(example_plan)
    for lc in lcs.values():
        kinds = {x.kind for x in lc.loads}
        if lc.analysis_type in ("static_structural", "thermo_structural"):
            assert kinds <= set(MECHANICAL) and lc.loads
            phase = [x for x in example_ctx.loads(example, None, lc.phase)]
            assert [x.effect for x in lc.loads] == [x.effect for x in phase if x.kind in MECHANICAL]
            assert lc.supports == [x.interface for x in phase if x.kind == "support"]
        elif lc.analysis_type == "thermal_steady":
            assert kinds <= set(THERMAL) and len(lc.loads) == 3 and not lc.supports
        elif lc.analysis_type == "flow":
            assert kinds == {"mass_flow", "pressure", "fluid_temperature"} and not lc.supports
        else:
            assert not lc.loads and not lc.supports
    # the proof test holds the valve in a fixture without load: no support; the pump holds it in operation
    assert lcs["LC-PROOF"].supports == [] and lcs["LC-FIRE"].supports == ["INT-INLET"]
    assert "No interface holds the part" in lcs["LC-PROOF"].statement
    assert "The interface INT-INLET holds the part" in lcs["LC-FIRE"].statement
    proof = {x.effect: x for x in lcs["LC-PROOF"].loads}
    burst = {x.effect: x for x in lcs["LC-BURST"].loads}
    assert proof["EF-IN-P-PROOF"].si == pytest.approx(1.5 * 10.5e6) and burst["EF-IN-P-BURST"].si == pytest.approx(2.5 * 10.5e6)
    fire = lcs["LC-TH-FIRE"]
    assert {x.effect: x.h_si for x in fire.loads}["EF-ENV-T"] == 12.0 and {x.si for x in fire.loads} == {90.0, 293.15}
    assert [x.effect for x in lcs["LC-FLOW-FIRE"].loads if x.kind == "mass_flow"] == ["EF-IN-FLOW-FIRE"]
    # each load case has its own copies of the loads
    assert all(a is not b for a in lcs["LC-FIRE"].loads for b in lcs["LC-FLOW-FIRE"].loads)


def test_example_fatigue(example_plan):
    lcs = cases(example_plan)
    fat = lcs["LC-FATIGUE"]
    assert fat.phase is None
    assert [(c.duty_cycle, c.count, c.states) for c in fat.cycles] == [
        ("DC-RUN", 65.0, ["LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN"]),
        ("DC-PROOF", 2.0, ["LC-PROOF"]),
        ("DC-LEAK", 200.0, ["LC-LEAK"])]
    # the leak check has no strength requirement: its load case is there for the fatigue analysis only
    leak = lcs["LC-LEAK"]
    assert leak.criteria == [] and ("derives_from", "DC-LEAK") in {(l.rel, l.to) for l in leak.links}
    d = decisions(example_plan)["DEC-LC-LEAK-SELECTED"]
    assert "No requirement gives a criterion" in d.rationale and "LC-FATIGUE needs the stress" in d.rationale
    assert "DC-LEAK" in linked(d)
    assert "65 cycles of DC-RUN, 2 cycles of DC-PROOF and 200 cycles of DC-LEAK" in fat.statement


def test_example_hand_calculations(example, example_plan):
    lcs = cases(example_plan)
    got = {i: [h.spec for h in example_plan.hand_calcs if h.load_case == i] for i in lcs}
    assert {i: s for i, s in got.items() if s} == HAND_CALCS
    assert [h.id for h in example_plan.hand_calcs] == [i for lc in lcs.values() for i in lc.hand_calcs]
    assert len({h.id for h in example_plan.hand_calcs}) == 12
    assert example_plan.journal.findings == []


def test_example_journal(example, example_plan):
    dec, asm = decisions(example_plan), assumptions(example_plan)
    assert set(dec) == ({make_id("DEC", i, "selected") for i in EXPECTED} | {"DEC-LC-BURST-SCALED", "DEC-LC-LEAK-SCALED",
                        "DEC-PH-TRANSPORT-NO-LOAD-CASE", "DEC-REQ-MOV-009-NO-LOAD-CASE"})
    assert all(d.origin.stage == "plan" and d.origin.by.startswith("rule:loadcases.") for d in dec.values())

    # a selected load case: the requirements and criteria, the phase, the kind of analysis and why
    fire = dec["DEC-LC-FIRE-SELECTED"]
    assert "thermo-structural analysis of the phase PH-FIRE (Steady firing) at limit loads" in fire.choice
    assert "criteria yield and ultimate of REQ-MOV-004 in the phase PH-FIRE" in fire.rationale
    for text in ("EF-IN-T-FIRE gives a fluid temperature of 90 K", "reference temperature of 293.15 K",
                 "difference of 203.15 K is more than the policy limit of 5 K", "temperature field of LC-TH-FIRE"):
        assert text in fire.rationale
    assert {"REQ-MOV-004", "PH-FIRE", "EF-IN-T-FIRE", "POLICY", "CMP-MOV-BODY", "DC-RUN"} <= linked(fire)
    assert any("static structural analysis without thermal strain" in a.option for a in fire.alternatives)
    assert any("envelope" in a.option and "loads can cancel" in a.why_not for a in fire.alternatives)
    proof = dec["DEC-LC-PROOF-SELECTED"]
    assert "no thermal effect on the part in this phase" in proof.rationale
    assert "LC-BURST and LC-LEAK get their results from this load case" in proof.rationale
    assert "REQ-MOV-007" in dec["DEC-LC-TH-FIRE-SELECTED"].rationale and "LC-FIRE needs the temperature field" in \
        dec["DEC-LC-TH-FIRE-SELECTED"].rationale
    assert "REQ-MOV-008" in dec["DEC-LC-FLOW-FIRE-SELECTED"].rationale
    assert "REQ-MOV-006" in dec["DEC-LC-FATIGUE-SELECTED"].rationale

    # a scaled load case: the scale, why it is exact, and the alternative of an own solve
    burst = dec["DEC-LC-BURST-SCALED"]
    assert burst.choice == "Do not solve LC-BURST. Multiply the results of LC-PROOF by 1.667."
    assert "EF-IN-P-BURST gives 26.25 MPa and the effect EF-IN-P-PROOF gives 15.75 MPa" in burst.rationale
    assert "linear and static" in burst.rationale and burst.basis == "calculation"
    assert burst.alternatives[0].option == "Solve LC-BURST with its own mesh study."
    assert "LC-PROOF" in linked(burst) and not burst.needs_review
    assert "0.6667" in dec["DEC-LC-LEAK-SCALED"].choice

    # what gets no load case, with the findings of the validation stage
    transport = dec["DEC-PH-TRANSPORT-NO-LOAD-CASE"]
    assert transport.subject == "PH-TRANSPORT" and transport.needs_review
    assert "FND-PH-TRANSPORT-UNCOVERED" in linked(transport) and "FND-PH-TRANSPORT-UNCOVERED" in transport.rationale
    assert "1 force, 1 moment and 1 acceleration" in transport.rationale
    assert {"EF-ACT-F-TRANSPORT", "EF-ACT-M-TRANSPORT", "EF-ENV-A-TRANSPORT"} <= linked(transport)
    assert any("envelope" in a.option for a in transport.alternatives)
    stiff = dec["DEC-REQ-MOV-009-NO-LOAD-CASE"]
    assert stiff.subject == "REQ-MOV-009" and stiff.needs_review and linked(stiff) == {"FND-REQ-MOV-009-NOT-VERIFIABLE"}
    # a requirement that a review verifies, and that has no criterion, is not a matter of the plan
    assert not any(d.subject in ("REQ-MOV-001", "REQ-MOV-005") for d in dec.values())

    # every thermal load case: the steady state is an assumption, and not a conservative one
    assert set(asm) == {make_id("ASM", i, "steady-state") for i in EXPECTED if i.startswith("LC-TH-")}
    for a in asm.values():
        assert a.conservative is False and "transient thermal analysis" in a.verify_by
        assert "thermal stress is higher" in a.impact and "not conservative" in a.impact
        assert "steady-state field" in a.statement
    assert "duration of 120 s" in asm["ASM-LC-TH-CHILL-STEADY-STATE"].statement
    assert "duration of 400 s" in asm["ASM-LC-TH-FIRE-STEADY-STATE"].statement
    assert "approximately 39 mm" in asm["ASM-LC-TH-FIRE-STEADY-STATE"].impact          # sqrt(k / (rho cp) x 400 s)
    assert "REQ-MOV-007" in asm["ASM-LC-TH-FIRE-STEADY-STATE"].impact
    assert "no duration" in asm["ASM-LC-TH-PRESTART-STEADY-STATE"].statement
    assert "LC-SHUTDOWN" in asm["ASM-LC-TH-SHUTDOWN-STEADY-STATE"].impact


def test_example_links_point_to_records(example, example_plan):
    known = set(example.inputs()) | set(EXPECTED) | set(decisions(example_plan)) | set(assumptions(example_plan))
    known |= {h.id for h in example_plan.hand_calcs}
    known |= {"FND-PH-TRANSPORT-UNCOVERED", "FND-REQ-MOV-009-NOT-VERIFIABLE"}       # made by the validation stage
    for rec in [*example_plan.load_cases, *example_plan.journal.decisions, *example_plan.journal.assumptions]:
        assert {l.to for l in rec.refs()} <= known, rec.id


def test_plan_is_the_same_on_every_run_and_leaves_its_inputs(example, example_ctx, example_plan):
    before = example_ctx.model_dump()
    again = make(example, ctx=example_ctx)
    dump = lambda p: [[r.model_dump(mode="json") for r in group] for group in  # noqa: E731
                      (p.load_cases, p.hand_calcs, p.journal.decisions, p.journal.assumptions, p.journal.findings)]
    assert dump(again) == dump(example_plan)
    assert example_ctx.model_dump() == before


def test_hand_calculations_of_the_example_with_the_real_engine(example, example_ctx, example_plan):
    real = sound(plan(example, example_ctx, None), example)
    lcs = cases(real)
    assert list(lcs) == list(EXPECTED)
    assert [f.statement for f in real.journal.findings] == []
    got = {i: [h.spec for h in real.hand_calcs if h.load_case == i] for i in lcs}
    assert {i: s for i, s in got.items() if s} == HAND_CALCS
    assert all(h.kind == "hand_calc" and h.outputs and h.id in lcs[h.load_case].hand_calcs for h in real.hand_calcs)
    assert len({h.id for h in real.hand_calcs}) == 12
    # the same plan as with the stand-in: the result keys come from the comparisons of the specs
    assert {i: set(lc.result_keys) for i, lc in lcs.items()} == RESULT_KEYS
    assert [d.model_dump() for d in real.journal.decisions] == [d.model_dump() for d in example_plan.journal.decisions]
    # the hand calculation of a scaled load case reads the loads of its own phase
    hoop = {h.load_case: next(o.si for o in h.outputs if o.name == "hoop_mean")
            for h in real.hand_calcs if h.spec == "HC-SHELL-HOOP"}
    assert hoop["LC-BURST"] / hoop["LC-PROOF"] == pytest.approx(2.5 / 1.5)
    assert hoop["LC-LEAK"] / hoop["LC-PROOF"] == pytest.approx(1 / 1.5)


def test_plan_stage_writes_the_plan(workproject):
    workproject.write_stage("geometry", geometry={"component": workproject.subject.id, "features": {}, "problems": []})
    out = pipeline.run(workproject, first="plan", last="plan", opts=pipeline.Options(log=lambda s: None))
    assert out["plan"]["load_cases"][0] == "LC-PROOF (static_structural, selected)"
    assert "LC-BURST (static_structural, scaled)" in out["plan"]["load_cases"] and out["plan"]["hand_calcs"] == 12
    read = Project.load(workproject.root)
    assert [lc.id for lc in read.collection("load_cases")] == list(EXPECTED)
    assert [lc.id for lc in pipeline.analysis_order(read.collection("load_cases"))] == list(EXPECTED)
    ids = {r.id for r in read.collection("decisions")} | {r.id for r in read.collection("assumptions")}
    assert {"DEC-LC-BURST-SCALED", "DEC-PH-TRANSPORT-NO-LOAD-CASE", "ASM-LC-TH-FIRE-STEADY-STATE"} <= ids
    assert len(read.collection("hand_calcs")) == 12


# --------------------------------------------------------------------------- the kind of structural analysis


def _without_thermal_effects(data, phase):
    for c in data["connections"]:
        kept = []
        for e in c.get("effects", []):
            if e["kind"] in THERMAL:
                phases = [p for p in ([e["phase"]] if isinstance(e["phase"], str) else e["phase"]) if p != phase]
                if not phases:
                    continue
                e["phase"] = phases
            kept.append(e)
        if "effects" in c:
            c["effects"] = kept


def test_phase_without_thermal_effects_is_static(workproject):
    project = edit(workproject, SYSTEM, lambda d: _without_thermal_effects(d, "PH-CHILL"))
    p = sound(make(project), project)
    lcs = cases(p)
    chill = lcs["LC-CHILL"]
    assert chill.analysis_type == "static_structural" and chill.uses == [] and chill.status == "selected"
    assert "LC-TH-CHILL" not in lcs and "ASM-LC-TH-CHILL-STEADY-STATE" not in assumptions(p)
    assert len(chill.loads) == 5 and chill.supports == ["INT-INLET"]
    assert "temperature field" not in chill.statement
    d = decisions(p)["DEC-LC-CHILL-SELECTED"]
    assert "static structural analysis" in d.choice and "no thermal effect on the part in this phase" in d.rationale
    # the other operation phases keep their thermal load cases, and the fatigue analysis uses the static one
    assert lcs["LC-FIRE"].analysis_type == "thermo_structural" and lcs["LC-FATIGUE"].cycles[0].states[0] == "LC-CHILL"
    # the chill-down loads are not a multiple of the proof loads (a seat force, a support): it is solved
    assert chill.scaled_from is None


def test_small_temperature_difference_is_static_with_an_assumption(workproject):
    def change(d):
        effect(d, "EF-IN-T-CHILL")["value"] = "290 K"           # 3.15 K under the reference temperature
        effect(d, "EF-OUT-T-CHILL")["value"] = "291 K"
    project = edit(workproject, SYSTEM, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert lcs["LC-CHILL"].analysis_type == "static_structural" and "LC-TH-CHILL" not in lcs
    assert lcs["LC-CHILL"].status == "selected"                 # a phase with a thermal effect is not scaled
    d = decisions(p)["DEC-LC-CHILL-SELECTED"]
    assert "largest temperature difference from the reference temperature of 293.15 K is 3.15 K (effect EF-IN-T-CHILL)" \
        in d.rationale
    assert "not more than the policy limit of 5 K" in d.rationale
    a = assumptions(p)["ASM-LC-CHILL-THERMAL-STRAIN"]
    assert a.conservative is False and a.subject == "LC-CHILL"
    assert "E alpha dT / (1 - nu) = 11.13 MPa" in a.impact      # 193 GPa x 13e-6 / K x 3.15 K / (1 - 0.29)
    assert {"POLICY", "EF-IN-T-CHILL", "MAT-316L"} <= linked(a)

    # exactly at the limit is not "more than" the limit
    project = edit(project, SYSTEM, lambda d: effect(d, "EF-IN-T-CHILL").update(value="288.15 K"))
    assert cases(make(project))["LC-CHILL"].analysis_type == "static_structural"
    project = edit(project, SYSTEM, lambda d: effect(d, "EF-IN-T-CHILL").update(value="288.1 K"))
    lcs = cases(make(project))
    assert lcs["LC-CHILL"].analysis_type == "thermo_structural" and lcs["LC-CHILL"].uses == ["LC-TH-CHILL"]


def test_the_policy_limit_decides_the_kind_of_analysis(workproject):
    project = edit(workproject, POLICY, lambda d: d.update(thermal_delta="250 K"))
    p = sound(make(project), project)
    lcs = cases(p)
    for i in ("LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN"):
        assert lcs[i].analysis_type == "static_structural" and lcs[i].uses == [] and lcs[i].status == "selected"
        assert make_id("ASM", i, "thermal-strain") in assumptions(p)
    # the temperature criterion of REQ-MOV-007 still needs the thermal load case of the firing phase
    assert [i for i in lcs if i.startswith("LC-TH-")] == ["LC-TH-FIRE"]
    a = assumptions(p)["ASM-LC-TH-FIRE-STEADY-STATE"]
    assert "REQ-MOV-007" in a.impact and "thermal stress is higher" not in a.impact
    assert "203.15 K" in assumptions(p)["ASM-LC-FIRE-THERMAL-STRAIN"].statement


def test_a_heat_flux_needs_the_thermal_load_case(workproject):
    def change(d):
        _without_thermal_effects(d, "PH-CHILL")
        conn = next(c for c in d["connections"] if c["id"] == "IF-ENV-MOV")
        conn["effects"].append({"id": "EF-ENV-Q-CHILL", "phase": "PH-CHILL", "kind": "heat_flux", "role": "external",
                                "value": "800 W/m^2"})
    project = edit(workproject, SYSTEM, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert lcs["LC-CHILL"].analysis_type == "thermo_structural" and lcs["LC-CHILL"].uses == ["LC-TH-CHILL"]
    assert [x.kind for x in lcs["LC-TH-CHILL"].loads] == ["heat_flux"]
    d = decisions(p)["DEC-LC-CHILL-SELECTED"]
    assert "EF-ENV-Q-CHILL gives a heat flux of 800 W/m²" in d.rationale and "EF-ENV-Q-CHILL" in linked(d)


# --------------------------------------------------------------------------- proportional load cases


def test_non_proportional_loads_are_not_scaled(workproject):
    # every load of the leak check is smaller than in the proof test, but the pattern is different:
    # the stem force is a third of the proof value, the pressures are two thirds
    def change(d):
        effect(d, "EF-ACT-F-LEAK")["value"] = {"expr": [0, 0, "0.5 * meop * a_stem"], "unit": "N"}
    project = edit(workproject, SYSTEM, change)
    p = sound(make(project), project)
    lcs = cases(p)
    leak = lcs["LC-LEAK"]
    assert (leak.status, leak.scaled_from, leak.scale) == ("selected", None, 1.0)
    assert "DEC-LC-LEAK-SCALED" not in decisions(p)
    alt = {a.option: a.why_not for a in decisions(p)["DEC-LC-LEAK-SELECTED"].alternatives}
    assert "No load case before this one has proportional loads" in alt[
        "Use the results of a different load case as an envelope for this phase."]
    # the burst test keeps its scale, and the base no longer carries the keys of the leak check alone
    assert lcs["LC-BURST"].scaled_from == "LC-PROOF" and lcs["LC-BURST"].scale == pytest.approx(2.5 / 1.5)
    assert "LC-BURST gets its results from this load case" in decisions(p)["DEC-LC-PROOF-SELECTED"].rationale


def test_a_different_support_is_not_proportional(workproject):
    project = edit(workproject, SYSTEM, lambda d: effect(d, "EF-IN-HOLD")["phase"].append("PH-LEAK"))
    p = sound(make(project), project)
    leak = cases(p)["LC-LEAK"]
    assert leak.status == "selected" and leak.scaled_from is None and leak.supports == ["INT-INLET"]
    assert cases(p)["LC-BURST"].status == "scaled"


def test_the_first_phase_of_the_system_is_the_base(workproject):
    def change(d):
        d["phases"].sort(key=lambda ph: ph["id"] != "PH-LEAK")      # the leak check comes first
    project = edit(workproject, SYSTEM, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert list(lcs)[:3] == ["LC-LEAK", "LC-PROOF", "LC-BURST"]
    assert lcs["LC-LEAK"].status == "selected"
    assert (lcs["LC-PROOF"].scaled_from, lcs["LC-BURST"].scaled_from) == ("LC-LEAK", "LC-LEAK")
    assert lcs["LC-PROOF"].scale == pytest.approx(1.5) and lcs["LC-BURST"].scale == pytest.approx(2.5)
    # the solved load case gives the results that its scaled load cases need
    assert {"von_mises.membrane_bending@scls", "von_mises.membrane@scls", SHELL, NOZZLE, FLOOR} == set(lcs["LC-LEAK"].result_keys)


# --------------------------------------------------------------------------- what gets no load case


def test_requirement_without_criteria_gets_no_load_case(workproject):
    project = edit(workproject, REQUIREMENTS, lambda d: requirement(d, "REQ-MOV-010").pop("criteria"))
    p = sound(make(project), project)
    lcs = cases(p)
    assert list(lcs) == list(EXPECTED)
    assert "radial_displacement.range@FT-SEAT-OUT-WALL" not in lcs["LC-PRESTART"].result_keys
    assert all(c.requirement != "REQ-MOV-010" for lc in lcs.values() for c in lc.criteria)
    d = decisions(p)["DEC-REQ-MOV-010-NO-LOAD-CASE"]
    assert d.needs_review and linked(d) == {"FND-REQ-MOV-010-NOT-VERIFIABLE"}
    assert "gives no criterion" in d.rationale and "FND-REQ-MOV-010-NOT-VERIFIABLE" in d.rationale


def test_requirement_with_another_verification_method_gets_no_load_case(workproject):
    project = edit(workproject, REQUIREMENTS, lambda d: requirement(d, "REQ-MOV-003").update(verification="test"))
    p = sound(make(project), project)
    lcs = cases(p)
    assert "LC-BURST" not in lcs and "LC-PROOF" in lcs
    assert "von_mises.membrane@scls" not in lcs["LC-PROOF"].result_keys
    d = decisions(p)["DEC-REQ-MOV-003-NO-LOAD-CASE"]
    assert "its verification method is test" in d.rationale and d.basis == "input" and not d.needs_review
    # the burst phase has loads and no structural load case: its decision names the requirement that applies
    ph = decisions(p)["DEC-PH-BURST-NO-LOAD-CASE"]
    assert "REQ-MOV-003 applies to the phase" in ph.rationale and ph.needs_review
    assert "FND-PH-BURST-UNCOVERED" not in linked(ph)


def test_uncovered_phase_gets_a_decision_and_a_requirement_gives_it_a_load_case(workproject):
    p = make(workproject)
    assert "DEC-PH-TRANSPORT-NO-LOAD-CASE" in decisions(p) and "LC-TRANSPORT" not in cases(p)

    def change(d):
        d["requirements"].append({
            "id": "REQ-MOV-011", "title": "Strength in transport", "kind": "strength",
            "text": "The valve body shall withstand the transport loads with a factor of safety of 1.25 against yield.",
            "applies": {"category": "handling"},
            "criteria": [{"id": "yield", "quantity": "von_mises", "measure": "peak", "location": "part",
                          "comparator": "<=", "allowable": "yield_strength", "factor": 1.25}]})
    project = edit(workproject, REQUIREMENTS, change)
    p = sound(make(project), project)
    lc = cases(p)["LC-TRANSPORT"]
    assert (lc.analysis_type, lc.status, lc.supports, lc.uses) == ("static_structural", "selected", ["INT-INLET"], [])
    assert [x.kind for x in lc.loads] == ["force", "moment", "acceleration"]
    assert [(c.requirement, c.criterion) for c in lc.criteria] == [("REQ-MOV-011", 0)]
    assert "DEC-PH-TRANSPORT-NO-LOAD-CASE" not in decisions(p)
    assert list(cases(p))[3] == "LC-TRANSPORT"                      # in the order of the phases of the system


def test_fatigue_alone_decides_the_leak_load_case(workproject):
    def change(d):
        d["requirements"] = [r for r in d["requirements"] if r["id"] != "REQ-MOV-006"]
    project = edit(workproject, REQUIREMENTS, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert "LC-FATIGUE" not in lcs and "LC-LEAK" not in lcs and len(lcs) == len(EXPECTED) - 2
    assert not any(l.to.startswith("DC-") for lc in lcs.values() for l in lc.links)
    # the leak check now has loads and no requirement; its loads are two thirds of the proof loads
    d = decisions(p)["DEC-PH-LEAK-NO-LOAD-CASE"]
    assert "0.6667 times the loads of PH-PROOF" in d.rationale and "the results of LC-PROOF are an envelope" in d.rationale
    assert not d.needs_review and "LC-PROOF" in linked(d) and "FND-PH-LEAK-UNCOVERED" not in linked(d)


def test_fatigue_of_one_duty_cycle(workproject):
    project = edit(workproject, REQUIREMENTS,
                   lambda d: requirement(d, "REQ-MOV-006").update(applies={"duty_cycles": ["DC-LEAK"]}))
    p = sound(make(project), project)
    fat = cases(p)["LC-FATIGUE"]
    assert [(c.duty_cycle, c.count, c.states) for c in fat.cycles] == [("DC-LEAK", 200.0, ["LC-LEAK"])]
    assert fat.uses == ["LC-LEAK"] and fat.title == "Fatigue analysis of the duty cycle DC-LEAK"
    assert not any(l.to == "DC-PROOF" for l in cases(p)["LC-PROOF"].links)


def test_two_sets_of_duty_cycles_give_two_fatigue_load_cases(workproject):
    def change(d):
        d["requirements"].append({
            "id": "REQ-MOV-012", "title": "Life in engine runs", "kind": "life", "text": "The valve body shall "
            "withstand 4 times the engine runs without a fatigue crack.", "applies": {"duty_cycles": ["DC-RUN"]},
            "criteria": [{"id": "damage", "quantity": "damage", "measure": "total", "location": "part",
                          "comparator": "<=", "limit": 1.0, "factor": 4}]})
    project = edit(workproject, REQUIREMENTS, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert [c.duty_cycle for c in lcs["LC-FATIGUE"].cycles] == ["DC-RUN", "DC-PROOF", "DC-LEAK"]
    assert [c.duty_cycle for c in lcs["LC-FATIGUE-2"].cycles] == ["DC-RUN"]
    assert [(c.requirement, c.criterion) for c in lcs["LC-FATIGUE-2"].criteria] == [("REQ-MOV-012", 0)]
    assert lcs["LC-FATIGUE-2"].uses == ["LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN"]


def test_criterion_the_plan_cannot_serve_is_a_finding(workproject):
    def change(d):
        requirement(d, "REQ-MOV-010")["criteria"][0]["quantity"] = "ovality"        # no such quantity
        requirement(d, "REQ-MOV-002")["applies"] = {}                                 # criteria, but no phase
    project = edit(workproject, REQUIREMENTS, change)
    p = sound(make(project), project)
    lcs, fnd, dec = cases(p), findings(p), decisions(p)
    f = fnd["FND-REQ-MOV-010-CRITERION-1-QUANTITY"]
    assert (f.severity, f.category, f.subject) == ("major", "model", "REQ-MOV-010") and "ovality" in f.statement
    assert linked(dec["DEC-REQ-MOV-010-NO-LOAD-CASE"]) == {f.id} and dec["DEC-REQ-MOV-010-NO-LOAD-CASE"].needs_review
    g = fnd["FND-REQ-MOV-002-NO-PHASE"]
    assert (g.severity, g.category) == ("major", "coverage") and "applies to no phase" in g.statement
    assert linked(dec["DEC-REQ-MOV-002-NO-LOAD-CASE"]) == {g.id}
    # the proof load case stays: the fatigue analysis needs it as a state
    assert lcs["LC-PROOF"].criteria == [] and lcs["LC-PROOF"].status == "selected"
    assert all(c.requirement not in ("REQ-MOV-002", "REQ-MOV-010") for lc in lcs.values() for c in lc.criteria)


def test_temperature_criterion_in_a_phase_without_thermal_effects_is_not_supported(workproject):
    project = edit(workproject, REQUIREMENTS,
                   lambda d: requirement(d, "REQ-MOV-007").update(applies={"phases": ["PH-PROOF"]}))
    p = sound(make(project), project)
    lcs = cases(p)
    th = lcs["LC-TH-PROOF"]
    assert th.status == "not_supported" and th.loads == [] and "no thermal effect on the part in the phase PH-PROOF" in th.note
    assert [(c.requirement, c.criterion) for c in th.criteria] == [("REQ-MOV-007", 0)]
    assert lcs["LC-PROOF"].analysis_type == "static_structural" and lcs["LC-PROOF"].uses == []
    f = findings(p)["FND-PH-PROOF-NO-THERMAL-LOAD"]
    assert (f.severity, f.category) == ("major", "coverage") and {"REQ-MOV-007", "LC-TH-PROOF"} <= linked(f)
    d = decisions(p)["DEC-LC-TH-PROOF-NOT-SUPPORTED"]
    assert d.needs_review and f.id in linked(d) and "ASM-LC-TH-PROOF-STEADY-STATE" not in assumptions(p)
    assert lcs["LC-TH-FIRE"].criteria == [] and lcs["LC-FIRE"].uses == ["LC-TH-FIRE"]
    # the pipeline turns such a load case into an analysis record that says why there is no result
    out = pipeline.analyze_one(project, context.build(project), {}, th, pipeline.Options(log=lambda s: None))
    assert out.record.status == "not_supported" and out.record.message == th.note


def test_flow_criterion_in_a_phase_without_mass_flow_is_not_supported(workproject):
    project = edit(workproject, REQUIREMENTS,
                   lambda d: requirement(d, "REQ-MOV-008").update(applies={"phases": ["PH-PRESTART"]}))
    p = sound(make(project), project)
    flow = cases(p)["LC-FLOW-PRESTART"]
    assert flow.status == "not_supported" and "no mass flow" in flow.note and "LC-FLOW-FIRE" not in cases(p)
    assert {x.kind for x in flow.loads} == {"pressure", "fluid_temperature"}
    assert "FND-PH-PRESTART-NO-MASS-FLOW" in findings(p) and "DEC-LC-FLOW-PRESTART-NOT-SUPPORTED" in decisions(p)


def test_fatigue_criterion_without_a_duty_cycle_is_not_supported(workproject):
    project = edit(workproject, REQUIREMENTS, lambda d: requirement(d, "REQ-MOV-006").update(applies={}))
    p = sound(make(project), project)
    lcs = cases(p)
    fat = lcs["LC-FATIGUE"]
    assert fat.status == "not_supported" and fat.cycles == [] and fat.uses == [] and "names no duty cycle" in fat.note
    assert "LC-LEAK" not in lcs
    f = findings(p)["FND-REQ-MOV-006-NO-DUTY-CYCLE"]
    assert f.severity == "major" and f.id in linked(decisions(p)["DEC-LC-FATIGUE-NOT-SUPPORTED"])


def test_phase_with_a_requirement_and_no_load_gets_no_load_case(workproject):
    def change(d):
        d["phases"].append({"id": "PH-STORAGE", "name": "Storage", "category": "storage"})
        d["phases"].append({"id": "PH-COAST", "name": "Coast"})                     # operation: REQ-MOV-004 applies
    project = edit(workproject, SYSTEM, change)
    p = sound(make(project), project)
    assert not any(lc.phase in ("PH-STORAGE", "PH-COAST") for lc in p.load_cases)
    storage, coast = decisions(p)["DEC-PH-STORAGE-NO-LOAD-CASE"], decisions(p)["DEC-PH-COAST-NO-LOAD-CASE"]
    assert "no load on the part" in storage.rationale and "No requirement applies" in storage.rationale
    assert not storage.needs_review
    assert "REQ-MOV-004 needs the stress" in coast.rationale and "zero stress" in coast.rationale and coast.needs_review


def test_phase_with_loads_and_only_a_thermal_requirement(workproject):
    def change(d):
        requirement(d, "REQ-MOV-004")["applies"] = {"phases": ["PH-CHILL", "PH-PRESTART", "PH-SHUTDOWN"]}
        requirement(d, "REQ-MOV-006")["applies"] = {"duty_cycles": ["DC-PROOF", "DC-LEAK"]}
    project = edit(workproject, REQUIREMENTS, change)
    p = sound(make(project), project)
    lcs = cases(p)
    assert "LC-FIRE" not in lcs and lcs["LC-TH-FIRE"].status == "selected" and "LC-FLOW-FIRE" in lcs
    d = decisions(p)["DEC-PH-FIRE-NO-LOAD-CASE"]
    assert d.title == "No structural load case for the phase PH-FIRE" and d.needs_review
    assert "REQ-MOV-007 and REQ-MOV-008 apply to the phase" in d.rationale
    assert "LC-TH-FIRE and LC-FLOW-FIRE" in d.rationale and "203.15 K" in d.rationale
    # the hand calculation of the neck has no load case now: a finding says so
    f = findings(p)["FND-HC-NECK-BENDING-NOT-USED"]
    assert (f.severity, f.category, f.subject) == ("minor", "model", "HC-NECK-BENDING")


def test_phase_ids_that_give_the_same_load_case_id_stop_the_plan(workproject):
    def change(d):
        d["phases"].append({"id": "PH-TH-FIRE", "name": "Throttled firing"})
        conn = next(c for c in d["connections"] if c["id"] == "IF-PUMP-MOV")
        conn["effects"].append({"id": "EF-IN-P-TH", "phase": "PH-TH-FIRE", "kind": "pressure", "value": "5 MPa"})
    project = edit(workproject, SYSTEM, change)
    with pytest.raises(pipeline.StageError) as e:
        make(project)
    assert "LC-TH-FIRE names two load cases" in str(e.value) and "PH-TH-FIRE" in str(e.value)


# --------------------------------------------------------------------------- hand calculations


def test_hand_calculation_that_fails_is_a_finding_and_the_plan_goes_on(workproject):
    p = sound(make(workproject, StandIn(fail={"HC-FLOOR-PLATE"})), workproject)
    lcs = cases(p)
    assert list(lcs) == list(EXPECTED)
    assert [h.spec for h in p.hand_calcs if h.load_case == "LC-PROOF"] == ["HC-SHELL-HOOP", "HC-NOZZLE-HOOP"]
    assert len(p.hand_calcs) == 9 and FLOOR not in lcs["LC-PROOF"].result_keys
    fnd = findings(p)
    assert set(fnd) == {make_id("FND", "HC-FLOOR-PLATE", f"{i}-failed") for i in ("LC-PROOF", "LC-BURST", "LC-LEAK")}
    f = fnd["FND-HC-FLOOR-PLATE-LC-BURST-FAILED"]
    assert (f.severity, f.category, f.subject, f.rule) == ("major", "model", "HC-FLOOR-PLATE", "plan.hand_calc")
    assert "ValueError: load('INT-NOWHERE', 'pressure'): this load case has no such load" in f.rationale
    assert f.values["load_case"] == "LC-BURST" and "LC-BURST" in linked(f)

    # a `when` condition that cannot be evaluated: one finding, and the other specs go on
    p = sound(make(workproject, StandIn(fail_when={"HC-ACT-TEMP"})), workproject)
    assert set(findings(p)) == {"FND-HC-ACT-TEMP-WHEN-FAILED"} and len(p.hand_calcs) == 11
    assert cases(p)["LC-TH-FIRE"].hand_calcs == [] and cases(p)["LC-TH-FIRE"].result_keys == ["temperature.min@INT-ACT"]


def test_hand_calculation_engine_with_the_same_id_twice_is_a_finding(workproject):
    class SameId(StandIn):
        def evaluate(self, spec, project, ctx, geo, lc):
            return super().evaluate(spec, project, ctx, geo, lc).model_copy(update={"id": make_id("HCR", spec.id)})
    p = make(workproject, SameId())
    assert len(p.hand_calcs) == 6 and len({h.id for h in p.hand_calcs}) == 6
    assert "FND-HC-SHELL-HOOP-LC-BURST-FAILED" in findings(p) and cases(p)["LC-BURST"].hand_calcs == []
