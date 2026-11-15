"""Tests of dossier.requirements.validate: the six defects of the example, their fixes, and the rules on small copies."""

import pytest

import dataclasses

from dossier import context
from dossier.model import Value
from dossier.requirements import validate as V

SUBJ = "mov_body"


def run(project):
    V.STRICT = True
    ctx = context.build(project)
    return V.validate(project, ctx, None)


def ids(j, rule=None, subject=None, severity=None):
    return {f.id for f in j.findings if (rule is None or f.rule == rule) and (subject is None or f.subject == subject)
            and (severity is None or f.severity == severity)}


def edit(project, req_id, **fields):
    """Replace fields of a requirement in the loaded project."""
    rs = project.requirements[project.file.subject]
    i = next(k for k, r in enumerate(rs) if r.id == req_id)
    rs[i] = rs[i].model_copy(update=fields)


def set_param(project, req_id, name, **fields):
    r = project.requirement(req_id)
    params = dict(r.parameters)
    if isinstance(fields.get("value"), str):                  # "10.8 MPa": read it as the project files do
        v = Value.model_validate(fields.pop("value"))
        fields.update(value=v.value, unit=v.unit)
    params[name] = params[name].model_copy(update=fields)
    edit(project, req_id, parameters=params)


@pytest.fixture(scope="module")
def base(example, example_ctx):
    V.STRICT = True
    return V.validate(example, example_ctx, None)


# ---- the example: the six defects, by rule and subject

def test_defect_1_meop(base):
    f = next(f for f in base.findings if f.rule == "req.meop_envelope")
    assert f.subject == "REQ-MOV-001" and f.severity == "major" and f.category == "consistency"
    assert f.values["requirement"] == pytest.approx(10.5e6)
    assert f.values["context"] == pytest.approx(10.7239e6, rel=1e-4)
    assert f.values["inputs"]["p_pump_max"] == pytest.approx(10.4e6)
    for text in ("PH-SHUTDOWN", "EF-IN-P-SHUTDOWN", "p_max_inlet", "p_pump_max", "dp_surge", "10.4 MPa"):
        assert text in f.rationale
    assert "10.73 MPa" in f.suggestion


def test_defect_2_yield_factor(base):
    f = next(f for f in base.findings if f.rule == "req.policy_factor")
    assert f.subject == "REQ-MOV-004" and f.severity == "major" and f.category == "policy"
    assert f.values["requirement"] == pytest.approx(1.1) and f.values["policy"] == pytest.approx(1.25)
    assert ids(base, "req.policy_factor") == {f.id}          # the ultimate factor 1.4 is fine


def test_defect_3_min_temperature(base):
    f = next(f for f in base.findings if f.rule == "req.temperature_range")
    assert f.subject == "REQ-MOV-005" and f.severity == "major"
    assert f.values["requirement"] == 233.0 and f.values["context"] == 90.0
    assert "PH-CHILL" in f.rationale and "t_lox" in f.rationale
    assert len(ids(base, "req.temperature_range")) == 1       # ambient air (role external) is not the fluid


def test_defect_4_param_check(base):
    f = next(f for f in base.findings if f.rule == "req.param_check")
    assert f.subject == "REQ-MOV-006" and f.severity == "major"
    assert f.values["requirement"] == 200.0 and f.values["context"] == 267.0
    assert ids(base, "req.design_cycles") == set()            # a parameter with a check is not judged twice


def test_defect_5_not_verifiable(base):
    assert "FND-REQ-MOV-009-NOT-VERIFIABLE" in ids(base, "req.not_verifiable", "REQ-MOV-009", "major")
    f = next(f for f in base.findings if f.rule == "req.vague_word")
    assert f.subject == "REQ-MOV-009" and '"sufficiently"' in f.statement


def test_defect_6_uncovered_phase(base):
    assert ids(base, "req.phase_uncovered") == {"FND-PH-TRANSPORT-UNCOVERED"}
    f = base.findings[[x.id for x in base.findings].index("FND-PH-TRANSPORT-UNCOVERED")]
    assert f.subject == "PH-TRANSPORT" and f.severity == "major" and f.category == "coverage"


def test_example_has_nothing_else_serious(base):
    majors = {f.id for f in base.findings if f.severity in ("major", "error")}
    assert majors == {"FND-REQ-MOV-009-NOT-VERIFIABLE", "FND-REQ-MOV-001-MEOP-LOW", "FND-REQ-MOV-005-MIN-TEMPERATURE",
                      "FND-REQ-MOV-006-CHECK-DESIGN-CYCLES", "FND-REQ-MOV-004-YIELD-FACTOR",
                      "FND-PH-TRANSPORT-UNCOVERED", "FND-REQ-MOV-009-VAGUE"}
    assert len({f.id for f in base.findings}) == len(base.findings)


def test_example_assumptions(base):
    a = {x.id for x in base.assumptions}
    assert "ASM-PH-LEAK-COVERED-BY-SCALE" in a and "ASM-MAT-316L-PROPERTIES-TEMPERATURE" in a


def test_records_are_complete_and_ste_shaped(base):
    for f in base.findings:
        assert f.rule.startswith("req.") and f.statement and f.rationale and f.suggestion and f.values
        assert f.subject, f.id
        assert isinstance(f.rationale, str) and f.statement.endswith(".")
        assert not any(w in f.statement.split() for w in ("shouldn't", "can't"))


def test_summary_lists_every_rule(base):
    rows = V.summary(base.findings, base.assumptions)
    assert [r["rule"] for r in rows] == [r.id for r in V.RULES]
    ran_silent = [r["rule"] for r in rows if not r["findings"] and not r["assumptions"]]
    assert "req.no_phase" in ran_silent
    assert all(r.id in V.__doc__ for r in V.RULES)


def test_a_failing_rule_does_not_hide_the_others(example, example_ctx, monkeypatch):
    def boom(env):
        raise RuntimeError("x")
    monkeypatch.setattr(V, "RULES", (dataclasses.replace(V.RULES[0], fn=boom), *V.RULES[1:]))
    monkeypatch.setattr(V, "STRICT", False)
    j = V.validate(example, example_ctx, None)
    assert any(f.rule == "req.not_verifiable" and f.severity == "error" for f in j.findings)
    assert "FND-REQ-MOV-001-MEOP-LOW" in ids(j)


# ---- the negative test of each defect: fix it and the finding is gone

def test_fix_1_meop(workproject):
    set_param(workproject, "REQ-MOV-001", "meop", value="10.8 MPa")
    j = run(workproject)
    assert not ids(j, "req.meop_envelope")


def test_meop_far_above_is_info(workproject):
    set_param(workproject, "REQ-MOV-001", "meop", value="20 MPa")
    j = run(workproject)
    f = [f for f in j.findings if f.rule == "req.meop_envelope"]
    assert len(f) == 1 and f[0].severity == "info"


def test_fix_2_yield_factor(workproject):
    set_param(workproject, "REQ-MOV-004", "yield_factor", value=1.25)
    assert not ids(run(workproject), "req.policy_factor")
    set_param(workproject, "REQ-MOV-004", "yield_factor", value=1.5)
    j = run(workproject)
    assert {f.severity for f in j.findings if f.rule == "req.policy_factor"} == {"info"}


def test_fix_3_temperature(workproject):
    set_param(workproject, "REQ-MOV-005", "t_fluid_min", value="90 K")
    assert not ids(run(workproject), "req.temperature_range")
    set_param(workproject, "REQ-MOV-005", "t_fluid_max", value="200 K")
    assert ids(run(workproject), "req.temperature_range") == {"FND-REQ-MOV-005-MAX-TEMPERATURE"}


def test_fix_4_check(workproject):
    set_param(workproject, "REQ-MOV-006", "design_cycles", value=267)
    assert not ids(run(workproject), "req.param_check")


def test_fix_5_criterion(workproject):
    r = workproject.requirement("REQ-MOV-009")
    crit = {"id": "deflection", "quantity": "displacement", "measure": "max", "location": "part", "comparator": "<=",
            "limit": "50 um"}
    from dossier.model import Criterion
    edit(workproject, "REQ-MOV-009", criteria=[Criterion(**crit)], applies=r.applies.model_copy(update={"phases": ["PH-PROOF"]}),
         text="The valve body shall have a deflection of 50 micrometres or less under the proof pressure.")
    j = run(workproject)
    assert not ids(j, "req.not_verifiable") and not ids(j, "req.vague_word")


def test_fix_6_phase(workproject):
    r = workproject.requirement("REQ-MOV-004")
    edit(workproject, "REQ-MOV-004", applies=r.applies.model_copy(update={"phases": ["PH-TRANSPORT"]}))
    assert not ids(run(workproject), "req.phase_uncovered")


# ---- the other rules, on small modified copies

def test_no_shall_and_not_singular(workproject):
    edit(workproject, "REQ-MOV-003", text="The valve body withstands a burst pressure of 2.5 times MEOP.")
    edit(workproject, "REQ-MOV-002", text="The valve body shall withstand 1.5 times MEOP and shall not leak.")
    j = run(workproject)
    assert ids(j, "req.no_shall") == {f.id for f in j.findings if f.rule == "req.no_shall"} and \
        any(f.subject == "REQ-MOV-003" for f in j.findings if f.rule == "req.no_shall")
    assert any(f.subject == "REQ-MOV-002" for f in j.findings if f.rule == "req.not_singular")


def test_vague_words(workproject):
    edit(workproject, "REQ-MOV-003", text="The valve body shall be robust and adequate, and/or TBD, about 2.5 times MEOP.")
    words = {w for f in run(workproject).findings if f.rule == "req.vague_word" and f.subject == "REQ-MOV-003"
             for w in f.values["words"]}
    assert {"robust", "adequate", "and/or", "TBD"} <= words


def test_criterion_errors(workproject):
    from dossier.model import Criterion
    bad = [Criterion(id="a", quantity="von_mises", measure="peak", location="part", comparator="<=", allowable="density"),
           Criterion(id="b", quantity="von_mises", measure="peak", location="FT-NOPE", comparator="<=", allowable="yield_strength"),
           Criterion(id="c", quantity="von_mises", measure="peak", location="part", comparator="<=", limit="5 K"),
           Criterion(id="d", quantity="von_mises", measure="peak", location="part", comparator="<=", allowable="yield_strength",
                     factor="no_such_factor"),
           Criterion(id="e", quantity="nonsense", measure="peak", location="part", comparator="<=", allowable="yield_strength")]
    edit(workproject, "REQ-MOV-004", criteria=bad)
    j = run(workproject)
    rules = {f.rule for f in j.findings if f.subject == "REQ-MOV-004" and f.severity == "error"}
    assert {"req.criterion_allowable", "req.criterion_location", "req.criterion_limit", "req.criterion_factor",
            "req.criterion_vocabulary"} <= rules


def test_number_not_in_text(workproject):
    set_param(workproject, "REQ-MOV-008", "dp_max", value="75 kPa")
    j = run(workproject)
    assert any(f.rule == "req.number_in_text" and f.subject == "REQ-MOV-008" and f.severity == "minor" for f in j.findings)


def test_number_unit_forms_are_silent(base):
    assert not [f for f in base.findings if f.rule == "req.number_in_text"]        # "20 micrometres" = 20 um


def test_rated_flow(workproject):
    set_param(workproject, "REQ-MOV-008", "rated_flow", value="6 kg/s")
    j = run(workproject)
    f = [f for f in j.findings if f.rule == "req.rated_flow"]
    assert len(f) == 1 and f[0].subject == "REQ-MOV-008" and f[0].severity == "major"
    assert f[0].values["requirement"] == pytest.approx(6.0)


def test_rated_flow_within_one_percent_is_silent(workproject):
    set_param(workproject, "REQ-MOV-008", "rated_flow", value="5.33 kg/s")
    assert not ids(run(workproject), "req.rated_flow")


def test_role_conflict(workproject):
    from dossier.model import ReqParameter
    r = workproject.requirement("REQ-MOV-003")
    edit(workproject, "REQ-MOV-003", parameters={**r.parameters, "meop_b": ReqParameter(value="11 MPa", role="meop")})
    j = run(workproject)
    assert any(f.rule == "req.role_conflict" and f.severity == "major" for f in j.findings)


def test_no_phase(workproject):
    edit(workproject, "REQ-MOV-003", applies=workproject.requirement("REQ-MOV-003").applies.model_copy(update={"phases": []}))
    j = run(workproject)
    assert any(f.rule == "req.no_phase" and f.subject == "REQ-MOV-003" for f in j.findings)


def test_duty_cycle_uncovered(workproject):
    r = workproject.requirement("REQ-MOV-006")
    edit(workproject, "REQ-MOV-006", applies=r.applies.model_copy(update={"duty_cycles": ["DC-RUN"]}))
    j = run(workproject)
    fs = [f for f in j.findings if f.rule == "req.duty_cycle_uncovered"]
    assert fs and all(f.severity == "minor" for f in fs)


def test_material_temperature(workproject):
    m = workproject.material_of()
    workproject.materials[m.id] = m.model_copy(update={"service_temperature": (Value.model_validate("150 K"),
                                                                              Value.model_validate("600 K"))})
    j = run(workproject)
    assert any(f.rule == "req.material_temperature" and f.severity == "major" for f in j.findings)


def test_material_in_range_is_silent(base):
    assert not [f for f in base.findings if f.rule in ("req.material_temperature", "req.material_compatibility")]
