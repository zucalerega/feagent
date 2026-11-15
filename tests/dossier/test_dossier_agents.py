"""The judgement roles of dossier (dossier/agents): the brief each role reads, the check of what it submits,
the records it leaves, the mock run and the command line.

No test here calls a language model. `plant` writes hand-made stage files (a rule finding, a small plan, one
analysis) the way the pipeline writes them, so the tests do not need the engines of the other stages.
"""

import json
import struct
import zlib

import pytest

from dossier import context, units
from dossier.agents import brief, loop, mock, roles
from dossier.agents import submit as sub
from dossier.agents.__main__ import main
from dossier.model import (AnalysisRecord, Artifact, BCItem, CalcInput, CalcOutput, Check, Comparison, CriterionRef,
                           HandCalcResult, LoadCase, MeshLevel, Origin, ResultValue)
from dossier.project import Project
from dossier.records import Journal

ROLES = ["requirements-reviewer", "plan-reviewer", "analysis-checker"]
RULE_FINDING = "FND-REQ-MOV-001-MEOP"
RULE_STATEMENT = ("The requirement gives an MEOP of 10.5 MPa. The system context gives 10.72 MPa at the inlet during "
                  "shutdown. The requirement value is too low.")


@pytest.fixture(autouse=True)
def no_ste(monkeypatch):
    """The STE checker belongs to another engine: here it reports nothing, so that the warnings a test counts
    are the ones the test plants. The tests of the STE warning replace it again."""
    monkeypatch.setattr(sub, "ste_issues", lambda text, kind="descriptive", terms=None: [])


def _png(path):
    """A valid 1 x 1 PNG file."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff")) + chunk(b"IEND", b""))


def plant(project, plan=True, analysis=True):
    """Stage files as the pipeline writes them: one rule finding, a plan of three load cases with a hand
    calculation, and one analysis with a picture. Every number is made up for the tests."""
    ctx = context.build(project)
    j = Journal("validate", "engine:validate")
    j.find("req.meop_envelope", "error", "consistency", "REQ-MOV-001", "meop", "The MEOP is too low", RULE_STATEMENT,
           "The effect EF-IN-P-SHUTDOWN gives 10.72 MPa. The requirement REQ-MOV-001 gives 10.5 MPa.",
           suggestion="Increase the MEOP.", values={"meop": 10.5e6, "envelope": 10.7239e6},
           links=[("evidence", "EF-IN-P-SHUTDOWN")])
    j.assume("h_stagnant", "assumed", "Assumed heat transfer coefficient", "The value of h_stagnant is correct.",
             "The system context gives the value with the basis assumed.", "The temperature field changes.",
             by="rule:sys.assumed_parameter")
    project.write_stage("validate", findings=j.findings, decisions=j.decisions, assumptions=j.assumptions)
    if not plan:
        return
    by = Origin(by="engine:loadcases", stage="plan")
    proof = [x for x in ctx.loads(project, None, "PH-PROOF")]
    fire = [x for x in ctx.loads(project, None, "PH-FIRE") if x.kind in ("pressure", "force", "moment", "support")]
    lcs = [
        LoadCase(id="LC-PROOF", title="Proof pressure test", analysis_type="static_structural",
                 component=project.subject.id, phase="PH-PROOF", statement="The valve body has the proof pressure.",
                 loads=proof, criteria=[CriterionRef(requirement="REQ-MOV-002", criterion=0)],
                 hand_calcs=["HCR-PROOF-SHELL"], origin=by,
                 result_keys=["von_mises.membrane_bending@scls", "max_principal.membrane@SCL-SHELL"]),
        LoadCase(id="LC-BURST", title="Burst pressure test", analysis_type="static_structural",
                 component=project.subject.id, phase="PH-BURST", loads=ctx.loads(project, None, "PH-BURST"),
                 criteria=[CriterionRef(requirement="REQ-MOV-003", criterion=0)], scaled_from="LC-PROOF",
                 scale=2.5 / 1.5, status="scaled", origin=by),
        LoadCase(id="LC-FIRE", title="Steady firing", analysis_type="thermo_structural", component=project.subject.id,
                 phase="PH-FIRE", loads=fire, supports=["INT-INLET"], origin=by,
                 criteria=[CriterionRef(requirement="REQ-MOV-004", criterion=0)]),
    ]
    hc = HandCalcResult(
        id="HCR-PROOF-SHELL", title="Hoop stress in the shell", spec="HC-SHELL-HOOP", model="thick_cylinder_pressure",
        load_case="LC-PROOF", reference="Lame equations of a thick-walled cylinder.", origin=by,
        inputs=[CalcInput(name="p", si=15.75e6, unit="Pa", display="15.75 MPa", source="the load case"),
                CalcInput(name="r_i", si=0.022, unit="m", display="22 mm")],
        outputs=[CalcOutput(name="hoop_mean", si=57.75e6, unit="Pa", display="57.75 MPa")],
        assumptions=["The cylinder is long."], warnings=["The shell is short."],
        comparisons=[Comparison(output="hoop_mean", result_key="max_principal.membrane@SCL-SHELL", hand_value=57.75e6,
                                fe_value=63.9e6, ratio=1.106, band=(0.6, 1.5), agree=True, why="Equilibrium.")])
    jp = Journal("plan", "rule:loadcases.proportional")
    jp.decide("LC-BURST", "scaled", "Scale of LC-BURST", "Scale the results of LC-PROOF for LC-BURST.",
              "Every load of PH-BURST is the same multiple of the load of PH-PROOF.",
              alternatives=[("Solve LC-BURST.", "The result is the same.")], basis="calculation")
    project.write_stage("plan", load_cases=lcs, hand_calcs=[hc], findings=jp.findings, decisions=jp.decisions,
                        assumptions=jp.assumptions)
    if not analysis:
        return
    ja = Journal("idealize", "engine:structural")
    ja.decide("LC-PROOF", "support", "Support of the proof test", "Hold the body at six degrees of freedom.",
              "The loads of the proof test are in equilibrium.",
              alternatives=[("Fix the inlet flange face.", "A fixed face adds stress that is not real.")])
    an = AnalysisRecord(
        id="AN-PROOF", title="Proof pressure test", load_case="LC-PROOF", analysis_type="static_structural",
        statement="A linear static analysis with the proof pressure.", solver="skfem", material={"E": 193e9, "nu": 0.29},
        origin=Origin(by="engine:structural", stage="analyze"), runtime_s=12.0,
        bcs=[BCItem(type="displacement", features=["FT-INLET-FLANGE-FACE"], value=[0.0, 0.0, 0.0], interface="INT-INLET",
                    decision="DEC-LC-PROOF-SUPPORT", description="Three points hold six degrees of freedom."),
             BCItem(type="pressure", features=["FT-INLET-BORE"], value=15.75e6, effect="EF-IN-P-PROOF",
                    interface="INT-INLET", description="The pressure acts on the wetted faces.")],
        mesh=[MeshLevel(level=1, h=0.004, nodes=5200, cells=3000, min_quality=0.21),
              MeshLevel(level=2, h=0.0028, nodes=12400, cells=7500, min_quality=0.24)],
        results={
            "von_mises.membrane_bending@scls": ResultValue(
                key="von_mises.membrane_bending@scls", value=121.4e6, unit="Pa", uncertainty=0.021, status="converged",
                per_level=[115.9e6, 121.4e6], extrapolated=122.8e6, order=1.9, near="SCL-FLOOR"),
            "max_principal.membrane@SCL-SHELL": ResultValue(
                key="max_principal.membrane@SCL-SHELL", value=63.9e6, unit="Pa", uncertainty=0.008, status="converged",
                per_level=[61.0e6, 63.9e6]),
            "von_mises.peak@part": ResultValue(
                key="von_mises.peak@part", value=212e6, unit="Pa", status="singular", per_level=[163e6, 212e6],
                near="the edge of the seat pocket", location=[0.027, 0.016, 0.0], note="The value grows."),
            "szz.mean@FT-NECK": ResultValue(key="szz.mean@FT-NECK", value=11.0e6, unit="Pa", uncertainty=0.004,
                                            status="converged", per_level=[10.9e6, 11.0e6])},
        checks=[Check(id="CHK-PROOF-EQUILIBRIUM", kind="equilibrium", passed=True, value=2e-4, limit=0.01,
                      statement="The reactions balance the applied loads."),
                Check(id="CHK-PROOF-ENERGY", kind="energy", passed=None, statement="The energy check did not run.")],
        artifacts=[Artifact(path="analyses/LC-PROOF/von_mises.png", kind="picture", caption="Von Mises stress."),
                   Artifact(path="analyses/LC-PROOF/fields_L2.npz", kind="field")])
    _png(project.out / "analyses/LC-PROOF/von_mises.png")
    project.write_stage("analyze", project.analysis_dir("LC-PROOF") / "analysis.json", analyses=[an],
                        findings=ja.findings, decisions=ja.decisions, assumptions=ja.assumptions)


def good(**change):
    """A submission that the check accepts on the example. `change` replaces fields of its finding."""
    finding = {
        "subject": "dp_surge", "key": "column-estimate", "title": "The surge estimate has no limit of validity",
        "severity": "major", "category": "context",
        "statement": "The system context derives the surge with a rigid-column estimate.",
        "rationale": "The number dp_surge is 0.3239 MPa. It uses the closing time t_close of 50 ms.",
        "suggestion": "Give the wave speed of the duct.", "evidence": ["dp_surge", "t_close", "EF-IN-P-SHUTDOWN"],
        "confidence": "medium"}
    finding.update(change)
    return {
        "summary": "A test submission.",
        "findings": [finding],
        "decisions": [{"subject": "PH-TRANSPORT", "key": "add-requirement", "title": "Requirement for transport",
                       "choice": "Add a strength requirement for the transport phase.",
                       "rationale": "The effect EF-ACT-F-TRANSPORT puts a load on the neck.",
                       "alternatives": [{"option": "Leave the phase out.", "why_not": "The loads are real."}],
                       "evidence": ["EF-ACT-F-TRANSPORT"], "confidence": "high"}],
        "assumptions": [{"subject": "IF-PUMP-MOV", "key": "rigid-pump", "title": "Rigid pump housing",
                         "statement": "The pump housing is rigid.",
                         "rationale": "The description of the connection says that the housing is much stiffer.",
                         "impact": "A soft housing changes the stress at the inlet flange.",
                         "verify_by": "Compare the stiffness of the two flanges.", "conservative": None,
                         "evidence": ["INT-INLET"], "confidence": "low"}]}


# =========================================================================== roles, schema


def test_three_roles_with_instructions():
    assert roles.names() == ROLES
    assert roles.get("req").name == "requirements-reviewer" and roles.get("plan").name == "plan-reviewer"
    assert [roles.get(r).stage for r in ROLES] == ["validate", "plan", "solve"]
    assert len({roles.get(r).tag for r in ROLES}) == 3
    with pytest.raises(KeyError):
        roles.get("nobody")
    for name in ROLES:
        r, text = roles.get(name), roles.instructions(name)
        for part in (r.who, *r.look_for, *r.not_this, *roles.MUST_NOT):
            assert part in text
        assert "Simplified Technical English" in text and "25 words" in text and "never give a review" in text
        assert "python -m dossier.agents submit" in text
        assert "python -m dossier.agents" not in roles.instructions(name, shell=False)
    assert roles.role_of("agent:plan-reviewer (claude-opus-5-5)") == "plan-reviewer"
    assert roles.model_of("agent:plan-reviewer (claude-opus-5-5)") == "claude-opus-5-5"
    assert roles.role_of("rule:req.meop") is None and roles.model_of("agent:plan-reviewer") == ""


@pytest.mark.parametrize("role", ROLES)
def test_schema_is_a_valid_json_schema(example, role):
    s = sub.schema(role)
    assert json.loads(json.dumps(s)) == s and s["type"] == "object" and role in s["title"]
    assert set(s["properties"]) == {"summary", "findings", "decisions", "assumptions"}
    f = s["$defs"]["FindingIn"]
    assert f["properties"]["severity"]["enum"] == list(roles.SEVERITIES)
    assert f["properties"]["category"]["enum"] == list(roles.CATEGORIES)
    assert f["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert f["additionalProperties"] is False and "subject" in f["required"] and "suggestion" not in f["required"]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(s)
    jsonschema.validate(mock.canned(example, role), s)
    jsonschema.validate({}, s)                                  # every list can be empty
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"findings": [{"subject": "REQ-MOV-001"}]}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(good(severity="critical"), s)


def test_schema_is_what_the_structured_call_can_send():
    anthropic = pytest.importorskip("anthropic")                # ace.llm.structured sends transform_schema(...)
    t = anthropic.transform_schema(sub.Submission.model_json_schema())
    assert t["type"] == "object" and t["additionalProperties"] is False and "$defs" in t


# =========================================================================== the brief


@pytest.mark.parametrize("role", ROLES)
def test_brief_of_the_example_has_the_dossier_and_the_instructions(example, role):
    text = brief.build(example, role)
    data = brief.dossier(example, role)
    for r in example.reqs():
        assert f"`{r.id}`" in data
    for ph in example.system.phases:
        assert f"`{ph.id}`" in data
    assert "10.5 MPa" in data and "10.72 MPa" in data           # the claim of the requirement, and the context
    assert "10.5 MPa" not in roles.instructions(role)           # they come from the data, not from the instructions
    r = roles.get(role)
    for part in (r.who, *r.look_for, *r.not_this):
        assert part in text
    assert text.startswith(f"# Brief for the role `{role}`") and data in text
    assert "Simplified Technical English" in text and "How to cite" in text
    assert "`EF-IN-P-SHUTDOWN`" in data and "`INT-INLET`" in data and "`MAT-316L`" in data
    assert len(text) < 60_000
    assert "{'" not in data and '{"' not in data                 # tables and sentences, no raw JSON


def test_brief_says_plainly_which_stage_has_not_run(workproject):
    rr = brief.dossier(workproject, "requirements-reviewer")
    assert "| validate | has NOT run" in rr and "The validate stage has not run" in rr
    assert "There is no finding in the dossier." in rr
    pr = brief.dossier(workproject, "plan-reviewer")
    assert "The plan stage has not run. There is no load-case plan to review." in pr
    ac = brief.dossier(workproject, "analysis-checker")
    assert "No analysis has run. There is nothing to check." in ac
    plant(workproject, plan=True, analysis=False)
    ac = brief.dossier(workproject, "analysis-checker")
    assert "| plan | has run: 3 load case(s), 1 hand calculation(s) |" in ac and "| analyze | has NOT run" in ac


def test_requirements_reviewer_reads_the_context_with_sources(example):
    data = brief.dossier(example, "requirements-reviewer")
    # a named number with its value, expression, basis, source and what rests on it
    assert "| `dp_surge` | 0.3239 MPa | rho_lox * l_line * v_lox / t_close | derived |" in data
    assert "| `p_pump_max` | 10.4 MPa |  | specified | ENG-ICD-002 rev B, table 4 |" in data
    assert "| `h_stagnant` | 400 W/(m²·K) |  | assumed | NO SOURCE |" in data
    row = next(x for x in data.splitlines() if x.startswith("| `n_pressure_cycles`"))
    assert "`REQ-MOV-006`" in row                                # the explicit check of the requirement uses it
    # an effect with its value, its expression and where it acts
    assert "| `EF-IN-P-SHUTDOWN` | `INT-INLET`: wetted | pressure | 10.72 MPa | p_max_inlet |" in data
    assert "| `EF-OUT-T-CHILL` | `INT-OUTLET`: wetted | fluid_temperature | 280 K, h = 50 W/(m²·K) |" in data
    # the requirements in full: the text, the numbers, the criteria as the pipeline reads them
    assert '"The valve body shall be sufficiently stiff to keep the seats leak-tight under all loads."' in data
    assert "yield_factor (1.1) × `von_mises.peak@part` <= yield_strength of the material (170 MPa)" in data
    assert "| `PH-TRANSPORT` | Transport | handling |  | ball: closed, temperature: ambient | NONE |" in data
    assert "| `yield` | 1.25 |" in data and "spiral-wound gasket" in data and "`SCL-NECK-ROOT`" in data


def test_briefs_show_the_records_of_the_stages(workproject):
    plant(workproject)
    rr = brief.dossier(workproject, "requirements-reviewer")
    assert f"`{RULE_FINDING}` [error, consistency; rule:req.meop_envelope; review: open] about `REQ-MOV-001`" in rr
    assert RULE_STATEMENT in rr and "`ASM-H-STAGNANT-ASSUMED`" in rr and "Do not repeat them" in rr
    pr = brief.dossier(workproject, "plan-reviewer")
    assert "#### `LC-BURST` [static_structural, scaled, scaled from `LC-PROOF` by 1.66667] phase `PH-BURST`" in pr
    assert "`DEC-LC-BURST-SCALED`" in pr and "Solve LC-BURST. (not chosen: The result is the same.)" in pr
    assert "Phases with no load case: `PH-LEAK`, `PH-TRANSPORT`, `PH-CHILL`, `PH-PRESTART`, `PH-SHUTDOWN`." in pr
    # what the plan left out of a phase that has a load case
    left = next(x for x in pr.splitlines() if x.startswith("- `PH-FIRE` (load cases `LC-FIRE`)"))
    assert "`EF-ENV-A-FIRE` acceleration" in left and "`EF-IN-T-FIRE` fluid_temperature 90 K" in left
    assert "`EF-IN-HOLD`" not in left                            # the support is in the load case
    assert "`HCR-PROOF-SHELL` for `LC-PROOF`: p = 15.75 MPa, r_i = 22 mm → `hoop_mean` = 57.75 MPa." in pr
    assert "WARNINGS (outside the validity range of the formula): The shell is short." in pr
    assert "compares `hoop_mean` with `max_principal.membrane@SCL-SHELL`, band 0.6 to 1.5" in pr
    ac = brief.dossier(workproject, "analysis-checker")
    assert "#### `AN-PROOF` [static_structural, done] of load case `LC-PROOF`, phase `PH-PROOF`" in ac
    assert "| `von_mises.membrane_bending@scls` | 121.4 MPa | converged | 2.1 % | 115.9 → 121.4 MPa | 122.8 MPa |" in ac
    assert "| `von_mises.peak@part` | 212 MPa | SINGULAR |" in ac
    # a load of the model with the value of its effect; a support with the decision that justifies it
    assert "`pressure` on `FT-INLET-BORE` 15.75 MPa [effect `EF-IN-P-PROOF`]" in ac
    assert "Decision `DEC-LC-PROOF-SUPPORT` [engine:structural; review: open]: Hold the body at six degrees" in ac
    assert "Fix the inlet flange face. (not chosen: A fixed face adds stress that is not real.)" in ac
    # the loads of the load case that the model does not have
    assert "LOADS OF THE LOAD CASE THAT NO SUPPORT OR LOAD OF THE MODEL NAMES: `EF-IN-F-PROOF`" in ac
    assert "`CHK-PROOF-ENERGY` (energy): passed: not evaluated" in ac
    assert "analysis 63.9 MPa, ratio 1.11; band 0.6 to 1.5: agree" in ac
    # picture files by path, for an agent with file access
    folder = (workproject.out / "analyses" / "LC-PROOF").resolve()
    assert f"Picture files in `{folder}`" in ac and "`von_mises.png` (Von Mises stress)" in ac
    assert brief.pictures(workproject, "analysis-checker") == [(folder / "von_mises.png", "Von Mises stress.")]
    assert "- `AN-BURST`" not in ac and "Load cases with no analysis record: `LC-BURST`, `LC-FIRE`." in ac
    for text in (rr, pr, ac):
        assert len(text) < 60_000


def test_brief_compact_form_and_focus(workproject):
    plant(workproject)
    full = brief.dossier(workproject, "analysis-checker")
    compact = brief.dossier(workproject, "analysis-checker", max_chars=1000)
    assert "COMPACT form" not in full and "`szz.mean@FT-NECK`" in full
    assert "COMPACT form" in compact and "`szz.mean@FT-NECK`" not in compact
    assert "1 converged result(s) of AN-PROOF that no criterion and no hand calculation uses" in compact
    assert "| `von_mises.peak@part` | 212 MPa | SINGULAR |" in compact      # nothing that is not converged is cut
    assert "passed: `CHK-PROOF-EQUILIBRIUM` (equilibrium, value 0.0002, limit 0.01)" in compact
    assert "`--focus ID`" in brief.build(workproject, "analysis-checker", max_chars=1000)
    focus = brief.dossier(workproject, "plan-reviewer", focus=["LC-FIRE"])
    assert "#### `LC-FIRE`" in focus and "#### `LC-PROOF`" not in focus and "- `LC-PROOF` [static_structural" in focus


def test_brief_on_a_full_synthetic_dossier(workproject):
    """The briefs on a complete set of stage files that another engine's fixture writes (when it is there)."""
    fixture = pytest.importorskip("fixture_records")
    fixture.make_records(workproject)
    p = Project.load(workproject.root)
    ids = {"plan-reviewer": [x.id for x in p.collection("load_cases")],
           "analysis-checker": [x.id for x in p.collection("analyses")],
           "requirements-reviewer": [x.id for x in p.collection("findings")]}
    for role in ROLES:
        text = brief.build(p, role)
        for i in ids[role]:
            assert f"`{i}`" in text
        assert len(text) < 110_000
    one = p.collection("analyses")[0].id
    assert len(brief.build(p, "analysis-checker", focus=[one])) < len(brief.build(p, "analysis-checker"))


# =========================================================================== submit: accepted


def test_submit_writes_records_that_the_project_reads(workproject):
    digest = workproject.inputs_digest()
    res = sub.submit(workproject, "requirements-reviewer", good(), by="claude-opus-5-5")
    assert res.ok and not res.warnings and res.path == workproject.root / "agent" / "requirements-reviewer.json"
    p = Project.load(workproject.root)
    assert res.path in p.stage_files()
    f = next(x for x in p.collection("findings") if x.id == "FND-A-RR-DP-SURGE-COLUMN-ESTIMATE")
    assert f.origin.by == "agent:requirements-reviewer (claude-opus-5-5)" and f.origin.stage == "validate"
    assert roles.role_of(f) == "requirements-reviewer" and roles.model_of(f) == "claude-opus-5-5"
    assert f.rule == "agent.requirements-reviewer" and f.severity == "major" and f.category == "context"
    assert f.subject == "dp_surge" and f.suggestion == "Give the wave speed of the duct."
    # links from the evidence (the subject is linked by the record itself)
    assert [(x.rel, x.to) for x in f.links] == [("evidence", "t_close"), ("evidence", "EF-IN-P-SHUTDOWN")]
    assert ("about", "dp_surge") in [(x.rel, x.to) for x in f.refs()]
    # the confidence is in the last sentence of the rationale; the numbers the dossier confirms are in `values`
    assert f.rationale.endswith("The agent gives this finding a medium confidence.")
    assert f.values["0.3239 MPa"] == pytest.approx(323_900, rel=1e-3) and f.values["50 ms"] == pytest.approx(0.05)
    d = next(x for x in p.collection("decisions") if x.id == "DEC-A-RR-PH-TRANSPORT-ADD-REQUIREMENT")
    assert d.needs_review and d.basis == "engineering_judgement" and d.stage == "validate"
    assert d.alternatives[0].why_not == "The loads are real." and d.origin.by.startswith("agent:requirements-reviewer")
    a = next(x for x in p.collection("assumptions") if x.id == "ASM-A-RR-IF-PUMP-MOV-RIGID-PUMP")
    assert a.conservative is None and a.verify_by and a.rationale.endswith("The agent gives this assumption a low confidence.")
    # the stage file also holds the confidence, the warnings and the submission as data
    doc = json.loads(res.path.read_text())
    assert doc["stage"] == "requirements-reviewer" and doc["agent"]["by"] == "claude-opus-5-5"
    assert doc["agent"]["records"][f.id] == {"confidence": "medium", "warnings": []}
    assert doc["submission"]["findings"][0]["key"] == "column-estimate"
    # a role proposes records only: no review, no change of an input file, one file written
    assert not p.reviews() and not p.reviews_path.exists() and p.inputs_digest() == digest
    assert sorted(x.name for x in (p.root / "agent").iterdir()) == ["requirements-reviewer.json"]
    assert "WRITTEN: 1 finding(s), 1 decision(s), 1 assumption(s)" in res.text() and f.id in res.text()


def test_origin_stage_and_id_follow_the_role(workproject):
    plant(workproject)
    for role, stage, tag, statement in (("plan-reviewer", "plan", "PR", "The load case has no thermal load."),
                                        ("analysis-checker", "solve", "AC", "The inlet support is too stiff.")):
        res = sub.submit(workproject, role, good(subject="LC-FIRE", evidence=["LC-FIRE", "AN-PROOF", RULE_FINDING],
                                                 statement=statement))
        assert res.ok, res.errors
        assert res.findings[0].id == f"FND-A-{tag}-LC-FIRE-COLUMN-ESTIMATE"
        assert res.findings[0].origin == Origin(by=f"agent:{role}", stage=stage)       # no model name given
        assert res.decisions[0].stage == stage and res.assumptions[0].stage == stage
    ids = {x.id for x in Project.load(workproject.root).collection("findings")}
    assert {"FND-A-PR-LC-FIRE-COLUMN-ESTIMATE", "FND-A-AC-LC-FIRE-COLUMN-ESTIMATE", RULE_FINDING} <= ids


def test_second_submission_replaces_the_first(workproject):
    first = sub.submit(workproject, "requirements-reviewer", good())
    assert first.ok
    again = {"findings": [dict(good()["findings"][0], key="second", subject="t_close", evidence=["t_close"],
                               statement="The closing time has no tolerance.")]}
    second = sub.submit(workproject, "requirements-reviewer", again)
    assert second.ok and second.path == first.path
    p = Project.load(workproject.root)
    assert [x.id for x in p.collection("findings")] == ["FND-A-RR-T-CLOSE-SECOND"]
    assert p.collection("decisions") == [] and p.collection("assumptions") == []
    # the earlier records of the role do not count as existing findings: the same submission passes again
    assert sub.submit(workproject, "requirements-reviewer", again).ok
    # an empty submission is a valid result; it leaves an empty file
    assert sub.submit(workproject, "requirements-reviewer", {}).ok and Project.load(workproject.root).records() == {}


def test_check_writes_nothing(workproject):
    res = sub.check(workproject, "requirements-reviewer", good())
    assert res.ok and res.path is None and len(res.records) == 3 and not (workproject.root / "agent").exists()
    assert "ACCEPTED (check only, nothing written)" in res.text()


def test_brief_shows_the_earlier_submission_of_the_role(workproject):
    plant(workproject, plan=False)
    assert sub.submit(workproject, "requirements-reviewer", good(), by="model-x").ok
    workproject.set_review("FND-A-RR-DP-SURGE-COLUMN-ESTIMATE", "rejected", "A. Engineer", "The duct is short.")
    mine = brief.dossier(workproject, "requirements-reviewer")
    earlier = mine.split("### Your earlier submission")[1]
    assert "`FND-A-RR-DP-SURGE-COLUMN-ESTIMATE`" in earlier and 'rejected by A. Engineer: "The duct is short."' in earlier
    assert "`FND-A-RR-DP-SURGE-COLUMN-ESTIMATE`" not in mine.split("### Your earlier submission")[0]
    other = brief.dossier(workproject, "plan-reviewer")              # for another role it is an existing finding
    assert "Your earlier submission" not in other
    assert "`FND-A-RR-DP-SURGE-COLUMN-ESTIMATE`" in other.split("### Existing findings")[1]
    assert "Roles that have a submission in the dossier: requirements-reviewer." in other


# =========================================================================== submit: rejected


def _rejected(project, data, role="requirements-reviewer"):
    res = sub.submit(project, role, data)
    assert not res.ok and res.path is None and not res.records
    assert not (project.root / "agent").exists()                # the whole submission is rejected: nothing is written
    assert res.text().startswith(f"REJECTED: the submission of {role} has {len(res.errors)} error(s)")
    return res.errors


def test_rejects_an_id_that_is_not_in_the_project(workproject):
    errors = _rejected(workproject, good(subject="REQ-MOV-011"))
    assert errors == ["findings[0].subject: 'REQ-MOV-011' is not a record or an input of the project "
                      "(nearest ids: REQ-MOV-010, REQ-MOV-001, REQ-MOV-009)"]
    errors = _rejected(workproject, good(evidence=["dp_surge", "EF-IN-P-SHUTDWN", "LC-FIRE"]))
    assert errors[0].startswith("findings[0].evidence[1]: 'EF-IN-P-SHUTDWN' is not a record or an input of the project "
                                "(nearest ids: EF-IN-P-SHUTDOWN")
    assert errors[1].startswith("findings[0].evidence[2]: 'LC-FIRE' is not a record")      # the plan has not run
    data = good()
    data["assumptions"][0]["subject"] = "the pump"
    data["decisions"][0]["evidence"] = ["PH-TRANSPORT", "FND-NOTHING"]
    errors = _rejected(workproject, data)
    assert len(errors) == 2 and errors[0].startswith("decisions[0].evidence[1]: 'FND-NOTHING'")
    assert errors[1].startswith("assumptions[0].subject: 'the pump'")


def test_accepts_ids_of_every_kind_of_record(workproject):
    plant(workproject)
    evidence = ["REQ-MOV-004", "PH-FIRE", "EF-IN-P-FIRE", "p_pump_max", "meop", "Sy", "INT-INLET", "FT-NECK",
                "SCL-SHELL", "HC-SHELL-HOOP", "MAT-316L", "POLICY", "DC-RUN", "EL-PUMP", "IF-PUMP-MOV", "CMP-MOV-BODY",
                "SYS-OXFEED", RULE_FINDING, "DEC-LC-BURST-SCALED", "LC-FIRE", "HCR-PROOF-SHELL", "AN-PROOF",
                "ASM-H-STAGNANT-ASSUMED"]
    data = good(evidence=evidence)
    # a record can rest on another record of the same submission
    data["decisions"][0]["evidence"] = ["FND-A-AC-DP-SURGE-COLUMN-ESTIMATE"]
    res = sub.check(workproject, "analysis-checker", data)
    assert res.ok, res.errors
    assert [x.to for x in res.findings[0].links] == evidence
    assert [x.to for x in res.decisions[0].links] == ["FND-A-AC-DP-SURGE-COLUMN-ESTIMATE"]


def test_rejects_an_empty_text_and_a_missing_field(workproject):
    errors = _rejected(workproject, good(statement="  ", title=""))
    assert errors == ["findings[0].title: the text is empty", "findings[0].statement: the text is empty"]
    data = good()
    del data["findings"][0]["rationale"]
    data["findings"][0]["severty"] = "major"
    data["decisions"][0]["alternatives"][0]["why_not"] = ""
    data["assumptions"][0]["impact"] = ""
    assert _rejected(workproject, data) == [
        "findings[0].rationale: the field is missing", "findings[0].severty: unknown field",
        "decisions[0].alternatives[0].why_not: the text is empty", "assumptions[0].impact: the text is empty"]
    assert _rejected(workproject, [good()]) == ["the submission must be one JSON object with the lists `findings`, "
                                                "`decisions` and `assumptions`"]
    assert _rejected(workproject, {"findings": "none"})[0].startswith("findings: ")


def test_rejects_a_word_outside_the_vocabulary(workproject):
    errors = _rejected(workproject, good(severity="critical", category="safety", confidence="certain"))
    assert errors == [
        "findings[0].severity: 'critical' is not in the vocabulary (error | major | minor | info)",
        "findings[0].category: 'safety' is not in the vocabulary (consistency | coverage | quality | policy | conflict "
        "| material | context | model)",
        "findings[0].confidence: 'certain' is not in the vocabulary (high | medium | low)"]
    data = good()
    del data["findings"][0]["confidence"]
    assert _rejected(workproject, data) == ["findings[0].confidence: the field is missing (high | medium | low)"]


def test_rejects_a_key_that_is_not_a_slug_and_an_id_used_twice(workproject):
    assert _rejected(workproject, good(key="The surge is wrong!"))[0].startswith(
        "findings[0].key: 'The surge is wrong!' is not a short slug")
    data = good()
    data["findings"].append(dict(data["findings"][0], statement="The duct has no wave speed."))
    errors = _rejected(workproject, data)
    assert errors == ["findings[1]: the subject 'dp_surge' and the key 'column-estimate' give the id "
                      "FND-A-RR-DP-SURGE-COLUMN-ESTIMATE, which findings[0] of this submission also has; use another key"]


def test_rejects_a_finding_that_repeats_an_existing_finding(workproject):
    plant(workproject, plan=False)
    about = dict(subject="REQ-MOV-001", key="meop-low", evidence=["REQ-MOV-001"])
    said = "The MEOP of 10.5 MPa is lower than the inlet pressure of 10.72 MPa during shutdown."
    res = sub.submit(workproject, "requirements-reviewer", good(**about, statement=said))
    assert not res.ok and len(res.errors) == 1 and not (workproject.root / "agent").exists()
    assert res.errors[0].startswith(f"findings[0]: the finding duplicates the existing finding {RULE_FINDING} "
                                    f"(rule:req.meop_envelope): the same subject REQ-MOV-001 and a similar statement")
    # the same subject with another statement is a new finding; the same statement about another record too
    new = good(**about, statement="The requirement does not say at which location the pressure is measured.")
    assert sub.check(workproject, "requirements-reviewer", new).ok
    assert sub.check(workproject, "requirements-reviewer", good(**dict(about, subject="meop"), statement=said)).ok
    # the finding of another role counts as an existing finding
    assert sub.submit(workproject, "requirements-reviewer", new).ok
    other = sub.check(workproject, "plan-reviewer", new)
    assert not other.ok and "duplicates the existing finding FND-A-RR-REQ-MOV-001-MEOP-LOW" in other.errors[0]
    # two findings of one submission that say the same
    twice = good()
    twice["findings"].append(dict(twice["findings"][0], key="again"))
    assert "duplicates findings[0] of this submission" in sub.check(workproject, "analysis-checker", twice).errors[0]


def test_similarity_is_the_share_of_the_words_of_the_shorter_statement():
    assert sub.similarity(RULE_STATEMENT, RULE_STATEMENT) == 1.0
    assert sub.similarity("The MEOP of 10.5 MPa is lower than the inlet pressure of 10.72 MPa during shutdown.",
                          RULE_STATEMENT) >= sub.SIMILAR
    assert sub.similarity("The requirement does not say at which location the pressure is measured.",
                          RULE_STATEMENT) < 0.2
    assert sub.similarity("", RULE_STATEMENT) == 0.0


# =========================================================================== submit: warnings


def test_warns_about_a_number_that_the_dossier_does_not_have(workproject):
    data = good(rationale="The number dp_surge is 0.3239 MPa. A wave speed of 900 m/s gives a surge of 9.7 MPa.")
    res = sub.submit(workproject, "requirements-reviewer", data)
    assert res.ok and res.path is not None                       # accepted, with warnings
    rid = "FND-A-RR-DP-SURGE-COLUMN-ESTIMATE"
    assert len(res.warnings) == 2 and res.record_warnings[rid] == res.warnings
    assert res.warnings[0].startswith(f"{rid}.rationale: the text gives 900 m/s; no number of the context or the "
                                      f"records is within 1 % of it (the nearest is 9.463 m/s, v_lox)")
    assert res.warnings[1].startswith(f"{rid}.rationale: the text gives 9.7 MPa;") and "10.35 MPa" in res.warnings[1]
    f = Project.load(workproject.root).records()[rid]
    assert f.rationale.endswith("The agent gives this finding a medium confidence. The submission check does not "
                                "find 900 m/s and 9.7 MPa in the dossier.")
    assert list(f.values) == ["0.3239 MPa"]                      # only the number that the dossier confirms
    doc = json.loads(res.path.read_text())
    assert doc["agent"]["records"][rid]["warnings"] == res.warnings and "2 warning(s)" in res.text()


@pytest.mark.parametrize("text, known", [
    ("The effect gives 10.72 MPa.", True),            # the display of 10.7239 MPa: within 1 %
    ("The effect gives 10.9 MPa.", False),            # 1.6 % from the nearest pressure
    ("The effect gives 10723.9 kPa.", True),          # another unit, the same number in SI
    ("The acceleration is 25 g.", True),              # 25 times standard gravity (a_lat), not 25 grams
    ("The acceleration is 245.2 m/s².", True),
    ("The coefficient is 15570 W/(m²·K).", True),
    ("The torque is 55 N·m.", True),
    ("The ambient temperature is 20 degC.", True),    # 293.15 K
    ("The pocket has a diameter of 32 mm.", True),    # from the description of a feature
    ("The limit is 20 µm.", True),
    ("The proof test takes 5 min.", True),            # the duration of the phase
    ("The policy allows an uncertainty of 5 %.", True),
    ("The margin is 37 %.", False),
    ("The wall has a thickness of 6.5 mm.", False),
    ("The requirement gives 233 K to 323 K.", True),
    ("The requirement has 4 parameters in 3 phases.", True),      # no unit: not checked
])
def test_number_check_against_the_context(workproject, text, known):
    res = sub.check(workproject, "requirements-reviewer", good(rationale=text))
    assert res.ok and (not res.warnings) == known, res.warnings


def test_number_check_reads_the_records_too(workproject):
    plant(workproject)
    ok = good(rationale="The result is 121.4 MPa with an uncertainty of 2.1 %. The coarse mesh gives 115.9 MPa. "
                        "The hand calculation gives 57.75 MPa, and the ratio is 1.106. The mesh size is 2.8 mm.")
    assert not sub.check(workproject, "analysis-checker", ok).warnings
    bad = sub.check(workproject, "analysis-checker", good(rationale="The result is 131.4 MPa."))
    assert len(bad.warnings) == 1 and "131.4 MPa" in bad.warnings[0]
    # the numbers of the statement of a finding are checked too; a suggestion can give a new number
    res = sub.check(workproject, "analysis-checker", good(statement="The peak is 300 MPa.", suggestion="Use 0.3 mm."))
    assert len(res.warnings) == 1 and ".statement: the text gives 300 MPa" in res.warnings[0]


def test_quantities_in_a_text():
    q = sub.quantities("The MEOP is 10.5 MPa (h = 15570 W/(m²·K), 2 %, 5 minutes, M6 bolts, 90-293 K, 1.3e-5 1/K).")
    assert [(x.text, x.unit) for x in q] == [("10.5 MPa", "Pa"), ("15570 W/(m²·K)", "W/(m^2*K)"), ("2 %", ""),
                                             ("293 K", "K"), ("1.3e-5 1/K", "1/K")]
    assert q[0].si == pytest.approx(10.5e6) and q[2].si == pytest.approx(0.02)
    assert sub.quantities("It is 5 in the list, 3 at most, 7 a day and section 3.2.4 K.") == []
    assert units.si(units.parse("25 g0")) == pytest.approx(sub.quantities("25 g")[0].alt[0])


def test_warns_about_an_id_in_a_text_that_the_dossier_does_not_have(workproject):
    data = good(rationale="The requirement REQ-MOV-014 gives no limit. The document ENG-ICD-002 gives the pressure. "
                          "The phase PH-FIRE has the effect EF-IN-P-FIRE.")
    res = sub.check(workproject, "requirements-reviewer", data)
    assert res.ok and len(res.warnings) == 1
    assert res.warnings[0].startswith("FND-A-RR-DP-SURGE-COLUMN-ESTIMATE.rationale: the text names REQ-MOV-014, which "
                                      "is not a record or an input of the project (nearest ids: REQ-MOV-01")
    assert res.findings[0].rationale.endswith("The submission check does not find the id REQ-MOV-014 in the dossier.")


def test_warns_about_ste_issues_and_a_decision_with_no_alternative(workproject, monkeypatch):
    import sys
    import types

    import dossier

    class Issue:
        def __init__(self, rule, message):
            self.rule, self.message = rule, message

    def fake_check(text, kind="descriptive", terms=None):
        assert kind == "descriptive" and terms == workproject.terms and "valve body" in terms["technical_names"]
        return [Issue("STE-9.1", "the sentence has 31 words"), Issue("STE-3.6", "'should' is not approved")] \
            if "should" in text else []

    monkeypatch.undo()                                           # the real ste_issues, with a stand-in for dossier.ste
    fake = types.ModuleType("dossier.ste")
    fake.check = fake_check
    monkeypatch.setitem(sys.modules, "dossier.ste", fake)
    monkeypatch.setattr(dossier, "ste", fake, raising=False)
    data = good(statement="The system context should give the wave speed of the duct for the surge estimate.")
    data["decisions"][0]["alternatives"] = []
    res = sub.submit(workproject, "requirements-reviewer", data)
    assert res.ok and len(res.warnings) == 2
    assert res.warnings[0] == ("FND-A-RR-DP-SURGE-COLUMN-ESTIMATE.statement: the STE check reports 2 issue(s): STE-9.1: "
                               "the sentence has 31 words; STE-3.6: 'should' is not approved")
    assert res.findings[0].rationale.endswith("The STE check reports 2 issues in the text of the agent.")
    assert res.warnings[1] == "DEC-A-RR-PH-TRANSPORT-ADD-REQUIREMENT: the decision gives no alternative"
    assert res.decisions[0].rationale.endswith("The agent gives no alternative.") and res.decisions[0].needs_review
    # without the checker of the other engine the submission is accepted, and the result says so
    monkeypatch.setattr(sub, "ste_issues", lambda text, kind="descriptive", terms=None: None)
    res = sub.check(workproject, "requirements-reviewer", good())
    assert res.ok and not res.warnings and "the STE checker (dossier.ste) is not installed" in res.notes[0]


def test_real_ste_checker_when_it_is_installed(monkeypatch):
    monkeypatch.undo()                                           # the real dossier.ste, when the other engine is there
    out = sub.ste_issues("The requirement gives a pressure of 10.5 MPa.", "descriptive", None)
    assert out is None or all(isinstance(x, str) for x in out)


# =========================================================================== run


@pytest.mark.parametrize("role", ROLES)
def test_run_mock_end_to_end(workproject, role):
    out = loop.run(workproject, role, mock=True)
    assert out.ok and out.model == "mock" and out.attempts == 1 and out.usd == 0.0 and not out.error
    assert out.result.path == workproject.root / "agent" / f"{role}.json"
    p = Project.load(workproject.root)
    recs = [x for x in p.records().values() if roles.role_of(x) == role]
    assert sorted(x.kind for x in recs) == ["assumption", "decision", "finding"]
    for x in recs:
        assert x.origin.by == f"agent:{role} (mock)" and x.origin.stage == roles.get(role).stage
        text = getattr(x, "statement", "") or x.choice
        assert text.startswith(mock.MARK) and x.rationale.startswith(mock.MARK)       # canned, and it says so
    assert not out.result.warnings, out.result.warnings          # the canned numbers and ids are real
    # the brief that the run read is kept beside the records
    kept = out.brief_path.read_text()
    assert out.brief_path == workproject.root / "agent" / f"{role}.brief.md" and roles.get(role).who in kept
    assert out.brief_path not in p.stage_files() and "`REQ-MOV-010`" in kept
    assert mock.canned(workproject, role) == mock.canned(workproject, role)


def test_mock_runs_of_all_roles_live_side_by_side(workproject):
    plant(workproject)
    for role in ROLES:
        assert loop.run(workproject, role, mock=True).ok
    p = Project.load(workproject.root)
    assert len([x for x in p.records().values() if roles.role_of(x)]) == 9
    subjects = {roles.role_of(x): x.subject for x in p.collection("findings") if roles.role_of(x)}
    assert subjects == {"requirements-reviewer": "REQ-MOV-001", "plan-reviewer": "LC-PROOF",
                        "analysis-checker": "AN-PROOF"}
    assert loop.run(workproject, "plan-reviewer", mock=True).ok          # again: it replaces its own file


def test_run_sends_the_errors_back_once(workproject):
    llm = mock.MockLLM(workproject, "requirements-reviewer", faults=1)
    out = loop.run(workproject, "requirements-reviewer", llm=llm)
    assert out.ok and out.attempts == 2 and len(llm.calls) == 2 and len(out.usage) == 2
    first, second = llm.calls[0]["user"], llm.calls[1]["user"]
    assert "The tool checked your submission and rejected it" not in first and second.startswith(first)
    assert "1. findings[0].evidence[2]: 'REQ-DOES-NOT-EXIST' is not a record or an input of the project" in second
    assert '"evidence": [' in second and "Give the complete submission again" in second
    assert llm.calls[0]["role"] == "dossier-requirements-reviewer"
    # the system prompt is the instructions, the user message is the dossier
    assert roles.get("requirements-reviewer").who in llm.calls[0]["system"] and "`REQ-MOV-010`" in first
    assert "python -m dossier.agents" not in llm.calls[0]["system"]
    # a second rejection ends the run with nothing written
    other = Project.load(workproject.root)
    llm = mock.MockLLM(other, "plan-reviewer", faults=2)
    out = loop.run(other, "plan-reviewer", llm=llm)
    assert not out.ok and out.attempts == 2 and len(llm.calls) == 2 and "Nothing was written" in out.error
    assert not (other.root / "agent" / "plan-reviewer.json").exists() and "REJECTED" in out.text()


def test_run_without_an_answer_of_the_model(workproject):
    class Silent:
        model = "silent"

        def structured(self, role, system, user, schema, records=None):
            return None, records

    out = loop.run(workproject, "analysis-checker", llm=Silent())
    assert not out.ok and out.attempts == 0 and "no answer that fits the schema" in out.error


def test_run_without_a_key_says_what_it_needs(workproject, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(loop.NoKey) as e:
        loop.run(workproject, "requirements-reviewer")
    for part in ("ANTHROPIC_API_KEY", "--mock", "python -m dossier.agents brief", "python -m dossier.agents submit"):
        assert part in str(e.value)
    assert main(["run", str(workproject.root), "plan"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err and not (workproject.root / "agent").exists()
    assert loop.model_id("opus") == "claude-opus-5-5" and loop.model_id("sonnet") == "claude-sonnet-5-5"
    with pytest.raises(ValueError):
        loop.model_id("gpt")


def test_run_registers_the_role_for_ace_llm_and_sends_the_pictures(workproject):
    ace_config = pytest.importorskip("ace.config")
    ace_llm = pytest.importorskip("ace.llm")
    assert loop.configure("analysis-checker", "sonnet") == "claude-sonnet-5-5"
    name = loop.llm_role("analysis-checker")
    assert ace_config.MODELS[name] == "claude-sonnet-5-5" and ace_config.EFFORT[name] == "high"
    assert ace_config.MAX_TOKENS[name] == loop.MAX_TOKENS and ace_llm.TIMEOUT[name] == loop.TIMEOUT
    plant(workproject)

    class Fake(mock.MockLLM):
        model = "fake-model"

    llm = Fake(workproject, "analysis-checker")
    out = loop.run(workproject, "analysis-checker", llm=llm)
    assert out.ok and out.model == "fake-model" and out.result.findings[0].origin.by == "agent:analysis-checker (fake-model)"
    user = llm.calls[0]["user"]                                  # text, the picture with its caption, the task
    assert [b["type"] for b in user] == ["text", "text", "image", "text"]
    assert "von_mises.png" in user[1]["text"] and user[2]["source"]["media_type"] == "image/png"
    assert "#### `AN-PROOF`" in user[0]["text"] and user[3]["text"] == loop.TASK


# =========================================================================== the command line


def test_command_line(workproject, tmp_path, capsys):
    root = str(workproject.root)
    assert main(["roles"]) == 0
    listed = capsys.readouterr().out
    assert all(r in listed for r in ROLES)
    assert main(["schema", "plan-reviewer"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "Submission of the role plan-reviewer"
    out = tmp_path / "briefs" / "rr.md"
    assert main(["brief", root, "requirements-reviewer", "--out", str(out)]) == 0
    assert out.read_text() == brief.build(workproject, "requirements-reviewer") and "rr.md" in capsys.readouterr().out
    assert main(["brief", root, "analysis"]) == 0
    assert capsys.readouterr().out == brief.build(workproject, "analysis-checker")
    # submit: check only, then write, with the name of the model
    file = tmp_path / "submission.json"
    file.write_text(json.dumps(good()))
    assert main(["submit", root, "req", str(file), "--check"]) == 0 and not (workproject.root / "agent").exists()
    assert main(["submit", root, "req", str(file), "--by", "claude-fable-5-2"]) == 0
    assert "WRITTEN: 1 finding(s)" in capsys.readouterr().out
    f = Project.load(root).records()["FND-A-RR-DP-SURGE-COLUMN-ESTIMATE"]
    assert f.origin.by == "agent:requirements-reviewer (claude-fable-5-2)"
    # a rejected submission: the error list, exit status 1, the earlier file stays
    file.write_text(json.dumps(good(subject="REQ-MOV-011")))
    assert main(["submit", root, "req", str(file)]) == 1
    assert "1. findings[0].subject: 'REQ-MOV-011' is not a record" in capsys.readouterr().err
    assert "FND-A-RR-DP-SURGE-COLUMN-ESTIMATE" in Project.load(root).records()
    file.write_text("{not json")
    assert main(["submit", root, "req", str(file)]) == 1 and "not valid JSON" in capsys.readouterr().err
    # run --mock, and the wrong command lines
    assert main(["run", root, "plan-reviewer", "--mock"]) == 0
    assert "plan-reviewer [mock]" in capsys.readouterr().out
    assert main(["brief", root, "nobody"]) == 2 and "no role 'nobody'" in capsys.readouterr().err
    assert main(["run", root, "plan", "--mock", "--model", "gpt"]) == 0       # the mock does not use the model
    assert main(["brief", str(tmp_path / "nothing"), "req"]) == 1
    assert "The project has problems" in capsys.readouterr().err


def test_package_names_do_not_hide_the_modules():
    from dossier import agents
    assert agents.submit is sub and agents.brief is brief and agents.check is sub.check
    assert agents.build_brief is brief.build and agents.run is loop.run and set(agents.ROLES) == set(ROLES)
