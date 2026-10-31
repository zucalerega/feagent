"""Diagnosis layers: oracles, stage invariants, counterfactual reruns, and the diagnoser agent
(in mock mode). Planted failures double as the evaluation set: each one has a known root cause."""

import pytest

from ace import fea, oracles, planted, stages
from ace.checks import reference_setup, tolerance
from ace.config import TASKS


@pytest.fixture(scope="module")
def cal():
    """Accuracy target per problem (the reference tolerance), keyed like the old calibration."""
    return {pid: {"target_err": tolerance(pid)} for pid in TASKS}


@pytest.fixture(scope="module")
def planted_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("planted")
    return {name: planted.make_run(name, root / name) for name in planted.PLANTED}


def _diagnose(run, cal):
    e_u = cal[run["problem_id"]]["target_err"]
    orc = stages.run_oracles(run, e_u)
    return orc, stages.stage_report(run, orc, e_u)


# --------------------------------------------------------------------------- layers 1 + 2


@pytest.mark.parametrize("name", list(planted.PLANTED))
def test_planted_failure_is_found_at_its_stage(name, planted_runs, cal):
    run = planted_runs[name]
    _, rep = _diagnose(run, cal)
    assert rep["first_failing_stage"] == run["label"]["true_stage"]


# The oracle that must flag each planted failure (the evidence the diagnoser starts from).
EXPECTED_ORACLE = {
    "load_on_wrong_face": "spec_compliance",
    "unit_error": "spec_compliance",
    "mesh_too_coarse": "mesh_convergence",
    "singular_peak": "symmetry",            # the point constraint replaced a symmetry plane
    "gamed_extra_support": "gaming",
    "gamed_dropped_load": "gaming",
}


@pytest.mark.parametrize("name,oracle", EXPECTED_ORACLE.items())
def test_planted_failure_trips_the_right_oracle(name, oracle, planted_runs, cal):
    orc, _ = _diagnose(planted_runs[name], cal)
    assert orc[oracle]["status"] == "fail", orc[oracle]


def test_missing_constraint_is_caught_before_solving(planted_runs, cal):
    orc, rep = _diagnose(planted_runs["missing_constraint"], cal)
    assert "rigid_body_mode" in orc["spec_compliance"]["why"]
    assert rep["solve"]["status"] == "fail"                       # a symptom, downstream of bcs


def test_unit_error_also_breaks_sanity_bounds(planted_runs, cal):
    orc, _ = _diagnose(planted_runs["unit_error"], cal)
    assert orc["sanity_bounds"]["status"] == "fail"


@pytest.mark.parametrize("pid", list(TASKS))
def test_no_oracle_fails_on_the_reference_setup(pid, cal, tmp_path):
    problem = fea.get_problem(pid)
    if fea.is_flow(problem):
        from feacore.solver._foam_env import find_openfoam

        if find_openfoam() is None:
            pytest.skip("flow problem: OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")
    req = planted.GOOD_MESH[pid]
    m = fea.make_mesh(problem, tmp_path / "m.msh", h=req["h"], order=req.get("order", 1),
                      refinements=req.get("refinements"), quads=req.get("quads", False))
    setup = reference_setup(problem)
    value = fea.run_solve(problem, m, setup, TASKS[pid]["qoi"])["value"]
    orc = oracles.evaluate(pid, m, req, setup, value, cal[pid]["target_err"])
    failed = {k: o["why"] for k, o in orc.items() if isinstance(o, dict) and o.get("status") == "fail"}
    assert not failed


def test_first_failing_stage_is_the_earliest():
    rep = {"geometry": {"status": "pass"}, "mesh": {"status": "pass"}, "bcs": {"status": "fail"},
           "solve": {"status": "fail"}, "postprocess": {"status": "fail"}}
    assert stages.first_failing_stage(rep) == "bcs"
    assert stages.first_failing_stage({s: {"status": "pass"} for s in stages.STAGES}) is None


# --------------------------------------------------------------------------- layer 3: counterfactual reruns

from ace import counterfactual as cf  # noqa: E402

# planted case -> (stage, the change that fixes it, a change that does not)
COUNTERFACTUALS = {
    "missing_constraint": ("bcs", {"name": "fix_constraint", "args": {"group": "sym_x0", "type": "symmetry"}},
                           {"name": "change_element_order", "args": {"order": 1}}),
    "load_on_wrong_face": ("bcs", {"name": "fix_load", "args": {"group": "edge_BC", "type": "pressure", "value": -1e7,
                                                                 "remove_from": "edge_DA"}},
                           {"name": "change_element_order", "args": {"order": 1}}),
    "unit_error": ("bcs", {"name": "convert_units", "args": {"target": "E", "factor": 1e6}},
                   {"name": "convert_units", "args": {"target": "loads", "factor": 2}}),
    "mesh_too_coarse": ("mesh", {"name": "set_local_mesh_size", "args": {"point": [2.0, 0.0], "size": 0.08, "radius": 0.6}},
                        {"name": "fix_constraint", "args": {"group": "edge_AB", "type": "symmetry"}}),
    "singular_peak": ("bcs", {"name": "fix_constraint", "args": {"group": "sym_y0", "type": "symmetry", "remove": [[0.1, 0.0]]}},
                      {"name": "change_element_order", "args": {"order": 1}}),
    "gamed_extra_support": ("bcs", {"name": "fix_constraint", "args": {"group": "edge_BC", "type": "free"}},
                            {"name": "change_element_order", "args": {"order": 1}}),
}


def _stage_of(change):
    return cf.INTERVENTIONS[change["name"]][0]


@pytest.mark.parametrize("name", list(COUNTERFACTUALS))
def test_right_fix_flips_wrong_change_does_not(name, planted_runs, cal, tmp_path):
    run = planted_runs[name]
    e_u = cal[run["problem_id"]]["target_err"]
    _, fix, wrong = COUNTERFACTUALS[name]
    assert cf.rerun_from_stage(run, _stage_of(fix), fix, e_u, tmp_path)["flipped"]
    assert not cf.rerun_from_stage(run, _stage_of(wrong), wrong, e_u, tmp_path)["flipped"]


def test_interventions_are_whitelisted_and_stage_checked(planted_runs, tmp_path):
    run = planted_runs["control_lame"]
    with pytest.raises(cf.InterventionError):
        cf.rerun_from_stage(run, "mesh", {"name": "rm -rf", "args": {}}, 0.04, tmp_path)
    with pytest.raises(cf.InterventionError):
        cf.rerun_from_stage(run, "mesh", {"name": "fix_load", "args": {}}, 0.04, tmp_path)


def test_rerun_reports_solver_time_and_mesh_used(planted_runs, cal, tmp_path):
    run = planted_runs["control_le1"]
    r = cf.rerun_from_stage(run, "mesh", {"name": "refine_mesh", "args": {"scope": "global", "factor": 1.2}},
                            cal["nafems_le1"]["target_err"], tmp_path)
    assert r["solver_seconds"] >= 0 and r["mesh_used"]["nodes"] > run["meshes"][0]["nodes"]


# --------------------------------------------------------------------------- the diagnoser (mock mode)

from ace import diagnoser  # noqa: E402
from ace.checks import check_run  # noqa: E402
from ace.curator import curate  # noqa: E402
from ace.mock import MockLLM, fake_rerun  # noqa: E402
from ace.playbook import Bullet, Playbook  # noqa: E402


def _prepared(planted_runs, cal, names):
    """Planted runs with the fields the loop adds: checks, oracles, stages."""
    out = []
    for n in names:
        r = dict(planted_runs[n])
        e_u = cal[r["problem_id"]]["target_err"]
        r["checks"] = check_run(r["problem_id"], r)
        r["oracles"] = stages.run_oracles(r, e_u)
        r["stages"] = stages.stage_report(r, r["oracles"], e_u)
        out.append(r)
    from ace.checks import score_population
    score_population(out)
    return out


@pytest.fixture(scope="module")
def lame_runs(planted_runs, cal):
    return _prepared(planted_runs, cal, ["control_lame", "missing_constraint", "unit_error", "singular_peak"])


def _session(runs, cal, tmp_path, flip):
    pid = runs[0]["problem_id"]
    return diagnoser.diagnose_problem(pid, runs, Playbook(), MockLLM(), cal[pid]["target_err"], tmp_path,
                                      rerun_fn=fake_rerun(flip))


def test_confirmed_rerun_keeps_the_claim(lame_runs, cal, tmp_path):
    res = _session(lame_runs, cal, tmp_path, flip=True)
    d = res["diagnosis"]
    assert d.runs[0].diagnosis_status == "confirmed"
    assert d.runs[0].hypotheses_tested[0].result == "confirmed"
    assert d.runs[0].hypotheses_tested[0].after["first_failing_stage"] is None   # filled from the real log
    assert any(les.evidence_type == "counterfactual" for les in d.lessons)
    assert d.verification_notes == []


def test_unflipped_rerun_downgrades_every_claim(lame_runs, cal, tmp_path):
    d = _session(lame_runs, cal, tmp_path, flip=False)["diagnosis"]
    assert d.runs[0].hypotheses_tested[0].result == "rejected"        # the mock claimed "confirmed"
    assert d.runs[0].diagnosis_status == "contrast_only"
    assert not any(les.evidence_type == "counterfactual" for les in d.lessons)
    assert len(d.verification_notes) >= 3


def test_claim_without_any_rerun_is_not_tested():
    d = diagnoser.Diagnosis(divergence="", lessons=[], runs=[diagnoser.RunDiagnosis(
        run_id="r1", diagnosis_status="confirmed", stage="mesh", category="mesh_too_coarse", cause="", evidence="",
        hypotheses_tested=[diagnoser.Hypothesis(stage="mesh", result="confirmed", intervention=diagnoser.Intervention(
            name="refine_mesh", args={"scope": "global", "factor": 2}))])])
    d = diagnoser.verify(d, reruns=[])
    assert d.runs[0].hypotheses_tested[0].result == "not_tested" and d.runs[0].diagnosis_status == "contrast_only"


def test_tool_and_rerun_limits(lame_runs, cal, tmp_path):
    tools = diagnoser.DiagnoserTools("lame_cylinder_plane_strain", lame_runs, Playbook(),
                                     cal["lame_cylinder_plane_strain"]["target_err"], tmp_path, fake_rerun(True))
    rid = lame_runs[1]["run_id"]
    change = {"name": "refine_mesh", "args": {"scope": "global", "factor": 2}}
    outs = [tools.call("rerun_from_stage", {"run_id": rid, "stage": "mesh", "change": change})[0]
            for _ in range(diagnoser.DIAGNOSER_MAX_RERUNS + 1)]
    assert "error" in outs[-1] and len(tools.reruns) == diagnoser.DIAGNOSER_MAX_RERUNS
    while tools.calls < diagnoser.DIAGNOSER_MAX_TOOL_CALLS:
        tools.call("list_runs", {})
    out, is_err = tools.call("list_runs", {})
    assert is_err and "limit" in out["error"]


def test_read_trace_is_paginated(lame_runs, cal, tmp_path):
    tools = diagnoser.DiagnoserTools("lame_cylinder_plane_strain", lame_runs, Playbook(), 0.04, tmp_path)
    page = tools.call("read_trace", {"run_id": lame_runs[0]["run_id"], "start": 0})[0]
    assert page["total_steps"] == 4 and len(page["steps"]) <= diagnoser.TRACE_PAGE


def test_curator_keeps_only_evidence_backed_lessons_and_counterfactual_goes_first():
    les = lambda et, c: diagnoser.DiagnosisLesson(section="meshing", scope="general", evidence="", content=c,  # noqa: E731
                                                  evidence_type=et, confidence="medium")
    d = diagnoser.Diagnosis(divergence="", runs=[], lessons=[les("contrast", "c"), les("oracle_only", "o"),
                                                              les("counterfactual", "x")])
    ops, _ = curate([("p", d, ["r1"])], Playbook(), MockLLM())
    assert sorted(o.evidence_type for o in ops if o.kind == "ADD") == ["contrast"] or \
        {o.evidence_type for o in ops} <= {"contrast", "counterfactual"}
    assert "oracle_only" not in {o.evidence_type for o in ops}
    pb = Playbook([Bullet("mesh-0001", "meshing", "contrast one", helpful=3, evidence_type="contrast"),
                   Bullet("mesh-0002", "meshing", "proven one", evidence_type="counterfactual")])
    assert pb.render().index("mesh-0002") < pb.render().index("mesh-0001")
