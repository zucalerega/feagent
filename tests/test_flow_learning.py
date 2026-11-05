"""--deep learning roles on flow tasks (ace/deep_teacher.py, ace/self_reflect.py, ace/process_brief.py)
on the mock: the flow brief goes with flow tasks only, the teacher's and the reflection's next mesh
call is a FlowMeshCall (quads, boundary layer, sweep; no order), and the experiment allocation varies
the cell structure instead of the element order. No API calls, no solves."""

import pytest

from ace import deep_teacher as dt
from ace import process_brief as pb
from ace import self_reflect as sr
from ace.mock import MockLLM

FLOW = "lid_driven_cavity_re100"
SOLID = "nafems_le1"


def _run(task: str, k: int) -> dict:
    return {"run_id": f"g1-{task}-a{k}", "problem_id": task, "trace": [],
            "checks": {"score": 0.1 * (k + 1), "rel_error": 0.05 / (k + 1), "dofs": 1000 * (k + 1)}}


@pytest.fixture(autouse=True)
def short_statement(monkeypatch):
    """The task statement without the automesher's numbers (in field mode they need a solve)."""
    monkeypatch.setattr(dt, "_task_statement", lambda task: f"Problem id: {task}")


def test_flow_brief_goes_with_flow_tasks_only():
    assert "FLOW (CFD)" not in pb.PROCESS_BRIEF and "FLOW (CFD)" not in pb.SINGLE_BRIEF
    assert "boundary_layer" not in pb.PROCESS_BRIEF            # the elasticity briefs are as they were
    assert dt.is_flow_task(FLOW) and not dt.is_flow_task(SOLID) and not dt.is_flow_task("no_such_task")

    for task, flow in ((FLOW, True), (SOLID, False)):
        runs = [_run(task, k) for k in range(2)]
        blocks, _ = dt.build_prefix(task, [{"run_id": r["run_id"]} for r in runs], runs, {}, "", None, None, None)
        assert ("## FLOW (CFD) TASKS" in blocks[0]["text"]) is flow
        user = sr.build_user({"run_id": runs[0]["run_id"]}, runs[0], "")
        assert ("## FLOW (CFD) TASKS" in user[0]["text"]) is flow
        ctx = pb.deep_context("a0", task, "S", "P", None, None, [], "", flow=flow)
        assert ("## FLOW (CFD) TASKS" in ctx) is flow
        if flow:   # right after the process brief, before the agent's own blocks
            assert ctx.index("HOW THIS SYSTEM WORKS") < ctx.index("## FLOW (CFD) TASKS") < ctx.index("You are agent a0")
    single = pb.single_context([FLOW], pb.Playbook(), pb.Playbook(), {}, flow=True)
    assert single.index("HOW THIS SYSTEM WORKS") < single.index("## FLOW (CFD) TASKS")


def test_flow_mesh_call_text():
    call = dt.FlowMeshCall.model_validate({
        "h": 0.025, "quads": True, "refinements": [{"group": "cylinder", "size": 0.004, "radius": 0.12}],
        "boundary_layer": {"groups": ["cylinder", "walls"], "first": 0.001, "ratio": 1.3, "layers": 4}}).model_dump()
    assert pb.mesh_call_text(call) == ("mesh(h=0.025, quads=true, refinements=[{group: cylinder, size: 0.004, "
                                       "radius: 0.12}], boundary_layer={groups: [cylinder, walls], first: 0.001, "
                                       "ratio: 1.3, layers: 4})")
    sweep = dt.FlowMeshCall.model_validate({"h": 0.001, "sweep": {"source": "inlet", "layers": 10}}).model_dump()
    assert pb.mesh_call_text(sweep) == "mesh(h=0.001, quads=false, refinements=[], sweep={source: inlet, layers: 10, ratio: 1.0})"
    solid = dt.MeshCall.model_validate({"h": 2.0, "order": 2, "refinements": []}).model_dump()
    assert pb.mesh_call_text(solid) == "mesh(h=2.0, order=2, refinements=[])"


def test_flow_allocation_varies_the_cell_structure_not_the_order():
    for n in range(2, 13):
        roles = dt.allocate([f"a{i}" for i in range(n)], flow=True)
        assert len(roles) == n and not any("order" in focus.lower() for _, _, focus in roles)
    small = dt.allocate(["a0", "a1", "a2", "a3"], flow=True)       # a task with 4 agents (--split)
    assert [r for _, r, _ in small] == ["control", "variant", "variant", "explorer"]
    assert "boundary layer" in small[2][2] and "CELL STRUCTURE" in small[3][2]
    assert dt.allocate([f"a{i}" for i in range(8)]) == dt.allocate([f"a{i}" for i in range(8)], flow=False)


def test_teacher_on_a_flow_task_keeps_the_flow_mesh_calls():
    runs = [_run(FLOW, k) for k in range(3)]
    fb = dt.teach(FLOW, [{"run_id": r["run_id"]} for r in runs], runs, {}, "", None, MockLLM(), [])
    assert isinstance(fb, dt.FlowTeacherFeedback) and len(fb.per_agent) == 3
    d = fb.model_dump()            # typed per_agent: the dump keeps the flow fields
    call = d["per_agent"][0]["next_mesh_call"]
    assert call["quads"] is True and call["boundary_layer"]["layers"] == 3 and "order" not in call
    assert {a["role"] for a in d["assignments"]} == {"control", "variant", "explorer"}
    assert "quads=true" in pb.mesh_call_text(call)
    solid = [_run(SOLID, k) for k in range(2)]
    fb = dt.teach(SOLID, [{"run_id": r["run_id"]} for r in solid], solid, {}, "", None, MockLLM(), [])
    assert type(fb) is dt.TeacherFeedback and fb.model_dump()["per_agent"][0]["next_mesh_call"]["order"] == 2


def test_reflection_on_a_flow_task_plans_a_flow_mesh_call():
    run = _run(FLOW, 0)
    out = sr.reflect_on_run({"run_id": run["run_id"]}, run, "", MockLLM(), [])
    assert isinstance(out, sr.FlowSelfReflection)
    assert out.model_dump()["next_mesh_call"]["boundary_layer"]["groups"] == ["walls"]
    solid = _run(SOLID, 0)
    assert type(sr.reflect_on_run({"run_id": solid["run_id"]}, solid, "", MockLLM(), [])) is sr.SelfReflection


def test_pre_teach_is_refused_on_flow_tasks():
    from ace.run_loop import _check_flow_tasks
    with pytest.raises(SystemExit, match="pre-teach"):
        _check_flow_tasks([FLOW], None, pre_teach=True)
