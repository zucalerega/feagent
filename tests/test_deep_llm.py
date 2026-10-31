"""--deep LLM components (ace/deep_teacher.py, ace/self_reflect.py, ace/organizer.py) on the mock:
schemas, prompts and the mock answers. No API calls."""

import base64

import pytest

from ace import deep_teacher as dt
from ace import organizer as org
from ace import self_reflect as sr
from ace.llm import image_block, text_block
from ace.mock import MockLLM

TASK = "nafems_le1"


@pytest.fixture
def png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1), dpi=20)
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "x.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _run(k: int) -> dict:
    rid = f"g1-{TASK}-a{k}"
    trace = [{"type": "thinking", "text": "the error will sit at the inner edge", "turn": 0},
             {"type": "tool_call", "name": "mesh", "input": {"h": 0.1, "order": 2, "using": ["mesh-0001"],
                                                            "refinements": [{"group": "edge_inner", "size": 0.02,
                                                                             "radius": 0.2}]}, "turn": 1},
             {"type": "tool_result", "name": "mesh", "result": {"ok": True, "nodes": 900, "within_cap": True}, "turn": 1},
             {"type": "tool_call", "name": "submit", "input": {"solve": 1}, "turn": 2}]
    return {"run_id": rid, "problem_id": TASK, "cited": ["mesh-0001"], "trace": trace,
            "checks": {"score": 0.1 * k, "rel_error": 0.05 / (k + 1), "dofs": 1000 * (k + 1)}}


def _report(run: dict, png) -> dict:
    # A minimal ace.run_report.build_run_report result, with its scores added.
    from ace.run_report import add_scores
    hot = {"centroid": [0.1, 0.2], "share": 0.4, "nearest_group": "edge_inner", "distance": 0.0, "local_h": 0.05,
           "vm_ref": 1e8, "vm_rel_error": 0.03}
    return add_scores({
        "run_id": run["run_id"], "agent": dt.agent_name(run["run_id"]), "task": TASK,
        "mesh_request": {"h": 0.1, "order": 2, "refinements": []},
        "mesh_stats": {"nodes": 900, "dofs": 1800, "n_elements": 400, "min_quality": 0.5},
        "setup": {"formulation": "plane_stress"},
        "error_map": {"median_h": 0.1, "field_error": 0.02, "probe_error": 0.01, "n_hotspots": 1,
                      "hotspot_radius": 0.1, "hotspots": [hot]},
        "mesh_layout": {"h_range": [0.01, 0.2], "groups": {}},
        "check_points": [{"i": 0, "xyz": [0.1, 0.2], "disclosed": True, "vm_ref": 1e8, "vm_run": 1.01e8, "rel_err": 0.01}],
        "region_grid": {"shape": [1, 1], "cell_size": [1.0, 1.0],
                        "cells": [{"centre": [0.5, 0.5], "n_elements": 400, "mean_h": 0.1, "error_share": 1.0,
                                   "max_rel_err": 0.05}]},
        "images": {"mesh": str(png), "error": str(png), "stress": str(png)}}, run)


def test_image_and_text_blocks(png):
    b = image_block(png)
    assert b["type"] == "image" and b["source"]["media_type"] == "image/png"
    assert base64.b64decode(b["source"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert text_block("hi") == {"type": "text", "text": "hi"}


def test_schemas_round_trip():
    call = dt.MeshCall(h=0.1, order=2, refinements=[dt.Refinement(group="edge", size=0.01, radius=0.1)])
    fb = dt.TeacherFeedback(task=TASK, task_summary="s", per_agent=[dt.AgentFeedback(
        agent="a0", run_id="r", diagnosis="d", next_mesh_call=call,
        notes=[dt.NoteItem(section="meshing", content="c", kind="fix")])], shared_lessons=[])
    assert dt.TeacherFeedback.model_validate_json(fb.model_dump_json()) == fb
    s = sr.SelfReflection(what_happened="w", what_i_learned=[], plan_next="p", next_mesh_call=call)
    assert sr.SelfReflection.model_validate_json(s.model_dump_json()) == s
    o = org.OrganizerOutput(notes=[org.OrganizedNote(id=None, section="bcs", content="c", kind="idea", source="self")])
    assert org.OrganizerOutput.model_validate_json(o.model_dump_json()) == o


def test_teacher_prompt_and_mock(png):
    runs = [_run(k) for k in range(4)]
    reports = [_report(r, png) for r in runs]
    reports[3] = {"run_id": runs[3]["run_id"], "error": "report failed"}     # a failed report still gets taught
    user = dt.build_user(TASK, reports, runs, {"a0": "[mesh-0001] my note"}, "shared text", None)
    text = "\n".join(b["text"] for b in user if b["type"] == "text")
    assert dt.PROCESS_BRIEF in dt.SYSTEM
    for r in runs:
        assert r["run_id"] in text
    assert "mesh(" in text and "edge_inner" in text          # the trace excerpt
    assert "region grid" in text and "no report (report failed)" in text
    images = [b for b in user if b["type"] == "image"]
    assert len(images) == min(dt.MAX_IMAGES, 3 * 3)           # 3 runs with pictures, 3 each

    records: list = []
    fb = dt.teach(TASK, reports, runs, {}, "", None, MockLLM(), records)
    assert isinstance(fb, dt.TeacherFeedback) and fb.task == TASK
    assert {a.run_id for a in fb.per_agent} == {r["run_id"] for r in runs}
    assert len(records) == 1 and records[0]["role"] == "teacher"


def test_self_reflect_prompt_and_mock(png):
    run = _run(1)
    user = sr.build_user(_report(run, png), run, "my notes")
    assert run["run_id"] in user[0]["text"] and sr.PROCESS_BRIEF in sr.SYSTEM
    assert sum(b["type"] == "image" for b in user) == 3
    out = sr.reflect_on_run(_report(run, png), run, "", MockLLM(), [])
    assert isinstance(out, sr.SelfReflection) and run["run_id"] in out.what_happened


def test_organizer_keeps_known_ids_only():
    current = [{"id": "a1-mesh-0001", "section": "meshing", "content": "old", "kind": "keep", "source": "self", "gen": 0},
               {"id": "a1-bc-0001", "section": "bcs", "content": "old bc", "kind": None, "source": None, "gen": 0}]
    incoming = [{"section": "meshing", "content": "new", "kind": "fix", "source": "teacher"}]
    assert org.PROCESS_BRIEF in org.SYSTEM
    out = org.organize_private("a1", current, incoming, MockLLM(), [])
    assert [n["id"] for n in out] == ["a1-mesh-0001", "a1-bc-0001", None]
    assert out[2]["content"] == "new"

    class Liar:   # returns an unknown id and a repeated id: both must become new notes
        def structured(self, role, system, user, schema, records=None):
            n = {"section": "meshing", "content": "c", "kind": "idea", "source": "x"}
            return schema(notes=[dict(n, id="a1-mesh-0001"), dict(n, id="a1-mesh-0001"), dict(n, id="made-up")]), []
    assert [n["id"] for n in org.organize("a1", current, incoming, Liar(), [], max_notes=2)] == ["a1-mesh-0001", None]
