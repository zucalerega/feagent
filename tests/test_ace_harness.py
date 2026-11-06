"""ace.harness: the opt-in feagent mesh harness in the ACE agent's tool results (ACE_HARNESS=1).
Mock LLM only: no API calls, no key needed."""

import hashlib
import json

import feagent.geometry
import feagent.quality
import pytest

from ace import agent, config, fea, harness
from ace.agent import TOOLS, Tools, run_agent, system_prompt, task_prompt, tools_for
from ace.mock import MockLLM

D3, D2 = "nafems_le10", "cooks_membrane"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _rounded(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return x


class SpyLLM(MockLLM):
    """MockLLM that keeps the tool schemas it was offered."""

    def __init__(self):
        self.offered = []

    def agent_turn(self, system, messages, tools, run_id="", role="generator"):
        self.offered.append(tools)
        return super().agent_turn(system, messages, tools, run_id=run_id, role=role)


def _mock_run(pid, tmp_path):
    llm = SpyLLM()
    record = run_agent(pid, f"g0-{pid}-0", 0, "", tmp_path / pid, llm)
    return record, [t["result"] for t in record["trace"] if t["type"] == "tool_result"], llm.offered


@pytest.fixture
def harness_on(monkeypatch):
    monkeypatch.setattr(config, "HARNESS", True)


def _never(*args, **kwargs):
    raise AssertionError("harness called with the switch off")


def test_switch_off_never_calls_the_harness_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HARNESS", False)
    monkeypatch.setattr(agent, "SCORE", "point")
    monkeypatch.setattr(harness, "inspect", _never)
    monkeypatch.setattr(harness, "quality", _never)
    for pid in (D3, D2):
        assert tools_for(fea.get_problem(pid)) is TOOLS          # the very same specs object
        record, results, offered = _mock_run(pid, tmp_path)
        assert record["stop"] == "submitted"
        assert "harness" not in results[0] and "mesh_gates" not in results[1]
        assert all(tools is TOOLS for tools in offered)
    assert "harness" not in system_prompt("") and "mesh_gates" not in task_prompt(D3)


def test_harness_on_3d_inspect_has_a_digest_labelled_by_group(tmp_path, harness_on):
    tools = Tools(D3, tmp_path)
    out, is_err = tools.call("inspect_geometry", {"using": []})
    assert not is_err and {"dim", "bbox", "groups"} <= set(out)
    h = out["harness"]
    assert set(h) == {"digest", "features", "hazards"}
    digest = h["digest"]
    assert "NAMED SURFACE GROUPS" in digest and "- face_hole: faces [" in digest
    faces = next(s for s in digest.split("\n\n") if s.startswith("FACES ("))
    assert faces.splitlines()[1].split()[-2:] == ["nbrs", "groups"]                  # the added groups column
    assert any(row.endswith("  face_hole") for row in faces.splitlines()[2:])   # the elliptic hole's face
    assert "spec:" not in digest and "detail=full" not in digest and len(digest) < 8000   # no feagent-only hints


def test_harness_on_3d_mesh_has_a_quality_table_and_the_same_mesh(tmp_path, harness_on, monkeypatch):
    request = {"h": 0.15, "order": 2, "using": []}
    on = Tools(D3, tmp_path / "on")
    out, is_err = on.call("mesh", request)
    assert not is_err
    gates = {g["id"]: g for g in out.pop("mesh_gates")}
    assert {"inverted", "sicn", "volume_fidelity"} <= set(gates)
    assert all(gates[i]["status"] == "pass" for i in ("inverted", "sicn", "volume_fidelity"))
    assert not [g for g in gates.values() if g["status"] == "fail"]
    assert set(next(iter(gates.values()))) == {"id", "status", "value", "threshold", "message"}
    monkeypatch.setattr(config, "HARNESS", False)
    off = Tools(D3, tmp_path / "off")
    assert off.call("mesh", request) == (out, False)                      # same result besides `quality`
    assert on.meshes[0]["sha256"] == off.meshes[0]["sha256"] and "mesh_gates" not in on.meshes[0]


def test_harness_on_3d_tool_descriptions_and_mock_run(tmp_path, harness_on):
    specs = tools_for(fea.get_problem(D3))
    assert [t["name"] for t in specs] == [t["name"] for t in TOOLS]
    changed = {t["name"] for t, old in zip(specs, TOOLS) if t != old}
    assert changed == {"inspect_geometry", "mesh"}
    assert "`harness`" in specs[0]["description"] and "`mesh_gates`" in specs[1]["description"]
    record, results, offered = _mock_run(D3, tmp_path)
    assert record["stop"] == "submitted" and offered[0] == specs
    assert "harness" in results[0] and "mesh_gates" in results[1]


def test_harness_on_2d_problem_is_unchanged(tmp_path, harness_on, monkeypatch):
    monkeypatch.setattr(harness, "inspect", _never)
    monkeypatch.setattr(harness, "quality", _never)
    monkeypatch.setattr(agent, "SCORE", "point")
    assert tools_for(fea.get_problem(D2)) is TOOLS
    _, results, offered = _mock_run(D2, tmp_path)
    assert "harness" not in results[0] and "mesh_gates" not in results[1]
    assert all(tools is TOOLS for tools in offered)


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_harness_failures_never_fail_the_tool_call(tmp_path, harness_on, monkeypatch):
    monkeypatch.setattr(feagent.geometry, "digest_text", _boom)
    monkeypatch.setattr(feagent.quality, "assess_mesh", _boom)
    tools = Tools(D3, tmp_path)
    out, is_err = tools.call("inspect_geometry", {"using": []})
    assert not is_err and out["groups"] and "boom" in out["harness"]["error"]
    request = {"h": 0.3, "order": 1, "using": []}
    out, is_err = tools.call("mesh", request)
    assert not is_err and "boom" in out.pop("mesh_gates")["error"]
    monkeypatch.setattr(config, "HARNESS", False)
    assert Tools(D3, tmp_path / "off").call("mesh", request) == (out, False)   # the mesh result is unchanged
