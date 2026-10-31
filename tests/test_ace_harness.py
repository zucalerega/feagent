"""ace.harness: the opt-in feagent mesh harness in the ACE agent's tool results (ACE_HARNESS=1).
Mock LLM only: no API calls, no key needed."""

import hashlib
import json

import feagent.geometry
import feagent.quality
import pytest

from ace import agent, config, fea, harness
from ace.agent import TOOLS, Tools, run_agent, system_prompt, task_prompt, tool_specs
from ace.mock import MockLLM

D3, D2 = "nafems_le10", "cooks_membrane"

# Recorded from the code before ace/harness.py existed (sha256 of sort_keys JSON; tool-result floats
# rounded to 6 significant digits). With the switch off all of these must stay exactly the same.
GOLDEN = {
    "tools": "65ce83f8abd350cd8b52c53fa3b32e983988f197bad492fd0f55f3e5ac34eb3b",
    "system": "5a9afb49352b618b4783a58d7c0c55b327ebccde4f80ead7ca7a6d192018f707",
    "task": "7a2a1c51b76807057fc821264ca52ebdd4ba0ff7f6c5d9ff22c28b312b146832",
    D3: "7b31aa0f5ffa9818e7ac37ac6218b1cc35c71c49578cc74eaa7ac8709d057562",
    D2: "96bc40efbf4d03c0c1824745cd3cd6c8ced767e31f06e3cd5a3e41febd98ea18",
}


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


def test_switch_off_is_byte_identical_to_before(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HARNESS", False)
    monkeypatch.setattr(agent, "SCORE", "point")
    monkeypatch.setattr(harness, "inspect", _never)
    monkeypatch.setattr(harness, "quality", _never)
    assert _sha(TOOLS) == GOLDEN["tools"]
    assert _sha([system_prompt(""), system_prompt("## meshing\n[mesh-0001] x")]) == GOLDEN["system"]
    assert _sha({p: task_prompt(p) for p in config.TASKS}) == GOLDEN["task"]
    for pid in (D3, D2):
        record, results, offered = _mock_run(pid, tmp_path)
        assert record["stop"] == "submitted"
        assert _sha(_rounded(results)) == GOLDEN[pid], pid
        assert all(tools is TOOLS for tools in offered)


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
    gates = {g["id"]: g for g in out.pop("quality")}
    assert {"inverted", "sicn", "volume_fidelity"} <= set(gates)
    assert all(gates[i]["status"] == "pass" for i in ("inverted", "sicn", "volume_fidelity"))
    assert not [g for g in gates.values() if g["status"] == "fail"]
    assert set(next(iter(gates.values()))) == {"id", "status", "value", "threshold", "message"}
    monkeypatch.setattr(config, "HARNESS", False)
    off = Tools(D3, tmp_path / "off")
    assert off.call("mesh", request) == (out, False)                      # same result besides `quality`
    assert on.meshes[0]["sha256"] == off.meshes[0]["sha256"] and "quality" not in on.meshes[0]


def test_harness_on_3d_tool_descriptions_and_mock_run(tmp_path, harness_on):
    specs = tool_specs(fea.get_problem(D3))
    assert [t["name"] for t in specs] == [t["name"] for t in TOOLS] and _sha(TOOLS) == GOLDEN["tools"]
    changed = {t["name"] for t, old in zip(specs, TOOLS) if t != old}
    assert changed == {"inspect_geometry", "mesh"}
    assert "`harness`" in specs[0]["description"] and "`quality`" in specs[1]["description"]
    record, results, offered = _mock_run(D3, tmp_path)
    assert record["stop"] == "submitted" and offered[0] == specs
    assert "harness" in results[0] and "quality" in results[1]


def test_harness_on_2d_problem_is_unchanged(tmp_path, harness_on, monkeypatch):
    monkeypatch.setattr(harness, "inspect", _never)
    monkeypatch.setattr(harness, "quality", _never)
    monkeypatch.setattr(agent, "SCORE", "point")
    assert tool_specs(fea.get_problem(D2)) is TOOLS
    _, results, offered = _mock_run(D2, tmp_path)
    assert "harness" not in results[0] and "quality" not in results[1]
    assert _sha(_rounded(results)) == GOLDEN[D2] and all(tools is TOOLS for tools in offered)


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
    assert not is_err and "boom" in out.pop("quality")["error"]
    monkeypatch.setattr(config, "HARNESS", False)
    assert Tools(D3, tmp_path / "off").call("mesh", request) == (out, False)   # the mesh result is unchanged
