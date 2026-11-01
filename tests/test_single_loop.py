"""Single-agent baseline (ace/single_loop.py) and the swarm comparison (ace/compare.py). Mock only."""

import json
import os
import subprocess
import sys
from pathlib import Path

from ace import deep_teacher, organizer, self_reflect
from ace.process_brief import PROCESS_BRIEF, SINGLE_BRIEF

REPO = Path(__file__).resolve().parents[1]
TASKS = ("cooks_membrane", "nafems_le11_axisym")


def test_single_brief_replaces_the_swarm_brief_in_every_learning_prompt():
    from ace.single_loop import SingleBrief

    seen = []

    class Fake:
        def structured(self, role, system, user, schema, records=None):
            seen.append(system)
            return None, records

    for system in (deep_teacher.SYSTEM, self_reflect.SYSTEM, organizer.SYSTEM):
        assert PROCESS_BRIEF in system                  # else the swap below would silently do nothing
        SingleBrief(Fake()).structured("teacher", system, "u", None)
    assert all(SINGLE_BRIEF in s and PROCESS_BRIEF not in s for s in seen)
    # and the critique of solve use is pitched for the baseline's solve budget, not the swarm's 4
    seen.clear()
    for system in (deep_teacher.SYSTEM, self_reflect.SYSTEM):
        SingleBrief(Fake(), 16).structured("teacher", system, "u", None)
    assert "at most 16 times per task" in seen[0] and "you had 16 solves on this task" in seen[1]


def test_a_crash_keeps_the_submitted_task():
    """A crash after one task was submitted errors only the open task (the submitted run is scored)."""
    from types import SimpleNamespace as NS

    from ace.single_loop import split_records

    tools = NS(tools={TASKS[0]: NS(submitted={"solve": 1}, meshes=[], solves=[]),
                      TASKS[1]: NS(submitted=None, meshes=[], solves=[])})
    recs = split_records(list(TASKS), 0, [], tools, [], "error", "529 overloaded")
    assert "error" not in recs[0] and recs[0]["stop"] == "submitted" and recs[0]["conversation_error"]
    assert recs[1]["error"] == "529 overloaded"


def test_a_turn_without_tool_calls_is_nudged_not_the_end():
    """tool_loop with on_stop goes on after a text-only turn; without it (the swarm) it stops there."""
    from types import SimpleNamespace as NS

    from ace.agent import tool_loop

    class Scripted:
        def __init__(self):
            self.n = 0

        def agent_turn(self, system, messages, tools, run_id="", **kw):
            self.n += 1
            if self.n == 1:          # a status line, no tool call
                return NS(content=[NS(type="text", text="Cook done, moving on.")], stop_reason="end_turn"), {}
            assert messages[-1]["content"][-1]["text"] == "continue"
            return NS(content=[NS(type="tool_use", id="t1", name="submit", input={})], stop_reason="tool_use"), {}

    done = []
    call = lambda name, args: (done.append(name) or {"ok": True}, False)   # noqa: E731
    trace, stop = tool_loop(Scripted(), "s", "m", [], call, lambda: bool(done), 5, "r", [])
    assert stop == "end_turn" and not done                                   # the swarm: unchanged
    trace, stop = tool_loop(Scripted(), "s", "m", [], call, lambda: bool(done), 5, "r", [],
                            on_stop=lambda reason: "continue")
    assert stop == "submitted" and done == ["submit"]


def test_single_loop_mock(tmp_path):
    out = tmp_path / "single"
    # A subprocess: field scoring is read at import (ACE_SCORE).
    p = subprocess.run([sys.executable, "-m", "ace.single_loop", "--mock", "--generations", "2",
                        "--problems", ",".join(TASKS), "--out", str(out), "--swarm-dir", str(out)],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "ACE_SCORE": "field"}, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    for g in range(2):
        recs = [json.loads((out / f"gen{g}" / "runs" / f"g{g}-{t}-big.json").read_text()) for t in TASKS]
        for t, r in zip(TASKS, recs):
            c = r["checks"]
            assert r["problem_id"] == t and not r.get("error") and r["submitted"] is not None
            assert c["base_score"] is not None and c["score"] == c["base_score"] and c["best_bonus"] == 0.0
            # its trace holds only its own tool calls
            assert {x["input"]["task"] for x in r["trace"] if x["type"] == "tool_call"} == {t}
            assert r["playbook_text"] == (out / f"gen{g}" / "context" / "big.md").read_text()
            assert (out / f"gen{g}" / "reports" / f"{r['run_id']}.json").exists()
        assert recs[0]["usage"] and not recs[1]["usage"]             # one conversation, counted once
    for t in TASKS:                                                  # gen 0 learned, gen 1 only evaluated
        assert (out / "gen0" / f"teacher_{t}.json").exists()
        assert (out / "gen0" / "reflections" / f"g0-{t}-big.json").exists()
    assert (out / "gen0" / "organizer" / "big.json").exists() and (out / "gen0" / "organizer" / "shared.json").exists()
    ctx0, ctx1 = ((out / f"gen{g}" / "context" / "big.md").read_text() for g in range(2))
    assert ctx0.startswith(SINGLE_BRIEF.strip()[:60]) and "(none yet)" in ctx0
    assert "YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task cooks_membrane" in ctx1
    assert len((out / "generations.jsonl").read_text().splitlines()) == 2
    assert (out / "summary.md").exists() and (out / "scores.png").exists()
    # the comparison (here the run against itself) has a row per generation and task
    assert (out / "compare.png").exists()
    rows = [line for line in (out / "compare.md").read_text().splitlines() if line.startswith("| 0 |") or line.startswith("| 1 |")]
    assert len(rows) == 4


def _has_openfoam() -> bool:
    try:
        from feacore.solver._foam_env import find_openfoam
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


def test_single_loop_takes_all_flow_tasks_and_refuses_a_mix(tmp_path):
    """One conversation has one tool list: flow tasks get TOOLS_FLOW (with the `task` argument), and a
    mix of flow and solid tasks is refused before anything runs."""
    import pytest

    from ace import single_loop
    from ace.agent import TOOLS_FLOW

    specs = single_loop.task_tool_specs(["poiseuille_channel", "taylor_couette_2d"])
    assert [t["name"] for t in specs] == [t["name"] for t in TOOLS_FLOW]
    mesh = next(t for t in specs if t["name"] == "mesh")["input_schema"]
    assert {"boundary_layer", "sweep", "task"} <= set(mesh["properties"]) and mesh["required"][0] == "task"
    assert single_loop.all_flow(["poiseuille_channel"]) and not single_loop.all_flow(["cooks_membrane"])
    env = {**os.environ, "ACE_SCORE": "field"}
    p = subprocess.run([sys.executable, "-m", "ace.single_loop", "--mock", "--problems", "poiseuille_channel,cooks_membrane",
                        "--generations", "1", "--out", str(tmp_path / "mix")], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=300)
    assert p.returncode != 0 and "all-solid or all-flow" in p.stderr + p.stdout
    if not _has_openfoam():
        pytest.skip("OpenFOAM not found: the flow run itself is not tried")
