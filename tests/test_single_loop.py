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
