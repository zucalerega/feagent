"""run_loop --personas for SOME agents (default off). Mock only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ace.process_brief import deep_context
from ace.run_loop import PERSONAS, persona_agents

REPO = Path(__file__).resolve().parents[1]
HEADER = "YOUR EXPLORATION STYLE"


def test_persona_agents():
    assert persona_agents(None, None, 8) == []                     # default: nobody
    assert persona_agents("all", None, 4) == [0, 1, 2, 3]          # --personas alone: everybody
    assert persona_agents("a5,a0, a3", None, 8) == [0, 3, 5]
    assert persona_agents(None, 3, 8) == [0, 1, 2]
    with pytest.raises(SystemExit):
        persona_agents("a9", None, 8)
    assert len(set(PERSONAS)) >= 8


def test_deep_context_persona_only_when_given():
    plain = deep_context("a1", "t", "S", "P", None, None, [], "")
    styled = deep_context("a1", "t", "S", "P", None, None, [], "", persona="STYLE X")
    assert HEADER not in plain and HEADER in styled
    assert styled.index("STYLE X") > styled.index("YOUR NOTES") and styled.index("STYLE X") < styled.index("THIS GENERATION")


def test_deep_loop_with_personas_for_some_agents(tmp_path):
    """Also the check that the --deep swarm still runs end to end (one learning generation)."""
    out = tmp_path / "deep"
    args = ["--mock", "--deep", "--split", "--problems", "cooks_membrane,nafems_le1", "--holdout", "",
            "--memory", "shared+private", "--agents", "4", "--concurrency", "4", "--generations", "2",
            "--personas", "a0,a3", "--out", str(out)]
    p = subprocess.run([sys.executable, "-m", "ace.run_loop", *args], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "ACE_SCORE": "field"}, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert p.stdout.count("context: private") == 1                  # gen 0 learns, gen 1 is evaluated
    assert json.loads((out / "personas.json").read_text()) == {"a0": PERSONAS[0], "a3": PERSONAS[1]}
    for g in range(2):
        ctx = {a: (out / f"gen{g}" / "context" / f"{a}.md").read_text() for a in ("a0", "a1", "a2", "a3")}
        assert PERSONAS[0] in ctx["a0"] and PERSONAS[1] in ctx["a3"]
        assert HEADER not in ctx["a1"] and HEADER not in ctx["a2"]
