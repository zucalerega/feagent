"""The Reflector: ONE LLM call per problem per generation.

Input: the task, all runs on it ranked best -> worst with their check results, condensed
traces, and the playbook bullets each run cited. Output (validated JSON): where best and worst
diverged, a root cause per failed run, 1-3 lessons, and a vote on every cited bullet.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from ace.agent import task_prompt
from ace.config import MAX_SOLVES

Section = Literal["meshing", "bcs", "geometry", "solver_setup", "postprocessing", "tool_use"]


class RootCause(BaseModel):
    run_id: str
    cause: str
    evidence: str = Field(description="run_id + turn, or a number from the checks")


class Lesson(BaseModel):
    section: Section
    content: str = Field(description='"When X, do Y, because Z." At most 40 words.')
    scope: Literal["general", "problem_class", "one_off"]
    evidence: str


class BulletVote(BaseModel):
    bullet_id: str
    vote: Literal["helpful", "harmful", "neutral"]
    reason: str


class Reflection(BaseModel):
    divergence: str = Field(description="where the best and worst runs first diverged")
    root_causes: list[RootCause]
    lessons: list[Lesson] = Field(description="1 to 3 lessons")
    bullet_votes: list[BulletVote]


SYSTEM = f"""\
You are the Reflector in a loop that improves an FEA setup agent through a playbook of short
lessons that is pasted into its system prompt. Several agent runs attempted the same problem.

How runs are scored (computed by code; treat these numbers as ground truth):
- The agent meshes the geometry (a node cap is only a safety limit), solves (at most {MAX_SOLVES}
  solves), and submits one solve; that solve is scored (its last successful solve if none).
- rel_error = |agent value - true value| / |true value| for one scored quantity.
- score = accuracy x efficiency (the team's speed/accuracy metric): accuracy = min(1, tolerance /
  rel_error); efficiency = DOFs of the cheapest run on the problem that met the tolerance / this
  run's DOFs. 1 = the cheapest mesh accurate enough; invalid runs (gate failed, no solve) score 0.
- `setup` lists differences from the reference setup (missing/unexpected BCs, sign flips,
  wrong formulation/thickness/material). The agent never saw the true value or reference setup.

Your job:
1. Compare the best and worst runs: where did they first diverge?
2. For each run that failed or scored clearly below the best, give a root cause with evidence
   (run_id + turn, or a number from the checks).
3. Write 1-3 lessons, "When X, do Y, because Z.", at most 40 words, that would have made the
   weaker runs better. Prefer lessons that transfer to other problems (scope general or
   problem_class); mark lessons that only apply to this exact problem as one_off.
   Never put this problem's true value or numeric answer in a lesson.
4. Vote on every bullet the runs cited: helpful if applying it went with better results,
   harmful if it misled the run, neutral if it made no visible difference.
If a run's reasoning relied on a remembered reference value or textbook formula to judge its
result, say so in its root cause; never turn that into a lesson.
"""


def condense_trace(trace: list[dict], max_chars: int = 12000) -> str:
    """One line per step, long payloads truncated, whole trace capped at max_chars."""
    lines = []
    for t in trace:
        if t["type"] == "text":
            body = t["text"].replace("\n", " ")[:400]
            lines.append(f"turn {t['turn']} said: {body}")
        elif t["type"] == "thinking":
            body = t["text"].replace("\n", " ")[:200]   # short: tool calls/results matter more
            lines.append(f"turn {t['turn']} reasoning (summary): {body}")
        elif t["type"] == "tool_call":
            lines.append(f"turn {t['turn']} call {t['name']}({json.dumps(t['input'])[:600]})")
        else:
            res = t["result"]
            if t["name"] == "inspect_geometry":   # long and identical across runs
                res = {"groups": [g["name"] for g in res.get("groups", [])]}
            tag = "ERROR" if t["is_error"] else "result"
            lines.append(f"turn {t['turn']} {tag} {t['name']}: {json.dumps(res)[:400]}")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars // 2] + "\n...\n" + out[-max_chars // 2:]


def build_prompt(problem_id: str, runs: list[dict], checks: dict[str, dict], playbook) -> str:
    ranked = sorted(runs, key=lambda r: (-checks[r["run_id"]]["score"], r["turns"]))
    from ace import evidence
    parts = [f"# Task given to the agent\n{task_prompt(problem_id)}\n"]
    if test := evidence.teacher_section(problem_id, runs):
        parts.append(f"# Physical test of this setup\n{test}\n")
    parts.append("# Runs, best to worst")
    for rank, run in enumerate(ranked, 1):
        # The Reflector can infer a TRAIN problem's true value (value + rel_error); that is fine
        # because held-out problems are never reflected on and lessons may not contain answers.
        c = dict(checks[run["run_id"]])
        cited = [f"[{b}] {playbook.get(b).content}" if playbook.get(b) else f"[{b}] (not in playbook)"
                 for b in run["cited"]]
        parts.append(
            f"\n## Rank {rank}: run {run['run_id']}  score={c['score']:.2f}\n"
            f"checks: {json.dumps(c)}\n"
            f"cited bullets: {cited or 'none'}\n"
            f"trace:\n{condense_trace(run['trace'])}"
        )
    return "\n".join(parts)


def reflect(problem_id: str, runs: list[dict], checks: dict[str, dict], playbook, llm, records=None):
    """Returns (Reflection or None if the output was invalid twice, usage records)."""
    prompt = build_prompt(problem_id, runs, checks, playbook)
    reflection, usage = llm.structured("reflector", SYSTEM, prompt, Reflection, records)
    if reflection is not None:
        reflection.lessons = reflection.lessons[:3]
    return reflection, usage
