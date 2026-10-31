"""The Curator: turns Reflector output into delta operations. It PROPOSES; ace.merge applies.

  ADD   a new bullet (section, content)
  VOTE  helpful/harmful +1 on an existing bullet ID

Bullet votes map straight to VOTE ops (no LLM needed). For lessons, one LLM call per generation
sees the existing bullets in the lessons' sections and says, per lesson, ADD (new idea) or
DUPLICATE of an existing ID (which becomes a helpful VOTE on it). one_off lessons are dropped.
If none of the lessons' sections has any bullets yet, there is nothing to compare against and
the LLM call is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


@dataclass
class Op:
    kind: str                       # "ADD" | "VOTE"
    section: str = ""
    content: str = ""
    bullet_id: str = ""
    vote: str = ""                  # "helpful" | "harmful"
    source_runs: list[str] = field(default_factory=list)


class Decision(BaseModel):
    lesson_index: int
    action: Literal["ADD", "DUPLICATE"]
    duplicate_of: str | None = None
    content: str                    # final wording (may tighten the lesson)


class CuratorOutput(BaseModel):
    decisions: list[Decision]


SYSTEM = """\
You are the Curator of a playbook of short lessons for an FEA setup agent. You receive new
candidate lessons and the existing bullets in the same sections. For each candidate decide:
- DUPLICATE (give duplicate_of = the existing bullet ID) if an existing bullet already says
  essentially the same thing, or
- ADD if it is a genuinely new idea.
For ADD you may tighten the wording, keeping the "When X, do Y, because Z." form and at most
40 words. Do not merge two candidates into one; return exactly one decision per candidate.
"""


def curate(reflections: list[tuple], playbook, llm) -> tuple[list[Op], list[dict]]:
    """reflections: [(problem_id, Reflection, run_ids)]. Returns (ops, usage records)."""
    ops: list[Op] = []
    candidates: list[tuple] = []        # (lesson, run_ids)
    for _pid, refl, run_ids in reflections:
        for v in refl.bullet_votes:
            if v.vote != "neutral" and playbook.get(v.bullet_id):
                ops.append(Op("VOTE", bullet_id=v.bullet_id, vote=v.vote, source_runs=run_ids))
        candidates += [(lesson, run_ids) for lesson in refl.lessons if lesson.scope != "one_off"]

    if not candidates:
        return ops, []
    sections = sorted({lesson.section for lesson, _ in candidates})
    existing = [b for s in sections for b in playbook.in_section(s)]
    if not existing:
        ops += [Op("ADD", section=l.section, content=l.content, source_runs=r) for l, r in candidates]
        return ops, []

    prompt = "# Existing bullets\n" + "\n".join(f"[{b.id}] ({b.section}) {b.content}" for b in existing)
    prompt += "\n\n# Candidate lessons\n" + "\n".join(
        f"{i}. ({l.section}) {l.content}" for i, (l, _) in enumerate(candidates))
    out, usage = llm.structured("curator", SYSTEM, prompt, CuratorOutput)

    decided = {d.lesson_index: d for d in out.decisions} if out else {}
    for i, (lesson, run_ids) in enumerate(candidates):
        d = decided.get(i)
        if d and d.action == "DUPLICATE" and d.duplicate_of and playbook.get(d.duplicate_of):
            ops.append(Op("VOTE", bullet_id=d.duplicate_of, vote="helpful", source_runs=run_ids))
        else:  # ADD, or the curator failed / skipped this one: fall back to adding it as-is
            content = d.content if d and d.action == "ADD" else lesson.content
            ops.append(Op("ADD", section=lesson.section, content=content, source_runs=run_ids))
    return ops, usage
