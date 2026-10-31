"""The Curator: turns Reflector output into delta operations. It PROPOSES; ace.merge applies.

  ADD   a new bullet (section, content)
  VOTE  helpful/harmful +1 on an existing bullet ID

Bullet votes map straight to VOTE ops (no LLM needed). For lessons, one LLM call per generation
sees the existing bullets in the lessons' sections plus all candidate lessons, and says, per
lesson, ADD (new idea), DUPLICATE of an existing ID (becomes a helpful VOTE on it), or DUPLICATE
of an earlier candidate (dropped). one_off lessons are dropped. The LLM call is skipped only when
there is nothing to compare: no existing bullets in those sections and no two candidates in the
same section.
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
    duplicate_of: str | None = None  # an existing bullet ID, or "#<i>" for an earlier candidate
    content: str                    # final wording (may tighten the lesson)


class CuratorOutput(BaseModel):
    decisions: list[Decision]


SYSTEM = """\
You are the Curator of a playbook of short lessons for an FEA setup agent. You receive new
candidate lessons (numbered) and the existing bullets in the same sections. For each candidate,
in order, decide:
- DUPLICATE with duplicate_of = an existing bullet ID, if that bullet already says essentially
  the same thing;
- DUPLICATE with duplicate_of = "#<number>" of an EARLIER candidate, if that candidate already
  says essentially the same thing; or
- ADD if it is a genuinely new idea.
For ADD you may tighten the wording, keeping the "When X, do Y, because Z." form and at most
40 words. Return exactly one decision per candidate.
"""


def curate(reflections: list[tuple], playbook, llm, records=None) -> tuple[list[Op], list[dict]]:
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
    sections = [lesson.section for lesson, _ in candidates]
    existing = [b for s in sorted(set(sections)) for b in playbook.in_section(s)]
    if not existing and len(set(sections)) == len(sections):   # nothing to compare against
        ops += [Op("ADD", section=les.section, content=les.content, source_runs=r) for les, r in candidates]
        return ops, []

    prompt = "# Existing bullets\n" + ("\n".join(f"[{b.id}] ({b.section}) {b.content}" for b in existing)
                                      or "(none in these sections)")
    prompt += "\n\n# Candidate lessons\n" + "\n".join(
        f"{i}. ({les.section}) {les.content}" for i, (les, _) in enumerate(candidates))
    out, usage = llm.structured("curator", SYSTEM, prompt, CuratorOutput, records)

    decided = {d.lesson_index: d for d in out.decisions} if out else {}
    for i, (lesson, run_ids) in enumerate(candidates):
        d = decided.get(i)
        dup = d.duplicate_of if d and d.action == "DUPLICATE" else None
        if dup and playbook.get(dup):
            ops.append(Op("VOTE", bullet_id=dup, vote="helpful", source_runs=run_ids))
        elif dup and dup.startswith("#") and dup[1:].isdigit() and int(dup[1:]) < i:
            continue                     # same idea as an earlier candidate: drop it
        else:  # ADD, or the curator failed / skipped this one: fall back to adding it as-is
            content = d.content if d and d.action == "ADD" and d.content else lesson.content
            ops.append(Op("ADD", section=lesson.section, content=content, source_runs=run_ids))
    return ops, usage
