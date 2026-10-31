"""The organizer (--deep): files all new material into an agent's notes (or the shared playbook).

It replaces the curator's gatekeeping (one lesson -> ADD or duplicate-of) with organizing: it reads
the current notes and ALL incoming items (the teacher's notes, the agent's self-reflection, peers'
highlights, shared lessons), merges duplicates, rewrites vague notes into specific ones, resolves
contradictions in favour of the newest evidence, and returns the new full list ordered by
importance. It keeps every useful specific idea; it drops only what is superseded or contradicted.

Notes are plain dicts here (Memory.replace_notes turns the result into bullets):
  current  {id, section, content, kind, source, gen}
  incoming {section, content, kind, source}     source: "teacher", "self", "peer:a2", "shared-lesson", ...
  result   {id (the kept/edited note's id, or None for a new note), section, content, kind, source}
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ace.deep_teacher import PROCESS_BRIEF, Kind
from ace.reflector import Section

PRIVATE_MAX = 45
SHARED_MAX = 40


class OrganizedNote(BaseModel):
    id: str | None = Field(description="the id of the current note this keeps or edits; null for a new note")
    section: Section
    content: str = Field(description="specific and actionable, <= 60 words")
    kind: Kind
    source: str = Field(description="where it came from (e.g. teacher, self, peer:a2); merged notes join "
                                    "their sources with '+'")


class OrganizerOutput(BaseModel):
    notes: list[OrganizedNote] = Field(description="the new full list, most important first")


SYSTEM = f"""\
You are the ORGANIZER of an FEA meshing agent's notes. You are an organizer, not a gatekeeper: the
incoming material is fresh evidence from the latest generation (a teacher who saw every agent's
whole mesh and error map, the agent's own reflection on its results, peers' best ideas), and your
job is to integrate ALL of it into one clean, well-ordered set of notes.

{PROCESS_BRIEF}

Rules:
1. Integrate every incoming item: each must end up as its own note or merged into another.
2. Merge duplicates and near-duplicates into one note that keeps the most specific details of each
   (locations, sizes, orders, radii, numbers).
3. Rewrite vague notes into specific ones using the evidence in the other notes (which group or
   point, what size or order, what effect). Never invent facts or numbers that appear nowhere in
   the material.
4. When notes contradict, keep the one backed by the newest evidence (incoming beats current) and
   drop or correct the older one.
5. Drop a current note only when it is superseded (merged into a better note) or contradicted.
6. Keep ids: a note that keeps or edits a current note carries that note's id (one output note per
   id; when you merge several current notes, keep the id of the main one). New notes have id null.
7. Order by importance (what most changes the next score first: setup fixes, then where to put
   elements, then ideas to test), grouped by section within that order. Keep notes <= 60 words.
8. Return at most the number of notes you are told; if there is more material, merge harder.
"""


def _line(note: dict, with_id: bool) -> str:
    """One note as a one-line JSON object (precise for the model, and parseable by the mock)."""
    keys = ("id", "section", "kind", "source", "content") if with_id else ("section", "kind", "source", "content")
    return json.dumps({k: note.get(k) for k in keys})


def build_user(owner: str, current: list[dict], incoming: list[dict], max_notes: int) -> str:
    cur = "\n".join(_line(n, True) for n in current) or "(none)"
    new = "\n".join(_line(n, False) for n in incoming) or "(none)"
    return (f"NOTES OF: {owner}\nAt most {max_notes} notes in the result.\n\n"
            f"CURRENT NOTES ({len(current)}):\n{cur}\n\n"
            f"INCOMING ({len(incoming)}, newest evidence):\n{new}\n\n"
            "Return the new full list of notes.")


def organize(owner: str, current: list[dict], incoming: list[dict], llm, records: list,
             max_notes: int) -> list[dict] | None:
    """The organized notes, or None if the model's output did not validate twice."""
    out, _ = llm.structured("organizer", SYSTEM, build_user(owner, current, incoming, max_notes),
                            OrganizerOutput, records)
    if out is None:
        return None
    known, seen, result = {n["id"] for n in current}, set(), []
    for n in out.notes[:max_notes]:
        d = n.model_dump()
        # An id the model made up, or a second note claiming the same id, becomes a new note
        # (so replace_notes never edits the wrong bullet or one bullet twice).
        if d["id"] not in known or d["id"] in seen:
            d["id"] = None
        seen.add(d["id"])
        result.append(d)
    return result


def organize_private(agent: str, current: list[dict], incoming: list[dict], llm, records: list) -> list[dict] | None:
    return organize(agent, current, incoming, llm, records, max_notes=PRIVATE_MAX)


def organize_shared(current: list[dict], incoming: list[dict], llm, records: list) -> list[dict] | None:
    return organize("shared playbook (all agents)", current, incoming, llm, records, max_notes=SHARED_MAX)
