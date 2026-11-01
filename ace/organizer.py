"""The organizer (--deep): files all new material into an agent's notes (or the shared playbook).

It replaces the curator's gatekeeping (one lesson -> ADD or duplicate-of) with organizing: it reads
the current notes and ALL incoming items (the teacher's notes, the agent's self-reflection, peers'
highlights, shared lessons), merges duplicates, rewrites vague notes into specific ones, resolves
contradictions in favour of the newest evidence, and returns the new full list ordered by
importance. It keeps every useful specific idea; it drops only what is superseded or contradicted.

Every note carries a task tag (a task id, or "general"). Notes for different tasks never contradict
each other (both are kept, each under its task). Agents switch tasks between generations, so the tag
is also written at the start of the content ("[task] ...") until memory stores it as a field.

Notes are plain dicts here (Memory.replace_notes turns the result into bullets):
  current  {id, section, content, kind, source, gen, task?}
  incoming {section, content, kind, source, task?}   source: "teacher", "self", "peer:a2", "shared-lesson", ...
  result   {id (the kept/edited note's id, or None for a new note), section, content, kind, source, task}
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
    task: str = Field(default="general", description="the task id the note is about, or 'general'")
    section: Section
    content: str | None = Field(description="specific and actionable, <= 50 words, without the [task] prefix; "
                                            "null = keep the current note's text unchanged (only with an id)")
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
4. Every note has a task: a task id or "general". Notes for DIFFERENT tasks never contradict each
   other: keep both, each tagged with its task. Within one task (or two general notes), when notes
   contradict, keep the one backed by the newest evidence (incoming beats current) and drop or
   correct the older one. A current note without a task tag: infer it from its content ("[task]"
   prefix or the task it names), else "general".
5. Drop a current note only when it is superseded (merged into a better note) or contradicted.
6. Keep ids: a note that keeps or edits a current note carries that note's id (one output note per
   id; when you merge several current notes, keep the id of the main one). New notes have id null.
7. Order: general notes first, then each task's notes together under that task; within each,
   by importance (setup fixes, then where to put elements, then ideas to test). Keep notes <= 50 words.
8. Return at most the number of notes you are told; if there is more material, merge harder.
9. Be compact (you are on the critical path): for a current note you keep unchanged, return its id,
   task, section, kind, source and content null. Prefer short rewrites over long ones.
"""


def _line(note: dict, with_id: bool) -> str:
    """One note as a one-line JSON object (precise for the model, and parseable by the mock)."""
    keys = ("id", "task", "section", "kind", "source", "content") if with_id else ("task", "section", "kind", "source", "content")
    return json.dumps({k: note.get(k) for k in keys if k != "task" or note.get("task")})


def _untag(content: str, task: str) -> str:
    p = f"[{task}] "
    return content[len(p):] if content.startswith(p) else content


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
    return clean_notes(out.notes, current, max_notes)


def clean_notes(notes: list[OrganizedNote], current: list[dict], max_notes: int) -> list[dict]:
    """A model's new note list made safe for Memory.replace_notes: unknown or repeated ids become new
    notes, "unchanged" (content null) takes the current text, and the task tag goes into the text.
    Also used by ace/pre_teach.py."""
    known, seen, result = {n["id"]: n for n in current}, set(), []
    for n in notes[:max_notes]:
        d = n.model_dump()
        # An id the model made up, or a second note claiming the same id, becomes a new note
        # (so replace_notes never edits the wrong bullet or one bullet twice).
        if d["id"] not in known or d["id"] in seen:
            d["id"] = None
        if d["content"] is None:            # "unchanged": the current text (a new note needs text)
            if d["id"] is None:
                continue
            d["content"] = known[d["id"]]["content"]
        seen.add(d["id"])
        task = d["task"] = (d["task"] or "general").strip()
        # The tag in the text, once (memory stores only the text for now).
        d["content"] = f"[{task}] " + _untag(d["content"], task)
        result.append(d)
    return result


def organize_private(agent: str, current: list[dict], incoming: list[dict], llm, records: list) -> list[dict] | None:
    return organize(agent, current, incoming, llm, records, max_notes=PRIVATE_MAX)


def organize_shared(current: list[dict], incoming: list[dict], llm, records: list) -> list[dict] | None:
    return organize("shared playbook (all agents)", current, incoming, llm, records, max_notes=SHARED_MAX)
