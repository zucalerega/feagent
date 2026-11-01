"""Pre-teaching (run_loop --pre-teach): before generation 0 on a NEW task, the teacher revises the
context a trained swarm brings with it (run_loop --init-from).

A swarm trained on other tasks carries lessons that may mislead on the new one (bc0's refinement
of one prong, for a one-sided load, is wrong for a symmetric compression). Before any agent runs,
the teacher (Opus, role "teacher") reads the experiment analysis of the new task (its teacher
brief, physical_tests/<task>/teacher_brief.md), the task statement the agents will see, the
shared playbook and every agent's private notes, persona and last run, and writes:

  shared      the revised shared playbook (general meshing lessons kept, lessons specific to the
              training tasks dropped or rewritten, guidance for the new task added), a short
              summary, the focus regions of the new task (shown to every agent where the
              contested regions of a task go) and the base mesh call every agent's first
              experiment starts from
  per agent   its revised private notes, a briefing (what is different here for it), its first
              mesh call, and its first experiment in the role allocate() gives it (control /
              variant / explorer; generation 0 has no ranking, so the order of --init-agents,
              the training rank, decides)

Like the deep teacher, this is one call for the shared playbook and one per agent over one cached
prefix; the shared call goes first (its base mesh call goes into every per-agent call), then the
per-agent calls in parallel. apply() files the result into memory (Memory.replace_notes) and into the
loop's --deep state, and writes <out>/pre_teach/: before.json, after.json, teacher.json, diff.md.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field

from ace import process_brief
from ace.deep_teacher import (NL_SCORING, PROCESS_BRIEF, Assignment, ContestedRegion, MeshCall, _task_statement,
                              allocate, is_nonlinear, teacher_brief)
from ace.llm import text_block
from ace.memory import Memory, notes_as_dicts
from ace.organizer import PRIVATE_MAX, SHARED_MAX, OrganizedNote, _line, clean_notes


# --------------------------------------------------------------------------- output schema


class RevisedShared(BaseModel):
    summary: str = Field(description="what you changed in the shared playbook and why, in a few sentences")
    notes: list[OrganizedNote] = Field(description="the new full shared playbook, most important first")
    focus_regions: list[ContestedRegion] = Field(description="the regions of the part that decide the score "
                                                             "on the new task, most important first")
    base_mesh_call: MeshCall = Field(description="the base recipe for this task built from the brief: symmetric "
                                                 "refinement of BOTH prongs' hex-pocket corners on BOTH z sides "
                                                 "(+-6.6 mm), plus the bore tension bands; global h for the bulk; "
                                                 "all sizes in metres")


class RevisedAgent(BaseModel):
    briefing: str = Field(description="to this agent (second person): what is different on the new task, which "
                                      "of its own lessons carry over, which would mislead it here and why")
    notes: list[OrganizedNote] = Field(description="its new full private notes, most important first")
    first_mesh_call: MeshCall = Field(description="the exact first mesh call it should try on the new task")
    assignment: Assignment = Field(description="its first experiment on the new task, in its role")


# --------------------------------------------------------------------------- the prompt

SYSTEM = f"""\
You are the TEACHER: an expert in finite element analysis and adaptive meshing, mentoring a swarm of
FEA meshing agents. The swarm was trained for several generations on other tasks and now moves to
a NEW task. Before any agent runs on it, you revise what every agent will read: the shared
playbook (all agents) and each agent's private notes. You have the analysis of the new task's
experiment (the teacher brief): use it, it is the evidence the agents do not have yet.

{PROCESS_BRIEF}

How to revise a note list (the same rules as the organizer that keeps the notes later):
1. KEEP general lessons that hold on the new task (tool use, how to compare solves, grading,
   element order, DOF economy). Return a kept note with its id and content null (unchanged).
2. DROP or REWRITE lessons specific to the training tasks that would mislead on the new one (other
   loads, other supports, other hotspots, recipes sized for another score). A training-task note
   that is still true and harmless may stay under its own task tag; never let one contradict the
   brief.
3. ADD guidance for the new task from the brief and its scoring: where the score is decided, what
   to refine and how to judge a mesh there. Never invent numbers that are not in the material.
4. Every note: WHERE (face group, edge, point or region) + WHAT (size, order, radius) + EXPECTED
   effect, <= 50 words, tagged with its task (the new task's id, a training task's id, or "general").
5. Keep ids for notes you keep or edit (one output note per id); new notes have id null. Most
   important first; at most the number of notes you are told.
Write densely; every sentence carries information from the inputs.
Mesh calls (h, size, radius, point, box) are in METRES (uniform 4 mm = h 0.004; x 12-31 mm = box x
0.012..0.031); the brief and the nonlinear hotspots (xyz_mm) are in mm.
"""

SHARED_JOB = """\
YOUR JOB NOW: the revised SHARED PLAYBOOK for task {task} (all agents read it). Return:
- notes: the new full shared playbook (at most {max_notes} notes), following the rules.
- summary: what you kept, dropped, rewrote and added, and why.
- focus_regions: the regions that decide the score on {task} (from the brief), most important
  first: region (with its face groups), why it matters, what to do there (sizes, grading).
- base_mesh_call: the base recipe for {task} from the brief, the one every agent's first experiment
  starts from (control runs it, variants change one factor of it, explorers depart from it).
Agents' private notes and first experiments are written in separate calls: do not write per-agent
advice here."""

AGENT_JOB = """\
YOUR JOB NOW: agent {agent} only (its notes, persona and last run are above). Return:
- notes: its new full private notes (at most {max_notes}), following the rules. Keep what is its
  own (its style, what it learned works for it) where it still holds; the shared playbook is
  revised separately in the same spirit, so do not just copy shared notes.
- briefing: to {agent} directly: what is different on {task}, which of its lessons carry over and
  which would mislead it here.
- first_mesh_call: the exact first mesh call (h, order, refinements) it should try on {task}.
- assignment: its first experiment on {task}.
FIRST EXPERIMENTS ON {task} (roles fixed in advance from the swarm's training rank, so that the agents
try DIFFERENT things in generation 0; every per-agent call sees this same table):
{table}
YOUR AGENT'S ROLE: {role}
BASE RECIPE (identical for every agent): {base}
control: run the base recipe with at most one small tweak.
variant: the base recipe with exactly ONE factor changed, the one named in the table; state the new
value. explorer: a structurally different mesh in the direction named in the table, chosen from
the brief. first_mesh_call must implement the assignment exactly.
Mesh calls (h, size, radius, point, box) are in METRES (uniform 4 mm = h 0.004; x 12-31 mm = box x
0.012..0.031); the brief and the nonlinear hotspots (xyz_mm) are in mm."""


def _owner_block(name: str, notes: list[dict]) -> str:
    return f"{name} ({len(notes)} notes):\n" + ("\n".join(_line(n, True) for n in notes) or "(none)")


def build_prefix(task: str, mem: Memory, agents: list[str], persona_of: dict, last_run: dict,
                 origin: str) -> list[dict]:
    """The user-message prefix every pre-teach call shares (last block marked for caching)."""
    head = [f"TASK {task} (the NEW task)",
            f"--- where the swarm comes from ---\n{origin}",
            f"--- problem statement as the agents will see it ---\n{_task_statement(task)}",
            f"--- the experiment analysis (teacher brief) ---\n{teacher_brief(task) or '(none)'}"]
    if is_nonlinear(task):
        head.append(f"--- how this task is scored ---\n{NL_SCORING}")
    blocks = [text_block("\n\n".join(head) + "\n"),
              text_block("\n=== SHARED PLAYBOOK (one JSON object per note) ===\n"
                         + _owner_block("shared playbook", notes_as_dicts(mem.shared)) + "\n")]
    for a in agents:
        k = int(a[1:])
        blocks.append(text_block(
            f"\n=== AGENT {a} ===\npersona: {persona_of.get(k) or '(none)'}\n"
            f"its last run (on a training task):\n{process_brief.run_summary_text(last_run.get(a))}\n"
            + _owner_block(f"{a}'s private notes", notes_as_dicts(mem.private[k])) + "\n"))
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}   # end of the shared prefix
    return blocks


# --------------------------------------------------------------------------- the calls


def ask(task: str, mem: Memory, persona_of: dict, last_run: dict, origin: str, llm, records: list) -> dict:
    """The teacher's revision: one call for the shared playbook (its base_mesh_call is every agent's
    BASE RECIPE), then one per agent, in parallel, over one cached prefix.
    Returns {"roles": [(agent, role, focus)], "shared": RevisedShared dict or None,
    "agents": {agent: RevisedAgent dict or None}}."""
    agents = [f"a{k}" for k in range(len(mem.private))]
    roles = allocate(agents)                        # a0 = best-ranked in training (--init-agents order)
    role_of = {a: role for a, role, _ in roles}
    table = "\n".join(f"  {a}: {role}: {focus}" for a, role, focus in roles) or "  (one agent: no roles)"
    prefix = build_prefix(task, mem, agents, persona_of, last_run, origin)

    def agent_user(a: str, base: str) -> list[dict]:
        return prefix + [text_block(f"FOR AGENT {a}\n" + AGENT_JOB.format(
            agent=a, task=task, max_notes=PRIVATE_MAX, table=table, role=role_of.get(a, "explorer"), base=base))]

    def call(user, schema):
        from ace.mock import MockLLM
        if isinstance(llm, MockLLM):                # the mock has no canned answer for these schemas
            return _mock(schema, user, mem)
        try:
            return llm.structured("teacher", SYSTEM, user, schema, records)[0]
        except Exception as e:  # noqa: BLE001  one failed call must not lose the others
            print(f"  pre-teach: a {schema.__name__} call failed: {e}")
            return None

    # the shared call first: its base mesh call is the one base every agent's experiment is relative to
    shared = call(prefix + [text_block(SHARED_JOB.format(task=task, max_notes=SHARED_MAX))], RevisedShared)
    base = (process_brief.mesh_call_text(shared.base_mesh_call.model_dump()) if shared is not None else
            f"(the shared call failed) the control agent's ({roles[0][0] if roles else 'a0'}) last submitted "
            f"mesh call above, adapted to {task} as the brief says")
    with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as pool:
        if len(agents) > 1:   # as in deep_teacher.teach: write the cache once, then the calls read it
            llm.warm_cache("teacher", SYSTEM, agent_user(agents[0], base), RevisedAgent, records)
        per = {a: pool.submit(call, agent_user(a, base), RevisedAgent) for a in agents}
        out = {"roles": roles, "shared": shared, "agents": {a: f.result() for a, f in per.items()}}
    for a, rev in out["agents"].items():           # the role allocate() gave, whatever the model wrote
        if rev is not None:
            rev.assignment.agent, rev.assignment.role = a, role_of.get(a, rev.assignment.role)
    out["shared"] = out["shared"].model_dump() if out["shared"] is not None else None
    out["agents"] = {a: rev.model_dump() if rev is not None else None for a, rev in out["agents"].items()}
    return out


def _mock(schema, user: list[dict], mem: Memory):
    """--mock: keep every note, add one, and give each agent a role-shaped experiment (no API call)."""
    text = "\n".join(b["text"] for b in user if b.get("type") == "text")
    new = {"id": None, "task": "general", "section": "meshing", "kind": "idea", "source": "teacher",
           "content": "mock pre-teach note: refine both prongs' pocket corners symmetrically"}
    if schema is RevisedShared:
        keep = [{"id": n["id"], "task": "general", "section": n["section"], "kind": n["kind"] or "keep",
                 "source": n["source"] or "mock", "content": None} for n in notes_as_dicts(mem.shared)]
        return RevisedShared.model_validate({"summary": "mock: kept every note, added one", "notes": [new] + keep,
                                             "focus_regions": [{"region": "mock: pocket corners f8/f38/f41",
                                                                "why_contested": "mock", "what_to_do": "mock"}],
                                             "base_mesh_call": {"h": 0.004, "order": 2, "refinements": [
                                                 {"box": [0.012, -0.01, 0.0, 0.031, 0.01, 0.0066], "size": 0.001,
                                                  "radius": 0.003}]}})
    a = text.split("FOR AGENT ", 1)[1].split()[0]
    role = text.split("YOUR AGENT'S ROLE: ", 1)[1].split()[0]
    keep = [{"id": n["id"], "task": "general", "section": n["section"], "kind": n["kind"] or "keep",
             "source": n["source"] or "mock", "content": None} for n in notes_as_dicts(mem.private[int(a[1:])])]
    return RevisedAgent.model_validate({
        "briefing": f"mock briefing for {a}", "notes": [new] + keep,
        "first_mesh_call": {"h": 0.006, "order": 2, "refinements": []},
        "assignment": {"agent": a, "role": role, "experiment": f"mock {role} experiment for {a}", "why": "mock"}})


# --------------------------------------------------------------------------- applying it


def _retag(notes: list[dict]) -> list[dict]:
    """A note moved to another task (or new) gets the source "pre-teach@<task>", so the context
    groups it under that task (process_brief.note_task reads the task from the source tags), and
    loses its old "[task] " tag from the text ("[new] [old] text" -> "[new] text")."""
    for n in notes:
        if m := re.match(r"\[([^\]]+)\] \[([^\]]+)\] ", n["content"]):
            n["content"] = f"[{m.group(1)}] " + n["content"][m.end():]
        if process_brief.note_task(SimpleNamespace(source=n.get("source"))) != n["task"]:
            n["source"] = f"pre-teach@{n['task']}"
    return notes


def apply(result: dict, task: str, mem: Memory, deep: dict, out_dir: Path) -> dict:
    """File the teacher's revision into memory and into run_loop's --deep state (deep["feedback"]
    per agent: the briefing, first mesh call and assignment the generation-0 context shows;
    deep["teacher"][task]: the summary and focus regions), and write before/after/diff to out_dir.
    A call that gave no output leaves that note list as it was. Returns the changes per owner."""
    out_dir.mkdir(parents=True, exist_ok=True)
    owners = {"shared": (mem.shared, result.get("shared"), SHARED_MAX)}
    owners.update({a: (mem.private[int(a[1:])], rev, PRIVATE_MAX) for a, rev in result["agents"].items()})
    before = {o: notes_as_dicts(pb) for o, (pb, _, _) in owners.items()}
    changes = {}
    for o, (pb, rev, max_notes) in owners.items():
        if rev is None:
            print(f"  pre-teach: no revision for {o}; its notes stay as they were")
            continue
        notes = clean_notes([OrganizedNote.model_validate(n) for n in rev["notes"]], before[o], max_notes)
        changes[o] = Memory.replace_notes(pb, _retag(notes), 0)
    after = {o: notes_as_dicts(pb) for o, (pb, _, _) in owners.items()}

    # what the generation-0 contexts show (ace.process_brief.deep_context)
    for a, rev in result["agents"].items():
        if rev is not None:
            deep["reflection"].pop(a, None)        # its reflection on a training-task run would mislead here
            deep["feedback"][a] = {
                "agent": a, "run_id": "pre-teach", "task": task,
                "diagnosis": f"Before your first run on {task} (the teacher's briefing): {rev['briefing']}",
                "step_critique": [], "mistakes": [], "strengths": [], "notes": [],
                "next_mesh_call": rev["first_mesh_call"], "assignment": rev["assignment"]}
    if result.get("shared") is not None:
        deep["teacher"][task] = {"task": task, "task_summary": result["shared"]["summary"], "setup_warnings": [],
                                 "contested_regions": [{**r, "which_agents_differ": [
                                     "(teacher's focus from the experiment analysis, before any run)"]}
                                     for r in result["shared"]["focus_regions"]], "highlights": [],
                                 "shared_lessons": [], "per_agent": []}

    # base_mesh_call: before, the control agent's last (training-task) call; after, the teacher's base recipe
    control = result["roles"][0][0] if result.get("roles") else "a0"
    old_base = (deep["last_run"].get(control) or {}).get("mesh_call")
    new_base = (result.get("shared") or {}).get("base_mesh_call")
    (out_dir / "before.json").write_text(json.dumps({**before, "base_mesh_call": old_base}, indent=1))
    (out_dir / "after.json").write_text(json.dumps({**after, "base_mesh_call": new_base}, indent=1))
    (out_dir / "teacher.json").write_text(json.dumps(result, indent=1, default=str))
    (out_dir / "diff.md").write_text(diff_markdown(task, result, before, changes))
    return changes


def diff_markdown(task: str, result: dict, before: dict, changes: dict) -> str:
    """The revision as a readable page: per owner what was removed, edited (old -> new), added."""
    old_text = {n["id"]: n["content"] for notes in before.values() for n in notes}
    L = [f"# Pre-teach for {task}", ""]
    for o, ch in changes.items():
        L.append(f"## {'Shared playbook' if o == 'shared' else 'Agent ' + o}")
        rev = result["shared"] if o == "shared" else result["agents"][o]
        if o == "shared":
            L += ["", f"Summary: {rev['summary']}", "",
                  f"Base mesh call: `{process_brief.mesh_call_text(rev['base_mesh_call'])}`", "", "Focus regions:"]
            L += [f"- {r['region']}: {r['why_contested']} What to do: {r['what_to_do']}" for r in rev["focus_regions"]]
        else:
            asg = rev["assignment"]
            L += ["", f"Briefing: {rev['briefing']}", "",
                  f"First mesh call: `{process_brief.mesh_call_text(rev['first_mesh_call'])}`", "",
                  f"Assignment ({asg['role']}): {asg['experiment']}  ", f"Why: {asg['why']}"]
        L += ["", f"{len(before[o])} notes before; {len(ch['kept'])} kept, {len(ch['edited'])} edited, "
                  f"{len(ch['removed'])} removed, {len(ch['added'])} added", ""]
        L += ["### Removed"] + [f"- `{i}` {c}" for i, c in ch["removed"]] + [""]
        L += ["### Edited"] + [f"- `{i}`\n  - before: {old_text.get(i)}\n  - after: {c}" for i, c in ch["edited"]] + [""]
        L += ["### Added"] + [f"- `{i}` {c}" for i, c in ch["added"]] + [""]
    return "\n".join(L)
