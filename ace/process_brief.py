"""--deep mode: the process brief every LLM in the loop reads, and the context block an agent sees.

PROCESS_BRIEF explains the whole system (generations, scoring, teacher, self-reflection, organizer)
to every role, so each knows what its output is used for. deep_context builds the block pasted into
a generator agent's system prompt (ace.agent.system_prompt) in deep mode: feedback on its last run
first, notes last, then an instruction to change something deliberately.
"""

from __future__ import annotations

import re

from ace.config import SECTIONS
from ace.playbook import Playbook

PROCESS_BRIEF = """\
## HOW THIS SYSTEM WORKS (read by every agent, teacher, reflector and organizer)

A population of persistent meshing agents (a0, a1, ...) learns over many generations to mesh FEA
problems well. Every generation the agents are split at random across the training tasks; each
agent inspects the geometry, meshes it (global size h, element order, local refinements at named
groups or points), solves within a small solve budget and submits one solve.

Scoring. The submitted solution is compared with a very fine reference solution: the error is the
larger of (a) the energy-norm stress error over the WHOLE part and (b) the RMS von Mises error at a
set of check points spread over the part (a few are disclosed to the agent, the rest are hidden).
The yardstick is an automesher (gmsh default sizes, quadratic elements). The target is half the
automesher's error: below that target the score falls quadratically, above it beating it keeps
paying. DOFs cost: the score is divided by DOFs relative to the automesher, so extra elements must
buy at least proportional accuracy. On each hidden check point, the single run with the smallest
error among that generation's runs on the task earns a bonus; a tie pays nobody, so copying a
peer's mesh exactly earns no bonus.

Learning. After each generation a teacher looks at every run on a task: the whole mesh, where the
error is (error maps and region grids), the setup and the agent's reasoning. It tells each agent in
depth what went wrong and exactly what to change, and tells everyone what each agent did really
well. Each agent also looks at its own final results and whole mesh and writes notes for itself.
An organizer then merges, rewrites and orders each agent's notes and the shared playbook: it keeps
every useful specific idea and drops only what newer evidence superseded or contradicted.

Notes are tagged by task (source "...@<task>" or "@general"): another task's note is a hint to
test. Address every point: the agent answers every item of its teacher feedback in its first
message, and the teacher checks next generation that each promise was kept and flags repeats.

Rules of the game. Notes are earned evidence from earlier runs: respect them, but test them, and
report when one fails. Every generation each agent must deliberately change at least one thing
compared with its last run, and say what and why. Use one solve as a controlled experiment (change
one thing, compare). Specific beats generic: name groups, points, sizes, radii and orders.
"""


def mesh_call_text(call: dict | None) -> str:
    """A MeshCall (ace.deep_teacher) as the agent's own tool call reads: mesh(h=..., order=..., refinements=[...])."""
    if not call:
        return "(none)"
    refs = ", ".join(
        "{" + ", ".join(f"{k}: {v}" for k, v in (("group", r.get("group")), ("point", r.get("point")),
                                                 ("size", r.get("size")), ("radius", r.get("radius")))
                        if v is not None) + "}"
        for r in call.get("refinements") or [])
    return f"mesh(h={call.get('h')}, order={call.get('order')}, refinements=[{refs}])"


def render_notes(playbook: Playbook) -> str:
    """Every note, grouped by section, in the organizer's order (most important first); each line
    carries the note's id (so the agent can cite it) and its kind. Empty -> "(empty)"."""
    lines = []
    for section in SECTIONS:
        chosen = [b for b in playbook.bullets if b.section == section]
        if chosen:
            lines.append(f"### {section}")
            lines += [f"[{b.id}]" + (f" ({b.kind})" if b.kind else "") + f" {b.content}" for b in chosen]
    return "\n".join(lines) or "(empty)"


def note_task(b) -> str:
    """The task a note was learned on, from its source tags ("teacher@p1+self@p1" -> "p1").
    No tag, "@general", or tags from several tasks -> "general"."""
    tags = {part.rsplit("@", 1)[1] for part in (b.source or "").split("+") if "@" in part}
    tags.discard("general")
    return tags.pop() if len(tags) == 1 else "general"


def render_notes_by_task(playbook: Playbook, task: str | None, other_budget: int | None = None) -> str:
    """render_notes, grouped by task: this task's notes first, then general ones, then other tasks'
    (cut to other_budget characters, whole notes, when given)."""
    groups: dict[str, list] = {}
    for b in playbook.bullets:
        groups.setdefault(note_task(b), []).append(b)
    def block(title, bullets):
        return f"#### {title}\n" + render_notes(Playbook(bullets=bullets))
    out = []
    if task in groups:
        out.append(block(f"notes from THIS task ({task})", groups.pop(task)))
    if "general" in groups:
        out.append(block("general notes", groups.pop("general")))
    others = [block(f"notes from task {t} (another task: test before trusting)", bs) for t, bs in groups.items()]
    if other_budget is not None:
        kept, used = [], 0
        for o in others:
            lines = o.split("\n")
            for j in range(len(lines), 0, -1):          # drop notes from the end until it fits
                part = "\n".join(lines[:j])
                if used + len(part) <= other_budget:
                    break
            else:
                part = ""
            if part.count("\n") >= 1:                   # header + at least one line
                kept.append(part); used += len(part)
        dropped = sum(len(bs) for bs in groups.values()) - sum(
            1 for k in kept for line in k.split("\n") if line.startswith("["))
        if dropped > 0:
            kept.append(f"({dropped} other-task notes not shown: context length)")
        others = kept
    return "\n\n".join(out + others) or "(empty)"


def run_summary_text(run: dict | None) -> str:
    """An agent's own previous run (run_loop._run_summary) as one short block."""
    if not run:
        return "(no previous run)"
    def f(x):
        return "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))
    rank = f"rank {run.get('rank')}/{run.get('group_size')}" if run.get("rank") else "rank -"
    errs = "; ".join(str(e) for e in (run.get("error"), run.get("failed_gate")) if e) or "none"
    return (f"Task {run.get('task')} (run {run.get('run_id')}): submitted {mesh_call_text(run.get('mesh_call'))}\n"
            f"Score {f(run.get('score'))}, {rank}; field error {f(run.get('field_error'))}, "
            f"check-point RMS error {f(run.get('probe_error'))}, DOFs {f(run.get('dofs'))}; errors: {errs}")


CLOSING = (
    "Before you mesh: for EACH item above (every step-critique correction, every mistake, every setup "
    "warning, the suggested mesh call, every contested region) write one line: what you will do about "
    "it, and answer 'could my solution fail in a way that makes the teacher tell me this again?' If yes, "
    "change your plan until the answer is no. Check your setup against every setup item before your first "
    "solve. Then follow your plan.")

CONTEXT_CHARS = 25_000      # rough cap on the whole block; other-task notes are cut first


def _task_of(d: dict) -> str:
    """The task a feedback/reflection dict is about: its "task" key, else parsed from its run id."""
    m = re.match(r"^g\d+-(.+)-a\d+$", d.get("run_id") or "")
    return d.get("task") or (m.group(1) if m else "?")


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items or []) or "- (none)"


def _critique_lines(items) -> str:
    out = []
    for c in items or []:
        c = c if isinstance(c, dict) else {"what_it_did": str(c)}
        out.append(f"- step {c.get('step')}: {c.get('what_it_did')} -> {c.get('verdict')}. "
                   f"Correction: {c.get('correction')}")
    return "\n".join(out) or "- (none)"


def _contested_lines(items) -> str:
    out = []
    for c in items or []:
        c = c if isinstance(c, dict) else {"region": str(c)}
        out.append(f"- {c.get('region')}: why contested: {c.get('why_contested')}; agents that differ: "
                   f"{c.get('which_agents_differ')}; what to do: {c.get('what_to_do')}")
    return "\n".join(out) or "- (none)"


def deep_context(agent: str, task: str | None, shared_text, private_text,
                 last_feedback: dict | None, last_reflection: dict | None, highlights: list[dict],
                 peers_text: str, last_run: dict | None = None, task_feedback: dict | None = None) -> str:
    """The full context block agent `agent` ("a3") sees on `task` (None: a held-out task) in deep
    mode. shared_text / private_text: a Playbook (rendered grouped by task, other tasks' notes cut
    first to keep the block under CONTEXT_CHARS) or ready text. last_feedback: the teacher's
    AgentFeedback for this agent from the last generation it ran (dict, with "task" or a run id);
    last_reflection: its own SelfReflection from that run; last_run: run_loop._run_summary of that
    run; task_feedback: the teacher's last TeacherFeedback dict on `task` (its setup warnings and
    contested regions); highlights: last generation's Highlights; peers_text: run_loop.peers_text."""
    parts = [PROCESS_BRIEF.strip(), f"You are agent {agent}" + (f", working on task {task}." if task else ".")]

    fb_head = "## YOUR LAST RUN\n" + run_summary_text(last_run)
    if last_feedback:
        f = last_feedback
        parts.append(f"{fb_head}\n\n## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task {_task_of(f)}, run {f.get('run_id')})\n"
                     f"Diagnosis: {f.get('diagnosis')}\n\n"
                     f"Step-by-step critique of your last run:\n{_critique_lines(f.get('step_critique'))}\n\n"
                     f"Mistakes:\n{_bullets(f.get('mistakes'))}\n\n"
                     f"Strengths:\n{_bullets(f.get('strengths'))}\n\n"
                     f"Suggested next mesh call: {mesh_call_text(f.get('next_mesh_call'))}")
    else:
        parts.append(f"{fb_head}\n\n## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN\n(none yet)")

    if task is not None:
        tf = task_feedback or {}
        parts.append(f"## SETUP WARNINGS FOR THIS TASK ({task})\n{_bullets(tf.get('setup_warnings'))}")
        parts.append(f"## CONTESTED REGIONS ON THIS TASK ({task}): where last generation's agents disagreed most\n"
                     f"{_contested_lines(tf.get('contested_regions'))}")

    if last_reflection:
        r = last_reflection
        parts.append(f"## YOUR OWN REFLECTION ON YOUR LAST RUN (task {_task_of(r)})\n"
                     f"Plan: {r.get('plan_next')}\n"
                     f"Mesh call you planned: {mesh_call_text(r.get('next_mesh_call'))}")
    else:
        parts.append("## YOUR OWN REFLECTION ON YOUR LAST RUN\n(none yet)")

    hl = [f"- {h.get('agent')} on {_task_of(h)}: {h.get('what')} Why it works: {h.get('why_it_works')} "
          f"How to copy: {h.get('how_to_copy')}" for h in highlights if h.get("agent") != agent]
    parts.append("## WHAT OTHER AGENTS DID WELL (last generation, all tasks)\n" + ("\n".join(hl) or "(none yet)"))
    if peers_text.strip():
        parts.append(peers_text.strip())
    closing = "## THIS GENERATION\n" + CLOSING + (
        "\nUse one solve as an experiment: change one thing and compare. Never copy a peer's mesh "
        "exactly: a tie on the hidden check points pays nothing.")

    # Notes last; other tasks' notes get whatever room is left under CONTEXT_CHARS (split evenly).
    def render(x, budget):
        return x if isinstance(x, str) else render_notes_by_task(x, task, budget)
    fixed = sum(len(p) + 2 for p in parts) + len(closing) + 200
    full = [render(x, None) for x in (shared_text, private_text)]
    if fixed + sum(map(len, full)) > CONTEXT_CHARS:
        base = [render(x, 0) for x in (shared_text, private_text)]
        room = max(0, CONTEXT_CHARS - fixed - sum(map(len, base)))
        full = [render(x, room // 2) for x in (shared_text, private_text)]
    parts.append("## SHARED PLAYBOOK (all agents)\n" + full[0])
    parts.append("## YOUR NOTES (only you see these)\n" + full[1])
    parts.append(closing)
    return "\n\n".join(parts)
