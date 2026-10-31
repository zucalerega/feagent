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


def _task_of(d: dict) -> str:
    """The task a feedback/reflection dict is about: its "task" key, else parsed from its run id."""
    m = re.match(r"^g\d+-(.+)-a\d+$", d.get("run_id") or "")
    return d.get("task") or (m.group(1) if m else "?")


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items or []) or "- (none)"


def deep_context(agent: str, task: str | None, shared_text: str, private_text: str,
                 last_feedback: dict | None, last_reflection: dict | None, highlights: list[dict],
                 peers_text: str) -> str:
    """The full context block agent `agent` ("a3") sees on `task` (None: a held-out task) in deep
    mode. last_feedback: the teacher's AgentFeedback for this agent from the last generation it ran
    (dict, with "task" or a run id); last_reflection: its own SelfReflection from that run;
    highlights: last generation's Highlights from every task; peers_text: run_loop.peers_text."""
    parts = [PROCESS_BRIEF.strip(), f"You are agent {agent}" + (f", working on task {task}." if task else ".")]

    if last_feedback:
        f = last_feedback
        parts.append(f"## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task {_task_of(f)}, run {f.get('run_id')})\n"
                     f"Diagnosis: {f.get('diagnosis')}\n\nMistakes:\n{_bullets(f.get('mistakes'))}\n\n"
                     f"Strengths:\n{_bullets(f.get('strengths'))}\n\n"
                     f"Suggested next mesh call: {mesh_call_text(f.get('next_mesh_call'))}")
    else:
        parts.append("## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN\n(none yet)")

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
    parts.append("## SHARED PLAYBOOK (all agents)\n" + shared_text)
    parts.append("## YOUR NOTES (only you see these)\n" + private_text)
    parts.append("## THIS GENERATION\n"
                 "In your first message, state which notes you will apply (by id) and what you will change "
                 "compared with your last run, and why. Use one solve as an experiment: change one thing and "
                 "compare. Never copy a peer's mesh exactly: a tie on the hidden check points pays nothing.")
    return "\n\n".join(parts)
