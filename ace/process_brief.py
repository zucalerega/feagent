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

Remembered values. Never use a published or remembered benchmark value (e.g. a NAFEMS target
stress) to judge or choose a solve: scoring is the error of the whole field, and a remembered
number can be wrong for this model. Choose your submitted solve by evidence from your own solves.

Scoring. The submitted solution is compared with a very fine reference solution: the error is the
larger of (a) the energy-norm stress error over the WHOLE part and (b) the RMS von Mises error at a
set of check points spread over the part (a few are disclosed to the agent, the rest are hidden).
The yardstick is an automesher (gmsh default sizes, quadratic elements). The target is half the
automesher's error: below that target the score falls quadratically, above it beating it keeps
paying. DOFs cost: the score is divided by DOFs relative to the automesher, so extra elements must
buy at least proportional accuracy. On each hidden check point, the single run with the smallest
error among that generation's runs on the task earns a bonus; a tie pays nobody, so copying a
peer's mesh exactly earns no bonus. A run whose mesh recipe differs structurally from every other
run on the task that generation (another order, h off by more than 1.5x, or mostly different
refinement targets) and whose base score is at least 90% of the best there earns a diversity bonus.
A task scored against a physical test (its statement says so, e.g. abc_00000065_compression) is
different: the submitted mesh is run through a frozen nonlinear simulation of the test and the error is
the mean relative force error at the test's checkpoints; the automesher yardstick, the score formula
and the diversity bonus are as above, and there is no be-the-best bonus.

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



TOOLS_BRIEF = """\
## THE AGENTS' TOOLS (every agent has them; full reference: ace/TOOLS.md)
- inspect_geometry(): every face (group f<tag>, type, area, centroid, normal, neighbours, named in
  the statement or not) and edge (tag, length, faces, dihedral angle, RE-ENTRANT or not: a likely
  stress singularity), smallest features, plain-language hints.
- mesh(h, order, refinements): refinements at a face {group}, a CAD edge {edge: tag}, a {point} or a
  {box: [xmin,ymin,zmin,xmax,ymax,zmax]}, each with size and radius. Returns quality, repaired_cells
  (folded curved cells straightened automatically) and folded_cells (must be 0, else the run is
  invalid), and element size near each named group.
- solve(setup): the scored quantity plus every statement quantity, reactions vs applied loads, max
  displacement and stress with locations, and a reference-free error estimate with the top 5
  hotspots (location, nearest face/edge, local element size, share of the error).
- view(solve, kind=mesh|stress|error, view=iso|x|y|z|-x|-y|-z): a picture the agent sees.
- compare(solve_a, solve_b): how much the stress field changed between two solves and where
  (convergence evidence without a reference).
- submit(solve).
- nonlinear_check(mesh): ONLY on a task scored by a nonlinear simulation of a physical test (e.g.
  abc_00000065_compression): runs that frozen scoring simulation (elastoplastic, large deformation)
  on one of the run's meshes and returns the force at each checkpoint vs the test, the mean relative
  error (the scored error), nodes, DOFs, peak plastic strain and hotspots (location, nearest faces).
  Minutes each, a few per run; on such a task the linear solve is a fast proxy for placing elements.
Good practice: inspect the geometry first; refine re-entrant edges and support/load edges; check
folded_cells is 0 and equilibrium holds; use the hotspots to place the next refinement; compare two
solves before choosing; submit by this evidence.
"""

# The single-agent baseline (ace/single_loop.py): PROCESS_BRIEF rewritten for ONE agent that works on
# every task in one conversation. Same scoring, same learning (teacher, self-reflection, organizer),
# no peers and no be-the-best bonus. ace/single_loop.py swaps it in for PROCESS_BRIEF in every
# prompt (the agent's context and the teacher, reflection and organizer system prompts).
SINGLE_BRIEF = """\
## HOW THIS SYSTEM WORKS (read by the agent, the teacher, the reflector and the organizer)

ONE persistent meshing agent learns over many generations to mesh FEA problems well. Every
generation it works on ALL the training tasks in one conversation: for each task it inspects the
geometry, meshes it (global size h, element order, local refinements at named groups or points),
solves within the task's solve budget and submits one solve. Every tool call names its task.

Remembered values. Never use a published or remembered benchmark value (e.g. a NAFEMS target
stress) to judge or choose a solve: scoring is the error of the whole field, and a remembered
number can be wrong for this model. Choose your submitted solve by evidence from your own solves.

Scoring. The submitted solution is compared with a very fine reference solution: the error is the
larger of (a) the energy-norm stress error over the WHOLE part and (b) the RMS von Mises error at a
set of check points spread over the part (a few are disclosed to the agent, the rest are hidden).
The yardstick is an automesher (gmsh default sizes, quadratic elements). The target is half the
automesher's error: below that target the score falls quadratically, above it beating it keeps
paying. DOFs cost: the score is divided by DOFs relative to the automesher, so extra elements must
buy at least proportional accuracy. There are no other agents and no bonus: only this score counts.

Learning. After each generation a teacher looks at the agent's run on every task: the whole mesh,
where the error is (error maps and region grids), the setup and the agent's reasoning. It tells the
agent in depth what went wrong and exactly what to change, and what it did really well. The agent
also looks at its own final results and whole mesh and writes notes for itself. An organizer then
merges, rewrites and orders the agent's notes and its playbook of task-level lessons: it keeps
every useful specific idea and drops only what newer evidence superseded or contradicted.

Notes are tagged by task (source "...@<task>" or "@general"): another task's note is a hint to
test. Address every point: the agent answers every item of its teacher feedback in its first
message, and the teacher checks next generation that each promise was kept and flags repeats.

Rules of the game. Notes are earned evidence from earlier runs: respect them, but test them, and
report when one fails. Every generation the agent must deliberately change at least one thing on
each task compared with its last run, and say what and why. Use solves as controlled experiments
(change one thing, compare). Specific beats generic: name groups, points, sizes, radii and orders.
"""

# Flow (CFD) tasks (ace.flow_mesh, ace.flow_feedback, ace.flow_field). The briefs above are written for
# elasticity and stay as they are, so a run without flow tasks reads exactly them. On a flow task this
# block goes with the task, as NL_SCORING (ace.deep_teacher) does for a task scored by a physical test:
# after the brief in the agent's context (deep_context, single_context) and next to the task statement
# for the teacher (deep_teacher.build_prefix) and the self-reflection (self_reflect.build_user).
FLOW_BRIEF = """\
## FLOW (CFD) TASKS
A flow task is steady laminar incompressible flow, solved with finite volumes (OpenFOAM, SIMPLEC).
The rest of the instructions is written for elasticity; on a flow task read it with these changes:
- Scoring: the error is the larger of (a) the relative L2 VELOCITY error over the whole domain and
  (b) the RMS SPEED error at check points in the bulk of the flow (away from walls), against the
  exact solution or a very fine reference. The automesher is the mesh tool's uniform triangles /
  tetrahedra at the task's base size with the second-order convection scheme (linearUpwind). The
  score formula and the bonuses are the same; the diversity bonus counts another cell kind (quads,
  boundary layer, sweep) where it counts another element order. Error maps, region grids and
  disagreement maps show the squared velocity error and the speed, not stress or error energy.
- No element order: finite-volume cells. On a smooth flow the error falls ~h^2 (2D: ~N^-1, 3D:
  ~N^-2/3, N = cells), slower at singular corners (the ends of a moving lid, a sharp step corner);
  quads and hexahedra converge more regularly than triangles and tetrahedra.
  DOFs = cells x unknowns per cell (3 in 2D, 4 in 3D).
- Mesh controls: h and refinements as for elasticity; quads (quad-dominant cells: 2D, or the swept
  face); boundary_layer {groups, first, ratio, layers} (thin cells along no-slip walls: first-cell
  height, growth per layer, number of layers; in 3D only with a sweep); sweep {source, layers, ratio}
  (3D: a straight duct or pipe as its end face extruded in layers, hexahedra or prisms, far fewer
  cells than tetrahedra for the same error).
- Gate: finite-volume quality (fv_ok: non-orthogonality <= 70 deg, skewness <= 4), not element
  shape: thin boundary-layer cells pass. There is no folded_cells check.
- Setup: formulation, fluid (rho; mu in Pa s: a viscosity taken as mPa s or cP is 1000x too large),
  convection scheme, and per boundary group velocity_inlet (its profile, pointing INTO the domain),
  pressure_outlet, wall (no slip; a moving wall carries its velocity), slip or symmetry (never in
  place of a no-slip wall). A setup error spreads error over the whole domain.
- Tools: mesh also returns fv_quality, fv_ok and per boundary group the first cells' wall distance;
  solve returns convergence, mass balance, wall forces, peak speed and pressure range, the cell
  Peclet number, reversed-flow zones and an error indicator (velocity jumps at faces) with its top
  5 hotspots; view(kind = mesh | speed | pressure | error; 3D: a section normal to x, y or z);
  compare gives the velocity difference between two solves and where it is.
"""

FLOW_MESH_KEYS = ("quads", "boundary_layer", "sweep")     # what a FlowMeshCall has instead of an order


def _inline(v) -> str:
    """A value as a mesh call reads: {k: v, ...}, [a, b], true / false, numbers and names bare."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_inline(x)}" for k, x in v.items() if x is not None) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(map(_inline, v)) + "]"
    return str(v).lower() if isinstance(v, bool) else str(v)


def mesh_call_text(call: dict | None) -> str:
    """A MeshCall (ace.deep_teacher) as the agent's own tool call reads: mesh(h=..., order=..., refinements=[...]);
    a FlowMeshCall (a flow task: no order) as mesh(h=..., quads=..., refinements=[...]) plus its
    boundary_layer={...} and sweep={...} when set."""
    if not call:
        return "(none)"
    refs = ", ".join(
        "{" + ", ".join(f"{k}: {v}" for k, v in (("group", r.get("group")), ("edge", r.get("edge")),
                                                 ("point", r.get("point")), ("box", r.get("box")),
                                                 ("size", r.get("size")), ("radius", r.get("radius")))
                        if v is not None) + "}"
        for r in call.get("refinements") or [])
    if "order" in call or not any(k in call for k in FLOW_MESH_KEYS):
        return f"mesh(h={call.get('h')}, order={call.get('order')}, refinements=[{refs}])"
    extra = "".join(f", {k}={_inline(call[k])}" for k in ("boundary_layer", "sweep") if call.get(k))
    return f"mesh(h={call.get('h')}, quads={_inline(bool(call.get('quads')))}, refinements=[{refs}]{extra})"


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
            f"Score {f(run.get('score'))}, {rank}; " + (
                f"force error vs the test {f(run.get('nonlinear_error'))}, "    # a nonlinear task (ace.nonlinear)
                if run.get("nonlinear_error") is not None else
                f"field error {f(run.get('field_error'))}, check-point RMS error {f(run.get('probe_error'))}, ")
            + f"DOFs {f(run.get('dofs'))}; errors: {errs}")


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


def _feedback_block(last_run: dict | None, last_feedback: dict | None) -> str:
    """The agent's last run and the teacher's feedback on it."""
    fb_head = "## YOUR LAST RUN\n" + run_summary_text(last_run)
    if not last_feedback:
        return f"{fb_head}\n\n## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN\n(none yet)"
    f = last_feedback
    return (f"{fb_head}\n\n## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task {_task_of(f)}, run {f.get('run_id')})\n"
            f"Diagnosis: {f.get('diagnosis')}\n\n"
            f"Step-by-step critique of your last run:\n{_critique_lines(f.get('step_critique'))}\n\n"
            f"Mistakes:\n{_bullets(f.get('mistakes'))}\n\n"
            f"Strengths:\n{_bullets(f.get('strengths'))}\n\n"
            f"Suggested next mesh call: {mesh_call_text(f.get('next_mesh_call'))}")


def _task_blocks(task: str, task_feedback: dict | None) -> list[str]:
    """The teacher's setup warnings and contested regions for a task."""
    tf = task_feedback or {}
    return [f"## SETUP WARNINGS FOR THIS TASK ({task})\n{_bullets(tf.get('setup_warnings'))}",
            f"## CONTESTED REGIONS ON THIS TASK ({task}): where last generation's agents disagreed most\n"
            f"{_contested_lines(tf.get('contested_regions'))}"]


def _reflection_block(last_reflection: dict | None) -> str:
    if not last_reflection:
        return "## YOUR OWN REFLECTION ON YOUR LAST RUN\n(none yet)"
    r = last_reflection
    return (f"## YOUR OWN REFLECTION ON YOUR LAST RUN (task {_task_of(r)})\n"
            f"Plan: {r.get('plan_next')}\n"
            f"Mesh call you planned: {mesh_call_text(r.get('next_mesh_call'))}")


# What each role asks of the agent (the teacher assigns them, ace.deep_teacher.allocate). Shown with
# the assignment, and alone when the agent moved to another task, where the experiment does not apply.
ROLE_MEANING = {
    "control": "re-run the best recipe known on your task with at most one small, deliberate tweak, so "
               "your score shows how noisy a score is",
    "variant": "start from the best recipe known on your task and change exactly ONE factor of it "
               "(h, the refinement sizes, radii or places, or the order), so the effect of that factor is clear",
    "explorer": "try a mesh STRUCTURALLY different from what agents have tried on your task (other "
                "refinement targets, element order or strategy), chosen from the error maps and contested "
                "regions; not a resized copy of anyone's recipe",
}


def assignment_block(assignment: dict | None, made_for: str, task: str | None) -> str | None:
    """The teacher's assignment for this generation (AgentFeedback.assignment), or None when there is
    none or the agent is on a held-out task (there it is judged, not experimenting)."""
    if not assignment or task is None:
        return None
    role = assignment.get("role")
    head = f"## YOUR ASSIGNMENT THIS GENERATION\nRole: {role}: {ROLE_MEANING.get(role, '')}.\n"
    if made_for == task:
        return head + (f"Experiment: {assignment.get('experiment')}\nWhy: {assignment.get('why')}\n"
                       "Carry it out: it is how this generation's agents cover different meshes.")
    return head + (f"The teacher's experiment was for task {made_for}, not your task {task}, so apply the "
                   f"role, not the experiment, here. (For {made_for} it was: {assignment.get('experiment')})")


def persona_block(persona: str) -> str:
    """A fixed exploration style for one agent (run_loop --personas)."""
    return "## YOUR EXPLORATION STYLE (yours alone)\n" + persona


def deep_context(agent: str, task: str | None, shared_text, private_text,
                 last_feedback: dict | None, last_reflection: dict | None, highlights: list[dict],
                 peers_text: str, last_run: dict | None = None, task_feedback: dict | None = None,
                 persona: str | None = None, flow: bool = False) -> str:
    """The full context block agent `agent` ("a3") sees on `task` (None: a held-out task) in deep
    mode. shared_text / private_text: a Playbook (rendered grouped by task, other tasks' notes cut
    first to keep the block under CONTEXT_CHARS) or ready text. last_feedback: the teacher's
    AgentFeedback for this agent from the last generation it ran (dict, with "task" or a run id);
    last_reflection: its own SelfReflection from that run; last_run: run_loop._run_summary of that
    run; task_feedback: the teacher's last TeacherFeedback dict on `task` (its setup warnings and
    contested regions); highlights: last generation's Highlights; peers_text: run_loop.peers_text;
    persona: this agent's fixed exploration style (run_loop --personas), None for most agents.
    flow: the agent's task is a flow (CFD) task (held out or not): FLOW_BRIEF follows the brief."""
    parts = [PROCESS_BRIEF.strip(), *([FLOW_BRIEF.strip()] if flow else []),
             f"You are agent {agent}" + (f", working on task {task}." if task else ".")]
    if block := assignment_block((last_feedback or {}).get("assignment"), _task_of(last_feedback or {}), task):
        parts.append(block)

    parts.append(_feedback_block(last_run, last_feedback))
    if task is not None:
        parts += _task_blocks(task, task_feedback)
    parts.append(_reflection_block(last_reflection))

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
    style = [persona_block(persona)] if persona else []     # just before the closing block
    fixed = sum(len(p) + 2 for p in parts + style) + len(closing) + 200
    full = [render(x, None) for x in (shared_text, private_text)]
    if fixed + sum(map(len, full)) > CONTEXT_CHARS:
        base = [render(x, 0) for x in (shared_text, private_text)]
        room = max(0, CONTEXT_CHARS - fixed - sum(map(len, base)))
        full = [render(x, room // 2) for x in (shared_text, private_text)]
    parts.append("## SHARED PLAYBOOK (all agents)\n" + full[0])
    parts.append("## YOUR NOTES (only you see these)\n" + full[1])
    parts += style
    parts.append(closing)
    return "\n\n".join(parts)


def single_context(tasks: list[str], playbook, notes, per_task: dict[str, dict], flow: bool = False) -> str:
    """The context block of the single-agent baseline (ace/single_loop.py): SINGLE_BRIEF, then for
    each task the same feedback blocks a swarm agent gets (its last run on that task, the teacher's
    feedback on it, setup warnings, contested regions, its own reflection), then its playbook and
    notes (Playbooks, grouped by task; no length cap: every task is the agent's own) and the closing
    checklist. per_task: {task: {"last_run", "feedback", "reflection", "task_feedback"}} (each may
    be missing). There are no peers' highlights: nobody else ran. flow: the tasks are flow (CFD) tasks
    (single_loop takes all-flow or all-solid tasks): FLOW_BRIEF follows the brief."""
    parts = [SINGLE_BRIEF.strip(), *([FLOW_BRIEF.strip()] if flow else []),
             f"You are the single agent, working on all of these tasks: {', '.join(tasks)}."]
    for t in tasks:
        d = per_task.get(t) or {}
        parts.append(f"# TASK {t}: WHAT HAPPENED LAST TIME")
        parts.append(_feedback_block(d.get("last_run"), d.get("feedback")))
        parts += _task_blocks(t, d.get("task_feedback"))
        parts.append(_reflection_block(d.get("reflection")))

    def by_task(pb) -> str:
        groups: dict[str, list] = {}
        for b in pb.bullets:
            groups.setdefault(note_task(b), []).append(b)
        order = ["general"] + [t for t in groups if t != "general"]
        return "\n\n".join(f"#### {'general notes' if t == 'general' else f'notes for task {t}'}\n"
                            + render_notes(Playbook(bullets=groups[t])) for t in order if t in groups) or "(empty)"
    parts.append("## YOUR PLAYBOOK (task-level lessons)\n" + by_task(playbook))
    parts.append("## YOUR NOTES\n" + by_task(notes))
    parts.append("## THIS GENERATION\n" + CLOSING + "\nDo this for every task. Use solves as experiments: "
                 "change one thing and compare.")
    return "\n\n".join(parts)


# Every role reads one of the briefs (agents via deep_context / single_context; teacher, reflector and
# organizer via their system prompts), so appending the tools section here tells all of them.
PROCESS_BRIEF = PROCESS_BRIEF.rstrip() + "\n\n" + TOOLS_BRIEF
SINGLE_BRIEF = SINGLE_BRIEF.rstrip() + "\n\n" + TOOLS_BRIEF
