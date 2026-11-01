"""The deep teacher (--deep): an expert review of EVERY run on a task, from everything they did.

For each run the teacher sees its numbers and rank, setup, mesh request, where its error is
(hotspots, a coarse grid over the WHOLE mesh, every check point), its full reasoning and tool calls,
the notes it had, pictures of its mesh, error and stress (ace/run_report.py), the disagreement map
across runs (ace.run_report.disagreement_map) and each agent's own previous run.
On a task scored by its nonlinear force curve against a physical test (is_nonlinear, ace.nonlinear)
the teacher also gets the experiment analysis (teacher_brief), each run's curve checkpoints and
plastic-strain hotspots with its curve and PEEQ pictures, and every run's error vs nodes so far;
there is no field error, check point or disagreement map there (NL_SCORING says so).
On a flow (CFD) task (is_flow_task) it also gets process_brief.FLOW_BRIEF (velocity instead of
stress, finite-volume cells instead of an element order), the per-agent schema's next mesh call is a
FlowMeshCall (quads, boundary_layer, sweep) and allocate() varies the cell structure.

To keep a generation fast, the review is split into parallel calls that share one cached prefix
(everything above): one call per agent (a step-by-step critique, notes, the next mesh call, a check
of what it promised last time) and one cross-run call (summary, setup warnings, contested regions,
highlights, shared lessons). Each per-agent call also writes the agent's experiment for next
generation in the role allocate() gave it (control / variant / explorer). teach() merges them into one
TeacherFeedback. ace/organizer.py then
files the notes into memory.

The schema below (NoteItem, MeshCall) is shared with ace/self_reflect.py and ace/organizer.py.
"""

from __future__ import annotations

import functools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ace import config
from ace.config import BEST_BONUS, MAX_SOLVES, PROBE_DISCLOSED, PROBE_POINTS
from ace.llm import image_block, text_block
from ace.reflector import Section

# The process brief (ace/process_brief.py) explains the whole loop to every model in it. The
# default below only covers a checkout where that module is missing.
try:
    from ace.process_brief import FLOW_BRIEF, PROCESS_BRIEF
except ImportError:
    FLOW_BRIEF = ""
    PROCESS_BRIEF = (
        "You are part of a loop that trains FEA meshing agents over generations. Each generation the "
        "agents are split at random across tasks; each meshes and solves its task and is scored against a "
        "gmsh-default automesher on the whole stress field and at check points, aiming at 2x its accuracy "
        "for as few DOFs as possible, plus a bonus for being uniquely best on hidden points (ties pay "
        "nothing). A teacher reviews every run's whole mesh and error map, each agent reflects on its own "
        "results, and an organizer keeps each agent's notes. Notes are evidence from earlier runs: respect "
        "them, but test them. Every generation each agent must change at least one thing on purpose.")

Kind = Literal["fix", "idea", "keep", "avoid", "highlight"]

MAX_IMAGES = 20          # images in the shared prefix: mesh + error for every run, stress while within this
REPORT_CHARS = 6000      # report_text per run
TRACE_CHARS = 14000      # reasoning and tool calls per run
THOUGHT_CHARS = 1500     # one thought (or said text) in a trace
NOTES_CHARS = 3000       # the notes a run had, per run


# --------------------------------------------------------------------------- output schema


class NoteItem(BaseModel):
    task: str = Field(default="general", description='the task id this note is about (e.g. "nafems_le1"), '
                                                      'or "general" if it applies to any task')
    section: Section
    content: str = Field(description="one specific, actionable note (<= 60 words): WHERE (group, point or "
                                     "region), WHAT (size, order, refinement, setup change) and the EXPECTED "
                                     "effect, citing the evidence. Never generic ('refine more').")
    kind: Kind = Field(description="fix = correct a mistake; idea = untested idea worth a try; keep = "
                                   "something that worked, keep doing it; avoid = something that hurt; "
                                   "highlight = a peer's good idea to copy or adapt")


class Refinement(BaseModel):
    group: str | None = Field(default=None, description="a named face group (f<tag>), or null")
    edge: int | None = Field(default=None, description="a CAD edge tag from inspect_geometry (refine along it), or null")
    point: list[float] | None = Field(default=None, description="[x, y] or [x, y, z], or null")
    box: list[float] | None = Field(default=None, description="[xmin, ymin, zmin, xmax, ymax, zmax], or null")
    size: float = Field(description="element size at the group/point")
    radius: float = Field(description="distance over which the size grows back to h")


class MeshCall(BaseModel):
    h: float
    order: int = Field(description="1 or 2")
    refinements: list[Refinement] = Field(default_factory=list)


# A flow (CFD) task's mesh call (ace.agent.TOOLS_FLOW): no element order, and the cell structure
# instead: quads, wall boundary layers and, in 3D, a sweep. Its own schema, so that the elasticity
# schemas (and prompts) stay exactly as they were.
class BoundaryLayerCall(BaseModel):
    groups: list[str] = Field(description="the no-slip wall groups to grow the layers on")
    first: float = Field(description="height of the first cell at the wall")
    ratio: float = Field(default=1.2, description="growth per layer, 1-3")
    layers: int = Field(description="number of layers, 1-40")


class SweepCall(BaseModel):
    source: str = Field(description="group of the planar end face to extrude (e.g. the inlet)")
    layers: int = Field(description="cell layers along the sweep, 1-400")
    ratio: float = Field(default=1.0, description="last / first layer length (> 1: the short layers at the source end)")


class FlowMeshCall(BaseModel):
    h: float
    quads: bool = Field(default=False, description="quad-dominant cells (2D, or the swept face: hexahedra)")
    refinements: list[Refinement] = Field(default_factory=list)
    boundary_layer: BoundaryLayerCall | None = Field(default=None, description="wall layers, or null")
    sweep: SweepCall | None = Field(default=None, description="3D straight duct or pipe only: the sweep, or null")


class StepCritique(BaseModel):
    step: str = Field(description='which step, e.g. "mesh call 2" or "solve 1 setup" or "choice of order"')
    what_it_did: str
    verdict: Literal["good", "mistake", "missed_opportunity"]
    correction: str = Field(description="for a mistake or missed opportunity: the named, concrete correction "
                                        "(what to do instead, with numbers); for good: why to keep it")


class PromiseCheck(BaseModel):
    item: str = Field(description="what the agent promised or was told to do after its previous run")
    status: Literal["done", "partly", "not_done", "unclear"]
    comment: str = Field(description="evidence from this run; say REPEATED MISTAKE if it repeated one")


class Assignment(BaseModel):
    """The experiment one agent runs next generation (allocate() picks the role, the teacher the rest)."""
    agent: str
    role: Literal["control", "variant", "explorer"]
    experiment: str = Field(description="one concrete thing to try, with locations (groups, edges, points, "
                                        "boxes) and sizes; for a variant name the one factor and its new value")
    why: str = Field(description="the evidence that makes this experiment worth a run")


class AgentFeedback(BaseModel):
    agent: str = Field(description='the agent, e.g. "a3" (the end of its run_id)')
    run_id: str
    diagnosis: str = Field(description="what happened and WHY, citing regions/groups/numbers from its report")
    step_critique: list[StepCritique] = Field(default_factory=list,
                                              description="every meaningful step of the run, in order")
    promised_items_check: list[PromiseCheck] = Field(default_factory=list,
                                                     description="each item from its previous run's plan/feedback")
    strengths: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    next_mesh_call: MeshCall = Field(description="the exact mesh call that should beat this run's score next time")
    notes: list[NoteItem] = Field(description="8-15 agent-specific, actionable notes, each tagged with its task")
    assignment: Assignment | None = Field(default=None, description="this agent's experiment next generation")


class AssignedFeedback(AgentFeedback):
    """AgentFeedback with the assignment required: the schema of the per-agent call when 2+ agents ran."""
    assignment: Assignment


class FlowAgentFeedback(AgentFeedback):
    """AgentFeedback on a flow task: the next mesh call is a FlowMeshCall."""
    next_mesh_call: FlowMeshCall = Field(description="the exact mesh call that should beat this run's score next time")


class FlowAssignedFeedback(FlowAgentFeedback):
    assignment: Assignment


class Highlight(BaseModel):
    agent: str
    run_id: str
    what: str
    why_it_works: str
    how_to_copy: str


class BulletRating(BaseModel):
    bullet_id: str
    rating: Literal["useful", "harmful", "not_useful"]
    why: str


class ContestedRegion(BaseModel):
    region: str = Field(description="the region of the disagreement map (and the nearest named group)")
    why_contested: str = Field(description="why the runs differ there (mesh size, order, setup, a singularity)")
    which_agents_differ: list[str] = Field(default_factory=list)
    what_to_do: str = Field(description="concrete: size/order/radius there, or the setup check to make")


class CrossRunFeedback(BaseModel):
    """The cross-run call's output: everything in TeacherFeedback except per_agent."""
    task_summary: str = Field(description="what separates the best runs from the worst on this task")
    setup_warnings: list[str] = Field(default_factory=list,
                                      description="every setup that differs from the problem statement, by run_id")
    contested_regions: list[ContestedRegion] = Field(default_factory=list,
                                                     description="the most contested regions, most contested first")
    highlights: list[Highlight] = Field(default_factory=list,
                                        description="the best idea of every agent where anything was good, "
                                                    "plus the best ideas overall")
    shared_lessons: list[NoteItem] = Field(description="6-12 lessons for every agent")
    bullet_ratings: list[BulletRating] = Field(default_factory=list,
                                               description="for playbook bullets the runs cited (optional)")


class TeacherFeedback(CrossRunFeedback):
    task: str
    per_agent: list[AgentFeedback] = Field(description="one per run")
    assignments: list[Assignment] = Field(default_factory=list, description="next generation's experiments, "
                                          "one per agent (copied from per_agent), best run's agent first")


class FlowTeacherFeedback(TeacherFeedback):
    """TeacherFeedback on a flow task (typed so that model_dump keeps the FlowMeshCalls: a field typed
    AgentFeedback would serialize them as MeshCalls and drop quads, boundary_layer and sweep)."""
    per_agent: list[FlowAgentFeedback] = Field(description="one per run")


# --------------------------------------------------------------------------- shared helpers


def agent_name(run_id: str) -> str:
    """"g3-nafems_le1-a5" -> "a5"."""
    return run_id.rsplit("-", 1)[-1]


def report_as_text(report: dict, max_chars: int = REPORT_CHARS) -> str:
    """ace.run_report.report_text, or (if that fails on a malformed report) the raw report as
    JSON, so a reader still gets whatever there is."""
    try:
        from ace.run_report import report_text
        return report_text(report, max_chars=max_chars)
    except Exception:      # noqa: BLE001  (a malformed report must not stop the teaching)
        pass
    slim = {k: v for k, v in report.items() if k != "images"}
    return json.dumps(slim, default=str)[:max_chars]


def report_images(report: dict, kinds: tuple[str, ...]) -> list[tuple[str, str]]:
    """(kind, path) for each of `kinds` the report has a picture of."""
    imgs = report.get("images") or {}
    return [(k, imgs[k]) for k in kinds if imgs.get(k)]


def _short(x, n: int) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def trace_excerpt(run: dict, max_chars: int = TRACE_CHARS) -> str:
    """The run's steps as short lines: its reasoning (thinking and text), every mesh call with what
    came back, every solve's setup and outcome, and the submit. Long thoughts are cut first."""
    lines = []
    for t in run.get("trace") or []:
        kind = t.get("type")
        if kind in ("text", "thinking"):
            lines.append(f"  [{'thought' if kind == 'thinking' else 'said'}] "
                         f"{_short(t.get('text', ''), THOUGHT_CHARS)}")
        elif kind == "tool_call":                       # full arguments: the setup and mesh are the decisions
            args = {k: v for k, v in (t.get("input") or {}).items() if k != "using"}
            used = (t.get("input") or {}).get("using") or []
            lines.append(f"  -> {t.get('name')}({json.dumps(args, default=str)})" + (f" using {used}" if used else ""))
        elif kind == "tool_result":
            out = t.get("result") or {}
            if t.get("name") == "mesh":
                keep = ("ok", "nodes", "elements", "order", "min_quality", "h_used", "within_cap", "error", "hint")
            elif t.get("name") == "solve":
                keep = ("ok", "qoi", "value", "solve", "nodes", "error", "warnings")
            elif t.get("name") == "inspect_geometry":
                continue                                     # the report already has the geometry
            else:
                keep = tuple(out) if isinstance(out, dict) else ()
            brief = {k: out[k] for k in keep if isinstance(out, dict) and k in out}
            lines.append(f"  <- {t.get('name')}: {_short(brief or out, 400)}")
    text = "\n".join(lines)
    if len(text) > max_chars:                            # keep the start and the end (the decisions)
        half = max_chars // 2
        text = text[:half] + "\n  ...[middle cut]...\n" + text[-half:]
    return text or "  (no trace)"


def is_nonlinear(task: str) -> bool:
    """Is the task scored by its nonlinear force curve against a physical test (ace.nonlinear)? The
    same test as ace.nonlinear.is_nonlinear, here so the teacher does not need that module to load."""
    return (config.TASKS.get(task) or {}).get("score") == "nonlinear"


@functools.lru_cache(maxsize=None)
def is_flow_task(task: str) -> bool:
    """Is the task a flow (CFD) task (ace.fea.is_flow)? Unknown ids are not."""
    from ace import fea                      # heavy (gmsh, the solvers): only when a task is checked
    try:
        return fea.is_flow(fea.get_problem(task))
    except KeyError:
        return False


def teacher_brief(task: str) -> str:
    """The experiment analysis written for the teacher (physical_tests/<task>/teacher_brief.md), or ""."""
    path = Path(__file__).resolve().parent.parent / "physical_tests" / task / "teacher_brief.md"
    return path.read_text() if path.exists() else ""


def run_error(run: dict) -> float | None:
    """The run's scored error: the nonlinear curve's mean error against the test on a nonlinear
    task, else the field-mode error (rel_error)."""
    c = run.get("checks") or {}
    if c.get("nonlinear"):
        return c["nonlinear"].get("mean_rel_err")
    return c.get("rel_error")


def _numbers(report: dict, run: dict) -> dict:
    """The run's headline numbers: from the report (add_scores), else from its checks."""
    n = report.get("numbers") or {}
    c = run.get("checks") or {}
    if c.get("nonlinear"):                               # the force-curve error, not a field error
        n = {**n, "field_error": run_error(run)}
    return {"score": n.get("score", c.get("score")), "error": n.get("field_error", c.get("rel_error")),
            "probe_error": n.get("probe_error", c.get("probe_error")), "dofs": n.get("dofs", c.get("dofs")),
            "gain": n.get("gain", c.get("gain")), "bonus": n.get("bonus", c.get("best_bonus")),
            "rank": n.get("rank_in_group", c.get("rank"))}


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def _task_statement(task: str) -> str:
    try:
        from ace.agent import task_prompt
        return task_prompt(task)
    except Exception as e:                    # noqa: BLE001  (a missing reference must not stop teaching)
        return f"Problem id: {task} (statement unavailable: {e})"


def prev_summary(prev: dict | None, max_chars: int = 2000) -> str:
    """The previous generation's feedback on this task, short (it mostly concerns OTHER agents: the
    agents are re-split each generation; each agent's own history is in continuity_text)."""
    if not prev:
        return "(none: this is the first generation with deep feedback on this task)"
    lines = [f"Summary: {prev.get('task_summary', '')}"]
    lines += [f"Lesson: {l['content']}" for l in prev.get("shared_lessons", [])[:8]]
    for a in prev.get("per_agent", []):
        call = a.get("next_mesh_call")
        lines.append(f"{a.get('agent')} ({a.get('run_id')}): {_short(a.get('diagnosis', ''), 250)} "
                     f"| suggested next mesh: {_short(call, 200)}")
    return "\n".join(lines)[:max_chars]


# --------------------------------------------------------------------------- the prompt


# For the generator's context (ace/process_brief.deep_context): the closing block after the
# teacher's feedback, so every item is answered before the agent meshes.
BEFORE_YOU_MESH = (
    "Before you mesh: for EACH item above (every step-critique correction, every mistake, every setup "
    "warning, the suggested mesh call, every contested region) write one line: what you will do about it, "
    "and answer 'could my solution fail in a way that makes the teacher tell me this again?' If yes, change "
    "your plan until the answer is no. Check your setup against every setup item before your first solve. "
    "Then follow your plan.")


SYSTEM = f"""\
You are the TEACHER: an expert in finite element analysis and adaptive meshing, mentoring a class of
FEA meshing agents. You see every run on one task: results, full reasoning and every tool call. Your
feedback is the main way the agents learn. Each agent will try again and the ONLY difference is the
context you write: write it so it cannot repeat any of its mistakes and does much, much better. Every
mistake gets a named, concrete correction.

{PROCESS_BRIEF}

USE THE TOOLS IN YOUR FEEDBACK. The agents have the tools listed above. Tell each agent which tool
calls to make and how: e.g. refine re-entrant edge 37 with {{edge: 37, size, radius}}; check
folded_cells; read the solve's error hotspots and view the error; compare solves 1 and 2 before
submitting. If an agent ignored a tool that would have shown its mistake, say so.

WHAT EACH AGENT DID. It inspected the geometry, called mesh(h, order, refinements) (refinements: an
element size `size` at a face `group`, an `edge`, a `point` or in a `box`, growing back to h over `radius`),
called solve(setup) at most {MAX_SOLVES} times, and submitted one solve. For each run you get: its
numbers and rank; its submitted setup; its mesh request and statistics; an error map (hotspots with
their share of the error energy, nearest named group, local element size); a REGION GRID over the
whole part (per cell: element count, mean size, share of error energy, max local error); every check
point ({PROBE_DISCLOSED} disclosed to the agent, {PROBE_POINTS - PROBE_DISCLOSED} hidden; you see all) with its error; its full reasoning
and tool calls; the notes it had; images of its mesh, its error (log error-energy density, hotspots
numbered) and, when they fit, its von Mises stress next to the reference. You also get a DISAGREEMENT
MAP (per region of a common grid: spread of the runs' von Mises = std/mean across runs, consensus =
median, outlier agents; it needs no reference) and each agent's OWN previous run with what it was
told and what it promised.

NO REMEMBERED REFERENCE VALUES. Never quote, use or suggest a published or remembered benchmark value
(e.g. a NAFEMS target stress) or tell an agent to compare a result with one. The score is the error of
the whole field, not any point value, and a remembered number can be wrong for this model. Judge only
from the data you are given, and tell agents to choose the solve they submit by evidence from their own
solves (how the field and the error map change between meshes), never by closeness to a remembered number.

HIDDEN CHECK POINTS ARE SECRET. Never write a hidden point's id (H<i>), coordinates or exact location
in anything you output: agents read all of it. You may use them to judge runs; speak of them only in
aggregate ("hidden-point RMS error 4%") or by region/group ("near the hole edge").

SCORING AND SIZING. e = max(field energy-norm error, RMS check-point error); gain g = automesher
error / e; c = DOFs / automesher DOFs; score = (g/2)^2 / c below 2x the automesher (g < 2) and
g / (2c) above, plus up to {BEST_BONUS:g} for the uniquely smallest hidden-point error (a tie pays nobody).
Consequences: first name the BINDING term (field error or check-point error): improving the other one
does nothing. With order 2 in 2D the energy error falls ~ N^-1 (N = DOFs) on smooth fields (in 3D
~ N^-2/3), slower near singular, re-entrant or clamped corners, so refine LOCALLY there. Below the
target, uniform refinement still raises the score (g^2 grows faster than c); above it, uniform
refinement only breaks even, so only moving DOFs to where the error energy is pays. DOFs ~ volume /
h^dim times a factor for order 2. When you estimate an effect, say it is an estimate.

SETUP CHECKING (generic). Translate EVERY expression in the problem statement (geometry, thickness,
material E/nu/alpha, each support and load: group, components, sign, magnitude, units, any
temperature or body-force field) into the model's own coordinates (e.g. in an axisymmetric model
x = r, y = z; in a symmetry model only part of the load acts), then check each run's setup against
it term by term. A setup error shows as error spread over the WHOLE part rather than concentrated at
features; runs with similar meshes but very different errors, or with different setups, are a strong
signal. A wrong setup is the first thing to fix.

CONTESTED REGIONS. From the disagreement map find the regions where runs disagree most, explain WHY
(element size or order there, a singularity, a setup difference), name the general traits of such
hard regions (re-entrant corners, load/support edges, holes, thin sections, material or load jumps)
and direct agents there with a concrete size/order/radius.

CONTINUITY. For each agent check every item it promised or was told after its previous run (the
suggested mesh call, its own plan): done, partly, not done. Flag every REPEATED mistake loudly.

STYLE. Write densely: no filler, every sentence carries information and numbers from the inputs.
Every note: WHERE (group, point or region) + WHAT (size, order, radius, setup change) + EXPECTED
effect. Never generic ("refine more", "check the setup"). Tag each note with its task id, or "general"
when it transfers to any part. Push each agent to try something DIFFERENT from its peers on purpose
(ties on hidden points pay nothing) and say which experiment to spend one solve on. Never invent
numbers you were not given. Work out the reasoning first, then write the result.
"""

AGENT_JOB = """\
YOUR JOB NOW: the AgentFeedback for run {rid} (agent {agent}) only. Judge EVERYTHING it did.
- diagnosis: where its error is and why (group/region, its local size there vs what is needed, the
  share of error energy there, the binding term), and how its DOFs were spent. Cite numbers.
- step_critique: every meaningful step in order (each mesh call, each solve and its setup, the choice
  of order and sizes, what it read or ignored, the submit): what it did, verdict good / mistake /
  missed_opportunity, and the named concrete correction.
- promised_items_check: each item from its previous run (see CONTINUITY for {agent}; empty if none).
- strengths, mistakes: short and specific.
- next_mesh_call: the exact call (h, order, refinements) you expect to beat its score, sized from the
  evidence (see SCORING AND SIZING).
- notes: 8-15, mixing fix / keep / avoid / idea / highlight (a peer's idea adapted to it), each tagged
  with task "{task}" or "general".
No hidden-point ids or coordinates anywhere."""

CROSS_JOB = """\
YOUR JOB NOW: the cross-run review of task {task} (not per-agent feedback; that is written separately).
- task_summary: what separates the best runs from the worst.
- setup_warnings: every setup that differs from the statement, by run_id, with the correct form.
- contested_regions: the most contested regions of the disagreement map (most first), why the runs
  differ there, which agents differ, what to do; then name the general traits of hard regions.
- highlights: the best idea of every agent that did anything well (what, why it works, how to copy it
  with concrete parameters), plus the best ideas overall.
- shared_lessons: 6-12 lessons for every agent, concrete, tagged with task "{task}" or "general".
- bullet_ratings (optional): playbook bullets the runs cited: useful / harmful / not_useful, why.
No hidden-point ids or coordinates anywhere."""


# --------------------------------------------------------------------------- experiment allocation
# Without this, every agent on a task ends up with nearly the same recipe (results/scaling_gold_2026-10-31):
# the swarm samples one point of mesh space several times. The allocation spreads the agents out: one
# control re-runs the best recipe (so we learn how noisy a score is), variants each move ONE factor of
# it (a coordinate search around the best point), explorers try structurally different meshes.
#
# The roles and each variant's factor are fixed here in code, BEFORE the teacher calls, from the
# ranking alone. That keeps the per-agent calls parallel (no waiting for the cross-run call, which
# has the longest answer), and makes the assignments distinct by construction. The teacher then
# chooses the concrete experiment inside that slot from the evidence.

VARIANT_FACTORS = [
    "the global size h (everything else as in the best recipe)",
    "the element size at the refinements (keep their places, radii, h and order)",
    "the refinement radii, i.e. how fast the size grows back to h (keep sizes, places, h and order)",
    "the placement of ONE refinement (move one, or add one where the best recipe leaves a hotspot or "
    "contested region coarse)",
    "the element order (re-size h so the DOFs stay comparable)",
]
EXPLORER_DIRECTIONS = [
    "different refinement TARGETS: refine regions the best recipes leave coarse (contested regions, error "
    "hotspots) and leave alone what they all refine",
    "a different STRATEGY: e.g. a coarse h with many small local refinements, edge or box refinements "
    "instead of points, or one strongly graded refinement",
    "a different element ORDER, with every size re-chosen for it",
]
# Flow tasks have no element order (finite-volume cells): the cell structure takes its place. Ordered
# by what a flow recipe gains most from, since a task often has only 2-4 agents (--split) and the
# factors / directions are handed out in this order: the explorers try another cell structure first.
VARIANT_FACTORS_FLOW = [
    "the global size h (everything else as in the best recipe)",
    "the wall boundary layer: its first-cell height, growth or number of layers (or add one along the walls "
    "with the steepest velocity gradients if the best recipe has none)",
    VARIANT_FACTORS[3],
    "the cell type: quads vs triangles; in 3D the sweep (its layers or grading, or a sweep instead of "
    "tetrahedra on a straight duct or pipe)",
    "the cell size at the refinements (keep their places, radii, h, cell type and boundary layer)",
    "the refinement radii, i.e. how fast the size grows back to h (keep sizes, places, h, cell type and "
    "boundary layer)",
]
EXPLORER_DIRECTIONS_FLOW = [
    "a different CELL STRUCTURE: quads instead of triangles (or back), a wall boundary layer, or in 3D a sweep "
    "of an end face, with every size re-chosen for it",
] + EXPLORER_DIRECTIONS[:2]


def allocate(agents: list[str], flow: bool = False) -> list[tuple[str, str, str]]:
    """[(agent, role, focus)] for the agents on one task, best run first. Fewer than 2 agents: none
    (a lone agent, e.g. the single-agent baseline, keeps the plain feedback). The best run's agent
    is the control; the lowest-ranked n//3 (at least 1 when n >= 3, more if the factors run out) are
    explorers, since they have least to lose; the rest are variants, one factor each. flow: a flow
    task, whose factors and directions vary the cell structure instead of the element order."""
    factors, directions = (VARIANT_FACTORS_FLOW, EXPLORER_DIRECTIONS_FLOW) if flow else (VARIANT_FACTORS,
                                                                                         EXPLORER_DIRECTIONS)
    n = len(agents)
    if n < 2:
        return []
    n_exp = max(n // 3, n - 1 - len(factors)) if n >= 3 else 0
    out = [(agents[0], "control", "re-run the best recipe so far with at most one small, deliberate tweak")]
    out += [(a, "variant", "change only " + factors[i]) for i, a in enumerate(agents[1:n - n_exp])]
    out += [(a, "explorer", directions[i % len(directions)])
            for i, a in enumerate(agents[n - n_exp:])]
    return out


ASSIGN_JOB = """EXPERIMENT ALLOCATION FOR TASK {task} next generation (roles fixed by rank before this review, so that
the agents on a task try DIFFERENT things; every per-agent review sees this same table):
{table}
YOUR ROLE: {role} (agent {agent})
- assignment: role "{role}"; experiment = ONE concrete thing to try, with groups/edges/points/boxes and
  sizes; why = the evidence for it.
  control: re-run the best recipe so far on this task (normally the top run's submitted mesh call) with
  at most one small, deliberate tweak, so its score measures run-to-run noise.
  variant: the best recipe with exactly ONE factor changed, the factor named for you in the table;
  state the new value. Everything else stays as in the best recipe.
  explorer: a structurally different mesh that no run above tried, in the direction named for you,
  chosen from the error maps and contested regions; not a resized copy of anyone's recipe.
  If another agent in the table has the same factor or direction, choose a different change from the
  one the agent listed first would obviously choose.
- next_mesh_call must implement the assignment exactly, and the step corrections and notes must not
  contradict it."""


# --------------------------------------------------------------------------- nonlinear tasks
# A task scored by its nonlinear force curve against a physical test (ace.nonlinear): the field
# error, check points and disagreement map of the SYSTEM prompt do not exist there. This block
# goes at the top of the prefix and says what replaces them.

_NL_TASK = config.TASKS.get("abc_00000065_compression") or {}    # the one nonlinear task: its budget
NL_SCORING = f"""\
THIS TASK IS SCORED BY A PHYSICAL TEST, NOT BY A REFERENCE FIELD. This overrides SCORING AND SIZING,
the check points, the disagreement map and SETUP CHECKING above.
- Budget here (not the one above): up to {_NL_TASK.get("max_solves", MAX_SOLVES)} linear solves (a fast proxy) and
  {_NL_TASK.get("nl_previews", 0)} nonlinear_check previews (the frozen nonlinear ramp of a chosen mesh, as scored).
- The analysis setup is FROZEN by the scorer (the agents do not choose it and their own setup is not
  scored): each run's scored MESH is run through one nonlinear CalculiX ramp (J2 plasticity from the
  resin's dogbone test, large deformation, sticking platens on the prong tops, base held), and the
  force F is read against crosshead travel u = u_part + delta + c*F (delta, c frozen).
- error e = mean over the checkpoints of |F_sim - F_test| / F_test; gain g = automesher error /
  max(e, 0.005); A = (g/2)^2 below g = 2, g/2 above; c = DOFs / automesher DOFs; score = A / c, plus
  the diversity bonus (no be-the-best bonus). So the binding term is always the curve error.
- Every run's block gives the five checkpoints (sim vs test), its mean error, nodes/DOFs, and the
  hotspots at the end of the ramp: where the equivalent plastic strain (PEEQ) and the largest
  principal stress peak, with their nearest faces. Its images are its force curve against the test
  and its deformed part coloured by PEEQ (the same camera for every run, so they compare directly).
- From the experiment analysis: the converged mesh is about 1 % from the test (model error: no mesh beats
  it); a coarse mesh lies ABOVE the test at every checkpoint (too stiff and too strong); the
  plastic zones decide the curve after yield. Size elements where the plastic strain is.
- Mesh calls (h, size, radius, point, box) are in METRES (uniform 4 mm = h 0.004; x 12-31 mm = box x 0.012..0.031); the brief and the nonlinear hotspots (xyz_mm) give mm.
- No hidden check points and no disagreement map exist on this task: ignore those instructions."""


def nonlinear_text(nl: dict | None) -> str:
    """A run's nonlinear check (checks["nonlinear"], ace.nonlinear.run_curve) as a few lines."""
    if not nl:
        return "nonlinear check: none"
    if not nl.get("ok"):
        return f"nonlinear check FAILED: {nl.get('error')}"
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"      # noqa: E731
    L = [f"nonlinear ramp: {nl.get('nodes')} nodes, {nl.get('dofs')} DOFs, {_fmt(nl.get('wall_s'))} s"
         + ("; " + str(nl.get("straightened_edges", "")) + " midside nodes CalculiX rejected were straightened" if nl.get("straightened") else ""),
         f"mean error vs the test {pct(nl.get('mean_rel_err'))}; peak PEEQ {_fmt(nl.get('peeq_max'))}",
         "checkpoint (crosshead mm) | test N | sim N | error (sim - test) / test"]
    for u, fm, fs, e in zip(nl.get("checkpoints_mm") or [], nl.get("force_meas_N") or [],
                            nl.get("force_sim_N") or [], nl.get("rel_err") or []):
        L.append(f"  {_fmt(u)} | {_fmt(fm)} | {_fmt(fs)} | {pct(e)}")
    for k, h in enumerate(nl.get("hotspots") or [], 1):
        xyz = ", ".join(_fmt(x) for x in h.get("xyz_mm") or [])
        L.append(f"  hotspot #{k} ({h.get('kind')}) at ({xyz}) mm, faces {', '.join(h.get('faces') or []) or '-'}: "
                 f"{_fmt(h.get('value'))}")
    return "\n".join(L)


def nl_history_text(runs: list[dict]) -> str:
    """Error vs nodes of every run so far on the task, one line per generation (best run first)."""
    by_gen: dict[int, list] = {}
    for r in runs:
        nl = (r.get("checks") or {}).get("nonlinear") or {}
        if not r.get("error") and nl.get("ok"):
            by_gen.setdefault(r["generation"], []).append(r)
    if not by_gen:
        return "(no scored runs yet)"
    lines = []
    for g in sorted(by_gen):
        rs = sorted(by_gen[g], key=lambda r: -((r["checks"].get("score")) or 0))
        lines.append(f"gen {g}: " + "; ".join(
            f"{agent_name(r['run_id'])} {100 * run_error(r):.2f}% at {r['checks']['nonlinear'].get('nodes')} nodes "
            f"(score {_fmt(r['checks'].get('score'))})" for r in rs))
    c = next((r["checks"] for rs in by_gen.values() for r in rs if r["checks"].get("auto_error") is not None), None)
    if c is not None:
        lines.append(f"automesher: {100 * c['auto_error']:.2f}% at {c.get('auto_dofs')} DOFs")
    return "\n".join(lines)


def _run_block(run: dict, report: dict, notes_text: str) -> str:
    rid = run["run_id"]
    cited = run.get("cited") or []
    nl = (run.get("checks") or {}).get("nonlinear")
    report_txt = report_as_text(report, REPORT_CHARS)
    if nl and (report.get("error") or not report.get("nonlinear")):   # the curve, even without a report
        report_txt += "\n" + nonlinear_text(nl)
    return (f"\n=== RUN {rid} (agent {agent_name(rid)}) ===\n"
            f"--- report ---\n{report_txt}\n"
            f"--- notes it had (playbook + own notes) ---\n{(notes_text or '(none)')[:NOTES_CHARS]}\n"
            f"--- bullets it cited --- {', '.join(cited) or '(none)'}\n"
            f"--- its steps (reasoning and every tool call) ---\n{trace_excerpt(run, TRACE_CHARS)}\n")


def disagreement_text(dmap: dict | None, max_cells: int = 30) -> str:
    """The disagreement map (ace.run_report.disagreement_map) as lines, most contested cells first."""
    if not dmap:
        return "(not available)"
    try:
        grid = sorted(dmap.get("grid") or [], key=lambda c: -(c.get("spread") or 0))
        q = dmap.get("quantity") or "von Mises"            # "speed" on a flow task
        lines = [f"region | centre | spread (std/mean of {q} across runs) | consensus {q} | outliers"]
        lines += [f"{c.get('region')} | {_short(c.get('centre'), 60)} | {_fmt(c.get('spread'))} | "
                  f"{_fmt(c.get('consensus_vm'))} | {', '.join(map(str, c.get('outliers') or [])) or '-'}"
                  for c in grid[:max_cells]]
        if len(grid) > max_cells:
            lines.append(f"... {len(grid) - max_cells} less contested cells omitted")
        if dmap.get("most_contested"):
            lines.append(f"most contested: {_short(dmap['most_contested'], 1500)}")
        return "\n".join(lines)
    except Exception:                    # noqa: BLE001  (a malformed map must not stop teaching)
        return _short(dmap, 4000)


def continuity_text(last_by_agent: dict | None, agents: list[str]) -> str:
    """Each agent's OWN previous run (any task) and what it was told / promised after it."""
    if not last_by_agent:
        return "(none: no agent has a previous deep run)"
    out = []
    for a in agents:
        last = last_by_agent.get(a)
        if not last:
            out.append(f"{a}: (no previous run)")
            continue
        out.append(f"{a}: " + "; ".join(f"{k}: {_short(v, 1200)}" for k, v in last.items() if v not in (None, "", [])))
    return "\n".join(out)


def build_prefix(task: str, reports: list[dict], runs: list[dict], notes_by_agent: dict[str, str],
                 shared_text: str, prev_feedback: dict | None, disagreement: dict | None,
                 last_by_agent: dict | None, history: list[dict] | None = None) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """The shared user-message prefix of every teacher call (its last block marked for caching), and
    the (run, report) pairs best first. On a nonlinear task (is_nonlinear) it also holds the
    teacher brief, the scoring that replaces the field error, and the error vs nodes of every run so
    far on the task (history: those runs, this generation's included)."""
    nl, flow = is_nonlinear(task), is_flow_task(task)
    by_id = {r["run_id"]: r for r in reports}
    pairs = [(run, by_id.get(run["run_id"], {"run_id": run["run_id"], "error": "no report"})) for run in runs]
    pairs.sort(key=lambda p: -(_numbers(p[1], p[0])["score"] or 0))      # best first

    from ace import evidence
    test = evidence.teacher_section(task, runs)          # physical test data, when the loop has some
    head = [f"TASK {task}\n--- problem statement as the agents saw it ---\n{_task_statement(task)}\n",
            *([f"--- this is a flow task ---\n{FLOW_BRIEF}"] if flow else []),
            *([f"--- physical test of this setup ---\n{test}\n"] if test else []),
            *([f"--- the experiment analysis (teacher brief) ---\n{teacher_brief(task) or '(none)'}\n",
               f"--- how this task is scored ---\n{NL_SCORING}\n",
               f"--- error vs nodes, every run so far on this task ---\n{nl_history_text(history or runs)}\n"]
              if nl else []),
            f"--- shared playbook (all agents saw it) ---\n{shared_text or '(empty)'}\n",
            f"--- last generation's review of this task (mostly other agents) ---\n{prev_summary(prev_feedback)}\n",
            "--- ranking (best first) ---\nrun_id | agent | score | error | probe_error | DOFs | gain | bonus"]
    for run, rep in pairs:
        n = _numbers(rep, run)
        head.append(" | ".join([run["run_id"], agent_name(run["run_id"])] +
                               [_fmt(n[k]) for k in ("score", "error", "probe_error", "dofs", "gain", "bonus")]))
    blocks = [text_block("\n".join(head))]
    blocks += [text_block(_run_block(run, rep, notes_by_agent.get(agent_name(run["run_id"]), "")))
               for run, rep in pairs]
    if not nl:                                       # no von Mises field to compare on a nonlinear task
        blocks.append(text_block(f"\n=== DISAGREEMENT MAP (across these runs) ===\n{disagreement_text(disagreement)}\n"))
    agents = [agent_name(run["run_id"]) for run, _ in pairs]
    blocks.append(text_block(f"\n=== CONTINUITY: each agent's own previous run and what it was told/promised ===\n"
                             f"{continuity_text(last_by_agent, agents)}\n"))

    # Images: mesh + error of every run, then stress pictures while the total stays within MAX_IMAGES.
    # Nonlinear task: each run's force curve vs the test and its PEEQ picture instead.
    first, more = (("curve", "peeq"), ()) if nl else (("mesh", "error"), ("stress", "speed"))   # speed: flow
    chosen = [(run["run_id"], k, path) for run, rep in pairs for k, path in report_images(rep, first)]
    for run, rep in pairs:
        for k, path in report_images(rep, more):
            if len(chosen) < MAX_IMAGES:
                chosen.append((run["run_id"], k, path))
    if chosen:
        blocks.append(text_block("\n=== IMAGES ===  (each is labelled with its run_id and kind)"))
    for rid, k, path in chosen:
        try:
            img = image_block(path)
        except OSError:
            continue                                   # a missing picture is not worth failing over
        blocks += [text_block(f"{rid}: {k}"), img]
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}   # end of the shared prefix
    return blocks, pairs


def teach(task: str, reports: list[dict], runs: list[dict], notes_by_agent: dict[str, str], shared_text: str,
          prev_feedback: dict | None, llm, records: list, disagreement: dict | None = None,
          last_by_agent: dict | None = None, history: list[dict] | None = None) -> TeacherFeedback | None:
    """The teacher for one task: warm the cache with the shared prefix, then one call per run and one
    cross-run call in parallel; merged. None if nothing validated. history: every run so far on the
    task (nonlinear tasks: the error-vs-nodes summary)."""
    prefix, pairs = build_prefix(task, reports, runs, notes_by_agent, shared_text, prev_feedback,
                                 disagreement, last_by_agent, history)
    cross_user = prefix + [text_block(CROSS_JOB.format(task=task))]

    rids = [run["run_id"] for run, _ in pairs]
    flow = is_flow_task(task)                    # FlowMeshCall instead of MeshCall; cell structure, not order
    roles = allocate([agent_name(r) for r in rids], flow)          # best first, like rids
    table = "\n".join(f"  {a}: {role}: {focus}" for a, role, focus in roles)
    role_of = {a: role for a, role, _ in roles}
    if flow:
        schema = FlowAssignedFeedback if roles else FlowAgentFeedback
    else:
        schema = AssignedFeedback if roles else AgentFeedback

    def agent_user(rid: str) -> list[dict]:
        a = agent_name(rid)
        job = AGENT_JOB.format(rid=rid, agent=a, task=task)
        if flow:
            job = job.replace("(h, order, refinements)", "(h, quads, refinements, boundary_layer, sweep)")
        if a in role_of:
            job += "\n" + ASSIGN_JOB.format(task=task, table=table, role=role_of[a], agent=a)
        return prefix + [text_block(f"FOR RUN {rid}\n" + job)]

    def call(user, schema):
        try:
            return llm.structured("teacher", SYSTEM, user, schema, records)[0]
        except Exception as e:           # noqa: BLE001  (one failed call must not lose the others)
            print(f"  teacher {task}: a {schema.__name__} call failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=len(rids) + 1) as pool:
        cross = pool.submit(call, cross_user, CrossRunFeedback)   # different schema: its own cache, start now
        # Kept blocking on purpose: a cache entry is readable only once its writer's answer has begun,
        # so per-agent calls sent at the same moment would each pay the full long prefix (images,
        # reports) instead of ~5% of it. The wait is mostly the prefix's processing, which every
        # call would otherwise do itself, so dropping it saves only a second or two.
        if len(rids) > 1:
            llm.warm_cache("teacher", SYSTEM, agent_user(rids[0]), schema, records)
        per = [pool.submit(call, agent_user(rid), schema) for rid in rids]
        per_agent = [f.result() for f in per]
        cross_fb = cross.result()

    got = []
    for rid, af in zip(rids, per_agent):
        if af is not None:
            af.run_id, af.agent = rid, agent_name(rid)   # the caller's ids, whatever the model wrote
            if af.agent not in role_of:                   # a lone agent gets no role: drop any the model wrote
                af.assignment = None
            elif af.assignment is not None:               # and the role allocate() gave, likewise
                af.assignment.agent, af.assignment.role = af.agent, role_of[af.agent]
            got.append(af)
    if not got and cross_fb is None:
        return None
    base = cross_fb.model_dump() if cross_fb is not None else {
        "task_summary": "(the cross-run review failed this generation)", "shared_lessons": []}
    return (FlowTeacherFeedback if flow else TeacherFeedback)(
        task=task, per_agent=got, assignments=[af.assignment for af in got if af.assignment], **base)


assert config.SECTIONS.keys() == set(Section.__args__)   # NoteItem.section must match the playbook
