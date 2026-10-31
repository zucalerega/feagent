"""The deep teacher (--deep): one expert review of EVERY run on a task, from everything they did.

Where ace/diagnoser.py wrote about one generic sentence per agent, this teacher sees, for each run:
its numbers and rank, its setup, its mesh request, where its error is (hotspots, a coarse grid over
the WHOLE mesh, every check point), its reasoning, the notes it had, and pictures of its mesh, its
error and its stress (ace/run_report.py builds all of that). In one structured call it compares the
runs side by side, checks every setup against the problem statement, and writes for each agent a
diagnosis, 8-15 concrete notes and an exact next mesh call, plus the best idea of every agent (so
the others can copy it) and shared lessons. ace/organizer.py then files the notes into memory.

The schema below (NoteItem, MeshCall) is shared with ace/self_reflect.py and ace/organizer.py.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from ace import config
from ace.config import BEST_BONUS, MAX_SOLVES, PROBE_DISCLOSED, PROBE_POINTS
from ace.llm import image_block, text_block
from ace.reflector import Section

# The process brief (ace/process_brief.py) explains the whole loop to every model in it. The
# default below only covers a checkout where that module is missing.
try:
    from ace.process_brief import PROCESS_BRIEF
except ImportError:
    PROCESS_BRIEF = (
        "You are part of a loop that trains FEA meshing agents over generations. Each generation the "
        "agents are split at random across tasks; each meshes and solves its task and is scored against a "
        "gmsh-default automesher on the whole stress field and at check points, aiming at 2x its accuracy "
        "for as few DOFs as possible, plus a bonus for being uniquely best on hidden points (ties pay "
        "nothing). A teacher reviews every run's whole mesh and error map, each agent reflects on its own "
        "results, and an organizer keeps each agent's notes. Notes are evidence from earlier runs: respect "
        "them, but test them. Every generation each agent must change at least one thing on purpose.")

Kind = Literal["fix", "idea", "keep", "avoid", "highlight"]

MAX_IMAGES = 12          # images per teacher call: mesh + error for every run, stress while within this
REPORT_CHARS = 6000      # report_text per run
TRACE_CHARS = 3000       # reasoning excerpt per run
NOTES_CHARS = 3000       # the notes a run had, per run


# --------------------------------------------------------------------------- output schema


class NoteItem(BaseModel):
    section: Section
    content: str = Field(description="one specific, actionable note (<= 60 words): WHERE (group, point or "
                                     "region), WHAT (size, order, refinement, setup change) and the EXPECTED "
                                     "effect, citing the evidence. Never generic ('refine more').")
    kind: Kind = Field(description="fix = correct a mistake; idea = untested idea worth a try; keep = "
                                   "something that worked, keep doing it; avoid = something that hurt; "
                                   "highlight = a peer's good idea to copy or adapt")


class Refinement(BaseModel):
    group: str | None = Field(default=None, description="a named group of the geometry, or null with a point")
    point: list[float] | None = Field(default=None, description="[x, y] or [x, y, z], or null with a group")
    size: float = Field(description="element size at the group/point")
    radius: float = Field(description="distance over which the size grows back to h")


class MeshCall(BaseModel):
    h: float
    order: int = Field(description="1 or 2")
    refinements: list[Refinement] = Field(default_factory=list)


class AgentFeedback(BaseModel):
    agent: str = Field(description='the agent, e.g. "a3" (the end of its run_id)')
    run_id: str
    diagnosis: str = Field(description="what happened and WHY, citing regions/groups/numbers from its report")
    strengths: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    next_mesh_call: MeshCall = Field(description="the exact mesh call that should beat this run's score next time")
    notes: list[NoteItem] = Field(description="8-15 agent-specific, actionable notes")


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


class TeacherFeedback(BaseModel):
    task: str
    task_summary: str = Field(description="what separates the best runs from the worst on this task")
    setup_warnings: list[str] = Field(default_factory=list,
                                      description="every setup that differs from the problem statement, by run_id")
    highlights: list[Highlight] = Field(default_factory=list,
                                        description="the best idea of every agent where anything was good, "
                                                    "plus the best ideas overall")
    per_agent: list[AgentFeedback] = Field(description="one per run")
    shared_lessons: list[NoteItem] = Field(description="6-12 lessons for every agent")
    bullet_ratings: list[BulletRating] = Field(default_factory=list,
                                               description="for playbook bullets the runs cited (optional)")


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
            lines.append(f"  [{'thought' if kind == 'thinking' else 'said'}] {_short(t.get('text', ''), 350)}")
        elif kind == "tool_call":
            args = {k: v for k, v in (t.get("input") or {}).items() if k != "using"}
            used = (t.get("input") or {}).get("using") or []
            lines.append(f"  -> {t.get('name')}({_short(args, 600)})" + (f" using {used}" if used else ""))
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
            lines.append(f"  <- {t.get('name')}: {_short(brief or out, 300)}")
    text = "\n".join(lines)
    if len(text) > max_chars:                            # keep the start and the end (the decisions)
        half = max_chars // 2
        text = text[:half] + "\n  ...[middle cut]...\n" + text[-half:]
    return text or "  (no trace)"


def _numbers(report: dict, run: dict) -> dict:
    """The run's headline numbers: from the report (add_scores), else from its checks."""
    n = report.get("numbers") or {}
    c = run.get("checks") or {}
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


def prev_summary(prev: dict | None, max_chars: int = 3000) -> str:
    """The previous generation's feedback on this task, short: what was said and what each agent was
    told to try, so the teacher can check whether the advice was followed and whether it worked."""
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


SYSTEM = f"""\
You are the TEACHER: an expert in finite element analysis and adaptive meshing, mentoring a class of
FEA meshing agents. You review every run on one task in a single pass and write feedback deep and
concrete enough that each agent can do clearly better next generation. Your feedback is the main way
the agents learn, so be generous with specific, evidence-based detail.

{PROCESS_BRIEF}

WHAT EACH AGENT DID. It inspected the geometry, called mesh(h, order, refinements) (refinements: an
element size `size` at a named `group` or at a `point`, growing linearly back to h over `radius`),
called solve(setup) at most {MAX_SOLVES} times, and submitted one solve. You get, for each run: its
numbers and rank; its submitted setup; its mesh request and mesh statistics; an error map (hotspots
of the stress error with their share of the error energy, nearest named group and local element
size); a REGION GRID over the whole part (per cell: element count, mean element size, share of the
error energy, max local error), which shows where its elements and its error are everywhere; every
check point ({PROBE_DISCLOSED} disclosed to the agent, {PROBE_POINTS - PROBE_DISCLOSED} hidden; you see all) with its error; its
reasoning and tool calls; the notes it had; and images: its mesh (element edges, named groups and
check points marked), its error (log error-energy density, top hotspots numbered) and, when they
fit, its von Mises stress next to the reference on one colour scale.

SCORING, in short: error e = max(field energy-norm error, RMS check-point error); gain g =
automesher error / e; A = (g/2)^2 below 2x the automesher, g/2 above; score = A / (DOFs / automesher
DOFs), plus up to {BEST_BONUS:g} for the smallest error on hidden check points (a tie pays nobody). So the
winning mesh puts small elements exactly where the error energy is and stays coarse elsewhere.

YOUR PROCEDURE
1. Compare all runs side by side first: scores, errors, DOFs, mesh requests (h, order, every
   refinement), region grids and images. Find what separates the best from the worst: element size
   in the high-error regions, order, wasted DOFs in low-error regions, setup mistakes.
2. Check the setup of EVERY run against the problem statement: formulation, thickness, material
   (E, nu, alpha), every support and load (group, components, sign, magnitude, units) and any
   temperature field. Translate coordinates where needed: e.g. in an axisymmetric model x = r and
   y = z, so a 3D temperature sqrt(x^2+y^2)+z becomes x + y. Report every difference in
   setup_warnings with the run_id and the correct form. A wrong setup is the first thing to fix.
3. For each run, write per_agent feedback:
   - diagnosis: where its error is and why (which group or region of the grid, the element size it
     had there vs the size needed, what share of the error energy sits there), and how its DOFs were
     spent. Cite the numbers.
   - strengths and mistakes: short, specific.
   - next_mesh_call: the exact mesh call (h, order, refinements with group or point, size, radius)
     you expect to beat its current score. Size it from the evidence: take the local size in the
     worst region and the error share there; with order p the error energy falls roughly like
     h^(2p) where the field is smooth. Keep the total DOFs in check (DOFs ~ volume / h^dim).
   - 8-15 notes. Every note must be concrete: a location (group name, point coordinates or grid
     region) + a change (size, order, radius, setup fix) + the expected effect, e.g. "Refine
     edge_inner to size 0.02 with radius 0.15 (it holds 45% of the error energy at local h 0.08);
     expect the field error to roughly halve." Never write generic notes like "refine more" or
     "check the setup". Mix kinds: fixes for its mistakes, keeps for what worked, avoids, ideas to
     test, and highlights of peers' good ideas adapted to it.
   - Push it to change something meaningful next generation, and to try something DIFFERENT from
     its peers: if everyone converges on one mesh, ties on the hidden points pay nothing. Tell
     each agent which experiment to spend one of its solves on.
4. highlights: for every agent that did anything well, its best idea: what, why it works, and how
   the others can copy or adapt it (concrete parameters). Include the best ideas overall.
5. shared_lessons: 6-12 notes for every agent, general enough to reuse on other parts but still
   concrete (what to look for, what to do, by how much).
6. bullet_ratings (optional): for playbook bullets the runs cited, whether each was useful,
   harmful or not useful here, and why.

Use the previous generation's feedback for continuity: say whether an agent followed its advice and
whether it worked, and adjust. Never invent numbers you were not given; when you estimate (e.g. the
effect of a size change), say it is an estimate. Write densely: no filler, every sentence carries
information. Work out the reasoning first, then write the result.
"""


def _run_block(run: dict, report: dict, notes_text: str) -> str:
    rid = run["run_id"]
    cited = run.get("cited") or []
    return (f"\n=== RUN {rid} (agent {agent_name(rid)}) ===\n"
            f"--- report ---\n{report_as_text(report, REPORT_CHARS)}\n"
            f"--- notes it had (playbook + own notes) ---\n{(notes_text or '(none)')[:NOTES_CHARS]}\n"
            f"--- bullets it cited --- {', '.join(cited) or '(none)'}\n"
            f"--- its steps ---\n{trace_excerpt(run, TRACE_CHARS)}\n")


def build_user(task: str, reports: list[dict], runs: list[dict], notes_by_agent: dict[str, str],
               shared_text: str, prev_feedback: dict | None) -> list[dict]:
    """The teacher's user message: problem, ranking, every run in full, then the images."""
    by_id = {r["run_id"]: r for r in reports}
    pairs = [(run, by_id.get(run["run_id"], {"run_id": run["run_id"], "error": "no report"})) for run in runs]
    pairs.sort(key=lambda p: -(_numbers(p[1], p[0])["score"] or 0))      # best first

    head = [f"TASK {task}\n--- problem statement as the agents saw it ---\n{_task_statement(task)}\n",
            f"--- shared playbook (all agents saw it) ---\n{shared_text or '(empty)'}\n",
            f"--- your feedback on this task last generation ---\n{prev_summary(prev_feedback)}\n",
            "--- ranking (best first) ---\nrun_id | agent | score | error | probe_error | DOFs | gain | bonus"]
    for run, rep in pairs:
        n = _numbers(rep, run)
        head.append(" | ".join([run["run_id"], agent_name(run["run_id"])] +
                               [_fmt(n[k]) for k in ("score", "error", "probe_error", "dofs", "gain", "bonus")]))
    blocks = [text_block("\n".join(head))]
    blocks += [text_block(_run_block(run, rep, notes_by_agent.get(agent_name(run["run_id"]), "")))
               for run, rep in pairs]

    # Images: mesh + error of every run, then stress pictures while the total stays within MAX_IMAGES.
    chosen = [(run["run_id"], k, path) for run, rep in pairs for k, path in report_images(rep, ("mesh", "error"))]
    for run, rep in pairs:
        for k, path in report_images(rep, ("stress",)):
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
    blocks.append(text_block(f"\nWrite the TeacherFeedback for task {task}: one per_agent entry for each of "
                             f"these runs: {', '.join(r['run_id'] for r, _ in pairs)}."))
    return blocks


def teach(task: str, reports: list[dict], runs: list[dict], notes_by_agent: dict[str, str], shared_text: str,
          prev_feedback: dict | None, llm, records: list) -> TeacherFeedback | None:
    """One teacher call for one task. None if the model's output did not validate twice."""
    user = build_user(task, reports, runs, notes_by_agent, shared_text, prev_feedback)
    fb, _ = llm.structured("teacher", SYSTEM, user, TeacherFeedback, records)
    if fb is not None:
        fb.task = task                                 # the caller's id, whatever the model wrote
    return fb


assert config.SECTIONS.keys() == set(Section.__args__)   # NoteItem.section must match the playbook
