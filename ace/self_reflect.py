"""Self-reflection (--deep): each agent looks at its own final results and its WHOLE mesh.

After scoring, every run gets one call in which the agent (the same kind of model, in the first
person) sees what the teacher sees about it alone: its numbers, setup, mesh request, where its error
is over the whole part (hotspots, region grid, check points), pictures of its mesh, error and
stress, and its own steps. It writes notes to its future self; ace/organizer.py files them together
with the teacher's.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ace.config import MAX_SOLVES
from ace.deep_teacher import (PROCESS_BRIEF, MeshCall, NoteItem, agent_name, report_as_text, report_images,
                              trace_excerpt)
from ace.llm import image_block, text_block

REPORT_CHARS = 8000      # one run only, so it can be read in more detail than in the teacher's call
TRACE_CHARS = 4000


class SelfReflection(BaseModel):
    what_happened: str = Field(description="first person: what I did, the result, and why it came out that way")
    what_i_learned: list[NoteItem] = Field(description="5-10 notes to my future self, each concrete: where, "
                                                       "what to change, by how much, expected effect")
    plan_next: str = Field(description="what I will deliberately change next generation, and which solve "
                                       "I will spend on an experiment")
    next_mesh_call: MeshCall = Field(description="the exact first mesh call I will try next time on this task")


SYSTEM = f"""\
You are an FEA meshing agent looking back at your own run, with full information you did not have
while working: your scored results, where your stress error actually is over the whole part, every
check point (including the hidden ones), and pictures of your mesh, your error and your stress next
to the reference. Write notes to your FUTURE SELF, in the first person ("I refined ... ; next time I
will ..."). Your future self will see these notes before its next run (possibly on another task),
so make them concrete and transferable.

{PROCESS_BRIEF}

How to reflect:
1. Find where your error is: which hotspots, which cells of the region grid hold most of the error
   energy, what element size you had there, and which check points were worst. Compare that with
   where you actually spent your elements (the grid's element counts and sizes, the mesh picture).
2. Check your setup against the problem statement (loads, supports, temperature field, material,
   formulation). A wrong setup outweighs any mesh choice.
3. Judge your decisions step by step (you had {MAX_SOLVES} solves): which ones paid, which wasted
   DOFs or solves, which notes you followed and whether they helped.
4. Write 5-10 notes, each with a location (group, point or grid region) + a change (size, order,
   radius, setup) + the expected effect, citing your numbers. Never generic ("refine more").
5. Plan the next run: one deliberate change you will make (and why), the experiment you will spend
   a solve on, and the exact first mesh call. Be bold enough to be different from the other agents:
   ties on the hidden check points pay nothing.
"""


def build_user(report: dict, run: dict, own_notes_text: str) -> list[dict]:
    rid = run["run_id"]
    blocks = [text_block(
        f"MY RUN {rid} (I am agent {agent_name(rid)}, task {run.get('problem_id')})\n"
        f"--- my final results and whole mesh ---\n{report_as_text(report, REPORT_CHARS)}\n"
        f"--- the notes I had ---\n{own_notes_text or '(none)'}\n"
        f"--- my steps ---\n{trace_excerpt(run, TRACE_CHARS)}\n")]
    for kind, path in report_images(report, ("mesh", "error", "stress")):
        try:
            blocks += [text_block(f"my {kind}"), image_block(path)]
        except OSError:
            continue
    blocks.append(text_block("Write my SelfReflection."))
    return blocks


def reflect_on_run(report: dict, run: dict, own_notes_text: str, llm, records: list) -> SelfReflection | None:
    out, _ = llm.structured("reflect", SYSTEM, build_user(report, run, own_notes_text), SelfReflection, records)
    return out
