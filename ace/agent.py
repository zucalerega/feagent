"""The Generator: a minimal tool-use FEA agent.

It sees a problem statement, three tools (inspect_geometry, mesh, solve) and a system prompt
= BASE_PROMPT + the current playbook. One call to run_agent = one run; it returns a JSON-able
run record (trace, meshes, solves, cited bullet IDs, token usage) for the checks and Reflector.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ace import fea
from ace.config import MAX_SOLVES, MAX_TURNS, TASKS

# Deliberately short and non-expert: the playbook is where expertise should accumulate.
# Keep it identical across runs (no problem-specific text) so prompt caching works.
BASE_PROMPT = """\
You set up and run linear-elastic finite-element analyses using tools.

Workflow: call inspect_geometry to see the named groups, call mesh to build a mesh, then call
solve with your analysis setup. You may re-mesh and solve again within the limits stated in the
task. Your answer is the scored quantity from your LAST successful solve; the solver computes it,
you do not report it. You are scored on how close it is to the true value, which you never see.
When you are finished, reply with a short summary and no tool call.

Playbook: lessons learned from earlier runs. When you apply a lesson, cite its ID inline in
your text, e.g. "[using mesh-0003]".
"""

BC_HELP = (
    "BC types: fixed (all displacement components 0); displacement (prescribed `components` "
    "[0=x,1=y,2=z] = `value`); symmetry (zero normal displacement on a planar axis-aligned group); "
    "pressure (traction = -value * outward normal, i.e. positive pushes into the body); traction "
    "(vector `value`, force/area); force (total force vector `value` spread over the group)."
)

TOOLS = [
    {
        "name": "inspect_geometry",
        "description": "List the named groups (points/curves/surfaces/volumes) with bounding boxes "
                       "and sizes (length/area/volume). BCs and refinements refer to these names.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "mesh",
        "description": "Mesh the geometry. Replaces the current mesh. Returns node count and the "
                       "minimum element quality (0 = degenerate, 1 = perfect). Quadratic (order 2) "
                       "meshes have mid-side nodes, which count toward the node cap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global element size"},
                "order": {"type": "integer", "enum": [1, 2], "description": "element order"},
                "refinements": {
                    "type": "array",
                    "description": "local refinement: element size `size` within distance `radius` "
                                   "of a named `group` or of a `point` [x, y] / [x, y, z], growing to h",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string"},
                            "point": {"type": "array", "items": {"type": "number"}},
                            "size": {"type": "number"},
                            "radius": {"type": "number"},
                        },
                        "required": ["size", "radius"],
                    },
                },
            },
            "required": ["h", "order"],
        },
    },
    {
        "name": "solve",
        "description": "Solve on the current mesh with the fixed linear-elastic solver and return the "
                       "scored quantity. " + BC_HELP,
        "input_schema": {
            "type": "object",
            "properties": {
                "formulation": {"type": "string", "enum": ["plane_stress", "plane_strain", "axisymmetric", "3d"]},
                "thickness": {"type": "number", "description": "plane stress/strain thickness"},
                "material": {
                    "type": "object",
                    "properties": {"E": {"type": "number"}, "nu": {"type": "number"}},
                    "required": ["E", "nu"],
                },
                "bcs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["fixed", "displacement", "symmetry",
                                                                "pressure", "traction", "force"]},
                            "group": {"type": "string"},
                            "components": {"type": "array", "items": {"type": "integer"}},
                            "value": {},
                        },
                        "required": ["type", "group"],
                    },
                },
            },
            "required": ["formulation", "material", "bcs"],
        },
    },
]

CITATION = re.compile(r"\[using ([a-z]+-\d{4})\]")


def system_prompt(playbook_text: str) -> str:
    return BASE_PROMPT + "\n" + (playbook_text or "(the playbook is empty)")


def task_prompt(problem_id: str) -> str:
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    q = fea.official_qoi(problem, task["qoi"])
    where = f" at point {q.point}" if q.point else ""
    return (
        f"Problem id: {problem_id}\n\n{problem.description()}\n\n"
        f"SCORED QUANTITY: {q.name} = {q.field}{where}. Only this quantity is scored; ignore any "
        f"other quantities the statement asks you to report.\n"
        f"LIMITS: the mesh you solve on must have at most {task['node_cap']} nodes. "
        f"You may call solve at most {MAX_SOLVES} times."
    )


def parse_citations(texts: list[str]) -> list[str]:
    return sorted({m for t in texts for m in CITATION.findall(t)})


class Tools:
    """Executes the agent's tool calls and keeps the run's meshes and solves."""

    def __init__(self, problem_id: str, workdir: Path):
        self.problem_id = problem_id
        self.task = TASKS[problem_id]
        self.problem = fea.get_problem(problem_id)
        self.workdir = Path(workdir)
        self.meshes: list[dict] = []
        self.solves: list[dict] = []

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Returns (result, is_error)."""
        if name == "inspect_geometry":
            return fea.geometry_summary(self.problem), False
        if name == "mesh":
            path = self.workdir / f"mesh_{len(self.meshes)}.msh"
            m = fea.make_mesh(self.problem, path, h=args.get("h", 0), order=args.get("order", 1),
                              refinements=args.get("refinements"))
            self.meshes.append(m)
            if m["ok"]:
                m = {**m, "node_cap": self.task["node_cap"], "within_cap": m["nodes"] <= self.task["node_cap"]}
                m.pop("path")
            return m, not m["ok"]
        if name == "solve":
            mesh = next((m for m in reversed(self.meshes) if m.get("ok")), None)
            if mesh is None:
                return {"error": "no mesh yet: call mesh first"}, True
            if mesh["nodes"] > self.task["node_cap"]:
                return {"error": f"current mesh has {mesh['nodes']} nodes, over the cap of "
                                 f"{self.task['node_cap']}; re-mesh first (this did not use a solve)"}, True
            if len(self.solves) >= MAX_SOLVES:
                return {"error": f"solve limit ({MAX_SOLVES}) reached; finish now"}, True
            result = fea.run_solve(self.problem, mesh, args, self.task["qoi"])
            self.solves.append({"mesh": mesh, "setup": args, "result": result})
            left = MAX_SOLVES - len(self.solves)
            return {**result, "nodes": mesh["nodes"], "solves_left": left}, not result["ok"]
        return {"error": f"unknown tool {name!r}"}, True


def _block_to_trace(block) -> dict | None:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_call", "id": block.id, "name": block.name, "input": block.input}
    return None  # thinking blocks etc. are not kept in the trace


def run_agent(problem_id: str, run_id: str, generation: int, playbook_text: str,
              workdir: Path, llm) -> dict:
    tools = Tools(problem_id, workdir)
    system = system_prompt(playbook_text)
    messages: list = [{"role": "user", "content": task_prompt(problem_id)}]
    trace: list[dict] = []
    usage: list[dict] = []
    stop = "max_turns"

    for turn in range(MAX_TURNS):
        resp, rec = llm.agent_turn(system, messages, TOOLS, run_id=run_id)
        usage.append(rec)
        # Append the response content unchanged (thinking blocks must be passed back as-is).
        messages.append({"role": "assistant", "content": resp.content})
        trace += [dict(t, turn=turn) for b in resp.content if (t := _block_to_trace(b))]

        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            stop = resp.stop_reason
            break
        results = []
        for b in calls:  # all results go back in ONE user message
            out, is_err = tools.call(b.name, b.input or {})
            trace.append({"type": "tool_result", "id": b.id, "name": b.name, "result": out,
                          "is_error": is_err, "turn": turn})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out), "is_error": is_err})
        messages.append({"role": "user", "content": results})

    texts = [t["text"] for t in trace if t["type"] == "text"]
    return {
        "problem_id": problem_id,
        "run_id": run_id,
        "generation": generation,
        "stop": stop,
        "turns": len(usage),
        "cited": parse_citations(texts),
        "trace": trace,
        "meshes": tools.meshes,
        "solves": tools.solves,
        "usage": usage,
    }
