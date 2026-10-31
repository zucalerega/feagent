"""The Generator: a minimal tool-use FEA agent.

It sees a problem statement, four tools (inspect_geometry, mesh, solve, submit) and a system
prompt = BASE_PROMPT + the current playbook. One call to run_agent = one run; it returns a
JSON-able run record (trace, meshes, solves, submitted solve, cited bullet IDs, token usage) for
the checks and the Reflector.
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
task. When you are done, call submit with the number of the solve you want scored; that ends
the run. If you never submit, your last successful solve is scored. The solver computes the
scored quantity; you do not report it. You are scored on how close it is to the true value,
which you never see. Judge your result from your own meshes and solves; do not rely on
reference values or closed-form estimates you remember, because the problem may differ from
any textbook case.

Playbook: lessons learned from earlier runs, each with an ID. Every tool call has a `using`
argument: list there the IDs of the playbook lessons that shaped that call (empty if none).
"""

BC_HELP = (
    "BC types: fixed (all displacement components 0); displacement (prescribed `components` "
    "[0=x,1=y,2=z] = `value`); symmetry (zero normal displacement on a planar axis-aligned group); "
    "pressure (traction = -value * outward normal, i.e. positive pushes into the body); traction "
    "(vector `value`, force/area); force (total force vector `value` spread over the group). "
    "Each BC applies to a named `group`, or (fixed/displacement only) to the single node "
    "nearest a `point` [x, y(, z)], e.g. to remove rigid-body motion."
)

USING = {"type": "array", "items": {"type": "string"},
         "description": "IDs of playbook lessons that shaped this call, e.g. [\"mesh-0003\"]; [] if none"}

TOOLS = [
    {
        "name": "inspect_geometry",
        "description": "List the named groups (points/curves/surfaces/volumes) with bounding boxes "
                       "and sizes (length/area/volume). BCs and refinements refer to these names.",
        "input_schema": {"type": "object", "properties": {"using": USING}},
    },
    {
        "name": "mesh",
        "description": "Mesh the geometry. Replaces the current mesh. Returns node count and the "
                       "minimum element quality (0 = degenerate, 1 = perfect). Quadratic (order 2) "
                       "meshes have mid-side nodes, which count toward the node cap. Node counts "
                       "change in steps, so tiny changes to h may give the same mesh.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global element size"},
                "order": {"type": "integer", "enum": [1, 2], "description": "element order"},
                "refinements": {
                    "type": "array",
                    "description": "local refinement: element size `size` at a named `group` or a "
                                   "`point` [x, y] / [x, y, z], growing linearly to h at distance `radius`",
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
                "using": USING,
            },
            "required": ["h", "order", "using"],
        },
    },
    {
        "name": "solve",
        "description": "Solve on the current mesh with the fixed linear-elastic solver and return the "
                       "scored quantity and the solve number. " + BC_HELP,
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
                            "point": {"type": "array", "items": {"type": "number"}},
                            "components": {"type": "array", "items": {"type": "integer"}},
                            "value": {},
                        },
                        "required": ["type"],
                    },
                },
                "using": USING,
            },
            "required": ["formulation", "material", "bcs", "using"],
        },
    },
    {
        "name": "submit",
        "description": "Choose which of your successful solves is scored, and end the run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "solve": {"type": "integer", "description": "solve number, as returned by solve"},
                "using": USING,
            },
            "required": ["solve", "using"],
        },
    },
]

CITATION = re.compile(r"\[using ([a-z]+-\d{4})\]")
BULLET_ID = re.compile(r"^[a-z]+-\d{4}$")


def system_prompt(playbook_text: str) -> str:
    return BASE_PROMPT + "\n" + (playbook_text or "(the playbook is empty)")


def task_prompt(problem_id: str) -> str:
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    q = fea.official_qoi(problem, task["qoi"])
    where = f" at point {q.point}" if q.point else (f" over group {q.group!r}" if q.group else "")
    return (
        f"Problem id: {problem_id}\n\n{problem.description()}\n\n"
        f"SCORED QUANTITY: {q.name} = {q.type} of {q.field}{where}. Only this quantity is scored; "
        f"ignore any other quantities the statement asks you to report.\n"
        f"LIMITS: the mesh you solve on must have at most {task['node_cap']} nodes. "
        f"You may call solve at most {MAX_SOLVES} times."
    )


def parse_citations(trace: list[dict]) -> list[str]:
    """Bullet IDs from the `using` argument of tool calls, plus any "[using id]" in the text."""
    ids = set()
    for t in trace:
        if t["type"] == "tool_call":
            ids |= {i for i in (t["input"].get("using") or []) if isinstance(i, str) and BULLET_ID.match(i)}
        elif t["type"] in ("text", "thinking"):
            ids |= set(CITATION.findall(t["text"]))
    return sorted(ids)


class Tools:
    """Executes the agent's tool calls and keeps the run's meshes, solves and submission."""

    def __init__(self, problem_id: str, workdir: Path):
        self.problem_id = problem_id
        self.task = TASKS[problem_id]
        self.problem = fea.get_problem(problem_id)
        self.workdir = Path(workdir)
        self.meshes: list[dict] = []
        self.solves: list[dict] = []
        self.submitted: int | None = None   # index into self.solves

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Returns (result, is_error)."""
        if name == "inspect_geometry":
            return fea.geometry_summary(self.problem), False
        if name == "mesh":
            return self._mesh(args)
        if name == "solve":
            return self._solve(args)
        if name == "submit":
            n = args.get("solve")
            if not isinstance(n, int) or not 1 <= n <= len(self.solves) or not self.solves[n - 1]["result"]["ok"]:
                ok = [i + 1 for i, s in enumerate(self.solves) if s["result"]["ok"]]
                return {"error": f"solve {n!r} is not a successful solve; successful solves: {ok}"}, True
            self.submitted = n - 1
            return {"submitted": n, "message": "run finished"}, False
        return {"error": f"unknown tool {name!r}"}, True

    def _mesh(self, args: dict) -> tuple[dict, bool]:
        path = self.workdir / f"mesh_{len(self.meshes)}.msh"
        m = fea.make_mesh(self.problem, path, h=args.get("h", 0), order=args.get("order", 1),
                          refinements=args.get("refinements"))
        self.meshes.append(m)
        if not m["ok"]:
            return m, True
        cap = self.task["node_cap"]
        out = {k: v for k, v in m.items() if k != "path"}
        out.update(node_cap=cap, within_cap=m["nodes"] <= cap)
        if m["nodes"] > cap:
            # Node count scales like (1/size)^dim, so this factor on all sizes roughly hits the cap.
            # Suggest concrete values based on the sizes actually USED (requests below the size
            # floor were clamped, so scaling the requested values could give the same mesh).
            factor = (m["nodes"] / cap) ** (1 / self.problem.dim)
            sizes = ", ".join(f"{s * factor:.4g}" for s in m["sizes_used"])
            out["hint"] = (f"over the cap by {m['nodes'] - cap} nodes; try h = {m['h_used'] * factor:.4g}"
                           + (f" and refinement sizes = [{sizes}]" if sizes else "")
                           + f" (the sizes used, times {factor:.2f})")
        return out, False

    def _solve(self, args: dict) -> tuple[dict, bool]:
        if not self.meshes:
            return {"error": "no mesh yet: call mesh first"}, True
        mesh = self.meshes[-1]   # always the latest mesh call, never an older one
        if not mesh.get("ok"):
            return {"error": "the latest mesh call failed; mesh again before solving"}, True
        if mesh["nodes"] > self.task["node_cap"]:
            return {"error": f"current mesh has {mesh['nodes']} nodes, over the cap of "
                             f"{self.task['node_cap']}; re-mesh first (this did not use a solve)"}, True
        if len(self.solves) >= MAX_SOLVES:
            return {"error": f"solve limit ({MAX_SOLVES}) reached; call submit"}, True
        setup = {k: v for k, v in args.items() if k != "using"}
        result = fea.run_solve(self.problem, mesh, setup, self.task["qoi"])
        self.solves.append({"mesh": mesh, "setup": setup, "result": result})
        return {**result, "solve": len(self.solves), "nodes": mesh["nodes"],
                "solves_left": MAX_SOLVES - len(self.solves)}, not result["ok"]


def _block_to_trace(block) -> dict | None:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_call", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "thinking" and getattr(block, "thinking", ""):
        return {"type": "thinking", "text": block.thinking}   # a summary of the reasoning
    return None


def run_agent(problem_id: str, run_id: str, generation: int, playbook_text: str,
              workdir: Path, llm, usage: list | None = None) -> dict:
    """`usage` collects one record per model call; the caller can pass its own list so the
    tokens of a run that crashes halfway are still counted."""
    tools = Tools(problem_id, workdir)
    system = system_prompt(playbook_text)
    messages: list = [{"role": "user", "content": task_prompt(problem_id)}]
    trace: list[dict] = []
    usage = [] if usage is None else usage
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
        if tools.submitted is not None:
            stop = "submitted"
            break

    return {
        "problem_id": problem_id,
        "run_id": run_id,
        "generation": generation,
        "stop": stop,
        "turns": len(usage),
        "cited": parse_citations(trace),
        "submitted": tools.submitted,
        "trace": trace,
        "meshes": tools.meshes,
        "solves": tools.solves,
        "usage": usage,
    }
