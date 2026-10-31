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

from ace import fea, ui_bridge
from ace.checks import tolerance
from ace.config import MAX_SOLVES, MAX_TURNS, SCORE, TASKS

# Deliberately short and non-expert: the playbook is where expertise should accumulate.
# Keep it identical across runs (no problem-specific text) so prompt caching works.
BASE_PROMPT = """\
You set up and run linear-elastic finite-element analyses using tools.

Workflow: call inspect_geometry to see the named groups, call mesh to build a mesh, then call
solve with your analysis setup. You may re-mesh and solve again within the limits stated in the
task. When you are done, call submit with the number of the solve you want scored; that ends
the run. If you never submit, your last successful solve is scored. The solver computes the
scored quantity; you do not report it.

How you are scored: the submitted solve must be ACCURATE ENOUGH (the scored quantity within the
tolerance stated in the task of the true value, which you never see) and CHEAP (few degrees of
freedom, DOFs). The best score goes to the cheapest mesh that is still within tolerance:
missing the tolerance loses accuracy, and using more DOFs than needed loses efficiency. Only the
submitted solve counts, so extra solves on other meshes cost nothing. Judge accuracy from your
own meshes and solves; do not rely on reference values or closed-form estimates you remember,
because the problem may differ from any textbook case.

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
                    "properties": {"E": {"type": "number"}, "nu": {"type": "number"},
                                   "alpha": {"type": "number", "description": "thermal expansion coefficient (1/K), "
                                                                             "needed only with a temperature load"}},
                    "required": ["E", "nu"],
                },
                "loads": {
                    "type": "array",
                    "description": "volume loads (optional): temperature = a temperature field T (number or an "
                                   "expression in x, y, z, e.g. 'x + y'; axisymmetric: x = r, y = z), causing "
                                   "thermal strain alpha * (T - T_ref); body_force = force per volume vector",
                    "items": {
                        "type": "object",
                        "properties": {"type": {"type": "string", "enum": ["temperature", "body_force"]},
                                       "value": {}, "group": {"type": "string", "description": "cell group; '*' = all"}},
                        "required": ["type", "value"],
                    },
                },
                "T_ref": {"type": "number", "description": "stress-free reference temperature (default 0)"},
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

# Checkpoint bookkeeping kept in the run record but not shown to the agent.
HIDDEN_KEYS = {"path", "request", "index", "sha256", "gmsh", "info"}

CITATION = re.compile(r"\[using ((?:[ap]\d+-)?[a-z]+-\d{4})\]")
BULLET_ID = re.compile(r"^(?:[ap]\d+-)?[a-z]+-\d{4}$")       # shared "mesh-0001" / "p1-mesh-0001" (task 1), private "a3-mesh-0001"


def system_prompt(playbook_text: str) -> str:
    return BASE_PROMPT + "\n" + (playbook_text or "(the playbook is empty)")


def task_prompt(problem_id: str, max_solves: int | None = None, bonus: bool = True) -> str:
    """The task statement, scoring and limits. max_solves: the solve budget stated (default
    MAX_SOLVES); bonus=False leaves out the be-the-best paragraph (the single-agent baseline,
    ace/single_loop.py, has no peers to beat on a task)."""
    max_solves = MAX_SOLVES if max_solves is None else max_solves
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    q = fea.official_qoi(problem, task["qoi"])
    where = f" at point {q.point}" if q.point else (f" over group {q.group!r}" if q.group else "")
    if SCORE == "field":
        from ace.config import AUTO_GAIN_TARGET as G
        from ace.field_error import automesher
        a = automesher(problem_id)
        from ace.config import BEST_BONUS, PROBE_DISCLOSED, PROBE_POINTS
        from ace.field_error import probe_points
        shown = "; ".join("(" + ", ".join(f"{v:.4g}" for v in pt[:problem.dim]) + ")"
                          for pt in probe_points(problem_id)["points"][:PROBE_DISCLOSED])
        scored = (f"SCORING FOR THIS TASK (replaces 'How you are scored' above). Scored: the whole stress field, "
                  f"two ways. (1) Field error: the energy-norm error of your stress field over the entire part, "
                  f"relative to a very fine reference solution (the strain energy of the stress difference, "
                  f"integrated everywhere). (2) Point error: the RMS relative error of the von Mises stress at "
                  f"{PROBE_POINTS} check points spread across the part (away from stress singularities). "
                  f"{PROBE_DISCLOSED} of them are {shown}; the other {PROBE_POINTS - PROBE_DISCLOSED} are hidden. "
                  f"Your error e = the larger of (1) and (2). The statement's {q.name}{where} is not scored; you "
                  f"may still use it as a check.\n"
                  f"You compete against an AUTOMESHER: gmsh with its default sizes and quadratic elements (scored "
                  f"the same way), which "
                  f"gets e = {a['error']:.2%} with {a['dofs']} DOFs.\n"
                  f"TARGET: {G:g}x the automesher's accuracy, i.e. e <= {a['error'] / G:.2%}.\n"
                  f"SCORE = A / c, where g = {a['error']:.4f} / e (your accuracy gain), c = your DOFs / {a['dofs']}, "
                  f"A = (g/{G:g})^2 if g < {G:g} (missing the target is penalised quadratically) else g/{G:g} "
                  f"(beating it keeps paying). Score 1 = {G:g}x the automesher's accuracy at its DOFs; the "
                  f"automesher itself scores {1 / G ** 2:g}. Extra DOFs pay only if they buy at least proportionally "
                  f"more accuracy, so put them where the error is.\n")
        if bonus:
            scored += (
                  f"BONUS for being the best: other agents attempt this task too. On each of the "
                  f"{PROBE_POINTS - PROBE_DISCLOSED} hidden check points, the agent with the smallest stress error "
                  f"there wins a share of up to +{BEST_BONUS:g} in total. A tie pays nobody, so copying the "
                  f"leader earns nothing. Trying something that beats the others pays.\n")
    else:
        scored = (f"SCORED QUANTITY: {q.name} = {q.type} of {q.field}{where}. Only this quantity is scored; "
                  f"ignore any other quantities the statement asks you to report.\n"
                  f"TARGET: {q.name} within {tolerance(problem_id):.0%} of the true value, with as few DOFs as possible.\n")
    return (
        f"Problem id: {problem_id}\n\n{problem.description()}\n\n"
        f"{scored}"
        f"LIMITS: the mesh you solve on must have at most {task['node_cap']} nodes (a safety limit, not a "
        f"target). You may call solve at most {max_solves} times."
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

    def __init__(self, problem_id: str, workdir: Path, ui: dict | None = None, max_solves: int | None = None):
        self.problem_id = problem_id
        self.max_solves = MAX_SOLVES if max_solves is None else max_solves   # larger for the single-agent baseline
        self.task = TASKS[problem_id]
        self.problem = fea.get_problem(problem_id)
        self.workdir = Path(workdir)
        self.meshes: list[dict] = []
        self.solves: list[dict] = []
        self.submitted: int | None = None   # index into self.solves
        self.ui = ui                        # ace.ui_bridge context of this run; None = off

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
        # Checkpoint: the request that produced this mesh, so it can be rebuilt or varied later.
        m["request"] = {k: v for k, v in args.items() if k != "using"}
        m["index"] = len(self.meshes)
        self.meshes.append(m)
        out = {k: v for k, v in m.items() if k not in HIDDEN_KEYS}
        if not m["ok"]:
            return out, True
        cap = self.task["node_cap"]
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
        if len(self.solves) >= self.max_solves:
            return {"error": f"solve limit ({self.max_solves}) reached; call submit"}, True
        setup = {k: v for k, v in args.items() if k != "using"}
        feacore_result = []   # filled with the feacore Result if the solve succeeds (for ui_bridge)
        result = fea.run_solve(self.problem, mesh, setup, self.task["qoi"], on_result=feacore_result.append)
        self.solves.append({"mesh": mesh, "mesh_index": len(self.meshes) - 1, "setup": setup, "result": result})
        ui_bridge.record_solve(self.ui, self, feacore_result[0] if feacore_result else None)
        return {**{k: v for k, v in result.items() if k not in HIDDEN_KEYS}, "solve": len(self.solves), "nodes": mesh["nodes"],
                "solves_left": self.max_solves - len(self.solves)}, not result["ok"]


def _block_to_trace(block) -> dict | None:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_call", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "thinking" and getattr(block, "thinking", ""):
        return {"type": "thinking", "text": block.thinking}   # a summary of the reasoning
    return None


def tool_loop(llm, system: str, first_message: str, tool_specs: list[dict], call, finished,
              max_turns: int, run_id: str, usage: list, role: str = "generator") -> tuple[list[dict], str]:
    """The tool-use conversation: ask the model, run its tool calls with `call(name, args) ->
    (result, is_error)`, send the results back, until it stops calling tools, `finished()` is true
    after a turn's calls, or max_turns. Shared by run_agent and the single-agent baseline
    (ace/single_loop.py). Fills `usage` (one record per model call). Returns (trace, stop)."""
    messages: list = [{"role": "user", "content": first_message}]
    trace: list[dict] = []
    stop = "max_turns"
    extra = {} if role == "generator" else {"role": role}   # the generator call is unchanged
    for turn in range(max_turns):
        resp, rec = llm.agent_turn(system, messages, tool_specs, run_id=run_id, **extra)
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
            out, is_err = call(b.name, b.input or {})
            trace.append({"type": "tool_result", "id": b.id, "name": b.name, "result": out,
                          "is_error": is_err, "turn": turn})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out), "is_error": is_err})
        messages.append({"role": "user", "content": results})
        if finished():
            stop = "submitted"
            break
    return trace, stop


def run_agent(problem_id: str, run_id: str, generation: int, playbook_text: str,
              workdir: Path, llm, usage: list | None = None, ui: dict | None = None) -> dict:
    """`usage` collects one record per model call; the caller can pass its own list so the
    tokens of a run that crashes halfway are still counted. `ui`: ace.ui_bridge.settings(...)
    to also record each solve in the feacore run store (None = off)."""
    tools = Tools(problem_id, workdir, ui_bridge.for_run(ui, run_id, generation))
    usage = [] if usage is None else usage
    trace, stop = tool_loop(llm, system_prompt(playbook_text), task_prompt(problem_id), TOOLS, tools.call,
                            lambda: tools.submitted is not None, MAX_TURNS, run_id, usage)

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
