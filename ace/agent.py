"""The Generator: a minimal tool-use FEA / CFD agent.

It sees a problem statement, six tools (inspect_geometry, mesh, solve, view, compare, submit) and
a system prompt = BASE_PROMPT + the current playbook. One call to run_agent = one run; it returns a
JSON-able run record (trace, meshes, solves, submitted solve, cited bullet IDs, token usage) for
the checks and the Reflector. Flow problems (physics "flow") get BASE_PROMPT_FLOW and TOOLS_FLOW:
the same four tools, with a finite-volume mesh (no element order; optional quads in 2D) and a flow
setup (fluid, convection scheme, flow BCs).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ace import fea, geometry_info, solve_feedback, ui_bridge
from ace.checks import tolerance
from ace.config import MAX_SOLVES, MAX_TURNS, MIN_QUALITY, SCORE, TASKS

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

Tools, and how a careful engineer uses them:
- inspect_geometry: the named groups (faces f<tag>), each face's type, area, centroid and neighbours,
  the CAD edges (curve tags) with their dihedral angles, the re-entrant (concave) edges, which are
  likely stress singularities, the smallest features, and a short plain-words summary. Read it first:
  find the loaded and supported faces and the re-entrant edges near them.
- mesh: a global size h plus local refinements at a face group, a CAD edge (curve tag), a point or a
  box. Put the small elements where the stress changes fast (re-entrant edges, fillets, small holes,
  edges of loaded or supported faces), not everywhere. It reports nodes, element quality, the mean
  element size near each group, and curved cells that folded: those are straightened automatically
  ("repaired_cells"); "folded_cells" must be 0 for a valid solve.
- solve: besides the solve number, feedback without any reference: every quantity the statement
  defines, support reactions vs the applied loads (equilibrium), the largest displacement and von
  Mises stress and where, and an error estimate (ZZ: the global relative energy-norm error and the 5
  hotspots holding most of it, with the nearest face and re-entrant edge and the local element size).
  Use the hotspots to decide the next mesh: refine where the estimated error is, coarsen elsewhere.
- view: a picture of a solve's mesh, stress or estimated error from a chosen direction.
- compare: how much two solves' stress fields differ, overall and in the 5 regions that differ most.
  Two meshes that agree closely are evidence of convergence; regions where they differ need refinement.
  view and compare do not use up solves.

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
        "description": "The geometry: named groups (points/curves/surfaces/volumes) with bounding boxes "
                       "and sizes (BCs and refinements refer to these names); faces (type, area, centroid, "
                       "normal if planar, adjacent faces, whether the statement names it); CAD edges (tag, "
                       "length, its two faces, dihedral angle, concave = re-entrant); the re-entrant and "
                       "sharp edges; the smallest feature size; and a plain-words summary (hints).",
        "input_schema": {"type": "object", "properties": {"using": USING}},
    },
    {
        "name": "mesh",
        "description": "Mesh the geometry. Replaces the current mesh. Returns node count, element "
                       "quality (min / 1st percentile / median; 0 = degenerate, 1 = perfect), the mean "
                       "element size near each named group, and for order 2: curved cells that folded "
                       "and were straightened (repaired_cells), cells still folded (folded_cells, must be "
                       "0) and the minimum Jacobian ratio. Quadratic (order 2) meshes have mid-side nodes, "
                       "which count toward the node cap. Node counts change in steps, so tiny changes to "
                       "h may give the same mesh.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global element size"},
                "order": {"type": "integer", "enum": [1, 2], "description": "element order"},
                "refinements": {
                    "type": "array",
                    "description": "local refinement: element size `size` at ONE of: a named `group`, "
                                   "a CAD `edge` (curve tag from inspect_geometry), a `point` [x, y] / "
                                   "[x, y, z], growing linearly to h at distance `radius`; or inside a `box` "
                                   "[xmin, ymin, zmin, xmax, ymax, zmax] (radius ignored)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string"},
                            "edge": {"type": "integer"},
                            "point": {"type": "array", "items": {"type": "number"}},
                            "box": {"type": "array", "items": {"type": "number"}},
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
                       "scored quantity, the solve number and `feedback`: the statement's quantities, "
                       "reactions vs applied loads, max displacement and von Mises stress with locations, "
                       "and a ZZ error estimate (global relative energy-norm error, top 5 hotspots). "
                       + BC_HELP,
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
        "name": "view",
        "description": "Look at a picture (PNG) of a solve: its mesh, von Mises stress, or ZZ estimated "
                       "error, seen from a direction (iso = oblique 3D view, or along an axis).",
        "input_schema": {
            "type": "object",
            "properties": {
                "solve": {"type": "integer", "description": "solve number, as returned by solve"},
                "kind": {"type": "string", "enum": ["mesh", "stress", "error"]},
                "view": {"type": "string", "enum": ["iso", "x", "y", "z", "-x", "-y", "-z"]},
                "using": USING,
            },
            "required": ["solve", "kind", "using"],
        },
    },
    {
        "name": "compare",
        "description": "Compare the stress fields of two solves (e.g. a coarse and a finer mesh): their "
                       "relative energy-norm-like difference over the part and the 5 regions where they "
                       "differ most. Convergence evidence without a reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "solve_a": {"type": "integer"},
                "solve_b": {"type": "integer"},
                "using": USING,
            },
            "required": ["solve_a", "solve_b", "using"],
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

BASE_PROMPT_FLOW = BASE_PROMPT.replace(
    "You set up and run linear-elastic finite-element analyses using tools.",
    "You set up and run steady incompressible laminar flow (CFD) analyses with a finite-volume solver "
    "using tools.").replace("degrees of\nfreedom, DOFs", "degrees of\nfreedom, DOFs = cells x unknowns per cell")

BC_HELP_FLOW = (
    "BC types: velocity_inlet (`value` = velocity vector [u_x, u_y(, u_z)] in m/s, each entry a number or "
    "an expression in x, y, z such as '0.06*y*(0.01 - y)'; or one number = speed into the domain along "
    "the normal); pressure_outlet (`value` = static pressure in Pa, number or expression); wall (no slip; "
    "optional `value` = wall velocity vector for a moving wall, numbers or expressions); slip (no "
    "penetration, no shear); symmetry. Each BC applies to a named boundary `group`; boundary groups "
    "without a BC are stationary no-slip walls. A domain without a pressure_outlet is closed: its "
    "pressure level is arbitrary (only differences are meaningful)."
)

# Flow agents get inspect / mesh / solve / submit: view and compare read stress fields (ace.solve_feedback).
_TOOL = {t["name"]: t for t in TOOLS}
TOOLS_FLOW = [
    _TOOL["inspect_geometry"],
    {
        "name": "mesh",
        "description": "Mesh the geometry with straight-sided finite-volume cells (triangles / tetrahedra; "
                       "quad-dominant in 2D with quads=true). Replaces the current mesh. Returns node count and "
                       "the minimum cell quality (0 = degenerate, 1 = perfect). Node counts change in steps, so "
                       "tiny changes to h may give the same mesh.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global cell size"},
                "quads": {"type": "boolean", "description": "2D only: recombine triangles into quads (default false)"},
                "refinements": _TOOL["mesh"]["input_schema"]["properties"]["refinements"],
                "using": USING,
            },
            "required": ["h", "using"],
        },
    },
    {
        "name": "solve",
        "description": "Solve on the current mesh with the fixed steady laminar flow solver (finite volumes, "
                       "SIMPLE) and return the scored quantity and the solve number. " + BC_HELP_FLOW,
        "input_schema": {
            "type": "object",
            "properties": {
                "formulation": {"type": "string", "enum": ["2d", "3d"]},
                "thickness": {"type": "number", "description": "2D: out-of-plane depth (forces and flow rates are per it)"},
                "fluid": {
                    "type": "object",
                    "properties": {"rho": {"type": "number", "description": "density, kg/m^3"},
                                   "mu": {"type": "number", "description": "dynamic viscosity, Pa s"}},
                    "required": ["rho", "mu"],
                },
                "convection": {"type": "string", "enum": ["upwind", "linearUpwind"],
                               "description": "convection scheme: upwind (first order, robust) or linearUpwind "
                                              "(second order, default)"},
                "bcs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["velocity_inlet", "pressure_outlet", "wall",
                                                                "slip", "symmetry"]},
                            "group": {"type": "string"},
                            "value": {},
                        },
                        "required": ["type", "group"],
                    },
                },
                "using": USING,
            },
            "required": ["formulation", "fluid", "bcs", "using"],
        },
    },
    _TOOL["submit"],
]

# Checkpoint bookkeeping kept in the run record but not shown to the agent.
HIDDEN_KEYS = {"path", "request", "index", "sha256", "gmsh", "info"}

CITATION = re.compile(r"\[using ((?:[ap]\d+-)?[a-z]+-\d{4})\]")
BULLET_ID = re.compile(r"^(?:[ap]\d+-)?[a-z]+-\d{4}$")       # shared "mesh-0001" / "p1-mesh-0001" (task 1), private "a3-mesh-0001"


def system_prompt(playbook_text: str, flow: bool = False) -> str:
    return (BASE_PROMPT_FLOW if flow else BASE_PROMPT) + "\n" + (playbook_text or "(the playbook is empty)")


def tools_for(problem) -> list[dict]:
    return TOOLS_FLOW if fea.is_flow(problem) else TOOLS


def task_prompt(problem_id: str, max_solves: int | None = None, bonus: bool = True) -> str:
    """The task statement, scoring and limits. max_solves: the solve budget stated (default
    MAX_SOLVES); bonus=False leaves out the be-the-best paragraph (the single-agent baseline,
    ace/single_loop.py, has no peers to beat on a task)."""
    max_solves = MAX_SOLVES if max_solves is None else max_solves
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    q = fea.official_qoi(problem, task["qoi"])
    where = f" at point {q.point}" if q.point else (f" over group {q.group!r}" if q.group else "")
    if q.point2:
        where += f" minus its value at point {q.point2}"
    if SCORE == "field" and not fea.is_flow(problem):    # field scoring is for solids
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
        f"target), and its minimum element quality (quality.min) must be at least {MIN_QUALITY}. You may call solve "
        f"at most {max_solves} times."
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
        self.results: dict = {}             # solve number -> feacore Result (in memory only, never in the record)
        self.n_views = 0
        self.ui = ui                        # ace.ui_bridge context of this run; None = off

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Returns (result, is_error)."""
        if name == "inspect_geometry":
            # The old group list (names BCs refer to) plus the face/edge description.
            return {**fea.geometry_summary(self.problem), **geometry_info.describe(self.problem)}, False
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
        if name in ("view", "compare"):
            try:
                return self._view(args) if name == "view" else self._compare(args)
            except Exception as e:  # noqa: BLE001  a plotting/sampling failure is the agent's information, not a crash
                return {"error": f"{name} failed: {type(e).__name__}: {e}"}, True
        return {"error": f"unknown tool {name!r}"}, True

    def _mesh(self, args: dict) -> tuple[dict, bool]:
        path = self.workdir / f"mesh_{len(self.meshes)}.msh"
        m = fea.make_mesh(self.problem, path, h=args.get("h", 0), order=args.get("order", 1),
                          refinements=args.get("refinements"), quads=bool(args.get("quads", False)))
        # Checkpoint: the request that produced this mesh, so it can be rebuilt or varied later.
        m["request"] = {k: v for k, v in args.items() if k != "using"}
        m["index"] = len(self.meshes)
        self.meshes.append(m)
        out = {k: v for k, v in m.items() if k not in HIDDEN_KEYS}
        if not m["ok"]:
            return out, True
        cap = self.task["node_cap"]
        out.update(node_cap=cap, within_cap=m["nodes"] <= cap)
        if m.get("repaired_cells"):
            out["message"] = (f"{m['repaired_cells']} curved (order 2) cells were folded or nearly folded and were "
                              f"straightened to keep the mesh valid; this is automatic and fine")
        if m["min_quality"] < MIN_QUALITY:
            # The scoring gate (ace.checks): a solve on this mesh cannot be scored.
            out["warning_quality"] = (f"min element quality {m['min_quality']} is below the gate {MIN_QUALITY}: a run "
                                      f"submitted on this mesh is invalid. Change h or the refinement sizes/radii a "
                                      f"little (smoother size transitions help)")
        if m.get("folded_cells"):
            out["warning"] = (f"{m['folded_cells']} cells are still folded (inverted): a solve on this mesh is invalid. "
                              f"Change h or the refinement sizes near curved faces, or use order 1")
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
        result = fea.run_solve(self.problem, mesh, setup, self.task["qoi"], keep_result=True)
        res = result.pop("_result", None)    # the feacore Result: kept in memory, never serialised
        self.solves.append({"mesh": mesh, "mesh_index": len(self.meshes) - 1, "setup": setup, "result": result})
        n = len(self.solves)
        ui_bridge.record_solve(self.ui, self, res)
        out = {**{k: v for k, v in result.items() if k not in HIDDEN_KEYS}, "solve": n, "nodes": mesh["nodes"],
               "solves_left": self.max_solves - n}
        if res is not None:
            self.results[n] = res
            if not fea.is_flow(self.problem):           # the feedback is stress-based (elasticity)
                try:
                    out["feedback"] = solve_feedback.feedback(self.problem, mesh, setup, res)
                except Exception as e:  # noqa: BLE001  the solve itself stands; say why there is no feedback
                    out["feedback"] = {"error": f"feedback failed: {type(e).__name__}: {e}"}
        return out, not result["ok"]

    def _result_of(self, n) -> tuple:
        """(feacore Result of solve n, None) or (None, an error for the agent)."""
        if n not in self.results:
            return None, ({"error": f"solve {n!r} is not a successful solve; successful solves: "
                                    f"{sorted(self.results)}"}, True)
        return self.results[n], None

    def _view(self, args: dict) -> tuple[dict, bool]:
        n, kind, view = args.get("solve"), args.get("kind", "stress"), args.get("view", "iso")
        res, err = self._result_of(n)
        if err:
            return err
        self.n_views += 1
        png = self.workdir / f"view_{self.n_views}_solve{n}_{kind}_{view.replace('-', 'm')}.png"
        path = solve_feedback.render(self.problem, self.solves[n - 1]["mesh"], res, kind, view, png)
        # tool_loop attaches the PNG at "image" as an image block; the record keeps only the path.
        return {"solve": n, "kind": kind, "view": view, "image": str(path)}, False

    def _compare(self, args: dict) -> tuple[dict, bool]:
        a, b = args.get("solve_a"), args.get("solve_b")
        ra, err = self._result_of(a)
        if err:
            return err
        rb, err = self._result_of(b)
        if err:
            return err
        return {"solve_a": a, "solve_b": b, **solve_feedback.compare(self.problem, ra, rb)}, False


def tool_result_content(out: dict):
    """A tool result as Anthropic tool_result content: its JSON text, plus the PNG at out["image"]
    (from the view tool) as an image block. The trace keeps `out` itself, i.e. the path, not the bytes."""
    text = json.dumps(out)
    if not (isinstance(out, dict) and out.get("image")):
        return text
    data = base64.b64encode(Path(out["image"]).read_bytes()).decode()
    return [{"type": "text", "text": text},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]


def _block_to_trace(block) -> dict | None:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_call", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "thinking" and getattr(block, "thinking", ""):
        return {"type": "thinking", "text": block.thinking}   # a summary of the reasoning
    return None


def tool_loop(llm, system: str, first_message: str, tool_specs: list[dict], call, finished,
              max_turns: int, run_id: str, usage: list, role: str = "generator",
              on_stop=None) -> tuple[list[dict], str]:
    """The tool-use conversation: ask the model, run its tool calls with `call(name, args) ->
    (result, is_error)`, send the results back, until it stops calling tools, `finished()` is true
    after a turn's calls, or max_turns. Shared by run_agent and the single-agent baseline
    (ace/single_loop.py). Fills `usage` (one record per model call). Returns (trace, stop).

    `on_stop(stop_reason) -> str | None` (the baseline only; None for the swarm, which then ends the
    run as before): when the model stops without finishing, a returned string is sent as a user
    message and the conversation goes on (the next model call counts as a turn)."""
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
            nudge = on_stop(resp.stop_reason) if on_stop else None
            if not nudge:
                stop = resp.stop_reason
                break
            # Go on. A tool call cut off by max_tokens still needs a tool_result before the text.
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": "not run: the turn was cut off",
                 "is_error": True} for b in calls] + [{"type": "text", "text": nudge}]})
            continue
        results = []
        for b in calls:  # all results go back in ONE user message
            out, is_err = call(b.name, b.input or {})
            trace.append({"type": "tool_result", "id": b.id, "name": b.name, "result": out,
                          "is_error": is_err, "turn": turn})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": tool_result_content(out), "is_error": is_err})
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
    flow = fea.is_flow(tools.problem)                 # flow problems get the CFD prompt and tools
    trace, stop = tool_loop(llm, system_prompt(playbook_text, flow=flow), task_prompt(problem_id),
                            tools_for(tools.problem), tools.call, lambda: tools.submitted is not None,
                            MAX_TURNS, run_id, usage)

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
