"""The Generator: a minimal tool-use FEA / CFD agent.

It sees a problem statement, six tools (inspect_geometry, mesh, solve, view, compare, submit; plus
nonlinear_check on a task scored by a nonlinear simulation of a physical test, ace/nonlinear.py) and
a system prompt = BASE_PROMPT + the current playbook. One call to run_agent = one run; it returns a
JSON-able run record (trace, meshes, solves, submitted solve, cited bullet IDs, token usage) for
the checks and the Reflector. Flow problems (physics "flow") get BASE_PROMPT_FLOW and TOOLS_FLOW:
the same six tools, with a finite-volume mesh (no element order; quads, wall boundary layers, swept
3D meshes: ace.flow_mesh), a flow setup (fluid, convection scheme, flow BCs) and velocity-based
feedback, view and compare (ace.flow_feedback).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ace import config, fea, flow_feedback, geometry_info, solve_feedback, ui_bridge
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

TOOLS_TEXT_FLOW = """\
Tools, and how a careful engineer uses them:
- inspect_geometry: the named groups, each face's type, area, centroid and neighbours, the CAD edges
  (curve tags) and the smallest features. Read it first: find the inlets, outlets and walls.
- mesh: a global cell size h plus local refinements at a group, a CAD edge (curve tag), a point or a
  box. Put the small cells where the velocity changes fast (near walls and corners, around bodies,
  where the flow develops), not everywhere. quads=true gives quad-dominant cells (2D, or the swept
  face). boundary_layer adds thin cells along no-slip walls (first-cell height, growth per layer,
  number of layers), where the velocity gradient is steepest. In 3D, sweep builds a straight duct or
  pipe as its end face meshed in 2D and extruded through the domain in layers (hexahedra or prisms):
  far fewer cells than tetrahedra for the same accuracy. It reports nodes, cells, the finite-volume
  quality (fv_quality: non-orthogonality, skewness, aspect ratio, cell volume ratio, and near_wall:
  per boundary group the distance from the wall to the first cell centres) with the gate fv_ok, and
  the mean cell size near each group.
- solve: the fixed steady laminar flow solver. Besides the solve number, feedback without any
  reference: every quantity the statement defines, convergence (iterations, residuals), the mass
  balance of inlets and outlets, the fluid force on each wall group, the peak speed and the pressure
  range and where they are, the cell Peclet number, reversed-flow (recirculation) zones, and an error
  indicator (the velocity jump between neighbouring cells' reconstructions at their faces: a global
  relative value and the 5 hotspots holding most of it, with the nearest group and the local cell
  size). Use the hotspots to decide the next mesh: refine where the indicator is, coarsen elsewhere.
- view: a picture of a solve's mesh, speed, pressure or error indicator (3D: a section).
- compare: how much two solves' velocity fields differ, overall and in the 5 regions that differ most.
  Two meshes that agree closely are evidence of convergence; regions where they differ need refinement.
  view and compare do not use up solves.

"""


def _flow_prompt(base: str) -> str:
    """BASE_PROMPT for flow: the CFD role line, DOFs per cell, and TOOLS_TEXT_FLOW instead of the
    elasticity tools section. Fails loudly if BASE_PROMPT changes shape."""
    head, tools_start, rest = base.partition("Tools, and how a careful engineer uses them:")
    _, playbook, tail = rest.partition("Playbook:")
    role = "You set up and run linear-elastic finite-element analyses using tools."
    dofs = "degrees of\nfreedom, DOFs"
    assert tools_start and playbook and role in head and dofs in head, "BASE_PROMPT changed: update _flow_prompt"
    head = head.replace(role, "You set up and run steady incompressible laminar flow (CFD) analyses with a "
                              "finite-volume solver using tools.")
    head = head.replace(dofs, "degrees of\nfreedom, DOFs = cells x unknowns per cell")
    return head + TOOLS_TEXT_FLOW + playbook + tail


BASE_PROMPT_FLOW = _flow_prompt(BASE_PROMPT)

BC_HELP_FLOW = (
    "BC types: velocity_inlet (`value` = velocity vector [u_x, u_y(, u_z)] in m/s, each entry a number or "
    "an expression in x, y, z such as '0.06*y*(0.01 - y)'; or one number = speed into the domain along "
    "the normal); pressure_outlet (`value` = static pressure in Pa, number or expression); wall (no slip; "
    "optional `value` = wall velocity vector for a moving wall, numbers or expressions); slip (no "
    "penetration, no shear); symmetry. Each BC applies to a named boundary `group`; boundary groups "
    "without a BC are stationary no-slip walls. A domain without a pressure_outlet is closed: its "
    "pressure level is arbitrary (only differences are meaningful)."
)

_BY_NAME = {t["name"]: t for t in TOOLS}
# Flow tasks get their own view / compare (ace.flow_feedback: velocity, pressure and its error indicator).
TOOLS_FLOW = [
    _BY_NAME["inspect_geometry"],
    {
        "name": "mesh",
        "description": "Mesh the geometry with straight-sided finite-volume cells (triangles / tetrahedra; "
                       "quad-dominant with quads=true; wall boundary layers; swept hexahedra / prisms in 3D). "
                       "Replaces the current mesh. Returns node and cell counts, the finite-volume quality "
                       "(fv_quality: max / mean non-orthogonality in degrees, max skewness, aspect ratio, cell "
                       "volume ratio across faces, near_wall: per boundary group the distance from the wall to "
                       "the first cell centres) and fv_ok, the quality gate (max non-orthogonality <= 70 deg, max "
                       "skewness <= 4). Node counts change in steps, so tiny changes to h may give the same mesh.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global cell size"},
                "quads": {"type": "boolean", "description": "recombine triangles into quads (2D, or the swept face "
                                                            "of a sweep: hexahedra; default false)"},
                "refinements": _BY_NAME["mesh"]["input_schema"]["properties"]["refinements"],
                "boundary_layer": {
                    "type": "object",
                    "description": "layers of thin cells along no-slip walls: the first cell is `first` high, each "
                                   "next layer `ratio` times thicker, `layers` deep (total thickness = first * "
                                   "(ratio^layers - 1) / (ratio - 1)). 2D; in 3D only with sweep (made on the swept "
                                   "face, along the walls it meets). Where a wall meets an inlet or outlet the "
                                   "layers end on it.",
                    "properties": {
                        "groups": {"type": "array", "items": {"type": "string"}, "description": "wall groups"},
                        "first": {"type": "number", "description": "height of the first cell at the wall"},
                        "ratio": {"type": "number", "description": "growth per layer, 1-3 (default 1.2)"},
                        "layers": {"type": "integer", "description": "number of layers, 1-40"},
                    },
                    "required": ["groups", "first", "layers"],
                },
                "sweep": {
                    "type": "object",
                    "description": "3D only: mesh a planar end face (e.g. the inlet) in 2D and extrude it through "
                                   "the domain in `layers` steps (hexahedra with quads=true, else prisms). Only for "
                                   "a straight duct or pipe of constant cross-section (checked); h, refinements, "
                                   "quads and boundary_layer then shape that face.",
                    "properties": {
                        "source": {"type": "string", "description": "group of the planar end face to sweep"},
                        "layers": {"type": "integer", "description": "cell layers along the sweep, 1-400"},
                        "ratio": {"type": "number", "description": "last / first layer length (default 1; > 1 puts "
                                                                   "the short layers at the source end)"},
                    },
                    "required": ["source", "layers"],
                },
                "using": USING,
            },
            "required": ["h", "using"],
        },
    },
    {
        "name": "solve",
        "description": "Solve on the current mesh with the fixed steady laminar flow solver (finite volumes, "
                       "SIMPLE) and return the scored quantity, the solve number and `feedback`: the statement's "
                       "quantities, convergence, mass balance, wall forces, peak speed and pressure range with "
                       "locations, cell Peclet number, reversed-flow zones, and an error indicator (global relative "
                       "value, top 5 hotspots). " + BC_HELP_FLOW,
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
    {
        "name": "view",
        "description": "Look at a picture (PNG) of a solve: its mesh, speed (with velocity arrows), pressure, or the "
                       "error indicator (the velocity jump between neighbouring cells' reconstructions at their "
                       "faces: large where the velocity curves sharply relative to the cell size). 2D: the "
                       "cells; 3D: a section through the middle of the domain normal to `view`.",
        "input_schema": {
            "type": "object",
            "properties": {
                "solve": {"type": "integer", "description": "solve number, as returned by solve"},
                "kind": {"type": "string", "enum": ["mesh", "speed", "pressure", "error"]},
                "view": {"type": "string", "enum": ["x", "y", "z"], "description": "3D: the section's normal"},
                "using": USING,
            },
            "required": ["solve", "kind", "using"],
        },
    },
    {
        "name": "compare",
        "description": "Compare the velocity fields of two solves (e.g. a coarse and a finer mesh): their relative "
                       "L2 difference over the domain, the 5 regions where they differ most, and every statement "
                       "quantity's change. Convergence evidence without a reference.",
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
    _BY_NAME["submit"],
]


# Only offered on a task scored by the nonlinear simulation of its physical test (tools_for).
NONLINEAR_TOOL = {
    "name": "nonlinear_check",
    "description": "Run the task's frozen NONLINEAR scoring simulation (CalculiX: elastoplastic resin, large "
                   "deformation, the platen ramp) on one of your meshes and compare its force-travel curve with "
                   "the test: the force at each checkpoint (simulated vs measured, relative error), the mean "
                   "relative error (the scored error), nodes, DOFs, the largest plastic strain and the hotspots "
                   "(where plastic strain and the largest principal stress peak, with the nearest faces). It takes "
                   "minutes and is limited per run (the task states how many); it does not use a solve. Your "
                   "boundary conditions do not matter here: the setup is fixed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mesh": {"type": "integer", "description": "mesh number, as returned by mesh"},
            "using": USING,
        },
        "required": ["mesh", "using"],
    },
}

# Checkpoint bookkeeping kept in the run record but not shown to the agent.
HIDDEN_KEYS = {"path", "request", "index", "sha256", "gmsh", "info"}
# Flow meshes are judged by their finite-volume quality (fv_quality, fv_ok); the element-shape
# measures (minSICN) and order-2 curving stats would only mislead: a boundary-layer cell is thin by design.
FLOW_HIDDEN_MESH_KEYS = {"min_quality", "quality", "repaired_cells", "folded_cells", "min_jacobian_ratio", "order"}

CITATION = re.compile(r"\[using ((?:[ap]\d+-)?[a-z]+-\d{4})\]")
BULLET_ID = re.compile(r"^(?:[ap]\d+-)?[a-z]+-\d{4}$")       # shared "mesh-0001" / "p1-mesh-0001" (task 1), private "a3-mesh-0001"


def is_nonlinear(problem_id: str) -> bool:
    """Is the task scored by the nonlinear simulation of its physical test? (Same test as
    ace.nonlinear.is_nonlinear, kept here so linear tasks never import that module.)"""
    return TASKS[problem_id].get("score") == "nonlinear"


def limits(problem_id: str) -> tuple[int, int]:
    """(max_turns, max_solves) of a task: its own entries in config.TASKS, else the global ones."""
    task = TASKS[problem_id]
    return task.get("max_turns", MAX_TURNS), task.get("max_solves", MAX_SOLVES)


def tools_for(problem) -> list[dict]:
    """The tool specs an agent gets on a task (a problem id or a problem): TOOLS_FLOW on a flow
    task; TOOLS, plus nonlinear_check on a nonlinear task."""
    problem = fea.get_problem(problem) if isinstance(problem, str) else problem
    if fea.is_flow(problem):
        return TOOLS_FLOW
    return TOOLS + [NONLINEAR_TOOL] if is_nonlinear(problem.id) else TOOLS


def system_prompt(playbook_text: str, flow: bool = False) -> str:
    return (BASE_PROMPT_FLOW if flow else BASE_PROMPT) + "\n" + (playbook_text or "(the playbook is empty)")


def task_prompt(problem_id: str, max_solves: int | None = None, bonus: bool = True) -> str:
    """The task statement, scoring and limits. max_solves: the solve budget stated (default
    MAX_SOLVES); bonus=False leaves out the be-the-best paragraph (the single-agent baseline,
    ace/single_loop.py, has no peers to beat on a task)."""
    max_solves = limits(problem_id)[1] if max_solves is None else max_solves
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    if is_nonlinear(problem_id):
        return _nonlinear_task_prompt(problem_id, problem, max_solves, bonus)
    q = fea.official_qoi(problem, task["qoi"])
    where = f" at point {q.point}" if q.point else (f" over group {q.group!r}" if q.group else "")
    if q.point2:
        where += f" minus its value at point {q.point2}"
    if SCORE == "field":
        scored = _field_scoring_text(problem_id, problem, q, where, bonus)
    else:
        scored = (f"SCORED QUANTITY: {q.name} = {q.type} of {q.field}{where}. Only this quantity is scored; "
                  f"ignore any other quantities the statement asks you to report.\n"
                  f"TARGET: {q.name} within {tolerance(problem_id):.0%} of the true value, with as few DOFs as possible.\n")
    if config.STATEMENT == "requirements":
        # Engineering requirements instead of the statement, and a neutral label instead of the id
        # (which names the benchmark): the agent works the boundary conditions out itself.
        from ace import requirements
        shown_id = requirements.label_for(problem_id)
        statement = f"{requirements.TASK_PREFACE}\n\n{requirements.agent_text(problem_id)}"
    else:
        shown_id, statement = problem_id, problem.description()
    from ace import evidence
    if test := evidence.agent_section(problem_id):    # physical test data, when the loop has some
        statement += "\n\n" + test
    return (
        f"Problem id: {shown_id}\n\n{statement}\n\n"
        f"{scored}"
        f"LIMITS: the mesh you solve on must have at most {task['node_cap']} nodes (a safety limit, not a "
        f"target), and {_quality_limit(problem)}. You may call solve at most {max_solves} times."
    )


def _field_scoring_text(problem_id: str, problem, q, where: str, bonus: bool) -> str:
    """The task's SCORING paragraph in field mode: the whole stress field for solids, the whole velocity
    field for flow (ace.field_error / ace.flow_field), against the automesher, plus the bonuses."""
    from ace.config import AUTO_GAIN_TARGET as G
    from ace.config import BEST_BONUS, DIVERSITY_BONUS, DIVERSITY_H_RATIO, DIVERSITY_MARGIN, PROBE_DISCLOSED, PROBE_POINTS
    from ace.field_error import automesher, probe_points
    a = automesher(problem_id)
    shown = "; ".join("(" + ", ".join(f"{v:.4g}" for v in pt[:problem.dim]) + ")"
                      for pt in probe_points(problem_id)["points"][:PROBE_DISCLOSED])
    hidden = PROBE_POINTS - PROBE_DISCLOSED
    if fea.is_flow(problem):
        from ace.flow_field import has_exact
        truth = "the exact solution" if has_exact(problem) else "a very fine reference solution"
        what = (f"Scored: the whole velocity field, two ways. (1) Field error: the relative L2 error of your velocity "
                f"field over the entire domain, against {truth}. (2) Point error: the RMS relative error of the "
                f"speed |u| at {PROBE_POINTS} check points in the bulk of the flow (away from the walls). "
                f"{PROBE_DISCLOSED} of them are {shown}; the other {hidden} are hidden. ")
        auto = (f"the mesh tool's uniform triangles / tetrahedra at h = {a.get('h', problem.params.get('h0')):g} with "
                f"linearUpwind")
        point_err, kind = "speed error", "cell kind (quads, boundary layer, sweep)"
    else:
        what = (f"Scored: the whole stress field, two ways. (1) Field error: the energy-norm error of your stress field "
                f"over the entire part, relative to a very fine reference solution (the strain energy of the stress "
                f"difference, integrated everywhere). (2) Point error: the RMS relative error of the von Mises stress "
                f"at {PROBE_POINTS} check points spread across the part (away from stress singularities). "
                f"{PROBE_DISCLOSED} of them are {shown}; the other {hidden} are hidden. ")
        auto = "gmsh with its default sizes and quadratic elements"
        point_err, kind = "stress error", "element order"
    scored = (f"SCORING FOR THIS TASK (replaces 'How you are scored' above). {what}"
              f"Your error e = the larger of (1) and (2). The statement's {q.name}{where} is not scored; you "
              f"may still use it as a check.\n"
              f"You compete against an AUTOMESHER: {auto} (scored the same way), which "
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
              f"{hidden} hidden check points, the agent with the smallest {point_err} "
              f"there wins a share of up to +{BEST_BONUS:g} in total. A tie pays nobody, so copying the "
              f"leader earns nothing. Trying something that beats the others pays.\n"
              f"DIVERSITY BONUS: +{DIVERSITY_BONUS:g} if your mesh recipe is structurally different from every "
              f"other agent's on this task in your generation (another {kind}, global h differing by "
              f"more than {DIVERSITY_H_RATIO:g}x, or mostly different refinement targets) and your base score "
              f"is at least {DIVERSITY_MARGIN:.0%} of the best one there.\n")
    return scored


def _quality_limit(problem) -> str:
    """The mesh quality gate in the task's LIMITS line: element shape for solids, the finite-volume
    gate (feacore.geometry.fv_quality) for flow."""
    if fea.is_flow(problem):
        from feacore.geometry.fv_quality import NON_ORTHO_MAX, SKEW_MAX
        return (f"its finite-volume quality must pass the gate (fv_ok: max non-orthogonality at most "
                f"{NON_ORTHO_MAX:.0f} degrees and max skewness at most {SKEW_MAX:.0f})")
    return f"its minimum element quality (quality.min) must be at least {MIN_QUALITY}"


def _nonlinear_task_prompt(problem_id: str, problem, max_solves: int, bonus: bool) -> str:
    """The task statement of a task scored against its physical test through a FROZEN nonlinear
    simulation (ace/nonlinear.py; abc_00000065_compression): the test, the frozen setup in plain
    words, that the agent's job is only the mesh, and the score vs the automesher per DOF. None of
    the field-mode wording (fine reference, probe points): nothing here is scored against a reference."""
    from ace import nonlinear
    from ace.config import DIVERSITY_BONUS, DIVERSITY_H_RATIO, DIVERSITY_MARGIN, FIELD_ERR_FLOOR
    sp = nonlinear.spec()
    mat = json.loads((Path(__file__).resolve().parents[1] / sp["material_file"]).read_text())   # repo-relative
    table = mat["plastic_table_true_stress_vs_plastic_strain"]        # [[true stress MPa, plastic strain], ...]
    checkpoints = ", ".join(f"{c:g}" for c in sp["checkpoints_mm"])
    u_max, n_inc = sp["u_part_max_mm"], sp["increments"]
    task = TASKS[problem_id]
    # The geometry line of the problem's statement (its supports, load and material are the linear
    # placeholder setup, replaced here by the frozen one).
    geometry = next((ln for ln in problem.description().splitlines() if ln.startswith("Geometry")), "")
    try:                              # cached in ace/field_refs; computing it needs CalculiX
        a = nonlinear.automesher_baseline(problem_id)
        auto = f"gets e = {a['mean_rel_err']:.2%} with {a['dofs']} DOFs ({a['nodes']} nodes)"
        e_auto, d_auto = f"{a['mean_rel_err']:.4f}", f"{a['dofs']}"
    except Exception as e:  # noqa: BLE001  no ccx and no cached baseline (a laptop): the formula without numbers
        auto, e_auto, d_auto = (f"has an error e_auto with DOFs_auto DOFs (not computed on this machine: "
                                f"{type(e).__name__})"), "e_auto", "DOFs_auto"
    statement = (
        f"{geometry}\n"
        f"THE REAL TEST: the printed part (Formlabs Alpha Fast resin) was compressed on an Instron between two flat "
        f"platens: the base face f33 (group base_bottom) stands on the lower platen and the upper platen pushes the "
        f"two prong tops f17 and f34 (group prong_tops) down, along -y. The machine recorded the force against the "
        f"crosshead travel (no extensometer) up to the peak, about 4.8 kN at about 2.5 mm; the resin yields well "
        f"before the peak. There is no reference solution: this test is the only check.")
    from ace import evidence
    if test := evidence.agent_section(problem_id):    # the test record (with --evidence): measured checkpoint forces
        statement += "\n\n" + test
    scored = (
        f"SCORING FOR THIS TASK (replaces 'How you are scored' above). The MESH of the solve you submit is run "
        f"through a FROZEN nonlinear simulation of the test; you do not choose any of its setup:\n"
        f"- sticking platens (friction stops the faces from sliding): base_bottom (f33) held in x, y and z; the prong "
        f"tops f17 and f34 held in x and z and pushed down in y by the same displacement u, ramped from 0 to "
        f"{u_max:g} mm in {n_inc} increments;\n"
        f"- the resin as measured on a dogbone of the same print: E = {mat['E'] / 1e3:.3g} GPa, Poisson's ratio "
        f"{mat['nu']:g}, von Mises (J2) plasticity with isotropic hardening, yield from {table[0][0]:.3g} MPa "
        f"hardening to {table[-1][0]:.3g} MPa;\n"
        f"- large deformation (geometric nonlinearity on);\n"
        f"- the force F is the total platen reaction on the prong tops, and the crosshead travel is the part's "
        f"shortening plus a seating gap plus the load train's deflection: u_crosshead = u + {sp['delta_mm']:g} mm "
        f"+ F x {sp['compliance_mm_per_N']:.3g} mm/N (both calibrated once and frozen);\n"
        f"- the simulated force is compared with the measured one at five crosshead travels, {checkpoints} mm: "
        f"e = the mean of |F_sim - F_test| / F_test over them.\n"
        f"YOUR JOB IS ONLY THE MESH: the physics, material, seating and compliance are fixed, so between meshes e "
        f"changes only through the mesh. The solve tool stays the fast LINEAR-elastic solver: use it, its error "
        f"estimate, view and compare as a quick proxy for where the mesh is under-resolved. Set your linear solves "
        f"up like the frozen setup: formulation 3d; base_bottom fixed; f17 and f34 each with displacement "
        f"components [0, 1, 2] = [0, -u, 0] (e.g. u = {u_max / 1e3:g}; the geometry is in metres); E = "
        f"{mat['E'] * 1e6:.4g} Pa, nu = {mat['nu']:g}. The linear solve cannot see plasticity or large "
        f"deformation, and its own scored quantity (u_top) is not scored here. nonlinear_check runs the real "
        f"scoring simulation on one of your meshes (minutes each; at most {task.get('nl_previews', 0)} per run; it "
        f"does not use a solve) and shows the force at each checkpoint vs the test, e, and where the plastic strain "
        f"and the largest principal stress concentrate. Quadratic meshes are fine: curved mid-side nodes CalculiX "
        f"rejects are straightened automatically.\n"
        f"You compete against an AUTOMESHER: gmsh with its default sizes and quadratic (straight-sided) elements, run "
        f"through the same simulation, which {auto}.\n"
        f"SCORE = A / c, where g = {e_auto} / e (your accuracy gain; e below {FIELD_ERR_FLOOR:.1%} counts as "
        f"{FIELD_ERR_FLOOR:.1%}), c = your DOFs / {d_auto}, A = (g/2)^2 if g < 2 (missing 2x the automesher's "
        f"accuracy is penalised quadratically) else g/2 (beating it keeps paying). Score 1 = half the automesher's "
        f"error at its DOFs; the automesher itself scores 0.25. Extra DOFs pay only if they buy at least "
        f"proportionally more accuracy, so put them where the error is.\n")
    if bonus:
        scored += (f"DIVERSITY BONUS: +{DIVERSITY_BONUS:g} if your mesh recipe is structurally different from every "
                   f"other agent's on this task in your generation (another element order, global h differing by more "
                   f"than {DIVERSITY_H_RATIO:g}x, or mostly different refinement targets) and your base score is at "
                   f"least {DIVERSITY_MARGIN:.0%} of the best one there.\n")
    return (
        f"Problem id: {problem_id}\n\n{statement}\n\n"
        f"{scored}"
        f"LIMITS: the mesh you submit must have at most {task['node_cap']} nodes (a safety limit, not a target), and "
        f"its minimum element quality (quality.min) must be at least {MIN_QUALITY}. You may call solve at most "
        f"{max_solves} times and nonlinear_check at most {task.get('nl_previews', 0)} times."
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
        self.task = TASKS[problem_id]
        # The task's own solve budget (config.TASKS), else MAX_SOLVES; larger for the single-agent baseline
        self.max_solves = limits(problem_id)[1] if max_solves is None else max_solves
        self.problem = fea.get_problem(problem_id)
        self.workdir = Path(workdir)
        self.meshes: list[dict] = []
        self.solves: list[dict] = []
        self.submitted: int | None = None   # index into self.solves
        self.results: dict = {}             # solve number -> feacore Result (in memory only, never in the record)
        self.n_views = 0
        self.nl_previews: list[dict] = []   # nonlinear_check results (the full ace.nonlinear.run_curve dicts)
        self.ui = ui                        # ace.ui_bridge context of this run; None = off

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Returns (result, is_error)."""
        if name == "inspect_geometry":
            # The old group list (names BCs refer to) plus the face/edge description, compacted for big parts.
            return geometry_info.for_agent({**fea.geometry_summary(self.problem), **geometry_info.describe(self.problem)}), False
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
        if name == "nonlinear_check":
            return self._nonlinear_check(args)
        if name in ("view", "compare"):
            try:
                return self._view(args) if name == "view" else self._compare(args)
            except Exception as e:  # noqa: BLE001  a plotting/sampling failure is the agent's information, not a crash
                return {"error": f"{name} failed: {type(e).__name__}: {e}"}, True
        return {"error": f"unknown tool {name!r}"}, True

    def _mesh(self, args: dict) -> tuple[dict, bool]:
        path = self.workdir / f"mesh_{len(self.meshes)}.msh"
        m = fea.make_mesh(self.problem, path, h=args.get("h", 0), order=args.get("order", 1),
                          refinements=args.get("refinements"), quads=bool(args.get("quads", False)),
                          boundary_layer=args.get("boundary_layer"), sweep=args.get("sweep"))
        # Checkpoint: the request that produced this mesh, so it can be rebuilt or varied later.
        m["request"] = {k: v for k, v in args.items() if k != "using"}
        m["index"] = len(self.meshes)
        self.meshes.append(m)
        hidden = HIDDEN_KEYS | (FLOW_HIDDEN_MESH_KEYS if fea.is_flow(self.problem) else set())
        out = {"mesh": len(self.meshes), **{k: v for k, v in m.items() if k not in hidden}}   # 1-based number
        if not m["ok"]:
            return out, True
        cap = self.task["node_cap"]
        out.update(node_cap=cap, within_cap=m["nodes"] <= cap)
        if m.get("repaired_cells"):
            out["message"] = (f"{m['repaired_cells']} curved (order 2) cells were folded or nearly folded and were "
                              f"straightened to keep the mesh valid; this is automatic and fine")
        if "fv_ok" in m:                  # flow: the finite-volume gate instead of the element shape
            if not m["fv_ok"]:
                out["warning_quality"] = (f"finite-volume quality gate failed ({m.get('fv_gate')}): a run submitted "
                                          f"on this mesh is invalid. Smoother size transitions, a thinner or shorter "
                                          f"boundary layer, or a different h usually fix it")
        elif m["min_quality"] < MIN_QUALITY:
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
                           + f" (the sizes used, times {factor:.2f})"
                           + (", or fewer sweep layers" if m.get("sweep") else ""))
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
        from ace import evidence
        check = evidence.solve_check(self.problem_id, res)   # this solve vs the physical test, when there is one
        self.solves.append({"mesh": mesh, "mesh_index": len(self.meshes) - 1, "setup": setup, "result": result,
                            **({"test_check": check} if check else {})})
        n = len(self.solves)
        ui_bridge.record_solve(self.ui, self, res)
        out = {**{k: v for k, v in result.items() if k not in HIDDEN_KEYS}, "solve": n, "nodes": mesh["nodes"],
               "solves_left": self.max_solves - n}
        if check:
            out["test_check"] = {"prediction_vs_test": f"{check['rel_err']:+.2%}", "sigma": round(check["sigma"], 2),
                                 "within_test_uncertainty": check["consistent"]}
        if res is not None:
            self.results[n] = res
            try:   # stress-based for solids (ace.solve_feedback), velocity-based for flow (ace.flow_feedback)
                out["feedback"] = self._feedback_module().feedback(self.problem, mesh, setup, res)
            except Exception as e:  # noqa: BLE001  the solve itself stands; say why there is no feedback
                out["feedback"] = {"error": f"feedback failed: {type(e).__name__}: {e}"}
        return out, not result["ok"]

    def _feedback_module(self):
        """ace.flow_feedback on a flow task, else ace.solve_feedback: the same feedback / render /
        compare interface."""
        return flow_feedback if fea.is_flow(self.problem) else solve_feedback

    def _nonlinear_check(self, args: dict) -> tuple[dict, bool]:
        """Mesh number n (1-based, as the mesh tool returns it) through the frozen nonlinear scoring
        simulation (ace.nonlinear.run_curve); a compact result for the agent, the full one kept."""
        if not is_nonlinear(self.problem_id):
            return {"error": "nonlinear_check is not available on this task"}, True
        cap = self.task.get("nl_previews", 0)
        if len(self.nl_previews) >= cap:
            return {"error": f"nonlinear_check limit ({cap}) reached for this run; choose your mesh from the "
                             f"checks you have and your linear solves"}, True
        n = args.get("mesh")
        if not isinstance(n, int) or not 1 <= n <= len(self.meshes) or not self.meshes[n - 1].get("ok"):
            ok = [i + 1 for i, m in enumerate(self.meshes) if m.get("ok")]
            return {"error": f"mesh {n!r} is not a successful mesh; successful meshes: {ok}"}, True
        mesh = self.meshes[n - 1]
        if mesh["nodes"] > self.task["node_cap"]:
            return {"error": f"mesh {n} has {mesh['nodes']} nodes, over the cap of {self.task['node_cap']} "
                             f"(this did not use a check)"}, True
        from ace import nonlinear
        workdir = self.workdir / f"nl_preview_{len(self.nl_previews) + 1}"
        try:
            r = nonlinear.run_curve(self.problem_id, mesh, workdir=workdir, threads=4, series=False)
        except Exception as e:  # noqa: BLE001  e.g. no CalculiX on this machine: the agent's information, not a crash
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.nl_previews.append({"mesh": n, **r})
        left = cap - len(self.nl_previews)
        if not r.get("ok"):
            return {"mesh": n, "error": f"nonlinear check failed: {r.get('error')}", "checks_left": left}, True
        table = [{"crosshead_mm": u, "test_N": round(fm), "sim_N": round(fs), "rel_err": f"{e:+.1%}"}
                 for u, fm, fs, e in zip(r["checkpoints_mm"], r["force_meas_N"], r["force_sim_N"], r["rel_err"])]
        return {"mesh": n, "mean_rel_err": f"{r['mean_rel_err']:.2%}", "checkpoints": table,
                "nodes": r["nodes"], "dofs": r["dofs"], "peeq_max": solve_feedback._sig(r["peeq_max"]),
                "hotspots": r.get("hotspots") or [], "straightened": r.get("straightened", False), "straightened_edges": r.get("straightened_edges", 0),
                "wall_s": round(r.get("wall_s") or 0), "checks_left": left}, False

    def _result_of(self, n) -> tuple:
        """(feacore Result of solve n, None) or (None, an error for the agent)."""
        if n not in self.results:
            return None, ({"error": f"solve {n!r} is not a successful solve; successful solves: "
                                    f"{sorted(self.results)}"}, True)
        return self.results[n], None

    def _view(self, args: dict) -> tuple[dict, bool]:
        flow = fea.is_flow(self.problem)
        n = args.get("solve")
        kind, view = args.get("kind", "speed" if flow else "stress"), args.get("view", "z" if flow else "iso")
        res, err = self._result_of(n)
        if err:
            return err
        self.n_views += 1
        png = self.workdir / f"view_{self.n_views}_solve{n}_{kind}_{view.replace('-', 'm')}.png"
        path = self._feedback_module().render(self.problem, self.solves[n - 1]["mesh"], res, kind, view, png)
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
        return {"solve_a": a, "solve_b": b, **self._feedback_module().compare(self.problem, ra, rb)}, False


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
    flow = fea.is_flow(tools.problem)          # flow tasks get the finite-volume prompt and tools
    trace, stop = tool_loop(llm, system_prompt(playbook_text, flow=flow), task_prompt(problem_id),
                            tools_for(problem_id), tools.call, lambda: tools.submitted is not None,
                            limits(problem_id)[0], run_id, usage)

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
        **({"nl_previews": tools.nl_previews} if is_nonlinear(problem_id) else {}),
        "usage": usage,
    }
