"""The analyst's tools: geometry -> idealization -> mesh -> solve -> convergence study -> report.

A `Session` is one analysis of one part in one workspace directory:

    <workspace>/geometry.step     the CAD (metres), the only geometry the agent has
    <workspace>/package.md       the design package (what the agent reads; not used here)
    <workspace>/session.json     everything done so far: meshes, solves, studies, the report
    <workspace>/mesh_<k>.msh, view_*.png

The same Session serves the automated loop (single_agent.agent: Anthropic tool use, in process) and
the command line (single_agent.cli: one tool call per process, state reloaded from session.json, a
solve's fields recomputed when a later call needs them). Nothing here reads a reference: the part is
a BlindPart (single_agent.part), and the evaluator (single_agent.evaluate) is a separate module.

Tools (TOOLS below has the schemas the model sees):
  inspect_geometry  faces, edges, re-entrant edges, smallest features (ace.geometry_info, no roles)
  find_faces        faces matching a geometric query (feagent's query language)
  view_geometry     a picture of the part with chosen faces highlighted and labelled
  mesh              gmsh mesh: global size, element order, local refinements, named face groups
  refine_mesh       the same recipe with every size divided by a factor (for convergence studies)
  solve             the fixed solver on a mesh with the agent's material, supports, loads and QoIs;
                    reference-free feedback (equilibrium, extremes, ZZ error estimate, hotspots)
  study             Richardson / GCI analysis of the agent's QoIs over 2+ solves of one setup
  view, compare     pictures of a solve; the difference between two solves
  submit_report     the deliverable; ends the analysis
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from single_agent import config
from single_agent.part import BlindPart, register

# --------------------------------------------------------------------------- tool schemas

VIEW_ENUM = ["iso", "x", "y", "z", "-x", "-y", "-z"]

QOI_HELP = (
    "Quantities of interest, each {name, type, field, group, point, component}: "
    "strain_energy (total, J); max / min / max_abs of a nodal field over a face group (group null = whole part); "
    "mean / integral of a field over a face group or the solid; reaction = support force on the part summed over "
    "a supported group, `component` 0/1/2 = x/y/z (N); point = a field at `point` [x, y, z]. "
    "Fields: ux, uy, uz, u_mag, sxx, syy, szz, sxy, syz, sxz, von_mises, tresca, s1, s2, s3. "
    "Stresses come from a recovered continuous field."
)
BC_HELP = (
    "Supports and loads (bcs), each on a named face `group` (f<tag> or a group you defined in mesh): "
    "fixed (all displacements 0); displacement (`components` [0=x,1=y,2=z] = `value`, scalar or one per "
    "component; optional `where`: an expression in x, y, z that keeps only part of the group's nodes); "
    "symmetry (zero normal displacement on a flat, axis-aligned face); pressure (`value` Pa, positive pushes "
    "into the part, normal to the face everywhere); traction (`value` [tx, ty, tz] Pa, a force per area in "
    "global axes); force (`value` [Fx, Fy, Fz] N, a total force spread uniformly per area over the group). "
    "fixed/displacement may instead take a `point` [x, y, z] (the nearest node) to remove rigid-body motion."
)

TOOLS = [
    {
        "name": "inspect_geometry",
        "description": "Describe the part: bounding box, volume, every face (f<tag>, type, area, centroid, outward "
                       "normal if flat, radius if curved, adjacent faces), the CAD edges with the material angle "
                       "between their faces (> 180 deg = re-entrant: stress singularities), the smallest features "
                       "and plain-words hints. Coordinates are in metres.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_faces",
        "description": "Faces matching a geometric query, with their description. Query keys (combine freely): kind "
                       "(plane, cylinder, cone, sphere, torus, bspline), normal [x,y,z] (flat faces facing that way; "
                       "tol_deg), axis [x,y,z] (cylinders/cones along it), radius (+ radius_tol) or radius_min/"
                       "radius_max, area_min/area_max, concave (true = a hole or fillet surface), at (xmin, xmax, ymin, "
                       "ymax, zmin, zmax: faces touching that side of the bounding box), box [[x0,y0,z0],[x1,y1,z1]] "
                       "(face inside), near [x,y,z] with within (distance), adjacent_to {query}, largest n / smallest n "
                       "(by area), any [queries], all [queries], not {query}, ids [tags]. Lengths in metres.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "object"}}, "required": ["query"]},
    },
    {
        "name": "view_geometry",
        "description": "A picture of the part (no mesh needed) seen from `view`, with the faces in `highlight` "
                       "coloured and labelled; `label_all` also labels every other visible face with its tag. Use it "
                       "to check that the faces you picked are the ones the design data describes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "highlight": {"type": "array", "items": {"type": "string"},
                              "description": "faces (f<tag>) or groups you defined, e.g. [\"f12\", \"f7\"]"},
                "view": {"type": "string", "enum": VIEW_ENUM},
                "label_all": {"type": "boolean"},
            },
        },
    },
    {
        "name": "mesh",
        "description": "Mesh the part with gmsh (tetrahedra). Returns the mesh number, nodes, element quality "
                       "(min / 1st percentile / median minSICN; 0 = degenerate), the mean element size near each face, "
                       "and for order 2: curved cells that had to be straightened (repaired_cells, automatic) and cells "
                       "still folded (folded_cells, must be 0). Optionally defines named groups (unions of faces) for "
                       "supports, loads, QoIs and refinements. Node counts change in steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "h": {"type": "number", "description": "global element size (m)"},
                "order": {"type": "integer", "enum": [1, 2], "description": "element order (2 = quadratic, curved)"},
                "refinements": {
                    "type": "array",
                    "description": "local sizes: `size` at ONE of a face `group`, a CAD `edge` (tag from "
                                   "inspect_geometry), a `point` [x,y,z] (growing linearly to h at distance `radius`), "
                                   "or inside a `box` [xmin,ymin,zmin,xmax,ymax,zmax]",
                    "items": {"type": "object", "properties": {
                        "group": {"type": "string"}, "edge": {"type": "integer"},
                        "point": {"type": "array", "items": {"type": "number"}},
                        "box": {"type": "array", "items": {"type": "number"}},
                        "size": {"type": "number"}, "radius": {"type": "number"}},
                        "required": ["size"]},
                },
                "groups": {"type": "object", "description": "named face groups to define, e.g. "
                                                            "{\"mount_pads\": [\"f9\", \"f18\"]}",
                           "additionalProperties": {"type": "array", "items": {"type": "string"}}},
            },
            "required": ["h", "order"],
        },
    },
    {
        "name": "refine_mesh",
        "description": "Re-mesh with the recipe of mesh `mesh` and every element size (global h and the refinement "
                       "sizes) divided by `factor` (refinement radii and groups unchanged). The systematic way to "
                       "build a convergence study: factors of 1.3-2 between levels.",
        "input_schema": {"type": "object", "properties": {"mesh": {"type": "integer"}, "factor": {"type": "number"}},
                         "required": ["mesh", "factor"]},
    },
    {
        "name": "solve",
        "description": "Static linear-elastic solve (the fixed solver) on mesh `mesh` (default: the latest). Returns "
                       "the solve number, your QoIs, DOFs and feedback without any reference: support reactions vs "
                       "applied loads (equilibrium), the largest displacement and von Mises stress and where, and a "
                       "ZZ error estimate (relative energy-norm error and the hotspots holding most of it, with the "
                       "nearest face, re-entrant edge, the rim of your own supports, and the local element size). "
                       + BC_HELP + " " + QOI_HELP,
        "input_schema": {
            "type": "object",
            "properties": {
                "mesh": {"type": "integer"},
                "material": {"type": "object", "properties": {"E": {"type": "number", "description": "Pa"},
                                                              "nu": {"type": "number"}},
                             "required": ["E", "nu"]},
                "bcs": {"type": "array", "items": {"type": "object", "properties": {
                    "type": {"type": "string", "enum": ["fixed", "displacement", "symmetry", "pressure", "traction",
                                                        "force"]},
                    "group": {"type": "string"}, "point": {"type": "array", "items": {"type": "number"}},
                    "components": {"type": "array", "items": {"type": "integer"}}, "value": {},
                    "where": {"type": "string"}},
                    "required": ["type"]}},
                "loads": {"type": "array", "description": "volume loads (optional): body_force [fx, fy, fz] N/m^3",
                          "items": {"type": "object", "properties": {
                              "type": {"type": "string", "enum": ["body_force"]}, "value": {}},
                              "required": ["type", "value"]}},
                "qois": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["strain_energy", "max", "min", "max_abs", "mean", "integral",
                                                        "reaction", "point"]},
                    "field": {"type": "string"}, "group": {"type": "string"},
                    "point": {"type": "array", "items": {"type": "number"}}, "component": {"type": "integer"}},
                    "required": ["name", "type"]}},
            },
            "required": ["material", "bcs", "qois"],
        },
    },
    {
        "name": "study",
        "description": "Convergence study of solves you already ran with ONE setup (same material, supports, loads) on "
                       "different meshes: for each common QoI the values against the representative element size, "
                       "the convergence type (monotone / oscillatory / divergent), observed order, Richardson-"
                       "extrapolated value and the Grid Convergence Index (Celik et al. 2008: the fine solution's "
                       "relative uncertainty, safety factor 1.25), plus a power-law fit with 4+ solves. Divergent "
                       "QoIs are usually singular (a stress at a re-entrant edge or at the rim of a support). Uses no "
                       "solves.",
        "input_schema": {"type": "object", "properties": {"solves": {"type": "array", "items": {"type": "integer"}}},
                         "required": ["solves"]},
    },
    {
        "name": "view",
        "description": "A picture of a solve: its mesh (your supports blue, loads red, re-entrant edges magenta), the "
                       "von Mises stress, or the ZZ estimated error density, seen from `view`.",
        "input_schema": {"type": "object", "properties": {
            "solve": {"type": "integer"}, "kind": {"type": "string", "enum": ["mesh", "stress", "error"]},
            "view": {"type": "string", "enum": VIEW_ENUM}}, "required": ["solve", "kind"]},
    },
    {
        "name": "compare",
        "description": "How far two solves' stress fields are apart (relative energy norm of the difference) and the "
                       "regions where they differ most, plus the change of every QoI. Evidence of convergence without "
                       "a reference. Uses no solves.",
        "input_schema": {"type": "object", "properties": {"solve_a": {"type": "integer"}, "solve_b": {"type": "integer"}},
                         "required": ["solve_a", "solve_b"]},
    },
    {
        "name": "submit_report",
        "description": "Submit the analysis report and end the analysis. It must answer every question of the design "
                       "review with a value from your solves (SI units), its relative discretization uncertainty from "
                       "your study, and a status: converged, not_converged, or singular (the value grows without bound "
                       "under refinement, so no mesh gives a trustworthy number).",
        "input_schema": {
            "type": "object",
            "properties": {
                "idealization": {"type": "array", "items": {"type": "object", "properties": {
                    "design_item": {"type": "string", "description": "what in the design data this represents"},
                    "faces": {"type": "array", "items": {"type": "string"}},
                    "model": {"type": "string", "description": "how you modelled it (support / load type, value)"},
                    "why": {"type": "string"}}, "required": ["design_item", "faces", "model"]}},
                "final_solve": {"type": "integer", "description": "the solve whose setup is your model of record"},
                "study_solves": {"type": "array", "items": {"type": "integer"}},
                "answers": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string", "description": "the review question id, e.g. Q1"},
                    "quantity": {"type": "string", "description": "the QoI of your setup the value comes from"},
                    "value": {"type": ["number", "null"], "description": "SI units; null if no trustworthy value"},
                    "units": {"type": "string"},
                    "uncertainty": {"type": ["number", "null"],
                                    "description": "relative discretization uncertainty (0.02 = 2%), from your study"},
                    "status": {"type": "string", "enum": ["converged", "not_converged", "singular"]},
                    "verdict": {"type": "string", "enum": ["meets", "fails", "not_applicable", "cannot_tell"]},
                    "comment": {"type": "string"}}, "required": ["id", "value", "status"]}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "rules_used": {"type": "array", "items": {"type": "string"},
                               "description": "ids of the rulebook rules that shaped your decisions, e.g. [\"A2\", \"L004\"]"},
                "recommendations": {"type": "array", "items": {"type": "string"},
                                    "description": "what the design team should do with these results"},
                "summary": {"type": "string"},
            },
            "required": ["idealization", "final_solve", "answers", "summary"],
        },
    },
]
TOOL_NAMES = [t["name"] for t in TOOLS]


# --------------------------------------------------------------------------- helpers


@contextlib.contextmanager
def quiet():
    """Silence C-level stdout/stderr (gmsh prints from C++), so the CLI's JSON stays clean."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def _sig(x, n: int = 4):
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{n}g}") if math.isfinite(x) else None


def canonical_setup(setup: dict) -> str:
    """The physics of a setup (material, supports, loads), order-independent: two solves with the same
    canonical setup differ only by their meshes, which is what a convergence study needs."""
    mat = setup.get("material") or {}
    bcs = sorted(json.dumps({k: v for k, v in b.items() if v is not None}, sort_keys=True) for b in setup.get("bcs", []))
    loads = sorted(json.dumps(ld, sort_keys=True) for ld in setup.get("loads", []) or [])
    return json.dumps({"E": mat.get("E"), "nu": mat.get("nu"), "bcs": bcs, "loads": loads}, sort_keys=True)


def build_case(mesh: dict, setup: dict):
    """The agent's setup -> a feacore Case on `mesh` (3D linear elasticity, FE order = mesh order)."""
    from feacore.case import BC, Analysis, Case, Load, Material, QoI
    mat = setup.get("material") or {}
    keys = {"type", "group", "point", "components", "value", "where"}
    return Case(
        mesh=mesh["path"],
        analysis=Analysis(physics="elasticity", formulation="3d", order=mesh["order"]),
        materials={"mat": Material(E=float(mat.get("E", 0.0)), nu=float(mat.get("nu", 0.0)))},
        bcs=[BC(**{k: v for k, v in b.items() if k in keys}) for b in setup.get("bcs", [])],
        loads=[Load(type=ld["type"], value=ld["value"], group=ld.get("group", "*")) for ld in setup.get("loads") or []],
        qois=[QoI(**{k: v for k, v in q.items() if k in ("name", "type", "field", "group", "point", "component")})
              for q in setup.get("qois", [])],
    )


def solve_setup(mesh: dict, setup: dict):
    """(feacore Result, None) or (None, error text). Validates the case first."""
    from feacore.solver import solve
    try:
        case = build_case(mesh, setup)
        problems = case.validate()
        if problems:
            return None, "invalid setup: " + "; ".join(problems)
        with quiet():
            return solve(case, backend="skfem"), None
    except Exception as e:  # noqa: BLE001  any failure is the agent's information
        return None, f"{type(e).__name__}: {e}"


def evaluate_qois(result, qois: list[dict]) -> dict:
    from feacore.case import QoI
    out = {}
    for q in qois:
        try:
            out[q["name"]] = float(result.evaluate_qoi(QoI(**{k: v for k, v in q.items()
                                                             if k in ("name", "type", "field", "group", "point",
                                                                      "component")})))
        except Exception as e:  # noqa: BLE001
            out[q["name"]] = None
            out.setdefault("_errors", {})[q["name"]] = f"{type(e).__name__}: {e}"
    return out


def gci_unequal(h: list[float], phi: list[float], p_theory: float | None, safety: float = 1.25) -> dict | None:
    """Three-mesh analysis for UNEQUAL refinement ratios, where feacore's gci3 misreads convergence.

    gci3 classifies by R = e21 / e32 alone, which is exact only when r21 = r32. With h1 < h2 < h3,
    e21 = f2 - f1, e32 = f3 - f2 and phi = phi0 + C h^p, the changes satisfy
        e32 / e21 = F(p) = r21^p (r32^p - 1) / (r21^p - 1),
    which tends to ln(r32) / ln(r21) as p -> 0 and grows with p. So a sequence of same-sign changes converges
    (some p > 0 fits) iff e32 / e21 > ln(r32) / ln(r21); with r21 = 1.34 and r32 = 1.21 a converging QoI can
    have R > 1, which gci3 calls divergent; and with r21 < r32 a stalling QoI can have R < 1, which gci3 calls
    monotone. Returns the gci3-shaped fields decided by the ratio-corrected test (bisection on p), or None when it
    does not apply (opposite-sign changes or nearly equal ratios: gci3 is right there)."""
    order_ = np.argsort(h)
    (h1, h2, h3), (f1, f2, f3) = np.asarray(h, float)[order_], np.asarray(phi, float)[order_]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 * e32 <= 0 or abs(r21 - r32) <= 0.05 * r21 or min(r21, r32) <= 1.0:
        return None
    e_a21 = abs(e21 / f1) if f1 else math.inf
    target = e32 / e21
    F = lambda p: r21 ** p * (r32 ** p - 1) / (r21 ** p - 1)   # noqa: E731
    lo, hi = 1e-6, 30.0
    flags = ["unequal_ratios: classified with the ratio-corrected test"]
    if min(r21, r32) < 1.3:
        flags.append("refinement_ratio_below_1.3")
    if target <= F(lo):                              # no positive order fits: not converging
        return {"convergence": "divergent", "p": None, "phi_ext": None, "gci_fine": None, "e_a21": e_a21,
                "gci_fine_conservative": None, "flags": flags + ["divergent"]}
    if target >= F(hi):
        lo = hi = 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if F(mid) < target else (lo, mid)
    p = 0.5 * (lo + hi)
    phi_ext = f1 + (f1 - f2) / (r21 ** p - 1)
    gci_fine = safety * e_a21 / (r21 ** p - 1)
    p_lim = min(max(p, 0.5), p_theory) if p_theory else p
    gci_cons = 3.0 * e_a21 / (r21 ** p_lim - 1)
    if p_theory:
        if p < 0.5 * p_theory:
            flags.append("p_far_below_theory")
        elif p < 0.8 * p_theory:
            flags.append("p_below_theory")
        elif p > 1.5 * p_theory + 0.5:
            flags.append("p_above_theory")
    return {"convergence": "monotone", "p": p, "phi_ext": phi_ext, "gci_fine": gci_fine, "e_a21": e_a21,
            "gci_fine_conservative": gci_cons, "flags": flags}


def expected_order(q: dict, order: int) -> float | None:
    """A-priori order in h of a QoI for P_order elements (feacore.verification.study)."""
    from feacore.case import QoI
    from feacore.verification.study import default_expected_order
    try:
        return default_expected_order(QoI(**{k: v for k, v in q.items() if k in ("name", "type", "field", "group",
                                                                                  "point", "component")}), order)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- the session


class Session:
    """One analysis of one part. All state is in self.state (saved to session.json after each call)."""

    def __init__(self, workdir: str | Path, label: str | None = None, limits: dict | None = None):
        self.dir = Path(workdir).resolve()
        self.path = self.dir / "session.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"label": label or self.dir.name, "meshes": [], "solves": [], "studies": [],
                          "views": 0, "report": None, "log": [],
                          "limits": {"node_cap": config.NODE_CAP, "max_solves": config.MAX_SOLVES,
                                     "max_views": config.MAX_VIEWS, **(limits or {})}}
        self.limits = self.state["limits"]
        self.part = register(BlindPart(self.dir / "geometry.step", self.state["label"]))
        self.results: dict[int, object] = {}    # solve number -> feacore Result (memory only)
        self._fmodel = None                      # feagent GeometryModel (find_faces), built on first use
        self._edges = None                       # CAD edges: tag -> (faces, sample points)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def finished(self) -> bool:
        return self.state["report"] is not None

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=1, default=float))

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Run one tool call; returns (result for the agent, is_error). Never raises."""
        t0 = time.time()
        if self.finished:
            out, err = {"error": "the report is already submitted; the analysis is over"}, True
        elif name not in TOOL_NAMES:
            out, err = {"error": f"unknown tool {name!r}; tools: {TOOL_NAMES}"}, True
        else:
            try:
                out, err = getattr(self, "_" + name)(dict(args or {}))
            except Exception as e:  # noqa: BLE001  a crash in a tool is the agent's information, not the run's end
                out, err = {"error": f"{name} failed: {type(e).__name__}: {e}"}, True
        self.state["log"].append({"tool": name, "error": err, "seconds": round(time.time() - t0, 2)})
        self.save()
        return out, err

    # ------------------------------------------------------------------ geometry
    def _inspect_geometry(self, args: dict) -> tuple[dict, bool]:
        from ace import fea, geometry_info
        with quiet():
            full = {**fea.geometry_summary(self.part), **geometry_info.describe(self.part)}
        return geometry_info.for_agent(full), False

    def _feagent_model(self):
        if self._fmodel is None:
            from feagent.geometry import analyze
            with quiet():
                self._fmodel = analyze(str(self.part.step_path))
        return self._fmodel

    def _find_faces(self, args: dict) -> tuple[dict, bool]:
        from feagent.errors import QueryError
        from feagent.select import describe_faces, select_faces
        q = args.get("query")
        if not isinstance(q, dict):
            return {"error": "query must be an object, e.g. {\"kind\": \"plane\", \"normal\": [0, 0, -1]}"}, True
        m = self._feagent_model()
        try:
            tags = select_faces(m, q)
        except QueryError as e:
            return {"error": str(e)}, True
        return {"faces": [f"f{t}" for t in tags], "count": len(tags),
                "details": describe_faces(m, tags).splitlines() if tags else []}, False

    def _resolve_faces(self, names: list[str]) -> tuple[list[int], str | None]:
        tags = []
        for n in names:
            if isinstance(n, str) and n.startswith("f") and n[1:].isdigit():
                tags.append(int(n[1:]))
            elif n in self.part.groups:
                tags += self.part.groups[n]
            else:
                return [], f"unknown face or group {n!r} (faces are f<tag>; groups are defined in mesh)"
        return tags, None

    def _view_geometry(self, args: dict) -> tuple[dict, bool]:
        if self.state["views"] >= self.limits["max_views"]:
            return {"error": f"picture limit ({self.limits['max_views']}) reached"}, True
        hl = args.get("highlight") or []
        tags, err = self._resolve_faces(hl)
        if err:
            return {"error": err}, True
        view = args.get("view", "iso")
        if view not in VIEW_ENUM:
            return {"error": f"view must be one of {VIEW_ENUM}"}, True
        self.state["views"] += 1
        png = self.dir / f"view_{self.state['views']}_geometry_{view.replace('-', 'm')}.png"
        from single_agent.render import render_geometry
        with quiet():
            seen = render_geometry(self.part, tags, view, bool(args.get("label_all")), png)
        return {"view": view, "highlighted": [f"f{t}" for t in tags], "visible_highlighted": seen,
                "image": str(png)}, False

    # ------------------------------------------------------------------ meshing
    def _mesh(self, args: dict) -> tuple[dict, bool]:
        groups = args.get("groups") or {}
        if not isinstance(groups, dict):
            return {"error": "groups must be an object {name: [faces]}"}, True
        resolved = {}
        for name, faces in groups.items():
            if not isinstance(name, str) or not name or (name.startswith("f") and name[1:].isdigit()) or name == "solid":
                return {"error": f"group name {name!r} is reserved or invalid"}, True
            tags, err = self._resolve_faces(list(faces or []))
            if err or not tags:
                return {"error": err or f"group {name!r} has no faces"}, True
            resolved[name] = sorted(set(tags))
        request = {"h": args.get("h"), "order": args.get("order", 2), "refinements": args.get("refinements") or [],
                   "groups": {k: [f"f{t}" for t in v] for k, v in resolved.items()}}
        return self._make_mesh(request, resolved)

    def _make_mesh(self, request: dict, resolved_groups: dict) -> tuple[dict, bool]:
        from ace import fea
        if not isinstance(request.get("h"), (int, float)) or request["h"] <= 0:
            return {"error": "h must be a positive number (metres)"}, True
        k = len(self.state["meshes"]) + 1
        path = self.dir / f"mesh_{k}.msh"
        self.part.groups = resolved_groups
        try:
            with quiet():
                m = fea.make_mesh(self.part, path, h=float(request["h"]), order=int(request["order"]),
                                  refinements=request["refinements"])
        finally:
            self.part.groups = {}
        m = {k2: v for k2, v in m.items() if k2 not in ("sha256", "gmsh")}
        m.update(number=k, request=request)
        self.state["meshes"].append(m)
        out = {"mesh": k, **{k2: v for k2, v in m.items() if k2 not in ("path", "request", "number")}}
        if not m.get("ok"):
            return out, True
        cap = self.limits["node_cap"]
        out["node_cap"] = cap
        if m["nodes"] > cap:
            f = (m["nodes"] / cap) ** (1 / 3)
            out["warning"] = (f"{m['nodes']} nodes is over the cap of {cap}: this mesh cannot be solved. Sizes about "
                              f"{f:.2f}x larger would fit (h = {m['h_used'] * f:.4g}).")
        if m.get("folded_cells"):
            out["warning_folded"] = f"{m['folded_cells']} cells are folded: a solve on this mesh is invalid"
        if m["min_quality"] < config.MIN_QUALITY:
            out["warning_quality"] = (f"min element quality {m['min_quality']} < {config.MIN_QUALITY}: answers resting "
                                      f"on this mesh are not trusted. Smoother size transitions usually fix it.")
        return out, False

    def _refine_mesh(self, args: dict) -> tuple[dict, bool]:
        k, f = args.get("mesh"), args.get("factor")
        meshes = self.state["meshes"]
        if not isinstance(k, int) or not 1 <= k <= len(meshes) or not meshes[k - 1].get("ok"):
            return {"error": f"mesh {k!r} is not a successful mesh"}, True
        if not isinstance(f, (int, float)) or not 1.05 <= f <= 4:
            return {"error": "factor must be between 1.05 and 4"}, True
        req = json.loads(json.dumps(meshes[k - 1]["request"]))
        req["h"] = req["h"] / f
        for r in req["refinements"]:
            r["size"] = r["size"] / f
        resolved = {name: self._resolve_faces(faces)[0] for name, faces in req["groups"].items()}
        out, err = self._make_mesh(req, resolved)
        out["refined_from"] = {"mesh": k, "factor": f}
        return out, err

    # ------------------------------------------------------------------ solving
    def _mesh_record(self, k) -> tuple[dict | None, str | None]:
        meshes = self.state["meshes"]
        if k is None:
            k = len(meshes)
        if not isinstance(k, int) or not 1 <= k <= len(meshes):
            return None, f"mesh {k!r} does not exist (meshes: 1..{len(meshes)})"
        m = meshes[k - 1]
        if not m.get("ok"):
            return None, f"mesh {k} failed; mesh again"
        if m["nodes"] > self.limits["node_cap"]:
            return None, f"mesh {k} has {m['nodes']} nodes, over the cap of {self.limits['node_cap']}"
        if m.get("folded_cells"):
            return None, f"mesh {k} has folded cells"
        return m, None

    def _solve(self, args: dict) -> tuple[dict, bool]:
        n_ok = len(self.state["solves"])
        if n_ok >= self.limits["max_solves"]:
            return {"error": f"solve limit ({self.limits['max_solves']}) reached: run study on the solves you have, "
                             f"then submit_report"}, True
        mesh, err = self._mesh_record(args.get("mesh"))
        if err:
            return {"error": err + " (this did not use a solve)"}, True
        setup = {"material": args.get("material"), "bcs": args.get("bcs") or [], "loads": args.get("loads") or [],
                 "qois": args.get("qois") or []}
        names = [q.get("name") for q in setup["qois"]]
        if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            return {"error": "every QoI needs a unique, non-empty name (this did not use a solve)"}, True
        result, err = solve_setup(mesh, setup)
        n = n_ok + 1
        rec = {"number": n, "mesh": mesh["number"], "setup": setup, "ok": result is not None}
        if result is None:
            rec["error"] = err
            self.state["solves"].append(rec)
            return {"solve": n, "ok": False, "error": err, "solves_left": self.limits["max_solves"] - n}, True
        self.results[n] = result
        values = evaluate_qois(result, setup["qois"])
        qerr = values.pop("_errors", None)
        info = result.info
        rec.update(qois=values, dofs=int(info.n_dofs), cells=int(info.n_cells), nodes=int(info.n_nodes),
                   h_rep=float(info.h_rep), order=int(info.order), wall_time=round(float(info.wall_time), 2))
        out = {"solve": n, "mesh": mesh["number"], "ok": True, "dofs": rec["dofs"], "h_rep": _sig(rec["h_rep"]),
               "qois": {k: _sig(v, 6) for k, v in values.items()}, "solves_left": self.limits["max_solves"] - n}
        if qerr:
            out["qoi_errors"] = qerr
        try:
            out["feedback"] = self._feedback(mesh, setup, result)
        except Exception as e:  # noqa: BLE001
            out["feedback"] = {"error": f"feedback failed: {type(e).__name__}: {e}"}
        rec["feedback"] = {k: out["feedback"].get(k) for k in ("equilibrium", "max_displacement", "max_von_mises",
                                                               "error_estimate", "notes")
                           if isinstance(out["feedback"], dict)}
        self.state["solves"].append(rec)
        return out, False

    def _feedback(self, mesh: dict, setup: dict, result) -> dict:
        from ace import solve_feedback
        self.part.agent_qois = []          # the QoIs are reported above; keep the feedback to the physics
        with quiet():
            fb = solve_feedback.feedback(self.part, mesh, setup, result)
        fb.pop("qois", None)
        supported = []
        for b in setup.get("bcs", []):
            if b.get("type") in ("fixed", "displacement", "symmetry") and b.get("group"):
                tags, err = self._resolve_faces([b["group"]])
                if not err:
                    supported += tags
        notes = fb.setdefault("notes", [])
        rims = self._support_rims(set(supported))
        if rims:
            near = []
            radius = 0.05 * solve_feedback._part(result).diag     # the hotspot cluster radius (ace.inspect_tools)
            for h in fb.get("error_estimate", {}).get("hotspots", []):
                x = np.asarray(list(h["at"]) + [0.0] * (3 - len(h["at"])))
                d, tag = min((float(np.linalg.norm(pts - x, axis=1).min()), t) for t, pts in rims.items())
                h["nearest_support_rim_edge"] = {"tag": tag, "distance": _sig(d, 3)}
                if d <= max(h.get("local_h") or 0.0, radius):
                    near.append(h)
            if near:
                notes.append(f"{sum(h['share'] for h in near):.0%} of the estimated error sits at the rim of your "
                             f"supports (edges {sorted({h['nearest_support_rim_edge']['tag'] for h in near})}): a rigid "
                             f"support's edge is a stress singularity, so peak stresses there grow under refinement")
        if not notes:
            fb.pop("notes")
        return fb

    def _support_rims(self, supported: set[int]) -> dict:
        """{edge tag: sample points} of the CAD edges between a supported face and an unsupported one."""
        if not supported:
            return {}
        if self._edges is None:
            import gmsh

            from feacore.geometry.gmsh_tools import gmsh_session
            edges = {}
            with quiet(), gmsh_session(self.part.id + "-edges"):
                self.part.build_geometry()
                vol = [t for _, t in gmsh.model.getEntities(3)]
                faces = {abs(t) for _, t in gmsh.model.getBoundary([(3, v) for v in vol], oriented=False)}
                for _, e in gmsh.model.getEntities(1):
                    up = [int(f) for f in gmsh.model.getAdjacencies(1, e)[0] if int(f) in faces]
                    if len(up) != 2:
                        continue
                    lo, hi = gmsh.model.getParametrizationBounds(1, e)
                    pts = np.asarray(gmsh.model.getValue(1, e, list(np.linspace(lo[0], hi[0], 48)))).reshape(-1, 3)
                    edges[e] = (up, pts)
            self._edges = edges
        return {e: pts for e, (up, pts) in self._edges.items() if (up[0] in supported) != (up[1] in supported)}

    def _result(self, n) -> tuple[object | None, str | None]:
        """The Result of solve n, recomputed from its record when this process has not got it (CLI)."""
        solves = self.state["solves"]
        if not isinstance(n, int) or not 1 <= n <= len(solves) or not solves[n - 1]["ok"]:
            ok = [s["number"] for s in solves if s["ok"]]
            return None, f"solve {n!r} is not a successful solve; successful solves: {ok}"
        if n not in self.results:
            rec = solves[n - 1]
            mesh = self.state["meshes"][rec["mesh"] - 1]
            res, err = solve_setup(mesh, rec["setup"])
            if res is None:
                return None, f"could not recompute solve {n}: {err}"
            self.results[n] = res
        return self.results[n], None

    # ------------------------------------------------------------------ verification
    def _study(self, args: dict) -> tuple[dict, bool]:
        from feacore.verification.gci import analyze
        nums = args.get("solves")
        if not isinstance(nums, list) or len(nums) < 2:
            return {"error": "give at least 2 solve numbers (3+ for a GCI)"}, True
        solves = self.state["solves"]
        recs = []
        for n in nums:
            if not isinstance(n, int) or not 1 <= n <= len(solves) or not solves[n - 1]["ok"]:
                return {"error": f"solve {n!r} is not a successful solve"}, True
            recs.append(solves[n - 1])
        setups = {canonical_setup(r["setup"]) for r in recs}
        if len(setups) > 1:
            return {"error": "these solves do not share one setup (material, supports and loads must be identical; "
                             "only the mesh may change)"}, True
        if len({r["mesh"] for r in recs}) < len(recs):
            return {"error": "two of these solves are on the same mesh"}, True
        recs = sorted(recs, key=lambda r: -r["h_rep"])                 # coarse -> fine
        orders = {r["order"] for r in recs}
        order = orders.pop() if len(orders) == 1 else None
        common = [q for q in recs[-1]["setup"]["qois"]
                  if all(r["qois"].get(q["name"]) is not None for r in recs)]
        h = [r["h_rep"] for r in recs]
        ratios = [h[i] / h[i + 1] for i in range(len(h) - 1)]
        out = {"solves": [r["number"] for r in recs], "h_rep": [_sig(x) for x in h], "dofs": [r["dofs"] for r in recs],
               "refinement_ratios": [_sig(x, 3) for x in ratios], "qois": {}}
        notes = []
        if order is None:
            notes.append("the solves mix element orders: not a systematic refinement, the orders below are unreliable")
        if min(ratios) < 1.15:
            notes.append("a refinement ratio is below 1.15: mesh-to-mesh changes are then mostly noise; use factors of "
                         "1.3-2 between levels")
        for q in common:
            phi = [r["qois"][q["name"]] for r in recs]
            p_th = expected_order(q, order) if order else None
            row = {"values": [_sig(v, 6) for v in phi], "p_theory": p_th}
            if len(recs) >= 3:
                try:
                    a = analyze(h, phi, p_theory=p_th)
                    g = a["gci"]
                except (ValueError, OverflowError, ZeroDivisionError) as e:
                    # feacore's gci3 can fail on unequal ratios (math domain error in its fixed-point order);
                    # the ratio-corrected test below still decides the case.
                    a, g = {}, {"convergence": "indeterminate", "p": None, "phi_ext": None, "gci_fine": None,
                                "e_a21": None, "flags": [f"gci3 failed: {type(e).__name__}: {e}"]}
                idx = np.argsort(h)[:3]                        # the three finest, as feacore's analyze uses
                fix = gci_unequal([h[i] for i in idx], [phi[i] for i in idx], p_th)
                if fix:                                        # unequal ratios: the ratio-corrected verdict wins
                    g = {**g, **fix}
                row.update(convergence=g["convergence"], p_observed=_sig(g["p"], 3), extrapolated=_sig(g["phi_ext"], 6),
                           gci_fine=_sig(g["gci_fine"], 3), gci_conservative=_sig(g.get("gci_fine_conservative"), 3),
                           change_fine_vs_medium=_sig(g["e_a21"], 3), flags=g["flags"])
                if a.get("fit"):
                    f = a["fit"]
                    row["power_fit"] = {"limit": _sig(f["phi0"], 6), "p": _sig(f["p"], 3), "n": f["n"]}
                row["reading"] = _reading(row)
            else:
                ch = abs(phi[1] - phi[0]) / max(abs(phi[1]), 1e-300)
                row["change"] = _sig(ch, 3)
                row["reading"] = (f"changed {ch:.2%} between the two meshes; a third mesh is needed for an order and a "
                                  f"GCI")
            out["qois"][q["name"]] = row
        if notes:
            out["notes"] = notes
        self.state["studies"].append({"solves": out["solves"], "qois": out["qois"]})
        return out, False

    # ------------------------------------------------------------------ pictures, comparisons
    def _view(self, args: dict) -> tuple[dict, bool]:
        if self.state["views"] >= self.limits["max_views"]:
            return {"error": f"picture limit ({self.limits['max_views']}) reached"}, True
        n, kind, view = args.get("solve"), args.get("kind", "stress"), args.get("view", "iso")
        res, err = self._result(n)
        if err:
            return {"error": err}, True
        from ace import solve_feedback
        rec = self.state["solves"][n - 1]
        mesh = self.state["meshes"][rec["mesh"] - 1]
        self.state["views"] += 1
        png = self.dir / f"view_{self.state['views']}_solve{n}_{kind}_{view.replace('-', 'm')}.png"
        self.part.agent_qois = []
        with quiet():
            solve_feedback.render(self.part, mesh, res, kind, view, png)
        return {"solve": n, "kind": kind, "view": view, "image": str(png)}, False

    def _compare(self, args: dict) -> tuple[dict, bool]:
        from feacore.case import QoI
        a, b = args.get("solve_a"), args.get("solve_b")
        ra, err = self._result(a)
        if err:
            return {"error": err}, True
        rb, err = self._result(b)
        if err:
            return {"error": err}, True
        from ace import solve_feedback
        qa = {q["name"]: q for q in self.state["solves"][a - 1]["setup"]["qois"]}
        qb = {q["name"]: q for q in self.state["solves"][b - 1]["setup"]["qois"]}
        self.part.agent_qois = [QoI(**{k: v for k, v in qa[n].items() if k in ("name", "type", "field", "group",
                                                                                "point", "component")})
                                for n in qa if n in qb]
        try:
            with quiet():
                out = solve_feedback.compare(self.part, ra, rb)
        finally:
            self.part.agent_qois = []
        return {"solve_a": a, "solve_b": b, **out}, False

    # ------------------------------------------------------------------ the deliverable
    def _submit_report(self, args: dict) -> tuple[dict, bool]:
        solves = self.state["solves"]
        n = args.get("final_solve")
        if not isinstance(n, int) or not 1 <= n <= len(solves) or not solves[n - 1]["ok"]:
            ok = [s["number"] for s in solves if s["ok"]]
            return {"error": f"final_solve {n!r} is not a successful solve; successful solves: {ok}"}, True
        answers = args.get("answers")
        if not isinstance(answers, list) or not answers:
            return {"error": "answers must list one entry per review question"}, True
        for a in answers:
            if not isinstance(a, dict) or not a.get("id") or a.get("status") not in ("converged", "not_converged",
                                                                                     "singular"):
                return {"error": f"each answer needs an id and a status (converged / not_converged / singular): {a!r}"}, True
            if a.get("value") is not None and not isinstance(a["value"], (int, float)):
                return {"error": f"answer {a['id']}: value must be a number or null"}, True
        self.state["report"] = {k: args.get(k) for k in ("idealization", "final_solve", "study_solves", "answers",
                                                         "assumptions", "rules_used", "recommendations", "summary")}
        return {"submitted": True, "message": "report received; the analysis is over"}, False


def _reading(row: dict) -> str:
    """One sentence on what a study row means for reporting the QoI."""
    conv, p, pt, gci = row.get("convergence"), row.get("p_observed"), row.get("p_theory"), row.get("gci_fine")
    if conv in ("divergent", "oscillatory_divergent"):
        return ("diverging: the changes grow under refinement. For a stress this is the signature of a singularity "
                "(re-entrant edge, support rim, point load); no mesh gives a trustworthy value")
    if conv == "converged":
        return f"converged within round-off (uncertainty {gci:.2g})" if gci is not None else "converged within round-off"
    if conv == "oscillatory":
        return ("oscillating between meshes: use the spread of the values as the uncertainty; the extrapolation is not "
                "reliable")
    if conv == "monotone" and p is not None:
        s = f"monotone, observed order {p:g}"
        if pt:
            s += f" (theory {pt:g})"
        if gci is not None:
            s += f"; the finest value is uncertain by about {gci:.2%} (GCI)"
        cons = row.get("gci_conservative")
        if pt and p < 0.5 * pt:
            s += ". The order is far below theory: a singularity nearby is polluting it, or the meshes are not in the " \
                 "asymptotic range; treat the GCI as optimistic"
        elif pt and p > 1.5 * pt + 0.5:
            s += (". The order is far ABOVE theory: the meshes are not yet in the asymptotic range or the refinement "
                  "ratios are too close, so the extrapolation and the GCI are unreliable and usually optimistic")
            if cons is not None:
                s += f"; the conservative GCI (safety 3, order capped at theory) is {cons:.2%}"
        return s
    return f"{conv}: no order could be computed"
