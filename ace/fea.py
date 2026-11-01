"""Plain-Python FEA plumbing behind the agent's tools: geometry summary, meshing, solving.

Two physics: linear elasticity (skfem backend) and steady incompressible flow (physics "flow",
openfoam backend). For flow the cells are always linear finite volumes (2D: counter-clockwise in the
xy-plane, optionally quads); the agent's "order" choice is the convection scheme instead
(upwind = 1, linearUpwind = 2), and the solver controls (laminar, iteration cap, residual target)
come from the problem, not the agent.

No LLM here. gmsh is process-global, so call these from one thread per process
(run_loop runs each agent run in its own process).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import gmsh
import numpy as np

from ace.config import MIN_QUALITY
from feacore.case import BC, Analysis, Case, CaseError, Load, Material
from feacore.geometry import fv_quality
from feacore.geometry.gmsh_tools import distance_threshold_field, ensure_domain_group, gmsh_session
from feacore.problems import registry
from feacore.problems.base import Problem
from feacore.problems.cfd._common import orient_ccw_2d
from feacore.solver import solve

from ace import flow_mesh


CONVECTION = {"upwind": 1, "linearUpwind": 2}       # flow: scheme name -> analysis.order


def get_problem(problem_id: str) -> Problem:
    """The problem as the loop sees it. In requirements mode (config.STATEMENT), groups whose
    names give a support or load away (e.g. "clamped") get the neutral names stored with the
    problem's requirements (ace.requirements); every module looks problems up here, so meshes,
    setups, reference setups and reports all use the same names."""
    from ace import config
    problem = registry.get(problem_id)
    if config.STATEMENT == "requirements":
        from ace import requirements
        aliases = requirements.aliases_for(problem_id)
        if aliases:
            return AliasedProblem(problem, aliases)
    return problem


class AliasedProblem:
    """A problem whose named groups are renamed (real name -> alias) everywhere the loop can see
    them: the geometry it builds, its reference case and its QoIs. Everything else is the wrapped
    problem's. Group names not in `aliases` are unchanged."""

    def __init__(self, problem: Problem, aliases: dict[str, str]):
        self._problem = problem
        self.aliases = dict(aliases)

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def alias(self, group: str | None) -> str | None:
        return self.aliases.get(group, group) if group else group

    def build_geometry(self) -> None:
        self._problem.build_geometry()
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            if name in self.aliases:
                gmsh.model.removePhysicalName(name)       # setPhysicalName alone keeps an existing name
                gmsh.model.setPhysicalName(dim, tag, self.aliases[name])

    def qois(self):
        return [dataclasses.replace(q, group=self.alias(q.group)) for q in self._problem.qois()]

    def case(self, mesh_path, order: int = 2) -> Case:
        c = self._problem.case(mesh_path, order)
        c.bcs = [dataclasses.replace(b, group=self.alias(b.group)) for b in c.bcs]
        c.loads = [dataclasses.replace(ld, group=self.alias(ld.group)) for ld in c.loads]
        c.regions = [dataclasses.replace(r, group=self.alias(r.group)) for r in c.regions]
        c.qois = [dataclasses.replace(q, group=self.alias(q.group)) for q in c.qois]
        return c


def is_flow(problem: Problem) -> bool:
    return problem.physics == "flow"


def official_qoi(problem: Problem, qoi_name: str):
    return next(q for q in problem.qois() if q.name == qoi_name)


def reference_value(problem: Problem, qoi_name: str) -> float:
    return next(r.value for r in problem.references() if r.qoi == qoi_name)


# --------------------------------------------------------------------------- geometry


def _bbox_of(dim_tags) -> list[float]:
    boxes = [gmsh.model.getBoundingBox(d, t) for d, t in dim_tags]
    lo = [min(b[i] for b in boxes) for i in range(3)]
    hi = [max(b[i + 3] for b in boxes) for i in range(3)]
    return [round(v, 6) for v in lo + hi]


def build_agent_geometry(problem: Problem, sweep: dict | None = None) -> dict | None:
    """In the current gmsh session: the geometry the agent inspects and meshes, with its named
    groups. A flow problem's is its CAD (problem.build_cad) with the reference geometry's group
    names (ace.flow_mesh.build; with `sweep` rebuilt as a swept face), not the block-split geometry
    its structured reference mesh is made on; returns flow_mesh's info then. Others: the problem's
    geometry."""
    if is_flow(problem):
        info = flow_mesh.build(problem, sweep)
        ensure_domain_group(problem.dim)
        return info
    problem.build_geometry()
    ensure_domain_group(problem.dim)
    return None


def geometry_summary(problem: Problem) -> dict:
    """Named groups (what BCs and refinements can refer to) with their extents."""
    with gmsh_session(problem.id):
        build_agent_geometry(problem)
        groups = []
        for dim, ptag in gmsh.model.getPhysicalGroups():
            ents = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
            dim_tags = [(dim, int(e)) for e in ents]
            measure = sum(gmsh.model.occ.getMass(dim, int(e)) for e in ents) if dim > 0 else 0.0
            groups.append({
                "name": gmsh.model.getPhysicalName(dim, ptag),
                "dim": dim,
                "kind": ["point", "curve", "surface", "volume"][dim],
                "n_entities": len(ents),
                "bbox": _bbox_of(dim_tags),
                "measure": round(measure, 6),   # length / area / volume
            })
        bbox = _bbox_of(gmsh.model.getEntities(problem.dim))
    return {"dim": problem.dim, "bbox": bbox, "groups": groups}


def _size_floor(problem: Problem, bbox: list[float]) -> float:
    """Smallest element size we allow, so a bad request can't produce a huge mesh that hangs."""
    diag = math.dist(bbox[:3], bbox[3:])
    return diag / (100.0 if problem.dim == 3 else 400.0)


def _local_floor(problem: Problem, bbox: list[float]) -> float:
    """Smallest local refinement size. Flow domains are long and their bodies small (the DFG
    cylinder is 1/22 of the channel), so local sizes may go 5x (2D) / 3x (3D) below the global
    floor there; h itself keeps the global floor, which is what bounds the mesh size."""
    floor = _size_floor(problem, bbox)
    if not is_flow(problem):
        return floor
    return floor / (3.0 if problem.dim == 3 else 5.0)


SWEEP_MAX_NODES = 2e6     # a swept mesh's estimated node count above which the request is refused

# The mesh request keys make_mesh takes; a recorded request (the checkpoint) rebuilds the mesh.
MESH_KEYS = ("h", "order", "refinements", "quads", "boundary_layer", "sweep")


def mesh_from_request(problem: Problem, path: Path, request: dict) -> dict:
    """make_mesh from a recorded mesh request (the agent's mesh tool arguments)."""
    return make_mesh(problem, path, **{k: request[k] for k in MESH_KEYS if k in request and request[k] is not None})


def scale_request(request: dict, s: float) -> dict:
    """The request with every length scaled by s (s < 1: finer), for refinement studies. Element
    sizes scale by s; a boundary layer keeps its thickness (first height x s, growth 1 + s (ratio - 1),
    layers / s); a sweep gets layers / s."""
    out = dict(request)
    out["h"] = request["h"] * s
    out["refinements"] = [dict(r, size=r["size"] * s) if "size" in r else dict(r)
                          for r in request.get("refinements") or []]
    bl = request.get("boundary_layer")
    if bl:
        out["boundary_layer"] = dict(bl, first=float(bl["first"]) * s,
                                     ratio=1.0 + s * (float(bl.get("ratio", 1.2)) - 1.0),
                                     layers=max(1, int(round(int(bl["layers"]) / s))))
    sw = request.get("sweep")
    if sw:
        out["sweep"] = dict(sw, layers=max(1, int(round(int(sw["layers"]) / s))))
    return out


# --------------------------------------------------------------------------- mesh


def make_mesh(problem: Problem, path: Path, h: float, order: int = 1,
              refinements: list[dict] | None = None, quads: bool = False,
              boundary_layer: dict | None = None, sweep: dict | None = None) -> dict:
    """Mesh the problem's geometry.

    h: global element size. order: 1 (linear) or 2 (quadratic, curved); flow meshes are always
    linear. refinements: [{"group": name | "edge": curve tag | "point": [x, y(, z)], "size": s,
      "radius": r}] element size s at the group, CAD curve or point, growing linearly to h at
      distance r; or {"box": [xmin, ymin, zmin, xmax, ymax, zmax], "size": s}: size s inside the box.
    Flow only (ace.flow_mesh):
      quads: recombine into quad-dominant cells (2D, or the swept face of a sweep: hexahedra);
      boundary_layer: {"groups": [wall groups], "first": height, "ratio": growth, "layers": n}
        layers of thin cells along the walls (2D, or on the swept face in 3D);
      sweep (3D): {"source": group of a planar end face, "layers": n, "ratio": last / first}
        the domain as that face extruded in n layers (hexahedra / prisms).
    Order 2 meshes are checked with the solver's Jacobian test; curved cells that fold or nearly
    fold are straightened (`repaired_cells`), so `folded_cells` should be 0. Flow meshes are judged
    by their finite-volume quality (`fv_quality`, feacore.geometry.fv_quality; `fv_ok`).
    Returns stats; `ok` is False with an `error` message on failure.
    """
    if order not in (1, 2):
        return {"ok": False, "error": "order must be 1 or 2"}
    flow = is_flow(problem)
    if flow:
        order = 1                           # finite volumes: straight-sided cells
    if not flow and (boundary_layer or sweep):
        return {"ok": False, "error": "boundary_layer and sweep are for flow meshes"}
    try:
        sweep = flow_mesh.sweep_spec(sweep) if flow else None
    except flow_mesh.FlowMeshError as e:
        return {"ok": False, "error": str(e)}
    if quads and not (flow and (problem.dim == 2 or sweep)):
        return {"ok": False, "error": "quads is only available for 2D flow meshes and swept 3D flow meshes"}
    if boundary_layer and flow and problem.dim == 3 and not sweep:
        return {"ok": False, "error": "boundary_layer in 3D needs a sweep (the layers are made on the swept face)"}
    # gmsh's default surface mesher sometimes leaves flat tets between boundary curves of a CAD
    # part (all four nodes on CAD edges, so no optimizer can move them); MeshAdapt surfaces avoid
    # that on the gold parts. Both can fail on a curved face much smaller than h ("overlapping
    # facets"); sizing by curvature resolves it at the price of extra elements there, so it is the
    # last resort, and it starts at few elements per 2 pi (8 per 2 pi puts ~7000 nodes on a gold
    # part whatever h is, which would forbid coarse meshes). The first mesh that passes the scoring
    # quality gate is kept (MIN_QUALITY; flow: the finite-volume gate), else the best one. Each
    # attempt is a fresh gmsh session (after a failed generate gmsh fails every later one) writing
    # its own file.
    path = Path(path)
    errors, best = [], None
    for i, mesher in enumerate(MESHERS):
        res = _mesh_once(problem, path.with_suffix(f".try{i}.msh"), h, order, refinements or [], mesher,
                         quads, boundary_layer, sweep)
        if not res["ok"]:
            if not res.pop("retry", False):
                return res               # a bad request: every mesher would refuse it
            errors.append(res["error"])
            continue
        if best is None or _better(res, best, flow):
            best = res
        if (res["fv_ok"] if flow else res["min_quality"] >= MIN_QUALITY):
            break
    for f in path.parent.glob(path.stem + ".try*.msh"):
        if best is None or str(f) != best["path"]:
            f.unlink()
    if best is None:
        return {"ok": False, "error": errors[0]}
    Path(best["path"]).replace(path)
    return best | {"path": str(path)}


def _better(a: dict, b: dict, flow: bool) -> bool:
    """Is mesh a better than mesh b by the gate's measure? Flow: passing the finite-volume gate,
    then the lower max non-orthogonality; else the higher min quality."""
    if not flow:
        return a["min_quality"] > b["min_quality"]
    if a["fv_ok"] != b["fv_ok"]:
        return a["fv_ok"]
    na = (a["fv_quality"].get("non_orthogonality") or {}).get("max") or 0.0
    nb = (b["fv_quality"].get("non_orthogonality") or {}).get("max") or 0.0
    return na < nb


def _mesh_once(problem: Problem, path: Path, h: float, order: int, refinements: list[dict],
               mesher: tuple[int, int], quads: bool = False, boundary_layer: dict | None = None,
               sweep: dict | None = None) -> dict:
    """make_mesh with one mesher (gmsh surface algorithm, curvature sizing). A gmsh failure comes
    back with "retry": True."""
    flow = is_flow(problem)
    with gmsh_session(problem.id):
        try:
            info = build_agent_geometry(problem, sweep)
        except flow_mesh.FlowMeshError as e:
            return {"ok": False, "error": str(e)}
        bbox = _bbox_of(gmsh.model.getEntities(problem.dim))
        floor = _size_floor(problem, bbox)
        local_floor = _local_floor(problem, bbox)
        sw = (info or {}).get("sweep")
        if sw:
            # A sweep meshes its face in 2D: the 2D floors of that face, and a bound on the node count
            # (face nodes x layers) instead of the 3D floor of the whole, long domain.
            faces = [(2, t) for t in sw["source_faces"]]
            fb = _bbox_of(faces)
            floor = math.dist(fb[:3], fb[3:]) / 200.0
            local_floor = floor / 5.0
            area = sum(gmsh.model.occ.getMass(2, t) for _, t in faces)
            est = area / max(float(h), floor) ** 2 * (sw["layers"] + 1)
            if est > SWEEP_MAX_NODES:
                return {"ok": False, "error": f"sweep: about {est:.3g} nodes (face area / h^2 x layers) is over "
                                              f"the {SWEEP_MAX_NODES:.0g} the mesher allows; use a larger h or "
                                              f"fewer layers"}
        h = max(float(h), floor)

        names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
        bl_applied = None
        if flow and boundary_layer:
            groups = info["groups"]
            try:
                bl = flow_mesh.bl_spec(boundary_layer, groups, problem.dim, math.dist(bbox[:3], bbox[3:]))
                flow_mesh.add_boundary_layer(bl, groups, problem.dim,
                                             (info.get("sweep") or {}).get("source_faces"))
            except flow_mesh.FlowMeshError as e:
                return {"ok": False, "error": str(e)}
            bl_applied = {k: bl[k] for k in ("groups", "first", "ratio", "layers", "thickness", "last")}
        fields, sizes_used = [], []
        for r in refinements:
            size = max(float(r.get("size", h)), local_floor)
            sizes_used.append(size)
            radius = float(r.get("radius", 0.0))
            if "box" not in r and radius <= 0:
                return {"ok": False, "error": f"refinement {r}: radius must be > 0"}
            if "group" in r:
                if r["group"] not in names:
                    return {"ok": False, "error": f"unknown group {r['group']!r}; groups: {sorted(names)}"}
                d, t = names[r["group"]]
                ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)]
                kw = {0: "points", 1: "curves", 2: "surfaces"}.get(d)
                if kw is None:
                    return {"ok": False, "error": f"cannot refine near a volume group ({r['group']!r})"}
                fields.append(distance_threshold_field(**{kw: ents}, size_min=size, size_max=h,
                                                       dist_min=0.0, dist_max=radius))
            elif "edge" in r:
                # A CAD curve by its tag (geometry_info lists them): re-entrant edges are where 3D
                # stresses are singular, and a face group refinement would refine whole faces instead.
                curves = {t for _, t in gmsh.model.getEntities(1)}
                if not isinstance(r["edge"], int) or r["edge"] not in curves:
                    return {"ok": False, "error": f"refinement {r}: 'edge' must be a curve tag; "
                                                  f"the geometry has curves 1..{max(curves, default=0)}"}
                fields.append(distance_threshold_field(curves=[r["edge"]], size_min=size, size_max=h,
                                                       dist_min=0.0, dist_max=radius))
            elif "box" in r:
                # Size `size` inside an axis-aligned box, h outside (radius is not used).
                b = [float(v) for v in r["box"]]
                if len(b) != 6 or b[0] > b[3] or b[1] > b[4] or b[2] > b[5]:
                    return {"ok": False, "error": f"refinement {r}: 'box' must be [xmin, ymin, zmin, xmax, ymax, zmax] "
                                                  f"(2D: any zmin <= 0 <= zmax)"}
                f = gmsh.model.mesh.field.add("Box")
                for k, v in zip(("XMin", "YMin", "ZMin", "XMax", "YMax", "ZMax"), b):
                    gmsh.model.mesh.field.setNumber(f, k, v)
                gmsh.model.mesh.field.setNumber(f, "VIn", size)
                gmsh.model.mesh.field.setNumber(f, "VOut", h)
                fields.append(f)
            elif "point" in r:
                # A Ball of zero radius whose transition layer is `radius` thick: size at the
                # point, growing linearly to h at distance radius, like the group Threshold above.
                x, y, z = (list(r["point"]) + [0.0, 0.0])[:3]
                f = gmsh.model.mesh.field.add("Ball")
                for k, v in {"XCenter": x, "YCenter": y, "ZCenter": z, "Radius": 0.0,
                             "Thickness": radius, "VIn": size, "VOut": h}.items():
                    gmsh.model.mesh.field.setNumber(f, k, v)
                fields.append(f)
            else:
                return {"ok": False, "error": f"refinement {r} needs 'group', 'edge', 'box' or 'point'"}

        if fields:
            m = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(m, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        for opt in ("MeshSizeFromPoints", "MeshSizeExtendFromBoundary"):   # curvature: see MESHERS
            gmsh.option.setNumber(f"Mesh.{opt}", 0)
        if quads:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)    # 2D cells / a sweep's face: quads (-> hexahedra)

        alg, curv = mesher
        gmsh.option.setNumber("Mesh.Algorithm", alg)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
        try:
            gmsh.model.mesh.generate(problem.dim)
        except Exception as e:  # noqa: BLE001  gmsh raises plain Exception
            return {"ok": False, "error": f"gmsh failed: {e}", "retry": True}
        if flow and problem.dim == 2:
            orient_ccw_2d()                 # the openfoam backend extrudes +z: cells must be counter-clockwise
        q = _qualities(problem.dim)
        if not len(q):
            return {"ok": False, "error": "gmsh failed: no elements generated", "retry": True}
        # Snapping boundary vertices onto curved CAD can leave a few nearly flat tets; gmsh's tet
        # optimizer + vertex relocation repairs those it can (as the gold generator does). Only when
        # needed, so meshes that are already fine stay exactly as gmsh made them. (A swept mesh has
        # no tets: its layers are exact.)
        if problem.dim == 3 and q.min() < REPAIR_TARGET and not (flow and sweep):
            for _ in range(3):
                gmsh.model.mesh.optimize("", niter=3)
                gmsh.model.mesh.optimize("Relocate3D", niter=3)
                q = _qualities(problem.dim)
                if q.min() >= REPAIR_TARGET:
                    break
        fv = None
        if flow:                                     # the finite-volume quality (the flow gate)
            try:
                fv = fv_quality.from_gmsh(problem.dim)
            except ValueError as e:                 # e.g. a non-manifold mesh
                return {"ok": False, "error": f"gmsh failed: {e}", "retry": True}
        # Quality and local sizes are measured on the straight-sided mesh, before curving to order 2.
        size_by_group = _size_by_group(problem.dim)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        repaired, jac = 0, {"folded": 0, "min_ratio": 1.0}   # straight cells: constant Jacobian
        if order == 2:
            gmsh.model.mesh.setOrder(2)
            # Curving boundary midside nodes onto the CAD can fold a thin cell. Straighten cells
            # whose scaled Jacobian is below the threshold (they lose their curvature, nothing else),
            # then check with the solver's own Jacobian test; escalate if any still fold, as a last
            # resort to an all straight-sided order-2 mesh (cannot fold where the corner tets are valid).
            straightened: set[int] = set()
            for threshold in (SJ_MIN, 0.2, math.inf):
                straightened |= _straighten_below(threshold, problem.dim)
                gmsh.write(str(path))
                jac = _jacobian_check(path, problem.dim)
                if jac["folded"] == 0:
                    break
            repaired = len(straightened)
        else:
            gmsh.write(str(path))
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        cells = flow_mesh.count_cell_types(problem.dim) if flow else None
    return {
        "ok": True,
        "path": str(path),
        "nodes": n_nodes,
        "elements": len(q),
        "order": order,
        **(_flow_stats(fv, quads, bl_applied, info, cells) if flow else {}),
        "min_quality": round(float(q.min()), 4),
        # minSICN of the straight-sided cells: 1 = regular, near 0 = flat
        "quality": {k: round(float(f(q)), 4) for k, f in
                    (("min", np.min), ("p01", lambda a: np.percentile(a, 1)), ("median", np.median))},
        "repaired_cells": repaired,                    # curved cells straightened to keep them valid
        "folded_cells": jac["folded"],                 # still folded (Jacobian changes sign): should be 0
        "min_jacobian_ratio": jac["min_ratio"],        # min over cells of min/max det J; <= 0 means folded
        "size_by_group": size_by_group,
        "mesher": MESHER_NAMES[mesher],    # +curvature: extra elements on small curved faces
        "h_used": h,                       # after clamping to the size floor
        "sizes_used": sizes_used,          # refinement sizes after clamping, in request order
        # Checkpoint fingerprint: a rebuild from the same request reproduces the mesh only with
        # the same code and gmsh version; the hash says whether it did.
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gmsh": gmsh.__version__,
    }


def _flow_stats(fv: dict, quads: bool, bl: dict | None, info: dict | None, cells: dict | None) -> dict:
    """The flow-only mesh stats: cell types, the finite-volume quality and its gate, and what the
    boundary layer / sweep made."""
    ok, why = fv_quality.passes(fv)
    out = {"quads": bool(quads), "cells": cells, "fv_quality": fv, "fv_ok": ok}
    if why:
        out["fv_gate"] = why
    if bl:
        out["boundary_layer"] = bl
    if info and info.get("sweep"):
        out["sweep"] = {k: v for k, v in info["sweep"].items() if k != "source_faces"}
    return out


REPAIR_TARGET = MIN_QUALITY   # min straight-cell quality (minSICN) the linear repair aims for: the scoring gate
# (gmsh Mesh.Algorithm, Mesh.MeshSizeFromCurvature) tried in turn until one passes the gate:
# Frontal-Delaunay surfaces (gmsh's default), MeshAdapt, then curvature sizing with 4, 6 and 8
# (as feacore.goldgen) elements per 2 pi. Nodes grow with the elements per 2 pi, quality does not
# reliably, so the coarse settings come first.
MESHERS = ((6, 0), (1, 0), (1, 4), (6, 4), (1, 6), (1, 8))
MESHER_NAMES = {(6, 0): "frontal-delaunay", (1, 0): "meshadapt", (1, 4): "meshadapt+curvature4",
                (6, 4): "frontal-delaunay+curvature4", (1, 6): "meshadapt+curvature6",
                (1, 8): "meshadapt+curvature8"}
SJ_MIN = 0.05          # curved cells below this min scaled Jacobian are straightened (as feacore.goldgen)
CURVED_TYPE = {2: 9, 3: 11}          # tri6, tet10
# (corner a, corner b, midside node) in gmsh's node ordering
MIDSIDE = {2: [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
           3: [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]}


def _qualities(dim: int) -> np.ndarray:
    """minSICN of every cell of dimension dim: all element types, so 2D flow meshes recombined into
    quads (quads + some triangles) count too."""
    _, tags, _ = gmsh.model.mesh.getElements(dim)
    tags = np.concatenate([np.asarray(t, dtype=np.int64) for t in tags]) if len(tags) else np.zeros(0)
    return np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN")) if len(tags) else np.zeros(0)


def _size_by_group(dim: int) -> dict:
    """{group: mean local element size near it}: each mesh node gets the mean length of the
    element edges touching it; a group's size is the mean over its nodes (straight mesh)."""
    ntags, X, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    pos = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
    pos[ntags] = np.arange(len(ntags))
    X = np.asarray(X).reshape(-1, 3)
    # every cell's edges (once per cell, as for simplices before), for any cell type: 2D flow meshes
    # recombined into quads must not count the quads' diagonals
    E = pos[np.concatenate([np.asarray(gmsh.model.mesh.getElementEdgeNodes(t, -1, True), dtype=np.int64)
                            for t in gmsh.model.mesh.getElementTypes(dim)]).reshape(-1, 2)]
    total, count = np.zeros(len(ntags)), np.zeros(len(ntags))
    length = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    for k in (E[:, 0], E[:, 1]):
        total += np.bincount(k, length, len(ntags))
        count += np.bincount(k, None, len(ntags))
    node_h = total / np.maximum(count, 1)
    out = {}
    for d, t in gmsh.model.getPhysicalGroups():
        nodes = np.asarray(gmsh.model.mesh.getNodesForPhysicalGroup(d, t)[0], dtype=np.int64)
        idx = pos[nodes][count[pos[nodes]] > 0]
        if len(idx):
            out[gmsh.model.getPhysicalName(d, t)] = float(f"{node_h[idx].mean():.4g}")
    return out


def _straighten_below(threshold: float, dim: int) -> set[int]:
    """Move the midside nodes of curved cells whose min scaled Jacobian is below `threshold` to
    their edge midpoints, repeatedly (a moved shared midside node changes the neighbours too).
    Returns the tags of the cells straightened."""
    done: set[int] = set()
    for _ in range(5):
        tags, conn = gmsh.model.mesh.getElementsByType(CURVED_TYPE[dim])
        sj = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
        bad = np.nonzero(sj < threshold)[0]
        if not len(bad):
            break
        conn = np.asarray(conn, dtype=np.int64).reshape(len(tags), -1)
        ntags, X, _ = gmsh.model.mesh.getNodes()
        coord = dict(zip(np.asarray(ntags, dtype=np.int64).tolist(), np.asarray(X).reshape(-1, 3)))
        for c in bad:
            for a, b, m in MIDSIDE[dim]:
                mid = 0.5 * (coord[conn[c, a]] + coord[conn[c, b]])
                gmsh.model.mesh.setNode(int(conn[c, m]), mid.tolist(), [])
        done |= {int(tags[c]) for c in bad}
        if math.isinf(threshold):
            break      # every cell is straight now
    return done


def _jacobian_check(path: Path, dim: int) -> dict:
    """The solver's validity test for curved cells (skfem_backend._check_jacobians): det J at the
    integration points of the order-2 analysis; a cell folds when det J changes sign in it."""
    from skfem import Basis, ElementTetP2, ElementTriP2

    from feacore.solver._skfem_mesh import read_mesh

    md = read_mesh(path, dim, 2)
    b = Basis(md.mesh, ElementTetP2() if dim == 3 else ElementTriP2(), intorder=4)
    det = b.mapping.detDF(b.X)                       # (cells, integration points), signed
    lo, hi = det.min(axis=1), det.max(axis=1)
    folded = lo * hi <= 0
    ratio = np.where(folded, lo / np.where(hi > 0, hi, 1.0), np.abs(det).min(axis=1) / np.abs(det).max(axis=1))
    return {"folded": int(folded.sum()), "min_ratio": round(float(ratio.min()), 4)}


# --------------------------------------------------------------------------- solve


def build_case(problem: Problem, mesh: dict, setup: dict, qoi_name: str) -> Case:
    """Agent's setup dict -> feacore Case. The mesh, FE order and the scored QoI are ours."""
    bc_keys = {"type", "group", "point", "components", "value"}
    if is_flow(problem):
        return _build_flow_case(problem, mesh, setup, qoi_name, bc_keys)
    mat = setup.get("material") or {}
    return Case(
        mesh=mesh["path"],
        analysis=Analysis(physics="elasticity", formulation=setup.get("formulation", ""),
                          order=mesh["order"], thickness=float(setup.get("thickness", 1.0)),
                          T_ref=float(setup.get("T_ref", 0.0))),
        materials={"mat": Material(E=float(mat.get("E", 0.0)), nu=float(mat.get("nu", 0.0)),
                                   alpha=float(mat.get("alpha", 0.0)))},
        bcs=[BC(**{k: v for k, v in bc.items() if k in bc_keys}) for bc in setup.get("bcs", [])],
        loads=[Load(**{k: v for k, v in ld.items() if k in ("type", "value", "group")}) for ld in setup.get("loads", [])],
        qois=[official_qoi(problem, qoi_name)],
    )


def _build_flow_case(problem: Problem, mesh: dict, setup: dict, qoi_name: str, bc_keys: set) -> Case:
    """Flow setup: {formulation, thickness, fluid: {rho, mu}, convection, bcs}. The solver controls
    (laminar, iteration cap, residual target) are the problem's: the agent does not tune the solver."""
    fluid = setup.get("fluid") or {}
    conv = setup.get("convection", "linearUpwind")
    if conv not in CONVECTION:
        raise CaseError(f"convection must be one of {sorted(CONVECTION)}, got {conv!r}")
    options = problem.options() if hasattr(problem, "options") else {"turbulence": "laminar"}
    return Case(
        mesh=mesh["path"],
        analysis=Analysis(physics="flow", formulation=setup.get("formulation", ""), order=CONVECTION[conv],
                          thickness=float(setup.get("thickness", 1.0)), options=dict(options)),
        materials={"fluid": Material(rho=float(fluid.get("rho", 0.0)), mu=float(fluid.get("mu", 0.0)))},
        bcs=[BC(**{k: v for k, v in bc.items() if k in bc_keys}) for bc in setup.get("bcs", [])],
        qois=[official_qoi(problem, qoi_name)],
    )


def run_solve(problem: Problem, mesh: dict, setup: dict, qoi_name: str, on_result=None,
              keep_result: bool = False) -> dict:
    """Solve with the fixed solver and evaluate the scored QoI ourselves. `on_result`, if
    given, receives the feacore Result of a successful solve (for ace.ui_bridge). With
    `keep_result` the Result is also returned under "_result" (in memory only: not JSON)."""
    try:
        case = build_case(problem, mesh, setup, qoi_name)
        result = solve(case, backend="openfoam" if is_flow(problem) else "skfem")
        value = float(result.evaluate_all()[qoi_name])
    except (CaseError, TypeError, ValueError, KeyError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001  solver failures go back to the agent as text
        return {"ok": False, "error": f"solver failed: {type(e).__name__}: {e}"}
    if on_result is not None:
        on_result(result)
    info = {"n_dofs": int(result.info.n_dofs), "wall_time": float(result.info.wall_time),
            "warnings": list(result.info.warnings)}
    out = {"ok": True, "qoi": qoi_name, "value": value, "info": info}
    if keep_result:
        out["_result"] = result
    return out
