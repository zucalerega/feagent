"""Plain-Python FEA plumbing behind the agent's tools: geometry summary, meshing, solving.

No LLM here. gmsh is process-global, so call these from one thread per process
(run_loop runs each agent run in its own process).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import gmsh
import numpy as np

from feacore.case import BC, Analysis, Case, CaseError, Load, Material
from feacore.geometry.gmsh_tools import distance_threshold_field, ensure_domain_group, gmsh_session
from feacore.problems import registry
from feacore.problems.base import Problem
from feacore.solver import solve


def get_problem(problem_id: str) -> Problem:
    return registry.get(problem_id)


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


def geometry_summary(problem: Problem) -> dict:
    """Named groups (what BCs and refinements can refer to) with their extents."""
    with gmsh_session(problem.id):
        problem.build_geometry()
        ensure_domain_group(problem.dim)
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


# --------------------------------------------------------------------------- mesh


def make_mesh(problem: Problem, path: Path, h: float, order: int = 1,
              refinements: list[dict] | None = None) -> dict:
    """Mesh the problem's geometry.

    h: global element size. order: 1 (linear) or 2 (quadratic, curved).
    refinements: [{"group": name | "edge": curve tag | "point": [x, y(, z)], "size": s, "radius": r}]
      element size s at the group, CAD curve or point, growing linearly to h at distance r;
      or {"box": [xmin, ymin, zmin, xmax, ymax, zmax], "size": s}: size s inside the box.
    Order 2 meshes are checked with the solver's Jacobian test; curved cells that fold or nearly
    fold are straightened (`repaired_cells`), so `folded_cells` should be 0.
    Returns stats; `ok` is False with an `error` message on failure.
    """
    if order not in (1, 2):
        return {"ok": False, "error": "order must be 1 or 2"}
    # gmsh's default surface mesher sometimes leaves flat tets between boundary curves of a CAD
    # part (all four nodes on CAD edges, so no optimizer can move them); MeshAdapt surfaces avoid
    # that on the gold parts. Both can fail on a curved face much smaller than h ("overlapping
    # facets"); sizing by curvature resolves it at the price of extra elements there, so it is the
    # last resort. The default is kept whenever it gives a good mesh. Each attempt is a fresh gmsh
    # session (after a failed generate gmsh fails every later one) writing its own file.
    path = Path(path)
    errors, best = [], None
    for i, mesher in enumerate(MESHERS):
        if mesher[1] and best is not None:
            break
        res = _mesh_once(problem, path.with_suffix(f".try{i}.msh"), h, order, refinements or [], mesher)
        if not res["ok"]:
            if not res.pop("retry", False):
                return res               # a bad request: every mesher would refuse it
            errors.append(res["error"])
            continue
        if best is None or res["min_quality"] > best["min_quality"]:
            best = res
        if res["min_quality"] >= REPAIR_TARGET:
            break
    for f in path.parent.glob(path.stem + ".try*.msh"):
        if best is None or str(f) != best["path"]:
            f.unlink()
    if best is None:
        return {"ok": False, "error": errors[0]}
    Path(best["path"]).replace(path)
    return best | {"path": str(path)}


def _mesh_once(problem: Problem, path: Path, h: float, order: int, refinements: list[dict],
               mesher: tuple[int, int]) -> dict:
    """make_mesh with one mesher (gmsh surface algorithm, curvature sizing). A gmsh failure comes
    back with "retry": True."""
    with gmsh_session(problem.id):
        problem.build_geometry()
        ensure_domain_group(problem.dim)
        bbox = _bbox_of(gmsh.model.getEntities(problem.dim))
        floor = _size_floor(problem, bbox)
        h = max(float(h), floor)

        names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
        fields, sizes_used = [], []
        for r in refinements:
            size = max(float(r.get("size", h)), floor)
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

        alg, curv = mesher
        gmsh.option.setNumber("Mesh.Algorithm", alg)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
        try:
            gmsh.model.mesh.generate(problem.dim)
        except Exception as e:  # noqa: BLE001  gmsh raises plain Exception
            return {"ok": False, "error": f"gmsh failed: {e}", "retry": True}
        q = _qualities(problem.dim)
        if not len(q):
            return {"ok": False, "error": "gmsh failed: no elements generated", "retry": True}
        # Snapping boundary vertices onto curved CAD can leave a few nearly flat tets; gmsh's tet
        # optimizer + vertex relocation repairs those it can (as the gold generator does). Only when
        # needed, so meshes that are already fine stay exactly as gmsh made them.
        if problem.dim == 3 and q.min() < REPAIR_TARGET:
            for _ in range(3):
                gmsh.model.mesh.optimize("", niter=3)
                gmsh.model.mesh.optimize("Relocate3D", niter=3)
                q = _qualities(problem.dim)
                if q.min() >= REPAIR_TARGET:
                    break
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
    return {
        "ok": True,
        "path": str(path),
        "nodes": n_nodes,
        "elements": len(q),
        "order": order,
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


REPAIR_TARGET = 0.05   # min straight-cell quality (minSICN) the linear repair aims for
# (gmsh Mesh.Algorithm, Mesh.MeshSizeFromCurvature) tried in turn: Frontal-Delaunay surfaces (gmsh's
# default), MeshAdapt, MeshAdapt with at least 8 elements per 2 pi of curvature (as feacore.goldgen)
MESHERS = ((6, 0), (1, 0), (1, 8))
MESHER_NAMES = {(6, 0): "frontal-delaunay", (1, 0): "meshadapt", (1, 8): "meshadapt+curvature"}
SJ_MIN = 0.05          # curved cells below this min scaled Jacobian are straightened (as feacore.goldgen)
LINEAR_TYPE = {2: 2, 3: 4}           # gmsh element types: tri3, tet4
CURVED_TYPE = {2: 9, 3: 11}          # tri6, tet10
# (corner a, corner b, midside node) in gmsh's node ordering
MIDSIDE = {2: [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
           3: [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]}


def _qualities(dim: int) -> np.ndarray:
    tags, _ = gmsh.model.mesh.getElementsByType(LINEAR_TYPE[dim])
    return np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN")) if len(tags) else np.zeros(0)


def _size_by_group(dim: int) -> dict:
    """{group: mean local element size near it}: each mesh node gets the mean length of the
    element edges touching it; a group's size is the mean over its nodes (straight mesh)."""
    ntags, X, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    pos = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
    pos[ntags] = np.arange(len(ntags))
    X = np.asarray(X).reshape(-1, 3)
    _, conn = gmsh.model.mesh.getElementsByType(LINEAR_TYPE[dim])
    T = pos[np.asarray(conn, dtype=np.int64).reshape(-1, dim + 1)]
    total, count = np.zeros(len(ntags)), np.zeros(len(ntags))
    for i in range(dim + 1):
        for j in range(i + 1, dim + 1):
            length = np.linalg.norm(X[T[:, i]] - X[T[:, j]], axis=1)
            for k in (T[:, i], T[:, j]):
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


def run_solve(problem: Problem, mesh: dict, setup: dict, qoi_name: str, on_result=None,
              keep_result: bool = False) -> dict:
    """Solve with the fixed solver and evaluate the scored QoI ourselves. `on_result`, if
    given, receives the feacore Result of a successful solve (for ace.ui_bridge). With
    `keep_result` the Result is also returned under "_result" (in memory only: not JSON)."""
    try:
        case = build_case(problem, mesh, setup, qoi_name)
        result = solve(case, backend="skfem")
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
