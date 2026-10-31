"""Plain-Python FEA plumbing behind the agent's tools: geometry summary, meshing, solving.

No LLM here. gmsh is process-global, so call these from one thread per process
(run_loop runs each agent run in its own process).
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

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
    refinements: [{"group": name | "point": [x, y(, z)], "size": s, "radius": r}]
      smaller elements (size s) within distance r of a named group or a point,
      growing to h further away.
    Returns stats; `ok` is False with an `error` message on failure.
    """
    refinements = refinements or []
    if order not in (1, 2):
        return {"ok": False, "error": "order must be 1 or 2"}
    with gmsh_session(problem.id):
        problem.build_geometry()
        ensure_domain_group(problem.dim)
        bbox = _bbox_of(gmsh.model.getEntities(problem.dim))
        floor = _size_floor(problem, bbox)
        h = max(float(h), floor)

        names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
        fields = []
        for r in refinements:
            size = max(float(r.get("size", h)), floor)
            radius = float(r.get("radius", 0.0))
            if radius <= 0:
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
            elif "point" in r:
                x, y, z = (list(r["point"]) + [0.0, 0.0])[:3]
                f = gmsh.model.mesh.field.add("Ball")
                for k, v in {"XCenter": x, "YCenter": y, "ZCenter": z, "Radius": radius,
                             "Thickness": radius, "VIn": size, "VOut": h}.items():
                    gmsh.model.mesh.field.setNumber(f, k, v)
                fields.append(f)
            else:
                return {"ok": False, "error": f"refinement {r} needs 'group' or 'point'"}

        if fields:
            m = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(m, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        for opt in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(f"Mesh.{opt}", 0)

        try:
            gmsh.model.mesh.generate(problem.dim)
        except Exception as e:  # noqa: BLE001  gmsh raises plain Exception
            return {"ok": False, "error": f"gmsh failed: {e}"}

        # Quality is measured on the straight-sided mesh, before curving to order 2.
        _, elem_tags, _ = gmsh.model.mesh.getElements(problem.dim)
        tags = [int(t) for block in elem_tags for t in block]
        q = gmsh.model.mesh.getElementQualities(tags, "minSICN") if tags else []
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(path))
    return {
        "ok": True,
        "path": str(path),
        "nodes": n_nodes,
        "elements": len(tags),
        "order": order,
        "min_quality": round(float(min(q)), 4) if len(q) else None,
        "h_used": h,
    }


# --------------------------------------------------------------------------- solve


def build_case(problem: Problem, mesh: dict, setup: dict, qoi_name: str) -> Case:
    """Agent's setup dict -> feacore Case. The mesh, FE order and the scored QoI are ours."""
    bc_keys = {"type", "group", "point", "components", "value"}
    mat = setup.get("material") or {}
    return Case(
        mesh=mesh["path"],
        analysis=Analysis(physics="elasticity", formulation=setup.get("formulation", ""),
                          order=mesh["order"], thickness=float(setup.get("thickness", 1.0))),
        materials={"mat": Material(E=float(mat.get("E", 0.0)), nu=float(mat.get("nu", 0.0)))},
        bcs=[BC(**{k: v for k, v in bc.items() if k in bc_keys}) for bc in setup.get("bcs", [])],
        loads=[Load(**ld) for ld in setup.get("loads", [])],
        qois=[official_qoi(problem, qoi_name)],
    )


def run_solve(problem: Problem, mesh: dict, setup: dict, qoi_name: str) -> dict:
    """Solve with the fixed solver and evaluate the scored QoI ourselves."""
    try:
        case = build_case(problem, mesh, setup, qoi_name)
        result = solve(case, backend="skfem")
        value = float(result.evaluate_all()[qoi_name])
    except (CaseError, TypeError, ValueError, KeyError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001  solver failures go back to the agent as text
        return {"ok": False, "error": f"solver failed: {type(e).__name__}: {e}"}
    return {"ok": True, "qoi": qoi_name, "value": value}
