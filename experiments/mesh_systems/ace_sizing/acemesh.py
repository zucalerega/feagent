"""ACE-style mesher (ace/fea.make_mesh on feacore.geometry.gmsh_tools) for a STEP part.

Same knobs as the ACE agents: global size h, refinements [{group|point, size, radius}], size floor
diag/100, order 1|2. Groups bottom/top/domain are chosen by geometry (planar, flat along the axis,
at the extreme position), never by tag.
"""
from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.geometry.gmsh_tools import (add_group, distance_threshold_field, ensure_domain_group,
                                         gmsh_session)

AX = {"x": 0, "y": 1, "z": 2}


def _bbox_of(dts):
    b = [gmsh.model.getBoundingBox(d, t) for d, t in dts]
    return [min(x[i] for x in b) for i in range(3)] + [max(x[i + 3] for x in b) for i in range(3)]


def build_geometry(step: str, axis: str) -> dict:
    """Import the STEP, name bottom/top/domain by geometry; return {'bottom': [tags], 'top': [tags]}."""
    gmsh.model.occ.importShapes(str(step), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    a = AX[axis]
    bb = _bbox_of(gmsh.model.getEntities(3))
    diag = math.dist(bb[:3], bb[3:])
    tol = 1e-4 * diag
    flat = []
    for _, t in gmsh.model.getEntities(2):
        if gmsh.model.getType(2, t) != "Plane":
            continue
        b = gmsh.model.getBoundingBox(2, t)
        if b[a + 3] - b[a] < tol:
            flat.append((t, b[a]))
    lo = min(p for _, p in flat); hi = max(p for _, p in flat)
    bottom = [t for t, p in flat if abs(p - lo) < tol]
    top = [t for t, p in flat if abs(p - hi) < tol]
    add_group(2, bottom, "bottom")
    add_group(2, top, "top")
    ensure_domain_group(3)
    return {"bottom": bottom, "top": top, "bbox": bb, "diag": diag}


def make_mesh(step, axis, path, h, order=2, refinements=None, floor_div=100.0, write_inp=None, verbosity=0, high_order_opt=False):
    """ace.fea.make_mesh, reproduced. Returns stats dict (ok False + error on failure)."""
    refinements = refinements or []
    with gmsh_session("ace_sizing", verbosity):
        gmsh.option.setNumber("General.NumThreads", 4)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 4)
        info = build_geometry(step, axis)
        floor = info["diag"] / floor_div
        h = max(float(h), floor)
        names = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
        fields, sizes = [], []
        for r in refinements:
            size = max(float(r.get("size", h)), floor); sizes.append(size)
            radius = float(r["radius"])
            if "group" in r:
                d, t = names[r["group"]]
                ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)]
                fields.append(distance_threshold_field(surfaces=ents, size_min=size, size_max=h,
                                                       dist_min=0.0, dist_max=radius))
            else:
                x, y, z = r["point"]
                f = gmsh.model.mesh.field.add("Ball")
                for k, v in {"XCenter": x, "YCenter": y, "ZCenter": z, "Radius": 0.0,
                             "Thickness": radius, "VIn": size, "VOut": h}.items():
                    gmsh.model.mesh.field.setNumber(f, k, v)
                fields.append(f)
        if fields:
            m = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(m, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        for opt in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(f"Mesh.{opt}", 0)
        gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
            if high_order_opt:   # deviation from ace.fea: untangle tets inverted by curving boundary edges
                gmsh.model.mesh.optimize("HighOrder")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(path))
        if write_inp:
            gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
            gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
            gmsh.write(str(write_inp))
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        _, et, _ = gmsh.model.mesh.getElements(3)
        n_el = sum(len(x) for x in et)
        groups = {}
        for name in ("bottom", "top"):
            d, t = names[name]
            tags = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)]
            cnt = 0
            for e in tags:
                _, ee, _ = gmsh.model.mesh.getElements(2, e); cnt += sum(len(x) for x in ee)
            area = sum(gmsh.model.occ.getMass(2, e) for e in tags)
            groups[name] = {"tags": tags, "elements": cnt, "area": area}
    return {"ok": True, "nodes": n_nodes, "elements": n_el, "order": order, "h_used": h, "floor": floor,
            "sizes_used": sizes, "groups": groups, "bbox": info["bbox"], "diag": info["diag"]}


def geometry_model(step, axis):
    """Harness GeometryModel of the same import (same session recipe -> same face tags)."""
    from feagent.geometry import analyze_current_model
    with gmsh_session("ace_geo"):
        build_geometry(step, axis)
        return analyze_current_model(source=str(step))
