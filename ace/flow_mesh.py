"""CFD meshing for the agents' flow mesh tool (ace.fea.make_mesh on a flow problem).

What a flow mesh needs that an elasticity mesh does not, and how the tool provides it:

* **The CAD, not the reference's blocks.** A flow problem's ``build_geometry`` is the domain split
  into the blocks of its structured reference mesh; its ``build_cad`` is the plain CAD the agent is
  given (one face or solid). Agents mesh the CAD (``build``). Its boundary entities get the
  reference geometry's group names by geometry: every sample point of a CAD entity must lie on
  entities of the group. Closed curves (circles: one vertex) are split into four arcs first, which
  gmsh's boundary-layer mesher handles better at the seam.
* **Wall boundary layers** (``add_boundary_layer``): layers of thin quads along wall groups, from a
  first-cell height growing by a ratio, ``layers`` deep (gmsh's BoundaryLayer field with an explicit
  thickness: its default, 0.01 m, makes the layers grow until they collide in narrow gaps). Where a
  wall meets a boundary without layers (an inlet), the layers end on it, so the inlet edge gets the
  graded nodes. 2D directly; in 3D on the source face of a sweep.
* **Swept 3D meshes** (``sweep``): a straight duct or pipe is the extrusion of its inlet (or any
  planar end face) along the axis. The domain is rebuilt as that extrusion (checked against the CAD
  volume), the face is meshed in 2D (quads, boundary layers, refinements) and extruded in ``layers``
  steps, optionally graded, into hexahedra and prisms. Structured layers along the flow avoid the
  unstructured-tet errors and cost far fewer cells for the same accuracy.

The finite-volume quality of the result (non-orthogonality, skewness, wall distances; the gate the
scoring uses) is ``feacore.geometry.fv_quality``.

gmsh is process-global: call these inside one gmsh session (``ace.fea`` does).
"""

from __future__ import annotations

import math

import gmsh
import numpy as np

SPLIT_CLOSED = 4            # arcs per closed curve (circle) before meshing
MATCH_TOL = 1e-5            # entity matching tolerance, relative to the bounding-box diagonal
BL_MAX_LAYERS = 40
BL_MAX_RATIO = 3.0
SWEEP_MAX_LAYERS = 400
SWEEP_MAX_RATIO = 50.0


class FlowMeshError(ValueError):
    """A request the flow mesher cannot honour (shown to the agent as the mesh error)."""


# --------------------------------------------------------------------------- the CAD with groups


def build(problem, sweep: dict | None = None) -> dict:
    """Build the agent-facing CAD of a flow problem in the current gmsh model, with the group names
    of its reference geometry; with ``sweep`` rebuild it as the swept source face first.

    Returns {"groups": {name: (dim, [tags])}, "sweep": sweep info or None}."""
    cad_model = gmsh.model.getCurrent()
    dim = problem.dim
    gmsh.model.add("__flow_reference")
    try:
        problem.build_geometry()
        gmsh.model.occ.synchronize()
        ref, all_named = _reference_groups(dim)
        diag = _diag(gmsh.model.getBoundingBox(-1, -1))
        gmsh.model.setCurrent(cad_model)
        _build_cad(problem)
        _split_closed_curves(dim)
        groups = _transfer(ref, all_named, dim, diag, cad_model)
        info = None
        if sweep is not None:
            if dim != 3:
                raise FlowMeshError("sweep is for 3D domains; in 2D the mesh is already one layer deep")
            info = _sweep(sweep, groups, diag)
            groups = _transfer(ref, all_named, dim, diag, cad_model)
            info["source_faces"] = [t for t in groups.get(sweep["source"], (2, []))[1]]
            if not info["source_faces"]:
                raise FlowMeshError(f"sweep: the rebuilt domain has no face in group {sweep['source']!r}")
        _add_groups(groups)
        return {"groups": groups, "sweep": info}
    finally:
        names = [n for n in _model_names() if n == "__flow_reference"]
        if names:
            gmsh.model.setCurrent("__flow_reference")
            gmsh.model.remove()
        gmsh.model.setCurrent(cad_model)


def _model_names() -> list[str]:
    try:
        return list(gmsh.model.list())
    except Exception:  # noqa: BLE001  older gmsh: no model.list
        return ["__flow_reference"]


def _build_cad(problem) -> None:
    build_cad = getattr(problem, "build_cad", None)
    if build_cad is None:
        raise FlowMeshError(f"{problem.id}: no agent-facing CAD (build_cad)")
    build_cad()
    gmsh.model.occ.synchronize()
    if gmsh.model.getPhysicalGroups():                       # the CAD names nothing: groups come from the reference
        for d, t in gmsh.model.getPhysicalGroups():
            gmsh.model.removePhysicalName(gmsh.model.getPhysicalName(d, t))
        gmsh.model.removePhysicalGroups()


def _diag(bb) -> float:
    return float(math.dist(bb[:3], bb[3:]))


def _reference_groups(dim: int) -> tuple[dict, bool]:
    """({name: (dim, [tags])}, whether every boundary entity is in a group) of the reference geometry
    in the current model."""
    out: dict = {}
    for d, t in gmsh.model.getPhysicalGroups():
        out[gmsh.model.getPhysicalName(d, t)] = (d, [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)])
    if not any(d == dim for d, _ in out.values()):
        raise FlowMeshError("the reference geometry has no group of the domain's dimension")
    named = {t for d, ts in out.values() if d == dim - 1 for t in ts}
    boundary = [t for _, t in gmsh.model.getEntities(dim - 1) if _on_boundary(dim - 1, t, dim)]
    # every boundary entity of the reference is named: then every one of the CAD must be too (_transfer)
    return out, bool(boundary) and all(t in named for t in boundary)


def _split_closed_curves(dim: int, n: int = SPLIT_CLOSED) -> int:
    """Split every closed curve (a single vertex, e.g. a circle) into n pieces."""
    occ = gmsh.model.occ
    pts = []
    for _, c in gmsh.model.getEntities(1):
        b = gmsh.model.getBoundary([(1, c)], oriented=False)
        if len(b) >= 2 and b[0][1] != b[1][1]:
            continue
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        for k in range(1, n):
            x = gmsh.model.getValue(1, c, [lo[0] + k * (hi[0] - lo[0]) / n])
            pts.append(occ.addPoint(*x))
    if pts:
        occ.fragment(gmsh.model.getEntities(dim), [(0, p) for p in pts])
        occ.synchronize()
    return len(pts)


def _samples(d: int, t: int) -> list[list[float]]:
    """A few points on the CAD entity (d, t), inside its trimmed extent."""
    if d == 0:
        return [list(gmsh.model.getValue(0, t, []))]
    lo, hi = gmsh.model.getParametrizationBounds(d, t)
    if d == 1:
        return [list(gmsh.model.getValue(1, t, [lo[0] + f * (hi[0] - lo[0])])) for f in (0.21, 0.5, 0.79)]
    inside = []
    for fu in np.linspace(0.05, 0.95, 10):
        for fv in np.linspace(0.05, 0.95, 10):
            uv = [lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1])]
            if gmsh.model.isInside(2, t, uv, parametric=True):
                inside.append(uv)
    if not inside:
        c = gmsh.model.occ.getCenterOfMass(2, t)
        return [list(gmsh.model.getClosestPoint(2, t, list(c))[0])]
    pick = [inside[0], inside[len(inside) // 2], inside[-1]]
    return [list(gmsh.model.getValue(2, t, uv)) for uv in pick]


def _curve_distance(e: int, x) -> float:
    """Distance from x to the trimmed curve e: nearest of 65 samples along its own parametrisation,
    then a golden-section search on the parameter around it. (gmsh's getClosestPoint projects onto
    the untrimmed curve, and on OCC circle arcs it can return the far side of the circle.)"""
    lo, hi = (v[0] for v in gmsh.model.getParametrizationBounds(1, e))
    ts = np.linspace(lo, hi, 65)
    X = np.reshape(gmsh.model.getValue(1, e, ts.tolist()), (-1, 3))
    i = int(np.argmin(np.linalg.norm(X - np.asarray(x), axis=1)))
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]

    def dist(t):
        return float(np.linalg.norm(np.subtract(gmsh.model.getValue(1, e, [t]), x)))
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = dist(c), dist(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = dist(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = dist(d)
    return min(fc, fd, float(np.linalg.norm(X[i] - np.asarray(x))))


def _on_entity(d: int, e: int, x, tol: float) -> bool:
    """Is point x on the reference entity (d, e) of the current model, within its trimmed extent?"""
    if d == 0:
        return float(np.linalg.norm(np.subtract(gmsh.model.getValue(0, e, []), x))) <= tol
    if d == 1:
        return _curve_distance(e, x) <= tol
    xc, _ = gmsh.model.getClosestPoint(d, e, list(x))
    if float(np.linalg.norm(np.subtract(xc, x))) > tol:
        return False
    return bool(gmsh.model.isInside(2, e, list(xc)))


def _transfer(ref: dict, all_named: bool, dim: int, diag: float, cad_model: str) -> dict:
    """{name: (dim, [CAD tags])}: each CAD entity of a lower dimension joins every reference group
    whose entities hold all its sample points; the domain group takes every CAD top-dim entity."""
    tol = MATCH_TOL * diag
    samples = {(d, t): _samples(d, t) for d in range(dim) for _, t in gmsh.model.getEntities(d)}
    gmsh.model.setCurrent("__flow_reference")
    member: dict = {}
    for name, (gd, ents) in ref.items():
        if gd == dim:
            continue
        for (d, t), pts in samples.items():
            if d == gd and all(any(_on_entity(gd, e, x, tol) for e in ents) for x in pts):
                member.setdefault(name, (gd, []))[1].append(int(t))
    gmsh.model.setCurrent(cad_model)
    top = [n for n, (d, _) in ref.items() if d == dim]
    if len(top) != 1:
        raise FlowMeshError(f"cannot map {len(top)} domain groups {top} onto the CAD (one expected)")
    member[top[0]] = (dim, [int(t) for _, t in gmsh.model.getEntities(dim)])
    missing = [n for n, (d, _) in ref.items() if d != dim and n not in member]
    if missing:
        raise FlowMeshError(f"groups {missing} match no entity of the CAD")
    if all_named:
        grouped = {t for n, (d, ts) in member.items() if d == dim - 1 for t in ts}
        loose = [t for _, t in gmsh.model.getEntities(dim - 1) if t not in grouped and _on_boundary(dim - 1, t, dim)]
        if loose:
            raise FlowMeshError(f"boundary entities {loose} of the CAD match no group of the reference geometry")
    return member


def _on_boundary(d: int, t: int, dim: int) -> bool:
    """A (dim-1) entity bounding exactly one top-dimensional entity is on the domain boundary."""
    up, _ = gmsh.model.getAdjacencies(d, t)
    return len(up) == 1


def _add_groups(groups: dict) -> None:
    for name, (d, tags) in groups.items():
        gmsh.model.addPhysicalGroup(d, sorted(set(tags)), name=name)


# --------------------------------------------------------------------------- sweep


def sweep_spec(raw) -> dict | None:
    """Validate the agent's sweep request: {"source": group, "layers": n, "ratio": r (default 1)}."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or "source" not in raw or "layers" not in raw:
        raise FlowMeshError('sweep needs {"source": <group of a planar end face>, "layers": <n>} '
                            '(optional "ratio": last / first layer length)')
    try:
        layers = int(raw["layers"])
        ratio = float(raw.get("ratio", 1.0))
    except (TypeError, ValueError):
        raise FlowMeshError("sweep: layers must be an integer and ratio a number") from None
    if not 1 <= layers <= SWEEP_MAX_LAYERS:
        raise FlowMeshError(f"sweep: layers must be 1..{SWEEP_MAX_LAYERS}")
    if not 1.0 / SWEEP_MAX_RATIO <= ratio <= SWEEP_MAX_RATIO:
        raise FlowMeshError(f"sweep: ratio must be in [{1 / SWEEP_MAX_RATIO:g}, {SWEEP_MAX_RATIO:g}]")
    return {"source": str(raw["source"]), "layers": layers, "ratio": ratio}


def _plane(faces: list[int]) -> tuple[np.ndarray, float, np.ndarray, float] | None:
    """(unit normal, offset n.x, area-weighted centroid, total area) if all faces lie in one plane."""
    occ = gmsh.model.occ
    normals, area, cent = [], 0.0, np.zeros(3)
    for f in faces:
        lo, hi = gmsh.model.getParametrizationBounds(2, f)
        uvs = [[lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1])]
               for a, b in ((0.3, 0.3), (0.7, 0.6), (0.4, 0.8))]
        for uv in uvs:
            normals.append(np.asarray(gmsh.model.getNormal(f, uv)))
        a = occ.getMass(2, f)
        area += a
        cent += a * np.asarray(occ.getCenterOfMass(2, f))
    n0 = normals[0] / np.linalg.norm(normals[0])
    if any(abs(abs(float(np.dot(n0, n / np.linalg.norm(n)))) - 1.0) > 1e-9 for n in normals):
        return None
    cent /= area
    pts = np.array([gmsh.model.getValue(0, p, []) for f in faces
                    for _, p in gmsh.model.getBoundary([(2, f)], recursive=True)])
    off = float(np.dot(n0, cent))
    if len(pts) and np.max(np.abs(pts @ n0 - off)) > 1e-9 * max(1.0, float(np.ptp(pts, axis=0).max())) + 1e-12:
        return None
    return n0, off, cent, area


def _sweep(spec: dict, groups: dict, diag: float) -> dict:
    """Rebuild the domain as the source group's planar face(s) extruded to the opposite end, with
    ``layers`` element layers along the sweep (graded so the last is ``ratio`` x the first)."""
    occ = gmsh.model.occ
    src = spec["source"]
    if src not in groups or groups[src][0] != 2:
        raise FlowMeshError(f"sweep: {src!r} is not a surface group; groups: "
                            f"{sorted(n for n, (d, _) in groups.items() if d == 2)}")
    faces = groups[src][1]
    pl = _plane(faces)
    if pl is None:
        raise FlowMeshError(f"sweep: the faces of {src!r} are not planar")
    n, off, c_src, a_src = pl
    vol = sum(occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))
    # the opposite end: the boundary faces parallel to the source, at another offset, with the same total area
    by_offset: dict = {}
    for _, f in gmsh.model.getEntities(2):
        if f in faces or not _on_boundary(2, f, 3):
            continue
        p = _plane([f])
        if p is None or abs(abs(float(np.dot(p[0], n))) - 1.0) > 1e-9:
            continue
        key = round(float(np.dot(n, p[2])) / (1e-7 * diag))
        by_offset.setdefault(key, []).append(f)
    target = None
    for fs in by_offset.values():
        pt = _plane(fs)
        if pt is not None and abs(pt[3] - a_src) <= 1e-6 * a_src and abs(float(np.dot(n, pt[2])) - off) > 1e-6 * diag:
            target = pt
            break
    if target is None:
        raise FlowMeshError(f"sweep: no end face opposite {src!r} with the same area: the domain is not a "
                            f"straight sweep of {src!r}")
    vec = target[2] - c_src
    height = abs(float(np.dot(vec, n)))
    if abs(a_src * height - vol) > 1e-6 * vol:
        raise FlowMeshError(f"sweep: the domain (volume {vol:.6g}) is not the extrusion of {src!r} "
                            f"(area x length = {a_src * height:.6g})")
    layers, ratio = spec["layers"], spec["ratio"]
    sizes = np.array([ratio ** (i / max(layers - 1, 1)) for i in range(layers)])
    heights = (np.cumsum(sizes) / sizes.sum()).tolist()
    heights[-1] = 1.0
    copies = occ.copy([(2, f) for f in faces])
    occ.remove(gmsh.model.getEntities(3), recursive=True)
    occ.synchronize()
    occ.extrude(copies, *vec.tolist(), numElements=[1] * layers, heights=heights, recombine=True)
    occ.synchronize()
    vol2 = sum(occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))
    if abs(vol2 - vol) > 1e-6 * vol:
        raise FlowMeshError(f"sweep: the extrusion's volume {vol2:.6g} differs from the domain's {vol:.6g}")
    return {"source": src, "vector": [float(f"{v:.6g}") for v in vec], "length": float(f"{height:.6g}"),
            "layers": layers, "ratio": ratio,
            "first_layer": float(f"{height * sizes[0] / sizes.sum():.4g}"),
            "last_layer": float(f"{height * sizes[-1] / sizes.sum():.4g}")}


# --------------------------------------------------------------------------- boundary layers


def bl_spec(raw, groups: dict, dim: int, diag: float) -> dict | None:
    """Validate the agent's boundary-layer request:
    {"groups": [wall groups], "first": first cell height, "ratio": growth (default 1.2), "layers": n}."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw.get("groups") or "first" not in raw or "layers" not in raw:
        raise FlowMeshError('boundary_layer needs {"groups": [<wall groups>], "first": <first cell height>, '
                            '"layers": <n>} (optional "ratio": growth per layer, default 1.2)')
    names = raw["groups"] if isinstance(raw["groups"], list) else [raw["groups"]]
    bad = [g for g in names if g not in groups or groups[g][0] != dim - 1]
    if bad:
        raise FlowMeshError(f"boundary_layer: {bad} are not boundary groups; boundary groups: "
                            f"{sorted(n for n, (d, _) in groups.items() if d == dim - 1)}")
    try:
        first, layers, ratio = float(raw["first"]), int(raw["layers"]), float(raw.get("ratio", 1.2))
    except (TypeError, ValueError):
        raise FlowMeshError("boundary_layer: first and ratio must be numbers, layers an integer") from None
    floor = diag * 1e-5
    if not first > 0:
        raise FlowMeshError("boundary_layer: first must be > 0")
    if not 1 <= layers <= BL_MAX_LAYERS:
        raise FlowMeshError(f"boundary_layer: layers must be 1..{BL_MAX_LAYERS}")
    if not 1.0 <= ratio <= BL_MAX_RATIO:
        raise FlowMeshError(f"boundary_layer: ratio must be in [1, {BL_MAX_RATIO:g}]")
    first = max(first, floor)
    thickness = sum(first * ratio ** i for i in range(layers))
    return {"groups": names, "first": first, "ratio": ratio, "layers": layers,
            "thickness": float(f"{thickness:.6g}"), "last": float(f"{first * ratio ** (layers - 1):.6g}")}


def add_boundary_layer(spec: dict, groups: dict, dim: int, source_faces: list[int] | None = None) -> dict:
    """gmsh BoundaryLayer field for the validated spec. 2D: on the curves of the groups. 3D (a sweep):
    on the source faces' boundary curves that border a face of the groups. Returns what was applied."""
    if dim == 3:
        if not source_faces:
            raise FlowMeshError("boundary_layer in 3D needs a sweep: the layers are made on the swept face")
        wall_faces = {t for g in spec["groups"] for t in groups[g][1]}
        curves = []
        for f in source_faces:
            for _, c in gmsh.model.getBoundary([(2, f)], oriented=False):
                up, _ = gmsh.model.getAdjacencies(1, abs(c))
                if any(int(s) in wall_faces for s in up):
                    curves.append(abs(c))
        boundary = {abs(c) for f in source_faces for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)}
    else:
        curves = [t for g in spec["groups"] for t in groups[g][1]]
        boundary = {t for _, t in gmsh.model.getEntities(1) if _on_boundary(1, t, 2)}
    curves = sorted(set(curves))
    if not curves:
        raise FlowMeshError(f"boundary_layer: groups {spec['groups']} border no edge of the meshed face")
    ends = set()
    bl = set(curves)
    for c in curves:
        for _, p in gmsh.model.getBoundary([(1, c)], oriented=False):
            adj = [int(x) for x in gmsh.model.getAdjacencies(0, p)[0]]
            if any(a in boundary and a not in bl for a in adj):
                ends.add(int(p))
    f = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(f, "CurvesList", curves)
    if ends:
        gmsh.model.mesh.field.setNumbers(f, "PointsList", sorted(ends))
    gmsh.model.mesh.field.setNumber(f, "Size", spec["first"])
    gmsh.model.mesh.field.setNumber(f, "Ratio", spec["ratio"])
    # an explicit thickness: the layers stop after `layers` (gmsh's default, 0.01 m, would grow them on)
    gmsh.model.mesh.field.setNumber(f, "Thickness", spec["thickness"] * (1 + 1e-4))
    gmsh.model.mesh.field.setNumber(f, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(f)
    return {"curves": len(curves), "ends": len(ends)}


def count_cell_types(dim: int) -> dict:
    """{cell type name: count} of the mesh in the current model."""
    from feacore.geometry.fv_quality import GMSH_CELLS
    out: dict = {}
    for t in gmsh.model.mesh.getElementTypes(dim):
        name = GMSH_CELLS.get(int(t), (gmsh.model.mesh.getElementProperties(t)[0], 0))[0]
        out[name] = out.get(name, 0) + len(gmsh.model.mesh.getElementsByType(t)[0])
    return out
