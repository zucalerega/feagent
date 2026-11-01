"""Element budget for automesh: estimate the tet count from a 1D mesh and relax the size controls.

A part with hundreds of small freeform faces (curvature sizing) or many thin walls (thin_wall_layers)
can ask for millions of elements. ``fit_size`` meshes the curves only (seconds), estimates the surface
triangles from them, adds the volume and thin-wall terms, and steps ``size`` down a ladder until the
estimated DOF count is under ``size.max_dofs``."""
from __future__ import annotations

from dataclasses import replace

import gmsh
import numpy as np

from .gmsh_util import pinned_options
from .spec import MeshSpec, SizeSpec
from .types import GeometryModel

TRI_AREA_FACTOR = 0.433      # area of an equilateral triangle of edge h is 0.433 h^2
TETS_PER_TRI = 3.2           # tets per boundary triangle on freeform parts (measured 2.5-3.3)
TET_VOLUME_FACTOR = 8.49     # a regular tet of edge h has volume h^3 / 8.49
NODES_PER_TET = {1: 0.25, 2: 1.52}  # measured: 3.84M nodes for 2.53M tet10
DOF_PER_NODE = {"fea": 3, "cfd": 4}
SAFETY = 1.15                # headroom for what the estimate ignores (user refine fields, grading)
# (curvature cap, size.min >= global / divisor): the sizing ladder, finest first
LADDER = ((12, 20), (8, 10), (6, 8), (4, 5), (3, 4), (2, 3))
THIN_FACTOR, THIN_REACH = 0.8, 2.0  # mirror mesher.THIN_WALL_SIZE_FACTOR / THIN_WALL_DISTANCE


def dofs_per_tet(spec: MeshSpec) -> float:
    return NODES_PER_TET[min(spec.order, 2)] * DOF_PER_NODE.get(spec.analysis, 3)


def _curve_lengths() -> dict[int, float]:
    """Mean 1D element length per curve, from the current mesh."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    pts, tags = xyz.reshape(-1, 3)[order], tags[order]
    out = {}
    for _, c in gmsh.model.getEntities(1):
        _, _, en = gmsh.model.mesh.getElements(1, c)
        if not len(en) or not len(en[0]):
            continue
        n = np.asarray(en[0], np.int64).reshape(-1, 2)
        a, b = (pts[np.searchsorted(tags, n[:, i])] for i in (0, 1))
        out[c] = float(np.linalg.norm(a - b, axis=1).mean())
    return out


def surface_triangles(model: GeometryModel, h: float) -> float:
    """Boundary triangles implied by the current 1D mesh: each face's area over the triangle area
    of its mean boundary edge length (capped at the global size)."""
    lens = _curve_lengths()
    total = 0.0
    for f in model.faces:
        ls = [lens[c] for _, c in gmsh.model.getBoundary([(2, f.tag)], oriented=False) if c in lens]
        hf = min(float(np.mean(ls)) if ls else h, h)
        total += f.area / (TRI_AREA_FACTOR * hf ** 2)
    return total


def thin_wall_tets(model: GeometryModel, size: SizeSpec, h: float) -> float:
    """Tets of the thin-wall refinement: wall volume (half per face, a wall has two) at the
    refined size."""
    if size.thin_wall_layers <= 0:
        return 0.0
    total = 0.0
    for z in model.hazards:
        if z.kind != "thin_wall" or not z.value:
            continue
        s = min(THIN_FACTOR * z.value / size.thin_wall_layers, h)
        area = sum(model.face(int(t)).area for d, t in z.entities if d == 2)
        total += 0.5 * area * z.value * TET_VOLUME_FACTOR / s ** 3
    return total


def _mesh_curves(size: SizeSpec) -> None:
    opts = {"Mesh.MeshSizeMax": size.global_, "Mesh.MeshSizeMin": size.min,
            "Mesh.MeshSizeFromCurvature": size.curvature, "Mesh.MeshSizeFromPoints": 0}
    with pinned_options(opts):
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(1)


def estimate_tets(spec: MeshSpec, model: GeometryModel, size: SizeSpec) -> float:
    """Estimated tets for `size` (user refine fields are not included)."""
    _mesh_curves(size)
    try:
        tris = surface_triangles(model, size.global_)
    finally:
        gmsh.model.mesh.clear()
    vol = sum(v.volume for v in model.volumes) * TET_VOLUME_FACTOR / size.global_ ** 3
    return vol + TETS_PER_TRI * tris + thin_wall_tets(model, size, size.global_)


def _candidates(size: SizeSpec):
    """The requested size, then the ladder rungs (thin-wall layers capped at 1 on all of them)."""
    yield size
    layers = min(size.thin_wall_layers, 1)
    for curv, div in LADDER:
        lo = max(size.min, size.global_ / div)
        yield replace(size, curvature=min(size.curvature, curv), min=lo, thin_wall_layers=layers)


def _change_note(old: SizeSpec, new: SizeSpec) -> str:
    return (f"size.min {old.min:.3g}->{new.min:.3g}, curvature {old.curvature}->{new.curvature}, "
            f"thin_wall_layers {old.thin_wall_layers}->{new.thin_wall_layers}")


def fit_size(spec: MeshSpec, model: GeometryModel) -> tuple[SizeSpec, list[str]]:
    """The finest rung of the ladder whose estimated DOFs fit ``spec.size.max_dofs`` (the spec's
    own size if it fits or the budget is off), with warnings describing any relaxation."""
    size = spec.size
    if size.max_dofs <= 0:
        return size, []
    limit = size.max_dofs / dofs_per_tet(spec) / SAFETY
    tets = 0.0
    for cand in _candidates(size):
        tets = estimate_tets(spec, model, cand)
        if tets <= limit:
            break
    dofs = int(tets * dofs_per_tet(spec))
    if cand == size:
        return size, []
    over = "" if tets <= limit else f" (still ~{dofs} DOFs: the part needs a larger size.global)"
    return cand, [f"element budget size.max_dofs={size.max_dofs}: the requested sizing would need "
                  f"more elements; relaxed {_change_note(size, cand)}, ~{dofs} DOFs{over}"]
