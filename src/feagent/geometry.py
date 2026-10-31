"""Geometry analysis of the current gmsh/OCC model and the text digest an agent reads before meshing.

Everything reported comes from the kernel (OCC via gmsh): areas, centres, bounding boxes, outward
normals, cylinder axes/radii, concavity, adjacency. A temporary 2D tessellation supplies exact surface
sample points (interior mesh nodes carry their uv), the outward orientation (ray parity) and the
triangles used for ray-cast wall thickness and gap estimates. The mesh is cleared before returning.
"""
from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import gmsh
import numpy as np

from .errors import FeagentError
from .gmsh_session import gmsh_messages, gmsh_session, load_cad
from .gmsh_util import element_blocks, entity_tags, node_table, pinned_options
from .graph import connected_groups
from .types import (BBox, EdgeInfo, FaceInfo, Feature, GeometryModel, Hazard, Tessellation,
                    VolumeInfo)
from .vec import bbox_diag, unit, unit_rows

SURFACE_KINDS = {
    "Plane": "plane", "Cylinder": "cylinder", "Cone": "cone", "Sphere": "sphere", "Torus": "torus",
    "BSpline surface": "bspline", "Bezier surface": "bspline", "Surface of Revolution": "revolution",
    "Surface of Extrusion": "extrusion",
}
CURVE_KINDS = {
    "Line": "line", "Circle": "circle", "Ellipse": "ellipse", "BSpline": "bspline",
    "BSpline curve": "bspline", "Bezier": "bspline", "Bezier curve": "bspline",
    "Hyperbola": "hyperbola", "Parabola": "parabola",
}
_REVOLVED = ("cylinder", "cone", "torus")   # u parameter = angle about the axis

_MAX_SAMPLES = 24          # exact surface samples per face used for normals / curvature
_EDGE_SAMPLES = 17         # points per boundary curve used for extents / angle span
_MAX_RAYS = 50             # thickness / gap rays per face
_PARITY_RAYS = 3           # orientation rays per face (majority vote)
_SPAN_FULL = 300.0         # angle_span (deg) at or above which a cylinder counts as a full revolution
_DIGEST_BUDGET = 7000      # summary digest target size (chars); FACES rows stop here
_SUMMARY_GROUPS = 10       # feature patterns / normal groups listed in a summary digest
_EMPTY3 = np.zeros((0, 3))
# Session-independent temporary tessellation: whatever a mesher set up, this run is linear triangles
# sized by MeshSizeMax + curvature. (Size fields cannot be switched off through the gmsh API; they
# can only refine it, because sizes are clamped to MeshSizeMax.)
_TESS_OPTIONS = {"Mesh.ElementOrder": 1, "Mesh.RecombineAll": 0, "Mesh.SubdivisionAlgorithm": 0,
                 "Mesh.MeshSizeFactor": 1, "Mesh.MeshSizeMin": 0, "Mesh.MeshSizeFromPoints": 0,
                 "Mesh.MeshSizeExtendFromBoundary": 1, "Mesh.Algorithm": 6,
                 "Mesh.MeshSizeFromCurvature": 12}
_TILTS = 0.2 * np.array([[0.31, 0.53, 0.79], [-0.72, 0.21, 0.66], [0.45, -0.83, 0.33]])


@dataclass
class _Face:
    """Working data for one face while the model is analyzed."""
    tag: int
    kind: str
    volumes: list[int]
    edges: list[int]
    xyz: np.ndarray = field(default_factory=lambda: _EMPTY3)   # exact samples (k,3)
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    raw: np.ndarray = field(default_factory=lambda: _EMPTY3)   # getNormal at samples
    nrm: np.ndarray = field(default_factory=lambda: _EMPTY3)   # outward normals at samples
    pts: np.ndarray = field(default_factory=lambda: _EMPTY3)   # every known point on the face
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    sign: float = 1.0          # raw * sign = outward normal
    uspan: float | None = None


@dataclass
class _Ctx:
    scale: float               # diag, or 1 for an empty model (tolerances)
    diag: float
    ref_size: float
    faces: dict[int, _Face]
    edge_pts: dict[int, np.ndarray]
    mesh_error: str | None = None
    unresolved: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------
# vector helpers

def _canonical_axis(a) -> np.ndarray:
    """Unit axis with its first non-negligible component positive; ~0 components snap to 0."""
    a = unit(a)
    lead = next((c for c in a if abs(c) > 1e-9), 1.0)
    a = -a if lead < 0 else a
    return np.where(np.abs(a) < 1e-12, 0.0, a)


def _num(x, nd: int = 12) -> float:
    """Round away last-bit noise (and -0.0) so reported values read cleanly."""
    return float(round(float(x), nd)) + 0.0


def _tup(v, nd: int = 12) -> tuple[float, float, float]:
    return tuple(_num(x, nd) for x in v)  # type: ignore[return-value]


def _bbox_tuple(b) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (_tup(b[:3]), _tup(b[3:6]))


def _axis_point(point, axis) -> np.ndarray:
    """Point on the axis line through `point` closest to the origin."""
    point = np.asarray(point, float)
    return point - np.dot(point, axis) * axis


def _dist_to_line(pts: np.ndarray, origin, axis) -> np.ndarray:
    d = pts - origin
    return np.linalg.norm(d - np.outer(d @ axis, axis), axis=1)


def _perp_basis(axis) -> tuple[np.ndarray, np.ndarray]:
    a = unit(axis)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(a, ref))
    return u, np.cross(a, u)


def _spread(n: int, k: int) -> np.ndarray:
    """Up to k evenly spread indices into range(n)."""
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int)) if n else np.zeros(0, int)


def _arc(pts: np.ndarray, origin, axis, rmin: float,
         exact_span: float | None = None) -> tuple[float, float] | None:
    """(start, span) in degrees of the points about an axis: span = 360 - largest empty sector,
    start = first angle after that sector. exact_span (from the face's u-range) overrides span."""
    u, v = _perp_basis(axis)
    d = pts - origin
    x, y = d @ u, d @ v
    keep = np.hypot(x, y) > rmin
    if keep.sum() < 2:
        return None
    ang = np.sort(np.degrees(np.arctan2(y[keep], x[keep])))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
    i = int(np.argmax(gaps))
    span = 360.0 - float(gaps[i]) if exact_span is None else exact_span
    return float(ang[(i + 1) % len(ang)] % 360.0), float(min(360.0, max(0.0, span)))


def _angle_span(pts, origin, axis, rmin: float, exact_span: float | None) -> float | None:
    if exact_span is not None:
        return _num(min(360.0, exact_span), 9)
    a = _arc(pts, origin, axis, rmin)
    return None if a is None else _num(a[1], 9)


def _union_span(arcs: list[tuple[float, float]]) -> float:
    """Total angle (deg) covered by arcs (start, span) on the circle."""
    if any(sp >= 360.0 - 1e-9 for _, sp in arcs):
        return 360.0
    segs = []
    for st, sp in arcs:
        st %= 360.0
        segs += [(st, 360.0), (0.0, st + sp - 360.0)] if st + sp > 360.0 else [(st, st + sp)]
    total, cur = 0.0, None
    for a, b in sorted(segs):
        if cur is not None and a <= cur[1]:
            cur[1] = max(cur[1], b)
            continue
        total += 0.0 if cur is None else cur[1] - cur[0]
        cur = [a, b]
    total += 0.0 if cur is None else cur[1] - cur[0]
    return float(min(360.0, total))


def _axis_point_from_normals(pts: np.ndarray, nrm: np.ndarray, axis) -> np.ndarray | None:
    """Least-squares meeting point (in the plane normal to axis) of the normal lines; for surfaces of
    revolution every normal line meets the axis."""
    a = unit(axis)
    P = pts - np.outer(pts @ a, a)
    M = nrm - np.outer(nrm @ a, a)
    ok = np.linalg.norm(M, axis=1) > 1e-9
    if ok.sum() < 2:
        return None
    P, M = P[ok], unit_rows(M[ok])
    Q = np.eye(3)[None] - M[:, :, None] * M[:, None, :]
    A = Q.sum(axis=0) + np.outer(a, a)  # pin the component along the axis to 0
    try:
        return np.linalg.solve(A, np.einsum("kij,kj->i", Q, P))
    except np.linalg.LinAlgError:
        return None


def _ray_triangle(o: np.ndarray, d: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Moller-Trumbore distance along unit d from o to each triangle (k,3,3); inf where missed.
    o and d are one ray (3,) or one ray per triangle (k,3)."""
    a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    good = np.abs(det) > 1e-300
    inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
    tv = o - a
    qv = np.cross(tv, e1)
    u = np.einsum("ij,ij->i", tv, pv) * inv
    v = np.einsum("ij,ij->i", qv, np.broadcast_to(d, qv.shape)) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = good & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return np.where(hit, t, np.inf)


def _distinct_count(t: np.ndarray, tol: float) -> int:
    """Number of distinct values (a ray through a shared edge hits both triangles once each)."""
    if not len(t):
        return 0
    t = np.sort(t)
    return int(1 + np.count_nonzero(np.diff(t) > tol))


# ----------------------------------------------------------------------------------------------
# gmsh helpers

def _adj(dim: int, tag: int) -> tuple[list[int], list[int]]:
    up, down = gmsh.model.getAdjacencies(dim, tag)
    return sorted(int(x) for x in up), sorted(int(x) for x in down)


def _u_span(f: int) -> float | None:
    """Exact angular extent (deg) of an OCC cylinder/cone/torus face: its u parameter is the angle."""
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, f)
    except Exception:
        return None
    span = math.degrees(float(hi[0] - lo[0]))
    return min(360.0, span) if span > 0 else None


def _curve_points(e: int) -> np.ndarray:
    try:
        lo, hi = gmsh.model.getParametrizationBounds(1, e)
        ts = np.linspace(lo[0], hi[0], _EDGE_SAMPLES)
        return np.asarray(gmsh.model.getValue(1, e, ts.tolist()), float).reshape(-1, 3)
    except Exception:
        return _EMPTY3


def _project(f: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, uv = gmsh.model.getClosestPoint(2, f, np.asarray(pts, float).ravel().tolist())
    return np.asarray(c, float).reshape(-1, 3), np.asarray(uv, float).reshape(-1, 2)


def _normals_at(f: int, uv: np.ndarray) -> np.ndarray:
    raw = gmsh.model.getNormal(f, np.asarray(uv, float).ravel().tolist())
    return unit_rows(np.asarray(raw, float).reshape(-1, 3))


def _uv_grid_samples(f: int, n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Exact samples without a mesh: a uv grid filtered to the trimmed face."""
    m = gmsh.model
    lo, hi = m.getParametrizationBounds(2, f)
    us, vs = (np.linspace(lo[i], hi[i], n + 2)[1:-1] for i in (0, 1))
    uv = np.array([(u, v) for u in us for v in vs])
    keep = np.array([bool(m.isInside(2, f, [float(u), float(v)], parametric=True)) for u, v in uv])
    uv = uv[keep][_spread(int(keep.sum()), _MAX_SAMPLES)]
    if len(uv) == 0:  # very thin face: project its centre of mass instead
        return _project(f, np.asarray(m.occ.getCenterOfMass(2, f), float)[None, :])
    return np.asarray(m.getValue(2, f, uv.ravel().tolist()), float).reshape(-1, 3), uv


def _curvatures(fd: _Face) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kmax, kmin, dmax, dmin = gmsh.model.getPrincipalCurvatures(fd.tag, fd.uv.ravel().tolist())
    return (np.asarray(kmax, float), np.asarray(kmin, float),
            np.asarray(dmax, float).reshape(-1, 3), np.asarray(dmin, float).reshape(-1, 3))


def _curvature_radius(fd: _Face) -> float | None:
    """1 / median of the largest |principal curvature| over the samples."""
    kmax, kmin, _, _ = _curvatures(fd)
    big = np.maximum(np.abs(kmax), np.abs(kmin))
    return float(1.0 / np.median(big[big > 0])) if (big > 0).any() else None


def _face_triangles(f: int) -> np.ndarray:
    """Corner-node triangles (node tags) of the face mesh, whatever element type it holds
    (quads are split, high-order elements reduced to their corners)."""
    out = []
    for t, _, nodes in element_blocks(2, f):
        npri = gmsh.model.mesh.getElementProperties(t)[5]
        c = nodes[:, :npri].astype(np.int64)
        out += [c] if npri == 3 else ([c[:, [0, 1, 2]], c[:, [0, 2, 3]]] if npri == 4 else [])
    return np.vstack(out) if out else np.zeros((0, 3), dtype=np.int64)


@contextmanager
def _temporary_mesh(opts: dict[str, float]):
    """Pin mesh options for the temporary tessellation; always clear the mesh, then restore them."""
    with pinned_options(opts):
        try:
            yield
        finally:
            gmsh.model.mesh.clear()


def _require_no_mesh() -> None:
    if len(gmsh.model.mesh.getNodes()[0]):
        raise FeagentError(
            "a mesh already exists in the current gmsh session; analyze_current_model needs a "
            "temporary tessellation and would destroy it",
            hint="analyze the geometry before meshing, call gmsh.model.mesh.clear() first, or pass "
                 "tessellate=False (no thickness/gap measures)")


# ----------------------------------------------------------------------------------------------
# public API

def analyze(path: str | Path, ref_size: float | None = None, heal: bool = False,
            tessellate: bool = True) -> GeometryModel:
    """Load a CAD file in a fresh gmsh session and analyze it."""
    with gmsh_session():
        load_cad(path, heal=heal)
        return analyze_current_model(ref_size=ref_size, source=str(path), tessellate=tessellate)


def analyze_current_model(ref_size: float | None = None, source: str = "",
                          tessellate: bool = True) -> GeometryModel:
    """Analyze the model of the CURRENT gmsh session; call it BEFORE meshing.

    ref_size (default diag/40, must be > 0 else ValueError) sets the hazard thresholds.
    tessellate=True builds a temporary linear triangulation with pinned mesh options (restored after)
    and clears it again; it raises FeagentError if the session already holds a mesh, rather than
    destroying it. tessellate=False never touches the mesh: no thickness/gap, no Tessellation, uv-grid
    samples and isInside probes for orientation (see _orient_faces). A failed or partial tessellation
    is reported as an `analysis_incomplete` hazard (also shown in the digest header)."""
    if tessellate:
        _require_no_mesh()
    bbox = _model_bbox()
    diag = bbox_diag(bbox)
    ctx = _Ctx(scale=diag if diag > 0 else 1.0, diag=diag, ref_size=_ref_size(ref_size, diag),
               faces=_face_records(), edge_pts={})
    volumes, edges = _volume_infos(), _edge_infos(ctx.scale)
    ctx.edge_pts = {e.tag: _curve_points(e.tag) for e in edges}
    points = _tessellate(ctx) if tessellate and ctx.faces else _EMPTY3
    for fd in ctx.faces.values():
        _prepare_samples(ctx, fd)
    tess = _build_tessellation(ctx.faces, points) if len(points) else None
    caster = _RayCaster(tess, ctx.scale) if tess is not None and volumes else None
    _orient_faces(ctx, volumes, tess, caster)
    faces = [_face_info(ctx, fd) for fd in ctx.faces.values()]
    _set_neighbors(faces, edges)
    features = _features(faces, ctx)
    if caster is not None:
        _ray_measures(faces, tess, caster, _feature_index(features))
    hazards = _hazards(faces, edges, volumes, features, ctx) + _hz_incomplete(ctx, tessellate)
    return GeometryModel(source=source, units="mm", bbox=bbox, diag=diag, ref_size=ctx.ref_size,
                         volumes=volumes, faces=faces, edges=edges, features=features,
                         hazards=hazards, tessellation=tess)


def _ref_size(ref_size: float | None, diag: float) -> float:
    if ref_size is None:
        return diag / 40.0 if diag > 0 else 1.0
    if not ref_size > 0:
        raise ValueError(f"ref_size must be > 0 (got {ref_size}); omit it for diag/40")
    return float(ref_size)


# ----------------------------------------------------------------------------------------------
# topology and exact measures

def _bbox(dim: int, tag: int) -> BBox:
    return _bbox_tuple(gmsh.model.getBoundingBox(dim, tag))


def _model_bbox():
    return _bbox(-1, -1) if gmsh.model.getEntities() else _bbox_tuple((0.0,) * 6)


def _volume_infos() -> list[VolumeInfo]:
    occ = gmsh.model.occ
    return [VolumeInfo(tag=v, volume=float(occ.getMass(3, v)), center=_tup(occ.getCenterOfMass(3, v)),
                       bbox=_bbox(3, v), faces=_adj(3, v)[1])
            for v in entity_tags(3)]


def _edge_infos(scale: float) -> list[EdgeInfo]:
    out = []
    for e in entity_tags(1):
        length = float(gmsh.model.occ.getMass(1, e))
        kind = ("degenerate" if length <= 1e-9 * scale
                else CURVE_KINDS.get(gmsh.model.getType(1, e), "other"))
        out.append(EdgeInfo(tag=e, kind=kind, length=length, faces=_adj(1, e)[0], bbox=_bbox(1, e)))
    return out


def _face_records() -> dict[int, _Face]:
    out = {}
    for f in entity_tags(2):
        up, down = _adj(2, f)
        kind = SURFACE_KINDS.get(gmsh.model.getType(2, f), "other")
        out[f] = _Face(tag=f, kind=kind, volumes=up, edges=down,
                       uspan=_u_span(f) if kind in _REVOLVED else None)
    return out


def _set_neighbors(faces: list[FaceInfo], edges: list[EdgeInfo]) -> None:
    edge_faces = {e.tag: e.faces for e in edges}
    for fi in faces:
        nb = {t for e in fi.edges for t in edge_faces.get(e, ())}
        fi.neighbors = sorted(nb - {fi.tag})


# ----------------------------------------------------------------------------------------------
# temporary tessellation

def _tessellate(ctx: _Ctx) -> np.ndarray:
    """2D-mesh the model, record per-face triangles and exact interior samples, clear the mesh.
    Returns the node coordinates the per-face triangle rows index into."""
    size = min(ctx.ref_size, ctx.diag / 25.0) if ctx.diag > 0 else ctx.ref_size
    with _temporary_mesh({**_TESS_OPTIONS, "Mesh.MeshSizeMax": size}):
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:  # keep whatever faces did mesh; reported as analysis_incomplete
            ctx.mesh_error = (str(e) + " " + "; ".join(gmsh_messages("Error")[-3:])).strip()
        _, points, row = node_table()
        for fd in ctx.faces.values() if len(points) else ():
            _read_face_mesh(fd, points, row)
    return points


def _read_face_mesh(fd: _Face, points: np.ndarray, row: np.ndarray) -> None:
    tris = _face_triangles(fd.tag)
    if len(tris):
        fd.tris = row[tris]
        fd.pts = points[np.unique(fd.tris)]
    itags, ixyz, iuv = gmsh.model.mesh.getNodes(2, fd.tag, includeBoundary=False,
                                                returnParametricCoord=True)
    if len(itags):
        pick = _spread(len(itags), _MAX_SAMPLES)
        fd.xyz = np.asarray(ixyz, float).reshape(-1, 3)[pick]
        fd.uv = np.asarray(iuv, float).reshape(-1, 2)[pick]
    elif len(fd.tris):  # no interior node (tiny face): project triangle centroids
        cen = points[fd.tris].mean(axis=1)
        fd.xyz, fd.uv = _project(fd.tag, cen[_spread(len(cen), _MAX_SAMPLES)])


def _build_tessellation(fdata: dict[int, _Face], points: np.ndarray) -> Tessellation | None:
    """Triangles of all faces wound along getNormal (flipped to outward later), compacted to the
    nodes they use."""
    parts = [(fd.tag, *_orient_face_tris(fd, points)) for fd in fdata.values() if len(fd.tris)]
    if not parts:
        return None
    tris = np.vstack([p[1] for p in parts])
    used, inv = np.unique(tris, return_inverse=True)
    return Tessellation(points=points[used].copy(), triangles=inv.reshape(-1, 3).astype(np.int64),
                        face_tag=np.concatenate([np.full(len(p[1]), p[0], dtype=np.int64) for p in parts]),
                        normals=np.vstack([p[2] for p in parts]))


def _orient_face_tris(fd: _Face, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wind triangles along the exact getNormal of the face; return (tris, unit normals)."""
    tris = fd.tris.copy()
    p = points[tris]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if fd.kind == "plane" and len(fd.raw):
        ref = np.broadcast_to(fd.raw[0], cr.shape)
    else:  # exact normal at each projected centroid
        ref = _normals_at(fd.tag, _project(fd.tag, p.mean(axis=1))[1])
    flip = np.einsum("ij,ij->i", cr, ref) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    cr[flip] *= -1
    ln = np.linalg.norm(cr, axis=1)
    nrm = np.where(ln[:, None] > 0, cr / np.maximum(ln, 1e-300)[:, None], unit_rows(ref))
    return tris, nrm


def _rows_by_face(tess: Tessellation | None) -> dict[int, np.ndarray]:
    if tess is None:
        return {}
    order = np.argsort(tess.face_tag, kind="stable")
    tags, starts = np.unique(tess.face_tag[order], return_index=True)
    return dict(zip(tags.tolist(), np.split(order, starts[1:])))


# ----------------------------------------------------------------------------------------------
# samples and outward orientation

def _prepare_samples(ctx: _Ctx, fd: _Face) -> None:
    """Exact samples (uv grid when the face has no mesh), known points and getNormal at samples."""
    if len(fd.xyz) == 0:
        try:
            fd.xyz, fd.uv = _uv_grid_samples(fd.tag)
        except Exception:
            fd.xyz, fd.uv = _EMPTY3, np.zeros((0, 2))
    own = [ctx.edge_pts[e] for e in fd.edges if len(ctx.edge_pts.get(e, _EMPTY3))]
    fd.pts = np.vstack([fd.pts, fd.xyz, *own])
    fd.raw = _normals_at(fd.tag, fd.uv) if len(fd.xyz) else _EMPTY3


def _orient_faces(ctx: _Ctx, volumes: list[VolumeInfo], tess: Tessellation | None,
                  caster: _RayCaster | None) -> None:
    """Outward sign per face from ray parity against its first volume's boundary triangles
    (O(F log T)). Parity is only trusted when that volume's boundary is fully tessellated (a hole in
    it breaks crossing counts); otherwise, or when the vote ties, isInside probe pairs decide: up to
    3 exact samples, each probed at 3 offsets along the normal; the first decisive pair wins and
    none keeps the getNormal sign."""
    rows = _rows_by_face(tess)
    closed = {v.tag: np.asarray(v.faces, dtype=np.int64) for v in volumes
              if caster is not None and all(f in rows for f in v.faces)}
    for fd in ctx.faces.values():
        if fd.volumes:
            fd.sign = _face_sign(ctx, fd, rows.get(fd.tag), closed.get(fd.volumes[0]), tess, caster)
        fd.nrm = fd.sign * fd.raw
        if fd.sign < 0 and fd.tag in rows:
            r = rows[fd.tag]
            tess.triangles[r] = tess.triangles[r][:, [0, 2, 1]]
            tess.normals[r] *= -1


def _face_sign(ctx: _Ctx, fd: _Face, rows, vol_faces, tess, caster) -> float:
    """Parity when the volume is closed (also for faces without exact samples); else isInside on
    up to 3 samples (a projected centroid of a sliver face may fall just off the face)."""
    votes = 0
    if rows is not None and vol_faces is not None:
        votes = _parity_votes(rows, vol_faces, tess, caster)
    for i in _spread(len(fd.xyz), 3) if votes == 0 else ():  # first decisive sample wins
        votes = _inside_vote(fd.volumes[0], fd.xyz[i], fd.raw[i], 1e-4 * ctx.scale)
        if votes:
            break
    return -1.0 if votes < 0 else 1.0


def _parity_votes(rows: np.ndarray, vol_faces, tess: Tessellation, caster: _RayCaster) -> int:
    """Per ray from a triangle centroid (tilted off its getNormal-side normal): +1 if it crosses the
    volume boundary an even number of times ahead and an odd number behind (normal points out), -1
    for the reverse, 0 (abstain) if both parities agree, i.e. the faceted boundary is not a
    consistent closed shell there (walls thinner than the chordal error)."""
    votes = 0
    for k, cid in enumerate(rows[_spread(len(rows), _PARITY_RAYS)]):
        n = tess.normals[cid]
        d = unit(n + _TILTS[k] - np.dot(_TILTS[k], n) * n)
        ahead, behind = (_crossings(caster, cid, s * d, vol_faces, tess) % 2 for s in (1.0, -1.0))
        votes += 0 if ahead == behind else (-1 if ahead else 1)
    return votes


def _crossings(caster: _RayCaster, cid, d: np.ndarray, vol_faces, tess: Tessellation) -> int:
    t, ids = caster.hits(caster.centroid(cid), d, int(cid))
    return _distinct_count(t[np.isin(tess.face_tag[ids], vol_faces)], caster.eps * 1e-3)


def _inside_vote(vol: int, p: np.ndarray, n: np.ndarray, delta: float) -> int:
    """isInside probe pair at p ± d·n: -1 if the normal points into the volume, +1 if out."""
    for d in (delta, 10 * delta, 0.1 * delta):
        ahead = bool(gmsh.model.isInside(3, vol, (p + d * n).tolist()))
        behind = bool(gmsh.model.isInside(3, vol, (p - d * n).tolist()))
        if ahead != behind:
            return -1 if ahead else 1
    return 0


# ----------------------------------------------------------------------------------------------
# per-face shape facts

def _face_info(ctx: _Ctx, fd: _Face) -> FaceInfo:
    center = np.asarray(gmsh.model.occ.getCenterOfMass(2, fd.tag), float)
    info = _shape_facts(ctx, fd, center)
    as_tup = lambda k: _tup(info[k]) if info.get(k) is not None else None  # noqa: E731
    return FaceInfo(
        tag=fd.tag, kind=fd.kind, area=float(gmsh.model.occ.getMass(2, fd.tag)), center=_tup(center),
        bbox=_bbox(2, fd.tag), normal=as_tup("normal"),
        radius=info.get("radius"), axis=as_tup("axis"), axis_point=as_tup("axis_point"),
        concave=info.get("concave"), angle_span=info.get("angle_span"),
        volumes=fd.volumes, edges=fd.edges, neighbors=[])


def _shape_facts(ctx: _Ctx, fd: _Face, center: np.ndarray) -> dict:
    """Normal at the sample nearest the centre plus kind-specific facts; failures are recorded in
    ctx.unresolved (reported as analysis_incomplete), never silently dropped."""
    if len(fd.xyz) == 0:
        ctx.unresolved.append(fd.tag)
        return {}
    rep = int(np.argmin(np.linalg.norm(fd.xyz - center, axis=1)))
    info = {"normal": fd.nrm[rep]}
    shape = _SHAPES.get(fd.kind)
    try:
        facts = shape(fd, rep, ctx.scale) if shape else {}
    except Exception:
        facts = {}
    if shape and not facts:
        ctx.unresolved.append(fd.tag)
    info.update(facts)
    return info


def _cylinder(fd: _Face, rep: int, scale: float) -> dict:
    """Axis = principal direction of the smallest |k|, radius = 1/largest |k|. The curvature sign does
    not separate holes from bosses, so concavity = which axis candidate p ± r·n fits the samples."""
    kmax, kmin, dmax, dmin = _curvatures(fd)
    big = max(abs(kmax[rep]), abs(kmin[rep]))
    if big <= 0:
        return {}
    axis = _canonical_axis(dmax[rep] if abs(kmax[rep]) < abs(kmin[rep]) else dmin[rep])
    r = _num(1.0 / big)
    p, n = fd.xyz[rep], fd.nrm[rep]
    err = {cv: float(np.mean(np.abs(_dist_to_line(fd.pts, p + s * r * n, axis) - r)))
           for cv, s in ((True, 1.0), (False, -1.0))}
    concave = err[True] < err[False]
    c = p + (r if concave else -r) * n
    return dict(radius=r, axis=axis, axis_point=_axis_point(c, axis), concave=bool(concave),
                angle_span=_angle_span(fd.pts, c, axis, 1e-3 * r, fd.uspan))


def _cone(fd: _Face, rep: int, scale: float) -> dict:
    """Cone normals make a constant angle with the axis (n_i·a = const): axis = null direction of
    the centred normals; radius measured at the representative sample."""
    if len(fd.xyz) < 3:
        return {}
    axis = _canonical_axis(np.linalg.svd(fd.nrm - fd.nrm.mean(axis=0))[2][-1])
    c = _axis_point_from_normals(fd.xyz, fd.nrm, axis)
    if c is None:
        return {}
    p, n = fd.xyz[rep], fd.nrm[rep]
    foot = c + np.dot(p - c, axis) * axis
    return dict(radius=float(np.linalg.norm(p - foot)), axis=axis, axis_point=_axis_point(c, axis),
                concave=bool(np.dot(n, foot - p) > 0),
                angle_span=_angle_span(fd.pts, c, axis, 1e-9 * scale, fd.uspan))


def _sphere(fd: _Face, rep: int, scale: float) -> dict:
    """axis_point carries the sphere centre (the candidate p ± r·n that fits the samples)."""
    r = _curvature_radius(fd)
    if r is None:
        return {}
    p, n = fd.xyz[rep], fd.nrm[rep]
    err = {cv: float(np.mean(np.abs(np.linalg.norm(fd.pts - (p + s * r * n), axis=1) - r)))
           for cv, s in ((True, 1.0), (False, -1.0))}
    concave = err[True] < err[False]
    return dict(radius=r, axis_point=p + (r if concave else -r) * n, concave=bool(concave))


def _torus(fd: _Face, rep: int, scale: float) -> dict:
    """radius = minor (tube) radius; tube centres p ± r·n are coplanar only for the right sign, and
    the normal of that plane is the main axis."""
    r = _curvature_radius(fd) if len(fd.xyz) >= 3 else None
    if r is None:
        return {}
    fits = []
    for concave, s in ((True, 1.0), (False, -1.0)):
        cen = fd.xyz + s * r * fd.nrm
        _, sv, vt = np.linalg.svd(cen - cen.mean(axis=0))
        fits.append((float(sv[-1]), concave, _canonical_axis(vt[-1])))
    _, concave, axis = min(fits, key=lambda t: t[0])
    out = dict(radius=r, axis=axis, concave=bool(concave))
    c = _axis_point_from_normals(fd.xyz, fd.nrm, axis)
    if c is not None:
        out.update(axis_point=_axis_point(c, axis),
                   angle_span=_angle_span(fd.pts, c, axis, 1e-9 * scale, fd.uspan))
    return out


_SHAPES = {"cylinder": _cylinder, "cone": _cone, "sphere": _sphere, "torus": _torus}


# ----------------------------------------------------------------------------------------------
# ray casting: orientation parity, thickness, gap

class _RayCaster:
    """Cell locator over the tessellation; distances are recomputed exactly per candidate triangle.
    (vtkOBBTree.IntersectWithLine returns no hits in VTK 9.7.1, hence vtkCellLocator.)"""

    def __init__(self, tess: Tessellation, scale: float):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkCellLocator, vtkPolyData
        vp = vtkPoints()
        vp.SetDataTypeToDouble()
        vp.SetData(numpy_to_vtk(np.ascontiguousarray(tess.points, dtype=float), deep=True))
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(tess.triangles) + 1, 3), deep=True),
                      numpy_to_vtkIdTypeArray(np.ascontiguousarray(tess.triangles.ravel()), deep=True))
        self._pd = vtkPolyData()
        self._pd.SetPoints(vp)
        self._pd.SetPolys(cells)
        self._loc = vtkCellLocator()
        self._loc.SetDataSet(self._pd)
        self._loc.BuildLocator()
        self._hp, self._hi = vtkPoints(), vtkIdList()
        self.tri = tess.points[tess.triangles]   # vertex positions: unaffected by later flips
        self.eps, self.length = 1e-6 * scale, 1.01 * scale

    def centroid(self, cid: int) -> np.ndarray:
        return self.tri[cid].mean(axis=0)

    def hits(self, origin: np.ndarray, direction: np.ndarray, source: int):
        """(distances, cells) of every triangle hit beyond eps along the ray, source cell excluded."""
        ids = self._candidates(origin, direction, source)
        t = _ray_triangle(origin, direction, self.tri[ids]) if len(ids) else np.zeros(0)
        ok = np.isfinite(t) & (t > self.eps)
        return t[ok], ids[ok]

    def _candidates(self, origin: np.ndarray, direction: np.ndarray, source: int) -> np.ndarray:
        self._hp.Reset()
        self._hi.Reset()
        self._loc.IntersectWithLine(origin.tolist(), (origin + self.length * direction).tolist(),
                                    self.eps * 1e-3, self._hp, self._hi, None)
        ids = np.array([self._hi.GetId(i) for i in range(self._hi.GetNumberOfIds())], dtype=np.int64)
        return ids[ids != source]

    def first_hits(self, origins: np.ndarray, dirs: np.ndarray, sources: np.ndarray):
        """Nearest hit per ray, batched: (distances (inf = none), cells (-1 = none))."""
        cand = [self._candidates(o, d, int(s)) for o, d, s in zip(origins, dirs, sources)]
        ray = np.repeat(np.arange(len(cand)), [len(c) for c in cand])
        cells = np.concatenate(cand) if cand else np.zeros(0, dtype=np.int64)
        t = _ray_triangle(origins[ray], dirs[ray], self.tri[cells]) if len(cells) else np.zeros(0)
        t = np.where(np.isfinite(t) & (t > self.eps), t, np.inf)
        best = np.full(len(cand), np.inf)
        np.minimum.at(best, ray, t)
        cell = np.full(len(cand), -1, dtype=np.int64)
        win = np.isfinite(t) & (t == best[ray])
        cell[ray[win]] = cells[win]
        return best, cell


def _ray_measures(faces: list[FaceInfo], tess: Tessellation, caster: _RayCaster,
                  group: dict[int, int]) -> None:
    """thickness = median / thickness_min = min inward first-hit distance over <=50 triangles per
    face; gap = min outward first-hit distance when the hit is on another face (and not on another
    face of the same feature, e.g. the other half of a split hole)."""
    rows = _rows_by_face(tess)
    members: dict[int, set[int]] = defaultdict(set)
    for t, g in group.items():
        members[g].add(t)
    for fi in faces:
        ids = rows.get(fi.tag)
        own = members[group[fi.tag]] | {fi.tag} if fi.tag in group else {fi.tag}
        if ids is not None and fi.volumes:
            _face_rays(fi, ids[_spread(len(ids), _MAX_RAYS)], caster, tess, own)


def _face_rays(fi: FaceInfo, ids, caster: _RayCaster, tess: Tessellation, own: set[int]) -> None:
    o, n = caster.tri[ids].mean(axis=1), tess.normals[ids]
    t, _ = caster.first_hits(o, -n, ids)
    if np.isfinite(t).any():
        thick = t[np.isfinite(t)]
        fi.thickness, fi.thickness_min = float(np.median(thick)), float(np.min(thick))
    if len(fi.volumes) != 1:
        return
    g, cell = caster.first_hits(o, n, ids)
    other = (cell >= 0) & ~np.isin(tess.face_tag[np.maximum(cell, 0)], list(own))
    if other.any():
        fi.gap = float(np.min(g[other]))


# ----------------------------------------------------------------------------------------------
# features

def _features(faces: list[FaceInfo], ctx: _Ctx) -> list[Feature]:
    """Group cylinders by (concavity, radius ±1e-3·r, collinear axis, touching) and classify. Pairs
    are only tested inside a spatial hash of the axis points, so this is ~linear in cylinders."""
    cyl = [f for f in faces if f.kind == "cylinder" and f.radius and f.axis is not None
           and f.axis_point is not None and f.concave is not None]
    ext = {f.tag: _axial_extent(ctx.faces[f.tag].pts, f.axis) for f in cyl}
    pairs = [(a.tag, b.tag) for a, b in _candidate_pairs(cyl, ctx.diag)
             if _same_cylinder(a, b, ext, ctx.diag)]
    by_tag = {f.tag: f for f in cyl}
    groups = [[by_tag[t] for t in g] for g in connected_groups([f.tag for f in cyl], pairs)]
    feats = [_feature(g, ctx) for g in groups]
    return [f for f in feats if f is not None]


def _candidate_pairs(cyl: list[FaceInfo], diag: float):
    """Cylinder pairs whose axis points share or touch a hash cell larger than the merge tolerance."""
    cell = 2e-3 * max([diag, 1e-9] + [f.radius for f in cyl])
    key = {f.tag: tuple(np.floor(np.asarray(f.axis_point) / cell).astype(int)) for f in cyl}
    grid: dict[tuple, list[FaceInfo]] = defaultdict(list)
    for f in cyl:
        grid[key[f.tag]].append(f)
    for a in cyl:
        k = key[a.tag]
        for off in product((-1, 0, 1), repeat=3):
            for b in grid.get((k[0] + off[0], k[1] + off[1], k[2] + off[2]), ()):
                if b.tag > a.tag:
                    yield a, b


def _axial_extent(pts: np.ndarray, axis) -> tuple[float, float]:
    t = pts @ np.asarray(axis)
    return float(t.min()), float(t.max())


def _same_cylinder(a: FaceInfo, b: FaceInfo, ext: dict, diag: float) -> bool:
    r = max(a.radius, b.radius)
    tol = 1e-3 * r + 1e-6 * diag
    aa = np.asarray(a.axis)
    if (a.concave != b.concave or abs(a.radius - b.radius) > 1e-3 * r
            or abs(float(aa @ np.asarray(b.axis))) < 1 - 1e-6
            or _dist_to_line(np.asarray([b.axis_point]), np.asarray(a.axis_point), aa)[0] > tol):
        return False
    (a0, a1), (b0, b1) = ext[a.tag], ext[b.tag]
    return b.tag in a.neighbors or (a0 <= b1 + tol and b0 <= a1 + tol)


def _feature(members: list[FaceInfo], ctx: _Ctx) -> Feature | None:
    f0 = members[0]
    axis, ap = np.asarray(f0.axis), np.asarray(f0.axis_point)
    r = _num(np.mean([f.radius for f in members]))
    arcs = [_arc(ctx.faces[f.tag].pts, ap, axis, 1e-3 * r, ctx.faces[f.tag].uspan) for f in members]
    span = _union_span([a for a in arcs if a is not None])
    lo, hi = _axial_extent(np.vstack([ctx.faces[f.tag].pts for f in members]), axis)
    kind = _classify(f0.concave, span, r, ctx.diag)
    if kind is None:
        return None
    return Feature(kind=kind, faces=sorted(f.tag for f in members),
                   params={"radius": r, "axis": [float(x) for x in axis],
                           "axis_point": [float(x) for x in ap], "depth": _num(hi - lo),
                           "angle_span": _num(span, 9)})


def _classify(concave: bool, span: float, r: float, diag: float) -> str | None:
    if span >= _SPAN_FULL:
        return "hole" if concave else "boss"
    if concave:
        return "fillet" if r < 0.1 * diag else None
    return "round"


def _feature_index(features: list[Feature]) -> dict[int, int]:
    """face tag -> index of the feature it belongs to."""
    return {t: i for i, ft in enumerate(features) for t in ft.faces}


# ----------------------------------------------------------------------------------------------
# hazards

def _id_text(tags, cap: int = 12) -> str:
    """Face/edge id list for messages: at most `cap` ids, then '+N more'."""
    tags = list(tags)
    more = f" +{len(tags) - cap} more" if len(tags) > cap else ""
    return "[" + ", ".join(str(t) for t in tags[:cap]) + more + "]"


def _hazards(faces, edges, volumes, features, ctx: _Ctx) -> list[Hazard]:
    h = ctx.ref_size
    edge_len = {e.tag: e.length for e in edges}
    seam = {e.tag for e in edges if len(e.faces) == 1}
    out = _hz_no_volume(faces) if not volumes else []
    for f in faces:
        per = sum(edge_len.get(e, 0.0) for e in f.edges if not (f.volumes and e in seam))
        out += _hz_small_or_sliver(f, per, h)
    out += [_hz_short_edge(e, h, ctx.edge_pts.get(e.tag, _EMPTY3)) for e in edges
            if e.kind != "degenerate" and e.length < 0.25 * h]
    out += _hz_walls(faces, "thickness", h) + _hz_walls(faces, "gap", h)
    out += [_hz_small_fillet(ft, h) for ft in features
            if ft.kind == "fillet" and ft.params["radius"] < 0.5 * h]
    out += [_hz_interface(f) for f in faces if len(f.volumes) >= 2]
    return out


def _hz_no_volume(faces) -> list[Hazard]:
    return [Hazard(
        kind="no_volume", entities=[(2, f.tag) for f in faces], value=None, threshold=None,
        location=None,
        message=f"model has no 3D volume ({len(faces)} surfaces only); a solid mesh cannot be built",
        suggestion="export the part as a closed solid from CAD, or retry with heal: true (sewing) if "
                   "the shell is closed but not stitched")]


def _hz_small_or_sliver(f: FaceInfo, perimeter: float, h: float) -> list[Hazard]:
    """small_face if A < (h/2)^2, else sliver_face if width 2A/P < h/4 (P without seam edges)."""
    small, thin = (0.5 * h) ** 2, 0.25 * h
    if f.area < small:
        return [Hazard(
            kind="small_face", entities=[(2, f.tag)], value=f.area, threshold=small, location=f.center,
            message=f"face {f.tag} ({f.kind}) area {_f(f.area)} < (0.5*h)^2 = {_f(small)}; the mesher "
                    f"must refine locally to resolve it",
            suggestion=f"defeature it in CAD or heal: true; if it is real, accept local refinement "
                       f"near face {f.tag} or lower size.global")]
    w = 2.0 * f.area / perimeter if perimeter > 0 else math.inf
    if w >= thin:
        return []
    return [Hazard(
        kind="sliver_face", entities=[(2, f.tag)], value=w, threshold=thin, location=f.center,
        message=f"face {f.tag} ({f.kind}) is a sliver: width 2A/P = {_f(w)} < 0.25*h = {_f(thin)} "
                f"(area {_f(f.area)})",
        suggestion=f"defeature it in CAD (usually a modelling step of ~{_f(w)}) or heal: true; "
                   f"otherwise expect elements of size ~{_f(w)} near face {f.tag}")]


def _hz_short_edge(e: EdgeInfo, h: float, pts: np.ndarray) -> Hazard:
    return Hazard(
        kind="short_edge", entities=[(1, e.tag)], value=e.length, threshold=0.25 * h,
        location=_tup(pts[len(pts) // 2]) if len(pts) else None,
        message=f"edge {e.tag} ({e.kind}) length {_f(e.length)} < 0.25*h = {_f(0.25 * h)} "
                f"(faces {_id_text(e.faces)})",
        suggestion="it forces elements of that size locally; defeature the tiny step in CAD or "
                   "heal: true, otherwise expect a local size jump (watch the sicn gate)")


def _hz_walls(faces, attr: str, h: float) -> list[Hazard]:
    """thin_wall (attr=thickness) / narrow_gap (attr=gap) below 2h, one hazard per value (3 s.f.)."""
    bucket: dict[float, list[FaceInfo]] = defaultdict(list)
    for f in faces:
        v = getattr(f, attr)
        if v is not None and v < 2.0 * h:
            bucket[float(f"{v:.3g}")].append(f)
    out = []
    for _, fs in sorted(bucket.items()):
        tags = sorted(f.tag for f in fs)
        val = float(np.median([getattr(f, attr) for f in fs]))
        msg, sug = (_thin_wall_text if attr == "thickness" else _narrow_gap_text)(tags, val, h)
        out.append(Hazard(kind="thin_wall" if attr == "thickness" else "narrow_gap",
                          entities=[(2, t) for t in tags], value=val, threshold=2.0 * h,
                          location=max(fs, key=lambda f: f.area).center, message=msg, suggestion=sug))
    return out


def _thin_wall_text(tags, t: float, h: float) -> tuple[str, str]:
    ids = _id_text(tags)
    return (f"wall thickness {_f(t)} < 2*h = {_f(2 * h)} on faces {ids}; at size {_f(h)} there are "
            f"fewer than 2 elements through the wall",
            f"local size <= t/2 = {_f(t / 2)} on faces {ids} (spec: refine: [{{select: <query for these "
            f"faces>, size: {_f(t / 2)}, distance: {_f(t)}}}]) or element order 2 (spec: element.order: "
            f"2); elements through the wall: size.thin_wall_layers")


def _narrow_gap_text(tags, g: float, h: float) -> tuple[str, str]:
    ids = _id_text(tags)
    return (f"gap {_f(g)} < 2*h = {_f(2 * h)} in front of faces {ids} (distance to the next surface "
            f"along the outward normal)",
            f"local size <= gap/2 = {_f(g / 2)} near faces {ids} (spec: refine with size {_f(g / 2)}), "
            f"or close/defeature the gap in CAD")


def _hz_small_fillet(ft: Feature, h: float) -> Hazard:
    r = ft.params["radius"]
    return Hazard(
        kind="small_fillet", entities=[(2, t) for t in ft.faces], value=r, threshold=0.5 * h,
        location=None,
        message=f"fillet r={_f(r)} < 0.5*h = {_f(0.5 * h)} on faces {_id_text(ft.faces)}; it will be "
                f"faceted by about one element or force local refinement",
        suggestion=f"if stress at the fillet matters, refine to size <= r/3 = {_f(r / 3)} (spec: "
                   f"refine: [{{select: {{feature: fillet}}, size: {_f(r / 3)}}}]); else defeature it")


def _hz_interface(f: FaceInfo) -> Hazard:
    return Hazard(
        kind="multi_volume_interface", entities=[(2, f.tag)], value=f.area, threshold=None,
        location=f.center,
        message=f"face {f.tag} is shared by volumes {_id_text(f.volumes)} (conformal interface after "
                f"fragment: nodes are shared, the bodies are bonded)",
        suggestion="fine for bonded multi-body parts; it is internal, so it is not a boundary-group "
                   "candidate (default_group takes only faces with one volume)")


def _hz_incomplete(ctx: _Ctx, tessellate: bool) -> list[Hazard]:
    """analysis_incomplete: failed/partial tessellation (no thickness/gap) and unresolved faces."""
    out = []
    missing = sorted(t for t, fd in ctx.faces.items() if not len(fd.tris)) if tessellate else []
    if ctx.mesh_error or missing:
        why = f"temporary tessellation failed ({ctx.mesh_error})" if ctx.mesh_error else \
            "temporary tessellation is partial"
        out.append(Hazard(
            kind="analysis_incomplete", entities=[(2, t) for t in missing], value=None, threshold=None,
            location=None,
            message=f"{why}: thickness/gap not measured on faces {_id_text(missing)}; thin_wall and "
                    f"narrow_gap hazards may be missing there",
            suggestion="retry with heal: true or a different ref_size; inspect those faces "
                       "(render/describe) before trusting the hazard list"))
    if ctx.unresolved:
        out.append(Hazard(
            kind="analysis_incomplete", entities=[(2, t) for t in ctx.unresolved], value=None,
            threshold=None, location=None,
            message=f"shape facts unresolved on faces {_id_text(ctx.unresolved)} (sampling/curvature "
                    f"query failed): normal/axis/radius/concavity may be missing",
            suggestion="select these faces by position (box/near/at) rather than by axis/radius"))
    return out


# ----------------------------------------------------------------------------------------------
# digest

def _f(x, ztol: float = 5e-13) -> str:
    """Compact number: 4 significant digits, integers >= 1e4 without decimals, |x| <= ztol -> 0."""
    if x is None:
        return "-"
    x = float(x)
    if abs(x) <= ztol:
        return "0"
    s = f"{x:.4g}"
    s = f"{x:.0f}" if "e+" in s else s
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _pt(v, ztol: float = 5e-13) -> str:
    return "(" + ",".join(_f(c, ztol) for c in v) + ")"


def _dir_label(v) -> str:
    """+x / -y / ... for axis-aligned unit vectors, else the components to 3 decimals."""
    v = np.asarray(v, float)
    for i, ax in enumerate("xyz"):
        if abs(abs(v[i]) - 1) < 1e-6:
            return ("+" if v[i] > 0 else "-") + ax
    return "(" + ",".join(_f(round(float(c), 3), 5e-4) for c in v) + ")"


def _ids(tags, cap: int = 24) -> str:
    tags = list(tags)
    return "[" + ",".join(str(t) for t in tags[:cap]) + (",…" if len(tags) > cap else "") + "]"


def _counts(items) -> str:
    c: dict[str, int] = defaultdict(int)
    for it in items:
        c[it.kind] += 1
    return ", ".join(f"{k} {n}" for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def _capped(lines: list[str], full: bool, what: str) -> list[str]:
    """Summary keeps the first _SUMMARY_GROUPS lines, then '… N more <what> (use detail=full)'."""
    if full or len(lines) <= _SUMMARY_GROUPS:
        return lines
    return lines[:_SUMMARY_GROUPS] + [f"… {len(lines) - _SUMMARY_GROUPS} more {what} (use detail=full)"]


def digest_text(model: GeometryModel, detail: str = "summary", max_faces: int = 60) -> str:
    """Deterministic text digest: header, FEATURES, PLANAR FACES, HAZARDS, FACES. detail="summary"
    is token-bounded: groups are capped and FACES stops at max_faces rows or ~_DIGEST_BUDGET chars;
    detail="full" lists every feature, face and edge."""
    if detail not in ("summary", "full"):
        raise ValueError(f"detail must be 'summary' or 'full', got {detail!r}")
    full = detail == "full"
    z = 1e-6 * model.diag  # kernel tolerance noise (bbox enlargement, mass integration) prints as 0
    sections = [_dg_header(model, z), _dg_features(model, z, full), _dg_planes(model, z, full),
                _dg_hazards(model, full)]
    used = sum(len(line) + 1 for s in sections for line in s)
    budget = math.inf if full else max(_DIGEST_BUDGET - used, 1500)
    sections.append(_dg_faces(model, z, max_faces if not full else len(model.faces), budget))
    if full:
        sections += [_dg_edges(model), _dg_measures(model, z)]
    return "\n\n".join("\n".join(s) for s in sections) + "\n"


def _dg_header(model: GeometryModel, z: float) -> list[str]:
    lo, hi = model.bbox
    size = np.subtract(hi, lo)
    h = model.ref_size
    vols = ", ".join(f"{v.tag}: V={_f(v.volume)} faces={len(v.faces)}" for v in model.volumes[:10])
    more = f", … {len(model.volumes) - 10} more" if len(model.volumes) > 10 else ""
    return [
        f"GEOMETRY {model.source or '(current model)'}  units={model.units}",
        f"bbox {_pt(lo, z)} .. {_pt(hi, z)}  size {_f(size[0])} x {_f(size[1])} x {_f(size[2])}  "
        f"diag {_f(model.diag)}",
        f"volumes {len(model.volumes)}" + (f" [{vols}{more}]" if vols else " (none)"),
        f"faces {len(model.faces)} ({_counts(model.faces)}); total area "
        f"{_f(sum(f.area for f in model.faces))}",
        f"edges {len(model.edges)} ({_counts(model.edges)})",
        f"ref_size h={_f(h)}  thresholds: small_face A<{_f((0.5 * h) ** 2)}, short_edge "
        f"L<{_f(0.25 * h)}, sliver 2A/P<{_f(0.25 * h)}, thin_wall/narrow_gap <{_f(2 * h)}, "
        f"small_fillet r<{_f(0.5 * h)}",
        *[f"NOTE: {hz.message}" for hz in model.hazards if hz.kind == "analysis_incomplete"]]


def _dg_features(model: GeometryModel, z: float, full: bool) -> list[str]:
    """Identical features collapse into one line ("4× hole r=3.25 axis +z depth 6 ..."); patterns
    are ordered by count, then total face area."""
    area = {f.tag: f.area for f in model.faces}
    pat: dict[tuple, list[Feature]] = defaultdict(list)
    for ft in model.features:
        p = ft.params
        pat[(ft.kind, _f(p["radius"]), _dir_label(p["axis"]), _f(p["depth"]),
             _f(round(p["angle_span"])))].append(ft)
    order = sorted(pat.items(), key=lambda kv: (-len(kv[1]),
                                                -sum(area.get(t, 0.0) for ft in kv[1] for t in ft.faces)))
    lines = []
    for (kind, r, ax, depth, span), fts in order:
        word = "depth" if kind in ("hole", "boss") else "length"
        at = (" at " + " ".join(_pt(ft.params["axis_point"], z) for ft in fts)) if len(fts) <= 8 else ""
        faces = sorted(t for ft in fts for t in ft.faces)
        lines.append(f"- {len(fts)}× {kind} r={r} axis {ax} {word} {depth} span {span}°{at} "
                     f"faces {_ids(faces)}")
    none = ["  none detected (holes/bosses/fillets/rounds come from cylindrical faces)"]
    return [f"FEATURES ({len(model.features)})"] + (_capped(lines, full, "feature patterns") or none)


def _dg_planes(model: GeometryModel, z: float, full: bool) -> list[str]:
    """Planar faces grouped by rounded outward normal, largest total area first."""
    groups: dict[str, list[FaceInfo]] = defaultdict(list)
    for f in model.faces:
        if f.kind == "plane" and f.normal is not None:
            groups[_dir_label(np.round(f.normal, 3))].append(f)
    lines = []
    for lab, fs in sorted(groups.items(), key=lambda kv: -sum(f.area for f in kv[1])):
        fs = sorted(fs, key=lambda f: -f.area)
        cap = len(fs) if full else 8
        parts = [f"{f.tag} (A={_f(f.area)}, {_plane_offset(f, lab, z)})" for f in fs[:cap]]
        parts += [f"… {len(fs) - cap} more"] if len(fs) > cap else []
        lines.append(f"{lab:<6} {len(fs)} face{'s' if len(fs) != 1 else ''}  "
                     f"ΣA={_f(sum(f.area for f in fs))}: " + "; ".join(parts))
    head = ["PLANAR FACES (by outward normal; offset = n·x)"]
    return head + (_capped(lines, full, "normal groups") or ["  none"])


def _plane_offset(f: FaceInfo, label: str, z: float) -> str:
    off = float(np.dot(f.normal, f.center))
    if len(label) == 2:  # axis-aligned: report the coordinate, e.g. z=6
        i = "xyz".index(label[1])
        return f"{label[1]}={_f(off * f.normal[i], z)}"
    return f"d={_f(off, z)}"


def _dg_hazards(model: GeometryModel, full: bool) -> list[str]:
    """Summary: up to 3 hazards of a kind are listed, more collapse into one line."""
    out = [f"HAZARDS ({len(model.hazards)})"]
    if not model.hazards:
        out.append(f"  none at ref_size h={_f(model.ref_size)}")
    by_kind: dict[str, list[Hazard]] = defaultdict(list)
    for hz in model.hazards:
        by_kind[hz.kind].append(hz)
    for kind, hs in by_kind.items():
        if full or len(hs) <= 3:
            out += [f"- {kind}: {hz.message} → {hz.suggestion}" for hz in hs]
            continue
        ents = [t for hz in hs for _, t in hz.entities]
        vals = [hz.value for hz in hs if hz.value is not None]
        rng = f" values {_f(min(vals))}..{_f(max(vals))}" if vals else ""
        dim = "edges" if hs[0].entities and hs[0].entities[0][0] == 1 else "faces"
        out.append(f"- {len(hs)}× {kind} on {dim} {_ids(ents)}{rng} (threshold "
                   f"{_f(hs[0].threshold)}) → e.g. {hs[0].suggestion}")
    return out


def _face_row(f: FaceInfo, ztol: float) -> str:
    dirn = ("a " + _dir_label(f.axis) if f.axis is not None
            else "n " + _dir_label(f.normal) if f.normal is not None else "-")
    cv = "-" if f.concave is None else ("y" if f.concave else "n")
    return (f"{f.tag:>5} {f.kind:<10} {_f(f.area):>10} {_pt(f.center, ztol):<22} {dirn:<16} "
            f"{_f(f.radius):>7} {cv:>7}  {_ids(f.neighbors, 12)}")


def _dg_faces(model: GeometryModel, z: float, max_rows: int, budget: float) -> list[str]:
    """FACES table, largest first, until max_rows rows or the char budget is used up."""
    faces = sorted(model.faces, key=lambda f: (-f.area, f.tag))
    rows: list[str] = []
    for f in faces[:max_rows]:
        row = _face_row(f, z)
        if rows and sum(len(r) + 1 for r in rows) + len(row) > budget:
            break
        rows.append(row)
    note = "" if len(rows) == len(faces) else f"; largest {len(rows)} by area"
    out = [f"FACES ({len(faces)}{note})",
           f"{'tag':>5} {'kind':<10} {'area':>10} {'center':<22} {'normal|axis':<16} {'r':>7} "
           f"{'concave':>7}  nbrs"] + rows
    if len(faces) > len(rows):
        out.append(f"… {len(faces) - len(rows)} more (use detail=full)")
    return out


def _dg_edges(model: GeometryModel) -> list[str]:
    out = [f"EDGES ({len(model.edges)})", f"{'tag':>5} {'kind':<10} {'length':>10}  faces"]
    out += [f"{e.tag:>5} {e.kind:<10} {_f(e.length):>10}  {_ids(e.faces)}"
            for e in sorted(model.edges, key=lambda e: e.tag)]
    return out


def _dg_measures(model: GeometryModel, z: float) -> list[str]:
    out = ["FACE MEASURES (thickness = median inward ray, gap = min outward ray)"]
    for f in sorted(model.faces, key=lambda f: f.tag):
        if f.thickness is None and f.gap is None and f.angle_span is None:
            continue
        out.append(f"{f.tag:>5} t={_f(f.thickness)} t_min={_f(f.thickness_min)} gap={_f(f.gap)}"
                   + (f" span={_f(f.angle_span)}°" if f.angle_span is not None else "")
                   + (f" axis_point={_pt(f.axis_point, z)}" if f.axis_point is not None else ""))
    return out
