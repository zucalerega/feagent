"""Geometry analysis of the current gmsh/OCC model and the text digest an agent reads before meshing.

Everything reported comes from the kernel (OCC via gmsh): areas, centres, bounding boxes, outward
normals, cylinder axes/radii, concavity, adjacency. A temporary 2D tessellation supplies exact surface
sample points (interior mesh nodes carry their uv) and the triangles used for ray-cast wall thickness
and gap estimates. The mesh is always cleared again before returning.
"""
from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from .gmsh_session import gmsh_session, load_cad
from .types import (EdgeInfo, FaceInfo, Feature, GeometryModel, Hazard, Tessellation,
                    VolumeInfo)

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
_SPAN_FULL = 300.0         # angle_span (deg) at or above which a cylinder counts as a full revolution
_EMPTY3 = np.zeros((0, 3))


@dataclass
class _Face:
    """Working data for one face while the model is analyzed."""
    tag: int
    kind: str
    volumes: list[int]
    edges: list[int]
    xyz: np.ndarray = field(default_factory=lambda: _EMPTY3)   # exact samples (k,3)
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    nrm: np.ndarray = field(default_factory=lambda: _EMPTY3)   # outward normals at samples
    pts: np.ndarray = field(default_factory=lambda: _EMPTY3)   # every known point on the face
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    sign: float = 1.0          # getNormal * sign = outward normal
    uspan: float | None = None


@dataclass
class _Ctx:
    scale: float               # diag, or 1 for an empty model (tolerances)
    diag: float
    ref_size: float
    faces: dict[int, _Face]
    edge_pts: dict[int, np.ndarray]


# ----------------------------------------------------------------------------------------------
# vector helpers

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _unit_rows(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a / np.maximum(np.linalg.norm(a, axis=1), 1e-300)[:, None]


def _canonical_axis(a) -> np.ndarray:
    """Unit axis with its first non-negligible component positive; ~0 components snap to 0."""
    a = _unit(a)
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
    a = _unit(axis)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, ref))
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
    a = _unit(axis)
    P = pts - np.outer(pts @ a, a)
    M = nrm - np.outer(nrm @ a, a)
    ok = np.linalg.norm(M, axis=1) > 1e-9
    if ok.sum() < 2:
        return None
    P, M = P[ok], _unit_rows(M[ok])
    Q = np.eye(3)[None] - M[:, :, None] * M[:, None, :]
    A = Q.sum(axis=0) + np.outer(a, a)  # pin the component along the axis to 0
    try:
        return np.linalg.solve(A, np.einsum("kij,kj->i", Q, P))
    except np.linalg.LinAlgError:
        return None


def _ray_triangle(o: np.ndarray, d: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Moller-Trumbore distance along unit d from o to each triangle (k,3,3); inf where missed."""
    a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    good = np.abs(det) > 1e-300
    inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
    tv = o - a
    qv = np.cross(tv, e1)
    u = np.einsum("ij,ij->i", tv, pv) * inv
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = good & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return np.where(hit, t, np.inf)


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
    return _unit_rows(np.asarray(raw, float).reshape(-1, 3))


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


@contextmanager
def _mesh_options(opts: dict[str, float]):
    saved = {k: gmsh.option.getNumber(k) for k in opts}
    try:
        for k, v in opts.items():
            gmsh.option.setNumber(k, v)
        yield
    finally:
        gmsh.model.mesh.clear()
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)


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
    """Analyze the model of the CURRENT gmsh session. Leaves no mesh behind and restores the mesh
    options it touches. ref_size (default diag/40) sets the hazard thresholds; tessellate=False skips
    the temporary mesh (no thickness/gap, no Tessellation; samples come from a uv grid)."""
    bbox = _model_bbox()
    diag = float(np.linalg.norm(np.subtract(bbox[1], bbox[0])))
    h = float(ref_size) if ref_size and ref_size > 0 else (diag / 40.0 if diag > 0 else 1.0)
    ctx = _Ctx(scale=diag if diag > 0 else 1.0, diag=diag, ref_size=h, faces=_face_records(),
               edge_pts={})
    volumes, edges = _volume_infos(), _edge_infos(ctx.scale)
    ctx.edge_pts = {e.tag: _curve_points(e.tag) for e in edges}
    points = _tessellate(ctx) if tessellate and ctx.faces else _EMPTY3
    faces = [_face_info(ctx, fd) for fd in ctx.faces.values()]
    _set_neighbors(faces, edges)
    tess = _build_tessellation(ctx.faces, points) if len(points) else None
    if tess is not None and volumes:
        _ray_measures(faces, tess, ctx.scale)
    features = _features(faces, ctx)
    hazards = _hazards(faces, edges, volumes, features, ctx)
    return GeometryModel(source=source, units="mm", bbox=bbox, diag=diag, ref_size=h,
                         volumes=volumes, faces=faces, edges=edges, features=features,
                         hazards=hazards, tessellation=tess)


# ----------------------------------------------------------------------------------------------
# topology and exact measures

def _model_bbox():
    return _bbox_tuple(gmsh.model.getBoundingBox(-1, -1) if gmsh.model.getEntities() else (0.0,) * 6)


def _volume_infos() -> list[VolumeInfo]:
    occ = gmsh.model.occ
    return [VolumeInfo(tag=v, volume=float(occ.getMass(3, v)), center=_tup(occ.getCenterOfMass(3, v)),
                       bbox=_bbox_tuple(gmsh.model.getBoundingBox(3, v)), faces=_adj(3, v)[1])
            for _, v in gmsh.model.getEntities(3)]


def _edge_infos(scale: float) -> list[EdgeInfo]:
    out = []
    for _, e in gmsh.model.getEntities(1):
        length = float(gmsh.model.occ.getMass(1, e))
        kind = ("degenerate" if length <= 1e-9 * scale
                else CURVE_KINDS.get(gmsh.model.getType(1, e), "other"))
        out.append(EdgeInfo(tag=e, kind=kind, length=length, faces=_adj(1, e)[0],
                            bbox=_bbox_tuple(gmsh.model.getBoundingBox(1, e))))
    return out


def _face_records() -> dict[int, _Face]:
    out = {}
    for _, f in gmsh.model.getEntities(2):
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
    with _mesh_options({"Mesh.MeshSizeMax": size, "Mesh.MeshSizeFromCurvature": 12}):
        try:
            gmsh.model.mesh.generate(2)
        except Exception:
            pass  # keep whatever faces did mesh; the rest fall back to uv-grid samples
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        if len(ntags) == 0:
            return _EMPTY3
        ntags = np.asarray(ntags, dtype=np.int64)
        points = np.asarray(ncoord, float).reshape(-1, 3)
        row = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        row[ntags] = np.arange(len(ntags))
        for fd in ctx.faces.values():
            _read_face_mesh(fd, points, row)
    return points


def _read_face_mesh(fd: _Face, points: np.ndarray, row: np.ndarray) -> None:
    etypes, _, enodes = gmsh.model.mesh.getElements(2, fd.tag)
    tris = [np.asarray(n, dtype=np.int64).reshape(-1, 3) for t, n in zip(etypes, enodes) if t == 2]
    if tris:
        fd.tris = row[np.vstack(tris)]
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
    """Outward-oriented triangles of all faces, compacted to the nodes they use."""
    parts = [(fd.tag, *_orient_face_tris(fd, points)) for fd in fdata.values() if len(fd.tris)]
    if not parts:
        return None
    tris = np.vstack([p[1] for p in parts])
    used, inv = np.unique(tris, return_inverse=True)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    for fd in fdata.values():
        fd.tris = remap[fd.tris] if len(fd.tris) else fd.tris
    return Tessellation(points=points[used].copy(), triangles=inv.reshape(-1, 3).astype(np.int64),
                        face_tag=np.concatenate([np.full(len(p[1]), p[0], dtype=np.int64) for p in parts]),
                        normals=np.vstack([p[2] for p in parts]))


def _orient_face_tris(fd: _Face, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip triangles whose winding disagrees with the exact outward normal; return (tris, normals)."""
    tris = fd.tris.copy()
    p = points[tris]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if fd.kind == "plane" and len(fd.nrm):
        ref = np.broadcast_to(fd.nrm[0], cr.shape)
    else:  # exact outward normal at each projected centroid
        ref = fd.sign * _normals_at(fd.tag, _project(fd.tag, p.mean(axis=1))[1])
    flip = np.einsum("ij,ij->i", cr, ref) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    cr[flip] *= -1
    ln = np.linalg.norm(cr, axis=1)
    nrm = np.where(ln[:, None] > 0, cr / np.maximum(ln, 1e-300)[:, None], _unit_rows(ref))
    fd.tris = tris
    return tris, nrm


# ----------------------------------------------------------------------------------------------
# per-face analysis

def _face_info(ctx: _Ctx, fd: _Face) -> FaceInfo:
    center = np.asarray(gmsh.model.occ.getCenterOfMass(2, fd.tag), float)
    info = _sample_and_orient(ctx, fd, center)
    as_tup = lambda k: _tup(info[k]) if info.get(k) is not None else None  # noqa: E731
    return FaceInfo(
        tag=fd.tag, kind=fd.kind, area=float(gmsh.model.occ.getMass(2, fd.tag)), center=_tup(center),
        bbox=_bbox_tuple(gmsh.model.getBoundingBox(2, fd.tag)), normal=as_tup("normal"),
        radius=info.get("radius"), axis=as_tup("axis"), axis_point=as_tup("axis_point"),
        concave=info.get("concave"), angle_span=info.get("angle_span"),
        volumes=fd.volumes, edges=fd.edges, neighbors=[])


def _sample_and_orient(ctx: _Ctx, fd: _Face, center: np.ndarray) -> dict:
    """Fill samples / outward normals / known points of fd; return the shape facts of the face."""
    if len(fd.xyz) == 0:
        fd.xyz, fd.uv = _uv_grid_samples(fd.tag)
    own = [ctx.edge_pts[e] for e in fd.edges if len(ctx.edge_pts.get(e, _EMPTY3))]
    fd.pts = np.vstack([fd.pts, fd.xyz, *own])
    if len(fd.xyz) == 0:
        return {}
    raw = _normals_at(fd.tag, fd.uv)
    fd.sign = _outward_sign(fd, raw, 1e-4 * ctx.scale) if fd.volumes else 1.0
    fd.nrm = fd.sign * raw
    rep = int(np.argmin(np.linalg.norm(fd.xyz - center, axis=1)))
    info = {"normal": fd.nrm[rep]}
    try:
        info.update(_SHAPES.get(fd.kind, lambda *_: {})(fd, rep, ctx.scale))
    except Exception:  # curvature queries can fail on odd surfaces; keep the face
        pass
    return info


def _outward_sign(fd: _Face, raw: np.ndarray, delta: float) -> float:
    """+1 if getNormal points out of the face's first volume, -1 if it points in (majority vote of
    isInside probes on both sides of up to 5 samples)."""
    votes = sum(_inside_vote(fd.volumes[0], fd.xyz[i], raw[i], delta) for i in _spread(len(fd.xyz), 5))
    return -1.0 if votes < 0 else 1.0


def _inside_vote(vol: int, p: np.ndarray, n: np.ndarray, delta: float) -> int:
    for d in (delta, 10 * delta, 0.1 * delta):
        ahead = bool(gmsh.model.isInside(3, vol, (p + d * n).tolist()))
        behind = bool(gmsh.model.isInside(3, vol, (p - d * n).tolist()))
        if ahead != behind:
            return -1 if ahead else 1
    return 0


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
# ray-cast thickness and gap

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
        self.tri = tess.points[tess.triangles]
        self.eps, self.length = 1e-6 * scale, 1.01 * scale

    def first_hit(self, origin: np.ndarray, direction: np.ndarray, source: int):
        """(distance, cell) of the first triangle hit beyond eps, excluding the source cell."""
        self._hp.Reset()
        self._hi.Reset()
        self._loc.IntersectWithLine(origin.tolist(), (origin + self.length * direction).tolist(),
                                    self.eps * 1e-3, self._hp, self._hi, None)
        ids = np.array([self._hi.GetId(i) for i in range(self._hi.GetNumberOfIds())], dtype=np.int64)
        ids = ids[ids != source]
        t = _ray_triangle(origin, direction, self.tri[ids]) if len(ids) else np.zeros(0)
        t = np.where(np.isfinite(t) & (t > self.eps), t, np.inf)
        if not len(t) or not np.isfinite(t.min()):
            return None
        k = int(np.argmin(t))
        return float(t[k]), int(ids[k])


def _ray_measures(faces: list[FaceInfo], tess: Tessellation, scale: float) -> None:
    """thickness = median / thickness_min = min inward first-hit distance over <=50 triangles per
    face; gap = min outward first-hit distance when the hit is on another face."""
    caster = _RayCaster(tess, scale)
    cen = tess.points[tess.triangles].mean(axis=1)
    order = np.argsort(tess.face_tag, kind="stable")
    tags, starts = np.unique(tess.face_tag[order], return_index=True)
    groups = dict(zip(tags.tolist(), np.split(order, starts[1:])))
    for fi in faces:
        ids = groups.get(fi.tag, np.zeros(0, int))
        if len(ids) and fi.volumes:
            _face_rays(fi, ids[_spread(len(ids), _MAX_RAYS)], caster, cen, tess)


def _face_rays(fi: FaceInfo, ids, caster: _RayCaster, cen, tess: Tessellation) -> None:
    thick, gaps = [], []
    for cid in ids:
        o, n = cen[cid], tess.normals[cid]
        hit = caster.first_hit(o, -n, int(cid))
        if hit is not None:
            thick.append(hit[0])
        hit = caster.first_hit(o, n, int(cid)) if len(fi.volumes) == 1 else None
        if hit is not None and int(tess.face_tag[hit[1]]) != fi.tag:
            gaps.append(hit[0])
    if thick:
        fi.thickness, fi.thickness_min = float(np.median(thick)), float(np.min(thick))
    if gaps:
        fi.gap = float(np.min(gaps))


# ----------------------------------------------------------------------------------------------
# features

def _features(faces: list[FaceInfo], ctx: _Ctx) -> list[Feature]:
    """Group cylinders by (concavity, radius ±1e-3·r, collinear axis, touching) and classify."""
    cyl = [f for f in faces if f.kind == "cylinder" and f.radius and f.axis is not None
           and f.axis_point is not None and f.concave is not None]
    ext = {f.tag: _axial_extent(ctx.faces[f.tag].pts, f.axis) for f in cyl}
    parent = {f.tag: f.tag for f in cyl}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cyl):
        for b in cyl[i + 1:]:
            if _same_cylinder(a, b, ext, ctx.diag):
                parent[find(b.tag)] = find(a.tag)
    groups: dict[int, list[FaceInfo]] = defaultdict(list)
    for f in cyl:
        groups[find(f.tag)].append(f)
    feats = [_feature(g, ctx) for g in sorted(groups.values(), key=lambda g: g[0].tag)]
    return [f for f in feats if f is not None]


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


# ----------------------------------------------------------------------------------------------
# hazards

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
                f"(faces {e.faces})",
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
    return (f"wall thickness {_f(t)} < 2*h = {_f(2 * h)} on faces {tags}; at size {_f(h)} there are "
            f"fewer than 2 elements through the wall",
            f"local size <= t/2 = {_f(t / 2)} on faces {tags} (spec: refine: [{{select: <query for these "
            f"faces>, size: {_f(t / 2)}, distance: {_f(t)}}}]) or element order 2 (spec: element.order: "
            f"2); elements through the wall: size.thin_wall_layers")


def _narrow_gap_text(tags, g: float, h: float) -> tuple[str, str]:
    return (f"gap {_f(g)} < 2*h = {_f(2 * h)} in front of faces {tags} (distance to the next surface "
            f"along the outward normal)",
            f"local size <= gap/2 = {_f(g / 2)} near faces {tags} (spec: refine with size {_f(g / 2)}), "
            f"or close/defeature the gap in CAD")


def _hz_small_fillet(ft: Feature, h: float) -> Hazard:
    r = ft.params["radius"]
    return Hazard(
        kind="small_fillet", entities=[(2, t) for t in ft.faces], value=r, threshold=0.5 * h,
        location=None,
        message=f"fillet r={_f(r)} < 0.5*h = {_f(0.5 * h)} on faces {ft.faces}; it will be faceted "
                f"by about one element or force local refinement",
        suggestion=f"if stress at the fillet matters, refine to size <= r/3 = {_f(r / 3)} (spec: "
                   f"refine: [{{select: {{feature: fillet}}, size: {_f(r / 3)}}}]); else defeature it")


def _hz_interface(f: FaceInfo) -> Hazard:
    return Hazard(
        kind="multi_volume_interface", entities=[(2, f.tag)], value=f.area, threshold=None,
        location=f.center,
        message=f"face {f.tag} is shared by volumes {f.volumes} (conformal interface after fragment: "
                f"nodes are shared, the bodies are bonded)",
        suggestion="fine for bonded multi-body parts; it is internal, so it is not a boundary-group "
                   "candidate (default_group takes only faces with one volume)")


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


def digest_text(model: GeometryModel, detail: str = "summary", max_faces: int = 60) -> str:
    """Deterministic text digest: header, FEATURES, PLANAR FACES, HAZARDS, FACES. detail="summary"
    truncates FACES to the max_faces largest; detail="full" lists every face and edge."""
    if detail not in ("summary", "full"):
        raise ValueError(f"detail must be 'summary' or 'full', got {detail!r}")
    full = detail == "full"
    z = 1e-6 * model.diag  # kernel tolerance noise (bbox enlargement, mass integration) prints as 0
    sections = [_dg_header(model, z), _dg_features(model, z), _dg_planes(model, z, full),
                _dg_hazards(model, full), _dg_faces(model, z, full, max_faces)]
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
        f"small_fillet r<{_f(0.5 * h)}"]


def _dg_features(model: GeometryModel, z: float) -> list[str]:
    """Identical features collapse into one line: "4× hole r=3.25 axis +z depth 6 ..."."""
    pat: dict[tuple, list[Feature]] = defaultdict(list)
    for ft in model.features:
        p = ft.params
        pat[(ft.kind, _f(p["radius"]), _dir_label(p["axis"]), _f(p["depth"]),
             _f(round(p["angle_span"])))].append(ft)
    out = [f"FEATURES ({len(model.features)})"]
    if not pat:
        out.append("  none detected (holes/bosses/fillets/rounds come from cylindrical faces)")
    for (kind, r, ax, depth, span), fts in pat.items():
        word = "depth" if kind in ("hole", "boss") else "length"
        at = (" at " + " ".join(_pt(ft.params["axis_point"], z) for ft in fts)) if len(fts) <= 8 else ""
        faces = sorted(t for ft in fts for t in ft.faces)
        out.append(f"- {len(fts)}× {kind} r={r} axis {ax} {word} {depth} span {span}°{at} "
                   f"faces {_ids(faces)}")
    return out


def _dg_planes(model: GeometryModel, z: float, full: bool) -> list[str]:
    """Planar faces grouped by rounded outward normal, largest group first."""
    groups: dict[str, list[FaceInfo]] = defaultdict(list)
    for f in model.faces:
        if f.kind == "plane" and f.normal is not None:
            groups[_dir_label(np.round(f.normal, 3))].append(f)
    out = ["PLANAR FACES (by outward normal; offset = n·x)"] + ([] if groups else ["  none"])
    for lab, fs in sorted(groups.items(), key=lambda kv: -sum(f.area for f in kv[1])):
        fs = sorted(fs, key=lambda f: -f.area)
        cap = len(fs) if full else 8
        parts = [f"{f.tag} (A={_f(f.area)}, {_plane_offset(f, lab, z)})" for f in fs[:cap]]
        parts += [f"… {len(fs) - cap} more"] if len(fs) > cap else []
        out.append(f"{lab:<6} {len(fs)} face{'s' if len(fs) != 1 else ''}  "
                   f"ΣA={_f(sum(f.area for f in fs))}: " + "; ".join(parts))
    return out


def _plane_offset(f: FaceInfo, label: str, z: float) -> str:
    off = float(np.dot(f.normal, f.center))
    if len(label) == 2:  # axis-aligned: report the coordinate, e.g. z=6
        i = "xyz".index(label[1])
        return f"{label[1]}={_f(off * f.normal[i], z)}"
    return f"d={_f(off, z)}"


def _dg_hazards(model: GeometryModel, full: bool) -> list[str]:
    out = [f"HAZARDS ({len(model.hazards)})"]
    if not model.hazards:
        out.append(f"  none at ref_size h={_f(model.ref_size)}")
    by_kind: dict[str, list[Hazard]] = defaultdict(list)
    for hz in model.hazards:
        by_kind[hz.kind].append(hz)
    for kind, hs in by_kind.items():
        if full or len(hs) <= 4:
            out += [f"- {kind}: {hz.message} → {hz.suggestion}" for hz in hs]
            continue
        ents = [t for hz in hs for _, t in hz.entities]
        vals = [hz.value for hz in hs if hz.value is not None]
        rng = f" values {_f(min(vals))}..{_f(max(vals))}" if vals else ""
        dim = "edges" if hs[0].entities and hs[0].entities[0][0] == 1 else "faces"
        out.append(f"- {len(hs)}× {kind} on {dim} {_ids(ents)}{rng} (threshold "
                   f"{_f(hs[0].threshold)}) → {hs[0].suggestion}")
    return out


def _face_row(f: FaceInfo, ztol: float) -> str:
    dirn = ("a " + _dir_label(f.axis) if f.axis is not None
            else "n " + _dir_label(f.normal) if f.normal is not None else "-")
    cv = "-" if f.concave is None else ("y" if f.concave else "n")
    return (f"{f.tag:>5} {f.kind:<10} {_f(f.area):>10} {_pt(f.center, ztol):<22} {dirn:<16} "
            f"{_f(f.radius):>7} {cv:>7}  {_ids(f.neighbors, 12)}")


def _dg_faces(model: GeometryModel, z: float, full: bool, max_faces: int) -> list[str]:
    faces = sorted(model.faces, key=lambda f: (-f.area, f.tag))
    shown = faces if full else faces[:max_faces]
    note = "" if len(shown) == len(faces) else f"; largest {max_faces} by area"
    out = [f"FACES ({len(faces)}{note})",
           f"{'tag':>5} {'kind':<10} {'area':>10} {'center':<22} {'normal|axis':<16} {'r':>7} "
           f"{'concave':>7}  nbrs"]
    out += [_face_row(f, z) for f in shown]
    if len(faces) > len(shown):
        out.append(f"… {len(faces) - len(shown)} more (use detail=full)")
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
