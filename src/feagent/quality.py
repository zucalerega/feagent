"""Mesh quality measurement and mesh gates.

Each gate cites its measured value and threshold and gives a hint naming the
spec key to change. ``QualitySpec`` is duck-typed (attributes only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import gmsh_util as gu
from .gates import failed, passed, skipped, threshold_gate, warned
from .types import Gate, GeometryModel, MeshResult, Vec3, _Base
from .vec import vec3

if TYPE_CHECKING:  # pragma: no cover
    from .meshdata import MeshData
    from .spec import QualitySpec

NON_ORTHO_WARN = 65.0
N_WORST = 10
_MEASURES = ("minSICN", "gamma", "minSJ", "volume")


@dataclass
class Stats(_Base):
    min: float
    p01: float
    p05: float
    median: float
    max: float
    bins: list[float]  # 11 bin edges
    hist: list[int]  # 10 counts


@dataclass
class WorstElement(_Base):
    tag: int
    sicn: float
    gamma: float
    centroid: Vec3
    faces: list[int]  # CAD faces touched (via node classification)


@dataclass
class QualityReport(_Base):
    n_nodes: int
    n_tets: int
    sicn: Stats
    gamma: Stats
    sj: Stats
    inverted: int
    worst: list[WorstElement]
    volume_cad: float
    volume_mesh: float
    volume_error: float
    group_area: dict[str, tuple[float, float, float]]  # name -> (cad, mesh, rel_err)
    components: int
    expected_components: int
    non_ortho: Stats | None
    skewness: Stats | None
    sicn_per_tet: np.ndarray = field(repr=False)  # not serialized
    gates: list[Gate] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.pop("sicn_per_tet", None)
        return d


def _stats(a: np.ndarray) -> Stats:
    a = np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=1e6, neginf=-1e6)
    if a.size == 0:
        return Stats(0.0, 0.0, 0.0, 0.0, 0.0, [0.0] * 11, [0] * 10)
    lo, hi = float(a.min()), float(a.max())
    if not hi - lo > 1e-9 * max(1.0, abs(hi)):  # degenerate range: put everything in one bin
        hi = lo + max(1e-9 * max(1.0, abs(lo)), 1e-12)
    hist, edges = np.histogram(a, bins=10, range=(lo, hi))
    p01, p05, med = np.percentile(a, [1, 5, 50])
    return Stats(float(a.min()), float(p01), float(p05), float(med), float(a.max()),
                 [float(e) for e in edges], [int(h) for h in hist])


def _mean(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else 0.0


# --- topology / OpenFOAM-style face metrics -------------------------------

_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))  # outward-oriented for positive tets


def _tet_faces(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All 4n tet faces, each face's unique-face index, and each unique face's count."""
    faces = np.concatenate([tets[:, list(f)] for f in _TET_FACES])  # (4n,3)
    _, inv, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True,
                               return_counts=True)
    return faces, inv.ravel(), counts


def _internal_faces(tets: np.ndarray):
    """Return (owner, neighbour, face_nodes[n,3]) for faces shared by exactly two tets."""
    n = len(tets)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int)
    faces, inv, counts = _tet_faces(tets)
    owner = np.tile(np.arange(n), 4)
    order = np.argsort(inv, kind="stable")
    first = np.searchsorted(inv[order], np.arange(len(counts)))
    two = np.nonzero(counts == 2)[0]
    a, b = order[first[two]], order[first[two] + 1]
    return owner[a], owner[b], faces[a]


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Tet faces not shared with another tet."""
    faces, inv, counts = _tet_faces(tets)
    return faces[counts[inv] == 1]


def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def tet_non_ortho_skewness(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per internal face (tets P, N): non-orthogonality [deg] and OpenFOAM skewness.

    non-orthogonality = angle between face area vector S_f and d = C_N - C_P.
    skewness = |x_f - x_i| / |d|, x_f face centroid, x_i intersection of line
    C_P->C_N with the face plane. OpenFOAM normalises by a face-size measure;
    we use |d| (cell-centre distance), which is the dimensionless form used by
    checkMesh for internal faces up to that choice.
    """
    P, N, fn = _internal_faces(np.asarray(tets))
    if len(P) == 0:
        return np.zeros(0), np.zeros(0)
    pts = np.asarray(points, float)
    cent = pts[tets].mean(axis=1)
    d = cent[N] - cent[P]
    s, xf = _oriented_face_vectors(pts[fn], d)
    sn, dn = np.linalg.norm(s, axis=1), np.linalg.norm(d, axis=1)
    ok = (sn > 0) & (dn > 0)
    n = s / np.where(ok, sn, 1.0)[:, None]
    dn = np.where(ok, dn, 1.0)
    non_ortho = np.degrees(np.arccos(np.clip(_rowdot(n, d) / dn, -1.0, 1.0)))
    skew = _skewness(cent[P], d, dn, n, xf)
    # degenerate faces are as bad as it gets
    return np.where(ok, non_ortho, 90.0), np.where(ok, skew, np.inf)


def _oriented_face_vectors(tri: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Face area vectors (pointing along ``d``) and centroids of triangles ``tri[n,3,3]``."""
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    s = 0.5 * np.cross(p1 - p0, p2 - p0)
    s[_rowdot(s, d) < 0] *= -1
    return s, (p0 + p1 + p2) / 3.0


def _skewness(cp: np.ndarray, d: np.ndarray, dn: np.ndarray, n: np.ndarray,
              xf: np.ndarray) -> np.ndarray:
    """|x_f - x_i| / |d| with x_i where the line cp + t*d meets the face plane."""
    dd = _rowdot(d, n)
    dd = np.where(np.abs(dd) < 1e-300, 1e-300, dd)
    t = _rowdot(xf - cp, n) / dd
    xi = cp + d * t[:, None]
    return np.linalg.norm(xf - xi, axis=1) / dn


def _components(tets: np.ndarray) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(tets)
    if n == 0:
        return 0
    P, N, _ = _internal_faces(tets)
    g = coo_matrix((np.ones(len(P)), (P, N)), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def _expected_components(geometry: GeometryModel | None) -> int:
    if geometry is None or not geometry.volumes:
        return 1
    parent = {v.tag: v.tag for v in geometry.volumes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in geometry.faces:
        vs = [v for v in f.volumes if v in parent]
        if len(vs) == 2:
            parent[find(vs[0])] = find(vs[1])
    return len({find(t) for t in parent})


# --- gmsh-side measurements -----------------------------------------------

@dataclass
class _GmshMeasures:
    qualities: dict[str, np.ndarray]  # gmsh measure name -> per-tet values
    worst: list[WorstElement]
    thin: dict[int, tuple[float, float | None]]  # face -> (thickness, mean edge length)


def _gmsh_measures(mesh_path: str | Path, md: MeshData,
                   thin_faces: dict[int, float]) -> _GmshMeasures:
    import gmsh

    from .gmsh_session import gmsh_session

    with gmsh_session():
        gmsh.open(str(mesh_path))
        qual = gu.element_qualities(md.tet_tags, _MEASURES)
        worst = _worst_elements(md, qual["minSICN"], qual["gamma"])
        edges = _face_edge_lengths(sorted(thin_faces), md.tets, md.points) if thin_faces else {}
    thin = {f: (t, edges.get(f)) for f, t in sorted(thin_faces.items())}
    return _GmshMeasures(qual, worst, thin)


def _worst_elements(md: MeshData, sicn: np.ndarray, gamma: np.ndarray) -> list[WorstElement]:
    return [WorstElement(tag=int(md.tet_tags[i]), sicn=float(sicn[i]), gamma=float(gamma[i]),
                         centroid=vec3(md.points[md.tets[i]].mean(axis=0)),
                         faces=gu.element_faces(md.tet_tags[i]))
            for i in np.argsort(sicn, kind="stable")[:N_WORST]]


def _thin_wall_faces(geo: GeometryModel | None) -> dict[int, float]:
    """CAD face -> wall thickness, from the geometry's thin-wall hazards."""
    if geo is None:
        return {}
    return {int(tag): float(h.value) for h in geo.hazards if h.kind == "thin_wall" and h.value
            for dim, tag in h.entities if dim == 2}


def _face_edge_lengths(face_tags: list[int], tets: np.ndarray,
                       points: np.ndarray) -> dict[int, float | None]:
    """Mean surface-triangle edge length per CAD face (None if no triangles found)."""
    node_tags, _, lookup = gu.node_table()
    boundary: np.ndarray | None = None  # tet-mesh skin, for msh files without face elements
    out: dict[int, float | None] = {}
    for ft in face_tags:
        tri = gu.corner_rows(2, ft, gu.TRI_TYPES, 3, lookup)
        if len(tri) == 0 and len(tets):
            boundary = _boundary_faces(tets) if boundary is None else boundary
            tri = _triangles_on_face(ft, boundary, node_tags)
        out[ft] = _mean_edge_length(tri, points) if len(tri) else None
    return out


def _triangles_on_face(face: int, tris: np.ndarray, node_tags: np.ndarray) -> np.ndarray:
    """Rows of ``tris`` whose nodes all lie on ``face`` or its boundary curves/points."""
    allowed = gu.closure(2, face)

    @cache
    def on_face(i: int) -> bool:
        return gu.node_entity(node_tags[i]) in allowed

    keep = [row for row in tris if all(on_face(int(i)) for i in row)]
    return np.array(keep, int).reshape(-1, 3)


def _mean_edge_length(tri: np.ndarray, points: np.ndarray) -> float:
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return float(np.linalg.norm(points[e[:, 0]] - points[e[:, 1]], axis=1).mean())


# --- report ---------------------------------------------------------------

def _volumes(tet_volume: np.ndarray, geo: GeometryModel | None) -> tuple[float, float, float]:
    """(CAD volume, mesh volume, relative error)."""
    mesh = float(tet_volume.sum())
    cad = float(sum(v.volume for v in geo.volumes)) if geo is not None else 0.0
    return cad, mesh, abs(mesh - cad) / cad if cad > 0 else 0.0


def _tri_area(points: np.ndarray, tri: np.ndarray) -> float:
    p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())


def _cad_area(geo: GeometryModel, tags: list[int]) -> float:
    total = 0.0
    for t in tags:
        try:
            total += geo.face(t).area
        except KeyError:
            pass
    return total


def _group_areas(md: MeshData, result: MeshResult) -> dict[str, tuple[float, float, float]]:
    """Surface group -> (CAD area, mesh area, relative error); a group absent from the mesh
    gets error 1.0 so it is never silently dropped."""
    geo = result.geometry
    if geo is None:
        return {}
    out = {}
    for name, gr in result.groups.items():
        if gr.dim == 2:
            mesh = _tri_area(md.points, md.tris.get(name, np.zeros((0, 3), int)))
            cad = _cad_area(geo, gr.tags)
            err = 1.0 if name not in md.tris else (abs(mesh - cad) / cad if cad > 0 else 0.0)
            out[name] = (float(cad), mesh, float(err))
    return out


def _report(md: MeshData, result: MeshResult, gm: _GmshMeasures,
            cfd: tuple[np.ndarray, np.ndarray] | None) -> QualityReport:
    qual = gm.qualities
    v_cad, v_mesh, v_err = _volumes(qual["volume"], result.geometry)
    return QualityReport(
        n_nodes=len(md.points), n_tets=len(md.tets), sicn=_stats(qual["minSICN"]),
        gamma=_stats(qual["gamma"]), sj=_stats(qual["minSJ"]),
        inverted=int(np.sum(qual["minSJ"] <= 0)),
        worst=gm.worst, volume_cad=v_cad, volume_mesh=v_mesh, volume_error=float(v_err),
        group_area=_group_areas(md, result), components=_components(md.tets),
        expected_components=_expected_components(result.geometry),
        non_ortho=_stats(cfd[0]) if cfd else None, skewness=_stats(cfd[1]) if cfd else None,
        sicn_per_tet=qual["minSICN"])


def assess_mesh(mesh_path: str | Path, result: MeshResult, q: "QualitySpec",
                analysis: str) -> QualityReport:
    """Measure a .msh file against the CAD in ``result`` and evaluate the mesh gates."""
    from .meshdata import read_msh

    md = read_msh(mesh_path)
    gm = _gmsh_measures(mesh_path, md, _thin_wall_faces(result.geometry))
    cfd = tet_non_ortho_skewness(md.points, md.tets) if analysis == "cfd" else None
    rep = _report(md, result, gm, cfd)
    no_mean, sk_mean = (_mean(cfd[0]), _mean(cfd[1])) if cfd else (0.0, 0.0)
    missing = {n for n in result.groups if n not in md.tris}
    ctx = _GateInputs(rep, q, missing, gm.thin, no_mean, sk_mean)
    rep.gates = [g(ctx) for g in _GATES + (_CFD_GATES if cfd else ())]
    return rep


# --- gates ----------------------------------------------------------------

@dataclass
class _GateInputs:
    rep: QualityReport
    q: "QualitySpec"
    missing_groups: set[str]  # result groups with no triangles in the mesh
    thin: dict[int, tuple[float, float | None]]  # face -> (thickness, mean edge length)
    non_ortho_mean: float
    skewness_mean: float


def _inverted_gate(c: _GateInputs) -> Gate:
    r = c.rep
    return threshold_gate(
        "inverted", "No inverted elements", float(r.inverted), 0.0,
        message_fmt="{n} of {n_tets} tets have non-positive scaled Jacobian (min SJ {sj:.3g})",
        n=r.inverted, n_tets=r.n_tets, sj=r.sj.min,
        hint="Inverted elements: lower size.global, try algorithm.mesh3d: 1 (Delaunay) or "
             "element.optimize: netgen; check the worst elements' faces for CAD defects "
             "(heal: true).")


def _sicn_gate(c: _GateInputs) -> Gate:
    s, faces = c.rep.sicn, c.rep.worst[0].faces if c.rep.worst else []
    return threshold_gate(
        "sicn", "Minimum SICN", s.min, c.q.min_sicn, worse="below",
        message_fmt="min SICN {value:.3f} (threshold {limit}); p01 {p01:.3f}, median {median:.3f}",
        p01=s.p01, median=s.median,
        hint=f"Worst elements touch faces {faces}: add a `refine` entry there, lower "
             "size.global, enable element.optimize: netgen, or relax quality.min_sicn.")


def _gamma_gate(c: _GateInputs) -> Gate:
    s = c.rep.gamma
    return threshold_gate(
        "gamma", "Minimum gamma", s.min, c.q.min_gamma, worse="below", fail_status="warn",
        message_fmt="min gamma {value:.3f} (threshold {limit}); median {median:.3f}",
        median=s.median,
        hint="Low inscribed/circumscribed radius ratio (slivers): lower size.global, set "
             "element.optimize: netgen, or adjust quality.min_gamma.")


def _volume_gate(c: _GateInputs) -> Gate:
    r, limit = c.rep, c.q.max_volume_error
    if r.volume_cad <= 0:
        return skipped("volume_fidelity", "Volume fidelity",
                       "no CAD volume available (geometry missing or zero volume)",
                       threshold=float(limit))
    return threshold_gate(
        "volume_fidelity", "Volume fidelity", r.volume_error, limit,
        message_fmt="mesh volume {mesh:.6g} vs CAD {cad:.6g}: relative error {value:.3g} "
                    "(threshold {limit})", mesh=r.volume_mesh, cad=r.volume_cad,
        hint="Geometry is poorly resolved (curved boundaries): raise size.curvature, lower "
             "size.global, use element.order: 2, or relax quality.max_volume_error.")


def _area_gate(c: _GateInputs) -> Gate:
    limit = c.q.max_area_error
    if not c.rep.group_area:
        return skipped("area_fidelity", "Group area fidelity", "no surface groups to compare",
                       threshold=float(limit))
    name, (cad, mesh, err) = max(c.rep.group_area.items(), key=lambda kv: kv[1][2])
    return threshold_gate(
        "area_fidelity", "Group area fidelity", err, limit,
        message_fmt="worst group '{group}'{missing}: mesh area {mesh:.6g} vs CAD {cad:.6g}, "
                    "relative error {value:.3g} (threshold {limit})",
        group=name, mesh=mesh, cad=cad,
        missing=" (missing from mesh)" if name in c.missing_groups else "",
        hint=f"Group '{name}' is poorly resolved: add a `refine` entry for it, raise "
             "size.curvature, lower size.global, or relax quality.max_area_error.")


def _connectivity_gate(c: _GateInputs) -> Gate:
    have, want = c.rep.components, c.rep.expected_components
    args = ("connectivity", "Mesh connectivity",
            f"mesh has {have} face-connected component(s), geometry expects {want}",
            float(have), float(want))
    if have == want:
        return passed(*args)
    return failed(*args, hint="Mesh is split (or merged) relative to the CAD: check for tiny gaps "
                  "or unfused bodies; set heal: true or lower size.global / size.min so thin "
                  "necks are resolved.")


def _thin_wall_ratio(thin: dict[int, tuple[float, float | None]],
                     n: int) -> tuple[float | None, list[tuple[int, float, float]], list[int]]:
    """(min thickness/edge ratio over measured faces or None, faces with fewer than ``n``
    elements across as (face, thickness, edge), unmeasured faces)."""
    measured = {f: (t, e) for f, (t, e) in thin.items() if e is not None}
    unmeasured = [f for f, (_, e) in thin.items() if e is None]
    ratio = min((t / e for t, e in measured.values()), default=None)  # elements across wall
    under = [(f, t, e) for f, (t, e) in measured.items() if e > t / n]
    return ratio, under, unmeasured


def _thin_wall_gate(c: _GateInputs) -> Gate:
    n, name = c.q.thin_wall_min_elements, "Thin-wall resolution"
    if not c.thin:
        return skipped("thin_walls", name, "no thin-wall hazards in geometry", threshold=float(n))
    ratio, under, unmeasured = _thin_wall_ratio(c.thin, n)
    if ratio is None:
        return skipped("thin_walls", name,
                       f"no surface triangles found on thin-wall faces {unmeasured}",
                       threshold=float(n))
    if under:
        faces = ", ".join(f"face {f} (t={t:.3g}, mean edge {e:.3g})" for f, t, e in under)
        return warned("thin_walls", name, f"fewer than {n} elements across thickness on: {faces}",
                      ratio, float(n), "Raise size.thin_wall_layers, add a `refine` entry with "
                      f"size <= thickness/{n} on those faces, or use element.order: 2.")
    msg = f"min thickness/edge ratio {ratio:.2f} (need >= {n})"
    msg += f"; unmeasured faces {unmeasured}" if unmeasured else ""
    return passed("thin_walls", name, msg, ratio, float(n))


def _non_ortho_gate(c: _GateInputs) -> Gate:
    mx, limit = c.rep.non_ortho.max if c.rep.non_ortho else 0.0, c.q.max_non_ortho
    make = failed if mx > limit else (warned if mx > NON_ORTHO_WARN else passed)
    return make(
        "non_ortho", "Non-orthogonality",
        f"max non-orthogonality {mx:.1f} deg (fail > {limit}, warn > {NON_ORTHO_WARN}); "
        f"mean {c.non_ortho_mean:.1f}", mx, float(limit),
        "" if make is passed else "Reduce non-orthogonality: lower size.global, add `refine` "
        "near the offending region, use algorithm.mesh3d: 1 / element.optimize: netgen, or "
        "adjust quality.max_non_ortho.")


def _skewness_gate(c: _GateInputs) -> Gate:
    mx = c.rep.skewness.max if c.rep.skewness else 0.0
    return threshold_gate(
        "skewness", "Skewness", mx, c.q.max_skewness,
        message_fmt="max skewness {value:.2f} (threshold {limit}); mean {mean:.2f}",
        mean=c.skewness_mean,
        hint="Reduce skewness: lower size.global, add `refine`, use element.optimize: netgen, "
             "or adjust quality.max_skewness.")


_GATES = (_inverted_gate, _sicn_gate, _gamma_gate, _volume_gate, _area_gate,
          _connectivity_gate, _thin_wall_gate)
_CFD_GATES = (_non_ortho_gate, _skewness_gate)
