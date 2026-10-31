"""Mesh quality measurement and mesh gates.

Each gate cites its measured value and threshold and gives a hint naming the
spec key to change. ``QualitySpec`` is duck-typed (attributes only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .types import Gate, MeshResult, Vec3, _Base

if TYPE_CHECKING:  # pragma: no cover
    from .spec import QualitySpec

NON_ORTHO_WARN = 65.0
N_WORST = 10


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


# --- topology / OpenFOAM-style face metrics -------------------------------

_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))  # outward-oriented for positive tets


def _internal_faces(tets: np.ndarray):
    """Return (owner, neighbour, face_nodes[n,3]) for faces shared by exactly two tets."""
    n = len(tets)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int)
    faces = np.concatenate([tets[:, list(f)] for f in _TET_FACES])  # (4n,3)
    owner = np.tile(np.arange(n), 4)
    key = np.sort(faces, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    first = np.searchsorted(sorted_inv, np.arange(len(counts)))
    two = np.nonzero(counts == 2)[0]
    a, b = order[first[two]], order[first[two] + 1]
    return owner[a], owner[b], faces[a]


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
    p0, p1, p2 = pts[fn[:, 0]], pts[fn[:, 1]], pts[fn[:, 2]]
    s = 0.5 * np.cross(p1 - p0, p2 - p0)
    flip = np.einsum("ij,ij->i", s, d) < 0
    s[flip] *= -1
    xf = (p0 + p1 + p2) / 3.0
    sn = np.linalg.norm(s, axis=1)
    dn = np.linalg.norm(d, axis=1)
    ok = (sn > 0) & (dn > 0)
    sn = np.where(ok, sn, 1.0)
    dn = np.where(ok, dn, 1.0)
    n = s / sn[:, None]
    cosang = np.clip(np.einsum("ij,ij->i", n, d) / dn, -1.0, 1.0)
    non_ortho = np.degrees(np.arccos(cosang))
    dd = np.einsum("ij,ij->i", d, n)
    dd = np.where(np.abs(dd) < 1e-300, 1e-300, dd)
    t = np.einsum("ij,ij->i", xf - cent[P], n) / dd
    xi = cent[P] + d * t[:, None]
    skew = np.linalg.norm(xf - xi, axis=1) / dn
    # degenerate faces are as bad as it gets
    non_ortho = np.where(ok, non_ortho, 90.0)
    skew = np.where(ok, skew, np.inf)
    return non_ortho, skew


def _components(tets: np.ndarray) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(tets)
    if n == 0:
        return 0
    P, N, _ = _internal_faces(tets)
    g = coo_matrix((np.ones(len(P)), (P, N)), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def _expected_components(geometry) -> int:
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

def _face_edge_lengths(face_tags: list[int], tets: np.ndarray, node_tag_to_idx: dict,
                       points: np.ndarray) -> dict[int, float | None]:
    """Mean surface-triangle edge length per CAD face (None if no triangles found)."""
    import gmsh

    out: dict[int, float | None] = {}
    # boundary triangles of the tet mesh, as fallback when the msh stores no face elements
    bfaces = None
    for ft in face_tags:
        tris = []
        types, _, enodes = gmsh.model.mesh.getElements(2, ft)
        for t, en in zip(types, enodes):
            if t in (2, 9):
                nn = gmsh.model.mesh.getElementProperties(t)[3]
                tris.append(np.array([[node_tag_to_idx[int(x)] for x in row[:3]]
                                      for row in np.asarray(en).reshape(-1, nn)]))
        tri = np.concatenate(tris) if tris else np.zeros((0, 3), int)
        if len(tri) == 0 and len(tets):
            if bfaces is None:
                allf = np.concatenate([tets[:, list(f)] for f in _TET_FACES])
                key = np.sort(allf, axis=1)
                _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
                bfaces = allf[cnt[inv.ravel()] == 1]
            allowed = {(2, ft)}
            for dim, tag in gmsh.model.getBoundary([(2, ft)], combined=False, oriented=False,
                                                   recursive=True):
                allowed.add((dim, tag))
            idx_to_tag = {v: k for k, v in node_tag_to_idx.items()}
            cls: dict[int, tuple[int, int]] = {}
            keep = []
            for row in bfaces:
                ok = True
                for i in row:
                    i = int(i)
                    if i not in cls:
                        _, _, dim, tag = gmsh.model.mesh.getNode(idx_to_tag[i])
                        cls[i] = (dim, tag)
                    if cls[i] not in allowed:
                        ok = False
                        break
                if ok:
                    keep.append(row)
            tri = np.array(keep, int).reshape(-1, 3)
        if len(tri) == 0:
            out[ft] = None
            continue
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        out[ft] = float(np.linalg.norm(points[e[:, 0]] - points[e[:, 1]], axis=1).mean())
    return out


def _g(id_, name, status, value, threshold, message, hint=""):
    return Gate(id=id_, name=name, status=status, value=value, threshold=threshold,
                message=message, hint=hint)


def assess_mesh(mesh_path: str | Path, result: MeshResult, q: "QualitySpec",
                analysis: str) -> QualityReport:
    import gmsh

    from .gmsh_session import gmsh_session
    from .meshdata import read_msh

    md = read_msh(mesh_path)
    tets, tags = md.tets, md.tet_tags
    n_tets = len(tets)

    with gmsh_session():
        gmsh.open(str(mesh_path))
        ntags, _, _ = gmsh.model.mesh.getNodes()
        node_tag_to_idx = {int(t): i for i, t in enumerate(ntags)}
        tl = [int(t) for t in tags]
        if n_tets:
            sicn = np.asarray(gmsh.model.mesh.getElementQualities(tl, "minSICN"), float)
            gamma = np.asarray(gmsh.model.mesh.getElementQualities(tl, "gamma"), float)
            sj = np.asarray(gmsh.model.mesh.getElementQualities(tl, "minSJ"), float)
            vol = np.asarray(gmsh.model.mesh.getElementQualities(tl, "volume"), float)
        else:
            sicn = gamma = sj = vol = np.zeros(0)

        worst: list[WorstElement] = []
        for i in np.argsort(sicn, kind="stable")[:N_WORST]:
            _, nodes, _, _ = gmsh.model.mesh.getElement(tl[i])
            fs = set()
            for nt in nodes:
                _, _, dim, tag = gmsh.model.mesh.getNode(int(nt))
                if dim == 2:
                    fs.add(int(tag))
            c = md.points[tets[i]].mean(axis=0)
            worst.append(WorstElement(tag=int(tags[i]), sicn=float(sicn[i]), gamma=float(gamma[i]),
                                      centroid=(float(c[0]), float(c[1]), float(c[2])),
                                      faces=sorted(fs)))

        geo = result.geometry
        hz_faces: dict[int, float] = {}
        if geo is not None:
            for h in geo.hazards:
                if h.kind == "thin_wall" and h.value:
                    for dim, tag in h.entities:
                        if dim == 2:
                            hz_faces[int(tag)] = float(h.value)
        edge_len = _face_edge_lengths(sorted(hz_faces), tets, node_tag_to_idx, md.points) \
            if hz_faces else {}

    inverted = int(np.sum(sj <= 0)) if n_tets else 0
    volume_mesh = float(vol.sum())
    volume_cad = float(sum(v.volume for v in geo.volumes)) if geo else 0.0
    volume_error = abs(volume_mesh - volume_cad) / volume_cad if volume_cad > 0 else 0.0

    group_area: dict[str, tuple[float, float, float]] = {}
    pts = md.points
    for name, gr in result.groups.items():
        if gr.dim != 2 or name not in md.tris or geo is None:
            continue
        tri = md.tris[name]
        if len(tri):
            a = 0.5 * np.linalg.norm(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                                              pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1).sum()
        else:
            a = 0.0
        cad = 0.0
        for t in gr.tags:
            try:
                cad += geo.face(t).area
            except KeyError:
                pass
        group_area[name] = (float(cad), float(a), float(abs(a - cad) / cad) if cad > 0 else 0.0)

    components = _components(tets)
    expected = _expected_components(geo)

    no_stats = sk_stats = None
    no_arr = sk_arr = np.zeros(0)
    if analysis == "cfd":
        no_arr, sk_arr = tet_non_ortho_skewness(md.points, tets)
        no_stats, sk_stats = _stats(no_arr), _stats(sk_arr)

    s_sicn, s_gamma, s_sj = _stats(sicn), _stats(gamma), _stats(sj)
    gates: list[Gate] = []

    gates.append(_g(
        "inverted", "No inverted elements", "fail" if inverted > 0 else "pass",
        float(inverted), 0.0,
        f"{inverted} of {n_tets} tets have non-positive scaled Jacobian (min SJ {s_sj.min:.3g})",
        "Inverted elements: lower size.global, try algorithm_3d 1 (Delaunay) or optimize: netgen; "
        "check the worst elements' faces for CAD defects (heal: true)." if inverted else ""))

    gates.append(_g(
        "sicn", "Minimum SICN", "fail" if s_sicn.min < q.min_sicn else "pass",
        s_sicn.min, float(q.min_sicn),
        f"min SICN {s_sicn.min:.3f} (threshold {q.min_sicn}); p01 {s_sicn.p01:.3f}, "
        f"median {s_sicn.median:.3f}",
        f"Worst elements touch faces {worst[0].faces if worst else []}: add a `refine` entry "
        "there, lower size.global, enable optimize: netgen, or relax quality.min_sicn."
        if s_sicn.min < q.min_sicn else ""))

    gates.append(_g(
        "gamma", "Minimum gamma", "warn" if s_gamma.min < q.min_gamma else "pass",
        s_gamma.min, float(q.min_gamma),
        f"min gamma {s_gamma.min:.3f} (threshold {q.min_gamma}); median {s_gamma.median:.3f}",
        "Low inscribed/circumscribed radius ratio (slivers): lower size.global, set "
        "optimize: netgen, or adjust quality.min_gamma." if s_gamma.min < q.min_gamma else ""))

    ve_bad = volume_error > q.max_volume_error
    gates.append(_g(
        "volume_fidelity", "Volume fidelity", "fail" if ve_bad else "pass",
        volume_error, float(q.max_volume_error),
        f"mesh volume {volume_mesh:.6g} vs CAD {volume_cad:.6g}: relative error "
        f"{volume_error:.3g} (threshold {q.max_volume_error})",
        "Geometry is poorly resolved (curved boundaries): raise size.curvature, lower "
        "size.global, use order: 2, or relax quality.max_volume_error." if ve_bad else ""))

    if not group_area:
        gates.append(_g("area_fidelity", "Group area fidelity", "skip", None,
                        float(q.max_area_error), "no surface groups to compare", ""))
    else:
        wn, (wc, wm, we) = max(group_area.items(), key=lambda kv: kv[1][2])
        bad = we > q.max_area_error
        gates.append(_g(
            "area_fidelity", "Group area fidelity", "fail" if bad else "pass", we,
            float(q.max_area_error),
            f"worst group '{wn}': mesh area {wm:.6g} vs CAD {wc:.6g}, relative error {we:.3g} "
            f"(threshold {q.max_area_error})",
            f"Group '{wn}' is poorly resolved: add a `refine` entry for it, raise size.curvature, "
            "lower size.global, or relax quality.max_area_error." if bad else ""))

    gates.append(_g(
        "connectivity", "Mesh connectivity",
        "fail" if components != expected else "pass",
        float(components), float(expected),
        f"mesh has {components} face-connected component(s), geometry expects {expected}",
        "Mesh is split (or merged) relative to the CAD: check for tiny gaps or unfused bodies; "
        "set heal: true or lower size.global / size.min so thin necks are resolved."
        if components != expected else ""))

    if not hz_faces:
        gates.append(_g("thin_walls", "Thin-wall resolution", "skip", None,
                        float(q.thin_wall_min_elements), "no thin-wall hazards in geometry", ""))
    else:
        bad_f, unmeasured, worst_ratio = [], [], None
        for ft, th in sorted(hz_faces.items()):
            el = edge_len.get(ft)
            if el is None:
                unmeasured.append(ft)
                continue
            ratio = th / el  # elements across the wall (approx.)
            worst_ratio = ratio if worst_ratio is None else min(worst_ratio, ratio)
            if el > th / q.thin_wall_min_elements:
                bad_f.append((ft, th, el))
        if bad_f:
            lst = ", ".join(f"face {f} (t={t:.3g}, mean edge {e:.3g})" for f, t, e in bad_f)
            gates.append(_g(
                "thin_walls", "Thin-wall resolution", "warn", worst_ratio,
                float(q.thin_wall_min_elements),
                f"fewer than {q.thin_wall_min_elements} elements across thickness on: {lst}",
                "Raise size.thin_wall_layers, add a `refine` entry with size <= thickness/"
                f"{q.thin_wall_min_elements} on those faces, or use order: 2."))
        elif worst_ratio is None:
            gates.append(_g("thin_walls", "Thin-wall resolution", "skip", None,
                            float(q.thin_wall_min_elements),
                            f"no surface triangles found on thin-wall faces {unmeasured}", ""))
        else:
            msg = (f"min thickness/edge ratio {worst_ratio:.2f} "
                   f"(need >= {q.thin_wall_min_elements})")
            if unmeasured:
                msg += f"; unmeasured faces {unmeasured}"
            gates.append(_g("thin_walls", "Thin-wall resolution", "pass", worst_ratio,
                            float(q.thin_wall_min_elements), msg, ""))

    if analysis == "cfd":
        mx = no_stats.max if no_stats else 0.0
        st = "fail" if mx > q.max_non_ortho else ("warn" if mx > NON_ORTHO_WARN else "pass")
        gates.append(_g(
            "non_ortho", "Non-orthogonality", st, mx, float(q.max_non_ortho),
            f"max non-orthogonality {mx:.1f} deg (fail > {q.max_non_ortho}, warn > "
            f"{NON_ORTHO_WARN}); mean {float(no_arr.mean()) if no_arr.size else 0.0:.1f}",
            "Reduce non-orthogonality: lower size.global, add `refine` near the offending region, "
            "use algorithm_3d 1 / optimize: netgen, or adjust quality.max_non_ortho."
            if st != "pass" else ""))
        sx = sk_stats.max if sk_stats else 0.0
        bad = sx > q.max_skewness
        gates.append(_g(
            "skewness", "Skewness", "fail" if bad else "pass", sx, float(q.max_skewness),
            f"max skewness {sx:.2f} (threshold {q.max_skewness}); mean "
            f"{float(sk_arr.mean()) if sk_arr.size else 0.0:.2f}",
            "Reduce skewness: lower size.global, add `refine`, use optimize: netgen, or adjust "
            "quality.max_skewness." if bad else ""))

    return QualityReport(
        n_nodes=len(md.points), n_tets=n_tets, sicn=s_sicn, gamma=s_gamma, sj=s_sj,
        inverted=inverted, worst=worst, volume_cad=volume_cad, volume_mesh=volume_mesh,
        volume_error=float(volume_error), group_area=group_area, components=components,
        expected_components=expected, non_ortho=no_stats, skewness=sk_stats,
        sicn_per_tet=sicn, gates=gates)
