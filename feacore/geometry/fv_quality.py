"""Finite-volume quality of a gmsh mesh, with OpenFOAM's formulas (what ``checkMesh`` reports).

A flow mesh is judged by how well its faces suit the finite-volume discretisation, not by the
element-shape measures a finite-element mesh is judged by:

* **non-orthogonality** of an internal face: the angle between its area vector and the line joining
  the two cell centres (``primitiveMeshTools::faceOrthogonality``). Above about 70 degrees the
  diffusion term needs non-orthogonal correction and accuracy drops (``checkMesh`` flags it).
* **skewness** of a face: how far the centre-to-centre line misses the face centre, relative to the
  face size (``primitiveMeshTools::faceSkewness``; boundary faces use the wall-normal line).
  ``checkMesh`` calls faces above 4 "highly skew".
* **aspect ratio** of a cell: longest over shortest distance from the cell centre to its face
  centres (1 for a square or a regular simplex, a / b for an a x b rectangle). Large values are
  fine in wall boundary layers that follow the flow, so this is reported, not gated.
* **volume ratio** across an internal face: the larger over the smaller of its two cell volumes
  (abrupt size jumps lose accuracy). Reported, not gated.
* **wall distance** per boundary group: the wall-normal distance from each boundary face's cell
  centre to the face, i.e. half the first cell's height for a wall-aligned cell. This is the
  distance the solver's wall gradient uses.

2D meshes (cells in the xy-plane) are extruded one layer, as the OpenFOAM backend does, so face
centres, face area vectors, cell centres and volumes are OpenFOAM's own (``FvMesh``); the
extruded front and back faces are left out of every statistic.

Use ``from_gmsh(dim)`` inside a gmsh session after meshing, or ``from_msh(path, dim)`` for a file.
"""

from __future__ import annotations

import numpy as np

# Thresholds checkMesh uses (OpenFOAM defaults): non-orthogonality above this many degrees is
# "severe"; faces above this skewness are "highly skew".
NON_ORTHO_MAX = 70.0
SKEW_MAX = 4.0

# gmsh element type -> (name, number of corner nodes); higher-order types are linearised
GMSH_CELLS = {2: ("triangle", 3), 3: ("quad", 4), 4: ("tetra", 4), 5: ("hexahedron", 8), 6: ("wedge", 6),
              7: ("pyramid", 5), 9: ("triangle", 3), 10: ("quad", 4), 16: ("quad", 4), 11: ("tetra", 4),
              12: ("hexahedron", 8), 17: ("hexahedron", 8), 13: ("wedge", 6), 18: ("wedge", 6),
              14: ("pyramid", 5), 19: ("pyramid", 5)}
GMSH_FACETS = {1: 2, 8: 2, 2: 3, 9: 3, 3: 4, 10: 4, 16: 4}   # boundary facet types -> corner nodes

# Faces of 3D cells in gmsh node order, each listed counter-clockwise seen from outside the cell
# (gmsh's positive orientation), so a face's area vector points out of its owner cell.
CELL_FACES = {
    "tetra": [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    "hexahedron": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}


def _keys(rows: np.ndarray, nmax: int) -> np.ndarray:
    """Set key per face row (n, 4); -1 pads triangles."""
    from feacore.solver._skfem_mesh import _row_keys
    return _row_keys(rows + 1, nmax + 2)


def _extrude(points: np.ndarray, cells: dict) -> tuple[np.ndarray, dict, int]:
    """2D cells (counter-clockwise or not) -> one layer of wedges / hexes of thickness 1 (z in [0, 1]).
    Returns (points, 3D cells by type, number of 2D points)."""
    n = len(points)
    P = np.vstack([np.column_stack([points[:, :2], np.zeros(n)]), np.column_stack([points[:, :2], np.ones(n)])])
    out: dict = {}
    for name, rows in cells.items():
        rows = np.asarray(rows, dtype=np.int64).copy()
        X = points[rows, :2]
        k = rows.shape[1]
        area = 0.5 * sum(X[:, i, 0] * X[:, (i + 1) % k, 1] - X[:, (i + 1) % k, 0] * X[:, i, 1] for i in range(k))
        rows[area < 0] = rows[area < 0][:, ::-1]               # counter-clockwise: positive volumes
        kind = "wedge" if name == "triangle" else "hexahedron"
        out.setdefault(kind, []).append(np.column_stack([rows, rows + n]))
    return P, {k: np.vstack(v) for k, v in out.items()}, n


# Node permutation that mirrors a cell (turns an inverted node order into a positive one).
MIRROR = {"tetra": (0, 2, 1, 3), "hexahedron": (0, 3, 2, 1, 4, 7, 6, 5), "wedge": (0, 2, 1, 3, 5, 4),
          "pyramid": (0, 3, 2, 1, 4)}


def _signed_volumes(points: np.ndarray, name: str, conn: np.ndarray) -> np.ndarray:
    """Volume of each cell from its outward-listed faces (divergence theorem, triangle fans)."""
    vol = np.zeros(len(conn))
    for fv in CELL_FACES[name]:
        X = points[conn[:, fv]]
        xm = X.mean(axis=1)
        for j in range(len(fv)):
            a, b = X[:, j], X[:, (j + 1) % len(fv)]
            vol += np.einsum("ij,ij->i", np.cross(a - xm, b - xm), xm) / 6.0
    return vol


def _orient(points: np.ndarray, cells: dict) -> dict:
    """Mirror cells whose node order makes them inside out (gmshToFoam re-orients them too)."""
    out = {}
    for name, conn in cells.items():
        conn = np.asarray(conn, dtype=np.int64).copy()
        neg = _signed_volumes(points, name, conn) < 0
        if neg.any():
            conn[neg] = conn[neg][:, list(MIRROR[name])]
        out[name] = conn
    return out


def _assemble(points: np.ndarray, cells: dict):
    """Faces (list of node tuples, oriented out of the owner), owner, neighbour (-1 for boundary
    faces), and the face key per face, for 3D cells by type."""
    rows, owner, cell0 = [], [], 0
    for name, conn in cells.items():
        conn = np.asarray(conn, dtype=np.int64)
        for fv in CELL_FACES[name]:
            r = conn[:, fv]
            if len(fv) == 3:
                r = np.column_stack([r, np.full(len(r), -1)])
            rows.append(r)
            owner.append(cell0 + np.arange(len(conn)))
        cell0 += len(conn)
    F = np.vstack(rows)
    own = np.concatenate(owner)
    key = _keys(F, len(points))
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.ones(len(ks), dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    grp = np.cumsum(first) - 1
    counts = np.bincount(grp)
    if np.any(counts > 2):
        raise ValueError(f"non-manifold mesh: {int((counts > 2).sum())} faces shared by more than two cells")
    head = order[first]                                    # one occurrence per face: its owner
    nb = np.full(len(head), -1, dtype=np.int64)
    second = order[~first]                                 # the other occurrence of internal faces
    nb[grp[~first]] = own[second]
    faces = [tuple(int(v) for v in r if v >= 0) for r in F[head]]
    return faces, own[head], nb, key[head]


def _cells_geometry(points: np.ndarray, faces, owner: np.ndarray, nb: np.ndarray, n_cells: int):
    """OpenFOAM face centres / area vectors, cell centres / volumes (primitiveMesh)."""
    from feacore.solver._foam_io import face_geometry
    Cf, Sf = face_geometry(points, faces)
    inner = nb >= 0
    cnt = np.bincount(owner, minlength=n_cells) + np.bincount(nb[inner], minlength=n_cells)
    cest = np.zeros((n_cells, 3))
    for k in range(3):
        cest[:, k] = np.bincount(owner, Cf[:, k], n_cells) + np.bincount(nb[inner], Cf[inner, k], n_cells)
    cest /= np.maximum(cnt, 1)[:, None]
    pv_o = np.einsum("ij,ij->i", Sf, Cf - cest[owner])
    pc_o = 0.75 * Cf + 0.25 * cest[owner]
    pv_n = np.einsum("ij,ij->i", Sf[inner], cest[nb[inner]] - Cf[inner])
    pc_n = 0.75 * Cf[inner] + 0.25 * cest[nb[inner]]
    vol = np.bincount(owner, pv_o, n_cells) + np.bincount(nb[inner], pv_n, n_cells)
    C = np.zeros((n_cells, 3))
    for k in range(3):
        C[:, k] = np.bincount(owner, pv_o * pc_o[:, k], n_cells) + np.bincount(nb[inner], pv_n * pc_n[:, k], n_cells)
    ok = np.abs(vol) > 1e-300
    C[ok] /= vol[ok][:, None]
    C[~ok] = cest[~ok]
    return Cf, Sf, C, vol / 3.0


def _skewness(points, faces, Cf, Sf, C, owner, nb) -> np.ndarray:
    """primitiveMeshTools::faceSkewness (internal) and boundaryFaceSkewness, vectorised by face size."""
    inner = nb >= 0
    Cpf = Cf - C[owner]
    d = np.where(inner[:, None], C[np.maximum(nb, 0)] - C[owner], 0.0)
    nrm = Sf / (np.linalg.norm(Sf, axis=1)[:, None] + 1e-300)
    d_b = nrm * np.einsum("ij,ij->i", nrm, Cpf)[:, None]
    d = np.where(inner[:, None], d, d_b)
    sv = Cpf - (np.einsum("ij,ij->i", Sf, Cpf) / (np.einsum("ij,ij->i", Sf, d) + 1e-300))[:, None] * d
    svhat = sv / (np.linalg.norm(sv, axis=1)[:, None] + 1e-300)
    fd = np.where(inner, 0.2, 0.4) * np.linalg.norm(d, axis=1) + 1e-300
    by_len: dict = {}
    for i, f in enumerate(faces):
        by_len.setdefault(len(f), []).append(i)
    for k, idx in by_len.items():
        idx = np.asarray(idx)
        Fv = np.asarray([faces[i] for i in idx], dtype=np.int64)
        ext = np.abs(np.einsum("ij,ikj->ik", svhat[idx], points[Fv] - Cf[idx][:, None, :])).max(axis=1)
        fd[idx] = np.maximum(fd[idx], ext)
    return np.linalg.norm(sv, axis=1) / fd


def _stats(v: np.ndarray, digits: int = 4) -> dict:
    if not len(v):
        return {"max": None, "mean": None}
    return {"max": float(f"{float(v.max()):.{digits}g}"), "mean": float(f"{float(v.mean()):.{digits}g}")}


def quality(points: np.ndarray, cells: dict, dim: int, facets: dict | None = None) -> dict:
    """FV quality of a mesh given as arrays.

    points (n, 2|3); cells {type name: (m, k) corner-node rows} with names of GMSH_CELLS (2D:
    triangle / quad in the xy-plane; 3D: tetra / hexahedron / wedge / pyramid); facets (optional)
    {boundary group: (m, 2|3|4) node rows, -1 padding allowed} for the per-group wall distance.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items() if len(v)}
    if not cells:
        raise ValueError("no cells")
    used = np.unique(np.concatenate([c.ravel() for c in cells.values()]))
    renum = np.full(len(points), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    pts = points[used]
    cells = {k: renum[v] for k, v in cells.items()}
    n2d = None
    if dim == 2:
        P, cells3, n2d = _extrude(pts, cells)
    else:
        P, cells3 = pts, _orient(pts, cells)
    n_cells = sum(len(v) for v in cells3.values())
    faces, owner, nb, fkeys = _assemble(P, cells3)
    Cf, Sf, C, V = _cells_geometry(P, faces, owner, nb, n_cells)
    magS = np.linalg.norm(Sf, axis=1)
    # 2D: the extruded copies (all nodes at z = 0 or all at z = 1) are the empty front / back
    if n2d is not None:
        fv = [np.asarray(f) for f in faces]
        lateral = np.array([not (np.all(f < n2d) or np.all(f >= n2d)) for f in fv])
    else:
        lateral = np.ones(len(faces), dtype=bool)
    inner = (nb >= 0) & lateral
    bnd = (nb < 0) & lateral
    d = C[nb[inner]] - C[owner[inner]]
    cosang = np.einsum("ij,ij->i", d, Sf[inner]) / (np.linalg.norm(d, axis=1) * magS[inner] + 1e-300)
    non_ortho = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    skew = _skewness(P, faces, Cf, Sf, C, owner, nb)[inner | bnd]
    # aspect ratio: longest / shortest cell-centre-to-face-centre distance over a cell's lateral faces
    dist = np.linalg.norm(Cf - C[owner], axis=1)
    big = np.zeros(n_cells)
    small = np.full(n_cells, np.inf)
    np.maximum.at(big, owner[lateral], dist[lateral])
    np.minimum.at(small, owner[lateral], dist[lateral])
    dist_n = np.linalg.norm(Cf[inner] - C[nb[inner]], axis=1)
    np.maximum.at(big, nb[inner], dist_n)
    np.minimum.at(small, nb[inner], dist_n)
    aspect = big / np.maximum(small, 1e-300)
    vr = np.abs(V[owner[inner]]) / np.maximum(np.abs(V[nb[inner]]), 1e-300)
    vr = np.maximum(vr, 1.0 / np.maximum(vr, 1e-300))
    out = {
        "cells": int(n_cells),
        "negative_volumes": int((V <= 0).sum()),
        # mean as checkMesh reports it: the angle of the mean cosine
        "non_orthogonality": {"max": _stats(non_ortho)["max"],
                              "mean": float(f"{np.degrees(np.arccos(np.clip(cosang.mean(), -1, 1))):.4g}")
                              if len(cosang) else None},
        "severe_non_orthogonal_faces": int((non_ortho > NON_ORTHO_MAX).sum()),
        "skewness": _stats(skew),
        "highly_skew_faces": int((skew > SKEW_MAX).sum()),
        "aspect_ratio": _stats(aspect),
        "volume_ratio": {**_stats(vr), "p99": float(f"{float(np.percentile(vr, 99)):.4g}") if len(vr) else None},
    }
    if facets:
        wall = {}
        bidx = np.nonzero(bnd)[0]
        bkeys = fkeys[bidx]
        sort = np.argsort(bkeys)
        for name, rows in facets.items():
            rows = np.asarray(rows, dtype=np.int64)
            if not len(rows):
                continue
            rows = np.where(rows >= 0, renum[np.maximum(rows, 0)], -1)
            if np.any((rows < 0) & (np.asarray(facets[name]) >= 0)):
                continue                                   # facets on nodes of no cell (other dims)
            if dim == 2:                                   # boundary edges -> the extruded quads
                r2 = rows[:, :2]
                q = np.column_stack([r2[:, 0], r2[:, 1], r2[:, 1] + n2d, r2[:, 0] + n2d])
            else:
                q = rows if rows.shape[1] == 4 else np.column_stack([rows, np.full(len(rows), -1)])
            k = _keys(q, len(P))
            pos = np.searchsorted(bkeys[sort], k)
            pos = np.minimum(pos, len(sort) - 1)
            hit = bkeys[sort][pos] == k
            if not hit.any():
                continue
            fi = bidx[sort[pos[hit]]]
            nrm = Sf[fi] / magS[fi][:, None]
            wd = np.abs(np.einsum("ij,ij->i", nrm, C[owner[fi]] - Cf[fi]))
            tang = magS[fi] if dim == 2 else np.sqrt(magS[fi])        # face size along the wall
            wall[name] = {"wall_distance": float(f"{wd.mean():.4g}"), "wall_distance_max": float(f"{wd.max():.4g}"),
                          "face_size": float(f"{tang.mean():.4g}"), "faces": int(len(fi))}
        out["near_wall"] = wall
    return out


def centres_volumes(points: np.ndarray, cells: dict, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenFOAM's cell centres (n, 3) and volumes (n,) of the cells (2D: areas, centres at z = 0.5 of
    the unit-depth extrusion), in the order of `cells` (by type, then row)."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items() if len(v)}
    if dim == 2:
        P, cells3, _ = _extrude(points, cells)
    else:
        P, cells3 = points, _orient(points, cells)
    n_cells = sum(len(v) for v in cells3.values())
    faces, owner, nb, _ = _assemble(P, cells3)
    _, _, C, V = _cells_geometry(P, faces, owner, nb, n_cells)
    return C, np.abs(V)


def passes(q: dict) -> tuple[bool, str | None]:
    """The flow quality gate: no inverted cells, max non-orthogonality <= NON_ORTHO_MAX, max
    skewness <= SKEW_MAX. Returns (ok, reason)."""
    if q.get("negative_volumes"):
        return False, f"{q['negative_volumes']} cells have zero or negative volume"
    no = (q.get("non_orthogonality") or {}).get("max")
    if no is not None and no > NON_ORTHO_MAX:
        return False, f"max non-orthogonality {no:.1f} deg > {NON_ORTHO_MAX:.0f}"
    sk = (q.get("skewness") or {}).get("max")
    if sk is not None and sk > SKEW_MAX:
        return False, f"max skewness {sk:.2f} > {SKEW_MAX:.0f}"
    return True, None


def from_gmsh(dim: int, groups: bool = True) -> dict:
    """FV quality of the mesh in the current gmsh model (all cells of dimension dim; the boundary
    physical groups of dimension dim - 1 for the wall distances)."""
    import gmsh

    ntags, X, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    pos = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
    pos[ntags] = np.arange(len(ntags))
    X = np.asarray(X).reshape(-1, 3)
    cells: dict = {}
    for t in gmsh.model.mesh.getElementTypes(dim):
        if t not in GMSH_CELLS:
            raise ValueError(f"unsupported cell type {gmsh.model.mesh.getElementProperties(t)[0]}")
        name, k = GMSH_CELLS[t]
        _, conn = gmsh.model.mesh.getElementsByType(t)
        n_nodes = gmsh.model.mesh.getElementProperties(t)[3]
        rows = pos[np.asarray(conn, dtype=np.int64).reshape(-1, n_nodes)[:, :k]]
        cells.setdefault(name, []).append(rows)
    cells = {k: np.vstack(v) for k, v in cells.items()}
    facets = {}
    if groups:
        for d, t in gmsh.model.getPhysicalGroups(dim - 1):
            rows = []
            for e in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                types, _, conns = gmsh.model.mesh.getElements(d, int(e))
                for typ, conn in zip(types, conns):
                    if typ not in GMSH_FACETS:
                        continue
                    k = GMSH_FACETS[typ]
                    nn = gmsh.model.mesh.getElementProperties(typ)[3]
                    r = pos[np.asarray(conn, dtype=np.int64).reshape(-1, nn)[:, :k]]
                    if k < 4 and dim == 3:
                        r = np.column_stack([r, np.full(len(r), -1)])
                    rows.append(r if dim == 3 else r[:, :2])
            if rows:
                facets[gmsh.model.getPhysicalName(d, t)] = np.vstack(rows)
    return quality(X if dim == 3 else X[:, :2], cells, dim, facets)


def from_msh(path, dim: int) -> dict:
    """FV quality of a gmsh .msh file (named boundary groups give the wall distances)."""
    import meshio

    m = meshio.read(str(path))
    lin = {"triangle": ("triangle", 3), "triangle6": ("triangle", 3), "quad": ("quad", 4), "quad8": ("quad", 4),
           "quad9": ("quad", 4), "tetra": ("tetra", 4), "tetra10": ("tetra", 4), "hexahedron": ("hexahedron", 8),
           "hexahedron20": ("hexahedron", 8), "hexahedron27": ("hexahedron", 8), "wedge": ("wedge", 6),
           "wedge15": ("wedge", 6), "pyramid": ("pyramid", 5), "pyramid13": ("pyramid", 5)}
    want = {"triangle", "quad"} if dim == 2 else {"tetra", "hexahedron", "wedge", "pyramid"}
    cells: dict = {}
    for c in m.cells:
        if c.type in lin and lin[c.type][0] in want:
            name, k = lin[c.type]
            cells.setdefault(name, []).append(np.asarray(c.data)[:, :k])
    cells = {k: np.unique(np.vstack(v), axis=0) for k, v in cells.items()}
    facets = {}
    fwant = ("line", "line3") if dim == 2 else ("triangle", "triangle6", "quad", "quad8", "quad9")
    for name, by_type in (m.cell_sets_dict or {}).items():
        if name.startswith("gmsh:"):
            continue
        rows = []
        for t, idx in by_type.items():
            if t not in fwant or not len(idx):
                continue
            k = 2 if dim == 2 else (3 if t.startswith("triangle") else 4)
            r = np.asarray(m.cells_dict[t])[idx][:, :k]
            if dim == 3 and k == 3:
                r = np.column_stack([r, np.full(len(r), -1)])
            rows.append(r)
        if rows:
            facets[name] = np.vstack(rows)
    pts = m.points if dim == 3 else m.points[:, :2]
    return quality(pts, cells, dim, facets)
