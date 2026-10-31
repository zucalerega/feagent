"""Viewer data: boundary triangulation of a mesh + nodal field values, stored as compact .npz.

3D meshes (tet4 / tet10): the boundary faces of the volume mesh (faces used by exactly one
cell), oriented outward. 2D meshes (tri3 / tri6): every cell. Quadratic faces are split into
four linear display triangles through the mid-side nodes, so P2 fields keep their resolution.

``fields.npz`` keys
    points     float32 (n, 3)  display vertices (2D: z = 0)
    triangles  uint32  (m, 3)  display triangles (outward / +z oriented)
    edges      uint32  (k, 2)  element edges on the boundary (mesh wireframe; curved for P2)
    disp       float32 (n, 3)  displacement (elasticity only)
    f_<name>   float32 (n,)    nodal value of field <name> at each display vertex
    meta       str             JSON: dim, cell_type, fields, stats, counts
"""

from __future__ import annotations

import base64
import itertools
import json
from pathlib import Path

import numpy as np

from feacore.case import ELASTICITY_FIELDS, HEAT_FIELDS

# local corner indices of the faces of a tetrahedron, and the opposite corner of each
_TET_FACES = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
_TET_OPPOSITE = np.array([3, 2, 1, 0])
# meshio (VTK) ordering of mid-side nodes, used if geometric detection is inconclusive
_VTK_MID = {
    "triangle6": {(0, 1): 3, (1, 2): 4, (0, 2): 5},
    "tetra10": {(0, 1): 4, (1, 2): 5, (0, 2): 6, (0, 3): 7, (1, 3): 8, (2, 3): 9},
}


# --------------------------------------------------------------------------- mesh -> display triangulation


def _midside_map(points: np.ndarray, cells: np.ndarray, n_corners: int, cell_type: str) -> dict:
    """{(a, b): local index of the mid-side node on corner edge a-b}, detected geometrically
    on a sample of cells (robust to reader-specific node orderings)."""
    sample = cells[: min(len(cells), 200)]
    P = points[sample]
    out = {}
    for k in range(n_corners, cells.shape[1]):
        best, best_d = None, np.inf
        for a, b in itertools.combinations(range(n_corners), 2):
            d = np.median(np.linalg.norm(P[:, k] - 0.5 * (P[:, a] + P[:, b]), axis=1))
            if d < best_d:
                best, best_d = (a, b), d
        out[best] = k
    if len(out) != cells.shape[1] - n_corners:
        return dict(_VTK_MID[cell_type])
    return out


def _unique_rows(keys: np.ndarray, n_nodes: int):
    """np.unique over sorted integer triples, via a scalar key when it fits in int64."""
    if n_nodes < 2**20:
        k = (keys[:, 0].astype(np.int64) * n_nodes + keys[:, 1]) * n_nodes + keys[:, 2]
        return np.unique(k, return_index=True, return_counts=True)[1:]
    return np.unique(keys, axis=0, return_index=True, return_counts=True)[1:]


def mesh_triangulation(points: np.ndarray, cells: np.ndarray, cell_type: str) -> dict:
    """Display triangulation for one block of top-dimensional cells.

    Returns ``{"node_ids": (n,) mesh node ids of the display vertices, "triangles": (m, 3)
    indices into node_ids, "edges": (k, 2) indices into node_ids, "dim": 2|3}``."""
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    if cell_type in ("tetra", "tetra10"):
        dim, nc = 3, 4
    elif cell_type in ("triangle", "triangle6"):
        dim, nc = 2, 3
    else:
        raise ValueError(f"unsupported cell type {cell_type!r} (need triangle/triangle6/tetra/tetra10)")
    quadratic = cells.shape[1] > nc
    mid = _midside_map(points, cells, nc, cell_type) if quadratic else {}

    if dim == 3:
        F = cells[:, _TET_FACES]                      # (nc, 4, 3) corner node ids
        F = F.reshape(-1, 3)
        cell_of = np.repeat(np.arange(len(cells)), 4)
        face_of = np.tile(np.arange(4), len(cells))
        first, counts = _unique_rows(np.sort(F, axis=1), len(points))
        sel = first[counts == 1]
        F, cell_of, face_of = F[sel], cell_of[sel], face_of[sel]
        # orient outward: normal must point away from the opposite corner
        opp = cells[cell_of, _TET_OPPOSITE[face_of]]
        p0, p1, p2, po = points[F[:, 0]], points[F[:, 1]], points[F[:, 2]], points[opp]
        flip = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), po - p0) > 0
        F[flip] = F[flip][:, [0, 2, 1]]
        local = _TET_FACES[face_of]
        local[flip] = local[flip][:, [0, 2, 1]]
    else:
        F = cells[:, :3].copy()
        p0, p1, p2 = points[F[:, 0]], points[F[:, 1]], points[F[:, 2]]
        d1, d2 = p1 - p0, p2 - p0
        flip = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] < 0
        F[flip] = F[flip][:, [0, 2, 1]]
        cell_of = np.arange(len(cells))
        local = np.tile(np.arange(3), (len(cells), 1))
        local[flip] = local[flip][:, [0, 2, 1]]

    if quadratic:
        def midnode(la, lb):
            key = np.where((la < lb)[:, None], np.stack([la, lb], 1), np.stack([lb, la], 1))
            idx = np.empty(len(la), dtype=np.int64)
            for (a, b), k in mid.items():
                m = (key[:, 0] == a) & (key[:, 1] == b)
                idx[m] = k
            return cells[cell_of, idx]

        c0, c1, c2 = F[:, 0], F[:, 1], F[:, 2]
        m01 = midnode(local[:, 0], local[:, 1])
        m12 = midnode(local[:, 1], local[:, 2])
        m20 = midnode(local[:, 2], local[:, 0])
        tris = np.concatenate([np.stack([c0, m01, m20], 1), np.stack([m01, c1, m12], 1),
                               np.stack([m20, m12, c2], 1), np.stack([m01, m12, m20], 1)])
        segs = np.concatenate([np.stack([c0, m01], 1), np.stack([m01, c1], 1), np.stack([c1, m12], 1),
                               np.stack([m12, c2], 1), np.stack([c2, m20], 1), np.stack([m20, c0], 1)])
    else:
        tris = F
        segs = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    segs = np.unique(np.sort(segs, axis=1), axis=0)
    node_ids, inv = np.unique(np.concatenate([tris.ravel(), segs.ravel()]), return_inverse=True)
    nt = tris.size
    return {
        "node_ids": node_ids,
        "triangles": inv[:nt].reshape(-1, 3),
        "edges": inv[nt:].reshape(-1, 2),
        "dim": dim,
        "quadratic": quadratic,
    }


def read_mesh_triangulation(mesh_path: str | Path) -> dict:
    """Read a gmsh mesh (meshio) and build the display triangulation of its top-dim cells."""
    import meshio

    m = meshio.read(str(mesh_path))
    blocks = {}
    for c in m.cells:
        blocks.setdefault(c.type, []).append(c.data)
    for ct in ("tetra10", "tetra", "triangle6", "triangle"):
        if ct in blocks:
            cell_type = ct
            break
    else:
        raise ValueError(f"no triangle/tetra cells in {mesh_path} (found {sorted(blocks)})")
    cells = np.concatenate(blocks[cell_type])
    tri = mesh_triangulation(m.points, cells, cell_type)
    tri["points"] = np.asarray(m.points, dtype=float)[tri["node_ids"]]
    if tri["points"].shape[1] == 2:
        tri["points"] = np.column_stack([tri["points"], np.zeros(len(tri["points"]))])
    tri["cell_type"] = cell_type
    tri["n_cells"] = int(len(cells))
    return tri


# --------------------------------------------------------------------------- result -> nodal values at vertices


class _NodalSampler:
    """Values of result fields at given points: exact match to the solution-space nodes
    (KD-tree), ``probe`` for the rest (e.g. P1 solution on a quadratic mesh)."""

    def __init__(self, result, points: np.ndarray, dim: int):
        self.result = result
        self.points = points[:, :dim] if dim == 2 else points
        self.dim = dim
        span = np.ptp(points, axis=0).max() if len(points) else 1.0
        self.tol = 1e-7 * max(span, 1e-300)
        self._trees: list = []  # [(coords, nearest index, matched mask)]

    def _index(self, coords: np.ndarray):
        from scipy.spatial import cKDTree

        for c, idx, ok in self._trees:
            if c.shape == coords.shape and (c is coords or np.array_equal(c, coords)):
                return idx, ok
        coords = np.asarray(coords, dtype=float)
        q = self.points if coords.shape[1] == self.points.shape[1] else (
            self.points[:, : coords.shape[1]] if coords.shape[1] < self.points.shape[1]
            else np.column_stack([self.points, np.zeros((len(self.points), coords.shape[1] - self.points.shape[1]))]))
        dist, idx = cKDTree(coords).query(q, k=1)
        ok = dist <= self.tol
        self._trees.append((coords, idx, ok))
        return idx, ok

    def __call__(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """(values at the display vertices, all nodal values) for one field."""
        coords, vals = self.result.nodal(field)
        coords = np.asarray(coords, dtype=float)
        vals = np.asarray(vals, dtype=float).ravel()
        idx, ok = self._index(coords)
        out = np.full(len(self.points), np.nan)
        out[ok] = vals[idx[ok]]
        if not ok.all():
            try:
                out[~ok] = np.asarray(self.result.probe(field, self.points[~ok]), dtype=float).ravel()
            except Exception:  # noqa: BLE001 - leave NaN (shown grey) rather than fail the run
                pass
        return out, vals


def candidate_fields(physics: str, dim: int, formulation: str = "") -> list[str]:
    if physics == "heat":
        return [f for f in HEAT_FIELDS if not (dim == 2 and f == "qz")]
    skip = {"uz", "syz", "sxz"} if dim == 2 else {"stt"}
    if dim == 2 and formulation != "axisymmetric":
        skip.add("stt")
    if formulation == "plane_stress":
        skip.add("szz")
    return [f for f in ELASTICITY_FIELDS if f not in skip]


def extract_view(result, mesh_path: str | Path, fields: list[str] | None = None) -> dict:
    """Boundary triangulation of ``mesh_path`` + nodal values of ``result``'s fields."""
    tri = read_mesh_triangulation(mesh_path)
    info = result.info
    a = result.case.analysis
    dim = tri["dim"]
    sampler = _NodalSampler(result, tri["points"], dim)
    names = fields or candidate_fields(a.physics, dim, a.formulation)
    values, stats, errors = {}, {}, {}
    for f in names:
        try:
            v, allv = sampler(f)
        except Exception as e:  # noqa: BLE001 - a backend may not provide every field
            errors[f] = f"{type(e).__name__}: {e}"
            continue
        finite = allv[np.isfinite(allv)]
        if finite.size == 0:
            continue
        values[f] = v
        vb = v[np.isfinite(v)]
        stats[f] = {
            "min": float(finite.min()), "max": float(finite.max()),
            "boundary_min": float(vb.min()) if vb.size else None,
            "boundary_max": float(vb.max()) if vb.size else None,
            "n_nan": int(np.count_nonzero(~np.isfinite(v))),
        }
    # drop identically-zero secondary fields (e.g. szz in plane stress)
    primary = {"ux", "uy", "u_mag", "von_mises", "T", "q_mag"}
    for f in list(values):
        s = stats[f]
        if f not in primary and s["min"] == 0.0 and s["max"] == 0.0:
            values.pop(f)
            stats.pop(f)
    disp = None
    if a.physics == "elasticity":
        comps = []
        for f in ("ux", "uy", "uz"):
            if f in values:
                comps.append(values[f])
            elif f == "uz" and dim == 2:
                comps.append(np.zeros(len(tri["points"])))
            else:
                try:
                    comps.append(sampler(f)[0])
                except Exception:  # noqa: BLE001
                    comps = None
                    break
        if comps is not None:
            disp = np.nan_to_num(np.column_stack(comps))
    meta = {
        "dim": dim,
        "cell_type": tri["cell_type"],
        "quadratic": bool(tri["quadratic"]),
        "n_cells": tri["n_cells"],
        "n_points": int(len(tri["points"])),
        "n_triangles": int(len(tri["triangles"])),
        "n_edges": int(len(tri["edges"])),
        "physics": a.physics,
        "formulation": a.formulation,
        "fields": list(values),
        "stats": stats,
        "field_errors": errors,
        "has_displacement": disp is not None,
        "backend": info.backend,
    }
    return {"points": tri["points"], "triangles": tri["triangles"], "edges": tri["edges"],
            "values": values, "disp": disp, "meta": meta}


def save_view(view: dict, path: str | Path, relto: str | Path | None = None) -> dict:
    """Write fields.npz; return the summary kept in the run record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "points": np.asarray(view["points"], dtype=np.float32),
        "triangles": np.asarray(view["triangles"], dtype=np.uint32),
        "edges": np.asarray(view["edges"], dtype=np.uint32),
        "meta": np.asarray(json.dumps(view["meta"])),
    }
    if view.get("disp") is not None:
        arrays["disp"] = np.asarray(view["disp"], dtype=np.float32)
    for f, v in view["values"].items():
        arrays[f"f_{f}"] = np.asarray(v, dtype=np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    m = view["meta"]
    file = str(path.relative_to(relto)) if relto is not None and path.is_relative_to(relto) else str(path)
    return {"file": file, "available": m["fields"], "stats": m["stats"], "dim": m["dim"],
            "n_points": m["n_points"], "n_triangles": m["n_triangles"],
            "has_displacement": m["has_displacement"], "cell_type": m["cell_type"]}


# --------------------------------------------------------------------------- transport encoding


def encode_array(a: np.ndarray, dtype: str | None = None, encoding: str = "b64") -> dict:
    """Typed array for JSON transport: little-endian base64 (``b64``) or plain list."""
    a = np.ascontiguousarray(a, dtype=dtype or a.dtype)
    if encoding == "list":  # NaN -> null happens in the JSON response (runstore.jsonable)
        return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tolist()}
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "b64": base64.b64encode(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()).decode("ascii")}


def decode_array(d: dict) -> np.ndarray:
    if "b64" in d:
        return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.dtype(d["dtype"]).newbyteorder("<")).reshape(d["shape"])
    return np.asarray(d["data"], dtype=d["dtype"]).reshape(d["shape"])
