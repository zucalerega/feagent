"""gmsh .msh -> scikit-fem mesh + named physical groups (private helper of the skfem backend).

Handles linear and quadratic simplices (tri3 / tri6 / tet4 / tet10), gmsh 4.1 (``cell_sets``)
and 2.2 (``gmsh:physical`` tags), drops the z = 0 coordinate of 2D meshes, removes nodes not
used by any domain cell, and maps every physical group to skfem entities by dimension:

  mesh dim      -> cells (element indices)
  mesh dim - 1  -> facets (edges in 2D, faces in 3D), found from their corner vertices
  1 (3D only)   -> edges
  0             -> vertices

Geometry order: a quadratic file with analysis order 2 gives an isoparametric (curved)
MeshTri2 / MeshTet2. Analysis order 1 always runs on the straight-sided corner mesh
(MeshTri1 / MeshTet1), i.e. P1 on a tri6 / tet10 file ignores the midside nodes.

Curved meshes use ``RobustIsoparametric``: skfem's inverse map (used for every facet quadrature
point by ``FacetBasis``) starts Newton at (0.5, 0.5, 0.5), outside the reference simplex, clips to
the unit cube and requires every point at once to converge to 1e-12; on valid but strongly curved
cells it gives up, so the whole mesh was rejected. This one starts from the exact inverse of the
straight-sided cell, never clips (Newton may wander far before converging) and accepts a residual
relative to the cell size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import meshio
import numpy as np
from skfem import MeshTet1, MeshTet2, MeshTri1, MeshTri2
from skfem.mapping import MappingIsoparametric

DOMAIN_TYPES = {2: ("triangle", "triangle6"), 3: ("tetra", "tetra10")}
FACET_TYPES = {2: ("line", "line3"), 3: ("triangle", "triangle6")}
EDGE_TYPES = ("line", "line3")
UNSUPPORTED_DOMAIN = {"quad", "quad8", "quad9", "hexahedron", "hexahedron20", "hexahedron27",
                      "wedge", "wedge15", "pyramid", "pyramid13", "tetra20", "triangle10"}


class MeshError(ValueError):
    """Unreadable / unsupported mesh or unknown physical group."""


class RobustIsoparametric(MappingIsoparametric):
    """Isoparametric mapping whose inverse (``invF``) works on strongly curved simplices."""

    RESIDUAL_ACCEPT = 1e-8      # |x - F(X)| / cell size at which a point counts as located

    def invF(self, x, tind=None, newton_max_iters=100, newton_tol=1e-12):
        mesh, dim = self.mesh, self.dim
        t = mesh.t if tind is None else mesh.t[:, tind]
        V = mesh.p[:, t[: dim + 1]]                                   # (dim, dim+1, n) corners
        A = np.transpose(V[:, 1:] - V[:, :1], (2, 0, 1))              # (n, dim, dim)
        h = np.linalg.norm(A, axis=(1, 2))[:, None]                   # cell size, (n, 1)
        X = np.einsum("nij,jnq->inq", np.linalg.inv(A), x - V[:, 0][:, :, None])   # straight-cell inverse
        for _ in range(newton_max_iters):
            r = x - self.F(X, tind)
            if (np.linalg.norm(r, axis=0) / h).max() < newton_tol:
                return X
            dX = np.einsum("ijkl,jkl->ikl", self.invDF(X, tind), r)
            # no clipping: on strongly curved cells Newton makes a large excursion before converging
            # quadratically (seen: residual 2.6 -> 32 -> ... -> 2e-13 cell sizes), and clipping stalls it
            X = X + dX
            if not np.all(np.isfinite(X)):
                break
            if np.abs(dX).max() < newton_tol:
                return X
        rel = np.linalg.norm(x - self.F(X, tind), axis=0) / h
        if rel.max() < self.RESIDUAL_ACCEPT:
            return X
        bad = np.unique(np.nonzero(rel >= self.RESIDUAL_ACCEPT)[0])
        raise MeshError(f"inverse isoparametric map failed in {len(bad)} cells (max residual {rel.max():.1e} of "
                        f"the cell size): folded or degenerate curved cells")


class _Robust:
    def _mapping(self):
        if not hasattr(self, "_cached_mapping"):
            self._cached_mapping = RobustIsoparametric(self, self.elem(), self.bndelem)
        return self._cached_mapping


class CurvedTri(_Robust, MeshTri2):
    """MeshTri2 with ``RobustIsoparametric``."""


class CurvedTet(_Robust, MeshTet2):
    """MeshTet2 with ``RobustIsoparametric``."""


@dataclass
class Group:
    name: str
    gdim: int           # dimension of the gmsh entities in the group
    kind: str           # "cells" | "facets" | "edges" | "vertices"
    index: np.ndarray   # skfem element / facet / edge / vertex indices


@dataclass
class MeshData:
    mesh: object                    # skfem mesh used for the analysis
    dim: int
    cell_type: str                  # domain cell type in the file, e.g. "triangle6"
    file_order: int                 # 1 or 2
    geom_order: int                 # geometry order actually used (1 straight, 2 curved)
    n_file_nodes: int               # nodes used by domain cells in the file
    groups: dict = field(default_factory=dict)
    bbox: np.ndarray | None = None  # (2, dim) min / max
    warnings: list = field(default_factory=list)

    @property
    def bbox_diag(self) -> float:
        return float(np.linalg.norm(self.bbox[1] - self.bbox[0]))

    def group(self, name: str) -> Group:
        if name not in self.groups:
            raise MeshError(f"unknown physical group {name!r}; available groups: {self.describe_groups()}")
        return self.groups[name]

    def describe_groups(self) -> str:
        if not self.groups:
            return "(none: the mesh has no named physical groups)"
        return ", ".join(f"{g.name} ({g.kind}, {len(g.index)})" for g in self.groups.values())


def _row_keys(a: np.ndarray, nmax: int | None = None) -> np.ndarray:
    """One sortable key per row of ``a`` (n, k), rows compared as sets: packed int64 when the
    indices fit (fast integer sorts), else a void view (slow but general)."""
    a = np.sort(np.asarray(a, dtype=np.int64), axis=1)
    k = a.shape[1]
    nmax = int(a.max()) + 1 if nmax is None and a.size else (nmax or 1)
    bits = max(1, int(nmax - 1).bit_length())
    if bits * k <= 62:
        key = np.zeros(len(a), dtype=np.int64)
        for j in range(k):
            key = (key << bits) | a[:, j]
        return key
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * k))).ravel()


def _lookup(table: np.ndarray, query: np.ndarray, what: str, name: str) -> np.ndarray:
    """Indices of rows ``query`` (n, k) in ``table`` (m, k), rows compared as sorted sets."""
    nmax = int(max(table.max(initial=0), query.max(initial=0))) + 1
    tk = _row_keys(table, nmax)
    order = np.argsort(tk, kind="stable")
    tk_sorted = tk[order]
    qk = _row_keys(query, nmax)
    pos = np.searchsorted(tk_sorted, qk)
    pos = np.clip(pos, 0, len(tk_sorted) - 1)
    ok = tk_sorted[pos] == qk
    if not np.all(ok):
        raise MeshError(f"group {name!r}: {int((~ok).sum())} of {len(ok)} {what} are not {what} of the "
                        f"domain mesh (non-conforming or unrelated elements)")
    return order[pos]


def _domain_type(m: meshio.Mesh, dim: int) -> str:
    types = {c.type for c in m.cells}
    bad = types & UNSUPPORTED_DOMAIN
    if bad:
        raise MeshError(f"unsupported cell types {sorted(bad)}: only tri3/tri6/tet4/tet10 meshes are supported")
    present = [t for t in DOMAIN_TYPES[dim] if t in types]
    if not present:
        other = 3 if dim == 2 else 2
        if any(t in types for t in DOMAIN_TYPES[other]):
            raise MeshError(f"the analysis needs a {dim}D mesh but the file contains {other}D cells "
                            f"({sorted(types)}); check analysis.formulation")
        raise MeshError(f"no {dim}D simplex cells in the mesh (cell types: {sorted(types)})")
    if len(present) > 1:
        raise MeshError(f"mixed linear and quadratic domain cells {present} are not supported")
    return present[0]


def _sets_from_physical(m: meshio.Mesh) -> dict:
    """gmsh 2.2 fallback: {name: {cell_type: indices}} from the gmsh:physical tags."""
    out: dict = {}
    phys = m.cell_data.get("gmsh:physical")
    if phys is None:
        return out
    tag_name = {(int(v[0]), int(v[1])): k for k, v in m.field_data.items()}
    gdim_of = {"vertex": 0, "line": 1, "line3": 1, "triangle": 2, "triangle6": 2, "tetra": 3, "tetra10": 3}
    offsets: dict = {}
    for block, tags in zip(m.cells, phys):
        off = offsets.get(block.type, 0)
        offsets[block.type] = off + len(block.data)
        gd = gdim_of.get(block.type)
        if gd is None:
            continue
        for tg in np.unique(tags):
            name = tag_name.get((int(tg), gd))
            if name is None:
                continue
            idx = off + np.nonzero(tags == tg)[0]
            d = out.setdefault(name, {})
            d[block.type] = np.concatenate([d.get(block.type, np.zeros(0, int)), idx])
    return out


def read_mesh(path, dim: int, order: int) -> MeshData:
    """Read a gmsh mesh for an analysis of mesh dimension ``dim`` and FE order ``order``."""
    try:
        m = meshio.read(str(path))
    except Exception as e:  # meshio raises many types
        raise MeshError(f"cannot read mesh {str(path)!r}: {e}") from e
    ctype = _domain_type(m, dim)
    file_order = 2 if ctype in ("triangle6", "tetra10") else 1
    nv = dim + 1
    cells_dict = m.cells_dict
    cells = np.asarray(cells_dict[ctype], dtype=np.int64)
    warnings: list[str] = []

    # gmsh 2.2 writes an element once per physical group: drop duplicates (4.1 never has them)
    dup_map = None
    first = np.arange(len(cells))
    if not m.cell_sets:
        keys = _row_keys(cells[:, :nv])
        _, first = np.unique(keys, return_index=True)
    if len(first) != len(cells):
        first = np.sort(first)
        warnings.append(f"mesh: {len(cells) - len(first)} duplicate domain cells removed")
        _, inv = np.unique(keys, return_inverse=True)
        uniq_first = np.full(inv.max() + 1, -1, dtype=np.int64)
        uniq_first[inv[first]] = np.arange(len(first))
        dup_map = uniq_first[inv]          # old cell index -> new cell index
        cells = cells[first]

    pts = np.asarray(m.points, dtype=float)
    if dim == 2:
        if pts.shape[1] > 2:
            zspan = float(np.ptp(pts[:, 2])) if len(pts) else 0.0
            xyspan = float(np.ptp(pts[:, :2], axis=0).max()) if len(pts) else 1.0
            if zspan > 1e-9 * max(xyspan, 1e-300):
                raise MeshError("2D analysis but the mesh is not planar in z (2D meshes must lie in the xy-plane)")
        pts = pts[:, :2]
    use_quadratic = file_order == 2 and order == 2
    conn = cells if use_quadratic else cells[:, :nv]
    used = np.unique(conn)
    renum = np.full(len(pts), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    p = np.ascontiguousarray(pts[used].T)
    t = np.ascontiguousarray(renum[conn].T)
    n_file_nodes = int(len(np.unique(cells)))

    if use_quadratic:
        mesh = (CurvedTri if dim == 2 else CurvedTet)(p, t)
        # skfem renumbers vertices as np.unique(t[:nv]) and puts the midside nodes after them
        uniq = np.unique(t[:nv])
        vmap_c = np.full(p.shape[1], -1, dtype=np.int64)
        vmap_c[uniq] = np.arange(len(uniq))
        geom_order = 2
    else:
        mesh = (MeshTri1 if dim == 2 else MeshTet1)(p, t)
        vmap_c = np.arange(p.shape[1], dtype=np.int64)
        geom_order = 1
    if file_order == 2 and order == 1:
        warnings.append("info: order 1 on a quadratic mesh: midside nodes ignored (straight-sided P1 cells)")
    # file point index -> skfem vertex index (-1: not a vertex of the analysis mesh)
    vmap = np.full(len(pts), -1, dtype=np.int64)
    vmap[used] = vmap_c
    # sanity check of the vertex mapping
    corner = cells[:, :nv].ravel()
    if not np.allclose(mesh.p[:, vmap[corner]], pts[corner].T, atol=1e-12 * max(1.0, float(np.abs(pts).max()))):
        raise MeshError("internal error: vertex renumbering mismatch")  # pragma: no cover

    # ---- physical groups
    if m.cell_sets:
        sets = {k: v for k, v in m.cell_sets_dict.items() if not k.startswith("gmsh:")}
    else:
        sets = _sets_from_physical(m)
    gdims = {k: int(v[1]) for k, v in m.field_data.items()} if m.field_data else {}
    groups: dict[str, Group] = {}
    for name, by_type in sets.items():
        gd = gdims.get(name)
        if gd is None:  # infer from cell types
            tps = set(by_type)
            gd = 0 if "vertex" in tps else 1 if tps & {"line", "line3"} else 2 if tps & {"triangle", "triangle6"} else 3
        if gd == dim:
            idx = np.concatenate([np.asarray(by_type[t_], dtype=np.int64) for t_ in by_type if t_ == ctype] or [np.zeros(0, int)])
            if dup_map is not None:
                idx = dup_map[idx]
            groups[name] = Group(name, gd, "cells", np.unique(idx))
        elif gd == dim - 1:
            rows = [np.asarray(cells_dict[t_], dtype=np.int64)[np.asarray(i_, dtype=np.int64)][:, :dim]
                    for t_, i_ in by_type.items() if t_ in FACET_TYPES[dim]]
            if not rows:
                continue
            q = vmap[np.concatenate(rows)]
            if np.any(q < 0):
                raise MeshError(f"group {name!r} uses nodes that belong to no domain cell")
            groups[name] = Group(name, gd, "facets", np.unique(_lookup(mesh.facets.T, q, "facets", name)))
        elif gd == 1 and dim == 3:
            rows = [np.asarray(cells_dict[t_], dtype=np.int64)[np.asarray(i_, dtype=np.int64)][:, :2]
                    for t_, i_ in by_type.items() if t_ in EDGE_TYPES]
            if not rows:
                continue
            q = vmap[np.concatenate(rows)]
            if np.any(q < 0):
                raise MeshError(f"group {name!r} uses nodes that belong to no domain cell")
            groups[name] = Group(name, gd, "edges", np.unique(_lookup(mesh.edges.T, q, "edges", name)))
        elif gd == 0:
            rows = [np.asarray(cells_dict[t_], dtype=np.int64)[np.asarray(i_, dtype=np.int64)][:, 0]
                    for t_, i_ in by_type.items() if t_ == "vertex"]
            if not rows:
                continue
            v = vmap[np.concatenate(rows)]
            if np.any(v < 0):
                raise MeshError(f"point group {name!r} is not a vertex of the domain mesh")
            groups[name] = Group(name, gd, "vertices", np.unique(v))
    # empty physical groups (named but without elements) are simply absent
    for name in gdims:
        if name not in groups and not name.startswith("gmsh:"):
            warnings.append(f"mesh: physical group {name!r} has no elements of a usable type")

    bbox = np.array([mesh.p.min(axis=1), mesh.p.max(axis=1)])
    return MeshData(mesh=mesh, dim=dim, cell_type=ctype, file_order=file_order, geom_order=geom_order,
                    n_file_nodes=n_file_nodes, groups=groups, bbox=bbox, warnings=warnings)
