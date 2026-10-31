"""Mesh handling for the CalculiX backend: gmsh .msh -> CalculiX-ordered FE model.

* Reads gmsh v2.2 / v4.1 (ASCII or binary) with meshio; named physical groups of any dimension.
* Domain cells (tri3/tri6 in 2D, tet4/tet10 in 3D) are deduplicated (msh2 repeats a cell once per
  physical group), re-oriented to positive Jacobian (CalculiX rejects clockwise 2D elements) and
  converted to the requested order:
    order 1 on a quadratic mesh -> midside nodes dropped (straight-sided linear elements),
    order 2 on a linear mesh    -> midside nodes inserted at straight edge midpoints.
* Node ordering. meshio returns cells in VTK ordering. For the types used here it coincides with
  CalculiX's:
    tri6  : 0,1,2 corners; 3 = (0,1), 4 = (1,2), 5 = (2,0)        (gmsh order == CalculiX CPS6/CAX6)
    tet10 : 0-3 corners; 4 = (0,1), 5 = (1,2), 6 = (2,0), 7 = (0,3), 8 = (1,3), 9 = (2,3)
            (gmsh stores (2,3) before (1,3) in slots 8/9; meshio swaps them -> CalculiX C3D10 order)
  ``FEModel.check_midside`` verifies this geometrically (midside nodes near edge midpoints).
* Faces (CalculiX numbering, used by *DLOAD Pk / *DFLUX Sk / *FILM Fk):
    triangle  : face 1 = 1-2, face 2 = 2-3, face 3 = 3-1
    tetrahedron: face 1 = 1-2-3, face 2 = 1-4-2, face 3 = 2-4-3, face 4 = 3-4-1
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# local corner pairs of the edges, in CalculiX midside-node order
TRI_EDGES = np.array([(0, 1), (1, 2), (2, 0)])
TET_EDGES = np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)])
# faces: corners first, then midside nodes (quadratic) in tri6 / line3 order
TRI_FACES = np.array([(0, 1, 3), (1, 2, 4), (2, 0, 5)])           # line3: [a, b, mid]
TET_FACES = np.array([(0, 1, 2, 4, 5, 6), (0, 3, 1, 7, 8, 4),
                      (1, 3, 2, 8, 9, 5), (2, 3, 0, 9, 7, 6)])     # tri6: [c0, c1, c2, m01, m12, m20]
# permutation that flips the orientation of a cell (keeps CalculiX ordering valid)
FLIP = {"triangle": [0, 2, 1], "triangle6": [0, 2, 1, 5, 4, 3],
        "tetra": [0, 2, 1, 3], "tetra10": [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]}

CELL_DIM = {"vertex": 0, "line": 1, "line3": 1, "triangle": 2, "triangle6": 2, "quad": 2, "quad8": 2,
            "quad9": 2, "tetra": 3, "tetra10": 3, "hexahedron": 3, "hexahedron20": 3,
            "hexahedron27": 3, "wedge": 3, "wedge15": 3, "wedge18": 3, "pyramid": 3,
            "pyramid13": 3, "pyramid14": 3}
DOMAIN_TYPES = {2: ("triangle", "triangle6"), 3: ("tetra", "tetra10")}
LINEAR_OF = {"triangle6": "triangle", "tetra10": "tetra", "line3": "line",
             "triangle": "triangle", "tetra": "tetra", "line": "line", "vertex": "vertex"}
N_CORNERS = {"vertex": 1, "line": 2, "line3": 2, "triangle": 3, "triangle6": 3, "tetra": 4, "tetra10": 4}


class MeshError(ValueError):
    """Mesh cannot be used for the requested analysis."""


@dataclass
class GmshMesh:
    points: np.ndarray                                   # (n, 3)
    groups: dict[str, list[tuple[str, np.ndarray]]]      # name -> [(meshio cell type, conn)]
    group_dim: dict[str, int]
    unnamed_domain: dict[int, list[tuple[str, np.ndarray]]] = field(default_factory=dict)


def read_gmsh(path) -> GmshMesh:
    """Read a gmsh mesh with meshio; return points and the cells of every named physical group."""
    import meshio

    m = meshio.read(str(path))
    pts = np.asarray(m.points, dtype=float)
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    names = {k: (int(v[0]), int(v[1])) for k, v in (m.field_data or {}).items()
             if np.size(v) == 2 and not k.startswith("gmsh:")}
    groups: dict[str, list] = {k: [] for k in names}
    group_dim = {k: d for k, (_, d) in names.items()}
    sets = {k: v for k, v in (m.cell_sets or {}).items() if k in names}
    phys = (m.cell_data or {}).get("gmsh:physical")
    used_sets = bool(sets) and any(
        any(s is not None and len(s) for s in v) for v in sets.values())
    unnamed: dict[int, list] = {}
    for ib, block in enumerate(m.cells):
        ctype, data = block.type, np.asarray(block.data, dtype=np.int64)
        if ctype not in CELL_DIM:
            raise MeshError(f"unsupported gmsh cell type {ctype!r}")
        cdim = CELL_DIM[ctype]
        if used_sets:
            for name, per_block in sets.items():
                idx = per_block[ib] if ib < len(per_block) else None
                if idx is not None and len(idx) and group_dim[name] == cdim:
                    groups[name].append((ctype, data[np.asarray(idx, dtype=np.int64)]))
        elif phys is not None:
            tags = np.asarray(phys[ib])
            for name, (tag, d) in names.items():
                if d == cdim:
                    sel = tags == tag
                    if sel.any():
                        groups[name].append((ctype, data[sel]))
        unnamed.setdefault(cdim, []).append((ctype, data))
    return GmshMesh(points=pts, groups=groups, group_dim=group_dim, unnamed_domain=unnamed)


def _signed_measure(X: np.ndarray, corners: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        a, b, c = (X[corners[:, i], :2] for i in range(3))
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    a, b, c, d = (X[corners[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _encode(keys: np.ndarray, base: int):
    """Sorted int rows -> hashable codes (int64 when it fits, else tuples)."""
    k = keys.shape[1]
    if float(base) ** k < 2.0 ** 62:
        code = np.zeros(len(keys), dtype=np.int64)
        for j in range(k):
            code = code * base + keys[:, j]
        return code
    return np.array([hash(tuple(r)) for r in keys.tolist()], dtype=np.int64)  # pragma: no cover


@dataclass
class FEModel:
    X: np.ndarray                  # (nn, 3) node coordinates; model node i -> CalculiX node i + 1
    conn: np.ndarray               # (ne, nen) connectivity in CalculiX order (model node indices)
    cell_type: str                 # triangle | triangle6 | tetra | tetra10
    dim: int
    order: int
    elem_groups: dict[str, np.ndarray]     # domain cell group -> element indices
    mesh: GmshMesh
    orig_to_model: np.ndarray      # original mesh point index -> model node index (-1 = unused)
    warnings: list[str] = field(default_factory=list)
    _face_index: dict | None = None
    _edge_mid: dict | None = None

    # ------------------------------------------------------------------ basic
    @property
    def n_nodes(self) -> int:
        return len(self.X)

    @property
    def n_elems(self) -> int:
        return len(self.conn)

    @property
    def n_corners(self) -> int:
        return self.dim + 1

    @property
    def faces_local(self) -> np.ndarray:
        f = TRI_FACES if self.dim == 2 else TET_FACES
        return f if self.order == 2 else f[:, : self.dim]

    def corner_conn(self) -> np.ndarray:
        return self.conn[:, : self.n_corners]

    def measures(self) -> np.ndarray:
        return np.abs(_signed_measure(self.X, self.corner_conn(), self.dim))

    def group_names(self) -> list[str]:
        return sorted(self.mesh.groups)

    def has_group(self, name: str) -> bool:
        return name == "*" or name in self.mesh.groups

    def group_dim(self, name: str) -> int:
        if name == "*":
            return self.dim
        if name not in self.mesh.groups:
            raise MeshError(f"unknown physical group {name!r}; mesh has {self.group_names()}")
        return self.mesh.group_dim[name]

    def elements_of(self, name: str | None) -> np.ndarray:
        if name in (None, "*"):
            return np.arange(self.n_elems)
        if name not in self.elem_groups:
            d = self.group_dim(name)
            raise MeshError(f"group {name!r} has dimension {d}; a cell group of dimension {self.dim} is needed")
        return self.elem_groups[name]

    # ------------------------------------------------------------------ faces
    def _faces(self):
        if self._face_index is None:
            fl = self.faces_local[:, : self.dim]
            corners = self.conn[:, fl]                       # (ne, nf, dim)
            ne, nf = corners.shape[:2]
            keys = np.sort(corners.reshape(-1, self.dim), axis=1)
            code = _encode(keys, self.n_nodes + 1)
            order = np.argsort(code, kind="stable")
            self._face_index = {"code": code[order], "elem": (order // nf), "face": (order % nf)}
        return self._face_index

    def match_facets(self, corner_nodes: np.ndarray):
        """Model-node corner rows of facets -> (elem, local face) arrays; all adjacent elements are
        returned for interior facets. Raises if a facet is not a face of any element."""
        fi = self._faces()
        keys = np.sort(corner_nodes, axis=1)
        code = _encode(keys, self.n_nodes + 1)
        lo = np.searchsorted(fi["code"], code, side="left")
        hi = np.searchsorted(fi["code"], code, side="right")
        if np.any(hi == lo):
            bad = int(np.argmax(hi == lo))
            raise MeshError(f"facet with nodes {corner_nodes[bad] + 1} is not a face of any domain element")
        el, fc, fac = [], [], []
        for i, (a, b) in enumerate(zip(lo, hi)):
            for j in range(a, b):
                el.append(fi["elem"][j]); fc.append(fi["face"][j]); fac.append(i)
        return np.array(el, dtype=np.int64), np.array(fc, dtype=np.int64), np.array(fac, dtype=np.int64)

    def facets_of(self, name: str, interior: str = "first"):
        """(elem, local face) of the facets of facet group ``name``. Interior facets (two adjacent
        elements) keep the first element unless ``interior == 'all'``."""
        if self.group_dim(name) != self.dim - 1:
            raise MeshError(f"group {name!r} has dimension {self.group_dim(name)}; "
                            f"a facet group of dimension {self.dim - 1} is needed")
        rows = []
        for ctype, data in self.mesh.groups[name]:
            c = self.orig_to_model[data[:, : N_CORNERS[ctype]]]
            if np.any(c < 0):
                raise MeshError(f"facet group {name!r} has nodes outside the domain mesh")
            rows.append(c)
        if not rows:
            raise MeshError(f"facet group {name!r} is empty")
        corners = np.unique(np.sort(np.vstack(rows), axis=1), axis=0)
        el, fc, fac = self.match_facets(corners)
        if interior != "all":
            keep = np.r_[True, fac[1:] != fac[:-1]]
            n_int = int(len(fac) - keep.sum())
            if n_int:
                self.warnings.append(f"group {name!r}: {n_int} facets are interior (two adjacent "
                                     "elements); surface loads use one side")
            el, fc = el[keep], fc[keep]
        return el, fc

    def face_nodes(self, elem: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Model nodes of element faces (corners first, then midside nodes)."""
        return self.conn[elem[:, None], self.faces_local[face]]

    # ------------------------------------------------------------------ node sets
    def _edges(self):
        if self._edge_mid is None:
            ed = TRI_EDGES if self.dim == 2 else TET_EDGES
            a = self.conn[:, ed[:, 0]].ravel()
            b = self.conn[:, ed[:, 1]].ravel()
            mid = self.conn[:, self.n_corners:].ravel()
            keys = np.sort(np.c_[a, b], axis=1)
            code = _encode(keys, self.n_nodes + 1)
            order = np.argsort(code)
            self._edge_mid = {"code": code[order], "mid": mid[order]}
        return self._edge_mid

    def nodes_of(self, name: str) -> np.ndarray:
        """All model nodes of a group of any dimension (midside nodes included)."""
        if name in (None, "*"):
            return np.arange(self.n_nodes)
        d = self.group_dim(name)
        if d == self.dim:
            return np.unique(self.conn[self.elements_of(name)])
        if d == self.dim - 1:
            el, fc = self.facets_of(name, interior="all")
            return np.unique(self.face_nodes(el, fc))
        nodes = []
        for ctype, data in self.mesh.groups[name]:
            c = self.orig_to_model[data[:, : N_CORNERS[ctype]]]
            c = c[np.all(c >= 0, axis=1)]
            nodes.append(c.ravel())
            if self.order == 2 and LINEAR_OF[ctype] == "line" and len(c):
                em = self._edges()
                code = _encode(np.sort(c[:, :2], axis=1), self.n_nodes + 1)
                pos = np.clip(np.searchsorted(em["code"], code), 0, len(em["code"]) - 1)
                ok = em["code"][pos] == code
                nodes.append(em["mid"][pos[ok]])
        out = np.unique(np.concatenate(nodes)) if nodes else np.zeros(0, dtype=np.int64)
        if len(out) == 0:
            raise MeshError(f"group {name!r} has no nodes in the domain mesh")
        return out

    def check_midside(self, tol: float = 0.25) -> float:
        """Max distance of midside nodes from their edge midpoints relative to the edge length."""
        if self.order != 2:
            return 0.0
        ed = TRI_EDGES if self.dim == 2 else TET_EDGES
        a = self.X[self.conn[:, ed[:, 0]]]
        b = self.X[self.conn[:, ed[:, 1]]]
        m = self.X[self.conn[:, self.n_corners:]]
        rel = np.linalg.norm(m - 0.5 * (a + b), axis=2) / np.maximum(np.linalg.norm(b - a, axis=2), 1e-300)
        worst = float(rel.max()) if rel.size else 0.0
        if worst > tol:
            raise MeshError(f"midside nodes are {worst:.2f} edge lengths from their edge midpoints: "
                            "node ordering problem or a badly distorted quadratic mesh")
        return worst


def build_model(gm: GmshMesh, dim: int, order: int, axisymmetric: bool = False) -> FEModel:
    """Domain cells of dimension ``dim`` -> CalculiX-ordered model of the requested order."""
    warnings: list[str] = []
    blocks: list[tuple[str, np.ndarray, str]] = []          # (type, conn, group name)
    for name, bl in gm.groups.items():
        if gm.group_dim[name] != dim:
            continue
        for ctype, data in bl:
            blocks.append((ctype, data, name))
    if not blocks:
        bl = gm.unnamed_domain.get(dim, [])
        if not bl:
            raise MeshError(f"mesh has no cells of dimension {dim}")
        warnings.append(f"no physical group of dimension {dim}: using all {dim}D cells as the domain")
        blocks = [(ctype, data, "*") for ctype, data in bl]
    types = {b[0] for b in blocks}
    bad = types - set(DOMAIN_TYPES[dim])
    if bad:
        raise MeshError(f"unsupported {dim}D cell types {sorted(bad)}: the ccx backend takes "
                        f"{' / '.join(DOMAIN_TYPES[dim])} meshes")
    mesh_quadratic = {"triangle6", "tetra10"} & types
    if len(types) > 1:
        raise MeshError(f"mixed linear and quadratic domain cells {sorted(types)}")
    ctype = next(iter(types))
    mesh_order = 2 if mesh_quadratic else 1

    # dedupe cells (msh2 lists a cell once per physical group) and record group membership
    allc = np.vstack([b[1] for b in blocks])
    keys = np.sort(allc[:, : dim + 1], axis=1)
    uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    conn = allc[first]
    elem_groups: dict[str, list] = {}
    start = 0
    for ct, data, name in blocks:
        n = len(data)
        if name != "*":
            elem_groups.setdefault(name, []).append(inv[start:start + n])
        start += n
    elem_groups_np = {k: np.unique(np.concatenate(v)) for k, v in elem_groups.items()}

    pts = gm.points
    if dim == 2:
        span = np.ptp(pts[np.unique(conn)], axis=0)
        if span[2] > 1e-9 * max(span.max(), 1e-300):
            raise MeshError("2D meshes must lie in the xy-plane (z = 0)")
    # orientation
    vol = _signed_measure(pts, conn[:, : dim + 1], dim)
    if np.any(vol == 0):
        raise MeshError(f"{int(np.sum(vol == 0))} degenerate (zero-measure) cells")
    neg = vol < 0
    if neg.any():
        conn[neg] = conn[neg][:, FLIP[ctype]]
        if dim == 3:
            warnings.append(f"{int(neg.sum())} cells had negative orientation and were flipped")

    if order == 1 and mesh_order == 2:
        conn = conn[:, : dim + 1]
        warnings.append("analysis.order = 1 on a quadratic mesh: midside nodes dropped "
                        "(straight-sided linear elements)")
    used = np.unique(conn)
    orig_to_model = -np.ones(len(pts), dtype=np.int64)
    orig_to_model[used] = np.arange(len(used))
    X = pts[used].copy()
    conn = orig_to_model[conn]
    if order == 2 and mesh_order == 1:
        ed = TRI_EDGES if dim == 2 else TET_EDGES
        e = np.sort(np.stack([conn[:, ed[:, 0]], conn[:, ed[:, 1]]], axis=2).reshape(-1, 2), axis=1)
        ue, einv = np.unique(e, axis=0, return_inverse=True)
        mids = len(X) + np.arange(len(ue))
        X = np.vstack([X, 0.5 * (X[ue[:, 0]] + X[ue[:, 1]])])
        conn = np.hstack([conn, mids[einv.ravel()].reshape(len(conn), -1)])
        warnings.append("analysis.order = 2 on a linear mesh: midside nodes inserted at straight "
                        "edge midpoints")
    cell_type = {(2, 1): "triangle", (2, 2): "triangle6", (3, 1): "tetra", (3, 2): "tetra10"}[(dim, order)]
    if axisymmetric and X[:, 0].min() < -1e-9 * max(np.ptp(X[:, 0]), 1e-300):
        raise MeshError("axisymmetric meshes need x = r >= 0")
    model = FEModel(X=X, conn=conn, cell_type=cell_type, dim=dim, order=order,
                    elem_groups=elem_groups_np, mesh=gm, orig_to_model=orig_to_model,
                    warnings=warnings)
    model.check_midside()
    return model
