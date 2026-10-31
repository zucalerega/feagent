"""Minimal gmsh ``.msh`` reader for the surrogate (2D triangle meshes, named physical groups).

Independent of the skfem backend's private reader on purpose (the surrogate must not depend on
solver internals). Returns node coordinates, domain cells (tri3 or tri6, gmsh node order:
corners 0,1,2 then midside nodes 3=(0,1), 4=(1,2), 5=(2,0)) and, per physical group, its
dimension, its node indices and (for facet groups) its line cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DOMAIN_TYPES = ("triangle", "triangle6")
GDIM = {"vertex": 0, "line": 1, "line3": 1, "triangle": 2, "triangle6": 2,
        "quad": 2, "quad8": 2, "quad9": 2, "tetra": 3, "tetra10": 3, "hexahedron": 3}


class MeshReadError(ValueError):
    pass


@dataclass
class SurrogateMesh:
    points: np.ndarray                 # (n, 2) node coordinates (only nodes used by domain cells)
    cells: np.ndarray                  # (m, 3) tri3 or (m, 6) tri6
    cell_type: str                     # "triangle" | "triangle6"
    group_dim: dict = field(default_factory=dict)     # name -> gmsh dimension of the group
    group_nodes: dict = field(default_factory=dict)   # name -> sorted node indices
    group_lines: dict = field(default_factory=dict)   # name -> (k, 2|3) line cells (facet groups)
    group_cells: dict = field(default_factory=dict)   # name -> domain cell indices (2D groups)

    @property
    def order(self) -> int:
        return 2 if self.cell_type == "triangle6" else 1

    def nodes_of(self, name: str) -> np.ndarray:
        if name in ("*", None):
            return np.arange(len(self.points))
        if name not in self.group_nodes:
            raise MeshReadError(f"unknown physical group {name!r}; available: {sorted(self.group_nodes)}")
        return self.group_nodes[name]


def _cell_sets(m) -> dict:
    """{name: {cell_type: indices into that cell type's array}} (gmsh 4.1 cell_sets or 2.2 tags)."""
    out: dict = {}
    if m.cell_sets:
        for name, per_type in m.cell_sets_dict.items():
            if name.startswith("gmsh:"):
                continue
            out[name] = {k: np.asarray(v, dtype=np.int64) for k, v in per_type.items()}
        if out:
            return out
    phys = m.cell_data.get("gmsh:physical")
    if phys is None:
        return out
    tag_name = {(int(v[0]), int(v[1])): k for k, v in m.field_data.items()}
    offsets: dict = {}
    for block, tags in zip(m.cells, phys):
        off = offsets.get(block.type, 0)
        offsets[block.type] = off + len(block.data)
        gd = GDIM.get(block.type)
        if gd is None:
            continue
        for tg in np.unique(tags):
            name = tag_name.get((int(tg), gd))
            if name is None:
                continue
            idx = off + np.nonzero(tags == tg)[0]
            d = out.setdefault(name, {})
            d[block.type] = np.concatenate([d.get(block.type, np.zeros(0, np.int64)), idx])
    return out


def read_msh(path: str | Path) -> SurrogateMesh:
    import meshio

    try:
        m = meshio.read(str(path))
    except Exception as e:  # noqa: BLE001 - meshio raises many types
        raise MeshReadError(f"cannot read mesh {str(path)!r}: {e}") from e
    types = {c.type for c in m.cells}
    if types & {"tetra", "tetra10", "hexahedron", "wedge", "pyramid"}:
        raise MeshReadError("3D cells found: the surrogate supports 2D triangle meshes only")
    present = [t for t in DOMAIN_TYPES if t in types]
    if not present:
        raise MeshReadError(f"no triangle cells in the mesh (cell types: {sorted(types)}); the surrogate "
                            f"supports tri3 / tri6 meshes only")
    if len(present) > 1:
        raise MeshReadError("mixed linear and quadratic triangles are not supported")
    ctype = present[0]
    cd = m.cells_dict
    cells = np.asarray(cd[ctype], dtype=np.int64)
    # gmsh 2.2 may repeat an element once per physical group
    keys = np.sort(cells[:, :3], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    dedup = None
    if len(first) != len(cells):
        dedup = first
        cells = cells[first]
    pts = np.asarray(m.points, dtype=float)
    if pts.shape[1] > 2:
        if np.ptp(pts[:, 2]) > 1e-9 * max(float(np.ptp(pts[:, :2], axis=0).max()), 1e-300):
            raise MeshReadError("mesh is not planar in z: 2D meshes must lie in the xy-plane")
        pts = pts[:, :2]
    used = np.unique(cells)
    renum = np.full(len(pts), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    mesh = SurrogateMesh(points=np.ascontiguousarray(pts[used]), cells=renum[cells], cell_type=ctype)

    for name, per_type in _cell_sets(m).items():
        nodes, lines, gcells, gdim = [], [], [], None
        for t, idx in per_type.items():
            if len(idx) == 0 or t not in cd:
                continue
            gd = GDIM.get(t)
            gdim = gd if gdim is None else max(gdim, gd)
            conn = np.asarray(cd[t], dtype=np.int64)[idx]
            if t == ctype:
                if dedup is not None:  # map to deduplicated cell indices via corner keys
                    lookup = {tuple(k): i for i, k in enumerate(np.sort(cells[:, :3], axis=1))}
                    idx = np.array([lookup[tuple(k)] for k in np.sort(conn[:, :3], axis=1)], dtype=np.int64)
                gcells.append(idx)
            elif t in ("line", "line3"):
                lines.append(renum[conn])
            nodes.append(renum[conn.ravel()])
        if gdim is None:
            continue
        nd = np.unique(np.concatenate(nodes)) if nodes else np.zeros(0, np.int64)
        nd = nd[nd >= 0]
        mesh.group_dim[name] = int(gdim)
        mesh.group_nodes[name] = nd
        if lines:
            ln = np.concatenate(lines)
            mesh.group_lines[name] = ln[(ln >= 0).all(axis=1)]
        if gcells:
            mesh.group_cells[name] = np.unique(np.concatenate(gcells))
    return mesh
