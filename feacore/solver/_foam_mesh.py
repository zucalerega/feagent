"""gmsh .msh -> gmsh 2.2 volume mesh for ``gmshToFoam`` (private helper of the OpenFOAM backend).

* 2D (flow formulation "2d"): triangles / quads in the xy-plane are extruded by one cell layer of
  ``thickness`` (z from 0 to thickness) into prisms / hexes. Edge groups become side patches, the
  two copies of the 2D mesh become the ``frontAndBack`` patch (type ``empty``).
* 3D: tets / hexes / prisms / pyramids are passed through (quadratic cells are linearised; finite
  volumes use straight faces anyway).

Every boundary face is written exactly once, so overlapping physical groups are split into
"atoms": a patch per distinct set of groups (``a__b`` for faces in groups ``a`` and ``b``). A
group's faces are then the union of the patches whose atom contains it. Boundary faces in no
group go to the patch ``unassigned`` (a no-slip wall unless a BC says otherwise). Facet groups
that lie inside the domain are kept (``interior_groups``) but cannot carry BCs.

Input: gmsh 2.2 or 4.1, ASCII or binary (read with meshio), with named physical groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import meshio
import numpy as np

from feacore.solver._skfem_mesh import MeshError, _row_keys

_LINEAR = {"triangle": "triangle", "triangle6": "triangle", "quad": "quad", "quad8": "quad", "quad9": "quad",
           "tetra": "tetra", "tetra10": "tetra", "hexahedron": "hexahedron", "hexahedron20": "hexahedron",
           "hexahedron27": "hexahedron", "wedge": "wedge", "wedge15": "wedge", "wedge18": "wedge",
           "pyramid": "pyramid", "pyramid13": "pyramid", "pyramid14": "pyramid",
           "line": "line", "line3": "line", "vertex": "vertex"}
_NV = {"vertex": 1, "line": 2, "triangle": 3, "quad": 4, "tetra": 4, "hexahedron": 8, "wedge": 6, "pyramid": 5}
_GDIM = {"vertex": 0, "line": 1, "triangle": 2, "quad": 2, "tetra": 3, "hexahedron": 3, "wedge": 3, "pyramid": 3}
_GMSH_ID = {"triangle": 2, "quad": 3, "tetra": 4, "hexahedron": 5, "wedge": 6, "pyramid": 7}
# faces of 3D cells in gmsh node order (orientation irrelevant: matched as vertex sets)
_FACES = {
    "tetra": [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    "hexahedron": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}
FRONT_BACK = "frontAndBack"
UNASSIGNED = "unassigned"


@dataclass
class FoamMesh:
    dim: int                         # analysis dimension (2: extruded one layer)
    thickness: float                 # 2D extrusion depth (1.0 in 3D)
    cell_type: str                   # e.g. "triangle", "quad", "triangle+quad", "tetra"
    n_nodes: int                     # nodes of the input (2D or 3D) mesh used by domain cells
    n_cells: int                     # input cells (2D cells for extruded cases)
    measure: float                   # area (2D) / volume (3D)
    centroids: np.ndarray            # (n_cells, dim) centroids of the input cells, in file order
    patches: dict = field(default_factory=dict)          # patch name -> tuple of group names
    group_patches: dict = field(default_factory=dict)    # facet group -> [patch names]
    cell_groups: dict = field(default_factory=dict)      # cell group -> cell indices (written order)
    interior_groups: dict = field(default_factory=dict)  # facet group -> (n, nv) node indices
    bbox: np.ndarray | None = None
    warnings: list = field(default_factory=list)

    def patches_of(self, group: str) -> list[str]:
        if group not in self.group_patches:
            if group in self.interior_groups:
                raise MeshError(f"group {group!r} lies inside the domain, not on its boundary")
            if group in self.cell_groups:
                raise MeshError(f"group {group!r} is a cell group, not a boundary group")
            raise MeshError(f"unknown boundary group {group!r}; available: {sorted(self.group_patches)}")
        return self.group_patches[group]


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s if re.match(r"[A-Za-z_]", s) else "g_" + s


def _sets(m: meshio.Mesh) -> dict:
    """{name: {cell_type: indices into m.cells_dict[cell_type]}} for gmsh 4.1 and 2.2 files."""
    if m.cell_sets:
        return {k: {t: np.asarray(i, dtype=np.int64) for t, i in v.items() if len(i)}
                for k, v in m.cell_sets_dict.items() if not k.startswith("gmsh:")}
    out: dict = {}
    phys = m.cell_data.get("gmsh:physical")
    if phys is None:
        return out
    tag_name = {(int(v[0]), int(v[1])): k for k, v in m.field_data.items()}
    offsets: dict = {}
    for block, tags in zip(m.cells, phys):
        off = offsets.get(block.type, 0)
        offsets[block.type] = off + len(block.data)
        lin = _LINEAR.get(block.type)
        if lin is None:
            continue
        for tg in np.unique(tags):
            name = tag_name.get((int(tg), _GDIM[lin]))
            if name is None:
                continue
            idx = off + np.nonzero(tags == tg)[0]
            d = out.setdefault(name, {})
            d[block.type] = np.concatenate([d.get(block.type, np.zeros(0, np.int64)), idx])
    return out


def _keys(rows: np.ndarray, nmax: int) -> np.ndarray:
    """Set keys of faces given as (n, k) node rows; -1 pads triangles in 4-column rows."""
    return _row_keys(rows + 1, nmax + 2)


def _tri_area(p, a, b, c):
    return 0.5 * ((p[b, 0] - p[a, 0]) * (p[c, 1] - p[a, 1]) - (p[c, 0] - p[a, 0]) * (p[b, 1] - p[a, 1]))


def _atoms(n_faces: int, member: dict, order: list[str], warnings: list) -> tuple[np.ndarray, dict, dict]:
    """Split boundary faces into atoms (distinct sets of groups). Returns (atom index per face,
    {patch name: groups}, {group: [patch names]})."""
    names = [g for g in order if g in member]
    M = np.zeros((n_faces, len(names)), dtype=bool)
    for j, g in enumerate(names):
        M[member[g], j] = True
    uniq, inv = np.unique(M, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    patches: dict = {}
    pnames = []
    used = set()
    for row in uniq:
        groups = tuple(names[j] for j in np.nonzero(row)[0])
        base = "__".join(_safe(g) for g in groups) if groups else UNASSIGNED
        name, k = base, 1
        while name in used or name == FRONT_BACK:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        pnames.append(name)
        patches[name] = groups
        if len(groups) > 1:
            warnings.append(f"mesh: {int((inv == len(pnames) - 1).sum())} boundary faces belong to several "
                            f"groups {list(groups)}; they form patch {name!r}")
    group_patches = {g: [pnames[i] for i, row in enumerate(uniq) if row[j]] for j, g in enumerate(names)}
    return inv, patches, group_patches


def _write_msh22(path: Path, pts: np.ndarray, vol: list, faces: list, phys_names: list) -> None:
    """vol / faces: lists of (gmsh type name, (n, nv) zero-based node rows, physical tag)."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n")
        f.write(f"{len(phys_names)}\n")
        for dim, tag, name in phys_names:
            f.write(f'{dim} {tag} "{name}"\n')
        f.write(f"$EndPhysicalNames\n$Nodes\n{len(pts)}\n")
        ids = np.arange(1, len(pts) + 1)
        np.savetxt(f, np.column_stack([ids, pts]), fmt="%d %.17g %.17g %.17g")
        n_el = sum(len(r) for _, r, _ in faces) + sum(len(r) for _, r, _ in vol)
        f.write(f"$EndNodes\n$Elements\n{n_el}\n")
        eid = 1
        for typ, rows, tag in faces + vol:
            if not len(rows):
                continue
            n = len(rows)
            head = np.column_stack([np.arange(eid, eid + n), np.full(n, _GMSH_ID[typ]), np.full(n, 2),
                                    np.full(n, tag), np.full(n, tag)])
            np.savetxt(f, np.column_stack([head, rows + 1]), fmt="%d")
            eid += n
        f.write("$EndElements\n")


def _domain_blocks(m: meshio.Mesh, dim: int) -> list:
    want = {"triangle", "quad"} if dim == 2 else {"tetra", "hexahedron", "wedge", "pyramid"}
    other = {"tetra", "hexahedron", "wedge", "pyramid"} if dim == 2 else set()
    types = {c.type for c in m.cells}
    blocks = [t for t in m.cells_dict if _LINEAR.get(t) in want]
    if not blocks:
        if any(_LINEAR.get(t) in other for t in types):
            raise MeshError(f"the analysis is 2D but the file contains 3D cells ({sorted(types)}); "
                            f"check analysis.formulation")
        raise MeshError(f"no {dim}D cells in the mesh (cell types: {sorted(types)})")
    unknown = {t for t in types if t not in _LINEAR}
    if unknown:
        raise MeshError(f"unsupported cell types {sorted(unknown)}")
    return blocks


def convert(path, dim: int, thickness: float, out_msh: Path) -> FoamMesh:
    """Read the gmsh mesh ``path`` and write the 3D gmsh 2.2 mesh ``out_msh`` for gmshToFoam."""
    try:
        m = meshio.read(str(path))
    except Exception as e:  # meshio raises many types
        raise MeshError(f"cannot read mesh {str(path)!r}: {e}") from e
    warnings: list[str] = []
    blocks = _domain_blocks(m, dim)
    sets = _sets(m)
    gdims = {k: int(v[1]) for k, v in m.field_data.items()} if m.field_data else {}
    pts = np.asarray(m.points, dtype=float)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    if dim == 2:
        zspan = float(np.ptp(pts[:, 2])) if len(pts) else 0.0
        if zspan > 1e-9 * max(float(np.ptp(pts[:, :2], axis=0).max()), 1e-300):
            raise MeshError("2D flow analysis but the mesh is not planar in z (2D meshes lie in the xy-plane)")

    # ---- domain cells (linearised, deduplicated: gmsh 2.2 repeats an element per physical group)
    cells = []          # (linear type, rows, original block type, kept-cell index of every file row)
    nmax = len(pts)
    for bt in blocks:
        lt = _LINEAR[bt]
        rows = np.asarray(m.cells_dict[bt], dtype=np.int64)[:, :_NV[lt]]
        _, first, inv = np.unique(_keys(rows, nmax), return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        if len(first) != len(rows):
            warnings.append(f"mesh: {len(rows) - len(first)} duplicate {bt} cells removed")
        order = np.argsort(first)            # keep file order of the first occurrences
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        cells.append((lt, rows[np.sort(first)], bt, rank[inv]))
    if any(bt != _LINEAR[bt] for bt in blocks):
        warnings.append("info: quadratic cells linearised (finite volumes use straight faces)")
    used = np.unique(np.concatenate([r.ravel() for _, r, _, _ in cells]))
    renum = np.full(len(pts), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    P = pts[used]
    cells = [(lt, renum[r], bt, first) for lt, r, bt, first in cells]
    n_cells = sum(len(r) for _, r, _, _ in cells)
    offsets = np.cumsum([0] + [len(r) for _, r, _, _ in cells])

    # cell groups (indices into the written cell order)
    cell_groups: dict = {}
    for name, by_type in sets.items():
        if gdims.get(name, dim) != dim:
            continue
        idx = []
        for k, (lt, rows, bt, kept) in enumerate(cells):
            if bt in by_type:
                idx.append(offsets[k] + kept[by_type[bt]])
        if idx:
            cell_groups[name] = np.unique(np.concatenate(idx))

    # facet groups as node rows
    fdim = dim - 1
    facet_rows: dict = {}
    for name, by_type in sets.items():
        gd = gdims.get(name)
        rows = []
        for t, i in by_type.items():
            lt = _LINEAR.get(t)
            if lt is None or _GDIM[lt] != fdim or (gd is not None and gd != fdim):
                continue
            r = np.asarray(m.cells_dict[t], dtype=np.int64)[i][:, :_NV[lt]]
            if np.any(renum[r] < 0):
                raise MeshError(f"group {name!r} uses nodes that belong to no domain cell")
            r = renum[r]
            if r.shape[1] == 3 and dim == 3:
                r = np.column_stack([r, np.full(len(r), -1)])
            rows.append(r)
        if rows:
            facet_rows[name] = np.concatenate(rows)

    # ---- all faces of the (input-dimension) cells -> boundary faces
    if dim == 2:
        segs, cent, area = [], [], 0.0
        for lt, rows, _, _ in cells:
            nv = rows.shape[1]
            a = _tri_area(P, rows[:, 0], rows[:, 1], rows[:, 2])
            if nv == 4:
                a = a + _tri_area(P, rows[:, 0], rows[:, 2], rows[:, 3])
            flip = a < 0
            if flip.any():  # orient counter-clockwise (positive prism / hex volume)
                rows[flip] = rows[flip][:, ::-1]
                a = np.abs(a)
            area += float(a.sum())
            for k in range(nv):
                segs.append(np.column_stack([rows[:, k], rows[:, (k + 1) % nv]]))
            # area centroid (quads: two triangles)
            if nv == 3:
                cent.append(P[rows, :2].mean(axis=1))
            else:
                a1 = np.abs(_tri_area(P, rows[:, 0], rows[:, 1], rows[:, 2]))[:, None]
                a2 = np.abs(_tri_area(P, rows[:, 0], rows[:, 2], rows[:, 3]))[:, None]
                c1 = P[rows[:, [0, 1, 2]], :2].mean(axis=1)
                c2 = P[rows[:, [0, 2, 3]], :2].mean(axis=1)
                cent.append((a1 * c1 + a2 * c2) / np.maximum(a1 + a2, 1e-300))
        allf = np.concatenate(segs)
        measure = area
        centroids = np.concatenate(cent)
    else:
        fl, cent = [], []
        for lt, rows, _, _ in cells:
            for fv in _FACES[lt]:
                r = rows[:, fv]
                if len(fv) == 3:
                    r = np.column_stack([r, np.full(len(r), -1)])
                fl.append(r)
            cent.append(P[rows].mean(axis=1))
        allf = np.concatenate(fl)
        centroids = np.concatenate(cent)
        measure = float("nan")  # filled from the finite-volume cell volumes
    keys = _keys(allf, len(P))
    uk, first, counts = np.unique(keys, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise MeshError(f"non-manifold mesh: {int((counts > 2).sum())} faces shared by more than two cells")
    bfaces = allf[first[counts == 1]]
    bkeys = uk[counts == 1]
    ikeys = uk[counts == 2]

    # ---- facet groups -> boundary-face membership
    member: dict = {}
    interior: dict = {}
    for name, rows in facet_rows.items():
        k = np.unique(_keys(rows, len(P)))
        pos = np.searchsorted(bkeys, k)
        on_b = (pos < len(bkeys)) & (bkeys[np.minimum(pos, len(bkeys) - 1)] == k)
        if on_b.any():
            member[name] = pos[on_b]
        if (~on_b).any():
            kin = k[~on_b]
            pi = np.searchsorted(ikeys, kin)
            inside = (pi < len(ikeys)) & (ikeys[np.minimum(pi, len(ikeys) - 1)] == kin)
            if not inside.all():
                raise MeshError(f"group {name!r}: {int((~inside).sum())} facets are not faces of the domain mesh "
                                f"(non-conforming or unrelated elements)")
            interior[name] = rows[np.isin(_keys(rows, len(P)), kin)]
            if on_b.any():
                warnings.append(f"mesh: group {name!r} has faces both on the boundary and inside the domain")
    for name in gdims:
        if name not in member and name not in interior and name not in cell_groups and not name.startswith("gmsh:"):
            warnings.append(f"mesh: physical group {name!r} has no usable elements")
    order = sorted(member)
    atom, patches, group_patches = _atoms(len(bfaces), member, order, warnings)
    if UNASSIGNED in patches:
        warnings.append(f"mesh: {int((atom == list(patches).index(UNASSIGNED)).sum())} boundary faces are in no "
                        f"named group: patch {UNASSIGNED!r}")

    # ---- write the volume mesh
    phys = []
    vol_out, face_out = [], []
    tag = 1
    pnames = list(patches)
    if dim == 2:
        n = len(P)
        z0 = P.copy()
        z0[:, 2] = 0.0
        z1 = z0.copy()
        z1[:, 2] = thickness
        X = np.vstack([z0, z1])
        for lt, rows, _, _ in cells:
            vol_out.append(("wedge" if lt == "triangle" else "hexahedron", np.column_stack([rows, rows + n]), 0))
            face_out.append((lt, rows, -1))            # front (z = 0)
            face_out.append((lt, rows + n, -1))        # back (z = thickness)
        for i, name in enumerate(pnames):
            e = bfaces[atom == i]
            quads = np.column_stack([e[:, 0], e[:, 1], e[:, 1] + n, e[:, 0] + n])
            face_out.append(("quad", quads, tag))
            phys.append((2, tag, name))
            tag += 1
        phys.append((2, tag, FRONT_BACK))
        face_out = [(t, r, tag if g == -1 else g) for t, r, g in face_out]
        tag += 1
    else:
        X = P
        for lt, rows, _, _ in cells:
            vol_out.append((lt, rows, 0))
        for i, name in enumerate(pnames):
            f = bfaces[atom == i]
            tri = f[:, 3] < 0
            if tri.any():
                face_out.append(("triangle", f[tri][:, :3], tag))
            if (~tri).any():
                face_out.append(("quad", f[~tri], tag))
            phys.append((2, tag, name))
            tag += 1
    phys.append((3, tag, "internal"))
    vol_out = [(t, r, tag) for t, r, _ in vol_out]
    _write_msh22(Path(out_msh), X, vol_out, face_out, phys)

    types = sorted({lt for lt, _, _, _ in cells})
    return FoamMesh(dim=dim, thickness=float(thickness) if dim == 2 else 1.0, cell_type="+".join(types),
                    n_nodes=len(P), n_cells=int(n_cells), measure=float(measure), centroids=centroids,
                    patches=patches, group_patches=group_patches, cell_groups=cell_groups,
                    interior_groups=interior, bbox=np.array([P[:, :dim].min(axis=0), P[:, :dim].max(axis=0)]),
                    warnings=warnings)
