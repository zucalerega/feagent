"""Case + mesh -> graph with nondimensional node / edge features (numpy only, no torch).

The SAME function (``build_graph``) produces the features for dataset generation, training and
the ``nn`` backend, so train and inference cannot drift apart.

Graph
-----
Nodes are all mesh nodes (tri6: corners and midside nodes). Edges are the edges of the linear
sub-triangulation (tri6 -> 4 sub-triangles; tri3 -> the triangle edges), both directions.

Nondimensionalisation (exact for linear elasticity with traction loads, one material)
------------------------------------------------------------------------------------
  L   = largest bounding-box extent of the mesh;  x* = (x - x_min) / L
  t0  = largest applied traction magnitude;       t* = t / t0
  stresses  s* = s / t0;   displacements  u* = u E / (t0 L)
so doubling the load, changing E or scaling the geometry leaves every network input unchanged
and the outputs scale exactly. Plane strain is mapped exactly to plane stress with
E' = E / (1 - nu^2), nu' = nu / (1 - nu) (szz = nu (sxx + syy) is added afterwards).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from feacore.case import Case, as_field

NODE_FEATURES = ("x", "y", "fixed", "loaded", "free", "interior", "fix_x", "fix_y",
                 "tx", "ty", "nx", "ny", "nu", "d_fixed", "d_loaded", "d_free",
                 # graph-level, broadcast to every node: resultant load force / moment, bbox extents
                 "Fx", "Fy", "Mz", "ex", "ey")
EDGE_FEATURES = ("dx", "dy", "length")
TARGETS = ("ux", "uy", "sxx", "syy", "sxy", "von_mises")
# continuous node features whose training range is checked at inference
RANGE_CHECKED = ("x", "y", "tx", "ty", "d_fixed", "d_loaded", "d_free", "Fx", "Fy", "Mz")
DESCRIPTORS = ("extent_x", "extent_y", "solid_fraction", "n_loops", "inner_perimeter", "min_ligament",
               "h_mean", "n_nodes", "load_angle", "nu", "fixed_fraction", "loaded_fraction")
INTEGER_DESCRIPTORS = ("n_loops",)
NO_BC_DISTANCE = 10.0  # d_* value when the corresponding node set is empty


class UnsupportedCase(NotImplementedError):
    """The case is outside what the surrogate can represent at all (not just out of distribution)."""


@dataclass
class Scales:
    L: float
    t0: float
    E: float                # modulus used for u* (plane strain: E / (1 - nu^2))
    nu: float               # Poisson ratio seen by the network (plane strain: nu / (1 - nu))
    origin: tuple
    formulation: str = "plane_stress"
    nu_material: float = 0.0  # Poisson ratio of the material (szz in plane strain)

    def to_dict(self) -> dict:
        return {"L": self.L, "t0": self.t0, "E": self.E, "nu": self.nu, "origin": list(self.origin),
                "formulation": self.formulation, "nu_material": self.nu_material}


@dataclass
class Graph:
    points: np.ndarray           # (n, 2) physical coordinates
    cells: np.ndarray            # (m, 3|6)
    cell_type: str
    senders: np.ndarray          # (E,) int64
    receivers: np.ndarray        # (E,)
    node_features: np.ndarray    # (n, len(NODE_FEATURES)) float32
    edge_features: np.ndarray    # (E, len(EDGE_FEATURES)) float32
    scales: Scales
    descriptors: dict
    notes: list = field(default_factory=list)
    no_load: bool = False

    @property
    def n_nodes(self) -> int:
        return len(self.points)


# ---------------------------------------------------------------------------- topology

_TRI6_SUBEDGES = np.array([[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0], [3, 4], [4, 5], [5, 3]])
_TRI3_EDGES = np.array([[0, 1], [1, 2], [2, 0]])


def graph_edges(cells: np.ndarray, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges (senders, receivers) of the linear (sub-)triangulation, both directions."""
    pairs = _TRI6_SUBEDGES if cell_type == "triangle6" else _TRI3_EDGES
    e = np.sort(cells[:, pairs].reshape(-1, 2), axis=1)
    e = np.unique(e, axis=0)
    s = np.concatenate([e[:, 0], e[:, 1]])
    r = np.concatenate([e[:, 1], e[:, 0]])
    return s.astype(np.int64), r.astype(np.int64)


@dataclass
class Boundary:
    a: np.ndarray        # (k,) end node 1 of each boundary edge
    b: np.ndarray        # (k,) end node 2
    mid: np.ndarray      # (k,) midside node or -1
    normal: np.ndarray   # (k, 2) outward unit normal
    length: np.ndarray   # (k,)
    node_mask: np.ndarray  # (n,) bool: node on the boundary
    node_normal: np.ndarray  # (n, 2) averaged outward normal (0 inside)


def _edge_key(a, b, n):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo.astype(np.int64) * n + hi


def boundary(points: np.ndarray, cells: np.ndarray, cell_type: str) -> Boundary:
    n = len(points)
    ab = cells[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    opp = cells[:, [2, 0, 1]].reshape(-1)
    mid = cells[:, [3, 4, 5]].reshape(-1) if cell_type == "triangle6" else np.full(len(ab), -1)
    key = _edge_key(ab[:, 0], ab[:, 1], n)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    on_b = cnt[inv] == 1
    a, b, c, m = ab[on_b, 0], ab[on_b, 1], opp[on_b], mid[on_b]
    d = points[b] - points[a]
    nrm = np.stack([d[:, 1], -d[:, 0]], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", nrm, points[c] - points[a]) > 0
    nrm[flip] *= -1
    if cell_type == "triangle6":
        length = np.linalg.norm(points[m] - points[a], axis=1) + np.linalg.norm(points[b] - points[m], axis=1)
    else:
        length = np.linalg.norm(d, axis=1)
    mask = np.zeros(n, bool)
    mask[a] = mask[b] = True
    acc = np.zeros((n, 2))
    np.add.at(acc, a, nrm * length[:, None])
    np.add.at(acc, b, nrm * length[:, None])
    if cell_type == "triangle6":
        mask[m] = True
        acc[m] += nrm
    nn = np.linalg.norm(acc, axis=1)
    node_normal = np.where(nn[:, None] > 0, acc / np.maximum(nn, 1e-300)[:, None], 0.0)
    return Boundary(a, b, m, nrm, length, mask, node_normal)


def _group_boundary_edges(mesh, bnd: Boundary, name: str) -> np.ndarray:
    """Indices into the boundary-edge arrays of the line cells of facet group ``name``."""
    lines = mesh.group_lines.get(name)
    if lines is None or len(lines) == 0:
        return np.zeros(0, np.int64)
    n = len(mesh.points)
    bkey = _edge_key(bnd.a, bnd.b, n)
    order = np.argsort(bkey)
    qk = _edge_key(lines[:, 0], lines[:, 1], n)
    pos = np.clip(np.searchsorted(bkey[order], qk), 0, len(order) - 1)
    ok = bkey[order][pos] == qk
    return order[pos[ok]]


# ---------------------------------------------------------------------------- BCs -> features


def _material(case: Case):
    used = {r.material for r in case.regions} if case.regions else set(case.materials)
    mats = {(float(case.materials[k].E), float(case.materials[k].nu)) for k in used}
    if len(mats) != 1:
        raise UnsupportedCase(f"surrogate supports a single homogeneous material; the case uses {len(mats)} "
                              f"different (E, nu) pairs")
    return mats.pop()


def check_supported(case: Case) -> None:
    a = case.analysis
    if a.physics != "elasticity":
        raise UnsupportedCase(f"surrogate supports linear elasticity only (got physics={a.physics!r})")
    if a.formulation not in ("plane_stress", "plane_strain"):
        raise UnsupportedCase(f"surrogate supports 2D plane stress / plane strain only (got formulation="
                              f"{a.formulation!r}; 3D and axisymmetric are not implemented)")
    for ld in case.loads:
        if ld.type == "temperature":
            raise UnsupportedCase("thermal loads are not supported by the surrogate")
        if ld.type == "body_force":
            vals = ld.value if isinstance(ld.value, list) else [ld.value]
            if any(isinstance(v, str) or float(v) != 0.0 for v in vals):
                raise UnsupportedCase("body forces are not supported by the surrogate")
    _material(case)


def _nearest_node(points, pt):
    pt = np.asarray(pt, dtype=float)[:2]
    return int(np.argmin(np.linalg.norm(points - pt, axis=1)))


def bc_arrays(case: Case, mesh, bnd: Boundary, L: float):
    """Per-node Dirichlet flags (n, 2) and physical tractions (n, 2) from the case BCs."""
    pts = mesh.points
    n = len(pts)
    fix = np.zeros((n, 2), bool)
    trac = np.zeros((n, 2))
    notes: list[str] = []
    for i, bc in enumerate(case.bcs):
        tag = f"bcs[{i}] ({bc.type} on {bc.group or bc.point})"
        if bc.point is not None:
            nodes = np.array([_nearest_node(pts, bc.point)])
        else:
            nodes = mesh.nodes_of(bc.group)
        x, y = pts[nodes, 0], pts[nodes, 1]
        if bc.type == "fixed":
            fix[nodes] = True
        elif bc.type == "displacement":
            vals = bc.value if isinstance(bc.value, list) else [bc.value] * len(bc.components)
            for comp, v in zip(bc.components, vals):
                if np.any(np.abs(as_field(v, x, y)) > 0):
                    raise UnsupportedCase(f"{tag}: nonzero prescribed displacements are not supported by the "
                                          f"surrogate (only zero-displacement constraints)")
                fix[nodes, comp] = True
        elif bc.type == "symmetry":
            if np.ptp(x) <= 1e-6 * L:
                fix[nodes, 0] = True
            elif np.ptp(y) <= 1e-6 * L:
                fix[nodes, 1] = True
            else:
                raise UnsupportedCase(f"{tag}: symmetry BC on a group that is not axis-aligned")
        elif bc.type in ("traction", "pressure", "force"):
            if mesh.group_dim.get(bc.group) != 1:
                raise UnsupportedCase(f"{tag}: loads must act on an edge group (point loads are singular and "
                                      f"not supported)")
            eidx = _group_boundary_edges(mesh, bnd, bc.group)
            if len(eidx) == 0:
                raise UnsupportedCase(f"{tag}: group has no boundary edges")
            if bc.type == "traction":
                for comp in (0, 1):
                    trac[nodes, comp] += as_field(bc.value[comp], x, y)
            elif bc.type == "force":
                length = float(bnd.length[eidx].sum())
                t = np.asarray([float(v) for v in bc.value[:2]]) / (length * case.analysis.thickness)
                trac[nodes] += t
            else:  # pressure: t = -p n with the group's own outward normals
                acc = np.zeros((n, 2))
                ln = bnd.length[eidx][:, None]
                np.add.at(acc, bnd.a[eidx], bnd.normal[eidx] * ln)
                np.add.at(acc, bnd.b[eidx], bnd.normal[eidx] * ln)
                if mesh.cell_type == "triangle6":
                    np.add.at(acc, bnd.mid[eidx], bnd.normal[eidx])
                nn = np.linalg.norm(acc, axis=1, keepdims=True)
                gn = np.where(nn > 0, acc / np.maximum(nn, 1e-300), 0.0)
                p = as_field(bc.value, x, y)
                trac[nodes] += -p[:, None] * gn[nodes]
        else:
            raise UnsupportedCase(f"{tag}: BC type not supported by the surrogate")
    trac[fix.any(axis=1)] = 0.0  # tractions on constrained nodes do no work in this model
    return fix, trac, notes


def _loops(points, bnd: Boundary):
    """Boundary loops: (labels per boundary edge, number of loops, index of the outer loop)."""
    n = len(points)
    adj = coo_matrix((np.ones(len(bnd.a)), (bnd.a, bnd.b)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    elab = lab[bnd.a]
    ids, elab = np.unique(elab, return_inverse=True)
    areas = []
    for k in range(len(ids)):
        p = points[np.concatenate([bnd.a[elab == k], bnd.b[elab == k]])]
        ext = np.ptp(p, axis=0)
        areas.append(ext[0] * ext[1])
    return elab, len(ids), int(np.argmax(areas)) if areas else 0


def build_graph(case: Case, mesh=None) -> Graph:
    """Graph + features for a case (reads ``case.mesh_path()`` unless ``mesh`` is given)."""
    from feacore.surrogate.mesh import read_msh

    check_supported(case)
    if mesh is None:
        mesh = read_msh(case.mesh_path())
    pts, cells, ctype = mesh.points, mesh.cells, mesh.cell_type
    n = len(pts)
    origin = pts.min(axis=0)
    ext = np.ptp(pts, axis=0)
    L = float(ext.max())
    E, nu = _material(case)
    form = case.analysis.formulation
    nu_mat = nu
    if form == "plane_strain":
        E, nu = E / (1.0 - nu**2), nu / (1.0 - nu)
    bnd = boundary(pts, cells, ctype)
    fix, trac, notes = bc_arrays(case, mesh, bnd, L)
    tmag = np.linalg.norm(trac, axis=1)
    t0 = float(tmag.max())
    no_load = t0 <= 0.0
    if no_load:
        t0 = 1.0
        notes.append("no applied traction: the solution is zero")

    fixed = fix.any(axis=1)
    loaded = (tmag > 0) & ~fixed
    free = bnd.node_mask & ~fixed & ~loaded
    interior = ~(fixed | loaded | free)
    xs = (pts - origin) / L

    def dist_to(mask):
        if not mask.any():
            return np.full(n, NO_BC_DISTANCE)
        d, _ = cKDTree(xs[mask]).query(xs)
        return d

    F = np.zeros((n, len(NODE_FEATURES)), dtype=np.float64)
    col = {k: i for i, k in enumerate(NODE_FEATURES)}
    F[:, col["x"]], F[:, col["y"]] = xs[:, 0], xs[:, 1]
    F[:, col["fixed"]], F[:, col["loaded"]] = fixed, loaded
    F[:, col["free"]], F[:, col["interior"]] = free, interior
    F[:, col["fix_x"]], F[:, col["fix_y"]] = fix[:, 0], fix[:, 1]
    F[:, col["tx"]], F[:, col["ty"]] = trac[:, 0] / t0, trac[:, 1] / t0
    F[:, col["nx"]], F[:, col["ny"]] = bnd.node_normal[:, 0], bnd.node_normal[:, 1]
    F[:, col["nu"]] = nu
    F[:, col["d_fixed"]] = dist_to(fixed)
    F[:, col["d_loaded"]] = dist_to(loaded)
    F[:, col["d_free"]] = dist_to(free)

    s, r = graph_edges(cells, ctype)
    d = (pts[r] - pts[s]) / L
    ef = np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True)], axis=1)

    # ---- graph-level descriptors (out-of-distribution checks)
    tri = pts[cells[:, :3]]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    elab, n_loops, outer = _loops(pts, bnd)
    inner = elab != outer
    inner_perim = float(bnd.length[inner].sum()) / L
    if n_loops > 1:
        on = np.unique(np.concatenate([bnd.a[~inner], bnd.b[~inner]]))
        inn = np.unique(np.concatenate([bnd.a[inner], bnd.b[inner]]))
        dmin, _ = cKDTree(pts[on]).query(pts[inn])
        min_lig = float(dmin.min()) / L
    else:
        min_lig = float("nan")
    tm = 0.5 * (trac[bnd.a] + trac[bnd.b])
    loaded_edge = loaded[bnd.a] & loaded[bnd.b]
    fe = tm[loaded_edge] * bnd.length[loaded_edge, None]                     # edge forces (per thickness)
    Fres = fe.sum(axis=0)
    arm = 0.5 * (pts[bnd.a[loaded_edge]] + pts[bnd.b[loaded_edge]]) - (origin + 0.5 * ext)
    Mres = float((arm[:, 0] * fe[:, 1] - arm[:, 1] * fe[:, 0]).sum())
    F[:, col["Fx"]], F[:, col["Fy"]] = Fres[0] / (t0 * L), Fres[1] / (t0 * L)
    F[:, col["Mz"]] = Mres / (t0 * L * L)
    F[:, col["ex"]], F[:, col["ey"]] = ext[0] / L, ext[1] / L
    load_angle = math.degrees(math.atan2(Fres[1], Fres[0])) if np.linalg.norm(Fres) > 0 else float("nan")
    blen = float(bnd.length.sum())
    fixed_edge = fixed[bnd.a] & fixed[bnd.b]
    desc = {
        "extent_x": float(ext[0] / L), "extent_y": float(ext[1] / L),
        "solid_fraction": float(area / max(ext[0] * ext[1], 1e-300)),
        "n_loops": float(n_loops), "inner_perimeter": inner_perim, "min_ligament": min_lig,
        "h_mean": float(ef[:, 2].mean()), "n_nodes": float(n), "load_angle": load_angle, "nu": float(nu),
        "fixed_fraction": float(bnd.length[fixed_edge].sum() / blen),
        "loaded_fraction": float(bnd.length[loaded_edge].sum() / blen),
    }
    if not fixed.any():
        notes.append("no Dirichlet constraint: the elasticity problem is singular (rigid-body modes)")
    scales = Scales(L=L, t0=t0, E=float(E), nu=float(nu), origin=tuple(map(float, origin)),
                    formulation=form, nu_material=float(nu_mat))
    return Graph(points=pts, cells=cells, cell_type=ctype, senders=s, receivers=r,
                 node_features=F.astype(np.float32), edge_features=ef.astype(np.float32), scales=scales,
                 descriptors=desc, notes=notes, no_load=no_load)


# ---------------------------------------------------------------------------- outputs


def nondimensionalise(fields: dict, scales: Scales) -> np.ndarray:
    """Physical nodal fields {ux, uy, sxx, syy, sxy, von_mises} -> (n, 6) nondimensional targets."""
    us = scales.E / (scales.t0 * scales.L)
    cols = []
    for f in TARGETS:
        v = np.asarray(fields[f], dtype=float)
        cols.append(v * us if f in ("ux", "uy") else v / scales.t0)
    return np.stack(cols, axis=1)


def physical_fields(y: np.ndarray, scales: Scales, vm_source: str = "head") -> dict:
    """(n, 6) nondimensional prediction -> every elasticity field (physical units)."""
    y = np.asarray(y, dtype=float)
    us = scales.t0 * scales.L / scales.E
    ux, uy = y[:, 0] * us, y[:, 1] * us
    sxx, syy, sxy = y[:, 2] * scales.t0, y[:, 3] * scales.t0, y[:, 4] * scales.t0
    z = np.zeros_like(ux)
    szz = scales.nu_material * (sxx + syy) if scales.formulation == "plane_strain" else z
    vm_c = np.sqrt(np.maximum(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy**2, 0.0))
    vm = np.maximum(y[:, 5], 0.0) * scales.t0 if (vm_source == "head" and scales.formulation == "plane_stress") \
        else vm_c
    c, R = 0.5 * (sxx + syy), np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
    pr = np.sort(np.stack([c + R, c - R, szz], axis=1), axis=1)[:, ::-1]
    return {
        "ux": ux, "uy": uy, "uz": z, "u_mag": np.hypot(ux, uy),
        "sxx": sxx, "syy": syy, "szz": szz, "sxy": sxy, "syz": z, "sxz": z, "stt": z,
        "von_mises": vm, "von_mises_components": vm_c,
        "tresca": pr[:, 0] - pr[:, 2], "s1": pr[:, 0], "s2": pr[:, 1], "s3": pr[:, 2],
    }
