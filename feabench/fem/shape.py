"""Isoparametric shape functions in CalculiX/Abaqus node order.

Reference elements
  tet4/tet10  : corners (0,0,0) (1,0,0) (0,1,0) (0,0,1); L1 = 1-x-y-z, L2 = x, L3 = y, L4 = z
                mid-edge order 5:(1,2) 6:(2,3) 7:(3,1) 8:(1,4) 9:(2,4) 10:(3,4)
  hex8/hex20  : [-1,1]^3, corners 1..8 = (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)
                mid-edges 9-12 bottom, 13-16 top, 17-20 vertical
  wedge6/15   : triangle (x,y) with L1 = 1-x-y, L2 = x, L3 = y; z in [-1,1], nodes 1-3 at z=-1
                mid-edges 7-9 bottom (1-2, 2-3, 3-1), 10-12 top, 13-15 vertical (1-4, 2-5, 3-6)
"""

from __future__ import annotations

import numpy as np

_HEX_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float
)
_HEX_MID = np.array(
    [[0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
     [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float
)
TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
WEDGE15_EDGES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]

N_CORNERS = {"tet4": 4, "tet10": 4, "hex8": 8, "hex20": 8, "wedge6": 6, "wedge15": 6}
N_NODES = {"tet4": 4, "tet10": 10, "hex8": 8, "hex20": 20, "wedge6": 6, "wedge15": 15}
CENTER = {
    "tet4": [0.25, 0.25, 0.25], "tet10": [0.25, 0.25, 0.25],
    "hex8": [0.0, 0.0, 0.0], "hex20": [0.0, 0.0, 0.0],
    "wedge6": [1 / 3, 1 / 3, 0.0], "wedge15": [1 / 3, 1 / 3, 0.0],
}


def shape(etype: str, xi: np.ndarray) -> np.ndarray:
    """Shape function values, shape (n_points, n_nodes), at local coordinates ``xi`` (n_points, 3)."""
    xi = np.atleast_2d(np.asarray(xi, float))
    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
    if etype in ("tet4", "tet10"):
        L = np.stack([1 - x - y - z, x, y, z], axis=1)
        if etype == "tet4":
            return L
        corners = L * (2 * L - 1)
        mids = np.stack([4 * L[:, a] * L[:, b] for a, b in TET10_EDGES], axis=1)
        return np.hstack([corners, mids])
    if etype in ("hex8", "hex20"):
        c = _HEX_CORNERS
        fx = 1 + np.outer(x, c[:, 0])
        fy = 1 + np.outer(y, c[:, 1])
        fz = 1 + np.outer(z, c[:, 2])
        if etype == "hex8":
            return fx * fy * fz / 8.0
        corners = fx * fy * fz * (np.outer(x, c[:, 0]) + np.outer(y, c[:, 1]) + np.outer(z, c[:, 2]) - 2) / 8.0
        mids = []
        for m in _HEX_MID:
            gx = (1 - x**2) if m[0] == 0 else (1 + x * m[0])
            gy = (1 - y**2) if m[1] == 0 else (1 + y * m[1])
            gz = (1 - z**2) if m[2] == 0 else (1 + z * m[2])
            mids.append(gx * gy * gz / 4.0)
        return np.hstack([corners, np.stack(mids, axis=1)])
    if etype in ("wedge6", "wedge15"):
        L = np.stack([1 - x - y, x, y], axis=1)
        zb, zt = (1 - z), (1 + z)
        if etype == "wedge6":
            return np.hstack([L * zb[:, None] / 2, L * zt[:, None] / 2])
        q = (1 - z**2)[:, None]
        cb = 0.5 * L * ((2 * L - 1) * zb[:, None] - q)
        ct = 0.5 * L * ((2 * L - 1) * zt[:, None] - q)
        tri = [(0, 1), (1, 2), (2, 0)]
        mb = np.stack([2 * L[:, a] * L[:, b] * zb for a, b in tri], axis=1)
        mt = np.stack([2 * L[:, a] * L[:, b] * zt for a, b in tri], axis=1)
        mv = L * q
        return np.hstack([cb, ct, mb, mt, mv])
    raise ValueError(f"unsupported element type {etype}")


def dshape(etype: str, xi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Derivatives dN/dxi, shape (n_points, 3, n_nodes), by central differences (exact enough)."""
    xi = np.atleast_2d(np.asarray(xi, float))
    out = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        out.append((shape(etype, xi + e) - shape(etype, xi - e)) / (2 * h))
    return np.stack(out, axis=1)


def inside(etype: str, xi: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    xi = np.atleast_2d(xi)
    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
    if etype in ("tet4", "tet10"):
        return np.min(np.stack([1 - x - y - z, x, y, z]), axis=0) >= -tol
    if etype in ("hex8", "hex20"):
        return np.max(np.abs(xi), axis=1) <= 1 + tol
    if etype in ("wedge6", "wedge15"):
        return (np.min(np.stack([1 - x - y, x, y]), axis=0) >= -tol) & (np.abs(z) <= 1 + tol)
    raise ValueError(etype)


def outside_measure(etype: str, xi: np.ndarray) -> np.ndarray:
    """How far outside the reference element (0 if inside); used to pick the best candidate."""
    xi = np.atleast_2d(xi)
    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
    if etype in ("tet4", "tet10"):
        return np.maximum(0, -np.min(np.stack([1 - x - y - z, x, y, z]), axis=0))
    if etype in ("hex8", "hex20"):
        return np.maximum(0, np.max(np.abs(xi), axis=1) - 1)
    m = np.maximum(0, -np.min(np.stack([1 - x - y, x, y]), axis=0))
    return np.maximum(m, np.maximum(0, np.abs(z) - 1))


def clamp(etype: str, xi: np.ndarray) -> np.ndarray:
    """Project local coordinates onto the reference element (approximate for simplices)."""
    xi = np.array(np.atleast_2d(xi), float)
    if etype in ("hex8", "hex20"):
        return np.clip(xi, -1, 1)
    if etype in ("tet4", "tet10"):
        xi = np.clip(xi, 0, None)
        s = xi.sum(axis=1)
        over = s > 1
        xi[over] /= s[over][:, None]
        return xi
    tri = np.clip(xi[:, :2], 0, None)
    s = tri.sum(axis=1)
    over = s > 1
    tri[over] /= s[over][:, None]
    return np.c_[tri, np.clip(xi[:, 2], -1, 1)]
