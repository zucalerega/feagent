"""Point location and interpolation of nodal fields on tri3 / tri6 meshes (numpy / scipy).

Cells are located with straight-sided corner triangles (barycentric coordinates); tri6 values are
interpolated with the quadratic Lagrange shape functions in those coordinates, tri3 linearly.
Points outside every candidate cell (e.g. marginally outside a curved boundary) are evaluated in
the nearest candidate with clamped barycentric coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class TriInterpolator:
    def __init__(self, points: np.ndarray, cells: np.ndarray, cell_type: str, k: int = 12):
        self.points = np.asarray(points, dtype=float)
        self.cells = np.asarray(cells, dtype=np.int64)
        self.quadratic = cell_type == "triangle6"
        c = self.cells[:, :3]
        a, b, d = self.points[c[:, 0]], self.points[c[:, 1]], self.points[c[:, 2]]
        T = np.stack([b - a, d - a], axis=2)            # (m, 2, 2), columns = edge vectors
        self.Tinv = np.linalg.inv(T)
        self.a = a
        self.tree = cKDTree((a + b + d) / 3.0)
        self.k = min(k, len(self.cells))

    def locate(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cell index, barycentric coordinates (n, 3), violation >= 0) for points q (n, 2)."""
        q = np.atleast_2d(np.asarray(q, dtype=float))[:, :2]
        _, idx = self.tree.query(q, k=self.k)
        idx = np.atleast_2d(idx).reshape(len(q), -1)
        rel = q[:, None, :] - self.a[idx]                                  # (n, k, 2)
        l12 = np.einsum("nkij,nkj->nki", self.Tinv[idx], rel)              # (n, k, 2)
        lam = np.concatenate([1.0 - l12.sum(axis=2, keepdims=True), l12], axis=2)
        viol = np.maximum(-lam.min(axis=2), 0.0)                           # 0 inside
        best = np.argmin(viol, axis=1)
        rows = np.arange(len(q))
        lam_b = np.clip(lam[rows, best], 0.0, None)
        lam_b /= lam_b.sum(axis=1, keepdims=True)
        return idx[rows, best], lam_b, viol[rows, best]

    def shape(self, lam: np.ndarray) -> np.ndarray:
        if not self.quadratic:
            return lam
        l0, l1, l2 = lam[:, 0], lam[:, 1], lam[:, 2]
        return np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                         4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0], axis=1)

    def __call__(self, values: np.ndarray, q: np.ndarray) -> np.ndarray:
        cell, lam, _ = self.locate(q)
        N = self.shape(lam)
        return (np.asarray(values)[self.cells[cell]] * N).sum(axis=1)
