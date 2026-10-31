"""Point location in (possibly curved) simplex meshes for ``Result.probe`` (private helper).

For every query point we look at the cells whose centroids are nearest (KD-tree), compute the
reference coordinates in each candidate (affine solve, then Newton on the isoparametric map
for curved cells) and keep the cell that contains the point. Points marginally outside the
discrete domain (curved boundaries approximated by chords, points on edges / vertices with
round-off) are assigned to the nearest cell and evaluated there (polynomial extension).
A point farther than ``tol_out`` (default 1% of the bounding-box diagonal) from the mesh raises.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class PointOutsideMesh(ValueError):
    pass


class Locator:
    def __init__(self, mesh, bbox_diag: float):
        self.mesh = mesh
        self.dim = mesh.p.shape[0]
        self.gelem = mesh.elem()
        self.gdofs = mesh.dofs.element_dofs          # geometry nodes per cell
        self.X = mesh.doflocs                         # geometry node coordinates
        self.ng = self.gdofs.shape[0]
        self.curved = self.ng > self.dim + 1
        corners = mesh.p[:, mesh.t]                   # (dim, dim+1, nel)
        self.x0 = corners[:, 0, :]                    # (dim, nel)
        self.A = np.transpose(corners[:, 1:, :] - corners[:, :1, :], (2, 0, 1))  # (nel, dim, dim)
        self.Ainv = np.linalg.inv(self.A)
        self.tree = cKDTree(corners.mean(axis=1).T)
        self.nel = mesh.t.shape[1]
        self.diag = float(bbox_diag)

    # -- geometry map of cells ``e`` at reference points ``xi`` (dim, n)
    def _map(self, e, xi):
        x = np.zeros((self.dim, len(e)))
        J = np.zeros((len(e), self.dim, self.dim))
        for j in range(self.ng):
            phi, dphi = self.gelem.lbasis(xi, j)
            Xj = self.X[:, self.gdofs[j, e]]          # (dim, n)
            x += phi * Xj
            J += np.einsum("in,jn->nij", Xj, dphi)
        return x, J

    def _ref(self, e, pts):
        """Reference coordinates (dim, n) of pts (dim, n) in cells e (n,)."""
        xi = np.einsum("nij,jn->in", self.Ainv[e], pts - self.x0[:, e])
        if self.curved:
            for _ in range(12):
                x, J = self._map(e, xi)
                r = x - pts
                try:
                    d = np.linalg.solve(J, r.T[..., None])[..., 0].T
                except np.linalg.LinAlgError:
                    break
                xi = xi - d
                if np.max(np.abs(d)) < 1e-13:
                    break
        return xi

    @staticmethod
    def _outside(xi):
        bary = np.vstack([1.0 - xi.sum(axis=0), xi])
        return np.maximum(0.0, -bary.min(axis=0))

    def _clamp(self, xi):
        bary = np.vstack([1.0 - xi.sum(axis=0), xi])
        bary = np.maximum(bary, 0.0)
        bary /= bary.sum(axis=0)
        return bary[1:]

    def locate(self, pts, tol_out: float | None = None):
        """pts (n, dim) -> (cells (n,), reference coords (dim, n), distance outside (n,))."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if pts.shape[1] != self.dim:
            if pts.shape[1] == 3 and self.dim == 2:
                pts = pts[:, :2]
            else:
                raise ValueError(f"points must have {self.dim} coordinates, got shape {pts.shape}")
        tol_out = 0.01 * self.diag if tol_out is None else tol_out
        n = len(pts)
        cell = np.full(n, -1, dtype=np.int64)
        xi_out = np.zeros((self.dim, n))
        dist = np.full(n, np.inf)
        todo = np.arange(n)
        k = min(self.nel, 16)
        while len(todo):
            _, cand = self.tree.query(pts[todo], k=k)
            cand = np.asarray(cand).reshape(len(todo), -1)
            kk = cand.shape[1]
            e = cand.ravel()
            P = np.repeat(pts[todo], kk, axis=0).T
            xi = self._ref(e, P)
            out = self._outside(xi).reshape(len(todo), kk)
            best = np.argmin(out, axis=1)
            inside = out[np.arange(len(todo)), best] <= 1e-10
            sel = todo[inside]
            bi = best[inside]
            cell[sel] = cand[inside, bi]
            xi_r = xi.reshape(self.dim, len(todo), kk)
            xi_out[:, sel] = xi_r[:, np.nonzero(inside)[0], bi]
            dist[sel] = 0.0
            rest = ~inside
            if not rest.any():
                break
            if kk >= self.nel or kk >= 1024:
                # nearest cell by physical distance to the clamped reference point
                ii = np.nonzero(rest)[0]
                xc = self._clamp(xi_r[:, ii, :].reshape(self.dim, -1))
                ee = cand[ii].ravel()
                if self.curved:
                    xm, _ = self._map(ee, xc)
                else:
                    xm = np.einsum("nij,jn->in", self.A[ee], xc) + self.x0[:, ee]
                d = np.linalg.norm(xm - np.repeat(pts[todo[ii]], kk, axis=0).T, axis=0).reshape(len(ii), kk)
                bj = np.argmin(d, axis=1)
                tgt = todo[ii]
                cell[tgt] = cand[ii, bj]
                xi_out[:, tgt] = xi_r[:, ii, bj]
                dist[tgt] = d[np.arange(len(ii)), bj]
                break
            todo = todo[rest]
            k = min(self.nel, kk * 8)
        far = dist > tol_out
        if far.any():
            i = int(np.nonzero(far)[0][0])
            raise PointOutsideMesh(
                f"{int(far.sum())} probe point(s) outside the mesh, e.g. {pts[i].tolist()} at distance "
                f"{dist[i]:.3g} (> tolerance {tol_out:.3g} = 1% of the bounding-box diagonal)")
        return cell, xi_out, dist
