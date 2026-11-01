"""Shape functions, quadrature, point location and integration on the CalculiX-ordered model.

Used only for post-processing (interpolating CalculiX nodal results at probe points, QoI
integrals) and for building consistent nodal loads; the solve itself is done by CalculiX.
Reference cells: line [0, 1] (nodes a, b, mid), triangle (0,0),(1,0),(0,1), unit tetrahedron;
node ordering as in CalculiX (see ``_ccx_mesh``).
"""

from __future__ import annotations

import numpy as np

# d(barycentric L_i)/d(reference xi_j)
_DL = {
    1: np.array([[-1.0], [1.0]]),
    2: np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
    3: np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}
_MID_PAIRS = {1: [(0, 1)], 2: [(0, 1), (1, 2), (2, 0)], 3: [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]}


def shape(sdim: int, order: int, xi: np.ndarray):
    """Shape functions of the simplex of dimension ``sdim`` at reference points xi (..., sdim).
    Returns N (..., nen) and dN/dxi (..., nen, sdim)."""
    xi = np.asarray(xi, dtype=float)
    L = np.concatenate([1.0 - xi.sum(axis=-1, keepdims=True), xi], axis=-1)   # (..., sdim+1)
    DL = _DL[sdim]                                                              # (sdim+1, sdim)
    if order == 1:
        N = L
        dN = np.broadcast_to(DL, L.shape[:-1] + DL.shape).copy()
        return N, dN
    pairs = _MID_PAIRS[sdim]
    Nc = L * (2.0 * L - 1.0)
    dNc = (4.0 * L - 1.0)[..., :, None] * DL                                    # (..., nc, sdim)
    Nm = np.stack([4.0 * L[..., i] * L[..., j] for i, j in pairs], axis=-1)
    dNm = np.stack([4.0 * (L[..., j, None] * DL[i] + L[..., i, None] * DL[j]) for i, j in pairs], axis=-2)
    return np.concatenate([Nc, Nm], axis=-1), np.concatenate([dNc, dNm], axis=-2)


def quadrature(sdim: int, degree: int):
    """(points (nq, sdim), weights (nq,)) exact for polynomials of the given degree."""
    from skfem.quadrature import get_quadrature
    from skfem.refdom import RefLine, RefTet, RefTri

    X, W = get_quadrature({1: RefLine, 2: RefTri, 3: RefTet}[sdim], max(int(degree), 1))
    return X.T.copy(), W.copy()


# trilinear hexahedron on [-1, 1]^3, CalculiX C3D8 node order
_HEX_SIGN = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                      (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=float)


def hex_shape(xi: np.ndarray):
    """Trilinear shape functions N (..., 8) and dN/dxi (..., 8, 3) at points xi (..., 3) of [-1, 1]^3."""
    xi = np.asarray(xi, dtype=float)
    f = 1.0 + xi[..., None, :] * _HEX_SIGN                     # (..., 8, 3)
    N = 0.125 * f.prod(axis=-1)
    dN = np.stack([0.125 * _HEX_SIGN[:, k] * np.prod(np.delete(f, k, axis=-1), axis=-1) for k in range(3)],
                  axis=-1)
    return N, dN


def hex_quadrature(degree: int):
    """Tensor Gauss-Legendre rule on [-1, 1]^3 exact for the given degree per direction."""
    g, w = np.polynomial.legendre.leggauss(max(1, (int(degree) + 2) // 2))
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return X, W


# ---------------------------------------------------------------------------- geometry at qps
def map_points(Xe: np.ndarray, N: np.ndarray, dN: np.ndarray):
    """Element node coordinates Xe (ne, nen, 3) -> physical points (ne, nq, 3) and Jacobians
    (ne, nq, 3, sdim) for shape values N (nq, nen), dN (nq, nen, sdim)."""
    x = np.einsum("qa,eai->eqi", N, Xe)
    J = np.einsum("qas,eai->eqis", dN, Xe)
    return x, J


def measure_factor(J: np.ndarray) -> np.ndarray:
    """|det J| for full-dimensional maps, or the area / length element for embedded facets."""
    sdim = J.shape[-1]
    if sdim == 1:
        return np.linalg.norm(J[..., 0], axis=-1)
    if sdim == 2:
        c = np.cross(J[..., 0], J[..., 1])
        if np.allclose(c[..., :2], 0.0):        # in-plane 2D cell
            return np.abs(c[..., 2])
        return np.linalg.norm(c, axis=-1)
    return np.abs(np.linalg.det(J))


# ---------------------------------------------------------------------------- point location
def _bary(Xc: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of p (m, d) in straight simplices Xc (m, d+1, d)."""
    T = np.swapaxes(Xc[:, 1:] - Xc[:, :1], 1, 2)            # (m, d, d)
    rhs = p - Xc[:, 0]
    try:
        lam = np.linalg.solve(T, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:                           # pragma: no cover
        lam = np.einsum("mij,mj->mi", np.linalg.pinv(T), rhs)
    return np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)


class Locator:
    """Find the cell and reference coordinates of points (nearest cell for outside points)."""

    def __init__(self, X: np.ndarray, conn: np.ndarray, dim: int, order: int):
        from scipy.spatial import cKDTree

        self.X, self.conn, self.dim, self.order = X[:, :dim], conn, dim, order
        self.nc = dim + 1
        self.Xc = self.X[conn[:, : self.nc]]                  # (ne, nc, d)
        self.cent = self.Xc.mean(axis=1)
        self.tree = cKDTree(self.cent)

    def locate(self, pts: np.ndarray, k: int = 24, tol: float = 1e-10):
        pts = np.asarray(pts, dtype=float)[:, : self.dim]
        m = len(pts)
        k = int(min(k, len(self.conn)))
        _, cand = self.tree.query(pts, k=k)
        cand = np.asarray(cand).reshape(m, k)
        lam = _bary(self.Xc[cand.ravel()], np.repeat(pts, k, axis=0)).reshape(m, k, self.nc)
        score = lam.min(axis=2)
        best = np.argmax(score, axis=1)
        elem = cand[np.arange(m), best]
        sc = score[np.arange(m), best]
        # brute force for points not inside any candidate (large size gradients / outside points)
        for i in np.nonzero(sc < -tol)[0]:
            lam_all = _bary(self.Xc, np.repeat(pts[i:i + 1], len(self.conn), axis=0))
            s = lam_all.min(axis=1)
            j = int(np.argmax(s))
            if s[j] > sc[i]:
                elem[i], sc[i] = j, s[j]
        lam_b = _bary(self.Xc[elem], pts)
        xi = lam_b[:, 1:]
        if self.order == 2:
            elem, xi, sc = self._curved(pts, elem, cand, score)
        return elem, xi, sc

    def _curved(self, pts, elem, cand, score, kk: int = 8):
        """Curved (isoparametric) cells: a point inside a cell's straight-sided simplex can lie in a
        neighbour's curved cell. Invert the map in the best few candidates and keep a converged
        inverse inside the reference cell (else the converged one closest to it)."""
        m = len(pts)
        kk = int(min(kk, cand.shape[1]))
        order_c = np.argsort(-score, axis=1)[:, :kk]
        cands = np.concatenate([elem[:, None], cand[np.arange(m)[:, None], order_c]], axis=1)
        E = cands.ravel()
        P = np.repeat(pts, cands.shape[1], axis=0)
        xi0 = _bary(self.Xc[E], P)[:, 1:]
        xi, res = self._newton(E, P, xi0, return_residual=True)
        lam = np.concatenate([1.0 - xi.sum(axis=1, keepdims=True), xi], axis=1).min(axis=1)
        size = np.linalg.norm(self.Xc[E, 1] - self.Xc[E, 0], axis=1)
        conv = res <= 1e-9 * np.maximum(size, 1e-300)
        key = np.where(conv, lam, -1e300 - res)              # converged first, then most inside
        key = key.reshape(m, -1)
        best = np.argmax(key, axis=1)
        j = np.arange(m) * cands.shape[1] + best
        return E[j], xi[j], lam[j]

    def _newton(self, elem, pts, xi, iters: int = 30, return_residual: bool = False):
        Xe = self.X[self.conn[elem]]                        # (m, nen, d)
        for _ in range(iters):
            N, dN = shape(self.dim, 2, xi)                  # (m, nen), (m, nen, d)
            x = np.einsum("ma,mai->mi", N, Xe)
            J = np.einsum("mas,mai->mis", dN, Xe)
            r = x - pts
            det = np.linalg.det(J)
            bad = np.abs(det) < 1e-300
            J[bad] = np.eye(self.dim)
            dx = np.linalg.solve(J, r[..., None])[..., 0]
            dx = np.clip(dx, -0.5, 0.5)                     # damp steps (folded maps far outside)
            xi = xi - dx
            if np.max(np.abs(dx)) < 1e-14:
                break
        if return_residual:
            N, _ = shape(self.dim, 2, xi)
            res = np.linalg.norm(np.einsum("ma,mai->mi", N, Xe) - pts, axis=1)
            return xi, res
        return xi


def interpolate(conn: np.ndarray, dim: int, order: int, values: np.ndarray, elem: np.ndarray, xi: np.ndarray):
    """Nodal values (nn, ...) -> values at (elem, xi)."""
    N, _ = shape(dim, order, xi)                            # (m, nen)
    return np.einsum("ma,ma...->m...", N, values[conn[elem]])
