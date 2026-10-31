"""Vectorised stiffness / mass assembly for Lagrange simplex bases (private helper).

scikit-fem's generic ``BilinearForm`` loops in Python over every pair of local basis functions
(900 pairs for a P2 tetrahedron with 3 components). Here the element matrices are formed with
batched BLAS products over the quadrature points instead, which is ~10x faster for P2 in 3D.
The skfem forms in ``skfem_backend`` remain the reference (``analysis.options["assembly"] =
"skfem"``) and the tests check that both agree. Batches are sized to keep the element-matrix
temporaries around 20 MB.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _shape_data(sb):
    """Scalar basis values N (nloc, nel, nqp) and physical gradients G (nloc, dim, nel, nqp)."""
    N = np.stack([np.asarray(sb.basis[a][0]) for a in range(sb.Nbfun)])
    G = np.stack([np.asarray(sb.basis[a][0].grad) for a in range(sb.Nbfun)])
    return N, G


def _chunk(m: int) -> int:
    return max(500, int(2.5e6 // (m * m)))


def _scatter(n, gd, element_matrices, max_entries: int = 20_000_000):
    """Sum element matrices into an (n, n) CSR matrix.

    gd: (nel, m) global dofs of each element; element_matrices(E) -> (len(E), m, m). Entries go
    into preallocated COO batches of up to ``max_entries`` (one COO -> CSR conversion per batch:
    fast, bounded memory); the batches are summed."""
    nel, m = gd.shape
    itype = np.int32 if n < 2**31 - 1 else np.int64
    per_batch = max(1, max_entries // (m * m))
    step = _chunk(m)
    K = None
    for b0 in range(0, nel, per_batch):
        b1 = min(nel, b0 + per_batch)
        V = np.empty((b1 - b0, m, m))
        for s0 in range(b0, b1, step):
            s1 = min(b1, s0 + step)
            V[s0 - b0:s1 - b0] = element_matrices(np.arange(s0, s1))
        g = gd[b0:b1].astype(itype)
        rows = np.broadcast_to(g[:, :, None], V.shape).reshape(-1)
        cols = np.broadcast_to(g[:, None, :], V.shape).reshape(-1)
        part = sp.coo_matrix((V.reshape(-1), (rows, cols)), shape=(n, n)).tocsr()
        del V, rows, cols
        K = part if K is None else K + part
    if K is None:
        return sp.csr_matrix((n, n))
    K.sum_duplicates()
    return K


def elastic_stiffness(sb, split, lam, mu, c, axi: bool, r=None):
    """Linear elasticity stiffness for the vector Lagrange space whose component-k dofs are
    ``split[k][scalar dof]``. lam, mu, c: (nel, nqp) arrays (c = thickness / 2 pi r / 1);
    axisymmetric adds the hoop strain u_r / r (r: (nel, nqp))."""
    N, G = _shape_data(sb)
    nloc, dim, nel, nqp = G.shape
    m = nloc * dim
    # global dof of local (a, i): split[i][element_dofs[a]] -> (nel, nloc*dim), (a, i) row-major
    gd = np.stack([split[i][sb.element_dofs] for i in range(dim)], axis=1)  # (nloc, dim, nel)
    gd = np.transpose(gd, (2, 0, 1)).reshape(nel, m)
    W = c * sb.dx

    def element_matrices(E):
        ne = len(E)
        B = np.transpose(G[:, :, E, :], (2, 0, 1, 3)).reshape(ne, m, nqp)   # (e, (a,i), q)
        Wl = (W[E] * lam[E])[:, None, :]
        Wm = (W[E] * mu[E])[:, None, :]
        if axi:
            H = np.transpose(N[:, E, :], (1, 0, 2)) / r[E][:, None, :]      # (e, a, q)
            Hv = np.zeros((ne, nloc, dim, nqp))
            Hv[:, :, 0, :] = H
            Hv = Hv.reshape(ne, m, nqp)
            D = B + Hv                                                       # divergence incl. hoop
            Ke = np.matmul(D * Wl, np.swapaxes(D, 1, 2))
            Ke += 2.0 * np.matmul(Hv * Wm, np.swapaxes(Hv, 1, 2))
        else:
            Ke = np.matmul(B * Wl, np.swapaxes(B, 1, 2))
        P = np.matmul(B * Wm, np.swapaxes(B, 1, 2)).reshape(ne, nloc, dim, nloc, dim)
        Ke = Ke.reshape(ne, nloc, dim, nloc, dim)
        Ke += np.transpose(P, (0, 1, 4, 3, 2))            # mu dN_a/dx_j dN_b/dx_i
        S = np.einsum("eakbk->eab", P)                     # mu grad N_a . grad N_b
        for i in range(dim):
            Ke[:, :, i, :, i] += S
        return Ke.reshape(ne, m, m)

    return _scatter(len(split[0]) * dim, gd, element_matrices)


def scalar_stiffness(sb, k, c):
    """Diffusion matrix int c k grad u . grad v for the scalar basis."""
    _, G = _shape_data(sb)
    nloc, dim, nel, nqp = G.shape
    W = c * k * sb.dx

    def element_matrices(E):
        B = np.transpose(G[:, :, E, :], (2, 0, 1, 3)).reshape(len(E), nloc, dim * nqp)
        Wb = np.repeat(W[E][:, None, :], dim, axis=1).reshape(len(E), 1, dim * nqp)
        return np.matmul(B * Wb, np.swapaxes(B, 1, 2))

    return _scatter(sb.N, sb.element_dofs.T, element_matrices)


def mass(sb, c=None):
    """Mass matrix int c u v for the scalar basis (c = 1 if None)."""
    N, _ = _shape_data(sb)
    W = sb.dx if c is None else c * sb.dx

    def element_matrices(E):
        B = np.transpose(N[:, E, :], (1, 0, 2))
        return np.matmul(B * W[E][:, None, :], np.swapaxes(B, 1, 2))

    return _scatter(sb.N, sb.element_dofs.T, element_matrices)
