"""Finite-volume post-processing of an OpenFOAM result in numpy (private helper of the backend).

OpenFOAM's own formulas, so the numbers match its function objects:

* face centres / area vectors and cell centres / volumes as ``primitiveMesh`` computes them
  (triangle fans about the face average point; pyramids about the estimated cell centre);
* ``Gauss linear`` gradients with ``surfaceInterpolation`` weights; the boundary value of the
  gradient gets its normal part from the patch ``snGrad`` (``gaussGrad::correctBoundaryConditions``),
  whose distance is the wall-normal one (``fvPatch::delta``);
* the ``forces`` function object: pressure force ``rho (p - pRef) Sf`` plus viscous force
  ``Sf . devRhoReff`` with ``devRhoReff = -rho nuEff dev(twoSymm(grad U))`` on the patch.

Why not the function objects themselves: the Ubuntu 24.04 ``openfoam`` package (v1912) aborts in
every function object (``error in IOstream "sha1"``), and doing it here keeps the results
identical across OpenFOAM versions and their output formats.
"""

from __future__ import annotations

import numpy as np

from feacore.solver._foam_io import face_geometry


class FvMesh:
    def __init__(self, points: np.ndarray, faces, owner: np.ndarray, neighbour: np.ndarray, patches: dict):
        self.points = points
        self.owner = np.asarray(owner, dtype=np.int64)
        self.neighbour = np.asarray(neighbour, dtype=np.int64)
        self.patches = patches                      # name -> (nFaces, startFace, type)
        self.n_int = len(self.neighbour)
        self.n_cells = int(max(self.owner.max(initial=-1), self.neighbour.max(initial=-1)) + 1)
        self.Cf, self.Sf = face_geometry(points, faces)
        self.magSf = np.linalg.norm(self.Sf, axis=1)
        self._cells()
        own, nei = self.owner[: self.n_int], self.neighbour
        Cf, Sf = self.Cf[: self.n_int], self.Sf[: self.n_int]
        sfd_own = np.abs(np.einsum("ij,ij->i", Sf, Cf - self.C[own]))
        sfd_nei = np.abs(np.einsum("ij,ij->i", Sf, self.C[nei] - Cf))
        self.w = sfd_nei / np.maximum(sfd_own + sfd_nei, 1e-300)   # owner weight

    def _cells(self) -> None:
        n = self.n_cells
        own, nei = self.owner, self.neighbour
        cnt = np.bincount(own, minlength=n) + np.bincount(nei, minlength=n)
        cest = np.zeros((n, 3))
        for k in range(3):
            cest[:, k] = (np.bincount(own, self.Cf[:, k], n) + np.bincount(nei, self.Cf[: self.n_int, k], n))
        cest /= np.maximum(cnt, 1)[:, None]
        # owner pyramids
        pv_o = np.einsum("ij,ij->i", self.Sf, self.Cf - cest[own])
        pc_o = 0.75 * self.Cf + 0.25 * cest[own]
        pv_n = np.einsum("ij,ij->i", self.Sf[: self.n_int], cest[nei] - self.Cf[: self.n_int])
        pc_n = 0.75 * self.Cf[: self.n_int] + 0.25 * cest[nei]
        vol = np.bincount(own, pv_o, n) + np.bincount(nei, pv_n, n)
        C = np.zeros((n, 3))
        for k in range(3):
            C[:, k] = np.bincount(own, pv_o * pc_o[:, k], n) + np.bincount(nei, pv_n * pc_n[:, k], n)
        ok = np.abs(vol) > 1e-300
        C[ok] /= vol[ok][:, None]
        C[~ok] = cest[~ok]
        self.C = C
        self.V = vol / 3.0

    def patch_slice(self, name: str) -> slice:
        n, start, _ = self.patches[name]
        return slice(start, start + n)

    def grad(self, cell: np.ndarray, bvals: dict) -> np.ndarray:
        """Gauss linear gradient of a cell field (n,) or (n, k) with boundary face values
        {patch: (nf,) or (nf, k)} (empty patches skipped). Returns (n, 3) or (n, 3, k):
        g[c, i, j] = d phi_j / d x_i (OpenFOAM's grad(U) layout)."""
        vec = cell.ndim == 2
        phi = cell if vec else cell[:, None]
        own, nei = self.owner[: self.n_int], self.neighbour
        w = self.w[:, None]
        pf = w * phi[own] + (1.0 - w) * phi[nei]
        S = self.Sf[: self.n_int]
        flux = S[:, :, None] * pf[:, None, :]                  # (nf, 3, k)
        n = self.n_cells
        k = phi.shape[1]
        g = np.zeros((n, 3, k))
        for i in range(3):
            for j in range(k):
                g[:, i, j] = np.bincount(own, flux[:, i, j], n) - np.bincount(nei, flux[:, i, j], n)
        for name, (nf, start, ptype) in self.patches.items():
            if nf == 0 or ptype == "empty" or name not in bvals:
                continue
            sl = slice(start, start + nf)
            vb = np.asarray(bvals[name], dtype=float).reshape(nf, k)
            fb = self.Sf[sl][:, :, None] * vb[:, None, :]
            ob = self.owner[sl]
            for i in range(3):
                for j in range(k):
                    g[:, i, j] += np.bincount(ob, fb[:, i, j], n)
        g /= self.V[:, None, None]
        return g if vec else g[:, :, 0]

    def boundary_grad(self, name: str, cell: np.ndarray, grad: np.ndarray, bval: np.ndarray) -> np.ndarray:
        """Gradient on patch faces: owner gradient with its normal part replaced by snGrad.
        deltaCoeffs = 1 / (n . (Cf - C_owner)), the wall-normal distance of the owner centre, as
        fvPatch::delta() / deltaCoeffs of a non-coupled patch (OpenFOAM >= 2.3, ESI v1912+). It is
        what the solver's own wall flux uses; 1 / |Cf - C_owner| differs on non-orthogonal wall cells
        (about 10 % on tets)."""
        sl = self.patch_slice(name)
        own = self.owner[sl]
        nrm = self.Sf[sl] / self.magSf[sl][:, None]
        delta = self.Cf[sl] - self.C[own]
        dc = 1.0 / np.maximum(np.abs(np.einsum("ij,ij->i", nrm, delta)), 1e-300)
        sn = dc[:, None] * (bval - cell[own])                   # (nf, k)
        G = grad[own]                                           # (nf, 3, k)
        nG = np.einsum("fi,fik->fk", nrm, G)
        return G + nrm[:, :, None] * (sn - nG)[:, None, :]

    def forces(self, names: list[str], p_b: dict, U: np.ndarray, gradU: np.ndarray, U_b: dict,
               nu_eff_b: dict, rho: float, p_ref: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(pressure force, viscous force) on the patches ``names``; p in kinematic units."""
        fp = np.zeros(3)
        fv = np.zeros(3)
        for name in names:
            sl = self.patch_slice(name)
            Sf = self.Sf[sl]
            fp += rho * np.sum(Sf * (np.asarray(p_b[name]) - p_ref)[:, None], axis=0)
            G = self.boundary_grad(name, U, gradU, U_b[name])   # (nf, 3, 3)
            T = G + np.transpose(G, (0, 2, 1))
            tr = np.trace(T, axis1=1, axis2=2)
            T = T - (tr / 3.0)[:, None, None] * np.eye(3)
            dev = -rho * np.asarray(nu_eff_b[name]).reshape(-1)[:, None, None] * T
            fv += np.einsum("fi,fij->j", Sf, dev)
        return fp, fv


DIRICHLET = ("fixedValue", "noSlip", "surfaceNormalFixedValue", "movingWallVelocity", "uniformFixedValue")


def _padded(faces) -> np.ndarray:
    """Face point lists -> (n_faces, kmax) array padded with -1."""
    lens = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    F = np.full((len(faces), int(lens.max(initial=0))), -1, dtype=np.int64)
    for k in np.unique(lens):
        idx = np.nonzero(lens == k)[0]
        F[idx, :k] = np.asarray([faces[i] for i in idx], dtype=np.int64)
    return F


class CellPointInterpolator:
    """Continuous, cell-order independent point interpolation of cell-centred fields, in the spirit
    of OpenFOAM's ``cellPoint``: vertex values from a weighted least-squares linear fit to the
    surrounding cell centres (exact for linear fields; the 2-ring of cells where the 1-ring is too
    small, e.g. on walls), or the mean of the adjacent Dirichlet face values on Dirichlet patches;
    then linear interpolation on the decomposition of every cell into tets (cell centre, face
    centre, face edge). On a face the result depends only on that face's vertex values, so it is
    continuous across faces and the same for any cell numbering. ``fit_dims`` = 2 for extruded 2D
    meshes (no fit in z)."""

    def __init__(self, mesh: FvMesh, faces, fit_dims: int, skip_patches=()):
        self.m = mesh
        self.fit_dims = fit_dims
        self.F = _padded(faces)
        nf, kmax = self.F.shape
        own, nei = mesh.owner, mesh.neighbour
        npts = len(mesh.points)
        # cell-point incidence
        cells = np.concatenate([np.repeat(own, kmax), np.repeat(nei, kmax)])
        pts = np.concatenate([self.F.ravel(), self.F[: mesh.n_int].ravel()])
        ok = pts >= 0
        key = np.unique(cells[ok] * npts + pts[ok])
        self.pc_cell, self.pc_point = key // npts, key % npts
        # tets (cell centre, face centre, edge a-b), faces of skipped (empty) patches left out
        use = np.ones(nf, dtype=bool)
        for name in skip_patches:
            use[mesh.patch_slice(name)] = False
        nc = mesh.n_cells
        tets = []
        lens = (self.F >= 0).sum(axis=1)
        for j in range(kmax):
            f = np.nonzero(use & (j < lens))[0]
            a = self.F[f, j]
            b = self.F[f, (j + 1) % lens[f]]
            tets.append(np.column_stack([own[f], nc + f, nc + nf + a, nc + nf + b]))
            i = f < mesh.n_int
            fi = f[i]
            tets.append(np.column_stack([nei[fi], nc + fi, nc + nf + a[i], nc + nf + b[i]]))
        self.tets = np.concatenate(tets)
        tet_cell = self.tets[:, 0]
        order = np.argsort(tet_cell, kind="stable")
        self.tets = self.tets[order]
        self.ptr = np.concatenate([[0], np.cumsum(np.bincount(tet_cell, minlength=nc))])
        fc = np.zeros((nf, 3))
        cnt = (self.F >= 0).sum(axis=1)
        for j in range(kmax):
            sel = self.F[:, j] >= 0
            fc[sel] += mesh.points[self.F[sel, j]]
        self.face_avg = fc / np.maximum(cnt, 1)[:, None]
        self.xyz = np.vstack([mesh.C, self.face_avg, mesh.points])
        self._ring2 = None

    # ---- vertex values
    def _fit(self, v, c, vals, npts):
        X = (self.m.C[c] - self.m.points[v])[:, : self.fit_dims]
        h2 = np.bincount(v, np.einsum("ij,ij->i", X, X), npts) / np.maximum(np.bincount(v, minlength=npts), 1)
        d = X / np.sqrt(np.maximum(h2[v], 1e-300))[:, None]
        w = 1.0 / (np.einsum("ij,ij->i", d, d) + 1e-2)
        R = np.column_stack([np.ones(len(v)), d])
        q = R.shape[1]
        M = np.zeros((npts, q, q))
        for i in range(q):
            for j in range(i, q):
                M[:, i, j] = M[:, j, i] = np.bincount(v, w * R[:, i] * R[:, j], npts)
        k = vals.shape[1]
        B = np.zeros((npts, q, k))
        for i in range(q):
            for j in range(k):
                B[:, i, j] = np.bincount(v, w * R[:, i] * vals[c, j], npts)
        tr = np.maximum(M[:, 0, 0], 1e-300)
        for i in range(1, q):
            M[:, i, i] += 1e-8 * tr                          # ridge: degenerate stencils -> mean
        empty = M[:, 0, 0] <= 1e-300
        M[empty] = np.eye(q)
        return np.linalg.solve(M, B)[:, 0, :]

    def _ring2_pairs(self, verts: np.ndarray):
        from scipy.sparse import csr_matrix

        m = self.m
        A = csr_matrix((np.ones(len(self.pc_cell)), (self.pc_cell, self.pc_point)),
                       shape=(m.n_cells, len(m.points)))
        sub = A.T.tocsr()[verts]                              # verts x cells (1-ring)
        ring2 = ((sub @ A) @ A.T).tocoo()                     # verts x cells sharing a point
        return verts[ring2.row], ring2.col

    def vertex_values(self, cell_vals: np.ndarray, dirichlet: dict) -> np.ndarray:
        """cell_vals (n, k); dirichlet {patch: (nf, k) face values} -> (n_points, k)."""
        npts = len(self.m.points)
        vals = cell_vals.reshape(len(cell_vals), -1)
        out = self._fit(self.pc_point, self.pc_cell, vals, npts)
        cnt = np.bincount(self.pc_point, minlength=npts)
        thin = np.nonzero((cnt > 0) & (cnt < self.fit_dims + 2))[0]
        if len(thin):
            if self._ring2 is None:
                self._ring2 = self._ring2_pairs(thin)
            v2, c2 = self._ring2
            out[thin] = self._fit(v2, c2, vals, npts)[thin]
        acc = np.zeros_like(out)
        n = np.zeros(npts)
        for name, fv in dirichlet.items():
            sl = self.m.patch_slice(name)
            Fp = self.F[sl]
            fv = np.asarray(fv, dtype=float).reshape(len(Fp), -1)
            for j in range(Fp.shape[1]):
                sel = Fp[:, j] >= 0
                np.add.at(acc, Fp[sel, j], fv[sel])
                np.add.at(n, Fp[sel, j], 1.0)
        hit = n > 0
        out[hit] = acc[hit] / n[hit][:, None]
        return out

    def node_values(self, cell_vals: np.ndarray, dirichlet: dict) -> np.ndarray:
        """Values at all interpolation nodes: cells, face centres (mean of the face's vertex values),
        points."""
        pv = self.vertex_values(cell_vals, dirichlet)
        F = self.F
        s = np.zeros((len(F), pv.shape[1]))
        for j in range(F.shape[1]):
            sel = F[:, j] >= 0
            s[sel] += pv[F[sel, j]]
        fv = s / np.maximum((F >= 0).sum(axis=1), 1)[:, None]
        return np.vstack([cell_vals.reshape(len(cell_vals), -1), fv, pv])

    # ---- point location
    def locate(self, X: np.ndarray, tree, hloc: np.ndarray, dims: int, tol: float = 1e-9):
        """(tet index, barycentric weights (n, 4)) of each point; raises ValueError for points
        outside the mesh (farther than 2 local cell sizes from every cell centre)."""
        tet_out = np.zeros(len(X), dtype=np.int64)
        lam_out = np.zeros((len(X), 4))
        for i, x in enumerate(X):
            best = None
            for k in (16, 64):
                k = min(k, self.m.n_cells)
                dist, cand = tree.query(x[:dims], k=k)
                cand = np.atleast_1d(cand)
                ids = np.concatenate([np.arange(self.ptr[c], self.ptr[c + 1]) for c in cand])
                T = self.xyz[self.tets[ids]]                       # (m, 4, 3)
                A = np.transpose(T[:, 1:] - T[:, :1], (0, 2, 1))   # columns = edges
                with np.errstate(all="ignore"):
                    lam = np.linalg.solve(A, (x - T[:, 0])[:, :, None])[:, :, 0]
                lam = np.column_stack([1.0 - lam.sum(axis=1), lam])
                lam[~np.isfinite(lam).all(axis=1)] = -np.inf
                j = int(np.argmax(lam.min(axis=1)))
                best = (ids[j], lam[j])
                if lam[j].min() >= -tol:
                    break
            if best[1].min() < -tol:
                d0 = float(np.atleast_1d(tree.query(x[:dims], k=1)[0])[0])
                c0 = int(np.atleast_1d(tree.query(x[:dims], k=1)[1])[0])
                if d0 > 2.0 * hloc[c0] + 1e-12:
                    raise ValueError(f"probe point {x[:dims].tolist()} is outside the mesh")
            tet_out[i], lam_out[i] = best
        return tet_out, lam_out
