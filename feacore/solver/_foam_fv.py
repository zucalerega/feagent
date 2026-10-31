"""Finite-volume post-processing of an OpenFOAM result in numpy (private helper of the backend).

OpenFOAM's own formulas, so the numbers match its function objects:

* face centres / area vectors and cell centres / volumes as ``primitiveMesh`` computes them
  (triangle fans about the face average point; pyramids about the estimated cell centre);
* ``Gauss linear`` gradients with ``surfaceInterpolation`` weights; the boundary value of the
  gradient gets its normal part from the patch ``snGrad`` (``gaussGrad::correctBoundaryConditions``);
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
        """Gradient on patch faces: owner gradient with its normal part replaced by snGrad
        (deltaCoeffs = 1 / |Cf - C_owner|, as fvPatch::deltaCoeffs)."""
        sl = self.patch_slice(name)
        own = self.owner[sl]
        nrm = self.Sf[sl] / self.magSf[sl][:, None]
        delta = self.Cf[sl] - self.C[own]
        dc = 1.0 / np.linalg.norm(delta, axis=1)
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
