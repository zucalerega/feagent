"""scikit-fem backend: the fixed solver.

``solve(case, **options) -> SkfemResult``. Linear elasticity on tri3/tri6/tet4/tet10 meshes
(``feacore.solver._skfem_mesh``) with P1/P2 Lagrange elements, isoparametric on quadratic files.
Stresses are recovered by L2 projection of the quadrature-point stresses onto the scalar P_k space
(continuous field used by ``probe`` / ``nodal`` / ``max``).

Supported here: formulations plane_stress, plane_strain, 3d; BCs fixed, displacement, symmetry,
pressure, traction, force; loads body_force. Not yet: axisymmetric, heat, thermal loads (they raise
NotImplementedError naming the missing piece).

Besides the ``Result`` interface the result keeps the raw system for ``feacore.diagnostics``:
``K`` / ``f`` (full, before Dirichlet elimination, thickness included), ``u``, ``D`` (constrained
dofs), ``basis`` / ``sbasis`` (vector / scalar), ``md`` (MeshData), ``sig_qp`` (quadrature-point
stresses per cell), ``dx`` (quadrature weights x thickness) and ``linear`` (LinearSolveReport).

Options: ``linear_solver`` (see ``feacore.solver.linsolve``), ``rtol`` for the iterative method.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from skfem import (Basis, BilinearForm, ElementTetP1, ElementTetP2, ElementTriP1, ElementTriP2,
                   ElementVector, FacetBasis, LinearForm, asm)
from skfem.helpers import dot

from feacore.case import BC, Case, CaseError, ELASTICITY_FIELDS, QoI, as_field
from feacore.solver._skfem_locate import Locator
from feacore.solver._skfem_mesh import MeshData, read_mesh
from feacore.solver.base import Result, SolveInfo
from feacore.solver.linsolve import solve_linear

__all__ = ["solve", "setup", "SkfemResult"]

_SUPPORTED = ("plane_stress", "plane_strain", "3d")
_VOIGT = ("sxx", "syy", "szz", "sxy", "syz", "sxz")
_SCALAR = {(2, 1): ElementTriP1, (2, 2): ElementTriP2, (3, 1): ElementTetP1, (3, 2): ElementTetP2}


def _lame(E: float, nu: float, formulation: str) -> tuple[float, float]:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    if formulation == "plane_stress":
        lam = 2 * lam * mu / (lam + 2 * mu)
    return lam, mu


def solve(case: Case, **options) -> "SkfemResult":
    t0 = time.perf_counter()
    case.check()
    a = case.analysis
    if a.physics != "elasticity":
        raise NotImplementedError("skfem backend: heat physics is not implemented yet")
    if a.formulation not in _SUPPORTED:
        raise NotImplementedError(f"skfem backend: formulation {a.formulation!r} is not implemented yet "
                                  f"(supported: {_SUPPORTED})")
    if any(l.type == "temperature" for l in case.loads):
        raise NotImplementedError("skfem backend: thermal loads are not implemented yet")
    res = SkfemResult(case, read_mesh(case.mesh_path(), case.dim, a.order), options)
    res.info.wall_time = time.perf_counter() - t0
    return res


def setup(case: Case) -> "SkfemResult":
    """Assembled system without solving (for setup checks that must run before a singular solve)."""
    case.check()
    return SkfemResult(case, read_mesh(case.mesh_path(), case.dim, case.analysis.order), {"setup_only": True})


class SkfemResult(Result):
    def __init__(self, case: Case, md: MeshData, options: dict):
        self.case, self.md = case, md
        a = case.analysis
        dim, order = case.dim, a.order
        self.dim = dim
        self.t = a.thickness if dim == 2 else 1.0
        self.basis = Basis(md.mesh, ElementVector(_SCALAR[(dim, order)]()), intorder=2 * order)
        self.sbasis = self.basis.with_element(_SCALAR[(dim, order)]())
        self.comp = self.basis.split_indices()
        self.warnings = list(md.warnings)
        self._fields: dict = {}
        self._locator = None

        self.cell_mat = self._cell_materials()
        self.K = self._stiffness()
        self.f = self._loads()
        D, xD = self._dirichlet()
        self.D, self.xD = D, xD
        self.setup_only = bool(options.get("setup_only", False))
        if self.setup_only:  # assembled system only (diagnostics prechecks); no solve, no fields
            self.u, self.linear = None, None
            self.dx = self.basis.dx * self.t
        else:
            self.u, self.linear = self._solve(options)
            self._recover()

        mesh = md.mesh
        measure = float(self.dx.sum() / self.t)
        ncells = mesh.t.shape[1]
        self.info = SolveInfo(
            backend="skfem", physics=a.physics, formulation=a.formulation, order=order, dim=dim,
            cell_type=md.cell_type, n_nodes=int(self.sbasis.N), n_cells=int(ncells),
            n_dofs=int(self.basis.N), measure=measure, h_rep=(measure / ncells) ** (1.0 / dim),
            wall_time=0.0, warnings=self.warnings + (self.linear.notes if self.linear else []))

    # ------------------------------------------------------------------ assembly
    def _cell_materials(self) -> np.ndarray:
        """(nel, 2) Lame parameters per cell; also sets ``cell_Enu`` (nel, 2)."""
        nel = self.md.mesh.t.shape[1]
        out = np.full((nel, 2), np.nan)
        self.cell_Enu = np.full((nel, 2), np.nan)
        regions = self.case.regions or [type("R", (), {"group": "*", "material": next(iter(self.case.materials))})]
        for r in reversed(regions):  # first matching region wins
            m = self.case.materials[r.material]
            cells = slice(None) if r.group == "*" else self.md.group(r.group).index
            out[cells] = _lame(m.E, m.nu, self.case.analysis.formulation)
            self.cell_Enu[cells] = (m.E, m.nu)
        if np.isnan(out).any():
            raise CaseError(f"{int(np.isnan(out[:, 0]).sum())} cells have no material region")
        return out

    def _qp(self, per_cell: np.ndarray) -> np.ndarray:
        return np.repeat(per_cell[:, None], self.basis.X.shape[1], axis=1)

    def _stiffness(self) -> sp.csr_matrix:
        lam, mu = self._qp(self.cell_mat[:, 0]), self._qp(self.cell_mat[:, 1])

        @BilinearForm
        def k(u, v, w):
            eu, ev = _sym(u.grad), _sym(v.grad)
            tr = sum(eu[i, i] for i in range(eu.shape[0]))
            trv = sum(ev[i, i] for i in range(ev.shape[0]))
            return w.lam * tr * trv + 2 * w.mu * np.einsum("ij...,ij...->...", eu, ev)

        return (asm(k, self.basis, lam=lam, mu=mu) * self.t).tocsr()

    def _facet_basis(self, group: str) -> FacetBasis:
        g = self.md.group(group)
        if g.kind != "facets":
            raise CaseError(f"group {group!r} must be a facet group for this BC (it is {g.kind})")
        return FacetBasis(self.md.mesh, self.basis.elem, facets=g.index, intorder=2 * self.case.analysis.order)

    def _loads(self) -> np.ndarray:
        f = np.zeros(self.basis.N)
        dim = self.dim
        for b in self.case.bcs:
            if b.type not in ("pressure", "traction", "force"):
                continue
            fb = self._facet_basis(b.group)
            if b.type == "pressure":
                @LinearForm
                def lf(v, w):
                    p = as_field(b.value, *w.x)
                    return -p * dot(w.n, v)
            else:
                vec = [float(c) for c in b.value]
                if b.type == "force":
                    area = asm(LinearForm(lambda v, w: v), fb.with_element(self.sbasis.elem)).sum() * self.t
                    if area <= 0:
                        raise CaseError(f"force BC on {b.group!r}: group has zero area")
                    vec = [c / area for c in vec]
                tr = np.array(vec)

                @LinearForm
                def lf(v, w):
                    return sum(tr[i] * v[i] for i in range(dim))
            f += asm(lf, fb) * self.t
        for l in self.case.loads:
            if l.type != "body_force":
                continue
            bf = np.array([float(c) for c in l.value])
            mask = np.zeros(self.md.mesh.t.shape[1])
            mask[slice(None) if l.group == "*" else self.md.group(l.group).index] = 1.0

            @LinearForm
            def lb(v, w):
                return w.mask * sum(bf[i] * v[i] for i in range(dim))
            f += asm(lb, self.basis, mask=self._qp(mask)) * self.t
        return f

    def group_dofs(self, group: str) -> np.ndarray:
        """All vector dofs on a group of any dimension (cells, facets, edges, vertices)."""
        g = self.md.group(group)
        b = self.basis
        if g.kind == "facets":
            return b.get_dofs(facets=g.index).flatten()
        if g.kind == "cells":
            return np.unique(b.element_dofs[:, g.index])
        if g.kind == "vertices":
            return b.nodal_dofs[:, g.index].ravel()
        verts = np.unique(self.md.mesh.edges[:, g.index])
        out = [b.nodal_dofs[:, verts].ravel()]
        if b.edge_dofs.size:
            out.append(b.edge_dofs[:, g.index].ravel())
        return np.concatenate(out)

    def _component_of(self) -> np.ndarray:
        c = np.empty(self.basis.N, dtype=int)
        for i, idx in enumerate(self.comp):
            c[idx] = i
        return c

    def bc_dofs(self, b: BC) -> np.ndarray:
        if b.point is not None:
            p = np.asarray(b.point[: self.dim], dtype=float)
            v = int(np.argmin(np.linalg.norm(self.md.mesh.p.T - p, axis=1)))
            return self.basis.nodal_dofs[:, v]
        return self.group_dofs(b.group)

    def _dirichlet(self) -> tuple[np.ndarray, np.ndarray]:
        comp = self._component_of()
        X = self.basis.doflocs
        xD = np.full(self.basis.N, np.nan)
        for b in self.case.bcs:
            if b.type not in ("fixed", "displacement", "symmetry"):
                continue
            dofs = self.bc_dofs(b)
            if b.type == "fixed":
                xD[dofs] = 0.0
            elif b.type == "symmetry":
                axis = self._symmetry_axis(b.group)
                xD[dofs[comp[dofs] == axis]] = 0.0
            else:
                vals = b.value if isinstance(b.value, list) else [b.value] * len(b.components)
                for c, v in zip(b.components, vals):
                    sel = dofs[comp[dofs] == c]
                    xD[sel] = as_field(v, *X[:, sel])
        D = np.nonzero(~np.isnan(xD))[0]
        return D, xD[D]

    def _symmetry_axis(self, group: str) -> int:
        fb = self._facet_basis(group)
        n = np.abs(fb.normals.reshape(self.dim, -1)).mean(axis=1)
        axis = int(np.argmax(n))
        if n[axis] < 0.999:
            raise CaseError(f"symmetry BC on {group!r}: the facets are not planar and axis-aligned "
                            f"(mean |n| = {np.round(n, 3).tolist()})")
        return axis

    def _solve(self, options):
        N = self.basis.N
        u = np.zeros(N)
        u[self.D] = self.xD
        I = np.setdiff1d(np.arange(N), self.D)
        rhs = self.f[I] - self.K[I][:, self.D] @ self.xD
        x, report = solve_linear(self.K[I][:, I], rhs, options.get("linear_solver", "auto"),
                                 symmetric=True, rtol=options.get("rtol", 1e-10), return_report=True)
        u[I] = x
        return u, report

    # ------------------------------------------------------------------ stresses
    def _recover(self):
        """Quadrature-point stresses (raw, per cell) and their L2 projection onto P_k (nodal)."""
        uh = self.basis.interpolate(self.u)
        G = np.asarray(uh.grad)                               # (dim, dim, nel, nqp)
        eps = 0.5 * (G + np.swapaxes(G, 0, 1))
        lam, mu = self._qp(self.cell_mat[:, 0]), self._qp(self.cell_mat[:, 1])
        tr = sum(eps[i, i] for i in range(self.dim))
        sig = {}
        names = ("x", "y", "z")
        for i in range(self.dim):
            sig["s" + names[i] * 2] = lam * tr + 2 * mu * eps[i, i]
        sig["sxy"] = 2 * mu * eps[0, 1]
        if self.dim == 3:
            sig["syz"], sig["sxz"] = 2 * mu * eps[1, 2], 2 * mu * eps[0, 2]
        else:
            zero = np.zeros_like(tr)
            sig["syz"] = sig["sxz"] = zero
            if self.case.analysis.formulation == "plane_strain":
                sig["szz"] = lam * tr          # nu (sxx + syy)
            else:
                sig["szz"] = zero
        self.sig_qp = sig
        self.eps_qp = eps
        self.dx = self.basis.dx * self.t                     # (nel, nqp)
        self.u_qp = np.asarray(uh)                      # (dim, nel, nqp)

        M = asm(BilinearForm(lambda u, v, w: u * v), self.sbasis).tocsc()
        Minv = factorized(M)
        rhs = LinearForm(lambda v, w: w.s * v)
        self.S = np.column_stack([Minv(asm(rhs, self.sbasis, s=sig[k])) for k in _VOIGT])  # (n_sdofs, 6)
        self.U = np.column_stack([self.u[c] for c in self.comp] + [np.zeros(self.sbasis.N)] * (3 - self.dim))

    def _tensor(self, s: np.ndarray) -> np.ndarray:
        T = np.zeros((len(s), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
        T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
        T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
        return T

    def derive(self, field: str, U: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Field from displacement rows U (n, 3) and stress rows S (n, 6)."""
        if field not in ELASTICITY_FIELDS:
            raise CaseError(f"unknown elasticity field {field!r}")
        if field in ("ux", "uy", "uz"):
            return U[:, "xyz".index(field[1])]
        if field == "u_mag":
            return np.linalg.norm(U, axis=1)
        if field in _VOIGT:
            return S[:, _VOIGT.index(field)]
        if field == "stt":
            raise CaseError("stt (hoop stress) exists only in axisymmetric analyses")
        T = self._tensor(S)
        if field == "von_mises":
            d = T - np.trace(T, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
            return np.sqrt(1.5 * np.einsum("nij,nij->n", d, d))
        ev = np.linalg.eigvalsh(T)[:, ::-1]
        if field == "tresca":
            return ev[:, 0] - ev[:, 2]
        return ev[:, {"s1": 0, "s2": 1, "s3": 2}[field]]

    def qp_field(self, field: str) -> np.ndarray:
        """Raw (unrecovered) field at quadrature points, (nel, nqp)."""
        shape = self.dx.shape
        U = np.zeros((self.dx.size, 3))
        U[:, : self.dim] = self.u_qp.reshape(self.dim, -1).T
        S = np.column_stack([self.sig_qp[k].ravel() for k in _VOIGT])
        return self.derive(field, U, S).reshape(shape)

    # ------------------------------------------------------------------ Result API
    def nodal(self, field: str):
        if field not in self._fields:
            self._fields[field] = self.derive(field, self.U, self.S)
        return self.sbasis.doflocs.T.copy(), self._fields[field].copy()

    def probe(self, field: str, points) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))[:, : self.dim]
        if self._locator is None:
            self._locator = Locator(self.md.mesh, self.md.bbox_diag)
        cell, xi, _ = self._locator.locate(pts)
        e = self.sbasis.elem
        edofs = self.sbasis.element_dofs
        U = np.zeros((len(pts), 3))
        S = np.zeros((len(pts), 6))
        for j in range(edofs.shape[0]):
            phi, _ = e.lbasis(xi, j)
            d = edofs[j, cell]
            U += phi[:, None] * self.U[d]
            S += phi[:, None] * self.S[d]
        return self.derive(field, U, S)

    def group_nodes(self, group: str | None) -> np.ndarray:
        """Scalar-space nodes of a group (None / "*" = all)."""
        if group in (None, "*"):
            return np.arange(self.sbasis.N)
        return np.unique(self._scalar_nodes(group))

    def _scalar_nodes(self, group: str) -> np.ndarray:
        g = self.md.group(group)
        sb = self.sbasis
        if g.kind == "facets":
            return sb.get_dofs(facets=g.index).flatten()
        if g.kind == "cells":
            return np.unique(sb.element_dofs[:, g.index])
        if g.kind == "vertices":
            return sb.nodal_dofs[:, g.index].ravel()
        verts = np.unique(self.md.mesh.edges[:, g.index])
        out = [sb.nodal_dofs[:, verts].ravel()]
        if sb.edge_dofs.size:
            out.append(sb.edge_dofs[:, g.index].ravel())
        return np.concatenate(out)

    def reaction_vector(self) -> np.ndarray:
        """Nodal reactions K u - f (non-zero only on constrained dofs, up to round-off)."""
        return self.K @ self.u - self.f

    def reaction(self, group: str, component: int = 0) -> float:
        r = self.reaction_vector()
        dofs = self.group_dofs(group)
        return float(r[np.intersect1d(dofs, self.comp[int(component)])].sum())

    def strain_energy(self) -> float:
        return float(0.5 * self.u @ (self.K @ self.u))

    def integral(self, field: str, group: str | None, mean: bool = False) -> float:
        vals = self.qp_field(field)
        w = self.dx
        if group not in (None, "*"):
            g = self.md.group(group)
            if g.kind != "cells":
                raise NotImplementedError("skfem backend: integrals over facet groups are not implemented yet")
            vals, w = vals[g.index], w[g.index]
        total = float((vals * w).sum())
        return total / float(w.sum()) if mean else total

    def evaluate(self, qoi: QoI) -> float:
        t = qoi.type
        if t == "point":
            return float(self.probe(qoi.field, [qoi.point])[0])
        if t in ("max", "min", "max_abs"):
            _, v = self.nodal(qoi.field)
            v = v[self.group_nodes(qoi.group)]
            return float(v.max() if t == "max" else v.min() if t == "min" else np.abs(v).max())
        if t == "reaction":
            return self.reaction(qoi.group, qoi.component or 0)
        if t in ("integral", "mean"):
            return self.integral(qoi.field, qoi.group, mean=(t == "mean"))
        if t == "strain_energy":
            return self.strain_energy()
        raise CaseError(f"unknown QoI type {t!r}")


def _sym(G):
    return 0.5 * (G + np.swapaxes(G, 0, 1))
