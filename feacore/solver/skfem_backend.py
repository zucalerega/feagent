"""scikit-fem backend: the FIXED solver (``feacore.solver.solve(case, backend="skfem")``).

Physics / formulations
  elasticity: plane_stress, plane_strain, axisymmetric (x = r, y = z), 3d
  heat:       2d, axisymmetric, 3d
Interpolation: Lagrange P1 / P2 (``analysis.order``), independent of the mesh order. A quadratic
mesh (tri6 / tet10) with order 2 is isoparametric (curved cells); order 1 always runs on the
straight-sided corner mesh. Plane formulations multiply stiffness and loads by
``analysis.thickness``; axisymmetric integrals carry 2*pi*r, so forces, reactions and energies
are full 360-degree totals.

Stress / flux recovery (default ``analysis.options["stress_recovery"] = "l2"``): global L2
projection of each stress (flux) component, computed at the quadrature points from the FE
solution, onto the continuous scalar Lagrange space of the solution order (mass matrix solved
with the configured direct solver). von Mises, Tresca and principal stresses are computed
pointwise from the recovered components.

Linear solves go through ``feacore.solver.linsolve.solve_linear`` (scipy / pardiso / cudss /
auto), selected by ``analysis.options["linear_solver"]``, a ``linear_solver=`` keyword, or the
``FEACORE_LINEAR_SOLVER`` environment variable. See docs/solver_skfem.md.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTetP1, ElementTetP2, ElementTriP1, ElementTriP2,
                   ElementVector, FacetBasis, Functional, LinearForm)
from skfem.helpers import ddot, dot, grad, sym_grad, trace

from feacore.case import ELASTICITY_FIELDS, HEAT_FIELDS, Case, CaseError, QoI, as_field
from feacore.solver import _skfem_assembly as fast
from feacore.solver._skfem_locate import Locator
from feacore.solver._skfem_mesh import MeshData, MeshError, read_mesh
from feacore.solver.base import Result, SolveInfo
from feacore.solver.linsolve import solve_linear

KNOWN_OPTIONS = ("linear_solver", "stress_recovery", "intorder", "axis_constraint", "assembly")
_ELEMS = {(2, 1): ElementTriP1, (2, 2): ElementTriP2, (3, 1): ElementTetP1, (3, 2): ElementTetP2}
_VTK = {(2, 1): "triangle", (2, 2): "triangle6", (3, 1): "tetra", (3, 2): "tetra10"}
TWO_PI = 2.0 * math.pi

# stress components stored per formulation (others are identically zero)
_STRESS_COMPS = {
    "plane_stress": ("sxx", "syy", "sxy"),
    "plane_strain": ("sxx", "syy", "sxy", "szz"),
    "axisymmetric": ("sxx", "syy", "sxy", "stt"),
    "3d": ("sxx", "syy", "szz", "sxy", "syz", "sxz"),
}


class SolverError(RuntimeError):
    """The solve failed numerically."""


class InsufficientConstraints(CaseError):
    """The Dirichlet BCs leave a rigid-body mode free (a setup error: the stiffness is singular)."""


# --------------------------------------------------------------------------- forms


@BilinearForm
def _k_elastic(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return w.c * (w.lam * trace(eu) * trace(ev) + 2.0 * w.mu * ddot(eu, ev))


@BilinearForm
def _k_axi(u, v, w):
    r = w.x[0]
    eu, ev = sym_grad(u), sym_grad(v)
    hu, hv = u[0] / r, v[0] / r
    tu, tv = trace(eu) + hu, trace(ev) + hv
    return w.c * (w.lam * tu * tv + 2.0 * w.mu * (ddot(eu, ev) + hu * hv))


def _dgrad_form(i):
    return LinearForm(lambda v, w: w.c * v.grad[i])


_F_DGRAD = [_dgrad_form(i) for i in range(3)]  # int c * d v / d x_i


@BilinearForm
def _k_heat(u, v, w):
    return w.c * w.k * dot(grad(u), grad(v))


@BilinearForm
def _m_weighted(u, v, w):
    return w.c * w.h * u * v


@BilinearForm
def _mass(u, v, w):
    return u * v


@LinearForm
def _f_scalar(v, w):
    return w.c * w.g * v


@LinearForm
def _rhs_proj(v, w):
    return w.s * v


@Functional
def _measure_w(w):
    return w.c


# --------------------------------------------------------------------------- helpers


def _qp(basis, arr):
    """Broadcast per-element values (nel,) to quadrature-point arrays (nel, nqp)."""
    return np.broadcast_to(np.asarray(arr, dtype=float)[:, None], (basis.nelems, basis.X.shape[1])).copy()


def _coords(basis):
    return np.asarray(basis.global_coordinates())  # (dim, nel, nqp)


def _eval(value, x):
    """Scalar / expression at coordinate arrays x (dim, ...)."""
    z = x[2] if x.shape[0] == 3 else None
    return as_field(value, x[0], x[1], z)


def _principal(s):
    """s: dict of full 3D components (arrays) -> (s1, s2, s3) sorted descending."""
    T = np.stack([np.stack([s["xx"], s["xy"], s["xz"]], -1),
                  np.stack([s["xy"], s["yy"], s["yz"]], -1),
                  np.stack([s["xz"], s["yz"], s["zz"]], -1)], -2)
    ev = np.linalg.eigvalsh(T)  # ascending
    return ev[..., 2], ev[..., 1], ev[..., 0]


def _derived_stress(comp: dict, name: str):
    """comp: field name -> array for the 7 components (zeros where absent)."""
    if name == "stt":
        return comp["stt"]
    full = _full_tensor(comp)
    if name == "von_mises":
        return np.sqrt(np.maximum(0.0, 0.5 * ((full["xx"] - full["yy"]) ** 2 + (full["yy"] - full["zz"]) ** 2
                                              + (full["zz"] - full["xx"]) ** 2)
                                  + 3.0 * (full["xy"] ** 2 + full["yz"] ** 2 + full["xz"] ** 2)))
    s1, s2, s3 = _principal(full)
    return {"s1": s1, "s2": s2, "s3": s3, "tresca": s1 - s3}[name]


def _full_tensor(comp):
    # axisymmetric: (r, z, theta) frame; the hoop stress is the third normal component
    if comp.get("_axi", False):
        zero = np.zeros_like(comp["sxx"])
        return {"xx": comp["sxx"], "yy": comp["syy"], "zz": comp["stt"], "xy": comp["sxy"],
                "yz": zero, "xz": zero}
    return {"xx": comp["sxx"], "yy": comp["syy"], "zz": comp["szz"], "xy": comp["sxy"],
            "yz": comp["syz"], "xz": comp["sxz"]}


# --------------------------------------------------------------------------- model


class _Model:
    """Mesh, bases, materials and group resolution shared by assembly, recovery and results."""

    def __init__(self, case: Case, opts: dict):
        self.case = case
        a = case.analysis
        self.a = a
        self.opts = opts
        self.physics = a.physics
        self.form = a.formulation
        self.axi = a.formulation == "axisymmetric"
        self.dim = case.dim
        self.order = int(a.order)
        self.warnings: list[str] = []
        unknown = sorted(set(opts) - set(KNOWN_OPTIONS))
        if unknown:
            self.warnings.append(f"unknown analysis options ignored: {unknown}")
        path = case.mesh_path()
        if not Path(path).exists():
            raise MeshError(f"mesh file not found: {str(path)!r}")
        self.md: MeshData = read_mesh(path, self.dim, self.order)
        self.warnings += self.md.warnings
        self.mesh = self.md.mesh
        if self.axi and self.mesh.p[0].min() < -1e-9 * self.md.bbox_diag:
            raise CaseError("axisymmetric mesh must lie in x = r >= 0 (axis at x = 0)")
        self.selem = _ELEMS[(self.dim, self.order)]()
        default_q = 2 * self.order + (1 if self.axi else 0)
        self.intorder = int(opts.get("intorder") or default_q)
        self.sb = Basis(self.mesh, self.selem, intorder=self.intorder)
        self._check_jacobians()
        self.ncomp = self.dim if self.physics == "elasticity" else 1
        ns = self.sb.N
        # vector unknowns are interleaved per node: dof = ncomp * scalar_dof + component
        self.split = [np.arange(ns, dtype=np.int64) * self.ncomp + k for k in range(self.ncomp)]
        self.N = ns * self.ncomp
        self.xq = _coords(self.sb)
        self.nel = self.mesh.nelements
        self._materials()

    def _check_jacobians(self):
        d = self.sb.mapping.detDF(self.sb.X)  # (nel, nqp), signed
        dmax = float(np.abs(d).max()) if d.size else 1.0
        flipped = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
        tiny = np.abs(d).min(axis=1) <= 1e-12 * dmax
        if flipped.any():
            self.warnings.append(f"mesh: {int(flipped.sum())} curved cells have a Jacobian that changes sign "
                                 f"(inverted / tangled high-order cells): results are unreliable")
        if tiny.any():
            self.warnings.append(f"mesh: {int(tiny.sum())} cells are degenerate (near-zero Jacobian)")

    # ---- integration weight: thickness (plane), 2*pi*r (axisymmetric), 1 (3d)
    def wfac(self, x):
        if self.axi:
            return TWO_PI * x[0]
        if self.dim == 2:
            return np.full(x.shape[1:], float(self.a.thickness))
        return np.ones(x.shape[1:])

    def cells_of(self, group: str | None) -> np.ndarray:
        if group in (None, "*"):
            return np.arange(self.nel)
        g = self.md.group(group)
        if g.kind != "cells":
            raise CaseError(f"group {group!r} is a {g.kind} group; a cell group ({self.dim}D) is required here. "
                            f"Available: {self.md.describe_groups()}")
        return g.index

    def facets_of(self, group: str) -> np.ndarray:
        g = self.md.group(group)
        if g.kind != "facets":
            raise CaseError(f"group {group!r} is a {g.kind} group; a facet group "
                            f"({'curves' if self.dim == 2 else 'surfaces'}) is required here. "
                            f"Available: {self.md.describe_groups()}")
        return g.index

    def sdofs_of(self, group: str | None) -> np.ndarray:
        """Scalar dofs (nodes of the solution space) of any group."""
        sb, mesh = self.sb, self.mesh
        if group in (None, "*"):
            return np.arange(sb.N)
        g = self.md.group(group)
        if g.kind == "cells":
            return np.unique(sb.element_dofs[:, g.index])
        if g.kind == "facets":
            return np.unique(np.asarray(sb.get_dofs(facets=g.index).all()))
        if g.kind == "edges":
            d = [sb.dofs.nodal_dofs[:, mesh.edges[:, g.index].ravel()].ravel()]
            if sb.dofs.edge_dofs.size:
                d.append(sb.dofs.edge_dofs[:, g.index].ravel())
            return np.unique(np.concatenate(d))
        if g.kind == "vertices":
            return np.unique(sb.dofs.nodal_dofs[:, g.index].ravel())
        raise CaseError(f"unsupported group kind {g.kind}")  # pragma: no cover

    def nearest_sdof(self, point) -> tuple[int, float]:
        p = np.asarray(point, dtype=float)[: self.dim]
        d = np.linalg.norm(self.sb.doflocs - p[:, None], axis=0)
        i = int(np.argmin(d))
        return i, float(d[i])

    def _materials(self):
        case = self.case
        names = list(case.materials)
        mat = np.full(self.nel, -1, dtype=np.int64)
        if not case.regions:
            if len(names) > 1:
                self.warnings.append(f"several materials but no regions: using {names[0]!r} everywhere")
            mat[:] = 0
        else:
            for r in reversed(case.regions):  # first matching region wins (Case.material_for)
                mat[self.cells_of(r.group)] = names.index(r.material)
            if (mat < 0).any():
                raise CaseError(f"{int((mat < 0).sum())} cells are not covered by any region; add a region "
                                f"for them (or group '*'). Cell groups: {self.md.describe_groups()}")
        self.mat = mat
        M = [case.materials[n] for n in names]
        self.E = np.array([m.E for m in M])[mat]
        self.nu = np.array([m.nu for m in M])[mat]
        self.alpha = np.array([m.alpha for m in M])[mat]
        self.k = np.array([m.k for m in M])[mat]
        if self.physics == "elasticity":
            E, nu = self.E, self.nu
            self.lam3 = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.mu = E / (2 * (1 + nu))
            if self.form == "plane_stress":
                self.lam = 2 * self.lam3 * self.mu / (self.lam3 + 2 * self.mu)
                self.beta = (2 * self.lam + 2 * self.mu) * self.alpha        # = E alpha / (1 - nu)
            else:
                self.lam = self.lam3
                self.beta = (3 * self.lam3 + 2 * self.mu) * self.alpha       # = E alpha / (1 - 2 nu)

    # ---- facet bases
    def facet_basis(self, facets):
        return FacetBasis(self.mesh, self.selem, facets=facets, intorder=self.intorder)


# --------------------------------------------------------------------------- Dirichlet collection


class _Constraints:
    def __init__(self):
        self.idx: list[np.ndarray] = []
        self.val: list[np.ndarray] = []
        self.by_group: dict[str, list[np.ndarray]] = {}

    def add(self, idx, val, group=None):
        idx = np.asarray(idx, dtype=np.int64).ravel()
        val = np.broadcast_to(np.asarray(val, dtype=float), idx.shape).ravel()
        self.idx.append(idx)
        self.val.append(val)
        if group is not None:
            self.by_group.setdefault(group, []).append(idx)

    def finalize(self, warnings: list):
        if not self.idx:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.concatenate(self.idx)
        val = np.concatenate(self.val)
        # last definition wins; warn about conflicting values
        order = np.arange(len(idx))
        srt = np.lexsort((order, idx))
        idx_s, val_s = idx[srt], val[srt]
        last = np.r_[idx_s[1:] != idx_s[:-1], True]
        first = np.r_[True, idx_s[1:] != idx_s[:-1]]
        grp_first_val = np.repeat(val_s[first], np.diff(np.r_[np.nonzero(first)[0], len(idx_s)]))
        scale = max(1e-300, float(np.abs(val_s).max()) if len(val_s) else 1.0)
        conflict = np.abs(val_s - grp_first_val) > 1e-12 * scale
        if conflict.any():
            n = len(np.unique(idx_s[conflict]))
            warnings.append(f"{n} dofs are constrained by several Dirichlet BCs with different values; "
                            f"the last BC in the list wins")
        return idx_s[last], val_s[last]


# --------------------------------------------------------------------------- elasticity


def _assemble_elasticity(md: _Model):
    case, sb = md.case, md.sb
    c = md.wfac(md.xq)
    lam, mu = _qp(sb, md.lam), _qp(sb, md.mu)
    if md.opts.get("assembly", "fast") == "skfem":  # reference implementation (slow)
        ub = Basis(md.mesh, ElementVector(md.selem), intorder=md.intorder)
        perm = np.empty(md.N, dtype=np.int64)
        for k, ik in enumerate(ub.split_indices()):
            perm[md.split[k]] = ik
        K = (_k_axi if md.axi else _k_elastic).assemble(ub, lam=lam, mu=mu, c=c)[perm][:, perm].tocsr()
    else:
        K = fast.elastic_stiffness(sb, md.split, lam, mu, c, md.axi, r=md.xq[0])
    f = np.zeros(md.N)
    dim = md.dim

    # body forces and thermal loads
    dT = np.zeros_like(c)
    for ld in case.loads:
        E = md.cells_of(ld.group)
        mask = np.zeros(md.nel, dtype=bool)
        mask[E] = True
        if ld.type == "body_force":
            for i in range(dim):
                gi = np.zeros_like(c)
                gi[mask] = _eval(ld.value[i], md.xq[:, mask])
                f[md.split[i]] += _f_scalar.assemble(sb, c=c, g=gi)
        elif ld.type == "temperature":
            dT[mask] = _eval(ld.value, md.xq[:, mask]) - md.a.T_ref
    sth = _qp(sb, md.beta) * dT
    if np.any(sth != 0):  # int c * beta * alpha * dT * div(v)  (+ v_r / r axisymmetric)
        for i in range(dim):
            f[md.split[i]] += _F_DGRAD[i].assemble(sb, c=c * sth)
        if md.axi:
            f[md.split[0]] += _f_scalar.assemble(sb, c=c, g=sth / md.xq[0])
    md.sth = sth
    md.dT = dT

    # Neumann BCs
    for i_bc, bc in enumerate(case.bcs):
        if bc.type not in ("pressure", "traction", "force"):
            continue
        F = md.facets_of(bc.group)
        fb = md.facet_basis(F)
        xf = _coords(fb)
        cf = md.wfac(xf)
        if bc.type == "pressure":
            if np.any(md.mesh.f2t[1, F] >= 0):
                raise CaseError(f"bcs[{i_bc}] pressure on group {bc.group!r}: it contains interior facets, "
                                f"so the outward normal is undefined; use a boundary group or traction")
            p = _eval(bc.value, xf)
            n = fb.normals
            g = [-p * n[i] for i in range(dim)]
        elif bc.type == "traction":
            g = [_eval(bc.value[i], xf) for i in range(dim)]
        else:  # force: total force spread uniformly per area (area incl. thickness / 2 pi r)
            meas = _measure_w.assemble(fb, c=cf)
            if meas <= 0:
                raise CaseError(f"bcs[{i_bc}] force: group {bc.group!r} has zero measure")
            g = [np.full(cf.shape, float(bc.value[i]) / meas) for i in range(dim)]
        for i in range(dim):
            f[md.split[i]] += _f_scalar.assemble(fb, c=cf, g=g[i])

    # Dirichlet BCs
    cons = _Constraints()
    diag = md.md.bbox_diag
    for i_bc, bc in enumerate(case.bcs):
        if bc.type in ("fixed", "displacement"):
            if bc.point is not None:
                s, d = md.nearest_sdof(bc.point)
                if d > 0.01 * diag:
                    md.warnings.append(f"bcs[{i_bc}] {bc.type} point {bc.point}: nearest node is {d:.3g} away")
                S = np.array([s])
                gname = None
            else:
                S = md.sdofs_of(bc.group)
                gname = bc.group
            if len(S) == 0:
                raise CaseError(f"bcs[{i_bc}] {bc.type}: group {bc.group!r} has no nodes")
            if bc.type == "fixed":
                for k in range(dim):
                    cons.add(md.split[k][S], 0.0, gname)
            else:
                comps = bc.components
                vals = bc.value if isinstance(bc.value, list) else [bc.value] * len(comps)
                X = md.sb.doflocs[:, S]
                for k, v in zip(comps, vals):
                    cons.add(md.split[k][S], _eval(v, X), gname)
        elif bc.type == "symmetry":
            F = md.facets_of(bc.group)
            ax = _symmetry_axis(md, F, bc.group, i_bc)
            S = md.sdofs_of(bc.group)
            cons.add(md.split[ax][S], 0.0, bc.group)
    if md.axi and md.opts.get("axis_constraint", True):
        on_axis = np.nonzero(np.abs(md.sb.doflocs[0]) <= 1e-9 * diag)[0]
        if len(on_axis):
            cons.add(md.split[0][on_axis], 0.0, None)
            md.warnings.append(f"info: axisymmetric: u_r = 0 imposed on {len(on_axis)} nodes on the axis")
    return K, f, cons


def _symmetry_axis(md: _Model, F, group, i_bc) -> int:
    mesh = md.mesh
    P = mesh.p[:, mesh.facets[:, F]]  # (dim, nv, nf)
    if md.dim == 2:
        t = P[:, 1] - P[:, 0]
        n = np.vstack([-t[1], t[0]])
    else:
        n = np.cross((P[:, 1] - P[:, 0]).T, (P[:, 2] - P[:, 0]).T).T
    n = n / np.linalg.norm(n, axis=0)
    a = np.abs(n).mean(axis=1)
    ax = int(np.argmax(a))
    S = md.sdofs_of(group)
    coord = md.sb.doflocs[ax, S]
    if np.abs(n[ax]).min() < 1 - 1e-6 or np.ptp(coord) > 1e-6 * md.md.bbox_diag:
        raise CaseError(f"bcs[{i_bc}] symmetry: group {group!r} is not a planar facet group normal to a "
                        f"coordinate axis; use a displacement BC with explicit components instead")
    return ax


def _recover_elasticity(md: _Model, u: np.ndarray, solver_opts: dict) -> dict:
    S, _ = qp_stress_elasticity(md, u)
    return _project(md, S, solver_opts)


def qp_stress_elasticity(md: _Model, u: np.ndarray) -> tuple[dict, np.ndarray]:
    """Raw (unrecovered) stresses at the quadrature points, {component: (nel, nqp)}, and the strain
    tensor e (dim, dim, nel, nqp). Also used by ``feacore.diagnostics`` (error estimation)."""
    Uk = [md.sb.interpolate(u[md.split[k]]) for k in range(md.dim)]
    G = np.stack([np.asarray(U.grad) for U in Uk])  # G[i, j] = d u_i / d x_j, (dim, dim, nel, nqp)
    lam, mu = _qp(md.sb, md.lam), _qp(md.sb, md.mu)
    sth = md.sth
    e = 0.5 * (G + np.swapaxes(G, 0, 1))
    S = {}
    if md.dim == 3:
        tr = e[0, 0] + e[1, 1] + e[2, 2]
        S["sxx"] = lam * tr + 2 * mu * e[0, 0] - sth
        S["syy"] = lam * tr + 2 * mu * e[1, 1] - sth
        S["szz"] = lam * tr + 2 * mu * e[2, 2] - sth
        S["sxy"] = 2 * mu * e[0, 1]
        S["syz"] = 2 * mu * e[1, 2]
        S["sxz"] = 2 * mu * e[0, 2]
    elif md.axi:
        ett = np.asarray(Uk[0]) / md.xq[0]
        tr = e[0, 0] + e[1, 1] + ett
        S["sxx"] = lam * tr + 2 * mu * e[0, 0] - sth
        S["syy"] = lam * tr + 2 * mu * e[1, 1] - sth
        S["sxy"] = 2 * mu * e[0, 1]
        S["stt"] = lam * tr + 2 * mu * ett - sth
    else:
        tr = e[0, 0] + e[1, 1]
        S["sxx"] = lam * tr + 2 * mu * e[0, 0] - sth
        S["syy"] = lam * tr + 2 * mu * e[1, 1] - sth
        S["sxy"] = 2 * mu * e[0, 1]
        if md.form == "plane_strain":
            S["szz"] = _qp(md.sb, md.lam3) * tr - sth
    return S, e


def _project(md: _Model, S: dict, solver_opts: dict) -> dict:
    method = md.opts.get("stress_recovery", "l2")
    if method != "l2":
        raise CaseError(f"unknown stress_recovery {method!r} (supported: 'l2')")
    if not hasattr(md, "_M"):
        md._M = _mass.assemble(md.sb) if md.opts.get("assembly", "fast") == "skfem" else fast.mass(md.sb)
    names = list(S)
    B = np.column_stack([_rhs_proj.assemble(md.sb, s=S[n]) for n in names])
    X, rep = solve_linear(md._M, B, symmetric=True, return_report=True, **solver_opts)
    X = X.reshape(md.sb.N, len(names))
    md.proj_report = rep
    return {n: X[:, i].copy() for i, n in enumerate(names)}


# --------------------------------------------------------------------------- heat


def _assemble_heat(md: _Model):
    case, sb = md.case, md.sb
    c = md.wfac(md.xq)
    if md.opts.get("assembly", "fast") == "skfem":
        K = _k_heat.assemble(sb, k=_qp(sb, md.k), c=c)
    else:
        K = fast.scalar_stiffness(sb, _qp(sb, md.k), c)
    f = np.zeros(md.N)
    for ld in case.loads:
        if ld.type != "source":
            continue
        E = md.cells_of(ld.group)
        mask = np.zeros(md.nel, dtype=bool)
        mask[E] = True
        g = np.zeros_like(c)
        g[mask] = _eval(ld.value, md.xq[:, mask])
        f += _f_scalar.assemble(sb, c=c, g=g)
    md.sth = None
    md.has_robin = False
    cons = _Constraints()
    for i_bc, bc in enumerate(case.bcs):
        if bc.type in ("flux", "convection"):
            F = md.facets_of(bc.group)
            fb = md.facet_basis(F)
            xf = _coords(fb)
            cf = md.wfac(xf)
            if bc.type == "flux":
                f += _f_scalar.assemble(fb, c=cf, g=_eval(bc.value, xf))
            else:
                h = np.full(cf.shape, float(bc.h))
                K = K + _m_weighted.assemble(fb, c=cf, h=h)
                md.has_robin = md.has_robin or float(bc.h) > 0
                f += _f_scalar.assemble(fb, c=cf, g=h * float(bc.T_inf))
        elif bc.type == "temperature":
            if bc.point is not None:
                s, d = md.nearest_sdof(bc.point)
                if d > 0.01 * md.md.bbox_diag:
                    md.warnings.append(f"bcs[{i_bc}] temperature point {bc.point}: nearest node is {d:.3g} away")
                S, gname = np.array([s]), None
            else:
                S, gname = md.sdofs_of(bc.group), bc.group
            cons.add(S, _eval(bc.value, md.sb.doflocs[:, S]), gname)
    return K, f, cons


def _recover_heat(md: _Model, T: np.ndarray, solver_opts: dict) -> dict:
    G = np.asarray(md.sb.interpolate(T).grad)  # (dim, nel, nqp)
    k = _qp(md.sb, md.k)
    S = {("qx", "qy", "qz")[i]: -k * G[i] for i in range(md.dim)}
    return _project(md, S, solver_opts)


# --------------------------------------------------------------------------- driver


def _check_rigid_modes(md: _Model, D: np.ndarray, has_robin: bool = False) -> None:
    """Raise a clear error when the Dirichlet set leaves a rigid-body mode free (singular K)."""
    X = md.sb.doflocs
    ctr = X.mean(axis=1, keepdims=True)
    L = max(md.md.bbox_diag, 1e-300)
    Y = (X - ctr) / L
    ns = md.sb.N
    modes, names = [], []
    if md.physics == "heat":
        if has_robin:
            return
        modes.append(np.ones(ns))
        names.append("constant temperature (no temperature or convection BC)")
    elif md.axi:
        z = np.zeros(2 * ns)
        z[md.split[1]] = 1.0
        modes.append(z)
        names.append("translation z")
    else:
        dim = md.dim
        for k in range(dim):
            m_ = np.zeros(dim * ns)
            m_[md.split[k]] = 1.0
            modes.append(m_)
            names.append("translation " + "xyz"[k])
        rots = [(0, 1, "z")] if dim == 2 else [(0, 1, "z"), (1, 2, "x"), (2, 0, "y")]
        for i, j, axis in rots:  # rotation in the (i, j) plane: u_i = -y_j, u_j = y_i
            m_ = np.zeros(dim * ns)
            m_[md.split[i]] = -Y[j]
            m_[md.split[j]] = Y[i]
            modes.append(m_)
            names.append("rotation about " + axis)
    R = np.column_stack(modes)[D] if len(D) else np.zeros((0, len(modes)))
    if R.shape[0] == 0:
        free = names
    else:
        # full_matrices only when R has fewer rows than modes: otherwise the reduced SVD already gives
        # the complete (modes x modes) vt, and the full one would build an n x n U (80 GB at n ~ 1e5)
        u, sv, vt = np.linalg.svd(R, full_matrices=R.shape[0] < R.shape[1])
        tol = 1e-8 * max(1.0, sv.max() if sv.size else 1.0)
        rank = int((sv > tol).sum())
        if rank == len(modes):
            return
        null = vt[rank:]  # combinations of modes that the constraints do not touch
        free = [names[int(np.argmax(np.abs(v)))] for v in null]
    raise InsufficientConstraints(f"insufficient constraints: the model can move as a rigid body ({', '.join(free)}). "
                      f"Add Dirichlet BCs (fixed / displacement / symmetry"
                      f"{' / temperature or convection' if md.physics == 'heat' else ''}) that remove it.")


def solve(case: Case, **options) -> "SkfemResult":
    t0 = time.perf_counter()
    case.check()
    if case.analysis.nonlinear:
        raise NotImplementedError(
            "skfem backend: nonlinear analyses (analysis.options.nonlinear / material plasticity) are not "
            "implemented; the fixed solver is linear elastic. Use backend='ccx' (opt-in elastoplastic, "
            "docs/solver_ccx_nonlinear.md)")
    opts = {**(case.analysis.options or {}), **options}
    md = _Model(case, opts)
    solver_opts = {"method": opts.get("linear_solver")}
    if md.physics == "elasticity":
        K, f, cons = _assemble_elasticity(md)
    else:
        K, f, cons = _assemble_heat(md)
    D, uD = cons.finalize(md.warnings)
    _check_rigid_modes(md, D, has_robin=getattr(md, "has_robin", False))
    N = md.N
    x = np.zeros(N)
    x[D] = uD
    free = np.ones(N, dtype=bool)
    free[D] = False
    I = np.nonzero(free)[0]
    rhs = f - K @ x
    Kc = K[I][:, I]
    xI, rep = solve_linear(Kc, rhs[I], symmetric=True, return_report=True, **solver_opts)
    x[I] = xI
    if rep.residual > 1e-6:
        raise SolverError(f"linear solve inaccurate (relative residual {rep.residual:.2e}): the system is "
                          f"probably singular (insufficient constraints / rigid-body modes)")
    md.warnings.append("info: " + rep.describe())
    md.warnings += [f"linear solver: {n}" for n in rep.notes]
    if rep.iterative:
        md.warnings.append(f"linear solver is ITERATIVE ({rep.detail}, rtol={rep.tolerance:g}): the solution "
                           f"carries an algebraic error on top of the discretisation error")
    if md.physics == "elasticity":
        rec = _recover_elasticity(md, x, solver_opts)
    else:
        rec = _recover_heat(md, x, solver_opts)
    md.warnings += [f"stress recovery: {n}" for n in md.proj_report.notes]
    reaction = K @ x - f
    wall = time.perf_counter() - t0

    measure = float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(md.sb))
    info = SolveInfo(
        backend="skfem", physics=md.physics, formulation=md.form, order=md.order, dim=md.dim,
        cell_type=md.md.cell_type, n_nodes=int(md.mesh.doflocs.shape[1]), n_cells=int(md.nel),
        n_dofs=int(N), measure=measure, h_rep=float((measure / md.nel) ** (1.0 / md.dim)),
        wall_time=float(wall), warnings=list(md.warnings),
    )
    return SkfemResult(case, info, md, x, rec, K, f, reaction, D)


# --------------------------------------------------------------------------- result


class SkfemResult(Result):
    def __init__(self, case, info, model: _Model, x, recovered, K, f, reaction, dirichlet_dofs):
        self.case = case
        self.info = info
        self.model = model
        self.x = x
        self.K = K
        self.f = f
        self.reaction_vector = reaction
        self.dirichlet_dofs = dirichlet_dofs
        md = model
        self.fields: dict[str, np.ndarray] = {}
        n = md.sb.N
        zero = np.zeros(n)
        if md.physics == "elasticity":
            for k, name in enumerate(("ux", "uy", "uz")):
                self.fields[name] = x[md.split[k]].copy() if k < md.dim else zero
            for name in ("sxx", "syy", "szz", "sxy", "syz", "sxz", "stt"):
                self.fields[name] = recovered.get(name, zero)
            if md.axi:  # out-of-plane component of the 2D formulation = hoop stress
                self.fields["szz"] = self.fields["stt"]
        else:
            self.fields["T"] = x.copy()
            for name in ("qx", "qy", "qz"):
                self.fields[name] = recovered.get(name, zero)
        self._locator = None

    # ---- field algebra on arrays of components
    def _check_field(self, field):
        allowed = ELASTICITY_FIELDS if self.model.physics == "elasticity" else HEAT_FIELDS
        if field not in allowed:
            raise CaseError(f"unknown field {field!r} for {self.model.physics}; available: {allowed}")

    def _combine(self, field, comp: dict):
        if field in comp and not field.startswith("_"):
            return comp[field]
        if field == "u_mag":
            return np.sqrt(comp["ux"] ** 2 + comp["uy"] ** 2 + comp["uz"] ** 2)
        if field == "q_mag":
            return np.sqrt(comp["qx"] ** 2 + comp["qy"] ** 2 + comp["qz"] ** 2)
        comp = dict(comp)
        comp["_axi"] = self.model.axi
        return _derived_stress(comp, field)

    def _needed(self, field):
        if field in self.fields:
            return [field]
        if field == "u_mag":
            return ["ux", "uy", "uz"]
        if field == "q_mag":
            return ["qx", "qy", "qz"]
        return ["sxx", "syy", "szz", "sxy", "syz", "sxz", "stt"]

    def nodal(self, field):
        self._check_field(field)
        comp = {n: self.fields[n] for n in self._needed(field)}
        return self.model.sb.doflocs.T.copy(), np.asarray(self._combine(field, comp), dtype=float)

    def probe(self, field, points):
        self._check_field(field)
        md = self.model
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if self._locator is None:
            self._locator = Locator(md.mesh, md.md.bbox_diag)
        cells, xi, _ = self._locator.locate(pts)
        sb = md.sb
        phi = np.stack([sb.elem.lbasis(xi, i)[0] for i in range(sb.element_dofs.shape[0])])  # (nloc, n)
        dofs = sb.element_dofs[:, cells]  # (nloc, n)
        comp = {n: np.sum(phi * self.fields[n][dofs], axis=0) for n in self._needed(field)}
        return np.asarray(self._combine(field, comp), dtype=float)

    # ---- quadrature-point values for integrals
    def _qp_values(self, field, basis):
        comp = {}
        for n in self._needed(field):
            comp[n] = np.asarray(basis.interpolate(self.fields[n]))
        return self._combine(field, comp)

    def _integral(self, field, group):
        md = self.model
        if group in (None, "*"):
            kind = "cells"
            idx = np.arange(md.nel)
        else:
            g = md.md.group(group)
            kind, idx = g.kind, g.index
        if kind == "cells":
            b = Basis(md.mesh, md.selem, intorder=md.intorder, elements=idx)
        elif kind == "facets":
            b = FacetBasis(md.mesh, md.selem, facets=idx, intorder=md.intorder)
        else:
            raise CaseError(f"integral/mean need a cell or facet group; {group!r} is a {kind} group")
        x = _coords(b)
        c = md.wfac(x)
        vals = self._qp_values(field, b)
        integ = float(np.sum(vals * c * b.dx))
        meas = float(np.sum(c * b.dx))
        return integ, meas

    def evaluate_all(self) -> dict[str, float]:
        """Like ``Result.evaluate_all`` but a QoI that cannot be evaluated (point outside the mesh,
        unknown group, ...) gives NaN plus a warning instead of failing the whole result."""
        out = {}
        for q in self.case.qois:
            try:
                out[q.name] = float(self.evaluate(q))
            except (CaseError, ValueError) as e:
                out[q.name] = float("nan")
                msg = f"qoi {q.name!r} could not be evaluated: {e}"
                if msg not in self.info.warnings:
                    self.info.warnings.append(msg)
        return out

    def summary(self) -> dict:
        qois = self.evaluate_all()  # first: it may append warnings to info
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta}

    def evaluate(self, qoi: QoI) -> float:
        if qoi.type == "point_difference":   # backend-independent, see Result.evaluate_qoi
            return Result.evaluate_qoi(self, qoi)
        md = self.model
        t = qoi.type
        if t == "point":
            return float(self.probe(qoi.field, [qoi.point[: md.dim]])[0])
        if t in ("max", "min", "max_abs"):
            _, v = self.nodal(qoi.field)
            S = md.sdofs_of(qoi.group)
            v = v[S]
            return float(v.max() if t == "max" else v.min() if t == "min" else np.abs(v).max())
        if t == "reaction":
            S = md.sdofs_of(qoi.group)
            k = int(qoi.component or 0)
            if md.physics == "heat" and k != 0:
                raise CaseError("heat reaction (heat flow into the body) has only component 0")
            idx = md.split[k][S]
            if not np.isin(idx, self.dirichlet_dofs).any():
                msg = (f"reaction QoI {qoi.name!r}: group {qoi.group!r} has no Dirichlet constraint on "
                       f"component {k}, so its reaction is ~0 (did you mean a supported group?)")
                if msg not in self.info.warnings:
                    self.info.warnings.append(msg)
            return float(self.reaction_vector[idx].sum())
        if t in ("integral", "mean"):
            self._check_field(qoi.field)
            integ, meas = self._integral(qoi.field, qoi.group)
            return integ if t == "integral" else integ / meas
        if t == "strain_energy":
            if md.physics != "elasticity":
                raise CaseError("strain_energy is elasticity-only")
            return float(0.5 * self.x @ (self.K @ self.x))
        raise CaseError(f"unknown QoI type {t!r}")

    # ---- output
    def write_vtu(self, path):
        import meshio

        md = self.model
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = md.sb.doflocs.T
        if md.dim == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        cells = [(_VTK[(md.dim, md.order)], md.sb.element_dofs.T.astype(np.int64))]
        fields = ELASTICITY_FIELDS if md.physics == "elasticity" else HEAT_FIELDS
        point_data = {}
        for fname in fields:
            point_data[fname] = self.nodal(fname)[1]
        if md.physics == "elasticity":
            point_data["displacement"] = np.column_stack([self.fields["ux"], self.fields["uy"], self.fields["uz"]])
        cell_data = {"material": [md.mat.astype(np.int32)]}
        meshio.Mesh(pts, cells, point_data=point_data, cell_data=cell_data).write(str(path))
        return path
