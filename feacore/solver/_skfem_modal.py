"""Modal analysis of the fixed solver: natural frequencies and mode shapes (``analysis.options["modal"]``).

    K phi = omega^2 M phi

K is the stiffness of the steady elasticity case (``skfem_backend._assemble_elasticity``, so the
formulations, materials and Dirichlet sets are the steady ones) and M the consistent mass matrix with
the density of each material region (thickness for the plane formulations, 2 pi r axisymmetric). Loads
are ignored and every Dirichlet BC acts as zero, a prescribed non-zero displacement included.

The ``n_modes`` lowest elastic modes above ``shift`` (Hz) come from shift-invert Lanczos
(``scipy.sparse.linalg.eigsh``) with ONE sparse factorization of K - sigma M, kept for all iterations:

  * restrained model, shift 0: sigma = 0 (K is positive definite).
  * a model with rigid-body modes, shift 0: K is singular, so sigma is a small negative number (a
    fraction of a geometric estimate of the first elastic eigenvalue), which makes K - sigma M positive
    definite. The rigid-body modes are known in closed form (translations and rotations that the
    Dirichlet set leaves free); the Lanczos operator is deflated by them (projected M-orthogonally), so
    they are never returned and the count ``rigid_modes`` is exact, not a threshold on eigenvalues.
  * shift f > 0: sigma = (2 pi f)^2 and the modes just above it are returned (K - sigma M is
    indefinite; an LDL^T / LU factorization).

Models with fewer than ``DENSE_BELOW`` free unknowns are solved with the dense LAPACK solver instead.

Mode shapes are mass-normalised (phi^T M phi = 1) with the sign that makes the largest component
positive. ``effective_mass[k, j]`` is the share of the total mass that mode k moves along axis j:
(phi_k^T M 1_j)^2 / m, with 1_j the unit translation along j. For a restrained model the shares of all
modes sum toward 1 (what is missing is the mass that sits on the supports); for a free body the elastic
modes are orthogonal to the translations and the shares are 0. Axisymmetric: only the axial share is
defined (the radial column is NaN).

See docs/solver_transient.md ("Modal analysis").
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from skfem import Functional

from feacore.case import Case, CaseError, QoI
from feacore.solver import _skfem_assembly as fast
from feacore.solver import linsolve
from feacore.solver._skfem_locate import Locator
from feacore.solver.base import ModalResult, SolveInfo
from feacore.solver.skfem_backend import _VTK, SolverError, _assemble_elasticity, _m_weighted, _Model, _qp

DENSE_BELOW = 400          # free unknowns below which the dense generalized eigensolver is used
_DIRICHLET = ("fixed", "displacement", "symmetry")
_MODE_FIELDS = ("ux", "uy", "uz", "u_mag")


# --------------------------------------------------------------------------- mass and rigid-body modes


def density(md: _Model) -> np.ndarray:
    """The density of every cell, (nel,), from the material of its region."""
    rho = np.array([m.rho for m in md.case.materials.values()], dtype=float)
    return rho[md.mat]


def mass_matrix(md: _Model) -> sp.csr_matrix:
    """Consistent mass matrix of the vector unknowns (dof = ncomp * node + component): int rho c u.v with
    c = thickness (plane), 2 pi r (axisymmetric) or 1 (3D). Cached on the model."""
    if getattr(md, "_M_dyn", None) is None:
        c = md.wfac(md.xq)
        rho = _qp(md.sb, density(md))
        if md.opts.get("assembly", "fast") == "skfem":      # reference implementation
            Ms = _m_weighted.assemble(md.sb, c=c, h=rho)
        else:
            Ms = fast.mass(md.sb, c=c * rho)
        md._M_dyn = sp.kron(sp.csr_matrix(Ms), sp.identity(md.ncomp), format="csr")
    return md._M_dyn


def rigid_body_modes(md: _Model) -> tuple[np.ndarray, list[str]]:
    """The rigid-body motions of the formulation as columns (N, n): translations and rotations (2D: 3,
    3D: 6), axisymmetric: the axial translation only. As in ``skfem_backend._check_rigid_modes``."""
    X = md.sb.doflocs
    Y = (X - X.mean(axis=1, keepdims=True)) / max(md.md.bbox_diag, 1e-300)
    N, dim = md.N, md.dim
    modes, names = [], []
    if md.axi:
        z = np.zeros(N)
        z[md.split[1]] = 1.0
        return z[:, None], ["translation z"]
    for k in range(dim):
        m_ = np.zeros(N)
        m_[md.split[k]] = 1.0
        modes.append(m_)
        names.append("translation " + "xyz"[k])
    rots = [(0, 1, "z")] if dim == 2 else [(0, 1, "z"), (1, 2, "x"), (2, 0, "y")]
    for i, j, axis in rots:          # rotation in the (i, j) plane: u_i = -y_j, u_j = y_i
        m_ = np.zeros(N)
        m_[md.split[i]] = -Y[j]
        m_[md.split[j]] = Y[i]
        modes.append(m_)
        names.append("rotation about " + axis)
    return np.column_stack(modes), names


def free_rigid_modes(md: _Model, D: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """The rigid-body motions that the constrained dofs D leave free: (N, n) columns that vanish on D,
    and a name for each (the motion it is closest to)."""
    R, names = rigid_body_modes(md)
    if len(D) == 0:
        return R, names
    RD = R[D]
    _, sv, vt = np.linalg.svd(RD, full_matrices=RD.shape[0] < RD.shape[1])
    tol = 1e-8 * max(1.0, sv.max() if sv.size else 1.0)
    rank = int((sv > tol).sum())
    null = vt[rank:]
    Z = R @ null.T
    Z[D] = 0.0
    return Z, [names[int(np.argmax(np.abs(v)))] for v in null]


# --------------------------------------------------------------------------- a factorization that is kept


class Factorization:
    """A sparse direct factorization of a symmetric matrix, kept for repeated solves (the Lanczos
    iterations here, the time steps of ``_skfem_dynamics``).

    ``method`` follows ``feacore.solver.linsolve`` ("scipy", "pardiso", "auto", or the
    ``FEACORE_LINEAR_SOLVER`` variable): MKL PARDISO when it is installed and selected, else SuperLU.
    cuDSS and cg are not used for a kept factorization (noted in ``notes``). ``definite`` = the matrix is
    positive definite (Cholesky-like); otherwise a symmetric indefinite / LU factorization."""

    def __init__(self, A, method: str | None = None, definite: bool = True):
        t0 = time.perf_counter()
        self.n = A.shape[0]
        self.nnz = A.nnz
        self.notes: list[str] = []
        self._pardiso = None
        self._lu = None
        m = linsolve.resolve_method(method)
        if m in ("cudss", "cg"):
            self.notes.append(f"linear solver {m!r} is not used for a factorization that is kept; "
                              f"using a CPU direct solver")
            m = "auto"
        if m == "auto":
            m = "pardiso" if linsolve._have_pypardiso() else "scipy"
        elif m == "pardiso" and not linsolve._have_pypardiso():
            self.notes.append("requested linear solver 'pardiso' unavailable; fell back to 'scipy'")
            m = "scipy"
        if m == "pardiso":
            try:
                self._factor_pardiso(A, definite)
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"'pardiso' failed ({type(e).__name__}: {e}); fell back to 'scipy'")
                self._pardiso = None
                m = "scipy"
        if m == "scipy":
            self._factor_scipy(A, definite)
        self.method = m
        self.time = time.perf_counter() - t0

    def _factor_pardiso(self, A, definite):
        linsolve._mkl_rt_hint()
        from pypardiso import PyPardisoSolver

        U = sp.triu(A, format="csr")
        U.sort_indices()
        for mtype, what in ((2, "SPD"), (-2, "symmetric indefinite")) if definite else ((-2, "symmetric indefinite"),):
            solver = PyPardisoSolver(mtype=mtype)
            try:
                solver.factorize(U)
                x = solver._call_pardiso(U, np.ones((self.n, 1)))      # a first solve: errors show here
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("non-finite solution")
            except Exception as e:  # noqa: BLE001
                self._free(solver)
                if mtype == -2:
                    raise
                self.notes.append(f"PARDISO SPD factorization failed ({e}); retried as symmetric indefinite")
                continue
            solver.set_phase(33)
            self._pardiso, self._A = solver, U
            self.detail = f"MKL PARDISO, mtype={mtype} ({what})"
            return

    def _factor_scipy(self, A, definite):
        from scipy.sparse.linalg import splu

        A = sp.csc_matrix(A)
        if definite:
            try:
                self._lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
                self.detail = "SuperLU, MMD_AT_PLUS_A ordering, symmetric mode"
                return
            except RuntimeError as e:
                self.notes.append(f"symmetric-mode SuperLU failed ({e}); retried with COLAMD + pivoting")
        self._lu = splu(A, permc_spec="COLAMD")
        self.detail = "SuperLU, COLAMD ordering, partial pivoting"

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if self._pardiso is not None:
            x = self._pardiso._call_pardiso(self._A, np.ascontiguousarray(b.reshape(self.n, -1), dtype=np.float64))
            return np.asarray(x).reshape(b.shape)
        return self._lu.solve(b)

    @staticmethod
    def _free(solver):
        try:
            solver.free_memory(everything=True)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        if self._pardiso is not None:
            self._free(self._pardiso)
            self._pardiso = None

    def __del__(self):
        self.close()

    def describe(self) -> str:
        return f"{self.detail}, n={self.n}, nnz={self.nnz}, factor t={self.time:.3g}s"


# --------------------------------------------------------------------------- the eigenproblem


def _m_orthonormal(Z: np.ndarray, M) -> np.ndarray:
    """Columns of Z made M-orthonormal (Z^T M Z = I)."""
    if Z.shape[1] == 0:
        return Z
    G = Z.T @ (M @ Z)
    L = np.linalg.cholesky(0.5 * (G + G.T))
    return np.linalg.solve(L, Z.T).T


def _sigma_free_body(md: _Model) -> float:
    """A negative shift for a model with rigid-body modes: minus a geometric estimate of the first
    elastic eigenvalue. With L the longest and t the shortest side of the bounding box and c the bar
    wave speed sqrt(E / rho), the first bending frequency of a free slender body is about t c / L^2 (Hz)
    and tends to c / L for a compact one. The estimate only sets the conditioning of K - sigma M and
    the speed of the Lanczos iteration, never the result."""
    X = md.sb.doflocs
    ext = np.ptp(X, axis=1)
    L, t = float(ext.max()), float(max(ext.min(), 1e-6 * ext.max()))
    c = math.sqrt(float(np.min(md.E / density(md))))
    return -((2.0 * math.pi * t * c / L**2) ** 2)


def _eigen(KI, MI, Z, n_modes: int, sigma: float, above: bool, method, notes: list[str]):
    """The n_modes lowest eigenpairs of (KI, MI) with eigenvalue > sigma (``above``) or all of them,
    M-orthogonal to the columns of Z (M-orthonormal rigid-body modes). Returns (lam, Phi, how)."""
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    n = KI.shape[0]
    nz = Z.shape[1]
    if n - nz < 1:
        raise CaseError("modal analysis: the model has no elastic degree of freedom left")
    n_modes = min(n_modes, n - nz)
    if n < DENSE_BELOW or n_modes + 2 >= n - nz:
        from scipy.linalg import eigh

        lam, Phi = eigh(KI.toarray(), MI.toarray())
        if nz:                                    # drop the modes that lie in the span of Z
            share = np.sum((Z.T @ (MI @ Phi)) ** 2, axis=0)
            keep = np.argsort(share, kind="stable")[: n - nz]
            keep.sort()
            lam, Phi = lam[keep], Phi[:, keep]
        if above:
            sel = lam > sigma
            lam, Phi = lam[sel], Phi[:, sel]
        return lam[:n_modes], Phi[:, :n_modes], f"dense LAPACK (n = {n})"
    A = (KI - sigma * MI).tocsr()
    fac = Factorization(A, method, definite=sigma < 0 or (sigma == 0 and nz == 0))
    notes += [f"linear solver: {s}" for s in fac.notes]
    MZ = MI @ Z if nz else None

    def op(x):
        y = fac.solve(x)
        if nz:
            y = y - Z @ (MZ.T @ y)
        return y

    v0 = np.random.default_rng(0).standard_normal(n)       # fixed: the same case gives the same modes
    if nz:
        v0 -= Z @ (MZ.T @ v0)
    k = min(n_modes + 2, n - nz - 1)                        # two spare vectors help a cluster to converge
    kw = dict(k=k, M=MI, sigma=sigma, which="LA" if above else "LM", v0=v0, tol=0.0,
              OPinv=LinearOperator((n, n), matvec=op, dtype=float), ncv=min(n, max(2 * k + 1, 24)))
    try:
        lam, Phi = eigsh(KI, **kw)
    except ArpackNoConvergence:
        kw["ncv"] = min(n, 4 * k + 40)
        try:
            lam, Phi = eigsh(KI, maxiter=50 * n, **kw)
        except ArpackNoConvergence as e:
            raise SolverError(f"modal analysis: the Lanczos iteration did not converge "
                              f"({len(e.eigenvalues)} of {k} eigenvalues); try analysis.options.modal.shift "
                              f"near the frequencies of interest") from e
    finally:
        how = f"shift-invert Lanczos, sigma = {sigma:.6g} (rad/s)^2, {fac.describe()}"
        fac.close()
    idx = np.argsort(lam)
    lam, Phi = lam[idx], Phi[:, idx]
    if above:
        sel = lam > sigma
        lam, Phi = lam[sel], Phi[:, sel]
    return lam[:n_modes], Phi[:, :n_modes], how


def solve(case: Case, **options) -> "SkfemModalResult":
    t0 = time.perf_counter()
    case.check()
    mo = case.analysis.modal
    if mo is None:
        raise CaseError("not a modal analysis: analysis.options.modal is missing")
    opts = {**(case.analysis.options or {}), **options}
    md = _Model(case, opts)
    if md.physics != "elasticity":
        raise CaseError("a modal analysis is defined for elasticity only")
    n_modes = int(mo["n_modes"])
    shift = float(mo.get("shift") or 0.0)

    # stiffness and Dirichlet set of the steady case, without its loads (they are ignored)
    n_neumann = sum(b.type not in _DIRICHLET for b in case.bcs)
    md.case = replace(case, loads=[], bcs=[b for b in case.bcs if b.type in _DIRICHLET])
    try:
        K, _, cons = _assemble_elasticity(md)
    finally:
        md.case = case
    if case.loads or n_neumann:
        md.warnings.append(f"modal analysis: {len(case.loads)} loads and {n_neumann} traction / pressure / force "
                           f"BCs are ignored (natural frequencies do not depend on the loads)")
    D, uD = cons.finalize(md.warnings)
    if np.any(uD != 0.0):
        md.warnings.append(f"modal analysis: {int(np.count_nonzero(uD))} prescribed non-zero displacements act as "
                           f"zero (a mode shape vanishes on every Dirichlet dof)")
    N = md.N
    free = np.ones(N, dtype=bool)
    free[D] = False
    I = np.nonzero(free)[0]
    if len(I) == 0:
        raise CaseError("modal analysis: every degree of freedom is constrained")
    M = mass_matrix(md)
    KI = K[I][:, I].tocsr()
    MI = M[I][:, I].tocsr()

    Zfull, znames = free_rigid_modes(md, D)
    Z = _m_orthonormal(Zfull[I], MI)
    nz = Z.shape[1]
    if nz:
        md.warnings.append(f"modal analysis: the model has {nz} rigid-body modes ({', '.join(znames)}) at 0 Hz; "
                           f"they are reported in rigid_modes and are not counted among the elastic modes")
    sigma = (2.0 * math.pi * shift) ** 2
    if shift == 0.0 and nz:
        sigma = _sigma_free_body(md)
    notes: list[str] = []
    try:
        lam, Phi, how = _eigen(KI, MI, Z, n_modes, sigma, shift > 0.0, opts.get("linear_solver"), notes)
    except (RuntimeError, linsolve.LinearSolverError) as e:
        if isinstance(e, SolverError) or sigma != 0.0:
            raise
        # K is singular although the Dirichlet set removes every rigid motion of the whole model: a part
        # that is not connected, or a mechanism. A negative shift still gives the modes; they are flagged below.
        notes.append(f"factorization of K failed ({e}); retried with a negative shift")
        sigma = _sigma_free_body(md)
        lam, Phi, how = _eigen(KI, MI, Z, n_modes, sigma, False, opts.get("linear_solver"), notes)
    md.warnings += notes
    if len(lam) < n_modes:
        md.warnings.append(f"modal analysis: {len(lam)} of the {n_modes} requested modes were found "
                           f"(the model has {len(I) - nz} elastic degrees of freedom"
                           + (f"; {shift:g} Hz is above most of them)" if shift > 0 else ")"))

    # mass-normalise, fix the sign, residuals
    modes = np.zeros((N, len(lam)))
    resid = np.zeros(len(lam))
    kscale = float(abs(KI).max())
    for k in range(len(lam)):
        phi = Phi[:, k]
        if nz:
            phi = phi - Z @ (Z.T @ (MI @ phi))
        phi = phi / math.sqrt(float(phi @ (MI @ phi)))
        j = int(np.argmax(np.abs(phi)))
        if phi[j] < 0:
            phi = -phi
        r = KI @ phi - lam[k] * (MI @ phi)
        if nz:                                      # the residual in the elastic subspace
            r = r - (MI @ Z) @ (Z.T @ r)
        resid[k] = float(np.linalg.norm(r)) / max(kscale * float(np.linalg.norm(phi)), 1e-300)
        modes[I, k] = phi
    if len(resid) and resid.max() > 1e-6:
        md.warnings.append(f"modal analysis: the eigenpair residual is {resid.max():.1e} (relative to |K| |phi|): "
                           f"the modes are not converged")
    lam_ref = float(np.abs(lam).max()) if len(lam) else 0.0
    zero = np.nonzero(lam <= 1e-9 * lam_ref)[0] if len(lam) > 1 else np.zeros(0, dtype=int)
    if len(zero):
        md.warnings.append(f"modal analysis: modes {[int(k) + 1 for k in zero]} have (near) zero frequency but are "
                           f"not rigid-body motions of the whole model: a mechanism, or a part that is not connected")
    freqs = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * math.pi)

    # effective mass: participation of each mode in a unit translation along each axis
    dim = md.dim
    part = np.zeros((len(lam), dim))
    total = np.zeros(dim)
    for j in range(dim):
        one = np.zeros(N)
        one[md.split[j]] = 1.0
        Mone = M @ one
        total[j] = float(one @ Mone)
        part[:, j] = modes.T @ Mone
    eff = part**2 / total[None, :]
    if md.axi:
        eff[:, 0] = np.nan                 # a radial "translation" is not a rigid motion of a body of revolution
    md.warnings.append(f"info: modal: {len(lam)} modes, {how}; rigid-body modes {nz}; "
                       f"largest eigenpair residual {resid.max() if len(resid) else 0.0:.1e}")
    wall = time.perf_counter() - t0
    measure = float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(md.sb))
    info = SolveInfo(
        backend="skfem", physics=md.physics, formulation=md.form, order=md.order, dim=md.dim,
        cell_type=md.md.cell_type, n_nodes=int(md.mesh.doflocs.shape[1]), n_cells=int(md.nel),
        n_dofs=int(N), measure=measure, h_rep=float((measure / md.nel) ** (1.0 / md.dim)),
        wall_time=float(wall), warnings=list(md.warnings),
    )
    return SkfemModalResult(case, info, md, freqs, lam, modes, nz, znames, Zfull, eff, part, float(total[-1]),
                            K, M, D, resid)


# --------------------------------------------------------------------------- result


class SkfemModalResult(ModalResult):
    """``frequencies`` (Hz), ``eigenvalues`` (omega^2), ``modes`` (N, n: mass-normalised, zero on the
    Dirichlet dofs), ``rigid_modes`` and their ``rigid_names`` / ``rigid_shapes`` (N, n_rigid),
    ``effective_mass`` (n, dim: shares of ``total_mass``), ``participation`` (n, dim: phi^T M 1_j),
    ``residuals`` (per mode, relative), and the matrices ``K``, ``M`` with ``dirichlet_dofs``."""

    def __init__(self, case, info, model, frequencies, eigenvalues, modes, rigid_modes, rigid_names, rigid_shapes,
                 effective_mass, participation, total_mass, K, M, dirichlet_dofs, residuals):
        self.case = case
        self.info = info
        self.model = model
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.modes = modes
        self.rigid_modes = int(rigid_modes)
        self.rigid_names = list(rigid_names)
        self.rigid_shapes = rigid_shapes
        self.effective_mass = effective_mass
        self.participation = participation
        self.total_mass = total_mass
        self.K = K
        self.M = M
        self.dirichlet_dofs = dirichlet_dofs
        self.residuals = residuals
        self._locator = None

    def _mode(self, mode: int) -> np.ndarray:
        n = len(self.frequencies)
        if not isinstance(mode, (int, np.integer)) or not 1 <= mode <= n:
            raise ValueError(f"mode {mode} is not one of the {n} modes that were computed (1 = lowest elastic mode)")
        return self.modes[:, int(mode) - 1]

    def _comp(self, mode: int) -> dict:
        md, phi = self.model, self._mode(mode)
        zero = np.zeros(md.sb.N)
        return {name: (phi[md.split[k]] if k < md.dim else zero) for k, name in enumerate(("ux", "uy", "uz"))}

    @staticmethod
    def _check_field(field):
        if field not in _MODE_FIELDS:
            raise CaseError(f"a mode shape has the displacement fields {_MODE_FIELDS}, not {field!r} (a mode has "
                            f"no amplitude, so no stress level)")

    def shape(self, mode: int = 1):
        md, phi = self.model, self._mode(mode)
        return md.sb.doflocs.T.copy(), np.column_stack([phi[md.split[k]] for k in range(md.dim)])

    def nodal(self, field: str, mode: int = 1):
        self._check_field(field)
        c = self._comp(mode)
        v = np.sqrt(c["ux"] ** 2 + c["uy"] ** 2 + c["uz"] ** 2) if field == "u_mag" else c[field]
        return self.model.sb.doflocs.T.copy(), np.asarray(v, dtype=float).copy()

    def probe(self, field: str, points, mode: int = 1) -> np.ndarray:
        self._check_field(field)
        md = self.model
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if self._locator is None:
            self._locator = Locator(md.mesh, md.md.bbox_diag)
        cells, xi, _ = self._locator.locate(pts)
        sb = md.sb
        phi = np.stack([sb.elem.lbasis(xi, i)[0] for i in range(sb.element_dofs.shape[0])])
        dofs = sb.element_dofs[:, cells]
        c = {n: np.sum(phi * v[dofs], axis=0) for n, v in self._comp(mode).items()}
        return np.asarray(np.sqrt(c["ux"] ** 2 + c["uy"] ** 2 + c["uz"] ** 2) if field == "u_mag" else c[field],
                          dtype=float)

    def evaluate_all(self) -> dict[str, float]:
        """A QoI that cannot be evaluated (a mode that was not computed) gives NaN and a warning."""
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

    def evaluate_qoi(self, qoi: QoI) -> float:
        return self.evaluate(qoi)

    def summary(self) -> dict:
        qois = self.evaluate_all()
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta,
                "modal": {"frequencies": [float(f) for f in self.frequencies], "rigid_modes": self.rigid_modes,
                          "rigid_names": self.rigid_names, "total_mass": self.total_mass,
                          "effective_mass": [[None if not np.isfinite(v) else float(v) for v in row]
                                             for row in self.effective_mass]}}

    def write_vtu(self, path):
        """One file: the mesh with a vector ``mode_<k>`` (and ``rigid_<k>``) per mode; the frequencies are
        in the field data."""
        import meshio

        md = self.model
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = md.sb.doflocs.T
        if md.dim == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        cells = [(_VTK[(md.dim, md.order)], md.sb.element_dofs.T.astype(np.int64))]

        def vec(phi):
            v = np.zeros((md.sb.N, 3))
            for k in range(md.dim):
                v[:, k] = phi[md.split[k]]
            return v

        point_data = {f"mode_{k + 1}": vec(self.modes[:, k]) for k in range(len(self.frequencies))}
        for k in range(self.rigid_modes):
            point_data[f"rigid_{k + 1}"] = vec(self.rigid_shapes[:, k])
        meshio.Mesh(pts, cells, point_data=point_data, cell_data={"material": [md.mat.astype(np.int32)]},
                    field_data={"frequencies_Hz": self.frequencies}).write(str(path))
        return path
