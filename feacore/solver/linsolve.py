"""Pluggable sparse linear solver for the fixed FE solver.

``solve_linear(K, f, method="auto") -> x`` solves ``K x = f`` for a sparse (normally symmetric
positive definite) stiffness / conductivity matrix ``K``.

Methods (all direct by default, so the answer does not depend on the hardware or on an
iterative tolerance that would contaminate a Richardson / GCI study):

  scipy    SciPy SuperLU with a symmetric ordering (MMD on A^T + A, diagonal pivoting,
           ``SymmetricMode``), which for SPD FE matrices behaves like a sparse Cholesky and
           gives far less fill than SciPy's default COLAMD. Single threaded; always available.
  pardiso  Intel MKL PARDISO via ``pypardiso`` (``pip install pypardiso``); multithreaded
           (``MKL_NUM_THREADS`` / ``OMP_NUM_THREADS``), much faster for 3D. Uses the SPD
           matrix type (mtype 2) when the matrix is symmetric, else the general type.
  cudss    NVIDIA cuDSS on a CUDA GPU via nvmath-python
           (``pip install nvmath-python[cu12]`` or ``[cu13]``):
           ``nvmath.sparse.advanced.direct_solver`` with ``DirectSolverMatrixType.SPD``.
  auto     cudss if a CUDA GPU and nvmath are available, else pardiso if installed, else scipy.
  cg       (opt-in, iterative) preconditioned CG: pyamg smoothed aggregation if installed
           (with ``near_nullspace`` = rigid-body modes when given), else Jacobi. Relative
           tolerance 1e-10 by default. Not used unless asked for explicitly.

Selection: an explicit ``method`` other than ``"auto"`` wins; otherwise the environment
variable ``FEACORE_LINEAR_SOLVER``; otherwise ``auto``. If the selected method is not usable
here (package missing, no GPU) the next direct method in the auto order is used and the
fallback is reported in ``LinearSolveReport.notes`` (``strict=True`` raises instead).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

METHODS = ("auto", "scipy", "pardiso", "cudss", "cg")
DIRECT_METHODS = ("cudss", "pardiso", "scipy")
ENV_VAR = "FEACORE_LINEAR_SOLVER"


class LinearSolverError(RuntimeError):
    """The linear solve failed (singular matrix, solver error, ...)."""


@dataclass
class LinearSolveReport:
    method: str                 # method actually used
    requested: str              # method requested (after env-var resolution)
    detail: str = ""            # e.g. ordering / matrix type
    n: int = 0
    nnz: int = 0
    time: float = 0.0           # seconds (factorisation + solve)
    residual: float = float("nan")   # ||K x - f|| / ||f||
    iterative: bool = False
    tolerance: float | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        s = f"linear_solver={self.method}"
        if self.detail:
            s += f" ({self.detail})"
        s += f", n={self.n}, nnz={self.nnz}, t={self.time:.3g}s, rel_residual={self.residual:.2e}"
        if self.iterative:
            s += f", ITERATIVE rtol={self.tolerance:g}"
        return s


# --------------------------------------------------------------------------- availability


def _have_pypardiso() -> bool:
    try:
        import pypardiso  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_device_count() -> int:
    """Number of visible CUDA devices (0 if no driver / runtime / GPU)."""
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        pass
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.6

        err, n = cudart.cudaGetDeviceCount()
        return int(n) if int(err) == 0 else 0
    except Exception:
        pass
    try:
        from cuda import cudart  # older cuda-python

        err, n = cudart.cudaGetDeviceCount()
        return int(n) if int(err) == 0 else 0
    except Exception:
        return 0


def _have_cudss() -> bool:
    if _cuda_device_count() < 1:
        return False
    try:
        from nvmath.sparse.advanced import direct_solver  # noqa: F401
    except Exception:
        return False
    return True


def available_methods() -> dict[str, bool]:
    return {"scipy": True, "pardiso": _have_pypardiso(), "cudss": _have_cudss(), "cg": True}


def _usable(method: str) -> bool:
    if method == "pardiso":
        return _have_pypardiso()
    if method == "cudss":
        return _have_cudss()
    return method in ("scipy", "cg")


def resolve_method(method: str | None = None) -> str:
    """Requested method after applying the env-var rule (may still be ``"auto"``)."""
    m = (method or "").strip().lower()
    if not m or m == "auto":
        m = os.environ.get(ENV_VAR, "").strip().lower() or "auto"
    if m not in METHODS:
        raise ValueError(f"unknown linear solver {m!r}; choose one of {METHODS}")
    return m


def _auto() -> str:
    for m in DIRECT_METHODS:
        if _usable(m):
            return m
    return "scipy"


# --------------------------------------------------------------------------- backends


def _is_symmetric(K: sp.csr_matrix, rtol: float = 1e-10) -> bool:
    d = K - K.T
    if d.nnz == 0:
        return True
    return float(abs(d).max()) <= rtol * float(abs(K).max())


def _solve_scipy(K, f, report: LinearSolveReport, symmetric: bool):
    from scipy.sparse.linalg import splu

    A = K.tocsc()
    if symmetric:
        try:
            lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options={"SymmetricMode": True})
            report.detail = "SuperLU, MMD_AT_PLUS_A ordering, symmetric mode"
        except RuntimeError as e:  # zero pivot without pivoting -> retry with partial pivoting
            report.notes.append(f"symmetric-mode SuperLU failed ({e}); retried with COLAMD + pivoting")
            lu = splu(A, permc_spec="COLAMD")
            report.detail = "SuperLU, COLAMD ordering"
    else:
        lu = splu(A, permc_spec="COLAMD")
        report.detail = "SuperLU, COLAMD ordering"
    return lu.solve(np.asarray(f, dtype=float))


_PARDISO: dict = {}


def _pardiso_solver(mtype: int):
    """One cached PyPardisoSolver per matrix type: creating one scans site-packages for
    libmkl_rt (~0.5 s), so point it at the library directly when we can find it cheaply."""
    if mtype not in _PARDISO:
        if not os.environ.get("PYPARDISO_MKL_RT"):
            import glob
            import site
            import sys

            dirs = [os.path.join(sys.prefix, "lib")]
            try:
                dirs.append(os.path.join(site.getuserbase(), "lib"))
            except Exception:
                pass
            for d in dirs:
                hits = sorted(glob.glob(os.path.join(d, "libmkl_rt.so*")))
                if hits:
                    os.environ["PYPARDISO_MKL_RT"] = hits[0]
                    break
        from pypardiso import PyPardisoSolver

        _PARDISO[mtype] = PyPardisoSolver(mtype=mtype)
    return _PARDISO[mtype]


def _pardiso_run(mtype: int, A, b):
    solver = _pardiso_solver(mtype)
    try:
        return solver.solve(A, b)
    finally:
        try:
            solver.remove_stored_factorization()
            solver.free_memory(everything=True)
        except Exception:
            pass


def _solve_pardiso(K, f, report: LinearSolveReport, symmetric: bool):
    b = np.asarray(f, dtype=float)
    if symmetric:
        A = sp.triu(K, format="csr")  # mtype 2 (real SPD): PARDISO reads the upper triangle
        A.sort_indices()
        try:
            x = _pardiso_run(2, A, b)
            report.detail = "MKL PARDISO, mtype=2 (SPD)"
            return np.asarray(x).reshape(b.shape)
        except Exception as e:  # not positive definite -> general LU
            report.notes.append(f"PARDISO SPD factorisation failed ({e}); retried with mtype=11")
    A = sp.csr_matrix(K)
    A.sort_indices()
    x = _pardiso_run(11, A, b)
    report.detail = "MKL PARDISO, mtype=11 (general)"
    return np.asarray(x).reshape(b.shape)


def _solve_cudss(K, f, report: LinearSolveReport, symmetric: bool):
    """cuDSS through nvmath-python. Operands stay on the host (SciPy CSR + NumPy RHS); nvmath
    copies them to device 0 and returns a NumPy array (the result lives where ``b`` lives)."""
    from nvmath.sparse.advanced import (DirectSolverMatrixType, DirectSolverMatrixViewType,
                                        DirectSolverOptions, direct_solver)

    A = sp.csr_matrix(K, dtype=np.float64)
    A.sort_indices()
    A.indptr = A.indptr.astype(np.int32)   # cuDSS needs 32-bit indices
    A.indices = A.indices.astype(np.int32)
    b = np.asfortranarray(np.asarray(f, dtype=np.float64))
    mtype = DirectSolverMatrixType.SPD if symmetric else DirectSolverMatrixType.GENERAL
    options = DirectSolverOptions(sparse_system_type=mtype,
                                  sparse_system_view=DirectSolverMatrixViewType.FULL)
    x = direct_solver(A, b, options=options, execution="cuda")
    report.detail = f"cuDSS via nvmath-python, {'SPD' if symmetric else 'GENERAL'}"
    return np.asarray(x).reshape(np.shape(f))


def _solve_cg(K, f, report: LinearSolveReport, symmetric: bool, rtol: float, near_nullspace):
    from scipy.sparse.linalg import cg

    A = sp.csr_matrix(K)
    b = np.asarray(f, dtype=float)
    M = None
    try:
        import pyamg

        B = None if near_nullspace is None else np.asarray(near_nullspace, dtype=float)
        ml = pyamg.smoothed_aggregation_solver(A, B=B)
        M = ml.aspreconditioner(cycle="V")
        report.detail = "CG + pyamg smoothed aggregation" + (" (rigid-body near-nullspace)" if B is not None else "")
    except Exception:
        d = A.diagonal()
        M = sp.diags(1.0 / np.where(d != 0, d, 1.0))
        report.detail = "CG + Jacobi (pyamg not installed)"
    cols = b.reshape(b.shape[0], -1)
    out = np.empty_like(cols)
    for j in range(cols.shape[1]):
        x, info = cg(A, cols[:, j], rtol=rtol, atol=0.0, maxiter=max(1000, 10 * A.shape[0]), M=M)
        if info != 0:
            raise LinearSolverError(f"CG did not converge (info={info})")
        out[:, j] = x
    report.iterative = True
    report.tolerance = rtol
    return out.reshape(b.shape)


# --------------------------------------------------------------------------- public API


def solve_linear(K, f, method: str | None = "auto", *, symmetric: bool | None = None,
                 strict: bool = False, rtol: float = 1e-10, near_nullspace=None,
                 return_report: bool = False):
    """Solve ``K x = f``. ``f`` may be a vector (n,) or a matrix (n, m) of right-hand sides.

    ``symmetric``: hint (FE stiffness matrices are SPD); ``None`` checks numerically.
    ``strict``: raise if the requested method is unusable instead of falling back.
    ``rtol`` / ``near_nullspace``: only for the iterative ``cg`` method.
    Returns ``x`` (and a :class:`LinearSolveReport` if ``return_report``).
    """
    K = sp.csr_matrix(K)
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"matrix must be square, got {K.shape}")
    f = np.asarray(f, dtype=float)
    requested = resolve_method(method)
    report = LinearSolveReport(method="", requested=requested, n=K.shape[0], nnz=K.nnz)
    if K.shape[0] == 0:
        report.method = "none"
        x = np.zeros_like(f)
        return (x, report) if return_report else x

    m = requested
    if m == "auto":
        m = _auto()
    elif not _usable(m):
        if strict:
            raise LinearSolverError(f"linear solver {m!r} is not available here "
                                    f"(available: {[k for k, v in available_methods().items() if v]})")
        fb = _auto()
        report.notes.append(f"requested linear solver {m!r} unavailable; fell back to {fb!r}")
        m = fb
    if symmetric is None:
        symmetric = _is_symmetric(K)

    t0 = time.perf_counter()
    try:
        if m == "scipy":
            x = _solve_scipy(K, f, report, symmetric)
        elif m == "pardiso":
            x = _solve_pardiso(K, f, report, symmetric)
        elif m == "cudss":
            x = _solve_cudss(K, f, report, symmetric)
        elif m == "cg":
            x = _solve_cg(K, f, report, symmetric, rtol, near_nullspace)
        else:  # pragma: no cover
            raise ValueError(m)
    except LinearSolverError:
        raise
    except Exception as e:
        if strict or m == "scipy":
            raise LinearSolverError(f"linear solve with {m!r} failed: {e}") from e
        report.notes.append(f"{m!r} failed ({type(e).__name__}: {e}); fell back to 'scipy'")
        m = "scipy"
        x = _solve_scipy(K, f, report, symmetric)
    report.time = time.perf_counter() - t0
    report.method = m

    x = np.asarray(x, dtype=float).reshape(f.shape)
    if not np.all(np.isfinite(x)):
        raise LinearSolverError("linear solve produced non-finite values (singular matrix: "
                                "missing constraints / rigid-body modes?)")
    r = K @ x - f
    fn = float(np.linalg.norm(f))
    report.residual = float(np.linalg.norm(r)) / fn if fn > 0 else float(np.linalg.norm(r))
    return (x, report) if return_report else x


__all__ = ["solve_linear", "available_methods", "resolve_method", "LinearSolveReport",
           "LinearSolverError", "METHODS", "ENV_VAR"]
