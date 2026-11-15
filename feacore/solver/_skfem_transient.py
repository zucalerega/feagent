"""Transient heat conduction for the scikit-fem backend (docs/solver_transient.md, "Transient heat conduction").

    rho cp dT/dt - div(k grad T) = q,    T(x, 0) = T0(x)

with the boundary conditions and the source of the steady heat case (``temperature``, ``flux``,
``convection``, ``source``), each of which may now depend on the time: an expression with ``t`` or a table
(``feacore/case.py``, "Time-dependent and modal analyses"); for a convection BC also ``h`` and ``T_inf``
(in time only, not in x, y, z). ``solve(case, **options)`` is the entry point; ``skfem_backend.solve`` sends
every heat case with ``analysis.options["transient"]`` here.

Space.  The same P1 / P2 space, quadrature, groups and conventions as the steady solve (thickness, 2 pi r).
The semi-discrete problem is ``M dT/dt + K(t) T = f(t)`` with the consistent capacity matrix
``M = int rho cp N N`` (rho cp per material region). K and f come from the steady assembler
``skfem_backend._assemble_heat``, never from a second implementation: it is called on sub-cases, once for
everything that does not depend on time, once with a unit value for a term whose value depends on the time
only (the term is then scaled per step: ``h(t)``, ``T_inf(t)``, a flux, a source or a temperature given by a
table or by an expression in t), and at every step only for a value that depends on x and t together. A
boundary term is assembled on its facets alone (one cell with zero conductivity stands in for the conduction
matrix); a source that depends on x and t together also pays for a zero conduction matrix each step.

Time.  A fixed step dt = t_end / n from ``Analysis.time_steps()``. With A(a, t) = a M + K(t) and the free
(not prescribed) unknowns:
    crank_nicolson (default)  A(2/dt, t1) T1 = (2/dt M - K(t0)) T0 + f(t0) + f(t1)          second order
    backward_euler            A(1/dt, t1) T1 = (1/dt) M T0 + f(t1)                           first order
    bdf2                      A(3/(2 dt), t1) T1 = M (2 T0 - T_1 / 2) / dt + f(t1)           second order
BDF2 takes its first step with backward Euler (one step of dt: its local error is O(dt^2), so the result
stays second order). A prescribed temperature g(t) enters with its value at the new time in T1 and, for
Crank-Nicolson and BDF2, through the old states, which hold g at the old times; so M (g(t1) - g(t0)) / dt
and K g are both in the right-hand side of the free unknowns. The matrix A is factored once and the
factorization solves every step; it is factored again only when it changes: a time-dependent ``h``, and the
second step of BDF2. ``linsolve.solve_linear`` has no reusable factorization, so this module keeps its own:
MKL PARDISO (SPD) through ``pypardiso`` when it is installed, else SciPy SuperLU in symmetric mode, the two
CPU methods of ``linsolve`` with the same settings. cuDSS and CG are not used here; asking for them is noted
in ``info.warnings``.

Start-up of Crank-Nicolson.  Crank-Nicolson does not damp the modes a step cannot resolve: it multiplies
them by nearly -1 each step. When the initial state does not satisfy a temperature BC at t = 0 (a quench: a
face suddenly held at another temperature) those modes are excited at full strength and the solution rings
from step to step near the face. The same happens, less visibly, when a flux or the fluid temperature of a
convection BC steps at t = 0: the temperature stays continuous, but the initial state does not satisfy the
boundary condition's flux, and Crank-Nicolson's surface temperature then alternates by up to 25 % of the
step for many steps (measured: the convective slab at Bi = 10, 100 steps to Fo = 1; 3 % at Bi = 1). This is
handled, not hidden: the first two steps are each taken as two backward-Euler half-steps (Rannacher's
start-up; four half-steps are what restores second order for discontinuous data: Rannacher, Numer. Math. 43
(1984); Luskin & Rannacher, SIAM J. Numer. Anal. 19 (1982); Giles & Carter, J. Comput. Finance 9 (2006)).
The half-step matrix 2/dt M + K is the Crank-Nicolson matrix, so the start-up costs no factorization.
When it is used (``cn_startup`` = "auto", the default): whenever the initial state is not a steady state of
the data at t = 0, i.e. the residual f(0) - K(0) T0 of the free unknowns is above 1e-9 of its terms (and so
always after a jump of a prescribed temperature, detected separately against 1e-9 of the largest
temperature and named in ``info.warnings``). Telling a flux that is incompatible with the initial gradient
from a compatible one would need a smoothness test of the initial rate that small meshes defeat; the
start-up costs two extra solves and keeps second order for smooth data too (verified: the manufactured
solution's observed order is 2.0 with it, and the projection of a smooth initial state onto the mesh is itself
a slightly rough start that Crank-Nicolson alone carries along: observed 1.6 there without it). The choice
over a warning: the ringing does not decay on a fine mesh, so a warning would leave a wrong answer.
"always" forces the start-up; "off" never uses it (an incompatible temperature BC is then a warning).
Backward Euler and BDF2 damp the unresolved modes and need no start-up.

Results.  ``TransientHeatResult``: ``times``, ``kept``, ``series(qoi)`` (every QoI of the case at every
step, evaluated while the solver runs, by ``SkfemResult``'s own evaluation), ``state(t)`` (a ``SkfemResult``
of a kept step: probe, nodal, evaluate, write_vtu), ``info``. The state at t = 0 is the initial state as
given. ``reaction`` (the heat flow into the body through a temperature BC) is M dT/dt + K T - f on the
constrained nodes, with dT/dt the backward difference of the scheme's order (backward Euler: two levels;
Crank-Nicolson and BDF2: three levels, two at the first step; at t = 0 the forward difference of the first
step).

Checks that end in ``info.warnings``: a value in time that changes between two steps by more than a straight
line between them (the midpoint differs by more than 2 % of the value's range: fewer than about 11 steps per
period), an incompatible initial state, and a first step whose penetration depth sqrt(alpha dt) is below the
cell size at a face whose temperature jumps (the consistent capacity matrix then over- and undershoots in
the first cells).
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import time

import numpy as np
import scipy.sparse as sp
from skfem import Basis, FacetBasis, Functional

from feacore.case import (BC, Case, CaseError, Load, QoI, _compile, as_field, table_value, uses_time)
from feacore.solver import _skfem_assembly as fast
from feacore.solver import linsolve
from feacore.solver import skfem_backend as skb
from feacore.solver._skfem_locate import Locator
from feacore.solver.base import SolveInfo, TransientResult

OWN_OPTIONS = ("cn_startup",)          # options of this module, beside skfem_backend.KNOWN_OPTIONS
STARTUP_STEPS = 2                      # Crank-Nicolson steps replaced by two backward-Euler half-steps each
JUMP_TOL = 1e-9                        # initial jump at a temperature BC, relative to the largest temperature
JUMP_RATIO = 3.0                       # a later jump: the data's change over a step against its neighbours'
RESOLVE_TOL = 0.02                     # a value in time: midpoint against the chord, relative to its range
FLUX_FIELDS = ("qx", "qy", "qz", "q_mag")
_SKIP_BC, _SKIP_LOAD = BC(type="-"), Load(type="-", value=0.0)     # ignored by the steady assembler


# --------------------------------------------------------------------------- values in time


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _names(value) -> set:
    return set(_compile(value).co_names) if isinstance(value, str) else set()


def _space_free(value) -> bool:
    """True for a number, a table, or an expression without x, y, z."""
    return not _names(value) & {"x", "y", "z"}


def _at(value, t: float) -> float:
    """A value without x, y, z at the time t."""
    return float(as_field(value, 0.0, 0.0, 0.0, t))


class _SetTime(ast.NodeTransformer):
    def __init__(self, t: float):
        self.t = float(t)

    def visit_Name(self, node):
        return ast.copy_location(ast.Constant(self.t), node) if node.id == "t" else node


def _freeze(value, t: float):
    """A value at the time t as a steady case takes it: a number, or an expression in x, y, z only (the
    name t is replaced by the number, written with repr, so no digit is lost)."""
    if isinstance(value, dict) and "table" in value:
        return table_value(value, t)
    if isinstance(value, str) and uses_time(value):
        tree = _SetTime(t).visit(ast.parse(value, mode="eval"))
        return ast.unparse(ast.fix_missing_locations(tree))
    return value


# --------------------------------------------------------------------------- the system in time


class _View:
    """A model with some attributes replaced (the case, the basis, the conductivity): the steady assembler
    reads these from the view and everything else from the model."""

    def __init__(self, model, **replaced):
        self.__dict__["_model"] = model
        self.__dict__.update(replaced)

    def __getattr__(self, name):          # only for names that the view does not hold itself
        return getattr(self._model, name)


@dataclasses.dataclass
class _At:
    """The data at one time: the convection coefficients that scale the Robin matrices, the load vector and
    the prescribed temperatures (on ``_System.D``)."""
    t: float
    h: np.ndarray
    f: np.ndarray
    g: np.ndarray

    @property
    def key(self) -> bytes:
        return self.h.tobytes()


class _System:
    """K(t), f(t) and the prescribed temperatures g(t) of M dT/dt + K(t) T = f(t), built with the steady
    assembler on sub-cases (module docstring, "Space")."""

    def __init__(self, md: skb._Model):
        self.md = md
        case = md.case
        one = Basis(md.mesh, md.selem, intorder=md.intorder, elements=np.array([0]))
        self._one = dict(sb=one, xq=skb._coords(one), k=np.zeros(1))     # boundary terms: no conduction matrix
        self.timed: list[tuple[str, object]] = []      # (label, value) of every value given in time only

        def label(i, bc, what="value"):
            return f"bcs[{i}] ({bc.type}) {what}"

        static_b, static_l = {}, {}
        self.scaled_f: list[tuple[object, np.ndarray]] = []         # value(t) * unit load vector
        self.conv: list[tuple[object, object, sp.csr_matrix, np.ndarray]] = []   # h(t), T_inf(t), unit Robin K, f
        self.gen_b, self.gen_l = {}, {}                             # values in x and t: assembled per step
        temps = {}
        for i, bc in enumerate(case.bcs):
            try:
                if bc.type == "temperature":
                    temps[i] = bc
                elif bc.type == "flux":
                    if not uses_time(bc.value):
                        static_b[i] = bc
                    elif _space_free(bc.value):
                        _, fu, _ = self._assemble(md.warnings, bcs={i: dataclasses.replace(bc, value=1.0)})
                        self.scaled_f.append((bc.value, fu))
                        self.timed.append((label(i, bc), bc.value))
                    else:
                        self.gen_b[i] = bc
                elif bc.type == "convection":
                    if _is_number(bc.h) and _is_number(bc.T_inf):
                        static_b[i] = bc
                        continue
                    for what, v in (("h", bc.h), ("T_inf", bc.T_inf)):
                        if not _space_free(v):
                            raise CaseError(f"{label(i, bc, what)} = {v!r}: h and T_inf of a convection BC may "
                                            f"depend on the time t, not on x, y, z")
                        if not _is_number(v):
                            self.timed.append((label(i, bc, what), v))
                    Ku, fu, _ = self._assemble(md.warnings, bcs={i: dataclasses.replace(bc, h=1.0, T_inf=1.0)})
                    Ku.eliminate_zeros()
                    self.conv.append((bc.h, bc.T_inf, Ku, fu))
            except SyntaxError as e:
                raise CaseError(f"{label(i, bc)}: bad expression: {e}") from e
        for i, ld in enumerate(case.loads):
            if ld.type != "source":
                continue
            try:
                timed = uses_time(ld.value)
            except SyntaxError as e:
                raise CaseError(f"loads[{i}] (source) value: bad expression: {e}") from e
            if not timed:
                static_l[i] = ld
            elif _space_free(ld.value):
                _, fu, _ = self._assemble(md.warnings, loads={i: dataclasses.replace(ld, value=1.0)})
                self.scaled_f.append((ld.value, fu))
                self.timed.append((f"loads[{i}] (source) value", ld.value))
            else:
                self.gen_l[i] = ld

        # everything that does not depend on time: the conduction matrix, constant convection, flux, source
        self.K0, self.f0, _ = skb._assemble_heat(_View(md, case=self._sub(static_b, static_l)))
        self.K0 = sp.csr_matrix(self.K0)
        self._K: tuple[bytes, sp.csr_matrix] | None = None

        # temperature BCs, in the order of the case (the last one wins on a shared node, as in a steady case)
        self.dir: list[dict] = []
        if temps:
            _, _, cons = self._assemble(md.warnings, bcs={i: dataclasses.replace(bc, value=_freeze(bc.value, 0.0))
                                                          for i, bc in temps.items()})
            for (i, bc), idx, val in zip(temps.items(), cons.idx, cons.val):
                try:
                    timed = uses_time(bc.value)
                except SyntaxError as e:
                    raise CaseError(f"{label(i, bc)}: bad expression: {e}") from e
                kind = "static" if not timed else "scaled" if _space_free(bc.value) else "generic"
                self.dir.append(dict(i=i, bc=bc, idx=idx, val=val, kind=kind,
                                     group=bc.group if bc.point is None else None))
                if kind == "scaled":
                    self.timed.append((label(i, bc), bc.value))
                elif kind == "generic":
                    self.gen_b[i] = bc
        self.time_dependent = bool(self.timed or self.gen_b or self.gen_l)
        self._last: _At | None = None
        d0 = self.at(0.0, md.warnings)
        self.D = self._D
        free = np.ones(md.N, dtype=bool)
        free[self.D] = False
        self.I = np.nonzero(free)[0]
        self._static = None if self.time_dependent else d0

    # ---- the steady assembler on a sub-case
    def _sub(self, bcs: dict, loads: dict) -> Case:
        """The case with only the given BCs and loads, each at its own index (so that an error message of
        the assembler names the BC of the full case)."""
        case = self.md.case
        return dataclasses.replace(case, bcs=[bcs.get(i, _SKIP_BC) for i in range(len(case.bcs))],
                                   loads=[loads.get(i, _SKIP_LOAD) for i in range(len(case.loads))])

    def _assemble(self, warnings: list, bcs: dict | None = None, loads: dict | None = None):
        """``_assemble_heat`` for these BCs (on their facets; the one cell with k = 0 gives a zero conduction
        matrix) or these sources (all cells, k = 0)."""
        case = self._sub(bcs or {}, loads or {})
        if loads:
            view = _View(self.md, case=case, k=np.zeros(self.md.nel), warnings=warnings)
        else:
            view = _View(self.md, case=case, warnings=warnings, **self._one)
        return skb._assemble_heat(view)

    # ---- the data at a time
    def at(self, t: float, warnings: list | None = None) -> _At:
        if self._last is not None and self._last.t == t:
            return self._last
        if getattr(self, "_static", None) is not None:
            return dataclasses.replace(self._static, t=t)
        scratch = [] if warnings is None else warnings
        f = self.f0.copy()
        for value, fu in self.scaled_f:
            f += _at(value, t) * fu
        h = np.array([_at(hv, t) for hv, _, _, _ in self.conv])
        for hj, (_, T_inf, _, fu) in zip(h, self.conv):
            f += hj * _at(T_inf, t) * fu
        generic = {}
        if self.gen_b:
            _, fb, cons = self._assemble(scratch, bcs={i: dataclasses.replace(bc, value=_freeze(bc.value, t))
                                                       for i, bc in self.gen_b.items()})
            f += fb
            generic = dict(zip([i for i, bc in self.gen_b.items() if bc.type == "temperature"], cons.val))
        if self.gen_l:
            _, fl, _ = self._assemble(scratch, loads={i: dataclasses.replace(ld, value=_freeze(ld.value, t))
                                                      for i, ld in self.gen_l.items()})
            f += fl
        cons = skb._Constraints()
        for e in self.dir:
            val = e["val"] if e["kind"] == "static" else _at(e["bc"].value, t) if e["kind"] == "scaled" \
                else generic[e["i"]]
            cons.add(e["idx"], val, e["group"])
        self._D, g = cons.finalize(scratch)
        self._last = _At(float(t), h, f, g)
        return self._last

    def K(self, d: _At) -> sp.csr_matrix:
        """The conductivity matrix at the time of ``d``: conduction and constant convection, plus h(t) times
        the unit Robin matrix of every convection BC given in time."""
        if self._K is None or self._K[0] != d.key:
            K = self.K0
            for hj, (_, _, Ku, _) in zip(d.h, self.conv):
                K = K + hj * Ku
            self._K = (d.key, sp.csr_matrix(K))
        return self._K[1]

    @property
    def matrix_in_time(self) -> bool:
        return any(not _is_number(h) and uses_time(h) for h, _, _, _ in self.conv)


# --------------------------------------------------------------------------- linear algebra


class _Factor:
    """A factorization of a symmetric positive definite matrix that solves many right-hand sides: MKL
    PARDISO (mtype 2, upper triangle) or SciPy SuperLU in symmetric mode, as ``linsolve`` sets them up."""

    def __init__(self, A: sp.csr_matrix, method: str):
        self.n, self.nnz = A.shape[0], A.nnz
        self.method, self.detail, self.notes = method, "", []
        self._pardiso = self._lu = None
        if self.n == 0:
            self.method = "none"
            return
        if method == "pardiso":
            try:
                from pypardiso import PyPardisoSolver

                self._A = sp.triu(A, format="csr")
                self._A.sort_indices()
                self._pardiso = PyPardisoSolver(mtype=2)
                self._pardiso.factorize(self._A)
                self.detail = "MKL PARDISO, mtype=2 (SPD)"
                return
            except Exception as e:  # noqa: BLE001  as linsolve: any PARDISO failure falls back to SciPy
                self.close()
                self.notes.append(f"PARDISO factorization failed ({type(e).__name__}: {e}); SciPy SuperLU used")
                self.method = "scipy"
        from scipy.sparse.linalg import splu

        try:
            self._lu = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                            options={"SymmetricMode": True})
            self.detail = "SuperLU, MMD_AT_PLUS_A ordering, symmetric mode"
        except RuntimeError as e:
            self.notes.append(f"symmetric-mode SuperLU failed ({e}); retried with COLAMD + pivoting")
            self._lu = splu(A.tocsc(), permc_spec="COLAMD")
            self.detail = "SuperLU, COLAMD ordering"

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if self.n == 0:
            return np.zeros_like(b)
        if self._pardiso is not None:
            return np.asarray(self._pardiso.solve(self._A, b)).reshape(b.shape)
        return self._lu.solve(b)

    def close(self) -> None:
        if self._pardiso is not None:
            try:
                self._pardiso.free_memory(everything=True)
            except Exception:  # noqa: BLE001
                pass
            self._pardiso = None

    def __del__(self):
        self.close()


def _direct_method(requested: str | None, n: int, warnings: list) -> str:
    """The CPU direct method that time stepping uses (module docstring, "Time")."""
    req = linsolve.resolve_method(requested)
    have = linsolve._have_pypardiso()
    cpu = "pardiso" if have else "scipy"
    if req == "scipy":
        return "scipy"
    if req == "pardiso" and not have:
        warnings.append("linear solver: requested 'pardiso' unavailable; fell back to 'scipy'")
    elif req in ("cudss", "cg"):
        warnings.append(f"linear solver: {req!r} is not used for time stepping (a factorization is kept and reused "
                        f"on the CPU); {cpu!r} used")
    elif req == "auto" and linsolve._have_cudss() and n >= linsolve.cudss_min_n():
        warnings.append(f"info: linear solver: time stepping keeps its factorization on the CPU ({cpu}); cuDSS is "
                        f"not used")
    return cpu


class _Stepper:
    """Solves (a M + K(t)) T = b with T = g(t) on the constrained nodes, keeping the factorization while
    a and K(t) stay the same."""

    def __init__(self, system: _System, M: sp.csr_matrix, method: str):
        self.sys, self.M, self.method = system, M, method
        self.key = None
        self.A = self.factor = None
        self.n_factor = 0
        self.max_residual = 0.0
        self.time_factor = self.time_solve = 0.0
        self.notes: list[str] = []

    def solve(self, a: float, d: _At, b: np.ndarray) -> np.ndarray:
        s = self.sys
        key = (float(a), d.key)
        if key != self.key:
            t0 = time.perf_counter()
            if self.factor is not None:
                self.factor.close()
            self.A = sp.csr_matrix(a * self.M + s.K(d))
            self.factor = _Factor(sp.csr_matrix(self.A[s.I][:, s.I]), self.method)
            self.key = key
            self.n_factor += 1
            self.notes += [n for n in self.factor.notes if n not in self.notes]
            self.time_factor += time.perf_counter() - t0
        t0 = time.perf_counter()
        x = np.zeros(s.md.N)
        x[s.D] = d.g
        rhs = (b - self.A @ x)[s.I]
        x[s.I] = self.factor.solve(rhs)
        self.time_solve += time.perf_counter() - t0
        if not np.all(np.isfinite(x)):
            raise skb.SolverError("transient heat: the linear solve produced non-finite values")
        res = (self.A @ x - b)[s.I]
        scale = float(np.linalg.norm(rhs))
        rel = float(np.linalg.norm(res)) / scale if scale > 0 else float(np.linalg.norm(res))
        self.max_residual = max(self.max_residual, rel)
        if rel > 1e-6:
            raise skb.SolverError(f"transient heat: linear solve inaccurate at t = {d.t:g} (relative residual "
                                  f"{rel:.2e})")
        return x

    def describe(self) -> str:
        f = self.factor
        return (f"linear_solver={f.method} ({f.detail}), n={f.n}, nnz={f.nnz}, factorizations={self.n_factor}, "
                f"t_factor={self.time_factor:.3g}s, t_solve={self.time_solve:.3g}s, "
                f"max rel_residual={self.max_residual:.2e}")


class _FluxRecovery:
    """q = -k grad T projected onto the solution space, as the steady ``_recover_heat`` does (global L2
    projection), with the mass matrix factored once for all steps."""

    def __init__(self, md: skb._Model, method: str):
        self.md, self.method, self.factor = md, method, None

    def __call__(self, T: np.ndarray) -> dict:
        md = self.md
        if self.factor is None:
            if not hasattr(md, "_M"):
                md._M = skb._mass.assemble(md.sb) if md.opts.get("assembly", "fast") == "skfem" else fast.mass(md.sb)
            self.factor = _Factor(sp.csr_matrix(md._M), self.method)
        G = np.asarray(md.sb.interpolate(T).grad)
        k = skb._qp(md.sb, md.k)
        names = ("qx", "qy", "qz")[: md.dim]
        B = np.column_stack([skb._rhs_proj.assemble(md.sb, s=-k * G[i]) for i in range(md.dim)])
        X = self.factor.solve(B)
        return {n: X[:, i].copy() for i, n in enumerate(names)}


# --------------------------------------------------------------------------- states and QoIs


class _LocatorMemo:
    """The point location of a probe, kept per set of points: a point QoI is located once, not every step."""

    def __init__(self, locator):
        self.locator, self.memo = locator, {}

    def locate(self, pts, *args, **kwargs):
        key = (np.asarray(pts).tobytes(), args, tuple(sorted(kwargs.items())))
        if key not in self.memo:
            self.memo[key] = self.locator.locate(pts, *args, **kwargs)
        return self.memo[key]


class HeatState(skb.SkfemResult):
    """The state of one time as a steady-style result. Every state of a run shares the point locator, the
    node sets of the groups and the quadrature of the integrals, so a QoI costs little per step."""

    def __init__(self, shared: dict, *args):
        super().__init__(*args)
        self._shared = shared
        self.model = shared["model"]
        self._locator = shared["locator"]
        self.time = 0.0

    def _integral(self, field, group):
        md, memo = self.model, self._shared["integrals"]
        if group not in memo:        # the basis of SkfemResult._integral, built once per group
            if group in (None, "*"):
                kind, idx = "cells", np.arange(md.nel)
            else:
                g = md.md.group(group)
                kind, idx = g.kind, g.index
            if kind == "cells":
                b = Basis(md.mesh, md.selem, intorder=md.intorder, elements=idx)
            elif kind == "facets":
                b = FacetBasis(md.mesh, md.selem, facets=idx, intorder=md.intorder)
            else:
                raise CaseError(f"integral/mean need a cell or facet group; {group!r} is a {kind} group")
            w = md.wfac(skb._coords(b)) * b.dx
            memo[group] = (b, w, float(np.sum(w)))
        b, w, meas = memo[group]
        return float(np.sum(self._qp_values(field, b) * w)), meas


def _qoi_key(q: QoI) -> tuple:
    """What a QoI evaluates at one time (its name and its reduction in time left out)."""
    return (q.type, q.field, None if q.point is None else tuple(float(v) for v in q.point), q.group, q.component,
            None if q.point2 is None else tuple(float(v) for v in q.point2))


def _needs_flux(q: QoI) -> bool:
    return q.type != "reaction" and q.field in FLUX_FIELDS


class TransientHeatResult(TransientResult):
    """A transient heat solution (module docstring, "Results")."""

    def __init__(self, case, info, model, system, shared, times, kept, series, temperatures, reactions,
                 recover, scheme, startup_steps):
        self.case, self.info, self.model = case, info, model
        self.times = np.asarray(times, dtype=float)
        self.kept = np.asarray(kept, dtype=np.int64)
        self.scheme, self.dt, self.startup_steps = scheme, float(times[1] - times[0]), int(startup_steps)
        self._system, self._shared, self._series = system, shared, series
        self._T, self._R, self._recover = temperatures, reactions, recover
        self._states: dict[int, HeatState] = {}

    def series(self, qoi: QoI) -> np.ndarray:
        key = _qoi_key(qoi)
        if key in self._series:
            return self._series[key].copy()
        if len(self.kept) == len(self.times):          # every step is kept: evaluate it on the states
            return np.array([self.state(t).evaluate_qoi(qoi) for t in self.times])
        raise CaseError(f"QoI {qoi.name!r} was not evaluated during the run: a QoI is evaluated at every step "
                        f"while the solver runs, so it must be in case.qois (or keep every step: "
                        f"transient.output_every = 1). This run kept {len(self.kept)} of {len(self.times)} steps.")

    def state(self, t: float | None = None) -> HeatState:
        k = int(self.kept[-1]) if t is None else int(self.kept[np.argmin(np.abs(self.times[self.kept] - float(t)))])
        if k not in self._states:
            T, s = self._T[k], self._system
            d = s.at(float(self.times[k]))
            R = np.zeros(self.model.N)
            R[s.D] = self._R[k]
            st = HeatState(self._shared, self.case, self.info, self.model, T, self._recover(T), s.K(d), d.f, R, s.D)
            st.time = float(self.times[k])
            self._states[k] = st
        return self._states[k]

    def temperature(self, t: float | None = None) -> np.ndarray:
        """Nodal temperatures of the kept state nearest to t (at the nodes of ``nodal("T")``)."""
        return self.state(t).fields["T"].copy()

    def evaluate_all(self) -> dict[str, float]:
        """Every QoI of the case, reduced in time as it asks; one that cannot be evaluated gives NaN and a
        warning, as in a steady result."""
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

    def write_vtu(self, path, t: float | None = None):
        """The fields of the kept state nearest to t (default: the last step)."""
        return self.state(t).write_vtu(path)


# --------------------------------------------------------------------------- set-up helpers


def _initial_temperature(md: skb._Model, value) -> np.ndarray:
    """The state at t = 0 at the nodes of the solution space: a number, an expression in x, y, z, or a field
    file, which goes to the nodes by the rule of the steady thermal load (``skfem_backend._field_at_qp``:
    nearest point, every node needs a field point within FIELD_MATCH_TOL of the bounding-box diagonal)."""
    X = md.sb.doflocs
    what = "analysis.options.transient.initial.T"
    if isinstance(value, dict) and "field" in value:
        if set(value) != {"field"} or not isinstance(value["field"], str) or not value["field"].strip():
            raise CaseError(f'{what}: a field must be {{"field": "<path to an .npz file with x (n, dim) and T (n,)>"}}')

        class _Nodes:                      # the load's "interpolation to the quadrature points" kept at the nodes
            element_dofs, doflocs, N = md.sb.element_dofs, md.sb.doflocs, md.sb.N
            interpolate = staticmethod(lambda nodal: nodal)

        notes: list[str] = []
        ld = Load(type="temperature", value=value, group="*")
        try:
            T0 = np.array(skb._field_at_qp(_View(md, sb=_Nodes, warnings=notes), ld, 0, np.arange(md.nel)), dtype=float)
        except CaseError as e:
            raise CaseError(str(e).replace("loads[0] temperature field", f"{what}: temperature field")) from e
        md.warnings += [n.replace("loads[0] temperature field", "initial temperature field") for n in notes]
        return T0
    if isinstance(value, dict):
        raise CaseError(f"{what} must be a number, an expression in x, y, z or {{\"field\": path}}, got {value!r}")
    try:
        T0 = as_field(value, X[0], X[1], X[2] if X.shape[0] == 3 else None)
    except (SyntaxError, TypeError, ValueError) as e:
        if isinstance(e, CaseError):
            raise
        raise CaseError(f"{what} = {value!r}: {e}") from e
    if not np.all(np.isfinite(T0)):
        raise CaseError(f"{what} = {value!r} is not finite at {int((~np.isfinite(T0)).sum())} nodes")
    return T0


def _kept_steps(tr: dict, times: np.ndarray) -> np.ndarray:
    n = len(times) - 1
    if tr.get("output_times") is not None:
        idx = [int(np.argmin(np.abs(times - float(t)))) for t in tr["output_times"]]
    elif tr.get("output_every") is not None:
        idx = list(range(0, n + 1, int(tr["output_every"])))
    else:                                  # 10 states evenly spaced in time after the initial one
        idx = np.rint(np.linspace(0, n, 11)).astype(int).tolist()
    return np.unique(np.array(list(idx) + [n], dtype=np.int64))


def _resolution_warnings(system: _System, times: np.ndarray, warnings: list) -> None:
    """A value in time must be resolved by the step: at the middle of a step it should lie on the straight
    line between its two step values (the schemes see the step values only)."""
    if not system.timed:
        return
    mid = 0.5 * (times[1:] + times[:-1])
    for label, value in system.timed:
        s = np.array([_at(value, t) for t in times])
        m = np.array([_at(value, t) for t in mid])
        dev = np.abs(m - 0.5 * (s[1:] + s[:-1]))
        rng = float(max(s.max(), m.max()) - min(s.min(), m.min()))
        if rng > 0 and float(dev.max()) > RESOLVE_TOL * rng:
            j = int(np.argmax(dev))
            warnings.append(
                f"{label} changes faster than the time step resolves: between t = {times[j]:g} and {times[j + 1]:g} "
                f"it leaves the straight line between its step values by {float(dev[j]):.3g} "
                f"({100 * float(dev[j]) / rng:.0f} % of its range {rng:.3g}); the scheme sees the step values only. "
                f"Reduce dt ({times[1] - times[0]:g}) or put a step where the value changes.")


def _data_jumps(system: _System, times: np.ndarray) -> list[int]:
    """The steps n over which the boundary or load data jump: between t_n and t_n+1 the change of the load
    vector f, of the prescribed temperatures g or of the convection coefficients h is above JUMP_TOL of its
    size and more than JUMP_RATIO times its change over each neighbouring step. Data that are smooth in time
    and resolved by the step change by about as much from one step to the next (a sine with 11 steps per
    period: at most 1.2 times), a kink (a ramp that starts) changes less than its neighbours, and a jump
    changes against neighbours that do not change at all."""
    n_steps = len(times) - 1
    if not system.time_dependent or n_steps < 2:
        return []
    deltas = {p: np.zeros(n_steps) for p in ("f", "g", "h")}
    sizes = dict.fromkeys(deltas, 0.0)
    prev = None
    for n, t in enumerate(times):
        d = system.at(float(t))
        cur = {"f": d.f[system.I], "g": d.g, "h": d.h}
        for p, v in cur.items():
            if v.size:
                sizes[p] = max(sizes[p], float(np.linalg.norm(v)))
                if prev is not None:
                    deltas[p][n - 1] = float(np.linalg.norm(v - prev[p]))
        prev = {p: v.copy() for p, v in cur.items()}
    jumps = set()
    for p, dl in deltas.items():
        if sizes[p] <= 0.0:
            continue
        nb = np.maximum(np.r_[0.0, dl[:-1]], np.r_[dl[1:], 0.0])
        big = (dl > JUMP_TOL * sizes[p]) & (dl > JUMP_RATIO * nb)
        jumps.update(int(n) for n in np.nonzero(big)[0])
    return sorted(jumps)


def _jump_is_late(system: _System, t0: float, t1: float) -> bool:
    """Whether a jump between t0 and t1 lies in the second half of the step (the data at the middle are
    still those at t0), e.g. ``where(t < t1, ...)`` with t1 a step time."""
    parts = [(d.f[system.I].copy(), d.g.copy(), d.h.copy()) for d in
             (system.at(t) for t in (t0, 0.5 * (t0 + t1), t1))]
    near_a = near_b = 0.0
    for a, m, b in zip(*parts):           # each part in its own units
        if a.size and np.linalg.norm(b - a) > 0.0:
            near_a += np.linalg.norm(m - a) / np.linalg.norm(b - a)
            near_b += np.linalg.norm(m - b) / np.linalg.norm(b - a)
    return bool(near_a < near_b)


def _first_cell_warning(md: skb._Model, rc: np.ndarray, nodes: np.ndarray, dt: float, warnings: list) -> None:
    """A face whose temperature jumps at t = 0: compare the penetration depth of one step, sqrt(alpha dt),
    with the size of the cells at the face."""
    if len(nodes) == 0:
        return
    mark = np.zeros(md.sb.N, dtype=bool)
    mark[nodes] = True
    cells = np.nonzero(mark[md.sb.element_dofs].any(axis=0))[0]
    if len(cells) == 0:
        return
    vol = np.asarray(Functional(lambda w: 1.0 + 0.0 * w.x[0]).elemental(md.sb))[cells]
    size = (np.abs(vol) * (2.0 if md.dim == 2 else 6.0)) ** (1.0 / md.dim)       # edge of the regular simplex's box
    depth = np.sqrt(md.k[cells] / rc[cells] * dt)
    ratio = depth / size
    if float(ratio.min()) < 1.0:
        j = int(np.argmin(ratio))
        warnings.append(
            f"the temperature jumps at t = 0 on a face whose cells are larger than the penetration depth of one "
            f"step: sqrt(alpha dt) = {float(depth[j]):.3g} against a cell size of {float(size[j]):.3g} "
            f"({int((ratio < 1.0).sum())} of {len(cells)} cells at the face). Expect over- and undershoots of the "
            f"jump in the first cells during the first steps; refine the mesh towards the face (or accept the "
            f"early times as unresolved).")


# --------------------------------------------------------------------------- driver


def solve(case: Case, **options) -> TransientHeatResult:
    t_start = time.perf_counter()
    case.check()
    a = case.analysis
    tr = a.transient
    if a.physics != "heat" or tr is None:
        raise CaseError("feacore.solver._skfem_transient solves transient heat conduction: physics 'heat' with "
                        "analysis.options.transient")
    opts = {**(a.options or {}), **options}
    startup = opts.get("cn_startup", "auto")
    if startup not in ("auto", "always", "off"):
        raise CaseError(f"analysis.options.cn_startup must be 'auto', 'always' or 'off', got {startup!r}")
    scheme = tr.get("scheme") or "crank_nicolson"
    initial = (tr.get("initial") or {}).get("T")
    if initial is None:
        raise CaseError('transient heat conduction needs the temperature at t = 0: '
                        'analysis.options.transient.initial = {"T": number | expression in x, y, z | '
                        '{"field": "<.npz file with x and T>"}}')
    times = a.time_steps()
    n_steps = len(times) - 1
    dt = float(times[-1]) / n_steps
    kept = _kept_steps(tr, times)

    md = skb._Model(case, {k: v for k, v in opts.items() if k not in OWN_OPTIONS})
    W = md.warnings
    N = md.N
    names = list(case.materials)
    rc = np.array([case.materials[n].rho * case.materials[n].cp for n in names])[md.mat]     # per cell
    c = md.wfac(md.xq)
    if md.opts.get("assembly", "fast") == "skfem":
        M = skb._m_weighted.assemble(md.sb, c=c, h=skb._qp(md.sb, rc))
    else:
        M = fast.mass(md.sb, c * skb._qp(md.sb, rc))
    M = sp.csr_matrix(M)
    system = _System(md)
    D = system.D
    T = _initial_temperature(md, initial)
    d = system.at(0.0)

    # the initial state against the temperature BCs at t = 0
    n_startup = 0
    jump = float(np.abs(d.g - T[D]).max()) if len(D) else 0.0
    scale = max(float(np.abs(T).max()), float(np.abs(d.g).max()) if len(D) else 0.0)
    incompatible = jump > JUMP_TOL * scale
    if incompatible:
        nodes = D[np.abs(d.g - T[D]) > JUMP_TOL * scale]
        what = f"the initial state does not satisfy the temperature BCs at t = 0 (largest jump {jump:.6g} " \
               f"at {len(nodes)} nodes)"
        if scheme != "crank_nicolson":
            W.append(f"info: {what}; {scheme} damps the modes a step cannot resolve")
        elif startup == "off":
            W.append(f"{what} and cn_startup is 'off': Crank-Nicolson does not damp the modes this excites, the "
                     f"solution rings from step to step near these faces")
        _first_cell_warning(md, rc, nodes, dt, W)
    # the initial state against the rest of the data at t = 0: is it a steady state (nothing to damp)?
    x0 = T.copy()
    x0[D] = d.g
    K0 = system.K(d)
    unbalanced = float(np.linalg.norm((d.f - K0 @ x0)[system.I]))
    terms = float(np.linalg.norm(d.f[system.I])) + float(np.linalg.norm((abs(K0) @ np.abs(x0))[system.I]))
    moving = unbalanced > JUMP_TOL * max(terms, 1e-300)       # |K| |x|: K x of a uniform state is round-off
    if scheme == "crank_nicolson" and (startup == "always" or (startup == "auto" and (incompatible or moving))):
        n_startup = min(STARTUP_STEPS, n_steps)
        why = (what if incompatible else "cn_startup = 'always'" if startup == "always" else
               "the initial state is not a steady state of the data at t = 0")
        W.append(f"info: {why}: Crank-Nicolson takes its first {n_startup} steps as {2 * n_startup} backward-Euler "
                 f"half-steps (Rannacher start-up)")
    half_steps = set(range(n_startup))
    held: set[int] = set()                    # steps taken with the data at their start (a jump at their end)
    # a jump of the data later in the run excites the same undamped modes: the same start-up after it
    jumps = [n for n in _data_jumps(system, times) if n > 0 or n_startup == 0] if scheme == "crank_nicolson" else []
    if jumps:
        at = ", ".join(f"{float(times[n]):g}..{float(times[n + 1]):g}" for n in jumps[:8]) + \
            (f" and {len(jumps) - 8} more" if len(jumps) > 8 else "")
        if startup == "off":
            W.append(f"the boundary or load data jump between t = {at} and cn_startup is 'off': Crank-Nicolson does "
                     f"not damp the modes a jump excites, the solution near those faces alternates from step to step "
                     f"after it (measured: 8.5 % of the step at Biot 10). Use cn_startup 'auto' or the scheme bdf2.")
        else:
            for n in jumps:
                # a jump in the second half of the step (at t_n+1 itself, say): the step ends at the left limit,
                # the old data held to t_n+1, and the start-up begins there, as at t = 0
                late = _jump_is_late(system, float(times[n]), float(times[n + 1]))
                if late:
                    held.add(n)
                first = n + 1 if late else n
                half_steps.update(range(first, min(first + STARTUP_STEPS, n_steps)))
            W.append(f"info: the boundary or load data jump between t = {at}: Crank-Nicolson takes the "
                     f"{STARTUP_STEPS} steps from each jump on as backward-Euler half-steps (Rannacher start-up "
                     f"after a jump; a jump in the second half of a step is taken at its end, the state reported "
                     f"there being the one just before the jump, as at t = 0)")
    _resolution_warnings(system, times, W)

    method = _direct_method(opts.get("linear_solver"), len(system.I), W)
    stepper = _Stepper(system, M, method)
    recover = _FluxRecovery(md, method)
    measure = float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(md.sb))
    info = SolveInfo(
        backend="skfem", physics=md.physics, formulation=md.form, order=md.order, dim=md.dim,
        cell_type=md.md.cell_type, n_nodes=int(md.mesh.doflocs.shape[1]), n_cells=int(md.nel), n_dofs=int(N),
        measure=measure, h_rep=float((measure / md.nel) ** (1.0 / md.dim)), wall_time=0.0, warnings=W)

    # QoIs: one evaluator state, refreshed every step
    shared = {"model": _View(md, sdofs_of=functools.lru_cache(maxsize=None)(md.sdofs_of)),
              "locator": _LocatorMemo(Locator(md.mesh, md.md.bbox_diag)), "integrals": {}}
    ev = HeatState(shared, case, info, md, T, {}, None, None, np.zeros(N), D)
    series: dict[tuple, np.ndarray] = {}
    qois: dict[tuple, QoI] = {}
    for q in case.qois:
        qois.setdefault(_qoi_key(q), q)
    with_flux = any(_needs_flux(q) for q in qois.values())
    failed: set[tuple] = set()

    def record(step: int, T_step: np.ndarray, R_D: np.ndarray | None, which: str) -> None:
        """Evaluate the QoIs of one step: ``which`` = "fields" (everything but reactions) or "reactions"."""
        todo = [(k, q) for k, q in qois.items() if (q.type == "reaction") == (which == "reactions") and k not in failed]
        if not todo:
            return
        if which == "fields":
            ev.fields["T"] = T_step
            if with_flux:
                ev.fields.update(recover(T_step))
        else:
            ev.reaction_vector = np.zeros(N)
            ev.reaction_vector[D] = R_D
        for k, q in todo:
            try:
                series.setdefault(k, np.full(n_steps + 1, np.nan))[step] = float(ev.evaluate_qoi(q))
            except (CaseError, ValueError) as e:
                failed.add(k)
                series[k] = np.full(n_steps + 1, np.nan)
                W.append(f"qoi {q.name!r} could not be evaluated: {e}")

    temperatures = {0: T.copy()} if 0 in kept else {}
    reactions: dict[int, np.ndarray] = {}
    record(0, T, None, "fields")
    MT_old = None
    MT = M @ T
    r = system.K(d) @ T - d.f                 # K T - f at the current step
    r0_D = r[D].copy()
    for n in range(n_steps):
        t_new = float(times[n + 1])
        d_new = system.at(t_new if n not in held else float(times[n]))     # held: the data before the jump
        if scheme == "backward_euler" or (scheme == "bdf2" and n == 0):
            T_new = stepper.solve(1.0 / dt, d_new, MT / dt + d_new.f)
        elif scheme == "bdf2":
            T_new = stepper.solve(1.5 / dt, d_new, (2.0 * MT - 0.5 * MT_old) / dt + d_new.f)
        elif n in half_steps:                 # two backward-Euler half-steps; same matrix as Crank-Nicolson
            d_half = d_new if n in held else system.at(float(times[n]) + 0.5 * dt)
            T_half = stepper.solve(2.0 / dt, d_half, (2.0 / dt) * MT + d_half.f)
            d_new = d_new if n in held else system.at(t_new)
            T_new = stepper.solve(2.0 / dt, d_new, (2.0 / dt) * (M @ T_half) + d_new.f)
        else:
            T_new = stepper.solve(2.0 / dt, d_new, (2.0 / dt) * MT - r + d_new.f)
        MT_new = M @ T_new
        r = system.K(d_new) @ T_new - d_new.f
        if scheme == "backward_euler" or n == 0:
            mdot = (MT_new - MT)[D] / dt
        else:
            mdot = (3.0 * MT_new - 4.0 * MT + MT_old)[D] / (2.0 * dt)
        if n == 0:                            # the reaction at t = 0: forward difference of the first step
            R0 = (MT_new - MT)[D] / dt + r0_D
            record(0, T, R0, "reactions")
            if 0 in temperatures:
                reactions[0] = R0
        R_D = mdot + r[D]
        record(n + 1, T_new, None, "fields")
        record(n + 1, T_new, R_D, "reactions")
        if (n + 1) in kept:
            temperatures[n + 1] = T_new.copy()
            reactions[n + 1] = R_D
        MT_old, MT, T = MT, MT_new, T_new

    W.append(f"info: transient heat: scheme={scheme}"
             + (", first step backward Euler" if scheme == "bdf2" else "")
             + f", steps={n_steps}, dt={dt:.6g}, t_end={float(times[-1]):.6g}, kept states={len(kept)}"
             + ("" if not system.time_dependent else
                f", data in time: {len(system.timed)} scaled, {len(system.gen_b) + len(system.gen_l)} assembled per step"))
    W.append("info: " + stepper.describe())
    W += [f"linear solver: {n}" for n in stepper.notes]
    if stepper.factor is not None:
        stepper.factor.close()
    info.wall_time = float(time.perf_counter() - t_start)
    return TransientHeatResult(case, info, md, system, shared, times, kept, series, temperatures, reactions,
                               recover, scheme, n_startup)
