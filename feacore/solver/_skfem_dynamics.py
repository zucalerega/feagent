"""Transient structural dynamics of the fixed solver (elasticity with ``analysis.options["transient"]``).

    M a + C v + K u = f(t),        C = a_R M + b_R K   (``analysis.options["damping"] = {"rayleigh": [a_R, b_R]}``)

K, the loads and the Dirichlet set are those of the steady elasticity case (``skfem_backend._assemble_elasticity``),
M is the consistent mass matrix of ``_skfem_modal.mass_matrix`` (density per material region; thickness for the
plane formulations, 2 pi r axisymmetric). A BC or load value may depend on time (an expression with ``t`` or a
table, ``feacore.case``); everything that does not is assembled once.

Schemes (``transient.scheme``), both in the Hilber-Hughes-Taylor form with alpha in [-1/3, 0]:

    M a_{n+1} + (1 + alpha) (C v_{n+1} + K u_{n+1}) - alpha (C v_n + K u_n) = (1 + alpha) f_{n+1} - alpha f_n
    beta = (1 - alpha)^2 / 4,   gamma = (1 - 2 alpha) / 2

* ``newmark`` (default): alpha = 0, the average-acceleration (trapezoidal) Newmark scheme: unconditionally
  stable, second order, no numerical damping, period elongation T_h / T - 1 = (omega dt)^2 / 12 + O(dt^4)
  (exactly: omega dt = 2 arctan(Omega / 2) with Omega = omega_h dt).
* ``hht``: ``transient.alpha`` (default -0.05): second order, damps the modes the step cannot resolve
  (spectral radius (1 + alpha) / (1 - alpha) for omega dt -> infinity).

The unknown of a step is the displacement u_{n+1}; with u_{n+1} known, a_{n+1} and v_{n+1} follow from the Newmark
relations. The effective matrix c0 M + (1 + alpha) (c1 C + K) (c0 = 1 / (beta dt^2), c1 = gamma / (beta dt)) is
constant for the fixed step: its free-free block is factored ONCE (``_skfem_modal.Factorization``) and reused at
every step. A prescribed displacement (``fixed`` / ``displacement`` / ``symmetry``, a value in time included) is
imposed on u_{n+1}; its velocity and acceleration are those the Newmark relations give for that displacement
history, so the prescribed motion and the scheme are consistent (at t = 0 they come from finite differences of
the prescribed value in time). The initial acceleration is the equilibrium one: M_II a_I = (f - C v - K u)_I -
M_ID a_D at t = 0 (one more factorization, of M_II).

A free body (rigid-body modes) is legal: M makes the effective matrix definite. The count is reported.

Results. ``SkfemDynamicsResult``: ``times``, ``kept``, ``series(qoi)`` (every QoI of the case at every step; the
steady QoI types: point / point_difference / max / min / max_abs of displacements and stresses, reaction (the
dynamic support reaction M a + C v + K u - f on the constrained dofs), integral / mean, strain_energy),
``state(t)`` (a kept step as a steady-style ``SkfemResult``: probe, nodal, evaluate, stress recovery, write_vtu;
with ``velocity`` and ``acceleration``), ``energy`` (kinetic, strain, external work, damping dissipation,
balance per step), ``evaluate`` reducing a series as ``QoI.time`` asks.

Not handled: a temperature load that depends on time (refused), contact, large rotations, nonlinear material.
See docs/solver_transient.md ("Transient dynamics").
"""

from __future__ import annotations

import ast
import time
from dataclasses import asdict, replace

import numpy as np
from skfem import Basis, FacetBasis, Functional

from feacore.case import Case, CaseError, QoI, as_field, table_value, uses_time
from feacore.solver import skfem_backend as skb
from feacore.solver._skfem_locate import Locator
from feacore.solver._skfem_modal import Factorization, free_rigid_modes, mass_matrix
from feacore.solver.base import SolveInfo, TransientResult

_DIRICHLET = ("fixed", "displacement", "symmetry")
_NEUMANN = ("pressure", "traction", "force")
_DISPLACEMENT_FIELDS = ("ux", "uy", "uz", "u_mag")
HHT_ALPHA_DEFAULT = -0.05
SMALL_FOR_SUPERLU = 30000     # unknowns below which the kept factorizations use SuperLU unless a solver is named


# --------------------------------------------------------------------------- values in time


class _SetTime(ast.NodeTransformer):
    def __init__(self, t: float):
        self.t = float(t)

    def visit_Name(self, node):
        return ast.copy_location(ast.Constant(self.t), node) if node.id == "t" else node


def _freeze(value, t: float):
    """A value at time t as the steady assembler takes it (a number or an expression in x, y, z)."""
    if isinstance(value, (list, tuple)):
        return [_freeze(v, t) for v in value]
    if isinstance(value, dict) and "table" in value:
        return table_value(value, t)
    if isinstance(value, str) and uses_time(value):
        tree = _SetTime(t).visit(ast.parse(value, mode="eval"))
        return ast.unparse(ast.fix_missing_locations(tree))
    return value


def _space_free(value) -> bool:
    if isinstance(value, str):
        return not {"x", "y", "z"} & set(compile(ast.parse(value, mode="eval"), "<e>", "eval").co_names)
    return True


def _scalar_at(value, t: float) -> float:
    return float(as_field(value, np.zeros(1), np.zeros(1), np.zeros(1), t)[0])


# --------------------------------------------------------------------------- loads and prescribed motion in time


class _LoadsInTime:
    """f(t) = the loads that do not depend on time (assembled once by the steady assembler) + the loads and
    Neumann BCs that do. A timed item whose value has no x, y, z is a fixed vector times a number of t
    (assembled once); one that varies in space and time is assembled at every step on a kept facet basis."""

    def __init__(self, md: skb._Model, f_static: np.ndarray, items: list):
        self.md, self.f_static = md, f_static
        self.parts = []                      # (kind, data)
        dim = md.dim
        for tag, obj in items:
            if obj.type == "body_force":
                E = md.cells_of(obj.group)
                mask = np.zeros(md.nel, dtype=bool)
                mask[E] = True
                geo = ("cells", md.sb, md.xq, md.wfac(md.xq), mask)
            else:
                F = md.facets_of(obj.group)
                fb = md.facet_basis(F)
                xf = skb._coords(fb)
                cf = md.wfac(xf)
                if obj.type == "pressure" and np.any(md.mesh.f2t[1, F] >= 0):
                    raise CaseError(f"{tag} pressure on group {obj.group!r}: it contains interior facets, so the "
                                    f"outward normal is undefined; use a boundary group or traction")
                meas = None
                if obj.type == "force":
                    meas = float(skb._measure_w.assemble(fb, c=cf))
                    if meas <= 0:
                        raise CaseError(f"{tag} force: group {obj.group!r} has zero measure")
                geo = ("facets", fb, xf, cf, meas)
            values = obj.value if obj.type != "pressure" else [obj.value]
            if all(_space_free(v) for v in values):
                units = [self._assemble(obj.type, geo, k, 1.0) for k in range(len(values))]
                self.parts.append(("separable", (values, units)))
            else:
                self.parts.append(("general", (obj.type, geo, values)))
        self.dim = dim

    def _assemble(self, typ, geo, k, value):
        """The load vector of component k (pressure: the whole pressure) of one item for the value ``value``
        (a number or an expression in x, y, z)."""
        md = self.md
        f = np.zeros(md.N)
        if geo[0] == "cells":
            _, sb, xq, c, mask = geo
            g = np.zeros_like(c)
            g[mask] = skb._eval(value, xq[:, mask])
            f[md.split[k]] += skb._f_scalar.assemble(sb, c=c, g=g)
            return f
        _, fb, xf, cf, meas = geo
        if typ == "pressure":
            p = skb._eval(value, xf)
            for i in range(md.dim):
                f[md.split[i]] += skb._f_scalar.assemble(fb, c=cf, g=-p * fb.normals[i])
        elif typ == "traction":
            f[md.split[k]] += skb._f_scalar.assemble(fb, c=cf, g=skb._eval(value, xf))
        else:
            f[md.split[k]] += skb._f_scalar.assemble(fb, c=cf, g=np.full(cf.shape, float(value) / meas))
        return f

    def __call__(self, t: float) -> np.ndarray:
        f = self.f_static.copy()
        for kind, data in self.parts:
            if kind == "separable":
                values, units = data
                for v, u in zip(values, units):
                    s = _scalar_at(v, t)
                    if s != 0.0:
                        f += s * u
            else:
                typ, geo, values = data
                for k, v in enumerate(values):
                    f += self._assemble(typ, geo, k, _freeze(v, t))
        return f


class _DirichletInTime:
    """The constrained dofs D (fixed in time) and their values u_D(t), in the order and with the rules of the
    steady assembler (the last BC wins; u_r = 0 on the axis)."""

    def __init__(self, md: skb._Model, case: Case):
        self.md = md
        self.entries = []                    # (dof indices, value, coordinates)
        diag = md.md.bbox_diag
        dim = md.dim
        for i_bc, bc in enumerate(case.bcs):
            if bc.type in ("fixed", "displacement"):
                if bc.point is not None:
                    s, _ = md.nearest_sdof(bc.point)
                    S = np.array([s])
                else:
                    S = md.where_filter(md.sdofs_of(bc.group), bc, i_bc)
                X = md.sb.doflocs[:, S]
                if bc.type == "fixed":
                    for k in range(dim):
                        self.entries.append((md.split[k][S], 0.0, X))
                else:
                    vals = bc.value if isinstance(bc.value, list) else [bc.value] * len(bc.components)
                    for k, v in zip(bc.components, vals):
                        self.entries.append((md.split[k][S], v, X))
            elif bc.type == "symmetry":
                ax = skb._symmetry_axis(md, md.facets_of(bc.group), bc.group, i_bc)
                S = md.sdofs_of(bc.group)
                self.entries.append((md.split[ax][S], 0.0, md.sb.doflocs[:, S]))
        if md.axi and md.opts.get("axis_constraint", True):
            on_axis = np.nonzero(np.abs(md.sb.doflocs[0]) <= 1e-9 * diag)[0]
            if len(on_axis):
                self.entries.append((md.split[0][on_axis], 0.0, md.sb.doflocs[:, on_axis]))
        self.timed = any(uses_time(v) for _, v, _ in self.entries)
        self.D, self._u0 = self._at(0.0)

    def _at(self, t: float):
        cons = skb._Constraints()
        for idx, v, X in self.entries:
            cons.add(idx, skb._eval(_freeze(v, t), X), None)
        return cons.finalize([])

    def __call__(self, t: float) -> np.ndarray:
        if not self.timed:
            return self._u0
        D, u = self._at(t)
        return u

    def rates0(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Velocity and acceleration of the prescribed values at t = 0, by one-sided differences (a table is
        constant before its first point, so no value before t = 0 is used)."""
        if not self.timed:
            z = np.zeros(len(self.D))
            return z, z
        h = 1e-4 * dt
        u = [self(k * h) for k in range(4)]
        v0 = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
        a0 = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / h**2
        return v0, a0


def _initial_field(md: skb._Model, value, what: str) -> np.ndarray:
    """``transient.initial`` "u" or "v": a list of dim components (numbers or expressions in x, y, z)."""
    out = np.zeros(md.N)
    if value is None:
        return out
    if not isinstance(value, (list, tuple)) or len(value) != md.dim:
        raise CaseError(f"analysis.options.transient.initial.{what} must be a list of {md.dim} components "
                        f"(numbers or expressions in x, y, z)")
    X = md.sb.doflocs
    for k, v in enumerate(value):
        out[md.split[k]] = skb._eval(v, X)
    return out


# --------------------------------------------------------------------------- states and QoIs


class _LocatorMemo:
    """A point QoI is located once per run, not at every step."""

    def __init__(self, locator):
        self.locator, self.memo = locator, {}

    def locate(self, pts, *args, **kwargs):
        key = (np.asarray(pts).tobytes(), args, tuple(sorted(kwargs.items())))
        if key not in self.memo:
            self.memo[key] = self.locator.locate(pts, *args, **kwargs)
        return self.memo[key]


class _Recovery:
    """Continuous L2 stress recovery as the steady solver does it, with the scalar mass matrix factored once."""

    def __init__(self, md: skb._Model, method):
        self.md = md
        Ms = skb._mass.assemble(md.sb) if md.opts.get("assembly", "fast") == "skfem" else skb.fast.mass(md.sb)
        self.fac = Factorization(Ms.tocsr(), method, definite=True)

    def __call__(self, u: np.ndarray) -> dict:
        S, _ = skb.qp_stress_elasticity(self.md, u)
        names = list(S)
        B = np.column_stack([skb._rhs_proj.assemble(self.md.sb, s=S[n]) for n in names])
        X = self.fac.solve(B).reshape(self.md.sb.N, len(names))
        return {n: X[:, i].copy() for i, n in enumerate(names)}


class DynamicState(skb.SkfemResult):
    """One time of a dynamic run as a steady-style result (``SkfemResult``); ``velocity``, ``acceleration``
    (N,) and ``time`` are added. The states of a run share the point locator and the integral bases."""

    def __init__(self, shared: dict, *args, velocity=None, acceleration=None, t: float = 0.0):
        super().__init__(*args)
        self._shared = shared
        self._locator = shared["locator"]
        self.velocity = velocity
        self.acceleration = acceleration
        self.time = float(t)

    def _integral(self, field, group):
        md, memo = self.model, self._shared["integrals"]
        if group not in memo:
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

    def write_vtu(self, path):
        import meshio

        p = super().write_vtu(path)
        if self.velocity is not None:
            m = meshio.read(str(p))
            md = self.model

            def vec(x):
                v = np.zeros((md.sb.N, 3))
                for k in range(md.dim):
                    v[:, k] = x[md.split[k]]
                return v

            m.point_data["velocity"] = vec(self.velocity)
            m.point_data["acceleration"] = vec(self.acceleration)
            m.field_data["time"] = np.array([self.time])
            m.write(str(p))
        return p


def _qoi_key(q: QoI) -> tuple:
    return (q.type, q.field, None if q.point is None else tuple(float(v) for v in q.point), q.group, q.component,
            None if q.point2 is None else tuple(float(v) for v in q.point2))


def _needs_stress(q: QoI) -> bool:
    return q.type not in ("reaction", "strain_energy") and q.field is not None and q.field not in _DISPLACEMENT_FIELDS


def _kept_steps(tr: dict, times: np.ndarray) -> np.ndarray:
    n = len(times) - 1
    if tr.get("output_times") is not None:
        idx = [int(np.argmin(np.abs(times - float(t)))) for t in tr["output_times"]]
    elif tr.get("output_every") is not None:
        idx = list(range(0, n + 1, int(tr["output_every"])))
    else:                                  # 10 states evenly spaced in time after the initial one
        idx = np.rint(np.linspace(0, n, 11)).astype(int).tolist()
    return np.unique(np.array(list(idx) + [n], dtype=np.int64))


# --------------------------------------------------------------------------- the solve


def solve(case: Case, **options) -> "SkfemDynamicsResult":
    t_start = time.perf_counter()
    case.check()
    a = case.analysis
    tr = a.transient
    if tr is None or a.physics != "elasticity":
        raise CaseError("not a transient elasticity (dynamic) analysis")
    opts = {**(a.options or {}), **options}
    md = skb._Model(case, opts)
    method = opts.get("linear_solver")
    step_method = method
    if method is None and md.N < SMALL_FOR_SUPERLU:
        step_method = "scipy"        # PARDISO's solve phase costs milliseconds even for a small system
    scheme = tr.get("scheme") or "newmark"
    alpha = float(tr["alpha"]) if tr.get("alpha") is not None else (HHT_ALPHA_DEFAULT if scheme == "hht" else 0.0)
    damp = (opts.get("damping") or {}).get("rayleigh", [0.0, 0.0])
    ra, rb = float(damp[0]), float(damp[1])
    times = a.time_steps()
    nst = len(times) - 1
    dt = float(times[1] - times[0])

    # split the BCs and loads into the parts that are fixed in time and those that are not
    timed_items, bcs0, loads0 = [], [], []
    for i, bc in enumerate(case.bcs):
        if bc.type in _NEUMANN and uses_time(bc.value):
            timed_items.append((f"bcs[{i}]", bc))
        elif bc.type in _DIRICHLET:
            bcs0.append(replace(bc, value=_freeze(bc.value, 0.0)) if uses_time(bc.value) else bc)
        else:
            bcs0.append(bc)
    for i, ld in enumerate(case.loads):
        if uses_time(ld.value):
            if ld.type != "body_force":
                raise CaseError(f"loads[{i}] ({ld.type}): a {ld.type} load that changes in time is not supported by "
                                f"the dynamic analysis (only body forces, pressures, tractions and forces may "
                                f"depend on t)")
            timed_items.append((f"loads[{i}]", ld))
        else:
            loads0.append(ld)
    md.case = replace(case, bcs=bcs0, loads=loads0)
    try:
        K, f_static, cons = skb._assemble_elasticity(md)
    finally:
        md.case = case
    D0, _ = cons.finalize(md.warnings)
    dirichlet = _DirichletInTime(md, case)
    D = dirichlet.D
    if not np.array_equal(D0, D):            # both follow the same rules; a difference is a bug here
        raise skb.SolverError("dynamics: the constrained dofs differ from those of the steady assembler")
    loads = _LoadsInTime(md, f_static, timed_items)
    M = mass_matrix(md)
    C = (ra * M + rb * K).tocsr() if (ra or rb) else None

    N = md.N
    free = np.ones(N, dtype=bool)
    free[D] = False
    I = np.nonzero(free)[0]
    if len(I) == 0:
        raise CaseError("dynamics: every degree of freedom is constrained")
    Z, znames = free_rigid_modes(md, D)
    if Z.shape[1]:
        md.warnings.append(f"dynamics: the model has {Z.shape[1]} rigid-body modes ({', '.join(znames)}); "
                           f"a net load accelerates it as a whole")
    for i, (tag, obj) in enumerate(timed_items):
        for v in (obj.value if isinstance(obj.value, list) else [obj.value]):
            if isinstance(v, dict) and "table" in v:
                tt = np.asarray(v["table"], dtype=float)[:, 0]
                if len(tt) > 1 and np.diff(tt).min() < dt * (1 - 1e-9):
                    md.warnings.append(f"dynamics: {tag} is a table with points {np.diff(tt).min():.3g} s apart, "
                                       f"closer than the step {dt:.3g} s: the step cannot follow it")

    # initial state
    u = _initial_field(md, (tr.get("initial") or {}).get("u"), "u")
    v = _initial_field(md, (tr.get("initial") or {}).get("v"), "v")
    uD0 = dirichlet(0.0)
    vD0, aD0 = dirichlet.rates0(dt)
    if (tr.get("initial") or {}) and (np.any(u[D] != uD0) or np.any(v[D] != vD0)):
        md.warnings.append("dynamics: the initial displacement / velocity differs from the prescribed motion on "
                           "constrained dofs; the prescribed motion wins there")
    u[D], v[D] = uD0, vD0
    f = loads(0.0)
    acc = np.zeros(N)
    acc[D] = aD0
    MI_I = M[I][:, I].tocsr()
    r0 = f - K @ u - (C @ v if C is not None else 0.0) - M[:, D] @ aD0
    if np.any(r0[I] != 0.0):
        mfac = Factorization(MI_I, step_method, definite=True)
        acc[I] = mfac.solve(r0[I])
        mfac.close()

    # the effective matrix, factored once
    beta, gamma = (1.0 - alpha) ** 2 / 4.0, (1.0 - 2.0 * alpha) / 2.0
    c0, c1 = 1.0 / (beta * dt * dt), gamma / (beta * dt)
    c2, c3 = 1.0 / (beta * dt), 1.0 / (2.0 * beta) - 1.0
    Keff = c0 * M + (1.0 + alpha) * (K + (c1 * C if C is not None else 0.0))
    Keff = Keff.tocsr()
    KII = Keff[I][:, I].tocsr()
    KID = Keff[I][:, D].tocsr()
    fac = Factorization(KII, step_method, definite=True)
    notes = [f"linear solver: {s}" for s in fac.notes]

    # QoIs evaluated at every step
    keys, qmap, need_stress = [], {}, False
    for q in case.qois:
        k = _qoi_key(q)
        if k not in qmap:
            qmap[k] = q
            keys.append(k)
        need_stress |= _needs_stress(q)
    recover = _Recovery(md, step_method)
    shared = {"locator": _LocatorMemo(Locator(md.mesh, md.md.bbox_diag)), "integrals": {}}
    kept = _kept_steps(tr, times)
    kept_set = set(int(k) for k in kept)
    series = {k: np.full(nst + 1, np.nan) for k in keys}
    qerr: dict[tuple, str] = {}
    states: dict[int, tuple] = {}
    en = {n: np.zeros(nst + 1) for n in ("kinetic", "strain", "external_work", "damping", "balance")}
    info0 = SolveInfo(backend="skfem", physics="elasticity", formulation=md.form, order=md.order, dim=md.dim,
                      cell_type=md.md.cell_type, n_nodes=int(md.mesh.doflocs.shape[1]), n_cells=int(md.nel),
                      n_dofs=int(N), measure=0.0, h_rep=0.0, wall_time=0.0, warnings=[])

    def reaction(u_, v_, a_, f_):
        r = K @ u_ + M @ a_ - f_
        if C is not None:
            r = r + C @ v_
        return r

    def record(n, u_, v_, a_, f_):
        R = reaction(u_, v_, a_, f_)
        en["kinetic"][n] = 0.5 * float(v_ @ (M @ v_))
        en["strain"][n] = 0.5 * float(u_ @ (K @ u_))
        if keys:
            rec = recover(u_) if need_stress else {}
            st = DynamicState(shared, case, info0, md, u_, rec, K, f_, R, D)
            for k in keys:
                if k in qerr:
                    continue
                try:
                    series[k][n] = st.evaluate_qoi(qmap[k])
                except (CaseError, ValueError) as e:
                    qerr[k] = str(e)
        if n in kept_set:
            states[n] = (u_.copy(), v_.copy(), a_.copy(), f_.copy())
        return R

    R = record(0, u, v, acc, f)
    fext = f.copy()
    fext[D] += R[D]
    t_loop = time.perf_counter()
    for n in range(nst):
        t1 = float(times[n + 1])
        f1 = loads(t1)
        u1 = np.zeros(N)
        u1[D] = dirichlet(t1)
        A = c0 * u + c2 * v + c3 * acc
        rhs = (1.0 + alpha) * f1 - alpha * f + M @ A + alpha * (K @ u)
        if C is not None:
            B = c1 * u - (1.0 - gamma / beta) * v - dt * (1.0 - gamma / (2.0 * beta)) * acc
            rhs += (1.0 + alpha) * (C @ B) + alpha * (C @ v)
        u1[I] = fac.solve(rhs[I] - KID @ u1[D])
        a1 = c0 * (u1 - u) - c2 * v - c3 * acc
        v1 = v + dt * ((1.0 - gamma) * acc + gamma * a1)
        R1 = record(n + 1, u1, v1, a1, f1)
        fext1 = f1.copy()
        fext1[D] += R1[D]
        du = u1 - u
        en["external_work"][n + 1] = en["external_work"][n] + 0.5 * float((fext + fext1) @ du)
        if C is not None:
            en["damping"][n + 1] = en["damping"][n] + 0.5 * float(du @ (C @ (v + v1)))
        u, v, acc, f, fext = u1, v1, a1, f1, fext1
    loop_time = time.perf_counter() - t_loop
    fac_desc = fac.describe()
    fac.close()
    recover.fac.close()
    e0 = en["kinetic"][0] + en["strain"][0]
    en["balance"] = en["kinetic"] + en["strain"] + en["damping"] - en["external_work"] - e0

    for k, msg in qerr.items():
        md.warnings.append(f"qoi {qmap[k].name!r} could not be evaluated: {msg}")
    md.warnings += notes
    md.warnings.append(f"info: dynamics: scheme {scheme} (alpha = {alpha:g}, beta = {beta:.6g}, gamma = {gamma:.6g}), "
                       f"{nst} steps of dt = {dt:.6g} s to t_end = {float(times[-1]):.6g} s, Rayleigh damping "
                       f"[{ra:g}, {rb:g}]; effective matrix factored once: {fac_desc}; stepping {loop_time:.3g} s")
    measure = float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(md.sb))
    info = SolveInfo(
        backend="skfem", physics="elasticity", formulation=md.form, order=md.order, dim=md.dim,
        cell_type=md.md.cell_type, n_nodes=int(md.mesh.doflocs.shape[1]), n_cells=int(md.nel),
        n_dofs=int(N), measure=measure, h_rep=float((measure / md.nel) ** (1.0 / md.dim)),
        wall_time=float(time.perf_counter() - t_start), warnings=list(md.warnings),
    )
    return SkfemDynamicsResult(case, info, md, times, kept, series, qmap, states, en, K, M, C, D, shared,
                               recover, scheme, alpha, (ra, rb), Z.shape[1])


# --------------------------------------------------------------------------- result


class SkfemDynamicsResult(TransientResult):
    """A transient dynamic solution (module docstring, "Results"). ``energy``: dict of per-step arrays
    "kinetic" (1/2 v.Mv), "strain" (1/2 u.Ku), "external_work" (loads and support reactions, trapezoidal in
    time), "damping" (dissipated by C), "balance" (kinetic + strain + damping - work - initial energy: zero to
    round-off for the average-acceleration scheme)."""

    def __init__(self, case, info, model, times, kept, series, qmap, states, energy, K, M, C, D, shared, recover,
                 scheme, alpha, rayleigh, rigid_modes):
        self.case, self.info, self.model = case, info, model
        self.times = np.asarray(times, dtype=float)
        self.kept = np.asarray(kept, dtype=np.int64)
        self.dt = float(self.times[1] - self.times[0])
        self._series, self._qmap, self._states = series, qmap, states
        self.energy = energy
        self.K, self.M, self.C, self.dirichlet_dofs = K, M, C, D
        self._shared, self._recover = shared, recover
        self.scheme, self.alpha, self.rayleigh, self.rigid_modes = scheme, alpha, rayleigh, rigid_modes
        self._cache: dict[int, DynamicState] = {}

    def series(self, qoi: QoI) -> np.ndarray:
        key = _qoi_key(qoi)
        if key in self._series:
            return self._series[key].copy()
        if len(self.kept) == len(self.times):
            return np.array([self.state(t).evaluate_qoi(qoi) for t in self.times])
        raise CaseError(f"QoI {qoi.name!r} was not evaluated during the run: a QoI is evaluated at every step "
                        f"while the solver runs, so it must be in case.qois (or keep every step: "
                        f"transient.output_every = 1). This run kept {len(self.kept)} of {len(self.times)} steps.")

    def state(self, t: float | None = None) -> DynamicState:
        k = int(self.kept[-1]) if t is None else int(self.kept[np.argmin(np.abs(self.times[self.kept] - float(t)))])
        if k not in self._cache:
            u, v, a, f = self._states[k]
            md = self.model
            R = self.K @ u + self.M @ a - f
            if self.C is not None:
                R = R + self.C @ v
            if self._recover.fac._lu is None and self._recover.fac._pardiso is None:
                self._recover = _Recovery(md, md.opts.get("linear_solver"))
            st = DynamicState(self._shared, self.case, self.info, md, u, self._recover(u), self.K, f, R,
                              self.dirichlet_dofs, velocity=v, acceleration=a, t=float(self.times[k]))
            self._cache[k] = st
        return self._cache[k]

    def displacement(self, t: float | None = None) -> np.ndarray:
        """The displacement vector (N,) of the kept state nearest to t."""
        return self.state(t).x

    def evaluate_all(self) -> dict[str, float]:
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
        qois = self.evaluate_all()
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta,
                "dynamics": {"scheme": self.scheme, "alpha": self.alpha, "dt": self.dt, "n_steps": len(self.times) - 1,
                             "rayleigh": list(self.rayleigh), "rigid_modes": self.rigid_modes,
                             "energy_balance_max": float(np.abs(self.energy["balance"]).max())}}

    def write_vtu(self, path, t: float | None = None):
        """The fields of the kept state nearest to t (default: the last step)."""
        return self.state(t).write_vtu(path)
