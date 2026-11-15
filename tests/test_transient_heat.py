"""Transient heat conduction in the fixed solver (``feacore/solver/_skfem_transient.py``,
docs/solver_transient.md, "Transient heat conduction").

What is checked, and the tolerance of each check (why it is what it is):

1. Closed forms (``feacore/problems/transient_heat/_series.py``):
   (a) a slab whose faces are suddenly held at a temperature (Fourier series), in the ``2d``, ``axisymmetric``
       (heat along the axis) and ``3d`` formulations;
   (b) a slab cooled by convection after a step of the fluid temperature at Bi = 0.1, 1 and 10 (roots of
       z tan z = Bi computed numerically), in 2d and once in 3d;
   (c) a lumped body (Bi = 1e-3) against the exponential;
   (d) a long solid cylinder cooled by convection (Bessel series, axisymmetric).
2. Orders of accuracy with a manufactured solution (sympy, as ``feacore/verification/mms.py``): in time on a
   fixed P2 mesh (Crank-Nicolson and BDF2 second order, backward Euler first order), in space at a small step
   (P1 second, P2 third order in the maximum error at probe points).
3. NAFEMS T3 (registered problem ``nafems_t3``): 36.60 degC at x = 0.08 m, t = 32 s.
4. Consistency: the long-time limit is the steady solution; nothing changing stays put; energy balance.
5. Features: h(t), tables, an initial field from a file, a Dirichlet value in time, the Crank-Nicolson
   start-up, the reductions of QoI.time, options and errors.
6. The registered benchmark problems (``nafems_t3``, ``convective_slab_chilldown``).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("skfem")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from feacore.case import BC, Analysis, Case, CaseError, Load, Material, QoI  # noqa: E402
from feacore.geometry.gmsh_tools import gmsh_session  # noqa: E402
from feacore.problems.transient_heat import _series as ser  # noqa: E402
from feacore.solver import solve  # noqa: E402

K, RHO, CP = 20.0, 8000.0, 500.0
ALPHA = K / (RHO * CP)


# --------------------------------------------------------------------------- meshes


def strip(path, L, W, nx, ny=1, order=1, x0=0.0):
    """Structured triangles on [x0, x0 + L] x [0, W]; groups left, right, bottom, top, domain."""
    with gmsh_session("strip"):
        geo = gmsh.model.geo
        p = [geo.addPoint(x0, 0, 0), geo.addPoint(x0 + L, 0, 0), geo.addPoint(x0 + L, W, 0), geo.addPoint(x0, W, 0)]
        ln = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        s = geo.addPlaneSurface([geo.addCurveLoop(ln)])
        for c, n in zip(ln, (nx, ny, nx, ny)):
            geo.mesh.setTransfiniteCurve(c, n + 1)
        geo.mesh.setTransfiniteSurface(s)
        geo.synchronize()
        for name, tag in (("bottom", ln[0]), ("right", ln[1]), ("top", ln[2]), ("left", ln[3])):
            gmsh.model.addPhysicalGroup(1, [tag], name=name)
        gmsh.model.addPhysicalGroup(2, [s], name="domain")
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def bar3d(path, L, W, nx, order=1):
    """Tetrahedra on the bar [0, L] x [0, W] x [0, W] (one layer of cells across); groups left (x = 0),
    right (x = L), domain."""
    with gmsh_session("bar"):
        geo = gmsh.model.geo
        p = [geo.addPoint(0, 0, 0), geo.addPoint(L, 0, 0), geo.addPoint(L, W, 0), geo.addPoint(0, W, 0)]
        ln = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        s = geo.addPlaneSurface([geo.addCurveLoop(ln)])
        for c, n in zip(ln, (nx, 1, nx, 1)):
            geo.mesh.setTransfiniteCurve(c, n + 1)
        geo.mesh.setTransfiniteSurface(s)
        out = geo.extrude([(2, s)], 0, 0, W, numElements=[1])
        geo.synchronize()
        vol = [t for d, t in out if d == 3]
        # extrude output: [top, volume, side of ln[0], ln[1], ln[2], ln[3]]
        sides = [t for d, t in out if d == 2][1:]
        gmsh.model.addPhysicalGroup(2, [sides[1]], name="right")
        gmsh.model.addPhysicalGroup(2, [sides[3]], name="left")
        gmsh.model.addPhysicalGroup(3, vol, name="domain")
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def heat(mesh, tr, bcs, qois=(), loads=(), form="2d", order=2, k=K, rho=RHO, cp=CP, **opts) -> Case:
    return Case(mesh=str(mesh), analysis=Analysis(physics="heat", formulation=form, order=order,
                                                  options={"transient": tr, **opts}),
                materials={"m": Material(k=k, rho=rho, cp=cp)}, bcs=list(bcs), loads=list(loads), qois=list(qois))


def P(name, x, y=0.0, z=None, **kw):
    pt = [x, y] if z is None else [x, y, z]
    return QoI(name=name, type="point", field="T", point=pt, **kw)


def info_line(result, start):
    return next(w for w in result.info.warnings if w.startswith(start))


# --------------------------------------------------------------------------- 1. closed forms


L_SLAB, TI, TS = 0.1, 300.0, 100.0
T_QUENCH = 0.2 * L_SLAB ** 2 / ALPHA           # Fo = 0.2: the centre has cooled by ~70 %


@pytest.mark.parametrize("form", ["2d", "axisymmetric", "3d"])
def test_quench_slab_fourier(form, tmp_path):
    """A slab at 300 K whose faces are held at 100 K from t = 0 (Fourier series). Crank-Nicolson with the
    automatic Rannacher start-up, 80 steps. Tolerance: 0.5 % of the 200 K jump at the centre and at x = L/8
    for t >= T_QUENCH / 4 (P2, 40 cells; P1 in 3d: 0.8 %). Measured: 2d 0.04 %, axisymmetric 0.04 %, 3d
    0.4 %. Early times near the face are excluded: the jump is a singularity in time there."""
    n = 40
    if form == "3d":
        mesh, order = bar3d(tmp_path / "bar.msh", L_SLAB, L_SLAB / n, n, order=1), 1
        lo, hi, pts = "left", "right", lambda x: (x, 0.5 * L_SLAB / n, 0.5 * L_SLAB / n)
    elif form == "axisymmetric":      # r in [0, W], z = the slab's thickness direction
        mesh, order = strip(tmp_path / "ax.msh", L_SLAB / n, L_SLAB, 1, n, order=2), 2
        lo, hi, pts = "bottom", "top", lambda x: (0.5 * L_SLAB / n, x)
    else:
        mesh, order = strip(tmp_path / "s.msh", L_SLAB, L_SLAB / n, n, order=2), 2
        lo, hi, pts = "left", "right", lambda x: (x, 0.5 * L_SLAB / n)
    xs = (L_SLAB / 2, L_SLAB / 8)
    q = [QoI(name=f"T{i}", type="point", field="T", point=list(pts(x))) for i, x in enumerate(xs)]
    c = heat(mesh, {"t_end": T_QUENCH, "n_steps": 80, "initial": {"T": TI}},
             [BC(type="temperature", group=lo, value=TS), BC(type="temperature", group=hi, value=TS)], q,
             form=form, order=order)
    r = solve(c)
    late = r.times >= T_QUENCH / 4
    errs = [np.abs(r.series(qq) - ser.slab_quench(x, r.times, L_SLAB, ALPHA, TI, TS))[late].max() / (TI - TS)
            for qq, x in zip(q, xs)]
    print(form, "max rel err", errs)
    assert max(errs) < (0.008 if form == "3d" else 0.005)
    assert any("Rannacher" in w for w in r.info.warnings)


@pytest.mark.parametrize("bi", [0.1, 1.0, 10.0])
def test_convective_slab_biot(bi, tmp_path):
    """Half a slab (x = 0 insulated mid-plane, x = L cooled) after a step of the fluid temperature, against the
    series in the roots of z tan z = Bi. P2, 20 cells, Crank-Nicolson, 100 steps to Fo = 1. Tolerance 0.1 %
    of the step at the surface and the centre at every step (measured 1e-5 .. 3e-4: the error is largest at
    Bi = 10, where the surface temperature drops fastest at the start). No start-up is used: the temperature
    stays continuous when only the fluid temperature jumps."""
    Lh, n = 0.05, 20
    mesh = strip(tmp_path / "c.msh", Lh, Lh / n, n, order=2)
    q = [P("Tc", 0.0), P("Ts", Lh)]
    c = heat(mesh, {"t_end": Lh ** 2 / ALPHA, "n_steps": 100, "initial": {"T": TI}},
             [BC(type="convection", group="right", h=bi * K / Lh, T_inf=TS)], q)
    r = solve(c)
    assert not any("Rannacher" in w for w in r.info.warnings)
    for qq, x in zip(q, (0.0, Lh)):
        err = np.abs(r.series(qq) - ser.slab_convection(x, r.times, Lh, ALPHA, bi, TI, TS)).max() / (TI - TS)
        print(bi, qq.name, err)
        assert err < 1e-3


def test_convective_slab_3d(tmp_path):
    """The convective slab at Bi = 1 on a 3D tet bar (P2, 20 cells along, 60 steps). Tolerance 0.2 % of the
    step (measured ~1e-4). Also prints the size and run time of a typical small 3D transient."""
    Lh, n = 0.05, 20
    mesh = bar3d(tmp_path / "b.msh", Lh, Lh / n, n, order=2)
    q = [P("Tc", 0.0, 0.5 * Lh / n, 0.5 * Lh / n), P("Ts", Lh, 0.5 * Lh / n, 0.5 * Lh / n)]
    c = heat(mesh, {"t_end": Lh ** 2 / ALPHA, "n_steps": 60, "initial": {"T": TI}},
             [BC(type="convection", group="right", h=K / Lh, T_inf=TS)], q, form="3d")
    r = solve(c)
    print("3d:", r.info.n_dofs, "dofs,", len(r.times) - 1, "steps,", round(r.info.wall_time, 2), "s")
    for qq, x in zip(q, (0.0, Lh)):
        err = np.abs(r.series(qq) - ser.slab_convection(x, r.times, Lh, ALPHA, 1.0, TI, TS)).max() / (TI - TS)
        assert err < 2e-3


def test_lumped_body(tmp_path):
    """Bi = h a / k = 1e-3 on a square cooled on all faces: T = Tf + (Ti - Tf) exp(-h P t / (rho cp A)).
    Tolerance 0.1 % of the step: the lumped model itself is off by O(Bi) (measured ~2e-4)."""
    a = 0.01
    mesh = strip(tmp_path / "sq.msh", a, a, 4, 4, order=1)
    h = 1e-3 * K / a
    tau = RHO * CP * a * a / (h * 4 * a)
    q = [P("Tc", a / 2, a / 2)]
    c = heat(mesh, {"t_end": 2 * tau, "n_steps": 40, "initial": {"T": TI}},
             [BC(type="convection", group=g, h=h, T_inf=TS) for g in ("left", "right", "bottom", "top")], q, order=1)
    r = solve(c)
    err = np.abs(r.series(q[0]) - (TS + (TI - TS) * np.exp(-r.times / tau))).max() / (TI - TS)
    print("lumped", err)
    assert err < 1e-3


def test_cylinder_bessel(tmp_path):
    """A long solid cylinder (axisymmetric strip 0 <= r <= R, insulated ends) cooled by convection at
    Bi = h R / k = 1, against the Bessel series. P2, 20 cells, 100 steps to Fo = 0.5. Tolerance 0.1 % of the
    step at the axis and the surface (measured ~1e-4)."""
    R, n = 0.02, 20
    mesh = strip(tmp_path / "cyl.msh", R, R / n, n, order=2)
    q = [P("T0", 0.0, 0.5 * R / n), P("TR", R, 0.5 * R / n)]
    c = heat(mesh, {"t_end": 0.5 * R ** 2 / ALPHA, "n_steps": 100, "initial": {"T": TI}},
             [BC(type="convection", group="right", h=K / R, T_inf=TS)], q, form="axisymmetric")
    r = solve(c)
    for qq, x in zip(q, (0.0, R)):
        err = np.abs(r.series(qq) - ser.cylinder_convection(x, r.times, R, ALPHA, 1.0, TI, TS)).max() / (TI - TS)
        print("cyl", qq.name, err)
        assert err < 1e-3


# --------------------------------------------------------------------------- 2. orders of accuracy


def _mms(T_expr: str):
    """Source and boundary data of a manufactured T(x, y, t) for k = rho = cp = 1."""
    sp = pytest.importorskip("sympy")
    from feacore.verification.mms import to_expr

    x, y, t = sp.symbols("x y t")
    T = sp.sympify(T_expr, locals={"x": x, "y": y, "t": t})
    q = sp.diff(T, t) - sp.diff(T, x, 2) - sp.diff(T, y, 2)
    exact = sp.lambdify((x, y, t), T, "numpy")
    return to_expr(T), to_expr(sp.simplify(q)), to_expr(T.subs(t, 0)), exact


MMS_T = "1 + exp(-t)*sin(pi*x)*sin(pi*y) + x*y**2*cos(3*t)"


def _mms_case(mesh, scheme, n, t_end, order, T_expr=MMS_T, qpts=((0.3, 0.6),)):
    Tx, q, T0, exact = _mms(T_expr)
    qois = [P(f"T{i}", *p) for i, p in enumerate(qpts)]
    c = heat(mesh, {"t_end": t_end, "n_steps": n, "scheme": scheme, "initial": {"T": T0}},
             [BC(type="temperature", group=g, value=Tx) for g in ("left", "right", "bottom", "top")], qois,
             loads=[Load(type="source", value=q, group="*")], order=order, k=1.0, rho=1.0, cp=1.0)
    return c, exact


@pytest.mark.parametrize("scheme,p", [("crank_nicolson", 2.0), ("bdf2", 2.0), ("backward_euler", 1.0)])
def test_time_order(scheme, p, tmp_path):
    """Halve the step three times (8 .. 64 steps to t = 1) on one P2 mesh (12 x 12). The observed order from
    three successive solutions, p = log2((f1 - f2) / (f2 - f3)), cancels the spatial error. Tolerance: within
    0.15 of the theory on the finest triple (measured: CN 2.00, BDF2 2.0, BE 1.0). The error against the
    exact solution at 64 steps is also bounded (spatial + temporal, 1e-3)."""
    mesh = strip(tmp_path / "u.msh", 1.0, 1.0, 12, 12, order=2)
    vals = []
    for n in (8, 16, 32, 64):
        c, exact = _mms_case(mesh, scheme, n, 1.0, 2)
        vals.append(solve(c).evaluate(c.qois[0]))
    d = np.diff(vals)
    orders = np.log2(np.abs(d[:-1] / d[1:]))
    print(scheme, vals, "orders", orders, "err", abs(vals[-1] - exact(0.3, 0.6, 1.0)))
    assert abs(orders[-1] - p) < 0.15
    assert abs(vals[-1] - exact(0.3, 0.6, 1.0)) < (3e-3 if scheme == "backward_euler" else 1e-4)


@pytest.mark.parametrize("order,p", [(1, 2.0), (2, 3.0)])
def test_space_order(order, p, tmp_path):
    """Refine the mesh (P1: 8, 16, 32 cells; P2: 4, 8, 16) at a small fixed step (Crank-Nicolson, 40 steps
    to t = 0.1) with T = (1 + t) (sin(pi x) cos(pi y / 2) + x^2 y): the time error of a solution linear in t
    is only that of the decaying projection mismatch. Max error over 30 interior probe points at t_end;
    observed order within 0.3 of the theory (measured P1 2.0, P2 3.0)."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.05, 0.95, size=(30, 2))
    T = "(1 + t)*(sin(pi*x)*cos(pi*y/2) + x**2*y)"
    errs = []
    for n in ((8, 16, 32) if order == 1 else (4, 8, 16)):
        mesh = strip(tmp_path / f"s{n}.msh", 1.0, 1.0, n, n, order=order)
        c, exact = _mms_case(mesh, "crank_nicolson", 40, 0.1, order, T_expr=T, qpts=())
        r = solve(c)
        errs.append(np.abs(r.probe("T", pts) - exact(pts[:, 0], pts[:, 1], 0.1)).max())
    obs = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(order, errs, obs)
    assert abs(obs[-1] - p) < 0.3


# --------------------------------------------------------------------------- 3. NAFEMS T3


T3_EXACT = 36.603


def test_t3_exact_series():
    """The Duhamel series reproduces the NAFEMS target 36.60 degC (to 0.01 %)."""
    from feacore.problems.transient_heat import nafems_t3 as t3

    g = lambda t: 100 * math.sin(math.pi * t / 40)  # noqa: E731
    dg = lambda t: 100 * math.pi / 40 * math.cos(math.pi * t / 40)  # noqa: E731
    v = ser.slab_face_history(0.08, 32.0, t3.L, t3.K / (t3.RHO * t3.CP), g, dg, n=200)
    assert abs(v - 36.60) < 0.01 and abs(v - T3_EXACT) < 1e-3


@pytest.mark.parametrize("order,tol", [(2, 0.002), (1, 0.01)])
def test_nafems_t3(order, tol, tmp_path):
    """The registered problem on its h0 mesh (P2: within 0.2 % of the target, measured 36.60; P1: within the
    NAFEMS 1 %). Also checks the step convergence of the T3 value: halving the step changes it by < 0.05 %."""
    from feacore.problems.registry import get

    pr = get("nafems_t3")
    mesh = pr.mesh(tmp_path / "t3.msh", pr.params["h0"], geom_order=order)
    c = pr.case(mesh, order=order)
    v = solve(c).evaluate_all()["T_x08"]
    print("T3", order, v)
    assert abs(v - 36.60) < tol * 36.60
    if order == 2:
        c2 = pr.case(mesh, order=2)
        c2.analysis.options["transient"]["n_steps"] *= 2
        v2 = solve(c2).evaluate_all()["T_x08"]
        assert abs(v2 - v) < 5e-4 * 36.6 and abs(v2 - T3_EXACT) < 0.02


# --------------------------------------------------------------------------- 4. consistency


def test_long_time_reaches_steady(tmp_path):
    """Faces at 100 and 400 K, a source, convection on top: after Fo = 3 the transient (backward Euler, which
    damps everything) equals the steady solve of the same case to 1e-6 of the range."""
    mesh = strip(tmp_path / "s.msh", 0.1, 0.02, 20, 4, order=2)
    bcs = [BC(type="temperature", group="left", value=100.0), BC(type="temperature", group="right", value=400.0),
           BC(type="convection", group="top", h=50.0, T_inf=300.0)]
    loads = [Load(type="source", value=1e5, group="*")]
    c = heat(mesh, {"t_end": 3 * 0.1 ** 2 / ALPHA * 20, "n_steps": 100, "scheme": "backward_euler",
                    "initial": {"T": 250.0}}, bcs, [P("T", 0.03, 0.01)], loads=loads)
    r = solve(c)
    steady = solve(Case(mesh=str(mesh), analysis=Analysis(physics="heat", formulation="2d", order=2),
                        materials={"m": Material(k=K)}, bcs=bcs, loads=loads))
    Tt, Ts_ = r.state().fields["T"], steady.fields["T"]
    assert np.abs(Tt - Ts_).max() < 1e-6 * 300


@pytest.mark.parametrize("scheme", ["crank_nicolson", "bdf2", "backward_euler"])
def test_nothing_changes(scheme, tmp_path):
    """Initial state = the BC values and no source: the state stays put to round-off (1e-12 relative)."""
    mesh = strip(tmp_path / "s.msh", 0.1, 0.02, 10, 2, order=2)
    c = heat(mesh, {"t_end": 100.0, "n_steps": 10, "scheme": scheme, "initial": {"T": 300.0}},
             [BC(type="temperature", group="left", value=300.0),
              BC(type="convection", group="right", h=100.0, T_inf=300.0)],
             [QoI(name="mx", type="max", field="T"), QoI(name="mn", type="min", field="T")])
    r = solve(c)
    for q in c.qois:
        assert np.abs(r.series(q) - 300.0).max() < 1e-12 * 300


@pytest.mark.parametrize("scheme,tol", [("backward_euler", 1e-9), ("crank_nicolson", 2e-3), ("bdf2", 2e-3)])
def test_energy_balance(scheme, tol, tmp_path):
    """The stored heat rho cp int T changes by the time integral of the heat flowing in: the reaction at a
    temperature BC and h (T_inf - T) over a convection face (an integral QoI of T on the face). Backward
    Euler with its own reaction (the backward difference) balances to round-off; Crank-Nicolson and BDF2
    reactions use a second-order difference and the trapezoidal rule integrates them: tolerance 0.2 % of the
    heat exchanged (measured ~1e-4 with the start-up)."""
    mesh = strip(tmp_path / "s.msh", 0.05, 0.01, 20, 2, order=2)
    h, Tf = 500.0, 400.0
    q = [QoI(name="E", type="integral", field="T"), QoI(name="R", type="reaction", group="left"),
         QoI(name="F", type="integral", field="T", group="right")]
    c = heat(mesh, {"t_end": 200.0, "n_steps": 200, "scheme": scheme, "initial": {"T": 300.0}},
             [BC(type="temperature", group="left", value=200.0), BC(type="convection", group="right", h=h, T_inf=Tf)], q)
    r = solve(c)
    t = r.times
    stored = RHO * CP * (r.series(q[0]) - r.series(q[0])[0])
    area = 0.01                                    # face length x thickness 1
    inflow = r.series(q[1]) + h * (Tf * area - r.series(q[2]))
    if scheme == "backward_euler":                 # rectangle rule of the scheme
        put_in = np.concatenate([[0.0], np.cumsum(inflow[1:] * np.diff(t))])
    else:
        put_in = np.concatenate([[0.0], np.cumsum(0.5 * (inflow[1:] + inflow[:-1]) * np.diff(t))])
    scale = np.abs(stored).max()
    print(scheme, np.abs(stored - put_in).max() / scale)
    assert np.abs(stored - put_in).max() < tol * scale


# --------------------------------------------------------------------------- 5. features


def _square(tmp_path, order=1, n=2):
    return strip(tmp_path / "sq.msh", 0.01, 0.01, n, n, order=order)


def test_h_in_time_lumped(tmp_path):
    """h(t) = h0 (1 + t / t1) on a lumped body: T = Tf + (Ti - Tf) exp(-(P / (rho cp A)) int h); matrix
    re-factored every step (it changes every step). Tolerance 0.1 % of the step (CN, 80 steps)."""
    a, h0 = 0.01, 2.0
    t1 = 2000.0
    mesh = _square(tmp_path)
    q = [P("T", a / 2, a / 2)]
    c = heat(mesh, {"t_end": 4000.0, "n_steps": 80, "initial": {"T": TI}},
             [BC(type="convection", group=g, h=f"{h0} * (1 + t / {t1})", T_inf=TS)
              for g in ("left", "right", "bottom", "top")], q)
    r = solve(c)
    H = h0 * (r.times + r.times ** 2 / (2 * t1))
    exact = TS + (TI - TS) * np.exp(-4 * a / (RHO * CP * a * a) * H)
    assert np.abs(r.series(q[0]) - exact).max() < 1e-3 * (TI - TS)
    assert "factorizations=80" in info_line(r, "info: linear_solver")


def test_h_table_refactors_only_on_change(tmp_path):
    """A table h that steps once: the Crank-Nicolson matrix is factored twice, not every step. The table's
    T_inf ramps; the answer matches a run with the same values as expressions (to 1e-10)."""
    mesh = _square(tmp_path)
    tab_h = {"table": [[0.0, 5.0], [1000.0, 5.0], [1000.0 + 1e-9, 50.0]]}
    tab_T = {"table": [[0.0, 300.0], [2000.0, 100.0]]}
    q = [P("T", 0.005, 0.005)]
    bcs = [BC(type="convection", group="left", h=tab_h, T_inf=tab_T)]
    r = solve(heat(mesh, {"t_end": 2000.0, "n_steps": 40, "initial": {"T": 300.0}}, bcs, q))
    assert "factorizations=2" in info_line(r, "info: linear_solver")
    expr = [BC(type="convection", group="left", h="5.0 + 45.0 * (t > 1000.0 + 0.5e-9)", T_inf="300.0 - 0.1 * t")]
    r2 = solve(heat(mesh, {"t_end": 2000.0, "n_steps": 40, "initial": {"T": 300.0}}, expr, q))
    assert np.abs(r.series(q[0]) - r2.series(q[0])).max() < 1e-10 * 300


def test_dirichlet_table_and_expression(tmp_path):
    """A face temperature given as a table and as the same ramp in t give the same history (1e-10); a
    ramp from the initial value is compatible (no start-up)."""
    mesh = strip(tmp_path / "s.msh", 0.05, 0.01, 10, 1, order=2)
    q = [P("T", 0.02, 0.005)]
    tr = {"t_end": 100.0, "n_steps": 20, "initial": {"T": 300.0}}
    a = solve(heat(mesh, tr, [BC(type="temperature", group="left", value={"table": [[0, 300.0], [50.0, 400.0]]})], q))
    b = solve(heat(mesh, tr, [BC(type="temperature", group="left", value="300.0 + 2.0 * min(t, 50.0)")], q))
    assert np.abs(a.series(q[0]) - b.series(q[0])).max() < 1e-10 * 300
    assert not any("Rannacher" in w for w in a.info.warnings)


def test_initial_field_from_file(tmp_path):
    """An initial field from an .npz (x, T at the mesh nodes) gives the same run as the same expression."""
    mesh = strip(tmp_path / "s.msh", 0.05, 0.01, 10, 2, order=2)
    expr = "300.0 + 1000.0 * x"
    tr = {"t_end": 50.0, "n_steps": 10, "initial": {"T": expr}}
    bcs = [BC(type="convection", group="right", h=100.0, T_inf=280.0)]
    q = [P("T", 0.01, 0.005)]
    a = solve(heat(mesh, tr, bcs, q))
    X, _ = a.state(0.0).nodal("T")
    path = tmp_path / "T0.npz"
    np.savez(path, x=X[:, :2], T=300.0 + 1000.0 * X[:, 0])
    b = solve(heat(mesh, {**tr, "initial": {"T": {"field": str(path)}}}, bcs, q))
    assert np.abs(a.series(q[0]) - b.series(q[0])).max() < 1e-9 * 300
    with pytest.raises(CaseError, match="not found"):
        solve(heat(mesh, {**tr, "initial": {"T": {"field": str(tmp_path / "missing.npz")}}}, bcs, q))


def test_crank_nicolson_startup(tmp_path):
    """A quench (initial state against the face temperature): without a start-up Crank-Nicolson rings near the
    face (the error alternates in sign from step to step and stays above 5 % of the jump over the first ten
    steps); with the automatic Rannacher start-up it is smooth and within 1 % from step 3 on. 'off' warns,
    'auto' says what it did."""
    L, n = 0.1, 40
    mesh = strip(tmp_path / "s.msh", L, L / n, n, order=2)
    q = [P("T", L / 20, L / (2 * n))]
    tr = {"t_end": T_QUENCH, "n_steps": 40, "initial": {"T": TI}}
    bcs = [BC(type="temperature", group="left", value=TS), BC(type="temperature", group="right", value=TS)]
    res = {}
    for mode in ("off", "auto"):
        r = solve(heat(mesh, tr, bcs, q, cn_startup=mode))
        res[mode] = (r, (r.series(q[0]) - ser.slab_quench(L / 20, r.times, L, ALPHA, TI, TS)) / (TI - TS))
    r_off, e_off = res["off"]
    r_auto, e_auto = res["auto"]
    assert any("cn_startup is 'off'" in w for w in r_off.info.warnings)
    assert any("Rannacher" in w for w in r_auto.info.warnings)
    flips = np.sum(np.sign(e_off[1:10]) != np.sign(e_off[2:11]))
    assert flips >= 6 and np.abs(e_off[1:11]).max() > 0.05
    assert np.abs(e_auto[3:]).max() < 0.01
    with pytest.raises(CaseError, match="cn_startup"):
        solve(heat(mesh, tr, bcs, q, cn_startup="sometimes"))


def test_qoi_time_reductions_and_outputs(tmp_path):
    """QoI.time: the value at a time (linear between steps), max over time, and a window; output_times keeps
    those states; state(t) is a steady-style result (probe, nodal, evaluate, write_vtu)."""
    mesh = strip(tmp_path / "s.msh", 0.05, 0.01, 10, 1, order=2)
    q = [P("Tend", 0.02, 0.005), P("Tmax", 0.02, 0.005, time="max"), P("T30", 0.02, 0.005, time=30.0),
         P("Tmean", 0.02, 0.005, time="mean", window=[20.0, 60.0]),
         QoI(name="qx", type="point", field="qx", point=[0.02, 0.005])]
    c = heat(mesh, {"t_end": 100.0, "n_steps": 20, "initial": {"T": 300.0}, "output_times": [30.0, 50.0]},
             [BC(type="temperature", group="left", value="300.0 + 100.0 * sin(pi * t / 100.0)")], q)
    r = solve(c)
    s = r.series(q[0])
    v = r.evaluate_all()
    assert v["Tmax"] == pytest.approx(s.max())
    assert v["T30"] == pytest.approx(s[6])                      # t = 30 is step 6
    assert v["Tend"] == pytest.approx(s[-1])
    w = (r.times >= 20.0) & (r.times <= 60.0)
    assert v["Tmean"] == pytest.approx(np.trapezoid(s[w], r.times[w]) / 40.0)
    assert list(r.times[r.kept]) == [30.0, 50.0, 100.0]
    st = r.state(30.0)
    assert st.evaluate_qoi(q[0]) == pytest.approx(s[6])
    assert st.probe("qx", np.array([[0.02, 0.005]]))[0] == pytest.approx(r.series(q[4])[6])
    st.write_vtu(tmp_path / "s30.vtu")
    assert (tmp_path / "s30.vtu").stat().st_size > 0
    assert any(w_.startswith("info: transient heat: scheme=crank_nicolson") for w_ in r.info.warnings)


def test_options_and_errors(tmp_path):
    """The initial state is required; an unknown option warns; a fast-changing value warns; h may not depend
    on x."""
    mesh = _square(tmp_path)
    q = [P("T", 0.005, 0.005)]
    bcs = [BC(type="convection", group="left", h=10.0, T_inf=300.0)]
    with pytest.raises(CaseError, match="initial"):
        solve(heat(mesh, {"t_end": 10.0, "n_steps": 2}, bcs, q))
    r = solve(heat(mesh, {"t_end": 10.0, "n_steps": 2, "initial": {"T": 300.0}}, bcs, q, foo=1))
    assert any("foo" in w for w in r.info.warnings)
    fast = [BC(type="convection", group="left", h=10.0, T_inf="300 + 10 * sin(2 * pi * t)")]
    r = solve(heat(mesh, {"t_end": 10.0, "n_steps": 20, "initial": {"T": 300.0}}, fast, q))
    assert any("faster than the time step" in w for w in r.info.warnings)
    with pytest.raises(CaseError, match="not on x, y, z"):
        solve(heat(mesh, {"t_end": 10.0, "n_steps": 2, "initial": {"T": 300.0}},
                   [BC(type="convection", group="left", h="10 * (1 + x) * (1 + t)", T_inf=300.0)], q))


def test_steady_case_unchanged(tmp_path):
    """A heat case without the transient option is the steady solve (not this module)."""
    mesh = _square(tmp_path, order=2)
    c = Case(mesh=mesh, analysis=Analysis(physics="heat", formulation="2d", order=2), materials={"m": Material(k=K)},
             bcs=[BC(type="temperature", group="left", value=100.0), BC(type="temperature", group="right", value=200.0)],
             qois=[P("T", 0.005, 0.005)])
    r = solve(c)
    assert type(r).__name__ == "SkfemResult"
    assert r.evaluate_all()["T"] == pytest.approx(150.0, abs=1e-9)


# --------------------------------------------------------------------------- 6. benchmark problems


@pytest.mark.parametrize("pid", ["nafems_t3", "convective_slab_chilldown"])
def test_problem_registered(pid, tmp_path):
    """Registered, valid case (JSON round trip), no reference value in the agent statement, mesh groups,
    STEP export."""
    meshio = pytest.importorskip("meshio")
    from feacore.problems.registry import all_problems, get

    pr = get(pid)
    assert pr in all_problems(tags=("transient",))
    c = pr.case("MESH.msh").check()
    assert Case.from_dict(json.loads(json.dumps(c.to_dict()))).to_dict() == c.to_dict()
    text = json.dumps(pr.statement())
    for ref in pr.references():
        assert ref.qoi in pr.statement()["description"]
        for s in {f"{ref.value:.4g}", f"{ref.value:.3g}", f"{ref.value:.2f}"}:
            assert s not in text, s
    assert len(pr.private()["study_sizes"]) >= 3
    m = meshio.read(pr.mesh(tmp_path / "m.msh", pr.params["h0"]))
    for b in c.bcs:
        assert b.group in m.field_data
    assert pr.export_step(tmp_path / "g.step").read_text(errors="ignore").startswith("ISO-10303-21")


def test_convective_slab_problem_solves(tmp_path):
    """The chill-down problem on its h0 mesh (P2) meets its references (0.2 %; measured < 0.01 %)."""
    from feacore.problems.registry import get
    from feacore.problems.transient_heat import convective_slab as cs

    pr = get("convective_slab_chilldown")
    v = solve(pr.case(pr.mesh(tmp_path / "w.msh", pr.params["h0"]))).evaluate_all()
    for ref in pr.references():
        print(ref.qoi, v[ref.qoi], ref.value)
        assert abs(v[ref.qoi] - ref.value) <= ref.tolerance * abs(ref.value)
    assert cs.BIOT == pytest.approx(1.25)
