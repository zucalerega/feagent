"""Transient structural dynamics of the fixed solver (``feacore/solver/_skfem_dynamics.py``,
docs/solver_transient.md "Transient dynamics").

References, all independent of the time integrator:
* the exact solution of the semi-discrete system M a + C v + K u = f(t), from the eigen-decomposition (dense
  LAPACK) of the solver's own K and M: modal superposition, undamped, Rayleigh-damped, step loads, base motion;
* the exact DISCRETE solution of the average-acceleration scheme: each mode keeps its amplitude and turns with
  omega_h dt = 2 arctan(omega dt / 2) (period elongation ~ (omega dt)^2 / 12);
* rigid-body motion under a constant force, u_cm = F t^2 / (2 m);
* the cantilever's first natural frequency from the modal solver and its static deflection;
* the 1D wave solution of a bar (``feacore.problems.dynamics.bar_wave``).
Small P2 meshes; each test takes a few seconds.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from scipy.linalg import eigh  # noqa: E402

from feacore.case import BC, Analysis, Case, CaseError, Load, Material, QoI  # noqa: E402
from feacore.problems.registry import discover, get  # noqa: E402
from feacore.solver import solve  # noqa: E402
from feacore.solver._skfem_dynamics import DynamicState  # noqa: E402

E, RHO = 2e11, 8000.0
C_BAR = math.sqrt(E / RHO)          # 5000 m/s
L_BAR, H_BAR = 1.0, 0.1
F1_BAR = C_BAR / (4 * L_BAR)        # fixed-free bar, nu = 0: 1250 Hz


@contextmanager
def _session(name):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add(name)
        yield
    finally:
        gmsh.finalize()


def rect(path, L, H, h):
    with _session("rect"):
        gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        gmsh.model.occ.synchronize()
        e = 1e-9
        names = {"left": [], "right": [], "bottom": [], "top": []}
        for _, t in gmsh.model.getEntities(1):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, t)
            key = "left" if abs(x) < e else "right" if abs(x - L) < e else "bottom" if abs(y) < e else "top"
            names[key].append(t)
        for n, tg in names.items():
            gmsh.model.addPhysicalGroup(1, tg, name=n)
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name="domain")
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("dynamics")


@pytest.fixture(scope="module")
def bar(tmp):
    return rect(tmp / "bar.msh", L_BAR, H_BAR, 0.05)


@pytest.fixture(scope="module")
def beam(tmp):
    return rect(tmp / "beam.msh", 1.0, 0.1, 0.05)


BAR_BCS = [BC(type="fixed", group="left"), BC(type="displacement", group="top", components=[1], value=0.0),
           BC(type="displacement", group="bottom", components=[1], value=0.0)]
TIP = [L_BAR, H_BAR / 2]


def bar_case(mesh, dt, t_end, bcs=None, initial=None, ray=None, loads=(), qois=None, **tr):
    o = {"transient": {"t_end": t_end, "dt": dt, "output_every": 1, **tr}}
    if initial is not None:
        o["transient"]["initial"] = initial
    if ray is not None:
        o["damping"] = {"rayleigh": ray}
    return Case(mesh=mesh, analysis=Analysis(formulation="plane_strain", order=2, options=o),
                materials={"s": Material(E=E, nu=0.0, rho=RHO)}, bcs=list(bcs if bcs is not None else BAR_BCS),
                loads=list(loads), qois=qois or [QoI("tip", "point", field="ux", point=TIP)])


class Modal:
    """The exact semi-discrete solution from a dense eigen-decomposition of the solver's K and M."""

    def __init__(self, r, dof):
        md = r.model
        self.I = np.setdiff1d(np.arange(md.N), r.dirichlet_dofs)
        KI = r.K[self.I][:, self.I].toarray()
        self.MI = r.M[self.I][:, self.I]
        self.lam, self.Phi = eigh(KI, self.MI.toarray())
        self.w = np.sqrt(self.lam)
        self.row = self.Phi[np.nonzero(self.I == dof)[0][0]]
        self.M, self.K = r.M, r.K

    def coords(self, u):
        return self.Phi.T @ (self.MI @ u[self.I])

    def free(self, t, q0, zeta=None, dt_discrete=None):
        w = self.w if dt_discrete is None else 2.0 / dt_discrete * np.arctan(self.w * dt_discrete / 2.0)
        if zeta is None:
            return (np.cos(np.outer(t, w)) * q0) @ self.row
        z = np.minimum(zeta, 0.9999)
        wd = w * np.sqrt(1 - z**2)
        env = np.exp(-np.outer(t, z * w))
        return (env * (np.cos(np.outer(t, wd)) + z / np.sqrt(1 - z**2) * np.sin(np.outer(t, wd))) * q0) @ self.row


def tip_dof(r, comp=0, point=TIP):
    X = r.model.sb.doflocs
    j = int(np.argmin((X[0] - point[0]) ** 2 + (X[1] - point[1]) ** 2))
    assert np.hypot(X[0, j] - point[0], X[1, j] - point[1]) < 1e-12
    return r.model.split[comp][j]


INITIAL = {"u": ["1e-4*sin(pi*x/2)", 0]}     # close to the first axial mode of the fixed-free bar


# --------------------------------------------------------------------------- exactness of the integrator


def test_newmark_is_the_exact_discrete_rotation(bar):
    """Average acceleration: every mode keeps its amplitude and turns by 2 arctan(omega dt / 2) per step. The run
    matches that to round-off (measured 2e-12 relative), at any step: the scheme is implemented exactly."""
    T = 1 / F1_BAR
    for nper in (10, 40):
        r = solve(bar_case(bar, T / nper, 2 * T, initial=INITIAL))
        mo = Modal(r, tip_dof(r))
        t, y = r.history(r.case.qois[0])
        ex = mo.free(t, mo.coords(r.state(0.0).x), dt_discrete=T / nper)
        assert np.abs(y - ex).max() < 1e-9 * 1e-4


def test_period_error_closed_form(bar):
    """The measured frequency of the first axial mode follows omega_h / omega = 2 arctan(omega dt / 2) / (omega dt):
    -3.1 %, -0.81 %, -0.20 % at 10, 20, 40 steps per period, ~ -(omega dt)^2 / 12."""
    for nper in (10, 20, 40):
        T = 1 / F1_BAR
        dt = T / nper
        r = solve(bar_case(bar, dt, 6 * T, initial=INITIAL,
                           qois=[QoI("f", "point", field="ux", point=TIP, time="frequency")]))
        w = 2 * math.pi * 1250.00001294        # this mesh's first eigenfrequency (modal solver)
        expect = 2 / dt * math.atan(w * dt / 2) / w - 1
        got = r.evaluate(r.case.qois[0]) / (w / (2 * math.pi)) - 1
        assert got == pytest.approx(expect, abs=3e-4 if nper == 10 else 5e-5)
        assert expect == pytest.approx(-(w * dt) ** 2 / 12, rel=0.07)


@pytest.mark.parametrize("scheme,ray", [("newmark", None), ("newmark", [200.0, 1e-7]), ("hht", None)])
def test_second_order_in_time(bar, scheme, ray):
    """Against the exact semi-discrete solution (modal superposition with the modal damping ratios
    zeta = a / (2 omega) + b omega / 2 of Rayleigh damping): the error falls by 4 per halving of dt
    (observed orders 1.96-1.99 newmark, 1.96-1.99 damped, 1.94-1.97 HHT alpha = -0.05)."""
    T = 1 / F1_BAR
    errs = []
    for nper in (10, 20, 40):
        extra = {"scheme": scheme}
        r = solve(bar_case(bar, T / nper, 2 * T, initial=INITIAL, ray=ray, **extra))
        mo = Modal(r, tip_dof(r))
        zeta = None if ray is None else ray[0] / (2 * mo.w) + ray[1] * mo.w / 2
        t, y = r.history(r.case.qois[0])
        errs.append(np.abs(y - mo.free(t, mo.coords(r.state(0.0).x), zeta=zeta)).max() / 1e-4)
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((p > 1.85) & (p < 2.15)), (errs, p)
    assert errs[-1] < 0.035


def test_rayleigh_damping_ratio_from_decay(bar):
    """The free decay of the first mode gives the modal damping ratio that C = a M + b K implies,
    zeta_1 = a / (2 omega_1) + b omega_1 / 2 (here 0.0131), from the logarithmic decrement (within 1 %)."""
    a_, b_ = 200.0, 1e-7
    T = 1 / F1_BAR
    r = solve(bar_case(bar, T / 80, 6 * T, initial=INITIAL, ray=[a_, b_]))
    t, y = r.history(r.case.qois[0])
    w = 2 * math.pi * F1_BAR
    zeta = a_ / (2 * w) + b_ * w / 2
    peaks = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 0]
    n = len(peaks) - 1
    delta = math.log(y[peaks[0]] / y[peaks[-1]]) / n
    assert delta / math.sqrt(4 * math.pi**2 + delta**2) == pytest.approx(zeta, rel=0.01)


def test_energy_conservation_and_balance(bar):
    """Undamped average acceleration conserves kinetic + strain energy to round-off in free vibration
    (measured 1e-13 of the energy); with damping and loads the balance energy = work - dissipation holds to
    round-off too (the trapezoidal rule is exact for this scheme). HHT dissipates."""
    T = 1 / F1_BAR
    r = solve(bar_case(bar, T / 20, 5 * T, initial=INITIAL))
    e = r.energy["kinetic"] + r.energy["strain"]
    assert np.abs(e / e[0] - 1).max() < 1e-11
    r = solve(bar_case(bar, T / 20, 3 * T, ray=[100.0, 1e-7],
                       bcs=BAR_BCS + [BC(type="traction", group="right", value=["1e6*sin(3000*t)", 0])]))
    assert np.abs(r.energy["balance"]).max() < 1e-10 * r.energy["external_work"].max()
    assert r.energy["damping"][-1] > 0
    r = solve(bar_case(bar, T / 20, 5 * T, initial=INITIAL, scheme="hht", alpha=-0.3))
    e = r.energy["kinetic"] + r.energy["strain"]
    assert e[-1] < e[0] * (1 - 1e-4) and np.all(np.diff(e) <= 1e-12 * e[0])


# --------------------------------------------------------------------------- loads, prescribed motion, free body


def test_prescribed_base_motion(bar):
    """The fixed end moves as u_D = U (1 - cos W t). Exact: u = u_D (rigid) + w with M w'' + K w = -M 1 a_D,
    i.e. q_k = -G_k U W^2 / (w_k^2 - W^2) (cos W t - cos w_k t). Second order in dt, so the prescribed
    velocity and acceleration (from the Newmark relations, and by differences at t = 0) are consistent."""
    U, W = 1e-5, 2 * math.pi * 500.0
    bcs = [BC(type="displacement", group="left", components=[0], value=f"{U}*(1 - cos({W}*t))"),
           BC(type="displacement", group="left", components=[1], value=0.0)] + BAR_BCS[1:]
    T = 1 / 500.0
    errs = []
    for nst in (40, 80, 160):
        r = solve(bar_case(bar, T / nst, T, bcs=bcs))
        mo = Modal(r, tip_dof(r))
        one = np.zeros(r.model.N)
        one[r.model.split[0]] = 1.0
        G = mo.Phi.T @ (r.M @ one)[mo.I]
        t, y = r.history(r.case.qois[0])
        q = (-G * U * W**2 / (mo.lam - W**2))[None, :] * (np.cos(W * t)[:, None] - np.cos(np.outer(t, mo.w)))
        ex = U * (1 - np.cos(W * t)) + q @ mo.row
        errs.append(np.abs(y - ex).max() / np.abs(ex).max())
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(p > 1.8), (errs, p)
    assert errs[-1] < 0.01


def test_free_body_rigid_acceleration(tmp, bar):
    """No support: a constant force F accelerates the body; the mean displacement (centre of mass, uniform
    density) is F t^2 / (2 m) exactly (Newmark integrates a constant acceleration exactly)."""
    F = 1000.0
    m = RHO * L_BAR * H_BAR * 1.0
    c = bar_case(bar, 1e-5, 1e-3, bcs=[BC(type="force", group="right", value=[F, 0.0])],
                 qois=[QoI("ucm", "mean", field="ux")])
    r = solve(c)
    assert r.rigid_modes == 3 and any("rigid-body" in w for w in r.info.warnings)
    t, y = r.history(c.qois[0])
    assert np.allclose(y, F * t**2 / (2 * m), rtol=1e-8, atol=1e-18)


def test_separable_and_general_time_loads_agree(bar):
    """A load that varies in time only is assembled once and scaled; one in x and t is reassembled each step.
    Both paths give the same answer; a table works like the expression it samples."""
    T = 1 / F1_BAR
    a = solve(bar_case(bar, T / 20, T, loads=[Load(type="body_force", value=["1e5*sin(5000*t)", 0])]))
    b = solve(bar_case(bar, T / 20, T, loads=[Load(type="body_force", value=["1e5*sin(5000*t) + 0*x", 0])]))
    ya, yb = a.series(a.case.qois[0]), b.series(b.case.qois[0])
    assert np.abs(ya - yb).max() < 1e-12 * np.abs(ya).max()
    tt = np.linspace(0, T, 21)
    tab = {"table": [[float(s), float(v)] for s, v in zip(tt, 1e6 * np.minimum(tt / (T / 2), 1.0))]}
    c1 = solve(bar_case(bar, T / 20, T, bcs=BAR_BCS + [BC(type="traction", group="right", value=[tab, 0])]))
    c2 = solve(bar_case(bar, T / 20, T, bcs=BAR_BCS + [BC(type="traction", group="right",
                                                         value=[f"1e6*minimum(t/{T / 2!r}, 1)", 0])]))
    assert np.allclose(c1.series(c1.case.qois[0]), c2.series(c2.case.qois[0]), rtol=1e-9, atol=1e-20)


def test_refuses_time_dependent_temperature(bar):
    c = bar_case(bar, 1e-5, 1e-4, loads=[Load(type="temperature", value="20 + t")])
    c.materials["s"].alpha = 1e-5
    with pytest.raises(CaseError, match="changes in time"):
        solve(c)


# --------------------------------------------------------------------------- the cantilever under a step load


def test_cantilever_step_load(tmp, beam):
    """A step tip load: the tip oscillates about the static deflection with a peak near twice it, at the first
    natural frequency (from the modal solver). Measured: peak / static 1.99 at 40 steps per period, frequency
    within 0.3 % of f1 (the higher modes the step excites shift the zero crossings a little); the mean over
    whole periods is the static value."""
    mat = {"s": Material(E=E, nu=0.3, rho=RHO)}
    bcs = [BC(type="fixed", group="left"), BC(type="force", group="right", value=[0.0, -1000.0])]
    q = QoI("tip", "point", field="uy", point=[1.0, 0.05])
    st = solve(Case(mesh=beam, analysis=Analysis(formulation="plane_stress", order=2, thickness=0.01),
                    materials=mat, bcs=bcs, qois=[q]))
    u_st = st.evaluate(q)
    mo = solve(Case(mesh=beam, analysis=Analysis(formulation="plane_stress", order=2, thickness=0.01,
                                                 options={"modal": {"n_modes": 1}}),
                    materials=mat, bcs=bcs[:1]))
    f1 = float(mo.frequencies[0])
    T = 1 / f1
    qs = [QoI("peak", "point", field="uy", point=[1.0, 0.05], time="min"),
          QoI("f", "point", field="uy", point=[1.0, 0.05], time="frequency"),
          QoI("mean", "point", field="uy", point=[1.0, 0.05], time="mean", window=[0.0, 4 * T]),
          QoI("at", "point", field="uy", point=[1.0, 0.05], time=T / 2),
          QoI("r", "reaction", group="left", component=1),
          QoI("smax", "max_abs", field="sxx", group="left", time="max"),
          QoI("se", "strain_energy", time="max")]
    c = Case(mesh=beam, analysis=Analysis(formulation="plane_stress", order=2, thickness=0.01,
                                          options={"transient": {"t_end": 4 * T, "dt": T / 40, "scheme": "hht",
                                                                 "alpha": -0.05}}),
             materials=mat, bcs=bcs, qois=qs)
    r = solve(c)
    v = r.evaluate_all()
    assert 1.9 < v["peak"] / u_st < 2.05
    assert v["f"] == pytest.approx(f1, rel=0.005)
    assert v["mean"] == pytest.approx(u_st, rel=0.02)
    assert v["at"] / u_st == pytest.approx(2.0, abs=0.1)          # half a period: the peak
    assert v["se"] > 0 and v["smax"] > 0
    t, y = r.history(qs[0])
    assert len(t) == 161 and t[-1] == pytest.approx(4 * T)
    assert v["at"] == pytest.approx(np.interp(T / 2, t, y))
    st_half = r.state(T / 2)
    assert isinstance(st_half, DynamicState) and st_half.time == pytest.approx(T / 2, rel=0.2)
    assert st_half.probe("von_mises", [[0.05, 0.0]])[0] > 0
    assert st_half.velocity.shape == st_half.x.shape
    assert "info: dynamics: scheme hht" in " ".join(r.info.warnings)
    p = r.write_vtu(tmp / "cant_dyn.vtu", t=T / 2)
    import meshio
    m = meshio.read(p)
    assert "velocity" in m.point_data and "sxx" in m.point_data
    with pytest.raises(CaseError):
        r.series(QoI("other", "point", field="ux", point=[0.5, 0.05]))


def test_quasi_static_reaction(beam):
    """A load ramped over many periods (with damping) gives the static support reaction: the dynamic reaction
    M a + C v + K u - f on the supports tends to minus the load."""
    mat = {"s": Material(E=E, nu=0.3, rho=RHO)}
    T = 1 / 80.3
    c = Case(mesh=beam, analysis=Analysis(formulation="plane_stress", order=2, thickness=0.01,
                                          options={"transient": {"t_end": 12 * T, "dt": T / 10},
                                                   "damping": {"rayleigh": [50.0, 0.0]}}),
             materials=mat,
             bcs=[BC(type="fixed", group="left"),
                  BC(type="force", group="right", value=[0.0, {"table": [[0.0, 0.0], [8 * T, -1000.0]]}])],
             qois=[QoI("r", "reaction", group="left", component=1)])
    r = solve(c)
    assert r.evaluate(c.qois[0]) == pytest.approx(1000.0, rel=0.01)


# --------------------------------------------------------------------------- the registered benchmark


def test_bar_wave_benchmark(tmp):
    """The step-loaded bar (HHT, alpha = -0.1): end displacement mean = static, peak = twice static (a corner,
    -1.4 % at h0), frequency c / (4 L); the stress front reaches mid-length at L / (2 c): before it the stress
    there is ~0, after it ~sigma."""
    discover()
    p = get("bar_wave_step")
    case = p.case(p.mesh(tmp / "wave.msh", p.params["h0"]))
    sig, c = p.params["sigma"], p.c
    case.qois += [QoI("before", "point", field="sxx", point=[0.5, 0.05], time=0.8 * 0.5 / c),
                  QoI("after", "point", field="sxx", point=[0.5, 0.05], time=1.5 * 0.5 / c)]
    r = solve(case)
    v = r.evaluate_all()
    for ref in p.references():
        assert v[ref.qoi] == pytest.approx(ref.value, rel=ref.tolerance), (ref.qoi, v[ref.qoi], ref.value)
    assert abs(v["before"]) < 0.05 * sig
    assert v["after"] == pytest.approx(sig, rel=0.15)
