"""Transient incompressible flow (``feacore/solver/_foam_transient.py``, pimpleFoam).

Skipped when OpenFOAM (ESI line, with pimpleFoam) or foamlib is missing. The fast tests run small
2D channels and boxes (a second or two each) end to end: the series, the kept states, the reductions
of ``QoI.time``, values in time (a table, an expression in t), the steady initial state, a run at rest,
mass conservation per step, the approach to the steady solution, and a 3D hex duct. The slow tests
(``-m slow``) measure the orders in time and in space on the Taylor-Green vortex and run the
registered benchmark problems (DFG 2D-2 takes several minutes). The numbers they check against are in
docs/solver_transient.md, "Transient incompressible flow".
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("foamlib")

from feacore.case import BC, Analysis, Case, CaseError, Material, QoI, Region, table_value  # noqa: E402
from feacore.solver import solve  # noqa: E402
from feacore.solver._foam_env import find_openfoam  # noqa: E402
from feacore.solver.base import TransientResult, reduce_series  # noqa: E402


def _has_pimple() -> bool:
    fe = find_openfoam()
    return fe is not None and shutil.which("pimpleFoam", path=fe.env.get("PATH", "")) is not None


pytestmark = pytest.mark.skipif(not _has_pimple(), reason="OpenFOAM (pimpleFoam) not installed")

MU, RHO, UM, H, L, T = 0.01, 1.0, 1.0, 0.1, 0.6, 0.01   # channel: Re = rho Um H / mu = 10
PROF = f"6*{UM}*y*({H}-y)/{H}**2"                        # parabolic profile, mean speed UM
Q = UM * H * T                                           # its flow rate (m^3/s)


# --------------------------------------------------------------------------- meshes and cases


def _rect(path, Lx, Ly, nx, ny, names):
    """Structured quads on [0, Lx] x [0, Ly]; names(x, y) -> group of the edge with that midpoint."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("r")
        s = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        gmsh.model.occ.synchronize()
        groups: dict = {}
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, c)
            groups.setdefault(names(x, y), []).append(c)
            horizontal = abs(y) < 1e-12 or abs(y - Ly) < 1e-12
            gmsh.model.mesh.setTransfiniteCurve(c, (nx if horizontal else ny) + 1)
        for name, cs in groups.items():
            gmsh.model.addPhysicalGroup(1, cs, name=name)
        gmsh.model.addPhysicalGroup(2, [s], name="fluid")
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _box(path, Lx, Ly, Lz, n):
    """Structured hexes on [0, Lx] x [0, Ly] x [0, Lz] (n cells per Ly); inlet x = 0, outlet x = Lx."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("b")
        v = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
        gmsh.model.occ.synchronize()
        for _, c in gmsh.model.getEntities(1):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, c)
            size = max(x1 - x0, y1 - y0, z1 - z0)
            gmsh.model.mesh.setTransfiniteCurve(c, max(1, round(n * size / Ly)) + 1)
        groups: dict = {}
        for _, s in gmsh.model.getEntities(2):
            x, _, _ = gmsh.model.occ.getCenterOfMass(2, s)
            name = "inlet" if abs(x) < 1e-9 else "outlet" if abs(x - Lx) < 1e-9 else "walls"
            groups.setdefault(name, []).append(s)
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)
        for name, ss in groups.items():
            gmsh.model.addPhysicalGroup(2, ss, name=name)
        gmsh.model.addPhysicalGroup(3, [v], name="fluid")
        gmsh.model.mesh.setTransfiniteVolume(v)
        gmsh.model.mesh.setRecombine(3, v)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _channel_names(x, y):
    return "inlet" if abs(x) < 1e-12 else "outlet" if abs(x - L) < 1e-12 else "walls"


def _channel_case(mesh, transient, inlet=None, extra_qois=(), **opts) -> Case:
    return Case(
        mesh=str(mesh),
        analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=T,
                          options={"transient": transient, **opts}),
        materials={"water": Material(rho=RHO, mu=MU)},
        bcs=[BC("velocity_inlet", group="inlet", value=inlet if inlet is not None else [PROF, 0.0]),
             BC("pressure_outlet", group="outlet", value=0.0),
             BC("wall", group="walls")],
        regions=[Region(group="fluid", material="water")],
        qois=[QoI("dp", "point_difference", field="p", point=[0.2, 0.5 * H], point2=[0.5, 0.5 * H]),
              QoI("q_in", "flow_rate", group="inlet"), QoI("q_out", "flow_rate", group="outlet"),
              QoI("fx_walls", "force", group="walls", component=0),
              QoI("u_mid", "point", field="ux", point=[0.4, 0.5 * H]),
              QoI("u_max", "max", field="u_mag", group="fluid"), *extra_qois],
    )


@pytest.fixture(scope="module")
def channel(tmp_path_factory):
    d = tmp_path_factory.mktemp("transient_channel")
    return _rect(d / "ch.msh", L, H, 24, 8, _channel_names)


@pytest.fixture(scope="module")
def steady_channel(channel):
    case = _channel_case(channel, {})
    case.analysis.options = {}
    return solve(case, backend="openfoam")


RAMP = {"table": [[0.0, 0.0], [0.05, UM]]}   # the inflow speed: 0 -> UM in 0.05 s, then constant


@pytest.fixture(scope="module")
def ramp(channel):
    """A uniform inflow speed ramped by a table; 50 steps of 2 ms, a state kept every 10 steps."""
    extra = (QoI("u_mid_at", "point", field="ux", point=[0.4, 0.5 * H], time=0.031),
             QoI("q_out_max", "flow_rate", group="outlet", time="max"),
             QoI("u_mid_mean_late", "point", field="ux", point=[0.4, 0.5 * H], time="mean", window=[0.05, 0.1]),
             QoI("u_mid_freq", "point", field="ux", point=[0.4, 0.5 * H], time="frequency"))
    case = _channel_case(channel, {"t_end": 0.1, "dt": 0.002, "output_every": 10}, inlet=RAMP, extra_qois=extra)
    res = solve(case, backend="openfoam")
    yield case, res
    res.cleanup()


# --------------------------------------------------------------------------- fast: end to end


def test_tiny_case_end_to_end(ramp):
    case, res = ramp
    assert isinstance(res, TransientResult)
    assert len(res.times) == 51 and res.times[0] == 0.0 and res.times[-1] == pytest.approx(0.1)
    np.testing.assert_allclose(np.diff(res.times), 0.002, rtol=1e-12)
    assert list(res.kept) == [0, 10, 20, 30, 40, 50]
    info = res.info
    assert info.scheme == "backward" and info.n_steps == 50 and info.dt == pytest.approx(0.002)
    assert info.converged and info.solver_steps == 50 and 0 < info.courant_max < 1
    text = "\n".join(info.warnings)
    assert "info: transient flow, pimpleFoam: scheme backward (second order in time), PIMPLE" in text
    assert "info: Courant number reached" in text and "wall time" in text
    assert "uniformFixedValue" in text            # the table became an OpenFOAM table
    for q in case.qois:
        s = res.series(q)
        assert s.shape == res.times.shape and np.isfinite(s).all()


def test_table_inlet_is_exact_at_every_step(ramp):
    """The inflow follows the table at every step (OpenFOAM interpolates the table linearly, as
    ``table_value`` does), and the outflow follows within the round-off of the pressure solve."""
    case, res = ramp
    want = -np.array([table_value(RAMP, t) for t in res.times]) * H * T
    q_in = res.series(next(q for q in case.qois if q.name == "q_in"))
    np.testing.assert_allclose(q_in, want, rtol=0, atol=1e-12 * Q)
    q_out = res.series(next(q for q in case.qois if q.name == "q_out"))
    np.testing.assert_allclose(q_out[1:], -want[1:], rtol=0, atol=1e-7 * Q)


def test_mass_conservation_per_step(ramp):
    """The sum of the face fluxes over the boundary vanishes at every step (incompressible), and
    OpenFOAM's continuity error stays at the level of the linear solver tolerance."""
    _, res = ramp
    assert np.abs(res.boundary_flux).max() < 1e-8 * Q
    assert res.continuity.max() < 1e-8


def test_time_reductions(ramp):
    """``QoI.time`` / ``QoI.window`` through ``result.evaluate`` reduce the series as reduce_series does."""
    case, res = ramp
    by = {q.name: q for q in case.qois}
    s = res.series(by["u_mid"])
    assert res.evaluate(by["u_mid"]) == pytest.approx(s[-1], rel=1e-12)
    assert res.evaluate(by["u_mid_at"]) == pytest.approx(np.interp(0.031, res.times, s), rel=1e-12)
    assert res.evaluate(by["q_out_max"]) == pytest.approx(res.series(by["q_out"]).max(), rel=1e-12)
    assert res.evaluate(by["u_mid_mean_late"]) == pytest.approx(
        reduce_series(res.times, s, "mean", [0.05, 0.1]), rel=1e-12)
    # several QoIs that measure the same thing share one series
    assert np.array_equal(res.series(by["u_mid_at"]), s)
    # a start-up has no frequency: evaluate raises CaseError, evaluate_all gives nan and a warning
    with pytest.raises(CaseError, match="u_mid_freq"):
        res.evaluate(by["u_mid_freq"])
    vals = res.evaluate_all()
    assert math.isnan(vals["u_mid_freq"]) and all(math.isfinite(v) for k, v in vals.items() if k != "u_mid_freq")
    assert any("u_mid_freq" in w for w in res.info.warnings)
    with pytest.raises(CaseError, match="not evaluated during the run"):
        res.series(QoI("other", "point", field="uy", point=[0.3, 0.05]))


def test_kept_states_are_steady_style_results(ramp):
    """``state(t)`` is the kept state nearest to t; its instantaneous QoIs (the full steady loader)
    equal the series (the fast loader used while the solver runs), and probe / nodal work."""
    case, res = ramp
    st = res.state(0.041)
    assert st.time == pytest.approx(0.04)
    i = int(np.argmin(np.abs(res.times - 0.04)))
    for q in case.qois:
        if q.time is None:
            assert st.evaluate_qoi(q) == pytest.approx(res.series(q)[i], rel=1e-9, abs=1e-14)
    ux = st.probe("ux", np.array([[0.4, 0.5 * H], [0.3, 0.02]]))
    assert ux.shape == (2,) and np.all(ux > 0)
    xy, v = st.nodal("u_mag")
    assert len(xy) == len(v) and v.max() > UM
    assert res.state().time == pytest.approx(0.1)      # default: the last step
    assert res.probe("ux", np.array([[0.4, 0.5 * H]]))[0] == pytest.approx(res.evaluate(case.qois[4]), rel=1e-9)


def test_expression_in_t_inlet(channel):
    """A profile times a ramp in t: one sample per face (timeVaryingMappedFixedValue) at every step."""
    case = _channel_case(channel, {"t_end": 0.02, "dt": 0.002}, inlet=[f"{PROF}*minimum(t/0.01, 1)", 0.0])
    res = solve(case, backend="openfoam")
    try:
        assert any("timeVaryingMappedFixedValue" in w for w in res.info.warnings)
        q_in = res.series(case.qois[1])
        q_face = q_in[-1]                     # the face-centre profile's flow rate (not exactly Q)
        assert q_face == pytest.approx(-Q, rel=1e-2)
        np.testing.assert_allclose(q_in, q_face * np.minimum(res.times / 0.01, 1.0), rtol=0, atol=1e-12 * Q)
    finally:
        res.cleanup()


# --------------------------------------------------------------------------- fast: consistency


def test_at_rest_stays_at_rest(tmp_path):
    """A closed box at rest with walls and symmetry planes: nothing moves, at every step."""
    m = _rect(tmp_path / "box.msh", 1.0, 1.0, 6, 6, lambda x, y: "sym" if abs(x) < 1e-12 else "walls")
    case = Case(mesh=str(m),
                analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=1.0,
                                  options={"transient": {"t_end": 0.1, "n_steps": 10, "scheme": "euler"}}),
                materials={"f": Material(rho=1.0, mu=0.01)},
                bcs=[BC("wall", group="walls"), BC("symmetry", group="sym")],
                regions=[Region(group="fluid", material="f")],
                qois=[QoI("u_max", "max", field="u_mag", group="fluid"),
                      QoI("p_max", "max", field="p", group="fluid")])
    res = solve(case, backend="openfoam")
    assert res.info.scheme == "euler" and len(res.times) == 11
    assert any("first order in time" in w for w in res.info.warnings)
    assert np.abs(res.series(case.qois[0])).max() < 1e-12
    assert np.abs(res.series(case.qois[1])).max() < 1e-9


def test_steady_inflow_converges_to_the_steady_solution(channel, steady_channel):
    """From rest with a constant inflow, the flow settles on the steady backend's solution (the
    viscous time H^2 / nu = 1 s at Re 10, but the start-up is mostly convective: 0.2 s suffice).

    The wall shear force differs by about 1e-4: pimpleFoam's face flux (Rhie-Chow interpolation with
    the time term in its coefficient) depends on the step, so its steady state is not simpleFoam's to
    round-off; the difference falls with the step (8.8e-5, 7.8e-5, 3.0e-5 at 4, 2, 1 ms) and is far
    below the discretization error."""
    case = _channel_case(channel, {"t_end": 0.2, "dt": 0.002, "output_every": 1000})
    res = solve(case, backend="openfoam")
    want = steady_channel.evaluate_all()
    got = res.evaluate_all()
    res.cleanup()
    for k in ("dp", "u_mid", "u_max"):
        assert got[k] == pytest.approx(want[k], rel=1e-5), k
    assert got["fx_walls"] == pytest.approx(want["fx_walls"], rel=2e-4)   # see the note below
    assert got["q_out"] == pytest.approx(want["q_out"], rel=1e-9)


def test_steady_initial_state(channel, steady_channel):
    """``initial = {"steady": true}`` starts on the steady solution, which then does not change."""
    case = _channel_case(channel, {"t_end": 0.01, "dt": 0.002, "initial": {"steady": True}})
    res = solve(case, backend="openfoam")
    want = steady_channel.evaluate_all()
    try:
        assert any("initial state = the steady solution" in w for w in res.info.warnings)
        for q in case.qois:
            if q.name in ("dp", "fx_walls", "u_mid", "u_max"):
                s = res.series(q)
                assert s[0] == pytest.approx(want[q.name], rel=1e-9), q.name
                tol = 2e-4 if q.name == "fx_walls" else 1e-5     # see the steady test above
                np.testing.assert_allclose(s, want[q.name], rtol=tol, err_msg=q.name)
        assert res.info.outer_iterations <= 3
    finally:
        res.cleanup()


def test_adaptive_step_lands_on_the_result_steps(channel):
    """``max_co``: the solver cuts its step to the Courant number, the result keeps the grid of dt."""
    case = _channel_case(channel, {"t_end": 0.02, "dt": 0.005, "max_co": 0.2})
    res = solve(case, backend="openfoam")
    try:
        np.testing.assert_allclose(res.times, [0.0, 0.005, 0.01, 0.015, 0.02], atol=1e-12)
        assert res.info.solver_steps > res.info.n_steps
        # the first step runs at dt from rest; the Courant number of each later step is at most max_co
        assert res.courant[2:].max() <= 0.2 * 1.05
        assert res.info.dt_min < 0.005
    finally:
        res.cleanup()


def test_3d_hex_duct(tmp_path):
    """A square duct of hexes, uniform inflow from rest: flow in = flow out at every step after the first."""
    m = _box(tmp_path / "duct.msh", 0.4, 0.1, 0.1, 4)
    case = Case(mesh=str(m),
                analysis=Analysis(physics="flow", formulation="3d", order=2,
                                  options={"transient": {"t_end": 0.02, "dt": 0.002}}),
                materials={"f": Material(rho=1.0, mu=0.01)},
                bcs=[BC("velocity_inlet", group="inlet", value=[0.5, 0.0, 0.0]),
                     BC("pressure_outlet", group="outlet", value=0.0), BC("wall", group="walls")],
                regions=[Region(group="fluid", material="f")],
                qois=[QoI("q_in", "flow_rate", group="inlet"), QoI("q_out", "flow_rate", group="outlet"),
                      QoI("drag", "force", group="walls", component=0)])
    res = solve(case, backend="openfoam")
    try:
        q_in, q_out = res.series(case.qois[0]), res.series(case.qois[1])
        np.testing.assert_allclose(q_in, -0.5 * 0.01, rtol=1e-12)
        np.testing.assert_allclose(q_out[1:], 0.5 * 0.01, rtol=1e-7)
        assert res.series(case.qois[2])[-1] > 0
        assert res.info.dim == 3 and res.info.converged
    finally:
        res.cleanup()


def test_refusals(channel):
    case = _channel_case(channel, {"t_end": 0.01, "dt": 0.002}, turbulence="kOmegaSST")
    with pytest.raises(NotImplementedError, match="laminar only"):
        solve(case, backend="openfoam")
    case = _channel_case(channel, {"t_end": 0.01, "dt": 0.002, "initial": {"steady": True, "p": 0}})
    with pytest.raises(CaseError):
        solve(case, backend="openfoam")


# --------------------------------------------------------------------------- slow: Taylor-Green orders

NU_TG = 0.05


def _tg_case(mesh, dt, scheme, t_end=1.0):
    from feacore.problems.registry import get

    p = get("taylor_green_vortex_2d")
    case = p.case(mesh)
    case.analysis.options["transient"].update({"dt": dt, "scheme": scheme, "t_end": t_end})
    case.qois = [case.qois[0]]
    return p, case


def _tg_mesh(tmp_path, n):
    from feacore.problems.registry import get

    p = get("taylor_green_vortex_2d")
    return p.mesh(tmp_path / f"tg{n}.msh", p.params["L"] / n)


def _order(q1, q2, q3, r=2.0):
    return math.log(abs(q1 - q2) / abs(q2 - q3)) / math.log(r)


@pytest.mark.slow
@pytest.mark.parametrize("scheme,order", [("backward", 2.0), ("euler", 1.0)])
def test_taylor_green_order_in_time(tmp_path, scheme, order):
    """Step halving (0.02, 0.01, 0.005 s; Courant 0.64 to 0.16) on 32 x 32 cells: the observed order
    (from the three values, so the spatial error cancels) is the scheme's, within 0.1 (measured
    backward 1.98, euler 0.99)."""
    m = _tg_mesh(tmp_path, 32)
    vals = []
    for dt in (0.02, 0.01, 0.005):
        _, case = _tg_case(m, dt, scheme)
        res = solve(case, backend="openfoam")
        assert res.info.converged
        vals.append(res.evaluate(case.qois[0]))
        res.cleanup()
    assert _order(*vals) == pytest.approx(order, abs=0.1)


@pytest.mark.slow
def test_taylor_green_order_in_space(tmp_path):
    """16, 32, 64 cells per side at step 0.005 s (BDF2; its error is 100 times below the spatial one):
    the error of ux at A against the exact value falls with order about 2 (measured 2.3, 2.2), and
    the observed order from the three values is 2 within 0.4."""
    p = None
    errs, vals = [], []
    for n in (16, 32, 64):
        p, case = _tg_case(_tg_mesh(tmp_path, n), 0.005, "backward")
        res = solve(case, backend="openfoam")
        v = res.evaluate(case.qois[0])
        res.cleanup()
        ref = p.references()[0].value
        vals.append(v)
        errs.append(abs(v / ref - 1.0))
    assert errs[0] < 0.012 and errs[2] < 0.001
    for e1, e2 in zip(errs, errs[1:]):
        assert math.log2(e1 / e2) > 1.7
    assert _order(*vals) == pytest.approx(2.0, abs=0.4)


@pytest.mark.slow
def test_taylor_green_problem_meets_its_references(tmp_path):
    from feacore.problems.registry import get

    p = get("taylor_green_vortex_2d")
    m = p.mesh(tmp_path / "tg.msh", p.params["h0"])
    res = solve(p.case(m), backend="openfoam")
    got = res.evaluate_all()
    res.cleanup()
    for r in p.references():
        assert got[r.qoi] == pytest.approx(r.value, rel=r.tolerance), r.qoi


# --------------------------------------------------------------------------- slow: DFG 2D-2


@pytest.mark.slow
def test_dfg_2d2_reference_mesh(tmp_path):
    """DFG 2D-2 on the reference mesh h0 (2954 cells) at the problem's step: shedding develops from
    the impulsive start, and the maximum drag, the maximum lift and the Strouhal number are within the
    registered tolerances (several minutes; the measured values are in docs/solver_transient.md)."""
    from feacore.problems.registry import get

    p = get("cylinder_channel_dfg_2d2")
    m = p.mesh(tmp_path / "dfg.msh", p.params["h0"])
    case = p.case(m)
    res = solve(case, backend="openfoam")
    got = res.evaluate_all()
    res.cleanup()
    for r in p.references():
        assert got[r.qoi] == pytest.approx(r.value, rel=r.tolerance), (r.qoi, got[r.qoi], r.value)
