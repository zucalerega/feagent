"""Compressible internal flow (physics "compressible" with total_inlet / pressure_outlet / mass_flow): the exact
quasi-1D nozzle solution (feacore/problems/compressible/quasi1d.py) against NACA Report 1135's tables, the
boundary conditions of feacore/solver/_foam_internal.py, and the converging-diverging nozzle
(feacore/problems/compressible/nozzle.py) solved with rhoCentralFoam in its three regimes
(docs/solver_transient.md, "Compressible internal flow")."""

import math

import numpy as np
import pytest

from feacore.case import BC, Analysis, Case, CaseError, Material, QoI, Region
from feacore.problems.compressible import quasi1d as q1
from feacore.problems.compressible.nozzle import REGIMES, TOL, CDNozzle2D, shock_position
from feacore.problems.registry import get


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam
    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


needs_foam = pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")


# --------------------------------------------------------------------------- the reference itself


# NACA Report 1135, Table I (isentropic flow, gamma = 1.4) and Table II (normal shock), 4-5 significant digits.
ISENTROPIC = [  # M, p/p0, T/T0, A/A*
    (0.5, 0.84302, 0.95238, 1.3398),
    (1.0, 0.52828, 0.83333, 1.0000),
    (2.0, 0.12780, 0.55556, 1.6875),
    (3.0, 0.027224, 0.35714, 4.2346),
]
SHOCK = [  # M1, p2/p1, M2, p02/p01
    (1.5, 2.4583, 0.70109, 0.92979),
    (2.0, 4.5000, 0.57735, 0.72087),
    (3.0, 10.333, 0.47519, 0.32834),
]


@pytest.mark.parametrize("M,p,T,a", ISENTROPIC)
def test_isentropic_tables(M, p, T, a):
    assert q1.p_ratio(M) == pytest.approx(p, rel=1e-4)
    assert q1.T_ratio(M) == pytest.approx(T, rel=1e-4)
    assert q1.area_ratio(M) == pytest.approx(a, rel=1e-4)
    if M != 1.0:      # the root finding inverts the area-Mach relation on the right branch
        assert q1.mach_from_area(q1.area_ratio(M), M > 1) == pytest.approx(M, rel=1e-10)
    assert q1.mach_from_p(q1.p_ratio(M)) == pytest.approx(M, rel=1e-10)


@pytest.mark.parametrize("M1,pr,M2,p0r", SHOCK)
def test_normal_shock_tables(M1, pr, M2, p0r):
    assert q1.shock_p_ratio(M1) == pytest.approx(pr, rel=1e-4)
    assert q1.shock_mach(M1) == pytest.approx(M2, rel=1e-4)
    assert q1.shock_p0_ratio(M1) == pytest.approx(p0r, rel=1e-4)


def test_choked_mass_flux():
    # mdot sqrt(T0) / (p0 A*) = sqrt(gamma / R) (2 / (gamma + 1))^3 = 0.040418 for air (NACA 1135 eq. 80 ff.)
    assert q1.choked_mass_flux(1e5, 300.0, 1.4, 287.0) * math.sqrt(300.0) / 1e5 == pytest.approx(0.040418, rel=1e-4)
    assert q1.mass_flux(1.0, 1e5, 300.0, 1.4, 287.0) == pytest.approx(q1.choked_mass_flux(1e5, 300.0, 1.4, 287.0),
                                                                      rel=1e-12)


def test_nozzle_regimes_exact():
    """The three registered regimes are in their intended ranges, and the shock solution is consistent: the
    pressure behind the shock follows from the normal-shock relation, the mass flow is the same on both sides."""
    for reg, pb in REGIMES.items():
        p = get(f"cd_nozzle_{reg}_2d")
        ex = p.exact_solution()
        assert ex.regime == reg and p.params["pb_ratio"] == pb
    p = get("cd_nozzle_shock_2d")
    ex = p.exact_solution()
    xs = ex.x_shock
    assert 0.3 < xs < 0.9
    M1, M2 = ex.mach(xs - 1e-9), ex.mach(xs + 1e-9)
    assert M2 == pytest.approx(q1.shock_mach(M1), rel=1e-6)
    assert ex.p(xs + 1e-9) / ex.p(xs - 1e-9) == pytest.approx(q1.shock_p_ratio(M1), rel=1e-6)
    assert ex.p(p.params["L"]) == pytest.approx(0.80 * p.params["p0"], rel=1e-10)
    for x in (-0.9, 0.0, 0.5, 0.9):     # rho u A is the same everywhere (the choked flow)
        g, R, T = 1.4, 287.0, ex.T(x)
        mf = ex.p(x) / (R * T) * ex.mach(x) * math.sqrt(g * R * T) * p.area(x)
        assert mf == pytest.approx(ex.mass_flow, rel=1e-8)
    sup = get("cd_nozzle_supersonic_2d").exact_solution()
    assert sup.mach(1.0) == pytest.approx(q1.mach_from_area(1.5, True), rel=1e-12)
    assert sup.mass_flow == pytest.approx(ex.mass_flow, rel=1e-12)       # choked: independent of pb


# --------------------------------------------------------------------------- schema and boundary dictionaries


def _internal_case(**kw) -> Case:
    bcs = kw.pop("bcs", None) or [BC("total_inlet", group="in", p0=1e5, T0=300.0),
                                  BC("pressure_outlet", group="out", value=8e4), BC("slip", group="wall")]
    return Case(mesh="m.msh", analysis=Analysis(physics="compressible", formulation="2d"),
                materials={"gas": Material(gamma=1.4, gas_constant=287.0)}, bcs=bcs,
                regions=[Region("fluid", "gas")], qois=[QoI("m", "mass_flow", group="out")], **kw)


def test_schema_internal():
    _internal_case().check()
    with pytest.raises(CaseError):
        _internal_case(bcs=[BC("total_inlet", group="in", p0=1e5), BC("pressure_outlet", group="out", value=8e4)]).check()
    with pytest.raises(CaseError):
        Case(mesh="m.msh", analysis=Analysis(physics="compressible", formulation="2d"),
             materials={"gas": Material(gamma=1.4, gas_constant=287.0)},
             bcs=[BC("total_inlet", group="in", p0=1e5, T0=300.0)], regions=[Region("fluid", "gas")],
             qois=[QoI("m", "mass_flow")]).check()


def test_reservoir_rules():
    from feacore.solver import _foam_internal as fint
    w = []
    r = fint.reservoir(_internal_case(), 1.4, 287.0, w)
    assert r.a_star == pytest.approx(math.sqrt(2 * 1.4 * 287 * 300 / 2.4))
    assert r.flux == pytest.approx(q1.choked_mass_flux(1e5, 300.0, 1.4, 287.0))
    assert not w
    with pytest.raises(CaseError, match="pressure_outlet"):     # nothing to drive the flow
        fint.reservoir(_internal_case(bcs=[BC("total_inlet", group="in", p0=1e5, T0=300.0),
                                           BC("outlet", group="out")]), 1.4, 287.0, [])
    fint.reservoir(_internal_case(bcs=[BC("total_inlet", group="in", p0=1e5, T0=300.0),
                                       BC("pressure_outlet", group="out", value=2e5)]), 1.4, 287.0, w)
    assert any("below the total pressure" in s for s in w)


def test_boundary_dicts():
    from feacore.solver import _foam_internal as fint
    U, P, T = fint.inlet_fields(BC("total_inlet", group="in", p0=1e5, T0=300.0), 1.4)
    assert (U["type"], P["type"], T["type"]) == ("pressureInletOutletVelocity", "totalPressure", "totalTemperature")
    assert P["psi"] == "thermo:psi" and P["gamma"] == 1.4      # the compressible (isentropic) total pressure
    U, P, T = fint.outlet_fields(np.full(3, 8e4), np.array([1.0, 0.0, 1.0]), 300.0)
    assert P["type"] == "mixed" and P["refValue"] == 8e4 and P["valueFraction"][0] == "nonuniform"
    # the switch: a normal shock at Mach 2 in the outlet plane raises p by 4.5
    assert fint.normal_shock_pressure(np.array([2.0]), np.array([1.0]), 1.4)[0] == pytest.approx(4.5)


# --------------------------------------------------------------------------- solves


def _solve(tmp_path, prob, k: float, **opts):
    from feacore.solver import solve
    m = prob.mesh(tmp_path / f"{prob.id}_{k:g}.msh", prob.params["h0"] / k)
    return solve(prob.case(m), backend="openfoam", **opts)


def _balance_line(res) -> str:
    return next(w for w in res.info.warnings if w.startswith("info: mass balance"))


@needs_foam
def test_nozzle_choked_coarse(tmp_path):
    """Fast end-to-end check (40 x 3 cells, a few seconds): the supersonic regime chokes at the exact mass
    flow within 1 % (the coarse mesh's error is -0.1 %), the outlet extrapolates on every face, and the mass
    balance is reported."""
    p = get("cd_nozzle_supersonic_2d")
    res = _solve(tmp_path, p, 0.5)
    ex = p.exact_qois()
    assert res.info.converged
    assert res.evaluate_all()["mass_flow"] == pytest.approx(ex["mass_flow"], rel=0.01)
    assert res.info.outlets == {"outlet": 0.0}
    assert res.info.mass_balance["imbalance"] < p.qoi_tol
    assert "NOT MET" not in _balance_line(res)
    # inflow is negative, outflow positive, and they balance
    inflow = res.evaluate(QoI("m_in", "mass_flow", group="inlet"))
    assert inflow == pytest.approx(-res.evaluate(QoI("m_out", "mass_flow", group="outlet")), rel=1e-4)


# Errors at h0 (80 x 6 cells), measured (docs/solver_transient.md): mass flow -0.04 % (choked) / -0.6 %
# (subsonic), shock position +0.16 % of L. The tolerances (nozzle.TOL) are the benchmark's.
@needs_foam
@pytest.mark.slow
@pytest.mark.parametrize("regime", list(REGIMES))
def test_nozzle_regimes(tmp_path, regime):
    p = get(f"cd_nozzle_{regime}_2d")
    res = _solve(tmp_path, p, 1.0)
    assert res.info.converged
    ex = p.exact_solution()
    q, exq = res.evaluate_all(), p.exact_qois()
    for name in ("mass_flow", "p_exit", "mach_exit"):
        assert q[name] == pytest.approx(exq[name], rel=p.tolerance(name)), name
    assert res.info.mass_balance["imbalance"] < p.qoi_tol     # the stopping rule's own band (a limit cycle)
    assert res.info.outlets["outlet"] == (0.0 if regime == "supersonic" else 1.0)
    # pressure and Mach number along the nozzle (section averages) against the exact solution, away from the
    # shock; 1 % of p0 and 0.02 in Mach cover the scheme's error at h0 and the 2D effects (~1e-3)
    from feacore.problems.compressible.nozzle import section_average
    x = np.linspace(-0.95, 0.95, 39)
    if ex.x_shock is not None:
        x = x[np.abs(x - ex.x_shock) > 0.1]
    pa, ma = section_average(res, p, x, "p"), section_average(res, p, x, "mach")
    assert np.max(np.abs(pa - [ex.p(v) for v in x])) < 0.01 * p.params["p0"]
    assert np.max(np.abs(ma - [ex.mach(v) for v in x])) < 0.02
    xs = shock_position(res, p)
    if regime == "shock":
        assert xs == pytest.approx(ex.x_shock, abs=TOL["x_shock"] * p.params["L"])
    else:
        assert xs is None


@needs_foam
@pytest.mark.slow
def test_supersonic_independent_of_back_pressure(tmp_path):
    """Below p2 the nozzle flow does not depend on the back pressure: an under-expanded (pb < p3) and an
    over-expanded (p3 < pb < p2) exit give the same mass flow and exit state, and the outlet imposes nothing.
    Same to the stopping rule's qoi_tol (1e-3; measured: 1e-5)."""
    out = []
    for pb in (0.05, 0.4):
        res = _solve(tmp_path, CDNozzle2D(id=f"nz_{pb}", pb_ratio=pb), 1.0)
        assert res.info.outlets["outlet"] == 0.0
        out.append(res.evaluate_all())
    for k in ("mass_flow", "p_exit", "mach_exit"):
        assert out[0][k] == pytest.approx(out[1][k], rel=CDNozzle2D.qoi_tol), k


@needs_foam
@pytest.mark.slow
def test_nozzle_mesh_convergence(tmp_path):
    """The choked mass flow and the shock position converge towards the exact values (h0 / 0.5, h0, h0 / 2)."""
    p = get("cd_nozzle_shock_2d")
    ex = p.exact_qois()
    errs_m, errs_x = [], []
    for k in (0.5, 1.0, 2.0):
        res = _solve(tmp_path, p, k)
        errs_m.append(abs(res.evaluate_all()["mass_flow"] / ex["mass_flow"] - 1))
        errs_x.append(abs(shock_position(res, p) - ex["x_shock"]))
    assert errs_m[2] < errs_m[0] and errs_m[2] < 2e-3
    assert errs_x[2] < 0.01 * p.params["L"]
