"""The shared schema of time-dependent and modal analyses (feacore/case.py, docs/solver_transient.md):
time in values, the transient and modal options, time reductions of a QoI, and that steady cases are
untouched."""

import json

import numpy as np
import pytest

from feacore.case import (BC, QOI_TIME, Analysis, Case, CaseError, Load, Material, QoI, as_field, eval_expr,
                          table_value, uses_time)
from feacore.solver import solve
from feacore.solver.base import dominant_frequency, reduce_series


def _heat(options=None, bcs=None, qois=None, material=None):
    return Case(mesh="m.msh", analysis=Analysis(physics="heat", formulation="3d", options=options or {}),
                materials={"m": material or Material(k=12.0, rho=7990.0, cp=400.0)},
                bcs=bcs or [BC(type="temperature", group="g", value=300.0)], qois=qois or [])


TR = {"t_end": 10.0, "dt": 0.5}


def test_time_in_expressions_and_tables():
    assert eval_expr("90 + 203 * exp(-t / 5)", [0.0], [0.0], t=5.0)[0] == pytest.approx(90 + 203 * np.exp(-1))
    assert eval_expr("x + 2 * y", [1.0, 2.0], [0.5, 0.5]).tolist() == [2.0, 3.0]          # no t: as before
    with pytest.raises(CaseError, match="uses the time t"):
        eval_expr("t * x", [1.0], [1.0])
    tab = {"table": [[0.0, 293.0], [10.0, 90.0]]}
    assert table_value(tab, 5.0) == pytest.approx(191.5) and table_value(tab, -1) == 293.0 and table_value(tab, 99) == 90.0
    assert as_field(tab, [0, 0], [0, 0], t=5.0).tolist() == [191.5, 191.5]
    with pytest.raises(CaseError, match="table"):
        as_field(tab, [0], [0])
    assert uses_time("x * t") and not uses_time("x * tan(y)") and uses_time(tab) and uses_time([0, "t", 1])
    assert not uses_time(3.0) and not uses_time([1.0, "x"])


def test_transient_options_are_validated():
    assert _heat({"transient": dict(TR, scheme="crank_nicolson", initial={"T": 293.15})}).validate() == []
    assert len(_heat({"transient": TR}).analysis.time_steps()) == 21
    assert len(_heat({"transient": {"t_end": 1.0, "n_steps": 4}}).analysis.time_steps()) == 5

    def errors(tr, **kw):
        return " | ".join(_heat({"transient": tr}, **kw).validate())

    assert "unknown keys ['schem']" in errors(dict(TR, schem="x"))
    assert "exactly one of dt" in errors({"t_end": 1.0}) and "exactly one of dt" in errors(dict(TR, n_steps=3))
    assert "scheme for heat" in errors(dict(TR, scheme="newmark"))
    assert "t_end must be a number > 0" in errors({"t_end": 0, "dt": 1})
    assert "rho > 0 (kg/m^3) and cp > 0" in errors(TR, material=Material(k=12.0))
    assert "cannot depend on t" in errors(dict(TR, initial={"T": "300 + t"}))
    assert "output_times or output_every" in errors(dict(TR, output_times=[1.0], output_every=2))
    assert "max_co is for flow" in errors(dict(TR, max_co=1.0))


def test_time_dependent_values_need_a_transient_analysis():
    timed = [BC(type="convection", group="g", h={"table": [[0, 100.0], [5, 600.0]]}, T_inf="293 - 200 * (1 - exp(-t))")]
    assert _heat({"transient": TR}, bcs=timed).validate() == []
    errs = " | ".join(_heat(bcs=timed).validate())
    assert "depends on the time t" in errs
    bad = [BC(type="temperature", group="g", value={"table": [[1, 2], [1, 3]]})]
    assert "table times must increase" in " | ".join(_heat({"transient": TR}, bcs=bad).validate())


def test_qoi_time_reductions_are_validated():
    q = [QoI(name="Tmin", type="min", field="T", time="min", window=[1.0, 10.0])]
    assert _heat({"transient": TR}, qois=q).validate() == []
    assert "need a transient analysis" in " | ".join(_heat(qois=q).validate())
    bad = [QoI(name="a", type="min", field="T", time="largest"), QoI(name="b", type="min", field="T", window=[2.0, 1.0])]
    errs = " | ".join(_heat({"transient": TR}, qois=bad).validate())
    assert f"one of {QOI_TIME}" in errs and "window must be [t0, t1]" in errs


def test_modal_options_and_qois():
    def modal(mo, qois, rho=7850.0):
        return Case(mesh="m.msh", analysis=Analysis(options={"modal": mo}), materials={"m": Material(E=2e11, nu=0.3, rho=rho)},
                    bcs=[BC(type="fixed", group="g")], qois=qois)

    f1 = QoI(name="f1", type="natural_frequency", mode=1)
    assert modal({"n_modes": 6}, [f1, QoI(name="mx", type="effective_mass", mode=1, component=0)]).validate() == []
    assert "n_modes must be an integer" in " | ".join(modal({"n_modes": 0}, [f1]).validate())
    assert "needs mode" in " | ".join(modal({"n_modes": 3}, [QoI(name="f", type="natural_frequency")]).validate())
    assert "natural_frequency and effective_mass only" in " | ".join(
        modal({"n_modes": 3}, [QoI(name="u", type="max", field="u_mag")]).validate())
    assert "rho > 0" in " | ".join(modal({"n_modes": 3}, [f1], rho=0.0).validate())
    steady = Case(mesh="m.msh", analysis=Analysis(), materials={"m": Material(E=2e11, nu=0.3)}, bcs=[BC(type="fixed", group="g")], qois=[f1])
    assert "needs a modal analysis" in " | ".join(steady.validate())


def test_steady_cases_serialise_as_before_and_round_trip():
    old = Case(mesh="m.msh", analysis=Analysis(), materials={"m": Material(E=2e11, nu=0.3)},
               bcs=[BC(type="fixed", group="g"), BC(type="pressure", group="p", value=1e6)],
               loads=[Load(type="body_force", value=[0, 0, -1])], qois=[QoI(name="u", type="max", field="u_mag")])
    d = old.to_dict()
    assert sorted(d["materials"]["m"]) == ["E", "alpha", "k", "mu", "nu", "rho"]            # no cp
    assert "p0" not in d["bcs"][0] and "time" not in d["qois"][0] and "mode" not in d["qois"][0]
    new = _heat({"transient": dict(TR, initial={"T": 293.15})},
                qois=[QoI(name="T", type="max", field="T", time="max", window=[0.0, 5.0])])
    again = Case.from_dict(json.loads(json.dumps(new.to_dict())))
    assert again.to_dict() == new.to_dict() and again.qois[0].time == "max" and again.materials["m"].cp == 400.0


def test_reductions_of_a_series():
    t = np.linspace(0.0, 2.0, 801)
    y = 3.0 + 0.7 * np.sin(2 * np.pi * 7.3 * t + 0.4)
    assert dominant_frequency(t, y) == pytest.approx(7.3, rel=1e-5)
    periods = [0.0, 10 / 7.3]
    assert reduce_series(t, y, "mean", periods) == pytest.approx(3.0, abs=2e-3)
    assert reduce_series(t, y, "rms", periods) == pytest.approx(np.sqrt(9 + 0.49 / 2), rel=2e-4)
    assert reduce_series(t, y, "max") == pytest.approx(3.7, abs=1e-4) and reduce_series(t, -y, "max_abs") == pytest.approx(3.7, abs=1e-4)
    assert reduce_series(t, t**2, 1.0) == pytest.approx(1.0, abs=1e-5) and reduce_series(t, t**2) == 4.0
    assert reduce_series(t, t**2, None, [0.0, 1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="two full periods"):
        dominant_frequency(t[:40], y[:40])
    with pytest.raises(ValueError, match="outside the series"):
        reduce_series(t, y, 5.0)


def test_a_capability_that_is_not_installed_says_so(tmp_path):
    """Nothing falls back to a steady solve: a transient or modal case either runs or refuses by name."""
    import importlib.util
    case = _heat({"transient": TR})
    if importlib.util.find_spec("feacore.solver._skfem_transient") is None:
        with pytest.raises(NotImplementedError, match="transient heat conduction"):
            solve(case, backend="skfem")
    with pytest.raises(NotImplementedError, match="transient and modal"):
        solve(case, backend="ccx")
