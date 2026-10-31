"""Code verification of the solver backends by the method of manufactured solutions.

Machinery: ``feacore/verification/mms.py``; results and method: ``docs/code_verification.md``
(regenerate with ``python scripts/run_code_verification.py``).

* ``test_expression_printer`` / ``test_mms_self_check``: the MMS machinery itself (no solver).
* ``test_patch``: linear fields reproduced on distorted meshes (P2: curved interior edges).
* ``test_convergence``: observed orders on a nested refinement study vs theory
  (u ~ k+1, recovered stress / flux ~ k, strain energy ~ 2k or the corner-limited value);
  pass if observed >= theory - 0.3 on the finest pair.

A backend that raises NotImplementedError (or is not importable) is skipped, not failed.
Heavy cases (3D, and all CalculiX convergence studies) are marked slow and skipped unless
``FEACORE_SLOW=1`` or ``-m slow`` is given. Scratch meshes / cases go to ``$FEACORE_CODEVER_DIR``
(default: a pytest temporary directory).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import sympy as sp

from feacore.verification import mms

BACKENDS = ("skfem", "ccx")
# CalculiX results come through the ASCII .frd file (~6 significant digits): that, not the
# solver, limits how exactly a patch test can be reproduced / how small a P2 error is measurable.
PATCH_TOL = {"skfem": (1e-9, 1e-8), "ccx": (5e-5, 5e-5)}
ROUNDOFF = {"skfem": 1e-10, "ccx": 3e-6}


def _run_slow(request) -> bool:
    return os.environ.get("FEACORE_SLOW") == "1" or "slow" in (request.config.getoption("-m") or "")


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    d = os.environ.get("FEACORE_CODEVER_DIR")
    return d if d else str(tmp_path_factory.mktemp("codever"))


# --------------------------------------------------------------------------- machinery only


def test_expression_printer():
    x, y, z = mms.x, mms.y, mms.z
    e = (1 / sp.sqrt(x**2 + y**2) + sp.Rational(1, 3) * x - sp.Float(0.5) * x ** sp.Float(-0.5) + sp.E
         + sp.Abs(x - 0.3) + sp.atan2(y, x) + sp.asin(x / 3) + sp.Max(x, y, 0.2) + sp.exp(-z)
         + sp.Piecewise((x, x < 0.5), (y, True)) + sp.tanh(x * y) + sp.log(1 + x))
    s = mms.to_expr(e)
    for bad in ("Abs", "asin", "atan2(", "E ", "Max", "Piecewise", "math", "numpy"):
        assert bad not in s.replace("arctan2(", "")
    pts = np.random.default_rng(0).uniform(0.1, 1.0, (200, 3))
    mms.check_expr(e, s, pts)
    assert mms.to_expr(sp.pi * 2) == pytest.approx(2 * np.pi)
    with pytest.raises(ValueError):
        mms.to_expr(sp.erf(x))


@pytest.mark.parametrize("pid", mms.CONVERGENCE_IDS + mms.PATCH_IDS)
def test_mms_self_check(pid):
    """Body force / source and BC data in the case (as the solver sees them) agree with the
    exact solution by finite differences and an independent numpy constitutive law."""
    p = mms.convergence_problem(pid) if pid in mms.CONVERGENCE_IDS else mms.patch_problem(pid)
    res = mms.self_check(p)
    assert max(res.values()) < 1e-5, res


def test_isoparametric_control(workdir):
    """Negative control: P2 on a straight-sided (tri3) quarter-annulus mesh must lose the
    optimal u order (geometry error O(h^2)), so the curved-boundary cases do test the
    isoparametric geometry."""
    p = mms.convergence_problem("E-pe-annulus-lame")
    try:
        r = mms.run_convergence(p, 2, "skfem", geom_order=1, workdir=workdir)
    except mms.FeatureMissing as e:
        pytest.skip(str(e))
    assert r["observed"]["u"] < 2.6, r["orders"]


# --------------------------------------------------------------------------- patch tests

_PATCH = {c["id"]: c for c in mms.patch_catalog()}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("pid", mms.PATCH_IDS)
def test_patch(pid, order, backend, workdir):
    spec = _PATCH[pid]
    tol_u, tol_s = PATCH_TOL[backend]
    try:
        r = mms.run_patch(mms.patch_problem(pid), order, backend, h=spec["h"], workdir=workdir,
                          s_scale=spec["s_scale"], tol_u=tol_u, tol_s=tol_s)
    except mms.FeatureMissing as e:
        pytest.skip(str(e))
    assert r["verdict"] == "pass", {k: r[k] for k in r if k.startswith("err") or k in ("checks", "case", "mesh")}


# --------------------------------------------------------------------------- convergence


def _conv_params():
    out = []
    for c in mms.convergence_catalog():
        for backend in BACKENDS:
            for order in (1, 2):
                slow = c["slow"] or backend == "ccx"
                marks = [pytest.mark.slow] if slow else []
                out.append(pytest.param(c["id"], order, backend, slow, marks=marks,
                                        id=f"{c['id']}-P{order}-{backend}"))
    return out


@pytest.mark.parametrize("pid,order,backend,slow", _conv_params())
def test_convergence(pid, order, backend, slow, workdir, request):
    if slow and not _run_slow(request):
        pytest.skip("slow (set FEACORE_SLOW=1 or use -m slow)")
    p = mms.convergence_problem(pid)
    try:
        r = mms.run_convergence(p, order, backend, workdir=workdir)
    except mms.FeatureMissing as e:
        pytest.skip(str(e))
    # re-judge with the backend's round-off / output-precision floor
    hs = [lv["h_rep"] for lv in r["levels"]]
    msg = {"orders": r["orders"], "theory": r["theory"], "cases": r["levels"][-1]["case"]}
    for key, theory in r["theory"].items():
        if key == "W" and (not p.energy_exact or p.energy_physical is not None):
            continue
        e = [lv.get(f"e_{key}") for lv in r["levels"]]
        assert all(v is not None and np.isfinite(v) for v in e), msg
        floor = ROUNDOFF[backend] if key != "W" else ROUNDOFF[backend] / 10
        pobs, _ = mms._finest_order(hs, e, floor)
        assert pobs is not None, msg
        assert pobs >= theory - 0.3, {"field": key, "observed": pobs, **msg}
