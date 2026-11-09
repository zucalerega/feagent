"""Three-mesh GCI (feacore.verification.gci.gci3, Celik et al. 2008), in particular unequal refinement ratios.

Independently remeshed sequences (what agents produce) practically never have r21 = r32. There the
convergence ratio R = e21 / e32 alone does not decide convergence: same-sign changes converge iff a
positive order p fits e32 / e21 = r21^p (r32^p - 1) / (r21^p - 1), i.e. iff R < ln r21 / ln r32.
"""

from __future__ import annotations

import math
import random
import warnings

import pytest

from feacore.verification.gci import _observed_order, gci3

UNEQUAL = [(1.0, 1.5, 3.0),           # r21 < r32: R < 1 is too lenient (a stalling QoI read as monotone)
           (1.0, 2.0, 3.0),           # r21 > r32: R < 1 is too strict (a converging QoI read as divergent)
           (1.0, 1.34, 1.34 * 1.21),  # the single_agent pilot's ratios
           (0.2, 0.3, 0.33)]          # r32 = 1.1: strongly unequal


def power_law(h, p, phi0=1.0, c=0.1):
    return [phi0 + c * x ** p for x in h]


def alternating(h, p, phi0=1.0, c=0.1):
    """An error that changes sign from mesh to mesh (Celik's s = -1 case)."""
    return [phi0 + c * (-1) ** k * x ** p for k, x in enumerate(h)]


def no_warnings(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # an overflow in r^p must not leak out as inf / nan
        return fn(*args, **kw)


# --------------------------------------------------------------------------- equal ratios: unchanged


@pytest.mark.parametrize("h", [(1.0, 2.0, 4.0), (0.1, 0.13, 0.169), (1.0, 1.5, 2.25)])
@pytest.mark.parametrize("p_true", [0.5, 1.0, 2.0, 3.0])
def test_equal_ratios_textbook_formulas(h, p_true):
    phi = power_law(h, p_true)
    g = gci3(h, phi, p_theory=2.0)
    r = h[1] / h[0]
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    assert g.convergence == "monotone"
    assert math.isclose(g.p, math.log(e32 / e21) / math.log(r), rel_tol=1e-12)
    assert math.isclose(g.p, p_true, rel_tol=1e-9)
    rp = r ** g.p
    assert math.isclose(g.phi_ext, (rp * phi[0] - phi[1]) / (rp - 1), rel_tol=1e-12)
    assert math.isclose(g.phi_ext, 1.0, rel_tol=1e-9)
    assert math.isclose(g.gci_fine, 1.25 * abs(e21 / phi[0]) / (rp - 1), rel_tol=1e-12)
    assert math.isclose(g.gci_medium, 1.25 * abs(e32 / phi[1]) / (rp - 1), rel_tol=1e-12)


@pytest.mark.parametrize("R, convergence, flags", [
    (0.5, "monotone", []),
    (0.999, "monotone", []),
    (1.0, "divergent", ["divergent"]),
    (2.0, "divergent", ["divergent"]),
    (-0.5, "oscillatory", ["oscillatory"]),
    (-1.0, "oscillatory_divergent", ["divergent"]),
    (-2.0, "oscillatory_divergent", ["divergent"]),
])
def test_equal_ratios_classify_by_R(R, convergence, flags):
    h = (1.0, 2.0, 4.0)
    e32 = 0.5
    phi = [10.0, 10.0 + R * e32, 10.0 + R * e32 + e32]
    g = gci3(h, phi)
    assert (g.convergence, g.flags) == (convergence, flags)
    assert math.isclose(g.R, R, rel_tol=1e-12)
    if convergence in ("monotone", "oscillatory"):     # p = ln|e32 / e21| / ln r
        assert math.isclose(g.p, math.log(abs(1 / g.R)) / math.log(2.0), rel_tol=1e-12)
    else:
        assert g.p is None and g.phi_ext is None and g.gci_fine is None


# --------------------------------------------------------------------------- unequal ratios


@pytest.mark.parametrize("h", UNEQUAL)
@pytest.mark.parametrize("p_true", [0.25, 0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("c", [0.1, -0.1])
def test_unequal_ratios_power_law_is_monotone(h, p_true, c):
    phi = power_law(h, p_true, c=c)
    g = no_warnings(gci3, h, phi, p_theory=2.0)
    assert g.convergence == "monotone", (g.R, math.log(h[1] / h[0]) / math.log(h[2] / h[1]))
    assert math.isclose(g.p, p_true, rel_tol=1e-9)
    assert math.isclose(g.phi_ext, 1.0, rel_tol=1e-9)
    # on an exact power law the two GCIs scale as the theory says: gci_medium / (r21^p gci_fine) = |phi1 / phi2|
    assert math.isclose(g.asymptotic_ratio, abs(phi[0] / phi[1]), rel_tol=1e-9)
    assert "divergent" not in g.flags


def test_unequal_ratios_converging_with_R_above_one():
    """r21 > r32: a QoI converging at p = 1 has R = e21 / e32 > 1; a ratio-blind test called it divergent."""
    h = (1.0, 1.34, 1.34 * 1.21)
    phi = [2.0 - 0.1 * x for x in h]
    g = gci3(h, phi, p_theory=2.0)
    assert g.R > 1.0
    assert g.convergence == "monotone" and "divergent" not in g.flags
    assert math.isclose(g.p, 1.0, rel_tol=1e-9)
    assert math.isclose(g.phi_ext, 2.0, rel_tol=1e-9)


@pytest.mark.parametrize("h", UNEQUAL + [(1.0, 1.1, 1.1 * 1.6)])
@pytest.mark.parametrize("p_true", [-0.3, -0.05])
def test_unequal_ratios_diverging_stays_divergent(h, p_true):
    """A singular peak (phi ~ h^-0.3) or a slowly creeping one (h^-0.05) is divergent whatever the ratios: with
    r21 < r32 its R is below 1, and Celik's |.| fixed point used to make up a positive order (or raise)."""
    g = no_warnings(gci3, h, [x ** p_true for x in h], p_theory=2.0)
    assert g.convergence == "divergent"
    assert g.flags[-1] == "divergent"
    assert g.p is None and g.phi_ext is None and g.gci_fine is None and g.gci_fine_conservative is None


def test_unequal_ratios_stalling_with_R_below_one():
    h = (1.0, 1.5, 3.0)
    phi = [x ** -0.3 for x in h]
    g = gci3(h, phi)
    assert 0 < g.R < 1                     # R < 1 alone would say monotone
    assert g.convergence == "divergent"


@pytest.mark.parametrize("h", UNEQUAL)
def test_unequal_ratio_threshold(h):
    r21, r32 = h[1] / h[0], h[2] / h[1]
    R_crit = math.log(r21) / math.log(r32)
    for factor, expected in ((0.999, "monotone"), (1.001, "divergent")):
        e32 = 0.01
        R = factor * R_crit
        g = gci3(h, [1.0, 1.0 + R * e32, 1.0 + (1.0 + R) * e32])
        assert g.convergence == expected, (factor, R_crit)


@pytest.mark.parametrize("h", UNEQUAL)
@pytest.mark.parametrize("p_true", [0.5, 1.0, 2.0])
def test_unequal_ratios_oscillatory(h, p_true):
    g = no_warnings(gci3, h, alternating(h, p_true))
    assert g.convergence == "oscillatory" and "oscillatory" in g.flags
    assert math.isclose(g.p, p_true, rel_tol=1e-9)
    g = no_warnings(gci3, h, alternating(h, -0.3))
    assert g.convergence == "oscillatory_divergent" and g.p is None


def test_absurd_order_from_nearly_coincident_meshes_stays_finite():
    """h2 and h3 0.05 % apart: the order that fits is in the thousands and r21^p overflows; the result takes the
    limits (phi_ext -> phi1) instead of turning into nan."""
    h = (1.0, 2.0, 2.001)
    g = no_warnings(gci3, h, [1.0, 1.1, 1.275], p_theory=2.0)
    assert g.convergence == "monotone" and g.p > 1000
    assert g.phi_ext == 1.0 and g.gci_fine == 0.0 and g.asymptotic_ratio is None
    assert math.isfinite(g.gci_medium) and math.isclose(g.gci_fine_conservative, 0.1, rel_tol=1e-12)
    assert {"refinement_ratio_below_1.3", "p_above_theory"} <= set(g.flags)


def test_random_unequal_ratios_never_raise_and_follow_the_criterion():
    rng = random.Random(20081)
    for _ in range(3000):
        x21, x32 = 1 + 10 ** rng.uniform(-2.5, 1), 1 + 10 ** rng.uniform(-2.5, 1)
        h = (1.0, x21, x21 * x32)
        phi = [1 + rng.uniform(-0.1, 0.1) for _ in range(3)]
        g = no_warnings(gci3, h, phi, p_theory=2.0)
        a, b = math.log(h[1] / h[0]), math.log(h[2] / h[1])
        R = (phi[1] - phi[0]) / (phi[2] - phi[1])
        if R > 0:
            expected = "monotone" if R < a / b else "divergent"
        else:
            expected = "oscillatory" if R > -1 else "oscillatory_divergent"
        assert g.convergence == expected, (h, phi)
        if expected in ("monotone", "oscillatory"):
            assert g.p > 0
            values = (g.phi_ext, g.e_ext21, g.gci_fine, g.gci_medium, g.gci_fine_conservative)
            assert all(math.isfinite(v) for v in values), (h, phi, values)
            if g.p * max(a, b) < 700:   # the order solves Celik's equation with both ratios
                if R > 0:
                    lhs = g.p * a + math.log(math.expm1(g.p * b) / math.expm1(g.p * a))
                else:
                    lhs = g.p * a + math.log((math.exp(g.p * b) + 1) / (math.exp(g.p * a) + 1))
                assert math.isclose(lhs, math.log(abs(1 / R)), rel_tol=1e-9, abs_tol=1e-9), (h, phi)
        else:
            assert g.p is None and g.flags[-1] == "divergent"


# --------------------------------------------------------------------------- the short-cut paths, unchanged


def test_exact_noise_floor_and_identical_paths_on_unequal_ratios():
    h = (1.0, 2.0, 3.0)
    g = gci3(h, [2.0, 2.0, 2.0])
    assert g.convergence == "exact" and g.gci_fine == 0.0
    g = gci3(h, [1.0, 1.0 + 1e-8, 1.0 + 3e-8])
    assert g.convergence == "converged" and "converged_within_noise" in g.flags and g.phi_ext == 1.0
    g = gci3(h, [1.0, 1.0, 1.5])
    assert g.convergence == "indeterminate" and "two_solutions_identical" in g.flags


def test_mesh_sizes_must_be_distinct():
    with pytest.raises(ValueError):
        gci3([1.0, 1.0, 2.0], [1.0, 1.1, 1.2])


# --------------------------------------------------------------------------- the order solve itself


def test_observed_order_brackets_the_root():
    # r32 > r21^2: Celik's fixed-point iteration diverged here (r32^p overflowed, then a math domain error)
    h = (1.0, 1.1, 1.1 * 1.6)
    phi = power_law(h, 2.0)
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    assert math.isclose(_observed_order(e21, e32, h[1] / h[0], h[2] / h[1], 1.0), 2.0, rel_tol=1e-9)
    # a diverging sequence has a negative root: no positive order
    phi = [x ** -0.3 for x in h]
    assert _observed_order(phi[1] - phi[0], phi[2] - phi[1], h[1] / h[0], h[2] / h[1], 1.0) is None
    # alternating (s = -1): the root is positive iff |e32| > |e21|, whatever the ratios
    phi = alternating(h, 1.5)
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    assert math.isclose(_observed_order(e21, e32, h[1] / h[0], h[2] / h[1], -1.0), 1.5, rel_tol=1e-9)
    assert _observed_order(1.0, -0.9, h[1] / h[0], h[2] / h[1], -1.0) is None
