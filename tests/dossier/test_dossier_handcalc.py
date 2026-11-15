"""The hand-calculation library of dossier.

A hand calculation is the check that finds a wrong finite-element model, so every formula of the library is
compared here with a number that does not come from the library: an exact solution of the repository
(feacore.problems.analytical), a numerical solution of the governing equation (scipy solve_bvp, quadrature, an
eigenvalue problem, the method of characteristics), a published value, or an identity. Then the engine
(evaluate, applies, compare) runs on the six specifications of the example project, with the numbers worked
out by hand in the test.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from scipy.integrate import dblquad, quad, solve_bvp
from scipy.linalg import eigh
from scipy.optimize import brentq

from dossier import handcalc, units
from dossier.handcalc import fluid, formula
from dossier.handcalc.base import HandCalcError
from dossier.model import (AnalysisRecord, Compare, HandCalcResult, HandCalcSpec, HandCalcWhen, LoadCase, LoadItem,
                           ResultValue, Value)
from dossier.project import make_id

approx = pytest.approx


def run(name: str, **values: float) -> dict[str, float]:
    """The SI numbers a model gives for SI inputs."""
    return handcalc.get(name).calculate(values).values


# Inputs for one run of every model (SI): used by the checks that go over the whole library.
SAMPLES: dict[str, dict[str, float]] = {
    "thick_cylinder_pressure": {"p": 15.75e6, "r_i": 0.022, "r_o": 0.028},
    "thin_cylinder_pressure": {"p": 1e6, "r": 0.1, "t": 0.005, "E": 2e11, "nu": 0.3},
    "sphere_pressure": {"p": 1e6, "r_i": 0.1, "r_o": 0.2},
    "circular_plate_pressure": {"p": 15.75e6, "a": 0.022, "t": 0.008, "nu": 0.29, "E": 193e9},
    "tube_bending_axial": {"M": 94.73, "F": 2091.04, "r_o": 0.012, "r_i": 0.008, "r_eval": 0.010},
    "beam_bending": {"M": -100.0, "Z": 1e-6},
    "bolt_circle_loads": {"F": 1000.0, "M": 90.0, "n": 6, "r_b": 0.032},
    "euler_buckling": {"E": 2e11, "I": 1e-8, "A": 1e-4, "L": 2.0, "K": 1.0, "Sy": 250e6},
    "fin_convective_tip": {"T_base": 90.0, "T_inf": 293.15, "h": 12.0, "k": 12.0, "r_o": 0.012, "r_i": 0.008,
                           "L": 0.078, "A_tip": 4476e-6},
    "thermal_stress_constrained": {"dT": 100.0, "E": 2e11, "alpha": 1.2e-5, "nu": 0.3},
    "hollow_cylinder_radial_gradient": {"dT": 50.0, "r_i": 0.1, "r_o": 0.12, "E": 2e11, "alpha": 1.2e-5, "nu": 0.3},
    "conduction_cylinder_wall": {"T_i": 400.0, "T_o": 300.0, "h_i": 1000.0, "h_o": 10.0, "r_i": 0.05, "r_o": 0.06,
                                 "k": 15.0},
    "dittus_boelter": {"mdot": 0.05, "d": 0.15, "mu": 208e-7, "k": 0.030, "cp": 1009.6, "n": 0.3},
    "pressure_drop_minor_loss": {"mdot": 5.3, "rho": 1141.0, "mu": 1.95e-4, "d": 0.025, "L": 0.15, "K": 0.2,
                                 "roughness": 3.2e-6},
    "rigid_column_surge": {"rho": 1141.0, "L": 1.5, "dv": 9.46, "t_close": 0.05, "a": 900.0},
    "joukowsky_surge": {"rho": 1000.0, "a": 1200.0, "dv": 1.0, "L": 100.0, "t_close": 0.05},
    "basquin_life": {"sigma_a": 100e6, "sigma_m": 97e6, "sigma_f": 1000e6, "b": -0.1, "Su": 485e6},
}
# Inputs outside the range of each theory: every warning text of the library.
OUTSIDE: dict[str, dict[str, float]] = {
    "thin_cylinder_pressure": {"p": 1e6, "r": 0.1, "t": 0.02, "E": 2e11, "nu": 0.3},
    "circular_plate_pressure": {"p": 400e6, "a": 0.022, "t": 0.001, "nu": 0.29, "E": 193e9},
    "tube_bending_axial": {"M": 94.73, "F": 2091.04, "r_o": 0.012, "r_i": 0.008, "r_eval": 0.020},
    "euler_buckling": {"E": 2e11, "I": 1e-8, "A": 1e-4, "L": 0.5, "K": 1.0, "Sy": 250e6},
    "fin_convective_tip": {"T_base": 90.0, "T_inf": 293.15, "h": 4000.0, "k": 12.0, "r_o": 0.012, "r_i": 0.008,
                           "L": 0.010, "A_tip": 0.0},
    "dittus_boelter": {"mdot": 0.001, "d": 0.15, "mu": 208e-7, "k": 0.030, "cp": 100.0, "n": 0.5},
    "pressure_drop_minor_loss": {"mdot": 0.01, "rho": 1000.0, "mu": 1e-3, "d": 0.004, "L": 0.01, "K": 0.0,
                                 "roughness": 4e-4},
    "rigid_column_surge": {"rho": 1141.0, "L": 1.5, "dv": 9.46, "t_close": 0.001, "a": 900.0},
    "joukowsky_surge": {"rho": 1000.0, "a": 1200.0, "dv": 1.0, "L": 100.0, "t_close": 2.0},
    "basquin_life": {"sigma_a": 400e6, "sigma_m": 97e6, "sigma_f": 1000e6, "b": -0.1, "Su": 485e6},
}


# =========================================================================== formulas as text


def test_formula_gives_the_number_and_the_latex_of_one_expression():
    v = {"a": 3.0, "b": 5.0, "c": 7.0, "x": 2.0, "r_i": 0.022, "r_o": 0.028, "p": 15.75e6, "M": -4.0, "Z": 2.0}
    cases = {                                               # precedence: what needs parentheses, and what does not
        "p * r_i / (r_o - r_i)": r"\frac{p\,r_{i}}{r_{o} - r_{i}}",
        "a - (b - c)": r"a - \left(b - c\right)",
        "a - b - c": "a - b - c",
        "a + (b - c)": "a + b - c",
        "a - (b + c)": r"a - \left(b + c\right)",
        "(a + b) * c": r"\left(a + b\right)\,c",
        "a * (b + c)": r"a\,\left(b + c\right)",
        "a - b * c": r"a - b\,c",
        "-(a + b)": r"-\left(a + b\right)",
        "-a * b": r"-a\,b",
        "a * -b": r"a\,\left(-b\right)",
        "a - -b": r"a - \left(-b\right)",
        "(a / b)**2": r"\left(\frac{a}{b}\right)^{2}",
        "a**(b + c)": "a^{b + c}",
        "(-a)**2": r"\left(-a\right)^{2}",
        "-a**2": "-a^{2}",
        "(a**b)**x": r"\left(a^{b}\right)^{x}",
        "x**0.5": r"\sqrt{x}",
        "sqrt(a**2 + b**2)": r"\sqrt{a^{2} + b^{2}}",
        "2 * 3 * x": r"2 \cdot 3\,x",
        "x * 2": r"x \cdot 2",
        "log(r_o / r_i)": r"\ln\left(\frac{r_{o}}{r_{i}}\right)",
        "log10(a) + exp(-x)": r"\log_{10}\left(a\right) + \exp\left(-x\right)",
        "cosh(x)**2": r"\left(\cosh\left(x\right)\right)^{2}",
        "abs(M) / Z": r"\frac{\left|M\right|}{Z}",
        "a / b / c": r"\frac{\frac{a}{b}}{c}",
        "a / (b / c)": r"\frac{a}{\frac{b}{c}}",
        "a / b * c": r"\frac{a}{b}\,c",
        "pi * (r_o**2 - r_i**2)": r"\pi\,\left(r_{o}^{2} - r_{i}^{2}\right)",
        "0.023 * a**0.8": r"0.023\,a^{0.8}",
    }
    env = {**v, "pi": math.pi, "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
           "cosh": math.cosh, "abs": abs}
    for expr, tex in cases.items():
        assert formula.latex(expr) == tex, expr
        assert formula.evaluate(expr, v) == approx(eval(expr, {"__builtins__": {}}, env), rel=1e-14), expr  # noqa: S307
    # a symbol of two tokens is not an atom: it gets parentheses as the base of a power
    assert formula.latex("dp**2 * rho", {"dp": r"\Delta p", "rho": r"\rho"}) == r"\left(\Delta p\right)^{2}\,\rho"
    assert formula.auto_symbol("sigma_a") == r"\sigma_{a}" and formula.auto_symbol("T_base") == r"T_{\mathrm{base}}"
    assert formula.names("p * r_i / (r_o - r_i) + sqrt(pi)") == ("p", "r_i", "r_o")
    for bad in ("__import__('os')", "a.real", "a if b else c", "[a]", "f(a)", "a < b", "'x'"):
        with pytest.raises(formula.FormulaError):
            formula.evaluate(bad, v)
    for bad, values in (("a / (b - b)", v), ("log(-a)", v), ("a + missing", v), ("sqrt(a - b)", v)):
        with pytest.raises(formula.FormulaError):
            formula.evaluate(bad, values)


def _all_runs():
    for name, values in SAMPLES.items():
        yield name, handcalc.get(name).calculate(values)
    for name, values in OUTSIDE.items():
        yield name, handcalc.get(name).calculate(values)


def test_every_model_is_complete_and_every_step_is_one_formula():
    assert set(SAMPLES) == set(handcalc.MODELS), "a model has no sample run in this test file"
    for name, m in handcalc.MODELS.items():
        assert m.name == name and m.title and m.assumptions and m.inputs and m.outputs
        assert re.search(r"\b(19|20)\d\d\b", m.reference), f"{name}: the reference has no year"
        for x in (*m.inputs, *m.outputs):
            assert x.description and units.Q(1.0, x.unit) is not None, (name, x.name)   # the SI unit parses
        for x in m.inputs:
            assert x.sign in ("positive", "nonnegative", "any")
    for name, calc in _all_runs():
        m = handcalc.get(name)
        assert all(math.isfinite(calc.values[o.name]) for o in m.outputs)
        assert len(calc.steps) >= len(m.outputs)
        for step in calc.steps:
            assert step.text and step.latex and step.result, (name, step)
            assert "=" in step.latex and "$" not in step.latex, (name, step.latex)
            assert len(re.split(r"(?<=[.!?])\s+", step.text.strip())) == 1, f"{name}: a step is one sentence: {step.text}"


def test_every_step_formula_is_latex_that_the_report_can_draw():
    mathtext = pytest.importorskip("matplotlib.mathtext")
    parser = mathtext.MathTextParser("path")
    seen = set()
    for name, calc in _all_runs():
        for step in calc.steps:
            if step.latex not in seen:
                seen.add(step.latex)
                parser.parse(f"${step.latex}$")            # raises ValueError when mathtext cannot read it
    assert len(seen) > 60


def test_every_theory_limit_gives_a_warning_with_the_numbers():
    for name, values in SAMPLES.items():
        expected = {"circular_plate_pressure": 1, "pressure_drop_minor_loss": 1}.get(name, 0)   # the example's own
        assert len(handcalc.get(name).calculate(values).warnings) == expected, name
    count = {"thin_cylinder_pressure": 1, "circular_plate_pressure": 2, "tube_bending_axial": 1, "euler_buckling": 1,
             "fin_convective_tip": 2, "dittus_boelter": 3, "pressure_drop_minor_loss": 3, "rigid_column_surge": 1,
             "joukowsky_surge": 1, "basquin_life": 2}
    for name, values in OUTSIDE.items():
        warnings = handcalc.get(name).calculate(values).warnings
        assert len(warnings) == count[name], (name, warnings)
        assert all(re.search(r"\d", w) for w in warnings), (name, warnings)       # a warning gives the number
    plate = handcalc.get("circular_plate_pressure")
    thick = plate.calculate({"p": 1e6, "a": 0.01, "t": 0.008, "nu": 0.3, "E": 2e11}).warnings
    assert len(thick) == 1 and "thick" in thick[0] and "2.5 times" in thick[0]
    assert plate.calculate({"p": 1e3, "a": 0.1, "t": 0.01, "nu": 0.3, "E": 2e11}).warnings == []
    no_speed = handcalc.get("rigid_column_surge").calculate({"rho": 1000.0, "L": 10.0, "dv": 1.0, "t_close": 1.0})
    assert len(no_speed.warnings) == 1 and "no wave speed" in no_speed.warnings[0]
    assert len(handcalc.get("joukowsky_surge").calculate({"rho": 1000.0, "a": 1200.0, "dv": 1.0}).warnings) == 1


# --------------------------------------------------------------------------- ASD-STE100 rules of CONTRACTS.md

_MODALS = {"should", "shall", "may", "might", "could", "would", "ought"}
_NOT_STE = ("etc.", "e.g.", "i.e.", "and/or", "therefore", "due to", "prior to", "has been", "have been")
_ING_NOUNS = {"during", "nothing", "ring"}
_PASSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+(not\s+)?(\w+ly\s+)?(\w+ed|given|taken|made|shown|known|held)\b")
_NOT_PARTICIPLES = {"fixed", "closed", "developed", "need"}        # adjectives of state: "a fixed end"


def ste_problems(text: str, words_max: int = 25) -> list[str]:
    """The explicit rules of dossier/CONTRACTS.md, "Text": short sentences, active voice, no "-ing" verb
    forms, no forbidden modal verbs, no contractions, no Latin abbreviations."""
    out = []
    body = re.sub(r'"[^"]*"|\b[\w.]+\([^)]*\)', "X", text)                   # quoted text and code are not prose
    body = re.sub(r"\b[A-Z][A-Z0-9]*(-[A-Z0-9]+)+\b", "X", body)              # record ids are names
    for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        low = [w.lower() for w in words]
        if len(words) > words_max:
            out.append(f"{len(words)} words: {sentence}")
        out += [f"modal verb {w!r}: {sentence}" for w in low if w in _MODALS]
        out += [f"contraction {w!r}: {sentence}" for w in words if "'" in w and not w.endswith("'s")]
        out += [f"-ing form {w!r}: {sentence}" for w in low if w.endswith("ing") and len(w) > 4 and w not in _ING_NOUNS]
        m = _PASSIVE.search(sentence)
        if m and m.group(4) not in _NOT_PARTICIPLES:
            out.append(f"passive voice {m.group(0)!r}: {sentence}")
    out += [f"not STE {w!r}: {text}" for w in _NOT_STE if w in text.lower()]
    return out


def test_text_of_the_models_follows_the_ste_rules():
    problems = []
    for name, m in handcalc.MODELS.items():
        for text in (*m.assumptions, m.title, *(x.description for x in (*m.inputs, *m.outputs))):
            problems += [f"{name}: {p}" for p in ste_problems(text)]
    for name, calc in _all_runs():
        for text in (*(s.text for s in calc.steps), *calc.warnings):
            problems += [f"{name}: {p}" for p in ste_problems(text)]
    assert not problems, "\n".join(sorted(set(problems)))
    assert ste_problems("The stress is calculated and may be high.")          # the check itself finds problems


# =========================================================================== pressure


def test_thick_cylinder_is_the_lame_solution_of_feacore():
    from feacore.problems.analytical._common import derived_field
    from feacore.problems.analytical.lame import LameCylinderAxisym

    a, b, p = 0.022, 0.028, 15.75e6
    v = run("thick_cylinder_pressure", p=p, r_i=a, r_o=b)
    lame = LameCylinderAxisym(id="hc_ref_cylinder", axial="free", a=a, b=b, p=p, E=193e9, nu=0.29)
    _, srr, stt, _ = lame._radial(np.array([a, b]))
    assert v["hoop_inner"] == approx(stt[0], rel=1e-12) and v["hoop_outer"] == approx(stt[1], rel=1e-12)
    assert v["radial_inner"] == approx(srr[0], rel=1e-12) == approx(-p)
    # the mean hoop stress is the integral of the exact hoop stress over the wall: equilibrium of a half cylinder
    mean = quad(lambda r: float(lame._radial(np.array([r]))[2][0]), a, b)[0] / (b - a)
    assert v["hoop_mean"] == approx(mean, rel=1e-10) == approx(p * a / (b - a), rel=1e-12) == approx(57.75e6)
    # closed ends: the wall carries the pressure force on the bore area
    assert v["axial_closed"] * math.pi * (b**2 - a**2) == approx(p * math.pi * a**2, rel=1e-12)
    # von Mises from the stress tensor (feacore's stress measures), and its closed form sqrt(3) p b^2 / (b^2 - a^2)
    vm = derived_field("von_mises", {"sxx": np.array([srr[0]]), "syy": np.array([stt[0]]),
                                     "szz": np.array([v["axial_closed"]]), "sxy": np.zeros(1)}, "3d")[0]
    assert v["von_mises_inner"] == approx(vm, rel=1e-12) == approx(math.sqrt(3) * p * b**2 / (b**2 - a**2), rel=1e-12)
    # the same pattern as feacore's exact von Mises of the open cylinder, K sqrt(3 b^4/a^4 + 1): closed ends drop the 1
    K = p * a**2 / (b**2 - a**2)
    assert lame.vm_max_exact() == approx(K * math.sqrt(3 * b**4 / a**4 + 1), rel=1e-12)
    assert v["von_mises_inner"] == approx(K * math.sqrt(3 * b**4 / a**4), rel=1e-12)
    with pytest.raises(HandCalcError, match="outer radius"):
        run("thick_cylinder_pressure", p=p, r_i=b, r_o=a)
    with pytest.raises(HandCalcError, match="more than zero"):
        run("thick_cylinder_pressure", p=p, r_i=-a, r_o=b)


def test_thin_cylinder_is_the_limit_of_the_lame_solution():
    from feacore.problems.analytical.lame import LameCylinderAxisym

    p, R, E, nu = 2e6, 0.5, 2e11, 0.3
    errors = []
    for ratio in (100.0, 1000.0):
        t = R / ratio
        a, b = R - t / 2, R + t / 2
        v = run("thin_cylinder_pressure", p=p, r=R, t=t, E=E, nu=nu)
        lame = LameCylinderAxisym(id="hc_ref_thin", axial="free", a=a, b=b, p=p, E=E, nu=nu)
        ur, _, stt, _ = lame._radial(np.array([a, R]))
        K = p * a**2 / (b**2 - a**2)
        # with the mean radius, p R / t is the largest (inner) Lamé hoop stress to second order in t / R
        assert v["hoop"] == approx(stt[0], rel=1.0 / ratio**2)
        assert v["axial_closed"] == approx(K, rel=1.5 / ratio)                     # first order in t / R
        vm_lame = math.sqrt(3) * p * b**2 / (b**2 - a**2)                          # closed ends, inner surface
        assert v["von_mises_closed"] == approx(vm_lame, rel=2.0 / ratio)
        # radial growth: the open-end Lamé displacement of feacore plus the Poisson effect of the axial stress
        growth = ur[1] - nu * K * R / E
        assert v["radial_growth_closed"] == approx(growth, rel=1.5 / ratio)
        assert v["radial_growth_closed"] == approx(p * R**2 / (E * t) * (1 - nu / 2), rel=1e-12)   # Roark 13.1, 1c
        errors.append(abs(v["axial_closed"] / K - 1))
    assert errors[1] == approx(errors[0] / 10, rel=0.05)                           # the error goes with t / R


def test_sphere_is_the_lame_solution_of_feacore():
    from feacore.problems.analytical.lame import LameSphereAxisym

    a, b, p = 0.1, 0.16, 3e6
    v = run("sphere_pressure", p=p, r_i=a, r_o=b)
    lame = LameSphereAxisym(id="hc_ref_sphere", a=a, b=b, p=p)
    _, srr, stt = lame._radial(np.array([a, b]))
    hoop = lambda r: float(lame._radial(np.array([r]))[2][0])  # noqa: E731
    assert v["hoop_inner"] == approx(stt[0], rel=1e-12) and v["hoop_outer"] == approx(stt[1], rel=1e-12)
    assert v["radial_inner"] == approx(srr[0], rel=1e-12)
    assert v["hoop_mean"] == approx(quad(hoop, a, b)[0] / (b - a), rel=1e-10)        # mean along a radial line
    # the equilibrium of a half sphere: the hoop stress on the cut ring carries the pressure on the bore circle
    force = quad(lambda r: hoop(r) * 2 * math.pi * r, a, b)[0]
    assert force == approx(p * math.pi * a**2, rel=1e-10)
    assert v["hoop_mean_area"] == approx(force / (math.pi * (b**2 - a**2)), rel=1e-10)
    assert v["von_mises_inner"] == approx(1.5 * lame._K() * b**3 / a**3, rel=1e-12)  # feacore's vm_max reference
    thin = run("sphere_pressure", p=p, r_i=1.0, r_o=1.001)                           # thin wall: p r / (2 t)
    for key in ("hoop_inner", "hoop_outer", "hoop_mean", "hoop_mean_area"):
        assert thin[key] == approx(p * 1.0005 / (2 * 0.001), rel=2e-3)


# =========================================================================== plates


def _plate_numerical(p, a, t, nu, E, edge):
    """Centre deflection and surface stresses of a circular plate from the axisymmetric plate equation,
    d/dr[(1/r) d/dr(r dw/dr)] = p r / (2 D), solved with solve_bvp. With dw/dr = r g and h = dg/dr the system
    is w' = r g, g' = h, h' = -3 h / r + p / (2 D), regular at r = 0 with h(0) = 0."""
    D = E * t**3 / (12 * (1 - nu**2))

    def rhs(r, y):
        return np.vstack([r * y[1], y[2], np.full_like(r, p / (2 * D))])

    def bc(y0, ya):
        edge_bc = ya[1] if edge == "clamped" else (1 + nu) * ya[1] + a * ya[2]      # slope = 0, or M_r = 0
        return np.array([y0[2], ya[0], edge_bc])

    r = np.linspace(0.0, a, 41)
    sol = solve_bvp(rhs, bc, r, np.zeros((3, r.size)), S=np.array([[0, 0, 0], [0, 0, 0], [0, 0, -3.0]]), tol=1e-10)
    assert sol.success, sol.message
    g0, ga, ha = sol.y[1, 0], sol.y[1, -1], sol.y[2, -1]
    m_centre = D * (1 + nu) * g0                     # M = -D (w'' + nu w'/r), and w'' = w'/r = g at the centre
    m_edge = D * (ga + a * ha + nu * ga)
    return abs(sol.y[0, 0]), 6 * abs(m_centre) / t**2, 6 * abs(m_edge) / t**2


def test_circular_plate_against_roark_coefficients_and_a_numerical_solution():
    # Roark's Formulas for Stress and Strain, 7th ed., Table 11.2, cases 10a and 10b, load over the whole plate,
    # nu = 0.3: y_c = K_y p a^4 / D and M = K_M p a^2 with
    #   fixed edge              K_y = -0.01563   K_M(centre) = 0.08125   K_M(edge, radial) = -0.125
    #   simply supported edge   K_y = -0.06370   K_M(centre) = 0.20625
    p, a, t, nu, E = 0.4e6, 0.15, 0.006, 0.3, 200e9
    D = E * t**3 / (12 * (1 - nu**2))
    v = run("circular_plate_pressure", p=p, a=a, t=t, nu=nu, E=E)
    assert v["D"] == approx(D)
    assert v["deflection_center_clamped"] * D / (p * a**4) == approx(0.01563, abs=6e-6)       # 1/64, five decimals
    assert v["sigma_center_clamped"] * t**2 / (6 * p * a**2) == approx(0.08125, abs=5e-6)
    assert v["sigma_edge_clamped"] * t**2 / (6 * p * a**2) == approx(0.125, abs=5e-6)
    assert v["deflection_center_simply_supported"] * D / (p * a**4) == approx(0.06370, abs=5e-6)
    assert v["sigma_center_simply_supported"] * t**2 / (6 * p * a**2) == approx(0.20625, abs=5e-6)
    # the handbook forms for nu = 0.3: 0.75 and 1.238 p a^2 / t^2, 0.171 and 0.696 p a^4 / (E t^3)
    assert v["sigma_edge_clamped"] == approx(0.75 * p * a**2 / t**2)
    assert v["sigma_center_simply_supported"] == approx(1.238 * p * a**2 / t**2, rel=5e-4)
    assert v["deflection_center_clamped"] == approx(0.171 * p * a**4 / (E * t**3), rel=3e-3)
    assert v["deflection_center_simply_supported"] == approx(0.696 * p * a**4 / (E * t**3), rel=1e-3)
    # Timoshenko, Theory of Plates and Shells, art. 16: for nu = 0.3 the simply supported plate deflects
    # "about four times" as much as the clamped plate; the centre stresses have the ratio (3 + nu) / (1 + nu)
    assert v["deflection_center_simply_supported"] / v["deflection_center_clamped"] == approx(4.0, rel=0.03)
    assert v["sigma_center_simply_supported"] / v["sigma_center_clamped"] == approx(3.3 / 1.3)
    # a numerical solution of the plate equation, for the example's floor (nu = 0.29) and for this plate
    for args in ((15.75e6, 0.022, 0.008, 0.29, 193e9), (p, a, t, nu, E)):
        v = run("circular_plate_pressure", **dict(zip(("p", "a", "t", "nu", "E"), args)))
        w, s_centre, s_edge = _plate_numerical(*args, "clamped")
        assert v["deflection_center_clamped"] == approx(w, rel=1e-7)
        assert v["sigma_center_clamped"] == approx(s_centre, rel=1e-7) and v["sigma_edge_clamped"] == approx(s_edge, rel=1e-7)
        w, s_centre, s_edge = _plate_numerical(*args, "simply supported")
        assert v["deflection_center_simply_supported"] == approx(w, rel=1e-7)
        assert v["sigma_center_simply_supported"] == approx(s_centre, rel=1e-7) and s_edge < 1e-6 * s_centre


# =========================================================================== beams


def test_tube_stress_carries_the_force_and_the_moment():
    M, F, ro, ri, re_ = 94.73, 2091.04, 0.012, 0.008, 0.010
    v = run("tube_bending_axial", M=M, F=F, r_o=ro, r_i=ri, r_eval=re_)
    # area and second moment of area of the ring by numerical integration in polar coordinates
    area = dblquad(lambda r, th: r, 0, 2 * math.pi, ri, ro)[0]
    second = dblquad(lambda r, th: (r * math.sin(th))**2 * r, 0, 2 * math.pi, ri, ro)[0]
    assert v["A"] == approx(area, rel=1e-10) and v["I"] == approx(second, rel=1e-10)
    assert v["I"] == approx(13069.0e-12, rel=1e-4)                              # pi/4 (12^4 - 8^4) mm^4, by hand
    # the linear stress field of the model, integrated over the ring, gives back the force and the moment
    sigma = lambda r, th: v["sigma_axial"] + v["sigma_bending_outer"] * r * math.sin(th) / ro  # noqa: E731
    assert dblquad(lambda r, th: sigma(r, th) * r, 0, 2 * math.pi, ri, ro)[0] == approx(F, rel=1e-9)
    assert dblquad(lambda r, th: sigma(r, th) * r * math.sin(th) * r, 0, 2 * math.pi, ri, ro)[0] == approx(M, rel=1e-9)
    assert v["sigma_bending"] == approx(v["sigma_bending_outer"] * re_ / ro)
    assert v["sigma_tension"] == approx(sigma(re_, math.pi / 2)) and v["sigma_compression"] == approx(sigma(re_, -math.pi / 2))
    assert v["sigma_tension"] == approx(F / area + M * re_ / second)
    # a thin ring: I = pi r^3 t, A = 2 pi r t
    thin = run("tube_bending_axial", M=1.0, F=1.0, r_o=1.0005, r_i=0.9995, r_eval=1.0)
    assert thin["I"] == approx(math.pi * 1e-3, rel=1e-6) and thin["A"] == approx(2 * math.pi * 1e-3, rel=1e-9)
    # the sign of the moment does not change which side is the tension side
    assert run("tube_bending_axial", M=-M, F=F, r_o=ro, r_i=ri, r_eval=re_)["sigma_tension"] == approx(v["sigma_tension"])
    solid = run("tube_bending_axial", M=10.0, F=0.0, r_o=0.01, r_i=0.0, r_eval=0.01)
    assert solid["sigma_bending"] == approx(32 * 10.0 / (math.pi * 0.02**3))    # 32 M / (pi d^3) of a solid bar


def test_beam_bending_with_the_section_modulus_of_a_rectangle():
    b, h, M = 0.02, 0.05, 300.0
    v = run("beam_bending", M=M, Z=b * h**2 / 6)
    # the linear stress field with that outer-fibre stress carries the moment
    moment = dblquad(lambda y, x: v["sigma_bending"] * (y / (h / 2)) * y, 0, b, -h / 2, h / 2)[0]
    assert moment == approx(M, rel=1e-10) and v["sigma_bending"] == approx(6 * M / (b * h**2))
    assert run("beam_bending", M=-M, Z=b * h**2 / 6)["sigma_bending"] == approx(v["sigma_bending"])


def _rigid_flange_bolt_forces(F, Mx, My, n, r, phase):
    """Bolt forces of a rigid flange on n equal springs on a circle: solve the three equilibrium equations
    for the axial displacement and the two rotations of the flange."""
    th = phase + 2 * math.pi * np.arange(n) / n
    x, y = r * np.cos(th), r * np.sin(th)
    shape = np.vstack([np.ones(n), y, -x])                  # force of bolt i = stiffness (w + rx y_i - ry x_i)
    dof = np.linalg.solve(shape @ shape.T, np.array([F, Mx, My]))
    return shape.T @ dof


def test_bolt_circle_against_the_equilibrium_of_a_rigid_flange():
    F, M, r = 1000.0, 90.0, 0.032
    rng = np.random.default_rng(1)
    for n in range(3, 13):
        v = run("bolt_circle_loads", F=F, M=M, n=n, r_b=r)
        assert v["force_axial"] == approx(F / n)
        # the moment in the plane of a bolt: that bolt is at the largest distance, and the formula is exact
        forces = _rigid_flange_bolt_forces(F, 0.0, -M, n, r, phase=0.0)
        assert forces.sum() == approx(F) and forces.max() == approx(v["force_max"], rel=1e-12)
        opposite = _rigid_flange_bolt_forces(F, 0.0, M, n, r, phase=0.0)
        assert opposite.min() == approx(v["force_min"], rel=1e-12)
        # any other direction of the moment gives a smaller largest force, and never more than the formula
        for phase in rng.uniform(0, 2 * math.pi, 20):
            forces = _rigid_flange_bolt_forces(F, 0.0, -M, n, r, phase)
            assert forces.max() <= v["force_max"] * (1 + 1e-12) and forces.min() >= v["force_min"] * (1 + 1e-12) - 1e-9
        # Brownell and Young: 4 M / (n d) with the bolt circle diameter d
        assert v["force_moment"] == approx(4 * M / (n * 2 * r))
    for bad in (2, 2.5, 6.5):
        with pytest.raises(HandCalcError, match="whole number of 3 or more"):
            run("bolt_circle_loads", F=F, M=M, n=bad, r_b=r)


def _column_critical_load(E, I, L, left, right, n=60):
    """Lowest critical load of a column from a finite-element model (Hermite beam elements: bending stiffness
    against geometric stiffness)."""
    le = L / n
    ke = E * I / le**3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                                   [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le**2, -6 * le, 4 * le**2]])
    kg = 1 / (30 * le) * np.array([[36, 3 * le, -36, 3 * le], [3 * le, 4 * le**2, -3 * le, -le**2],
                                   [-36, -3 * le, 36, -3 * le], [3 * le, -le**2, -3 * le, 4 * le**2]])
    size = 2 * (n + 1)
    K, G = np.zeros((size, size)), np.zeros((size, size))
    for e in range(n):
        idx = np.arange(2 * e, 2 * e + 4)
        K[np.ix_(idx, idx)] += ke
        G[np.ix_(idx, idx)] += kg
    held = [2 * node + d for end, node in ((left, 0), (right, n))
            for d in ((0, 1) if end == "fixed" else (0,) if end == "pinned" else ())]
    free = [i for i in range(size) if i not in held]
    return float(eigh(K[np.ix_(free, free)], G[np.ix_(free, free)], eigvals_only=True)[0])


def test_euler_load_against_the_eigenvalue_of_a_finite_element_column():
    E, I, A, L = 2e11, 1e-8, 1e-4, 1.2
    fixed_pinned = math.pi / brentq(lambda x: math.tan(x) - x, 4.4, 4.6)           # 0.6992: root of tan(kL) = kL
    assert fixed_pinned == approx(0.7, abs=1e-3)
    for (left, right), factor in ((("pinned", "pinned"), 1.0), (("fixed", "free"), 2.0), (("fixed", "fixed"), 0.5),
                                  (("fixed", "pinned"), fixed_pinned)):
        v = run("euler_buckling", E=E, I=I, A=A, L=L, K=factor, Sy=250e6)
        assert v["P_cr"] == approx(_column_critical_load(E, I, L, left, right), rel=1e-6), (left, right)
        assert v["sigma_cr"] == approx(v["P_cr"] / A)
        assert v["slenderness"] == approx(factor * L / math.sqrt(I / A))
        assert v["sigma_cr"] == approx(math.pi**2 * E / v["slenderness"]**2)       # the other form of the formula
    # the limit of the formula (Shigley 4-45): the Euler stress is one half of the yield strength
    limit = math.sqrt(2 * math.pi**2 * E / 250e6) * math.sqrt(I / A)               # effective length at the limit
    euler = handcalc.get("euler_buckling")
    assert euler.calculate({"E": E, "I": I, "A": A, "L": 1.01 * limit, "K": 1.0, "Sy": 250e6}).warnings == []
    assert len(euler.calculate({"E": E, "I": I, "A": A, "L": 0.99 * limit, "K": 1.0, "Sy": 250e6}).warnings) == 1


# =========================================================================== thermal


def _fin_numerical(T_base, T_inf, h, k, r_o, r_i, L, A_tip):
    """Tip temperature and base heat flow from the fin equation theta'' = (h P / k A_c) theta with
    theta(0) = T_base - T_inf and -k A_c theta'(L) = h A_tip theta(L), solved with solve_bvp."""
    area, perimeter = math.pi * (r_o**2 - r_i**2), 2 * math.pi * r_o
    m2 = h * perimeter / (k * area)
    x = np.linspace(0.0, L, 41)
    sol = solve_bvp(lambda s, y: np.vstack([y[1], m2 * y[0]]),
                    lambda y0, yl: np.array([y0[0] - (T_base - T_inf), k * area * yl[1] + h * A_tip * yl[0]]),
                    x, np.zeros((2, x.size)), tol=1e-8, max_nodes=100000)
    assert sol.success, sol.message
    surface_loss = quad(lambda s: h * perimeter * float(sol.sol(s)[0]), 0.0, L)[0] + h * A_tip * sol.y[0, -1]
    return T_inf + sol.y[0, -1], -k * area * sol.y[1, 0], surface_loss


def test_fin_against_a_numerical_solution_of_the_fin_equation():
    cases = [
        dict(T_base=90.0, T_inf=293.15, h=12.0, k=12.0, r_o=0.012, r_i=0.008, L=0.078, A_tip=4476e-6),   # the example
        dict(T_base=400.0, T_inf=300.0, h=60.0, k=180.0, r_o=0.005, r_i=0.0, L=0.12, A_tip=0.0),
        dict(T_base=350.0, T_inf=290.0, h=25.0, k=45.0, r_o=0.02, r_i=0.017, L=0.3, A_tip=2e-2),
    ]
    for case in cases:
        v = run("fin_convective_tip", **case)
        t_tip, q_base, loss = _fin_numerical(**case)
        assert v["T_tip"] == approx(t_tip, rel=1e-8)
        assert v["heat_flow_base"] == approx(q_base, rel=1e-6)
        assert v["heat_flow_base"] == approx(loss, rel=1e-6)          # energy balance: base heat flow = surface loss
    # the example by hand: m^2 = h 2 pi r_o / (k pi (r_o^2 - r_i^2)) = 2 * 0.012 / (0.012^2 - 0.008^2) = 300 1/m^2
    v = run("fin_convective_tip", **cases[0])
    assert v["m"] == approx(math.sqrt(300.0)) and v["mL"] == approx(math.sqrt(300.0) * 0.078)
    beta = 12.0 * 4476e-6 / (math.sqrt(300.0) * 12.0 * math.pi * 80e-6)
    assert v["T_tip"] == approx(293.15 - 203.15 / (math.cosh(v["mL"]) + beta * math.sinh(v["mL"]))) == approx(241.22, abs=0.01)
    assert v["heat_flow_base"] < 0                                    # the base is cold: heat flows into it
    # Incropera, Table 3.4: case B (adiabatic tip) is A_tip = 0; case A (tip of area A_c) has h / (m k)
    case = dict(T_base=400.0, T_inf=300.0, h=60.0, k=180.0, r_o=0.005, r_i=0.0, L=0.12)
    area = math.pi * 0.005**2
    m = math.sqrt(60.0 * 2 * math.pi * 0.005 / (180.0 * area))
    big_m = math.sqrt(60.0 * 2 * math.pi * 0.005 * 180.0 * area) * 100.0
    adiabatic = run("fin_convective_tip", **case, A_tip=0.0)
    assert adiabatic["T_tip"] == approx(300.0 + 100.0 / math.cosh(m * 0.12))
    assert adiabatic["heat_flow_base"] == approx(big_m * math.tanh(m * 0.12))
    tip = run("fin_convective_tip", **case, A_tip=area)
    hmk = 60.0 / (m * 180.0)
    c, s = math.cosh(m * 0.12), math.sinh(m * 0.12)
    assert tip["T_tip"] == approx(300.0 + 100.0 / (c + hmk * s))
    assert tip["heat_flow_base"] == approx(big_m * (s + hmk * c) / (c + hmk * s))
    # a long fin: the base heat flow of the infinite fin (case D), sqrt(h P k A_c) theta_b
    assert run("fin_convective_tip", **{**case, "L": 3.0}, A_tip=area)["heat_flow_base"] == approx(big_m, rel=1e-9)


def test_constrained_thermal_stress_against_hookes_law():
    E, alpha, nu, dT = 193e9, 13e-6, 0.29, -203.15
    v = run("thermal_stress_constrained", dT=dT, E=E, alpha=alpha, nu=nu)
    compliance = np.array([[1, -nu, -nu], [-nu, 1, -nu], [-nu, -nu, 1]]) / E
    for held, key in (([0], "sigma_uniaxial"), ([0, 1], "sigma_biaxial")):
        # total strain = compliance * stress + alpha dT is zero in the held directions; the other stresses are zero
        sub = compliance[np.ix_(held, held)]
        stress = np.linalg.solve(sub, -alpha * dT * np.ones(len(held)))
        assert v[key] == approx(stress[0], rel=1e-12) and np.allclose(stress, stress[0])
    assert v["sigma_uniaxial"] == approx(193e9 * 13e-6 * 203.15) and v["sigma_uniaxial"] > 0     # cooling: tension
    assert v["sigma_biaxial"] == approx(v["sigma_uniaxial"] / 0.71)
    with pytest.raises(HandCalcError, match="0.5 or less"):
        run("thermal_stress_constrained", dT=dT, E=E, alpha=alpha, nu=0.6)


def _cylinder_thermal_numerical(dT, a, b, E, alpha, nu):
    """Stresses (radial, hoop, axial) at both surfaces of a long hollow cylinder with the steady temperature
    T(r) = dT ln(b / r) / ln(b / a), from the equilibrium equation in terms of the radial displacement u,
    u'' + u'/r - u/r^2 = (1 + nu) / (1 - nu) alpha T', with free surfaces and a uniform axial strain e0 that
    makes the axial force zero (the unknown parameter of solve_bvp)."""
    log = math.log(b / a)
    c = E / ((1 + nu) * (1 - 2 * nu))

    def stresses(r, u, du, e0):
        thermal = (1 + nu) * alpha * dT * np.log(b / r) / log
        return (c * ((1 - nu) * du + nu * u / r + nu * e0 - thermal),
                c * ((1 - nu) * u / r + nu * du + nu * e0 - thermal),
                c * (nu * (du + u / r) + (1 - nu) * e0 - thermal))

    def rhs(r, y, prm):
        axial = stresses(r, y[0], y[1], prm[0])[2]
        return np.vstack([y[1], -y[1] / r + y[0] / r**2 - (1 + nu) / (1 - nu) * alpha * dT / (r * log),
                          axial * r / (E * a * alpha * abs(dT))])            # y[2]: the axial force up to r, scaled

    def bc(ya, yb, prm):
        return np.array([stresses(a, ya[0], ya[1], prm[0])[0] / E, stresses(b, yb[0], yb[1], prm[0])[0] / E,
                         ya[2], yb[2]])

    r = np.linspace(a, b, 81)
    guess = np.zeros((3, r.size))
    guess[0] = alpha * dT * r
    sol = solve_bvp(rhs, bc, r, guess, p=[alpha * dT], tol=1e-11, max_nodes=20000)
    assert sol.success, sol.message
    return (stresses(a, sol.y[0, 0], sol.y[1, 0], sol.p[0]), stresses(b, sol.y[0, -1], sol.y[1, -1], sol.p[0]))


def test_thermal_stress_of_the_hollow_cylinder_against_two_numerical_solutions():
    for dT, a, b, E, alpha, nu in ((50.0, 0.10, 0.12, 200e9, 1.2e-5, 0.3), (-203.0, 0.0125, 0.020, 193e9, 13e-6, 0.29),
                                   (120.0, 0.02, 0.08, 70e9, 2.3e-5, 0.33)):
        v = run("hollow_cylinder_radial_gradient", dT=dT, r_i=a, r_o=b, E=E, alpha=alpha, nu=nu)
        # 1. the displacement form of equilibrium, solved numerically (plane sections, zero axial force)
        inner, outer = _cylinder_thermal_numerical(dT, a, b, E, alpha, nu)
        scale = E * alpha * abs(dT)
        assert abs(inner[0]) < 1e-9 * scale and abs(outer[0]) < 1e-9 * scale        # free surfaces
        assert v["hoop_inner"] == approx(inner[1], rel=1e-8) and v["hoop_outer"] == approx(outer[1], rel=1e-8)
        assert v["axial_inner"] == approx(inner[2], rel=1e-8) and v["axial_outer"] == approx(outer[2], rel=1e-8)
        # 2. Timoshenko and Goodier's integral formula for any radial temperature, by quadrature:
        #    hoop(r) = alpha E / ((1 - nu) r^2) [ (r^2 + a^2) / (b^2 - a^2) int_a^b T r dr + int_a^r T r dr - T r^2 ]
        T = lambda r: dT * math.log(b / r) / math.log(b / a)  # noqa: E731
        whole = quad(lambda r: T(r) * r, a, b)[0]
        hoop = lambda r: alpha * E / ((1 - nu) * r**2) * ((r**2 + a**2) / (b**2 - a**2) * whole  # noqa: E731
                                                          + quad(lambda s: T(s) * s, a, r)[0] - T(r) * r**2)
        assert v["hoop_inner"] == approx(hoop(a), rel=1e-9) and v["hoop_outer"] == approx(hoop(b), rel=1e-9)
        assert (v["hoop_inner"] < 0) == (dT > 0) and (v["hoop_outer"] > 0) == (dT > 0)     # the hot side is in compression
    # a thin wall: -/+ E alpha dT / (2 (1 - nu))  (Roark, section 16.6: linear gradient through a thin wall)
    thin = run("hollow_cylinder_radial_gradient", dT=40.0, r_i=1.0, r_o=1.0004, E=200e9, alpha=1.2e-5, nu=0.3)
    limit = 200e9 * 1.2e-5 * 40.0 / (2 * 0.7)
    assert thin["hoop_inner"] == approx(-limit, rel=1e-3) and thin["hoop_outer"] == approx(limit, rel=1e-3)
    assert run("hollow_cylinder_radial_gradient", dT=0.0, r_i=0.1, r_o=0.12, E=2e11, alpha=1e-5, nu=0.3)["hoop_inner"] == 0.0


def test_heat_flow_through_a_cylinder_wall_against_a_numerical_solution():
    case = dict(T_i=90.0, T_o=293.15, h_i=15572.0, h_o=12.0, r_i=0.0125, r_o=0.020, k=12.0)
    for case in (case, dict(T_i=400.0, T_o=300.0, h_i=1000.0, h_o=10.0, r_i=0.05, r_o=0.06, k=15.0)):
        v = run("conduction_cylinder_wall", **case)
        k, ri, ro = case["k"], case["r_i"], case["r_o"]
        # (r T')' = 0 with a fluid film on each surface: y = [T, r T']
        r = np.linspace(ri, ro, 21)
        sol = solve_bvp(lambda s, y: np.vstack([y[1] / s, np.zeros_like(s)]),
                        lambda ya, yb: np.array([-k * ya[1] / ri - case["h_i"] * (case["T_i"] - ya[0]),
                                                 -k * yb[1] / ro - case["h_o"] * (yb[0] - case["T_o"])]),
                        r, np.vstack([np.full_like(r, 0.5 * (case["T_i"] + case["T_o"])), np.zeros_like(r)]), tol=1e-10)
        assert sol.success
        assert v["heat_flow_per_length"] == approx(-2 * math.pi * k * sol.y[1, 0], rel=1e-8)
        assert v["T_wall_inner"] == approx(sol.y[0, 0], rel=1e-10) and v["T_wall_outer"] == approx(sol.y[0, -1], rel=1e-10)
        assert v["dT_wall"] == approx(sol.y[0, 0] - sol.y[0, -1], rel=1e-6)
    v = run("conduction_cylinder_wall", **case)
    assert v["heat_flow_per_length"] > 0 and v["T_wall_inner"] > v["T_wall_outer"]      # from the hot fluid outward


def test_dittus_boelter_against_a_textbook_example_and_the_example_project(example_ctx):
    # Incropera et al., Fundamentals of Heat and Mass Transfer, 6th ed., example 8.5: hot air, 0.050 kg/s, in a
    # duct of diameter 0.15 m; mu = 208e-7 N s/m^2, k = 0.030 W/(m K), Pr = 0.70; the air cools (n = 0.3).
    # The book gives Re = 20 404, Nu = 57.9 and h = 11.6 W/(m^2 K).
    mu, k, pr = 208e-7, 0.030, 0.70
    v = run("dittus_boelter", mdot=0.050, d=0.15, mu=mu, k=k, cp=pr * k / mu, n=0.3)
    assert v["reynolds"] == approx(20404, abs=0.5) and v["prandtl"] == approx(0.70)
    assert v["nusselt"] == approx(57.9, abs=0.05) and v["h"] == approx(11.6, abs=0.05)
    assert handcalc.get("dittus_boelter").calculate(dict(mdot=0.050, d=0.15, mu=mu, k=k, cp=pr * k / mu, n=0.3)).warnings == []
    # the heat transfer coefficient of the flowing LOX in the example's system.yaml (its own expression)
    p = {n: example_ctx.params[n].si for n in ("mdot_ox", "d_bore", "mu_lox", "k_lox", "cp_lox", "re_lox", "pr_lox", "h_lox")}
    v = run("dittus_boelter", mdot=p["mdot_ox"], d=p["d_bore"], mu=p["mu_lox"], k=p["k_lox"], cp=p["cp_lox"], n=0.4)
    assert v["reynolds"] == approx(p["re_lox"]) and v["prandtl"] == approx(p["pr_lox"]) and v["h"] == approx(p["h_lox"])


# =========================================================================== fluid


def test_friction_factor_against_published_values_and_formulas():
    # F. M. White, Fluid Mechanics, 7th ed., section 6.6: smooth pipes (Prandtl's law, equation 6.38)
    for re_, f in ((4000, 0.0399), (1e4, 0.0309), (1e5, 0.0180), (1e6, 0.0116), (1e7, 0.0081), (1e8, 0.0059)):
        assert fluid.colebrook(re_, 0.0) == approx(f, abs=5e-5), re_
        # the Blasius formula, 0.316 Re^(-1/4), for smooth pipes with 4000 < Re < 1e5
        if re_ <= 1e5:
            assert fluid.colebrook(re_, 0.0) == approx(0.316 * re_**-0.25, rel=0.03), re_
    # fully rough flow (the right side of the Moody chart): 1 / sqrt(f) = 2 log10(3.7 d / roughness)
    for rel, moody in ((0.05, 0.072), (0.01, 0.038), (0.001, 0.0196), (0.0001, 0.0120)):
        f = fluid.colebrook(1e9, rel)
        assert f == approx((2 * math.log10(3.7 / rel))**-2, rel=2e-3) and f == approx(moody, rel=0.01)
    # the root satisfies the Colebrook formula, and the Haaland formula is within 2 % of it (White, equation 6.49)
    for re_ in (4e3, 2.3e4, 1e5, 1.384e6, 1e8):
        for rel in (0.0, 1e-5, 1.28e-4, 2e-3, 0.05):
            f = fluid.colebrook(re_, rel)
            assert 1 / math.sqrt(f) == approx(-2 * math.log10(rel / 3.7 + 2.51 / (re_ * math.sqrt(f))), rel=1e-11)
            haaland = (-1.8 * math.log10((rel / 3.7)**1.11 + 6.9 / re_))**-2
            assert f == approx(haaland, rel=0.02), (re_, rel)
    with pytest.raises(HandCalcError):
        fluid.colebrook(0.0, 0.0)


def test_pressure_drop_in_each_flow_regime():
    # the example by hand: v = 4 mdot / (rho pi d^2), Re = rho v d / mu, dp = (f L / d + K) rho v^2 / 2
    mdot, rho, mu, d, L, K, eps = 5.3, 1141.0, 1.95e-4, 0.025, 0.15, 0.2, 3.2e-6
    v = run("pressure_drop_minor_loss", mdot=mdot, rho=rho, mu=mu, d=d, L=L, K=K, roughness=eps)
    vel = 4 * mdot / (rho * math.pi * d**2)
    assert v["velocity"] == approx(vel) == approx(9.463, rel=1e-4)
    assert v["reynolds"] == approx(4 * mdot / (math.pi * d * mu)) == approx(1.384e6, rel=1e-3)
    f = v["friction_factor"]
    assert 1 / math.sqrt(f) == approx(-2 * math.log10(eps / d / 3.7 + 2.51 / (v["reynolds"] * math.sqrt(f))), rel=1e-11)
    assert f == approx(0.01352, rel=1e-3)
    q = 0.5 * rho * vel**2
    assert v["dp_friction"] == approx(f * L / d * q) and v["dp_minor"] == approx(K * q)
    assert v["dp_total"] == approx((f * L / d + K) * q) == approx(14.36e3, rel=1e-3)
    # laminar flow: the model must give the Hagen-Poiseuille pressure drop, 32 mu L v / d^2
    lam = run("pressure_drop_minor_loss", mdot=0.01, rho=900.0, mu=0.1, d=0.02, L=3.0, K=0.0, roughness=1e-5)
    vel = 0.01 / (900.0 * math.pi * 0.01**2)
    assert lam["reynolds"] == approx(900.0 * vel * 0.02 / 0.1) and lam["reynolds"] < 2300
    assert lam["dp_total"] == approx(32 * 0.1 * 3.0 * vel / 0.02**2) and lam["dp_minor"] == 0.0
    # the regime changes at Re = 2300: laminar below, the Colebrook formula from there
    for re_, laminar in ((2299.0, True), (2301.0, False)):
        mdot_ = re_ * math.pi * 0.02 * 1e-3 / 4
        got = run("pressure_drop_minor_loss", mdot=mdot_, rho=1000.0, mu=1e-3, d=0.02, L=1.0, K=0.0, roughness=0.0)
        assert got["reynolds"] == approx(re_)
        assert got["friction_factor"] == approx(64 / re_ if laminar else fluid.colebrook(re_, 0.0))
    steps = handcalc.get("pressure_drop_minor_loss").calculate(
        dict(mdot=mdot, rho=rho, mu=mu, d=d, L=L, K=K, roughness=eps)).steps
    assert any("Colebrook" in s.text and r"\log_{10}" in s.latex for s in steps)


def _valve_pressure_by_characteristics(rho, a, L, v0, t_close, t_end, n=120):
    """Pressure rise at a valve at the end of a liquid column without friction (reservoir at the other end),
    when the velocity at the valve goes linearly from v0 to zero in t_close: the method of characteristics
    for the wave equations of the liquid column (p + rho a v and p - rho a v are constant along dx/dt = +a, -a)."""
    dt = L / n / a
    steps = int(round(t_end / dt))
    p, v = np.zeros(n + 1), np.full(n + 1, float(v0))
    z = rho * a
    out = np.zeros(steps + 1)
    for s in range(1, steps + 1):
        t = s * dt
        v_valve = 0.0 if t >= t_close else v0 * (1 - t / t_close)
        pn, vn = np.empty_like(p), np.empty_like(v)
        pn[1:-1] = 0.5 * (p[:-2] + p[2:] + z * (v[:-2] - v[2:]))
        vn[1:-1] = 0.5 * (v[:-2] + v[2:] + (p[:-2] - p[2:]) / z)
        pn[0], vn[0] = 0.0, v[1] - p[1] / z                    # reservoir: constant pressure
        vn[-1], pn[-1] = v_valve, p[-2] + z * (v[-2] - v_valve)    # valve: the velocity is given
        p, v = pn, vn
        out[s] = p[-1]
    return np.arange(steps + 1) * dt, out


def test_surge_formulas_against_the_method_of_characteristics():
    rho, a, L, v0 = 1141.0, 900.0, 1.5, 9.4628
    travel = 2 * L / a
    # a closure faster than the wave travel time: the Joukowsky pressure, rho a dv
    jou = handcalc.get("joukowsky_surge")
    for t_close in (0.0, 0.5 * travel, travel):
        _, dp = _valve_pressure_by_characteristics(rho, a, L, v0, t_close, t_end=6 * travel)
        got = jou.calculate({"rho": rho, "a": a, "dv": v0, "L": L, "t_close": max(t_close, 1e-9)})
        assert dp.max() == approx(got.values["dp"], rel=1e-9) == approx(rho * a * v0)
        assert got.warnings == []
    # a slow, linear closure: the pressure at the valve is a triangle wave between zero and the Michaud value;
    # its mean over the closure is the rigid-column value
    rigid = handcalc.get("rigid_column_surge")
    for t_close in (16 * travel, 0.05, 40 * travel):
        t, dp = _valve_pressure_by_characteristics(rho, a, L, v0, t_close, t_end=t_close)
        got = rigid.calculate({"rho": rho, "L": L, "dv": v0, "t_close": t_close, "a": a})
        assert got.warnings == [] and got.values["t_wave"] == approx(travel)
        assert dp.max() == approx(got.values["dp_michaud"], rel=1e-9) == approx(2 * rho * L * v0 / t_close)
        assert dp.min() >= -1e-6 * dp.max()
        if round(t_close / (2 * travel)) == approx(t_close / (2 * travel)):      # whole periods of the wave
            mean = float(np.sum(0.5 * (dp[1:] + dp[:-1]) * np.diff(t))) / t_close
            assert mean == approx(got.values["dp"], rel=1e-9) == approx(rho * L * v0 / t_close)
    # the surge of the example's system.yaml (dp_surge = rho l v / t_close) is the rigid-column value
    assert rigid.calculate({"rho": rho, "L": L, "dv": v0, "t_close": 0.05}).values["dp"] == approx(323.91e3, rel=1e-4)
    # a slow closure in the Joukowsky model, and a fast closure in the rigid-column model, give a warning
    assert len(jou.calculate({"rho": rho, "a": a, "dv": v0, "L": L, "t_close": 0.05}).warnings) == 1
    fast = rigid.calculate({"rho": rho, "L": L, "dv": v0, "t_close": 0.5 * travel, "a": a})
    assert len(fast.warnings) == 1 and "joukowsky_surge" in fast.warnings[0]
    with pytest.raises(HandCalcError, match="together"):
        jou.calculate({"rho": rho, "a": a, "dv": v0, "L": L})


# =========================================================================== fatigue


def test_basquin_life_against_a_root_solve_of_the_stress_life_curve():
    sigma_f, b, su = 1000e6, -0.10, 485e6
    # whole numbers: amplitude 500 MPa at zero mean stress -> 2 N = (1/2)^(-10) = 1024
    assert run("basquin_life", sigma_a=500e6, sigma_m=0.0, sigma_f=sigma_f, b=b, Su=su)["cycles"] == approx(512.0)
    for sigma_a, sigma_m in ((120e6, 97e6), (80e6, 240e6), (200e6, 0.0), (150e6, -60e6)):
        v = run("basquin_life", sigma_a=sigma_a, sigma_m=sigma_m, sigma_f=sigma_f, b=b, Su=su)
        # the cycle is on the Goodman line of the life N: sigma_a / S(N) + sigma_m / Su = 1, S(N) = sigma_f (2N)^b
        mean = max(sigma_m, 0.0)                                   # no benefit from a compressive mean stress
        life = brentq(lambda logn: sigma_a / (sigma_f * (2 * 10**logn)**b) + mean / su - 1.0, -2, 30)
        assert math.log10(v["cycles"]) == approx(life, rel=1e-9)
        assert v["sigma_ar"] == approx(sigma_f * (2 * v["cycles"])**b)
    v = run("basquin_life", sigma_a=400e6, sigma_m=97e6, sigma_f=sigma_f, b=b, Su=su)
    assert v["sigma_ar"] == approx(500e6) and v["cycles"] == approx(512.0)     # 400 / (1 - 97 / 485) = 500 MPa
    with pytest.raises(HandCalcError, match="negative"):
        run("basquin_life", sigma_a=1e8, sigma_m=0.0, sigma_f=sigma_f, b=0.1, Su=su)
    with pytest.raises(HandCalcError, match="less than the ultimate"):
        run("basquin_life", sigma_a=1e8, sigma_m=500e6, sigma_f=sigma_f, b=b, Su=su)


# =========================================================================== the engine on the example project


def _load_case(example, example_ctx, case_id: str, phase: str, analysis_type: str) -> LoadCase:
    return LoadCase(id=case_id, title=case_id, analysis_type=analysis_type, component=example.subject.id, phase=phase,
                    loads=example_ctx.loads(example, None, phase))


@pytest.fixture(scope="module")
def cases(example, example_ctx):
    return {"proof": _load_case(example, example_ctx, "LC-PROOF", "PH-PROOF", "static_structural"),
            "fire": _load_case(example, example_ctx, "LC-FIRE", "PH-FIRE", "thermo_structural"),
            "th_fire": _load_case(example, example_ctx, "LC-TH-FIRE", "PH-FIRE", "thermal_steady"),
            "flow": _load_case(example, example_ctx, "LC-FLOW-FIRE", "PH-FIRE", "flow"),
            "transport": _load_case(example, example_ctx, "LC-TRANSPORT", "PH-TRANSPORT", "static_structural")}


@pytest.fixture(scope="module")
def specs(example):
    return {s.id: s for s in example.subject.hand_calcs}


def test_registry_serves_the_example_and_names_unknown_models(specs):
    assert len(specs) == 6
    needed = {"thick_cylinder_pressure": {"hoop_inner", "hoop_outer", "hoop_mean", "radial_inner", "axial_closed",
                                          "von_mises_inner"},
              "circular_plate_pressure": {"sigma_center_clamped", "sigma_edge_clamped", "sigma_center_simply_supported",
                                          "deflection_center_clamped", "deflection_center_simply_supported"},
              "tube_bending_axial": {"sigma_bending", "sigma_bending_outer", "sigma_axial", "sigma_tension",
                                     "sigma_compression"},
              "fin_convective_tip": {"T_tip", "heat_flow_base", "m", "mL"},
              "pressure_drop_minor_loss": {"velocity", "reynolds", "friction_factor", "dp_friction", "dp_minor",
                                           "dp_total"}}
    for name, outputs in needed.items():
        assert {o.name for o in handcalc.get(name).outputs} == outputs
    for spec in specs.values():
        m = handcalc.get(spec.model)
        assert set(spec.inputs) <= {i.name for i in m.inputs} and all(c.output in {o.name for o in m.outputs}
                                                                     for c in spec.compare)
    for name in ("thin_cylinder_pressure", "sphere_pressure", "beam_bending", "bolt_circle_loads",
                 "thermal_stress_constrained", "hollow_cylinder_radial_gradient", "conduction_cylinder_wall",
                 "dittus_boelter", "rigid_column_surge", "joukowsky_surge", "basquin_life", "euler_buckling"):
        assert handcalc.MODELS[name] is handcalc.get(name)
    with pytest.raises(HandCalcError) as e:
        handcalc.get("thick_cylinder")
    assert "thick_cylinder_pressure" in str(e.value) and "fin_convective_tip" in str(e.value)


def test_applies_reads_the_when_of_the_specification(specs, cases):
    def which(case):
        return {s for s, spec in specs.items() if handcalc.applies(spec, cases[case])}

    assert which("proof") == {"HC-SHELL-HOOP", "HC-NOZZLE-HOOP", "HC-FLOOR-PLATE"}
    assert which("fire") == {"HC-NECK-BENDING"}          # thermo_structural: the pressure models ask for static_structural
    assert "HC-ACT-TEMP" in which("th_fire") and "HC-PRESSURE-DROP" in which("flow")
    assert which("transport") == set()                   # no pressure; the neck model is for PH-FIRE only
    # needs: the load must be in the load case, with a value
    spec = specs["HC-NECK-BENDING"]
    no_moment = cases["fire"].model_copy(update={"loads": [x for x in cases["fire"].loads
                                                           if not (x.kind == "moment" and x.interface == "INT-ACT")]})
    assert handcalc.applies(spec, cases["fire"]) and not handcalc.applies(spec, no_moment)
    empty = [x.model_copy(update={"si": None}) if x.kind == "moment" else x for x in cases["fire"].loads]
    assert not handcalc.applies(spec, cases["fire"].model_copy(update={"loads": empty}))
    # an empty `when` applies to every load case; the kind alone means any interface; a support has no value
    anything = HandCalcSpec(id="HC-X", model="beam_bending", title="x")
    assert all(handcalc.applies(anything, c) for c in cases.values())

    def when(**kw):
        return HandCalcSpec(id="HC-X", model="beam_bending", title="x", when=HandCalcWhen(**kw))

    assert handcalc.applies(when(needs=["pressure"]), cases["proof"])
    assert not handcalc.applies(when(needs=["pressure"]), cases["transport"])
    assert not handcalc.applies(when(needs=["pressure@INT-ACT"]), cases["proof"])
    assert handcalc.applies(when(needs=["support@INT-INLET"]), cases["fire"])
    assert not handcalc.applies(when(needs=["support@INT-INLET"]), cases["proof"])     # a fixture holds the valve
    held = cases["proof"].model_copy(update={"supports": ["INT-OUTLET"]})
    assert handcalc.applies(when(needs=["support@INT-OUTLET"]), held)
    assert handcalc.applies(when(phases=["PH-PROOF", "PH-BURST"]), cases["proof"])
    assert not handcalc.applies(when(phases=["PH-BURST"]), cases["proof"])
    assert not handcalc.applies(when(analysis=["flow"]), cases["proof"])
    fatigue = LoadCase(id="LC-FATIGUE", analysis_type="fatigue", component="CMP-MOV-BODY")
    assert not handcalc.applies(when(phases=["PH-FIRE"]), fatigue) and handcalc.applies(when(analysis=["fatigue"]), fatigue)


def test_the_six_example_specifications_end_to_end(example, example_ctx, specs, cases):
    ctx = example_ctx

    def out(hc: HandCalcResult) -> dict[str, float]:
        return {o.name: o.si for o in hc.outputs}

    def inp(hc: HandCalcResult) -> dict[str, float]:
        return {i.name: i.si for i in hc.inputs}

    # ---- HC-SHELL-HOOP at proof: p = 1.5 x 10.5 MPa = 15.75 MPa, r_i = 22 mm, r_o = 28 mm
    hc = handcalc.evaluate(specs["HC-SHELL-HOOP"], example, ctx, None, cases["proof"])
    assert hc.id == "HC-SHELL-HOOP-LC-PROOF" == make_id("HC", "SHELL-HOOP", "LC-PROOF")
    assert hc.spec == "HC-SHELL-HOOP" and hc.model == "thick_cylinder_pressure" and hc.load_case == "LC-PROOF"
    assert hc.origin.by == "engine:handcalc" and hc.origin.stage == "plan" and hc.kind == "hand_calc"
    assert inp(hc) == approx({"p": 15.75e6, "r_i": 0.022, "r_o": 0.028})
    o = out(hc)
    assert o["hoop_mean"] == approx(15.75e6 * 22 / 6) == approx(57.75e6)
    assert o["hoop_inner"] == approx(15.75e6 * (28**2 + 22**2) / (28**2 - 22**2)) == approx(66.57e6)
    assert o["hoop_outer"] == approx(2 * 15.75e6 * 22**2 / 300) == approx(50.82e6)
    assert o["radial_inner"] == approx(-15.75e6) and o["axial_closed"] == approx(15.75e6 * 484 / 300)
    assert o["von_mises_inner"] == approx(math.sqrt(3) * 15.75e6 * 784 / 300)
    cmp = hc.comparisons[0]
    assert (cmp.output, cmp.result_key, cmp.band, cmp.tolerance_si) == ("hoop_mean", "max_principal.membrane@SCL-SHELL",
                                                                       (0.6, 1.5), None)
    assert cmp.hand_value == approx(57.75e6) and cmp.fe_value is None and cmp.agree is None and "equilibrium" in cmp.why
    assert hc.reference and "Timoshenko" in hc.reference and len(hc.assumptions) == 5 and hc.warnings == []
    assert [s.result for s in hc.steps][2] == units.display(units.Q(57.75, "MPa")) and len(hc.steps) == 6
    by_name = {i.name: i for i in hc.inputs}
    assert by_name["p"].display == units.display(units.Q(15.75, "MPa")) and by_name["p"].unit == "Pa"
    assert 'load("INT-INLET", "pressure")' in by_name["p"].source and "EF-IN-P-PROOF" in by_name["p"].source
    assert by_name["r_i"].display == "22 mm" and "HC-SHELL-HOOP" in by_name["r_i"].source
    assert [(l.rel, l.to, l.note) for l in hc.links] == [("uses", "EF-IN-P-PROOF", "input p")]
    assert ("about", "LC-PROOF") in [(l.rel, l.to) for l in hc.refs()]
    assert HandCalcResult.model_validate_json(hc.model_dump_json()) == hc          # the record is plain JSON

    # ---- HC-NOZZLE-HOOP at proof: r_i = 12.5 mm, r_o = 20 mm
    hc = handcalc.evaluate(specs["HC-NOZZLE-HOOP"], example, ctx, None, cases["proof"])
    assert out(hc)["hoop_mean"] == approx(15.75e6 * 12.5 / 7.5) == approx(26.25e6)
    assert hc.comparisons[0].band == (0.7, 1.4) and "EF-OUT-P-PROOF" in hc.inputs[0].source

    # ---- HC-FLOOR-PLATE at proof: a = 22 mm, t = 8 mm, nu = 0.29 and E from the material
    hc = handcalc.evaluate(specs["HC-FLOOR-PLATE"], example, ctx, None, cases["proof"])
    unit_stress = 15.75e6 * (22 / 8)**2                              # p a^2 / t^2 = 119.1 MPa
    o = out(hc)
    assert o["sigma_center_clamped"] == approx(3 * 1.29 / 8 * unit_stress) == approx(57.62e6, rel=1e-4)
    assert o["sigma_edge_clamped"] == approx(0.75 * unit_stress) == approx(89.33e6, rel=1e-4)
    assert o["sigma_center_simply_supported"] == approx(3 * 3.29 / 8 * unit_stress) == approx(146.95e6, rel=1e-4)
    rigidity = 193e9 * 0.008**3 / (12 * (1 - 0.29**2))
    assert o["deflection_center_clamped"] == approx(15.75e6 * 0.022**4 / (64 * rigidity)) == approx(6.41e-6, rel=1e-3)
    assert o["deflection_center_simply_supported"] == approx(5.29 / 1.29 * o["deflection_center_clamped"])
    assert inp(hc) == approx({"p": 15.75e6, "a": 0.022, "t": 0.008, "nu": 0.29, "E": 193e9})
    by_name = {i.name: i for i in hc.inputs}
    assert "MAT-316L" in by_name["nu"].source and "no value" in by_name["nu"].source      # the model's default
    assert {l.to for l in hc.links} == {"EF-IN-P-PROOF", "nu", "E"}
    assert len(hc.warnings) == 1 and "5.5 times" in hc.warnings[0]                         # a thick plate
    assert hc.comparisons[0].hand_value == approx(o["sigma_center_clamped"]) and hc.comparisons[0].band == (0.8, 2.6)

    # ---- HC-NECK-BENDING in firing: the actuator (2.8 kg, 25 g lateral, centre of gravity 62 mm above the
    #      face) bends the neck at the section 76 mm under the face; the stem pressure force pulls it
    hc = handcalc.evaluate(specs["HC-NECK-BENDING"], example, ctx, None, cases["fire"])
    lateral = 2.8 * 25 * 9.80665                                     # 686.47 N
    moment = lateral * 0.062 + lateral * 0.076                       # 94.73 N m
    pull = 10.4e6 * math.pi * 0.008**2                               # 2091 N
    assert inp(hc) == approx({"M": moment, "F": pull, "r_o": 0.012, "r_i": 0.008, "r_eval": 0.010})
    area, second = math.pi * (0.012**2 - 0.008**2), math.pi / 4 * (0.012**4 - 0.008**4)
    o = out(hc)
    assert o["sigma_axial"] == approx(pull / area) == approx(8.32e6)
    assert o["sigma_bending"] == approx(moment * 0.010 / second) == approx(72.49e6, rel=1e-4)
    assert o["sigma_bending_outer"] == approx(1.2 * o["sigma_bending"])
    assert o["sigma_tension"] == approx(pull / area + moment * 0.010 / second) == approx(80.81e6, rel=1e-4)
    assert o["sigma_compression"] == approx(pull / area - moment * 0.010 / second)
    assert hc.id == "HC-NECK-BENDING-LC-FIRE" and hc.comparisons[0].result_key == "szz.membrane@SCL-NECK-ROOT"
    assert {(l.to, l.note) for l in hc.links} == {("EF-ACT-M-FIRE", "input M"), ("EF-ACT-F-FIRE", "inputs M, F")}
    assert {i.name: i for i in hc.inputs}["M"].display == units.display(units.Q(moment, "N*m"))

    # ---- HC-ACT-TEMP in firing: the neck as a fin from the LOX temperature to the pad in ambient air
    hc = handcalc.evaluate(specs["HC-ACT-TEMP"], example, ctx, None, cases["th_fire"])
    m = math.sqrt(12.0 * 2 * math.pi * 0.012 / (12.0 * area))        # sqrt(300) = 17.32 1/m
    beta = 12.0 * 4476e-6 / (m * 12.0 * area)
    t_tip = 293.15 + (90.0 - 293.15) / (math.cosh(m * 0.078) + beta * math.sinh(m * 0.078))
    o = out(hc)
    assert o["T_tip"] == approx(t_tip) == approx(241.22, abs=0.01)
    assert o["m"] == approx(math.sqrt(300.0)) and o["mL"] == approx(1.351, rel=1e-4)
    assert o["heat_flow_base"] == approx(-10.63, rel=1e-3)
    cmp = hc.comparisons[0]
    assert (cmp.result_key, cmp.band, cmp.tolerance_si) == ("temperature.min@INT-ACT", None, 15.0)
    by_name = {i.name: i for i in hc.inputs}
    assert "t_lox" in by_name["T_base"].source and "ENG-ICD-002 rev B, table 2" in by_name["T_base"].source
    assert "basis: assumed" in by_name["h"].source                    # the film coefficient is an assumption
    assert "MAT-316L" in by_name["k"].source
    assert {l.to for l in hc.links} == {"t_lox", "t_amb", "h_amb", "k_th"} and hc.warnings == []

    # ---- HC-PRESSURE-DROP: 5.3 kg/s of LOX through 150 mm of the 25 mm bore, K = 0.07
    hc = handcalc.evaluate(specs["HC-PRESSURE-DROP"], example, ctx, None, cases["flow"])
    velocity = 5.3 / (1141 * math.pi * 0.0125**2)
    o = out(hc)
    assert o["velocity"] == approx(velocity) == approx(ctx.params["v_lox"].si)
    assert o["reynolds"] == approx(ctx.params["re_lox"].si)
    assert o["friction_factor"] == approx(fluid.colebrook(o["reynolds"], 3.2e-6 / 0.025)) == approx(0.01352, rel=1e-3)
    dynamic = 0.5 * 1141 * velocity**2
    assert o["dp_friction"] == approx(o["friction_factor"] * 6 * dynamic) and o["dp_minor"] == approx(0.07 * dynamic)
    assert o["dp_total"] == approx(14.36e3, rel=1e-3) and hc.comparisons[0].band == (0.5, 2.0)
    assert len(hc.warnings) == 1 and "6 diameters" in hc.warnings[0]

    # ---- without a load case the record has the id of the specification
    hc = handcalc.evaluate(specs["HC-PRESSURE-DROP"], example, ctx, None, None)
    assert hc.id == "HC-PRESSURE-DROP" and hc.load_case is None and hc.title == specs["HC-PRESSURE-DROP"].title
    assert out(hc)["dp_total"] == approx(o["dp_total"]) and all(l.to != hc.id for l in hc.links)


def _spec(model: str, inputs: dict, compare: list | None = None, **kw) -> HandCalcSpec:
    return HandCalcSpec.model_validate({"id": "HC-T", "model": model, "title": "A test", "inputs": inputs,
                                        "compare": compare or [], **kw})


def test_evaluate_reads_inputs_defaults_units_and_the_policy_band(example, example_ctx, cases):
    ctx = example_ctx
    # a literal in another unit, an expression with a unit, a default from the material, the policy's band
    spec = _spec("circular_plate_pressure", {"p": "2 bar", "a": {"expr": "d_bore / 2"}, "t": "0.2 in"},
                 [{"output": "sigma_edge_clamped", "with": "max_principal.peak@part"}])
    hc = handcalc.evaluate(spec, example, ctx, None, None)
    values = {i.name: i.si for i in hc.inputs}
    assert values == approx({"p": 2e5, "a": 0.0125, "t": 0.00508, "nu": 0.29, "E": 193e9})
    assert hc.comparisons[0].band == tuple(example.policy.handcalc_band) == (0.8, 1.25)
    assert {i.name: i for i in hc.inputs}["t"].display == units.display(units.Q(0.2, "in"))
    # feat(): the geometry gives the radius; without a geometry the error says so
    geo = {"features": {"FT-CAVITY-WALL": {"radius": 0.022, "area": 1e-3}}}
    spec = _spec("thick_cylinder_pressure", {"p": {"expr": 'load("INT-INLET", "pressure")'},
                                             "r_i": {"expr": 'feat("FT-CAVITY-WALL", "radius")'}, "r_o": "28 mm"})
    hc = handcalc.evaluate(spec, example, ctx, geo, cases["proof"])
    assert {o.name: o.si for o in hc.outputs}["hoop_mean"] == approx(57.75e6) and hc.id == "HC-T-LC-PROOF"
    assert {l.to for l in hc.links} == {"EF-IN-P-PROOF", "FT-CAVITY-WALL"}
    assert "FT-CAVITY-WALL" in {i.name: i for i in hc.inputs}["r_i"].source
    with pytest.raises(HandCalcError, match="no geometry"):
        handcalc.evaluate(spec, example, ctx, None, cases["proof"])
    with pytest.raises(HandCalcError, match="no load case"):
        handcalc.evaluate(spec, example, ctx, geo, None)
    # a tolerance is a difference in the unit of the output; a temperature in degC is a clear error
    fin = {"T_base": "-183 degC", "T_inf": {"expr": "t_amb"}, "h": {"expr": "h_amb"}, "r_o": "12 mm", "r_i": "8 mm",
           "L": "78 mm", "A_tip": "4476 mm^2"}
    hc = handcalc.evaluate(_spec("fin_convective_tip", fin, [{"output": "T_tip", "with": "temperature.min@INT-ACT",
                                                              "tolerance": "15 delta_degC"}]), example, ctx, None, None)
    assert {i.name: i.si for i in hc.inputs}["T_base"] == approx(90.15) and hc.comparisons[0].tolerance_si == approx(15.0)
    assert {i.name: i.si for i in hc.inputs}["k"] == approx(12.0)                      # k_th of the material
    with pytest.raises(HandCalcError, match="temperature difference"):
        handcalc.evaluate(_spec("fin_convective_tip", fin, [{"output": "T_tip", "with": "temperature.min@INT-ACT",
                                                             "tolerance": "15 degC"}]), example, ctx, None, None)
    with pytest.raises(HandCalcError, match="temperature difference"):
        handcalc.evaluate(_spec("thermal_stress_constrained", {"dT": "20 degC"}), example, ctx, None, None)
    hc = handcalc.evaluate(_spec("thermal_stress_constrained", {"dT": {"expr": "t_lox - t_amb"}}), example, ctx, None, None)
    assert {o.name: o.si for o in hc.outputs}["sigma_uniaxial"] == approx(193e9 * 13e-6 * 203.15)
    # the fatigue constants come from the material when the specification does not give them
    hc = handcalc.evaluate(_spec("basquin_life", {"sigma_a": "400 MPa", "sigma_m": "97 MPa"}), example, ctx, None, None)
    values = {i.name: i.si for i in hc.inputs}
    assert values == approx({"sigma_a": 400e6, "sigma_m": 97e6, "sigma_f": 1000e6, "b": -0.10, "Su": 485e6})
    assert {o.name: o.si for o in hc.outputs}["cycles"] == approx(512.0)
    assert "MAT-316L" in {i.name: i for i in hc.inputs}["sigma_f"].source and "MAT-316L" in {l.to for l in hc.links}
    # an optional input: the calculation runs without it and says what it cannot show
    hc = handcalc.evaluate(_spec("rigid_column_surge", {"rho": {"expr": "rho_lox"}, "L": {"expr": "l_line"},
                                                        "dv": {"expr": "v_lox"}, "t_close": {"expr": "t_close"}}),
                           example, ctx, None, None)
    assert {o.name: o.si for o in hc.outputs}["dp"] == approx(ctx.params["dp_surge"].si)
    assert [i.name for i in hc.inputs] == ["rho", "L", "dv", "t_close"] and len(hc.warnings) == 1


def test_evaluate_errors_name_the_specification_the_input_and_the_cause(example, example_ctx, cases):
    ctx = example_ctx

    def fails(model, inputs, compare=None, case=None):
        with pytest.raises(HandCalcError) as e:
            handcalc.evaluate(_spec(model, inputs, compare), example, ctx, None, case)
        return str(e.value)

    assert "thick_cylinder_pressure" in fails("thick_cylinder", {})                              # unknown model
    msg = fails("thick_cylinder_pressure", {"p": "1 MPa", "r_i": "22 mm"})                       # a missing input
    assert "HC-T" in msg and "r_o" in msg and "Outer radius" in msg
    msg = fails("thick_cylinder_pressure", {"p": "1 MPa", "r_i": "22 mm", "r_o": "28 mm", "ro": "1 mm"})
    assert "ro" in msg and "does not have" in msg and "r_i, r_o" in msg                          # an unknown input
    msg = fails("thick_cylinder_pressure", {"p": "22 mm", "r_i": "22 mm", "r_o": "28 mm"})       # a wrong dimension
    assert "input p" in msg and "Pa" in msg and "22 mm" in msg
    msg = fails("tube_bending_axial", {"M": {"expr": 'load("INT-ACT", "moment")'}, "F": "0 N", "r_o": "12 mm",
                                       "r_i": "8 mm", "r_eval": "10 mm"}, case=cases["fire"])      # a vector
    assert "one number" in msg and "[0]" in msg
    msg = fails("thick_cylinder_pressure", {"p": {"expr": "p_unknown"}, "r_i": "22 mm", "r_o": "28 mm"})
    assert "input p" in msg and "p_unknown" in msg                                                # a bad expression
    msg = fails("thick_cylinder_pressure", {"p": {"expr": 'load("INT-INLET", "moment")'}, "r_i": "22 mm",
                                            "r_o": "28 mm"}, case=cases["proof"])
    assert "no such load" in msg
    msg = fails("thick_cylinder_pressure", {"p": "1 MPa", "r_i": "28 mm", "r_o": "22 mm"})       # impossible numbers
    assert "HC-T" in msg and "outer radius" in msg
    msg = fails("thick_cylinder_pressure", {"p": "1 MPa", "r_i": "-2 mm", "r_o": "22 mm"})
    assert "more than zero" in msg and "-2 mm" in msg
    msg = fails("dittus_boelter", {"mdot": "1 kg/s", "d": "25 mm", "mu": "1e-3 Pa*s", "cp": "4180 J/(kg*K)", "n": 0.4})
    assert "input k" in msg and "Thermal conductivity of the fluid" in msg         # no default: k_th is the solid's
    ok = {"p": "1 MPa", "r_i": "22 mm", "r_o": "28 mm"}
    msg = fails("thick_cylinder_pressure", ok, [{"output": "hoop", "with": "x.y@z"}])            # an unknown output
    assert "hoop_mean" in msg and "no output" in msg
    msg = fails("thick_cylinder_pressure", ok, [{"output": "hoop_mean", "with": "x.y@z", "tolerance": "5 mm"}])
    assert "tolerance" in msg and "Pa" in msg
    msg = fails("thick_cylinder_pressure", ok, [{"output": "hoop_mean", "with": "x.y@z", "band": [1.5, 0.6]}])
    assert "[low, high]" in msg
    for text in (msg, fails("thick_cylinder_pressure", {"p": "1 MPa", "r_i": "22 mm"})):
        assert text.rstrip().endswith(".") and text[0].isupper()                 # an error is a sentence


def _analysis(results: dict[str, ResultValue] | None = None, analysis_id: str = "AN-PROOF") -> AnalysisRecord:
    return AnalysisRecord(id=analysis_id, load_case="LC-" + analysis_id[3:], analysis_type="static_structural",
                          results=results or {})


def _result(key: str, value, unit: str = "Pa", **kw) -> ResultValue:
    return ResultValue(key=key, value=value, unit=unit, **{"status": "converged", **kw})


def test_compare_gives_one_honest_check_per_comparison(example, example_ctx, specs, cases):
    hc = handcalc.evaluate(specs["HC-SHELL-HOOP"], example, example_ctx, None, cases["proof"])
    key = "max_principal.membrane@SCL-SHELL"
    hand = 57.75e6
    # inside the band [0.6, 1.5]
    an = _analysis({key: _result(key, 62.1e6)})
    (check,) = handcalc.compare(hc, an, example)
    assert check.id == make_id("CHK", "AN-PROOF", hc.id, "hoop_mean") == "CHK-AN-PROOF-HC-SHELL-HOOP-LC-PROOF-HOOP-MEAN"
    assert check.kind == "hand_calc" and check.link == hc.id and check.passed is True
    assert check.value == approx(62.1 / 57.75) and check.limit == 1.5
    for part in ("AN-PROOF", hc.id, key, "hoop_mean", units.display(units.Q(62.1, "MPa")),
                 units.display(units.Q(57.75, "MPa")), "1.075", "0.6 to 1.5", "agree"):
        assert part in check.statement, part
    assert "do not agree" not in check.statement and not ste_problems(check.statement)
    # outside the band, on each side: the limit is the end of the band on the side of the ratio
    (low,) = handcalc.compare(hc, _analysis({key: _result(key, 0.5 * hand)}), example)
    (high,) = handcalc.compare(hc, _analysis({key: _result(key, 1.6 * hand)}), example)
    assert low.passed is False and low.value == approx(0.5) and low.limit == 0.6 and "do not agree" in low.statement
    assert high.passed is False and high.value == approx(1.6) and high.limit == 1.5
    (edge,) = handcalc.compare(hc, _analysis({key: _result(key, 1.5 * hand)}), example)
    assert edge.passed is True                                               # the ends of the band are inside
    (negative,) = handcalc.compare(hc, _analysis({key: _result(key, -hand)}), example)
    assert negative.passed is False and negative.value == approx(-1.0)       # the wrong sign is not agreement
    # a result that is not converged is compared, and the statement says so
    (rough,) = handcalc.compare(hc, _analysis({key: _result(key, 62.1e6, status="not_converged")}), example)
    assert rough.passed is True and "not_converged" in rough.statement and not ste_problems(rough.statement)
    # no such result, or a result without a value: the check has no verdict and says why
    (missing,) = handcalc.compare(hc, _analysis({"von_mises.peak@part": _result("von_mises.peak@part", 1e8)}), example)
    assert missing.passed is None and missing.value is None and missing.link == hc.id
    assert f"gives no result {key}" in missing.statement and not ste_problems(missing.statement)
    (empty,) = handcalc.compare(hc, _analysis({key: _result(key, None, status="not_run")}), example)
    assert empty.passed is None and "no value" in empty.statement and "not_run" in empty.statement
    # a result that is itself a hand calculation, or that has another dimension, is no independent check
    (circular,) = handcalc.compare(hc, _analysis({key: _result(key, hand, status="estimate", method="hand_calc")}), example)
    assert circular.passed is None and "independent" in circular.statement
    (wrong_unit,) = handcalc.compare(hc, _analysis({key: _result(key, 0.004, unit="m")}), example)
    assert wrong_unit.passed is None and "not possible" in wrong_unit.statement
    # the comparisons with the analysis numbers filled; the record does not change
    (filled,) = handcalc.compared(hc, an, example)
    assert filled.fe_value == approx(62.1e6) and filled.ratio == approx(62.1 / 57.75) and filled.agree is True
    assert hc.comparisons[0].fe_value is None and hc.comparisons[0].agree is None
    assert handcalc.compared(hc, _analysis(), example)[0].agree is None

    # a tolerance: the absolute difference in the unit of the output
    hc = handcalc.evaluate(specs["HC-ACT-TEMP"], example, example_ctx, None, cases["th_fire"])
    key = "temperature.min@INT-ACT"
    t_hand = hc.comparisons[0].hand_value
    (warm,) = handcalc.compare(hc, _analysis({key: _result(key, t_hand + 9.5, unit="K")}, "AN-TH-FIRE"), example)
    assert warm.passed is True and warm.value == approx(9.5) and warm.limit == 15.0
    assert units.display(units.Q(9.5, "K")) in warm.statement and units.display(units.Q(15.0, "K")) in warm.statement
    assert warm.id == "CHK-AN-TH-FIRE-HC-ACT-TEMP-LC-TH-FIRE-T-TIP" and not ste_problems(warm.statement)
    (cold,) = handcalc.compare(hc, _analysis({key: _result(key, t_hand - 20.0, unit="K")}, "AN-TH-FIRE"), example)
    assert cold.passed is False and cold.value == approx(20.0) and "do not agree" in cold.statement

    # a hand value of zero has no ratio; the policy's band serves a comparison without band and tolerance
    spec = _spec("tube_bending_axial", {"M": "0 N*m", "F": "0 N", "r_o": "12 mm", "r_i": "8 mm", "r_eval": "10 mm"},
                 [{"output": "sigma_tension", "with": "szz.membrane@SCL-NECK-ROOT"},
                  {"output": "sigma_tension", "with": "szz.membrane_bending@SCL-NECK-ROOT"}])
    hc = handcalc.evaluate(spec, example, example_ctx, None, None)
    an = _analysis({"szz.membrane@SCL-NECK-ROOT": _result("szz.membrane@SCL-NECK-ROOT", 1e6)})
    zero, other = handcalc.compare(hc, an, example)
    assert zero.passed is None and "zero" in zero.statement and other.passed is None
    assert zero.id != other.id and zero.id.endswith("SZZ-MEMBRANE-SCL-NECK-ROOT")     # one output, two results
    spec = _spec("beam_bending", {"M": "100 N*m", "Z": "1000 mm^3"}, [{"output": "sigma_bending", "with": "sxx.max@part"}])
    hc = handcalc.evaluate(spec, example, example_ctx, None, None)
    for value, passed in ((100e6, True), (81e6, True), (79e6, False), (126e6, False)):
        (check,) = handcalc.compare(hc, _analysis({"sxx.max@part": _result("sxx.max@part", value)}), example)
        assert check.passed is passed and "0.8 to 1.25" in check.statement


def test_sources_of_inputs_follow_the_ste_rules(example, example_ctx, specs, cases):
    problems = []
    for spec_id, case in (("HC-SHELL-HOOP", "proof"), ("HC-FLOOR-PLATE", "proof"), ("HC-NECK-BENDING", "fire"),
                          ("HC-ACT-TEMP", "th_fire"), ("HC-PRESSURE-DROP", "flow")):
        hc = handcalc.evaluate(specs[spec_id], example, example_ctx, None, cases[case])
        for i in hc.inputs:
            # the text an engineer wrote (the source of a parameter) is quoted as it is: check the engine's part
            own = i.source.split(" The source is: ")[0]
            problems += [f"{spec_id}.{i.name}: {p}" for p in ste_problems(own)]
            assert i.source.endswith(".")
        assert hc.title == f"{specs[spec_id].title}, load case {hc.load_case}"       # the engineer's title, as it is
    assert not problems, "\n".join(problems)


def test_load_items_feed_the_expressions(example, example_ctx):
    # a hand-built load case: two pressures on one interface add, and a factor scales a load
    loads = [LoadItem(effect="EF-A", interface="INT-INLET", kind="pressure", si=4e6, unit="Pa"),
             LoadItem(effect="EF-B", interface="INT-INLET", kind="pressure", si=1e6, unit="Pa", factor=2.0)]
    case = LoadCase(id="LC-X", analysis_type="static_structural", component="CMP-MOV-BODY", loads=loads)
    spec = HandCalcSpec(id="HC-SUM", model="thick_cylinder_pressure", title="Sum of two pressures",
                        inputs={"p": Value(expr='load("INT-INLET", "pressure")'), "r_i": Value(value=0.02, unit="m"),
                                "r_o": Value(value=0.03, unit="m")},
                        compare=[Compare.model_validate({"output": "hoop_mean", "with": "max_principal.membrane@SCL-SHELL"})])
    hc = handcalc.evaluate(spec, example, example_ctx, None, case)
    assert hc.inputs[0].si == approx(6e6) and hc.comparisons[0].hand_value == approx(6e6 * 2.0)
    assert [l.to for l in hc.links] == ["EF-A", "EF-B"] and "EF-A, EF-B" in hc.inputs[0].source
    assert hc.id == "HC-SUM-LC-X"
