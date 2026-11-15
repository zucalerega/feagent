"""Flow in pipes: pressure drop, and the pressure surge of a valve closure.

Sources
    [White] F. M. White, Fluid Mechanics, 7th edition, McGraw-Hill, 2011. Section 6.3, equation 6.10 (the
            Darcy-Weisbach equation). Section 6.4, equation 6.13 (laminar flow, f = 64 / Re). Section 6.6:
            equation 6.38 (Prandtl's law for smooth pipes, with a table of f for Re from 4000 to 1e8),
            equation 6.48 (the Colebrook formula) and equation 6.49 (the Haaland formula, within 2 % of
            Colebrook). Section 6.9 (minor losses, dp = K rho v^2 / 2).
    [Cole]  C. F. Colebrook, Turbulent flow in pipes, with particular reference to the transition region
            between the smooth and rough pipe laws, Journal of the Institution of Civil Engineers, volume 11,
            1939, pages 133 to 156.
    [WS]    E. B. Wylie and V. L. Streeter, Fluid Transients in Systems, Prentice Hall, 1993, chapter 1: the
            Joukowsky relation dp = rho a dv for a change of the velocity that is faster than the travel time
            of a pressure wave, 2 L / a.
    [Parm]  J. Parmakian, Waterhammer Analysis, Dover, 1963, chapter 2 (rigid water column theory):
            dp = rho L dv/dt.

The friction factor (Darcy). Re < 2300: laminar, f = 64 / Re. Re >= 2300: the Colebrook formula,
    1 / sqrt(f) = -2 log10( (roughness / d) / 3.7 + 2.51 / (Re sqrt(f)) ),
solved by fixed-point iteration to 1e-13 (`colebrook`). Between Re = 2300 and 4000 the flow is in transition
and the model gives a warning. The formula covers a relative roughness up to 0.05.

The surge of a slow closure. With pressure waves (speed a) and a velocity that decreases linearly in the
time T > 2 L / a, the pressure rise at the valve is a triangle wave in time, between zero and
2 rho L dv / T (the Michaud formula); its mean is the rigid-column value rho L dv / T. The model
`rigid_column_surge` gives both numbers.

Verification (tests): the friction factor against the table of White for smooth pipes, the fully rough limit
of the Moody chart, the Blasius and Haaland formulas, and the residual of the Colebrook formula; the two surge
formulas against a method-of-characteristics solution of the wave equations of a liquid column.
"""

from __future__ import annotations

import math

from dossier.handcalc.base import Calc, HandCalcError, Input, Output, model, shown

_WHITE = "F. M. White, Fluid Mechanics, 7th edition, McGraw-Hill, 2011"

RE_LAMINAR = 2300.0         # under this Reynolds number the flow is laminar
RE_TURBULENT = 4000.0       # from this Reynolds number the flow is turbulent [White, section 6.6]
ROUGHNESS_MAX = 0.05        # largest relative roughness of the Colebrook formula and of the Moody chart
DEVELOPED_LENGTH = 10.0     # L / d under which the friction factor of developed flow is too small


def colebrook(reynolds: float, relative_roughness: float) -> float:
    """The Darcy friction factor of turbulent flow: the root of the Colebrook formula."""
    if reynolds <= 0.0 or relative_roughness < 0.0:
        raise HandCalcError("The Colebrook formula needs a Reynolds number of more than zero and a roughness of "
                            "zero or more.")
    x = 8.0                                         # 1 / sqrt(f); the iteration contracts by about 0.87 / x
    for _ in range(200):
        new = -2.0 * math.log10(relative_roughness / 3.7 + 2.51 * x / reynolds)
        if abs(new - x) < 1e-13:
            return 1.0 / new**2
        x = new
    raise HandCalcError(f"The iteration of the Colebrook formula did not converge for a Reynolds number of "
                        f"{reynolds:g}.")


def _pressure_drop_validity(v) -> list[str]:
    out = []
    re = v["reynolds"]
    if RE_LAMINAR <= re < RE_TURBULENT:
        out.append(f"The Reynolds number is {shown(re, '')}. The flow is in the transition range from "
                   f"{RE_LAMINAR:g} to {RE_TURBULENT:g}, and the friction factor is not certain there. The model "
                   f"uses the Colebrook formula for turbulent flow, which gives the larger pressure drop.")
    rel = v["roughness"] / v["d"]
    if rel > ROUGHNESS_MAX:
        out.append(f"The relative roughness is {rel:.3g}. The Colebrook formula is for a relative roughness of "
                   f"{ROUGHNESS_MAX:g} or less.")
    if 0.0 < v["L"] < DEVELOPED_LENGTH * v["d"]:
        out.append(f"The pipe is short: its length is {v['L'] / v['d']:.3g} diameters. The friction factor is for "
                   f"fully developed flow, which needs more than {DEVELOPED_LENGTH:g} diameters. The true friction "
                   f"loss is larger.")
    return out


@model(
    name="pressure_drop_minor_loss",
    title="Pressure drop of a pipe with friction and a loss coefficient",
    reference=f"{_WHITE}, section 6.3 (the Darcy-Weisbach equation, 6.10), section 6.6 (the Colebrook formula, "
              f"6.48) and section 6.9 (minor losses). C. F. Colebrook, Journal of the Institution of Civil "
              f"Engineers, volume 11, 1939, pages 133 to 156.",
    inputs=[
        Input("mdot", "Mass flow", "kg/s", symbol=r"\dot{m}"),
        Input("rho", "Density of the fluid", "kg/m^3", symbol=r"\rho"),
        Input("mu", "Dynamic viscosity of the fluid", "Pa*s", symbol=r"\mu"),
        Input("d", "Inner diameter of the pipe", "m"),
        Input("L", "Length of the pipe", "m", sign="nonnegative"),
        Input("K", "Loss coefficient of the local losses, for the velocity in the pipe", "", sign="nonnegative"),
        Input("roughness", "Roughness of the pipe wall", "m", sign="nonnegative", symbol=r"\varepsilon"),
    ],
    outputs=[
        Output("velocity", "Mean velocity in the pipe", "m/s", "v"),
        Output("reynolds", "Reynolds number of the flow", "", r"\mathrm{Re}"),
        Output("friction_factor", "Darcy friction factor", "", "f"),
        Output("dp_friction", "Pressure drop from wall friction", "Pa", r"\Delta p_{f}"),
        Output("dp_minor", "Pressure drop from the local losses", "Pa", r"\Delta p_{K}"),
        Output("dp_total", "Total pressure drop", "Pa", r"\Delta p"),
    ],
    assumptions=[
        "The flow is steady and incompressible, with constant fluid properties.",
        "The pipe is straight and circular, with a constant diameter.",
        "The flow is fully developed. The friction factor has no allowance for the entry length.",
        "The loss coefficient K is for the dynamic pressure in the pipe of diameter d.",
    ],
    validity=_pressure_drop_validity,
    symbols={"A": "A", "q_dyn": "q"},
)
def pressure_drop_minor_loss(c: Calc) -> None:
    c.let("A", "pi * d**2 / 4", "The diameter gives the flow area.", unit="m^2")
    c.let("velocity", "mdot / (rho * A)", "The mass flow, the density and the flow area give the mean velocity.")
    re = c.let("reynolds", "rho * velocity * d / mu",
               "The velocity, the diameter and the fluid properties give the Reynolds number.")
    if re < RE_LAMINAR:
        c.let("friction_factor", "64 / reynolds", "The flow is laminar, and the Hagen-Poiseuille law gives the "
              "friction factor.")
    else:
        c.given("friction_factor", colebrook(re, c["roughness"] / c["d"]),
                "The flow is turbulent, and the model solves the Colebrook formula for the friction factor by "
                "iteration.",
                r"\frac{1}{\sqrt{f}} = -2\,\log_{10}\left(\frac{\varepsilon}{3.7\,d} + "
                r"\frac{2.51}{\mathrm{Re}\,\sqrt{f}}\right)")
    c.let("q_dyn", "rho * velocity**2 / 2", "The density and the velocity give the dynamic pressure.", unit="Pa")
    c.let("dp_friction", "friction_factor * L / d * q_dyn",
          "The Darcy-Weisbach equation gives the pressure drop from wall friction.")
    c.let("dp_minor", "K * q_dyn", "The loss coefficient times the dynamic pressure gives the pressure drop of the "
          "local losses.")
    c.let("dp_total", "dp_friction + dp_minor", "The sum of the two is the total pressure drop.")


def _rigid_column_validity(v) -> list[str]:
    if v.get("a") is None:
        return ["The specification gives no wave speed a. The model cannot compare the closure time with the "
                "travel time of a pressure wave."]
    if v["t_close"] < v["t_wave"]:
        return [f"The closure is fast: the closure time, {shown(v['t_close'], 's')}, is less than the travel time "
                f"of a pressure wave, {shown(v['t_wave'], 's')}. The rigid-column model is not applicable. Use the "
                f"model joukowsky_surge."]
    return []


@model(
    name="rigid_column_surge",
    title="Pressure surge of a slow valve closure (rigid liquid column)",
    reference="J. Parmakian, Waterhammer Analysis, Dover, 1963, chapter 2 (rigid water column theory). The "
              "largest pressure with pressure waves is the Michaud formula. The documentation of "
              "dossier.handcalc.fluid gives its derivation.",
    inputs=[
        Input("rho", "Density of the liquid", "kg/m^3", symbol=r"\rho"),
        Input("L", "Length of the liquid column between the valve and the open end", "m"),
        Input("dv", "Decrease of the velocity of the liquid", "m/s", symbol=r"\Delta v"),
        Input("t_close", "Time in which the velocity decreases", "s", symbol=r"t_{c}"),
        Input("a", "Speed of the pressure waves in the liquid column", "m/s", optional=True),
    ],
    outputs=[
        Output("dp", "Pressure rise at the valve from the inertia of the rigid liquid column", "Pa", r"\Delta p"),
        Output("dp_michaud", "Largest pressure rise at the valve with pressure waves, for a linear decrease of the "
               "velocity", "Pa", r"\Delta p_{\max}"),
    ],
    assumptions=[
        "The closure is slow: the closure time is longer than the travel time of a pressure wave, 2 L / a.",
        "The liquid column has a constant cross-section and no friction.",
        "The velocity decreases at a constant rate in the closure time.",
        "The pressure at the open end of the column is constant.",
    ],
    validity=_rigid_column_validity,
    symbols={"t_wave": r"t_{w}"},
)
def rigid_column_surge(c: Calc) -> None:
    c.let("dp", "rho * L * dv / t_close",
          "Newton's second law for the rigid liquid column gives the pressure rise at the valve.")
    c.let("dp_michaud", "2 * rho * L * dv / t_close",
          "With pressure waves and a linear decrease of the velocity, the largest pressure rise is two times "
          "this value.")
    if "a" in c:
        c.let("t_wave", "2 * L / a", "A pressure wave moves to the open end and back in this time.", unit="s")


def _joukowsky_validity(v) -> list[str]:
    if v.get("t_wave") is None:
        return ["The specification gives no length L and no closure time t_close. The model cannot compare the "
                "closure time with the travel time of a pressure wave. The result is an upper limit."]
    if v["t_close"] > v["t_wave"]:
        return [f"The closure is slow: the closure time, {shown(v['t_close'], 's')}, is more than the travel time "
                f"of a pressure wave, {shown(v['t_wave'], 's')}. The Joukowsky pressure is an upper limit. The "
                f"model rigid_column_surge gives the pressure rise of a slow closure."]
    return []


@model(
    name="joukowsky_surge",
    title="Pressure surge of a fast valve closure (Joukowsky relation)",
    reference="E. B. Wylie and V. L. Streeter, Fluid Transients in Systems, Prentice Hall, 1993, chapter 1 (the "
              "Joukowsky relation, dp = rho a dv).",
    inputs=[
        Input("rho", "Density of the liquid", "kg/m^3", symbol=r"\rho"),
        Input("a", "Speed of the pressure waves in the liquid column", "m/s"),
        Input("dv", "Decrease of the velocity of the liquid", "m/s", symbol=r"\Delta v"),
        Input("L", "Length of the liquid column between the valve and the open end", "m", optional=True),
        Input("t_close", "Time in which the velocity decreases", "s", optional=True, symbol=r"t_{c}"),
    ],
    outputs=[
        Output("dp", "Pressure rise at the valve", "Pa", r"\Delta p"),
    ],
    assumptions=[
        "The closure is fast: the closure time is shorter than the travel time of a pressure wave, 2 L / a.",
        "The liquid column has a constant cross-section and no friction.",
        "The wave speed a is that of the liquid in the elastic pipe.",
    ],
    validity=_joukowsky_validity,
    symbols={"t_wave": r"t_{w}"},
)
def joukowsky_surge(c: Calc) -> None:
    c.require(("L" in c) == ("t_close" in c), "The model joukowsky_surge needs the length L and the closure time "
                                              "t_close together, or none of the two.")
    c.let("dp", "rho * a * dv", "The Joukowsky relation gives the pressure rise of a fast change of the velocity.")
    if "L" in c:
        c.let("t_wave", "2 * L / a", "A pressure wave moves to the open end and back in this time.", unit="s")


__all__ = ["pressure_drop_minor_loss", "rigid_column_surge", "joukowsky_surge", "colebrook",
           "RE_LAMINAR", "RE_TURBULENT", "ROUGHNESS_MAX", "DEVELOPED_LENGTH"]
