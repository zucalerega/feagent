"""Heat conduction, convection and thermal stress.

Sources
    [Inc]   F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass
            Transfer, 6th edition, Wiley, 2007. Section 3.3.1 (the cylinder: thermal resistances of the wall
            and of the fluid films in series). Section 3.6.2 and Table 3.4 (fins of uniform cross-section;
            case A is the fin with convection at the tip). Section 8.5, equation 8.60 (the Dittus-Boelter
            correlation, for 0.7 <= Pr <= 160, Re >= 10 000 and L / d >= 10).
    [TG]    S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd edition, McGraw-Hill, 1970,
            chapter 13 (Thermal stress): article "The long circular cylinder" gives the stress of a hollow
            cylinder with a steady radial heat flow.
    [Roark] W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill,
            2002, section 16.6 (Temperature stresses), cases 1 and 2: a bar and a plate with rigid supports.

The fin with a body at its tip. Table 3.4, case A, of [Inc] has a tip that loses heat from the area A_c of
the cross-section. Here the tip carries a body of surface area A_tip at the tip temperature, thus the tip
condition is -k A_c dT/dx = h A_tip (T - T_inf), and the tip parameter h / (m k) of case A becomes
beta = h A_tip / (m k A_c):
    (T(x) - T_inf) / (T_base - T_inf) = (cosh m(L - x) + beta sinh m(L - x)) / (cosh mL + beta sinh mL)
    heat flow at the base = k A_c m (T_base - T_inf) (sinh mL + beta cosh mL) / (cosh mL + beta sinh mL)
with m^2 = h P / (k A_c). A_tip = 0 gives case B (adiabatic tip), A_tip = A_c gives case A.

The hollow cylinder with a steady radial heat flow (T_i - T_o = dT, free ends, generalized plane strain):
    C = E alpha dT / (2 (1 - nu) ln(r_o / r_i))
    hoop stress at r_i = C (1 - 2 r_o^2 ln(r_o / r_i) / (r_o^2 - r_i^2))     (compressive when the inside is hot)
    hoop stress at r_o = C (1 - 2 r_i^2 ln(r_o / r_i) / (r_o^2 - r_i^2))
and the axial stress is equal to the hoop stress at each surface. A thin wall gives -/+ E alpha dT / (2 (1 - nu)).

Verification (tests): the fin against a numerical solution of the fin equation (scipy.integrate.solve_bvp)
and against the two cases of Table 3.4; the cylinder wall against a numerical solution of the conduction
equation; the thermal stress of the cylinder against a numerical solution of the equilibrium equation in
terms of the displacement, and against the integral formulas of [TG]; the Dittus-Boelter correlation against
example 8.4 of [Inc] and against the heat transfer coefficient of the example project.
"""

from __future__ import annotations

from dossier.handcalc.base import Calc, Input, Output, model, shown

_INC = ("F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass Transfer, "
        "6th edition, Wiley, 2007")
_TG = "S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd edition, McGraw-Hill, 1970"
_ROARK = "W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill, 2002"

_K = "W/(m*K)"
_H = "W/(m^2*K)"

BIOT_LIMIT = 0.1        # h (r_o - r_i) / k above which the temperature is not constant across the fin wall


def _fin_validity(v) -> list[str]:
    out = []
    biot = v["h"] * (v["r_o"] - v["r_i"]) / v["k"]
    if biot > BIOT_LIMIT:
        out.append(f"The Biot number of the wall, h (r_o - r_i) / k, is {biot:.3g}. The fin model needs a value of "
                   f"{BIOT_LIMIT:g} or less. The temperature is not constant across the wall.")
    if v["L"] < 2.0 * v["r_o"]:
        out.append(f"The fin is not slender: its length, {shown(v['L'], 'm')}, is less than its outer diameter, "
                   f"{shown(2.0 * v['r_o'], 'm')}. The heat flow near the base is not in one direction.")
    return out


@model(
    name="fin_convective_tip",
    title="Hollow circular fin with a body at its tip that loses heat",
    reference=f"{_INC}, section 3.6.2 and Table 3.4, case A (fin of uniform cross-section with convection at the "
              f"tip). The model puts the surface area A_tip of the body at the tip in the tip condition of case A; "
              f"the docstring of dossier.handcalc.thermal gives the solution.",
    inputs=[
        Input("T_base", "Temperature of the base of the fin", "K", symbol=r"T_{b}"),
        Input("T_inf", "Temperature of the fluid around the fin", "K", symbol=r"T_{\infty}"),
        Input("h", "Heat transfer coefficient to the fluid", _H),
        Input("k", "Thermal conductivity of the fin", _K, default="k_th"),
        Input("r_o", "Outer radius of the fin", "m"),
        Input("r_i", "Inner radius of the fin (zero for a solid fin)", "m", sign="nonnegative"),
        Input("L", "Length of the fin", "m"),
        Input("A_tip", "Surface area of the body at the tip that loses heat", "m^2", sign="nonnegative"),
    ],
    outputs=[
        Output("T_tip", "Temperature of the tip of the fin and of the body on it", "K", r"T_{L}"),
        Output("heat_flow_base", "Heat flow from the base into the fin (negative when the base is colder than the "
               "fluid)", "W", r"Q_{b}"),
        Output("m", "Fin parameter", "1/m", "m"),
        Output("mL", "Fin parameter times the length of the fin", "", "m L"),
    ],
    assumptions=[
        "The temperature is constant across the wall of the fin. It changes only along the fin.",
        "The state is steady. The conductivity and the heat transfer coefficient are constant.",
        "Only the outer surface of the fin loses heat. The inner surface has no heat flow.",
        "The body at the tip has the temperature of the tip. Its surface loses heat with the same coefficient.",
        "The base of the fin has the given temperature.",
        "There is no heat radiation.",
    ],
    validity=_fin_validity,
    symbols={"A_c": "A_{c}", "P": "P", "beta": r"\beta", "theta_b": r"\theta_{b}", "theta_tip": r"\theta_{L}"},
)
def fin_convective_tip(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.let("A_c", "pi * (r_o**2 - r_i**2)", "The two radii give the area of the cross-section of the fin.", unit="m^2")
    c.let("P", "2 * pi * r_o", "The outer surface loses heat, thus its perimeter is the perimeter of the fin.",
          unit="m")
    c.let("m", "sqrt(h * P / (k * A_c))",
          "The fin parameter compares the heat loss of the surface with the conduction along the fin.")
    c.let("mL", "m * L", "The fin parameter times the length is the dimensionless length of the fin.")
    c.let("beta", "h * A_tip / (m * k * A_c)",
          "The tip parameter compares the heat loss of the body at the tip with the conduction along the fin.",
          unit="")
    c.let("theta_b", "T_base - T_inf", "The base temperature minus the fluid temperature is the excess temperature "
          "of the base.", unit="K")
    c.let("theta_tip", "theta_b / (cosh(mL) + beta * sinh(mL))",
          "The solution of the fin equation with heat loss at the tip gives the excess temperature of the tip.",
          unit="K")
    c.let("T_tip", "T_inf + theta_tip", "The fluid temperature plus the excess temperature is the temperature of "
          "the tip.")
    c.let("heat_flow_base", "k * A_c * m * theta_b * (sinh(mL) + beta * cosh(mL)) / (cosh(mL) + beta * sinh(mL))",
          "The temperature gradient at the base gives the heat flow from the base into the fin.")


@model(
    name="thermal_stress_constrained",
    title="Thermal stress of a part with rigid supports (one and two directions)",
    reference=f"{_ROARK}, section 16.6 (Temperature stresses), case 1 (bar with fixed ends) and case 2 (plate "
              f"with fixed edges). {_TG}, chapter 13, article \"The simplest cases of thermal stress distribution\".",
    inputs=[
        Input("dT", "Temperature change of the part (positive for an increase)", "K", sign="any", difference=True,
              symbol=r"\Delta T"),
        Input("E", "Modulus of elasticity", "Pa", default="E"),
        Input("alpha", "Coefficient of thermal expansion", "1/K", default="alpha", symbol=r"\alpha"),
        Input("nu", "Poisson's ratio", "", default="nu", sign="nonnegative", symbol=r"\nu"),
    ],
    outputs=[
        Output("sigma_uniaxial", "Stress of a bar with no thermal strain along its axis (compression is negative)",
               "Pa", r"\sigma_{1}"),
        Output("sigma_biaxial", "Stress of a plate with no thermal strain in its plane, in each direction "
               "(compression is negative)", "Pa", r"\sigma_{2}"),
    ],
    assumptions=[
        "The supports are rigid: they stop all of the thermal strain in the given directions.",
        "The temperature change is the same in all of the part.",
        "The material is linear elastic and isotropic. Its properties do not change with the temperature.",
        "The part has no stress before the temperature change.",
    ],
)
def thermal_stress_constrained(c: Calc) -> None:
    c.require(c["nu"] <= 0.5, f"Poisson's ratio nu must be 0.5 or less. It is {c.show('nu')}.")
    c.let("sigma_uniaxial", "-E * alpha * dT",
          "The supports stop the thermal strain in one direction, and Hooke's law gives the stress.")
    c.let("sigma_biaxial", "-E * alpha * dT / (1 - nu)",
          "The supports stop the thermal strain in two directions, and Hooke's law gives the two equal stresses.")


@model(
    name="hollow_cylinder_radial_gradient",
    title="Thermal stress of a hollow cylinder with a steady radial temperature difference",
    reference=f"{_TG}, chapter 13 (Thermal stress), article \"The long circular cylinder\": hollow cylinder with a "
              f"steady heat flow. The docstring of dossier.handcalc.thermal gives the formulas at the two surfaces.",
    inputs=[
        Input("dT", "Temperature of the inner surface minus the temperature of the outer surface", "K", sign="any",
              difference=True, symbol=r"\Delta T"),
        Input("r_i", "Inner radius of the cylinder", "m"),
        Input("r_o", "Outer radius of the cylinder", "m"),
        Input("E", "Modulus of elasticity", "Pa", default="E"),
        Input("alpha", "Coefficient of thermal expansion", "1/K", default="alpha", symbol=r"\alpha"),
        Input("nu", "Poisson's ratio", "", default="nu", sign="nonnegative", symbol=r"\nu"),
    ],
    outputs=[
        Output("hoop_inner", "Thermal hoop stress at the inner surface", "Pa", r"\sigma_{\theta,i}"),
        Output("hoop_outer", "Thermal hoop stress at the outer surface", "Pa", r"\sigma_{\theta,o}"),
        Output("axial_inner", "Thermal axial stress at the inner surface", "Pa", r"\sigma_{z,i}"),
        Output("axial_outer", "Thermal axial stress at the outer surface", "Pa", r"\sigma_{z,o}"),
    ],
    assumptions=[
        "The cylinder is long, and the section is far from its ends.",
        "The cylinder has free ends: there is no axial force on it.",
        "The heat flow is steady and radial, with a constant conductivity. The temperature is then a logarithm of "
        "the radius.",
        "The temperature does not change along the cylinder or around it.",
        "The material is linear elastic and isotropic. Its properties do not change with the temperature.",
        "The model gives the thermal stress only. The stress from pressure adds to it.",
    ],
    symbols={"C": "C"},
)
def hollow_cylinder_radial_gradient(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.require(c["nu"] <= 0.5, f"Poisson's ratio nu must be 0.5 or less. It is {c.show('nu')}.")
    c.let("C", "E * alpha * dT / (2 * (1 - nu) * log(r_o / r_i))",
          "The elastic constants, the temperature difference and the two radii give the stress factor.", unit="Pa")
    c.let("hoop_inner", "C * (1 - 2 * r_o**2 * log(r_o / r_i) / (r_o**2 - r_i**2))",
          "The solution for a steady radial heat flow gives the hoop stress at the inner surface.")
    c.let("hoop_outer", "C * (1 - 2 * r_i**2 * log(r_o / r_i) / (r_o**2 - r_i**2))",
          "The solution for a steady radial heat flow gives the hoop stress at the outer surface.")
    c.let("axial_inner", "hoop_inner",
          "The radial stress is zero at the inner surface, and the axial stress is equal to the hoop stress there.")
    c.let("axial_outer", "hoop_outer",
          "The radial stress is zero at the outer surface, and the axial stress is equal to the hoop stress there.")


@model(
    name="conduction_cylinder_wall",
    title="Steady heat flow through a cylinder wall with a fluid on each side",
    reference=f"{_INC}, section 3.3.1 (The cylinder): the thermal resistance of the wall, ln(r_o / r_i) / (2 pi k L), "
              f"in series with the resistances 1 / (2 pi r L h) of the two fluid films.",
    inputs=[
        Input("T_i", "Temperature of the fluid in the cylinder", "K", symbol=r"T_{i}"),
        Input("T_o", "Temperature of the fluid around the cylinder", "K", symbol=r"T_{o}"),
        Input("h_i", "Heat transfer coefficient on the inner surface", _H, symbol=r"h_{i}"),
        Input("h_o", "Heat transfer coefficient on the outer surface", _H, symbol=r"h_{o}"),
        Input("r_i", "Inner radius of the cylinder", "m"),
        Input("r_o", "Outer radius of the cylinder", "m"),
        Input("k", "Thermal conductivity of the wall", _K, default="k_th"),
    ],
    outputs=[
        Output("heat_flow_per_length", "Heat flow from the inner fluid to the outer fluid, for one metre of length",
               "W/m", r"q_{l}"),
        Output("T_wall_inner", "Temperature of the inner surface of the wall", "K", r"T_{w,i}"),
        Output("T_wall_outer", "Temperature of the outer surface of the wall", "K", r"T_{w,o}"),
        Output("dT_wall", "Temperature of the inner surface minus the temperature of the outer surface", "K",
               r"\Delta T_{w}"),
    ],
    assumptions=[
        "The state is steady, and the wall has no heat source.",
        "The heat flow is radial: the cylinder is long, and the conditions do not change along it or around it.",
        "The conductivity and the two heat transfer coefficients are constant.",
        "There is no heat radiation.",
    ],
    symbols={"R_i": "R_{i}", "R_w": "R_{w}", "R_o": "R_{o}"},
)
def conduction_cylinder_wall(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.let("R_i", "1 / (2 * pi * r_i * h_i)", "The fluid film on the inner surface has this thermal resistance for "
          "one metre of length.", unit="m*K/W")
    c.let("R_w", "log(r_o / r_i) / (2 * pi * k)", "The wall has this thermal resistance for one metre of length.",
          unit="m*K/W")
    c.let("R_o", "1 / (2 * pi * r_o * h_o)", "The fluid film on the outer surface has this thermal resistance for "
          "one metre of length.", unit="m*K/W")
    c.let("heat_flow_per_length", "(T_i - T_o) / (R_i + R_w + R_o)",
          "The three resistances in series give the heat flow from the inner fluid to the outer fluid.")
    c.let("T_wall_inner", "T_i - heat_flow_per_length * R_i",
          "The temperature drop in the inner fluid film gives the temperature of the inner surface.")
    c.let("T_wall_outer", "T_o + heat_flow_per_length * R_o",
          "The temperature drop in the outer fluid film gives the temperature of the outer surface.")
    c.let("dT_wall", "heat_flow_per_length * R_w",
          "The heat flow through the resistance of the wall gives the temperature difference across the wall.")


RE_MIN, PR_MIN, PR_MAX = 1.0e4, 0.7, 160.0      # range of the Dittus-Boelter correlation [Inc, equation 8.60]


def _dittus_boelter_validity(v) -> list[str]:
    out = []
    if v["reynolds"] < RE_MIN:
        out.append(f"The Reynolds number is {shown(v['reynolds'], '')}. The Dittus-Boelter correlation is for "
                   f"turbulent flow with a Reynolds number of {RE_MIN:g} or more.")
    if not (PR_MIN <= v["prandtl"] <= PR_MAX):
        out.append(f"The Prandtl number is {shown(v['prandtl'], '')}. The Dittus-Boelter correlation is for Prandtl "
                   f"numbers from {PR_MIN:g} to {PR_MAX:g}.")
    if min(abs(v["n"] - 0.4), abs(v["n"] - 0.3)) > 1e-9:
        out.append(f"The exponent n is {v['n']:g}. The Dittus-Boelter correlation has n = 0.4 when the wall heats "
                   f"the fluid and n = 0.3 when the wall cools the fluid.")
    return out


@model(
    name="dittus_boelter",
    title="Heat transfer coefficient of turbulent flow in a tube (Dittus-Boelter correlation)",
    reference=f"{_INC}, section 8.5, equation 8.60 (Dittus-Boelter), with its range of application.",
    inputs=[
        Input("mdot", "Mass flow in the tube", "kg/s", symbol=r"\dot{m}"),
        Input("d", "Inner diameter of the tube", "m"),
        Input("mu", "Dynamic viscosity of the fluid", "Pa*s", symbol=r"\mu"),
        Input("k", "Thermal conductivity of the fluid", _K),
        Input("cp", "Specific heat capacity of the fluid", "J/(kg*K)", symbol=r"c_{p}"),
        Input("n", "Exponent of the Prandtl number: 0.4 when the wall heats the fluid, 0.3 when the wall cools the "
              "fluid", ""),
    ],
    outputs=[
        Output("reynolds", "Reynolds number of the flow", "", r"\mathrm{Re}"),
        Output("prandtl", "Prandtl number of the fluid", "", r"\mathrm{Pr}"),
        Output("nusselt", "Nusselt number", "", r"\mathrm{Nu}"),
        Output("h", "Heat transfer coefficient between the fluid and the wall", _H, "h"),
    ],
    assumptions=[
        "The flow is turbulent and fully developed: the tube is longer than 10 diameters.",
        "The tube is circular and smooth.",
        "The fluid properties are those at the mean temperature of the fluid.",
        "The difference between the wall temperature and the fluid temperature is moderate.",
    ],
    validity=_dittus_boelter_validity,
)
def dittus_boelter(c: Calc) -> None:
    c.let("reynolds", "4 * mdot / (pi * d * mu)", "The mass flow, the diameter and the viscosity give the Reynolds "
          "number.")
    c.let("prandtl", "cp * mu / k", "The fluid properties give the Prandtl number.")
    c.let("nusselt", "0.023 * reynolds**0.8 * prandtl**n",
          "The Dittus-Boelter correlation gives the Nusselt number of the turbulent flow.")
    c.let("h", "nusselt * k / d", "The Nusselt number, the conductivity and the diameter give the heat transfer "
          "coefficient.")


__all__ = ["fin_convective_tip", "thermal_stress_constrained", "hollow_cylinder_radial_gradient",
           "conduction_cylinder_wall", "dittus_boelter", "BIOT_LIMIT", "RE_MIN", "PR_MIN", "PR_MAX"]
