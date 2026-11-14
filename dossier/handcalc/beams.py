"""Beams, columns and bolt circles.

Sources
    [GG]    J. M. Gere and B. J. Goodno, Mechanics of Materials, 8th edition, Cengage Learning, 2013.
            Section 5.5 (the flexure formula, sigma = M y / I, and the section modulus S = I / c) and section
            5.12 (beams with axial loads: sigma = N / A + M y / I).
    [Roark] W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill,
            2002. Section 8.1 (straight beams) and Table A.1 (properties of sections: the hollow circle has
            A = pi (R^2 - R_i^2) and I = pi (R^4 - R_i^4) / 4).
    [TGe]   S. P. Timoshenko and J. M. Gere, Theory of Elastic Stability, 2nd edition, McGraw-Hill, 1961,
            article 2.1 (Euler's column formula, and the reduced length for other end conditions).
    [Shig]  R. G. Budynas and J. K. Nisbett, Shigley's Mechanical Engineering Design, 10th edition,
            McGraw-Hill, 2015. Sections 4-12 and 4-13: the Euler formula applies when the critical stress is
            not more than one half of the yield strength; the Johnson formula applies above that.
    [BY]    L. E. Brownell and E. H. Young, Process Equipment Design, Wiley, 1959, chapter 10: the largest
            bolt load of a bolt circle with a moment M is 4 M / (n d) for n bolts on the diameter d.

The bolt circle. A rigid flange that turns about the centre line of the bolt circle gives each bolt a force in
proportion to its distance y_i from that line: F_i = M y_i / sum(y_j^2). For n >= 3 bolts with equal pitch
on the radius r_b, sum(y_j^2) = n r_b^2 / 2 for any direction of the moment, thus the largest force is
2 M / (n r_b) when a bolt is at the largest distance.

Verification (tests): the stresses of the tube against numerical integrals of the stress over the ring
(force and moment equilibrium); the bolt forces against a direct solution of the equilibrium of a rigid
flange on n springs; the Euler load against the lowest eigenvalue of a finite-element model of the column
for four end conditions.
"""

from __future__ import annotations

from dossier.handcalc.base import Calc, Input, Output, model, shown

_GG = "J. M. Gere and B. J. Goodno, Mechanics of Materials, 8th edition, Cengage Learning, 2013"
_ROARK = "W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill, 2002"

_BEAM = "The part is a slender beam: plane cross-sections stay plane."
_ELASTIC = "The material is linear elastic."


def _tube_validity(v) -> list[str]:
    if not (v["r_i"] <= v["r_eval"] <= v["r_o"]):
        return [f"The radius r_eval, {shown(v['r_eval'], 'm')}, is not in the wall. The wall is from "
                f"{shown(v['r_i'], 'm')} to {shown(v['r_o'], 'm')}."]
    return []


@model(
    name="tube_bending_axial",
    title="Tube with a moment and an axial force (beam theory)",
    reference=f"{_GG}, section 5.5 (the flexure formula) and section 5.12 (beams with axial loads). "
              f"{_ROARK}, Table A.1 (area and second moment of area of the hollow circle).",
    inputs=[
        Input("M", "Moment on the cross-section", "N*m", sign="any"),
        Input("F", "Axial force on the cross-section (tension is positive)", "N", sign="any"),
        Input("r_o", "Outer radius of the tube", "m"),
        Input("r_i", "Inner radius of the tube (zero for a solid bar)", "m", sign="nonnegative"),
        Input("r_eval", "Radius at which the model gives the stress", "m", symbol=r"r_{e}"),
    ],
    outputs=[
        Output("sigma_bending", "Stress from the moment at the radius r_eval", "Pa", r"\sigma_{b}"),
        Output("sigma_bending_outer", "Stress from the moment at the outer surface", "Pa", r"\sigma_{b,o}"),
        Output("sigma_axial", "Stress from the axial force", "Pa", r"\sigma_{a}"),
        Output("sigma_tension", "Axial stress at r_eval on the side that the moment puts in tension", "Pa",
               r"\sigma_{t}"),
        Output("sigma_compression", "Axial stress at r_eval on the side that the moment puts in compression", "Pa",
               r"\sigma_{c}"),
    ],
    assumptions=[
        _BEAM,
        "The cross-section is a circular ring with constant radii.",
        _ELASTIC,
        "The section is far from the ends, the supports and the points where the loads act.",
        "The axial force acts on the axis of the tube.",
        "The model gives the axial stress only. It has no shear stress and no hoop stress.",
    ],
    validity=_tube_validity,
    symbols={"A": "A", "I": "I"},
)
def tube_bending_axial(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.let("A", "pi * (r_o**2 - r_i**2)", "The two radii give the area of the cross-section.", unit="m^2")
    c.let("I", "pi * (r_o**4 - r_i**4) / 4", "The two radii give the second moment of area of the cross-section.",
          unit="m^4", show="mm^4")
    c.let("sigma_axial", "F / A", "The axial force gives a constant stress on the cross-section.")
    c.let("sigma_bending", "abs(M) * r_eval / I",
          "The flexure formula gives the stress from the moment at the radius r_eval.")
    c.let("sigma_bending_outer", "abs(M) * r_o / I",
          "The flexure formula gives the largest stress from the moment, at the outer surface.")
    c.let("sigma_tension", "sigma_axial + sigma_bending",
          "The sum is the stress at r_eval on the side that the moment puts in tension.")
    c.let("sigma_compression", "sigma_axial - sigma_bending",
          "The difference is the stress at r_eval on the side that the moment puts in compression.")


@model(
    name="beam_bending",
    title="Beam with a moment (section modulus)",
    reference=f"{_GG}, section 5.5 (the flexure formula with the section modulus, sigma = M / S).",
    inputs=[
        Input("M", "Moment on the cross-section", "N*m", sign="any"),
        Input("Z", "Section modulus of the cross-section for the outer fibre", "m^3"),
    ],
    outputs=[
        Output("sigma_bending", "Largest stress from the moment, at the outer fibre", "Pa", r"\sigma_{b}"),
    ],
    assumptions=[
        _BEAM,
        _ELASTIC,
        "The moment acts about a principal axis of the cross-section.",
        "The section is far from the supports and the points where the loads act.",
    ],
)
def beam_bending(c: Calc) -> None:
    c.let("sigma_bending", "abs(M) / Z", "The flexure formula gives the largest stress from the moment.")


@model(
    name="bolt_circle_loads",
    title="Bolt circle with an axial force and a moment (largest bolt load)",
    reference="L. E. Brownell and E. H. Young, Process Equipment Design, Wiley, 1959, chapter 10: the largest bolt "
              "load of a bolt circle with a moment is 4 M / (n d), with the diameter d of the bolt circle. The "
              "docstring of dossier.handcalc.beams gives the derivation.",
    inputs=[
        Input("F", "Axial force on the joint (tension is positive)", "N", sign="any"),
        Input("M", "Moment on the joint", "N*m", sign="any"),
        Input("n", "Number of bolts", ""),
        Input("r_b", "Radius of the bolt circle", "m"),
    ],
    outputs=[
        Output("force_axial", "External load at each bolt from the axial force", "N", r"F_{b,F}"),
        Output("force_moment", "Largest external load at a bolt from the moment", "N", r"F_{b,M}"),
        Output("force_max", "Largest external load at a bolt", "N", r"F_{b,\max}"),
        Output("force_min", "Smallest external load at a bolt", "N", r"F_{b,\min}"),
    ],
    assumptions=[
        "The bolts have the same size and an equal pitch on one circle.",
        "The flange is rigid. The joint stays closed, and it turns about the centre line of the bolt circle.",
        "One bolt is at the largest distance from that centre line. This gives the largest load for any "
        "direction of the moment.",
        "The model gives the external load at each bolt. The preload and the stiffness of the joint are not in "
        "the model.",
    ],
)
def bolt_circle_loads(c: Calc) -> None:
    n = c["n"]
    c.require(n >= 3 and float(n).is_integer(),
              f"The number of bolts n must be a whole number of 3 or more. It is {n:g}.")
    c.let("force_axial", "F / n", "Each bolt location carries an equal part of the axial force.")
    c.let("force_moment", "2 * abs(M) / (n * r_b)",
          "The moment gives each bolt a load in proportion to its distance from the centre line, and this is "
          "the largest one.")
    c.let("force_max", "force_axial + force_moment", "The sum is the load at the bolt on the tension side of the moment.")
    c.let("force_min", "force_axial - force_moment",
          "The difference is the load at the bolt on the compression side of the moment.")


def _euler_validity(v) -> list[str]:
    if v["sigma_cr"] > 0.5 * v["Sy"]:
        return [f"The column is not slender: the Euler stress, {shown(v['sigma_cr'], 'Pa')}, is more than one half "
                f"of the yield strength, {shown(0.5 * v['Sy'], 'Pa')}. The column fails at a smaller load. Use the "
                f"Johnson formula."]
    return []


@model(
    name="euler_buckling",
    title="Elastic stability of a column (Euler formula)",
    reference="S. P. Timoshenko and J. M. Gere, Theory of Elastic Stability, 2nd edition, McGraw-Hill, 1961, "
              "article 2.1 (Euler's column formula, with the reduced length for other end conditions). "
              "R. G. Budynas and J. K. Nisbett, Shigley's Mechanical Engineering Design, 10th edition, McGraw-Hill, "
              "2015, sections 4-12 and 4-13 (the limit of the Euler formula).",
    inputs=[
        Input("E", "Modulus of elasticity", "Pa", default="E"),
        Input("I", "Smallest second moment of area of the cross-section", "m^4"),
        Input("A", "Area of the cross-section", "m^2"),
        Input("L", "Length of the column", "m"),
        Input("K", "Effective length factor: 1 for two pinned ends, 2 for one fixed and one free end, 0.5 for two "
              "fixed ends, 0.7 for one fixed and one pinned end", ""),
        Input("Sy", "Yield strength (for the limit of the Euler formula)", "Pa", default="Sy", symbol="S_{y}"),
    ],
    outputs=[
        Output("P_cr", "Critical axial load of the column", "N", r"P_{\mathrm{cr}}"),
        Output("sigma_cr", "Mean stress at the critical load", "Pa", r"\sigma_{\mathrm{cr}}"),
        Output("slenderness", "Slenderness ratio: effective length divided by the radius of gyration", "",
               r"\lambda"),
    ],
    assumptions=[
        "The column is straight and has a constant cross-section.",
        "The load acts on the axis of the column.",
        _ELASTIC,
        "The column bends about the axis with the smallest second moment of area.",
    ],
    validity=_euler_validity,
    symbols={"L_e": "L_{e}", "i_g": "i"},
)
def euler_buckling(c: Calc) -> None:
    c.let("L_e", "K * L", "The end conditions give the effective length of the column.", unit="m")
    c.let("P_cr", "pi**2 * E * I / L_e**2", "The Euler formula gives the critical load of the column.")
    c.let("sigma_cr", "P_cr / A", "The critical load on the area of the cross-section gives the Euler stress.")
    c.let("i_g", "sqrt(I / A)", "The second moment of area and the area give the radius of gyration.", unit="m")
    c.let("slenderness", "L_e / i_g", "The effective length and the radius of gyration give the slenderness ratio.")


__all__ = ["tube_bending_axial", "beam_bending", "bolt_circle_loads", "euler_buckling"]
