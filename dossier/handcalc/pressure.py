"""Pressure vessels: thick and thin cylinders, the sphere.

Sources
    [TG]    S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd edition, McGraw-Hill, 1970.
            Article 28, "Stress distribution symmetrical about an axis": the Lamé solution of the hollow
            cylinder. Article "Symmetrical stress distribution in a sphere" (chapter 12): the hollow sphere.
    [Roark] W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill,
            2002. Table 13.1, case 1c (thin cylinder with closed ends); Table 13.5, cases 1a and 1b (thick
            cylinder) and case 2a (thick sphere); section 13.1 (a wall is thin when its thickness is less
            than about one tenth of the radius).

Mean stresses. A stress classification line gives the mean of a stress along a radial line through the wall
(its membrane stress). For the cylinder this mean follows from the equilibrium of one half of the cylinder,
p r_i / (r_o - r_i), for any wall thickness. For the sphere the equilibrium of one half gives the mean over
the cut area, p r_i^2 / (r_o^2 - r_i^2), which is not the mean along a line; `sphere_pressure` gives both.

Verification (tests/dossier/test_dossier_handcalc.py): the Lamé stresses against the exact solutions of
feacore.problems.analytical.lame, the mean stresses against numerical integrals of those solutions, the
thin-wall formulas against the Lamé solution of a thin wall.
"""

from __future__ import annotations

from dossier.handcalc.base import Calc, Input, Output, model, shown

_TG = "S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd edition, McGraw-Hill, 1970"
_ROARK = "W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill, 2002"

_ELASTIC = "The material is linear elastic and isotropic."


@model(
    name="thick_cylinder_pressure",
    title="Thick-walled cylinder with internal pressure (Lamé solution)",
    reference=f"{_TG}, article 28 (Stress distribution symmetrical about an axis): the Lamé solution. "
              f"{_ROARK}, Table 13.5, cases 1a and 1b. The mean hoop stress and the axial stress follow "
              f"from equilibrium.",
    inputs=[
        Input("p", "Internal pressure", "Pa", sign="any"),
        Input("r_i", "Inner radius of the cylinder", "m"),
        Input("r_o", "Outer radius of the cylinder", "m"),
    ],
    outputs=[
        Output("hoop_inner", "Hoop stress at the inner surface (the largest hoop stress)", "Pa", r"\sigma_{\theta,i}"),
        Output("hoop_outer", "Hoop stress at the outer surface", "Pa", r"\sigma_{\theta,o}"),
        Output("hoop_mean", "Mean hoop stress across the wall (the membrane stress)", "Pa", r"\sigma_{\theta,m}"),
        Output("radial_inner", "Radial stress at the inner surface", "Pa", r"\sigma_{r,i}"),
        Output("axial_closed", "Axial stress of the cylinder with closed ends", "Pa", r"\sigma_{z}"),
        Output("von_mises_inner", "Von Mises stress at the inner surface of the cylinder with closed ends", "Pa",
               r"\sigma_{\mathrm{vM},i}"),
    ],
    assumptions=[
        "The cylinder is long: its ends do not change the stress at the section.",
        "The wall has a constant thickness and no holes.",
        _ELASTIC,
        "The pressure acts on the inner surface. The outer surface has no load.",
        "The axial stress and the von Mises stress are for a cylinder with closed ends.",
    ],
)
def thick_cylinder_pressure(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.let("hoop_inner", "p * (r_o**2 + r_i**2) / (r_o**2 - r_i**2)",
          "The Lamé solution gives the hoop stress at the inner surface, which is the largest hoop stress.")
    c.let("hoop_outer", "2 * p * r_i**2 / (r_o**2 - r_i**2)",
          "The Lamé solution gives the hoop stress at the outer surface.")
    c.let("hoop_mean", "p * r_i / (r_o - r_i)",
          "The equilibrium of one half of the cylinder gives the mean hoop stress across the wall.")
    c.let("radial_inner", "-p", "The radial stress at the inner surface is the pressure, as a compressive stress.")
    c.let("axial_closed", "p * r_i**2 / (r_o**2 - r_i**2)",
          "The pressure force on the closed ends gives the axial stress in the wall.")
    c.let("von_mises_inner",
          "sqrt(((hoop_inner - radial_inner)**2 + (hoop_inner - axial_closed)**2 + (axial_closed - radial_inner)**2) / 2)",
          "The hoop, radial and axial stresses at the inner surface give the von Mises stress of the closed cylinder.")


def _thin_cylinder_validity(v) -> list[str]:
    ratio = v["r"] / v["t"]
    if ratio < 10.0:
        return [f"The wall is not thin: the ratio r / t is {ratio:.3g}. The thin-wall formulas need a ratio of 10 "
                f"or more. Use the model thick_cylinder_pressure."]
    return []


@model(
    name="thin_cylinder_pressure",
    title="Thin-walled cylinder with internal pressure (membrane stress)",
    reference=f"{_ROARK}, Table 13.1, case 1c (cylinder with uniform internal pressure and closed ends), and "
              f"section 13.1 for the limit of a thin wall.",
    inputs=[
        Input("p", "Internal pressure", "Pa", sign="any"),
        Input("r", "Mean radius of the wall", "m"),
        Input("t", "Wall thickness", "m"),
        Input("E", "Modulus of elasticity", "Pa", default="E"),
        Input("nu", "Poisson's ratio", "", default="nu", sign="nonnegative", symbol=r"\nu"),
    ],
    outputs=[
        Output("hoop", "Hoop stress in the wall", "Pa", r"\sigma_{\theta}"),
        Output("axial_closed", "Axial stress of the cylinder with closed ends", "Pa", r"\sigma_{z}"),
        Output("von_mises_closed", "Von Mises stress of the cylinder with closed ends", "Pa", r"\sigma_{\mathrm{vM}}"),
        Output("radial_growth_closed", "Increase of the radius of the cylinder with closed ends", "m", r"\Delta r"),
    ],
    assumptions=[
        "The wall is thin: the stress is constant across the wall.",
        "The radial stress is negligible in comparison with the hoop stress.",
        "The cylinder is long: its ends do not change the stress at the section.",
        _ELASTIC,
    ],
    validity=_thin_cylinder_validity,
)
def thin_cylinder_pressure(c: Calc) -> None:
    c.require(c["nu"] <= 0.5, f"Poisson's ratio nu must be 0.5 or less. It is {c.show('nu')}.")
    c.let("hoop", "p * r / t", "The equilibrium of one half of the cylinder gives the hoop stress of the thin wall.")
    c.let("axial_closed", "p * r / (2 * t)", "The pressure force on the closed ends gives the axial stress.")
    c.let("von_mises_closed", "sqrt(hoop**2 - hoop * axial_closed + axial_closed**2)",
          "The hoop stress and the axial stress give the von Mises stress of the wall.")
    c.let("radial_growth_closed", "r * (hoop - nu * axial_closed) / E",
          "The hoop strain from Hooke's law gives the increase of the radius.")


@model(
    name="sphere_pressure",
    title="Thick-walled sphere with internal pressure (Lamé solution)",
    reference=f"{_TG}, article \"Symmetrical stress distribution in a sphere\". {_ROARK}, Table 13.5, case 2a. "
              f"The two mean stresses are integrals of the Lamé hoop stress.",
    inputs=[
        Input("p", "Internal pressure", "Pa", sign="any"),
        Input("r_i", "Inner radius of the sphere", "m"),
        Input("r_o", "Outer radius of the sphere", "m"),
    ],
    outputs=[
        Output("hoop_inner", "Hoop stress at the inner surface (the largest hoop stress)", "Pa", r"\sigma_{\theta,i}"),
        Output("hoop_outer", "Hoop stress at the outer surface", "Pa", r"\sigma_{\theta,o}"),
        Output("hoop_mean", "Mean hoop stress along a radial line through the wall (the membrane stress)", "Pa",
               r"\sigma_{\theta,m}"),
        Output("hoop_mean_area", "Mean hoop stress over the cut area of one half of the sphere", "Pa",
               r"\sigma_{\theta,A}"),
        Output("radial_inner", "Radial stress at the inner surface", "Pa", r"\sigma_{r,i}"),
        Output("von_mises_inner", "Von Mises stress at the inner surface", "Pa", r"\sigma_{\mathrm{vM},i}"),
    ],
    assumptions=[
        "The sphere is complete, with a constant wall thickness and no holes.",
        _ELASTIC,
        "The pressure acts on the inner surface. The outer surface has no load.",
    ],
)
def sphere_pressure(c: Calc) -> None:
    c.require(c["r_o"] > c["r_i"], f"The outer radius r_o ({c.show('r_o')}) must be more than the inner radius "
                                   f"r_i ({c.show('r_i')}).")
    c.let("hoop_inner", "p * (r_o**3 + 2 * r_i**3) / (2 * (r_o**3 - r_i**3))",
          "The Lamé solution gives the hoop stress at the inner surface, which is the largest hoop stress.")
    c.let("hoop_outer", "3 * p * r_i**3 / (2 * (r_o**3 - r_i**3))",
          "The Lamé solution gives the hoop stress at the outer surface.")
    c.let("hoop_mean", "p * r_i * (4 * r_i**2 + r_i * r_o + r_o**2) / (4 * (r_o - r_i) * (r_i**2 + r_i * r_o + r_o**2))",
          "The mean of the Lamé hoop stress along a radial line through the wall is the membrane stress.")
    c.let("hoop_mean_area", "p * r_i**2 / (r_o**2 - r_i**2)",
          "The equilibrium of one half of the sphere gives the mean hoop stress over the cut area.")
    c.let("radial_inner", "-p", "The radial stress at the inner surface is the pressure, as a compressive stress.")
    c.let("von_mises_inner", "abs(hoop_inner - radial_inner)",
          "The two equal hoop stresses and the radial stress give the von Mises stress at the inner surface.")


__all__ = ["thick_cylinder_pressure", "thin_cylinder_pressure", "sphere_pressure", "shown"]
