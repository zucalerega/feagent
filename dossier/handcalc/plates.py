"""Plates: the circular plate with uniform pressure (thin-plate theory).

Sources
    [TWK]   S. P. Timoshenko and S. Woinowsky-Krieger, Theory of Plates and Shells, 2nd edition, McGraw-Hill,
            1959. Article 16, "Uniformly loaded circular plates": deflection and moments of the plate with a
            clamped edge and of the plate with a simply supported edge.
    [Roark] W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill,
            2002. Table 11.2, case 10a (simply supported) and case 10b (fixed), with their coefficients for
            a Poisson's ratio of 0.3. Section 11.1 gives the limits of the theory: the thickness is not more
            than about one quarter of the least transverse dimension, and the deflection is not more than
            about one half of the thickness.
    [VK]    E. Ventsel and T. Krauthammer, Thin Plates and Shells, Marcel Dekker, 2001, section 1.1: a plate
            is thin when its least in-plane dimension is 8 to 10 times its thickness, or more.

The bending stress is 6 M / t^2 with the moment per unit length M:
    clamped edge            M_centre = (1 + nu) p a^2 / 16,   M_edge = p a^2 / 8 (radial),  w = p a^4 / (64 D)
    simply supported edge   M_centre = (3 + nu) p a^2 / 16,   w = (5 + nu) p a^4 / (64 (1 + nu) D)
A real edge is between the two. The centre stress of the simply supported plate is (3 + nu) / (1 + nu) times
that of the clamped plate (2.54 for nu = 0.3).

Verification (tests): the coefficients of Roark's Table 11.2 for nu = 0.3, and a numerical solution of the
axisymmetric plate equation (scipy.integrate.solve_bvp) for both edge conditions.
"""

from __future__ import annotations

from dossier.handcalc.base import Calc, Input, Output, model, shown

_TWK = ("S. P. Timoshenko and S. Woinowsky-Krieger, Theory of Plates and Shells, 2nd edition, McGraw-Hill, 1959, "
        "article 16 (Uniformly loaded circular plates)")
_ROARK = ("W. C. Young and R. G. Budynas, Roark's Formulas for Stress and Strain, 7th edition, McGraw-Hill, 2002, "
          "Table 11.2, cases 10a and 10b")

THIN_LIMIT = 4.0        # diameter / thickness under which thin-plate theory is not applicable [Roark 11.1]
THIN_GOOD = 10.0        # diameter / thickness from which a plate is thin [VK 1.1]


def _plate_validity(v) -> list[str]:
    out = []
    ratio = 2.0 * v["a"] / v["t"]
    if ratio < THIN_LIMIT:
        out.append(f"The plate is thick: its diameter is {ratio:.3g} times its thickness. The thin-plate theory "
                   f"needs a ratio of {THIN_LIMIT:g} or more. The results are only a first estimate.")
    elif ratio < THIN_GOOD:
        out.append(f"The plate is moderately thick: its diameter is {ratio:.3g} times its thickness. A thin plate "
                   f"has a ratio of {THIN_GOOD:g} or more. The thin-plate theory has no shear deflection: the true "
                   f"deflection is larger. The stress is an estimate.")
    half = 0.5 * v["t"]
    for name, edge in (("deflection_center_clamped", "clamped"), ("deflection_center_simply_supported",
                                                                 "simply supported")):
        if abs(v[name]) > half:
            out.append(f"The deflection of the plate with a {edge} edge is {shown(abs(v[name]), 'm')}. This is more "
                       f"than one half of the thickness, {shown(half, 'm')}. The linear theory has no membrane "
                       f"stress: the true deflection is smaller.")
    return out


@model(
    name="circular_plate_pressure",
    title="Circular plate with uniform pressure (thin-plate theory)",
    reference=f"{_TWK}. {_ROARK}.",
    inputs=[
        Input("p", "Pressure on one side of the plate", "Pa", sign="any"),
        Input("a", "Radius of the plate", "m"),
        Input("t", "Thickness of the plate", "m"),
        Input("nu", "Poisson's ratio", "", default="nu", sign="nonnegative", symbol=r"\nu"),
        Input("E", "Modulus of elasticity", "Pa", default="E"),
    ],
    outputs=[
        Output("sigma_center_clamped", "Stress at the centre of the plate with a clamped edge, on the side without "
               "pressure", "Pa", r"\sigma_{c,\mathrm{cl}}"),
        Output("sigma_edge_clamped", "Radial stress at the clamped edge, on the side with pressure (the largest "
               "stress of the clamped plate)", "Pa", r"\sigma_{e,\mathrm{cl}}"),
        Output("sigma_center_simply_supported", "Stress at the centre of the plate with a simply supported edge, "
               "on the side without pressure", "Pa", r"\sigma_{c,\mathrm{ss}}"),
        Output("deflection_center_clamped", "Deflection at the centre of the plate with a clamped edge", "m",
               r"w_{\mathrm{cl}}"),
        Output("deflection_center_simply_supported", "Deflection at the centre of the plate with a simply supported "
               "edge", "m", r"w_{\mathrm{ss}}"),
    ],
    assumptions=[
        "The plate is flat and thin, with a constant thickness.",
        "The deflection is small in comparison with the thickness: the plate has no membrane stress.",
        "The pressure is uniform on one side of the plate.",
        "The edge is a circle. It has a clamped support or a simple support. A real edge is between the two.",
        "The material is linear elastic and isotropic.",
        "The stresses are the surface stresses from the moments. They are tensile on the side that the "
        "description of each output gives.",
    ],
    validity=_plate_validity,
    symbols={"D": "D"},
)
def circular_plate_pressure(c: Calc) -> None:
    c.require(c["nu"] <= 0.5, f"Poisson's ratio nu must be 0.5 or less. It is {c.show('nu')}.")
    c.let("D", "E * t**3 / (12 * (1 - nu**2))",
          "The thickness and the elastic constants give the flexural rigidity of the plate.", unit="N*m")
    c.let("sigma_center_clamped", "3 * (1 + nu) * p * a**2 / (8 * t**2)",
          "The moment at the centre of the plate with a clamped edge gives the stress at the centre.")
    c.let("sigma_edge_clamped", "3 * p * a**2 / (4 * t**2)",
          "The radial moment at the clamped edge gives the largest stress of the clamped plate.")
    c.let("sigma_center_simply_supported", "3 * (3 + nu) * p * a**2 / (8 * t**2)",
          "The moment at the centre of the plate with a simply supported edge gives the stress at the centre.")
    c.let("deflection_center_clamped", "p * a**4 / (64 * D)",
          "The solution of the plate equation gives the deflection at the centre of the clamped plate.")
    c.let("deflection_center_simply_supported", "(5 + nu) * p * a**4 / (64 * (1 + nu) * D)",
          "The solution of the plate equation gives the deflection at the centre of the simply supported plate.")


__all__ = ["circular_plate_pressure", "THIN_LIMIT", "THIN_GOOD"]
