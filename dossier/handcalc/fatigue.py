"""Fatigue: the stress-life relation of Basquin with the mean-stress correction of Goodman.

Sources
    [Dow]   N. E. Dowling, Mechanical Behavior of Materials, 4th edition, Pearson, 2013, chapter 9 (Fatigue of
            materials: introduction and stress-based approach): the stress-life curve
            sigma_a = sigma_f' (2 N_f)^b, and the modified Goodman relation
            sigma_a / sigma_ar + sigma_m / sigma_u = 1 for the equivalent amplitude sigma_ar at zero mean stress.
    [Basq]  O. H. Basquin, The exponential law of endurance tests, Proceedings of the American Society for
            Testing and Materials, volume 10, 1910, pages 625 to 630.

The convention is that of dossier.model.Fatigue: the stress amplitude at zero mean stress is
S_a = sigma_f (2 N)^b, with b negative, thus N = (S_a / sigma_f)^(1 / b) / 2. The inputs sigma_f and b come
from the fatigue data of the subject's material when the specification does not give them.

A mean stress that is not tensile gives no benefit: the equivalent amplitude is then the stress amplitude.
The relation has no endurance limit.

Verification (tests): a root solve of the stress-life curve on the Goodman line (scipy.optimize.brentq), and
a case with whole numbers (sigma_f = 1000 MPa, b = -0.1, amplitude 500 MPa: 2 N = 2^10).
"""

from __future__ import annotations

from dossier.handcalc.base import Calc, Input, Output, model, shown

LOW_CYCLE_LIMIT = 1000.0        # under this number of cycles the stress-life relation is not applicable


def _basquin_validity(v) -> list[str]:
    out = []
    if v["cycles"] < LOW_CYCLE_LIMIT:
        out.append(f"The life is {shown(v['cycles'], '')} cycles, which is less than {LOW_CYCLE_LIMIT:g} cycles. "
                   f"The stress-life relation is for high-cycle fatigue. Use a strain-life analysis for this range.")
    top = v["sigma_a"] + v["sigma_m"]
    if top > v["Su"]:
        out.append(f"The largest stress of the cycle, {shown(top, 'Pa')}, is more than the ultimate strength, "
                   f"{shown(v['Su'], 'Pa')}.")
    return out


@model(
    name="basquin_life",
    title="Fatigue life from the stress amplitude and the mean stress (Basquin and Goodman)",
    reference="N. E. Dowling, Mechanical Behavior of Materials, 4th edition, Pearson, 2013, chapter 9: the "
              "stress-life curve sigma_a = sigma_f' (2 N_f)^b and the modified Goodman relation. O. H. Basquin, "
              "Proceedings of the American Society for Testing and Materials, volume 10, 1910, pages 625 to 630.",
    inputs=[
        Input("sigma_a", "Stress amplitude of the cycle", "Pa", symbol=r"\sigma_{a}"),
        Input("sigma_m", "Mean stress of the cycle (tension is positive)", "Pa", sign="any", symbol=r"\sigma_{m}"),
        Input("sigma_f", "Fatigue strength coefficient of the material", "Pa", default="fatigue.sigma_f",
              symbol=r"\sigma_{f}"),
        Input("b", "Fatigue strength exponent of the material (negative)", "", default="fatigue.b", sign="any"),
        Input("Su", "Ultimate strength of the material", "Pa", default="Su", symbol="S_{u}"),
    ],
    outputs=[
        Output("sigma_ar", "Equivalent stress amplitude at zero mean stress", "Pa", r"\sigma_{ar}"),
        Output("cycles", "Number of cycles to failure", "", "N"),
    ],
    assumptions=[
        "The stress cycle has a constant amplitude and a constant mean stress.",
        "The stress is uniaxial, or it is an equivalent uniaxial stress.",
        "The fatigue constants are for the material, the surface and the temperature of the part.",
        "The relation has no endurance limit: each cycle causes damage.",
        "A mean stress that is not tensile gives no benefit.",
    ],
    validity=_basquin_validity,
)
def basquin_life(c: Calc) -> None:
    c.require(c["b"] < 0.0, f"The fatigue strength exponent b must be negative. It is {c['b']:g}.")
    c.require(c["sigma_m"] < c["Su"], f"The mean stress ({c.show('sigma_m')}) must be less than the ultimate "
                                      f"strength Su ({c.show('Su')}).")
    if c["sigma_m"] > 0.0:
        c.let("sigma_ar", "sigma_a / (1 - sigma_m / Su)",
              "The Goodman relation gives the equivalent stress amplitude at zero mean stress.")
    else:
        c.let("sigma_ar", "sigma_a",
              "The mean stress is not tensile, thus the equivalent stress amplitude is the stress amplitude.")
    c.let("cycles", "(sigma_ar / sigma_f)**(1 / b) / 2",
          "The Basquin relation gives the number of cycles to failure.")


__all__ = ["basquin_life", "LOW_CYCLE_LIMIT"]
