"""Exact quasi-one-dimensional flow of a perfect gas through a duct of slowly varying area: the isentropic
relations, the area-Mach relation and the normal-shock relations (NACA Report 1135, eqs. 44, 45, 80, 93, 94,
96, 99), solved with scipy root finding. The reference solution of the converging-diverging nozzle
(feacore/problems/compressible/nozzle.py); tests/test_compressible_internal.py checks every function against
values tabulated in NACA Report 1135."""

from __future__ import annotations

import math
from dataclasses import dataclass

NACA1135 = ("NACA Report 1135, 'Equations, tables, and charts for compressible flow', Ames Research Staff (1953): "
            "isentropic relations eqs. 44-45, area-Mach relation eq. 80, normal shock eqs. 93-99")


def p_ratio(M: float, g: float = 1.4) -> float:
    """p / p0 of an isentropic flow at Mach M."""
    return (1.0 + 0.5 * (g - 1.0) * M * M) ** (-g / (g - 1.0))


def T_ratio(M: float, g: float = 1.4) -> float:
    """T / T0."""
    return 1.0 / (1.0 + 0.5 * (g - 1.0) * M * M)


def area_ratio(M: float, g: float = 1.4) -> float:
    """A / A* (the area-Mach relation)."""
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area(a: float, supersonic: bool, g: float = 1.4) -> float:
    """The Mach number at A / A* = a (>= 1) on the subsonic or the supersonic branch."""
    from scipy.optimize import brentq

    if a < 1.0 - 1e-12:
        raise ValueError(f"A / A* = {a} < 1")
    if a <= 1.0:
        return 1.0
    if supersonic:
        return brentq(lambda M: area_ratio(M, g) - a, 1.0, 50.0, xtol=1e-14, rtol=1e-14)
    return brentq(lambda M: area_ratio(M, g) - a, 1e-8, 1.0, xtol=1e-14, rtol=1e-14)


def mach_from_p(pr: float, g: float = 1.4) -> float:
    """The Mach number at p / p0 = pr (isentropic)."""
    return math.sqrt(2.0 / (g - 1.0) * (pr ** (-(g - 1.0) / g) - 1.0))


def shock_p_ratio(M: float, g: float = 1.4) -> float:
    """p2 / p1 across a normal shock at upstream Mach M."""
    return 1.0 + 2.0 * g / (g + 1.0) * (M * M - 1.0)


def shock_mach(M: float, g: float = 1.4) -> float:
    """Mach number behind a normal shock."""
    return math.sqrt((1.0 + 0.5 * (g - 1.0) * M * M) / (g * M * M - 0.5 * (g - 1.0)))


def shock_p0_ratio(M: float, g: float = 1.4) -> float:
    """p02 / p01 across a normal shock."""
    a = ((g + 1.0) * M * M / ((g - 1.0) * M * M + 2.0)) ** (g / (g - 1.0))
    b = ((g + 1.0) / (2.0 * g * M * M - (g - 1.0))) ** (1.0 / (g - 1.0))
    return a * b


def choked_mass_flux(p0: float, T0: float, g: float, R: float) -> float:
    """rho* a* = p0 sqrt(g / (R T0)) (2 / (g + 1))^((g + 1) / (2 (g - 1))) (kg/(m^2 s))."""
    return p0 * math.sqrt(g / (R * T0)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mass_flux(M: float, p0: float, T0: float, g: float, R: float) -> float:
    """rho u at Mach M in an isentropic flow from (p0, T0)."""
    T = T0 * T_ratio(M, g)
    return p0 * p_ratio(M, g) / (R * T) * M * math.sqrt(g * R * T)


@dataclass
class NozzleRegimes:
    """The three critical back pressures of a converging-diverging nozzle with exit area ratio Ae / At."""
    p1: float      # choked, subsonic everywhere downstream of the throat (above: no choking)
    p2: float      # choked, normal shock in the exit plane
    p3: float      # choked, shock-free supersonic exit (design pressure)


def critical_pressures(ae_at: float, p0: float, g: float = 1.4) -> NozzleRegimes:
    Ms = mach_from_area(ae_at, True, g)
    p3 = p0 * p_ratio(Ms, g)
    return NozzleRegimes(p0 * p_ratio(mach_from_area(ae_at, False, g), g), p3 * shock_p_ratio(Ms, g), p3)


class Quasi1DNozzle:
    """Exact quasi-1D solution in a duct of area A(x) with a single throat (the smallest area at x_t),
    inflow from the reservoir (p0, T0) at x_in, back pressure pb at the exit x_out.

    ``regime``: "subsonic" (pb > p1: not choked), "shock" (p2 < pb <= p1: choked with a normal shock in the
    diverging part, at ``x_shock``), "supersonic" (pb <= p2: the supersonic branch to the exit; below p3 the
    jet expands outside, between p3 and p2 an oblique shock system stands outside the exit)."""

    def __init__(self, area, x_in: float, x_t: float, x_out: float, p0: float, T0: float, pb: float,
                 g: float = 1.4, R: float = 287.0):
        from scipy.optimize import brentq

        self.area, self.x_in, self.x_t, self.x_out = area, x_in, x_t, x_out
        self.p0, self.T0, self.pb, self.g, self.R = p0, T0, pb, g, R
        self.At, self.Ae = area(x_t), area(x_out)
        self.crit = critical_pressures(self.Ae / self.At, p0, g)
        self.x_shock = None
        self.p02 = p0
        if pb > self.crit.p1:
            self.regime = "subsonic"
            Me = mach_from_p(pb / p0, g)
            self.A_star = self.Ae / area_ratio(Me, g)       # the fictitious sonic area, > At
        else:
            self.A_star = self.At
            if pb > self.crit.p2:
                self.regime = "shock"

                def exit_p(xs):
                    M1 = mach_from_area(area(xs) / self.At, True, g)
                    p02 = p0 * shock_p0_ratio(M1, g)
                    Me = mach_from_area(self.Ae * p02 / (self.At * p0), False, g)
                    return p02 * p_ratio(Me, g) - pb

                self.x_shock = brentq(exit_p, x_t + 1e-9 * (x_out - x_t), x_out, xtol=1e-13)
                M1 = mach_from_area(area(self.x_shock) / self.At, True, g)
                self.p02 = p0 * shock_p0_ratio(M1, g)
            else:
                self.regime = "supersonic"
        self.mdot_per_area = mass_flux(1.0, p0, T0, g, R)      # at the (possibly fictitious) sonic area

    @property
    def mass_flow(self) -> float:
        """Through the duct (kg/s per unit of the area function): rho* a* A*."""
        return choked_mass_flux(self.p0, self.T0, self.g, self.R) * self.A_star

    def mach(self, x: float) -> float:
        a = self.area(x)
        if self.regime == "subsonic":
            return mach_from_area(a / self.A_star, False, self.g)
        if x <= self.x_t:
            return mach_from_area(a / self.At, False, self.g)
        if self.regime == "shock" and x > self.x_shock:
            return mach_from_area(a * self.p02 / (self.At * self.p0), False, self.g)
        return mach_from_area(a / self.At, True, self.g)

    def p0_at(self, x: float) -> float:
        return self.p02 if (self.regime == "shock" and x > self.x_shock) else self.p0

    def p(self, x: float) -> float:
        return self.p0_at(x) * p_ratio(self.mach(x), self.g)

    def T(self, x: float) -> float:
        return self.T0 * T_ratio(self.mach(x), self.g)


__all__ = ["NACA1135", "p_ratio", "T_ratio", "area_ratio", "mach_from_area", "mach_from_p", "shock_p_ratio",
           "shock_mach", "shock_p0_ratio", "choked_mass_flux", "mass_flux", "critical_pressures", "NozzleRegimes",
           "Quasi1DNozzle"]
