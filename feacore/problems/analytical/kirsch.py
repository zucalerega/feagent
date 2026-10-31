"""Kirsch problem with exact tractions: infinite plate with a circular hole (radius a) under remote
uniaxial tension sigma0 along x, posed on the finite quarter domain [0, L]^2 minus the quarter disk,
with the EXACT Kirsch tractions imposed on the outer edges x = L and y = L (expression BCs). The
Kirsch field is then the exact solution of the finite problem everywhere: a test of curved
boundaries, expression tractions and stress recovery with no modelling error.

Stresses (Kirsch 1898; Timoshenko & Goodier, Ch. 4, "The effect of circular holes on stress
distributions in plates"), in Cartesian form with r, t = polar coordinates:
    sxx = s0 [1 - a^2/r^2 (3/2 cos 2t + cos 4t) + 3/2 a^4/r^4 cos 4t]
    syy = s0 [ -a^2/r^2 (1/2 cos 2t - cos 4t) - 3/2 a^4/r^4 cos 4t]
    sxy = s0 [ -a^2/r^2 (1/2 sin 2t + sin 4t) + 3/2 a^4/r^4 sin 4t]
Displacements (plane stress, mu = E / (2 (1 + nu)), kappa = (3 - nu) / (1 + nu)):
    ux = s0 a / (8 mu) [ r/a (kappa + 1) cos t + 2 a/r ((1 + kappa) cos t + cos 3t) - 2 a^3/r^3 cos 3t ]
    uy = s0 a / (8 mu) [ r/a (kappa - 3) sin t + 2 a/r ((1 - kappa) sin t + sin 3t) - 2 a^3/r^3 sin 3t ]
(the displacement form used e.g. in the XFEM literature, Sukumar et al. 2001; derived from the
Kolosov-Muskhelishvili potentials, Barber, Elasticity, Ch. 8 / Ch. 19). The displacement field was
checked with sympy against the stresses (strain-displacement + plane-stress Hooke: residual ~1e-16)
and is zero-normal on both symmetry lines, so no rigid-body offset is needed.
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import (
    BARBER, STEEL, TG, ExactProblem, group_entities, num, plate_with_quarter_hole, refine_near,
)
from feacore.problems.base import Reference
from feacore.problems.registry import register

_SRC = (f"Kirsch (1898), Z. VDI 42:797-807; {TG}, Ch. 4 'The effect of circular holes on stress "
        f"distributions in plates'; plane-stress displacement field from the Kolosov-Muskhelishvili "
        f"potentials ({BARBER}; also Sukumar et al. 2001, CMAME 190:6183)")


def kirsch_stress(x, y, a, s0):
    r2 = x * x + y * y
    t = np.arctan2(y, x)
    ar2, ar4 = a * a / r2, a**4 / r2**2
    c2, c4, s2, s4 = np.cos(2 * t), np.cos(4 * t), np.sin(2 * t), np.sin(4 * t)
    sxx = s0 * (1 - ar2 * (1.5 * c2 + c4) + 1.5 * ar4 * c4)
    syy = s0 * (-ar2 * (0.5 * c2 - c4) - 1.5 * ar4 * c4)
    sxy = s0 * (-ar2 * (0.5 * s2 + s4) + 1.5 * ar4 * s4)
    return sxx, syy, sxy


def kirsch_displacement(x, y, a, s0, E, nu):
    mu = E / (2 * (1 + nu))
    kap = (3 - nu) / (1 + nu)
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    c = s0 * a / (8 * mu)
    ux = c * (r / a * (kap + 1) * np.cos(t) + 2 * a / r * ((1 + kap) * np.cos(t) + np.cos(3 * t))
              - 2 * a**3 / r**3 * np.cos(3 * t))
    uy = c * (r / a * (kap - 3) * np.sin(t) + 2 * a / r * ((1 - kap) * np.sin(t) + np.sin(3 * t))
              - 2 * a**3 / r**3 * np.sin(3 * t))
    return ux, uy


def kirsch_stress_expressions(a: float, s0: float, x: float | None = None, y: float | None = None) -> dict[str, str]:
    """The Kirsch stresses as feacore expressions in x, y. Give ``x`` (or ``y``) to substitute a
    constant coordinate, e.g. the expressions restricted to the edge x = L."""
    S, A2, A4 = num(s0), num(a * a), num(a**4)
    X = "x" if x is None else f"({num(x)})"
    Y = "y" if y is None else f"({num(y)})"
    r2 = f"({X}**2 + {Y}**2)"
    t = f"arctan2({Y}, {X})"
    c2, c4, s2, s4 = (f"cos(2*{t})", f"cos(4*{t})", f"sin(2*{t})", f"sin(4*{t})")
    return {
        "sxx": f"{S}*(1 - {A2}/{r2}*(1.5*{c2} + {c4}) + 1.5*{A4}/{r2}**2*{c4})",
        "syy": f"{S}*(-{A2}/{r2}*(0.5*{c2} - {c4}) - 1.5*{A4}/{r2}**2*{c4})",
        "sxy": f"{S}*(-{A2}/{r2}*(0.5*{s2} + {s4}) + 1.5*{A4}/{r2}**2*{s4})",
    }


def kirsch_edge_tractions(a: float, s0: float, L: float) -> dict[str, list[str]]:
    """Traction vectors (global axes) on the edges x = L (normal +x) and y = L (normal +y)."""
    r = kirsch_stress_expressions(a, s0, x=L)
    t = kirsch_stress_expressions(a, s0, y=L)
    return {"right": [r["sxx"], r["sxy"]], "top": [t["sxy"], t["syy"]]}


class KirschExactTraction(ExactProblem):
    id = "kirsch_exact_traction"
    title = "Plate with a circular hole, exact Kirsch tractions on a finite quarter domain (plane stress)"
    family = "kirsch"
    tags = ("2d", "plane_stress", "analytical", "stress_concentration", "curved_boundary",
            "expression_bc", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "stress_concentration"
    default_params = {"a": 0.05, "L": 0.25, "sigma0": 100e6, **STEEL, "thickness": 0.002, "h0": 0.04}

    def description(self) -> str:
        p = self.params
        tr = kirsch_edge_tractions(p["a"], p["sigma0"], p["L"])
        return (
            "Steel plate with a circular hole, plane stress. The plate stands for the classical infinite plate "
            "under remote uniaxial tension along x; a finite quarter of it is modelled and the outer edges "
            "carry the tractions that the infinite plate transmits across those lines, so the finite model "
            "behaves like the infinite plate.\n"
            f"Geometry (STEP, planar face in the xy-plane): square [0, L] x [0, L] with L = {p['L']} m minus "
            f"the quarter disk of radius a = {p['a']} m centred at the origin. Edges: x = 0 (from (0, a) to "
            "(0, L)), y = 0 (from (a, 0) to (L, 0)), x = L, y = L, and the quarter-circle hole edge.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}, thickness {p['thickness']} m.\n"
            f"Loads: remote stress sigma0 = {p['sigma0']:.4g} Pa ({p['sigma0'] / 1e6:g} MPa) along x, applied "
            "through these tractions (force per area in Pa, global axes, y and x in m):\n"
            f"  edge x = L: t_x(y) = {tr['right'][0]}\n"
            f"              t_y(y) = {tr['right'][1]}\n"
            f"  edge y = L: t_x(x) = {tr['top'][0]}\n"
            f"              t_y(x) = {tr['top'][1]}\n"
            "  The hole edge is traction free.\n"
            "Supports: symmetry on x = 0 (u_x = 0) and on y = 0 (u_y = 0).\n"
            "Report (SI units):\n"
            "  sxx_hole_top   sigma_xx at the top of the hole (0, a) [Pa]\n"
            "  syy_hole_side  sigma_yy at the side of the hole (a, 0) [Pa]\n"
            "  ux_hole_side   u_x at (a, 0) [m]\n"
            "  uy_hole_top    u_y at (0, a) [m]\n"
            "  ux_right       u_x at (L, 0) [m]\n"
            "  vm_max         maximum von Mises stress [Pa]\n"
            "  strain_energy  total strain energy of the modelled quarter (given thickness) [J]"
        )

    def build_geometry(self) -> None:
        p = self.params
        g = plate_with_quarter_hole(p["a"], p["L"], p["L"])
        add_group(1, [g["hole"]], "hole")
        add_group(1, [g["right"]], "right")
        add_group(1, [g["top"]], "top")
        add_group(1, [g["x0"]], "sym_x0")
        add_group(1, [g["y0"]], "sym_y0")
        add_group(2, [g["surface"]], "domain")

    def apply_sizing(self, h: float) -> None:
        a = self.params["a"]
        refine_near(group_entities("hole"), h, fine=0.25, coarse=1.0, dist_min=0.0, dist_max=3 * a)

    def qois(self) -> list[QoI]:
        p = self.params
        a, L = p["a"], p["L"]
        return [
            QoI("sxx_hole_top", "point", field="sxx", point=[0.0, a]),
            QoI("syy_hole_side", "point", field="syy", point=[a, 0.0]),
            QoI("ux_hole_side", "point", field="ux", point=[a, 0.0]),
            QoI("uy_hole_top", "point", field="uy", point=[0.0, a]),
            QoI("ux_right", "point", field="ux", point=[L, 0.0]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        p = self.params
        tr = kirsch_edge_tractions(p["a"], p["sigma0"], p["L"])
        bcs = [
            BC("symmetry", group="sym_x0"),
            BC("symmetry", group="sym_y0"),
            BC("traction", group="right", value=tr["right"]),
            BC("traction", group="top", value=tr["top"]),
        ]
        return self.make_case(mesh_path, order, bcs, thickness=p["thickness"])

    def _primary(self, x, y, z):
        p = self.params
        sxx, syy, sxy = kirsch_stress(x, y, p["a"], p["sigma0"])
        ux, uy = kirsch_displacement(x, y, p["a"], p["sigma0"], p["E"], p["nu"])
        return {"ux": ux, "uy": uy, "sxx": sxx, "syy": syy, "sxy": sxy}

    def strain_energy_exact(self, n: int = 200) -> float:
        """U = 1/2 * thickness * int (t . u) ds over the loaded edges x = L and y = L (the hole is
        traction free; on the symmetry lines the normal displacement and the shear traction vanish)."""
        p = self.params
        L = p["L"]
        g, w = np.polynomial.legendre.leggauss(n)
        s = 0.5 * L * (g + 1)
        w = 0.5 * L * w
        f = self._primary
        r = f(np.full_like(s, L), s, 0 * s)   # x = L, n = +x
        t = f(s, np.full_like(s, L), 0 * s)   # y = L, n = +y
        work = np.sum(w * (r["sxx"] * r["ux"] + r["sxy"] * r["uy"])) + \
            np.sum(w * (t["sxy"] * t["ux"] + t["syy"] * t["uy"]))
        return 0.5 * p["thickness"] * float(work)

    def references(self) -> list[Reference]:
        p = self.params
        return [
            self.point_reference("sxx_hole_top", _SRC + "; s_tt(a, 90 deg) = 3 sigma0", 0.02),
            self.point_reference("syy_hole_side", _SRC + "; s_tt(a, 0 deg) = -sigma0", 0.02),
            self.point_reference("ux_hole_side", _SRC + "; u_x(a, 0)", 0.01),
            self.point_reference("uy_hole_top", _SRC + "; u_y(0, a)", 0.01),
            self.point_reference("ux_right", _SRC + "; u_x(L, 0)", 0.01),
            Reference("vm_max", 3 * p["sigma0"], "analytical",
                      _SRC + "; the maximum of the Kirsch field is the hoop stress 3 sigma0 at (0, a), "
                      "where the stress state is uniaxial", 0.02),
            Reference("strain_energy", self.strain_energy_exact(), "analytical",
                      "Clapeyron: U = 1/2 t int (traction . u) ds over the loaded edges, Gauss-Legendre "
                      "(200 pts per edge) of the exact Kirsch fields; " + _SRC, 0.01),
        ]


register(KirschExactTraction())

__all__ = ["KirschExactTraction", "kirsch_stress", "kirsch_displacement", "kirsch_stress_expressions",
           "kirsch_edge_tractions"]
