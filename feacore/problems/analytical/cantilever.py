"""Timoshenko-Goodier cantilever with a parabolic end shear (plane stress), exact everywhere.

Beam 0 <= x <= L, -D/2 <= y <= D/2, thickness t, I = t D^3 / 12, end load P (downward, -y)
applied at x = L as the parabolic shear traction t_y = sxy = -P/(2I) (D^2/4 - y^2). Exact solution
(Timoshenko & Goodier, Ch. 3, "Bending of a cantilever loaded at the end"; in the form of
Augarde & Deeks, Finite Elem. Anal. Des. 44 (2008) 595-601):

    ux = P y / (6 E I) [ (6 L - 3 x) x + (2 + nu) (y^2 - D^2/4) ]
    uy = -P / (6 E I) [ 3 nu y^2 (L - x) + (4 + 5 nu) D^2 x / 4 + (3 L - x) x^2 ]
    sxx = P (L - x) y / I,  syy = 0,  sxy = -P / (2 I) (D^2/4 - y^2)

The exact displacement is prescribed at the root x = 0 (both components, expressions), which makes
the FE problem's exact solution the field above everywhere (no clamped-corner singularity). The
field was checked with sympy (strain-displacement + plane-stress Hooke reproduce the stresses).
Tip deflection uy(L, 0) = -P L (8 L^2 + (4 + 5 nu) D^2) / (24 E I); ux(L, D/2) = P D L^2 / (4 E I).
(No shear-stress QoI: the prescribed tip traction already gives sxy, which does not depend on x.)
Strain energy: U = t/2 int_A (sxx^2 / E + sxy^2 / G) dA = P^2 L^3 / (6 E I) + 3 P^2 L / (5 G t D).
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import STEEL, TG, ExactProblem, num
from feacore.problems.base import Reference
from feacore.problems.registry import register

_SRC = (f"{TG}, Ch. 3 'Bending of a cantilever loaded at the end'; displacement field as in "
        "C.E. Augarde & A.J. Deeks, Finite Elem. Anal. Des. 44 (2008) 595-601")


class TimoshenkoCantilever(ExactProblem):
    id = "timoshenko_cantilever"
    title = "Cantilever with parabolic end shear, exact Timoshenko-Goodier solution (plane stress)"
    family = "timoshenko_cantilever"
    tags = ("2d", "plane_stress", "analytical", "beam", "expression_bc")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "smooth"
    default_params = {"L": 1.0, "D": 0.2, "P": 100e3, **STEEL, "thickness": 0.1, "h0": 0.05}

    @property
    def I(self) -> float:  # noqa: E743
        p = self.params
        return p["thickness"] * p["D"] ** 3 / 12

    def _root_expressions(self) -> tuple[str, str]:
        """Exact displacement at x = 0 (valid on the root edge)."""
        p = self.params
        E, nu, D, L, P = p["E"], p["nu"], p["D"], p["L"], p["P"]
        c = P / (6 * E * self.I)
        ux = f"{num(c * (2 + nu))}*y*(y**2 - {num(D * D / 4)})"
        uy = f"{num(-c * 3 * nu * L)}*y**2"
        return ux, uy

    def _tip_traction(self) -> str:
        p = self.params
        return f"{num(-p['P'] / (2 * self.I))}*({num(p['D'] ** 2 / 4)} - y**2)"

    def description(self) -> str:
        p = self.params
        ux, uy = self._root_expressions()
        return (
            "Steel cantilever beam, plane stress, with the end load applied as a parabolic shear traction "
            "(the classical Timoshenko-Goodier cantilever). The root displacement is prescribed exactly as "
            "in the elasticity solution instead of a rigid clamp.\n"
            f"Geometry (STEP, planar face in the xy-plane): rectangle 0 <= x <= L, -D/2 <= y <= D/2 with "
            f"L = {p['L']} m, D = {p['D']} m. Thickness t = {p['thickness']} m (out of plane).\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}.\n"
            f"Loads: total end load P = {p['P']:.4g} N in the -y direction on the free end x = L, applied as "
            "the shear traction (force per area, Pa, y in m)\n"
            f"  t_y(y) = {self._tip_traction()}    (t_x = 0)\n"
            f"which is -P/(2I) (D^2/4 - y^2) with I = t D^3/12 = {self.I:.6g} m^4. The top and bottom faces "
            "y = +-D/2 are traction free.\n"
            "Supports: on the root x = 0 prescribe both displacement components (m, y in m):\n"
            f"  u_x(0, y) = {ux}\n"
            f"  u_y(0, y) = {uy}\n"
            "Report (SI units):\n"
            "  tip_deflection  u_y at (L, 0) [m]\n"
            "  sxx_top_mid     sigma_xx at (L/2, D/2) [Pa]\n"
            "  ux_tip_top      u_x at the top corner of the loaded end (L, D/2) [m]\n"
            "  vm_max          maximum von Mises stress [Pa]\n"
            "  strain_energy   total strain energy (given thickness) [J]"
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        L, D = p["L"], p["D"]
        occ = gmsh.model.occ
        s = occ.addRectangle(0, -D / 2, 0, L, D)
        occ.synchronize()
        tags = {}
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            x, y, _ = occ.getCenterOfMass(1, c)
            if abs(x) < 1e-9 * L:
                tags["root"] = c
            elif abs(x - L) < 1e-9 * L:
                tags["tip"] = c
            elif y > 0:
                tags["top"] = c
            else:
                tags["bottom"] = c
        for name in ("root", "tip", "top", "bottom"):
            add_group(1, [tags[name]], name)
        add_group(2, [s], "domain")

    def qois(self) -> list[QoI]:
        p = self.params
        L, D = p["L"], p["D"]
        return [
            QoI("tip_deflection", "point", field="uy", point=[L, 0.0]),
            QoI("sxx_top_mid", "point", field="sxx", point=[L / 2, D / 2]),
            QoI("ux_tip_top", "point", field="ux", point=[L, D / 2]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        ux, uy = self._root_expressions()
        bcs = [BC("displacement", group="root", components=[0, 1], value=[ux, uy]),
               BC("traction", group="tip", value=[0.0, self._tip_traction()])]
        return self.make_case(mesh_path, order, bcs, thickness=self.params["thickness"])

    def _primary(self, x, y, z):
        p = self.params
        E, nu, D, L, P, I = p["E"], p["nu"], p["D"], p["L"], p["P"], self.I
        ux = P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y**2 - D**2 / 4))
        uy = -P / (6 * E * I) * (3 * nu * y**2 * (L - x) + (4 + 5 * nu) * D**2 * x / 4 + (3 * L - x) * x**2)
        return {"ux": ux, "uy": uy, "sxx": P * (L - x) * y / I, "syy": 0 * x,
                "sxy": -P / (2 * I) * (D**2 / 4 - y**2)}

    def strain_energy_exact(self) -> float:
        p = self.params
        E, nu, D, L, P, t = p["E"], p["nu"], p["D"], p["L"], p["P"], p["thickness"]
        G = E / (2 * (1 + nu))
        return P**2 * L**3 / (6 * E * self.I) + 3 * P**2 * L / (5 * G * t * D)

    def references(self) -> list[Reference]:
        p = self.params
        return [
            self.point_reference("tip_deflection", _SRC + "; uy(L,0) = -P L (8L^2 + (4+5nu) D^2)/(24 E I)", 0.01),
            self.point_reference("sxx_top_mid", _SRC + "; sxx = P (L - x) y / I", 0.02),
            self.point_reference("ux_tip_top", _SRC + "; ux(L, D/2) = P D L^2 / (4 E I)", 0.01),
            Reference("vm_max", p["P"] * p["L"] * p["D"] / (2 * self.I), "analytical",
                      _SRC + "; max von Mises = |sxx| = P L D / (2 I) at the root fibres (0, +-D/2), where "
                      "sxy = 0", 0.02),
            Reference("strain_energy", self.strain_energy_exact(), "analytical",
                      _SRC + "; U = t/2 int (sxx^2/E + sxy^2/G) dA = P^2 L^3/(6EI) + 3 P^2 L/(5 G t D)", 0.01),
        ]


register(TimoshenkoCantilever())

__all__ = ["TimoshenkoCantilever"]
