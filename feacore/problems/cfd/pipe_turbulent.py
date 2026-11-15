"""Fully developed turbulent flow in a smooth circular pipe (3D, RANS k-omega SST with wall functions).

Pipe of radius R along the x-axis, 0 <= x <= L = 60 D, smooth no-slip wall, static pressure 0 at the
outlet. The inlet carries a power-law profile u = u_max (1 - r/R)^(1/n) with n = 1 / sqrt(f) (a developed
shape, so that the flow develops within a short length) and the mean velocity U. The QoI is the pressure
difference between two axis points in the developed region, A at 42 D and B at 57 D.

Reference: the Darcy-Weisbach relation with the Colebrook formula for a smooth wall (Prandtl's universal law
of friction, 1 / sqrt(f) = -2 log10(2.51 / (Re sqrt(f))))

    p(A) - p(B) = f (x_B - x_A) / D * rho U^2 / 2

The formula is an empirical fit to measurements (Nikuradse, Princeton superpipe); it agrees with them to
about 2 % at these Reynolds numbers. So the reference is "handbook", not exact, and the tolerance (10 %)
is the acceptance of the RANS validation, not a solver precision.

Default case: Re = rho U D / mu = 1e5 (D = 25 mm, water-like rho = 1000 kg/m3, mu = 1e-3 Pa s, U = 4 m/s).
The mesh is the O-grid of pipe_poiseuille_3d (structured hexahedra). The wall functions need the first
cell centre in the log layer: the study sizes keep y+ between about 85 and 200, inside the validated range
(60 to 1600). Do not refine further without a lower Reynolds number check: y+ under 30 leaves the range of
the wall functions. Validation and its limits: docs/solver_transient.md, "Turbulent flow (RANS)".
"""

from __future__ import annotations

import math

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import WHITE, num
from feacore.problems.cfd.pipe import PipePoiseuille3D
from feacore.problems.registry import register

_SRC = ("Darcy-Weisbach relation with the Colebrook formula for a smooth wall (Prandtl's law); "
        "C. F. Colebrook, J. Inst. Civil Eng. 11 (1939) 133-156; F. M. White, Fluid Mechanics, Sec. 6.6 "
        "(section number from memory)")


def colebrook(reynolds: float, relative_roughness: float = 0.0) -> float:
    """Darcy friction factor of turbulent pipe flow (Colebrook; a smooth wall gives Prandtl's law)."""
    x = 8.0
    for _ in range(200):
        new = -2.0 * math.log10(relative_roughness / 3.7 + 2.51 * x / reynolds)
        if abs(new - x) < 1e-13:
            break
        x = new
    return 1.0 / new**2


class PipeTurbulent3D(PipePoiseuille3D):
    id = "pipe_turbulent_3d"
    title = "Developed turbulent flow in a smooth pipe, Re 1e5 (3D RANS k-omega SST, wall functions)"
    family = "pipe_turbulent"
    tags = ("cfd", "3d", "flow", "turbulent", "rans", "empirical", "internal_flow", "expression_bc")
    max_iterations = 3000
    residual_tol = 1e-5
    default_params = {"R": 0.0125, "L": 1.5, "U": 4.0, "rho": 1000.0, "mu": 1e-3, "h0": 0.004}
    n_side0 = 4
    n_rad0 = 3
    n_axial0 = 80
    x_a, x_b = 42.0, 57.0       # the developed window, in diameters from the inlet

    @property
    def friction_factor(self) -> float:
        return colebrook(self.reynolds)

    def options(self) -> dict:
        p = self.params
        return {"turbulence": "kOmegaSST", "max_iterations": self.max_iterations,
                "residual_tol": self.residual_tol, "skew_correction": False,
                "turbulence_intensity": 0.16 * self.reynolds ** -0.125, "turbulence_velocity": p["U"],
                "turbulence_length": 0.07 * 2.0 * p["R"]}

    def points(self) -> dict[str, list[float]]:
        D = 2.0 * self.params["R"]
        return {"A": [round(self.x_a * D, 12), 0.0, 0.0], "B": [round(self.x_b * D, 12), 0.0, 0.0]}

    def inlet_profile(self) -> str:
        p = self.params
        n = 1.0 / math.sqrt(self.friction_factor)
        umax = p["U"] * (n + 1.0) * (2.0 * n + 1.0) / (2.0 * n * n)
        return f"{num(umax)}*maximum(1 - sqrt(y**2 + z**2)/{num(p['R'])}, 1e-9)**{num(1.0 / n)}"

    def description(self) -> str:
        p = self.params
        pts = self.points()
        return (
            "Steady turbulent flow of an incompressible Newtonian fluid in a straight smooth circular pipe, 3D. "
            f"Reynolds number rho U D / mu = {self.reynolds:.4g}.\n"
            f"Geometry (STEP solid): cylinder of radius R = {p['R']} m along the x-axis, 0 <= x <= L with "
            f"L = {p['L']} m (axis y = z = 0).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Model: RANS, k-omega SST, wall functions (smooth wall).\n"
            "Boundary conditions:\n"
            f"  inlet x = 0: power-law profile with mean velocity U = {p['U']} m/s,\n"
            f"    u_x = {self.inlet_profile()}  [m/s, y and z in m],  u_y = u_z = 0\n"
            "  outlet x = L: static pressure p = 0 Pa\n"
            "  pipe wall r = R: no slip (stationary, smooth)\n"
            "Report (SI units):\n"
            f"  dp_AB             static pressure difference p(A) - p(B) between the axis points "
            f"A = ({pts['A'][0]}, 0, 0) and B = ({pts['B'][0]}, 0, 0) [Pa]\n"
            "  flow_rate_outlet  volumetric flow rate through the outlet, positive out of the pipe [m^3/s]\n"
            "The flow-rate QoI refers to the boundary group outlet: use this name for the mesh group."
        )

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        # r = 1.5 (cells per quarter of the wall 4 / 6 / 9): y+ about 200 / 130 / 85 at Re 1e5. A ratio of 2
        # would put the third level at y+ ~ 50, near the lower end of the validated range.
        return [h0, h0 / 1.5, h0 / 2.25]

    def qois(self) -> list[QoI]:
        pts = self.points()
        return [QoI("dp_AB", "point_difference", field="p", point=pts["A"], point2=pts["B"]),
                QoI("flow_rate_outlet", "flow_rate", group="outlet")]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("velocity_inlet", group="inlet", value=[self.inlet_profile(), 0.0, 0.0]),
               BC("pressure_outlet", group="outlet", value=0.0),
               BC("wall", group="wall")]
        return self.make_case(mesh_path, order, bcs)

    def exact(self, field: str, points):
        return None

    def references(self) -> list[Reference]:
        p = self.params
        D = 2.0 * p["R"]
        f = self.friction_factor
        dp = f * (self.x_b - self.x_a) * 0.5 * p["rho"] * p["U"] ** 2
        return [
            Reference("dp_AB", dp, "handbook",
                      _SRC + f"; f = {f:.5f} at Re = {self.reynolds:.4g}; dp = f (x_B - x_A) / D rho U^2 / 2",
                      tolerance=0.10,
                      notes="k-omega SST with wall functions gave -3 % to -5 % on this case (y+ 85 to 200); "
                            "the tolerance is the acceptance of the turbulence validation"),
            Reference("flow_rate_outlet", math.pi * p["R"] ** 2 * p["U"], "analytical",
                      "Mass conservation of incompressible flow (continuity equation, exact): the outlet flow "
                      "rate equals the inlet's, Q = pi R^2 U; continuity in " + WHITE, tolerance=0.02,
                      notes=f"inscribed polygon section: the mesh area is O(h^2) short of pi R^2 (D = {D} m)"),
        ]


register(PipeTurbulent3D())

__all__ = ["PipeTurbulent3D", "colebrook"]
