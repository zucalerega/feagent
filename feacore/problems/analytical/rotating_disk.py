"""Rotating solid disk of uniform thickness (plane stress), centrifugal body force, quarter model.

Body force f = rho omega^2 (x, y) (force per volume). Exact solution (Timoshenko & Goodier, Ch. 4,
"Rotating disks"; plane stress), with c = rho omega^2 and outer radius b:
    s_rr = (3 + nu)/8 c (b^2 - r^2)
    s_tt = (3 + nu)/8 c b^2 - (1 + 3 nu)/8 c r^2
    u_r  = c r (1 - nu) / (8 E) [ (3 + nu) b^2 - (1 + nu) r^2 ]
Checked with sympy (equilibrium d s_rr/dr + (s_rr - s_tt)/r + c r = 0, Hooke).
At the centre s_rr = s_tt = (3 + nu)/8 c b^2 (the maximum of both and of von Mises).
Strain energy (Clapeyron, U = 1/2 int f.u dV; the rim is free and the symmetry lines do no work):
    U(quarter) = pi t c^2 (1 - nu) (7 + nu) b^6 / (384 E)
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, Load, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import STEEL, TG, ExactProblem, num
from feacore.problems.base import Reference
from feacore.problems.registry import register

_SRC = f"{TG}, Ch. 4 'Rotating disks' (disk of uniform thickness, plane stress)"


class RotatingDisk(ExactProblem):
    id = "rotating_disk"
    title = "Rotating solid disk, centrifugal body force (plane stress, quarter model)"
    family = "rotating_disk"
    tags = ("2d", "plane_stress", "analytical", "body_force", "expression_load", "curved_boundary", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "smooth"
    default_params = {"b": 0.25, "rho": 7850.0, "omega": 1000.0, **STEEL, "thickness": 0.02, "h0": 0.05}

    @property
    def c(self) -> float:
        return self.params["rho"] * self.params["omega"] ** 2

    def description(self) -> str:
        p = self.params
        return (
            "Solid steel disk of uniform thickness spinning about its axis at constant angular velocity, plane "
            "stress (thin disk). By symmetry only the quarter x >= 0, y >= 0 is modelled (STEP: planar quarter "
            "disk in the xy-plane, axis of rotation = z axis through the origin).\n"
            f"Geometry: quarter disk of radius b = {p['b']} m centred at the origin. Edges: y = 0 from (0, 0) to "
            "(b, 0), the rim arc r = b, x = 0 from (0, b) to (0, 0). Thickness "
            f"{p['thickness']} m.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}, density rho = {p['rho']:g} kg/m^3.\n"
            f"Loads: rotation at omega = {p['omega']:g} rad/s, i.e. the centrifugal body force (N/m^3) "
            f"f = rho omega^2 (x, y) = ({num(self.c)}*x, {num(self.c)}*y). No other loads; the rim is free.\n"
            "Supports: symmetry on x = 0 (u_x = 0) and y = 0 (u_y = 0).\n"
            "Report (SI units):\n"
            "  ur_rim         radial displacement at the rim, u_x at (b, 0) [m]\n"
            "  s_center       sigma_xx at the centre (0, 0) [Pa]\n"
            "  hoop_rim       hoop stress at the rim, sigma_yy at (b, 0) [Pa]\n"
            "  vm_max         maximum von Mises stress [Pa]\n"
            "  strain_energy  total strain energy of the modelled quarter (given thickness) [J]"
        )

    def build_geometry(self) -> None:
        import gmsh

        b = self.params["b"]
        occ = gmsh.model.occ
        o = occ.addPoint(0, 0, 0)
        p1, p2 = occ.addPoint(b, 0, 0), occ.addPoint(0, b, 0)
        y0 = occ.addLine(o, p1)
        rim = occ.addCircleArc(p1, o, p2)
        x0 = occ.addLine(p2, o)
        s = occ.addPlaneSurface([occ.addCurveLoop([y0, rim, x0])])
        occ.synchronize()
        add_group(1, [rim], "rim")
        add_group(1, [x0], "sym_x0")
        add_group(1, [y0], "sym_y0")
        add_group(2, [s], "domain")

    def qois(self) -> list[QoI]:
        b = self.params["b"]
        return [
            QoI("ur_rim", "point", field="ux", point=[b, 0.0]),
            QoI("s_center", "point", field="sxx", point=[0.0, 0.0]),
            QoI("hoop_rim", "point", field="syy", point=[b, 0.0]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        c = num(self.c)
        bcs = [BC("symmetry", group="sym_x0"), BC("symmetry", group="sym_y0")]
        loads = [Load("body_force", value=[f"{c}*x", f"{c}*y"], group="domain")]
        return self.make_case(mesh_path, order, bcs, loads=loads, thickness=self.params["thickness"])

    def _primary(self, x, y, z):
        p = self.params
        b, E, nu, c = p["b"], p["E"], p["nu"], self.c
        r2 = x * x + y * y
        srr = (3 + nu) / 8 * c * (b * b - r2)
        stt = (3 + nu) / 8 * c * b * b - (1 + 3 * nu) / 8 * c * r2
        ur_over_r = c * (1 - nu) / (8 * E) * ((3 + nu) * b * b - (1 + nu) * r2)
        r = np.sqrt(r2)
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = np.where(r > 0, x / np.where(r > 0, r, 1), 1.0)
            sn = np.where(r > 0, y / np.where(r > 0, r, 1), 0.0)
        return {"ux": ur_over_r * x, "uy": ur_over_r * y,
                "sxx": srr * cs**2 + stt * sn**2, "syy": srr * sn**2 + stt * cs**2,
                "sxy": (srr - stt) * sn * cs}

    def strain_energy_exact(self) -> float:
        p = self.params
        return math.pi * p["thickness"] * self.c**2 * (1 - p["nu"]) * (7 + p["nu"]) * p["b"] ** 6 / (384 * p["E"])

    def references(self) -> list[Reference]:
        p = self.params
        smax = (3 + p["nu"]) / 8 * self.c * p["b"] ** 2
        return [
            self.point_reference("ur_rim", _SRC + "; u_r(b) = c b^3 (1 - nu) / (4 E)", 0.01),
            self.point_reference("s_center", _SRC + "; s_rr(0) = (3 + nu)/8 rho omega^2 b^2", 0.02),
            self.point_reference("hoop_rim", _SRC + "; s_tt(b) = (1 - nu)/4 rho omega^2 b^2", 0.02),
            Reference("vm_max", smax, "analytical",
                      _SRC + "; max von Mises at the centre, where s_rr = s_tt = (3 + nu)/8 rho omega^2 b^2", 0.02),
            Reference("strain_energy", self.strain_energy_exact(), "analytical",
                      _SRC + "; U = 1/2 int f.u dV = pi t c^2 (1-nu)(7+nu) b^6 / (384 E) (quarter)", 0.01),
        ]


register(RotatingDisk())

__all__ = ["RotatingDisk"]
