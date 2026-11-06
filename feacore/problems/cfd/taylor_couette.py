"""Circular Couette flow between a rotating inner cylinder and a fixed outer one (2D, exact).

Annulus R1 <= r <= R2 about the origin in the xy-plane; the inner wall turns counter-clockwise at
angular speed Omega, the outer wall is fixed, and there is no inlet or outlet. The steady
Navier-Stokes equations have the exact solution (u_r = 0; the only inertia is the centripetal
acceleration, balanced by the radial pressure gradient):

    u_theta(r) = A r + B / r,   A = -Omega R1^2 / (R2^2 - R1^2),   B = Omega R1^2 R2^2 / (R2^2 - R1^2)
    dp/dr = rho u_theta^2 / r  =>  p(r) = rho (A^2 r^2 / 2 + 2 A B ln r - B^2 / (2 r^2)) + const

Derivation: the theta-momentum equation reduces to d/dr (1/r d(r u)/dr) = 0, whose solutions are
A r + B / r; u(R1) = Omega R1 and u(R2) = 0 fix A and B. Textbook form: White, Viscous Fluid Flow,
Sec. 3-2 (rotating concentric cylinders). The 2D solution is exact for any Reynolds number; in 3D
it is stable below the Taylor number for Taylor vortices, far above the default case.

QoIs (all exact): the tangential velocity at mid-gap, the speed nearer the inner wall at another
angle, and the pressure difference across the gap. The domain is closed, so only pressure
differences are meaningful.

Default case (Re = rho Omega R1 (R2 - R1) / mu = 10): R1 = 1 cm, R2 = 2 cm, Omega = 10 rad/s,
rho = 1000 kg/m^3, mu = 0.1 Pa s. The wall velocity is the rigid rotation (-Omega y, Omega x).
Reference mesh: structured quad O-grid, four 90-degree blocks with radial lines and arcs, cell
counts scaled with 1/h; the polygonal walls converge with O(h^2) like the rest of the solution.
Checked with the fixed solver (OpenFOAM v1912) on 10, 20 and 40 cells across the gap: u_theta_mid
is 1.25 %, 0.31 %, 0.08 % low, u_mag_inner 0.81 %, 0.26 %, 0.06 %, dp_gap 0.99 %, 0.69 %, 0.17 %
(second order once the pressure is in its asymptotic range).
"""

from __future__ import annotations

import math

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (WHITE, FlowProblem, TransfiniteBlocks, count, group, num,
                                          refinement_factor, set_edge, transfinite_all)
from feacore.problems.registry import register

_SRC = ("Exact solution of the steady Navier-Stokes equations for circular Couette flow, derived in "
        f"feacore/problems/cfd/taylor_couette.py; textbook form in {WHITE}, Sec. 3-2 'flow between "
        "rotating concentric cylinders' (section title from memory)")


class TaylorCouette2D(FlowProblem):
    id = "taylor_couette_2d"
    title = "Circular Couette flow: rotating inner cylinder in a fixed outer one (2D, laminar, exact)"
    family = "taylor_couette"
    tags = ("cfd", "2d", "flow", "laminar", "analytical", "closed_cavity", "moving_wall", "curved_boundary",
            "expression_bc")
    expected_behavior = "smooth"
    max_iterations = 5000
    default_params = {"R1": 0.01, "R2": 0.02, "Omega": 10.0, "rho": 1000.0, "mu": 0.1, "h0": 0.001}

    # ---- exact solution
    @property
    def AB(self) -> tuple[float, float]:
        p = self.params
        d = p["R2"] ** 2 - p["R1"] ** 2
        return -p["Omega"] * p["R1"] ** 2 / d, p["Omega"] * p["R1"] ** 2 * p["R2"] ** 2 / d

    def u_theta(self, r: float) -> float:
        A, B = self.AB
        return A * r + B / r

    def pressure(self, r: float) -> float:
        """Exact pressure up to a constant [Pa]."""
        A, B = self.AB
        return self.params["rho"] * (A * A * r * r / 2 + 2 * A * B * math.log(r) - B * B / (2 * r * r))

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["Omega"] * p["R1"] * (p["R2"] - p["R1"]) / p["mu"]

    def radii(self) -> dict[str, float]:
        p = self.params
        gap = p["R2"] - p["R1"]
        return {"mid": p["R1"] + 0.5 * gap, "a": p["R1"] + 0.25 * gap, "b": p["R1"] + 0.75 * gap}

    def points(self) -> dict[str, list[float]]:
        r = self.radii()
        s = math.sqrt(0.5)
        pts = {"M": [r["mid"], 0.0],                     # theta = 0: u_theta = u_y
               "Q": [-r["a"] * s, r["a"] * s],            # theta = 135 deg
               "A": [0.0, -r["a"]], "B": [0.0, -r["b"]]}  # theta = -90 deg, across the gap
        return {k: [round(v, 12) for v in xy] for k, xy in pts.items()}

    def wall_velocity(self) -> list[str]:
        w = num(self.params["Omega"])
        return [f"-{w}*y", f"{w}*x"]

    # ---- statement
    def description(self) -> str:
        p = self.params
        pts = self.points()
        u = self.wall_velocity()
        return (
            "Steady laminar flow of an incompressible Newtonian fluid in the gap between two concentric "
            "cylinders (2D, circular Couette flow).\n"
            f"Geometry (STEP, planar face in the xy-plane): annulus R1 <= r <= R2 about the origin with "
            f"R1 = {p['R1']} m and R2 = {p['R2']} m. Model it as 2D (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  inner wall r = R1: rotates counter-clockwise about the origin at Omega = {p['Omega']} rad/s, "
            f"i.e. wall velocity (u_x, u_y) = ({u[0]}, {u[1]}) m/s (no slip)\n"
            "  outer wall r = R2: stationary, no slip\n"
            "The gap is closed (no inlet or outlet), so only pressure differences are meaningful.\n"
            "Report (SI units):\n"
            f"  u_theta_mid   u_y at M = ({pts['M'][0]}, {pts['M'][1]}), the mid-gap point on the +x axis [m/s]\n"
            f"  u_mag_inner   flow speed |u| at Q = ({pts['Q'][0]}, {pts['Q'][1]}) [m/s]\n"
            f"  dp_gap        static pressure difference p(B) - p(A) across the gap, B = ({pts['B'][0]}, "
            f"{pts['B'][1]}) and A = ({pts['A'][0]}, {pts['A'][1]}) [Pa]\n"
            "The mesh groups for the walls must be named inner and outer."
        )

    # ---- geometry and mesh
    def _corner(self, r: float, k: int) -> tuple[float, float]:
        t = 0.5 * math.pi * k
        return (r * math.cos(t), r * math.sin(t))

    def build_geometry(self) -> None:
        """Four 90-degree blocks (radial lines + arcs) so the reference mesh can be a structured O-grid."""
        p = self.params
        tb = TransfiniteBlocks(tol=1e-12 * p["R2"])
        inner, outer = [], []
        for k in range(4):
            a, b = self._corner(p["R1"], k), self._corner(p["R2"], k)
            c, d = self._corner(p["R2"], k + 1), self._corner(p["R1"], k + 1)
            arc_out, arc_in = tb.arc(b, (0.0, 0.0, 0.0), c), tb.arc(d, (0.0, 0.0, 0.0), a)
            tb.block([tb.line(a, b), arc_out, tb.line(c, d), arc_in])
            outer.append(abs(arc_out))
            inner.append(abs(arc_in))
        tb.finish()
        group(1, inner, "inner")
        group(1, outer, "outer")
        group(2, tb.blocks, "fluid")

    def build_cad(self) -> None:
        import gmsh

        p = self.params
        disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, p["R2"], p["R2"])
        hole = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, p["R1"], p["R1"])
        gmsh.model.occ.cut([(2, disk)], [(2, hole)])
        gmsh.model.occ.synchronize()

    def apply_sizing(self, h: float) -> None:
        """Square-ish cells: (R2 - R1) / h across the gap, a quarter arc at mid-gap / h around
        each block (10 x 24 per block at h0)."""
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        n_r = count(round((p["R2"] - p["R1"]) / p["h0"]), k)
        n_t = count(round(0.25 * math.pi * (p["R1"] + p["R2"]) / p["h0"]), k)
        for _, c in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            straight = gmsh.model.getType(1, c) == "Line"
            set_edge(c, n_r if straight else n_t)
        transfinite_all(2)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("u_theta_mid", "point", field="uy", point=pts["M"]),
            QoI("u_mag_inner", "point", field="u_mag", point=pts["Q"]),
            QoI("dp_gap", "point_difference", field="p", point=pts["B"], point2=pts["A"]),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("wall", group="inner", value=self.wall_velocity()), BC("wall", group="outer")]
        return self.make_case(mesh_path, order, bcs)

    def exact(self, field: str, points):
        import numpy as np

        pts = np.atleast_2d(np.asarray(points, dtype=float))
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x, y)
        A, B = self.AB
        ut = A * r + B / r
        vals = {"ux": -ut * y / r, "uy": ut * x / r, "uz": 0 * x, "u_mag": np.abs(ut)}
        return vals.get(field)

    def references(self) -> list[Reference]:
        r = self.radii()
        return [
            Reference("u_theta_mid", self.u_theta(r["mid"]), "analytical",
                      _SRC + "; u_theta(r) = A r + B / r at r = (R1 + R2) / 2", tolerance=0.01),
            Reference("u_mag_inner", abs(self.u_theta(r["a"])), "analytical",
                      _SRC + "; |u| = |u_theta(r)| at r = R1 + (R2 - R1) / 4", tolerance=0.01),
            Reference("dp_gap", self.pressure(r["b"]) - self.pressure(r["a"]), "analytical",
                      _SRC + "; p(r) = rho (A^2 r^2 / 2 + 2 A B ln r - B^2 / (2 r^2)) + const", tolerance=0.02),
        ]


register(TaylorCouette2D())

__all__ = ["TaylorCouette2D"]
