"""Hagen-Poiseuille flow: fully developed laminar flow in a circular pipe (3D, exact).

Pipe of radius R along the x-axis, 0 <= x <= L, no-slip wall, fully developed inflow at x = 0,
static pressure 0 at the outlet x = L. With the parabolic profile imposed at the inlet the flow is
fully developed everywhere and the steady Navier-Stokes equations have the exact solution

    u_x(r) = 2 U (1 - r^2 / R^2),  u_y = u_z = 0,  p(x) = G (L - x),  G = -dp/dx = 8 mu U / R^2

(U = mean velocity, r^2 = y^2 + z^2). Derivation: (mu / r) d/dr (r du/dr) = -G with u(R) = 0 and
du/dr(0) = 0 gives u = G (R^2 - r^2) / (4 mu), whose mean over the section is G R^2 / (8 mu) = U.
Hence

    centreline velocity  u(0) = 2 U
    pressure drop        p(x_A) - p(x_B) = 8 mu U (x_B - x_A) / R^2   (points on the axis)
    wall shear stress    tau_w = 4 mu U / R; x-force of the fluid on the wall F = tau_w 2 pi R L = 8 pi mu U L
    flow rate            Q = pi R^2 U

Force balance check: F = G L pi R^2.

Default case (Re = rho U D / mu = 10): R = 5 mm, L = 5 cm, rho = 1000 kg/m^3, mu = 0.01 Pa s,
U = 1 cm/s. Reference mesh: structured hexahedra, a "butterfly" (O-grid) section of 5 blocks (a
central square and 4 blocks out to the wall) extruded along the axis. The section is a polygon
inscribed in the circle, so the discrete wetted area and flow cross-section converge with O(h^2)
like the rest of the second-order solution.
"""

from __future__ import annotations

import math

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (WHITE, FlowProblem, TransfiniteBlocks, count, group, num,
                                          refinement_factor, set_edge)
from feacore.problems.registry import register

_SRC = ("Exact solution of the steady Navier-Stokes equations for fully developed laminar pipe flow "
        "(Hagen-Poiseuille flow), derived in feacore/problems/cfd/pipe.py; textbook form in "
        f"{WHITE}, Sec. 3-3 'Hagen-Poiseuille flow in a circular pipe' (section title from memory)")


class PipePoiseuille3D(FlowProblem):
    id = "pipe_poiseuille_3d"
    title = "Hagen-Poiseuille flow in a circular pipe, fully developed inflow (3D laminar, exact)"
    family = "pipe_poiseuille"
    tags = ("cfd", "3d", "flow", "laminar", "analytical", "internal_flow", "expression_bc")
    dim = 3
    formulation = "3d"
    expected_behavior = "smooth"
    # The cross-stream velocity is zero to round-off, so its normalised residual stalls (~1e-7); the
    # backend accepts such a stalled component once u_x and p reach residual_tol (a few hundred
    # iterations here). The cap only bounds the cost if that detection ever fails.
    max_iterations = 3000
    default_params = {"R": 0.005, "L": 0.05, "U": 0.01, "rho": 1000.0, "mu": 0.01, "h0": 0.00125}
    # block counts at h0: per side of the central square (= per quarter of the wall), square -> wall,
    # and along the axis (axial cells 5 h0 long: the exact flow does not vary along x)
    n_side0 = 4
    n_rad0 = 4
    n_axial0 = 8
    core = 0.5          # half-width of the central square / R

    @property
    def G(self) -> float:
        p = self.params
        return 8.0 * p["mu"] * p["U"] / p["R"] ** 2

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["U"] * 2 * p["R"] / p["mu"]

    def points(self) -> dict[str, list[float]]:
        """Probe points on the axis, downstream of the first cells after the inlet: a finite-volume
        solution with a fixed-velocity inlet (zero-gradient pressure) and long axial cells has an
        odd-even pressure wiggle there that decays within ~3 cells."""
        L = self.params["L"]
        return {"A": [round(0.5 * L, 12), 0.0, 0.0], "B": [round(0.9 * L, 12), 0.0, 0.0],
                "C": [round(0.5 * L, 12), 0.0, 0.0]}

    def inlet_profile(self) -> str:
        p = self.params
        return f"{num(2.0 * p['U'])}*(1 - (y**2 + z**2)/{num(p['R'] ** 2)})"

    def description(self) -> str:
        p = self.params
        pts = self.points()
        return (
            "Steady laminar flow of an incompressible Newtonian fluid in a straight circular pipe "
            "(Hagen-Poiseuille flow), 3D.\n"
            f"Geometry (STEP solid): cylinder of radius R = {p['R']} m along the x-axis, 0 <= x <= L with "
            f"L = {p['L']} m (axis y = z = 0).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  inlet x = 0: fully developed parabolic profile with mean velocity U = {p['U']} m/s,\n"
            f"    u_x = 2 U (1 - (y^2 + z^2)/R^2) = {self.inlet_profile()}  [m/s, y and z in m],  u_y = u_z = 0\n"
            "  outlet x = L: static pressure p = 0 Pa\n"
            "  pipe wall r = R: no slip (stationary)\n"
            "Report (SI units):\n"
            f"  dp_AB             static pressure difference p(A) - p(B) between the axis points "
            f"A = ({pts['A'][0]}, 0, 0) and B = ({pts['B'][0]}, 0, 0) [Pa]\n"
            f"  u_centreline      u_x on the axis at C = ({pts['C'][0]}, 0, 0) [m/s]\n"
            "  wall_force        x-component of the force of the fluid (pressure + viscous) on the pipe wall [N]\n"
            "  flow_rate_outlet  volumetric flow rate through the outlet, positive out of the pipe [m^3/s]"
        )

    # ---- geometry
    def build_cad(self) -> None:
        import gmsh

        p = self.params
        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, p["L"], 0.0, 0.0, p["R"])
        gmsh.model.occ.synchronize()

    def build_geometry(self) -> None:
        """Butterfly section (central square + 4 outer blocks) in the plane x = 0, extruded to x = L."""
        import gmsh

        p = self.params
        R, L = p["R"], p["L"]
        a = self.core * R
        b = TransfiniteBlocks()

        def sq(t):
            return (0.0, round(a * math.sqrt(2) * math.cos(math.radians(t)), 15),
                    round(a * math.sqrt(2) * math.sin(math.radians(t)), 15))

        def circ(t):
            return (0.0, R * math.cos(math.radians(t)), R * math.sin(math.radians(t)))

        angles = (45.0, 135.0, 225.0, 315.0)
        b.block([b.line(sq(angles[i]), sq(angles[(i + 1) % 4])) for i in range(4)])
        for i in range(4):
            t1, t2 = angles[i], angles[(i + 1) % 4]
            b.block([b.line(sq(t1), circ(t1)), b.arc(circ(t1), (0.0, 0.0, 0.0), circ(t2)),
                     b.line(circ(t2), sq(t2)), b.line(sq(t2), sq(t1))])
        b.finish()
        occ = gmsh.model.occ
        occ.extrude([(2, s) for s in b.blocks], L, 0.0, 0.0)
        occ.removeAllDuplicates()       # glue the lateral faces shared by neighbouring blocks
        occ.synchronize()
        tol = 1e-4 * L          # OCC bounding boxes are padded by ~1e-7
        inlet, outlet, wall = [], [], []
        for _, s in gmsh.model.getEntities(2):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
            if x1 < tol:
                inlet.append(s)
            elif x0 > L - tol:
                outlet.append(s)
            elif self._is_wall(s):
                wall.append(s)
        group(2, inlet, "inlet")
        group(2, outlet, "outlet")
        group(2, wall, "wall")
        group(3, [t for _, t in gmsh.model.getEntities(3)], "fluid")

    def _is_wall(self, s: int) -> bool:
        """A lateral face lies on the pipe wall if its points are at r = R."""
        import gmsh

        R = self.params["R"]
        for _, pt in gmsh.model.getBoundary([(2, s)], combined=False, recursive=True):
            y, z = gmsh.model.getValue(0, pt, [])[1:]
            if abs(math.hypot(y, z) - R) > 1e-6 * R:
                return False
        return True

    def apply_sizing(self, h: float) -> None:
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        R = p["R"]
        n_side, n_rad, n_ax = count(self.n_side0, k), count(self.n_rad0, k), count(self.n_axial0, k)
        for _, c in gmsh.model.getEntities(1):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, c)
            if x1 - x0 > 1e-3 * p["L"]:                 # axial edge (bounding boxes are padded ~1e-7)
                set_edge(c, n_ax)
                continue
            ends = [gmsh.model.getValue(0, pt, [])[1:] for _, pt in gmsh.model.getBoundary([(1, c)])]
            radii = sorted(math.hypot(y, z) for y, z in ends)
            if abs(radii[0] - radii[1]) < 1e-9 * R:      # square side or wall arc
                set_edge(c, n_side)
            else:                                        # square corner -> wall
                set_edge(c, n_rad)
        for _, s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)
        for _, v in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(v)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        # r = 2 (640 / 5120 / 40960 hexes): the axis probes then stay mesh vertices on every level,
        # so the point-reconstruction error keeps the same h^2 coefficient (r = 1.5 would give 4 / 6 /
        # 9 cells per quarter and put the axis at a vertex, a vertex, then a cell centre)
        return [h0, h0 / 2, h0 / 4]

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("dp_AB", "point_difference", field="p", point=pts["A"], point2=pts["B"]),
            QoI("u_centreline", "point", field="ux", point=pts["C"]),
            QoI("wall_force", "force", group="wall", component=0),
            QoI("flow_rate_outlet", "flow_rate", group="outlet"),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [
            BC("velocity_inlet", group="inlet", value=[self.inlet_profile(), 0.0, 0.0]),
            BC("pressure_outlet", group="outlet", value=0.0),
            BC("wall", group="wall"),
        ]
        return self.make_case(mesh_path, order, bcs)

    def exact(self, field: str, points):
        import numpy as np

        p = self.params
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        ux = 2.0 * p["U"] * (1.0 - (y**2 + z**2) / p["R"] ** 2)
        vals = {"ux": ux, "uy": 0 * x, "uz": 0 * x, "u_mag": np.abs(ux), "p": self.G * (p["L"] - x)}
        return vals.get(field)

    def references(self) -> list[Reference]:
        p = self.params
        pts = self.points()
        return [
            Reference("dp_AB", self.G * (pts["B"][0] - pts["A"][0]), "analytical",
                      _SRC + "; p(A) - p(B) = 8 mu U (x_B - x_A) / R^2", tolerance=0.02),
            Reference("u_centreline", 2.0 * p["U"], "analytical", _SRC + "; u(0) = 2 U", tolerance=0.02),
            Reference("wall_force", 8.0 * math.pi * p["mu"] * p["U"] * p["L"], "analytical",
                      _SRC + "; F = tau_w 2 pi R L = 8 pi mu U L", tolerance=0.02),
            Reference("flow_rate_outlet", math.pi * p["R"] ** 2 * p["U"], "analytical",
                      _SRC + "; Q = pi R^2 U", tolerance=0.02),
        ]


register(PipePoiseuille3D())

__all__ = ["PipePoiseuille3D"]
