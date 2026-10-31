"""Plane Poiseuille flow: fully developed laminar flow between parallel walls (2D, exact).

Channel 0 <= x <= L, 0 <= y <= H, walls y = 0 and y = H (no slip), fully developed inflow at
x = 0, static pressure 0 at the outlet x = L. With the parabolic profile imposed at the inlet the
flow is fully developed everywhere, and the steady incompressible Navier-Stokes equations have the
exact solution (convection u . grad u vanishes identically):

    u(y) = 6 U (y / H) (1 - y / H),   v = 0,   p(x) = G (L - x),   G = -dp/dx = 12 mu U / H^2

(U = mean velocity). Derivation: mu u'' = dp/dx = -G with u(0) = u(H) = 0 gives u = G y (H - y) /
(2 mu), whose mean over the gap is G H^2 / (12 mu) = U. Hence (per metre of depth t):

    centreline velocity   u(H/2) = 1.5 U
    pressure drop         p(x_A) - p(x_B) = G (x_B - x_A)        (points on the centreline)
    wall shear stress     tau_w = mu du/dy|_wall = 6 mu U / H; x-force of the fluid on each wall
                          F = tau_w L t (+x, along the flow)
    flow rate             Q = U H t through the outlet (positive out of the domain)

Force balance check: 2 F = G L H t (pressure drop x area = total wall shear).

Default case (Re = rho U H / mu = 10): H = 1 cm, L = 5 cm, rho = 1000 kg/m^3, mu = 0.01 Pa s
(a light oil), U = 1 cm/s. The reference mesh is a uniform structured quad grid (square cells,
H / h cells across the gap); a second-order finite-volume solution has O(h^2) errors in every QoI
(for this flow the wall cells give a uniform shift of order (h/H)^2 in u and in the pressure
gradient), so the study shows clean order-2 convergence.
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (WHITE, FlowProblem, boundary_curves_where, count, group, near,
                                          num, refinement_factor, set_edge, transfinite_all)
from feacore.problems.registry import register

_SRC = (f"Exact solution of the steady Navier-Stokes equations for fully developed plane channel flow "
        f"(plane Poiseuille flow), derived in feacore/problems/cfd/poiseuille.py; textbook form in {WHITE}, "
        "Sec. 3-3 'Poiseuille flow between parallel plates' (section title from memory)")


class PoiseuilleChannel(FlowProblem):
    id = "poiseuille_channel"
    title = "Plane Poiseuille flow in a 2D channel, fully developed inflow (laminar, exact)"
    family = "poiseuille_channel"
    tags = ("cfd", "2d", "flow", "laminar", "analytical", "internal_flow", "expression_bc")
    expected_behavior = "smooth"
    # The cross-stream velocity is zero to round-off, so its normalised residual stalls (~1e-7); the
    # backend accepts such a stalled component once u_x and p reach residual_tol (a few hundred
    # iterations here). The cap only bounds the cost if that detection ever fails.
    max_iterations = 3000
    default_params = {"H": 0.01, "L": 0.05, "U": 0.01, "rho": 1000.0, "mu": 0.01, "h0": 0.00125}

    # ---- derived quantities
    @property
    def G(self) -> float:
        """Pressure gradient magnitude -dp/dx [Pa/m]."""
        p = self.params
        return 12.0 * p["mu"] * p["U"] / p["H"] ** 2

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["U"] * p["H"] / p["mu"]

    def points(self) -> dict[str, list[float]]:
        p = self.params
        pts = {"A": [0.25 * p["L"], 0.5 * p["H"]], "B": [0.75 * p["L"], 0.5 * p["H"]],
               "C": [0.5 * p["L"], 0.5 * p["H"]]}
        return {k: [round(v, 12) for v in xy] for k, xy in pts.items()}

    def inlet_profile(self) -> str:
        p = self.params
        return f"{num(6.0 * p['U'] / p['H'] ** 2)}*y*({num(p['H'])} - y)"

    # ---- statement
    def description(self) -> str:
        p = self.params
        pts = self.points()
        return (
            "Steady laminar flow of an incompressible Newtonian fluid in a straight 2D channel "
            "(plane Poiseuille flow).\n"
            f"Geometry (STEP, planar face in the xy-plane): rectangle 0 <= x <= L, 0 <= y <= H with "
            f"L = {p['L']} m (flow direction) and H = {p['H']} m (gap between the walls). Model it as 2D; "
            "report forces and flow rates per metre of depth (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  inlet x = 0: fully developed parabolic velocity profile with mean velocity U = {p['U']} m/s,\n"
            f"    u_x(y) = 6 U (y/H)(1 - y/H) = {self.inlet_profile()}  [m/s, y in m],  u_y = 0\n"
            "  outlet x = L: static pressure p = 0 Pa\n"
            "  walls y = 0 and y = H: no slip (stationary)\n"
            "Report (SI units):\n"
            f"  dp_AB             static pressure difference p(A) - p(B) between the centreline points "
            f"A = ({pts['A'][0]}, {pts['A'][1]}) and B = ({pts['B'][0]}, {pts['B'][1]}) [Pa]\n"
            f"  u_centreline      u_x on the centreline at C = ({pts['C'][0]}, {pts['C'][1]}) [m/s]\n"
            "  wall_force_bottom x-component of the force of the fluid (pressure + viscous) on the wall "
            "y = 0 [N per m depth]\n"
            "  wall_force_top    the same on the wall y = H [N per m depth]\n"
            "  flow_rate_outlet  volumetric flow rate through the outlet, positive out of the channel "
            "[m^3/s per m depth]\n"
            "The force and flow-rate QoIs refer to the boundary groups wall_bottom, wall_top and outlet: "
            "use these names for the corresponding mesh groups."
        )

    # ---- geometry and mesh
    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        L, H = p["L"], p["H"]
        s = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
        gmsh.model.occ.synchronize()
        tol = 1e-4 * H     # OCC bounding boxes are padded by ~1e-7
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0, tol)), "inlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x0, L, tol)), "outlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0, tol)), "wall_bottom")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(y0, H, tol)), "wall_top")
        group(2, [s], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Uniform structured grid of square cells: H/h cells across, L/h along (at h0: 8 x 40)."""
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        ny = count(round(p["H"] / p["h0"]), k)
        nx = count(round(p["L"] / p["h0"]), k)
        for _, c in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            set_edge(c, nx if abs(x1 - x0) > abs(y1 - y0) else ny)
        transfinite_all(2)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4, h0 / 8]     # 8, 16, 32, 64 cells across the gap

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("dp_AB", "point_difference", field="p", point=pts["A"], point2=pts["B"]),
            QoI("u_centreline", "point", field="ux", point=pts["C"]),
            QoI("wall_force_bottom", "force", group="wall_bottom", component=0),
            QoI("wall_force_top", "force", group="wall_top", component=0),
            QoI("flow_rate_outlet", "flow_rate", group="outlet"),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [
            BC("velocity_inlet", group="inlet", value=[self.inlet_profile(), 0.0]),
            BC("pressure_outlet", group="outlet", value=0.0),
            BC("wall", group="wall_bottom"),
            BC("wall", group="wall_top"),
        ]
        return self.make_case(mesh_path, order, bcs)

    def exact(self, field: str, points):
        import numpy as np

        p = self.params
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        x, y = pts[:, 0], pts[:, 1]
        ux = 6.0 * p["U"] * (y / p["H"]) * (1.0 - y / p["H"])
        vals = {"ux": ux, "uy": 0.0 * x, "uz": 0.0 * x, "u_mag": np.abs(ux), "p": self.G * (p["L"] - x)}
        return vals.get(field)

    def references(self) -> list[Reference]:
        p = self.params
        pts = self.points()
        t = self.thickness
        tau_w = 6.0 * p["mu"] * p["U"] / p["H"]
        return [
            Reference("dp_AB", self.G * (pts["B"][0] - pts["A"][0]), "analytical",
                      _SRC + "; p(A) - p(B) = 12 mu U (x_B - x_A) / H^2", tolerance=0.01),
            Reference("u_centreline", 1.5 * p["U"], "analytical", _SRC + "; u(H/2) = 1.5 U", tolerance=0.01),
            Reference("wall_force_bottom", tau_w * p["L"] * t, "analytical",
                      _SRC + "; tau_w = 6 mu U / H, F = tau_w L t per wall", tolerance=0.01),
            Reference("wall_force_top", tau_w * p["L"] * t, "analytical",
                      _SRC + "; tau_w = 6 mu U / H, F = tau_w L t per wall", tolerance=0.01),
            Reference("flow_rate_outlet", p["U"] * p["H"] * t, "analytical",
                      _SRC + "; mass conservation Q = U H t", tolerance=0.01),
        ]


register(PoiseuilleChannel())

__all__ = ["PoiseuilleChannel"]
