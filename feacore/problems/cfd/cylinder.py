"""DFG benchmark 2D-1 (Schaefer & Turek 1996): steady laminar flow around a cylinder in a channel, Re = 20.

Channel 0 <= x <= 2.2, 0 <= y <= 0.41 (m) with a circular cylinder of diameter D = 0.1 centred at
(0.2, 0.2), slightly below the channel axis (so the lift is small but not zero). Fluid: rho = 1
kg/m^3, kinematic viscosity nu = 1e-3 m^2/s (mu = 1e-3 Pa s). Inflow x = 0: parabolic profile
u = 4 U_m y (H - y) / H^2, v = 0 with U_m = 0.3 m/s; no slip on y = 0, y = H and the cylinder;
outflow x = 2.2 (the benchmark's "do-nothing" condition; here static pressure 0, equivalent for
the fully developed outflow 2 m downstream). Mean inflow velocity U_bar = 2 U_m / 3 = 0.2 m/s,
Re = U_bar D / nu = 20; the flow is steady.

Benchmark outputs: drag and lift coefficients c_D = 2 F_D / (rho U_bar^2 D), c_L = 2 F_L /
(rho U_bar^2 D) (F = force of the fluid on the cylinder per unit depth) and the pressure difference
dp = p(0.15, 0.2) - p(0.25, 0.2) between the front and rear points of the cylinder. Here the QoIs are
the forces per metre of depth, F = c rho U_bar^2 D / 2 = 0.002 c [N/m], and dp [Pa] (= kinematic
pressure, rho = 1).

High-accuracy reference values (FEATFLOW benchmark page for 2D-1, "reference results", computed with
high-order methods): c_D = 5.57953523384, c_L = 0.010618948146, dp = 0.11752016697. The original
paper gives the acceptance intervals c_D in [5.57, 5.59], c_L in [0.0104, 0.0110], dp in
[0.1172, 0.1176] (intervals quoted from memory).

Reference mesh: 12 transfinite quad blocks. A square box [0.1, 0.3]^2 around the cylinder is an
O-grid (4 blocks between the circle and the box, graded towards the cylinder); the rest of the
channel is cut by the lines x = 0.1, x = 0.3, y = 0.1, y = 0.3 into 8 rectangular blocks, graded
downstream of the box. All counts scale with 1/h (64 cells around the cylinder at h0).
"""

from __future__ import annotations

import math

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (FEATFLOW_2D1, SCHAEFER_TUREK, FlowProblem, TransfiniteBlocks,
                                          boundary_curves_where, count, group, near, num,
                                          refinement_factor, set_edge, transfinite_all)
from feacore.problems.registry import register

# Reference values (FEATFLOW 2D-1 page, verified through search-engine extracts of that page and of
# arXiv:1807.07380 Table 1; the FEATFLOW host is not reachable from the build session).
CD_REF = 5.57953523384
CL_REF = 0.010618948146
DP_REF = 0.11752016697

_SRC = (f"{SCHAEFER_TUREK} (problem 2D-1, Re = 20); high-accuracy reference values from the FEATFLOW "
        f"benchmark page {FEATFLOW_2D1} ('reference results', computed with high-order methods)")


class CylinderChannelDFG2D1(FlowProblem):
    id = "cylinder_channel_dfg_2d1"
    title = "DFG benchmark 2D-1: steady flow around a cylinder in a channel, Re = 20 (Schaefer & Turek 1996)"
    family = "cylinder_channel_dfg"
    tags = ("cfd", "2d", "flow", "laminar", "published_benchmark", "external_flow", "bluff_body",
            "expression_bc")
    expected_behavior = "smooth"
    default_params = {"L": 2.2, "H": 0.41, "xc": 0.2, "yc": 0.2, "D": 0.1, "Um": 0.3, "rho": 1.0, "mu": 1e-3,
                      "box": 0.1, "h0": 0.0125}
    # block counts at h0 (box spacing 0.2 / 16 = h0) and gradings (last / first cell size)
    n_arc0 = 16           # cells per quarter of the cylinder = per side of the box
    n_rad0 = 12           # cylinder -> box corner
    ratio_rad = 3.0
    n_right0 = 50         # box -> outlet
    ratio_right = 7.0

    # ---- derived quantities
    @property
    def U_mean(self) -> float:
        return 2.0 * self.params["Um"] / 3.0

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * self.U_mean * p["D"] / p["mu"]

    @property
    def force_scale(self) -> float:
        """rho U_bar^2 D / 2 (times the depth): force per coefficient unit [N]."""
        p = self.params
        return 0.5 * p["rho"] * self.U_mean ** 2 * p["D"] * self.thickness

    def inlet_profile(self) -> str:
        p = self.params
        return f"{num(4.0 * p['Um'] / p['H'] ** 2)}*y*({num(p['H'])} - y)"

    def points(self) -> dict[str, list[float]]:
        p = self.params
        r = 0.5 * p["D"]
        return {"front": [round(p["xc"] - r, 12), p["yc"]], "rear": [round(p["xc"] + r, 12), p["yc"]]}

    # ---- statement
    def description(self) -> str:
        p = self.params
        pts = self.points()
        return (
            "Steady laminar flow of an incompressible Newtonian fluid around a circular cylinder in a 2D "
            "channel (the DFG / Schaefer-Turek 'flow around a cylinder' benchmark, steady case).\n"
            f"Geometry (STEP, planar face in the xy-plane): channel 0 <= x <= {p['L']}, 0 <= y <= {p['H']} m "
            f"with a circular hole (the cylinder) of diameter D = {p['D']} m centred at ({p['xc']}, {p['yc']}). "
            "Model it as 2D; report forces per metre of depth (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  inlet x = 0: parabolic velocity profile u_x(y) = 4 U_m y (H - y) / H^2 with U_m = {p['Um']} m/s,\n"
            f"    i.e. u_x = {self.inlet_profile()}  [m/s, y in m],  u_y = 0\n"
            f"  outlet x = {p['L']}: static pressure p = 0 Pa\n"
            f"  channel walls y = 0 and y = {p['H']}, and the cylinder surface: no slip\n"
            "Report (SI units):\n"
            "  drag            x-component of the total force (pressure + viscous) of the fluid on the "
            "cylinder [N per m depth]\n"
            "  lift            y-component of that force [N per m depth]\n"
            f"  dp_front_rear   static pressure difference p(front) - p(rear) between the points "
            f"({pts['front'][0]}, {pts['front'][1]}) and ({pts['rear'][0]}, {pts['rear'][1]}) on the cylinder "
            "surface [Pa]\n"
            "The force QoIs refer to the boundary group cylinder: use this name for the cylinder surface "
            "group of the mesh."
        )

    # ---- geometry
    def build_cad(self) -> None:
        """Agent-facing CAD: one face (channel minus disk), no meshing blocks."""
        import gmsh

        p = self.params
        occ = gmsh.model.occ
        rect = occ.addRectangle(0.0, 0.0, 0.0, p["L"], p["H"])
        disk = occ.addDisk(p["xc"], p["yc"], 0.0, 0.5 * p["D"], 0.5 * p["D"])
        occ.cut([(2, rect)], [(2, disk)])
        occ.synchronize()

    def _corner(self, ang_deg: float, radius: float) -> tuple[float, float]:
        p = self.params
        a = math.radians(ang_deg)
        return (p["xc"] + radius * math.cos(a), p["yc"] + radius * math.sin(a))

    def build_geometry(self) -> None:
        """Reference geometry: the same face, split into 12 four-sided blocks for the structured mesh."""
        import gmsh

        p = self.params
        L, H, xc, yc, a, r = p["L"], p["H"], p["xc"], p["yc"], p["box"], 0.5 * p["D"]
        X = [0.0, xc - a, xc + a, L]
        Y = [0.0, yc - a, yc + a, H]
        b = TransfiniteBlocks()
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                p00, p10, p11, p01 = (X[i], Y[j]), (X[i + 1], Y[j]), (X[i + 1], Y[j + 1]), (X[i], Y[j + 1])
                b.block([b.line(p00, p10), b.line(p10, p11), b.line(p11, p01), b.line(p01, p00)])
        rb = a * math.sqrt(2.0)
        for t1 in (-45.0, 45.0, 135.0, 225.0):
            t2 = t1 + 90.0
            c1, c2 = self._corner(t1, r), self._corner(t2, r)
            b1, b2 = self._corner(t1, rb), self._corner(t2, rb)
            b1 = (round(b1[0], 12), round(b1[1], 12))
            b2 = (round(b2[0], 12), round(b2[1], 12))
            b.block([b.line(c1, b1), b.line(b1, b2), b.line(b2, c2), b.arc(c2, (xc, yc, 0.0), c1)])
        b.finish()
        self._blocks = b
        tol = 1e-6
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0, tol)), "inlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x0, L, tol)), "outlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0, tol) or near(y0, H, tol)),
              "walls")
        group(1, [abs(t) for t in b.arcs.values()], "cylinder")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Edge counts from the refinement factor k = h0 / h; every block edge is classified by its
        geometry (horizontal / vertical straight edge, radial diagonal, cylinder arc)."""
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        L, H, xc, yc, a = p["L"], p["H"], p["xc"], p["yc"], p["box"]
        X = [0.0, xc - a, xc + a, L]
        Y = [0.0, yc - a, yc + a, H]
        h_box = 2 * a / self.n_arc0                       # box spacing at h0 (= h0 by default)
        n_arc = count(self.n_arc0, k)
        nx = [count(round((X[1] - X[0]) / h_box), k), n_arc, count(self.n_right0, k)]
        ny = [count(round((Y[1] - Y[0]) / h_box), k), n_arc, count(round((Y[3] - Y[2]) / h_box), k)]
        n_rad = count(self.n_rad0, k)
        b = self._blocks
        done = set()
        tol = 1e-9

        def interval(lo, hi, grid):
            for i in range(3):
                if abs(lo - grid[i]) < tol and abs(hi - grid[i + 1]) < tol:
                    return i
            raise RuntimeError(f"{self.id}: edge [{lo}, {hi}] matches no block interval of {grid}")

        for (pa, pb), tag in b.lines.items():
            (xa, ya, _), (xb, yb, _) = b.coords[pa], b.coords[pb]
            if abs(ya - yb) < tol:                         # horizontal: x-interval, graded downstream
                i = interval(min(xa, xb), max(xa, xb), X)
                signed = tag if xb > xa else -tag
                done.add(set_edge(signed, nx[i], self.ratio_right if i == 2 else 1.0))
            elif abs(xa - xb) < tol:                       # vertical: y-interval, uniform
                done.add(set_edge(tag, ny[interval(min(ya, yb), max(ya, yb), Y)]))
            else:                                          # diagonal: graded away from the cylinder
                ra, rb = math.hypot(xa - xc, ya - yc), math.hypot(xb - xc, yb - yc)
                done.add(set_edge(tag if rb > ra else -tag, n_rad, self.ratio_rad))
        for tag in b.arcs.values():
            done.add(set_edge(tag, n_arc))
        missing = [t for _, t in gmsh.model.getEntities(1) if t not in done]
        if missing:
            raise RuntimeError(f"{self.id}: edges without a transfinite count: {missing}")
        transfinite_all(2)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("drag", "force", group="cylinder", component=0),
            QoI("lift", "force", group="cylinder", component=1),
            QoI("dp_front_rear", "point_difference", field="p", point=pts["front"], point2=pts["rear"]),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [
            BC("velocity_inlet", group="inlet", value=[self.inlet_profile(), 0.0]),
            BC("pressure_outlet", group="outlet", value=0.0),
            BC("wall", group="walls"),
            BC("wall", group="cylinder"),
        ]
        return self.make_case(mesh_path, order, bcs)

    def references(self) -> list[Reference]:
        s = self.force_scale
        conv = f"F = c rho U_bar^2 D / 2 per m depth, rho U_bar^2 D / 2 = {s:.6g} N/m"
        return [
            Reference("drag", CD_REF * s, "published_numerical", f"{_SRC}; c_D = {CD_REF}; {conv}", tolerance=0.01,
                      notes="Schaefer & Turek (1996) acceptance interval c_D in [5.57, 5.59] (from memory)."),
            Reference("lift", CL_REF * s, "published_numerical", f"{_SRC}; c_L = {CL_REF}; {conv}", tolerance=0.05,
                      notes="Lift is ~0.2 % of drag and sensitive to the mesh; Schaefer & Turek (1996) acceptance "
                            "interval c_L in [0.0104, 0.0110] (+-3 %, from memory)."),
            Reference("dp_front_rear", DP_REF, "published_numerical",
                      f"{_SRC}; dp = p(0.15, 0.2) - p(0.25, 0.2) = {DP_REF} (rho = 1, so Pa)", tolerance=0.01,
                      notes="Schaefer & Turek (1996) acceptance interval [0.1172, 0.1176] (from memory)."),
        ]


register(CylinderChannelDFG2D1())

__all__ = ["CylinderChannelDFG2D1", "CD_REF", "CL_REF", "DP_REF"]
