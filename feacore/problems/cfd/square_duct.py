"""Developing laminar flow in a straight square duct (3D); exact fully developed values downstream.

Duct 0 <= x <= L with square section |y|, |z| <= s / 2, no-slip walls, UNIFORM inflow U at x = 0,
static pressure 0 at the outlet x = L. The flow develops over the first few duct widths and is
then fully developed, with the exact series solution of the Poisson equation mu (u_yy + u_zz) = -G
for a rectangle |y| <= a,
|z| <= b (White, Viscous Fluid Flow, eq. 3-48; Shah & London 1978):

    u(y, z) = 16 a^2 G / (mu pi^3) sum_{i = 1, 3, 5, ...} (-1)^((i-1)/2)
              [1 - cosh(i pi z / 2a) / cosh(i pi b / 2a)] cos(i pi y / 2a) / i^3
    Q       = 4 b a^3 G / (3 mu) [1 - 192 a / (pi^5 b) sum_{i odd} tanh(i pi b / 2a) / i^5]

For a square (a = b = s / 2) and mean velocity U = Q / s^2 the series give u_max / U = 2.096256
(at the axis), u(y = s/4, z = 0) / U = 1.631416 and G s^2 / (mu U) = 28.4542, i.e. a Fanning
friction factor f Re = 14.227 (the classical Shah & London value). These are evaluated here to full
precision; the probes sit in the developed region (x = 0.5 L .. 0.9 L with L = 10 s), far from both
the inlet and the outlet. The outlet flow rate is U s^2 by mass conservation.

Default case (Re = rho U s / mu = 10): s = 1 cm, L = 10 cm, rho = 1000 kg/m^3, mu = 0.01 Pa s,
U = 1 cm/s. Reference mesh: structured hexahedra, s / h cells across each side and L / h along.
Checked with the fixed solver (OpenFOAM v1912): on 8 and 16 cells across, u_axis is 6.69 % then
1.78 % low and dp_AB 5.55 % then 1.47 % (second order); the Richardson estimate is within 0.15 % of
the exact developed values, so the probes are in fully developed flow.
"""

from __future__ import annotations

import numpy as np

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (WHITE, FlowProblem, count, group, refinement_factor, set_edge,
                                          transfinite_all)
from feacore.problems.registry import register

_SRC = ("Exact series solution for fully developed laminar flow in a rectangular duct, "
        f"{WHITE}, eq. 3-48 (equation number from memory); R.K. Shah & A.L. London, Laminar Flow Forced "
        "Convection in Ducts, Academic Press (1978) (square duct f Re = 14.227); evaluated in "
        "feacore/problems/cfd/square_duct.py")


def duct_series(a: float, b: float, y: float, z: float, terms: int = 400) -> tuple[float, float]:
    """(u mu / G at (y, z), Q mu / G) for fully developed flow in |y| <= a, |z| <= b."""
    i = np.arange(1, 2 * terms, 2, dtype=float)
    c = i * np.pi / (2 * a)
    z = abs(z)
    ratio = np.exp(c * (z - b)) * (1 + np.exp(-2 * c * z)) / (1 + np.exp(-2 * c * b))   # cosh(c z) / cosh(c b)
    u = 16 * a * a / np.pi ** 3 * np.sum((-1.0) ** ((i - 1) / 2) * (1 - ratio) * np.cos(c * y) / i ** 3)
    q = 4 * b * a ** 3 / 3 * (1 - 192 * a / (np.pi ** 5 * b) * np.sum(np.tanh(i * np.pi * b / (2 * a)) / i ** 5))
    return float(u), float(q)


class SquareDuct3D(FlowProblem):
    id = "square_duct_3d"
    title = "Developing laminar flow in a square duct, uniform inflow (3D, exact developed values)"
    family = "square_duct"
    tags = ("cfd", "3d", "flow", "laminar", "analytical", "internal_flow", "developing_flow")
    dim = 3
    formulation = "3d"
    expected_behavior = "smooth"   # the inlet edges (uniform inflow meets no-slip) are far from the probes
    max_iterations = 5000
    default_params = {"s": 0.01, "L": 0.1, "U": 0.01, "rho": 1000.0, "mu": 0.01, "h0": 0.00125}

    # ---- exact developed flow
    def developed(self) -> dict[str, float]:
        """u_axis / U, u(s/4, 0) / U and G (= -dp/dx) of the fully developed flow."""
        p = self.params
        a = p["s"] / 2
        u0, q = duct_series(a, a, 0.0, 0.0)
        u1, _ = duct_series(a, a, a / 2, 0.0)
        mean = q / (4 * a * a)                 # U per unit G / mu
        return {"axis": u0 / mean, "quarter": u1 / mean, "G": p["mu"] * p["U"] / mean}

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["U"] * p["s"] / p["mu"]

    def points(self) -> dict[str, list[float]]:
        p = self.params
        L, s = p["L"], p["s"]
        pts = {"A": [0.5 * L, 0.0, 0.0], "B": [0.9 * L, 0.0, 0.0], "C": [0.75 * L, 0.0, 0.0],
               "D": [0.75 * L, s / 4, 0.0]}
        return {k: [round(v, 12) for v in xyz] for k, xyz in pts.items()}

    # ---- statement
    def description(self) -> str:
        p = self.params
        pts = self.points()
        h = p["s"] / 2
        return (
            "Steady laminar flow of an incompressible Newtonian fluid through a straight duct of square "
            "cross-section (3D).\n"
            f"Geometry (STEP, one solid): 0 <= x <= L along the duct, -{h} <= y <= {h} and -{h} <= z <= {h} "
            f"m across it, with L = {p['L']} m (side s = {p['s']} m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  inlet x = 0: uniform velocity (u_x, u_y, u_z) = ({p['U']}, 0, 0) m/s\n"
            "  outlet x = L: static pressure p = 0 Pa\n"
            "  the four side walls: no slip (stationary)\n"
            "Report (SI units):\n"
            f"  u_axis            u_x on the duct axis at C = ({pts['C'][0]}, {pts['C'][1]}, {pts['C'][2]}) [m/s]\n"
            f"  u_offaxis         u_x at D = ({pts['D'][0]}, {pts['D'][1]}, {pts['D'][2]}) [m/s]\n"
            f"  dp_AB             static pressure difference p(A) - p(B) between the axis points "
            f"A = ({pts['A'][0]}, 0, 0) and B = ({pts['B'][0]}, 0, 0) [Pa]\n"
            "  flow_rate_outlet  volumetric flow rate through the outlet, positive out of the duct [m^3/s]\n"
            "The mesh groups must be named inlet, outlet and walls."
        )

    # ---- geometry and mesh
    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        L, s = p["L"], p["s"]
        vol = gmsh.model.occ.addBox(0.0, -s / 2, -s / 2, L, s, s)
        gmsh.model.occ.synchronize()
        tol = 1e-4 * s     # OCC bounding boxes are padded by ~1e-7
        faces = {"inlet": [], "outlet": [], "walls": []}
        for _, f in gmsh.model.getEntities(2):
            x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(2, f)
            key = "inlet" if x1 < tol else "outlet" if x0 > L - tol else "walls"
            faces[key].append(f)
        for name, tags in faces.items():
            group(2, tags, name)
        group(3, [vol], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Structured hexahedra: s / h cells across, L / h along (8 x 8 x 80 at h0)."""
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        n_s = count(round(p["s"] / p["h0"]), k)
        n_l = count(round(p["L"] / p["h0"]), k)
        for _, c in gmsh.model.getEntities(1):
            x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(1, c)
            set_edge(c, n_l if x1 - x0 > 0.5 * p["s"] else n_s)
        transfinite_all(3)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("u_axis", "point", field="ux", point=pts["C"]),
            QoI("u_offaxis", "point", field="ux", point=pts["D"]),
            QoI("dp_AB", "point_difference", field="p", point=pts["A"], point2=pts["B"]),
            QoI("flow_rate_outlet", "flow_rate", group="outlet"),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [
            BC("velocity_inlet", group="inlet", value=[self.params["U"], 0.0, 0.0]),
            BC("pressure_outlet", group="outlet", value=0.0),
            BC("wall", group="walls"),
        ]
        return self.make_case(mesh_path, order, bcs)

    def references(self) -> list[Reference]:
        p = self.params
        d = self.developed()
        pts = self.points()
        return [
            Reference("u_axis", d["axis"] * p["U"], "analytical",
                      _SRC + f"; developed u_max / U = {d['axis']:.6f}", tolerance=0.02),
            Reference("u_offaxis", d["quarter"] * p["U"], "analytical",
                      _SRC + f"; developed u(s/4, 0) / U = {d['quarter']:.6f}", tolerance=0.02),
            Reference("dp_AB", d["G"] * (pts["B"][0] - pts["A"][0]), "analytical",
                      _SRC + f"; developed G s^2 / (mu U) = {d['G'] * p['s'] ** 2 / (p['mu'] * p['U']):.4f}",
                      tolerance=0.02),
            Reference("flow_rate_outlet", p["U"] * p["s"] ** 2, "analytical",
                      _SRC + "; mass conservation: Q = U s^2 (the uniform inflow leaves through the outlet)",
                      tolerance=0.01),
        ]


register(SquareDuct3D())

__all__ = ["SquareDuct3D", "duct_series"]
