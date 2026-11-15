"""Decaying Taylor-Green vortex in a closed square (2D, unsteady, exact).

The unit square 0 <= x, y <= L (L = 1 m) holds one cell of the classical array of counter-rotating
vortices (k = pi / L):

    u = U0 sin(k x) cos(k y) F(t),   v = -U0 cos(k x) sin(k y) F(t),   F(t) = exp(-2 nu k^2 t)
    p = (rho U0^2 / 4) (cos(2 k x) + cos(2 k y)) F(t)^2 (+ a constant)

It is an exact solution of the unsteady incompressible Navier-Stokes equations for any Reynolds
number (the convection is a gradient, balanced by the pressure; Taylor & Green 1937). On the edges
of the square the normal velocity, the tangential shear and the normal pressure gradient vanish,
so the four edges are symmetry planes and the square is a closed domain with no inflow. The kinetic
energy decays as F(t)^2 = exp(-4 nu k^2 t).

Default case: U0 = 1 m/s, rho = 1 kg/m^3, mu = 0.05 Pa s (nu = 0.05 m^2/s, Re = U0 L / nu = 20),
t_end = 1 s (F = exp(-0.1 pi^2) = 0.373), step 0.005 s. The fluid starts in the exact state at t = 0.

QoIs (all exact): ux at (0.5, 0.25) at t_end; the time mean of the same velocity over the run,
U0 sin(k/2) cos(k/4) (1 - F(t_end)) / (2 nu k^2 t_end); the pressure difference p(0.5, 0.5) -
p(0.25, 0.5) = -rho U0^2 F(t_end)^2 / 4 at t_end (the vortex centre C is the pressure minimum).

Reference mesh: uniform square quads, L / h cells per side (16 at h0). The fixed solver (OpenFOAM
v1912, pimpleFoam, BDF2, linearUpwind) at step 0.005 s: ux at t_end is 0.94 %, 0.19 %, 0.042 % low on
16, 32 and 64 cells per side (second order); BDF2 converges with order 2.0 and Euler with order 1.0
under step halving (docs/solver_transient.md, "Transient incompressible flow").

The edges are "symmetry", not "slip": OpenFOAM's slip condition on a plain patch is first-order
accurate here (the decay is too fast by 5.6 % of the viscous rate at 16 cells, 2.9 % at 32), while
the symmetry condition converges with second order (docs/solver_transient.md).
"""

from __future__ import annotations

import math

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (FlowProblem, count, group, num, refinement_factor, set_edge,
                                          transfinite_all)
from feacore.problems.registry import register

TAYLOR_GREEN = ("G.I. Taylor & A.E. Green, 'Mechanism of the production of small eddies from large ones', "
                "Proc. R. Soc. Lond. A 158 (1937) 499-521")
_SRC = (f"Exact solution of the unsteady incompressible Navier-Stokes equations (the decaying Taylor-Green "
        f"vortex, {TAYLOR_GREEN}; 2D form and decay rate derived in feacore/problems/cfd/taylor_green.py)")


class TaylorGreenVortex2D(FlowProblem):
    id = "taylor_green_vortex_2d"
    title = "Decaying Taylor-Green vortex in a closed square (2D, unsteady, laminar, exact)"
    family = "taylor_green"
    tags = ("cfd", "2d", "flow", "laminar", "transient", "analytical", "closed_cavity", "symmetry_bc",
            "expression_ic")
    expected_behavior = "smooth"
    default_params = {"L": 1.0, "U0": 1.0, "rho": 1.0, "mu": 0.05, "t_end": 1.0, "dt": 0.005, "h0": 1.0 / 16}

    # ---- exact solution
    @property
    def k(self) -> float:
        return math.pi / self.params["L"]

    @property
    def nu(self) -> float:
        return self.params["mu"] / self.params["rho"]

    def decay(self, t: float) -> float:
        """F(t) = exp(-2 nu k^2 t), the decay of the velocity."""
        return math.exp(-2.0 * self.nu * self.k ** 2 * t)

    def exact(self, x: float, y: float, t: float) -> tuple[float, float, float]:
        """(u, v, p) at (x, y, t); p up to a constant."""
        p, k, F = self.params, self.k, self.decay(t)
        u = p["U0"] * math.sin(k * x) * math.cos(k * y) * F
        v = -p["U0"] * math.cos(k * x) * math.sin(k * y) * F
        pr = 0.25 * p["rho"] * p["U0"] ** 2 * (math.cos(2 * k * x) + math.cos(2 * k * y)) * F ** 2
        return u, v, pr

    def points(self) -> dict[str, list[float]]:
        L = self.params["L"]
        return {"A": [0.5 * L, 0.25 * L], "C": [0.5 * L, 0.5 * L], "B": [0.25 * L, 0.5 * L]}

    def initial(self) -> dict:
        p, k = self.params, num(self.k)
        U0, q = num(p["U0"]), num(0.25 * p["rho"] * p["U0"] ** 2)
        return {"U": [f"{U0}*sin({k}*x)*cos({k}*y)", f"-{U0}*cos({k}*x)*sin({k}*y)"],
                "p": f"{q}*(cos(2*{k}*x) + cos(2*{k}*y))"}

    def options(self) -> dict:
        p = self.params
        return {**super().options(),
                "transient": {"t_end": p["t_end"], "dt": p["dt"], "scheme": "backward", "initial": self.initial()}}

    # ---- statement
    def description(self) -> str:
        p = self.params
        pts = self.points()
        L = p["L"]
        ini = self.initial()
        return (
            "Unsteady laminar flow of an incompressible Newtonian fluid in a closed square: the decay of a "
            "Taylor-Green vortex cell.\n"
            f"Geometry (STEP, planar face in the xy-plane): the square 0 <= x <= {L}, 0 <= y <= {L} m. "
            "Model it as 2D (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions: all four edges are symmetry planes (zero normal velocity, zero shear); "
            "group name walls.\n"
            "Initial state at t = 0 [m/s, Pa, x and y in m]:\n"
            f"  u_x = {ini['U'][0]}\n  u_y = {ini['U'][1]}\n  p = {ini['p']}\n"
            f"Time: simulate from t = 0 to t = {p['t_end']} s (a time-dependent analysis).\n"
            "Report (SI units):\n"
            f"  ux_A        the velocity component u_x at A = ({pts['A'][0]}, {pts['A'][1]}) at t = {p['t_end']} s\n"
            f"  ux_A_mean   the time mean of u_x at A over 0 <= t <= {p['t_end']} s\n"
            f"  dp_CB       the pressure difference p(C) - p(B) at t = {p['t_end']} s, C = ({pts['C'][0]}, "
            f"{pts['C'][1]}), B = ({pts['B'][0]}, {pts['B'][1]}) [Pa]"
        )

    # ---- geometry
    def build_geometry(self) -> None:
        import gmsh

        L = self.params["L"]
        s = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
        gmsh.model.occ.synchronize()
        group(1, [t for _, t in gmsh.model.getEntities(1)], "walls")
        group(2, [s], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Uniform square cells, L / h per side (16 at h0)."""
        import gmsh

        p = self.params
        n = count(round(p["L"] / p["h0"]), refinement_factor(p["h0"], h))
        for _, c in gmsh.model.getEntities(1):
            set_edge(c, n)
        transfinite_all(2)

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("ux_A", "point", field="ux", point=pts["A"]),
            QoI("ux_A_mean", "point", field="ux", point=pts["A"], time="mean"),
            QoI("dp_CB", "point_difference", field="p", point=pts["C"], point2=pts["B"]),
        ]

    def case(self, mesh_path, order: int = 2):
        return self.make_case(mesh_path, order, [BC("symmetry", group="walls")])

    def references(self) -> list[Reference]:
        p = self.params
        T, k, nu = p["t_end"], self.k, self.nu
        A = self.points()["A"]
        ux_a = self.exact(A[0], A[1], T)[0]
        amp = self.exact(A[0], A[1], 0.0)[0]
        mean = amp * (1.0 - self.decay(T)) / (2.0 * nu * k ** 2 * T)
        C, B = self.points()["C"], self.points()["B"]
        dp = self.exact(C[0], C[1], T)[2] - self.exact(B[0], B[1], T)[2]
        return [
            Reference("ux_A", ux_a, "analytical", f"{_SRC}: u = U0 sin(kx) cos(ky) exp(-2 nu k^2 t)",
                      tolerance=0.02, notes="the reference mesh h0 is 0.94 % low; second order in h and in the step"),
            Reference("ux_A_mean", mean, "analytical",
                      f"{_SRC}: the time mean of U0 sin(kx) cos(ky) exp(-2 nu k^2 t) over [0, t_end]",
                      tolerance=0.02, notes="the mean is taken with the trapezoidal rule over the steps"),
            Reference("dp_CB", dp, "analytical",
                      f"{_SRC}: p = (rho U0^2 / 4)(cos 2kx + cos 2ky) exp(-4 nu k^2 t)", tolerance=0.02),
        ]


register(TaylorGreenVortex2D())

__all__ = ["TaylorGreenVortex2D", "TAYLOR_GREEN"]
