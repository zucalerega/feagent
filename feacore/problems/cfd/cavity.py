"""Lid-driven square cavity at Re = 100 (2D, steady, laminar): Ghia, Ghia & Shin (1982).

Unit square 0 <= x, y <= 1, lid y = 1 moving at U = 1 m/s in +x, the other three walls at rest,
rho = 1 kg/m^3, mu = 0.01 Pa s, so Re = rho U L / mu = 100. Closed domain: the pressure level is
arbitrary, so every QoI is a velocity.

References: Ghia, Ghia & Shin (1982), Table I (u along the vertical line through the geometric
centre, x = 0.5) and Table II (v along the horizontal line through the centre, y = 0.5), Re = 100
column. Their 129 x 129 uniform-grid values sit on grid points y_j = (j - 1) / 128, which the
tables print rounded to 4 decimals (0.4531 = 58/128, 0.2344 = 30/128, 0.8047 = 103/128). The QoIs
use the exact grid coordinates. The tabulated extremes along the two centrelines are

    u(0.5, 58/128)  = -0.21090   (minimum of the u column)
    v(30/128, 0.5)  =  0.17527   (maximum of the v column)
    v(103/128, 0.5) = -0.24533   (minimum of the v column)
    u(0.5, 0.5)     = -0.20581   (centre)

The table values were checked against the public transcription in PySPH
(pysph/examples/ghia_cavity_data.py). The extremes of the continuous profiles lie within one table
spacing of these points, where the profiles are flat.

Ghia's Re = 100 values are 1.4-3.5 % smaller in magnitude than mesh-converged solutions: the fixed
solver's Richardson-extrapolated values (32/64/128 grids, observed order ~2, GCI <= 0.25 %) are
u(0.5, 58/128) = -0.21392, v(30/128, 0.5) = 0.17951, v(103/128, 0.5) = -0.25379 and
u(0.5, 0.5) = -0.20914; a 4-level study up to 256 x 256 agrees to 1e-4. Fine-grid literature
deviates from Ghia in the same direction (Sahin & Owens 2003). The tolerances (3-5 %) cover that
table error plus the agent's own discretization error.

Mesh: uniform structured quad grid, n x n cells (32 / 64 / 128 for the reference study). The lid
corners are singular (velocity jump), which pollutes the solution only weakly at the centrelines.
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (GHIA, GHIA_COPY, FlowProblem, boundary_curves_where, count, group,
                                          near, refinement_factor, set_edge, transfinite_all)
from feacore.problems.registry import register

_SRC = f"{GHIA}, Re = 100 column; table values as transcribed in {GHIA_COPY}"

# (QoI name, field, point, Ghia value, table, tolerance, converged value of the fixed solver)
# Tolerances: Ghia's Re = 100 values are smaller in magnitude than mesh-converged solutions by
# 1.4 % (u) to 3.5 % (v_min) (fixed solver, Richardson-extrapolated 32/64/128 study, GCI <= 0.25 %;
# the same direction as the fine-grid finite-volume values of Sahin & Owens 2003). The tolerance is
# that table error plus ~1.5 % for the agent's own discretization error.
_GHIA_POINTS = (
    ("u_vcl_min", "ux", (0.5, 58 / 128), -0.21090, "Table I (u on x = 0.5), y = 0.4531", 0.03, -0.21392),
    ("v_hcl_max", "uy", (30 / 128, 0.5), 0.17527, "Table II (v on y = 0.5), x = 0.2344", 0.04, 0.17951),
    ("v_hcl_min", "uy", (103 / 128, 0.5), -0.24533, "Table II (v on y = 0.5), x = 0.8047", 0.05, -0.25379),
    ("u_centre", "ux", (0.5, 0.5), -0.20581, "Table I (u on x = 0.5), y = 0.5000", 0.03, -0.20914),
)


class LidDrivenCavityRe100(FlowProblem):
    id = "lid_driven_cavity_re100"
    title = "Lid-driven square cavity, Re = 100 (Ghia, Ghia & Shin 1982)"
    family = "lid_driven_cavity"
    tags = ("cfd", "2d", "flow", "laminar", "published_benchmark", "closed_cavity", "moving_wall")
    expected_behavior = "smooth"   # corner singularities at the lid ends, weak at the centrelines
    default_params = {"L": 1.0, "U": 1.0, "rho": 1.0, "mu": 0.01, "h0": 1.0 / 32}

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["U"] * p["L"] / p["mu"]

    def _scaled(self, xy):
        L = self.params["L"]
        return [xy[0] * L, xy[1] * L]

    def description(self) -> str:
        p = self.params
        L, U = p["L"], p["U"]
        lines = []
        for name, f, xy, *_ in _GHIA_POINTS:
            x, y = self._scaled(xy)
            comp = "u_x (horizontal velocity)" if f == "ux" else "u_y (vertical velocity)"
            lines.append(f"  {name:<10} {comp} at ({x:.7g}, {y:.7g}) [m/s]")
        return (
            "Steady laminar flow of an incompressible Newtonian fluid in a lid-driven square cavity "
            "(2D).\n"
            f"Geometry (STEP, planar face in the xy-plane): square 0 <= x <= {L}, 0 <= y <= {L} m. "
            "Model it as 2D (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions:\n"
            f"  lid y = {L}: wall moving tangentially with velocity (u_x, u_y) = ({U}, 0) m/s (no slip "
            "relative to the lid)\n"
            f"  bottom y = 0 and side walls x = 0, x = {L}: stationary no-slip walls\n"
            "The cavity is closed (no inlet or outlet), so only velocities are reported.\n"
            "Report (SI units), velocities at points on the two centrelines:\n" + "\n".join(lines)
        )

    def build_geometry(self) -> None:
        import gmsh

        L = self.params["L"]
        s = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
        gmsh.model.occ.synchronize()
        tol = 1e-6 * L
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(y0, L, tol)), "lid")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: not near(y0, L, tol)), "walls")
        group(2, [s], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Uniform n x n grid, n = L / h (32 at h0)."""
        import gmsh

        p = self.params
        n = count(round(p["L"] / p["h0"]), refinement_factor(p["h0"], h))
        for _, c in gmsh.model.getEntities(1):
            set_edge(c, n)
        transfinite_all(2)

    def qois(self) -> list[QoI]:
        return [QoI(name, "point", field=f, point=self._scaled(xy)) for name, f, xy, *_ in _GHIA_POINTS]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("wall", group="lid", value=[self.params["U"], 0.0]),
               BC("wall", group="walls")]
        return self.make_case(mesh_path, order, bcs)

    def references(self) -> list[Reference]:
        U = self.params["U"]
        refs = []
        for name, _, xy, v, table, tol, conv in _GHIA_POINTS:
            refs.append(Reference(
                name, v * U, "published_numerical", f"{_SRC}; {table}", tolerance=tol,
                notes=(f"Ghia et al.: finite-difference stream function-vorticity multigrid solution on "
                       f"a uniform grid (129 x 129 for Re = 100, from memory); grid-point values to 5 "
                       f"decimals. The table is {abs(conv - v) / abs(v):.1%} smaller in magnitude than the "
                       f"mesh-converged value {conv} of the fixed solver (OpenFOAM, Richardson-extrapolated "
                       f"32/64/128 study, GCI <= 0.25 %), consistent with fine-grid literature (Sahin & "
                       f"Owens 2003); the tolerance covers that table error plus the agent's discretization "
                       f"error. See docs/benchmarks_cfd.md.")))
        return refs


register(LidDrivenCavityRe100())

__all__ = ["LidDrivenCavityRe100"]
