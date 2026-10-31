"""Re-entrant corner problems.

lshape_poisson: Laplace equation (heat physics, k = 1, no source) on the L-shaped domain
(-1, 1)^2 minus [0, 1] x [-1, 0], re-entrant corner (270 degrees) at the origin. Exact solution
    u = r^(2/3) sin(2 theta / 3),  theta in [0, 3 pi / 2] measured from the +x axis,
imposed as Dirichlet data (u = 0 on the two edges meeting at the corner). grad u ~ r^(-1/3) is
unbounded at the corner (Grisvard, Elliptic Problems in Nonsmooth Domains, 1985; the standard
adaptive-FEM benchmark, e.g. Morin, Nochetto & Siebert, SIAM J. Numer. Anal. 38 (2000) 466).
On quasi-uniform meshes the energy error is O(h^(2/3)) for any element order and smooth
functionals (point values away from the corner, integrals) converge at O(h^(4/3)) (pollution by
the corner singularity, duality argument), so expected orders are 4/3, not order + 1.

The branch cut of the expression is rotated into the removed quadrant: with
psi = arctan2(-x - y, y - x) = theta - 3 pi/4 in (-pi, pi), u = r^(2/3) cos(2 psi / 3), smooth on the
whole closed domain (no sign problems for nodes with y = -0.0 on the edge theta = 0).

lbracket_reentrant: plane-stress L-shaped bracket with a sharp re-entrant corner, one leg clamped,
the other loaded. The stress is singular at the corner (Williams 1952: for a 270 degree traction-free
corner the leading exponent is lambda = 0.5445, sigma ~ r^(lambda - 1)), so the maximum von Mises
stress diverges under refinement while the tip displacement and the strain energy converge, at the
reduced rate 2 lambda ~ 1.09 on uniform meshes. No reference value: the convergence study is the
check (singularity detection).
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import ALUMINIUM, ExactProblem, num, polygon
from feacore.problems.base import Reference
from feacore.problems.registry import register

LSHAPE_INTEGRAL = 1.5839289449053858  # int_Omega r^(2/3) sin(2 theta/3) dA, see LShapePoisson.references
WILLIAMS_270 = 0.5444837367824632      # smallest root of sin(2 lambda alpha) = -lambda sin(2 alpha), alpha = 3 pi / 4


class LShapePoisson(ExactProblem):
    id = "lshape_poisson"
    title = "Laplace equation on an L-shaped domain with a re-entrant corner (exact r^(2/3) solution)"
    family = "lshape_poisson"
    tags = ("2d", "heat", "poisson", "analytical", "singular", "reentrant_corner", "expression_bc")
    dim = 2
    physics = "heat"
    formulation = "2d"
    expected_behavior = "singular"
    default_params = {"k": 1.0, "h0": 0.2}
    material_name = "unit"

    EXPR = "(x**2 + y**2)**(1/3)*cos(2/3*arctan2(-x - y, y - x))"

    def material(self):
        from feacore.case import Material

        return Material(k=self.params["k"])

    def description(self) -> str:
        return (
            "Steady heat conduction / Laplace equation (k = 1, no heat source) on an L-shaped plate with a "
            "re-entrant corner. Units: lengths in m, T in K (the problem is dimensionless in effect).\n"
            "Geometry (STEP, planar face in the xy-plane): the square (-1, 1) x (-1, 1) m with the quadrant "
            "[0, 1] x [-1, 0] removed. Vertices (0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1); the "
            "re-entrant (270 degree) corner is at the origin.\n"
            f"Material: conductivity k = {self.params['k']} W/(m K).\n"
            "Boundary conditions (Dirichlet everywhere):\n"
            "  the two edges meeting at the re-entrant corner, y = 0 for 0 <= x <= 1 and x = 0 for "
            "-1 <= y <= 0: T = 0;\n"
            f"  the four outer edges x = 1, y = 1, x = -1, y = -1: T(x, y) = {self.EXPR}\n"
            "  (this is r^(2/3) sin(2 theta/3) with theta measured from the +x axis, 0 <= theta <= 3 pi/2).\n"
            "Report:\n"
            "  T_point     temperature at (-0.5, 0.5) [K]\n"
            "  q_max       maximum heat-flux magnitude |q| = k |grad T| over the domain [W/m^2]\n"
            "  T_integral  integral of T over the domain [K m^2]"
        )

    def build_geometry(self) -> None:
        pts = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1)]
        lines, s = polygon(pts)
        add_group(1, [lines[0], lines[5]], "reentrant_edges")
        add_group(1, lines[1:5], "outer")
        add_group(2, [s], "domain")

    def qois(self) -> list[QoI]:
        return [QoI("T_point", "point", field="T", point=[-0.5, 0.5]),
                QoI("q_max", "max", field="q_mag"),
                QoI("T_integral", "integral", field="T", group="domain")]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("temperature", group="reentrant_edges", value=0.0),
               BC("temperature", group="outer", value=self.EXPR)]
        return self.make_case(mesh_path, order, bcs)

    def _primary(self, x, y, z):
        r2 = x * x + y * y
        psi = np.arctan2(-x - y, y - x)
        r = np.sqrt(r2)
        T = np.cbrt(r2) * np.cos(2 * psi / 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            # grad u in polar: du/dr = 2/3 r^(-1/3) cos(2psi/3), (1/r) du/dpsi = -2/3 r^(-1/3) sin(2psi/3);
            # theta = psi + 3pi/4, e_r = (cos theta, sin theta), e_t = (-sin theta, cos theta)
            th = psi + 0.75 * math.pi
            gr = 2.0 / 3.0 * np.cos(2 * psi / 3) / np.cbrt(r)
            gt = -2.0 / 3.0 * np.sin(2 * psi / 3) / np.cbrt(r)
            k = self.params["k"]
            qx = -k * (gr * np.cos(th) - gt * np.sin(th))
            qy = -k * (gr * np.sin(th) + gt * np.cos(th))
        qx = np.where(r > 0, qx, np.inf)
        qy = np.where(r > 0, qy, np.inf)
        return {"T": T, "qx": qx, "qy": qy}

    def expectations(self) -> dict[str, str]:
        return {"q_max": "diverge"}

    def expected_order(self, qoi: QoI, order: int) -> float | None:
        if qoi.name in ("T_point", "T_integral"):
            return 4.0 / 3.0
        return None

    def references(self) -> list[Reference]:
        src = ("exact harmonic function u = r^(2/3) sin(2 theta/3) on the L-shaped domain (re-entrant corner "
               "singularity, Grisvard 1985), imposed as Dirichlet data")
        return [
            self.point_reference("T_point", src + "; T(-0.5, 0.5) = 0.5^(1/3)", 0.005),
            Reference("T_integral", LSHAPE_INTEGRAL, "analytical",
                      src + "; integral over the domain by tanh-sinh quadrature (mpmath, 30 digits) on the "
                      "three unit squares, confirmed by an independent polar-coordinate quadrature "
                      "(int_0^R r^(5/3) dr = 3/8 R^(8/3)); the two agree to 30 digits", 0.005),
        ]

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]


class LBracketReentrant(ExactProblem):
    id = "lbracket_reentrant"
    title = "L-shaped bracket with a sharp re-entrant corner (plane stress, singular stress)"
    family = "lbracket_reentrant"
    tags = ("2d", "plane_stress", "singular", "reentrant_corner", "no_reference")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "singular"
    material_name = "aluminium"
    default_params = {"b": 0.04, "H": 0.12, "L": 0.12, "F": 1000.0, **ALUMINIUM, "thickness": 0.01, "h0": 0.01}

    def description(self) -> str:
        p = self.params
        b, H, L = p["b"], p["H"], p["L"]
        return (
            "Aluminium L-shaped bracket, plane stress. The inner (re-entrant) corner is sharp (no fillet).\n"
            f"Geometry (STEP, planar face in the xy-plane): the union of a vertical leg [0, {b}] x [0, {H}] m "
            f"and a horizontal leg [0, {L}] x [0, {b}] m. Vertices (0, 0), ({L}, 0), ({L}, {b}), ({b}, {b}), "
            f"({b}, {H}), (0, {H}); the re-entrant corner is at ({b}, {b}). Thickness {p['thickness']} m.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}.\n"
            f"Supports: the top end of the vertical leg, y = {H} m (0 <= x <= {b}), is fully clamped.\n"
            f"Loads: total force F = {p['F']:g} N in the -y direction, uniformly distributed over the free end "
            f"of the horizontal leg, x = {L} m (0 <= y <= {b}). All other edges are free.\n"
            "Report (SI units):\n"
            f"  vm_max         maximum von Mises stress [Pa]\n"
            f"  tip_uy         u_y at the middle of the loaded end, ({L}, {b / 2:g}) [m]\n"
            f"  strain_energy  total strain energy (given thickness) [J]"
        )

    def build_geometry(self) -> None:
        p = self.params
        b, H, L = p["b"], p["H"], p["L"]
        lines, s = polygon([(0, 0), (L, 0), (L, b), (b, b), (b, H), (0, H)])
        add_group(1, [lines[4]], "clamped")   # (b, H) -> (0, H)
        add_group(1, [lines[1]], "loaded")    # (L, 0) -> (L, b)
        add_group(2, [s], "domain")

    def qois(self) -> list[QoI]:
        p = self.params
        return [QoI("vm_max", "max", field="von_mises"),
                QoI("tip_uy", "point", field="uy", point=[p["L"], p["b"] / 2]),
                QoI("strain_energy", "strain_energy")]

    def case(self, mesh_path, order: int = 2):
        p = self.params
        bcs = [BC("fixed", group="clamped"), BC("force", group="loaded", value=[0.0, -p["F"]])]
        return self.make_case(mesh_path, order, bcs, thickness=p["thickness"])

    def exact(self, field, points):
        return None

    def expectations(self) -> dict[str, str]:
        return {"vm_max": "diverge"}

    def expected_order(self, qoi: QoI, order: int) -> float | None:
        if qoi.name in ("tip_uy", "strain_energy"):
            return 2 * WILLIAMS_270
        return None

    def references(self) -> list[Reference]:
        return []


def _williams_root(alpha: float = 0.75 * math.pi) -> float:
    """Mode-I eigenvalue for a traction-free wedge of half-angle alpha: sin(2 l a) + l sin(2 a) = 0."""
    from scipy.optimize import brentq

    return brentq(lambda l: math.sin(2 * l * alpha) + l * math.sin(2 * alpha), 0.51, 0.7)


register(LShapePoisson())
register(LBracketReentrant())

__all__ = ["LShapePoisson", "LBracketReentrant", "LSHAPE_INTEGRAL", "WILLIAMS_270", "num"]
