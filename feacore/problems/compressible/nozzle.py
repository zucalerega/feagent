"""Converging-diverging nozzle (2D planar, inviscid): compressible internal flow from a reservoir through a
throat, in its three regimes. Physics "compressible": the OpenFOAM backend's rhoCentralFoam path with a
total_inlet and a pressure_outlet (feacore/solver/_foam_compressible.py, docs/solver_transient.md,
"Compressible internal flow").

Geometry: the upper half of a planar nozzle, -L <= x <= L, 0 <= y <= h(x), with

    h(x) = h_t (1 + k (x / L)^2),     A(x) / A_t = h(x) / h_t,

the throat at x = 0, inlet and exit area ratios 1 + k (1.5 for the default k = 0.5). The wall slope is at
most 2 k h_t / L = 0.1 (5.7 deg) and the throat's radius of curvature is L^2 / (2 k h_t) = 10 h_t, so the flow
is close to quasi-one-dimensional: the departures (the wall-normal velocity, the curvature of the sonic line and
of the shock) are of order (h')^2 and h h'' ~ 1e-2 locally; measured, the choked mass flow at h0 / 4 is within
1.2e-4 of the 1D value, and the unchoked one converges to -0.33 % of it (the flow angle at the exit). y = 0 is
the symmetry plane, the curved wall slips (no viscosity), out-of-plane thickness 1 m: mass flows are per metre
of depth through the half nozzle.

Gas: gamma = 1.4, R = 287 J/(kg K), reservoir p0 = 1e5 Pa, T0 = 300 K. Regimes (back pressure pb at x = L;
critical values for Ae / At = 1.5: p1 / p0 = 0.8805 (choked, subsonic exit), p2 / p0 = 0.6157 (normal shock in
the exit plane), p3 / p0 = 0.1602 (shock-free supersonic exit)):

    subsonic    pb / p0 = 0.92   not choked: subsonic everywhere, the area-Mach relation with the fictitious
                                 sonic area A* = Ae / (A/A*)(Me), Me from pb / p0
    shock       pb / p0 = 0.80   choked; a normal shock in the diverging part at x_s / L = 0.5864 (M1 = 1.494)
    supersonic  pb / p0 = 0.10   choked, supersonic to the exit, Me = 1.8541; the exit flow and the mass flow
                                 do not depend on pb below p2 (here the jet expands outside the domain)

QoIs: mass_flow (kg/s through the outlet, per metre of depth of the half nozzle), the area-averaged static
pressure and Mach number on the exit plane, and (shock regime) the shock position from ``shock_position``.
References: the exact quasi-1D solution (``quasi1d``), verified against NACA Report 1135 in
tests/test_compressible_internal.py.
"""

from __future__ import annotations


import numpy as np

from feacore.case import BC, Analysis, Case, Material, QoI, Region
from feacore.problems.base import Reference
from feacore.problems.cfd._common import FlowProblem, count, group, refinement_factor, set_edge, transfinite_all
from feacore.problems.compressible.quasi1d import NACA1135, Quasi1DNozzle
from feacore.problems.registry import register

REGIMES = {"subsonic": 0.92, "shock": 0.80, "supersonic": 0.10}
# Tolerances (relative) of the references. The scheme's errors at 2 h0 .. h0 / 4 and the 2D effects of the
# planar nozzle are measured in docs/solver_transient.md, "Compressible internal flow" (Minmod, at h0: choked mass
# flow -0.06 %, exit pressure +0.4 % (supersonic), exit Mach number -0.23 / +0.09 %, shock position +0.17 % of L);
# every tolerance is at least twice the error at h0 and covers the 2D effects (exit pressure ~0.3 %).
TOL = {"mass_flow": 0.005, "p_exit": 0.01, "mach_exit": 0.01, "x_shock": 0.02}
# Not choked, the mass flow follows from the small pressure difference p0 - pb (8 % of p0) and takes the
# scheme's total-pressure loss with it: -0.64 % at h0, -0.41 % at h0 / 2, converging to the 2D value -0.33 %.
TOL_SUBSONIC = {**TOL, "mass_flow": 0.01}


class CDNozzle2D(FlowProblem):
    id = "cd_nozzle_2d"
    title = "Converging-diverging nozzle, inviscid compressible flow (2D planar)"
    family = "cd_nozzle"
    physics = "compressible"
    formulation = "2d"
    dim = 2
    units = "SI (m, s, kg, Pa, N, K)"
    material_name = "gas"
    default_params = {"L": 1.0, "h_t": 0.1, "k": 0.5, "p0": 1.0e5, "T0": 300.0, "pb_ratio": REGIMES["shock"],
                      "gamma": 1.4, "gas_constant": 287.0, "h0": 1.0 / 40}
    n_x0, n_y0 = 80, 6          # cells at h0 along the nozzle and across its half height
    n_wall = 201                # spline points of the curved wall
    max_flow_times = 200
    steady_tol = 1e-6
    qoi_tol = 1e-3

    def __init__(self, id: str | None = None, regime: str | None = None, **params):
        if regime is not None:
            params.setdefault("pb_ratio", REGIMES[regime])
        super().__init__(id, **params)
        p = self.params
        if not (0 < p["k"] <= 2 and p["h_t"] > 0 and p["L"] > 0 and 0 < p["pb_ratio"] < 1):
            raise ValueError(f"{type(self).__name__}: needs 0 < k <= 2, h_t, L > 0 and 0 < pb_ratio < 1")
        self.expected_behavior = "shock" if self.exact_solution().regime == "shock" else "smooth"
        reg = self.exact_solution().regime
        self.tags = ("compressible", "2d", "inviscid", "internal_flow", "nozzle", reg, "analytical")

    # ---- geometry
    def half_height(self, x):
        p = self.params
        return p["h_t"] * (1.0 + p["k"] * (np.asarray(x, dtype=float) / p["L"]) ** 2)

    def area(self, x: float) -> float:
        """A(x) per metre of depth of the half nozzle (= h(x))."""
        return float(self.half_height(x))

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        L = p["L"]
        occ = gmsh.model.occ
        xs = np.linspace(-L, L, self.n_wall)
        wall_pts = [occ.addPoint(float(x), float(self.half_height(x)), 0.0) for x in xs]
        a, b = occ.addPoint(-L, 0.0, 0.0), occ.addPoint(L, 0.0, 0.0)
        axis = occ.addLine(a, b)
        outlet = occ.addLine(b, wall_pts[-1])
        wall = occ.addSpline(wall_pts[::-1])
        inlet = occ.addLine(wall_pts[0], a)
        s = occ.addPlaneSurface([occ.addCurveLoop([axis, outlet, wall, inlet])])
        occ.synchronize()
        self._edges = {"x": [axis, wall], "y": [outlet, inlet]}
        group(1, [inlet], "inlet")
        group(1, [outlet], "outlet")
        group(1, [wall], "wall")
        group(1, [axis], "axis")
        group(2, [s], "fluid")

    def apply_sizing(self, h: float) -> None:
        k = refinement_factor(self.params["h0"], h)
        for c in self._edges["x"]:
            set_edge(c, count(self.n_x0, k))
        for c in self._edges["y"]:
            set_edge(c, count(self.n_y0, k))
        transfinite_all(2)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- exact solution
    def exact_solution(self) -> Quasi1DNozzle:
        p = self.params
        return Quasi1DNozzle(self.area, -p["L"], 0.0, p["L"], p["p0"], p["T0"], p["pb_ratio"] * p["p0"],
                             p["gamma"], p["gas_constant"])

    # ---- statement
    def description(self) -> str:
        p = self.params
        L, ht, k = p["L"], p["h_t"], p["k"]
        return (
            "Steady flow of an inviscid perfect gas through a converging-diverging nozzle (2D planar).\n"
            f"Geometry (STEP, planar face in the xy-plane): the upper half of the nozzle, -{L:g} <= x <= {L:g} m, "
            f"0 <= y <= h(x) with h(x) = {ht:g} (1 + {k:g} (x / {L:g})^2) m (throat at x = 0). The plane y = 0 is "
            "the nozzle's symmetry plane. Model it as 2D with an out-of-plane thickness of 1 m.\n"
            f"Gas: perfect gas, gamma = {p['gamma']}, specific gas constant R = {p['gas_constant']} J/(kg K), "
            "inviscid (no viscosity, no heat conduction).\n"
            "Boundary conditions:\n"
            f"  inlet (x = -{L:g} m): reservoir, total pressure p0 = {p['p0']:g} Pa, total temperature T0 = "
            f"{p['T0']:g} K, flow along +x\n"
            f"  outlet (x = {L:g} m): static back pressure pb = {p['pb_ratio'] * p['p0']:g} Pa where the gas "
            "leaves subsonically (nothing imposed where it leaves supersonically)\n"
            "  wall (the curved boundary): inviscid wall (slip)\n"
            "  axis (y = 0): symmetry plane\n"
            "Report (SI units, per metre of depth of the half nozzle):\n"
            "  mass_flow   mass flow rate out through the outlet [kg/s]\n"
            "  p_exit      area-averaged static pressure on the outlet [Pa]\n"
            "  mach_exit   area-averaged Mach number on the outlet [-]\n"
            "and, if a normal shock stands in the nozzle, its position x_shock [m].\n"
            "The mesh groups must be named inlet, outlet, wall, axis and fluid."
        )

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        return [QoI("mass_flow", "mass_flow", group="outlet"),
                QoI("p_exit", "mean", field="p", group="outlet"),
                QoI("mach_exit", "mean", field="mach", group="outlet")]

    def material(self) -> Material:
        p = self.params
        return Material(gamma=p["gamma"], gas_constant=p["gas_constant"], mu=0.0)

    def options(self) -> dict:
        return {"max_flow_times": self.max_flow_times, "steady_tol": self.steady_tol, "qoi_tol": self.qoi_tol}

    def case(self, mesh_path, order: int = 2) -> Case:
        p = self.params
        bcs = [BC("total_inlet", group="inlet", p0=p["p0"], T0=p["T0"]),
               BC("pressure_outlet", group="outlet", value=p["pb_ratio"] * p["p0"]),
               BC("slip", group="wall"),
               BC("symmetry", group="axis")]
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="compressible", formulation="2d", order=order, thickness=self.thickness,
                              options=self.options()),
            materials={self.material_name: self.material()},
            bcs=bcs,
            regions=[Region(group="fluid", material=self.material_name)],
            qois=self.qois(),
            meta={"problem": self.id, "setup": "reference"},
        )

    def exact_qois(self) -> dict[str, float]:
        ex = self.exact_solution()
        L = self.params["L"]
        out = {"mass_flow": ex.mass_flow, "p_exit": ex.p(L), "mach_exit": ex.mach(L)}
        if ex.x_shock is not None:
            out["x_shock"] = ex.x_shock
        return out

    def tolerance(self, q: str) -> float:
        return (TOL_SUBSONIC if self.exact_solution().regime == "subsonic" else TOL)[q]

    def references(self) -> list[Reference]:
        ex = self.exact_solution()
        out = []
        notes = {"mass_flow": "rho* a* A* (choked: A* = A_t; not choked: the fictitious sonic area of the exit state)",
                 "p_exit": "exit static pressure of the quasi-1D solution",
                 "mach_exit": "exit Mach number of the quasi-1D solution",
                 "x_shock": "normal-shock position: the exit pressure behind the shock equals pb"}
        for q, v in self.exact_qois().items():
            if q == "x_shock":
                continue              # not a solver QoI: checked with shock_position (tolerance relative to L)
            out.append(Reference(q, v, "analytical", NACA1135, tolerance=self.tolerance(q),
                                 notes=f"{notes[q]}; regime {ex.regime}, quasi-1D theory (2D effects ~1e-4 in the "
                                       "mass flow, docs/solver_transient.md)"))
        return out


# --------------------------------------------------------------------------- post-processing


def section_average(res, prob: CDNozzle2D, x: np.ndarray, field: str = "p", n: int = 8) -> np.ndarray:
    """Area average of ``field`` over the cross-section at every x (n midpoint samples over 0 < y < h(x))."""
    x = np.asarray(x, dtype=float)
    fr = (np.arange(n) + 0.5) / n
    h = prob.half_height(x)
    pts = np.column_stack([np.repeat(x, n), (h[:, None] * fr[None, :]).ravel()])
    return res.probe(field, pts).reshape(len(x), n).mean(axis=1)


def shock_position(res, prob: CDNozzle2D, n: int = 2001) -> float | None:
    """The shock position from the solution alone: along the diverging part, the section-averaged pressure
    falls to its minimum p_lo, then jumps to p_hi (its maximum downstream of the minimum); the shock is where it
    crosses (p_lo + p_hi) / 2, linearly interpolated. None when the pressure falls monotonically (no shock)."""
    L = prob.params["L"]
    x = np.linspace(0.0, L * (1 - 1e-9), n)
    p = section_average(res, prob, x, "p")
    m = section_average(res, prob, x[::20], "mach")
    if m.max() < 1.02 or m[-1] >= 1.0:       # no supersonic region that ends subsonic: no shock in the nozzle
        return None
    i = int(np.argmin(p))
    if i >= n - 2:
        return None
    j = i + int(np.argmax(p[i:]))
    lo, hi = p[i], p[j]
    if hi - lo < 0.05 * prob.params["p0"]:
        return None
    mid = 0.5 * (lo + hi)
    k = i + int(np.argmax(p[i:j + 1] >= mid))
    return float(x[k - 1] + (mid - p[k - 1]) * (x[k] - x[k - 1]) / (p[k] - p[k - 1]))


for _reg in REGIMES:
    register(CDNozzle2D(id=f"cd_nozzle_{_reg}_2d", regime=_reg))

__all__ = ["CDNozzle2D", "REGIMES", "TOL", "TOL_SUBSONIC", "section_average", "shock_position"]
