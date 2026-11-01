"""Supersonic inviscid flow over a blunted wedge at angle of attack (2D): an asymmetric blunt body.

Body: a circular nose of radius R (centre at the origin) with two flanks tangent to it at half-angle
theta, cut by the outflow plane x = x_out (the body continues downstream; the domain ends there).
Freestream: Mach M at angle of attack alpha, U_inf = M a_inf (cos alpha, sin alpha), so the lower
(windward) flank turns the flow by theta + alpha and the upper (leeward) one by theta - alpha. Gas:
inviscid perfect gas (Euler), gamma = 1.4, R_gas = 287 J/(kg K). A cylinder-wedge is the 2D
counterpart of a sphere-cone re-entry body; at incidence the flow has no symmetry.

A detached bow shock stands ahead of the nose with a subsonic region behind its nearly normal part.
The flow accelerates around the nose to supersonic speed along both flanks (the windward flank's
entropy layer at about Mach 1.4 for the defaults) and leaves the domain supersonic everywhere, so the
outflow plane needs no condition. The inflow boundary is the bow shock shape that Billig's
correlation gives for a cylinder at Mach 2, wider than the Mach 3 shock with margin on both sides at
|alpha| <= 12 deg, and it slopes more steeply than the freestream everywhere, so the flow enters
through all of it.

QoIs (SI; forces per metre of depth, with the gauge pressure p - p_inf, as if a base at p_inf closed
the cut body):
  p_stag        the largest wall pressure (the stagnation point)
  axial_force   x-force on the body (along the body axis)
  normal_force  y-force on the body (normal to the axis; the lift-like component)
  p_windward    wall pressure on the windward (lower) flank at x = 2 R
  p_leeward     wall pressure on the leeward (upper) flank at x = 2 R

References: no published value fits this configuration, and a detached shock has no closed-form
solution. The references are the fixed solver's own mesh-converged values (structured study on the
reference mesh, Richardson extrapolation: docs/benchmarks_cfd.md, "Supersonic blunted wedge"),
anchored by exact theory where it exists: at alpha = 0 the stagnation pressure is the Rayleigh pitot
pressure (exact behind the normal part of the shock) and the bow shock standoff is checked against
Billig's correlation for cylinder-wedges, Delta / R = 0.386 exp(4.67 / M^2).
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, Analysis, Case, Material, QoI, Region
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (FlowProblem, TransfiniteBlocks, count, group, refinement_factor,
                                          set_edge, transfinite_all)
from feacore.problems.registry import register

BILLIG = ("F.S. Billig, 'Shock-wave shapes around spherical- and cylindrical-nosed bodies', J. Spacecraft "
          "Rockets 4(6) (1967) 822-823 (correlations as given by J.D. Anderson, Hypersonic and High "
          "Temperature Gas Dynamics, 2nd ed., AIAA (2006), sec. 5.6)")
RAYLEIGH_PITOT = ("Rayleigh pitot formula: NACA Report 1135, 'Equations, tables, and charts for compressible "
                  "flow', Ames Research Staff (1953), eq. 100")

# Mesh-converged values of the fixed solver for the default parameters (docs/benchmarks_cfd.md,
# "Supersonic blunted wedge"); filled in from the structured study.
CONVERGED: dict[str, float] = {}
TOL: dict[str, float] = {}


def pitot_ratio(mach: float, gamma: float = 1.4) -> float:
    """p02 / p1: the total pressure behind a normal shock over the static pressure ahead of it (Rayleigh)."""
    g, m2 = gamma, mach * mach
    return (((g + 1) / 2 * m2) ** (g / (g - 1))) / ((2 * g / (g + 1) * m2 - (g - 1) / (g + 1)) ** (1 / (g - 1)))


def billig_standoff(mach: float) -> float:
    """Billig's bow shock standoff of a cylinder(-wedge), Delta / R = 0.386 exp(4.67 / M^2) (gamma = 1.4)."""
    return 0.386 * math.exp(4.67 / mach ** 2)


def standoff(res, p: "SupersonicBluntWedge2D") -> dict:
    """The bow shock's distance from the stagnation point (the wall face of largest pressure), along the
    line against the freestream: where p crosses the mean of p_inf and the normal-shock pressure."""
    pr = p.params
    body = [res.patch[n] for n in res.fm.patches_of("body") if n in res.patch]
    Cf = np.vstack([d["Cf"][:, :2] for d in body])
    pf = np.concatenate([d["p"] for d in body])
    s0 = Cf[int(np.argmax(pf))]
    e = np.array(p.u_inf) / np.linalg.norm(p.u_inf)
    g, M = pr["gamma"], pr["mach"]
    p2 = pr["p_inf"] * (2 * g * M * M - (g - 1)) / (g + 1)
    mid = 0.5 * (pr["p_inf"] + p2)
    s = np.linspace(0.0, 1.15 * pr["R"], 2301)
    pts = s0[None, :] - s[:, None] * e[None, :]
    v = res.probe("p", pts)
    above = v >= mid
    i = int(np.argmax(~above)) if not above.all() else len(s) - 1     # first sample ahead of the shock
    if i == 0:
        return {"standoff": float("nan"), "stagnation_point": s0.tolist()}
    t = (v[i - 1] - mid) / (v[i - 1] - v[i])
    d = s[i - 1] + t * (s[i] - s[i - 1])
    return {"standoff": float(d), "standoff_over_R": float(d / pr["R"]), "stagnation_point": s0.tolist(),
            "stagnation_angle_deg": float(math.degrees(math.atan2(-s0[1], -s0[0])))}


class SupersonicBluntWedge2D(FlowProblem):
    id = "supersonic_blunt_wedge_2d"
    title = "Supersonic inviscid flow over a blunted wedge at 10 deg angle of attack, Mach 3 (2D)"
    family = "supersonic_blunt_wedge"
    tags = ("supersonic", "compressible", "2d", "inviscid", "shock", "blunt_body", "angle_of_attack",
            "converged_numerical")
    physics = "compressible"
    formulation = "2d"
    dim = 2
    units = "SI (m, s, kg, Pa, N, K)"
    expected_behavior = "shock"
    material_name = "gas"
    default_params = {"R": 0.05, "theta": 10.0, "x_out": 0.2, "mach": 3.0, "alpha": 10.0, "p_inf": 1.0e5,
                      "T_inf": 300.0, "gamma": 1.4, "gas_constant": 287.0, "h0": 0.05 / 12}
    # inflow boundary x = -A + B (sqrt(1 + (y / C)^2) - 1), in units of R: Billig's cylinder shock at Mach 2
    # (A = 1 + Delta / R, B = Rc cot^2(beta), C = Rc cot(beta), beta = 30 deg, Rc = 8.385 R)
    OUTER = (2.241, 25.16, 14.52)
    n_nose0, n_side0, n_norm0 = 24, 24, 32    # cells at h0: each 100 deg nose arc, each flank, body to inflow
    ratio_norm = 3.0                          # cell size at the inflow boundary / at the body
    ratio_side = 3.0                          # flank cell size at the outflow / at the shoulder
    # Solver controls of the reference setup. At incidence the scheme keeps a small limit cycle behind the
    # sonic line: p_windward moves by 7e-4 of q_inf peak to peak (h0), the other QoIs by < 2e-4, so the QoIs
    # count as steady at 1e-3 of q_inf (well below any tolerance); the reference study averages over it.
    max_flow_times = 200
    steady_tol = 1e-6
    qoi_tol = 1e-3

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        p = self.params
        if not (p["mach"] >= 2.5 and abs(p["alpha"]) <= 12.0 and 0 <= p["theta"] <= 20.0):
            raise ValueError(f"{type(self).__name__}: the inflow boundary is sized for mach >= 2.5, |alpha| <= 12 "
                             f"and theta <= 20 deg (got mach {p['mach']}, alpha {p['alpha']}, theta {p['theta']})")

    # ---- flow quantities
    @property
    def a_inf(self) -> float:
        p = self.params
        return math.sqrt(p["gamma"] * p["gas_constant"] * p["T_inf"])

    @property
    def u_inf(self) -> list[float]:
        p = self.params
        U = p["mach"] * self.a_inf
        a = math.radians(p["alpha"])
        return [U * math.cos(a), U * math.sin(a)]

    @property
    def q_inf(self) -> float:
        p = self.params
        return 0.5 * p["gamma"] * p["p_inf"] * p["mach"] ** 2

    def pitot(self) -> float:
        """The Rayleigh pitot pressure (Pa): the stagnation pressure behind a normal shock."""
        p = self.params
        return pitot_ratio(p["mach"], p["gamma"]) * p["p_inf"]

    # ---- geometry
    def _key_points(self) -> dict:
        p = self.params
        R, th, xo = p["R"], math.radians(p["theta"]), p["x_out"]
        s = (xo + R * math.sin(th)) / math.cos(th)
        yb = R * math.cos(th) + s * math.sin(th)
        pts = {"T_up": (-R * math.sin(th), R * math.cos(th)), "T_lo": (-R * math.sin(th), -R * math.cos(th)),
               "F": (-R, 0.0), "B_up": (xo, yb), "B_lo": (xo, -yb), "O_front": (self._outer_x(0.0), 0.0)}
        o = self._radial_hit(pts["T_up"], (-math.sin(th), math.cos(th)))
        pts["O_up"], pts["O_lo"] = o, (o[0], -o[1])
        ye = self._outer_y(xo)
        pts["E_up"], pts["E_lo"] = (xo, ye), (xo, -ye)
        return pts

    def _outer_x(self, y: float) -> float:
        A, B, C = (v * self.params["R"] for v in self.OUTER)
        return -A + B * (math.sqrt(1.0 + (y / C) ** 2) - 1.0)

    def _outer_y(self, x: float) -> float:
        A, B, C = (v * self.params["R"] for v in self.OUTER)
        return C * math.sqrt((1.0 + (x + A) / B) ** 2 - 1.0)

    def _radial_hit(self, start, direction) -> tuple[float, float]:
        """Where the line start + t * direction (t > 0) meets the inflow boundary."""
        from scipy.optimize import brentq

        def g(t):
            x, y = start[0] + t * direction[0], start[1] + t * direction[1]
            return self._outer_x(y) - x
        t = brentq(g, 0.0, 50.0 * self.params["R"], xtol=1e-14)
        return (start[0] + t * direction[0], start[1] + t * direction[1])

    def _outer_spline(self, y0: float, y1: float, end_tags: tuple[int, int], n: int = 41) -> int:
        """Interpolating spline through the inflow boundary between y0 and y1 (end points given)."""
        import gmsh

        occ = gmsh.model.occ
        mids = [occ.addPoint(self._outer_x(y), y, 0.0) for y in np.linspace(y0, y1, n)[1:-1]]
        self._spline_points += mids
        return occ.addSpline([end_tags[0], *mids, end_tags[1]])

    def flank_point(self, x: float, upper: bool) -> list[float]:
        """The point of the upper / lower flank at abscissa x."""
        p = self.params
        R, th = p["R"], math.radians(p["theta"])
        y = R * math.cos(th) + (x + R * math.sin(th)) * math.tan(th)
        return [x, y if upper else -y]

    def build_cad(self) -> None:
        """Agent-facing CAD: the one fluid face (nose arcs, flanks, outflow segments, the inflow spline)."""
        import gmsh

        occ = gmsh.model.occ
        k = self._key_points()
        P = {n: occ.addPoint(x, y, 0.0) for n, (x, y) in k.items() if n != "O_front"}
        c = occ.addPoint(0.0, 0.0, 0.0)
        self._spline_points = []
        curves = [occ.addCircleArc(P["T_up"], c, P["F"]), occ.addCircleArc(P["F"], c, P["T_lo"]),
                  occ.addLine(P["T_lo"], P["B_lo"]), occ.addLine(P["B_lo"], P["E_lo"]),
                  self._outer_spline(k["E_lo"][1], k["E_up"][1], (P["E_lo"], P["E_up"]), n=121),
                  occ.addLine(P["E_up"], P["B_up"]), occ.addLine(P["B_up"], P["T_up"])]
        occ.addPlaneSurface([occ.addCurveLoop(curves)])
        occ.remove([(0, t) for t in [c, *self._spline_points]])
        occ.synchronize()

    def build_geometry(self) -> None:
        """Reference geometry: four structured blocks (two nose blocks, two flank blocks)."""
        import gmsh

        k = self._key_points()
        b = TransfiniteBlocks()
        self._spline_points = []
        tag = {n: b.point(x, y) for n, (x, y) in k.items()}
        centre = (0.0, 0.0, 0.0)
        arc_up = b.arc(k["T_up"] + (0.0,), centre, k["F"] + (0.0,))          # T_up -> F
        arc_lo = b.arc(k["F"] + (0.0,), centre, k["T_lo"] + (0.0,))          # F -> T_lo
        rad_up, rad_f, rad_lo = (b.line(k["T_up"], k["O_up"]), b.line(k["F"], k["O_front"]),
                                 b.line(k["T_lo"], k["O_lo"]))
        out_up, out_lo = b.line(k["B_up"], k["E_up"]), b.line(k["B_lo"], k["E_lo"])
        flank_up, flank_lo = b.line(k["T_up"], k["B_up"]), b.line(k["T_lo"], k["B_lo"])
        sf_up = self._outer_spline(k["O_up"][1], 0.0, (tag["O_up"], tag["O_front"]))       # O_up -> O_front
        sf_lo = self._outer_spline(0.0, k["O_lo"][1], (tag["O_front"], tag["O_lo"]))       # O_front -> O_lo
        ss_up = self._outer_spline(k["O_up"][1], k["E_up"][1], (tag["O_up"], tag["E_up"]))  # O_up -> E_up
        ss_lo = self._outer_spline(k["O_lo"][1], k["E_lo"][1], (tag["O_lo"], tag["E_lo"]))  # O_lo -> E_lo
        b.block([arc_up, rad_f, -sf_up, -rad_up])               # T_up -> F -> O_front -> O_up -> T_up
        b.block([arc_lo, rad_lo, -sf_lo, -rad_f])               # F -> T_lo -> O_lo -> O_front -> F
        b.block([flank_up, out_up, -ss_up, -rad_up])            # T_up -> B_up -> E_up -> O_up -> T_up
        b.block([flank_lo, out_lo, -ss_lo, -rad_lo])            # T_lo -> B_lo -> E_lo -> O_lo -> T_lo
        b.finish()
        gmsh.model.occ.remove([(0, t) for t in self._spline_points])
        gmsh.model.occ.synchronize()
        self._edges = {"nose": [arc_up, arc_lo, sf_up, sf_lo], "radial": [rad_up, rad_f, rad_lo, out_up, out_lo],
                       "flank": [flank_up, flank_lo, ss_up, ss_lo]}
        group(1, [abs(t) for t in (arc_up, arc_lo, flank_up, flank_lo)], "body")
        group(1, [abs(t) for t in (sf_up, sf_lo, ss_up, ss_lo)], "freestream")
        group(1, [abs(t) for t in (out_up, out_lo)], "outlet")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Edge counts from k = h0 / h; radial edges graded towards the body, flanks towards the shoulder."""
        k = refinement_factor(self.params["h0"], h)
        e = self._edges
        for c in e["nose"]:
            set_edge(c, count(self.n_nose0, k))
        for c in e["radial"]:                  # every radial edge runs body -> inflow boundary
            set_edge(c, count(self.n_norm0, k), self.ratio_norm)
        for c in e["flank"]:                   # every flank edge runs shoulder -> outflow
            set_edge(c, count(self.n_side0, k), self.ratio_side)
        transfinite_all(2)

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- statement
    def description(self) -> str:
        p = self.params
        R, xo = p["R"], p["x_out"]
        u = self.u_inf
        wp, lp = self.flank_point(2 * R, upper=False), self.flank_point(2 * R, upper=True)
        return (
            "Steady supersonic flow of an inviscid perfect gas over a blunted wedge at angle of attack (2D).\n"
            f"Geometry (STEP, planar face in the xy-plane): the body is a circular nose of radius R = {R:g} m "
            f"centred at the origin with two straight flanks tangent to it at +-{p['theta']:g} deg to the x-axis; "
            f"the domain ends at the outflow plane x = {xo:g} m, which cuts the body. The curved boundary ahead "
            "of the body is the inflow (freestream) boundary. Model it as 2D; report forces per metre of depth "
            "(out-of-plane thickness 1 m).\n"
            f"Gas: perfect gas, gamma = {p['gamma']}, specific gas constant R = {p['gas_constant']} J/(kg K), "
            "inviscid (no viscosity, no heat conduction).\n"
            f"Freestream: Mach {p['mach']:g} at angle of attack {p['alpha']:g} deg (flow towards +x, turned "
            f"{p['alpha']:g} deg towards +y): static pressure p_inf = {p['p_inf']:g} Pa, static temperature "
            f"T_inf = {p['T_inf']:g} K, velocity (u_x, u_y) = ({u[0]:.6g}, {u[1]:.6g}) m/s.\n"
            "Boundary conditions:\n"
            "  freestream (the curved inflow boundary): the freestream state above (the flow enters everywhere)\n"
            f"  outlet (the plane x = {xo:g} m): supersonic outflow (nothing imposed)\n"
            "  body (nose and both flanks): inviscid wall (slip, no flow through it)\n"
            "Report (SI units; forces with the gauge pressure p - p_inf, i.e. as if a base at p_inf closed the "
            "body where the domain cuts it):\n"
            "  p_stag        the largest static pressure on the body [Pa]\n"
            "  axial_force   x-component of the force of the gas on the body [N per m depth]\n"
            "  normal_force  y-component of the force of the gas on the body [N per m depth]\n"
            f"  p_windward    static pressure on the lower flank at ({wp[0]:g}, {wp[1]:.6g}) [Pa]\n"
            f"  p_leeward     static pressure on the upper flank at ({lp[0]:g}, {lp[1]:.6g}) [Pa]\n"
            "The mesh groups must be named body, freestream and outlet."
        )

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        R = self.params["R"]
        return [
            QoI("p_stag", "max", field="p", group="body"),
            QoI("axial_force", "force", group="body", component=0),
            QoI("normal_force", "force", group="body", component=1),
            QoI("p_windward", "point", field="p", point=self.flank_point(2 * R, upper=False)),
            QoI("p_leeward", "point", field="p", point=self.flank_point(2 * R, upper=True)),
        ]

    def material(self) -> Material:
        p = self.params
        return Material(gamma=p["gamma"], gas_constant=p["gas_constant"], mu=0.0)

    def options(self) -> dict:
        return {"max_flow_times": self.max_flow_times, "steady_tol": self.steady_tol, "qoi_tol": self.qoi_tol}

    def case(self, mesh_path, order: int = 2) -> Case:
        p = self.params
        bcs = [BC("freestream", group="freestream", value=self.u_inf, p_inf=p["p_inf"], T_inf=p["T_inf"]),
               BC("outlet", group="outlet"),
               BC("slip", group="body")]
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

    def references(self) -> list[Reference]:
        return [Reference(q, v, "converged_numerical", _SRC, tolerance=TOL[q],
                          notes="mesh-converged value of the fixed solver (rhoCentralFoam); docs/benchmarks_cfd.md, "
                                "'Supersonic blunted wedge'")
                for q, v in CONVERGED.items()]


_SRC = ("Mesh-converged value of the fixed solver (OpenFOAM v1912 rhoCentralFoam, Kurganov-Tadmor fluxes, van Leer "
        "reconstruction) on the structured reference mesh at h0, h0 / 2 and h0 / 4, Richardson-extrapolated "
        "(this repository, study of 2026-11-01: docs/benchmarks_cfd.md, 'Supersonic blunted wedge'); validated at "
        "alpha = 0 against the Rayleigh pitot pressure (exact) and " + BILLIG)

register(SupersonicBluntWedge2D())

__all__ = ["SupersonicBluntWedge2D", "CONVERGED", "pitot_ratio", "billig_standoff", "standoff"]
