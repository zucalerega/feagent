"""Laminar flow over a backward-facing step (2D, steady): separation at a sharp corner and a
recirculation bubble behind it. The references are the fixed solver's own mesh-converged values.

Geometry (step height s, expansion ratio 2): an inlet channel -L_in <= x <= 0, s <= y <= 2 s, then
the channel widens over the step face x = 0, 0 <= y <= s into 0 <= x <= L_out, 0 <= y <= 2 s. The
sharp corner (0, s) is re-entrant: the flow separates there, the shear layer leaving it reattaches
on the floor y = 0 a few step heights downstream, and the bubble under it recirculates.

Flow: fully developed parabolic inflow of mean velocity U in the inlet channel, static pressure 0
at the outlet, no slip elsewhere. Re = rho U D_h / mu with the inlet channel's hydraulic diameter
D_h = 2 s (Armaly et al. 1983); the default Re = 200 is steady and two-dimensional.

QoIs (SI, per metre of depth):
  u_bubble      u_x at (2 s, 0.25 s): inside the bubble, near the floor (negative: reversed flow)
  v_shear       u_y at (2 s, s): level with the step corner, where the shear layer bends down
  floor_force   x-force of the fluid on the floor y = 0 (wall shear, negative under the bubble)
(A pressure difference across the step converged non-monotonically in the reference study, by 0.5 %,
so it is not offered.)

References: the fixed solver (OpenFOAM, linearUpwind) on the structured reference mesh (quads,
graded towards the step), refined 2x and 4x, Richardson-extrapolated (docs/benchmarks_cfd.md,
"Backward-facing step"). There is no published value for this exact configuration. Reference mesh: five transfinite blocks (the inlet channel; the downstream
channel cut at y = s and at x = 10 s), graded towards the step in x; s / h0 = 12 cells per step
height in y.
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import (FlowProblem, TransfiniteBlocks, boundary_curves_where, count, group,
                                          near, num, refinement_factor, set_edge, transfinite_all)
from feacore.problems.registry import register

# Mesh-converged values of the fixed solver for the default parameters: Richardson extrapolation of the
# structured study at h0, h0 / 2, h0 / 4 (3120 / 12480 / 49920 cells; docs/benchmarks_cfd.md,
# "Backward-facing step"). Observed order and GCI (fine): u_bubble 1.21, 1.0 %; v_shear 1.72, 0.6 %;
# floor_force 1.51, 0.2 %. u_bubble converges below second order (the shear layer from the singular
# corner), hence its 3 % tolerance.
CONVERGED: dict[str, float | None] = {"u_bubble": -1.62691e-3, "v_shear": -8.92529e-4, "floor_force": 3.10536e-4}
TOL = {"u_bubble": 0.03, "v_shear": 0.02, "floor_force": 0.02}

_SRC = ("Mesh-converged value of the fixed solver (OpenFOAM v1912, SIMPLEC, linearUpwind) on the structured "
        "reference mesh at h0, h0 / 2 and h0 / 4, Richardson-extrapolated (this repository, study of "
        "2026-11-01: docs/benchmarks_cfd.md, 'Backward-facing step'); the configuration follows B.F. Armaly, "
        "F. Durst, J.C.F. Pereira & B. Schoenung, 'Experimental and theoretical investigation of backward-facing "
        "step flow', J. Fluid Mech. 127 (1983) 473-496 (Reynolds number on the inlet hydraulic diameter)")


class BackwardFacingStep2D(FlowProblem):
    id = "backward_facing_step_2d"
    title = "Laminar flow over a backward-facing step, expansion ratio 2, Re = 200 (2D)"
    family = "backward_facing_step"
    tags = ("cfd", "2d", "flow", "laminar", "internal_flow", "separation", "recirculation", "expression_bc",
            "converged_numerical")
    expected_behavior = "smooth"     # the step corner is singular, the QoIs are away from it
    default_params = {"s": 0.01, "L_in": 0.04, "L_out": 0.3, "U": 0.01, "rho": 1000.0, "mu": 0.001,
                      "h0": 0.01 / 12}
    x_split = 10.0                   # the near / far block boundary, in step heights
    n_in0, ratio_in = 30, 4.0        # inlet channel cells (finer at the step)
    n_near0, ratio_near = 75, 4.0    # 0 .. 10 s
    n_far0, ratio_far = 40, 4.0      # 10 s .. L_out

    @property
    def reynolds(self) -> float:
        p = self.params
        return p["rho"] * p["U"] * 2.0 * p["s"] / p["mu"]

    def inlet_profile(self) -> str:
        """u_x = 6 U (y - s)(2 s - y) / s^2 in the inlet channel s <= y <= 2 s."""
        p = self.params
        s = p["s"]
        return f"{num(6.0 * p['U'] / s ** 2)}*(y - {num(s)})*({num(2 * s)} - y)"

    def points(self) -> dict[str, list[float]]:
        p = self.params
        s = p["s"]
        return {"bubble": [2.0 * s, 0.25 * s], "shear": [2.0 * s, s]}

    # ---- statement
    def description(self) -> str:
        p = self.params
        s, Lin, Lout = p["s"], p["L_in"], p["L_out"]
        pts = self.points()
        return (
            "Steady laminar flow of an incompressible Newtonian fluid over a backward-facing step (2D).\n"
            f"Geometry (STEP, planar face in the xy-plane): an inlet channel {-Lin:g} <= x <= 0, {s:g} <= y <= "
            f"{2 * s:g} m, then the channel widens over a step of height s = {s:g} m at x = 0 into "
            f"0 <= x <= {Lout:g}, 0 <= y <= {2 * s:g} m. Model it as 2D; report forces per metre of depth "
            "(out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s "
            f"(Re = rho U (2 s) / mu = {self.reynolds:g}).\n"
            "Boundary conditions:\n"
            f"  inlet x = {-Lin:g}: fully developed parabolic profile with mean velocity U = {p['U']} m/s,\n"
            f"    u_x = 6 U (y - s)(2 s - y) / s^2 = {self.inlet_profile()}  [m/s, y in m],  u_y = 0\n"
            f"  outlet x = {Lout:g}: static pressure p = 0 Pa\n"
            "  every other boundary (the channel walls, the step face and the floor y = 0): no slip\n"
            "Report (SI units):\n"
            f"  u_bubble     u_x at ({pts['bubble'][0]:g}, {pts['bubble'][1]:g}) [m/s] (behind the step, near the floor)\n"
            f"  v_shear      u_y at ({pts['shear'][0]:g}, {pts['shear'][1]:g}) [m/s] (level with the step corner)\n"
            "  floor_force  x-component of the force of the fluid (pressure + viscous) on the floor y = 0 "
            "[N per m depth]\n"
            "The mesh groups must be named inlet, outlet, floor (y = 0) and walls (every other wall)."
        )

    # ---- geometry
    def _corners(self):
        p = self.params
        s, Lin, Lout = p["s"], p["L_in"], p["L_out"]
        return [(-Lin, s), (0.0, s), (0.0, 0.0), (Lout, 0.0), (Lout, 2 * s), (-Lin, 2 * s)]

    def build_cad(self) -> None:
        """Agent-facing CAD: the one L-shaped face, no meshing blocks."""
        import gmsh

        occ = gmsh.model.occ
        pts = [occ.addPoint(x, y, 0.0) for x, y in self._corners()]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.synchronize()

    def build_geometry(self) -> None:
        """Reference geometry: the same face as five four-sided blocks."""
        import gmsh

        p = self.params
        s, Lin, Lout = p["s"], p["L_in"], p["L_out"]
        X1 = self.x_split * s
        b = TransfiniteBlocks()

        def block(x0, x1, y0, y1):
            p00, p10, p11, p01 = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
            b.block([b.line(p00, p10), b.line(p10, p11), b.line(p11, p01), b.line(p01, p00)])

        block(-Lin, 0.0, s, 2 * s)
        for x0, x1 in ((0.0, X1), (X1, Lout)):
            block(x0, x1, 0.0, s)
            block(x0, x1, s, 2 * s)
        b.finish()
        self._blocks = b
        tol = 1e-6                     # OCC bounding boxes are padded by ~1e-7
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x1, -Lin, tol)), "inlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(x0, Lout, tol)), "outlet")
        group(1, boundary_curves_where(lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0, tol)), "floor")
        wall = boundary_curves_where(lambda x0, y0, z0, x1, y1, z1:
                                     near(y0, 2 * s, tol)                                  # top wall
                                     or (near(y0, s, tol) and near(y1, s, tol) and x1 <= tol)   # inlet channel floor
                                     or (near(x0, 0.0, tol) and near(x1, 0.0, tol) and y1 <= s + tol))  # step face
        group(1, wall, "walls")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "fluid")

    def apply_sizing(self, h: float) -> None:
        """Edge counts from k = h0 / h; x edges graded towards the step, y edges uniform."""
        import gmsh

        p = self.params
        k = refinement_factor(p["h0"], h)
        s, Lin = p["s"], p["L_in"]
        X1 = self.x_split * s
        ny = count(round(s / p["h0"]), k)
        b = self._blocks
        done = set()
        for (pa, pb), tag in b.lines.items():
            (xa, ya, _), (xb, yb, _) = b.coords[pa], b.coords[pb]
            if abs(xa - xb) < 1e-12:                                   # vertical: s tall
                done.add(set_edge(tag, ny))
                continue
            lo, hi = min(xa, xb), max(xa, xb)
            forward = tag if xb > xa else -tag                          # oriented +x
            if hi <= 1e-12:                                             # inlet channel: finer at x = 0
                done.add(set_edge(-forward, count(self.n_in0, k), self.ratio_in))
            elif hi <= X1 + 1e-12:                                      # near the step: finer at x = 0
                done.add(set_edge(forward, count(self.n_near0, k), self.ratio_near))
            else:
                done.add(set_edge(forward, count(self.n_far0, k), self.ratio_far))
        missing = [t for _, t in gmsh.model.getEntities(1) if t not in done]
        if missing:
            raise RuntimeError(f"{self.id}: edges without a transfinite count: {missing}")
        transfinite_all(2)
        del Lin

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        pts = self.points()
        return [
            QoI("u_bubble", "point", field="ux", point=pts["bubble"]),
            QoI("v_shear", "point", field="uy", point=pts["shear"]),
            QoI("floor_force", "force", group="floor", component=0),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [
            BC("velocity_inlet", group="inlet", value=[self.inlet_profile(), 0.0]),
            BC("pressure_outlet", group="outlet", value=0.0),
            BC("wall", group="floor"),
            BC("wall", group="walls"),
        ]
        return self.make_case(mesh_path, order, bcs)

    def references(self) -> list[Reference]:
        return [Reference(q, v, "converged_numerical", _SRC, tolerance=TOL[q],
                          notes="mesh-converged value of the fixed solver itself (no published value for this "
                                "configuration); docs/benchmarks_cfd.md, 'Backward-facing step'")
                for q, v in CONVERGED.items() if v is not None]


register(BackwardFacingStep2D())

__all__ = ["BackwardFacingStep2D", "CONVERGED"]
