"""DFG benchmark 2D-2 (Schaefer & Turek 1996): unsteady laminar flow around a cylinder, Re = 100.

The channel, the cylinder, the fluid and the reference mesh are those of the steady case 2D-1
(feacore/problems/cfd/cylinder.py): channel 0 <= x <= 2.2, 0 <= y <= 0.41 m, cylinder of diameter
D = 0.1 m centred at (0.2, 0.2), rho = 1 kg/m^3, nu = 1e-3 m^2/s. Only the inflow differs: the
parabolic profile u = 4 U_m y (H - y) / H^2 has U_m = 1.5 m/s, so the mean inflow velocity is
U_bar = 2 U_m / 3 = 1 m/s and Re = U_bar D / nu = 100. A von Karman vortex street sheds from the
cylinder; drag and lift become periodic.

The run starts from rest with the full inflow at t = 0 (an impulsive start) and goes to t_end = 8 s;
the periodic state is reached after about 4 s (the cylinder sits 1 cm below the channel axis, so
shedding starts without an artificial perturbation). The QoIs are reduced over the last 2 s
(about 6 periods): the maximum of the drag and of the lift, and the dominant frequency of the lift.

Benchmark outputs: c_D = 2 F_D / (rho U_bar^2 D), c_L = 2 F_L / (rho U_bar^2 D), the Strouhal number
St = D f / U_bar (f = the frequency of the lift). The QoIs are the forces per metre of depth,
F = c rho U_bar^2 D / 2 = 0.05 c [N/m], and f [Hz] = St U_bar / D = 10 St.

Reference values: the acceptance intervals of the original paper (Schaefer & Turek 1996, Table 4,
quoted by the task brief): c_D,max in [3.22, 3.24], c_L,max in [0.99, 1.01], St in [0.295, 0.305];
the references are the midpoints. Higher-accuracy values exist (FEATFLOW benchmark page for 2D-2;
John, Int. J. Numer. Meth. Fluids 44 (2004) 777-788) and lie inside these intervals; they are not
used here because they could not be checked from this machine.

Tolerances are set from what the fixed solver gives on the reference mesh sequence, not tuned to the
intervals. Measured (OpenFOAM v1912, BDF2, linearUpwind; docs/solver_transient.md):

    mesh            cells   step     c_D,max   c_L,max   St
    h0              2954    2 ms     3.2392    1.0261    0.2872
    h0              2954    1 ms     3.2415    1.0326    0.2874
    h0 / 1.5        6702    1.25 ms  3.2399    1.0310    0.2951
    h0 / 2         11816    1 ms     3.2379    1.0232    0.2980

c_D,max is inside the interval on every mesh; St converges with order 2 in h towards 0.3016 (inside);
c_L,max stays 2 to 3 % high. The registered case (h0, 2 ms) is within 2 % (drag), 5 % (lift) and 5 %
(frequency: h0 is 4.3 % low) of the midpoints.
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.problems.base import Reference
from feacore.problems.cfd._common import SCHAEFER_TUREK
from feacore.problems.cfd.cylinder import CylinderChannelDFG2D1
from feacore.problems.registry import register

CD_MAX_REF = 3.23        # midpoint of [3.22, 3.24]
CL_MAX_REF = 1.00        # midpoint of [0.99, 1.01]
ST_REF = 0.300           # midpoint of [0.295, 0.305]

_SRC = (f"{SCHAEFER_TUREK}, problem 2D-2 (Re = 100, periodic), Table 4: acceptance intervals "
        "c_D,max in [3.22, 3.24], c_L,max in [0.99, 1.01], St in [0.295, 0.305]; reference = the midpoint")


class CylinderChannelDFG2D2(CylinderChannelDFG2D1):
    id = "cylinder_channel_dfg_2d2"
    title = "DFG benchmark 2D-2: unsteady flow around a cylinder in a channel, Re = 100 (Schaefer & Turek 1996)"
    family = "cylinder_channel_dfg"
    tags = ("cfd", "2d", "flow", "laminar", "transient", "published_benchmark", "external_flow", "bluff_body",
            "vortex_shedding", "expression_bc")
    expected_behavior = "periodic"
    default_params = {**CylinderChannelDFG2D1.default_params, "Um": 1.5, "t_end": 8.0, "t_window": 6.0,
                      "dt": 0.002}

    def options(self) -> dict:
        p = self.params
        return {**super().options(),
                "transient": {"t_end": p["t_end"], "dt": p["dt"], "scheme": "backward", "output_every": 500}}

    @property
    def window(self) -> list[float]:
        return [float(self.params["t_window"]), float(self.params["t_end"])]

    # ---- statement
    def description(self) -> str:
        p = self.params
        w = self.window
        return (
            "Unsteady laminar flow of an incompressible Newtonian fluid around a circular cylinder in a 2D "
            "channel (the DFG / Schaefer-Turek 'flow around a cylinder' benchmark, periodic case with vortex "
            "shedding).\n"
            f"Geometry (STEP, planar face in the xy-plane): channel 0 <= x <= {p['L']}, "
            f"0 <= y <= {p['H']} m with a circular hole (the cylinder) of diameter D = {p['D']} m centred "
            f"at ({p['xc']}, {p['yc']}). "
            "Model it as 2D; report forces per metre of depth (out-of-plane thickness 1 m).\n"
            f"Fluid: density rho = {p['rho']} kg/m^3, dynamic viscosity mu = {p['mu']} Pa s.\n"
            "Boundary conditions (constant in time):\n"
            "  inlet x = 0: parabolic velocity profile u_x(y) = 4 U_m y (H - y) / H^2 with "
            f"U_m = {p['Um']} m/s,\n"
            f"    i.e. u_x = {self.inlet_profile()}  [m/s, y in m],  u_y = 0\n"
            f"  outlet x = {p['L']}: static pressure p = 0 Pa\n"
            f"  channel walls y = 0 and y = {p['H']}, and the cylinder surface: no slip\n"
            f"Time: the fluid is at rest at t = 0 and the inflow is on from t = 0; simulate to t = {p['t_end']} s "
            f"(a time-dependent analysis). The flow becomes periodic; evaluate over {w[0]} <= t <= {w[1]} s.\n"
            "Report (SI units), with F the total force (pressure + viscous) of the fluid on the cylinder "
            "[N per m depth]:\n"
            "  drag_max        the maximum over that window of the x-component of F\n"
            "  lift_max        the maximum over that window of the y-component of F\n"
            "  lift_frequency  the frequency of the periodic lift over that window [Hz]\n"
            "The force QoIs refer to the boundary group cylinder: use this name for the cylinder surface "
            "group of the mesh."
        )

    # ---- QoIs, case, references
    def qois(self) -> list[QoI]:
        w = self.window
        return [
            QoI("drag_max", "force", group="cylinder", component=0, time="max", window=w),
            QoI("lift_max", "force", group="cylinder", component=1, time="max", window=w),
            QoI("lift_frequency", "force", group="cylinder", component=1, time="frequency", window=w),
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
        f_scale = self.U_mean / self.params["D"]
        conv = f"F = c rho U_bar^2 D / 2 per m depth, rho U_bar^2 D / 2 = {s:.6g} N/m"
        return [
            Reference("drag_max", CD_MAX_REF * s, "published_numerical", f"{_SRC}; c_D,max = {CD_MAX_REF}; {conv}",
                      tolerance=0.02,
                      notes="interval half-width 0.3 %. The fixed solver: c_D,max = 3.2392, 3.2399, 3.2379 on "
                            "h0, h0/1.5, h0/2 (docs/solver_transient.md); 2 % leaves room for other setups"),
            Reference("lift_max", CL_MAX_REF * s, "published_numerical", f"{_SRC}; c_L,max = {CL_MAX_REF}; {conv}",
                      tolerance=0.05,
                      notes="interval half-width 1 %. The fixed solver: c_L,max = 1.026, 1.031, 1.023 on h0, "
                            "h0/1.5, h0/2 (2 to 3 % high, no clear trend; docs/solver_transient.md)"),
            Reference("lift_frequency", ST_REF * f_scale, "published_numerical",
                      f"{_SRC}; St = {ST_REF}; f = St U_bar / D with U_bar = {self.U_mean:g} m/s, D = "
                      f"{self.params['D']:g} m", tolerance=0.05,
                      notes="interval half-width 1.7 %. The fixed solver: St = 0.2872, 0.2951, 0.2980 on h0, "
                            "h0/1.5, h0/2, second order in h towards 0.3016; h0 is 4.3 % low, hence 5 %"),
        ]


register(CylinderChannelDFG2D2())

__all__ = ["CylinderChannelDFG2D2", "CD_MAX_REF", "CL_MAX_REF", "ST_REF"]
