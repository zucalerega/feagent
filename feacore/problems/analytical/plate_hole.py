"""Finite-width plate with a central circular hole in tension (quarter model), d/W = 0.1, 0.3, 0.5.

Reference: net-section stress concentration factor of Pilkey & Pilkey, Peterson's Stress
Concentration Factors, 3rd ed., Wiley (2008), Chart 4.1 (Howland 1930 solution for an infinitely
long strip with a central circular hole), curve fit

    K_tn = 2 + 0.284 (1 - d/W) - 0.600 (1 - d/W)^2 + 1.32 (1 - d/W)^3,   0 <= d/W <= 1
    sigma_max = K_tn * sigma_net,  sigma_net = sigma0 W / (W - d)   (sigma0 = gross remote stress)

R.C.J. Howland, "On the stresses in the neighbourhood of a circular hole in a strip under
tension", Phil. Trans. R. Soc. Lond. A 229 (1930) 49-86. The fit reproduces K_tn -> 3 (Kirsch) as
d/W -> 0 and 2 as d/W -> 1. It is a handbook fit, not an exact solution: tolerance 2 %.

The strip modelled has half-length 2 W from the hole centre (total length 4 W). The hole
disturbance decays like exp(-4.2 x / W) along a strip (smallest Papkovich-Fadle eigenvalue of a
strip of half-width W/2, Re(lambda) ~ 2.1), so end effects at the hole are ~1e-3 or smaller.
"""

from __future__ import annotations

from feacore.case import BC, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import STEEL, ExactProblem, group_entities, plate_with_quarter_hole, refine_near
from feacore.problems.base import Reference
from feacore.problems.registry import register

_SRC = ("W.D. Pilkey & D.F. Pilkey, Peterson's Stress Concentration Factors, 3rd ed., Wiley (2008), "
        "Chart 4.1 (finite-width tension strip with a central circular hole; Howland 1930, Phil. Trans. R. "
        "Soc. A 229:49-86): K_tn = 2 + 0.284(1-d/W) - 0.600(1-d/W)^2 + 1.32(1-d/W)^3, "
        "sigma_max = K_tn sigma0 W/(W-d)")


def peterson_ktn(d_over_w: float) -> float:
    s = 1.0 - d_over_w
    return 2.0 + 0.284 * s - 0.600 * s**2 + 1.32 * s**3


class PlateHoleFiniteWidth(ExactProblem):
    """Not an exact-solution problem (handbook reference); ExactProblem is used for the case helpers."""

    title = "Finite-width plate with a central circular hole in tension (plane stress, quarter model)"
    family = "plate_hole_finite_width"
    tags = ("2d", "plane_stress", "stress_concentration", "handbook", "curved_boundary", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "stress_concentration"
    # h0 = size at the hole edge on the coarsest mesh; far-field size = far * h
    default_params = {"W": 0.2, "d_over_W": 0.3, "length_over_W": 4.0, "sigma0": 100e6, **STEEL,
                      "thickness": 0.01, "h0": 0.006, "far": 4.0}

    @property
    def r(self) -> float:
        return 0.5 * self.params["d_over_W"] * self.params["W"]

    @property
    def half_length(self) -> float:
        return 0.5 * self.params["length_over_W"] * self.params["W"]

    def description(self) -> str:
        p = self.params
        W, d, Lh = p["W"], 2 * self.r, self.half_length
        return (
            "Long flat steel bar (strip) with a central circular through-hole, pulled along its length, plane "
            "stress. By double symmetry only the quarter x >= 0, y >= 0 is modelled (STEP: planar face in "
            "the xy-plane); the hole centre is at the origin.\n"
            f"Geometry: full bar width W = {W} m (y from -W/2 to W/2), hole diameter d = {d:g} m "
            f"(d/W = {p['d_over_W']}), full length {2 * Lh:g} m (x from -{Lh:g} to {Lh:g} m). Quarter model: "
            f"rectangle [0, {Lh:g}] x [0, {W / 2:g}] m minus the quarter disk of radius {self.r:g} m. Edges: "
            f"y = 0 (from ({self.r:g}, 0) to ({Lh:g}, 0)), x = {Lh:g} (loaded end), y = {W / 2:g} (free side), "
            f"x = 0 (from (0, {self.r:g}) to (0, {W / 2:g})), and the quarter-circle hole edge.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}, thickness {p['thickness']} m.\n"
            f"Loads: uniform tensile traction sigma0 = {p['sigma0']:.4g} Pa ({p['sigma0'] / 1e6:g} MPa) along "
            f"+x on the end x = {Lh:g} m (gross-section stress). The side y = W/2 and the hole are free.\n"
            "Supports: symmetry on x = 0 (u_x = 0) and on y = 0 (u_y = 0).\n"
            "Report (SI units):\n"
            f"  s_peak   sigma_xx at the edge of the hole on the transverse axis, (0, {self.r:g}) [Pa]\n"
            "  vm_max   maximum von Mises stress [Pa]"
        )

    def build_geometry(self) -> None:
        p = self.params
        g = plate_with_quarter_hole(self.r, self.half_length, 0.5 * p["W"])
        add_group(1, [g["hole"]], "hole")
        add_group(1, [g["right"]], "end")
        add_group(1, [g["top"]], "side")
        add_group(1, [g["x0"]], "sym_x0")
        add_group(1, [g["y0"]], "sym_y0")
        add_group(2, [g["surface"]], "domain")

    def apply_sizing(self, h: float) -> None:
        p = self.params
        refine_near(group_entities("hole"), h, fine=1.0, coarse=p["far"], dist_min=0.0,
                    dist_max=min(4 * self.r, 0.5 * p["W"]))

    def qois(self) -> list[QoI]:
        return [QoI("s_peak", "point", field="sxx", point=[0.0, self.r]),
                QoI("vm_max", "max", field="von_mises")]

    def case(self, mesh_path, order: int = 2):
        p = self.params
        bcs = [BC("symmetry", group="sym_x0"), BC("symmetry", group="sym_y0"),
               BC("traction", group="end", value=[p["sigma0"], 0.0])]
        return self.make_case(mesh_path, order, bcs, thickness=p["thickness"])

    def exact(self, field, points):  # no exact field: handbook reference only
        return None

    def peak_stress(self) -> float:
        dw = self.params["d_over_W"]
        return peterson_ktn(dw) * self.params["sigma0"] / (1.0 - dw)

    def references(self) -> list[Reference]:
        dw = self.params["d_over_W"]
        note = (f"K_tn({dw}) = {peterson_ktn(dw):.4f}. Curve fit to Howland's series solution for an "
                "infinite strip; Peterson gives no error bound for the fit, 2 % covers fit + digitisation. "
                "The model's ends are 2 W from the hole (end effects ~1e-3).")
        return [
            Reference("s_peak", self.peak_stress(), "handbook", _SRC, 0.02, note),
            Reference("vm_max", self.peak_stress(), "handbook",
                      _SRC + "; the maximum von Mises stress is the uniaxial hoop stress at (0, d/2)", 0.02, note),
        ]


for _dw, _h0, _far in ((0.1, 0.002, 10.0), (0.3, 0.006, 4.0), (0.5, 0.01, 3.0)):
    register(PlateHoleFiniteWidth(id=f"plate_hole_finite_width_d{int(round(_dw * 10)):02d}", d_over_W=_dw,
                                  h0=_h0, far=_far))

__all__ = ["PlateHoleFiniteWidth", "peterson_ktn"]
