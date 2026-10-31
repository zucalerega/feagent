"""Flat bar with two opposite U-notches in tension (plane stress).

Reference: the converged FE peak notch-root stress (``_numerical_refs``). The Peterson / Pilkey
curve fit for opposite U-shaped notches in a finite-width flat bar in tension
(``_common.kt_opposite_u_notches_tension``) times the net-section nominal stress P / ((H - 2t) b)
is 8-11 % lower; it is kept in the reference notes and in ``handbook_reference()``.

Variants: notch depth t in {5, 10} mm x root radius r in {2, 4} mm on a 50 mm wide, 5 mm thick
steel bar. The reference mesh is the full bar (2D is cheap); agents may use the two symmetry
lines.
"""

from __future__ import annotations

from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.base import Reference
from feacore.problems.parts._common import (
    MM, STEEL, PartProblem, add_group, finish_sizing, fmt_mm, kt_opposite_u_notches_tension,
    material, threshold,
)
from feacore.problems.parts._numerical_refs import SOURCE as NUM_SOURCE
from feacore.problems.parts._numerical_refs import numerical_reference
from feacore.problems.registry import register

KT_SOURCE = (
    "Pilkey & Pilkey, Peterson's Stress Concentration Factors, 3rd ed. (Wiley 2008), Chapter 2, "
    "chart for a flat tension bar with opposite U-shaped notches and its curve fit "
    "Kt = C1 + C2(2t/H) + C3(2t/H)^2 + C4(2t/H)^3, C_i(sqrt(t/r), t/r) for 0.1<=t/r<=2 and "
    "2<=t/r<=50; same fit in Young & Budynas, Roark's Formulas for Stress and Strain, 7th ed., "
    "Table 17.1 (two U-notches in a member of rectangular section, elementary tension). "
    "Peak stress = Kt * P / ((H - 2t) b)."
)


class NotchedBar(PartProblem):
    family = "notched_bar"
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "stress_concentration"
    default_params = {
        "length": 200 * MM,      # bar length (x from -L/2 to L/2)
        "width": 50 * MM,        # H (y from -H/2 to H/2)
        "thickness": 5 * MM,     # b
        "notch_depth": 5 * MM,   # t, measured from the edge to the notch root
        "notch_radius": 2 * MM,  # r, semicircular root; flanks parallel, notch width 2r
        "force": 20e3,           # axial tensile force [N]
        "E": STEEL["E"],
        "nu": STEEL["nu"],
        "h0": 4 * MM,
    }

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        p = self.params
        if p["notch_depth"] < p["notch_radius"]:
            raise ValueError("U-notch needs depth >= radius")
        self.tags = ("part", "2d", "plane_stress", "notch", "tension", "stress_concentration")
        self.title = (f"Flat bar with opposite U-notches in tension, t = {fmt_mm(p['notch_depth'])}, "
                      f"r = {fmt_mm(p['notch_radius'])}")

    def nominal_stress(self) -> float:
        p = self.params
        return p["force"] / ((p["width"] - 2 * p["notch_depth"]) * p["thickness"])

    def kt(self) -> float:
        p = self.params
        return kt_opposite_u_notches_tension(p["width"], p["notch_depth"], p["notch_radius"])

    def description(self) -> str:
        p = self.params
        L, H, b, t, r = p["length"], p["width"], p["thickness"], p["notch_depth"], p["notch_radius"]
        return (
            "Flat steel bar with two opposite U-notches, in tension. Plane stress.\n"
            "Geometry file (STEP): the planar outline of the bar in the xy-plane (the thickness is "
            "not modelled in the STEP).\n"
            f"Purpose: a {fmt_mm(b)} thick flat tension strap, {fmt_mm(L)} long (x from {fmt_mm(-L / 2)} "
            f"to {fmt_mm(L / 2)}) and H = {fmt_mm(H)} wide (y from {fmt_mm(-H / 2)} to {fmt_mm(H / 2)}), "
            f"with two opposite U-shaped notches at mid-length (x = 0), one in each long edge. Each "
            f"notch is {fmt_mm(t)} deep measured from the edge, has parallel flanks {fmt_mm(2 * r)} "
            f"apart and a semicircular root of radius r = {fmt_mm(r)}; the net width between the "
            f"notch roots is {fmt_mm(H - 2 * t)}.\n"
            f"Material: steel, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, Poisson's "
            f"ratio {p['nu']:g}. Thickness {fmt_mm(b)} (plane stress).\n"
            f"Supports: the end x = {fmt_mm(-L / 2)} is held in the x direction only (free to contract "
            "in y; a frictionless support); remove the remaining rigid-body motion (y translation) "
            "with a constraint that does not stress the bar.\n"
            f"Load: a total tensile force P = {p['force'] / 1e3:g} kN in +x, applied as a uniform "
            f"traction on the end x = {fmt_mm(L / 2)}.\n"
            "Analysis: linear static, small displacements.\n"
            "Report:\n"
            "  peak_stress_notch: the maximum value of the maximum principal stress s1 on the "
            "semicircular notch roots [Pa];\n"
            f"  elongation: the average x-displacement of the loaded end x = {fmt_mm(L / 2)} [m]."
        )

    # ---- geometry: outline traced counter-clockwise from the lower-left corner
    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        p = self.params
        L, H, t, r = p["length"], p["width"], p["notch_depth"], p["notch_radius"]
        x0, x1, y0, y1 = -L / 2, L / 2, -H / 2, H / 2
        yb = y0 + t - r      # bottom notch: arc centre height
        yt = y1 - t + r      # top notch: arc centre height
        P = lambda x, y: occ.addPoint(x, y, 0)  # noqa: E731
        pts = {
            "ll": P(x0, y0), "b1": P(-r, y0), "b2": P(-r, yb), "bc": P(0, yb), "b3": P(r, yb),
            "b4": P(r, y0), "lr": P(x1, y0), "ur": P(x1, y1), "t1": P(r, y1), "t2": P(r, yt),
            "tc": P(0, yt), "t3": P(-r, yt), "t4": P(-r, y1), "ul": P(x0, y1),
        }
        flanks = t > r + 1e-12
        if not flanks:  # semicircular notch: flank points coincide with the edge points
            raise NotImplementedError("t == r not used by the registered variants")
        c = self._c = {}
        c["edge_bottom_l"] = occ.addLine(pts["ll"], pts["b1"])
        c["flank_b1"] = occ.addLine(pts["b1"], pts["b2"])
        c["root_bottom"] = occ.addCircleArc(pts["b2"], pts["bc"], pts["b3"])
        c["flank_b2"] = occ.addLine(pts["b3"], pts["b4"])
        c["edge_bottom_r"] = occ.addLine(pts["b4"], pts["lr"])
        c["end_right"] = occ.addLine(pts["lr"], pts["ur"])
        c["edge_top_r"] = occ.addLine(pts["ur"], pts["t1"])
        c["flank_t1"] = occ.addLine(pts["t1"], pts["t2"])
        c["root_top"] = occ.addCircleArc(pts["t2"], pts["tc"], pts["t3"])
        c["flank_t2"] = occ.addLine(pts["t3"], pts["t4"])
        c["edge_top_l"] = occ.addLine(pts["t4"], pts["ul"])
        c["end_left"] = occ.addLine(pts["ul"], pts["ll"])
        loop = occ.addCurveLoop(list(c.values()))
        self._surf = occ.addPlaneSurface([loop])
        occ.remove([(0, pts["bc"]), (0, pts["tc"])])
        occ.synchronize()

    def _groups(self, full: bool) -> None:
        c = self._c
        add_group(1, [c["end_left"]], "end_left")
        add_group(1, [c["end_right"]], "end_right")
        add_group(1, [c["root_bottom"], c["root_top"]], "notch_roots")
        add_group(1, [c["flank_b1"], c["flank_b2"], c["flank_t1"], c["flank_t2"]], "notch_flanks")
        add_group(1, [c["edge_bottom_l"], c["edge_bottom_r"], c["edge_top_l"], c["edge_top_r"]], "long_edges")
        add_group(2, [self._surf], "bar")

    def apply_sizing(self, h: float) -> None:
        p = self.params
        r, h0 = p["notch_radius"], p["h0"]
        kf = min(1.0, r / (4.0 * h0))
        f = threshold(kf * h, h, 0.5 * r, 0.5 * r + 4.0 * h0,
                      curves=[self._c["root_bottom"], self._c["root_top"]])
        finish_sizing([f], h)

    # ---- analysis
    def qois(self) -> list[QoI]:
        return [QoI(name="peak_stress_notch", type="max", field="s1", group="notch_roots"),
                QoI(name="elongation", type="mean", field="ux", group="end_right")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        L, H = p["length"], p["width"]
        traction = p["force"] / (H * p["thickness"])
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_stress", order=order,
                              thickness=p["thickness"]),
            materials={"steel": material(p)},
            bcs=[
                BC(type="symmetry", group="end_left"),
                BC(type="displacement", point=[-L / 2, 0.0], components=[1], value=0.0),
                BC(type="traction", group="end_right", value=[traction, 0.0]),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "full"},
        )

    def handbook_reference(self) -> Reference:
        """Peterson / Pilkey curve-fit value (not the judged reference: see references())."""
        p = self.params
        kt = self.kt()
        return Reference(
            qoi="peak_stress_notch", value=kt * self.nominal_stress(), kind="handbook",
            source=KT_SOURCE, tolerance=0.05,
            notes=(f"2t/H = {2 * p['notch_depth'] / p['width']:.3g}, t/r = "
                   f"{p['notch_depth'] / p['notch_radius']:.3g}: Kt = {kt:.4f} on the net-section "
                   f"sigma_nom = {self.nominal_stress() / 1e6:.2f} MPa."),
        )

    def references(self) -> list[Reference]:
        hb = self.handbook_reference()
        num = numerical_reference(self.id, "peak_stress_notch")
        if num is None:
            return [hb]
        diff = (num["value"] - hb.value) / hb.value
        ccx = (f"; CalculiX on the finest mesh, which models plane stress as a {self.params['thickness'] * 1e3:g} mm "
               f"slab and so includes a through-thickness effect at the notch root: "
               f"{num['ccx_fine'] / 1e6:.2f} MPa" if num.get("ccx_fine") else "")
        return [Reference(
            qoi="peak_stress_notch", value=num["value"], kind="converged_numerical",
            source=NUM_SOURCE + f" Study of {self.id}{ccx}.", tolerance=0.02,
            notes=(f"Uncertainty ~{num['unc'] * 100:.2g} %. Handbook (Peterson / Pilkey fit, {hb.notes}) "
                   f"gives {hb.value / 1e6:.1f} MPa, {diff * 100:.1f} % below this value; the fit is close "
                   "to Neuber's nomogram, which body-force solutions (Nisitani & Noda) show to "
                   "underestimate opposite-notch Kt by ~10 %. Long bar (L/H = 4): no short-bar effect. "
                   f"Kt(FE) = {num['value'] / self.nominal_stress():.3f}."),
        )]


for _t in (5, 10):
    for _r in (2, 4):
        register(NotchedBar(id=f"notched_bar_t{_t}_r{_r}", notch_depth=_t * MM, notch_radius=_r * MM))
