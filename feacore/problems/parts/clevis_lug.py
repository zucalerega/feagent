"""Pin-loaded lug (clevis eye) in bearing: a straight shank with a semicircular end concentric
with the pin hole. The pin load is modelled as a cosine-distributed radial PRESSURE on the half
of the bore facing away from the base, p(theta) = p0 cos(theta), theta measured from the load
direction at the hole centre, p0 = 2 P / (pi a b) so that the resultant is P; the pressure
vanishes at theta = +-90 degrees, so the load has no traction jump at the net section.

Variants: plane stress with hole diameter 20 mm and 28 mm (W = 40 mm); one 3D variant
(hole 20 mm) whose reference mesh is the quarter x >= 0, z <= b/2 with symmetry planes. No
handbook reference (handbook lug data assume pin contact with clearance, not this load
model); GCI consensus is the check.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.parts._common import (
    MM, STEEL, PartProblem, add_group, com, entities_on_plane, finish_sizing, fmt_mm, material,
    threshold,
)
from feacore.problems.registry import register


class _LugBase(PartProblem):
    family = "clevis_lug"
    physics = "elasticity"
    expected_behavior = "stress_concentration"
    default_params = {
        "width": 40 * MM,          # shank width W = 2 x outer end radius
        "hole_diameter": 20 * MM,  # D = 2a
        "hole_y": 60 * MM,         # hole centre (and end-arc centre) height above the base
        "thickness": 10 * MM,      # b
        "force": 20e3,             # pin load P [N], +y (away from the base)
        "E": STEEL["E"],
        "nu": STEEL["nu"],
        "h0": 4 * MM,
    }

    @property
    def dims(self):
        p = self.params
        return p["width"], p["hole_diameter"] / 2.0, p["hole_y"], p["thickness"]

    def p0(self) -> float:
        W, a, yc, b = self.dims
        return 2.0 * self.params["force"] / (math.pi * a * b)

    def pressure_expr(self) -> str:
        W, a, yc, b = self.dims
        return f"maximum(0.0, {self.p0():.10g} * (y - {yc:.10g}) / {a:.10g})"

    def _text(self, plane: bool) -> str:
        p = self.params
        W, a, yc, b = self.dims
        geo = ("Geometry file (STEP): the planar outline of the lug in the xy-plane (thickness not "
               "modelled in the STEP)." if plane else
               f"Geometry file (STEP): the full 3D lug, thickness b = {fmt_mm(b)} along z (z from 0 to "
               f"{fmt_mm(b)}).")
        return (
            f"{geo}\n"
            f"Purpose: a steel lug (the eye of a clevis joint) of thickness b = {fmt_mm(b)}. A straight "
            f"shank of width W = {fmt_mm(W)} (x from {fmt_mm(-W / 2)} to {fmt_mm(W / 2)}) rises from its "
            f"base at y = 0 and ends in a semicircular head of outer radius {fmt_mm(W / 2)} centred at "
            f"(x, y) = (0, {fmt_mm(yc)}). A pin hole of diameter D = {fmt_mm(2 * a)} is concentric "
            f"with the head. The bore is split into its upper half (y >= {fmt_mm(yc)}, the side away "
            "from the base) and its lower half.\n"
            f"Material: steel, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, Poisson's ratio "
            f"{p['nu']:g}.\n"
            "Supports: the base (y = 0) is fixed, all displacement components zero (the lug is "
            "machined from a much stiffer block).\n"
            f"Load: the pin pulls the lug in +y with a total force P = {p['force'] / 1e3:g} kN. Model "
            "the pin bearing as a radial pressure on the upper half of the bore only, cosine "
            "distributed: p = p0 cos(theta), where theta is the angle from the +y direction measured "
            f"at the hole centre (so cos(theta) = (y - {fmt_mm(yc)}) / a with a = {fmt_mm(a)} the hole "
            "radius), and p0 = 2P / (pi a b) so that the resultant of the pressure is exactly P. The "
            "pressure pushes the material away from the hole centre; the lower half of the bore is "
            "free.\n"
            "Analysis: linear static, small displacements, no contact (the pin is not modelled)."
        )


class ClevisLug2D(_LugBase):
    dim = 2
    formulation = "plane_stress"

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        W, a, _, _ = self.dims
        self.tags = ("part", "2d", "plane_stress", "lug", "pin_loaded", "bearing",
                     "expression_load", "stress_concentration")
        self.title = f"Pin-loaded lug, D/W = {2 * a / W:g} (plane stress)"

    def description(self) -> str:
        return (
            "Pin-loaded lug (clevis eye) in bearing. Plane stress.\n" + self._text(True) + "\n"
            "Report:\n"
            "  peak_hoop_stress: the maximum value of the maximum principal stress s1 on the bore "
            "(both halves); at the free bore surface this is the hoop (tangential) stress [Pa];\n"
            "  lug_elongation: the average y-displacement of the loaded upper half of the bore [m]."
        )

    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        W, a, yc, b = self.dims
        R = W / 2.0
        P = lambda x, y: occ.addPoint(x, y, 0)  # noqa: E731
        p_bl, p_br, p_r, p_top, p_l = P(-R, 0), P(R, 0), P(R, yc), P(0, yc + R), P(-R, yc)
        c0 = P(0, yc)
        c = self._c = {}
        c["base"] = occ.addLine(p_bl, p_br)
        c["side_right"] = occ.addLine(p_br, p_r)
        c["head_r"] = occ.addCircleArc(p_r, c0, p_top)
        c["head_l"] = occ.addCircleArc(p_top, c0, p_l)
        c["side_left"] = occ.addLine(p_l, p_bl)
        outer = occ.addCurveLoop([c["base"], c["side_right"], c["head_r"], c["head_l"], c["side_left"]])
        h_r, h_t, h_l, h_b = P(a, yc), P(0, yc + a), P(-a, yc), P(0, yc - a)
        c["bore_ur"] = occ.addCircleArc(h_r, c0, h_t)
        c["bore_ul"] = occ.addCircleArc(h_t, c0, h_l)
        c["bore_ll"] = occ.addCircleArc(h_l, c0, h_b)
        c["bore_lr"] = occ.addCircleArc(h_b, c0, h_r)
        inner = occ.addCurveLoop([c["bore_ur"], c["bore_ul"], c["bore_ll"], c["bore_lr"]])
        self._surf = occ.addPlaneSurface([outer, inner])
        occ.remove([(0, c0)])
        occ.synchronize()

    def _groups(self, full: bool) -> None:
        c = self._c
        add_group(1, [c["base"]], "base")
        add_group(1, [c["bore_ur"], c["bore_ul"]], "bore_loaded")
        add_group(1, [c["bore_ll"], c["bore_lr"]], "bore_free")
        add_group(1, [c["bore_ur"], c["bore_ul"], c["bore_ll"], c["bore_lr"]], "bore")
        add_group(1, [c["side_right"], c["side_left"], c["head_r"], c["head_l"]], "outer_edge")
        add_group(2, [self._surf], "lug")

    def apply_sizing(self, h: float) -> None:
        W, a, yc, b = self.dims
        c = self._c
        f = threshold(0.25 * h, h, 0.1 * a, 0.1 * a + 3.0 * self.params["h0"],
                      curves=[c["bore_ur"], c["bore_ul"], c["bore_ll"], c["bore_lr"]])
        finish_sizing([f], h)

    def qois(self) -> list[QoI]:
        return [QoI(name="peak_hoop_stress", type="max", field="s1", group="bore"),
                QoI(name="lug_elongation", type="mean", field="uy", group="bore_loaded")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        W, a, yc, b = self.dims
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_stress", order=order, thickness=b),
            materials={"steel": material(self.params)},
            bcs=[BC(type="fixed", group="base"),
                 BC(type="pressure", group="bore_loaded", value=self.pressure_expr())],
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "full",
                  "load_model": "cosine bearing pressure on the upper half bore, p0 = 2P/(pi a b)"},
        )


class ClevisLug3D(ClevisLug2D):
    """Extruded lug. Reference mesh: quarter x >= 0, z <= b/2 (symmetry planes x = 0, z = b/2)."""

    dim = 3
    formulation = "3d"
    reference_portion = "quarter"
    default_params = {**_LugBase.default_params, "h0": 4 * MM}

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        W, a, _, _ = self.dims
        self.tags = ("part", "3d", "lug", "pin_loaded", "bearing", "expression_load",
                     "stress_concentration")
        self.title = f"Pin-loaded lug, D/W = {2 * a / W:g} (3D solid)"

    def description(self) -> str:
        return (
            "Pin-loaded lug (clevis eye) in bearing. 3D solid.\n" + self._text(False) + "\n"
            "The pressure is uniform through the thickness. Exploiting symmetry is allowed.\n"
            "Report:\n"
            "  peak_hoop_stress: the maximum value of the maximum principal stress s1 on the bore "
            "surface (both halves) [Pa];\n"
            "  lug_elongation: the average y-displacement of the loaded upper half of the bore [m]."
        )

    def _shape(self, full: bool) -> None:
        super()._shape(full)
        occ = gmsh.model.occ
        W, a, yc, b = self.dims
        out = occ.extrude([(2, self._surf)], 0, 0, b)
        vol = [t for d, t in out if d == 3][0]
        if not full:
            box = occ.addBox(0, -1 * MM, -1 * MM, W, yc + W, b / 2 + 1 * MM)
            out, _ = occ.intersect([(3, vol)], [(3, box)])
            vol = out[0][1]
        occ.synchronize()
        self._vol = vol

    def _bore_faces(self, upper: bool) -> list[int]:
        W, a, yc, b = self.dims
        out = []
        for _, t in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, t) != "Cylinder":
                continue
            x, y, z = com(2, t)
            if math.hypot(x, y - yc) < a + 1e-6 and ((y > yc) == upper):
                out.append(t)
        return out

    def _groups(self, full: bool) -> None:
        W, a, yc, b = self.dims
        up, lo = self._bore_faces(True), self._bore_faces(False)
        add_group(2, entities_on_plane(2, 1, 0.0), "base")
        add_group(2, up, "bore_loaded")
        add_group(2, lo, "bore_free")
        add_group(2, up + lo, "bore")
        if not full:
            add_group(2, entities_on_plane(2, 0, 0.0), "sym_x0")
            add_group(2, entities_on_plane(2, 2, b / 2), "sym_mid")
        add_group(3, [self._vol], "lug")

    def apply_sizing(self, h: float) -> None:
        W, a, yc, b = self.dims
        f = threshold(0.3 * h, h, 0.1 * a, 0.1 * a + 2.5 * self.params["h0"],
                      surfaces=self._bore_faces(True) + self._bore_faces(False))
        finish_sizing([f], h)

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        W, a, yc, b = self.dims
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"steel": material(self.params)},
            bcs=[BC(type="fixed", group="base"),
                 BC(type="pressure", group="bore_loaded", value=self.pressure_expr()),
                 BC(type="symmetry", group="sym_x0"),
                 BC(type="symmetry", group="sym_mid")],
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "quarter (x>=0, z<=b/2) with symmetry",
                  "load_model": "cosine bearing pressure on the upper half bore, p0 = 2P/(pi a b)"},
        )


register(ClevisLug2D(id="clevis_lug_2d_D20"))
register(ClevisLug2D(id="clevis_lug_2d_D28", hole_diameter=28 * MM))
register(ClevisLug3D(id="clevis_lug_3d_D20"))
