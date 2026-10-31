"""Stepped round bar (shaft) with a shoulder fillet in axial tension.

Reference: Kt from the Peterson / Pilkey curve fit for a stepped bar of circular cross section
with a shoulder fillet in tension (see ``_common.kt_shoulder_fillet_round_bar_tension``) times
the nominal stress on the small diameter, sigma_nom = 4 P / (pi d^2).

Variants: axisymmetric over D/d in {1.5, 2.0} x r/d in {0.05, 0.1, 0.2}; one sharp-shoulder
trap (r = 0, axisymmetric, singular re-entrant corner); one full-3D variant (D/d = 2, r/d = 0.1)
whose reference mesh is a 90-degree sector with symmetry planes.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.base import Reference
from feacore.problems.parts._common import (
    MM, STEEL, PartProblem, add_group, com, entities_on_plane, finish_sizing, fmt_mm,
    kt_shoulder_fillet_round_bar_tension, material, threshold,
)
from feacore.problems.registry import register

SIGMA_NOM = 100e6                        # target nominal stress on the small diameter [Pa]
D_SMALL = 20 * MM
FORCE = SIGMA_NOM * math.pi * D_SMALL**2 / 4.0   # 31.416 kN

KT_SOURCE = (
    "Pilkey & Pilkey, Peterson's Stress Concentration Factors, 3rd ed. (Wiley 2008), Chapter 3, "
    "chart for a stepped bar of circular cross section with a shoulder fillet in tension "
    "(Chart 3.4; photoelastic data of Allison 1961) and its curve fit "
    "Kt = C1 + C2(2h/D) + C3(2h/D)^2 + C4(2h/D)^3, h = (D-d)/2, C_i(sqrt(h/r), h/r) for "
    "0.1<=h/r<=2 and 2<=h/r<=20; same fit in Young & Budynas, Roark's Formulas for Stress and "
    "Strain, 7th ed., Table 17.1. Peak stress = Kt * 4P/(pi d^2)."
)


class _ShaftBase(PartProblem):
    family = "stepped_shaft"
    physics = "elasticity"
    expected_behavior = "stress_concentration"
    default_params = {
        "d": D_SMALL,             # small diameter [m]
        "D_over_d": 2.0,
        "r_over_d": 0.1,          # 0 -> sharp shoulder (singular trap)
        "L_small_over_d": 3.0,    # small section: end face z = 0 to the shoulder plane
        "L_large_over_D": 1.5,    # large section length beyond the shoulder plane
        "force": FORCE,           # axial tensile force [N]
        "E": STEEL["E"],
        "nu": STEEL["nu"],
        "h0": 2.5 * MM,
    }

    # ---- derived dimensions
    @property
    def dims(self):
        p = self.params
        d = p["d"]
        D = p["D_over_d"] * d
        r = p["r_over_d"] * d
        zs = p["L_small_over_d"] * d
        L = zs + p["L_large_over_D"] * D
        return d, D, r, zs, L

    @property
    def sharp(self) -> bool:
        return self.params["r_over_d"] <= 0.0

    def nominal_stress(self) -> float:
        d = self.params["d"]
        return self.params["force"] / (math.pi * d**2 / 4.0)

    def kt(self) -> float:
        d, D, r, _, _ = self.dims
        return kt_shoulder_fillet_round_bar_tension(D, d, r)

    # ---- agent-facing text shared by the variants
    def _common_text(self) -> str:
        d, D, r, zs, L = self.dims
        p = self.params
        fillet = (f"The step is blended by a fillet of radius r = {fmt_mm(r)}, tangent to the small "
                  f"cylinder and to the flat annular shoulder face in the plane z = {fmt_mm(zs)}."
                  if not self.sharp else
                  f"The step is SHARP as drawn (no fillet): the small cylinder meets the flat "
                  f"annular shoulder face in the plane z = {fmt_mm(zs)} at a 90-degree inside corner.")
        return (
            f"Purpose: a turned steel tension rod whose diameter steps from d = {fmt_mm(d)} to "
            f"D = {fmt_mm(D)}. {fillet} The small-diameter section runs from the end face at z = 0 "
            f"to the shoulder; the large-diameter section runs from the shoulder plane to the end "
            f"face at z = {fmt_mm(L)}.\n"
            f"Material: steel, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, "
            f"Poisson's ratio {p['nu']:g}.\n"
            f"Load: a total axial tensile force P = {p['force'] / 1e3:.3f} kN, applied as a uniform "
            f"axial traction (pulling outward) on the large end face z = {fmt_mm(L)}. The nominal "
            f"stress on the small diameter, 4P/(pi d^2), is {self.nominal_stress() / 1e6:.0f} MPa.\n"
            "Analysis: linear static, small displacements."
        )


class SteppedShaftAxi(_ShaftBase):
    dim = 2
    formulation = "axisymmetric"

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        tags = ["part", "axisymmetric", "tension", "stepped_shaft"]
        tags += ["sharp_corner", "singular_trap"] if self.sharp else ["fillet", "stress_concentration", "handbook"]
        self.tags = tuple(tags)
        self.expected_behavior = "singular" if self.sharp else "stress_concentration"
        d, D, r, _, _ = self.dims
        self.title = (f"Stepped shaft in tension, D/d = {D / d:g}, "
                      + ("sharp shoulder (r = 0)" if self.sharp else f"r/d = {r / d:g}") + " (axisymmetric)")

    def description(self) -> str:
        qoi = ("peak_stress_fillet: the maximum value of the maximum principal stress s1 on the "
               "fillet surface [Pa]" if not self.sharp else
               "peak_stress_shoulder: the maximum value of the maximum principal stress s1 on the "
               "small-cylinder surface and the shoulder face, i.e. at the inside corner where they "
               "meet [Pa]; report it, and say whether you trust it")
        return (
            "Stepped round bar (shaft) with a shoulder, in axial tension. Axisymmetric problem.\n"
            "Geometry file (STEP): the meridional half-section of the solid of revolution, drawn in "
            "the xy-plane: x is the radius (x >= 0, axis of revolution x = 0) and y is the axial "
            "coordinate z.\n" + self._common_text() + " Model it as axisymmetric.\n"
            "Supports: the small end face z = 0 is supported axially only: zero axial displacement, "
            "free to contract radially (a frictionless support).\n"
            "Report (in an axisymmetric model, x = r and y = z):\n"
            f"  {qoi};\n"
            "  elongation: the average axial displacement of the large end face (z = "
            f"{fmt_mm(self.dims[4])}) [m]."
        )

    # ---- geometry
    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        d, D, r, zs, L = self.dims
        a, b = d / 2.0, D / 2.0
        p1 = occ.addPoint(0, 0, 0)
        p2 = occ.addPoint(a, 0, 0)
        if self.sharp:
            p3 = occ.addPoint(a, zs, 0)
            p4 = p3
        else:
            p3 = occ.addPoint(a, zs - r, 0)
            pc = occ.addPoint(a + r, zs - r, 0)
            p4 = occ.addPoint(a + r, zs, 0)
        p5 = occ.addPoint(b, zs, 0)
        p6 = occ.addPoint(b, L, 0)
        p7 = occ.addPoint(0, L, 0)
        self._c = c = {}
        c["end_small"] = occ.addLine(p1, p2)
        c["surface_small"] = occ.addLine(p2, p3)
        if not self.sharp:
            c["fillet"] = occ.addCircleArc(p3, pc, p4)
        c["shoulder"] = occ.addLine(p4, p5)
        c["surface_large"] = occ.addLine(p5, p6)
        c["end_large"] = occ.addLine(p6, p7)
        c["axis"] = occ.addLine(p7, p1)
        order = ["end_small", "surface_small"] + ([] if self.sharp else ["fillet"]) + \
                ["shoulder", "surface_large", "end_large", "axis"]
        loop = occ.addCurveLoop([c[k] for k in order])
        self._surf = occ.addPlaneSurface([loop])
        if not self.sharp:
            occ.remove([(0, pc)])
        occ.synchronize()

    def _groups(self, full: bool) -> None:
        for name, tag in self._c.items():
            add_group(1, [tag], name)
        if self.sharp:
            add_group(1, [self._c["surface_small"], self._c["shoulder"]], "shoulder_corner")
        add_group(2, [self._surf], "shaft")

    def apply_sizing(self, h: float) -> None:
        d, D, r, zs, L = self.dims
        h0 = self.params["h0"]
        if self.sharp:
            corner = [t for _, t in gmsh.model.getEntitiesInBoundingBox(
                d / 2 - 1e-9, zs - 1e-9, -1e-9, d / 2 + 1e-9, zs + 1e-9, 1e-9, 0)]
            f = threshold(0.1 * h, h, 0.0, 4.0 * h0, points=corner)
        else:
            kf = min(1.0, r / (4.0 * h0))
            f = threshold(kf * h, h, 0.5 * r, 0.5 * r + 4.0 * h0, curves=[self._c["fillet"]])
        finish_sizing([f], h)

    # ---- analysis
    def qois(self) -> list[QoI]:
        peak = (QoI(name="peak_stress_shoulder", type="max", field="s1", group="shoulder_corner")
                if self.sharp else QoI(name="peak_stress_fillet", type="max", field="s1", group="fillet"))
        return [peak, QoI(name="elongation", type="mean", field="uy", group="end_large")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        d, D, r, zs, L = self.dims
        p = self.params
        traction = p["force"] / (math.pi * D**2 / 4.0)
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="axisymmetric", order=order),
            materials={"steel": material(p)},
            bcs=[
                BC(type="symmetry", group="end_small"),
                BC(type="displacement", group="axis", components=[0], value=0.0),
                BC(type="traction", group="end_large", value=[0.0, traction]),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "full"},
        )

    def references(self) -> list[Reference]:
        if self.sharp:
            return []
        d, D, r, _, _ = self.dims
        kt = self.kt()
        return [Reference(
            qoi="peak_stress_fillet", value=kt * self.nominal_stress(), kind="handbook",
            source=KT_SOURCE, tolerance=0.05,
            notes=(f"D/d = {D / d:g}, r/d = {r / d:g}, h/r = {(D - d) / 2 / r:.3g}: Kt = {kt:.4f} on "
                   f"sigma_nom = {self.nominal_stress() / 1e6:.2f} MPa. The fit summarises "
                   "photoelastic charts; allow ~5 %. Tipton, Sorem & Rolovic (J. Mech. Des. 118, "
                   "1996) report FE values above the published charts for some shoulder geometries."),
        )]

    def expectations(self) -> dict[str, str]:
        return {"peak_stress_shoulder": "diverge"} if self.sharp else {}


class SteppedShaft3D(_ShaftBase):
    """Same part as a 3D solid. The reference mesh is the 90-degree sector x >= 0, y >= 0 with
    symmetry planes x = 0 and y = 0 (the STEP is the full bar)."""

    dim = 3
    formulation = "3d"
    reference_portion = "quarter"
    default_params = {**_ShaftBase.default_params, "h0": 4.0 * MM}

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        if self.sharp:
            raise ValueError("SteppedShaft3D needs r > 0")
        d, D, r, _, _ = self.dims
        self.tags = ("part", "3d", "tension", "stepped_shaft", "fillet", "stress_concentration",
                     "handbook", "axisymmetric_part")
        self.title = f"Stepped shaft in tension, D/d = {D / d:g}, r/d = {r / d:g} (3D solid)"

    def description(self) -> str:
        return (
            "Stepped round bar (shaft) with a shoulder fillet, in axial tension. 3D solid.\n"
            "Geometry file (STEP): the full 3D bar; its axis is the z axis, the small end face lies "
            "in the plane z = 0.\n" + self._common_text() + "\n"
            "Supports: the small end face z = 0 is supported axially only (zero axial displacement, "
            "free to move in its plane: a frictionless support). Add only the minimum extra "
            "constraints needed to remove the remaining rigid-body motions (in-plane translation, "
            "rotation about the axis) without stressing the part. Exploiting symmetry is allowed.\n"
            "Report:\n"
            "  peak_stress_fillet: the maximum value of the maximum principal stress s1 on the "
            "fillet surface [Pa];\n"
            f"  elongation: the average axial displacement uz of the large end face (z = "
            f"{fmt_mm(self.dims[4])}) [m]."
        )

    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        d, D, r, zs, L = self.dims
        a, b = d / 2.0, D / 2.0
        c1 = occ.addCylinder(0, 0, 0, 0, 0, zs, a)
        c2 = occ.addCylinder(0, 0, zs, 0, 0, L - zs, b)
        out, _ = occ.fuse([(3, c1)], [(3, c2)])
        occ.synchronize()
        vol = out[0][1]
        edge = [t for _, t in gmsh.model.getEntitiesInBoundingBox(
            -a - 1e-6, -a - 1e-6, zs - 1e-6, a + 1e-6, a + 1e-6, zs + 1e-6, 1)]
        out = occ.fillet([vol], edge, [r])
        vol = out[0][1]
        if not full:
            box = occ.addBox(0, 0, -1e-3, b + 1e-3, b + 1e-3, L + 2e-3)
            out, _ = occ.intersect([(3, vol)], [(3, box)])
            vol = out[0][1]
        occ.synchronize()
        self._vol = vol

    def _fillet_faces(self) -> list[int]:
        d, D, r, zs, L = self.dims
        out = []
        for _, t in gmsh.model.getEntities(2):  # OCC bounding boxes of curved faces are loose: use COM
            x, y, z = com(2, t)
            if zs - r < z < zs and gmsh.model.getType(2, t) != "Plane":
                out.append(t)
        return out

    def _groups(self, full: bool) -> None:
        d, D, r, zs, L = self.dims
        add_group(2, entities_on_plane(2, 2, 0.0), "end_small")
        add_group(2, entities_on_plane(2, 2, L), "end_large")
        add_group(2, self._fillet_faces(), "fillet")
        if not full:
            add_group(2, entities_on_plane(2, 0, 0.0), "sym_x0")
            add_group(2, entities_on_plane(2, 1, 0.0), "sym_y0")
        add_group(3, [self._vol], "shaft")

    def apply_sizing(self, h: float) -> None:
        d, D, r, zs, L = self.dims
        h0 = self.params["h0"]
        kf = min(1.0, r / (2.0 * h0))
        f = threshold(kf * h, h, 0.5 * r, 0.5 * r + 3.0 * h0, surfaces=self._fillet_faces())
        finish_sizing([f], h)

    def qois(self) -> list[QoI]:
        return [QoI(name="peak_stress_fillet", type="max", field="s1", group="fillet"),
                QoI(name="elongation", type="mean", field="uz", group="end_large")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        d, D, r, zs, L = self.dims
        p = self.params
        traction = p["force"] / (math.pi * D**2 / 4.0)
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"steel": material(p)},
            bcs=[
                BC(type="symmetry", group="end_small"),
                BC(type="symmetry", group="sym_x0"),
                BC(type="symmetry", group="sym_y0"),
                BC(type="traction", group="end_large", value=[0.0, 0.0, traction]),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "quarter (x>=0, y>=0) with symmetry"},
        )

    def references(self) -> list[Reference]:
        d, D, r, _, _ = self.dims
        kt = self.kt()
        return [Reference(
            qoi="peak_stress_fillet", value=kt * self.nominal_stress(), kind="handbook",
            source=KT_SOURCE, tolerance=0.05,
            notes=f"D/d = {D / d:g}, r/d = {r / d:g}: Kt = {kt:.4f} on sigma_nom = "
                  f"{self.nominal_stress() / 1e6:.2f} MPa (same part as the axisymmetric variant).",
        )]


def _tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


for _Dd in (1.5, 2.0):
    for _rd in (0.05, 0.1, 0.2):
        register(SteppedShaftAxi(id=f"stepped_shaft_axi_Dd{_tag(_Dd)}_rd{_tag(_rd)}",
                                 D_over_d=_Dd, r_over_d=_rd))
register(SteppedShaftAxi(id="stepped_shaft_axi_Dd2_sharp", D_over_d=2.0, r_over_d=0.0))
register(SteppedShaft3D(id="stepped_shaft_3d_Dd2_rd0p1", D_over_d=2.0, r_over_d=0.1))
