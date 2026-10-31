"""Cylindrical pressure vessel with hemispherical heads under internal pressure (axisymmetric).

QoIs: the hoop stress at the inner surface at mid-length (reference: Lame thick-cylinder
solution, exact far from the heads: the head-junction disturbance decays like
exp(-beta x), beta = [3(1-nu^2)]^0.25 / sqrt(R t), i.e. by ~1e-4 at mid-length) and the peak
von Mises stress in the wall (no reference; the head-to-cylinder junction disturbs it).

The reference mesh is the half y >= 0 with a symmetry line at mid-length (the STEP is the full
meridional section). Variants: equal head thickness, and a thinner head (t_head = 0.6 t) with
the inside surfaces flush and a 3:1 outside taper in the cylinder (a common ASME-style detail).
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.base import Reference
from feacore.problems.parts._common import (
    MM, STEEL, PartProblem, add_group, finish_sizing, fmt_mm, lame_hoop_stress, material, threshold,
)
from feacore.problems.registry import register


class PressureVesselAxi(PartProblem):
    family = "pressure_vessel_axisym"
    dim = 2
    physics = "elasticity"
    formulation = "axisymmetric"
    reference_portion = "half"
    default_params = {
        "inner_radius": 250 * MM,
        "wall_thickness": 20 * MM,       # cylinder
        "head_thickness": 20 * MM,       # hemispherical heads
        "taper_ratio": 3.0,              # length / thickness step of the outside taper (thin heads)
        "cylinder_length": 1000 * MM,    # straight length between the two head junctions
        "pressure": 5e6,                 # internal pressure [Pa]
        "E": STEEL["E"],
        "nu": STEEL["nu"],
        "h0": 10 * MM,
    }

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        p = self.params
        self.thin_head = p["head_thickness"] < p["wall_thickness"] - 1e-12
        self.tags = ("part", "axisymmetric", "pressure_vessel", "internal_pressure", "analytical",
                     "shell_junction") + (("thickness_transition",) if self.thin_head else ())
        self.title = ("Pressure vessel with hemispherical heads" +
                      (", thin heads with tapered transition" if self.thin_head else ", equal thickness") +
                      " (axisymmetric)")

    @property
    def dims(self):
        p = self.params
        ri, t, th = p["inner_radius"], p["wall_thickness"], p["head_thickness"]
        return ri, ri + t, ri + th, p["cylinder_length"] / 2.0

    def description(self) -> str:
        p = self.params
        ri, ro, roh, hl = self.dims
        t, th = p["wall_thickness"], p["head_thickness"]
        if self.thin_head:
            taper = (t - th) * p["taper_ratio"]
            heads = (f"The hemispherical heads have inner radius {fmt_mm(ri)} and a thinner wall of "
                     f"{fmt_mm(th)}; the inside surface is flush across the junction. On the outside, "
                     f"the cylinder wall tapers linearly from {fmt_mm(t)} to {fmt_mm(th)} over the last "
                     f"{fmt_mm(taper)} before each junction (a {p['taper_ratio']:g}:1 taper).")
        else:
            heads = (f"The hemispherical heads have inner radius {fmt_mm(ri)} and the same wall thickness "
                     f"{fmt_mm(th)}; inside and outside surfaces are tangent-continuous at the junctions.")
        return (
            "Steel pressure vessel: a cylindrical shell closed by two hemispherical heads, under "
            "internal pressure. Axisymmetric problem.\n"
            "Geometry file (STEP): the meridional section of the vessel wall, drawn in the xy-plane: "
            "x is the radius (x >= 0, axis of revolution x = 0) and y is the axial coordinate z, with "
            "z = 0 at mid-length.\n"
            f"Cylinder: inner radius {fmt_mm(ri)}, wall thickness {fmt_mm(t)}, straight length "
            f"{fmt_mm(2 * hl)} between the head junctions (z from {fmt_mm(-hl)} to {fmt_mm(hl)}). {heads}\n"
            f"Material: steel, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, Poisson's ratio "
            f"{p['nu']:g}.\n"
            f"Load: internal pressure {p['pressure'] / 1e6:g} MPa on every inner surface (cylinder and "
            "heads). No other loads; the vessel is free (self-equilibrated): constrain only the axial "
            "rigid-body translation, without stressing the vessel.\n"
            "Analysis: linear static, small displacements, axisymmetric.\n"
            "Report (in an axisymmetric model, x = r and y = z):\n"
            f"  hoop_stress_mid: the hoop (circumferential) stress at the inner surface at mid-length, "
            f"the point (r, z) = ({fmt_mm(ri)}, 0) [Pa];\n"
            "  peak_von_mises: the maximum von Mises stress anywhere in the vessel wall [Pa]."
        )

    # ---- geometry
    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        p = self.params
        ri, ro, roh, hl = self.dims
        t, th = p["wall_thickness"], p["head_thickness"]
        taper = (t - th) * p["taper_ratio"] if self.thin_head else 0.0
        P = lambda x, y: occ.addPoint(x, y, 0)  # noqa: E731
        c = self._c = {"inner": [], "outer": [], "axis": [], "sym": []}

        def half(sign: int, start_inner, start_outer):
            """Head + cylinder wall for y*sign >= 0, from the mid-plane points outward."""
            yj = sign * hl
            ctr = P(0, yj)
            pi_j, pi_apex = P(ri, yj), P(0, yj + sign * ri)
            po_apex = P(0, yj + sign * roh)
            po_j = P(roh, yj)
            c["inner"].append(occ.addLine(start_inner, pi_j))
            c["inner"].append(occ.addCircleArc(pi_j, ctr, pi_apex))
            c["axis"].append(occ.addLine(pi_apex, po_apex))
            c["outer"].append(occ.addCircleArc(po_apex, ctr, po_j))
            if self.thin_head:
                pt = P(ro, yj - sign * taper)
                c["outer"].append(occ.addLine(po_j, pt))
                c["outer"].append(occ.addLine(pt, start_outer))
            else:
                c["outer"].append(occ.addLine(po_j, start_outer))
            return ctr

        centres = []
        if full:
            # trace: inner (0 -> +), top head, outer (+ -> 0) ... and the mirrored half
            pm_i, pm_o = P(ri, 0.0), P(ro, 0.0)
            centres.append(half(+1, pm_i, pm_o))
            centres.append(half(-1, pm_i, pm_o))
            loop_curves = self._loop_full()
        else:
            pm_i, pm_o = P(ri, 0.0), P(ro, 0.0)
            centres.append(half(+1, pm_i, pm_o))
            c["sym"].append(occ.addLine(pm_o, pm_i))
            loop_curves = c["inner"] + c["axis"] + c["outer"] + c["sym"]
        loop = occ.addCurveLoop(loop_curves)
        self._surf = occ.addPlaneSurface([loop])
        occ.remove([(0, t) for t in centres])
        occ.synchronize()

    def _loop_full(self) -> list[int]:
        c = self._c
        n = 3 if self.thin_head else 2
        top_in, bot_in = c["inner"][0:2], c["inner"][2:4]
        top_out, bot_out = c["outer"][0:n], c["outer"][n:2 * n]
        # mid-inner -> top junction -> top apex -> (axis) -> top outer ... -> mid-outer, then the
        # bottom half traversed backwards from mid-outer to mid-inner.
        return (top_in + [c["axis"][0]] + top_out + [-t for t in reversed(bot_out)] +
                [-c["axis"][1]] + [-t for t in reversed(bot_in)])

    def _groups(self, full: bool) -> None:
        c = self._c
        add_group(1, c["inner"], "inner_surface")
        add_group(1, c["outer"], "outer_surface")
        add_group(1, c["axis"], "axis")
        if not full:
            add_group(1, c["sym"], "sym_mid")
        add_group(2, [self._surf], "wall")

    def apply_sizing(self, h: float) -> None:
        # thickness-resolving size everywhere (h0 = t/2), finer near the junction(s)
        ri, ro, roh, hl = self.dims
        pts = [t for _, t in gmsh.model.getEntitiesInBoundingBox(ri - 1e-6, hl - 1e-6, -1e-6, ri + 1e-6,
                                                                  hl + 1e-6, 1e-6, 0)]
        f = threshold(0.5 * h, h, 0.0, 0.2, points=pts)
        finish_sizing([f], h)

    # ---- analysis
    def qois(self) -> list[QoI]:
        ri = self.params["inner_radius"]
        return [QoI(name="hoop_stress_mid", type="point", field="stt", point=[ri, 0.0]),
                QoI(name="peak_von_mises", type="max", field="von_mises", group=None)]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        bcs = [BC(type="pressure", group="inner_surface", value=p["pressure"]),
               BC(type="displacement", group="axis", components=[0], value=0.0)]
        if self.reference_portion == "half":
            bcs.append(BC(type="symmetry", group="sym_mid"))
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="axisymmetric", order=order),
            materials={"steel": material(p)},
            bcs=bcs,
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "half (z >= 0) with symmetry at z = 0"},
        )

    def references(self) -> list[Reference]:
        p = self.params
        ri, ro, _, hl = self.dims
        R, t = 0.5 * (ri + ro), ro - ri
        beta = (3 * (1 - p["nu"] ** 2)) ** 0.25 / math.sqrt(R * t)
        return [Reference(
            qoi="hoop_stress_mid", value=lame_hoop_stress(p["pressure"], ri, ro, ri), kind="analytical",
            source=("Lame thick-walled cylinder under internal pressure: sigma_theta(r) = p ri^2 (1 + "
                    "ro^2/r^2) / (ro^2 - ri^2), evaluated at r = ri (Timoshenko & Goodier, Theory of "
                    "Elasticity, 3rd ed., sec. 33); closed ends add a uniform axial stress that does not "
                    "change sigma_theta."),
            tolerance=0.01,
            notes=(f"Exact far from the heads; junction disturbance at mid-length ~ exp(-beta L/2) = "
                   f"{math.exp(-beta * hl):.1e} (beta = {beta:.3g} 1/m)."),
        )]


register(PressureVesselAxi(id="pressure_vessel_axi_equal"))
register(PressureVesselAxi(id="pressure_vessel_axi_thin_head", head_thickness=12 * MM))
