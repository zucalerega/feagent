"""3D steel angle (L) bracket: one leg bolted (bolt-hole bores fixed), a transverse load on the
end of the other leg. Variants over the inside fillet radius r in {0, 2, 5, 10} mm.

r = 0 is the deliberate trap: the sharp re-entrant inside corner is a stress singularity, so the
peak von Mises stress there grows without bound under mesh refinement (expectation "diverge").
There is no handbook reference; mesh-refinement consensus (GCI) is the check. The reference mesh
is the half y <= W/2 with a symmetry plane (the STEP is the full bracket).
"""

from __future__ import annotations

from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.parts._common import (
    MM, STEEL, PartProblem, add_group, com, entities_on_plane, finish_sizing, fmt_mm, material,
    threshold,
)
from feacore.problems.registry import register


class LBracket(PartProblem):
    family = "l_bracket"
    dim = 3
    physics = "elasticity"
    formulation = "3d"
    reference_portion = "half"
    default_params = {
        "thickness": 8 * MM,        # both legs
        "width": 50 * MM,           # along y
        "base_length": 80 * MM,     # bolted leg, along x
        "upright_height": 60 * MM,  # loaded leg, along z
        "fillet_radius": 5 * MM,    # inside corner; 0 = sharp
        "hole_diameter": 9 * MM,    # M8 clearance
        "hole_x": 55 * MM,
        "hole_y": (13 * MM, 37 * MM),
        "force": 1000.0,            # [N], in -x on the upright's top end face
        "E": STEEL["E"],
        "nu": STEEL["nu"],
        "h0": 4 * MM,
    }

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        r = self.params["fillet_radius"]
        self.sharp = r <= 0.0
        tags = ["part", "3d", "bolted", "bending", "l_bracket"]
        tags += ["sharp_corner", "singular_trap"] if self.sharp else ["fillet", "stress_concentration"]
        self.tags = tuple(tags)
        self.expected_behavior = "singular" if self.sharp else "stress_concentration"
        self.title = ("Bolted L-bracket, sharp inside corner (r = 0)" if self.sharp else
                      f"Bolted L-bracket, inside fillet r = {fmt_mm(r)}")

    @property
    def corner_group(self) -> str:
        return "corner_edge" if self.sharp else "fillet"

    def description(self) -> str:
        p = self.params
        t, W, LA, LB, r = p["thickness"], p["width"], p["base_length"], p["upright_height"], p["fillet_radius"]
        hx, (hy1, hy2), hd = p["hole_x"], p["hole_y"], p["hole_diameter"]
        corner = (f"The inside corner between the legs is blended by a fillet of radius r = {fmt_mm(r)} "
                  f"(axis along y, tangent to the top face of the base leg z = {fmt_mm(t)} and the inner "
                  f"face of the upright x = {fmt_mm(t)})." if not self.sharp else
                  f"The inside corner between the legs is SHARP as drawn (no fillet): the top face of the "
                  f"base leg z = {fmt_mm(t)} meets the inner face of the upright x = {fmt_mm(t)} at a "
                  f"90-degree inside edge along the line x = z = {fmt_mm(t)}.")
        qoi = ("max_vm_corner: the maximum von Mises stress on the fillet surface [Pa]" if not self.sharp
               else "max_vm_corner: the maximum von Mises stress along the sharp inside corner edge "
                    f"(x = z = {fmt_mm(t)}) [Pa]; report it, and say whether you trust it")
        return (
            "Bolted steel angle bracket (L-bracket) with a transverse load on its upright leg. 3D solid.\n"
            "Geometry file (STEP): the full bracket in mm-scale dimensions written in metres.\n"
            f"Purpose: a machined L-bracket of constant thickness {fmt_mm(t)} and width {fmt_mm(W)} "
            f"(y from 0 to {fmt_mm(W)}). The base leg lies on z = 0 .. {fmt_mm(t)} and extends from "
            f"x = 0 to x = {fmt_mm(LA)}; the upright leg lies on x = 0 .. {fmt_mm(t)} and rises from "
            f"z = 0 to z = {fmt_mm(LB)}. The outside corner (x = 0, z = 0) is square. {corner} Two "
            f"through holes of diameter {fmt_mm(hd)} (M8 bolt clearance) pass vertically through "
            f"the base leg, centred at (x, y) = ({fmt_mm(hx)}, {fmt_mm(hy1)}) and "
            f"({fmt_mm(hx)}, {fmt_mm(hy2)}).\n"
            f"Material: steel, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, Poisson's ratio "
            f"{p['nu']:g}.\n"
            "Supports: the base leg is bolted down; model the bolts by fixing the two bolt-hole bores "
            "(the cylindrical hole surfaces): all displacement components zero there.\n"
            f"Load: a total force of {p['force'] / 1e3:g} kN in the -x direction (pulling the upright "
            f"away from the base leg, opening the angle), uniformly distributed over the top end face "
            f"of the upright leg (the face z = {fmt_mm(LB)}).\n"
            "Analysis: linear static, small displacements.\n"
            "Report:\n"
            f"  {qoi};\n"
            f"  load_displacement: the average x-displacement of the loaded face z = {fmt_mm(LB)} [m]."
        )

    # ---- geometry
    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        p = self.params
        t, W, LA, LB, r = p["thickness"], p["width"], p["base_length"], p["upright_height"], p["fillet_radius"]
        a = occ.addBox(0, 0, 0, LA, W, t)
        b = occ.addBox(0, 0, 0, t, W, LB)
        out, _ = occ.fuse([(3, a)], [(3, b)])
        occ.synchronize()
        vol = out[0][1]
        if not self.sharp:
            e = 1e-6
            edge = [tg for _, tg in gmsh.model.getEntitiesInBoundingBox(t - e, -e, t - e, t + e, W + e, t + e, 1)]
            out = occ.fillet([vol], edge, [r])
            vol = out[0][1]
        holes = [(3, occ.addCylinder(p["hole_x"], y, -1 * MM, 0, 0, t + 2 * MM, p["hole_diameter"] / 2))
                 for y in p["hole_y"]]
        out, _ = occ.cut([(3, vol)], holes)
        vol = out[0][1]
        if not full:
            box = occ.addBox(-1 * MM, -1 * MM, -1 * MM, LA + 2 * MM, W / 2 + 1 * MM, LB + 2 * MM)
            out, _ = occ.intersect([(3, vol)], [(3, box)])
            vol = out[0][1]
        occ.synchronize()
        self._vol = vol

    def _hole_faces(self) -> list[int]:
        p = self.params
        out = []
        for _, tg in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, tg) != "Cylinder":
                continue
            x, y, z = com(2, tg)
            if abs(x - p["hole_x"]) < 1e-4 and z < p["thickness"]:
                out.append(tg)
        return out

    def _fillet_faces(self) -> list[int]:
        p = self.params
        t, r = p["thickness"], p["fillet_radius"]
        out = []
        for _, tg in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, tg) == "Plane":
                continue
            x, y, z = com(2, tg)
            if t < x < t + r and t < z < t + r:
                out.append(tg)
        return out

    def _corner_curves(self) -> list[int]:
        t, W = self.params["thickness"], self.params["width"]
        e = 1e-6
        return [tg for _, tg in gmsh.model.getEntitiesInBoundingBox(t - e, -e, t - e, t + e, W + e, t + e, 1)]

    def _groups(self, full: bool) -> None:
        p = self.params
        add_group(2, self._hole_faces(), "bolt_holes")
        add_group(2, entities_on_plane(2, 2, p["upright_height"]), "load_face")
        if self.sharp:
            add_group(1, self._corner_curves(), "corner_edge")
        else:
            add_group(2, self._fillet_faces(), "fillet")
        if not full:
            add_group(2, entities_on_plane(2, 1, p["width"] / 2), "sym_mid")
        add_group(3, [self._vol], "bracket")

    def apply_sizing(self, h: float) -> None:
        p = self.params
        r, h0 = p["fillet_radius"], p["h0"]
        if self.sharp:
            fc = threshold(0.25 * h, h, 0.0, 3.0 * h0, curves=self._corner_curves())
        else:
            kf = min(0.5, r / (2.0 * h0))
            fc = threshold(kf * h, h, 0.5 * r, 0.5 * r + 2.5 * h0, surfaces=self._fillet_faces())
        fh = threshold(0.6 * h, h, 0.0, 2.0 * h0, surfaces=self._hole_faces())
        finish_sizing([fc, fh], h)

    # ---- analysis
    def qois(self) -> list[QoI]:
        return [QoI(name="max_vm_corner", type="max", field="von_mises", group=self.corner_group),
                QoI(name="load_displacement", type="mean", field="ux", group="load_face")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        half = self.reference_portion == "half"
        F = p["force"] * (0.5 if half else 1.0)
        bcs = [BC(type="fixed", group="bolt_holes"),
               BC(type="force", group="load_face", value=[-F, 0.0, 0.0])]
        if half:
            bcs.append(BC(type="symmetry", group="sym_mid"))
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"steel": material(p)},
            bcs=bcs,
            qois=self.qois(),
            meta={"problem": self.id, "reference_portion": "half (y <= W/2) with symmetry; force halved"
                  if half else "full"},
        )

    def expectations(self) -> dict[str, str]:
        return {"max_vm_corner": "diverge"} if self.sharp else {}


for _r in (0, 2, 5, 10):
    register(LBracket(id=f"l_bracket_r{_r}", fillet_radius=_r * MM))
