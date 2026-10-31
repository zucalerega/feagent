"""Bolted mounting bracket (in the spirit of the GE jet-engine bracket challenge): a base plate
with four bolt holes, an upright web with a rounded top and a load (pin) hole, and a triangular
stiffening gusset behind the web, all junctions filleted. Aluminium 6061-T6.

Load cases (separate ids, same part): vertical pull-off (+z), horizontal (+y, out of the web
plane, pulling the web away from the gusset), and 45 degrees in the y-z plane. The load is a
total force spread uniformly (per area) over the load-hole bore (the ``force`` BC). The four
bolt-hole bores are fixed. The reference mesh is the half x >= 0 with a symmetry plane (the
part and all three loads are symmetric about x = 0); the STEP is the full bracket.

QoIs: max von Mises on the fillet surfaces (away from the fixed bores, whose edges are weakly
singular) and the mean displacement magnitude of the load-hole bore. No handbook reference;
the GCI consensus is the check.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, QoI
from feacore.problems.parts._common import (
    AL6061, MM, PartProblem, add_group, com, entities_on_plane, finish_sizing, fmt_mm, material,
    threshold,
)
from feacore.problems.registry import register

LOAD_CASES = {
    # name: (unit direction (x, y, z), magnitude [N], words)
    "vertical": ((0.0, 0.0, 1.0), 8000.0, "vertically upward (+z), pulling the bracket off its base"),
    "horizontal": ((0.0, 1.0, 0.0), 3000.0,
                   "horizontally in +y, normal to the web, pulling the web away from the gusset"),
    "diag45": ((0.0, math.sqrt(0.5), math.sqrt(0.5)), 5000.0,
               "at 45 degrees in the y-z plane, direction (0, 1, 1)/sqrt(2) (up and away from the gusset)"),
}


class MountingBracket(PartProblem):
    family = "mounting_bracket"
    dim = 3
    physics = "elasticity"
    formulation = "3d"
    expected_behavior = "stress_concentration"
    reference_portion = "half"
    default_params = {
        "base_x": 100 * MM, "base_y": 70 * MM, "base_t": 8 * MM,
        "bolt_d": 9 * MM, "bolt_x": 38 * MM, "bolt_y": 25 * MM,      # holes at (+-bolt_x, +-bolt_y)
        "web_w": 50 * MM, "web_t": 8 * MM, "web_h": 42 * MM,        # straight height above the base
        "load_hole_d": 14 * MM,                                      # at the centre of the rounded top
        "gusset_t": 8 * MM, "gusset_len": 30 * MM, "gusset_h": 30 * MM,
        "fillet_r": 4 * MM,
        "load_case": "vertical",
        "E": AL6061["E"], "nu": AL6061["nu"],
        "h0": 6 * MM,
    }

    def __init__(self, id: str | None = None, **params):
        super().__init__(id, **params)
        if self.params["load_case"] not in LOAD_CASES:
            raise ValueError(f"unknown load_case {self.params['load_case']!r}")
        self.tags = ("part", "3d", "bolted", "fillet", "gusset", "bracket", "multi_load_case",
                     "stress_concentration", f"load_{self.params['load_case']}")
        self.title = f"Bolted mounting bracket with gusset, {self.params['load_case']} load"

    # ---- derived
    @property
    def z_hole(self) -> float:
        p = self.params
        return p["base_t"] + p["web_h"]

    def force_vector(self) -> list[float]:
        u, F, _ = LOAD_CASES[self.params["load_case"]]
        return [F * c for c in u]

    def description(self) -> str:
        p = self.params
        u, F, words = LOAD_CASES[p["load_case"]]
        bx, by, bt = p["base_x"], p["base_y"], p["base_t"]
        ww, wt, zh = p["web_w"], p["web_t"], self.z_hole
        return (
            "Bolted aluminium mounting bracket with an upright web and a stiffening gusset. 3D solid.\n"
            "Geometry file (STEP): the full bracket in mm-scale dimensions written in metres.\n"
            f"Base plate: {fmt_mm(bx)} x {fmt_mm(by)} x {fmt_mm(bt)} (x from {fmt_mm(-bx / 2)} to "
            f"{fmt_mm(bx / 2)}, y from {fmt_mm(-by / 2)} to {fmt_mm(by / 2)}, z from 0 to {fmt_mm(bt)}), "
            f"with four through bolt holes of diameter {fmt_mm(p['bolt_d'])} at (x, y) = "
            f"(+-{fmt_mm(p['bolt_x'])}, +-{fmt_mm(p['bolt_y'])}).\n"
            f"Upright web: a {fmt_mm(wt)} thick plate in the x-z plane (y from {fmt_mm(-wt / 2)} to "
            f"{fmt_mm(wt / 2)}), {fmt_mm(ww)} wide (x from {fmt_mm(-ww / 2)} to {fmt_mm(ww / 2)}), rising "
            f"from the base to a semicircular top of radius {fmt_mm(ww / 2)} centred at (x, z) = (0, "
            f"{fmt_mm(zh)}). The load hole, diameter {fmt_mm(p['load_hole_d'])}, passes through the web "
            f"along y at that centre.\n"
            f"Gusset: a triangular rib {fmt_mm(p['gusset_t'])} thick (x from {fmt_mm(-p['gusset_t'] / 2)} "
            f"to {fmt_mm(p['gusset_t'] / 2)}) behind the web on the -y side, joining the base over "
            f"{fmt_mm(p['gusset_len'])} in y and the web over {fmt_mm(p['gusset_h'])} in z. All "
            f"web/base, gusset/base and gusset/web junctions are blended by fillets of radius "
            f"{fmt_mm(p['fillet_r'])}.\n"
            f"Material: aluminium 6061-T6, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, "
            f"Poisson's ratio {p['nu']:g}.\n"
            "Supports: the four bolt-hole bores are fixed (all displacement components zero), "
            "representing the bolts. The underside of the base is otherwise free (no contact).\n"
            f"Load ({p['load_case']}): a total force of {F / 1e3:g} kN acting {words}, applied to the "
            "bracket through the load-hole bore as a uniform traction (the total force divided by the "
            "bore area, the same vector at every point of the bore).\n"
            "Analysis: linear static, small displacements.\n"
            "Report:\n"
            "  max_vm_fillets: the maximum von Mises stress on the fillet surfaces (the blends at the "
            "web/base, gusset/base and gusset/web junctions) [Pa];\n"
            "  load_displacement: the average displacement magnitude |u| over the load-hole bore [m]."
        )

    # ---- geometry
    def _concave_edges(self) -> list[int]:
        p = self.params
        bt, wt, ww, gt = p["base_t"], p["web_t"], p["web_w"], p["gusset_t"]
        gl, gh = p["gusset_len"], p["gusset_h"]
        e = 1e-6
        out = set()
        for _, c in gmsh.model.getEntities(1):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, c)
            on_base = abs(z0 - bt) < e and abs(z1 - bt) < e
            if on_base and abs(y0 - y1) < e and abs(abs(y0) - wt / 2) < e and x0 > -ww / 2 - e and x1 < ww / 2 + e:
                out.add(c)  # web / base
            if on_base and abs(x0 - x1) < e and abs(abs(x0) - gt / 2) < e and y0 > -wt / 2 - gl - e and y1 < -wt / 2 + e:
                out.add(c)  # gusset / base
            if abs(x0 - x1) < e and abs(abs(x0) - gt / 2) < e and abs(y0 + wt / 2) < e and abs(y1 + wt / 2) < e \
                    and z0 > bt - e and z1 < bt + gh + e:
                out.add(c)  # gusset / web
        return sorted(out)

    def _shape(self, full: bool) -> None:
        occ = gmsh.model.occ
        p = self.params
        bx, by, bt = p["base_x"], p["base_y"], p["base_t"]
        ww, wt, wh = p["web_w"], p["web_t"], p["web_h"]
        gt, gl, gh = p["gusset_t"], p["gusset_len"], p["gusset_h"]
        zh = self.z_hole
        base = occ.addBox(-bx / 2, -by / 2, 0, bx, by, bt)
        web = occ.addBox(-ww / 2, -wt / 2, bt, ww, wt, wh)
        top = occ.addCylinder(0, -wt / 2, zh, 0, wt, 0, ww / 2)
        # gusset: triangle in the plane x = -gt/2, extruded by gt along +x
        q1 = occ.addPoint(-gt / 2, -wt / 2, bt)
        q2 = occ.addPoint(-gt / 2, -wt / 2 - gl, bt)
        q3 = occ.addPoint(-gt / 2, -wt / 2, bt + gh)
        tri = occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(q1, q2), occ.addLine(q2, q3),
                                                     occ.addLine(q3, q1)])])
        gus = [t for d, t in occ.extrude([(2, tri)], gt, 0, 0) if d == 3]
        out, _ = occ.fuse([(3, base)], [(3, web), (3, top)] + [(3, t) for t in gus])
        occ.synchronize()
        vol = out[0][1]
        edges = self._concave_edges()
        out = occ.fillet([vol], edges, [p["fillet_r"]])
        vol = out[0][1]
        holes = [(3, occ.addCylinder(sx * p["bolt_x"], sy * p["bolt_y"], -1 * MM, 0, 0, bt + 2 * MM,
                                     p["bolt_d"] / 2)) for sx in (-1, 1) for sy in (-1, 1)]
        holes.append((3, occ.addCylinder(0, -wt / 2 - 1 * MM, zh, 0, wt + 2 * MM, 0, p["load_hole_d"] / 2)))
        out, _ = occ.cut([(3, vol)], holes)
        vol = out[0][1]
        if not full:
            box = occ.addBox(0, -by, -1 * MM, bx, 2 * by, zh + ww)
            out, _ = occ.intersect([(3, vol)], [(3, box)])
            vol = out[0][1]
        occ.synchronize()
        self._vol = vol

    def _bolt_faces(self) -> list[int]:
        p = self.params
        out = []
        for _, t in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, t) != "Cylinder":
                continue
            x, y, z = com(2, t)
            if z < p["base_t"] and abs(abs(x) - p["bolt_x"]) < 2 * MM and abs(abs(y) - p["bolt_y"]) < 2 * MM:
                out.append(t)
        return out

    def _load_hole_faces(self) -> list[int]:
        p = self.params
        out = []
        for _, t in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, t) != "Cylinder":
                continue
            x, y, z = com(2, t)
            if math.hypot(x, z - self.z_hole) < p["load_hole_d"] / 2 + 1e-6 and abs(y) < p["web_t"]:
                out.append(t)
        return out

    def _fillet_faces(self) -> list[int]:
        """Blend faces: non-planar faces that are neither hole bores nor the web's rounded top."""
        p = self.params
        skip = set(self._bolt_faces()) | set(self._load_hole_faces())
        out = []
        for _, t in gmsh.model.getEntities(2):
            if t in skip or gmsh.model.getType(2, t) == "Plane":
                continue
            x, y, z = com(2, t)
            if z < p["base_t"] + max(p["gusset_h"], p["fillet_r"]) + 1e-6 and abs(y) < p["web_t"] / 2 + p["gusset_len"] + 1e-6:
                out.append(t)
        return out

    def _groups(self, full: bool) -> None:
        add_group(2, self._bolt_faces(), "bolt_holes")
        add_group(2, self._load_hole_faces(), "load_hole")
        add_group(2, self._fillet_faces(), "fillets")
        if not full:
            add_group(2, entities_on_plane(2, 0, 0.0), "sym_x0")
        add_group(3, [self._vol], "bracket")

    def apply_sizing(self, h: float) -> None:
        h0 = self.params["h0"]
        ff = threshold(0.4 * h, h, 0.0, 2.0 * h0, surfaces=self._fillet_faces())
        fl = threshold(0.4 * h, h, 0.0, 2.0 * h0, surfaces=self._load_hole_faces())
        fb = threshold(0.6 * h, h, 0.0, 1.5 * h0, surfaces=self._bolt_faces())
        finish_sizing([ff, fl, fb], h)

    # ---- analysis
    def qois(self) -> list[QoI]:
        return [QoI(name="max_vm_fillets", type="max", field="von_mises", group="fillets"),
                QoI(name="load_displacement", type="mean", field="u_mag", group="load_hole")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        half = self.reference_portion == "half"
        F = [f * (0.5 if half else 1.0) for f in self.force_vector()]
        bcs = [BC(type="fixed", group="bolt_holes"), BC(type="force", group="load_hole", value=F)]
        if half:
            bcs.append(BC(type="symmetry", group="sym_x0"))
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"al6061": material(self.params)},
            bcs=bcs,
            qois=self.qois(),
            meta={"problem": self.id, "load_case": self.params["load_case"],
                  "reference_portion": "half (x >= 0) with symmetry; force halved" if half else "full"},
        )


for _lc in LOAD_CASES:
    register(MountingBracket(id=f"mounting_bracket_{_lc}", load_case=_lc))
