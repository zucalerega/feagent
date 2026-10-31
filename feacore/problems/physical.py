"""Physical-test problems: printed parts set up the way they are tested on the Instron, so the FEA of
the loop can be compared with measured data (ace/evidence.py).

abc_00000003_pull
    The gold part abc_00000003 (ABC dataset, rescaled to a 100 mm bounding-box diagonal): a tube
    along z, closed at the top, with two opposed stubs on bosses along x. Both stubs are lengthened
    by 40 mm (same diameter and axis) so wedge grips can hold them; the tube, bosses and original
    stubs are the gold geometry. The print files of the same part come from
    scripts/physical_parts.py (branch physical-validation): gold STEP + a 40 mm cylinder on each
    stub end face (OCC faces 15 and 17).
    Test: printed in Formlabs Rigid 10K (TDS: E = 10 GPa UV-cured; Poisson's ratio not given, 0.35
    assumed), the -x stub in the fixed wedge grip over the added 40 mm, the +x stub in the moving
    grip, pulled along +x. A video extensometer tracks two marks on the front of the original stubs,
    x = -27 and +27 mm (a 54 mm gauge length between the bosses and the grips).
    Scored: ext_marks = ux(B) - ux(A) between the marks (a point_difference QoI). Its reference is a
    mesh study of this setup (REFERENCES below); ace.evidence compares the loop with the measured
    value once the test exists. Signal: about 1.58e-7 m per newton (47 um at the nominal 300 N):
    use the video extensometer, the crosshead cannot resolve it.
    Marks on the bosses (x = +-16 mm) were rejected: they sit 3.5 mm from the sharp re-entrant edge
    where each boss shoulder meets its stub, and converge erratically (GCI and power-law fit 4 %
    apart). That edge is also the stress peak away from the grips (21.7 MPa at 500 N on a 1.8 mm
    mesh and still rising: singular in the CAD, finite on the print): keep the pull at or below
    the nominal 300 N for the elastic runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.problems.base import Problem, Reference
from feacore.problems.registry import register

MM = 1e-3

# Mesh study of the default setup: P2 tets, uniform h = 5, 3.5, 2.5, 1.8 mm (6.6k-66k nodes), ace.fea
# meshing, skfem. ext_marks at 500 N: 7.1674e-5, 7.4965e-5, 7.6703e-5, 7.7654e-5 m. Richardson on the
# three finest (gci3): p = 1.74, extrapolated 7.8889e-5, fine-grid GCI 2.0 %; the 4-mesh power-law fit
# gives 7.8949e-5 (p = 1.69), 0.08 % away. Linear in the load, so at the nominal 300 N: 4.7333e-5 m.
REFERENCES = {
    "abc_00000003_pull": {
        "value": 7.8889e-5 * 300.0 / 500.0, "tolerance": 0.03,
        "source": "feacore mesh study (P2, h = 5 to 1.8 mm), Richardson extrapolation of the three finest "
                  "meshes; the 4-mesh power-law fit agrees to 0.08 %",
        "notes": "for the default parameters only (300 N, E = 10 GPa, nu = 0.35, marks at x = +-27 mm)",
    },
}


def _gold_step(gold_set: str) -> Path:
    from feacore.problems.gold import gold_root
    return gold_root() / gold_set / "public" / "geometry.step"


class GoldPullTest(Problem):
    """A gold part with its stubs lengthened for wedge grips, pulled along the stub axis."""

    dim = 3
    physics = "elasticity"
    formulation = "3d"
    units = "SI (m, N, Pa)"
    expected_behavior = "smooth"
    tags = ("physical", "3d", "pull_test", "printed")      # not "gold": that tag means an imported gold set
    family = "physical"
    default_params = {
        "gold_set": "abc_00000003_bc0",       # the gold set whose STEP holds the part
        "stub_end_faces": (15, 17),           # OCC tags of the stub end faces in that STEP
        "extension": 40 * MM,                 # stub length added per side, all of it in the grip
        "force": 300.0,                       # total pull of the moving grip (N)
        "E": 10e9,                            # Rigid 10K, UV-cured (Formlabs TDS 2001479)
        "nu": 0.35,                           # not on the TDS: assumed
        "mark_neg": [-27 * MM, 7.656 * MM, 30.588 * MM],   # extensometer marks, front of the stubs
        "mark_pos": [27 * MM, 7.656 * MM, 30.588 * MM],
        "h0": 4 * MM,
    }
    title = "Printed gold part abc_00000003, pulled by its lengthened stubs (Instron)"

    # ------------------------------------------------------------- geometry
    def build_geometry(self) -> None:
        p = self.params
        gmsh.model.occ.importShapes(str(_gold_step(p["gold_set"])))
        gmsh.model.occ.synchronize()
        (vol,) = gmsh.model.getEntities(3)
        ends, tools = [], []
        for face in p["stub_end_faces"]:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, face)
            xc, yc, zc, r = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (y1 - y0) / 2
            ends.append((xc, yc, zc, r))
            tools.append((3, gmsh.model.occ.addCylinder(xc, yc, zc, math.copysign(p["extension"], xc), 0, 0, r)))
        gmsh.model.occ.fuse([vol], tools)
        gmsh.model.occ.synchronize()
        eps = 1e-6
        for xc, yc, zc, r in ends:
            side = "pos" if xc > 0 else "neg"
            xa, xb = sorted((xc, xc + math.copysign(p["extension"], xc)))
            lateral, end = [], []
            for _, f in gmsh.model.getEntitiesInBoundingBox(xa - eps, yc - r - eps, zc - r - eps,
                                                            xb + eps, yc + r + eps, zc + r + eps, 2):
                b = gmsh.model.getBoundingBox(2, f)
                if b[3] - b[0] < eps:          # planar face normal to x: the new stub end
                    if abs(b[0] - (xb if side == "pos" else xa)) < eps:
                        end.append(f)
                else:
                    lateral.append(f)
            gmsh.model.addPhysicalGroup(2, lateral, name=f"stub_{side}_grip")
            gmsh.model.addPhysicalGroup(2, end, name=f"stub_{side}_end")
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")

    # ------------------------------------------------------------- statement
    def description(self) -> str:
        p = self.params
        a, b = ([round(v / MM, 3) for v in p[k]] for k in ("mark_neg", "mark_pos"))
        ext = p["extension"] / MM
        return (
            "Printed pull test (Instron) of the gold part abc_00000003 with lengthened stubs "
            "(3D linear elastic solid, static).\n"
            "Geometry (the STEP file holds it in metres): the gold part abc_00000003, a tube along z, "
            "closed at its top, with two opposed stubs of diameter 15.3 mm on bosses along x, axis at "
            f"y = 0, z = 30.59 mm. Both stubs are lengthened by {ext:g} mm (same diameter and axis), so the "
            f"part spans x from -{34.85 + ext:g} to {34.85 + ext:g} mm.\n"
            f"Material: Formlabs Rigid 10K, linear elastic and isotropic, E = {p['E'] / 1e9:g} GPa, Poisson's "
            f"ratio {p['nu']:g}.\n"
            f"Supports: the added {ext:g} mm of the -x stub (its cylindrical surface, x from -{34.85 + ext:g} to "
            "-34.85 mm) is held in the fixed wedge grip: all displacement components zero on that surface.\n"
            f"Load: the added {ext:g} mm of the +x stub (cylindrical surface, x from 34.85 to {34.85 + ext:g} mm) "
            f"is held in the moving grip, which pulls it along +x with a total force of {p['force']:g} N, spread "
            "uniformly over that surface, and holds it against moving in y and z (zero y- and z-displacement "
            "there). The stub end faces and the rest of the part are free. No gravity.\n"
            f"Report: ext_marks = ux(B) - ux(A), the x-elongation between the video-extensometer marks "
            f"A = ({a[0]:g}, {a[1]:g}, {a[2]:g}) mm and B = ({b[0]:g}, {b[1]:g}, {b[2]:g}) mm on the front of the "
            "original stubs, in m."
        )

    def qois(self) -> list[QoI]:
        p = self.params
        return [QoI(name="ext_marks", type="point_difference", field="ux",
                    point=list(p["mark_pos"]), point2=list(p["mark_neg"])),
                QoI(name="ux_grip_pos", type="mean", field="ux", group="stub_pos_grip"),
                QoI(name="strain_energy", type="strain_energy")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"rigid10k": Material(E=p["E"], nu=p["nu"])},
            bcs=[BC(type="fixed", group="stub_neg_grip"),
                 BC(type="displacement", group="stub_pos_grip", components=[1, 2], value=0.0),
                 BC(type="force", group="stub_pos_grip", value=[p["force"], 0.0, 0.0])],
            qois=self.qois(),
            meta={"problem": self.id, "test": "Instron pull, wedge grips over the added stub length"},
        )

    def references(self) -> list[Reference]:
        ref = REFERENCES.get(self.id)
        if ref is None:
            return []
        return [Reference(qoi="ext_marks", value=ref["value"], kind="converged_numerical",
                          source=ref["source"], tolerance=ref["tolerance"], notes=ref.get("notes", ""))]


register(GoldPullTest(id="abc_00000003_pull"))


# --------------------------------------------------------------------------- platen compression

class GoldPlatenCompression(Problem):
    """A gold part pressed between the platens of a testing machine: faces pushed by the upper
    platen with a total force, faces on the lower platen held. No reference solution: the physical
    test (ace.evidence) is the only check, so the loop scores runs against the test.

    abc_00000065_compression
        The gold part abc_00000065 (a clamp body: a block with a bore and two prongs above it,
        60.7 x 77.8 x 16.2 mm), printed as is in Formlabs Alpha Fast resin (Form Studio 1, post-cured
        2 min) and compressed on the Instron: the upper platen pushes the two prong tops (faces f17
        and f34, plane y = 33.84 mm), the bottom face (f33, plane y = -43.96 mm) sits on the lower
        platen. Material from the Alpha Fast TDS, default post-cure: tensile modulus 2230 MPa (ASTM
        D638 Type 1), flexural 2110 MPa; Poisson's ratio not given (0.35 assumed).
        Scored: u_top, the mean y-displacement of the prong tops (negative under compression).
    """

    dim = 3
    physics = "elasticity"
    formulation = "3d"
    units = "SI (m, N, Pa)"
    expected_behavior = "smooth"
    tags = ("physical", "3d", "compression_test", "printed")
    family = "physical"
    default_params = {
        "gold_set": "abc_00000065_bc0",       # the gold set whose STEP holds the part
        "top_faces": (17, 34),                # pushed by the upper platen: the prong tops
        "bottom_faces": (33,),                # on the lower platen: the bottom face
        "force": 1000.0,                      # total compression (N)
        "E": 2.23e9,                          # Alpha Fast, default post-cure (TDS, ASTM D638 Type 1)
        "nu": 0.35,                           # not on the TDS: assumed
        "h0": 3 * MM,
    }
    title = "Printed gold part abc_00000065, compressed between platens (Instron)"

    def build_geometry(self) -> None:
        p = self.params
        gmsh.model.occ.importShapes(str(_gold_step(p["gold_set"])))
        gmsh.model.occ.synchronize()
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.addPhysicalGroup(2, [tag], name=f"f{tag}")       # every face, as the gold sets
        gmsh.model.addPhysicalGroup(2, list(p["top_faces"]), name="prong_tops")
        gmsh.model.addPhysicalGroup(2, list(p["bottom_faces"]), name="base_bottom")
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")

    def description(self) -> str:
        p = self.params
        tops = " and ".join(f"f{t}" for t in p["top_faces"])
        return (
            "Printed compression test (Instron) of the gold part abc_00000065 (3D linear elastic solid, static).\n"
            "Geometry (the STEP file holds it in metres; faces are named f<tag> by their face tag): the gold "
            "part abc_00000065 as printed, a clamp body 60.7 x 77.8 x 16.2 mm: a block with a bore of radius "
            "19.67 mm along z and two prongs above it. The prongs' top faces (" + tops + ", on the plane "
            "y = 33.84 mm; group prong_tops) carry the load; the bottom face (f33, on the plane y = -43.96 mm; "
            "group base_bottom) rests on the machine.\n"
            f"Material: Formlabs Alpha Fast resin, printed on a Form Studio 1 and post-cured 2 min; linear elastic "
            f"and isotropic, E = {p['E'] / 1e9:g} GPa (datasheet tensile modulus, default post-cure), Poisson's "
            f"ratio {p['nu']:g} (not on the datasheet: assumed).\n"
            "Supports: the bottom face f33 (group base_bottom) sits on the lower compression platen: all "
            "displacement components zero on it.\n"
            f"Load: the upper compression platen pushes the two prong tops (group prong_tops) down, along -y, "
            f"with a total force of {p['force']:g} N, spread uniformly over both faces. Everything else is free. "
            "No gravity.\n"
            "Report: u_top = the mean y-displacement of the prong tops (group prong_tops), in m (negative: they "
            "move down).\n"
            "There is no reference solution: the physical test of this part is the only check."
        )

    def qois(self) -> list[QoI]:
        return [QoI(name="u_top", type="mean", field="uy", group="prong_tops"),
                QoI(name="reaction_y", type="reaction", group="base_bottom", component=1),
                QoI(name="strain_energy", type="strain_energy")]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"alpha_fast": Material(E=p["E"], nu=p["nu"])},
            bcs=[BC(type="fixed", group="base_bottom"),
                 BC(type="force", group="prong_tops", value=[0.0, -p["force"], 0.0])],
            qois=self.qois(),
            meta={"problem": self.id, "test": "Instron compression between flat platens"},
        )

    def references(self) -> list[Reference]:
        return []          # no ground truth: the loop scores against the physical test (ace.evidence)


register(GoldPlatenCompression(id="abc_00000065_compression"))
