"""NAFEMS LE10: thick elliptic plate under uniform pressure (3D solid).

Definition (NAFEMS TNSB Rev. 3, test LE10, sheet dated 15-6-90/2), verified against:
  * the NAFEMS LE10 data sheet reproduced in the FeenoX repository,
    doc/nafems-le10-problem-input.svg (https://github.com/seamplex/feenox): geometry, 1 MPa on the
    upper surface, supports, output "direct stress sigma_yy at point D", target 5.38 MPa;
  * FeenoX example examples/nafems-le10.fee + nafems-le10-cad.geo (z = -0.3..0.3, mid-plane line of
    the outer face imprinted in the CAD, sigma_y at D = (2, 0, 0.3) = -5.380 MPa);
  * Fino case 012 https://www.seamplex.com/fino/cases/012-nafems-le10 (-5.38 MPa).

Point labels follow the NAFEMS sheet: A, B, C, D on the upper surface z = +0.3, primed points on
the lower surface. (The Abaqus benchmark manual calls the mid-plane points E / E'.)
"""

from __future__ import annotations

import math
from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.geometry.gmsh_tools import distance_threshold_field, use_min_field
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import (
    ELL_INNER, ELL_OUTER, LE10_SHEET, STEEL, TNSB, entities_where, group, near, point_tag,
    quarter_elliptic_annulus,
)
from feacore.problems.registry import register

PRESSURE = 1e6       # Pa on the upper surface, pushing into the plate
HALF_T = 0.3         # m, half thickness


class NafemsLE10(Problem):
    id = "nafems_le10"
    title = "NAFEMS LE10: thick plate under pressure (3D)"
    family = "nafems_le10"
    tags = ("nafems", "3d", "solid", "plate_bending", "curved_boundary", "symmetry", "line_support")
    dim = 3
    physics = "elasticity"
    formulation = "3d"
    expected_behavior = "stress_concentration"
    default_params = {"h0": 0.3}

    def description(self) -> str:
        return (
            "NAFEMS benchmark LE10, thick plate under pressure (3D linear elastic solid, static).\n"
            "Geometry (metres; the STEP file holds this quarter model): a plate of thickness 0.6 m, "
            "-0.3 <= z <= 0.3, whose plan (x >= 0, y >= 0) is a quarter of an elliptic annulus. Inner "
            "boundary (hole): the ellipse (x/2.0)^2 + (y/1.0)^2 = 1; outer boundary: the ellipse "
            "(x/3.25)^2 + (y/2.75)^2 = 1. Corner points on the upper surface z = +0.3: A = (0, 1.0, 0.3), "
            "B = (0, 2.75, 0.3), C = (3.25, 0, 0.3), D = (2.0, 0, 0.3); the primed points A', B', C', D' "
            "are the same (x, y) on the lower surface z = -0.3. Face ABA'B' lies on x = 0 and face "
            "DCD'C' on y = 0 (symmetry planes of the full plate). The outer curved face BCB'C' is split "
            "along its mid-plane line (the outer ellipse at z = 0); this edge is in the CAD.\n"
            "Material: isotropic linear elastic, E = 210 GPa, Poisson's ratio 0.3.\n"
            "Load: uniform normal pressure of 1 MPa on the upper surface (z = +0.3), pushing down into "
            "the plate (towards -z). All other faces carry no load.\n"
            "Supports: face DCD'C' (y = 0): zero y-displacement. Face ABA'B' (x = 0): zero "
            "x-displacement. Outer curved face BCB'C': zero x- and y-displacement on the whole face, and "
            "zero z-displacement only along its mid-plane line (outer ellipse at z = 0). The elliptic "
            "hole face and the lower surface are free.\n"
            "Report: syy_D = direct stress sigma_yy at point D = (2.0, 0.0, 0.3) on the upper surface, "
            "in Pa."
        )

    # ------------------------------------------------------------------ geometry
    def build_geometry(self) -> None:
        import gmsh

        occ = gmsh.model.occ
        a, b = ELL_INNER
        c, d = ELL_OUTER
        surf = quarter_elliptic_annulus(-HALF_T)
        ext = occ.extrude([(2, surf)], 0.0, 0.0, 2 * HALF_T)
        vol = [t for dim, t in ext if dim == 3]
        # imprint the mid-plane line of the outer face so it exists as an edge (support line)
        mid = occ.addEllipse(0.0, 0.0, 0.0, c, d, angle1=0.0, angle2=math.pi / 2)
        occ.fragment([(3, vol[0])], [(1, mid)])
        occ.synchronize()
        if len(gmsh.model.getEntities(3)) != 1:
            raise RuntimeError("LE10 geometry: expected a single volume")

        def faces(pred):
            return entities_where(2, pred)

        group(2, faces(lambda x0, y0, z0, x1, y1, z1: near(z0, HALF_T) and near(z1, HALF_T)), "face_upper")
        group(2, faces(lambda x0, y0, z0, x1, y1, z1: near(z0, -HALF_T) and near(z1, -HALF_T)), "face_lower")
        group(2, faces(lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "face_ABA'B'_x0")
        group(2, faces(lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0)), "face_DCD'C'_y0")
        # vertical extent excludes the upper / lower faces, whose bounding boxes also reach (c, d)
        outer = faces(lambda x0, y0, z0, x1, y1, z1: near(x1, c) and near(y1, d) and z1 - z0 > 0.1)
        if len(outer) != 2:
            raise RuntimeError(f"LE10 geometry: outer face should be split in two, got {outer}")
        group(2, outer, "face_BCB'C'_outer")
        group(2, faces(lambda x0, y0, z0, x1, y1, z1: near(x1, a) and near(y1, b) and z1 - z0 > 0.5),
              "face_hole")
        midline = entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(z0, 0.0) and near(z1, 0.0)
                                 and near(x1, c) and near(y1, d))
        group(1, midline, "line_outer_midplane")
        group(3, [t for _, t in gmsh.model.getEntities(3)], "plate")

    def apply_sizing(self, h: float) -> None:
        d_pt = point_tag(ELL_INNER[0], 0.0, HALF_T)
        f = distance_threshold_field(points=[d_pt], size_min=h / 3, size_max=h, dist_min=0.2, dist_max=1.5)
        use_min_field([f], h_max=h)

    # ------------------------------------------------------------------ analysis
    def qois(self) -> list[QoI]:
        return [QoI(name="syy_D", type="point", field="syy", point=[2.0, 0.0, HALF_T])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order),
            materials={"steel": Material(**STEEL)},
            bcs=[
                BC(type="pressure", group="face_upper", value=PRESSURE),
                BC(type="symmetry", group="face_DCD'C'_y0"),
                BC(type="symmetry", group="face_ABA'B'_x0"),
                BC(type="displacement", group="face_BCB'C'_outer", components=[0, 1], value=0.0),
                BC(type="displacement", group="line_outer_midplane", components=[2], value=0.0),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, N, Pa)"},
        )

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="syy_D", value=-5.38e6, kind="published", tolerance=0.02,
            source=(f"{TNSB}, test LE10, output sigma_yy at D, target 5.38 MPa (compressive); data sheet "
                    f"reproduced at {LE10_SHEET}; sign as in FeenoX examples/nafems-le10.fee (-5.380 MPa) and "
                    "https://www.seamplex.com/fino/cases/012-nafems-le10 (-5.38 MPa)"),
            notes="The NAFEMS sheet prints the magnitude 5.38 MPa; the stress at D (upper surface) is "
                  "compressive. Some task descriptions call this point E; NAFEMS calls it D.",
        )]


register(NafemsLE10())
