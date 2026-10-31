"""Cook's membrane: tapered plane-stress panel, clamped on one edge, shear load on the other.

Problem: R.D. Cook, "Improved two-dimensional finite element", J. Struct. Div. ASCE 100(ST9),
1851-1863 (1974); the standard compressible variant (E = 1, nu = 1/3, thickness 1, total shear 1)
as used e.g. by P.G. Bergan & C.A. Felippa, "A triangular membrane element with rotational degrees
of freedom", Comput. Methods Appl. Mech. Engrg. 50 (1985) 25-69. Not a NAFEMS test; it lives in this
suite as a classic published-reference membrane benchmark.

There is no closed-form solution: the reference is the converged (fine-mesh) vertical displacement
at C = (48, 52), the MIDPOINT of the loaded edge (23.96; the classic Q4 table 18.30 / 22.08 / 23.43
for 4x4 / 8x8 / 16x16 meshes is reproduced exactly at this point, not at the top corner (48, 60),
whose displacement converges to ~25.2). Stresses are singular at the clamped corners (0, 0) and
(0, 44), which limits the observed convergence order.
"""

from __future__ import annotations

from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.geometry.gmsh_tools import distance_threshold_field, use_min_field
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import entities_where, group, near, point_tag
from feacore.problems.registry import register

CORNERS = [(0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)]


class CooksMembrane(Problem):
    id = "cooks_membrane"
    title = "Cook's membrane: tapered panel under end shear (plane stress)"
    family = "cooks_membrane"
    tags = ("nafems", "published_benchmark", "2d", "plane_stress", "bending", "corner_singularity")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    units = "consistent, dimensionless (E = 1)"
    expected_behavior = "singular"
    default_params = {"h0": 4.0}

    def description(self) -> str:
        return (
            "Cook's membrane (classic plane-stress benchmark; linear elastic, static). All quantities "
            "are in one consistent set of units (dimensionless).\n"
            "Geometry: a tapered quadrilateral panel in the xy-plane with corners (0, 0), (48, 44), "
            "(48, 60) and (0, 44). The left edge x = 0 runs from (0, 0) to (0, 44); the right edge "
            "x = 48 runs from (48, 44) to (48, 60); the bottom and top edges are the straight lines "
            "joining (0, 0)-(48, 44) and (0, 44)-(48, 60). Thickness 1. The STEP file holds this panel "
            "in the plane z = 0.\n"
            "Material: isotropic linear elastic, E = 1, Poisson's ratio 1/3; plane stress.\n"
            "Supports: the left edge (x = 0) is fully clamped (both displacement components zero).\n"
            "Load: a shear force of total magnitude 1 in the +y direction, uniformly distributed along "
            "the right edge (x = 48, length 16, i.e. 1/16 per unit length). The top and bottom edges "
            "are free.\n"
            "Report: uy_C = vertical (y) displacement at point C = (48, 52), the midpoint of the loaded "
            "right edge."
        )

    def build_geometry(self) -> None:
        import gmsh

        occ = gmsh.model.occ
        pts = [occ.addPoint(x, y, 0.0) for x, y in CORNERS]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.synchronize()
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "clamped")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x0, 48.0)), "loaded")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y0, 0.0) and near(x1, 48.0)), "bottom")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y1, 60.0) and near(x0, 0.0)), "top")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "panel")

    def apply_sizing(self, h: float) -> None:
        # grade towards the singular clamped corners; distances fixed, sizes scale with h
        corners = [point_tag(0.0, 0.0), point_tag(0.0, 44.0)]
        f = distance_threshold_field(points=corners, size_min=h / 8, size_max=h, dist_min=0.5, dist_max=16.0)
        use_min_field([f], h_max=h)

    def qois(self) -> list[QoI]:
        return [QoI(name="uy_C", type="point", field="uy", point=[48.0, 52.0])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_stress", order=order, thickness=1.0),
            materials={"mat": Material(E=1.0, nu=1.0 / 3.0)},
            bcs=[
                BC(type="fixed", group="clamped"),
                BC(type="force", group="loaded", value=[0.0, 1.0]),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "units": "dimensionless"},
        )

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="uy_C", value=23.96, kind="converged_numerical", tolerance=0.01,
            source=("Cook (1974) J. Struct. Div. ASCE 100(ST9):1851-1863 (problem); converged vertical "
                    "displacement at the mid-point of the loaded edge 23.96 (64x64 incompatible Q4), 23.92 "
                    "(64x64 compatible Q4), best value 23.9 per Bergan & Felippa (1985) CMAME 50:25-69, as "
                    "reproduced in ENERCALC 3D verification C-04 "
                    "(https://media.enercalc.com/ec3d_verification/c-04-(cook-membrane-problem).htm)"),
            notes="No closed form. Literature values 23.9 (Bergan & Felippa), 23.91 (smoothed-FEM papers), "
                  "23.92-23.96 (64x64 meshes); 1% tolerance covers the spread. Checked here: structured "
                  "compatible Q4 meshes reproduce the published 4x4/8x8/16x16/64x64 values (18.29, 22.08, "
                  "23.43, 23.92) at (48, 52); Q9 32x32 gives 23.961.",
        )]


register(CooksMembrane())
