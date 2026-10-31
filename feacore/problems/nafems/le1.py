"""NAFEMS LE1: elliptic membrane under outward pressure (plane stress).

Definition (NAFEMS TNSB Rev. 3, test LE1), verified against public reproductions:
  * FeenoX example ``examples/nafems-le1.geo`` / ``nafems-le1.fee`` (geometry, supports,
    10 MPa tension on BC, target 92.7 MPa): https://github.com/seamplex/feenox
  * COMSOL Application Gallery "Elliptic Membrane" (sigma_y at D = (2, 0): 92.7 MPa; their
    converged value 92.658 MPa): https://doc.comsol.com/6.4/doc/com.comsol.help.models.sme.elliptic_membrane/elliptic_membrane.html
  * Altair OptiStruct verification OS-V: 0010 (0.1 m thick plate, 10 MPa):
    https://help.altair.com/hwsolvers/os/topics/solvers/os/nafems_test_problem_le1_r.htm

Quarter model, x >= 0, y >= 0. A = (0, 1), B = (0, 2.75), C = (3.25, 0), D = (2, 0).
"""

from __future__ import annotations

from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.geometry.gmsh_tools import distance_threshold_field, use_min_field
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import (
    ELL_INNER, ELL_OUTER, STEEL, TNSB, entities_where, group, near, point_tag,
    quarter_elliptic_annulus,
)
from feacore.problems.registry import register

PRESSURE = 10e6      # Pa, outward (tensile) on the outer edge
THICKNESS = 0.1      # m


class NafemsLE1(Problem):
    id = "nafems_le1"
    title = "NAFEMS LE1: elliptic membrane (plane stress)"
    family = "nafems_le1"
    tags = ("nafems", "2d", "plane_stress", "stress_concentration", "curved_boundary", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "stress_concentration"
    default_params = {"h0": 0.2}

    def description(self) -> str:
        return (
            "NAFEMS benchmark LE1, elliptic membrane (linear elastic, plane stress, static).\n"
            "Geometry (metres; the STEP file holds this quarter in the plane z = 0): a flat plate of "
            f"thickness {THICKNESS} m whose plan is the quarter (x >= 0, y >= 0) of an elliptic annulus. "
            "Inner boundary: the ellipse (x/2.0)^2 + (y/1.0)^2 = 1, from A = (0, 1.0) to D = (2.0, 0). "
            "Outer boundary: the ellipse (x/3.25)^2 + (y/2.75)^2 = 1, from B = (0, 2.75) to C = (3.25, 0). "
            "Straight edges AB (on x = 0) and CD (on y = 0) lie on the symmetry planes of the full "
            "membrane.\n"
            "Material: isotropic linear elastic, E = 210 GPa, Poisson's ratio 0.3.\n"
            "Supports: edge AB (x = 0): zero x-displacement. Edge CD (y = 0): zero y-displacement.\n"
            "Load: uniform outward (tensile) normal traction of 10 MPa on the outer elliptic edge BC, "
            "i.e. the edge is pulled away from the body along its outward normal. The inner elliptic "
            "edge DA is traction-free. Plane stress state.\n"
            "Report: syy_D = direct stress sigma_yy at point D = (2.0, 0.0), in Pa."
        )

    # ------------------------------------------------------------------ geometry
    def build_geometry(self) -> None:
        import gmsh

        quarter_elliptic_annulus(0.0)
        gmsh.model.occ.synchronize()
        a, b = ELL_INNER
        c, d = ELL_OUTER
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "edge_AB")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0)), "edge_CD")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, c) and near(y1, d)), "edge_BC")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, a) and near(y1, b)), "edge_DA")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "membrane")

    def apply_sizing(self, h: float) -> None:
        # refine around the stress concentration at D; distances fixed, sizes scale with h
        d_pt = point_tag(ELL_INNER[0], 0.0)
        f = distance_threshold_field(points=[d_pt], size_min=h / 4, size_max=h, dist_min=0.15, dist_max=1.2)
        use_min_field([f], h_max=h)

    # ------------------------------------------------------------------ analysis
    def qois(self) -> list[QoI]:
        return [QoI(name="syy_D", type="point", field="syy", point=[2.0, 0.0])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_stress", order=order,
                              thickness=THICKNESS),
            materials={"steel": Material(**STEEL)},
            bcs=[
                BC(type="symmetry", group="edge_AB"),
                BC(type="symmetry", group="edge_CD"),
                BC(type="pressure", group="edge_BC", value=-PRESSURE),  # negative = tensile
            ],
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, N, Pa)"},
        )

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="syy_D", value=92.7e6, kind="published", tolerance=0.02,
            source=(f"{TNSB}, test LE1, target sigma_yy(D) = 92.7 MPa; reproduced in the FeenoX example "
                    "examples/nafems-le1.fee (https://github.com/seamplex/feenox) and COMSOL 'Elliptic "
                    "Membrane' (https://doc.comsol.com/6.4/doc/com.comsol.help.models.sme.elliptic_membrane/"
                    "elliptic_membrane.html)"),
            notes="Target quoted to 3 digits; COMSOL reports a mesh-converged 92.658 MPa.",
        )]


register(NafemsLE1())
