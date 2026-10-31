"""NAFEMS T4: two-dimensional steady heat conduction with convection.

Definition (NAFEMS TNSB Rev. 3, test T4), verified against public reproductions:
  * COMSOL Application Gallery "Steady-State 2D Heat Transfer with Conduction" (heat_convection_2d):
    0.6 m x 1.0 m, left edge insulated, lower edge 100 degC, upper and right edges convect to
    0 degC with h = 750 W/(m^2 degC); T(0.6, 0.2) = 18.25 degC (their 9x15 quadratic mesh: 18.265):
    https://doc.comsol.com/6.4/doc/com.comsol.help.models.mph.heat_convection_2d/heat_convection_2d.html
  * Abaqus Benchmarks Manual 4.3.4 "T4: Two-dimensional heat transfer with convection"
    (k = 52 W/(m degC), h = 750 W/(m^2 degC), target 18.3 degC at E).
  * Altair OptiStruct verification OS-V: 0120 "Two-Dimensional Heat Transfer with Convection".
"""

from __future__ import annotations

from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import TNSB, entities_where, group, near
from feacore.problems.registry import register

W, H = 0.6, 1.0
K, H_CONV, T_FIX, T_INF = 52.0, 750.0, 100.0, 0.0


class NafemsT4(Problem):
    id = "nafems_t4"
    title = "NAFEMS T4: 2D steady heat conduction with convection"
    family = "nafems_t4"
    tags = ("nafems", "2d", "heat", "steady", "convection")
    dim = 2
    physics = "heat"
    formulation = "2d"
    units = "SI (m, W, degC)"
    expected_behavior = "smooth"
    default_params = {"h0": 0.05}

    def description(self) -> str:
        return (
            "NAFEMS benchmark T4, two-dimensional steady-state heat conduction with convection.\n"
            "Geometry (metres; the STEP file holds it in the plane z = 0): the rectangle "
            "0 <= x <= 0.6, 0 <= y <= 1.0. Two-dimensional conduction (no variation through the "
            "thickness, no heat flow out of the plane).\n"
            "Material: isotropic thermal conductivity k = 52 W/(m degC). No internal heat generation.\n"
            "Boundary conditions: bottom edge (y = 0): prescribed temperature 100 degC. Left edge "
            "(x = 0): insulated (zero heat flux). Right edge (x = 0.6) and top edge (y = 1.0): "
            "convection to an ambient temperature of 0 degC with film coefficient "
            "h = 750 W/(m^2 degC).\n"
            "Report: T_E = temperature at point E = (0.6, 0.2) on the right edge, in degC."
        )

    def build_geometry(self) -> None:
        import gmsh

        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W, H)
        gmsh.model.occ.synchronize()
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0)), "bottom")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "left")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x0, W)), "right")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y0, H)), "top")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "plate")

    # Reference study on quasi-uniform meshes (default sizing): P2 converges cleanly at p ~ 3 at E.
    # (Grading towards the corner (0.6, 0), where the boundary condition jumps from fixed
    # temperature to convection, makes the already tiny changes non-monotone.)

    def qois(self) -> list[QoI]:
        return [QoI(name="T_E", type="point", field="T", point=[W, 0.2])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="heat", formulation="2d", order=order),
            materials={"solid": Material(k=K)},
            bcs=[
                BC(type="temperature", group="bottom", value=T_FIX),
                BC(type="convection", group="right", h=H_CONV, T_inf=T_INF),
                BC(type="convection", group="top", h=H_CONV, T_inf=T_INF),
            ],
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, W, degC)"},
        )

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="T_E", value=18.25, kind="published", tolerance=0.01,
            source=(f"{TNSB}, test T4, target T(E) = 18.25 degC at E = (0.6, 0.2); reproduced in COMSOL "
                    "'Steady-State 2D Heat Transfer with Conduction' (https://doc.comsol.com/6.4/doc/"
                    "com.comsol.help.models.mph.heat_convection_2d/heat_convection_2d.html) and the Abaqus "
                    "Benchmarks Manual 4.3.4 (T4, target 18.3 degC)"),
            notes="COMSOL's 9x15 quadratic mesh gives 18.265 degC.",
        )]


register(NafemsT4())
