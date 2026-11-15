"""NAFEMS T3: one-dimensional transient heat transfer.

Definition (NAFEMS, "The Standard NAFEMS Benchmarks", TNSB Rev. 3, 1990, test T3), as reproduced in public
verification manuals:
  * Abaqus Benchmarks Manual, "T3: One-dimensional transient heat transfer" (NAFEMS T3): a bar 0.1 m long,
    k = 35.0 W/(m degC), cp = 440.5 J/(kg degC), rho = 7200 kg/m^3, initially at 0 degC; end A (x = 0)
    held at 0 degC, end B (x = 0.1 m) at 100 sin(pi t / 40) degC; target T = 36.60 degC at x = 0.08 m,
    t = 32 s (https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-t3.htm).
  * Altair OptiStruct verification OS-V: 0110 "One Dimensional Transient Heat Transfer" (same data and
    target; 5- and 10-element meshes):
    https://help.altair.com/hwsolvers/os/topics/solvers/os/nafems_test_problem_t3_r.htm
  * MOOSE heat-transfer module, "NAFEMS Benchmark 1D Transient Heat Conduction T3":
    https://mooseframework.inl.gov/modules/heat_transfer/nafems_t3_verif.html
The NAFEMS booklet itself is not public. The target agrees with the exact solution (Duhamel's theorem,
``_series.slab_face_history``): 36.603 degC.
"""

from __future__ import annotations

from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import TNSB, entities_where, group, near
from feacore.problems.registry import register

L, W = 0.1, 0.01                   # bar length; the 2D strip's width (the sides are insulated)
K, CP, RHO = 35.0, 440.5, 7200.0
X_PROBE, T_PROBE, T_END = 0.08, 32.0, 32.0
END_B = "100.0 * sin(pi * t / 40.0)"


class NafemsT3(Problem):
    id = "nafems_t3"
    title = "NAFEMS T3: 1D transient heat transfer"
    family = "nafems_t3"
    tags = ("nafems", "2d", "heat", "transient")
    dim = 2
    physics = "heat"
    formulation = "2d"
    units = "SI (m, s, W, J, kg, degC)"
    expected_behavior = "smooth"
    default_params = {"h0": 0.005, "n_steps": 64}

    def description(self) -> str:
        return (
            "NAFEMS benchmark T3, one-dimensional transient heat conduction.\n"
            "Geometry (metres; the STEP file holds it in the plane z = 0): a bar 0 <= x <= 0.1, modelled as the "
            "strip 0 <= x <= 0.1, 0 <= y <= 0.01 with insulated long sides (y = 0 and y = 0.01), so the heat "
            "flows along x only.\n"
            "Material: k = 35.0 W/(m degC), specific heat cp = 440.5 J/(kg degC), density rho = 7200 kg/m^3.\n"
            "Initial state: the whole bar at 0 degC at t = 0.\n"
            "Boundary conditions: end A (x = 0) held at 0 degC; end B (x = 0.1) held at "
            "T_B(t) = 100 sin(pi t / 40) degC (t in seconds). No heat generation.\n"
            "Analysis: transient from t = 0 to t = 32 s.\n"
            "Report: T_x08 = temperature at x = 0.08 m at t = 32 s, in degC."
        )

    def build_geometry(self) -> None:
        import gmsh

        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, W)
        gmsh.model.occ.synchronize()
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "end_a")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x0, L)), "end_b")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0) or near(y0, W)), "sides")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "bar")

    def qois(self) -> list[QoI]:
        return [QoI(name="T_x08", type="point", field="T", point=[X_PROBE, 0.5 * W], time=T_PROBE)]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="heat", formulation="2d", order=order, options={"transient": {
                "t_end": T_END, "n_steps": int(self.params["n_steps"]), "scheme": "crank_nicolson",
                "initial": {"T": 0.0}}}),
            materials={"bar": Material(k=K, rho=RHO, cp=CP)},
            bcs=[BC(type="temperature", group="end_a", value=0.0),
                 BC(type="temperature", group="end_b", value=END_B)],
            qois=self.qois(),
            meta={"problem": self.id, "units": self.units},
        )

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="T_x08", value=36.60, kind="published", tolerance=0.01,
            source=(f"{TNSB}, test T3, target T = 36.60 degC at x = 0.08 m, t = 32 s; reproduced in the Abaqus "
                    "Benchmarks Manual 'T3: One-dimensional transient heat transfer' (https://docs.software.vt.edu/"
                    "abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-t3.htm) and Altair OptiStruct OS-V: 0110"),
            notes=("Exact (Duhamel series, feacore/problems/transient_heat/_series.py): 36.603 degC. The 1 % "
                   "tolerance is the usual NAFEMS acceptance; coarse meshes and large steps of the published "
                   "solutions fall 1-8 % low."),
        )]


register(NafemsT3())
