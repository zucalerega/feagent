"""Chill-down of a wall: a slab cooled on one face by a fluid whose temperature steps down at t = 0.

The model problem of a cryogenic valve's chill-down: a stainless-steel wall at room temperature, wetted on
one face by liquid nitrogen from t = 0, insulated on the other face. Constant properties and a constant film
coefficient (the real chill-down has strongly temperature-dependent cp, k and boiling regimes; this is the
linear model that a code has to get right first).

Exact solution: the slab with convection on one face and the other insulated is half of the symmetric slab
cooled on both faces (Carslaw & Jaeger, "Conduction of Heat in Solids", 2nd ed. (1959), sec. 3.11;
Incropera et al., "Fundamentals of Heat and Mass Transfer", 7th ed. (2011), sec. 5.5.1, eq. 5.39), with the
roots of z tan z = Bi computed numerically (``_series.slab_convection``).
"""

from __future__ import annotations

from pathlib import Path

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import entities_where, group, near
from feacore.problems.registry import register
from feacore.problems.transient_heat._series import slab_convection

THICK, HEIGHT = 0.01, 0.002                 # wall thickness (x); the strip's height (insulated sides)
K, RHO, CP = 16.0, 8000.0, 500.0            # austenitic stainless steel near room temperature
H, T_INIT, T_FLUID = 2000.0, 293.15, 77.4   # film coefficient; wall at 20 degC; liquid nitrogen
T_END = 30.0
ALPHA = K / (RHO * CP)
BIOT = H * THICK / K                        # 1.25
SOURCE = ("H.S. Carslaw & J.C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford (1959), sec. 3.11 "
          "(slab cooled by convection: series in the roots of z tan z = Bi); F.P. Incropera et al., Fundamentals "
          "of Heat and Mass Transfer, 7th ed. (2011), sec. 5.5.1, eq. 5.39")


def exact(x_from_wetted: float, t: float) -> float:
    """The exact temperature at a distance from the wetted face at the time t."""
    return float(slab_convection(THICK - x_from_wetted, t, THICK, ALPHA, BIOT, T_INIT, T_FLUID))


class ConvectiveSlab(Problem):
    id = "convective_slab_chilldown"
    title = "Wall chill-down: slab cooled by a fluid step (transient convection)"
    family = "convective_slab"
    tags = ("analytical", "2d", "heat", "transient", "convection", "cryogenic")
    dim = 2
    physics = "heat"
    formulation = "2d"
    units = "SI (m, s, W, J, kg, K)"
    expected_behavior = "smooth"
    default_params = {"h0": 0.001, "n_steps": 120}

    def description(self) -> str:
        return (
            "Chill-down of a wall: transient conduction through a stainless-steel wall wetted by liquid "
            "nitrogen on one face.\n"
            "Geometry (metres; the STEP file holds it in the plane z = 0): the wall thickness is the strip "
            "0 <= x <= 0.01; the strip's height 0 <= y <= 0.002 is a slice of a large wall, its sides y = 0 and "
            "y = 0.002 are insulated (no heat flows along the wall).\n"
            "Material (constant): k = 16 W/(m K), rho = 8000 kg/m^3, cp = 500 J/(kg K).\n"
            "Initial state: the whole wall at 293.15 K.\n"
            "Boundary conditions from t = 0: the wetted face x = 0 convects to liquid nitrogen at 77.4 K with "
            "a film coefficient h = 2000 W/(m^2 K); the outer face x = 0.01 is insulated.\n"
            "Analysis: transient from t = 0 to t = 30 s.\n"
            "Report: T_wet_10s = temperature of the wetted face (x = 0) at t = 10 s, and T_outer_30s = "
            "temperature of the outer face (x = 0.01) at t = 30 s, both in K."
        )

    def build_geometry(self) -> None:
        import gmsh

        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, THICK, HEIGHT)
        gmsh.model.occ.synchronize()
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x1, 0.0)), "wetted")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(x0, THICK)), "outer")
        group(1, entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y1, 0.0) or near(y0, HEIGHT)), "sides")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "wall")

    def qois(self) -> list[QoI]:
        return [QoI(name="T_wet_10s", type="point", field="T", point=[0.0, 0.5 * HEIGHT], time=10.0),
                QoI(name="T_outer_30s", type="point", field="T", point=[THICK, 0.5 * HEIGHT], time=30.0)]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="heat", formulation="2d", order=order, options={"transient": {
                "t_end": T_END, "n_steps": int(self.params["n_steps"]), "scheme": "crank_nicolson",
                "initial": {"T": T_INIT}}}),
            materials={"wall": Material(k=K, rho=RHO, cp=CP)},
            bcs=[BC(type="convection", group="wetted", h=H, T_inf=T_FLUID)],
            qois=self.qois(),
            meta={"problem": self.id, "units": self.units},
        )

    def exact(self, field, points):
        return None                       # a field in time; the references give the reported values

    def references(self) -> list[Reference]:
        tol = 0.002                       # 0.2 % of ~100-200 K: ~0.3 K, well above the reference mesh's error
        return [
            Reference(qoi="T_wet_10s", value=round(exact(0.0, 10.0), 4), kind="analytical", tolerance=tol,
                      source=SOURCE, notes=f"Bi = h L / k = {BIOT:g}, Fo = {ALPHA * 10.0 / THICK ** 2:g}; "
                                           "80 terms of the series"),
            Reference(qoi="T_outer_30s", value=round(exact(THICK, 30.0), 4), kind="analytical", tolerance=tol,
                      source=SOURCE, notes=f"Bi = {BIOT:g}, Fo = {ALPHA * 30.0 / THICK ** 2:g}; 80 terms"),
        ]


register(ConvectiveSlab())
