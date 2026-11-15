"""Transient dynamics benchmark: a step end load on a fixed-free bar (a wave travelling at sqrt(E / rho)).

The bar has nu = 0 and u_y = 0 on its long edges, so the continuum solution is exactly one-dimensional:
a stress front sigma travels from the loaded end at c = sqrt(E / rho), reflects at the fixed end (doubling)
and at the free end (cancelling). The end displacement is a triangle wave in time: it grows linearly
(u = sigma c t / E) to 2 sigma L / E at t = 2 L / c, falls back to 0 at 4 L / c, and repeats with the
period 4 L / c (the fundamental frequency c / (4 L)); its mean over a period is the static value sigma L / E.
Source: K.F. Graff, Wave Motion in Elastic Solids, Dover (1991), section 2.1 (the d'Alembert solution and
reflection at fixed and free ends).

The peak is a corner of the exact history, so a discrete solution reaches it from below and converges
slowly (P2, HHT alpha = -0.1, 50 steps per transit: -1.4 % at h = L/20, -0.9 % at L/40); the mean and the
frequency converge fast.
"""

from __future__ import annotations

import math

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.geometry.gmsh_tools import add_group, in_box
from feacore.problems.base import Problem, Reference
from feacore.problems.registry import register

GRAFF = "K.F. Graff, Wave Motion in Elastic Solids, Dover (1991), section 2.1 (waves in a rod, end reflections)"


class BarWaveStep(Problem):
    id = "bar_wave_step"
    title = "Step end load on a fixed-free bar: elastic wave, transient dynamics (exact 1D reference)"
    family = "bar_wave_step"
    tags = ("analytical", "2d", "plane_strain", "transient", "dynamics", "wave")
    dim = 2
    formulation = "plane_strain"
    expected_behavior = "smooth"
    default_params = {"E": 200e9, "nu": 0.0, "rho": 8000.0, "L": 1.0, "H": 0.1, "sigma": 1e6, "periods": 3,
                      "steps_per_period": 200, "h0": 0.05}

    @property
    def c(self) -> float:
        return math.sqrt(self.params["E"] / self.params["rho"])

    @property
    def period(self) -> float:
        return 4.0 * self.params["L"] / self.c

    def description(self) -> str:
        p = self.params
        return (
            "Transient dynamics: a bar loaded suddenly at its free end.\n"
            f"Geometry (metres): a rectangle {p['L']:g} long in x (x from 0 to {p['L']:g}) and {p['H']:g} high "
            f"(y from 0 to {p['H']:g}); plane strain.\n"
            f"Material: isotropic linear elastic, E = {p['E']:g} Pa, Poisson's ratio {p['nu']:g}, density "
            f"{p['rho']:g} kg/m^3. No damping.\n"
            f"Supports: the edge x = 0 is fixed (u_x = u_y = 0); the edges y = 0 and y = {p['H']:g} have u_y = 0 "
            "(u_x free).\n"
            f"Load: from t = 0 on, a uniform traction of {p['sigma']:g} Pa in +x on the edge x = {p['L']:g} "
            "(a step). The bar starts at rest.\n"
            f"Simulate {p['periods']} periods of the fundamental axial mode: t from 0 to "
            f"{p['periods'] * self.period:.6g} s.\n"
            "Report, for u_x at the point (L, H/2): u_max = its maximum over time (m), u_mean = its mean over "
            f"the first {self.period:.6g} s (m), and f_tip = the dominant frequency of its history (Hz)."
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        s = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, p["L"], p["H"])
        gmsh.model.occ.synchronize()
        add_group(1, in_box(1, xmax=0.0), "fixed_end")
        add_group(1, in_box(1, xmin=p["L"]), "loaded_end")
        add_group(1, in_box(1, ymax=0.0), "bottom")
        add_group(1, in_box(1, ymin=p["H"]), "top")
        add_group(2, [s], "bar")

    def qois(self) -> list[QoI]:
        p = self.params
        pt = [p["L"], p["H"] / 2]
        return [QoI("u_max", "point", field="ux", point=pt, time="max"),
                QoI("u_mean", "point", field="ux", point=pt, time="mean", window=[0.0, self.period]),
                QoI("f_tip", "point", field="ux", point=pt, time="frequency")]

    def case(self, mesh_path, order: int = 2) -> Case:
        p = self.params
        T = self.period
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_strain", order=order,
                              options={"transient": {"t_end": p["periods"] * T,
                                                     "n_steps": int(p["periods"] * p["steps_per_period"]),
                                                     "scheme": "hht", "alpha": -0.1}}),
            materials={"bar": Material(E=p["E"], nu=p["nu"], rho=p["rho"])},
            bcs=[BC(type="fixed", group="fixed_end"),
                 BC(type="displacement", group="bottom", components=[1], value=0.0),
                 BC(type="displacement", group="top", components=[1], value=0.0),
                 BC(type="traction", group="loaded_end", value=[p["sigma"], 0.0])],
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, kg, s, Pa, Hz)"},
        )

    def references(self) -> list[Reference]:
        p = self.params
        us = p["sigma"] * p["L"] / p["E"]
        return [
            Reference("u_max", 2.0 * us, "analytical", f"{GRAFF}: 2 sigma L / E at t = 2 L / c", tolerance=0.02,
                      notes="A corner of the exact history: reached from below, slow convergence (~ -1 % at the "
                            "reference setup)."),
            Reference("u_mean", us, "analytical", f"{GRAFF}: the mean of the triangle wave = the static sigma L / E",
                      tolerance=0.002),
            Reference("f_tip", self.c / (4.0 * p["L"]), "analytical", f"{GRAFF}: period 4 L / c",
                      tolerance=0.005, notes="The average-acceleration / HHT period error at 200 steps per period "
                                             "is (omega dt)^2 / 12 = 8e-5; the mesh adds less."),
        ]

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 2, h0 / 4]


register(BarWaveStep())

__all__ = ["BarWaveStep"]
