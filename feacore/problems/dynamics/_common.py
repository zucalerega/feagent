"""Shared pieces of the structural dynamics benchmarks (``feacore.problems.dynamics``).

* ``ModalProblem``: a ``Problem`` whose QoIs are the natural frequencies ``f1 .. fn`` of the elastic
  modes (rigid-body modes are not counted); eigenvalues of a smooth problem converge with h^(2p).
* ``lamb_frequency``: the exact frequency of a standing Rayleigh-Lamb wave in a plate, the reference of
  the simply supported beam and plate problems.
* the NAFEMS free-vibration material and citations.

The NAFEMS booklets are not public. The reference frequencies of FV32, FV41 and FV52 were taken from
search-engine extracts of the public reproductions named in each problem module (the Abaqus Benchmarks
Guide and the Altair OptiStruct and Ansys verification manuals; their hosts are not reachable from the
build session). Dimensions that the extracts do not state are marked in the modules; the solver
reproduces every published frequency of FV32 and FV41 to 0.05 % with them.
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import Analysis, Case, Material, QoI
from feacore.problems.base import Problem

NAFEMS_FV_STEEL = {"E": 200e9, "nu": 0.3, "rho": 8000.0}     # every NAFEMS FV test

TNSB = 'NAFEMS, "The Standard NAFEMS Benchmarks", publication TNSB Rev. 3 (October 1990)'
R0015 = ('F. Abbassian, D.J. Dawswell & N.C. Knowles, "Selected Benchmarks for Natural Frequency Analysis", '
         'NAFEMS R0015 (1987)')
ABAQUS_BMK = "Abaqus Benchmarks Guide, section 'NAFEMS benchmarks: free vibration'"
LAMB = ("H. Lamb, Proc. R. Soc. Lond. A 93 (1917) 114-128; K.F. Graff, Wave Motion in Elastic Solids, Dover (1991), "
        "section 8.1 (Rayleigh-Lamb frequency equations)")


def lamb_frequency(k: float, half: float, c_l: float, c_t: float, branch: str = "A") -> float:
    """The lowest circular frequency (rad/s) of a Rayleigh-Lamb wave with wavenumber ``k`` in a plate of
    half thickness ``half`` (free faces), for longitudinal and shear wave speeds ``c_l`` and ``c_t``:
    branch "A" antisymmetric (flexural), "S" symmetric (extensional).

    With p^2 = w^2 / c_l^2 - k^2 and q^2 = w^2 / c_t^2 - k^2 the frequency equations are
      A: 4 k^2 p q sin(q h) cos(p h) + (q^2 - k^2)^2 sin(p h) cos(q h) = 0
      S: (q^2 - k^2)^2 sin(q h) cos(p h) + 4 k^2 p q sin(p h) cos(q h) = 0
    (Graff, eqs. 8.1.56-57). Dividing A by p and S by q gives functions that are even in p and q, so real
    for every real w, also where p or q is imaginary. A standing wave sin(k x) of this kind is the exact
    solution of a simply supported strip (transverse and tangential displacement zero on the end faces,
    no normal stress there): plane strain with the 3D wave speeds, plane stress with
    c_l^2 = E / (rho (1 - nu^2)). w = 0 is a trivial root; the search starts at 0.3 of the thin-plate
    estimate, which holds for k * half up to about 2."""
    from scipy.optimize import brentq

    def g(w):
        p = np.sqrt(complex(w * w / c_l**2 - k * k))
        q = np.sqrt(complex(w * w / c_t**2 - k * k))
        sp_, sq_ = np.sin(p * half) / p, np.sin(q * half) / q
        cp, cq = np.cos(p * half), np.cos(q * half)
        if branch == "A":
            v = 4 * k * k * q * q * sq_ * cp + (q * q - k * k) ** 2 * cq * sp_
        else:
            v = (q * q - k * k) ** 2 * sq_ * cp + 4 * k * k * p * p * sp_ * cq
        return float(v.real)

    c_pl = 2 * c_t * math.sqrt(1 - (c_t / c_l) ** 2)             # plate (plane-stress) longitudinal speed
    w_est = k * k * (2 * half) / math.sqrt(12) * c_pl if branch == "A" else k * c_pl
    ws = np.linspace(0.3, 1.02, 400) * w_est
    gs = np.array([g(w) for w in ws])
    cross = np.nonzero(np.sign(gs[:-1]) * np.sign(gs[1:]) < 0)[0]
    if cross.size == 0:
        raise ValueError(f"lamb_frequency: no root between 0.3 and 1.02 of the thin-plate estimate "
                         f"(k * half = {k * half:g} is outside the range of this search)")
    i = int(cross[0])
    return float(brentq(g, ws[i], ws[i + 1], xtol=1e-14, rtol=1e-14))


def wave_speeds(E: float, nu: float, rho: float, plane_stress: bool = False) -> tuple[float, float]:
    """(c_l, c_t): longitudinal and shear wave speeds; ``plane_stress`` gives the plate speed
    sqrt(E / (rho (1 - nu^2))) for c_l (a plane-stress sheet), else the 3D / plane-strain one."""
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    c_l = math.sqrt(E / (rho * (1 - nu * nu))) if plane_stress else math.sqrt((lam + 2 * mu) / rho)
    return c_l, math.sqrt(mu / rho)


class ModalProblem(Problem):
    """A modal benchmark: ``n_modes`` natural frequencies ``f1 .. fn`` (Hz) of the elastic modes."""

    n_modes: int = 6
    expected_behavior = "smooth"

    def material(self) -> Material:
        p = self.params
        return Material(E=p["E"], nu=p["nu"], rho=p["rho"])

    def bcs(self) -> list:
        return []

    def qois(self) -> list[QoI]:
        return [QoI(f"f{k}", "natural_frequency", mode=k) for k in range(1, self.n_modes + 1)]

    def expected_order(self, qoi: QoI, order: int) -> float | None:
        return 2.0 * order          # eigenvalues converge twice as fast as the mode shapes in energy

    def case(self, mesh_path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation=self.formulation, order=order,
                              thickness=float(self.params.get("thickness", 1.0)),
                              options={"modal": {"n_modes": self.n_modes}}),
            materials={"material": self.material()},
            bcs=self.bcs(),
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, kg, s, Hz)"},
        )
