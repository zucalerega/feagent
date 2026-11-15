"""Natural frequencies with an exact elasticity reference (Rayleigh-Lamb standing waves).

ss_strip_modes        a simply supported thick strip in plane strain (2D, one rigid-body mode)
ss_thick_plate_modes  a simply supported thick square plate as a 3D solid (the well-posed counterpart of
                      NAFEMS FV52, whose line supports have no mesh-converged answer)

"Simply supported" here means the sliding support that makes a standing Lamb wave sin(k x) an exact
solution of 3D elasticity: on an end face the transverse and the in-face tangential displacement are zero
and the normal displacement is free (so the normal stress vanishes there). The frequencies then follow from
the Rayleigh-Lamb equations with k = n pi / L (``_common.lamb_frequency``), for every thickness: no plate
or beam theory is involved. Eigenvalues of P2 elements converge with h^4 (observed 3.8-4.6 on the strip).
"""

from __future__ import annotations

import math

from feacore.case import BC
from feacore.geometry.gmsh_tools import add_group, in_box
from feacore.problems.base import Reference
from feacore.problems.dynamics._common import LAMB, ModalProblem, lamb_frequency, wave_speeds
from feacore.problems.registry import register

_STEEL = {"E": 200e9, "nu": 0.3, "rho": 8000.0}


def strip_frequencies(L: float, t: float, E: float, nu: float, rho: float, n: int) -> list[tuple[float, str, int]]:
    """The n lowest elastic frequencies (Hz) of the plane-strain strip, as (f, branch, half waves)."""
    cl, ct = wave_speeds(E, nu, rho)
    out = []
    for m in range(1, 2 * n + 2):
        k = m * math.pi / L
        for b in "AS":
            try:
                out.append((lamb_frequency(k, t / 2, cl, ct, b) / (2 * math.pi), b, m))
            except ValueError:
                pass
    out.sort()
    if out and out[min(n, len(out)) - 1][0] >= ct / (2 * t):
        raise ValueError("strip_frequencies: a requested mode lies above the first thickness cut-off, where "
                         "higher Lamb branches (not searched) start")
    return out[:n]


def plate_frequencies(a: float, t: float, E: float, nu: float, rho: float, n: int) -> list[tuple[float, str]]:
    """The n lowest elastic frequencies (Hz) of the square plate a x a x t with sliding supports on the four
    side faces: antisymmetric Lamb modes with k = pi sqrt(m^2 + q^2) / a (m, q >= 1), symmetric Lamb modes
    (in-plane, m, q >= 1), and the shear-horizontal modes u = sin(m pi x / a) e_y (and x <-> y) at
    m c_t / (2 a), which are exact for any thickness."""
    cl, ct = wave_speeds(E, nu, rho)
    out = []
    for m in range(0, 6):
        for q in range(0, 6):
            if m == 0 and q == 0:
                continue
            k = math.pi * math.hypot(m, q) / a
            if m >= 1 and q >= 1:
                for b in "AS":
                    out.append((lamb_frequency(k, t / 2, cl, ct, b) / (2 * math.pi), f"{b}({m},{q})"))
            if q == 0 or m == 0:                    # shear horizontal, along x or along y
                out.append((max(m, q) * ct / (2 * a), f"SH({m},{q})"))
    out.sort()
    if out[n - 1][0] >= ct / (2 * t):
        raise ValueError("plate_frequencies: a requested mode lies above the first thickness cut-off")
    return out[:n]


class SSStripModes(ModalProblem):
    id = "ss_strip_modes"
    title = "Simply supported thick strip: natural frequencies (plane strain, exact Rayleigh-Lamb reference)"
    family = "ss_strip_modes"
    tags = ("analytical", "2d", "plane_strain", "modal", "dynamics", "rigid_body_mode")
    dim = 2
    formulation = "plane_strain"
    n_modes = 6
    default_params = {**_STEEL, "L": 10.0, "t": 1.0, "h0": 0.5}

    def description(self) -> str:
        p = self.params
        return (
            "Free vibration (natural frequencies) of a thick strip, plane strain, in-plane motion.\n"
            f"Geometry (metres): a rectangle {p['L']:g} long in x (x from 0 to {p['L']:g}) and {p['t']:g} thick "
            f"(y from {-p['t'] / 2:g} to {p['t'] / 2:g}).\n"
            f"Material: isotropic linear elastic, E = {p['E']:g} Pa, Poisson's ratio {p['nu']:g}, density "
            f"{p['rho']:g} kg/m^3.\n"
            "Supports: on the two end edges (x = 0 and x = L) the transverse displacement u_y is zero; u_x is "
            "free there. Nothing else is restrained, so the strip keeps one rigid-body mode (translation in x) "
            "at 0 Hz; do not count it.\n"
            "Report the six lowest natural frequencies of the ELASTIC modes in Hz, in ascending order: f1 .. f6."
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        s = gmsh.model.occ.addRectangle(0.0, -p["t"] / 2, 0.0, p["L"], p["t"])
        gmsh.model.occ.synchronize()
        add_group(1, in_box(1, xmax=0.0), "left")
        add_group(1, in_box(1, xmin=p["L"]), "right")
        add_group(2, [s], "strip")

    def bcs(self):
        return [BC(type="displacement", group=g, components=[1], value=0.0) for g in ("left", "right")]

    def references(self) -> list[Reference]:
        p = self.params
        modes = strip_frequencies(p["L"], p["t"], p["E"], p["nu"], p["rho"], self.n_modes)
        return [Reference(f"f{k}", f, "analytical",
                          f"{LAMB}; plane strain, wavenumber k = {m} pi / L, branch {b} "
                          f"({'flexural' if b == 'A' else 'extensional'})", tolerance=0.002,
                          notes="Exact for 3D elasticity in plane strain; P2 meshes converge with h^4 "
                                "(error below 0.5 % at h = 0.5, 2e-4 at h = 0.25).")
                for k, (f, b, m) in enumerate(modes, start=1)]


class SSThickPlateModes(ModalProblem):
    id = "ss_thick_plate_modes"
    title = "Simply supported thick square plate as a 3D solid: natural frequencies (exact Rayleigh-Lamb reference)"
    family = "ss_thick_plate_modes"
    tags = ("analytical", "3d", "modal", "dynamics")
    dim = 3
    formulation = "3d"
    n_modes = 6
    default_params = {**_STEEL, "a": 10.0, "t": 1.0, "h0": 1.0}

    def description(self) -> str:
        p = self.params
        return (
            "Free vibration (natural frequencies) of a thick square plate modelled as a 3D solid.\n"
            f"Geometry (metres): x and y from 0 to {p['a']:g}, z from 0 to {p['t']:g}.\n"
            f"Material: isotropic linear elastic, E = {p['E']:g} Pa, Poisson's ratio {p['nu']:g}, density "
            f"{p['rho']:g} kg/m^3.\n"
            "Supports (sliding simple supports over the whole side faces): on the faces x = 0 and x = a, "
            "u_y = u_z = 0 (u_x free); on the faces y = 0 and y = a, u_x = u_z = 0 (u_y free). The top and "
            "bottom faces are free.\n"
            "Report the six lowest natural frequencies in Hz, in ascending order: f1 .. f6."
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        a, t = p["a"], p["t"]
        b = gmsh.model.occ.addBox(0.0, 0.0, 0.0, a, a, t)
        gmsh.model.occ.synchronize()
        add_group(2, in_box(2, xmax=0.0), "x0")
        add_group(2, in_box(2, xmin=a), "x1")
        add_group(2, in_box(2, ymax=0.0), "y0")
        add_group(2, in_box(2, ymin=a), "y1")
        add_group(3, [b], "plate")

    def bcs(self):
        return ([BC(type="displacement", group=g, components=[1, 2], value=0.0) for g in ("x0", "x1")]
                + [BC(type="displacement", group=g, components=[0, 2], value=0.0) for g in ("y0", "y1")])

    def study_sizes(self, order: int = 2) -> list[float]:
        return [1.0, 0.7, 0.5]

    def references(self) -> list[Reference]:
        p = self.params
        modes = plate_frequencies(p["a"], p["t"], p["E"], p["nu"], p["rho"], self.n_modes)
        return [Reference(f"f{k}", f, "analytical", f"{LAMB}; 3D plate, mode {name}", tolerance=0.02,
                          notes="Exact for 3D elasticity. With one or two P2 cells through the thickness the "
                                "flexural modes converge slowly from above (+0.5 % at h = 1 for f1).")
                for k, (f, name) in enumerate(modes, start=1)]


for _p in (SSStripModes(), SSThickPlateModes()):
    register(_p)

__all__ = ["SSStripModes", "SSThickPlateModes", "strip_frequencies", "plate_frequencies"]
