"""Shared pieces of the test parts: materials, handbook Kt fits, gmsh helpers, a base class.

Handbook curve fits
-------------------
``kt_shoulder_fillet_round_bar_tension`` and ``kt_opposite_u_notches_tension`` are the
four-term polynomial fits Kt = C1 + C2 x + C3 x^2 + C4 x^3 published with Peterson's charts
(Pilkey & Pilkey, *Peterson's Stress Concentration Factors*, 3rd ed., Wiley 2008) and
reproduced in Young & Budynas, *Roark's Formulas for Stress and Strain*, 7th ed., Table 17.1.
Each fit has two ranges of the depth / radius ratio that meet at 2.0; the two coefficient
sets agree there to < 1 % (shoulder fillet; U-notches for 2t/H <= 0.4), a useful check
on the transcription (see ``tests/test_problems_parts.py``).
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

from feacore.case import Material
from feacore.geometry.gmsh_tools import add_group, distance_threshold_field, gmsh_session, use_min_field
from feacore.problems.base import Problem

# Linear-elastic properties (SI). Steel: generic structural / alloy steel. Aluminium 6061-T6:
# E = 68.9 GPa, nu = 0.33 (ASM / MMPDS handbook values), rounded.
STEEL = {"E": 200e9, "nu": 0.3}
AL6061 = {"E": 69e9, "nu": 0.33}

MM = 1e-3  # metres per millimetre


def material(props: dict) -> Material:
    return Material(E=props["E"], nu=props["nu"])


def fmt_mm(x_m: float) -> str:
    """Length in metres -> compact millimetre string ("12.5 mm")."""
    v = x_m / MM
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{s} mm"


def fmt_si(x: float, unit: str, scale: float = 1.0, digits: int = 4) -> str:
    v = x / scale
    return f"{v:.{digits}g} {unit}"


# --------------------------------------------------------------------------- handbook fits


def _cubic(c: tuple[float, float, float, float], x: float) -> float:
    return c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3


def kt_shoulder_fillet_round_bar_tension(D: float, d: float, r: float) -> float:
    """Kt (based on the small-diameter nominal stress 4P / (pi d^2)) for a stepped round bar
    with a shoulder fillet of radius r in axial tension. Peterson / Pilkey curve fit,
    h = (D - d) / 2, valid for 0.1 <= h/r <= 20 (and D/d <= 2)."""
    h = (D - d) / 2.0
    hr = h / r
    if not 0.1 <= hr <= 20.0:
        raise ValueError(f"h/r = {hr:.3g} outside the fit range [0.1, 20]")
    s = math.sqrt(hr)
    if hr <= 2.0:
        c = (0.926 + 1.157 * s - 0.099 * hr,
             0.012 - 3.036 * s + 0.961 * hr,
             -0.302 + 3.977 * s - 1.744 * hr,
             0.365 - 2.098 * s + 0.878 * hr)
    else:
        c = (1.200 + 0.860 * s - 0.022 * hr,
             -1.805 - 0.346 * s - 0.038 * hr,
             2.198 - 0.486 * s + 0.165 * hr,
             -0.593 - 0.028 * s - 0.106 * hr)
    return _cubic(c, 2.0 * h / D)


def kt_opposite_u_notches_tension(H: float, t: float, r: float) -> float:
    """Kt (based on the net-section nominal stress P / ((H - 2t) b)) for a flat bar of width H
    with two opposite U-notches of depth t and root radius r in tension. Peterson / Pilkey
    curve fit, valid for 0.1 <= t/r <= 50."""
    tr = t / r
    if not 0.1 <= tr <= 50.0:
        raise ValueError(f"t/r = {tr:.3g} outside the fit range [0.1, 50]")
    s = math.sqrt(tr)
    if tr <= 2.0:
        c = (0.955 + 2.169 * s - 0.081 * tr,
             -1.557 - 4.046 * s + 1.032 * tr,
             4.013 + 0.424 * s - 0.748 * tr,
             -2.461 + 1.538 * s - 0.236 * tr)
    else:
        c = (1.037 + 1.991 * s + 0.002 * tr,
             -1.886 - 2.181 * s - 0.048 * tr,
             0.649 + 1.086 * s + 0.142 * tr,
             1.218 - 0.922 * s - 0.086 * tr)
    return _cubic(c, 2.0 * t / H)


def lame_hoop_stress(p: float, ri: float, ro: float, r: float) -> float:
    """Hoop stress of a thick-walled cylinder under internal pressure p (Lame)."""
    return p * ri**2 / (ro**2 - ri**2) * (1.0 + ro**2 / r**2)


# --------------------------------------------------------------------------- gmsh helpers

TOL = 1e-7


def faces_of_type(kind: str, dim: int = 2) -> list[int]:
    return [t for _, t in gmsh.model.getEntities(dim) if gmsh.model.getType(dim, t) == kind]


def com(dim: int, tag: int) -> tuple[float, float, float]:
    return gmsh.model.occ.getCenterOfMass(dim, tag)


def bbox(dim: int, tag: int) -> tuple[float, float, float, float, float, float]:
    return gmsh.model.getBoundingBox(dim, tag)


def entities_on_plane(dim: int, axis: int, value: float, tol: float = 1e-6) -> list[int]:
    """Entities of ``dim`` lying in the plane coordinate[axis] == value (bounding-box test)."""
    out = []
    for _, t in gmsh.model.getEntities(dim):
        b = bbox(dim, t)
        if abs(b[axis] - value) < tol and abs(b[axis + 3] - value) < tol:
            out.append(t)
    return out


def threshold(size_min: float, size_max: float, dist_min: float, dist_max: float,
              curves=None, surfaces=None, points=None, sampling: int = 100) -> int:
    """Distance/Threshold field. Sizes must be proportional to the global h; the distances are
    geometric (fixed), so that halving h halves the size field everywhere."""
    return distance_threshold_field(curves=curves, surfaces=surfaces, points=points,
                                    size_min=size_min, size_max=size_max, dist_min=dist_min,
                                    dist_max=dist_max, sampling=sampling)


def finish_sizing(fields: list[int], h: float) -> None:
    use_min_field(fields, h)


# --------------------------------------------------------------------------- base class


class PartProblem(Problem):
    """A test part. ``_shape(full)`` builds the OCC solid (or face, 2D) and synchronizes;
    ``_groups(full)`` adds the named physical groups. The STEP handed to agents is always the
    complete part (``full=True``). The reference mesh (``build_geometry``) may use a symmetric
    portion of it (``reference_portion``) with symmetry BCs in ``case``; agents are free to
    model the full part or exploit the symmetry themselves."""

    reference_portion: str = "full"  # full | half | quarter

    def _shape(self, full: bool) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _groups(self, full: bool) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def build_geometry(self) -> None:
        full = self.reference_portion == "full"
        self._shape(full)
        self._groups(full)

    def export_step(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(self.id):
            self._shape(True)
            gmsh.write(str(path))
        return path

    # convenience
    @property
    def p(self) -> dict:
        return self.params


__all__ = [
    "AL6061", "MM", "PartProblem", "STEEL", "TOL", "add_group", "bbox", "com",
    "entities_on_plane", "faces_of_type", "finish_sizing", "fmt_mm", "fmt_si",
    "kt_opposite_u_notches_tension", "kt_shoulder_fillet_round_bar_tension", "lame_hoop_stress",
    "material", "threshold",
]
