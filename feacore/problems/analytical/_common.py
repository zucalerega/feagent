"""Shared helpers for the analytical benchmark problems (``feacore.problems.analytical``).

* ``ExactProblem``: a ``Problem`` with an exact solution. Subclasses implement
  ``_primary(x, y, z)`` returning the primary fields (displacements + stress components, or
  T + flux); ``exact`` derives the rest (u_mag, von_mises, tresca, principal stresses, q_mag)
  and ``point_reference`` turns a point QoI into a ``Reference`` from the exact field.
* geometry helpers (gmsh OCC) used by several problems.
* ``num`` formats a float for the restricted expression language with full precision.
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, Analysis, Case, Load, Material, QoI, Region
from feacore.problems.base import Problem, Reference

STEEL = {"E": 200e9, "nu": 0.3}
ALUMINIUM = {"E": 70e9, "nu": 0.33}

# Citations used by several problems (kept short; equation numbers are given where the
# formula is quoted in the problem module).
TG = "S.P. Timoshenko & J.N. Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill (1970)"
BARBER = "J.R. Barber, Elasticity, 3rd ed., Springer (2010)"


def num(v: float) -> str:
    """Float literal for expressions (full double precision, never locale dependent)."""
    return repr(float(v))


# --------------------------------------------------------------------------- stress measures


def stress_tensor(prim: dict, formulation: str) -> np.ndarray:
    """(n, 3, 3) stress tensors from primary components. Axisymmetric: (r, z, theta) axes with
    the hoop stress ``stt`` as the out-of-plane normal stress."""
    sxx = np.asarray(prim["sxx"], dtype=float)
    z = np.zeros_like(sxx)
    syy, sxy = prim.get("syy", z), prim.get("sxy", z)
    if formulation == "axisymmetric":
        s33, syz, sxz = prim["stt"], z, z
    else:
        s33, syz, sxz = prim.get("szz", z), prim.get("syz", z), prim.get("sxz", z)
    S = np.empty(sxx.shape + (3, 3))
    S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = sxx, syy, s33
    S[..., 0, 1] = S[..., 1, 0] = sxy
    S[..., 1, 2] = S[..., 2, 1] = syz
    S[..., 0, 2] = S[..., 2, 0] = sxz
    return S


def derived_field(field: str, prim: dict, formulation: str) -> np.ndarray | None:
    """Value of any feacore field from the primary fields (None if not defined here)."""
    if field in prim:
        return np.asarray(prim[field], dtype=float)
    if field == "u_mag" and "ux" in prim:
        return np.sqrt(sum(np.asarray(prim[c], dtype=float) ** 2 for c in ("ux", "uy", "uz") if c in prim))
    if field == "q_mag" and "qx" in prim:
        return np.sqrt(sum(np.asarray(prim[c], dtype=float) ** 2 for c in ("qx", "qy", "qz") if c in prim))
    if field in ("uz", "syz", "sxz", "qz") and ("ux" in prim or "qx" in prim):
        ref = prim.get("ux", prim.get("qx"))
        return np.zeros_like(np.asarray(ref, dtype=float))
    if field == "szz" and formulation in ("plane_stress", "axisymmetric") and "sxx" in prim:
        return np.zeros_like(np.asarray(prim["sxx"], dtype=float))
    if field in ("von_mises", "tresca", "s1", "s2", "s3") and "sxx" in prim:
        S = stress_tensor(prim, formulation)
        if field == "von_mises":
            d = S - np.trace(S, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0
            return np.sqrt(1.5 * np.einsum("...ij,...ij->...", d, d))
        ev = np.linalg.eigvalsh(S)  # ascending
        if field == "tresca":
            return ev[..., 2] - ev[..., 0]
        return ev[..., {"s1": 2, "s2": 1, "s3": 0}[field]]
    return None


# --------------------------------------------------------------------------- problem base


class ExactProblem(Problem):
    """A problem with a closed-form solution everywhere in the (reference) domain."""

    material_name = "steel"

    # -- to implement
    def _primary(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
        raise NotImplementedError

    # -- exact fields
    def exact(self, field: str, points) -> np.ndarray | None:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        x, y = pts[:, 0], pts[:, 1]
        z = pts[:, 2] if pts.shape[1] > 2 else np.zeros_like(x)
        return derived_field(field, self._primary(x, y, z), self.formulation)

    def exact_at(self, field: str, point) -> float:
        v = self.exact(field, np.asarray(point, dtype=float)[None, :])
        if v is None:
            raise KeyError(f"{self.id}: no exact field {field!r}")
        return float(v[0])

    def point_reference(self, qoi_name: str, source: str, tolerance: float, notes: str = "") -> Reference:
        q = next(q for q in self.qois() if q.name == qoi_name)
        assert q.type == "point", q
        return Reference(qoi=qoi_name, value=self.exact_at(q.field, q.point), kind="analytical",
                         source=source, tolerance=tolerance, notes=notes)

    # -- case helpers
    def material(self) -> Material:
        p = self.params
        return Material(E=p["E"], nu=p["nu"])

    def make_case(self, mesh_path, order: int, bcs: list[BC], loads: list[Load] | None = None,
                  thickness: float = 1.0) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics=self.physics, formulation=self.formulation, order=order,
                              thickness=thickness),
            materials={self.material_name: self.material()},
            bcs=bcs,
            loads=list(loads or []),
            regions=[Region(group="domain", material=self.material_name)],
            qois=self.qois(),
            meta={"problem": self.id, "setup": "reference"},
        )


# --------------------------------------------------------------------------- geometry helpers


def quarter_annulus(a: float, b: float) -> dict:
    """Quarter annulus a <= r <= b in x, y >= 0 from OCC primitives. Returns curve/surface tags:
    y0 (edge on y = 0), outer (arc r = b), x0 (edge on x = 0), inner (arc r = a), surface."""
    import gmsh

    occ = gmsh.model.occ
    c = occ.addPoint(0, 0, 0)
    p1, p2 = occ.addPoint(a, 0, 0), occ.addPoint(b, 0, 0)
    p3, p4 = occ.addPoint(0, b, 0), occ.addPoint(0, a, 0)
    y0 = occ.addLine(p1, p2)
    outer = occ.addCircleArc(p2, c, p3)
    x0 = occ.addLine(p3, p4)
    inner = occ.addCircleArc(p4, c, p1)
    s = occ.addPlaneSurface([occ.addCurveLoop([y0, outer, x0, inner])])
    occ.remove([(0, c)])
    occ.synchronize()
    return {"y0": y0, "outer": outer, "x0": x0, "inner": inner, "surface": s}


def polygon(points: list[tuple[float, float]]) -> tuple[list[int], int]:
    """Closed polygon (counter-clockwise points) -> (line tags in order p0p1, p1p2, ..., surface)."""
    import gmsh

    occ = gmsh.model.occ
    pts = [occ.addPoint(x, y, 0) for x, y in points]
    lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    s = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    occ.synchronize()
    return lines, s


def plate_with_quarter_hole(r: float, lx: float, ly: float) -> dict:
    """Rectangle [0, lx] x [0, ly] minus the quarter disk of radius r at the origin.
    Returns tags: y0 (y = 0, r..lx), right (x = lx), top (y = ly), x0 (x = 0, r..ly), hole, surface."""
    import gmsh

    occ = gmsh.model.occ
    c = occ.addPoint(0, 0, 0)
    p1, p2, p3 = occ.addPoint(r, 0, 0), occ.addPoint(lx, 0, 0), occ.addPoint(lx, ly, 0)
    p4, p5 = occ.addPoint(0, ly, 0), occ.addPoint(0, r, 0)
    y0 = occ.addLine(p1, p2)
    right = occ.addLine(p2, p3)
    top = occ.addLine(p3, p4)
    x0 = occ.addLine(p4, p5)
    hole = occ.addCircleArc(p5, c, p1)
    s = occ.addPlaneSurface([occ.addCurveLoop([y0, right, top, x0, hole])])
    occ.remove([(0, c)])
    occ.synchronize()
    return {"y0": y0, "right": right, "top": top, "x0": x0, "hole": hole, "surface": s}


def refine_near(curves: list[int], h: float, fine: float, coarse: float, dist_min: float,
                dist_max: float) -> None:
    """Size h*fine within dist_min of the curves, growing linearly to h*coarse at dist_max.
    Every size scales with h; distances are geometric (fixed), so refinement is self-similar."""
    from feacore.geometry.gmsh_tools import distance_threshold_field, use_min_field

    t = distance_threshold_field(curves=curves, size_min=fine * h, size_max=coarse * h,
                                 dist_min=dist_min, dist_max=dist_max, sampling=400)
    use_min_field([t], h_max=coarse * h)


def check_close(a: float, b: float, rel: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


__all__ = [
    "STEEL", "ALUMINIUM", "TG", "BARBER", "num", "stress_tensor", "derived_field", "ExactProblem",
    "quarter_annulus", "polygon", "plate_with_quarter_hole", "refine_near", "BC", "Load", "QoI",
    "Reference",
]
