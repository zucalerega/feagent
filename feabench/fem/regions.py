"""Geometric region predicates on node coordinates.

Regions are how tasks tell agents *where* boundary conditions go and how the evaluator finds
the same places in any agent's mesh, without relying on CAD face ids or mesh node numbers::

    {"type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]}
    {"type": "cylinder", "point": [0, 0, 0], "axis": [0, 0, 1], "radius": 50.0, "range": [0, 200]}
    {"type": "sphere", "center": [0, 0, 0], "radius": 10.0}
    {"type": "box", "min": [..], "max": [..]}
    {"type": "point", "xyz": [..]}                       (nearest node, within tolerance)
    {"type": "all"} | {"type": "and"|"or", "regions": [...]} | {"type": "not", "region": {...}}

Optional on every region: ``"tol"`` (absolute) -- default is 1e-5 x model bounding-box diagonal
(gmsh puts boundary nodes, including high-order ones, on the CAD surface; 1e-5 also absorbs
coordinates written with 6 significant digits), and ``"bounds"``
(``{"min": [...], "max": [...]}``) to clip any region to a box.
"""

from __future__ import annotations

import numpy as np


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def select_nodes(coords: np.ndarray, region: dict, scale: float | None = None) -> np.ndarray:
    """Boolean mask of nodes inside ``region``."""
    X = np.asarray(coords, float)
    if scale is None:
        scale = float(np.linalg.norm(X.max(0) - X.min(0))) if len(X) else 1.0
    tol = float(region.get("tol", 1e-5 * scale))
    t = region["type"]
    if t == "all":
        m = np.ones(len(X), bool)
    elif t == "plane":
        n = _unit(region["normal"])
        m = np.abs((X - np.asarray(region["point"], float)) @ n) <= tol
    elif t == "cylinder":
        a = _unit(region["axis"])
        d = X - np.asarray(region["point"], float)
        s = d @ a
        r = np.linalg.norm(d - np.outer(s, a), axis=1)
        m = np.abs(r - float(region["radius"])) <= tol
        if "range" in region:
            lo, hi = region["range"]
            m &= (s >= lo - tol) & (s <= hi + tol)
    elif t == "elliptic_cylinder":
        # axis parallel to z through "center" (x, y); semi-axes (a along x, b along y)
        cx, cy = region["center"][:2]
        a, b = region["semi_axes"]
        rho = np.sqrt(((X[:, 0] - cx) / a) ** 2 + ((X[:, 1] - cy) / b) ** 2)
        m = np.abs(rho - 1.0) * min(a, b) <= tol
        if "range" in region:
            lo, hi = region["range"]
            m &= (X[:, 2] >= lo - tol) & (X[:, 2] <= hi + tol)
    elif t == "sphere":
        r = np.linalg.norm(X - np.asarray(region["center"], float), axis=1)
        m = np.abs(r - float(region["radius"])) <= tol
    elif t == "box":
        lo, hi = np.asarray(region["min"], float), np.asarray(region["max"], float)
        m = np.all((X >= lo - tol) & (X <= hi + tol), axis=1)
    elif t == "point":
        d = np.linalg.norm(X - np.asarray(region["xyz"], float), axis=1)
        m = np.zeros(len(X), bool)
        if len(d) and d.min() <= max(tol, float(region.get("tol", 0))):
            m[int(np.argmin(d))] = True
    elif t == "and":
        m = np.ones(len(X), bool)
        for sub in region["regions"]:
            m &= select_nodes(X, sub, scale)
    elif t == "or":
        m = np.zeros(len(X), bool)
        for sub in region["regions"]:
            m |= select_nodes(X, sub, scale)
    elif t == "not":
        m = ~select_nodes(X, region["region"], scale)
    else:
        raise ValueError(f"unknown region type {t!r}")
    if "bounds" in region:
        lo, hi = np.asarray(region["bounds"]["min"], float), np.asarray(region["bounds"]["max"], float)
        m &= np.all((X >= lo - tol) & (X <= hi + tol), axis=1)
    return m


def describe(region: dict, unit: str = "") -> str:
    """Human-readable description used in task briefs."""
    t = region["type"]
    u = f" {unit}" if unit else ""
    if "label" in region:
        return region["label"]
    if t == "plane":
        return f"the planar face(s) through {region['point']}{u} with normal {region['normal']}"
    if t == "cylinder":
        rng = f", axial range {region['range']}{u}" if "range" in region else ""
        return f"the cylindrical face(s) of radius {region['radius']}{u} about axis {region['axis']} through {region['point']}{u}{rng}"
    if t == "sphere":
        return f"the spherical face(s) of radius {region['radius']}{u} centred at {region['center']}{u}"
    if t == "elliptic_cylinder":
        return (f"the elliptic-cylindrical face(s) (x/{region['semi_axes'][0]})^2 + (y/{region['semi_axes'][1]})^2 = 1"
                f" (axis parallel to z)")
    if t == "box":
        return f"the geometry inside the box {region['min']}..{region['max']}{u}"
    if t == "point":
        return f"the point {region['xyz']}{u}"
    if t in ("and", "or"):
        return f" {t} ".join(describe(r, unit) for r in region["regions"])
    if t == "not":
        return f"everything except ({describe(region['region'], unit)})"
    return t
