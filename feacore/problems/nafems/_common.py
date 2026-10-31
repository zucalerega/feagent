"""Shared helpers for the NAFEMS (and similar published-reference) benchmark problems.

Problem definitions follow "The Standard NAFEMS Benchmarks", NAFEMS publication TNSB Rev. 3
(October 1990). The booklet is not public; every problem module cites the public sources used to
verify geometry, boundary conditions and target values (see ``docs/benchmarks_nafems.md``).
"""

from __future__ import annotations

import math

import gmsh

from feacore.geometry.gmsh_tools import add_group, in_box

STEEL = {"E": 210e9, "nu": 0.3}          # E = 210 x 10^3 MPa, nu = 0.3 (LE1, LE10, LE11)

TNSB = 'NAFEMS, "The Standard NAFEMS Benchmarks", publication TNSB Rev. 3 (October 1990)'
FEENOX = "https://github.com/seamplex/feenox"
# NAFEMS data sheets reproduced (as images) in the FeenoX repository
LE10_SHEET = f"{FEENOX}/blob/main/doc/nafems-le10-problem-input.svg"
LE11_SHEET = f"{FEENOX}/blob/main/doc/nafems-le11-problem.png"

# Elliptic membrane / thick plate (LE1, LE10): quarter of an elliptic annulus.
ELL_INNER = (2.0, 1.0)      # (x/2)^2 + y^2 = 1          (D = (2, 0), A = (0, 1))
ELL_OUTER = (3.25, 2.75)    # (x/3.25)^2 + (y/2.75)^2 = 1 (C = (3.25, 0), B = (0, 2.75))


def quarter_elliptic_annulus(z: float = 0.0) -> int:
    """OCC surface: first-quadrant part of the elliptic annulus between ELL_INNER and ELL_OUTER,
    in the plane at height ``z``. Returns the surface tag (not synchronized)."""
    occ = gmsh.model.occ
    outer = occ.addDisk(0.0, 0.0, z, *ELL_OUTER)
    inner = occ.addDisk(0.0, 0.0, z, *ELL_INNER)
    ring, _ = occ.cut([(2, outer)], [(2, inner)])
    box = occ.addRectangle(0.0, 0.0, z, ELL_OUTER[0] + 1.0, ELL_OUTER[1] + 1.0)
    quarter, _ = occ.intersect(ring, [(2, box)])
    if len(quarter) != 1:
        raise RuntimeError(f"expected one quarter-annulus surface, got {quarter}")
    return quarter[0][1]


def point_tag(x: float, y: float, z: float = 0.0, tol: float = 1e-6) -> int:
    """Tag of the geometric vertex at (x, y, z) (after synchronize)."""
    tags = in_box(0, x, y, z, x, y, z, tol=tol)
    if len(tags) != 1:
        raise RuntimeError(f"expected one vertex at {(x, y, z)}, found {tags}")
    return tags[0]


def entities_where(dim: int, predicate) -> list[int]:
    """Tags of entities of ``dim`` whose bounding box satisfies ``predicate(xmin, ymin, zmin,
    xmax, ymax, zmax)`` (after synchronize). Bounding boxes are robust for curved entities,
    whose centre of mass need not lie on them."""
    out = []
    for _, tag in gmsh.model.getEntities(dim):
        if predicate(*gmsh.model.getBoundingBox(dim, tag)):
            out.append(tag)
    return out


def near(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def group(dim: int, tags, name: str) -> int:
    return add_group(dim, tags, name)


def mpa(v: float) -> str:
    return f"{v / 1e6:g} MPa"


SQRT_HALF = math.sqrt(0.5)
