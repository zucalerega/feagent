"""Small gmsh helpers shared by the reference problem builders (not the agents' mesher)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import gmsh


@contextmanager
def gmsh_session(name: str = "model", verbosity: int = 0):
    """Initialise gmsh, yield the module, always finalise. gmsh is process-global: one
    session at a time per process."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbosity else 0)
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add(name)
        yield gmsh
    finally:
        gmsh.finalize()


def in_box(dim: int, xmin=-1e30, ymin=-1e30, zmin=-1e30, xmax=1e30, ymax=1e30, zmax=1e30,
           tol: float = 1e-7) -> list[int]:
    """Tags of entities of ``dim`` whose bounding box lies inside the (tolerance-padded) box."""
    ents = gmsh.model.getEntitiesInBoundingBox(xmin - tol, ymin - tol, zmin - tol,
                                               xmax + tol, ymax + tol, zmax + tol, dim)
    return [t for _, t in ents]


def where(dim: int, predicate: Callable[[float, float, float], bool]) -> list[int]:
    """Tags of entities of ``dim`` whose centre of mass satisfies ``predicate(x, y, z)``."""
    out = []
    for _, tag in gmsh.model.getEntities(dim):
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
        if predicate(x, y, z):
            out.append(tag)
    return out


def add_group(dim: int, tags: Iterable[int], name: str) -> int:
    tags = sorted(set(int(t) for t in tags))
    if not tags:
        raise ValueError(f"physical group {name!r} would be empty")
    return gmsh.model.addPhysicalGroup(dim, tags, name=name)


def ensure_domain_group(dim: int, name: str = "domain") -> None:
    """Make sure every entity of the top dimension is in some physical group."""
    covered = set()
    for d, ptag in gmsh.model.getPhysicalGroups(dim):
        covered.update(gmsh.model.getEntitiesForPhysicalGroup(d, ptag))
    missing = [t for _, t in gmsh.model.getEntities(dim) if t not in covered]
    if missing:
        if covered:
            raise ValueError(f"entities {missing} of dim {dim} are in no physical group")
        add_group(dim, missing, name)


def set_uniform_size(h: float) -> None:
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def distance_threshold_field(curves: list[int] | None = None, surfaces: list[int] | None = None,
                             points: list[int] | None = None, size_min: float = 0.1,
                             size_max: float = 1.0, dist_min: float = 0.0, dist_max: float = 1.0,
                             sampling: int = 200) -> int:
    """Distance+Threshold field (size_min near the entities growing to size_max at dist_max).
    Returns the Threshold field tag; combine with ``use_min_field``. Scale every argument
    with the global h so refinement stays geometrically similar across a convergence study."""
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    if points:
        f.setNumbers(d, "PointsList", points)
    if curves:
        f.setNumbers(d, "CurvesList", curves)
        f.setNumber(d, "Sampling", sampling)
    if surfaces:
        f.setNumbers(d, "SurfacesList", surfaces)
        f.setNumber(d, "Sampling", sampling)
    t = f.add("Threshold")
    f.setNumber(t, "InField", d)
    f.setNumber(t, "SizeMin", size_min)
    f.setNumber(t, "SizeMax", size_max)
    f.setNumber(t, "DistMin", dist_min)
    f.setNumber(t, "DistMax", dist_max)
    return t


def use_min_field(field_tags: list[int], h_max: float) -> None:
    """Background mesh = min of the given fields, capped at h_max; disables other size sources."""
    f = gmsh.model.mesh.field
    m = f.add("Min")
    f.setNumbers(m, "FieldsList", field_tags)
    f.setAsBackgroundMesh(m)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
