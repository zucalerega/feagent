"""Helpers for repeated gmsh model/mesh access patterns (call inside a gmsh session)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np

TRI_TYPES = (2, 9)  # gmsh element types: tri3, tri6
TET_TYPES = (4, 11)  # tet4, tet10


def entity_tags(dim: int) -> list[int]:
    import gmsh

    return [int(t) for _, t in gmsh.model.getEntities(dim)]


def node_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (node tags, coords [n, 3], lookup) with ``lookup[tag]`` = row of that node."""
    import gmsh

    ntags, coords, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, np.int64)
    lookup = np.full(int(ntags.max()) + 1 if len(ntags) else 1, -1, dtype=np.int64)
    lookup[ntags] = np.arange(len(ntags))
    return ntags, np.asarray(coords).reshape(-1, 3), lookup


def element_blocks(dim: int, tag: int, types: Iterable[int] | None = None
                   ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (type, element tags, node tags [n, nodes per element]) of the wanted types (all if None)."""
    import gmsh

    wanted = None if types is None else set(types)
    for t, etags, enodes in zip(*gmsh.model.mesh.getElements(dim, tag)):
        if wanted is None or t in wanted:
            nn = gmsh.model.mesh.getElementProperties(t)[3]
            yield int(t), np.asarray(etags), np.asarray(enodes).reshape(-1, nn)


def corner_rows(dim: int, tag: int, types: Iterable[int], corners: int,
                lookup: np.ndarray) -> np.ndarray:
    """Corner-node rows (via ``lookup``) of all elements of ``types`` on one entity."""
    rows = [lookup[nodes[:, :corners]] for _, _, nodes in element_blocks(dim, tag, types)]
    return np.concatenate(rows) if rows else np.zeros((0, corners), np.int64)


def physical_groups() -> Iterator[tuple[int, str, list[int]]]:
    """Yield (dim, name, entity tags) for every physical group."""
    import gmsh

    for dim, pg in gmsh.model.getPhysicalGroups():
        ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(dim, pg)]
        yield dim, gmsh.model.getPhysicalName(dim, pg), ents


def node_entity(node_tag: int) -> tuple[int, int]:
    """(dim, tag) of the model entity a mesh node is classified on."""
    import gmsh

    _, _, dim, tag = gmsh.model.mesh.getNode(int(node_tag))
    return int(dim), int(tag)


def element_faces(element_tag: int) -> list[int]:
    """Sorted CAD faces that the nodes of one element are classified on."""
    import gmsh

    _, nodes, _, _ = gmsh.model.mesh.getElement(int(element_tag))
    return sorted({tag for dim, tag in map(node_entity, nodes) if dim == 2})


def closure(dim: int, tag: int) -> set[tuple[int, int]]:
    """The entity plus every lower-dimensional entity on its boundary (e.g. curves, points)."""
    import gmsh

    out, level = {(dim, tag)}, [(dim, tag)]
    while level and level[0][0] > 0:
        bnd = gmsh.model.getBoundary(level, combined=False, oriented=False, recursive=False)
        level = [(int(d), int(t)) for d, t in bnd]
        out.update(level)
    return out


def element_qualities(tags: Iterable[int], measures: Iterable[str]) -> dict[str, np.ndarray]:
    """Per-element quality arrays keyed by gmsh measure name (empty arrays if no tags)."""
    import gmsh

    tl = [int(t) for t in tags]
    get = gmsh.model.mesh.getElementQualities
    return {m: np.asarray(get(tl, m), float) if tl else np.zeros(0) for m in measures}


def integrated_measure(dim: int, tag: int = -1, rule: str = "Gauss4") -> float:
    """Length / area / volume of the mesh elements of dimension ``dim`` on entity ``tag`` (all if -1)
    by Gauss quadrature of the Jacobian determinant: exact for curved (order-2) elements, where
    corner-node formulas measure the straight-sided element instead."""
    import gmsh

    total = 0.0
    for etype in gmsh.model.mesh.getElementTypes(dim, tag):
        pts, w = gmsh.model.mesh.getIntegrationPoints(etype, rule)
        _, det, _ = gmsh.model.mesh.getJacobians(etype, pts, tag)
        total += float((np.asarray(det).reshape(-1, len(w)) @ np.asarray(w)).sum())
    return total


@contextmanager
def pinned_options(opts: dict[str, float]) -> Iterator[None]:
    """Set numeric gmsh options for the block; the previous values are restored afterwards."""
    import gmsh

    saved = {k: gmsh.option.getNumber(k) for k in opts}
    try:
        for k, v in opts.items():
            gmsh.option.setNumber(k, v)
        yield
    finally:
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)

