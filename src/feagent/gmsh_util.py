"""Helpers for repeated gmsh model/mesh access patterns (call inside a gmsh session)."""
from __future__ import annotations

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


def element_blocks(dim: int, tag: int,
                   types: Iterable[int]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (type, element tags, node tags [n, nodes per element]) of the wanted types."""
    import gmsh

    wanted = set(types)
    for t, etags, enodes in zip(*gmsh.model.mesh.getElements(dim, tag)):
        if t in wanted:
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


def with_boundary_points(dim: int, tag: int) -> set[tuple[int, int]]:
    """The entity plus its boundary points (gmsh's recursive boundary stops at dim 0)."""
    import gmsh

    bnd = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False, recursive=True)
    return {(dim, tag), *((int(d), int(t)) for d, t in bnd)}


def element_qualities(tags: Iterable[int], measures: Iterable[str]) -> dict[str, np.ndarray]:
    """Per-element quality arrays keyed by gmsh measure name (empty arrays if no tags)."""
    import gmsh

    tl = [int(t) for t in tags]
    get = gmsh.model.mesh.getElementQualities
    return {m: np.asarray(get(tl, m), float) if tl else np.zeros(0) for m in measures}
