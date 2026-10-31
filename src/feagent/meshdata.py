"""Read a .msh file into numpy arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np

from . import gmsh_util as gu
from .gmsh_session import gmsh_session


@dataclass
class MeshData:
    points: np.ndarray
    tets: np.ndarray
    tet_tags: np.ndarray
    tet_volume: np.ndarray
    tris: dict[str, np.ndarray]
    tri_face: dict[str, np.ndarray]
    volume_groups: dict[str, list[int]]
    order: int


def read_msh(path: str | Path) -> MeshData:
    """Read nodes, linear tet corners, surface-group triangles and volume groups."""
    with gmsh_session():
        gmsh.open(str(path))
        _, points, lookup = gu.node_table()
        tets, tet_tags, tet_volume, order = _read_tets(lookup)
        tris, tri_face, volume_groups = _read_groups(lookup)
    return MeshData(points=points, tets=tets, tet_tags=tet_tags, tet_volume=tet_volume,
                    tris=tris, tri_face=tri_face, volume_groups=volume_groups, order=order)


def _cat(parts: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(shape, np.int64)


def _read_tets(lookup: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(corner rows, element tags, owning volume per tet, element order)."""
    rows, tags, vols, order = [], [], [], 1
    for vt in gu.entity_tags(3):
        for etype, etags, nodes in gu.element_blocks(3, vt, gu.TET_TYPES):
            order = 2 if etype == 11 else order
            rows.append(lookup[nodes[:, :4]])
            tags.append(etags)
            vols.append(np.full(len(etags), vt))
    return _cat(rows, (0, 4)), _cat(tags, (0,)), _cat(vols, (0,)), order


def _read_groups(lookup: np.ndarray) -> tuple[dict, dict, dict]:
    """Surface groups -> (triangle corner rows, CAD face per triangle); volume groups -> tags."""
    tris: dict[str, np.ndarray] = {}
    tri_face: dict[str, np.ndarray] = {}
    volume_groups: dict[str, list[int]] = {}
    for dim, name, ents in gu.physical_groups():
        if dim == 3:
            volume_groups[name] = ents
        elif dim == 2:
            tris[name], tri_face[name] = _group_triangles(ents, lookup)
    return tris, tri_face, volume_groups


def _group_triangles(faces: list[int], lookup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = [gu.corner_rows(2, f, gu.TRI_TYPES, 3, lookup) for f in faces]
    owner = [np.full(len(r), f) for f, r in zip(faces, rows)]
    return _cat(rows, (0, 3)), _cat(owner, (0,))
