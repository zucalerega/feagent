"""Read a .msh file into numpy arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np

from .gmsh_session import gmsh_session

_TET = {4: 4, 11: 4}  # gmsh element type -> corner count (tet4, tet10)


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
    with gmsh_session():
        gmsh.open(str(path))
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        idx = np.full(int(ntags.max()) + 1, -1, dtype=np.int64) if len(ntags) else np.zeros(1, np.int64)
        idx[ntags] = np.arange(len(ntags))

        tets, tags, vol = [], [], []
        order = 1
        for _, vt in gmsh.model.getEntities(3):
            types, etags, enodes = gmsh.model.mesh.getElements(3, vt)
            for t, et, en in zip(types, etags, enodes):
                if t not in _TET:
                    continue
                if t == 11:
                    order = 2
                nn = gmsh.model.mesh.getElementProperties(t)[3]
                arr = idx[np.asarray(en).reshape(-1, nn)[:, :4]]
                tets.append(arr)
                tags.append(np.asarray(et))
                vol.append(np.full(len(et), vt))

        def cat(parts, shape, dt):
            return np.concatenate(parts).astype(dt) if parts else np.zeros(shape, dt)

        tris: dict[str, np.ndarray] = {}
        tri_face: dict[str, np.ndarray] = {}
        vgroups: dict[str, list[int]] = {}
        for dim, pg in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, pg)
            ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(dim, pg)]
            if dim == 3:
                vgroups[name] = ents
            elif dim == 2:
                tp, fp = [], []
                for e in ents:
                    types, _, enodes = gmsh.model.mesh.getElements(2, e)
                    for t, en in zip(types, enodes):
                        if t not in (2, 9):
                            continue
                        nn = gmsh.model.mesh.getElementProperties(t)[3]
                        a = idx[np.asarray(en).reshape(-1, nn)[:, :3]]
                        tp.append(a)
                        fp.append(np.full(len(a), e))
                tris[name] = cat(tp, (0, 3), np.int64)
                tri_face[name] = cat(fp, (0,), np.int64)
        return MeshData(
            points=coords, tets=cat(tets, (0, 4), np.int64),
            tet_tags=cat(tags, (0,), np.int64), tet_volume=cat(vol, (0,), np.int64),
            tris=tris, tri_face=tri_face, volume_groups=vgroups, order=order,
        )
