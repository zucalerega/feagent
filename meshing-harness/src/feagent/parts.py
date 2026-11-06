"""Procedural test parts, written as STEP files (mm)."""
from __future__ import annotations

from pathlib import Path

import gmsh

from .gmsh_session import gmsh_session
from .gmsh_util import entity_tags


def _box(x, y, z, dx, dy, dz):
    return [(3, gmsh.model.occ.addBox(x, y, z, dx, dy, dz))]


def _cyl(x, y, z, dx, dy, dz, r):
    return [(3, gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r))]


def _fuse(a, b):
    out, _ = gmsh.model.occ.fuse(a, b)
    return out


def _cut(a, b):
    out, _ = gmsh.model.occ.cut(a, b)
    return out


def _cantilever() -> None:
    _box(0, 0, 0, 100, 10, 10)


def _plate_hole() -> None:
    _cut(_box(0, 0, 0, 200, 100, 5), _cyl(100, 50, -1, 0, 0, 7, 10))


def _bracket() -> None:
    occ = gmsh.model.occ
    body = _fuse(_box(0, 0, 0, 80, 40, 6), _box(0, 0, 0, 6, 40, 50))
    occ.synchronize()
    body = occ.fillet([body[0][1]], [_bracket_inner_edge()], [4])
    holes = [h for x, y in ((25, 10), (25, 30), (65, 10), (65, 30))
             for h in _cyl(x, y, -1, 0, 0, 8, 3.25)]
    _cut(body, holes + _cyl(-1, 20, 40, 8, 0, 0, 5))


def _bracket_inner_edge() -> int:
    """The straight edge x=6, z=6 along y (the inside corner of the L)."""
    want = (6, 0, 6, 6, 40, 6)  # xmin, ymin, zmin, xmax, ymax, zmax
    edges = [t for t in entity_tags(1)
             if all(abs(b - w) < 1e-6 for b, w in zip(gmsh.model.getBoundingBox(1, t), want))]
    if not edges:
        raise RuntimeError("bracket inner edge not found")
    return edges[-1]


def _thin_cup() -> None:
    _cut(_box(0, 0, 0, 60, 40, 20), _box(1, 1, 1, 58, 38, 20))


def _sliver() -> None:
    _fuse(_box(0, 0, 0, 50, 50, 10), _box(10, 10, 0, 20, 20, 10.05))


def _tee() -> None:
    _fuse(_cyl(0, 0, 0, 100, 0, 0, 10), _cyl(50, 0, 0, 0, 50, 0, 6))


def _bullet() -> None:
    _fuse(_cyl(5, 0, 0, 30, 0, 0, 5), [(3, gmsh.model.occ.addSphere(5, 0, 0, 5))])


_BUILDERS = {"cantilever": _cantilever, "plate_hole": _plate_hole, "bracket": _bracket,
             "thin_cup": _thin_cup, "sliver": _sliver, "tee": _tee, "bullet": _bullet}
PARTS: tuple[str, ...] = tuple(_BUILDERS)


def make_part(name: str, path: str | Path) -> Path:
    if name not in PARTS:
        raise ValueError(f"unknown part '{name}'; choose from {PARTS}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gmsh_session(name):
        _BUILDERS[name]()
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    return path


def make_all(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    return {n: make_part(n, out / f"{n}.step") for n in PARTS}
