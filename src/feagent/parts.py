"""Procedural test parts, written as STEP files (mm)."""
from __future__ import annotations

from pathlib import Path

import gmsh

from .gmsh_session import gmsh_session

PARTS: tuple[str, ...] = ("cantilever", "plate_hole", "bracket", "thin_cup", "sliver", "tee", "bullet")


def _fuse(a, b):
    out, _ = gmsh.model.occ.fuse(a, b)
    return out


def _cut(a, b):
    out, _ = gmsh.model.occ.cut(a, b)
    return out


def _build(name: str) -> None:
    occ = gmsh.model.occ
    box = lambda x, y, z, dx, dy, dz: [(3, occ.addBox(x, y, z, dx, dy, dz))]
    if name == "cantilever":
        box(0, 0, 0, 100, 10, 10)
    elif name == "plate_hole":
        _cut(box(0, 0, 0, 200, 100, 5), [(3, occ.addCylinder(100, 50, -1, 0, 0, 7, 10))])
    elif name == "bracket":
        body = _fuse(box(0, 0, 0, 80, 40, 6), box(0, 0, 0, 6, 40, 50))
        occ.synchronize()
        edge = None
        for _, t in gmsh.model.getEntities(1):
            b = gmsh.model.getBoundingBox(1, t)
            if (abs(b[0] - 6) < 1e-6 and abs(b[3] - 6) < 1e-6 and abs(b[2] - 6) < 1e-6
                    and abs(b[5] - 6) < 1e-6 and abs(b[1]) < 1e-6 and abs(b[4] - 40) < 1e-6):
                edge = t
        if edge is None:
            raise RuntimeError("bracket inner edge not found")
        body = occ.fillet([body[0][1]], [edge], [4])
        holes = [(3, occ.addCylinder(x, y, -1, 0, 0, 8, 3.25))
                 for x, y in ((25, 10), (25, 30), (65, 10), (65, 30))]
        holes.append((3, occ.addCylinder(-1, 20, 40, 8, 0, 0, 5)))
        _cut(body, holes)
    elif name == "thin_cup":
        _cut(box(0, 0, 0, 60, 40, 20), box(1, 1, 1, 58, 38, 20))
    elif name == "sliver":
        _fuse(box(0, 0, 0, 50, 50, 10), box(10, 10, 0, 20, 20, 10.05))
    elif name == "tee":
        _fuse([(3, occ.addCylinder(0, 0, 0, 100, 0, 0, 10))],
              [(3, occ.addCylinder(50, 0, 0, 0, 50, 0, 6))])
    elif name == "bullet":
        _fuse([(3, occ.addCylinder(5, 0, 0, 30, 0, 0, 5))], [(3, occ.addSphere(5, 0, 0, 5))])
    occ.synchronize()


def make_part(name: str, path: str | Path) -> Path:
    if name not in PARTS:
        raise ValueError(f"unknown part '{name}'; choose from {PARTS}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gmsh_session(name):
        _build(name)
        gmsh.write(str(path))
    return path


def make_all(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    return {n: make_part(n, out / f"{n}.step") for n in PARTS}
