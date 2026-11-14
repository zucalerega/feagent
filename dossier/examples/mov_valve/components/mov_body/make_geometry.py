"""Build geometry.step of the example valve body (gmsh OpenCASCADE kernel).

    python dossier/examples/mov_valve/components/mov_body/make_geometry.py [--no-fillets]

CAD frame (metres): x = flow direction (inlet at -x, outlet at +x), z = stem axis (up), origin at the ball
centre. Dimensions below are in mm, as on drawing MOV-100-001 rev A of the example.

    shell      vertical cylinder, outer radius 28, cavity radius 22, floor and roof 8 thick (z -30 .. 42)
    nozzles    along x, outer radius 20, flow bore radius 12.5, flange faces at x = -75 and x = +75
    flanges    radius 42, 12 thick, six bolt holes of diameter 6.6 on a 64 pitch circle
    seat pockets  counterbores of radius 16 from the cavity to x = -27 and x = +27
    neck       tube, outer radius 12, stem bore radius 8, from the roof (z 42) to the pad (z 120)
    pad        56 x 56 x 8 actuator mounting face (top at z 128), four holes of diameter 5.5 at (+-20, +-20)
    fillets    radius 3 at the cavity floor and roof, radius 3 at the neck root, radius 4 under the pad
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh

MM = 1e-3
R_SHELL_O, R_CAV = 28.0, 22.0
Z_BOT, Z_FLOOR, Z_ROOF, Z_TOP = -30.0, -22.0, 34.0, 42.0
R_NOZ_O, R_BORE = 20.0, 12.5
X_FACE, T_FLANGE, R_FLANGE = 75.0, 12.0, 42.0
R_PCD, R_BOLT, N_BOLT = 32.0, 3.3, 6
R_POCKET, X_POCKET = 16.0, 27.0
R_NECK_O, R_STEM = 12.0, 8.0
Z_PAD, T_PAD, W_PAD = 120.0, 8.0, 56.0
PAD_HOLE_XY, R_PAD_HOLE = 20.0, 2.75
R_FILLET_CAVITY, R_FILLET_NECK, R_FILLET_PAD = 3.0, 3.0, 4.0


def _edges_where(pred) -> list[int]:
    """Curves of the model whose bounding box satisfies pred(xmin, ymin, zmin, xmax, ymax, zmax) (mm)."""
    out = []
    for _, t in gmsh.model.getEntities(1):
        b = [v / MM for v in gmsh.model.getBoundingBox(1, t)]
        if pred(*b):
            out.append(t)
    return out


def _circle_edges(z: float, r: float, tol: float = 0.05) -> list[int]:
    """Circular edges about the z axis at height z with radius r."""
    return _edges_where(lambda x0, y0, z0, x1, y1, z1: abs(z0 - z) < tol and abs(z1 - z) < tol
                        and abs(x1 - r) < tol and abs(x0 + r) < tol and abs(y1 - r) < tol and abs(y0 + r) < tol)


def build(fillets: bool = True) -> int:
    """Build the body in the current gmsh model; returns the volume tag."""
    occ = gmsh.model.occ
    cyl = lambda x, y, z, dx, dy, dz, r: (3, occ.addCylinder(x * MM, y * MM, z * MM, dx * MM, dy * MM, dz * MM, r * MM))  # noqa: E731

    solids = [
        cyl(0, 0, Z_BOT, 0, 0, Z_TOP - Z_BOT, R_SHELL_O),                              # shell
        cyl(-X_FACE, 0, 0, 2 * X_FACE, 0, 0, R_NOZ_O),                                 # both nozzles
        cyl(-X_FACE, 0, 0, T_FLANGE, 0, 0, R_FLANGE),                                  # inlet flange
        cyl(X_FACE - T_FLANGE, 0, 0, T_FLANGE, 0, 0, R_FLANGE),                        # outlet flange
        cyl(0, 0, Z_TOP, 0, 0, Z_PAD + T_PAD - Z_TOP, R_NECK_O),                       # neck
        (3, occ.addBox(-W_PAD / 2 * MM, -W_PAD / 2 * MM, Z_PAD * MM, W_PAD * MM, W_PAD * MM, T_PAD * MM)),   # pad
    ]
    body, _ = occ.fuse([solids[0]], solids[1:])
    cuts = [
        cyl(0, 0, Z_FLOOR, 0, 0, Z_ROOF - Z_FLOOR, R_CAV),                             # cavity
        cyl(-X_FACE - 1, 0, 0, 2 * X_FACE + 2, 0, 0, R_BORE),                          # flow bore
        cyl(-X_POCKET, 0, 0, 2 * X_POCKET, 0, 0, R_POCKET),                            # seat pockets
        cyl(0, 0, Z_ROOF - 1, 0, 0, Z_PAD + T_PAD - Z_ROOF + 2, R_STEM),               # stem bore
    ]
    for k in range(N_BOLT):
        a = 2 * math.pi * k / N_BOLT
        y, z = R_PCD * math.cos(a), R_PCD * math.sin(a)
        cuts.append(cyl(-X_FACE - 1, y, z, T_FLANGE + 2, 0, 0, R_BOLT))
        cuts.append(cyl(X_FACE - T_FLANGE - 1, y, z, T_FLANGE + 2, 0, 0, R_BOLT))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(cyl(sx * PAD_HOLE_XY, sy * PAD_HOLE_XY, Z_PAD - 1, 0, 0, T_PAD + 2, R_PAD_HOLE))
    body, _ = occ.cut(body, cuts)
    occ.synchronize()
    vol = body[0][1]
    if fillets:
        # one fillet at a time: OpenCASCADE renumbers the edges after each
        for (z, r_edge), r in (((Z_FLOOR, R_CAV), R_FILLET_CAVITY),        # cavity floor corner
                               ((Z_TOP, R_NECK_O), R_FILLET_NECK),         # neck root on the roof
                               ((Z_PAD, R_NECK_O), R_FILLET_PAD)):         # neck under the pad
            edges = _circle_edges(z, r_edge)
            if not edges:
                raise RuntimeError(f"fillet edge not found: circle of radius {r_edge} mm at z = {z} mm")
            out = occ.fillet([vol], edges, [r * MM])
            occ.synchronize()
            vol = [t for d, t in out if d == 3][0]
    return vol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fillets", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("geometry.step")))
    a = ap.parse_args()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mov_body")
    vol = build(fillets=not a.no_fillets)
    mass = gmsh.model.occ.getMass(3, vol)
    faces = gmsh.model.getBoundary([(3, vol)], oriented=False)
    gmsh.write(a.out)
    gmsh.finalize()
    print(f"{a.out}: volume {mass * 1e9:.0f} mm^3, {len(faces)} faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
