"""Build flow_domain.step: the fluid volume of the OPEN example valve (gmsh OpenCASCADE kernel).

    python dossier/examples/mov_valve/components/mov_body/make_flow_domain.py

CAD frame (metres) as geometry.step: x = flow direction (inlet at -x), z = stem axis, origin at the ball
centre. Dimensions in mm. The body dimensions are those of make_geometry.py (drawing MOV-100-001 rev A).

What the volume is
    The wetted volume of the open valve from the inlet flange face (x = -75) to the outlet flange face
    (x = +75) through the bore of the ball: a 25 mm bore with one annular groove at each seat pocket.

The internal parts (ball, seat rings) are not on the body drawing. ASSUMED dimensions (EL-INTERNALS):
    ball            sphere of radius 20 (the cavity has radius 22), full-port bore of radius 12.5 along x
    seat rings      inner radius 12.5 (full port), from the pocket shoulder (|x| = 27) to |x| = 17
Thus the part of each seat pocket (radius 16) that is open to the flow is the annulus between the face of the
seat ring (|x| = 17) and the sphere of the ball (|x| = sqrt(20^2 - r^2), 12 to 15.6 mm).

Simplifications (the flow engine analyses a stepped bore of revolution: planes across the axis, cylinders
about the axis)
    1. The open annulus of each pocket is a groove of radius 16 from |x| = 17 to |x| = X_GROOVE. X_GROOVE gives
       the groove the volume of the true annulus (the sphere of the ball becomes a plane).
    2. The cavity around the ball (radius 22 about the stem axis) is not in the volume. In the open position
       the ball floats on the downstream seat, thus no flow goes around the ball; the cavity is dead fluid that
       touches the groove only through the clearance between ball and pocket wall.
    3. The ball bore and the seat bores have exactly the diameter of the nozzle bores and are on one axis:
       no step at the ball, no offset, no wear.
    4. The fillets of the body are not wetted by the flow in the open position and are not in the volume.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gmsh

MM = 1e-3
X_FACE = 75.0           # flange faces (make_geometry.py)
R_BORE = 12.5           # flow bore, ball bore and seat bore
R_POCKET = 16.0         # seat pocket (make_geometry.py)
X_SEAT_FACE = 17.0      # ASSUMED: face of the seat ring toward the ball
R_BALL = 20.0           # ASSUMED: ball radius


def groove_end() -> float:
    """|x| of the plane that gives the groove the volume of the open annulus between seat face and ball."""
    n = 2000
    vol = 0.0
    dr = (R_POCKET - R_BORE) / n
    for i in range(n):
        r = R_BORE + (i + 0.5) * dr
        vol += 2.0 * math.pi * r * (X_SEAT_FACE - math.sqrt(R_BALL**2 - r * r)) * dr
    return X_SEAT_FACE - vol / (math.pi * (R_POCKET**2 - R_BORE**2))


X_GROOVE = groove_end()


def build() -> int:
    occ = gmsh.model.occ
    bore = occ.addCylinder(-X_FACE * MM, 0, 0, 2 * X_FACE * MM, 0, 0, R_BORE * MM)
    w = X_SEAT_FACE - X_GROOVE
    grooves = [(3, occ.addCylinder(-X_SEAT_FACE * MM, 0, 0, w * MM, 0, 0, R_POCKET * MM)),
               (3, occ.addCylinder(X_GROOVE * MM, 0, 0, w * MM, 0, 0, R_POCKET * MM))]
    out, _ = occ.fuse([(3, bore)], grooves)
    occ.synchronize()
    return out[0][1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("flow_domain.step")))
    a = ap.parse_args()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mov_body_flow")
    vol = build()
    mass = gmsh.model.occ.getMass(3, vol)
    faces = gmsh.model.getBoundary([(3, vol)], oriented=False)
    gmsh.write(a.out)
    gmsh.finalize()
    print(f"{a.out}: volume {mass * 1e9:.0f} mm^3, {len(faces)} faces, groove from {X_SEAT_FACE} to "
          f"{X_GROOVE:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
