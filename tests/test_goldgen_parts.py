"""goldgen face descriptions: outward normals of planar faces whose centroid lies off the face.

abc_00000003's boss shoulders f18 / f19 are rings around a pin: the centroid is on the pin's axis,
inside the solid on both sides, so probing there gave an arbitrary sign (both were stated inward).
Its 1 mm ring f24 under the bore is thin enough that every coarse triangle's centroid is in the bore.
"""

from __future__ import annotations

import gmsh
import numpy as np
import pytest

from feacore.geometry.gmsh_tools import gmsh_session


def _plane_at(z: float) -> list[int]:
    """Planar faces lying in the plane at height z (OCC pads bounding boxes by ~1e-7)."""
    out = []
    for _, t in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, t)
        if gmsh.model.getType(2, t) == "Plane" and abs(b[2] - z) < 1e-6 and abs(b[5] - z) < 1e-6:
            out.append(t)
    return out


def _meshed(diag: float) -> None:
    gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 8)
    gmsh.model.mesh.generate(3)


def test_ring_around_a_pin_faces_away_from_the_block():
    """A block with a pin standing on its top face: the top face is a ring whose centroid is on the pin's
    axis, inside the solid above (pin) and below (block)."""
    from feacore.goldgen.parts import _outward_normal, face_points

    with gmsh_session("pin"):
        occ = gmsh.model.occ
        block = occ.addBox(0, 0, 0, 1, 1, 0.2)
        pin = occ.addCylinder(0.5, 0.5, 0.2, 0, 0, 0.3, 0.2)
        (vol,), _ = occ.fuse([(3, block)], [(3, pin)])
        occ.synchronize()
        _meshed(1.5)
        (ring,) = _plane_at(0.2)
        c = np.asarray(gmsh.model.occ.getCenterOfMass(2, ring))
        assert np.allclose(c, [0.5, 0.5, 0.2])
        # the centroid is no point of the face, and the solid is on both sides of it
        assert not gmsh.model.isInside(2, ring, c.tolist())
        assert all(gmsh.model.isInside(3, vol[1], (c + s * np.array([0, 0, 1e-3])).tolist()) for s in (1, -1))
        pts = face_points(ring)
        assert pts and all(gmsh.model.isInside(2, ring, p.tolist()) for p in pts)
        assert np.allclose(_outward_normal(ring, vol[1], 1.5), [0, 0, 1])
        (top,) = _plane_at(0.5)                               # the pin's end: centroid on the face
        assert np.allclose(_outward_normal(top, vol[1], 1.5), [0, 0, 1])
        (bottom,) = _plane_at(0.0)
        assert np.allclose(_outward_normal(bottom, vol[1], 1.5), [0, 0, -1])


def test_thin_ring_is_oriented_from_points_on_it():
    """A tube with a 3 % wall: coarse triangles of its end rings span the hole's edge, their centroids lie
    in the hole (outside the solid on both sides), so the probes must be points on the ring."""
    from feacore.goldgen.parts import _outward_normal, face_points

    with gmsh_session("tube"):
        occ = gmsh.model.occ
        outer = occ.addCylinder(0, 0, 0, 0, 0, 1, 1.0)
        inner = occ.addCylinder(0, 0, -0.1, 0, 0, 1.2, 0.97)
        (vol,), _ = occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        _meshed(2.3)
        for z, nz in ((0.0, -1), (1.0, 1)):
            (ring,) = _plane_at(z)
            pts = face_points(ring)
            assert pts and all(0.97 < np.hypot(*p[:2]) < 1.0 for p in pts)
            assert np.allclose(_outward_normal(ring, vol[1], 2.3), [0, 0, nz])


def test_screen_states_the_shoulders_of_abc_00000003_outward():
    """The gold part itself: f18 at x = -19.5 mm faces -x, f19 at x = +19.5 mm faces +x."""
    from feacore.goldgen.parts import plane_normals
    from feacore.problems.gold import load_all

    p = next((p for p in load_all()[0] if p.family == "abc_00000003"), None)
    if p is None:
        pytest.skip("abc_00000003 not in the gold root")
    got = plane_normals(p.step_path, [18, 19])
    assert np.allclose(got[18], [-1, 0, 0]) and np.allclose(got[19], [1, 0, 0])
    faces = {f["tag"]: f for f in p.part["faces"]}
    assert np.allclose(faces[18]["normal"], got[18]) and np.allclose(faces[19]["normal"], got[19])
