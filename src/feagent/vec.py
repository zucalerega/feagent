"""Small 3-vector and bounding-box helpers.

Inputs may be tuples, lists or numpy arrays; outputs are plain floats/tuples.
"""
from __future__ import annotations

import math
from typing import Sequence

from .types import BBox, Vec3

_AXIS = {"x": 0, "y": 1, "z": 2}


def vec3(v: Sequence[float]) -> Vec3:
    """Plain-float tuple of a 3-vector."""
    return (float(v[0]), float(v[1]), float(v[2]))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle between two non-zero vectors, in degrees [0, 180]."""
    c = _dot(a, b) / (_norm(a) * _norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def axis_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle between two lines (direction sign ignored), in degrees [0, 90]."""
    ang = angle_deg(a, b)
    return min(ang, 180.0 - ang)


def box_dist(p: Sequence[float], bb: BBox) -> float:
    """Distance from point ``p`` to box ``bb`` (0 inside)."""
    return math.sqrt(sum(max(bb[0][i] - p[i], 0.0, p[i] - bb[1][i]) ** 2 for i in range(3)))


def bbox_diag(bb: BBox) -> float:
    return dist(bb[0], bb[1])


def extreme_gap(inner: BBox, outer: BBox, side: str) -> float:
    """How far ``inner`` is from lying on ``outer``'s extreme plane ``side`` (e.g. "xmin").

    Zero when inner's whole extent along that axis sits on the plane.
    """
    i = _AXIS[side[0]]
    if side.endswith("min"):
        return abs(inner[1][i] - outer[0][i])
    return abs(inner[0][i] - outer[1][i])
