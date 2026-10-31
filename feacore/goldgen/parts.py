"""Load a third-party STEP file, screen it, normalise its size, and describe its faces.

A part is accepted when it is exactly one closed solid with a sane shape and a coarse tet mesh
succeeds. Geometry is rescaled so the bounding-box diagonal is ``DIAG_M`` metres (ABC models
carry arbitrary units); the rescaled STEP is what agents get, so statement and CAD agree.
Faces keep their OCC tags, and every mesh of the part names each face ``f<tag>``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from feacore.geometry.gmsh_tools import gmsh_session

DIAG_M = 0.1                      # bounding-box diagonal after rescaling (m)
LIMITS = {
    "faces": (4, 300),            # too simple / too detailed to be an interesting benchmark
    "aspect": 25.0,               # longest / shortest bbox side
    "min_face_area_frac": 1e-6,   # sliver faces this small break meshing
}


@dataclass
class Face:
    tag: int
    kind: str                     # plane | cylinder | other
    area: float                   # m^2
    centroid: list[float]
    normal: list[float] | None = None    # outward, planes only
    axis: list[float] | None = None      # cylinders only
    radius: float | None = None


@dataclass
class Part:
    id: str
    source: str                   # original file name / dataset id
    scale: float                  # applied to the source units
    volume: float                 # m^3
    bbox: list[float]             # xmin, ymin, zmin, xmax, ymax, zmax (m)
    faces: list[Face] = field(default_factory=list)

    @property
    def diag(self) -> float:
        b = np.asarray(self.bbox)
        return float(np.linalg.norm(b[3:] - b[:3]))

    def face(self, tag: int) -> Face:
        return next(f for f in self.faces if f.tag == tag)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Part":
        d = json.loads(text)
        d["faces"] = [Face(**f) for f in d["faces"]]
        return cls(**d)


class Rejected(ValueError):
    pass


def import_scaled(step: str | Path, scale: float = 1.0) -> list[tuple[int, int]]:
    """Import into the current gmsh model, scaled about the origin (call inside a session)."""
    gmsh.option.setNumber("Geometry.OCCScaling", scale)
    ents = gmsh.model.occ.importShapes(str(step), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    return ents


def _outward_normal(tag: int, centroid: np.ndarray, vol_tag: int, diag: float) -> np.ndarray:
    """Outward unit normal of a planar face: the side where a small step leaves the solid."""
    uv = gmsh.model.getParametrization(2, tag, centroid.tolist())
    n = np.asarray(gmsh.model.getNormal(tag, uv), float)[:3]
    n /= np.linalg.norm(n)
    probe = gmsh.model.getClosestPoint(2, tag, centroid.tolist())[0]
    p = np.asarray(probe) + 1e-4 * diag * n
    inside = gmsh.model.isInside(3, vol_tag, p.tolist())
    return -n if inside else n


def screen(step: str | Path, part_id: str) -> Part:
    """Accept or reject (``Rejected``) a STEP file and describe the rescaled part."""
    with gmsh_session(part_id):
        try:
            ents = import_scaled(step)
        except Exception as e:  # noqa: BLE001 - OCC raises plain Exceptions
            raise Rejected(f"import failed: {e}") from e
        vols = [t for d, t in ents if d == 3]
        if len(vols) != 1:
            raise Rejected(f"expected one solid, found {len(vols)} solids / {len(ents)} entities")
        b = np.asarray(gmsh.model.getBoundingBox(3, vols[0]))
        sides = b[3:] - b[:3]
        diag0 = float(np.linalg.norm(sides))
        if diag0 <= 0 or sides.max() / max(sides.min(), 1e-300) > LIMITS["aspect"]:
            raise Rejected(f"bad bounding box {sides.tolist()}")
    scale = DIAG_M / diag0

    with gmsh_session(part_id):
        (_, vol), = [e for e in import_scaled(step, scale) if e[0] == 3]
        mass = gmsh.model.occ.getMass(3, vol)
        bbox = list(gmsh.model.getBoundingBox(3, vol))
        diag = float(np.linalg.norm(np.subtract(bbox[3:], bbox[:3])))
        if mass <= 0:
            raise Rejected("non-positive volume (open shell?)")
        tags = [t for _, t in gmsh.model.getBoundary([(3, vol)], oriented=False)]
        if not (LIMITS["faces"][0] <= len(tags) <= LIMITS["faces"][1]):
            raise Rejected(f"{len(tags)} faces outside {LIMITS['faces']}")
        total = sum(gmsh.model.occ.getMass(2, t) for t in tags)
        faces = []
        for t in tags:
            area = gmsh.model.occ.getMass(2, t)
            if area < LIMITS["min_face_area_frac"] * total:
                raise Rejected(f"sliver face {t} (area fraction {area / total:.1e})")
            c = np.asarray(gmsh.model.occ.getCenterOfMass(2, t))
            typ = gmsh.model.getType(2, t)
            f = Face(tag=t, kind="other", area=area, centroid=c.tolist())
            if typ == "Plane":
                f.kind = "plane"
                f.normal = _outward_normal(t, c, vol, diag).round(9).tolist()
            elif typ == "Cylinder":
                f.kind = "cylinder"
                f.axis, f.radius = _cylinder_axis(t)
            faces.append(f)
        # a coarse mesh must succeed, otherwise the fine ones will not
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 8)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:  # noqa: BLE001
            raise Rejected(f"coarse mesh failed: {e}") from e
        if len(gmsh.model.mesh.getElementsByType(4)[0]) == 0:
            raise Rejected("coarse mesh produced no tetrahedra")
    return Part(id=part_id, source=Path(step).name, scale=scale, volume=mass, bbox=bbox, faces=faces)


def _cylinder_axis(tag: int) -> tuple[list[float], float]:
    """Axis direction (the zero-curvature principal direction) and radius of a cylindrical face."""
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    uv = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]
    k1, k2, d1, d2 = gmsh.model.getPrincipalCurvatures(tag, uv)   # ordered by signed value
    (ka, da), (kb, _) = sorted([(abs(k1[0]), d1), (abs(k2[0]), d2)], key=lambda t: t[0])
    axis = np.asarray(da[:3], float)
    axis /= np.linalg.norm(axis)
    return axis.round(9).tolist(), float(1.0 / kb) if kb else math.nan


def export_scaled_step(step: str | Path, part: Part, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gmsh_session(part.id):
        import_scaled(step, part.scale)
        gmsh.write(str(out))
    return out
