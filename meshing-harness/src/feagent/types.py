"""Shared data types. All dataclasses round-trip through to_dict()/from_dict()."""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

Vec3 = tuple[float, float, float]
BBox = tuple[Vec3, Vec3]
# FaceInfo.kind values (geometry.SURFACE_KINDS maps gmsh surface types onto them)
FACE_KINDS = ("plane", "cylinder", "cone", "sphere", "torus", "bspline", "revolution", "extrusion",
              "other")


def _v(x):
    return None if x is None else tuple(x)


def _bb(x) -> BBox:
    return (tuple(x[0]), tuple(x[1]))


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class _Base:
    def to_dict(self) -> dict:
        """Every field except the tessellation and private (``_``-prefixed) caches."""
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)  # type: ignore[arg-type]
                if f.name != "tessellation" and not f.name.startswith("_")}


@dataclass
class FaceInfo(_Base):
    tag: int
    kind: str
    area: float
    center: Vec3
    bbox: BBox
    normal: Vec3 | None
    radius: float | None
    axis: Vec3 | None
    axis_point: Vec3 | None
    concave: bool | None
    angle_span: float | None
    volumes: list[int]
    edges: list[int]
    neighbors: list[int]
    thickness: float | None = None  # low-percentile inward ray length (geometry.THICKNESS_PERCENTILE)
    thickness_min: float | None = None
    gap: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> FaceInfo:
        d = dict(d)
        d["center"] = tuple(d["center"])
        d["bbox"] = _bb(d["bbox"])
        for k in ("normal", "axis", "axis_point"):
            d[k] = _v(d.get(k))
        return cls(**d)


@dataclass
class EdgeInfo(_Base):
    tag: int
    kind: str
    length: float
    faces: list[int]
    bbox: BBox

    @classmethod
    def from_dict(cls, d: dict) -> EdgeInfo:
        d = dict(d)
        d["bbox"] = _bb(d["bbox"])
        return cls(**d)


@dataclass
class VolumeInfo(_Base):
    tag: int
    volume: float
    center: Vec3
    bbox: BBox
    faces: list[int]

    @classmethod
    def from_dict(cls, d: dict) -> VolumeInfo:
        d = dict(d)
        d["center"] = tuple(d["center"])
        d["bbox"] = _bb(d["bbox"])
        return cls(**d)


@dataclass
class Feature(_Base):
    kind: str  # "hole", "boss", "fillet", "round"
    faces: list[int]
    params: dict

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        return cls(**d)


@dataclass
class Hazard(_Base):
    kind: str
    entities: list[tuple[int, int]]
    value: float | None
    threshold: float | None
    location: Vec3 | None
    message: str
    suggestion: str

    @classmethod
    def from_dict(cls, d: dict) -> Hazard:
        d = dict(d)
        d["entities"] = [tuple(e) for e in d["entities"]]
        d["location"] = _v(d.get("location"))
        return cls(**d)


@dataclass
class Tessellation:
    """Not serialized."""
    points: np.ndarray
    triangles: np.ndarray
    face_tag: np.ndarray
    normals: np.ndarray


@dataclass
class GeometryModel(_Base):
    source: str
    units: str
    bbox: BBox
    diag: float
    ref_size: float
    volumes: list[VolumeInfo]
    faces: list[FaceInfo]
    edges: list[EdgeInfo]
    features: list[Feature]
    hazards: list[Hazard]
    tessellation: Tessellation | None = None
    _by_tag: dict[int, FaceInfo] | None = field(default=None, init=False, repr=False, compare=False)

    def face(self, tag: int) -> FaceInfo:
        """The face with ``tag`` (KeyError if none); a dict built on first use, rebuilt if the
        faces list changed length."""
        if self._by_tag is None or len(self._by_tag) != len(self.faces):
            self._by_tag = {f.tag: f for f in self.faces}
        try:
            return self._by_tag[tag]
        except KeyError:
            raise KeyError(tag) from None

    @classmethod
    def from_dict(cls, d: dict) -> GeometryModel:
        return cls(
            source=d["source"], units=d["units"], bbox=_bb(d["bbox"]),
            diag=d["diag"], ref_size=d["ref_size"],
            volumes=[VolumeInfo.from_dict(x) for x in d["volumes"]],
            faces=[FaceInfo.from_dict(x) for x in d["faces"]],
            edges=[EdgeInfo.from_dict(x) for x in d["edges"]],
            features=[Feature.from_dict(x) for x in d["features"]],
            hazards=[Hazard.from_dict(x) for x in d["hazards"]],
            tessellation=None,
        )


@dataclass
class Gate(_Base):
    id: str
    name: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    value: float | None
    threshold: float | None
    message: str
    hint: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Gate:
        return cls(**d)


@dataclass
class GroupResult(_Base):
    name: str
    dim: int
    tags: list[int]
    elements: int
    area: float | None

    @classmethod
    def from_dict(cls, d: dict) -> GroupResult:
        return cls(**d)


@dataclass
class MeshResult(_Base):
    ok: bool
    spec_path: str
    out_dir: str
    mesh_path: str | None
    exports: dict[str, str]
    groups: dict[str, GroupResult]
    n_nodes: int
    n_elements: int
    order: int
    size_scale: float
    elapsed_s: float
    errors: list[str]
    warnings: list[str]
    gates: list[Gate]
    geometry: GeometryModel | None

    @classmethod
    def from_dict(cls, d: dict) -> MeshResult:
        d = dict(d)
        d["groups"] = {k: GroupResult.from_dict(v) for k, v in d["groups"].items()}
        d["gates"] = [Gate.from_dict(g) for g in d["gates"]]
        g = d.get("geometry")
        d["geometry"] = GeometryModel.from_dict(g) if g is not None else None
        return cls(**d)


def overall_status(gates: list[Gate]) -> str:
    return "fail" if any(g.status == "fail" for g in gates) else "pass"
