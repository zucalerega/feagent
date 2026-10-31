"""YAML mesh spec: strict parsing and validation. Errors name the field path."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any, Callable

import yaml

from feagent.errors import SpecError

_FALLBACK_QUERY_KEYS = (
    "ids", "kind", "normal", "axis", "tol_deg", "radius", "radius_tol", "radius_min",
    "radius_max", "area_min", "area_max", "concave", "at", "box", "near", "within",
    "feature", "adjacent_to", "largest", "smallest", "any", "all", "not",
)
_EXPORTS = ("msh", "msh22", "inp", "vtk", "unv", "su2")
_OPTIMIZE = ("none", "default", "netgen")
_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


@dataclass
class GroupSpec:
    name: str
    select: dict
    expect: int | tuple[int, int] | None = None


@dataclass
class RefineSpec:
    size: float
    distance: float
    group: str | None = None
    select: dict | None = None


@dataclass
class SizeSpec:
    global_: float
    min: float = 0.0
    curvature: int = 12
    thin_wall_layers: int = 2


@dataclass
class DomainSpec:
    type: str = "internal"
    flow_axis: str = "+x"
    upstream: float = 3.0
    downstream: float = 6.0
    lateral: float = 2.0


@dataclass
class QualitySpec:
    min_sicn: float = 0.1
    min_gamma: float = 0.05
    max_volume_error: float = 0.01
    max_area_error: float = 0.02
    max_non_ortho: float = 70.0
    max_skewness: float = 4.0
    thin_wall_min_elements: int = 2


@dataclass
class FeaLoad:
    group: str
    force: tuple[float, float, float]


@dataclass
class FeaBcs:
    """Structural boundary conditions declared for the downstream solver."""
    fixed: list[str]
    loads: list[FeaLoad]
    E: float = 210000.0
    nu: float = 0.3


@dataclass
class CfdBcs:
    """Flow boundary conditions declared for the downstream solver."""
    inlet: list[str]
    outlet: list[str]
    velocity: float = 1.0


@dataclass
class MeshSpec:
    path: Path
    name: str
    geometry: Path
    analysis: str
    heal: bool
    domain: DomainSpec
    groups: dict[str, GroupSpec]
    default_group: str | None
    size: SizeSpec
    refine: list[RefineSpec]
    order: int
    optimize: str
    algorithm_2d: int
    algorithm_3d: int
    export: list[str]
    quality: QualitySpec
    bcs_fea: FeaBcs | None
    bcs_cfd: CfdBcs | None
    output: Path = field(default_factory=lambda: Path("out"))

    def out_dir(self) -> Path:
        return (self.path.parent / self.output).resolve()


def _query_keys() -> tuple[str, ...]:
    try:
        from feagent.select import QUERY_KEYS

        return tuple(QUERY_KEYS)
    except ImportError:
        return _FALLBACK_QUERY_KEYS


def _err(path: str, msg: str, hint: str = "") -> SpecError:
    return SpecError(f"{path}: {msg}" if path else msg, hint)


def _join(p: str, k: str) -> str:
    return f"{p}.{k}" if p else k


def _mapping(v: Any, path: str, allowed: tuple[str, ...]) -> dict:
    if v is None:
        v = {}
    if not isinstance(v, dict):
        raise _err(path, f"expected a mapping, got {type(v).__name__}")
    for k in v:
        if k not in allowed:
            raise _err(_join(path, str(k)), "unknown key", f"allowed keys: {', '.join(allowed)}")
    return v


def _num(v: Any, path: str, *, positive: bool = False, nonneg: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(path, f"expected a number, got {v!r}")
    if positive and v <= 0:
        raise _err(path, f"must be > 0, got {v}")
    if nonneg and v < 0:
        raise _err(path, f"must be >= 0, got {v}")
    return float(v)


def _int(v: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _err(path, f"expected an integer, got {v!r}")
    if minimum is not None and v < minimum:
        raise _err(path, f"must be >= {minimum}, got {v}")
    return v


def _choice(v: Any, path: str, options: tuple) -> Any:
    if v not in options:
        raise _err(path, f"invalid value {v!r}", f"allowed: {', '.join(map(str, options))}")
    return v


def _str_list(v: Any, path: str) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise _err(path, "expected a list of group names")
    return list(v)


def _list(v: Any, path: str) -> list:
    if not isinstance(v, list):
        raise _err(path, "expected a list")
    return v


def _text(v: Any, path: str, what: str = "a non-empty string") -> str:
    if not isinstance(v, str) or not v:
        raise _err(path, f"expected {what}")
    return v


def _require(d: dict, path: str, keys: tuple[str, ...], hint: str = "") -> None:
    for k in keys:
        if k not in d:
            raise _err(_join(path, k), "required field missing", hint)


def _group_ref(name: Any, path: str, groups: dict) -> str:
    if not isinstance(name, str) or name not in groups:
        raise _err(path, f"unknown group {name!r}",
                   f"defined groups: {', '.join(groups) or '(none)'}")
    return name


def _group_refs(v: Any, path: str, groups: dict) -> list[str]:
    names = _str_list(v, path)
    for i, n in enumerate(names):
        _group_ref(n, f"{path}[{i}]", groups)
    return names


def _check_select(sel: Any, path: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(sel, dict) or not sel:
        raise _err(path, "expected a non-empty mapping of query keys")
    for k in sel:
        if k not in keys:
            raise _err(_join(path, str(k)), "unknown query key", f"allowed keys: {', '.join(keys)}")
    return sel


# --- scalar sections: key -> validator(value, path) -------------------------

_Validator = Callable[[Any, str], Any]


def _values(d: dict, path: str, table: dict[str, _Validator]) -> dict:
    """Validated values of the keys of ``table`` present in ``d`` (in table order)."""
    return {k: check(d[k], _join(path, k)) for k, check in table.items() if k in d}


def _section(v: Any, path: str, table: dict[str, _Validator],
             required: tuple[str, ...] = ()) -> dict:
    """A mapping whose keys are exactly those of ``table``, validated."""
    d = _mapping(v, path, tuple(table))
    _require(d, path, required)
    return _values(d, path, table)


def _poisson(v: Any, path: str) -> float:
    nu = _num(v, path)
    if not 0 <= nu < 0.5:
        raise _err(path, f"must be in [0, 0.5), got {nu}")
    return nu


def _element_order(v: Any, path: str) -> int:
    return _choice(_int(v, path), path, (1, 2))


_positive = partial(_num, positive=True)
_nonneg = partial(_num, nonneg=True)
_count = partial(_int, minimum=0)
_DOMAIN: dict[str, _Validator] = {
    "type": partial(_choice, options=("internal", "external")),
    "flow_axis": partial(_choice, options=_AXES),
    "upstream": _positive, "downstream": _positive, "lateral": _positive,
}
_SIZE: dict[str, _Validator] = {"global": _positive, "min": _nonneg, "curvature": _count,
                                "thin_wall_layers": _count}
_ELEMENT: dict[str, _Validator] = {"order": _element_order,
                                   "optimize": partial(_choice, options=_OPTIMIZE)}
_ALGORITHM: dict[str, _Validator] = {"mesh2d": partial(_int, minimum=1),
                                     "mesh3d": partial(_int, minimum=1)}
_QUALITY: dict[str, _Validator] = {f.name: _count if isinstance(f.default, int) else _nonneg
                                   for f in fields(QualitySpec)}
_FEA_SCALARS: dict[str, _Validator] = {"E": _positive, "nu": _poisson}


# --- sections: parser(data, parsed so far) -> MeshSpec fields ---------------

def _parse_name(data: dict, done: dict) -> dict:
    return {"name": _text(data.get("name", done["path"].resolve().parent.name), "name")}


def _parse_geometry(data: dict, done: dict) -> dict:
    _require(data, "", ("geometry",))
    if not isinstance(data["geometry"], str):
        raise _err("geometry", "expected a path string")
    geometry = (done["path"].parent / data["geometry"]).resolve()
    if not geometry.is_file():
        raise _err("geometry", f"file not found: {geometry}", "path is relative to the spec file")
    return {"geometry": geometry}


def _parse_analysis(data: dict, done: dict) -> dict:
    _require(data, "", ("analysis",), "allowed: fea, cfd")
    return {"analysis": _choice(data["analysis"], "analysis", ("fea", "cfd"))}


def _parse_heal(data: dict, done: dict) -> dict:
    heal = data.get("heal", False)
    if not isinstance(heal, bool):
        raise _err("heal", f"expected true/false, got {heal!r}")
    return {"heal": heal}


def _parse_domain(data: dict, done: dict) -> dict:
    return {"domain": DomainSpec(**_section(data.get("domain"), "domain", _DOMAIN))}


def _parse_groups(data: dict, done: dict) -> dict:
    gd = data.get("groups")
    gd = {} if gd is None else gd
    if not isinstance(gd, dict):
        raise _err("groups", "expected a mapping of group name to {select, expect}")
    qkeys = _query_keys()
    return {"groups": {str(n): _parse_group(str(n), g, qkeys) for n, g in gd.items()}}


def _parse_group(name: str, g: Any, qkeys: tuple[str, ...]) -> GroupSpec:
    gp = f"groups.{name}"
    g = _mapping(g, gp, ("select", "expect"))
    _require(g, gp, ("select",))
    sel = _check_select(g["select"], f"{gp}.select", qkeys)
    return GroupSpec(name=name, select=sel, expect=_parse_expect(g.get("expect"), f"{gp}.expect"))


def _parse_expect(ex: Any, path: str) -> int | tuple[int, int] | None:
    if ex is None:
        return None
    if not isinstance(ex, list):
        return _int(ex, path, minimum=0)
    if len(ex) != 2:
        raise _err(path, "range must be [min, max]")
    lo, hi = _int(ex[0], f"{path}[0]", minimum=0), _int(ex[1], f"{path}[1]", minimum=0)
    if lo > hi:
        raise _err(path, f"min {lo} > max {hi}")
    return (lo, hi)


def _parse_default_group(data: dict, done: dict) -> dict:
    dg = data.get("default_group", "walls" if done["analysis"] == "cfd" else None)
    if dg is not None:
        _text(dg, "default_group", "a non-empty string or null")
        if dg in done["groups"]:
            raise _err("default_group", f"{dg!r} collides with a group of the same name",
                       _collision_hint(dg, "default_group" in data, done["analysis"]))
    return {"default_group": dg}


def _collision_hint(dg: str, explicit: bool, analysis: str) -> str:
    if explicit or analysis != "cfd":
        return "rename default_group or the group"
    return (f"analysis: cfd sets default_group to {dg!r} (all faces not in a group) unless you "
            f"set it: rename your group, set default_group: <other name>, or set "
            f"default_group: null to keep your own {dg!r} group and leave no default")


def _parse_size(data: dict, done: dict) -> dict:
    _require(data, "", ("size",), "e.g. size: {global: 3.0}")
    s = _section(data["size"], "size", _SIZE, required=("global",))
    return {"size": SizeSpec(global_=s.pop("global"), **s)}


def _parse_refine(data: dict, done: dict) -> dict:
    items = data.get("refine")
    items = _list([] if items is None else items, "refine")
    qkeys = _query_keys()
    return {"refine": [_parse_refine_item(r, f"refine[{i}]", done["groups"], qkeys)
                       for i, r in enumerate(items)]}


def _parse_refine_item(r: Any, rp: str, groups: dict, qkeys: tuple[str, ...]) -> RefineSpec:
    r = _mapping(r, rp, ("group", "select", "size", "distance"))
    if ("group" in r) == ("select" in r):
        raise _err(rp, "exactly one of 'group' or 'select' is required")
    _require(r, rp, ("size", "distance"))
    by_group = "group" in r
    grp = _group_ref(r["group"], f"{rp}.group", groups) if by_group else None
    sel = None if by_group else _check_select(r["select"], f"{rp}.select", qkeys)
    return RefineSpec(size=_positive(r["size"], f"{rp}.size"),
                      distance=_positive(r["distance"], f"{rp}.distance"), group=grp, select=sel)


def _parse_element(data: dict, done: dict) -> dict:
    e = _section(data.get("element"), "element", _ELEMENT)
    return {"order": e.get("order", 1), "optimize": e.get("optimize", "netgen")}


def _parse_algorithm(data: dict, done: dict) -> dict:
    a = _section(data.get("algorithm"), "algorithm", _ALGORITHM)
    return {"algorithm_2d": a.get("mesh2d", 6), "algorithm_3d": a.get("mesh3d", 1)}


def _parse_export(data: dict, done: dict) -> dict:
    export = data.get("export", ["msh"])
    if not isinstance(export, list) or not export:
        raise _err("export", "expected a non-empty list", f"allowed: {', '.join(_EXPORTS)}")
    return {"export": [_choice(x, f"export[{i}]", _EXPORTS) for i, x in enumerate(export)]}


def _parse_quality(data: dict, done: dict) -> dict:
    return {"quality": QualitySpec(**_section(data.get("quality"), "quality", _QUALITY))}


def _parse_bcs(data: dict, done: dict) -> dict:
    bc = _mapping(data.get("bcs"), "bcs", ("fea", "cfd"))
    return {"bcs_fea": _bcs_of(bc, "fea", done, _parse_fea),
            "bcs_cfd": _bcs_of(bc, "cfd", done, _parse_cfd)}


def _bcs_of(bc: dict, kind: str, done: dict, parse: Callable[[Any, dict], Any]) -> Any:
    if bc.get(kind) is None:
        return None
    if done["analysis"] != kind:
        raise _err(f"bcs.{kind}", f"only allowed with analysis: {kind}")
    return parse(bc[kind], done["groups"])


def _parse_fea(v: Any, groups: dict) -> FeaBcs:
    fd = _mapping(v, "bcs.fea", ("E", "nu", "fixed", "loads"))
    fixed = _group_refs(fd.get("fixed", []), "bcs.fea.fixed", groups)
    loads = [_parse_load(ld, f"bcs.fea.loads[{i}]", groups)
             for i, ld in enumerate(_list(fd.get("loads", []), "bcs.fea.loads"))]
    return FeaBcs(fixed=fixed, loads=loads, **_values(fd, "bcs.fea", _FEA_SCALARS))


def _parse_load(ld: Any, lp: str, groups: dict) -> FeaLoad:
    ld = _mapping(ld, lp, ("group", "force"))
    _require(ld, lp, ("group", "force"))
    group = _group_ref(ld["group"], f"{lp}.group", groups)
    f = ld["force"]
    if not isinstance(f, list) or len(f) != 3:
        raise _err(f"{lp}.force", "expected [fx, fy, fz]")
    return FeaLoad(group, tuple(_num(c, f"{lp}.force[{j}]") for j, c in enumerate(f)))


def _parse_cfd(v: Any, groups: dict) -> CfdBcs:
    cd = _mapping(v, "bcs.cfd", ("inlet", "outlet", "velocity"))
    inlet = _group_refs(cd.get("inlet", []), "bcs.cfd.inlet", groups)
    outlet = _group_refs(cd.get("outlet", []), "bcs.cfd.outlet", groups)
    return CfdBcs(inlet=inlet, outlet=outlet, **_values(cd, "bcs.cfd", {"velocity": _positive}))


def _parse_output(data: dict, done: dict) -> dict:
    return {"output": Path(_text(data.get("output", "out"), "output", "a directory name string"))}


_SECTIONS = (_parse_name, _parse_geometry, _parse_analysis, _parse_heal, _parse_domain,
             _parse_groups, _parse_default_group, _parse_size, _parse_refine, _parse_element,
             _parse_algorithm, _parse_export, _parse_quality, _parse_bcs, _parse_output)
_TOP = ("name", "geometry", "analysis", "heal", "domain", "groups", "default_group", "size",
        "refine", "element", "algorithm", "export", "quality", "bcs", "output")
_RENAMED = {"solve": "bcs: (boundary conditions for the solver; same fea/cfd sub-keys)"}


def load_spec(path: str | Path) -> MeshSpec:
    """Read and validate a YAML mesh spec; raises SpecError naming the bad field."""
    p = Path(path)
    if not p.is_file():
        raise _err("", f"spec file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise _err("", f"invalid YAML in {p}: {exc}") from exc
    return parse_spec(data, p)


def parse_spec(data: dict, path: Path) -> MeshSpec:
    """Validate spec ``data`` (paths relative to ``path``), section by section in order."""
    parsed: dict[str, Any] = {"path": Path(path)}
    _reject_renamed(data)
    data = _mapping(data, "", _TOP)
    for parse in _SECTIONS:
        parsed.update(parse(data, parsed))
    return MeshSpec(**parsed)


def _reject_renamed(data: Any) -> None:
    for old, new in _RENAMED.items():
        if isinstance(data, dict) and old in data:
            raise _err(old, "unknown key (renamed)", f"use {new}")
