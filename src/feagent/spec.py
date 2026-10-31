"""YAML mesh spec: strict parsing and validation. Errors name the field path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class FeaSolveSpec:
    fixed: list[str]
    loads: list[FeaLoad]
    E: float = 210000.0
    nu: float = 0.3


@dataclass
class CfdSolveSpec:
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
    fea: FeaSolveSpec | None
    cfd: CfdSolveSpec | None
    output: Path = field(default_factory=lambda: Path("out"))

    def out_dir(self) -> Path:
        return (self.path.parent / self.output).resolve()


def _query_keys() -> tuple[str, ...]:
    try:
        from feagent.select import QUERY_KEYS

        return tuple(QUERY_KEYS)
    except Exception:
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


def _choice(v: Any, path: str, options: tuple[str, ...]) -> str:
    if v not in options:
        raise _err(path, f"invalid value {v!r}", f"allowed: {', '.join(options)}")
    return v


def _str_list(v: Any, path: str) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise _err(path, "expected a list of group names")
    return list(v)


def _fill(cls, data: dict, path: str, *, nonneg_ints: tuple[str, ...] = ()):
    """Build a dataclass of numeric fields from a validated mapping."""
    kwargs = {}
    for k, v in data.items():
        default = getattr(cls, k)
        p = _join(path, k)
        kwargs[k] = _int(v, p, minimum=0) if isinstance(default, int) else _num(v, p, nonneg=True)
    return cls(**kwargs)


def _check_select(sel: Any, path: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(sel, dict) or not sel:
        raise _err(path, "expected a non-empty mapping of query keys")
    for k in sel:
        if k not in keys:
            raise _err(_join(path, str(k)), "unknown query key", f"allowed keys: {', '.join(keys)}")
    return sel


def load_spec(path: str | Path) -> MeshSpec:
    p = Path(path)
    if not p.is_file():
        raise _err("", f"spec file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise _err("", f"invalid YAML in {p}: {exc}") from exc
    return parse_spec(data, p)


def parse_spec(data: dict, path: Path) -> MeshSpec:
    path = Path(path)
    top = ("name", "geometry", "analysis", "heal", "domain", "groups", "default_group", "size",
           "refine", "element", "algorithm", "export", "quality", "solve", "output")
    data = _mapping(data, "", top)
    qkeys = _query_keys()

    name = data.get("name", path.resolve().parent.name)
    if not isinstance(name, str) or not name:
        raise _err("name", "expected a non-empty string")

    if "geometry" not in data:
        raise _err("geometry", "required field missing")
    if not isinstance(data["geometry"], str):
        raise _err("geometry", "expected a path string")
    geometry = (path.parent / data["geometry"]).resolve()
    if not geometry.is_file():
        raise _err("geometry", f"file not found: {geometry}", "path is relative to the spec file")

    if "analysis" not in data:
        raise _err("analysis", "required field missing", "allowed: fea, cfd")
    analysis = _choice(data["analysis"], "analysis", ("fea", "cfd"))

    heal = data.get("heal", False)
    if not isinstance(heal, bool):
        raise _err("heal", f"expected true/false, got {heal!r}")

    dd = _mapping(data.get("domain"), "domain", ("type", "flow_axis", "upstream", "downstream", "lateral"))
    domain = DomainSpec()
    if "type" in dd:
        domain.type = _choice(dd["type"], "domain.type", ("internal", "external"))
    if "flow_axis" in dd:
        domain.flow_axis = _choice(dd["flow_axis"], "domain.flow_axis", _AXES)
    for k in ("upstream", "downstream", "lateral"):
        if k in dd:
            setattr(domain, k, _num(dd[k], f"domain.{k}", positive=True))

    gd = data.get("groups")
    if gd is None:
        gd = {}
    if not isinstance(gd, dict):
        raise _err("groups", "expected a mapping of group name to {select, expect}")
    groups: dict[str, GroupSpec] = {}
    for gname, g in gd.items():
        gp = f"groups.{gname}"
        g = _mapping(g, gp, ("select", "expect"))
        if "select" not in g:
            raise _err(f"{gp}.select", "required field missing")
        sel = _check_select(g["select"], f"{gp}.select", qkeys)
        ex = g.get("expect")
        if ex is not None:
            if isinstance(ex, list):
                if len(ex) != 2:
                    raise _err(f"{gp}.expect", "range must be [min, max]")
                lo, hi = _int(ex[0], f"{gp}.expect[0]", minimum=0), _int(ex[1], f"{gp}.expect[1]", minimum=0)
                if lo > hi:
                    raise _err(f"{gp}.expect", f"min {lo} > max {hi}")
                ex = (lo, hi)
            else:
                ex = _int(ex, f"{gp}.expect", minimum=0)
        groups[str(gname)] = GroupSpec(name=str(gname), select=sel, expect=ex)

    default_group = data.get("default_group", "walls" if analysis == "cfd" else None)
    if default_group is not None:
        if not isinstance(default_group, str) or not default_group:
            raise _err("default_group", "expected a non-empty string or null")
        if default_group in groups:
            raise _err("default_group", f"{default_group!r} collides with a group of the same name",
                       "rename default_group or the group")

    if "size" not in data:
        raise _err("size", "required field missing", "e.g. size: {global: 3.0}")
    sd = _mapping(data["size"], "size", ("global", "min", "curvature", "thin_wall_layers"))
    if "global" not in sd:
        raise _err("size.global", "required field missing")
    size = SizeSpec(global_=_num(sd["global"], "size.global", positive=True))
    if "min" in sd:
        size.min = _num(sd["min"], "size.min", nonneg=True)
    if "curvature" in sd:
        size.curvature = _int(sd["curvature"], "size.curvature", minimum=0)
    if "thin_wall_layers" in sd:
        size.thin_wall_layers = _int(sd["thin_wall_layers"], "size.thin_wall_layers", minimum=0)

    rl = data.get("refine")
    if rl is None:
        rl = []
    if not isinstance(rl, list):
        raise _err("refine", "expected a list")
    refine: list[RefineSpec] = []
    for i, r in enumerate(rl):
        rp = f"refine[{i}]"
        r = _mapping(r, rp, ("group", "select", "size", "distance"))
        if ("group" in r) == ("select" in r):
            raise _err(rp, "exactly one of 'group' or 'select' is required")
        for k in ("size", "distance"):
            if k not in r:
                raise _err(f"{rp}.{k}", "required field missing")
        grp = sel = None
        if "group" in r:
            grp = r["group"]
            if not isinstance(grp, str) or grp not in groups:
                raise _err(f"{rp}.group", f"unknown group {grp!r}", f"defined groups: {', '.join(groups) or '(none)'}")
        else:
            sel = _check_select(r["select"], f"{rp}.select", qkeys)
        refine.append(RefineSpec(size=_num(r["size"], f"{rp}.size", positive=True),
                                 distance=_num(r["distance"], f"{rp}.distance", positive=True),
                                 group=grp, select=sel))

    ed = _mapping(data.get("element"), "element", ("order", "optimize"))
    order = _int(ed.get("order", 1), "element.order")
    if order not in (1, 2):
        raise _err("element.order", f"invalid value {order!r}", "allowed: 1, 2")
    optimize = _choice(ed.get("optimize", "netgen"), "element.optimize", _OPTIMIZE)

    ad = _mapping(data.get("algorithm"), "algorithm", ("mesh2d", "mesh3d"))
    alg2 = _int(ad.get("mesh2d", 6), "algorithm.mesh2d", minimum=1)
    alg3 = _int(ad.get("mesh3d", 1), "algorithm.mesh3d", minimum=1)

    export = data.get("export", ["msh"])
    if not isinstance(export, list) or not export:
        raise _err("export", "expected a non-empty list", f"allowed: {', '.join(_EXPORTS)}")
    for i, x in enumerate(export):
        _choice(x, f"export[{i}]", _EXPORTS)
    export = list(export)

    qd = _mapping(data.get("quality"), "quality", tuple(QualitySpec.__dataclass_fields__))
    quality = _fill(QualitySpec, qd, "quality")

    sv = _mapping(data.get("solve"), "solve", ("fea", "cfd"))
    fea = cfd = None

    def need(names: list[str], p: str) -> None:
        for i, n in enumerate(names):
            if n not in groups:
                raise _err(f"{p}[{i}]", f"unknown group {n!r}", f"defined groups: {', '.join(groups) or '(none)'}")

    if sv.get("fea") is not None:
        if analysis != "fea":
            raise _err("solve.fea", "only allowed with analysis: fea")
        fd = _mapping(sv["fea"], "solve.fea", ("E", "nu", "fixed", "loads"))
        fixed = _str_list(fd.get("fixed", []), "solve.fea.fixed")
        need(fixed, "solve.fea.fixed")
        loads = []
        ll = fd.get("loads", [])
        if not isinstance(ll, list):
            raise _err("solve.fea.loads", "expected a list")
        for i, ld in enumerate(ll):
            lp = f"solve.fea.loads[{i}]"
            ld = _mapping(ld, lp, ("group", "force"))
            for k in ("group", "force"):
                if k not in ld:
                    raise _err(f"{lp}.{k}", "required field missing")
            if not isinstance(ld["group"], str) or ld["group"] not in groups:
                raise _err(f"{lp}.group", f"unknown group {ld['group']!r}",
                           f"defined groups: {', '.join(groups) or '(none)'}")
            f = ld["force"]
            if not isinstance(f, list) or len(f) != 3:
                raise _err(f"{lp}.force", "expected [fx, fy, fz]")
            loads.append(FeaLoad(ld["group"], tuple(_num(c, f"{lp}.force[{j}]") for j, c in enumerate(f))))
        fea = FeaSolveSpec(fixed=fixed, loads=loads)
        if "E" in fd:
            fea.E = _num(fd["E"], "solve.fea.E", positive=True)
        if "nu" in fd:
            fea.nu = _num(fd["nu"], "solve.fea.nu")
            if not 0 <= fea.nu < 0.5:
                raise _err("solve.fea.nu", f"must be in [0, 0.5), got {fea.nu}")

    if sv.get("cfd") is not None:
        if analysis != "cfd":
            raise _err("solve.cfd", "only allowed with analysis: cfd")
        cd = _mapping(sv["cfd"], "solve.cfd", ("inlet", "outlet", "velocity"))
        inlet = _str_list(cd.get("inlet", []), "solve.cfd.inlet")
        outlet = _str_list(cd.get("outlet", []), "solve.cfd.outlet")
        need(inlet, "solve.cfd.inlet")
        need(outlet, "solve.cfd.outlet")
        cfd = CfdSolveSpec(inlet=inlet, outlet=outlet)
        if "velocity" in cd:
            cfd.velocity = _num(cd["velocity"], "solve.cfd.velocity", positive=True)

    output = data.get("output", "out")
    if not isinstance(output, str) or not output:
        raise _err("output", "expected a directory name string")

    return MeshSpec(
        path=path, name=name, geometry=geometry, analysis=analysis, heal=heal, domain=domain,
        groups=groups, default_group=default_group, size=size, refine=refine, order=order,
        optimize=optimize, algorithm_2d=alg2, algorithm_3d=alg3, export=export, quality=quality,
        fea=fea, cfd=cfd, output=Path(output),
    )
