"""Face query language: select faces by geometry instead of remembered tags."""
from __future__ import annotations

import difflib

from .errors import QueryError
from .types import FACE_KINDS, FaceInfo, GeometryModel
from .vec import angle_deg, axis_angle_deg, box_dist, dist, extreme_gap

QUERY_KEYS = ("ids", "kind", "normal", "axis", "tol_deg", "radius", "radius_tol", "radius_min",
              "radius_max", "area_min", "area_max", "concave", "at", "box", "near", "within",
              "feature", "adjacent_to", "largest", "smallest", "any", "all", "not")
_MODIFIERS = {"tol_deg", "radius_tol", "within"}
_POST = {"largest", "smallest"}
_AT = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_NUM = (int, float)


def _bad(msg: str, hint: str = "") -> QueryError:
    return QueryError(msg, hint)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, _NUM) and not isinstance(v, bool)


def _vec(v, key: str) -> tuple[float, float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(map(_is_num, v)):
        raise _bad(f"query key '{key}' must be [x, y, z], got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


# --- validation: one checker per query key --------------------------------

def _did_you_mean(word: str, valid: tuple[str, ...], what: str) -> str:
    close = difflib.get_close_matches(word, valid, n=3, cutoff=0.6)
    return (f"did you mean {', '.join(repr(c) for c in close)}? " if close else "") + \
        f"valid {what}: {', '.join(valid)}"


def _validate(q, path: str = "query") -> None:
    if not isinstance(q, dict):
        raise _bad(f"{path} must be a dict, got {type(q).__name__}")
    for k in q:
        if k not in QUERY_KEYS:
            raise _bad(f"unknown query key '{k}'", _did_you_mean(str(k), QUERY_KEYS, "keys"))
    for k, v in q.items():
        _VALIDATORS[k](k, v, path)


def _v_ids(k, v, path) -> None:
    if not isinstance(v, (list, tuple)) or not all(map(_is_int, v)):
        raise _bad("query key 'ids' must be a list of ints")


def _v_kind(k, v, path) -> None:
    if not (isinstance(v, str) or (isinstance(v, (list, tuple))
                                   and all(isinstance(i, str) for i in v))):
        raise _bad("query key 'kind' must be a string or list of strings")
    for kind in [v] if isinstance(v, str) else v:
        if kind not in FACE_KINDS:
            raise _bad(f"unknown face kind {kind!r}", _did_you_mean(kind, FACE_KINDS, "kinds"))


def _v_direction(k, v, path) -> None:
    if not any(_vec(v, k)):
        raise _bad(f"query key '{k}' must be a non-zero vector")


def _v_point(k, v, path) -> None:
    _vec(v, k)


def _v_number(k, v, path) -> None:
    if not _is_num(v):
        raise _bad(f"query key '{k}' must be a number, got {v!r}")


def _v_concave(k, v, path) -> None:
    if not isinstance(v, bool):
        raise _bad("query key 'concave' must be true or false")


def _v_at(k, v, path) -> None:
    if v not in _AT:
        raise _bad(f"query key 'at' must be one of {list(_AT)}, got {v!r}",
                   f"valid values: {', '.join(_AT)}; each picks the faces lying flat on that side "
                   "of the model bounding box (e.g. zmin = faces in the plane z = bbox min z, "
                   "zmax = the top plane); for a plane inside the part use box: or near:")


def _v_box(k, v, path) -> None:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise _bad("query key 'box' must be [[x0,y0,z0],[x1,y1,z1]]")
    _vec(v[0], "box"), _vec(v[1], "box")


def _v_feature(k, v, path) -> None:
    if not isinstance(v, str):
        raise _bad("query key 'feature' must be a string (feature kind, e.g. 'hole')")


def _v_subquery(k, v, path) -> None:
    _validate(v, f"{path}.{k}")


def _v_subqueries(k, v, path) -> None:
    if not isinstance(v, (list, tuple)):
        raise _bad(f"query key '{k}' must be a list of queries")
    for i, s in enumerate(v):
        _validate(s, f"{path}.{k}[{i}]")


def _v_count(k, v, path) -> None:
    if not _is_int(v) or v < 1:
        raise _bad(f"query key '{k}' must be a positive int")


_VALIDATORS = {
    "ids": _v_ids, "kind": _v_kind, "normal": _v_direction, "axis": _v_direction,
    "near": _v_point, "concave": _v_concave, "at": _v_at, "box": _v_box,
    "feature": _v_feature, "adjacent_to": _v_subquery, "not": _v_subquery,
    "any": _v_subqueries, "all": _v_subqueries, "largest": _v_count, "smallest": _v_count,
    **dict.fromkeys(("tol_deg", "radius", "radius_tol", "radius_min", "radius_max",
                     "area_min", "area_max", "within"), _v_number),
}


# --- local checks: (ok, detail, ratio); ratio = tolerances away (>1 fails) --

_MISS = 1e3  # ratio for pass/fail-only checks


def _within(d: float, tol: float, detail: str, floor: float = 1e-12):
    return d <= tol, detail, d / max(tol, floor)


def _lacks(what: str, f: FaceInfo):
    return False, f"has no {what} (kind {f.kind})", _MISS


def _relaxed_slack(lim: float, relax: float) -> float:
    return abs(lim) * 0.1 if relax > 1 else 0.0


def _c_kind(m, f, q, key, relax):
    kinds = [q[key]] if isinstance(q[key], str) else list(q[key])
    return f.kind in kinds, f"kind is {f.kind}, wanted {'/'.join(kinds)}", _MISS


def _c_direction(m, f, q, key, relax):
    tol = float(q.get("tol_deg", 5.0)) * (3.0 if relax > 1 else 1.0)
    have = f.normal if key == "normal" else f.axis
    if have is None:
        return _lacks(key, f)
    want = _vec(q[key], key)
    ang = angle_deg(have, want) if key == "normal" else axis_angle_deg(have, want)
    return _within(ang, tol, f"{key} off by {ang:.1f}° (tol {tol:g}°)", floor=1e-9)


def _c_radius(m, f, q, key, relax):
    r = float(q[key])
    tol = (float(q["radius_tol"]) if "radius_tol" in q else max(1e-3 * r, 1e-6 * m.diag)) * relax
    if f.radius is None:
        return _lacks("radius", f)
    d = abs(f.radius - r)
    return _within(d, tol, f"radius {f.radius:.4g} vs {r:.4g} (tol {tol:.3g})")


def _limit(name: str, have: float, key: str, lim: float, relax: float):
    d = (lim - have) if key.endswith("_min") else (have - lim)
    return (d <= _relaxed_slack(lim, relax), f"{name} {have:.4g} violates {key}={lim:g}",
            1 + max(d, 0) / max(abs(lim), 1e-9))


def _c_radius_limit(m, f, q, key, relax):
    if f.radius is None:
        return _lacks("radius", f)
    return _limit("radius", f.radius, key, float(q[key]), relax)


def _c_area_limit(m, f, q, key, relax):
    return _limit("area", f.area, key, float(q[key]), relax)


def _c_concave(m, f, q, key, relax):
    return f.concave is q[key], f"concave is {f.concave}, wanted {q[key]}", _MISS


# OpenCASCADE enlarges bounding boxes by its tolerance (1e-7 model units per side), so two bboxes
# of the same plane differ by ~2e-7 whatever the part size: metre-scale parts need this floor.
OCC_BBOX_SLACK = 1e-6


def _c_at(m, f, q, key, relax):
    tol = (max(1e-6 * m.diag, OCC_BBOX_SLACK) + 1e-9) * relax
    d = extreme_gap(f.bbox, m.bbox, q[key])  # face lies on the extreme plane
    return _within(d, tol, f"{q[key]} gap {d:.4g}")


def _c_box(m, f, q, key, relax):
    box = (_vec(q[key][0], "box"), _vec(q[key][1], "box"))
    d = box_dist(f.center, box)
    return _within(d, 1e-6 * m.diag * relax, f"center {d:.4g} outside box")


def _c_near(m, f, q, key, relax):
    p = _vec(q[key], "near")
    tol = (float(q["within"]) if "within" in q else 1e-3 * m.diag) * relax
    d = min(dist(p, f.center), box_dist(p, f.bbox))
    return _within(d, tol, f"{d:.4g} from point (within {tol:.3g})")


def _c_ids(m, f, q, key, relax):
    return f.tag in q[key], "tag not in ids", _MISS


def _c_feature(m, f, q, key, relax):
    ok = any(ft.kind == q[key] and f.tag in ft.faces for ft in m.features)
    return ok, f"not part of a '{q[key]}' feature", _MISS


_CHECKS = {
    "kind": _c_kind, "normal": _c_direction, "axis": _c_direction, "radius": _c_radius,
    "radius_min": _c_radius_limit, "radius_max": _c_radius_limit,
    "area_min": _c_area_limit, "area_max": _c_area_limit, "concave": _c_concave,
    "at": _c_at, "box": _c_box, "near": _c_near, "ids": _c_ids, "feature": _c_feature,
}


def _check(m: GeometryModel, f: FaceInfo, q: dict, key: str, relax: float):
    return _CHECKS[key](m, f, q, key, relax)


_LOCAL = set(_CHECKS)


def _nonlocal(m: GeometryModel, q: dict, key: str, relax: float) -> set[int]:
    v = q[key]
    if key == "adjacent_to":
        src = _eval(m, v, relax)
        return {n for t in src for n in m.face(t).neighbors}
    if key == "not":
        return {f.tag for f in m.faces} - _eval(m, v, relax)
    if key == "any":
        out: set[int] = set()
        for s in v:
            out |= _eval(m, s, relax)
        return out
    out = {f.tag for f in m.faces}
    for s in v:
        out &= _eval(m, s, relax)
    return out


def _eval(m: GeometryModel, q: dict, relax: float = 1.0) -> set[int]:
    tags = {f.tag for f in m.faces}
    for key in _filter_keys(q):
        if key in _LOCAL:
            tags &= {f.tag for f in m.faces if f.tag in tags and _check(m, f, q, key, relax)[0]}
        else:
            tags &= _nonlocal(m, q, key, relax)
    for key, rev in (("largest", True), ("smallest", False)):
        if key in q:
            ordered = sorted(tags, key=lambda t: (-m.face(t).area if rev else m.face(t).area, t))
            tags = set(ordered[: int(q[key])])
    return tags


def select_faces(model: GeometryModel, query: dict) -> list[int]:
    """Sorted tags of the faces matching ``query`` (all keys AND-ed)."""
    _validate(query)
    return sorted(_eval(model, query))


def _fmt3(v) -> str:
    return "(" + ",".join(f"{x:.4g}" for x in v) + ")"


def _face_line(f: FaceInfo) -> str:
    parts = [f"tag={f.tag}", f"kind={f.kind}", f"area={f.area:.4g}", f"center={_fmt3(f.center)}"]
    if f.normal is not None:
        parts.append(f"normal={_fmt3(f.normal)}")
    elif f.axis is not None:
        parts.append(f"axis={_fmt3(f.axis)}")
    if f.radius is not None:
        parts.append(f"radius={f.radius:.4g}")
    if f.concave is not None:
        parts.append(f"concave={str(f.concave).lower()}")
    parts.append("neighbors=[" + ",".join(map(str, f.neighbors)) + "]")
    return " ".join(parts)


def describe_faces(model: GeometryModel, tags: list[int]) -> str:
    """One line per face: tag, kind, area, center, normal/axis, radius, neighbours."""
    lines = []
    for t in tags:
        try:
            lines.append(_face_line(model.face(t)))
        except KeyError:
            raise _bad(f"no face with tag {t}",
                       f"valid tags: {sorted(x.tag for x in model.faces)}") from None
    return "\n".join(lines)


# --- explain_miss ---------------------------------------------------------

def explain_miss(model: GeometryModel, query: dict, expected: int | tuple[int, int] | None) -> str:
    """Explain why ``query`` matched an unexpected number of faces."""
    _validate(query)
    got = sorted(_eval(model, query))
    out = [f"query {query}: expected {_expected_text(expected)} face(s), {len(got)} matched."]
    if got:
        return "\n".join(out + _matched_lines(model, got))
    nl_sets = {k: _nonlocal(model, query, k, 1.0) for k in _filter_keys(query) if k not in _LOCAL}
    out.append("each key alone (strict / relaxed: tol_deg x3, tolerances x10):")
    out += _key_alone_lines(model, query, "  ")
    out += _nearest_lines(model, query, nl_sets)
    out += _subquery_lines(model, query, nl_sets)
    return "\n".join(out)


def _expected_text(expected: int | tuple[int, int] | None) -> str:
    if expected is None:
        return "at least 1"
    if isinstance(expected, tuple):
        return f"{expected[0]}-{expected[1]}"
    return str(expected)


def _matched_lines(model: GeometryModel, got: list[int]) -> list[str]:
    lines = ["matched faces:\n" + describe_faces(model, got[:10])]
    if len(got) > 10:
        lines.append(f"... and {len(got) - 10} more")
    return lines


def _filter_keys(q: dict) -> list[str]:
    """Keys that filter faces on their own (not modifiers or largest/smallest)."""
    return [k for k in q if k not in _MODIFIERS and k not in _POST]


def _key_alone_lines(model: GeometryModel, q: dict, indent: str) -> list[str]:
    """Match counts (strict / relaxed) of each filter key with the query's modifiers."""
    mods = {m_: q[m_] for m_ in _MODIFIERS if m_ in q}
    lines = []
    for k in _filter_keys(q):
        one = {k: q[k], **mods}
        lines.append(f"{indent}{k}: {len(_eval(model, one))} matched / "
                     f"{len(_eval(model, one, 10.0))} relaxed")
    return lines


def _nearest_lines(model: GeometryModel, query: dict, nl_sets: dict[str, set[int]]) -> list[str]:
    local = [k for k in _filter_keys(query) if k in _LOCAL]
    scored = sorted((_score(model, f, query, local, nl_sets) for f in model.faces),
                    key=lambda s: s[:3])
    lines = ["nearest candidates:"]
    for _, _, tag, fails in scored[:5]:
        lines.append(describe_faces(model, [tag]))
        lines.append("    fails: " + ("; ".join(fails) if fails else
                                      "only the largest/smallest filter"))
    return lines


def _score(model: GeometryModel, f: FaceInfo, query: dict, local: list[str],
           nl_sets: dict[str, set[int]]) -> tuple[int, float, int, list[str]]:
    """(failed key count, summed tolerance ratios, tag, failure details) for one face."""
    fails, total = [], 0.0
    for k in local:
        ok, detail, ratio = _check(model, f, query, k, 1.0)
        if not ok:
            fails.append(detail)
            total += min(ratio, _MISS)
    for k, st in nl_sets.items():
        if f.tag not in st:
            fails.append(f"fails '{k}' sub-query (see below)")
            total += _MISS
    return len(fails), total, f.tag, fails


def _subquery_lines(model: GeometryModel, query: dict, nl_sets: dict[str, set[int]]) -> list[str]:
    lines = []
    for k in nl_sets:
        subs = list(enumerate(query[k])) if k in ("any", "all") else [(None, query[k])]
        for i, sub in subs:
            label = k if i is None else f"{k}[{i}]"
            lines.append(f"sub-query '{label}' {sub}: {len(_eval(model, sub))} matched; "
                         "each key alone:")
            lines += _key_alone_lines(model, sub, "    ")
    return lines
