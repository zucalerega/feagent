"""Face query language: select faces by geometry instead of remembered tags."""
from __future__ import annotations

import difflib
import math

from .errors import QueryError
from .types import FaceInfo, GeometryModel

QUERY_KEYS = ("ids", "kind", "normal", "axis", "tol_deg", "radius", "radius_tol", "radius_min",
              "radius_max", "area_min", "area_max", "concave", "at", "box", "near", "within",
              "feature", "adjacent_to", "largest", "smallest", "any", "all", "not")
_MODIFIERS = {"tol_deg", "radius_tol", "within"}
_POST = {"largest", "smallest"}
_AT = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_NUM = (int, float)


def _bad(msg: str, hint: str = "") -> QueryError:
    return QueryError(msg, hint)


def _num(q: dict, key: str) -> float:
    v = q[key]
    if isinstance(v, bool) or not isinstance(v, _NUM):
        raise _bad(f"query key '{key}' must be a number, got {v!r}")
    return float(v)


def _vec(v, key: str) -> tuple[float, float, float]:
    if (not isinstance(v, (list, tuple)) or len(v) != 3
            or any(isinstance(x, bool) or not isinstance(x, _NUM) for x in v)):
        raise _bad(f"query key '{key}' must be [x, y, z], got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _validate(q, path: str = "query") -> None:
    if not isinstance(q, dict):
        raise _bad(f"{path} must be a dict, got {type(q).__name__}")
    for k in q:
        if k not in QUERY_KEYS:
            close = difflib.get_close_matches(str(k), QUERY_KEYS, n=3, cutoff=0.6)
            hint = (f"did you mean {', '.join(repr(c) for c in close)}? " if close else "")
            raise _bad(f"unknown query key '{k}'", hint + f"valid keys: {', '.join(QUERY_KEYS)}")
    for k, v in q.items():
        if k == "ids":
            if not isinstance(v, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int) for i in v):
                raise _bad("query key 'ids' must be a list of ints")
        elif k == "kind":
            if not (isinstance(v, str) or (isinstance(v, (list, tuple)) and all(isinstance(i, str) for i in v))):
                raise _bad("query key 'kind' must be a string or list of strings")
        elif k in ("normal", "axis", "near"):
            x = _vec(v, k)
            if k != "near" and not any(x):
                raise _bad(f"query key '{k}' must be a non-zero vector")
        elif k in ("tol_deg", "radius", "radius_tol", "radius_min", "radius_max", "area_min",
                   "area_max", "within"):
            _num(q, k)
        elif k == "concave":
            if not isinstance(v, bool):
                raise _bad("query key 'concave' must be true or false")
        elif k == "at":
            if v not in _AT:
                raise _bad(f"query key 'at' must be one of {list(_AT)}, got {v!r}")
        elif k == "box":
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise _bad("query key 'box' must be [[x0,y0,z0],[x1,y1,z1]]")
            _vec(v[0], "box"), _vec(v[1], "box")
        elif k == "feature":
            if not isinstance(v, str):
                raise _bad("query key 'feature' must be a string (feature kind, e.g. 'hole')")
        elif k in ("adjacent_to", "not"):
            _validate(v, f"{path}.{k}")
        elif k in ("any", "all"):
            if not isinstance(v, (list, tuple)):
                raise _bad(f"query key '{k}' must be a list of queries")
            for i, s in enumerate(v):
                _validate(s, f"{path}.{k}[{i}]")
        elif k in _POST:
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise _bad(f"query key '{k}' must be a positive int")


def _angle(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    c = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _box_dist(p, bb) -> float:
    return math.sqrt(sum(max(bb[0][i] - p[i], 0.0, p[i] - bb[1][i]) ** 2 for i in range(3)))


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# A local check returns (ok, detail, ratio); ratio = how many tolerances away (>1 means failing).
def _check(m: GeometryModel, f: FaceInfo, q: dict, key: str, relax: float):
    v = q[key]
    diag = m.diag
    if key == "kind":
        kinds = [v] if isinstance(v, str) else list(v)
        return f.kind in kinds, f"kind is {f.kind}, wanted {'/'.join(kinds)}", 1e3
    if key in ("normal", "axis"):
        tol = (float(q["tol_deg"]) if "tol_deg" in q else 5.0) * (3.0 if relax > 1 else 1.0)
        want = _vec(v, key)
        have = f.normal if key == "normal" else f.axis
        if have is None:
            return False, f"has no {key} (kind {f.kind})", 1e3
        ang = _angle(have, want)
        if key == "axis":
            ang = min(ang, 180.0 - ang)
        return ang <= tol, f"{key} off by {ang:.1f}° (tol {tol:g}°)", ang / max(tol, 1e-9)
    if key == "radius":
        r = float(v)
        tol = (float(q["radius_tol"]) if "radius_tol" in q else max(1e-3 * r, 1e-6 * diag)) * relax
        if f.radius is None:
            return False, f"has no radius (kind {f.kind})", 1e3
        d = abs(f.radius - r)
        return d <= tol, f"radius {f.radius:.4g} vs {r:.4g} (tol {tol:.3g})", d / max(tol, 1e-12)
    if key in ("radius_min", "radius_max"):
        if f.radius is None:
            return False, f"has no radius (kind {f.kind})", 1e3
        lim = float(v)
        d = (lim - f.radius) if key == "radius_min" else (f.radius - lim)
        sl = abs(lim) * 0.1 if relax > 1 else 0.0
        return d <= sl, f"radius {f.radius:.4g} violates {key}={lim:g}", 1 + max(d, 0) / max(abs(lim), 1e-9)
    if key in ("area_min", "area_max"):
        lim = float(v)
        d = (lim - f.area) if key == "area_min" else (f.area - lim)
        sl = abs(lim) * 0.1 if relax > 1 else 0.0
        return d <= sl, f"area {f.area:.4g} violates {key}={lim:g}", 1 + max(d, 0) / max(abs(lim), 1e-9)
    if key == "concave":
        return f.concave is v, f"concave is {f.concave}, wanted {v}", 1e3
    if key == "at":
        i = "xyz".index(v[0])
        tol = (1e-6 * diag + 1e-9) * relax
        # face lies on the extreme plane: its whole extent along the axis is within tol of it
        d = abs(f.bbox[1][i] - m.bbox[0][i]) if v.endswith("min") else abs(f.bbox[0][i] - m.bbox[1][i])
        return d <= tol, f"{v} gap {d:.4g}", d / tol
    if key == "box":
        lo, hi = _vec(v[0], "box"), _vec(v[1], "box")
        d = _box_dist(f.center, (lo, hi))
        tol = 1e-6 * diag * relax
        return d <= tol, f"center {d:.4g} outside box", d / max(tol, 1e-12)
    if key == "near":
        p = _vec(v, "near")
        tol = (float(q["within"]) if "within" in q else 1e-3 * diag) * relax
        d = min(_dist(p, f.center), _box_dist(p, f.bbox))
        return d <= tol, f"{d:.4g} from point (within {tol:.3g})", d / max(tol, 1e-12)
    if key == "ids":
        return f.tag in v, "tag not in ids", 1e3
    if key == "feature":
        ok = any(ft.kind == v and f.tag in ft.faces for ft in m.features)
        return ok, f"not part of a '{v}' feature", 1e3
    raise AssertionError(key)


_LOCAL = {"kind", "normal", "axis", "radius", "radius_min", "radius_max", "area_min", "area_max",
          "concave", "at", "box", "near", "ids", "feature"}


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
    for key in q:
        if key in _MODIFIERS or key in _POST:
            continue
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
    _validate(query)
    return sorted(_eval(model, query))


def _fmt3(v) -> str:
    return "(" + ",".join(f"{x:.4g}" for x in v) + ")"


def describe_faces(model: GeometryModel, tags: list[int]) -> str:
    lines = []
    for t in tags:
        try:
            f = model.face(t)
        except KeyError:
            raise _bad(f"no face with tag {t}", f"valid tags: {sorted(x.tag for x in model.faces)}") from None
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
        lines.append(" ".join(parts))
    return "\n".join(lines)


def explain_miss(model: GeometryModel, query: dict, expected: int | tuple[int, int] | None) -> str:
    _validate(query)
    got = sorted(_eval(model, query))
    if expected is None:
        exp = "at least 1"
    elif isinstance(expected, tuple):
        exp = f"{expected[0]}-{expected[1]}"
    else:
        exp = str(expected)
    out = [f"query {query}: expected {exp} face(s), {len(got)} matched."]
    if got:
        out.append("matched faces:\n" + describe_faces(model, got[:10]))
        if len(got) > 10:
            out.append(f"... and {len(got) - 10} more")
        return "\n".join(out)
    keys = [k for k in query if k not in _MODIFIERS and k not in _POST]
    local = [k for k in keys if k in _LOCAL]
    out.append("each key alone (strict / relaxed: tol_deg x3, tolerances x10):")
    for k in keys:
        sub = {k: query[k]}
        for mod in _MODIFIERS:
            if mod in query:
                sub[mod] = query[mod]
        out.append(f"  {k}: {len(_eval(model, sub))} matched / {len(_eval(model, sub, 10.0))} relaxed")
    nl_sets = {k: _nonlocal(model, query, k, 1.0) for k in keys if k not in _LOCAL}
    scored = []
    for f in model.faces:
        fails, total = [], 0.0
        for k in local:
            ok, detail, ratio = _check(model, f, query, k, 1.0)
            if not ok:
                fails.append(detail)
                total += min(ratio, 1e3)
        for k, st in nl_sets.items():
            if f.tag not in st:
                fails.append(f"fails '{k}' sub-query (see below)")
                total += 1e3
        scored.append((len(fails), total, f.tag, fails))
    scored.sort(key=lambda s: s[:3])
    out.append("nearest candidates:")
    for n, _, tag, fails in scored[:5]:
        out.append(describe_faces(model, [tag]))
        out.append("    fails: " + ("; ".join(fails) if fails else "only the largest/smallest filter"))
    for k in nl_sets:
        subs = query[k] if k in ("any", "all") else [query[k]]
        for i, sub in enumerate(subs):
            label = k if k not in ("any", "all") else f"{k}[{i}]"
            out.append(f"sub-query '{label}' {sub}: {len(_eval(model, sub))} matched; each key alone:")
            for sk in sub:
                if sk in _MODIFIERS or sk in _POST:
                    continue
                one = {sk: sub[sk], **{m_: sub[m_] for m_ in _MODIFIERS if m_ in sub}}
                out.append(f"    {sk}: {len(_eval(model, one))} matched / {len(_eval(model, one, 10.0))} relaxed")
    return "\n".join(out)
