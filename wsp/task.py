"""The student's meshing task: one prompt in, one JSON mesh recipe out, scored by the fixed solver.

A small model is not a good tool-use agent, so the student does what the ACE agents do in their
`mesh(h, order, refinements)` call, in one shot: it reads the statement and a compact description
of the part and writes the mesh recipe. The harness meshes it (ace.fea.make_mesh), solves it with
the problem's correct setup (the statement spells the setup out; meshing is the skill under test),
and scores it in field mode exactly as the swarm is scored (ace.checks._score_vs_automesher):

    e = max(energy-norm field error, RMS von Mises error at 12 probe points), floored at 0.5 %
    g = e_auto / e,  c = DOFs / DOFs_auto,  A = (g/2)^2 if g < 2 else g/2,  score = A / c
    x_auto = score / 0.25            (the gmsh default mesh scores 1x, as in the README figures)

Normalized units. Lengths are given to the student in u = (part diagonal) / 1000, measured from
the lower-left corner of the bounding box, so every part looks the same size: h ~ 20-100 u, a
notch-root size ~ 1-5 u. A sub-1B model learns that far more easily than raw sizes that range from
0.0005 m to 4 (Cook's membrane). Refinements refer to a corner or curve by its tag, so the student
never has to copy coordinates. The harness converts back to absolute sizes and positions.

An invalid answer (no JSON, unknown tag, mesh failure, over the node cap, quality gate, folded
cells, solve failure) scores 0.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from wsp import problems

UNITS = 1000.0                       # the part diagonal in u
MAX_REFINEMENTS = 12
CACHE_DIR = Path(os.environ.get("WSP_CACHE") or Path(__file__).resolve().parents[1] / "wsp_runs" / "cache")

SYSTEM = ("You set up finite-element meshes. You read a part and write a gmsh mesh recipe as one JSON "
          "object. Answer with the JSON object only.")


# --------------------------------------------------------------------------- geometry in u


@dataclass
class Geometry:
    pid: str
    dim: int
    origin: list[float]              # bounding-box minimum (absolute)
    scale: float                     # metres (problem units) per u
    size_u: list[float]              # bounding-box extents in u
    curves: dict[int, dict] = field(default_factory=dict)
    corners: dict[int, dict] = field(default_factory=dict)
    groups: list[str] = field(default_factory=list)
    floor_u: float = 0.0             # smallest element size ace.fea allows, in u
    raw: dict = field(default_factory=dict)

    def u(self, x: float) -> float:
        return x / self.scale

    def xy_u(self, p) -> list[float]:
        return [(p[i] - self.origin[i]) / self.scale for i in range(self.dim)]

    def abs_point(self, p_u) -> list[float]:
        return [self.origin[i] + float(p_u[i]) * self.scale for i in range(self.dim)]


@functools.lru_cache(maxsize=None)
def geometry(pid: str) -> Geometry:
    from ace import fea, geometry_info
    p = fea.get_problem(pid)
    g = geometry_info.for_agent({**fea.geometry_summary(p), **geometry_info.describe(p)})
    bbox = g["bbox"]
    dim = g["dim"]
    lo = [bbox[i] for i in range(dim)]
    hi = [bbox[dim + i] for i in range(dim)] if len(bbox) == 2 * dim else [bbox[3 + i] for i in range(dim)]
    diag = math.dist(lo, hi)
    scale = diag / UNITS
    floor = diag / (100.0 if dim == 3 else 400.0)          # = ace.fea._size_floor
    return Geometry(pid=pid, dim=dim, origin=lo + [0.0] * (3 - dim), scale=scale,
                    size_u=[(h - l) / scale for l, h in zip(lo, hi)],
                    curves={int(c["tag"]): c for c in g["curves"]},
                    corners={int(k["tag"]): k for k in g["corners"]},
                    groups=[gr["name"] for gr in g["groups"]], floor_u=floor / scale, raw=g)


def _n(x: float) -> str:
    """A length in u, short: 3 significant digits, at most one decimal."""
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.2g}" if x < 1 else f"{x:.1f}".rstrip("0").rstrip(".")


def geometry_text(geo: Geometry) -> str:
    g = geo.raw
    reent = set(g.get("reentrant_corners") or [])
    sharp = set(g.get("sharp_corners") or [])
    sup = set(g.get("support_boundary") or [])
    load = set(g.get("load_boundary") or [])
    from ace import fea
    unit = "m" if "SI" in (fea.get_problem(geo.pid).units or "") else "length units"
    lines = [f"PART GEOMETRY. Lengths in u, where 1 u = {geo.scale:.4g} {unit} (the part diagonal is 1000 u); "
             f"coordinates are measured from the lower-left corner of the bounding box.",
             f"{geo.dim}D part, bounding box " + " x ".join(_n(s) for s in geo.size_u) + " u.",
             "Curves (c = curve tag):"]
    for tag, c in sorted(geo.curves.items()):
        bits = [f"  c{tag} {c['type']}", f"length {_n(geo.u(c['length']))}"]
        if c.get("radius"):
            bits.append(f"radius {_n(geo.u(c['radius']))}")
        if c.get("centroid"):
            bits.append("near (" + ", ".join(_n(v) for v in geo.xy_u(c["centroid"])) + ")")
        tail = []
        if c.get("group"):
            tail.append(f"group {c['group']}")
        if c.get("role") and c["role"] != "free":
            tail.append(c["role"])
        lines.append(", ".join(bits) + (f" [{'; '.join(tail)}]" if tail else ""))
    notable = []
    for tag, k in sorted(geo.corners.items()):
        flags = []
        if tag in reent or k.get("concave"):
            flags.append(f"RE-ENTRANT {k['interior_angle_deg']:.0f} deg (stress singularity)")
        elif tag in sharp:
            flags.append(f"acute {k['interior_angle_deg']:.0f} deg")
        if tag in sup:
            flags.append("support end")
        if tag in load:
            flags.append("load end")
        if flags:
            notable.append(f"  k{tag} at (" + ", ".join(_n(v) for v in geo.xy_u(k["point"])) + "): " + "; ".join(flags))
    other = len(geo.corners) - len(notable)
    if notable:
        lines.append("Notable corners (k = corner tag):")
        lines += notable
        if other:
            lines.append(f"  ({other} other corners: smooth or convex, not singular)")
        if any(t in sup for t in geo.corners):
            lines.append("  A support end is singular only where a clamp (both directions fixed) meets a free edge; the "
                         "ends of symmetry, roller, axis or exactly prescribed supports are smooth. A load end is a "
                         "traction jump: milder and local.")
    else:
        lines.append(f"Notable corners: none ({other} corners, all smooth or convex, not singular).")
    sf = g.get("smallest_features") or {}
    small = []
    if "smallest_radius" in sf:
        small.append(f"smallest radius {_n(geo.u(sf['smallest_radius']['radius']))} u ({sf['smallest_radius']['curve']})")
    if "shortest_curve" in sf:
        small.append(f"shortest curve {_n(geo.u(sf['shortest_curve']['length']))} u (c{sf['shortest_curve']['tag']})")
    if "thinnest_wall" in sf:
        small.append(f"thinnest wall {_n(geo.u(sf['thinnest_wall']['thickness']))} u")
    if small:
        lines.append("Smallest features: " + "; ".join(small) + ".")
    lines.append("Named groups: " + ", ".join(geo.groups) + ".")
    return "\n".join(lines)


def statement(pid: str) -> str:
    """The problem statement without its 'Report:' list (field mode scores the whole field)."""
    from ace import fea
    text = fea.get_problem(pid).description()
    return re.split(r"\n\s*Report:", text, maxsplit=1)[0].strip()


def scoring_text(pid: str) -> str:
    from ace import config
    from ace.field_error import automesher
    a = automesher(pid)
    geo = geometry(pid)
    cap = config.TASKS[pid]["node_cap"]
    return (
        f"SCORING. Your mesh is solved with the correct setup and its whole stress field is compared with a very "
        f"fine reference solution (error e = the larger of the energy-norm error over the part and the RMS von "
        f"Mises error at hidden check points). The automesher (gmsh defaults, quadratic elements) gets "
        f"e = {a['error']:.2%} with {a['dofs']} DOFs and scores 1x. Your score, in x the automesher, is 4 A / c with "
        f"g = {a['error']:.4f} / e, A = (g/2)^2 if g < 2 else g/2, c = your DOFs / {a['dofs']}. Accuracy per DOF "
        f"wins: put small elements where the stress changes fast (notch roots, holes, fillets, re-entrant corners, "
        f"ends of supports) and large ones elsewhere.\n"
        f"LIMITS. At most {cap} nodes. Sizes below {_n(geo.floor_u)} u are raised to {_n(geo.floor_u)} u."
    )


ANSWER_FORMAT = (
    'ANSWER. Only a JSON object of this form (the numbers and tags here are an example, not advice):\n'
    '{"order": 2, "h": 50, "refinements": [{"curve": 7, "size": 4, "radius": 60}, '
    '{"corner": 2, "size": 2.5, "radius": 100}]}\n'
    'order: 1 (linear) or 2 (quadratic). h: the global element size in u. Each refinement sets the element size '
    '"size" (u) at a curve ("curve": tag), a corner ("corner": tag) or a named group ("group": name), growing '
    'back to h over the distance "radius" (u). "refinements": [] gives a uniform mesh.'
)


def prompt(pid: str, playbook: str = "") -> str:
    """The user message the student gets. `playbook`: lessons to put in context (the playbook method)."""
    parts = [f"TASK. Mesh this part for a linear static stress analysis.\n\n{statement(pid)}",
             geometry_text(geometry(pid)), scoring_text(pid)]
    if playbook:
        parts.append("LESSONS FROM AN EXPERT (apply the ones that fit this part):\n" + playbook.strip())
    parts.append(ANSWER_FORMAT)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- parse + convert


class SpecError(ValueError):
    pass


def parse(text: str) -> tuple[dict | None, str | None]:
    """The first JSON object in the model's answer (code fences and a thinking block are skipped)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj, None
                    break
        start = text.find("{", start + 1)
    return None, "no JSON object in the answer"


def _num(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise SpecError(f"{what} must be a number, got {x!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise SpecError(f"{what} must be positive, got {x!r}")
    return v


def _tag_of(x, prefix: str) -> int:
    if isinstance(x, str):
        x = x.strip().lstrip(prefix)
    try:
        return int(x)
    except (TypeError, ValueError):
        raise SpecError(f"bad {prefix} tag {x!r}") from None


def to_request(pid: str, spec: dict) -> dict:
    """Student spec (u, tags) -> ace.fea.make_mesh arguments (absolute). Raises SpecError."""
    geo = geometry(pid)
    if not isinstance(spec, dict):
        raise SpecError("the answer must be a JSON object")
    order = spec.get("order", 2)
    try:
        order = int(order)
    except (TypeError, ValueError):
        raise SpecError(f"order must be 1 or 2, got {order!r}") from None
    if order not in (1, 2):
        raise SpecError(f"order must be 1 or 2, got {order!r}")
    if "h" not in spec:
        raise SpecError("missing h")
    h = _num(spec["h"], "h")
    refs = spec.get("refinements") or []
    if not isinstance(refs, list):
        raise SpecError("refinements must be a list")
    if len(refs) > MAX_REFINEMENTS:
        raise SpecError(f"at most {MAX_REFINEMENTS} refinements")
    out = []
    for r in refs:
        if not isinstance(r, dict):
            raise SpecError(f"refinement {r!r} must be an object")
        size = _num(r.get("size"), "refinement size")
        if "box" in r:
            b = r["box"]
            if not isinstance(b, list) or len(b) != 2 * geo.dim:
                raise SpecError(f"box must be [{'xmin, ymin, xmax, ymax' if geo.dim == 2 else 'xmin..zmax'}] in u")
            lo, hi = geo.abs_point(b[:geo.dim]), geo.abs_point(b[geo.dim:])
            if geo.dim == 2:
                lo, hi = lo + [-1.0], hi + [1.0]
            out.append({"box": lo + hi, "size": size * geo.scale})
            continue
        radius = _num(r.get("radius"), "refinement radius")
        item = {"size": size * geo.scale, "radius": radius * geo.scale}
        if "corner" in r:
            k = _tag_of(r["corner"], "k")
            if k not in geo.corners:
                raise SpecError(f"unknown corner k{k}; corners: {sorted(geo.corners)}")
            item["point"] = list(geo.corners[k]["point"])
        elif "curve" in r or "edge" in r:
            c = _tag_of(r.get("curve", r.get("edge")), "c")
            if c not in geo.curves:
                raise SpecError(f"unknown curve c{c}; curves: {sorted(geo.curves)}")
            item["edge"] = c
        elif "group" in r:
            if r["group"] not in geo.groups:
                raise SpecError(f"unknown group {r['group']!r}; groups: {geo.groups}")
            item["group"] = r["group"]
        elif "point" in r:
            pt = r["point"]
            if not isinstance(pt, list) or len(pt) < geo.dim:
                raise SpecError("point must be [x, y] in u")
            item["point"] = geo.abs_point(pt)
        else:
            raise SpecError(f"refinement {r} needs 'corner', 'curve', 'group' or 'point'")
        out.append(item)
    return {"h": h * geo.scale, "order": order, "refinements": out}


def canonical(spec: dict) -> dict:
    """A spec rounded the way it is shown (3 significant digits), with integer tags: the training target."""
    def r3(x):
        x = float(x)
        return float(f"{x:.3g}") if x < 1000 else round(x)

    out = {"order": int(spec.get("order", 2)), "h": r3(spec["h"]), "refinements": []}
    for r in spec.get("refinements") or []:
        item = {}
        for key in ("corner", "curve", "group", "point", "box"):
            if key in r:
                item[key] = r[key] if key in ("group", "point", "box") else int(_tag_of(r[key], key[0]))
                break
        item["size"] = r3(r["size"])
        if "radius" in r:
            item["radius"] = r3(r["radius"])
        out["refinements"].append(item)
    for v in out["refinements"]:
        for k in ("size", "radius"):
            if k in v and float(v[k]).is_integer():
                v[k] = int(v[k])
    if float(out["h"]).is_integer():
        out["h"] = int(out["h"])
    return out


def dumps(spec: dict) -> str:
    return json.dumps(canonical(spec), separators=(", ", ": "))


# --------------------------------------------------------------------------- score


def _cache_path(pid: str, request: dict) -> Path:
    import gmsh
    key = json.dumps([pid, request, gmsh.__version__, 2], sort_keys=True)
    return CACHE_DIR / "score" / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def invalid(pid: str, why: str, **extra) -> dict:
    return {"pid": pid, "valid": False, "error": why, "x_auto": 0.0, "score": 0.0, **extra}


def score(pid: str, spec: dict | None, *, keep_mesh: str | Path | None = None, use_cache: bool = True) -> dict:
    """Mesh, solve and score one spec. Returns a flat dict (JSON-safe)."""
    from ace import config, fea, field_error
    from ace.checks import reference_setup
    problems.ensure_tasks([pid])
    if spec is None:
        return invalid(pid, "no spec")
    try:
        req = to_request(pid, spec)
    except SpecError as e:
        return invalid(pid, f"bad spec: {e}", spec=spec)
    f = _cache_path(pid, req)
    if use_cache and f.exists() and keep_mesh is None:
        return json.loads(f.read_text())
    t0 = time.perf_counter()
    problem = fea.get_problem(pid)
    cap = config.TASKS[pid]["node_cap"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.msh"
        try:
            mesh = fea.make_mesh(problem, path, h=req["h"], order=req["order"], refinements=req["refinements"])
        except Exception as e:  # noqa: BLE001  gmsh can raise on odd requests
            mesh = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        base = {"spec": canonical(spec), "request": req}
        if not mesh.get("ok"):
            out = invalid(pid, f"mesh failed: {mesh.get('error')}", **base)
        elif mesh["nodes"] > cap:
            out = invalid(pid, f"over the node cap ({mesh['nodes']} > {cap})", nodes=mesh["nodes"], **base)
        elif mesh["min_quality"] < config.MIN_QUALITY:
            out = invalid(pid, f"min quality {mesh['min_quality']} < {config.MIN_QUALITY}", nodes=mesh["nodes"], **base)
        elif mesh.get("folded_cells"):
            out = invalid(pid, f"{mesh['folded_cells']} folded cells", nodes=mesh["nodes"], **base)
        else:
            try:
                res = field_error._solve(problem, mesh, reference_setup(problem), config.TASKS[pid]["qoi"])
                errs = field_error.errors_of_result(pid, res)
                dofs = int(res.info.n_dofs)
            except Exception as e:  # noqa: BLE001
                res, errs = None, {"error_msg": f"{type(e).__name__}: {e}"}
            if res is None:
                out = invalid(pid, f"solve failed: {errs['error_msg']}", nodes=mesh["nodes"], **base)
            else:
                auto = field_error.automesher(pid)
                e = max(errs["error"], config.FIELD_ERR_FLOOR)
                g = auto["error"] / e
                c = dofs / auto["dofs"]
                G = config.AUTO_GAIN_TARGET
                A = (g / G) ** 2 if g < G else g / G
                out = {"pid": pid, "valid": True, "error": None, "score": A / c, "x_auto": A / c * G ** 2,
                       "gain": g, "cost": c, "e": errs["error"], "field": errs["field"], "points": errs["points"],
                       "nodes": mesh["nodes"], "dofs": dofs, "order": req["order"], "min_quality": mesh["min_quality"],
                       "auto_error": auto["error"], "auto_dofs": auto["dofs"], **base}
                if keep_mesh is not None:
                    Path(keep_mesh).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, keep_mesh)
                    out["mesh_path"] = str(keep_mesh)
    out["seconds"] = round(time.perf_counter() - t0, 2)
    if use_cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmpf = f.with_suffix(f".{os.getpid()}.tmp")
        tmpf.write_text(json.dumps({k: v for k, v in out.items() if k != "mesh_path"}))
        os.replace(tmpf, f)
    return out


def _score_job(job: tuple) -> dict:
    pid, spec, keep = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return score(pid, spec, keep_mesh=keep)


def score_many(jobs: list[tuple], workers: int = 3) -> list[dict]:
    """[(pid, spec, keep_mesh_path_or_None)] -> results, in order (gmsh is per process: one job per worker)."""
    if workers <= 1 or len(jobs) <= 1:
        return [_score_job(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_score_job, jobs))


def score_text(pid: str, text: str, **kw) -> dict:
    """Parse a model answer and score it."""
    spec, err = parse(text)
    if spec is None:
        return invalid(pid, err or "unparseable", answer=text[:2000])
    out = score(pid, spec, **kw)
    return out


# --------------------------------------------------------------------------- CLI (the teacher's tool)


def main(argv=None) -> int:
    """python -m wsp.task prompt <pid>            print the student's prompt
    python -m wsp.task score <pid> '<json spec>'  score a spec (what the teacher uses to verify)
    python -m wsp.task list                       the train and test ids"""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "list":
        print("train:", " ".join(problems.train_ids()))
        print("test:", " ".join(problems.test_ids()))
        return 0
    cmd, pid = argv[0], argv[1]
    problems.ensure_tasks([pid])
    if cmd == "prompt":
        print(prompt(pid))
        return 0
    if cmd == "score":
        # optional: --log FILE appends {"pid", "spec", result..., "note"} (the teacher's attempt log)
        log = argv[argv.index("--log") + 1] if "--log" in argv else None
        note = argv[argv.index("--note") + 1] if "--note" in argv else ""
        spec = json.loads(argv[2])
        r = score(pid, spec)
        keys = ("valid", "error", "x_auto", "gain", "cost", "e", "field", "points", "nodes", "dofs", "seconds")
        short = {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in keys if k in r}
        print(json.dumps(short))
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(json.dumps({"pid": pid, "spec": canonical(spec) if r.get("spec") else spec, **short,
                                     "note": note}) + "\n")
        return 0
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
