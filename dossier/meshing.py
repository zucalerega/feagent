"""The meshes of the component: one study of quadratic tetrahedra, made once and shared by every engine.

    levels(project, geo, opts, extra_refinements=None) -> [MeshLevel]         coarse -> fine
    levels_with_journal(...)                           -> ([MeshLevel], Journal)   + the decision about the sizes
    part_for(project, component_id=None)               -> the registered feacore Problem of the component
    GROUP_SOLID = "solid"

The structural and the thermal engine call `levels()` with the same arguments and get the same files.

Physical groups in every mesh
    f<tag>             every CAD face (the tags of geo["faces"])
    <feature id>       every feature that has faces, for example FT-INLET-BORE
    <INT-ID>:<role>    every interface role that has faces, with "*" resolved, for example INT-ENV:external
                       (an addition to the contract: one group for a boundary condition on a whole role)
    solid              the cells
A feature without faces has no group; the journal then holds an error finding for it.

Sizes
    The meshes are made with ace.fea.make_mesh (order 2; it tries several surface meshers, straightens curved
    cells that fold, and reports the quality). `MeshLevel.level` is the index in the list: 0 is the coarsest.
    The levels are one size field scaled: the sizes of level k are those of level k + 1 times a scale, chosen by
    trial meshes (MAX_LEVEL_TRIALS) so that the representative size h_rep = (volume / cells)^(1/3) of level k is
    RATIO (1.4) times that of level k + 1, within LEVEL_RATIO_TOL (6 %). The scale is not 1.4, because the local
    refinement does not coarsen with the global size; a global size ratio of 1.4 gave a ratio of 1.17 of h_rep
    on the quick study. The field:
    a global size h, and a local size rho x h within a fixed distance of
      * every re-entrant edge (geo["edges"]["reentrant"]), and
      * every small fillet: a fillet face (a concave torus, or a concave part of a cylinder) whose quarter arc
        has fewer than ARC_ELEMENTS (3) elements of the global size of the finest level.
    rho is RHO (0.5), or larger when the mesher's size floor does not allow it: ace.fea.make_mesh makes no
    element smaller than 1 % of the bounding-box diagonal, and this module keeps that floor. The distance is
    1.5 times the wall thickness at the edge (geo["faces"][..]["thickness"], an estimate), not less than what
    keeps the size change per unit length under GRADING (0.4) on the coarsest level, and not more than 3 h.
    The global size of the finest level is found by trial meshes so that its node count is under
    opts.node_cap and, when the trials allow, over 70 % of it (the trials aim at 87 %). The first trial size
    comes from nodes ~ k (2.84 V / h^3 + 5.53 A / h^2) (constants fitted to uniform meshes of the example
    body); later trials fit that form, or a power law, to the trials made. A trial that is accepted is the
    finest level.

    `extra_refinements` are refinements in the format of ace.fea.make_mesh ({"group" | "edge" | "point" | "box",
    "size", "radius"}), with the sizes of the finest level; a coarser level scales the sizes, not the
    distances. They are part of the cache key, so a call with them makes its own meshes.

Cache
    <project.out>/meshes/index.json holds one entry per key = hash of (CAD sha256, length unit, groups,
    node_cap, number of levels, ratio, re-entrant edges, fillets, extra refinements, gmsh version, recipe
    version). An entry lists the levels with their sizes, refinements, node counts and the sha256 of each
    file. A call with a known key whose files are present and unchanged returns the cached levels without
    meshing. Entries of another CAD file or recipe are removed with their files. A lock file serializes
    calls of several processes on one project.

Checks on every level (results in the index; failures are findings in the journal)
    * the fixed solver's reader (feacore.solver._skfem_mesh.read_mesh) loads the file, and every group is there
      with at least one facet;
    * the volume of the curved mesh against the CAD volume (a minor finding over VOLUME_TOL = 1e-3, a major
      one over 1e-2; a quick run uses 1e-2 and 5e-2, see VOLUME_TOL_QUICK). Elements that are large against the radius of a hole or a fillet do not follow it: the
      coarse levels of the example differ by 0.2 % to 0.6 %, the levels with a global size under 5 mm by less
      than 0.1 %;
    * min element quality (gmsh minSICN of the straight cells) against ace.config.MIN_QUALITY;
    * no folded cell (MeshingError);
    * the ratio of the representative sizes of successive levels (finding under 1.15).

Limits, said plainly
    * The size floor (1 % of the diagonal: 2.42 mm for the example) is the smallest local size. A fillet of
      radius 3 mm then has at most 2 quadratic elements along its arc, and a 4 mm wall has fewer than 2
      elements through it. A peak stress at such a fillet is not resolved to better than a few percent; the
      decision about the sizes says how many elements the fillets and the thinnest wall get and asks for a
      review. A smaller floor needs a change in ace/fea.py.
    * gmsh is not bitwise reproducible across versions and machines: the cache key holds the gmsh version, and
      a mesh made again elsewhere has other node counts.
    * The node-count model for the first trial is fitted to one part. It only costs trials when it is wrong.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

from dossier import geometry, units
from dossier.model import MeshLevel
from dossier.records import Journal

GROUP_SOLID = "solid"
RATIO = 1.4                 # size ratio between successive levels
ORDER = 2                   # quadratic tetrahedra
RHO = 0.5                   # local size / global size at re-entrant edges and small fillets (target)
ARC_ELEMENTS = 3            # a fillet is small when its quarter arc has fewer global elements than this
GRADING = 0.4               # largest size change per unit length in a refined zone, on the coarsest level
FILL = (0.70, 0.87)         # the finest mesh: accepted over FILL[0] x node_cap, aimed at FILL[1] x node_cap
MAX_TRIALS = 6
MAX_LEVEL_TRIALS = 4       # trial meshes per coarser level, to get the size ratio RATIO
LEVEL_RATIO_TOL = 0.06     # the ratio of the representative sizes of successive levels: RATIO within 6 %
VOLUME_TOL = 1e-3           # mesh volume against CAD volume: a minor finding over it, a major one over 10 times it
# With opts.quick the meshes are coarse on purpose (about 2 times the size of the coarse level of a full run, and the
# curved volume error of a mesh goes with the square of the size). The coarse level of the example then differs by
# 1.8 % from the CAD volume. So a quick run uses a limit of 1 % for a minor finding and 5 % for a major one, and says so
# in the finding. A full run keeps VOLUME_TOL.
VOLUME_TOL_QUICK = 1e-2
VOLUME_MAJOR_QUICK = 5e-2
MIN_LEVEL_RATIO = 1.15      # successive levels must differ at least by this in representative size
RECIPE_VERSION = 2          # change it when the sizing rules change: cached meshes are then made again
C_VOLUME, C_SURFACE = 2.84, 5.53      # first guess: nodes ~ C_VOLUME V / h^3 + C_SURFACE A / h^2 (P2 tetrahedra)
INDEX_SCHEMA = 1


class MeshingError(RuntimeError):
    """The meshes cannot be made. The message says why."""


# =========================================================================== public


def part_for(project, component_id: str | None = None):
    """The registered feacore Problem of the component (a single_agent BlindPart of its CAD in metres), with
    the groups f<tag>, every feature id and `solid`: for ace.fea, ace.geometry_info and the renderers."""
    return geometry.part_for(project, component_id)


def levels(project, geo: dict, opts, extra_refinements: list[dict] | None = None) -> list[MeshLevel]:
    """The mesh levels of the component, coarse to fine (made on the first call, then read from the cache).
    `MeshLevel.path` is relative to project.out. Raises MeshingError."""
    return levels_with_journal(project, geo, opts, extra_refinements)[0]


def levels_with_journal(project, geo: dict, opts, extra_refinements: list[dict] | None = None
                        ) -> tuple[list[MeshLevel], Journal]:
    """`levels()` plus the records about the meshes: the decision about the sizes (id
    DEC-<component>-MESH-SIZES, the same on every call with the same options) and findings of the checks."""
    log = getattr(opts, "log", None) or (lambda s: None)
    comp_id = geo["component"]
    cad = geometry.si_cad(project, comp_id)
    if geo["cad"]["sha256"] != cad.sha256:
        raise MeshingError(f"the geometry record is for another CAD file than {cad.source.name}: run the geometry "
                           f"stage again")
    groups, empty = _groups(geo)
    plan = _plan(geo, cad, groups, opts, _check_extra(extra_refinements))
    key = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    mesh_dir = project.out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with _locked(mesh_dir):
        index = _read_index(mesh_dir)
        entry = index["entries"].get(key)
        cached = entry is not None and _files_ok(project, entry)
        if not cached:
            if entry is not None:
                log(f"meshes: the cached files of {key} are missing or changed; making them again")
            _remove(project, entry)
            entry = _make(project, geo, cad, plan, key, groups, log)
            index["entries"][key] = entry
            for other in [k for k, e in index["entries"].items()
                          if e["plan"]["cad"] != plan["cad"] or e["plan"]["recipe"] != RECIPE_VERSION]:
                _remove(project, index["entries"].pop(other))
            _write_index(mesh_dir, index)
    out = [MeshLevel(level=lv["level"], h=lv["h"], nodes=lv["nodes"], cells=lv["cells"], order=lv["order"],
                     min_quality=lv["min_quality"], h_rep=lv["h_rep"], path=lv["path"]) for lv in entry["levels"]]
    log(f"meshes ({'cached' if cached else 'made in %.0f s' % entry['seconds']}): "
        + ", ".join(f"L{m.level} {m.nodes} nodes" for m in out))
    return out, _journal(geo, entry, empty)


# =========================================================================== what decides the meshes


def _groups(geo: dict) -> tuple[dict[str, list[int]], list[str]]:
    """The named face groups of the meshes, and the features that have no faces (no group)."""
    groups = {fid: [int(t) for t in f["faces"]] for fid, f in geo["features"].items() if f["faces"]}
    empty = [fid for fid, f in geo["features"].items() if not f["faces"]]
    for iid, roles in geo.get("interfaces", {}).items():
        for role, tags in roles.items():
            if tags:
                groups[f"{iid}:{role}"] = [int(t) for t in tags]
    return groups, empty


def _check_extra(extra: list[dict] | None) -> list[dict]:
    out = []
    for r in extra or []:
        if not isinstance(r, dict) or sum(k in r for k in ("group", "edge", "point", "box")) != 1 \
                or not isinstance(r.get("size"), (int, float)) or r["size"] <= 0:
            raise MeshingError(f"extra refinement {r!r}: give one of group / edge / point / box, and a size > 0 (m)")
        if "box" not in r and not (isinstance(r.get("radius"), (int, float)) and r["radius"] > 0):
            raise MeshingError(f"extra refinement {r!r}: give a radius > 0 (m)")
        out.append(json.loads(json.dumps(r)))
    return out


def _plan(geo: dict, cad, groups: dict, opts, extra: list[dict]) -> dict:
    """Everything that decides the meshes, as plain data: its hash is the cache key."""
    import gmsh
    faces = geo["faces"]

    def wall(tags) -> float | None:
        t = [faces[str(k)].get("thickness") for k in tags if str(k) in faces]
        t = [x for x in t if x]
        return min(t) if t else None

    edges = {str(e["tag"]): wall(e["faces"]) for e in geo["edges"]["reentrant"]}
    fillets = {}
    for tag, f in faces.items():
        if f.get("fillet", f["kind"] == "torus") and f.get("radius"):
            fillets[tag] = {"radius": f["radius"], "wall": wall([tag])}
    walls = [f["thickness"] for f in faces.values() if f.get("thickness")]
    n_levels = int(getattr(opts, "levels", 3))
    cap = int(getattr(opts, "node_cap", 120_000))
    if n_levels < 1 or cap < 1:
        raise MeshingError(f"options: levels = {n_levels} and node_cap = {cap} must be 1 or more")
    return {"recipe": RECIPE_VERSION, "cad": cad.sha256, "length_unit": cad.length_unit, "component": geo["component"],
            "groups": groups, "node_cap": cap, "levels": n_levels, "ratio": RATIO, "order": ORDER,
            "edges": edges, "fillets": fillets, "extra": extra, "volume": geo["cad"]["volume"],
            "area": geo["cad"].get("area") or sum(f["area"] for f in faces.values()),
            "diagonal": geo["cad"].get("diagonal") or math.dist(*geo["cad"]["bbox"]),
            "thinnest_wall": min(walls) if walls else None, "gmsh": gmsh.__version__}


def _recipe(plan: dict, h: float) -> dict:
    """The size field of the finest level for the global size h: the local size and where it applies."""
    floor = plan["diagonal"] / 100.0                # the size floor of ace.fea.make_mesh (3D)
    h = max(h, floor)
    rho = min(1.0, max(RHO, floor / h))
    refs: list[dict] = []
    edges, small = [], []
    if rho < 0.95:
        r_min = (1.0 - rho) * h * RATIO ** (plan["levels"] - 1) / GRADING

        def radius(wall: float | None) -> float:
            r = 1.5 * wall if wall else 2.0 * h
            return float(f"{min(max(r, r_min), max(3.0 * h, r_min)):.6g}")

        for tag, wall in plan["edges"].items():
            refs.append({"edge": int(tag), "size": rho * h, "radius": radius(wall)})
            edges.append(int(tag))
        for tag, f in plan["fillets"].items():
            if 0.5 * math.pi * f["radius"] / h < ARC_ELEMENTS:
                refs.append({"group": f"f{tag}", "size": rho * h, "radius": radius(f["wall"])})
                small.append(int(tag))
    return {"h": h, "rho": rho, "floor": floor, "refinements": refs + [dict(r) for r in plan["extra"]],
            "edges": edges, "small_fillets": small}


def _scaled(recipe: dict, factor: float) -> tuple[float, list[dict]]:
    """(global size, refinements) of the level that is `factor` times coarser than the finest."""
    return recipe["h"] * factor, [dict(r, size=r["size"] * factor) for r in recipe["refinements"]]


# =========================================================================== making the meshes


def _mesh(part, path: Path, h: float, refinements: list[dict]) -> dict:
    from ace import fea
    t0 = time.time()
    with geometry._quiet():
        m = fea.make_mesh(part, path, h=h, order=ORDER, refinements=refinements)
    if not m.get("ok"):
        raise MeshingError(f"the mesher failed at a global size of {h:.4g} m: {m.get('error')}")
    if m.get("folded_cells"):
        raise MeshingError(f"the mesh with a global size of {h:.4g} m has {m['folded_cells']} folded cells")
    m["seconds"] = round(time.time() - t0, 1)
    return m


def _model_nodes(plan: dict, h: float) -> float:
    return C_VOLUME * plan["volume"] / h ** 3 + C_SURFACE * plan["area"] / h ** 2


def _size_for(fn, nodes: float, lo: float, hi: float) -> float:
    """The size h in [lo, hi] with fn(h) = nodes, for fn decreasing in h (bisection on log h)."""
    if fn(lo) <= nodes:
        return lo
    if fn(hi) >= nodes:
        return hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if fn(mid) > nodes else (lo, mid)
    return math.sqrt(lo * hi)


def _next_size(plan: dict, trials: list[dict], aim: float, lo: float, hi: float) -> float:
    """The next trial size for `aim` nodes, from the trials so far."""
    if len(trials) == 1:
        k = trials[0]["nodes"] / _model_nodes(plan, trials[0]["h"])
        return _size_for(lambda h: k * _model_nodes(plan, h), aim, lo, hi)
    # the two trials nearest to the aim in log(nodes); a pair that brackets the aim when there is one
    by = sorted(trials, key=lambda t: abs(math.log(t["nodes"] / aim)))
    a = by[0]
    b = next((t for t in by[1:] if (t["nodes"] - aim) * (a["nodes"] - aim) < 0), by[1])
    (h1, n1), (h2, n2) = (a["h"], a["nodes"]), (b["h"], b["nodes"])
    if abs(math.log(h1 / h2)) < 1e-6:
        return h1 * (n1 / aim) ** (1 / 2.5)
    # nodes = p / h^3 + q / h^2 through both trials, when both terms are positive; else a power law
    det = h1 ** -3 * h2 ** -2 - h2 ** -3 * h1 ** -2
    p = (n1 * h2 ** -2 - n2 * h1 ** -2) / det
    q = (h1 ** -3 * n2 - h2 ** -3 * n1) / det
    if p >= 0 and q >= 0:
        return _size_for(lambda h: p / h ** 3 + q / h ** 2, aim, lo, hi)
    e = min(3.5, max(1.5, math.log(n1 / n2) / math.log(h2 / h1)))
    return min(hi, max(lo, h1 * (n1 / aim) ** (1 / e)))


def _search(part, plan: dict, work: Path, log) -> tuple[dict, dict, list[dict]]:
    """Trial meshes of the finest level until one has between FILL[0] x cap and cap nodes (or the size floor
    is reached). Returns (recipe, make_mesh result of the accepted trial, the trials)."""
    cap = plan["node_cap"]
    floor, h_max = plan["diagonal"] / 100.0, plan["diagonal"] / 4.0
    aim = FILL[1] * cap
    trials: list[dict] = []
    # first guess: local refinement about doubles the nodes of a coarse mesh; a large mesh gets a cheap trial first
    k0 = 2.0 if plan["edges"] or plan["fillets"] or plan["extra"] else 1.2
    h = _size_for(lambda x: k0 * _model_nodes(plan, x), aim if aim <= 40_000 else aim / 3.0, floor, h_max)
    best = None
    for k in range(MAX_TRIALS):
        recipe = _recipe(plan, h)
        res = _mesh(part, work / f"trial{k}.msh", recipe["h"], recipe["refinements"])
        t = {"h": recipe["h"], "nodes": res["nodes"], "seconds": res["seconds"], "recipe": recipe, "result": res}
        trials.append(t)
        log(f"meshes: trial {k + 1}: h = {recipe['h'] * 1e3:.3g} mm, {res['nodes']} nodes ({res['seconds']:.0f} s)")
        if res["nodes"] <= cap and (best is None or res["nodes"] > best["nodes"]):
            best = t
        if best is t and (res["nodes"] >= FILL[0] * cap or recipe["h"] <= floor * (1 + 1e-9)):
            break
        if res["nodes"] > cap and recipe["h"] >= h_max * (1 - 1e-9):
            break
        h_new = _next_size(plan, trials, aim, floor, h_max)
        if any(abs(h_new / x["h"] - 1.0) < 5e-3 for x in trials):     # nothing new to learn from this size
            if best is not None:
                break
            h_new = max(x["h"] for x in trials) * 1.15
        h = h_new
    if best is None:
        raise MeshingError(f"no mesh under the node cap of {cap}: the coarsest trial has "
                           f"{min(t['nodes'] for t in trials)} nodes at a global size of "
                           f"{max(t['h'] for t in trials):.4g} m. Use a larger node cap.")
    return best["recipe"], best["result"], trials


def _verify(path: Path, groups: dict, cad_volume: float) -> dict:
    """Load the mesh with the fixed solver's reader, check the groups, and measure the curved volume."""
    from skfem import Basis, ElementTetP2

    from feacore.solver._skfem_mesh import MeshError, read_mesh
    try:
        md = read_mesh(path, 3, ORDER)
    except MeshError as e:
        raise MeshingError(f"the fixed solver cannot read {path.name}: {e}") from e
    missing = [g for g in (*groups, GROUP_SOLID) if g not in md.groups or not len(md.groups[g].index)]
    if missing:
        raise MeshingError(f"{path.name} has no elements in the group(s) {', '.join(missing)}")
    if md.groups[GROUP_SOLID].kind != "cells" or any(md.groups[g].kind != "facets" for g in groups):
        raise MeshingError(f"{path.name}: a face group is not a set of facets, or `{GROUP_SOLID}` is not the cells")
    volume = float(Basis(md.mesh, ElementTetP2(), intorder=3).dx.sum())
    return {"volume": volume, "volume_error": abs(volume / cad_volume - 1.0), "groups": len(md.groups)}


def _make(project, geo: dict, cad, plan: dict, key: str, groups: dict, log) -> dict:
    """Search the sizes, make every level, check it, and return the index entry."""
    t0 = time.time()
    part = geometry.part_for(project, geo["component"], groups=groups)
    mesh_dir = project.out / "meshes"
    work = mesh_dir / f"{key}.work"
    work.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        recipe, finest, trials = _search(part, plan, work, log)
        n = plan["levels"]
        records = []
        factor, prev_rep = 1.0, None
        for level in range(n - 1, -1, -1):                     # the finest first: it exists already
            path = mesh_dir / f"{key}_L{level}.msh"
            written.append(path)
            if level == n - 1:
                h, refs = _scaled(recipe, 1.0)
                Path(finest["path"]).replace(path)
                res = finest
                check = _verify(path, groups, plan["volume"])
            else:
                # The local refinement does not coarsen with h, so a size ratio of RATIO gives a smaller ratio
                # of the representative sizes. The scale of the level is adjusted until the representative
                # size (volume / cells)^(1/3) is RATIO times that of the next finer level.
                factor *= RATIO
                for attempt in range(MAX_LEVEL_TRIALS):
                    h, refs = _scaled(recipe, factor)
                    res = _mesh(part, path, h, refs)
                    check = _verify(path, groups, plan["volume"])
                    got = (check["volume"] / res["elements"]) ** (1.0 / 3.0) / prev_rep
                    log(f"meshes: level {level}, trial {attempt + 1}: h = {res['h_used'] * 1e3:.3g} mm, "
                        f"{res['nodes']} nodes, size ratio {got:.2f} ({res['seconds']:.0f} s)")
                    if abs(got / RATIO - 1.0) <= LEVEL_RATIO_TOL:
                        break
                    factor *= (RATIO / got) ** 2.0          # the size ratio goes about with the square root of the scale
            prev_rep = (check["volume"] / res["elements"]) ** (1.0 / 3.0)
            records.append({
                "level": level, "h": float(f"{res['h_used']:.6g}"), "nodes": int(res["nodes"]),
                "cells": int(res["elements"]), "order": ORDER, "min_quality": float(res["min_quality"]),
                "h_rep": float(f"{(check['volume'] / res['elements']) ** (1.0 / 3.0):.6g}"),
                "path": str(path.relative_to(project.out)), "sha256": geometry.sha256_of(path),
                "volume": check["volume"], "volume_error": float(f"{check['volume_error']:.3g}"),
                "groups": check["groups"], "quality": res["quality"], "repaired_cells": int(res["repaired_cells"]),
                "mesher": res["mesher"], "seconds": res["seconds"],
                "refinements": [dict(r, size=float(f"{s:.6g}")) for r, s in zip(refs, res["sizes_used"])],
            })
        records.reverse()
    except BaseException:
        for f in written:                   # no entry in the index: leave no file behind
            f.unlink(missing_ok=True)
        raise
    finally:
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()
    return {"key": key, "plan": plan, "made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1), "floor": recipe["floor"], "rho": float(f"{recipe['rho']:.4g}"),
            "edges": recipe["edges"], "small_fillets": recipe["small_fillets"],
            "trials": [{"h": float(f"{t['h']:.6g}"), "nodes": t["nodes"], "seconds": t["seconds"]} for t in trials],
            "levels": records}


# =========================================================================== the cache


@contextlib.contextmanager
def _locked(mesh_dir: Path):
    """One process at a time makes or reads the meshes of a project."""
    try:
        import fcntl
    except ImportError:                 # no file locks on this platform: one process per project is assumed
        yield
        return
    with open(mesh_dir / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_index(mesh_dir: Path) -> dict:
    path = mesh_dir / "index.json"
    if path.is_file():
        try:
            index = json.loads(path.read_text())
            if index.get("schema") == INDEX_SCHEMA and isinstance(index.get("entries"), dict):
                return index
        except ValueError:
            pass
    return {"schema": INDEX_SCHEMA, "entries": {}}


def _write_index(mesh_dir: Path, index: dict) -> None:
    tmp = mesh_dir / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1) + "\n")
    os.replace(tmp, mesh_dir / "index.json")


def _files_ok(project, entry: dict) -> bool:
    for lv in entry["levels"]:
        path = project.out / lv["path"]
        if not path.is_file() or geometry.sha256_of(path) != lv["sha256"]:
            return False
    return True


def _remove(project, entry: dict | None) -> None:
    for lv in (entry or {}).get("levels", []):
        (project.out / lv["path"]).unlink(missing_ok=True)


# =========================================================================== the records


def _mm(x: float) -> str:
    return units.fmt(units.from_si(float(x), "m"), "mm", 3)


def _journal(geo: dict, entry: dict, empty_features: list[str]) -> Journal:
    """The decision about the sizes and the findings of the checks, from the index entry (so that a cached
    call gives the same records as the call that made the meshes)."""
    from ace.config import MIN_QUALITY
    j = Journal("mesh", "engine:meshing")
    comp, plan, lv = geo["component"], entry["plan"], entry["levels"]
    n, fine = len(lv), entry["levels"][-1]
    sizes = [_mm(x["h"]) for x in lv]
    nodes = [str(x["nodes"]) for x in lv]

    def listing(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    if n == 1:
        choice = (f"The mesh study has 1 level of quadratic tetrahedra. The global size is {sizes[0]}. "
                  f"The mesh has {nodes[0]} nodes.")
    else:
        choice = (f"The mesh study has {n} levels of quadratic tetrahedra. The global sizes are {listing(sizes)}. "
                  f"The meshes have {listing(nodes)} nodes.")
    n_e, n_f = len(entry["edges"]), len(entry["small_fillets"])
    places = ([f"{n_e} re-entrant edge{'' if n_e == 1 else 's'}"] if n_e else []) \
        + ([f"{n_f} small fillet{'' if n_f == 1 else 's'}"] if n_f else [])
    where = listing(places) if places else ""
    if where:
        choice += f" The local size at {where} is {entry['rho']:.2f} times the global size."
    else:
        choice += " The meshes have no local refinement."
    if plan["extra"]:
        choice += f" The engine added {len(plan['extra'])} local refinement{'' if len(plan['extra']) == 1 else 's'}."
    choice += f" The mesh stage made {len(entry['trials'])} trial mesh{'' if len(entry['trials']) == 1 else 'es'} to find the sizes."

    floor_limits = bool(where) and entry["rho"] > RHO + 1e-9
    rationale = [f"The finest mesh must have fewer than {plan['node_cap']} nodes, and the representative element "
                 f"size of each level is {RATIO:g} times that of the next finer level.",
                 "The stress changes most at re-entrant edges and at small fillets, and the elements are smaller there."]
    if floor_limits or (not where and (plan["edges"] or plan["fillets"])):
        rationale.append(f"The mesher does not make elements smaller than {_mm(entry['floor'])}, which is 1 % of the "
                         f"diagonal of the part. This limit sets the local size.")
    arcs = None
    if plan["fillets"]:
        r_min = min(f["radius"] for f in plan["fillets"].values())
        local = fine["h"] * (entry["rho"] if entry["small_fillets"] else 1.0)
        arcs = 0.5 * math.pi * r_min / local
        rationale.append(f"The smallest fillet has a radius of {_mm(r_min)} and gets {arcs:.1f} elements along its arc "
                         f"on the finest mesh.")
    through = None
    if plan.get("thinnest_wall"):
        through = plan["thinnest_wall"] / fine["h"]
        rationale.append(f"The thinnest wall is approximately {_mm(plan['thinnest_wall'])}, and the finest mesh has "
                         f"{through:.1f} elements through it.")
    under = fine["nodes"] < FILL[0] * plan["node_cap"]
    j.decide(
        subject=comp, key="mesh-sizes", title="Sizes of the meshes", choice=choice, rationale=" ".join(rationale),
        alternatives=[("Use one global size with no local refinement.",
                       "Smaller elements at the re-entrant edges and the fillets give a smaller error for the same "
                       "number of nodes."),
                      ("Use a higher node limit.",
                       "The option node_cap of the run sets the limit. A higher limit gives a longer solve time.")],
        basis="calculation",
        needs_review=bool(floor_limits or under or (arcs is not None and arcs < 2.0) or (through is not None and through < 1.0)))

    for x in lv:
        if x["min_quality"] < MIN_QUALITY:
            j.find(rule="mesh.quality", severity="major", category="model", subject=comp, key=f"mesh-quality-L{x['level']}",
                   title=f"Element quality of mesh level {x['level']}",
                   statement=f"The mesh level {x['level']} has an element with a quality of {x['min_quality']:.2g}. "
                             f"The limit is {MIN_QUALITY:g}.",
                   rationale="The quality is the gmsh measure minSICN of the straight element. A flat element gives a "
                             "poor stress result near it.",
                   suggestion="Examine the results near the poor elements, or change the node limit to get other meshes.",
                   values={"level": x["level"], "min_quality": x["min_quality"], "limit": MIN_QUALITY})
    quick = bool(getattr(opts, "quick", False))
    vol_tol = VOLUME_TOL_QUICK if quick else VOLUME_TOL
    vol_major = VOLUME_MAJOR_QUICK if quick else 10 * VOLUME_TOL
    off = [x for x in lv if x["volume_error"] > vol_tol]
    if off:
        which = listing([str(x["level"]) for x in off])
        by = listing(["%.2g %%" % (100 * x["volume_error"]) for x in off])
        j.find(rule="mesh.volume", severity="major" if max(x["volume_error"] for x in off) > vol_major else "minor",
               category="model", subject=comp, key="mesh-volume", title="Volume of the meshes",
               statement=f"The volume of the mesh level{'' if len(off) == 1 else 's'} {which} differs from the CAD "
                         f"volume by {by}. The limit of this check is {100 * vol_tol:g} %."
                         + (f" The run is a quick run with coarse meshes, thus the limit is {100 * vol_tol:g} % and not "
                            f"{100 * VOLUME_TOL:g} %. A difference over {100 * vol_major:g} % is a major finding."
                            if quick else ""),
               rationale="An element that is large against the radius of a hole or of a fillet does not follow the "
                         "curved face. The mesher also makes a curved element straight when it folds.",
               suggestion="Use a result near a small hole or a small fillet only from a level that agrees with the CAD volume.",
               values={"cad_volume": plan["volume"], "levels": {str(x["level"]): x["volume"] for x in off}})
    for a, b in zip(lv, lv[1:]):
        ratio = a["h_rep"] / b["h_rep"]
        if ratio < MIN_LEVEL_RATIO:
            j.find(rule="mesh.ratio", severity="minor", category="model", subject=comp,
                   key=f"mesh-ratio-L{a['level']}-L{b['level']}", title=f"Size ratio of the mesh levels {a['level']} and {b['level']}",
                   statement=f"The element size of the mesh levels {a['level']} and {b['level']} differs by a factor of "
                             f"{ratio:.2f}. A mesh study needs a factor of {MIN_LEVEL_RATIO:g} or more.",
                   rationale="Small features of the part set the element size on the two levels. The change between "
                             "the levels then shows little about the convergence.",
                   suggestion="Use a higher node limit, or fewer levels.",
                   values={"h_rep": [a["h_rep"], b["h_rep"]]})
    for fid in empty_features:
        j.find(rule="mesh.feature_group", severity="error", category="model", subject=fid, key="no-mesh-group",
               title=f"No mesh group for the feature {fid}",
               statement=f"The meshes have no group for the feature {fid}, because the feature has no face.",
               rationale="A load or a support on this feature cannot go into an analysis.",
               suggestion="Correct the selection of the feature. Then make the meshes again.")
    return j
