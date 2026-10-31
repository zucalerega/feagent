"""Mesher: agent-edited YAML spec -> gmsh volume mesh with named, verified boundary groups.

One gmsh session per run: load CAD (cfd external: box minus body) -> analyze geometry -> resolve
face groups by geometric query (count- and overlap-checked) -> size fields -> mesh -> physical
groups -> exports + manifest.json. A group that does not match its expected count fails the run
with a near-miss explanation and no mesh is written: a wrong boundary condition never ships.

CLI: ``python -m feagent.mesher SPEC --scale S --json`` writes one MeshResult JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np
import yaml

from .errors import CadLoadError, FeagentError, QueryError, SpecError
from .gates import gate
from .geometry import analyze, analyze_current_model
from .gmsh_session import gmsh_messages, gmsh_session, load_cad
from .gmsh_util import entity_tags
from .select import explain_miss, select_faces
from .spec import DomainSpec, GroupSpec, MeshSpec, load_spec
from .types import Gate, GeometryModel, GroupResult, MeshResult

DOMAIN_GROUP = "domain"
_STAGES = {"geometry": "CAD geometry", "groups": "Face groups", "mesh_generated": "Mesh generated"}
_GATE_NAMES = {**_STAGES, "spec": "Mesh spec"}
_EXPORT_FILES = {"msh22": "mesh_v22.msh", "inp": "mesh.inp", "vtk": "mesh.vtk",
                 "unv": "mesh.unv", "su2": "mesh.su2"}
_EXPORT_OPTIONS = {"msh22": {"Mesh.MshFileVersion": 2.2},
                   "inp": {"Mesh.SaveGroupsOfNodes": 1, "Mesh.SaveGroupsOfElements": 1}}
_MESH_FILES = ("mesh.msh", *_EXPORT_FILES.values())
_CONTRACT = (*_MESH_FILES, "manifest.json", "mesh.log")

RefineTarget = tuple[list[int], float, float]  # (face tags, size at the faces, distance)


@dataclass
class _Run:
    """Mutable state of one meshing run inside a gmsh session."""
    spec: MeshSpec
    scale: float
    out: Path
    t0: float = field(default_factory=time.perf_counter)
    model: GeometryModel | None = None
    faces: dict[str, list[int]] = field(default_factory=dict)
    refine: list[RefineTarget] = field(default_factory=list)
    groups: dict[str, GroupResult] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)
    mesh_path: str | None = None
    n_nodes: int = 0
    n_elements: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    @property
    def h(self) -> float:
        return self.spec.size.global_ * self.scale


# ----------------------------------------------------------------------------------------------
# public API

def build_domain(spec: MeshSpec) -> list[tuple[int, int]]:
    """Load the CAD into the current session; for cfd external flow return box minus body."""
    vols = load_cad(spec.geometry, heal=spec.heal)
    if spec.analysis != "cfd" or spec.domain.type != "external" or not vols:
        return vols
    lo, hi = _bounds(vols)
    box = _domain_box(spec.domain, lo, hi)
    fluid, _ = gmsh.model.occ.cut([(3, box)], vols)
    gmsh.model.occ.synchronize()
    return fluid


def resolve_groups(spec: MeshSpec, model: GeometryModel) -> dict[str, list[int]]:
    """Face tags per group (spec groups, then default_group = remaining boundary faces).

    Raises QueryError on count mismatches (one near-miss explanation per failing group) or a
    face in two groups, or a group named like the reserved volume group."""
    _check_reserved(spec)
    out, misses = {}, []
    for name, g in spec.groups.items():
        try:
            out[name] = _resolve_one(model, g)
        except QueryError as e:
            misses.append(e)
    if misses:
        hint = misses[0].hint if len(misses) == 1 else "fix select (or expect) of each group above"
        raise QueryError("\n\n".join(e.message for e in misses), hint=hint)
    _check_overlaps(out)
    rest = _unassigned_boundary(model, out)
    if spec.default_group and rest:
        out[spec.default_group] = rest
    return out


def run_mesh(spec: MeshSpec, size_scale: float = 1.0, out_dir: Path | None = None) -> MeshResult:
    """Mesh `spec` at global size x `size_scale`; writes the file contract into the out dir."""
    run = _Run(spec=spec, scale=float(size_scale), out=_out_dir(spec, size_scale, out_dir))
    _clean(run.out)
    with gmsh_session():
        try:
            _ = _stage_geometry(run) and _stage_groups(run) and _stage_mesh(run)
        finally:
            _write_log(run.out)
    res = _result(run)
    (run.out / "manifest.json").write_text(_dumps(res, indent=1))
    return res


def run_mesh_subprocess(spec_path: str | Path, size_scale: float = 1.0,
                        timeout: float = 900) -> MeshResult:
    """Run the mesher in a child process (isolates gmsh crashes/hangs) and parse its JSON."""
    cmd = [sys.executable, "-m", "feagent.mesher", str(spec_path), "--scale",
           repr(float(size_scale)), "--json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _failed(spec_path, size_scale, f"mesh timed out after {timeout:g}s",
                       "mesh_generated", elapsed=float(timeout))
    lines = p.stdout.strip().splitlines()
    try:
        return MeshResult.from_dict(json.loads(lines[-1]))
    except (IndexError, ValueError, KeyError, TypeError):
        tail = " | ".join(p.stderr.strip().splitlines()[-5:])
        return _failed(spec_path, size_scale, f"mesher process exited with code {p.returncode} "
                       f"without a result: {tail}", "mesh_generated")


_AUTO_HEADER = ("# Auto-generated by feagent.mesher.auto_spec from a quick geometry analysis.\n"
                "# It is a starting point meant to be edited: add groups (with expect counts),\n"
                "# refine regions and sizes, then re-run the mesher.\n")


def auto_spec(geometry: str | Path, analysis: str = "fea",
              spec_path: str | Path | None = None) -> dict:
    """Default spec dict for `geometry` (size from its diagonal); written as YAML if spec_path."""
    if analysis not in ("fea", "cfd"):
        raise ValueError(f"analysis must be 'fea' or 'cfd', got {analysis!r}")
    geo = Path(geometry).resolve()
    h = float(f"{analyze(geo, tessellate=False).diag / 40.0:.2g}")
    fea = analysis == "fea"
    # cfd meshes are linear: 24 elements per 2*pi keeps curved walls within the 1 % volume gate
    # (tee: 1.24 % at 12, 0.66 % at 24); fea is quadratic and fine at 12.
    size = {"global": h, "min": float(f"{h / 20.0:.2g}"), "curvature": 12 if fea else 24,
            "thin_wall_layers": 2}
    d: dict = {"name": geo.stem, "geometry": str(geo), "analysis": analysis, "groups": {},
               "size": size, "element": {"order": 2 if fea else 1, "optimize": "netgen"},
               "export": ["msh", "inp"] if fea else ["msh", "msh22", "su2"]}
    if not fea:
        d.update(default_group="walls", domain={"type": "internal"})
    if spec_path is not None:
        sp = Path(spec_path).resolve()
        d["geometry"] = os.path.relpath(geo, sp.parent)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(_AUTO_HEADER + yaml.safe_dump(d, sort_keys=False))
    return d


def auto_mesh(geometry: str | Path, analysis: str = "fea",
              out_dir: str | Path | None = None) -> MeshResult:
    """Auto-generate a spec for `geometry` and mesh it.

    The spec is written to `<out_dir>/mesh.yaml` (default out_dir: `<geometry dir>/<stem>_mesh`);
    the mesh files go to the spec's output dir, `<out_dir>/out`."""
    geo = Path(geometry)
    d = Path(out_dir) if out_dir is not None else geo.resolve().parent / f"{geo.stem}_mesh"
    spec_path = d / "mesh.yaml"
    auto_spec(geo, analysis, spec_path=spec_path)
    return run_mesh(load_spec(spec_path))


# ----------------------------------------------------------------------------------------------
# domain

def _bounds(dimtags: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    bb = np.array([gmsh.model.occ.getBoundingBox(d, t) for d, t in dimtags], float)
    return bb[:, :3].min(axis=0), bb[:, 3:].max(axis=0)


def _domain_box(d: DomainSpec, lo: np.ndarray, hi: np.ndarray) -> int:
    """Box around the body: up/downstream in body lengths along the flow axis, lateral padding
    in multiples of the largest lateral extent."""
    i = "xyz".index(d.flow_axis[1])
    ext = hi - lo
    length, width = ext[i], max(ext[j] for j in range(3) if j != i)
    pad_lo, pad_hi = np.full(3, d.lateral * width), np.full(3, d.lateral * width)
    up, down = d.upstream * length, d.downstream * length
    pad_lo[i], pad_hi[i] = (up, down) if d.flow_axis[0] == "+" else (down, up)
    a, b = lo - pad_lo, hi + pad_hi
    return gmsh.model.occ.addBox(*(float(x) for x in a), *(float(x) for x in b - a))


# ----------------------------------------------------------------------------------------------
# groups

def _count_ok(n: int, expect: int | tuple[int, int] | None) -> bool:
    if expect is None:
        return n >= 1
    if isinstance(expect, tuple):
        return expect[0] <= n <= expect[1]
    return n == expect


def _resolve_one(model: GeometryModel, g: GroupSpec) -> list[int]:
    tags = select_faces(model, g.select)
    if not _count_ok(len(tags), g.expect):
        raise QueryError(f"group '{g.name}': " + explain_miss(model, g.select, g.expect),
                         hint=f"fix groups.{g.name}.select (or groups.{g.name}.expect if the "
                              f"listed faces are right)")
    return tags


def _check_reserved(spec: MeshSpec) -> None:
    if DOMAIN_GROUP in spec.groups:
        raise QueryError(f"group name '{DOMAIN_GROUP}' is reserved for the volume group",
                         hint=f"rename groups.{DOMAIN_GROUP}")
    if spec.default_group == DOMAIN_GROUP:
        raise QueryError(f"default_group '{DOMAIN_GROUP}' is reserved for the volume group",
                         hint="set default_group to another name, e.g. walls or boundary")


def _check_overlaps(groups: dict[str, list[int]]) -> None:
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(groups[a]) & set(groups[b]))
            if common:
                raise QueryError(f"faces {common} are in groups {a} and {b}",
                                 hint="a face carries one boundary condition; narrow one query "
                                      "(e.g. add a 'not:' clause)")


def _unassigned_boundary(model: GeometryModel, groups: dict[str, list[int]]) -> list[int]:
    taken = {t for tags in groups.values() for t in tags}
    return sorted(f.tag for f in model.faces if len(f.volumes) == 1 and f.tag not in taken)


def _refine_targets(run: _Run) -> list[RefineTarget]:
    out = []
    for i, r in enumerate(run.spec.refine):
        tags = run.faces.get(r.group, []) if r.group else select_faces(run.model, r.select)
        if not tags:
            run.warnings.append(f"refine[{i}] matched no faces; ignored")
            continue
        out.append((tags, r.size * run.scale, r.distance))
    return out + _thin_wall_targets(run)


def _thin_wall_targets(run: _Run) -> list[RefineTarget]:
    """One target per thin_wall hazard: size thickness/layers within 2*thickness of its faces."""
    layers = run.spec.size.thin_wall_layers
    if layers <= 0:
        return []
    out = []
    for z in run.model.hazards:
        if z.kind != "thin_wall" or not z.value:
            continue
        tags = [int(t) for d, t in z.entities if d == 2]
        # Hazards come from the unscaled ref size and the size scales with size_scale, so every
        # scale of a study refines the same faces by the same factor (a systematic refinement).
        size = min(float(z.value) / layers * run.scale, run.h)
        out.append((tags, size, 2.0 * float(z.value)))
        run.warnings.append(f"thin wall t={z.value:.3g} on faces {tags}: auto-refined to size "
                            f"{size:.3g} (size.thin_wall_layers={layers})")
    return out


# ----------------------------------------------------------------------------------------------
# stages (each appends its gate; False stops the pipeline)

def _gate(gid: str, status: str, message: str, value: float | None = None,
          threshold: float | None = None, hint: str = "") -> Gate:
    return gate(status, gid, _GATE_NAMES[gid], message, value, threshold, hint)


def _fail(run: _Run, gid: str, message: str, hint: str = "") -> bool:
    run.errors.append(f"{message} (hint: {hint})" if hint else message)
    run.gates.append(_gate(gid, "fail", message, hint=hint))
    return False


def _gmsh_error(what: str, e: Exception) -> str:
    lines = gmsh_messages("Error")[-5:]
    return f"{what}: {e}" + ("; gmsh: " + " | ".join(lines) if lines else "")


def _stage_geometry(run: _Run) -> bool:
    try:
        vols = build_domain(run.spec)
        # unscaled ref size: hazards (thin-wall refinement targets) must not depend on size_scale
        run.model = analyze_current_model(ref_size=run.spec.size.global_,
                                          source=str(run.spec.geometry))
    except CadLoadError as e:
        return _fail(run, "geometry", e.message, e.hint)
    except Exception as e:
        return _fail(run, "geometry", _gmsh_error("building the domain failed", e),
                     "check the CAD (heal: true) and the domain settings")
    no_vol = [z for z in run.model.hazards if z.kind == "no_volume"]
    if no_vol:
        return _fail(run, "geometry", no_vol[0].message, no_vol[0].suggestion)
    run.gates.append(_gate("geometry", "pass", f"{len(vols)} volume(s), {len(run.model.faces)} "
                           f"faces", value=float(len(vols)), threshold=1.0))
    return True


def _stage_groups(run: _Run) -> bool:
    try:
        run.faces = resolve_groups(run.spec, run.model)
        run.refine = _refine_targets(run)
    except FeagentError as e:
        return _fail(run, "groups", e.message, e.hint)
    dg = run.spec.default_group
    if dg and dg not in run.faces:
        run.warnings.append(f"default_group '{dg}' is empty: every boundary face is in a group")
    summary = ", ".join(f"{k}={len(v)}" for k, v in run.faces.items()) or "none"
    run.gates.append(_gate("groups", "pass", f"{len(run.faces)} group(s) resolved (faces: "
                           f"{summary})", value=float(len(run.faces))))
    return True


def _stage_mesh(run: _Run) -> bool:
    try:
        _physical_groups(run.faces)
        _apply_options(run)
        _generate(run.spec)
        n_tets = _count_elements(3)
        if n_tets:
            _collect(run, n_tets)
            _export(run)
    except Exception as e:
        _remove_mesh_files(run)
        return _fail(run, "mesh_generated", _gmsh_error("gmsh meshing failed", e),
                     "try size.global smaller/larger, heal: true, or another algorithm.mesh3d")
    if not n_tets:
        return _fail(run, "mesh_generated", _gmsh_error("gmsh produced 0 volume elements",
                                                        RuntimeError("empty mesh")))
    run.gates.append(_gate("mesh_generated", "pass", f"{n_tets} volume elements, {run.n_nodes} "
                           f"nodes", value=float(n_tets), threshold=1.0))
    return True


# ----------------------------------------------------------------------------------------------
# gmsh meshing

def _physical_groups(faces: dict[str, list[int]]) -> None:
    for name, tags in faces.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, entity_tags(3), name=DOMAIN_GROUP)


def _apply_options(run: _Run) -> None:
    s, k = run.spec.size, run.scale
    opts = {"Mesh.MeshSizeMax": s.global_ * k, "Mesh.MeshSizeMin": s.min * k,
            "Mesh.MeshSizeFromCurvature": s.curvature, "Mesh.MeshSizeFromPoints": 0,
            "Mesh.Algorithm": run.spec.algorithm_2d, "Mesh.Algorithm3D": run.spec.algorithm_3d}
    for key, v in opts.items():
        gmsh.option.setNumber(key, float(v))
    fields = [_threshold(tags, size, dist, run.h) for tags, size, dist in run.refine]
    if fields:
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)


def _threshold(tags: list[int], size: float, dist: float, size_max: float) -> int:
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    f.setNumbers(d, "SurfacesList", tags)
    t = f.add("Threshold")
    f.setNumber(t, "InField", d)
    for key, v in (("SizeMin", size), ("SizeMax", size_max), ("DistMin", 0.0), ("DistMax", dist)):
        f.setNumber(t, key, float(v))
    return t


def _generate(spec: MeshSpec) -> None:
    m = gmsh.model.mesh
    m.generate(3)
    if spec.optimize != "none":
        m.optimize("Netgen" if spec.optimize == "netgen" else "")
    if spec.order > 1:
        m.setOrder(spec.order)
        m.optimize("HighOrder")


def _count_elements(dim: int, tag: int = -1) -> int:
    _, tags, _ = gmsh.model.mesh.getElements(dim, tag)
    return int(sum(len(t) for t in tags))


def _collect(run: _Run, n_tets: int) -> None:
    run.n_nodes = int(len(gmsh.model.mesh.getNodes()[0]))
    run.n_elements = n_tets
    for name, tags in run.faces.items():
        run.groups[name] = GroupResult(
            name=name, dim=2, tags=[int(t) for t in tags],
            elements=sum(_count_elements(2, t) for t in tags),
            area=float(sum(run.model.face(t).area for t in tags)))
    run.groups[DOMAIN_GROUP] = GroupResult(name=DOMAIN_GROUP, dim=3, tags=entity_tags(3),
                                           elements=n_tets, area=None)


# ----------------------------------------------------------------------------------------------
# files

@contextmanager
def _options(opts: dict[str, float]):
    saved = {k: gmsh.option.getNumber(k) for k in opts}
    try:
        for k, v in opts.items():
            gmsh.option.setNumber(k, v)
        yield
    finally:
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)


def _export(run: _Run) -> None:
    run.out.mkdir(parents=True, exist_ok=True)
    msh = run.out / "mesh.msh"
    with _options({"Mesh.MshFileVersion": 4.1, "Mesh.Binary": 0}):
        gmsh.write(str(msh))
    run.mesh_path = run.exports["msh"] = str(msh)
    for name in run.spec.export:
        if name != "msh":
            path = run.out / _EXPORT_FILES[name]
            with _options(_EXPORT_OPTIONS.get(name, {})):
                gmsh.write(str(path))
            run.exports[name] = str(path)


def _remove_mesh_files(run: _Run) -> None:
    """A failed run leaves only manifest.json and mesh.log (no partial mesh/exports)."""
    for name in _MESH_FILES:
        (run.out / name).unlink(missing_ok=True)
    run.mesh_path, run.exports = None, {}


def _out_dir(spec: MeshSpec, scale: float, out_dir: Path | None) -> Path:
    base = Path(out_dir) if out_dir is not None else spec.out_dir()
    return base if scale == 1 else base / f"scale_{scale:g}"


def _clean(out: Path) -> None:
    """Remove a previous run's contract files so a failed run never leaves a stale mesh."""
    out.mkdir(parents=True, exist_ok=True)
    for name in _CONTRACT:
        (out / name).unlink(missing_ok=True)


def _write_log(out: Path) -> None:
    try:
        lines = gmsh.logger.get()
    except Exception:
        lines = []
    (out / "mesh.log").write_text("\n".join(lines) + "\n")


def _np_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def _dumps(res: MeshResult, indent: int | None = None) -> str:
    return json.dumps(res.to_dict(), indent=indent, default=_np_default)


# ----------------------------------------------------------------------------------------------
# results

def _result(run: _Run) -> MeshResult:
    done = {g.id for g in run.gates}
    run.gates += [_gate(gid, "skip", "skipped: an earlier stage failed")
                  for gid in _STAGES if gid not in done]
    ok = run.mesh_path is not None and all(g.status != "fail" for g in run.gates)
    return MeshResult(
        ok=ok, spec_path=str(Path(run.spec.path).resolve()), out_dir=str(run.out),
        mesh_path=run.mesh_path, exports=dict(run.exports), groups=run.groups,
        n_nodes=run.n_nodes, n_elements=run.n_elements, order=run.spec.order,
        size_scale=run.scale, elapsed_s=round(time.perf_counter() - run.t0, 3),
        errors=run.errors, warnings=run.warnings, gates=run.gates, geometry=run.model)


def _failed(spec_path: str | Path, scale: float, message: str, gid: str,
            elapsed: float = 0.0) -> MeshResult:
    """A MeshResult for a run that never produced one (bad spec, timeout, crashed process)."""
    return MeshResult(
        ok=False, spec_path=str(spec_path), out_dir="", mesh_path=None, exports={}, groups={},
        n_nodes=0, n_elements=0, order=1, size_scale=float(scale), elapsed_s=elapsed,
        errors=[message], warnings=[], gates=[_gate(gid, "fail", message)], geometry=None)


# ----------------------------------------------------------------------------------------------
# CLI

def _summary(res: MeshResult) -> str:
    head = (f"ok: {res.n_elements} elements, {res.n_nodes} nodes -> {res.mesh_path}" if res.ok
            else "FAILED")
    lines = [f"  {g.id}: {g.status} - {g.message}" + (f"\n    hint: {g.hint}" if g.hint else "")
             for g in res.gates]
    return "\n".join([head, *lines])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feagent.mesher", description=__doc__.split("\n")[0])
    ap.add_argument("spec", help="path to the YAML mesh spec")
    ap.add_argument("--scale", type=float, default=1.0, help="size scale factor (default 1)")
    ap.add_argument("--json", action="store_true", help="write the MeshResult as JSON to stdout")
    a = ap.parse_args(argv)
    try:
        res = run_mesh(load_spec(a.spec), a.scale)
    except SpecError as e:
        res = _failed(a.spec, a.scale, str(e), "spec")
    except Exception as e:  # --json must always yield exactly one document
        res = _failed(a.spec, a.scale, f"mesher crashed: {type(e).__name__}: {e}",
                      "mesh_generated")
    sys.stdout.write((_dumps(res) if a.json else _summary(res)) + "\n")
    sys.stdout.flush()
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
