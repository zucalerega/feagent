"""Job bodies, run in a fresh child process (gmsh is process-global and not thread-safe).

    python -m feacore.ui.worker '{"kind": "solve", "root": "runs", "run_id": "run-..."}'
    python -m feacore.ui.worker '{"kind": "study", "root": "runs", "study_id": "study-..."}'
    python -m feacore.ui.worker '{"kind": "geometry", "problem_id": "...", "out": "x.json"}'
    python -m feacore.ui.worker '{"kind": "backends"}'          # prints JSON

Results go to the run store (or ``out``); the exit code is 0 on success.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

# --------------------------------------------------------------------------- extra (out-of-repo) problems

_LOADED: set[str] = set()


def load_extra_problems() -> list[str]:
    """Import modules listed in ``$FEACORE_EXTRA_PROBLEMS`` (comma-separated module names or
    .py paths); they register problems with ``feacore.problems.registry.register``. Used for
    scratch problems that are not part of the repo. Returns errors."""
    errors = []
    for item in filter(None, (s.strip() for s in os.environ.get("FEACORE_EXTRA_PROBLEMS", "").split(","))):
        if item in _LOADED:
            continue
        _LOADED.add(item)
        try:
            if item.endswith(".py"):
                name = "feacore_extra_" + Path(item).stem
                spec = importlib.util.spec_from_file_location(name, item)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            else:
                importlib.import_module(item)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{item}: {type(e).__name__}: {e}")
    return errors


# --------------------------------------------------------------------------- solve


def run_solve(root: str, run_id: str, fields: bool = True) -> int:
    from feacore.case import Case
    from feacore.runstore import RunStore, now_iso
    from feacore.solver import solve

    store = RunStore(root)
    rec = store.update_run(run_id, status="running", started_at=now_iso(), pid=os.getpid())
    try:
        case = Case.from_dict(rec["case"], base_dir=store.run_dir(run_id))
        res = solve(case, backend=rec["backend"])
        qois = {k: float(v) for k, v in res.evaluate_all().items()}
        update = {"info": asdict(res.info), "qois": qois}
        if fields:
            try:
                update["fields"] = store.write_fields(run_id, res, case.mesh_path())
            except Exception as e:  # noqa: BLE001 - viewer data is optional
                update["warnings"] = [f"field extraction failed: {type(e).__name__}: {e}"]
                traceback.print_exc()
        store.update_run(run_id, status="succeeded", finished_at=now_iso(), **update)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        store.update_run(run_id, status="failed", finished_at=now_iso(), error=f"{type(e).__name__}: {e}",
                         traceback=traceback.format_exc())
        return 1


# --------------------------------------------------------------------------- study


class _ProblemProxy:
    """Delegates to a Problem; reports when each mesh level starts (progress)."""

    def __init__(self, problem, on_mesh):
        self._problem = problem
        self._on_mesh = on_mesh

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def mesh(self, path, h, geom_order=2, refine_steps=0):
        self._on_mesh(Path(path), float(h), int(refine_steps))
        if refine_steps:  # nested sequence: mesh once at the coarse size, then split
            return self._problem.mesh(path, h, geom_order=geom_order, refine_steps=refine_steps)
        return self._problem.mesh(path, h, geom_order=geom_order)


@contextmanager
def _hook_solve(on_before, on_solved):
    """Wrap the solver used by ``feacore.verification.study`` so each Result can be stored
    (fields for the viewer) while the study runs. If the study module stops using a
    module-level ``solve``, the hooks simply never fire and the study still runs."""
    import feacore.verification.study as study_mod

    real = getattr(study_mod, "solve", None)
    if real is None:
        yield
        return

    def wrapped(case, backend="skfem", **kw):
        try:
            on_before(case)
        except Exception:  # noqa: BLE001 - progress reporting must never break the study
            traceback.print_exc()
        t0 = time.perf_counter()
        res = real(case, backend=backend, **kw)
        try:
            on_solved(case, res, time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 - recording must never break the study
            traceback.print_exc()
        return res

    study_mod.solve = wrapped
    try:
        yield
    finally:
        study_mod.solve = real


def run_study(root: str, study_id: str) -> int:
    import inspect

    from feacore.problems.registry import get
    from feacore.runstore import RunStore, now_iso
    from feacore.ui.analysis import interpret_study
    from feacore.verification.study import convergence_study

    store = RunStore(root)
    st = store.get_study(study_id)
    if st.get("status") == "cancelled":
        return 0
    levels: list[dict] = []
    t_start = time.time()
    progress = {"done": 0, "total": None, "stage": "starting", "message": "loading problem", "levels": levels,
                "stage_started_at": t_start, "started_at": t_start}
    store.update_study(study_id, status="running", started_at=now_iso(), progress=progress, pid=os.getpid())
    run_ids: list[str] = []

    def stage(name: str, message: str, **kw) -> None:
        progress.update(stage=name, message=message, stage_started_at=time.time(), **kw)
        store.update_study(study_id, progress=progress)

    try:
        problem = get(st["problem_id"])
        order = int(st["order"])
        nested = bool(st.get("nested"))
        sizes = st.get("sizes") or problem.study_sizes(order)
        sizes = sorted((float(h) for h in sizes), reverse=True)
        if nested:  # same sequence convergence_study uses: coarse mesh split once per level
            sizes = [sizes[0] / 2**i for i in range(len(sizes))]
        levels.extend({"index": i, "h": h, "status": "pending"} for i, h in enumerate(sizes))
        progress.update(total=len(sizes))
        store.update_study(study_id, sizes=sizes, progress=progress)
        n = len(sizes)
        state = {"i": None, "t0": None}

        def on_mesh(path: Path, h: float, refine_steps: int):
            i = refine_steps if nested else min(range(n), key=lambda k: abs(math.log(sizes[k] / h)))
            state.update(i=i, t0=time.perf_counter())
            levels[i].update(status="meshing", started_at=time.time())
            how = f"split {refine_steps}x" if refine_steps else f"h = {h:.4g}"
            stage("meshing", f"mesh {i + 1}/{n}: meshing ({how})")

        def on_before(case):
            i = state["i"] if state["i"] is not None else len(run_ids)
            levels[i]["status"] = "solving"
            try:
                size_mb = case.mesh_path().stat().st_size / 1e6
                extra = f", mesh file {size_mb:.1f} MB"
            except OSError:
                extra = ""
            stage("solving", f"mesh {i + 1}/{n}: solving ({st['backend']}, P{order}{extra})")

        def on_solved(case, res, dt):
            i = state["i"] if state["i"] is not None else len(run_ids)
            info = res.info
            levels[i]["status"] = "storing"
            stage("storing", f"mesh {i + 1}/{n}: solved {info.n_dofs:,} dofs in {info.wall_time:.1f} s; "
                             f"storing fields for the viewer")
            rid = store.record_result(res, mesh_path=case.mesh_path(), problem_id=st["problem_id"],
                                      study_id=study_id, level=i, h_nominal=sizes[i], copy_mesh=False,
                                      source="ui")
            run_ids.append(rid)
            levels[i].update(status="done", run_id=rid, n_cells=info.n_cells, n_dofs=info.n_dofs,
                             h_rep=info.h_rep, wall_time=info.wall_time,
                             level_time=time.perf_counter() - (state["t0"] or time.perf_counter()),
                             qois=store.get_run(rid)["qois"])
            nxt = f"; mesh {i + 2} next" if i + 1 < n else "; analysing"
            progress["done"] = sum(l["status"] == "done" for l in levels)
            stage("solved", f"mesh {i + 1}/{n} done: {info.n_dofs:,} dofs, solve {info.wall_time:.1f} s{nxt}",
                  run_ids=run_ids)
            store.update_study(study_id, run_ids=run_ids)

        workdir = store.study_dir(study_id) / "meshes"
        workdir.mkdir(parents=True, exist_ok=True)
        kwargs = {}
        if nested:
            if "nested" not in inspect.signature(convergence_study).parameters:
                raise RuntimeError("this feacore.verification has no nested refinement")
            kwargs["nested"] = True
        with _hook_solve(on_before, on_solved):
            result = convergence_study(_ProblemProxy(problem, on_mesh), sizes=sizes, order=order,
                                       backend=st["backend"], geom_order=st.get("geom_order"), workdir=workdir,
                                       **kwargs)
        if not run_ids:  # hook did not fire: record the levels without viewer fields
            for i, r in enumerate(result.get("runs", [])):
                rec = store.create_run(backend=st["backend"], problem_id=st["problem_id"], study_id=study_id,
                                       level=i, h_nominal=r.get("h_nominal"), source="ui", status="succeeded")
                store.update_run(rec["run_id"], info=r.get("info"), qois=r.get("qois", {}), finished_at=now_iso())
                run_ids.append(rec["run_id"])
        result["problem"] = st["problem_id"]
        progress.update(done=n, stage="done", message="finished", stage_started_at=time.time())
        store.update_study(study_id, status="succeeded", finished_at=now_iso(), result=result,
                           interpretation=interpret_study(result), run_ids=run_ids, progress=progress,
                           elapsed=time.time() - t_start)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        for l in levels:
            if l["status"] in ("meshing", "solving", "storing"):
                l["status"] = "failed"
        progress.update(stage="failed", message=f"{type(e).__name__}: {e}", stage_started_at=time.time())
        store.update_study(study_id, status="failed", finished_at=now_iso(), error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc(), progress=progress, run_ids=run_ids,
                           elapsed=time.time() - t_start)
        return 1


# --------------------------------------------------------------------------- geometry preview


def geometry_preview(problem) -> dict:
    """Coarse surface triangulation of the problem's CAD (gmsh), with physical groups."""
    import gmsh
    import numpy as np

    from feacore.geometry.gmsh_tools import gmsh_session
    from feacore.ui.viewdata import encode_array

    t0 = time.perf_counter()
    with gmsh_session(f"preview_{problem.id}"):
        problem.build_geometry()
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) or 1.0
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 30)
        gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 3000)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        lut = np.full(int(tags.max()) + 1 if tags.size else 1, -1, dtype=np.int64)
        lut[tags] = np.arange(tags.size)

        def elements(dim, etype, nper):
            conn, ent = [], []
            for _, tag in gmsh.model.getEntities(dim):
                types, _, ntags = gmsh.model.mesh.getElements(dim, tag)
                for t, nt in zip(types, ntags):
                    if t == etype:
                        c = lut[np.asarray(nt, dtype=np.int64).reshape(-1, nper)]
                        conn.append(c)
                        ent.append(np.full(len(c), tag, dtype=np.int32))
            if not conn:
                return np.zeros((0, nper), dtype=np.int64), np.zeros(0, dtype=np.int32)
            return np.concatenate(conn), np.concatenate(ent)

        tris, tri_ent = elements(2, 2, 3)
        lines, line_ent = elements(1, 1, 2)
        groups = []
        entity_groups: dict[str, dict[str, list[int]]] = {"0": {}, "1": {}, "2": {}, "3": {}}
        for d, ptag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(d, ptag) or f"group_{d}_{ptag}"
            ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, ptag)]
            gi = len(groups)
            g = {"name": name, "dim": int(d), "entities": ents}
            if d == 0:
                g["points"] = [list(gmsh.model.getValue(0, e, [])) for e in ents]
            groups.append(g)
            for e in ents:
                entity_groups[str(d)].setdefault(str(e), []).append(gi)
        used = np.unique(np.concatenate([tris.ravel(), lines.ravel()])) if (tris.size or lines.size) else np.zeros(0, np.int64)
        remap = np.full(len(coords), -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        pts = coords[used]
        out = {
            "problem_id": problem.id,
            "dim": int(problem.dim),
            "bbox": [[xmin, ymin, zmin], [xmax, ymax, zmax]],
            "points": encode_array(pts, "float32"),
            "triangles": encode_array(remap[tris], "uint32"),
            "tri_entity": encode_array(tri_ent, "int32"),
            "lines": encode_array(remap[lines], "uint32"),
            "line_entity": encode_array(line_ent, "int32"),
            "groups": groups,
            "entity_groups": entity_groups,
            "n_triangles": int(len(tris)),
            "n_lines": int(len(lines)),
            "gmsh_version": gmsh.__version__ if hasattr(gmsh, "__version__") else None,
        }
    out["time"] = time.perf_counter() - t0
    return out


def run_geometry(problem_id: str, out: str) -> int:
    from feacore.problems.registry import get
    from feacore.runstore import write_json

    try:
        data = geometry_preview(get(problem_id))
        write_json(Path(out), data)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        write_json(Path(out).with_suffix(".error.json"), {"error": f"{type(e).__name__}: {e}",
                                                          "traceback": traceback.format_exc()})
        return 1


# --------------------------------------------------------------------------- backend import check


def check_backends() -> dict:
    import shutil

    from feacore.solver import BACKENDS

    out = {}
    for name, modname in BACKENDS.items():
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(modname)
            doc = (mod.__doc__ or "").strip().splitlines()
            out[name] = {"ok": hasattr(mod, "solve"), "module": modname, "doc": doc[0] if doc else "",
                         "import_time": round(time.perf_counter() - t0, 2)}
            if not hasattr(mod, "solve"):
                out[name]["error"] = "module has no solve()"
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "module": modname, "error": f"{type(e).__name__}: {e}"}
    out.setdefault("ccx", {})["executable"] = shutil.which("ccx")
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(argv[0])
    load_extra_problems()
    kind = job["kind"]
    if kind == "solve":
        return run_solve(job["root"], job["run_id"], fields=job.get("fields", True))
    if kind == "study":
        return run_study(job["root"], job["study_id"])
    if kind == "geometry":
        return run_geometry(job["problem_id"], job["out"])
    if kind == "backends":
        print("FEACORE_JSON " + json.dumps(check_backends()))
        return 0
    raise SystemExit(f"unknown job kind {kind!r}")


if __name__ == "__main__":
    sys.exit(main())
