"""HTTP API + static frontend for feacore (run: ``python -m feacore.ui --host 0.0.0.0 --port 8000``).

All heavy work (gmsh, solves) runs in child processes via ``feacore.ui.jobs``; request
handlers only read and write the run store. See docs/ui.md for the API reference.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

import feacore
from feacore.runstore import RunStore, check_id, jsonable, read_json
from feacore.ui import analysis
from feacore.ui.jobs import JobRunner, finalize_run, finalize_study
from feacore.ui.viewdata import encode_array

STATIC_DIR = Path(__file__).parent / "static"
THREE_VERSION = "0.170.0"
CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
VENDOR_FILES = {  # local name -> pinned CDN url
    "three.module.min.js": f"{CDN}/build/three.module.min.js",
    "OrbitControls.js": f"{CDN}/examples/jsm/controls/OrbitControls.js",
}


NPM_TARBALL = f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz"
NPM_PATHS = {"three.module.min.js": "package/build/three.module.min.js",
             "OrbitControls.js": "package/examples/jsm/controls/OrbitControls.js"}


def vendor_dir() -> Path:
    """Where ``--fetch-vendor`` writes (and the server looks first)."""
    return Path(os.environ.get("FEACORE_UI_VENDOR_DIR", STATIC_DIR / "vendor" / f"three-{THREE_VERSION}"))


def fetch_vendor(dest: Path | None = None) -> Path:
    """Download the pinned three.js files: jsDelivr, else the npm registry tarball."""
    import io
    import tarfile
    import urllib.request

    dest = Path(dest or vendor_dir())
    dest.mkdir(parents=True, exist_ok=True)

    def put(name: str, data: bytes) -> None:
        tmp = dest / f".{name}.tmp"
        tmp.write_bytes(data)
        tmp.replace(dest / name)

    missing = [n for n in VENDOR_FILES if not (dest / n).exists()]
    try:
        for name in missing:
            with urllib.request.urlopen(VENDOR_FILES[name], timeout=30) as r:  # noqa: S310 - pinned https URL
                put(name, r.read())
    except Exception:  # noqa: BLE001 - CDN blocked: same files from the npm registry
        with urllib.request.urlopen(NPM_TARBALL, timeout=180) as r:  # noqa: S310
            tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
        for name in VENDOR_FILES:
            if not (dest / name).exists():
                put(name, tar.extractfile(NPM_PATHS[name]).read())
    return dest


class SafeJSONResponse(JSONResponse):
    """Strict JSON (NaN/inf -> null, numpy -> python)."""

    def render(self, content) -> bytes:
        return json.dumps(jsonable(content), allow_nan=False, separators=(",", ":")).encode("utf-8")


class NoCacheStatic(StaticFiles):
    """No build step and no hashed file names: make browsers revalidate (ETag -> 304)."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# --------------------------------------------------------------------------- request models


class StudyRequest(BaseModel):
    problem_id: str
    backend: str = "skfem"
    order: int = 2
    sizes: list[float] | None = Field(None, description="mesh sizes h (any order); default: problem.study_sizes(order)")
    levels: int | None = Field(None, ge=2, le=8, description="number of levels from the coarsest default size")
    ratio: float | None = Field(None, gt=1.0, le=4.0, description="refinement ratio between levels (with levels)")
    geom_order: int | None = None
    meta: dict = Field(default_factory=dict)


class SubmissionRequest(BaseModel):
    problem_id: str
    agent_id: str
    run_ids: list[str] = Field(default_factory=list)
    levels: list[dict] = Field(default_factory=list, description="[{h, qois: {name: value}, n_dofs?, order?}]")
    order: int | None = None
    notes: str = ""
    meta: dict = Field(default_factory=dict)
    submission_id: str | None = None


# --------------------------------------------------------------------------- helpers


def _problems_api():
    from feacore.problems import registry

    return registry


def _problem_or_404(problem_id: str):
    reg = _problems_api()
    try:
        return reg.get(problem_id)
    except KeyError as e:
        raise HTTPException(404, str(e).strip('"')) from None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return default if default is not None else {"error": f"{type(e).__name__}: {e}"}


def _versions() -> dict:
    out = {}
    for pkg in ("numpy", "scipy", "scikit-fem", "meshio", "gmsh", "fastapi", "starlette", "uvicorn",
                "nvmath-python", "cupy", "torch"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _cpu_info() -> dict:
    info = {"count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        info["load"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = round(int(v.split()[0]) / 1024**2, 1)
        info["memory_gb"] = {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable")}
    except OSError:
        pass
    info["threads_env"] = {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                           if k in os.environ}
    return info


def _gpu_info() -> dict:
    exe = shutil.which("nvidia-smi")
    out = {"nvidia_smi": exe, "gpus": [],
           "libs": {m: importlib.util.find_spec(m) is not None for m in ("nvmath", "cupy", "torch")}}
    if exe:
        try:
            r = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    out["gpus"].append({"name": parts[0], "memory": parts[1], "driver": parts[2],
                                        "utilization": parts[3] if len(parts) > 3 else None})
        except (OSError, subprocess.SubprocessError) as e:
            out["error"] = str(e)
    return out


class _BackendCheck:
    """Import check of the solver backends, in a subprocess (heavy imports stay out of the
    server), refreshed in the background at most every ``ttl`` seconds."""

    def __init__(self, ttl: float = 60.0):
        self.ttl = ttl
        self.result: dict | None = None
        self.checked_at: float = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _run(self):
        try:
            r = subprocess.run([sys.executable, "-m", "feacore.ui.worker", json.dumps({"kind": "backends"})],
                               capture_output=True, text=True, timeout=180)
            line = next((l for l in r.stdout.splitlines() if l.startswith("FEACORE_JSON ")), None)
            res = json.loads(line[len("FEACORE_JSON "):]) if line else {"error": (r.stderr or r.stdout)[-2000:]}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        self.result, self.checked_at = res, time.time()

    def get(self, wait: bool = False, refresh: bool = False) -> dict:
        with self._lock:
            stale = refresh or (time.time() - self.checked_at > self.ttl)
            if stale and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            t = self._thread
        if wait and t is not None:
            t.join(200)
        return {"backends": self.result, "checked_at": self.checked_at or None,
                "checking": bool(t is not None and t.is_alive())}


def _run_summary(r: dict) -> dict:
    info = r.get("info") or {}
    f = r.get("fields") or {}
    return {
        "run_id": r.get("run_id"), "created_at": r.get("created_at"), "status": r.get("status"),
        "source": r.get("source"), "problem_id": r.get("problem_id"), "agent_id": r.get("agent_id"),
        "submission_id": r.get("submission_id"), "study_id": r.get("study_id"), "level": r.get("level"),
        "h_nominal": r.get("h_nominal"), "backend": r.get("backend"), "qois": r.get("qois") or {},
        "error": r.get("error"),
        "info": {k: info.get(k) for k in ("physics", "formulation", "order", "dim", "cell_type", "n_nodes",
                                          "n_cells", "n_dofs", "h_rep", "wall_time")} if info else None,
        "fields": f.get("available") if f else None,
    }


def _study_summary(s: dict) -> dict:
    res = s.get("result") or {}
    interp = analysis.interpret_study(res) if res else (s.get("interpretation") or {})
    prog = s.get("progress") or {}
    return {
        "study_id": s.get("study_id"), "created_at": s.get("created_at"), "finished_at": s.get("finished_at"),
        "status": s.get("status"), "problem_id": s.get("problem_id"), "backend": s.get("backend"),
        "order": s.get("order"), "sizes": s.get("sizes"), "source": s.get("source"),
        "progress": {k: prog.get(k) for k in ("done", "total", "stage", "message")},
        "passed": res.get("passed"), "error": s.get("error"),
        "verdicts": {k: {"label": v.get("label"), "status": v.get("status"), "verdict": v.get("verdict")}
                     for k, v in interp.items()},
    }


def _validation_studies(problem_id: str) -> list[dict]:
    """Reference studies exported by scripts/validate.py (benchmarks/<id>/private/validation_*.json)."""
    from feacore.problems.store import DEFAULT_ROOT

    out = []
    d = Path(os.environ.get("FEACORE_BENCHMARKS_DIR", DEFAULT_ROOT)) / problem_id / "private"
    for p in sorted(d.glob("validation_*.json")) if d.exists() else []:
        try:
            st = read_json(p)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(st, dict) and "qois" in st:
            out.append({"study_id": p.stem, "source": "validation", "file": p.name, "result": st,
                        "interpretation": analysis.interpret_study(st)})
    return out


class _NpzCache:
    def __init__(self, size: int = 6):
        self.size = size
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> dict:
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data["_meta"] = json.loads(str(data.pop("meta")))
        with self._lock:
            self._d[key] = data
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return data


# --------------------------------------------------------------------------- app


def create_app(runs_dir: str | Path | None = None, quick_workers: int | None = None,
               study_workers: int | None = None) -> FastAPI:
    store = RunStore(runs_dir)
    runner = JobRunner(store, quick_workers=quick_workers, study_workers=study_workers)
    backend_check = _BackendCheck()
    npz_cache = _NpzCache()
    from feacore.ui.worker import load_extra_problems

    extra_errors = load_extra_problems()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.ensure()
        store.mark_interrupted()
        backend_check.get()
        yield
        runner.stop()

    app = FastAPI(title="feacore", version=feacore.__version__, lifespan=lifespan,
                  default_response_class=SafeJSONResponse,
                  description="Fixed FEA solver service, solution verification (Richardson / GCI), "
                              "benchmark library and multi-agent consensus. See docs/ui.md.")
    app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=3)
    app.state.store = store
    app.state.runner = runner

    # ------------------------------------------------------------------ frontend
    @app.get("/", include_in_schema=False)
    def index():
        html = (STATIC_DIR / "index.html").read_text()
        # same-origin URLs: the browser never needs to reach the CDN itself (see /vendor below)
        imports = {"three": "/vendor/three.module.min.js",
                   "three/addons/controls/OrbitControls.js": "/vendor/OrbitControls.js"}
        html = html.replace("__IMPORTMAP__", json.dumps({"imports": imports}))
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    vendor_state = {"failed_at": 0.0, "lock": threading.Lock()}

    @app.get("/vendor/{name}", include_in_schema=False)
    def vendor(name: str):
        """Pinned three.js: a vendored copy (``--fetch-vendor``), else a copy the server fetched
        once into the run store cache, else a redirect to the pinned CDN URL."""
        if name not in VENDOR_FILES:
            raise HTTPException(404, "unknown vendor file")
        cache = store.cache_dir / "vendor" / f"three-{THREE_VERSION}"
        for d in (vendor_dir(), cache):
            if (d / name).exists():
                return FileResponse(d / name, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})
        with vendor_state["lock"]:
            if not (cache / name).exists() and time.time() - vendor_state["failed_at"] > 300:
                try:
                    fetch_vendor(cache)
                except Exception:  # noqa: BLE001 - offline: let the browser try the CDN
                    vendor_state["failed_at"] = time.time()
        if (cache / name).exists():
            return FileResponse(cache / name, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})
        return RedirectResponse(VENDOR_FILES[name])

    app.mount("/static", NoCacheStatic(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health(wait: bool = False, refresh: bool = False):
        reg = _problems_api()
        errors = reg.discover()
        bc = backend_check.get(wait=wait, refresh=refresh)
        return {
            "ok": True,
            "feacore": feacore.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "versions": _versions(),
            "backends": bc["backends"],
            "backends_checked_at": bc["checked_at"],
            "backends_checking": bc["checking"],
            "cpu": _cpu_info(),
            "gpu": _gpu_info(),
            "ccx": shutil.which("ccx"),
            "store": {"root": str(store.root.resolve())},
            "jobs": runner.status(),
            "problems": {"count": len(reg.all_problems()), "import_errors": errors, "extra_errors": extra_errors},
        }

    # ------------------------------------------------------------------ problems
    @app.get("/api/problems")
    def problems(tag: list[str] = Query(default=[]), family: str | None = None):
        reg = _problems_api()
        errors = reg.discover()
        everything = reg.all_problems()
        studies = store.list_studies()
        subs = store.list_submissions()
        tag_counts: dict[str, int] = {}
        for p in everything:
            for t in p.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        items = []
        for p in reg.all_problems(tags=tag or None, family=family):
            items.append({
                "id": p.id, "title": p.title, "family": p.family, "tags": list(p.tags), "dim": p.dim,
                "physics": p.physics, "formulation": p.formulation, "expected_behavior": p.expected_behavior,
                "n_qois": _safe(lambda: len(p.qois()), 0), "n_references": _safe(lambda: len(p.references()), 0),
                "n_studies": sum(s.get("problem_id") == p.id for s in studies),
                "n_submissions": sum(s.get("problem_id") == p.id for s in subs),
            })
        return {"problems": items, "tags": dict(sorted(tag_counts.items())), "total": len(everything),
                "import_errors": errors, "extra_errors": extra_errors}

    @app.get("/api/problems/{problem_id}")
    def problem_detail(problem_id: str, order: int = 2):
        p = _problem_or_404(problem_id)
        statement = _safe(p.statement)
        refs = analysis.reference_dicts(p)
        return {
            "id": p.id,
            "statement": statement,
            "evaluator": {
                "expected_behavior": p.expected_behavior,
                "references": refs,
                "expectations": _safe(p.expectations, {}),
                "study_sizes": _safe(lambda: p.study_sizes(order)),
                "reference_case": _safe(lambda: p.case("MESH.msh", order=order).to_dict()),
                "validation_studies": [{"study_id": v["study_id"], "file": v["file"],
                                        "interpretation": v["interpretation"],
                                        "backend": v["result"].get("backend"), "order": v["result"].get("order"),
                                        "passed": v["result"].get("passed")}
                                       for v in _validation_studies(problem_id)],
            },
            "studies": [_study_summary(s) for s in store.list_studies(problem_id=problem_id)],
            "n_submissions": len(store.list_submissions(problem_id=problem_id)),
        }

    @app.get("/api/problems/{problem_id}/geometry")
    async def problem_geometry(problem_id: str, refresh: bool = False):
        p = _problem_or_404(problem_id)
        key = hashlib.sha1(json.dumps(_safe(p.statement), sort_keys=True, default=str).encode()).hexdigest()[:12]
        gdir = store.cache_dir / "geometry"
        gdir.mkdir(parents=True, exist_ok=True)
        out = gdir / f"{problem_id}-{key}.json"
        err = out.with_suffix(".error.json")
        if refresh:
            out.unlink(missing_ok=True)
            err.unlink(missing_ok=True)
        if not out.exists() and not err.exists():
            job = runner.submit("geometry", {"problem_id": problem_id, "out": str(out)},
                                log=out.with_suffix(".log"), dedupe=str(out))
            try:
                await asyncio.wait_for(asyncio.wrap_future(job.future), timeout=320)
            except asyncio.TimeoutError:
                raise HTTPException(504, "geometry preview is still being generated; retry shortly") from None
        if out.exists():
            return FileResponse(out, media_type="application/json")
        detail = read_json(err) if err.exists() else {"error": "geometry job failed (see log)"}
        return SafeJSONResponse({"detail": detail.get("error"), "traceback": detail.get("traceback")}, status_code=500)

    # ------------------------------------------------------------------ solve service
    @app.post("/api/solve")
    async def solve_endpoint(request: Request):
        """multipart/form-data: ``case`` (JSON file or text), ``mesh`` (.msh file); optional
        ``backend`` (skfem), ``agent_id``, ``problem_id``, ``submission_id``, ``wait`` (true),
        ``fields`` (true: store viewer data), ``meta`` (JSON object)."""
        from feacore.case import Case, CaseError
        from feacore.solver import BACKENDS

        form = await request.form(max_part_size=16 * 1024 * 1024)
        case_part = form.get("case")
        mesh_part = form.get("mesh")
        if case_part is None:
            raise HTTPException(422, "missing form field 'case' (case JSON as a file or a text field)")
        if not isinstance(mesh_part, UploadFile):
            raise HTTPException(422, "missing file field 'mesh' (gmsh .msh with named physical groups)")
        raw = (await case_part.read()).decode("utf-8") if isinstance(case_part, UploadFile) else str(case_part)
        try:
            case_dict = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"case is not valid JSON: {e}") from None
        if not isinstance(case_dict, dict):
            raise HTTPException(422, "case JSON must be an object")
        backend = str(form.get("backend") or "skfem")
        if backend not in BACKENDS:
            raise HTTPException(422, f"unknown backend {backend!r}; available: {sorted(BACKENDS)}")
        original_mesh = case_dict.get("mesh")
        case_dict["mesh"] = "mesh.msh"
        try:
            errs = Case.from_dict(case_dict).validate()
        except CaseError as e:
            errs = [str(e)]
        if errs:
            return SafeJSONResponse({"detail": "invalid case", "errors": errs}, status_code=422)
        meta = {}
        if form.get("meta"):
            try:
                meta = json.loads(str(form.get("meta")))
            except json.JSONDecodeError:
                raise HTTPException(422, "meta must be a JSON object") from None
        meta.update({"original_mesh": original_mesh, "mesh_filename": mesh_part.filename})

        def opt(name):
            v = form.get(name)
            return str(v) if v not in (None, "") else None

        def flag(name, default=True):
            v = form.get(name)
            return default if v in (None, "") else str(v).lower() in ("1", "true", "yes", "on")

        for name in ("agent_id", "problem_id", "submission_id"):
            if opt(name) is not None and len(opt(name)) > 200:
                raise HTTPException(422, f"{name} too long")
        rec = store.create_run(backend=backend, case=case_dict, problem_id=opt("problem_id"),
                               agent_id=opt("agent_id"), submission_id=opt("submission_id"),
                               source="api", meta=meta)
        rid = rec["run_id"]
        dst = store.run_dir(rid) / "mesh.msh"

        def _copy():
            mesh_part.file.seek(0)
            with open(dst, "wb") as f:
                shutil.copyfileobj(mesh_part.file, f, 1024 * 1024)

        await run_in_threadpool(_copy)
        store.update_run(rid, mesh_file=store.relpath(dst))
        job = runner.submit("solve", {"root": str(store.root), "run_id": rid, "fields": flag("fields")},
                            log=store.run_dir(rid) / "worker.log", on_exit=finalize_run(store))
        if not flag("wait"):
            rec = store.get_run(rid)
            rec["poll"] = f"/api/runs/{rid}"
            return SafeJSONResponse(rec, status_code=202)
        timeout = float(form.get("timeout") or runner.timeouts["solve"] + 30)
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(job.future)), timeout=timeout)
        except asyncio.TimeoutError:
            rec = store.get_run(rid)
            rec["poll"] = f"/api/runs/{rid}"
            return SafeJSONResponse(rec, status_code=202)
        rec = store.get_run(rid)
        return SafeJSONResponse(rec, status_code=200 if rec["status"] == "succeeded" else 422)

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs(problem_id: str | None = None, agent_id: str | None = None, study_id: str | None = None,
             submission_id: str | None = None, status: str | None = None, limit: int = 200):
        rs = store.list_runs(problem_id=problem_id, agent_id=agent_id, study_id=study_id,
                             submission_id=submission_id, status=status, limit=limit)
        return {"runs": [_run_summary(r) for r in rs]}

    @app.post("/api/runs", status_code=201)
    def import_run(record: dict):
        """Register a run solved elsewhere (no fields): {qois, info?, case?, problem_id?, agent_id?, backend?}."""
        try:
            rid = store.save_run(record)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(422, str(e)) from None
        return store.get_run(rid)

    def _run_or_404(run_id: str) -> dict:
        try:
            return store.get_run(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        rec = _run_or_404(run_id)
        if rec.get("status") == "queued":
            for j in list(runner.jobs.values()):
                if j.payload.get("run_id") == run_id:
                    rec["queue_position"] = runner.queue_position(j)
        return rec

    @app.get("/api/runs/{run_id}/field")
    def run_field(run_id: str, name: str | None = None, geometry: bool = True,
                  encoding: str = Query("b64", pattern="^(b64|list)$")):
        rec = _run_or_404(run_id)
        p = store.fields_path(run_id)
        if not p.exists():
            raise HTTPException(404, f"run {run_id} has no field data (status {rec.get('status')})")
        data = npz_cache.get(p)
        meta = data["_meta"]
        fields = meta["fields"]
        if not fields:
            raise HTTPException(404, "no fields stored for this run")
        if name is None:
            name = next((f for f in ("von_mises", "T", "u_mag") if f in fields), fields[0])
        if name not in fields:
            return SafeJSONResponse({"detail": f"unknown field {name!r}", "available": fields}, status_code=404)
        vals = data[f"f_{name}"]
        fin = vals[np.isfinite(vals)]
        out = {
            "run_id": run_id, "field": name, "fields": fields, "dim": meta["dim"],
            "physics": meta.get("physics"), "formulation": meta.get("formulation"),
            "cell_type": meta.get("cell_type"), "n_points": meta["n_points"], "n_triangles": meta["n_triangles"],
            "stats": meta["stats"].get(name), "range": [float(fin.min()), float(fin.max())] if fin.size else None,
            "values": encode_array(vals, "float32", encoding),
        }
        if geometry:
            out["points"] = encode_array(data["points"], "float32", encoding)
            out["triangles"] = encode_array(data["triangles"], "uint32", encoding)
            out["edges"] = encode_array(data["edges"], "uint32", encoding)
            out["displacement"] = encode_array(data["disp"], "float32", encoding) if "disp" in data else None
        return SafeJSONResponse(out)

    @app.get("/api/runs/{run_id}/mesh", include_in_schema=False)
    def run_mesh(run_id: str):
        rec = _run_or_404(run_id)
        if not rec.get("mesh_file"):
            raise HTTPException(404, "no mesh stored")
        p = store.abspath(rec["mesh_file"])
        if not p.exists():
            raise HTTPException(404, "mesh file missing")
        return FileResponse(p, filename=f"{run_id}.msh", media_type="application/octet-stream")

    @app.get("/api/runs/{run_id}/log", include_in_schema=False)
    def run_log(run_id: str):
        _run_or_404(run_id)
        p = store.run_dir(run_id) / "worker.log"
        return PlainTextResponse(p.read_text(errors="replace")[-20000:] if p.exists() else "")

    # ------------------------------------------------------------------ studies
    @app.post("/api/studies", status_code=202)
    def create_study(req: StudyRequest):
        from feacore.solver import BACKENDS

        p = _problem_or_404(req.problem_id)
        if req.backend not in BACKENDS:
            raise HTTPException(422, f"unknown backend {req.backend!r}; available: {sorted(BACKENDS)}")
        if req.order not in (1, 2):
            raise HTTPException(422, "order must be 1 or 2")
        sizes = req.sizes
        if sizes is None and req.levels:
            try:
                h0 = max(p.study_sizes(req.order))
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"problem has no default sizes ({e}); pass sizes") from None
            r = req.ratio or (2.0 if p.dim == 2 else 1.5)
            sizes = [h0 / r**k for k in range(req.levels)]
        if sizes is not None:
            if not sizes or any((not np.isfinite(h)) or h <= 0 for h in sizes):
                raise HTTPException(422, "sizes must be positive numbers")
            if len(set(sizes)) != len(sizes):
                raise HTTPException(422, "sizes must be distinct")
            sizes = sorted(sizes, reverse=True)
        else:
            try:
                sizes = sorted(p.study_sizes(req.order), reverse=True)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"problem has no default study sizes ({e}); pass sizes") from None
        rec = store.create_study(req.problem_id, backend=req.backend, order=req.order, sizes=sizes,
                                 geom_order=req.geom_order, meta=req.meta, source="ui")
        sid = rec["study_id"]
        runner.submit("study", {"root": str(store.root), "study_id": sid},
                      log=store.study_dir(sid) / "worker.log", on_exit=finalize_study(store))
        return rec

    @app.get("/api/studies")
    def studies(problem_id: str | None = None, status: str | None = None, limit: int = 100):
        return {"studies": [_study_summary(s) for s in store.list_studies(problem_id=problem_id, status=status,
                                                                          limit=limit)]}

    @app.get("/api/studies/{study_id}")
    def study_detail(study_id: str):
        try:
            st = store.get_study(study_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None
        runs_ = []
        for rid in st.get("run_ids") or []:
            try:
                runs_.append(_run_summary(store.get_run(rid)))
            except KeyError:
                pass
        st["runs"] = runs_
        if st.get("result"):  # always with the current interpretation rules
            st["interpretation"] = analysis.interpret_study(st["result"])
        try:
            p = _problems_api().get(st["problem_id"])
            st["references"] = analysis.reference_dicts(p)
            st["problem_title"] = p.title
            st["qoi_defs"] = _safe(lambda: {q.name: dataclasses.asdict(q) for q in p.qois()}, {})
        except KeyError:
            st["references"] = []
        if st.get("status") == "queued":
            for j in list(runner.jobs.values()):
                if j.payload.get("study_id") == study_id:
                    st["queue_position"] = runner.queue_position(j)
        return st

    @app.get("/api/studies/{study_id}/log", include_in_schema=False)
    def study_log(study_id: str):
        p = store.study_dir(check_id(study_id)) / "worker.log"
        return PlainTextResponse(p.read_text(errors="replace")[-20000:] if p.exists() else "")

    # ------------------------------------------------------------------ submissions + consensus
    @app.post("/api/submissions", status_code=201)
    def create_submission(req: SubmissionRequest):
        try:
            return store.create_submission(req.problem_id, req.agent_id, run_ids=req.run_ids, levels=req.levels,
                                           notes=req.notes, meta=req.meta, order=req.order,
                                           submission_id=req.submission_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/api/submissions")
    def submissions(problem_id: str | None = None, agent_id: str | None = None):
        return {"submissions": store.list_submissions(problem_id=problem_id, agent_id=agent_id)}

    @app.get("/api/submissions/{submission_id}")
    def submission_detail(submission_id: str):
        try:
            return store.get_submission(submission_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None

    @app.get("/api/consensus/{problem_id}")
    def consensus_view(problem_id: str):
        subs = store.list_submissions(problem_id=problem_id)
        try:
            p = _problems_api().get(problem_id)
        except KeyError:
            p = None
        if p is None and not subs:
            raise HTTPException(404, f"unknown problem {problem_id!r} and no submissions for it")
        refs = analysis.reference_dicts(p) if p is not None else []
        ref_studies = []
        latest = {}
        for s in store.list_studies(problem_id=problem_id, status="succeeded"):
            key = (s.get("backend"), s.get("order"))
            if key not in latest:  # newest first
                latest[key] = s
        ref_studies.extend(latest.values())
        ref_studies.extend(_validation_studies(problem_id))
        names = _safe(lambda: [q.name for q in p.qois()], []) if p is not None else []
        out = analysis.consensus(problem_id, subs, refs, ref_studies, qoi_names=names)
        out["title"] = p.title if p is not None else problem_id
        out["registered"] = p is not None
        return out

    return app
