"""HTTP API of the dashboard's Gold page (gold benchmarks and their study meshes) and, through
feacore.ui.cad_api, of the Your CAD page. ``feacore.ui.server`` mounts ``router`` once.

    GET  /api/gold                             parts -> BC sets (references, singular QoIs, DOFs and
                                               seconds per level, the generator's QoIs per level) and
                                               the study meshes with their local status
    GET  /api/gold/meshes/{part}/{level}       a stored study mesh (gold root _meshes/, e.g. L0)
    GET  /api/gold/regenerated/{part}/{level}  a mesh regenerated on this machine (<runs>/gold_regen/)
    POST /api/gold/solve                       {"set_id", "level", "source": "stored" | "regenerated"}:
                                               solve the set on that mesh, for the field viewer
    PUT  /api/gold/archives/{name}             upload a generator archive (raw body) to <runs>/gold_archives/
    GET  /api/gold/archives                    the archives there (for the import tool)

Mesh status: ``missing`` (not on this machine), ``ok`` (sha256 as in index.json), ``mismatch``
(present, other bytes), ``unlisted`` (present, not in the index). The dashboard is the evaluator's
view, so references are shown. Nothing here writes into a gold set or _meshes/; the tools do that
on request (feacore/ui/tools/gold.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tarfile
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse

PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
LEVEL_RE = re.compile(r"^L\d{1,2}$")
ARCHIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(tgz|tar\.gz|tar)$")
DOFS_PER_CELL = 4.6          # tet10 (feacore.goldgen.__main__)


def view_max_dofs() -> int:
    """Largest solve the Gold page starts on this machine (L1 ~150k DOFs; L2 ~1M needs a GPU box)."""
    return int(os.environ.get("FEACORE_GOLD_VIEW_MAX_DOFS") or 400_000)


class _Sha:
    """sha256 of files, cached by (path, size, mtime)."""

    def __init__(self):
        self._d: dict[tuple, str] = {}
        self._lock = threading.Lock()

    def __call__(self, path: Path) -> str:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            if key in self._d:
                return self._d[key]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        with self._lock:
            self._d[key] = h.hexdigest()
        return self._d[key]


def bc_short(b: dict) -> str:
    """'clamp f21', 'roller x f19', 'force 7.8 kN f18', 'pressure 6 MPa f3' (the README's words)."""
    g = b.get("group") or "a point"
    t = b.get("type")
    if t == "fixed":
        return f"clamp {g}"
    if t == "displacement":
        return f"roller {''.join('xyz'[k] for k in b.get('components') or [])} {g}"
    if t == "pressure":
        return f"pressure {float(b['value']) / 1e6:.3g} MPa {g}"
    if t in ("force", "traction"):
        mag = sum(float(v) ** 2 for v in b["value"]) ** 0.5
        return f"{t} {mag / 1e3:.3g} kN {g}" if t == "force" else f"traction {mag / 1e6:.3g} MPa {g}"
    return f"{t} {g}"


def _level_key(name: str) -> int:
    return int(name[1:]) if LEVEL_RE.match(name) else 99


def gold_runs(store) -> dict[str, list[dict]]:
    """Solves recorded for gold sets (scripts/gold.py render, POST /api/gold/solve), by problem."""
    out: dict[str, list[dict]] = {}
    for r in store.list_runs():
        if r.get("source") != "gold" or not r.get("problem_id"):
            continue
        meta, info = r.get("meta") or {}, r.get("info") or {}
        out.setdefault(r["problem_id"], []).append({
            "run_id": r["run_id"], "level": r.get("level"), "agent_id": r.get("agent_id"), "status": r.get("status"),
            "created_at": r.get("created_at"), "error": r.get("error"), "mesh_source": meta.get("mesh_source", "stored"),
            "n_dofs": info.get("n_dofs"), "wall_time": info.get("wall_time"), "fields": bool(r.get("fields")),
            "qois": r.get("qois") or {}})
    return out


def router(store, runner) -> APIRouter:
    from feacore.problems import gold
    from feacore.ui import cad_api
    from feacore.ui.jobs import finalize_run

    r = APIRouter()
    r.include_router(cad_api.router(store, runner))
    sha = _Sha()
    regen_root = store.root / "gold_regen"
    archives = store.root / "gold_archives"

    def index_of(root: Path) -> dict:
        p = root / "index.json"
        try:
            index = json.loads(p.read_text()) if p.is_file() else {}
        except (OSError, json.JSONDecodeError):
            index = {}
        index.setdefault("sets", {})
        index.setdefault("parts", {})
        return index

    def regen_of(part: str) -> dict:
        p = regen_root / part / "regen.json"
        try:
            return json.loads(p.read_text()) if p.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def sync_registry(problems) -> None:
        """Sets imported while the server runs become registered problems (Benchmarks, studies)."""
        from feacore.problems import registry

        registry.discover()
        for p in problems:
            if p.id not in registry._REGISTRY:   # noqa: SLF001
                registry.register(p)

    def mesh_view(root: Path, part: str, level: str, entry: dict | None, regen: dict) -> dict:
        path = root / "_meshes" / part / f"{level}.msh"
        out = {"level": level, "k": _level_key(level)}
        out.update({k: (entry or {}).get(k) for k in ("h", "cells", "scheme", "min_sj", "bytes", "sha256", "archive")})
        if not path.is_file():
            out["status"] = "missing"
        else:
            out.update(local_bytes=path.stat().st_size, local_sha256=sha(path))
            out["status"] = ("unlisted" if not (entry or {}).get("sha256") else
                             "ok" if out["local_sha256"] == entry["sha256"] else "mismatch")
        rg = (regen.get("levels") or {}).get(level)
        if rg:
            rg = dict(rg)
            rg["available"] = (regen_root / part / f"{level}.msh").is_file()
            out["regenerated"] = rg
        return out

    def set_view(p, row: dict, runs: list[dict]) -> dict:
        ref = p._private()   # noqa: SLF001 - evaluator view
        study = p.study()
        return {"id": p.id, "title": p.title, "bcs": [bc_short(b) for b in ref["bcset"]["bcs"]],
                "words": ref["bcset"].get("words", []), "references": [asdict(x) for x in p.references()],
                "singular": p.singular_qois(), "scored_qoi": p.scored_qoi(),
                "dofs": row.get("dofs") or [x.get("n_dofs") for x in study["runs"]],
                "seconds": row.get("seconds"), "archive": row.get("archive"),
                "levels": [{"level": x.get("level"), "h": x.get("h"), "n_dofs": x.get("n_dofs"), "qois": x.get("qois")}
                           for x in study["runs"]],
                "runs": sorted(runs, key=lambda x: x.get("created_at") or "", reverse=True)}

    @r.get("/api/gold")
    def overview():
        root = gold.gold_root()
        index = index_of(root)
        problems, errors = gold.load_all(root)
        sync_registry(problems)
        runs = gold_runs(store)
        by_part: dict[str, list] = {}
        for p in problems:
            by_part.setdefault(p.family, []).append(p)
        parts = []
        for part in sorted(set(by_part) | set(index["parts"])):
            ip = index["parts"].get(part, {})
            stored = ip.get("meshes") or {}
            on_disk = {m.stem for m in (root / "_meshes" / part).glob("L*.msh")} if (root / "_meshes" / part).is_dir() else set()
            regen = regen_of(part)
            levels = sorted(set(stored) | {x for x in on_disk if LEVEL_RE.match(x)} | set(regen.get("levels") or {}),
                            key=_level_key)
            sets = [set_view(p, index["sets"].get(p.id, {}), runs.get(p.id, [])) for p in by_part.get(part, [])]
            first = by_part.get(part, [None])[0]
            parts.append({
                "part": part, "source": ip.get("source") or (first.part.get("source") if first else None),
                "faces": ip.get("faces") or (len(first.part["faces"]) if first else None),
                "meshes": [mesh_view(root, part, lv, stored.get(lv), regen) for lv in levels],
                "regenerated": {k: regen.get(k) for k in ("input", "h0_frac", "updated_at")} if regen else None,
                "sets": sets})
        n_mesh = sum(len(p["meshes"]) for p in parts)
        return {"root": str(root.resolve()), "index_found": (root / "index.json").is_file(),
                "runs_dir": str(store.root.resolve()), "n_sets": len(problems), "n_parts": len(parts),
                "n_meshes": n_mesh, "n_present": sum(m["status"] in ("ok", "mismatch", "unlisted")
                                                     for p in parts for m in p["meshes"]),
                "n_ok": sum(m["status"] == "ok" for p in parts for m in p["meshes"]),
                "view_max_dofs": view_max_dofs(), "errors": errors, "parts": parts}

    def safe_mesh(base: Path, part: str, level: str) -> Path:
        if not PART_RE.match(part) or ".." in part or not LEVEL_RE.match(level):
            raise HTTPException(404, "no such mesh")
        p = (base / part / f"{level}.msh").resolve()
        if base.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404, f"{part} {level}: not on this machine")
        return p

    @r.get("/api/gold/meshes/{part}/{level}")
    def stored_mesh(part: str, level: str):
        p = safe_mesh(gold.gold_root() / "_meshes", part, level)
        return FileResponse(p, filename=f"{part}_{level}.msh", media_type="application/octet-stream")

    @r.get("/api/gold/regenerated/{part}/{level}")
    def regenerated_mesh(part: str, level: str):
        p = safe_mesh(regen_root, part, level)
        return FileResponse(p, filename=f"{part}_{level}_regenerated.msh", media_type="application/octet-stream")

    @r.post("/api/gold/solve", status_code=202)
    def solve_level(body: dict = Body(..., examples=[{"set_id": "abc_00000003_bc0", "level": 0}])):
        root = gold.gold_root()
        sid, source = str(body.get("set_id") or ""), str(body.get("source") or "stored")
        try:
            k = int(body.get("level", 0))
        except (TypeError, ValueError):
            raise HTTPException(422, "level must be an integer") from None
        if source not in ("stored", "regenerated") or not 0 <= k <= 9:
            raise HTTPException(422, "source must be stored or regenerated, level 0-9")
        p = next((x for x in gold.load_all(root)[0] if x.id == sid), None)
        if p is None:
            raise HTTPException(404, f"no gold set {sid!r} under {root}")
        lv = f"L{k}"
        if source == "stored":
            mesh = p.mesh_file(k)
            if mesh is None:
                raise HTTPException(404, f"{p.family} {lv} is not on this machine: import its archive with the "
                                         f"meshes, or regenerate the level and solve on that")
        else:
            mesh = regen_root / p.family / f"{lv}.msh"
            if not mesh.is_file():
                raise HTTPException(404, f"{p.family} {lv} has not been regenerated here")
        entry = index_of(root)["parts"].get(p.family, {}).get("meshes", {}).get(lv, {})
        cells = ((regen_of(p.family).get("levels") or {}).get(lv, {}).get("cells") if source == "regenerated"
                 else entry.get("cells"))
        listed = index_of(root)["sets"].get(sid, {}).get("dofs") or []
        dofs = listed[k] if k < len(listed) and source == "stored" else None
        est = dofs or (cells or 0) * DOFS_PER_CELL
        if est > view_max_dofs():
            raise HTTPException(422, f"{lv} has about {est / 1e6:.2g}M DOFs, more than this machine solves for the "
                                     f"viewer ({view_max_dofs():,}, FEACORE_GOLD_VIEW_MAX_DOFS); render it on a GPU box")
        mesh = Path(mesh).resolve()
        h = next((x.get("h") for x in p.study()["runs"] if x.get("level") == k), entry.get("h"))
        rec = store.create_run(backend="skfem", case=p.case(mesh).to_dict(), problem_id=p.id,
                               agent_id=f"gold{'-regen' if source == 'regenerated' else ''}:{lv}", level=k,
                               h_nominal=h, source="gold",
                               meta={"source": "gold", "part": p.family, "mesh_level": k, "mesh_source": source,
                                     "mesh_sha256": sha(mesh)})
        rid = rec["run_id"]
        store.update_run(rid, mesh_file=store.relpath(mesh))
        runner.submit("solve", {"root": str(store.root), "run_id": rid, "fields": True},
                      log=store.run_dir(rid) / "worker.log", on_exit=finalize_run(store))
        return {**store.get_run(rid), "poll": f"/api/runs/{rid}", "viewer": f"#/viewer/{rid}"}

    @r.put("/api/gold/archives/{name}", status_code=201)
    async def put_archive(name: str, request: Request):
        if not ARCHIVE_RE.match(name):
            raise HTTPException(422, "archive names are like goldgen-l2_0001.tgz (.tgz, .tar.gz or .tar)")
        limit = int(os.environ.get("FEACORE_GOLD_ARCHIVE_MAX_BYTES") or 8e9)
        archives.mkdir(parents=True, exist_ok=True)
        tmp = archives / f".{name}.{secrets.token_hex(3)}.part"
        n = 0
        try:
            with open(tmp, "wb") as f:
                async for chunk in request.stream():
                    n += len(chunk)
                    if n > limit:
                        raise HTTPException(413, f"archive larger than {limit / 1e9:.1f} GB")
                    f.write(chunk)
            if n == 0:
                raise HTTPException(422, "empty upload")
            if not tarfile.is_tarfile(tmp):
                raise HTTPException(422, "not a tar archive")
            tmp.replace(archives / name)
        finally:
            tmp.unlink(missing_ok=True)
        return {"name": name, "bytes": n, "path": f"@runs/gold_archives/{name}"}

    @r.get("/api/gold/archives")
    def list_archives():
        files = sorted(archives.glob("*")) if archives.is_dir() else []
        return {"archives": [{"name": f.name, "bytes": f.stat().st_size, "path": f"@runs/gold_archives/{f.name}"}
                             for f in files if ARCHIVE_RE.match(f.name) and f.is_file()]}

    return r
