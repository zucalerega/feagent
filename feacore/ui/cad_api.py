"""HTTP API of the dashboard's "Your CAD" page: uploaded STEP files that become registered feacore
problems (feacore/problems/custom.py). Mounted by feacore.ui.gold_api.router; uploads go to
``<run store>/cad/``, which every child process of the server scans too.

    GET    /api/cad                            uploads with their BC sets, roots, limits, trained swarms
    POST   /api/cad/uploads                    multipart: file (STEP), units (si | as_is | normalize)
    GET    /api/cad/uploads/{upload}           one upload, with its faces
    GET    /api/cad/uploads/{upload}/geometry  3D preview: surface triangulation, faces f<tag>
    GET    /api/cad/uploads/{upload}/step      the part in metres (the geometry.step agents get)
    DELETE /api/cad/uploads/{upload}           the upload and every BC set made from it
    POST   /api/cad/uploads/{upload}/sets      {"mode": "auto", "count": n} or {"mode": "manual", ...}
                                               (feacore.problems.custom.define_set): new problem ids
    DELETE /api/cad/problems/{problem}         one BC set
    POST   /api/cad/problems/{problem}/adopt   {"study_id"}: certify a finished study as the reference
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from feacore.problems import custom

REPO_ROOT = Path(__file__).resolve().parents[2]


def trained_swarms() -> list[dict]:
    """Exported swarms in the repo (ace.export_swarm), for the agent commands on the page."""
    out = []
    for m in sorted((REPO_ROOT / "results" / "trained_swarms").glob("*/manifest.json")):
        try:
            d = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"path": str(m.parent.relative_to(REPO_ROOT)), "agents": d.get("agents", []),
                    "tasks": d.get("tasks", []), "generation": d.get("generation")})
    return out


def set_view(p: custom.CustomProblem, studies: list[dict]) -> dict:
    ref = p._private()   # noqa: SLF001 - the evaluator's view: references are shown
    from feacore.problems import registry

    mine = [s for s in studies if s.get("problem_id") == p.id]
    return {
        "id": p.id, "title": p.title, "kind": ref.get("kind"), "created_at": ref.get("created_at"),
        "words": ref["bcset"].get("words", []), "bcs": ref["bcset"]["bcs"], "material": p.material,
        "node_cap": p.node_cap, "scored_qoi": p.scored_qoi(), "references": [asdict(r) for r in p.references()],
        "singular": p.singular_qois(), "reference_study": ref.get("reference_study"),
        "levels": [{k: r.get(k) for k in ("level", "h", "n_dofs", "solve_s", "run_id")} for r in p.study()["runs"]],
        "registered": registry._REGISTRY.get(p.id) is not None,   # noqa: SLF001
        "studies": [{"study_id": s["study_id"], "status": s.get("status"), "created_at": s.get("created_at"),
                     "backend": s.get("backend"), "order": s.get("order"), "nested": s.get("nested", False),
                     "n_sizes": len(s.get("sizes") or [])} for s in mine],
    }


def upload_view(updir: Path, studies: list[dict], faces: bool = False) -> dict:
    up = custom.read_upload(updir)
    keys = ("id", "name", "bytes", "sha256", "units", "scale", "created_at", "n_faces", "diag_mm", "size_mm",
            "volume_cm3", "warnings", "meshes", "mesh_plan")
    out = {k: up.get(k) for k in keys}
    out["dir"] = str(updir)
    out["sets"] = [set_view(p, studies) for p in custom._sets_in(updir)[0]]   # noqa: SLF001
    if faces:
        out["faces"] = up["part"]["faces"]
        out["bbox"] = up["part"]["bbox"]
    return out


def router(store, runner) -> APIRouter:
    root = store.root / "cad"
    custom.add_root(root)        # this process and every worker / task it starts scan it
    r = APIRouter()

    def updir_or_404(uid: str) -> Path:
        try:
            return custom.find_upload(uid)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None

    @r.get("/api/cad")
    def overview():
        info = custom.refresh()
        studies = store.list_studies()
        return {"root": str(root.resolve()), "runs_dir": str(store.root.resolve()),
                "roots": [str(x) for x in custom.cad_roots()], "max_bytes": custom.max_bytes(),
                "units": custom.UNITS, "defaults": {"node_cap": custom.DEFAULT_NODE_CAP, **custom.STEEL},
                "uploads": [upload_view(d, studies) for d in custom.upload_dirs()],
                "errors": info["errors"], "swarms": trained_swarms()}

    @r.post("/api/cad/uploads", status_code=201)
    async def upload(request: Request, response: Response):
        limit = custom.max_bytes()
        n = request.headers.get("content-length", "")
        if n.isdigit() and int(n) > limit + 1_000_000:
            raise HTTPException(413, f"the file is larger than the {limit / 1e6:.0f} MB limit (FEACORE_CAD_MAX_BYTES)")
        form = await request.form(max_files=1, max_fields=10)
        f = form.get("file")
        if not isinstance(f, UploadFile):
            raise HTTPException(422, "missing file field 'file' (a STEP file)")
        data = await f.read(limit + 1)
        try:
            rec = await run_in_threadpool(custom.create_upload, f.filename, data, str(form.get("units") or "si"), root)
        except custom.CadError as e:
            raise HTTPException(422, str(e)) from None
        custom.refresh()
        if rec.get("existing"):
            response.status_code = 200
        return {**upload_view(custom.find_upload(rec["id"]), store.list_studies(), faces=True),
                "existing": bool(rec.get("existing"))}

    @r.get("/api/cad/uploads/{uid}")
    def upload_detail(uid: str):
        return upload_view(updir_or_404(uid), store.list_studies(), faces=True)

    @r.get("/api/cad/uploads/{uid}/geometry")
    def upload_geometry(uid: str):
        p = updir_or_404(uid) / "preview.json"
        if not p.is_file():
            raise HTTPException(404, "no preview for this upload")
        return FileResponse(p, media_type="application/json")

    @r.get("/api/cad/uploads/{uid}/step")
    def upload_step(uid: str):
        return FileResponse(updir_or_404(uid) / "geometry.step", filename=f"{uid}.step",
                            media_type="application/step")

    @r.delete("/api/cad/uploads/{uid}")
    def upload_delete(uid: str):
        updir_or_404(uid)
        return {"deleted": uid, "problems": custom.delete_upload(uid)}

    @r.post("/api/cad/uploads/{uid}/sets", status_code=201)
    def make_sets(uid: str, body: dict = Body(...)):
        updir_or_404(uid)
        try:
            if body.get("mode") == "auto":
                ids = custom.generate_sets(uid, int(body.get("count") or 1), body)
            elif body.get("mode") == "manual":
                ids = [custom.define_set(uid, body)]
            else:
                raise custom.CadError("mode must be 'auto' (generated BC sets) or 'manual'")
        except (custom.CadError, TypeError, ValueError) as e:
            raise HTTPException(422, str(e)) from None
        custom.refresh()
        return {"problem_ids": ids}

    @r.delete("/api/cad/problems/{pid}")
    def set_delete(pid: str):
        try:
            custom.delete_set(pid)
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None
        return {"deleted": pid}

    @r.post("/api/cad/problems/{pid}/adopt")
    def adopt(pid: str, body: dict = Body(...)):
        try:
            custom.find_set(pid)
            st = store.get_study(str(body.get("study_id")))
        except KeyError as e:
            raise HTTPException(404, str(e).strip('"')) from None
        if st.get("problem_id") != pid:
            raise HTTPException(422, f"study {st['study_id']} is of problem {st.get('problem_id')}, not {pid}")
        if st.get("status") != "succeeded":
            raise HTTPException(422, f"study {st['study_id']} has not succeeded (status {st.get('status')})")
        if st.get("backend") != "skfem" or int(st.get("order") or 0) != 2:
            raise HTTPException(422, "a reference needs a study with the fixed solver (skfem) and P2 elements")
        try:
            out = custom.adopt_study(pid, custom.study_levels(st), {
                "kind": "convergence study", "study_id": st["study_id"], "nested": bool(st.get("nested")),
                "sizes": st.get("sizes")})
        except custom.CadError as e:
            raise HTTPException(422, str(e)) from None
        custom.refresh()
        return out

    return r
