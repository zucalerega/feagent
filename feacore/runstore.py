"""File-based store for solve runs, convergence studies and agent submissions.

Layout (root = ``$FEACORE_RUNS_DIR``, else ``<repo>/runs``; a ``.gitignore`` with ``*`` is
written on first use so nothing in it is ever committed)::

    runs/
      runs/<run_id>/run.json             one solve: case, SolveInfo, QoI values, status
      runs/<run_id>/mesh.msh             the input mesh (uploads / record_result(copy_mesh=True))
      runs/<run_id>/fields.npz           boundary triangulation + nodal fields for the viewer
      studies/<study_id>/study.json      convergence study: run_ids, progress, GCI analysis
      studies/<study_id>/meshes/*.msh    the study's meshes
      submissions/<submission_id>.json   one agent's set of runs on one problem + GCI analysis
      batches/<batch_id>.json            a benchmark batch: settings + one study per problem
      cache/                             derived data (geometry previews)

Every record is a JSON object with a ``schema`` key (``feacore.run/1``, ``feacore.study/1``,
``feacore.submission/1``). Records are written atomically (temp file + rename), so readers in
other processes never see a partial file. See docs/ui.md for the field-by-field format.

Python use (e.g. from an agent harness)::

    from feacore.runstore import RunStore
    store = RunStore()
    run_id = store.record_result(result, mesh_path="part_h2.msh", problem_id="plate_hole",
                                 agent_id="agent-07")          # result = feacore.solver.solve(...)
    sub = store.create_submission("plate_hole", "agent-07", run_ids=[r1, r2, r3])
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

RUN_SCHEMA = "feacore.run/1"
STUDY_SCHEMA = "feacore.study/1"
SUBMISSION_SCHEMA = "feacore.submission/1"
BATCH_SCHEMA = "feacore.batch/1"

RUN_STATUSES = ("queued", "running", "succeeded", "failed", "interrupted")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# --------------------------------------------------------------------------- helpers


def default_root() -> Path:
    env = os.environ.get("FEACORE_RUNS_DIR")
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[1]
    if (repo / "pyproject.toml").exists():
        return repo / "runs"
    return Path.cwd() / "runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def new_id(prefix: str) -> str:
    """Sortable, unique id: ``<prefix>-YYYYmmdd-HHMMSS-<6 hex>``."""
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def check_id(record_id: str) -> str:
    if not isinstance(record_id, str) or not _ID_RE.match(record_id) or ".." in record_id:
        raise KeyError(f"invalid id {record_id!r}")
    return record_id


def jsonable(obj: Any) -> Any:
    """Recursively convert to strict-JSON types: numpy -> python, dataclasses -> dict,
    Path -> str, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data: Any) -> Path:
    """Atomic JSON write (strict JSON: NaN/inf become null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=1, allow_nan=False))
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> Any:
    # A writer replaces the file atomically; retry briefly in case it vanished mid-replace.
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise
        except (json.JSONDecodeError, OSError):
            if attempt == 4:
                raise
            time.sleep(0.02)


# --------------------------------------------------------------------------- store


class RunStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_root()

    # ---- layout
    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def studies_dir(self) -> Path:
        return self.root / "studies"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def batches_dir(self) -> Path:
        return self.root / "batches"

    def ensure(self) -> "RunStore":
        for d in (self.runs_dir, self.studies_dir, self.submissions_dir, self.cache_dir, self.batches_dir):
            d.mkdir(parents=True, exist_ok=True)
        gi = self.root / ".gitignore"
        if not gi.exists():
            gi.write_text("# feacore run store: never commit\n*\n")
        return self

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / check_id(run_id)

    def study_dir(self, study_id: str) -> Path:
        return self.studies_dir / check_id(study_id)

    def relpath(self, path: str | Path) -> str:
        """Path relative to the store root when inside it (records stay relocatable)."""
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    def abspath(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    # ================================================================== runs
    def create_run(self, *, backend: str = "skfem", case: dict | None = None,
                   problem_id: str | None = None, agent_id: str | None = None,
                   submission_id: str | None = None, study_id: str | None = None,
                   level: int | None = None, h_nominal: float | None = None,
                   source: str = "python", status: str = "queued", meta: dict | None = None,
                   run_id: str | None = None, **extra) -> dict:
        self.ensure()
        run_id = check_id(run_id) if run_id else new_id("run")
        d = self.run_dir(run_id)
        if (d / "run.json").exists():
            raise ValueError(f"run {run_id!r} already exists")
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "status": status,
            "source": source,
            "problem_id": problem_id,
            "agent_id": agent_id,
            "submission_id": submission_id,
            "study_id": study_id,
            "level": level,
            "h_nominal": h_nominal,
            "backend": backend,
            "case": case,
            "mesh_file": None,
            "info": None,
            "qois": {},
            "fields": None,
            "error": None,
            "traceback": None,
            "meta": meta or {},
        }
        rec.update(extra)
        write_json(d / "run.json", rec)
        return rec

    def get_run(self, run_id: str) -> dict:
        p = self.run_dir(run_id) / "run.json"
        if not p.exists():
            raise KeyError(f"unknown run {run_id!r}")
        return read_json(p)

    def update_run(self, run_id: str, **fields) -> dict:
        rec = self.get_run(run_id)
        rec.update(fields)
        write_json(self.run_dir(run_id) / "run.json", rec)
        return rec

    def list_runs(self, problem_id: str | None = None, agent_id: str | None = None,
                  study_id: str | None = None, submission_id: str | None = None,
                  status: str | None = None, limit: int | None = None) -> list[dict]:
        """Run records, newest first (ids sort by creation time)."""
        if not self.runs_dir.exists():
            return []
        out = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            p = d / "run.json"
            if not p.exists():
                continue
            try:
                r = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            if problem_id is not None and r.get("problem_id") != problem_id:
                continue
            if agent_id is not None and r.get("agent_id") != agent_id:
                continue
            if study_id is not None and r.get("study_id") != study_id:
                continue
            if submission_id is not None and r.get("submission_id") != submission_id:
                continue
            if status is not None and r.get("status") != status:
                continue
            out.append(r)
            if limit is not None and len(out) >= limit:
                break
        return out

    def save_run(self, record: dict) -> str:
        """Import a run solved elsewhere (no fields). Needs at least ``qois`` and preferably
        ``info`` (SolveInfo dict: ``h_rep`` and ``order`` are used by the GCI analysis)."""
        record = dict(record)
        qois = record.pop("qois", None)
        if not isinstance(qois, dict):
            raise ValueError("run record needs a 'qois' dict {name: value}")
        keep = {k: record.pop(k) for k in list(record) if k in (
            "backend", "case", "problem_id", "agent_id", "submission_id", "study_id", "level",
            "h_nominal", "meta", "run_id")}
        info = record.pop("info", None)
        status = record.pop("status", "succeeded")
        rec = self.create_run(source=record.pop("source", "import"), status=status, **keep)
        return self.update_run(rec["run_id"], info=info, qois=qois,
                               finished_at=record.pop("finished_at", now_iso()),
                               **{k: v for k, v in record.items() if k not in ("schema", "created_at")})["run_id"]

    def record_result(self, result, *, mesh_path: str | Path | None = None,
                      problem_id: str | None = None, agent_id: str | None = None,
                      submission_id: str | None = None, study_id: str | None = None,
                      level: int | None = None, h_nominal: float | None = None,
                      fields: bool = True, copy_mesh: bool = True, source: str = "python",
                      meta: dict | None = None, run_id: str | None = None) -> str:
        """Store a ``feacore.solver.Result``: case, SolveInfo, QoI values and (``fields``)
        the viewer data. ``mesh_path`` defaults to the case's mesh."""
        case = result.case
        mesh_path = Path(mesh_path) if mesh_path is not None else case.mesh_path()
        rec = self.create_run(backend=result.info.backend, case=case.to_dict(), problem_id=problem_id,
                              agent_id=agent_id, submission_id=submission_id, study_id=study_id,
                              level=level, h_nominal=h_nominal, source=source, status="running",
                              meta=meta, run_id=run_id)
        rid = rec["run_id"]
        mesh_file = None
        if mesh_path is not None and mesh_path.exists():
            if copy_mesh:
                dst = self.run_dir(rid) / "mesh.msh"
                shutil.copyfile(mesh_path, dst)
                mesh_file = self.relpath(dst)
            else:
                mesh_file = self.relpath(mesh_path)
        qois = {k: float(v) for k, v in result.evaluate_all().items()}
        update = dict(info=dataclasses.asdict(result.info), qois=qois, mesh_file=mesh_file,
                      status="succeeded", finished_at=now_iso())
        if fields and mesh_path is not None and mesh_path.exists():
            try:
                update["fields"] = self.write_fields(rid, result, mesh_path)
            except Exception as e:  # noqa: BLE001 - the viewer data is optional
                update.setdefault("warnings", []).append(f"field extraction failed: {type(e).__name__}: {e}")
        self.update_run(rid, **update)
        return rid

    # ---- viewer fields (.npz)
    def write_fields(self, run_id: str, result, mesh_path: str | Path) -> dict:
        """Extract the boundary triangulation + nodal fields of ``result`` and save
        ``fields.npz``; returns the summary stored under ``run["fields"]``."""
        from feacore.ui.viewdata import extract_view, save_view

        view = extract_view(result, mesh_path)
        return save_view(view, self.run_dir(run_id) / "fields.npz", relto=self.root)

    def fields_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "fields.npz"

    def load_fields(self, run_id: str) -> dict[str, np.ndarray]:
        p = self.fields_path(run_id)
        if not p.exists():
            raise KeyError(f"run {run_id!r} has no field data")
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    # ================================================================== studies
    def create_study(self, problem_id: str, backend: str = "skfem", order: int = 2,
                     sizes: list[float] | None = None, geom_order: int | None = None,
                     meta: dict | None = None, study_id: str | None = None,
                     source: str = "ui", nested: bool = False, batch_id: str | None = None) -> dict:
        self.ensure()
        study_id = check_id(study_id) if study_id else new_id("study")
        d = self.study_dir(study_id)
        if (d / "study.json").exists():
            raise ValueError(f"study {study_id!r} already exists")
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "schema": STUDY_SCHEMA,
            "study_id": study_id,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "status": "queued",
            "source": source,
            "problem_id": problem_id,
            "backend": backend,
            "order": int(order),
            "geom_order": geom_order,
            "sizes": sizes,
            "nested": bool(nested),
            "batch_id": batch_id,
            "progress": {"done": 0, "total": len(sizes) if sizes else None, "stage": "queued",
                         "message": "waiting for a worker", "levels": []},
            "run_ids": [],
            "result": None,          # feacore.verification.study.convergence_study(...) output
            "interpretation": None,  # per-QoI verdicts + flag explanations (feacore.ui.analysis)
            "error": None,
            "traceback": None,
            "meta": meta or {},
        }
        write_json(d / "study.json", rec)
        return rec

    def save_study(self, study: dict, problem_id: str | None = None, run_ids: list[str] | None = None,
                   meta: dict | None = None) -> str:
        """Import a finished ``convergence_study`` dict computed elsewhere."""
        rec = self.create_study(problem_id or study.get("problem"), backend=study.get("backend", "skfem"),
                                order=study.get("order", 2), sizes=study.get("sizes"),
                                geom_order=study.get("geom_order"), meta=meta, source="import")
        from feacore.ui.analysis import interpret_study

        n = len(study.get("runs", []))
        self.update_study(rec["study_id"], status="succeeded", result=study, run_ids=run_ids or [],
                          finished_at=now_iso(), interpretation=interpret_study(study),
                          progress={"done": n, "total": n, "stage": "done", "message": "imported",
                                    "levels": []})
        return rec["study_id"]

    def get_study(self, study_id: str) -> dict:
        p = self.study_dir(study_id) / "study.json"
        if not p.exists():
            raise KeyError(f"unknown study {study_id!r}")
        return read_json(p)

    def update_study(self, study_id: str, **fields) -> dict:
        rec = self.get_study(study_id)
        rec.update(fields)
        write_json(self.study_dir(study_id) / "study.json", rec)
        return rec

    def list_studies(self, problem_id: str | None = None, status: str | None = None,
                     limit: int | None = None) -> list[dict]:
        if not self.studies_dir.exists():
            return []
        out = []
        for d in sorted(self.studies_dir.iterdir(), reverse=True):
            p = d / "study.json"
            if not p.exists():
                continue
            try:
                s = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            if problem_id is not None and s.get("problem_id") != problem_id:
                continue
            if status is not None and s.get("status") != status:
                continue
            out.append(s)
            if limit is not None and len(out) >= limit:
                break
        return out

    # ================================================================== submissions
    def create_submission(self, problem_id: str, agent_id: str, run_ids: Iterable[str] | None = None,
                          levels: list[dict] | None = None, notes: str = "", meta: dict | None = None,
                          submission_id: str | None = None, order: int | None = None) -> dict:
        """An agent's answer to one problem: its runs on a sequence of its own meshes.

        Give ``run_ids`` (records in this store) and/or ``levels`` = ``[{"h": h_rep,
        "qois": {name: value}, "n_dofs": ..., "order": ...}, ...]`` for runs solved elsewhere.
        The per-QoI Richardson/GCI analysis is computed and stored with the record."""
        from feacore.ui.analysis import analyze_submission

        self.ensure()
        submission_id = check_id(submission_id) if submission_id else new_id("sub")
        path = self.submissions_dir / f"{submission_id}.json"
        if path.exists():
            raise ValueError(f"submission {submission_id!r} already exists")
        run_ids = list(run_ids or [])
        lv = [self._level_from_run(self.get_run(r)) for r in run_ids]
        for i, l in enumerate(levels or []):
            if "qois" not in l or "h" not in l:
                raise ValueError(f"levels[{i}] needs 'h' and 'qois'")
            lv.append({"run_id": l.get("run_id"), "h": float(l["h"]), "qois": dict(l["qois"]),
                       "n_dofs": l.get("n_dofs"), "n_cells": l.get("n_cells"), "order": l.get("order"),
                       "backend": l.get("backend")})
        if not lv:
            raise ValueError("a submission needs run_ids or levels")
        rec = {
            "schema": SUBMISSION_SCHEMA,
            "submission_id": submission_id,
            "created_at": now_iso(),
            "problem_id": problem_id,
            "agent_id": agent_id,
            "run_ids": run_ids,
            "levels": lv,
            "order": order,
            "notes": notes,
            "meta": meta or {},
            "analysis": analyze_submission(problem_id, lv, order=order),
        }
        write_json(path, rec)
        for r in run_ids:  # back-link (best effort)
            try:
                if self.get_run(r).get("submission_id") is None:
                    self.update_run(r, submission_id=submission_id)
            except KeyError:
                pass
        return rec

    @staticmethod
    def _level_from_run(run: dict) -> dict:
        if run.get("status") != "succeeded":
            raise ValueError(f"run {run['run_id']} has status {run.get('status')!r}; only succeeded runs "
                             "can be part of a submission")
        info = run.get("info") or {}
        h = info.get("h_rep", run.get("h_nominal"))
        if h is None:
            raise ValueError(f"run {run['run_id']} has no info.h_rep (or h_nominal)")
        return {"run_id": run["run_id"], "h": float(h), "qois": run.get("qois") or {},
                "n_dofs": info.get("n_dofs"), "n_cells": info.get("n_cells"), "order": info.get("order"),
                "backend": run.get("backend")}

    def get_submission(self, submission_id: str) -> dict:
        p = self.submissions_dir / f"{check_id(submission_id)}.json"
        if not p.exists():
            raise KeyError(f"unknown submission {submission_id!r}")
        return read_json(p)

    def list_submissions(self, problem_id: str | None = None, agent_id: str | None = None) -> list[dict]:
        if not self.submissions_dir.exists():
            return []
        out = []
        for p in sorted(self.submissions_dir.glob("*.json"), reverse=True):
            try:
                s = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            if problem_id is not None and s.get("problem_id") != problem_id:
                continue
            if agent_id is not None and s.get("agent_id") != agent_id:
                continue
            out.append(s)
        return out

    # ================================================================== batches
    def create_batch(self, items: list[dict], settings: dict, name: str = "", meta: dict | None = None,
                     batch_id: str | None = None) -> dict:
        """A benchmark batch: ``items`` = ``[{"problem_id", "study_id"}]`` (one convergence study
        per problem), ``settings`` = the shared study options (backend, order, nested, levels...).
        Progress and results are derived from the studies when read, so this record is static."""
        self.ensure()
        batch_id = check_id(batch_id) if batch_id else new_id("batch")
        path = self.batches_dir / f"{batch_id}.json"
        if path.exists():
            raise ValueError(f"batch {batch_id!r} already exists")
        rec = {"schema": BATCH_SCHEMA, "batch_id": batch_id, "created_at": now_iso(), "name": name,
               "settings": settings, "items": items, "meta": meta or {}}
        write_json(path, rec)
        return rec

    def update_batch(self, batch_id: str, **fields) -> dict:
        rec = self.get_batch(batch_id)
        rec.update(fields)
        write_json(self.batches_dir / f"{check_id(batch_id)}.json", rec)
        return rec

    def get_batch(self, batch_id: str) -> dict:
        p = self.batches_dir / f"{check_id(batch_id)}.json"
        if not p.exists():
            raise KeyError(f"unknown batch {batch_id!r}")
        return read_json(p)

    def list_batches(self, limit: int | None = None) -> list[dict]:
        if not self.batches_dir.exists():
            return []
        out = []
        for p in sorted(self.batches_dir.glob("*.json"), reverse=True):
            try:
                out.append(read_json(p))
            except (OSError, json.JSONDecodeError):
                continue
            if limit is not None and len(out) >= limit:
                break
        return out

    # ================================================================== maintenance
    def mark_interrupted(self, sources: tuple[str, ...] = ("api", "ui")) -> int:
        """After a server restart, jobs that were queued/running in the old process are dead."""
        n = 0
        for r in self.list_runs():
            if r.get("status") in ("queued", "running") and r.get("source") in sources:
                self.update_run(r["run_id"], status="interrupted", error="server restarted before the job finished")
                n += 1
        for s in self.list_studies():
            if s.get("status") in ("queued", "running") and s.get("source") in sources:
                self.update_study(s["study_id"], status="interrupted",
                                  error="server restarted before the study finished")
                n += 1
        return n
