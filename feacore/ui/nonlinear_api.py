"""Nonlinear runs for the dashboard (#/nonlinear): where the nonlinear tools write, their force-displacement
curves against the physical test, their pictures and videos, the committed gallery, and what this box can run.

Every run is a directory, named by a ref (feacore.ui.media: ``@runs/...`` in the run store, else a committed
path in the repo). Kinds, by the files a directory holds:

  crush   experiments/physical/crush.py, scripts/ccx_crush.py: force_displacement.csv (+ _full for a resumed run),
          run.json, summary.json (at the end), progress.log, result_NNNN.vtu. Dashboard: <runs>/nonlinear/crush/,
          the CLI default: <runs>/nonlinear/<system>/<part>/<run-id>/
  ramp    scripts/ccx_compression_ramp.py: <material>_<platen>.json (history), curve_<name>.csv, summary.json,
          compare.png. Dashboard: <runs>/nonlinear/ramp/, CLI: <runs>/ccx_ramp/, committed:
          physical_tests/abc_00000065_compression/ccx_ramp/
  score   ace.nonlinear.run_curve, the agents' frozen scoring ramp: summary.json (u_part_mm, u_cross_mm, force_N,
          mean_rel_err), curve.csv, series/. scripts/nl_baselines.py writes one per mesh into a group directory
          (baselines.json): dashboard <runs>/nonlinear/score/, committed results/real_data_finetune/baselines/
          and talk_plots/act3_real_data/videos/{before,first}_run/

Renderers (feacore/ui/tools/nonlinear.py) write to <runs>/nonlinear/renders/<task>/; a run's media are the
pictures in its own directory plus the outputs of the render tasks that name it. Crush meshes live in
<runs>/nonlinear/meshes/.

API:
  GET /api/nonlinear/capabilities         CalculiX, ffmpeg, off-screen rendering on this server
  GET /api/nonlinear/runs                 every run, newest first: kind, status, peak, error vs the test
  GET /api/nonlinear/run?ref=             one run: its curves (mm, N), facts, media and files
  GET /api/nonlinear/test                 the C-bracket's Instron curve and the frozen crosshead mapping
  GET /api/nonlinear/meshes               crush meshes in the run store (for the crush form)
  GET /api/nonlinear/gallery              committed figures and videos with their captions
  GET /api/nonlinear/file?ref=&download=  one file (pictures, videos, CSV, ...), path-checked
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import os
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from feacore.ui.media import REPO_ROOT, RefError, file_response, kind_of, media_files, resolve_ref, to_ref

TEST_DIR = REPO_ROOT / "physical_tests" / "abc_00000065_compression"
SPEC_FILE = TEST_DIR / "nonlinear_spec.json"
STORE_SCAN = ("nonlinear", "ccx_ramp")
SKIP_DIRS = {"ccx_work", "series", "harness", "vtu", "meshes", "renders", "media"}
COMMITTED_RUNS = {
    "physical_tests/abc_00000065_compression/ccx_ramp": "Elastoplastic ramp of the test (committed, 41k-node mesh)",
    "results/real_data_finetune/baselines": "Frozen scoring ramp: reference meshes (committed)",
    "talk_plots/act3_real_data/videos/before_run": "Before post-training: frictionless platens, automesher (committed)",
    "talk_plots/act3_real_data/videos/first_run": "The team's first nonlinear simulation (committed)",
}
SCORE_LABELS = {"auto": "automesher", "swarm_bc0": "swarm bc0 mesh", "h4": "uniform 4 mm", "h25": "uniform 2.5 mm",
                "h20": "uniform 2 mm", "before_run": "before post-training", "first_run": "first simulation"}
RUN_TOOLS = {"nonlinear.crush": "crush", "nonlinear.ramp": "ramp", "nonlinear.score": "score"}
RENDER_TOOLS = ("nonlinear.render_ramp", "nonlinear.render_crush", "nonlinear.render_score")
MAX_POINTS = 6000


# --------------------------------------------------------------------------- the test and the frozen mapping


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _test(spec_mtime: float) -> dict:
    from ace.evidence import read_table

    spec = json.loads(SPEC_FILE.read_text())
    t = read_table(REPO_ROOT / spec["test_csv"], {"load": "Force", "displacement": "Displacement",
                                                  "load_unit": "N", "displacement_unit": "mm"})
    u_all, F_all = t["meas"] * 1e3, t["load"]
    peak = int(np.argmax(F_all))
    order = np.argsort(u_all[: peak + 1], kind="stable")     # digitized: keep the loading branch increasing
    u, F = u_all[: peak + 1][order], F_all[: peak + 1][order]
    ck = [float(c) for c in spec["checkpoints_mm"]]
    return {"test_id": "cload-compression-1", "problem_id": spec["problem_id"],
            "label": "Instron test (printed C-bracket, crosshead travel)",
            "curve": np.column_stack([u_all, F_all]).round(4).tolist(), "peak_N": float(F_all[peak]),
            "peak_mm": float(u_all[peak]), "loading_mm": u.tolist(), "loading_N": F.tolist(),
            "checkpoints_mm": ck, "checkpoints_N": [float(f) for f in np.interp(ck, u, F)],
            "delta_mm": float(spec["delta_mm"]), "compliance_mm_per_N": float(spec["compliance_mm_per_N"]),
            "mapping": "u_cross = u_part + delta + compliance x F (seating gap and load-train compliance, fitted "
                       "once on a fine mesh and frozen: physical_tests/abc_00000065_compression/nonlinear_spec.json)"}


def test_curve() -> dict:
    return _test(_mtime(SPEC_FILE))


def checkpoint_error(u_part, F, u_cross=None) -> float | None:
    """Mean |F_sim - F_test| / F_test at the test's checkpoints (crosshead travel), as ace.nonlinear scores a
    run: the simulated travel mapped to the crosshead with the frozen seating gap and compliance, and a
    checkpoint the run never reaches counting 100 %."""
    try:
        t = test_curve()
    except Exception:  # noqa: BLE001 - no test data: no error
        return None
    u_part, F = np.asarray(u_part, float), np.asarray(F, float)
    if len(F) < 2:
        return None
    if u_cross is None:
        u_cross = np.where(u_part > 0, u_part + t["delta_mm"] + t["compliance_mm_per_N"] * F, 0.0)
    x = np.maximum.accumulate(np.asarray(u_cross, float))     # erosion drops the force at a held platen
    ck, Fm = np.array(t["checkpoints_mm"]), np.array(t["checkpoints_N"])
    rel = np.where(ck <= x[-1], np.abs(np.interp(ck, x, F) - Fm) / Fm, 1.0)
    return float(rel.mean())


# --------------------------------------------------------------------------- reading runs


def _read_json(p: Path, max_bytes: int = 20_000_000):
    try:
        if p.stat().st_size > max_bytes:
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            out.append(float("nan"))
    return out


def _is_score_dir(d: Path) -> bool:
    f = d / "curve.csv"
    if f.exists():
        try:
            with open(f) as fh:
                return fh.readline().startswith("u_part_mm")
        except OSError:
            return False
    s = _read_json(d / "summary.json", 2_000_000) if (d / "summary.json").exists() else None
    return isinstance(s, dict) and "mesh_sha256" in s


def _ramp_names(d: Path) -> list[str]:
    names = {f.stem for f in d.glob("*_s*.json") if f.stem.rsplit("_", 1)[-1] in ("slide", "stick")}
    names |= {f.stem[len("curve_"):] for f in d.glob("curve_*.csv")
              if f.stem.rsplit("_", 1)[-1] in ("slide", "stick")}
    return sorted(names)


def _subdirs(d: Path) -> list[Path]:
    try:
        return sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".") and c.name not in SKIP_DIRS)
    except OSError:
        return []


def detect(d: Path) -> str | None:
    """crush | ramp | score | score_group | None."""
    if (d / "force_displacement.csv").exists():
        return "crush"
    if _is_score_dir(d):
        return "score"
    if _ramp_names(d):
        return "ramp"
    if (d / "baselines.json").exists() or any(_is_score_dir(c) for c in _subdirs(d)):
        return "score_group"
    return None


def scan(root: Path, max_depth: int = 5) -> list[tuple[Path, str]]:
    """Run directories under root (a run directory's own subdirectories are not searched)."""
    found: list[tuple[Path, str]] = []

    def walk(d: Path, depth: int):
        k = detect(d)
        if k:
            found.append((d, k))
            return
        if depth < max_depth:
            for c in _subdirs(d):
                if not c.name.endswith("_frames"):
                    walk(c, depth + 1)

    if root.is_dir():
        walk(root, 0)
    return found


def _thin(*cols, keep=()):
    """Every column cut to at most MAX_POINTS points (evenly), keeping the indices in ``keep``."""
    n = len(cols[0])
    if n <= MAX_POINTS:
        return cols
    idx = np.unique(np.concatenate([np.linspace(0, n - 1, MAX_POINTS).astype(int), np.asarray(keep, int)]))
    return tuple([c[i] for i in idx] for c in cols)


def _crush_rows(d: Path, store_root: Path, depth: int = 0) -> list[dict]:
    """A crush run's CSV rows; a resumed run that has not finished yet (no force_displacement_full.csv) gets its
    parents' rows up to the step it continues from, as crush.py writes them at the end. Parents are followed
    only inside the run store or the repo."""
    full = d / "force_displacement_full.csv"
    try:
        rows = list(csv.DictReader(open(full if full.exists() else d / "force_displacement.csv", newline="")))
    except OSError:
        return []
    meta = _read_json(d / "run.json") or {}
    parent, at = meta.get("continued_from"), meta.get("continued_at_step")
    if full.exists() or not parent or at is None or depth > 20:
        return rows
    p = Path(parent)
    if not p.is_dir() or to_ref(p, store_root) is None:
        return rows
    older = [r for r in _crush_rows(p, store_root, depth + 1) if _floats([r], "step")[0] <= float(at)]
    return older + rows


def crush_curves(d: Path, store_root: Path) -> tuple[list[dict], dict]:
    rows = _crush_rows(d, store_root)
    u, F, nf = _floats(rows, "platen_travel_mm"), _floats(rows, "force_N"), _floats(rows, "n_failed")
    ok = [i for i in range(len(u)) if np.isfinite(u[i]) and np.isfinite(F[i])]
    u, F, nf = [u[i] for i in ok], [F[i] for i in ok], [nf[i] if np.isfinite(nf[i]) else 0 for i in ok]
    if u and u[0] > 0:
        u, F, nf = [0.0, *u], [0.0, *F], [0, *nf]
    meta = _read_json(d / "run.json") or {}
    summ = _read_json(d / "summary.json") or {}
    part = meta.get("part") or summ.get("part")
    facts: dict = {"part": part, "system": meta.get("system"), "platen": meta.get("platen") or summ.get("platen"),
                   "failure": (meta.get("failure") or {}).get("criterion"), "release": meta.get("release"),
                   "threads": meta.get("threads"), "continued_from": meta.get("continued_from"),
                   "increments": len(u) - 1 if u else 0}
    first = next((i for i, n in enumerate(nf) if n > 0), None)
    if F:
        ip = int(np.nanargmax(F))
        facts.update(peak_N=F[ip], travel_at_peak_mm=u[ip], travel_mm=u[-1], force_N=F[-1],
                     n_failed=int(nf[-1]) if nf else 0,
                     first_failure_mm=u[first] if first is not None else None)
    for k in ("stop_reason", "separated", "wall_s", "n_nodes", "n_cells", "n_detached", "travel_at_separation_mm",
              "height_m"):
        if k in summ:
            facts[k] = summ[k]
    if "n_nodes" not in facts and meta.get("mesh"):       # still running: the mesh's own report has the counts
        rep = _read_json(Path(meta["mesh"]).parent / "report.json") or {}
        facts.update({k: v for k, v in (("n_nodes", rep.get("n_nodes")), ("n_cells", rep.get("n_elements"))) if v})
    if isinstance(summ.get("first_failure"), dict):
        facts["first_failure_height_fraction"] = summ["first_failure"].get("height_fraction")
    label = f"crush: {facts.get('system') or 'mesh'} / {part or '?'} ({facts.get('platen') or 'bonded'}, " \
            f"{facts.get('failure') or 'peeq'} erosion)"
    keep = [i for i in (first, int(np.nanargmax(F)) if F else None) if i is not None]
    u2, F2, n2 = _thin(u, F, nf, keep=keep)
    curve = {"name": "crush", "label": label, "u": u2, "F": F2, "n_failed": n2, "x": "platen travel",
             "part": part, "first_failure": [u[first], F[first]] if first is not None else None,
             "error": checkpoint_error(u, F) if part == "abc_00000065" else None}
    return [curve], facts


def ramp_curves(d: Path) -> tuple[list[dict], dict]:
    summ = _read_json(d / "summary.json") or {}
    curves, facts = [], {"part": "abc_00000065", "problem_id": "abc_00000065_compression"}
    for name in _ramp_names(d):
        r = _read_json(d / f"{name}.json")
        s = (summ.get("runs") or {}).get(name, {})
        if r and r.get("history"):
            u = [0.0] + [h["u_mm"] for h in r["history"]]
            F = [0.0] + [h["F_N"] for h in r["history"]]
            mat, platen, nodes = r.get("material", {}), r.get("platen"), r.get("n_nodes")
        else:
            u, F = [], []
            try:
                for row in csv.reader(open(d / f"curve_{name}.csv", newline="")):
                    try:
                        u.append(float(row[0]))
                        F.append(float(row[1]))
                    except (ValueError, IndexError):
                        continue
            except OSError:
                pass
            mat, platen, nodes = s.get("material", {}), s.get("platen"), s.get("n_nodes")
        if not F:
            continue
        err = checkpoint_error(u, F)
        lab = f"{name}: E {mat.get('E', 0):.0f} MPa, {platen} platen" + (f", {nodes:,} nodes" if nodes else "")
        curves.append({"name": name, "label": lab, "u": u, "F": F, "x": "platen travel", "part": "abc_00000065",
                       "error": err, "first_yield": [s["first_yield"]["u_mm"], s["first_yield"]["F_N"]]
                       if isinstance(s.get("first_yield"), dict) else None})
        facts[name] = {"F_max_N": float(np.max(F)), "u_reached_mm": u[-1], "nodes": nodes, "error": err,
                       "initial_stiffness_N_per_mm": s.get("initial_stiffness_N_per_mm"), "wall_s": s.get("wall_s")
                       or (r or {}).get("wall_s"), "converged": s.get("converged", (r or {}).get("converged"))}
    return curves, facts


def score_curve(d: Path, name: str | None = None) -> dict | None:
    s = _read_json(d / "summary.json")
    name = name or d.name
    label = SCORE_LABELS.get(name, name)
    if not isinstance(s, dict):
        return None
    if not s.get("ok") or not s.get("force_N"):
        return {"name": name, "label": f"{label}: failed", "u": [], "F": [], "error": None,
                "failed": s.get("error") or "the run failed"}
    err = s.get("mean_rel_err")
    return {"name": name, "label": f"{label}: {s.get('nodes', 0):,} nodes", "u": s["u_part_mm"],
            "u_cross": s.get("u_cross_mm"), "F": s["force_N"], "x": "part shortening", "part": "abc_00000065",
            "error": err, "nodes": s.get("nodes"), "dofs": s.get("dofs"), "wall_s": s.get("wall_s"),
            "converged": s.get("converged"), "rel_err": s.get("rel_err"), "peeq_max": s.get("peeq_max"),
            "has_series": bool(s.get("series_dir")) and (d / "series").is_dir(), "dir": d}


def score_curves(d: Path, group: bool) -> tuple[list[dict], dict]:
    if not group:
        c = score_curve(d)
        curves = [c] if c else []
    else:
        curves = [c for c in (score_curve(x) for x in _subdirs(d)) if c]
    facts = {"part": "abc_00000065", "problem_id": "abc_00000065_compression",
             "meshes": {c["name"]: {k: c.get(k) for k in ("nodes", "dofs", "error", "wall_s", "converged", "peeq_max")}
                        for c in curves}}
    return curves, facts


# --------------------------------------------------------------------------- run records


def _task_maps(tasks) -> tuple[dict, dict]:
    """{run dir: newest task that wrote it}, {run dir: [render tasks that name it]}."""
    by_out, renders = {}, {}
    if tasks is None:
        return by_out, renders
    for t in tasks.tasks.list(group="nonlinear", limit=2000):         # newest first
        if t.get("out_abs"):
            by_out.setdefault(str(Path(t["out_abs"]).resolve()), t)
        if t.get("tool") in RENDER_TOOLS and (t.get("args") or {}).get("run"):
            arg = str(t["args"]["run"])                     # absolute, repo-relative or @runs/...
            p = tasks.store.root / arg[len("@runs/"):] if arg.startswith("@runs/") else Path(arg)
            try:
                run = str((p if p.is_absolute() else REPO_ROOT / p).resolve())
            except OSError:
                continue
            renders.setdefault(run, []).append(t)
    return by_out, renders


def _task_brief(t: dict | None) -> dict | None:
    if not t:
        return None
    return {k: t.get(k) for k in ("id", "tool", "title", "state", "created_at", "started_at", "finished_at", "error")}


def _status(kind: str, d: Path, task: dict | None) -> str:
    state = (task or {}).get("state")
    if state in ("queued", "running"):
        return "running"
    if kind == "crush":
        if (d / "summary.json").exists():
            return "finished"
        log = d / "progress.log"
        if time.time() - _mtime(log if log.exists() else d / "force_displacement.csv") < 300 and not state:
            return "running"                 # a CLI run still writing
        if (d / "checkpoint.json").exists():
            return "stopped"                 # killed or failed after a finished block: it can be resumed
        return "failed" if state == "failed" else "stopped"
    if state == "failed":
        return "failed"
    return "finished"


def _title(kind: str, d: Path, ref: str, facts: dict) -> str:
    if ref in COMMITTED_RUNS:
        return COMMITTED_RUNS[ref]
    if kind == "crush":
        return f"Crush: {facts.get('system') or 'mesh'} / {facts.get('part') or '?'}"
    if kind == "ramp":
        return "Elastoplastic ramp of the test"
    return "Frozen scoring ramp vs the test"


COMMITTED_SHORT = {"physical_tests/abc_00000065_compression/ccx_ramp": "committed ramp",
                   "results/real_data_finetune/baselines": "baselines",
                   "talk_plots/act3_real_data/videos/before_run": "before", "talk_plots/act3_real_data/videos/first_run": "first"}


def _short(d: Path, ref: str) -> str:
    """A short name for legends: 'ramp 11-06 00:19' for a dashboard run, the folder name otherwise."""
    if ref in COMMITTED_SHORT:
        return COMMITTED_SHORT[ref]
    m = re.fullmatch(r"([a-z-]+)-\d{4}(\d\d)(\d\d)-(\d\d)(\d\d)\d\d-[0-9a-f]+", d.name)
    return f"{m.group(1)} {m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else d.name


def _files(d: Path, store_root: Path) -> list[dict]:
    out = []
    try:
        entries = sorted(d.iterdir())
    except OSError:
        return out
    for p in entries:
        if p.is_file() and p.suffix.lower() in (".csv", ".json", ".log", ".txt", ".png", ".mp4", ".msh", ".pdf"):
            ref = to_ref(p, store_root)
            if ref:
                out.append({"name": p.name, "ref": ref, "bytes": p.stat().st_size})
    return out


def run_record(d: Path, kind: str, store_root: Path, task: dict | None = None, renders=(), full: bool = False) -> dict:
    ref = to_ref(d, store_root)
    if kind == "crush":
        curves, facts = crush_curves(d, store_root)
    elif kind == "ramp":
        curves, facts = ramp_curves(d)
    else:
        curves, facts = score_curves(d, kind == "score_group")
    stamp = max([_mtime(d)] + [_mtime(p) for p in (d / "force_displacement.csv", d / "summary.json",
                                                   d / "progress.log", d / "baselines.json")])
    source = "committed" if not (ref or "").startswith("@runs/") else ("dashboard" if task else "run store")
    errs = [c["error"] for c in curves if c.get("error") is not None]
    peaks = [max(c["F"]) for c in curves if c.get("F")]
    rec = {"ref": ref, "kind": "score" if kind == "score_group" else kind, "title": _title(kind, d, ref, facts),
           "short": _short(d, ref or ""),
           "source": source, "status": _status(kind, d, task), "modified": stamp, "task": _task_brief(task),
           "part": facts.get("part"), "n_curves": len(curves), "peak_N": max(peaks) if peaks else None,
           "best_error": min(errs) if errs else None, "abs": str(d),
           "curves_brief": [{"name": c["name"], "label": c["label"], "points": len(c.get("F") or []),
                             "error": c.get("error"), "failed": c.get("failed")} for c in curves]}
    if kind == "crush":
        rec.update({k: facts.get(k) for k in ("travel_mm", "n_failed", "separated", "stop_reason")})
        # stopped or killed before its end, with the checkpoint of its last block: crush.py --resume continues it
        rec["resumable"] = rec["status"] == "stopped" and (d / "checkpoint.json").exists()
    if not full:
        return rec
    for c in curves:
        if "dir" in c:
            sub = c.pop("dir")
            c["ref"] = to_ref(sub, store_root)
            c["abs"] = str(sub)
    rec.update(curves=curves, facts=facts, files=_files(d, store_root),
               media=media_files(d, store_root), renders=[])
    for t in renders:
        out = Path(t["out_abs"]) if t.get("out_abs") else None
        items = media_files(out, store_root, skip_dirs=()) if out and out.is_dir() else []
        if any(m["kind"] == "video" for m in items):
            items = [m for m in items if m["kind"] != "image" or "_frames/" not in m["name"]]
        rec["renders"].append({"task": _task_brief(t), "media": items})
    log = d / "progress.log"
    if log.exists():
        rec["log_tail"] = _tail(log, 4000)
    return rec


def _tail(p: Path, n: int) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def list_runs(store_root: Path, tasks=None) -> list[dict]:
    by_out, renders = _task_maps(tasks)
    seen, out = set(), []
    roots = [Path(store_root) / s for s in STORE_SCAN] + [REPO_ROOT / r for r in COMMITTED_RUNS]
    for root in roots:
        for d, kind in scan(root):
            key = str(d.resolve())
            if key in seen or to_ref(d, store_root) is None:
                continue
            seen.add(key)
            try:
                out.append(run_record(d, kind, store_root, by_out.get(key), renders.get(key, ())))
            except Exception as e:  # noqa: BLE001 - one unreadable run must not hide the others
                out.append({"ref": to_ref(d, store_root), "kind": kind, "title": d.name, "status": "unreadable",
                            "error": f"{type(e).__name__}: {e}", "modified": _mtime(d), "curves_brief": []})
    for key, t in by_out.items():           # dashboard runs that have not written a curve yet
        if key in seen or t.get("tool") not in RUN_TOOLS:
            continue
        ref = to_ref(Path(key), store_root)
        if ref is None:
            continue
        out.append({"ref": ref, "kind": RUN_TOOLS[t["tool"]], "title": t.get("title"), "source": "dashboard",
                    "status": "running" if t.get("state") in ("queued", "running") else t.get("state"),
                    "modified": _mtime(Path(key)), "task": _task_brief(t), "curves_brief": [], "n_curves": 0,
                    "abs": key})
    out.sort(key=lambda r: -(r.get("modified") or 0))
    return out


def meshes(store_root: Path) -> list[dict]:
    """Crush meshes (mesh.msh + report.json or imperfection.json) under <runs>/nonlinear/meshes."""
    root = Path(store_root) / "nonlinear" / "meshes"
    out = []
    for m in sorted(root.glob("**/mesh.msh")) if root.is_dir() else []:
        d = m.parent
        if any(p in SKIP_DIRS - {"meshes"} for p in d.relative_to(root).parts):
            continue
        imp = _read_json(d / "imperfection.json") or {}
        rep = _read_json(d / "report.json") or {}
        if not rep and imp.get("source_mesh"):          # a moved mesh: the report of the mesh it was made from
            rep = _read_json(Path(imp["source_mesh"]).parent / "report.json") or {}
        crit = imp.get("critical_loads_N") or []
        out.append({"ref": to_ref(d, store_root), "abs": str(d), "name": d.name, "modified": _mtime(m),
                    "part": rep.get("part") or imp.get("part"), "system": rep.get("system"),
                    "n_nodes": rep.get("n_nodes"), "n_elements": rep.get("n_elements"),
                    "h_mm": round(rep["knobs"]["h"] * 1e3, 3) if isinstance(rep.get("knobs"), dict)
                    and rep["knobs"].get("h") else None,
                    "straight": (rep.get("knobs") or {}).get("straight_edges"),
                    "gates": {g["id"]: g["status"] for g in rep.get("gates") or []
                              if isinstance(g, dict) and g.get("status") in ("fail", "warn")},
                    "imperfection": {"amp_mm": imp.get("amplitude_mm"), "mode": imp.get("mode"),
                                     "platen": imp.get("platen"), "critical_load_N": crit[0] if crit else None}
                    if imp else None,
                    "image": to_ref(d / "mesh.png", store_root) if (d / "mesh.png").exists() else None})
    out.sort(key=lambda r: -r["modified"])
    return out


# --------------------------------------------------------------------------- capabilities


def capabilities() -> dict:
    from feacore.ui.offscreen import display_mode

    ccx = os.environ.get("FEACORE_CCX") or shutil.which("ccx")
    bundled = importlib.util.find_spec("imageio_ffmpeg") is not None
    ffmpeg = "system" if shutil.which("ffmpeg") else ("bundled (imageio-ffmpeg)" if bundled else None)
    disp = display_mode()
    pyvista = importlib.util.find_spec("pyvista") is not None
    render = {"display": "a display", "xvfb": "a virtual display (xvfb-run, software OpenGL)",
              "none": "EGL / OSMesa if VTK has them (no display and no xvfb-run)"}[disp]
    return {"ccx": ccx, "ffmpeg": ffmpeg, "display": disp, "pyvista": pyvista, "cpus": os.cpu_count(),
            "notes": {"ccx": "CalculiX runs every nonlinear analysis" if ccx else
                      "no ccx on this server: the nonlinear tools cannot run",
                      "render": f"pyvista renderers draw through {render}" if pyvista else "no pyvista: no 3D renders",
                      "rtx": "Omniverse RTX renders need the DGX Spark (Isaac Sim): committed media only"}}


# --------------------------------------------------------------------------- the committed gallery


def _strip(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _md_caption(md: Path) -> tuple[str | None, str | None]:
    """(title, caption) of a talk_plots figure note: '# name: "Title"' and its '**Caption.**' paragraph."""
    try:
        text = md.read_text()
    except OSError:
        return None, None
    m = re.match(r"#\s*[^:\n]+:\s*\"?([^\"\n]+)\"?", text)
    c = re.search(r"\*\*Caption\.\*\*\s*(.+?)(?:\n\s*\n|$)", text, re.S)
    return (m.group(1).strip() if m else None), (re.sub(r"\s+", " ", c.group(1)).strip() if c else None)


def _html_videos(page: Path) -> list[dict]:
    """Videos of a generated HTML page: the <h2> before each <video> is its title, the next <p class="cap"> its
    caption."""
    try:
        text = page.read_text()
    except OSError:
        return []
    out = []
    for m in re.finditer(r'<video src="([^"]+)"', text):
        before, after = text[: m.start()], text[m.end():]
        h2 = re.findall(r"<h2>(.*?)</h2>", before, re.S)
        cap = re.search(r'<p class="cap">(.*?)</p>', after, re.S)
        out.append({"src": (page.parent / m.group(1)).resolve(), "title": _strip(h2[-1]) if h2 else None,
                    "caption": _strip(cap.group(1)) if cap else None})
    return out


def _poster(video: Path) -> Path | None:
    for c in (video.with_name(video.stem + "_middle.png"), video.with_name(video.stem + ".jpg"),
              REPO_ROOT / "docs" / "gallery" / "crush" / (video.stem + ".jpg")):
        if c.exists():
            return c
    return None


def _item(p: Path, store_root: Path, title=None, caption=None) -> dict | None:
    ref = to_ref(p, store_root)
    if ref is None or not p.exists():
        return None
    poster = _poster(p) if kind_of(p) == "video" else None
    return {"ref": ref, "kind": kind_of(p), "title": title or p.stem.replace("_", " "), "caption": caption,
            "poster": to_ref(poster, store_root) if poster else None, "rtx": "_rtx" in p.stem,
            "bytes": p.stat().st_size}


def _readme_sections(readme: Path, skip=()) -> list[dict]:
    """'## ' sections of a gallery README: the text, and every picture / video it links, in order."""
    try:
        text = readme.read_text()
    except OSError:
        return []
    out = []
    for block in re.split(r"\n(?=## )", text)[1:]:
        lines = block.splitlines()
        title = lines[0].lstrip("# ").strip()
        if any(s in title.lower() for s in skip):
            continue
        prose = " ".join(x.strip() for x in lines[1:] if x.strip() and not x.lstrip().startswith(("|", "!["))
                         and not re.fullmatch(r"\[MP4\]\(.*?\)\s*(\(.*\))?\.?", x.strip()))
        links = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", block)
        out.append({"title": title, "text": _strip(prose) or None,
                    "files": [(readme.parent / x).resolve() for x in dict.fromkeys(links)]})
    return out


@lru_cache(maxsize=2)
def _gallery(stamp: float, store_root: str) -> dict:
    sr = Path(store_root)
    act3 = REPO_ROOT / "talk_plots" / "act3_real_data"
    sections = []

    figs = []
    for png in sorted(act3.glob("fig*.png")):
        title, cap = _md_caption(png.with_suffix(".md"))
        figs.append(_item(png, sr, title, cap))
    readme = (REPO_ROOT / "README.md").read_text() if (REPO_ROOT / "README.md").exists() else ""
    for src, alt in re.findall(r'<img src="(talk_plots/act3_real_data/photos/[^"]+)"[^>]*alt="([^"]*)"', readme):
        figs.append(_item(REPO_ROOT / src, sr, "the printed part after the test", alt))
    sections.append({"id": "act3", "title": "The real compression test against the simulation (Result 3)",
                     "text": "A 3D-printed C-bracket (abc_00000065, Formlabs Alpha Fast) crushed on an Instron, and "
                             "the simulation that learned the test conditions and was post-trained on it: from 33 % "
                             "to 2.0 % off the measured force curve.",
                     "source": "talk_plots/act3_real_data/README.md", "items": [x for x in figs if x]})

    vids = [_item(v["src"], sr, v["title"], v["caption"]) for v in _html_videos(act3 / "videos.html")]
    sections.append({"id": "act3_videos", "title": "Videos: the simulation against the real test",
                     "text": "Force vs crosshead travel filling in over the Instron curve (black); the part coloured "
                             "by plastic strain PEEQ, the same range in every video.",
                     "source": "talk_plots/act3_real_data/videos.html", "items": [x for x in vids if x]})

    brk = [_item(v["src"], sr, v["title"], v["caption"]) for v in _html_videos(act3 / "break" / "index.html")]
    t_title, _ = _md_caption(act3 / "break" / "break_table.md")
    brk += [_item(act3 / "break" / "break_table.png", sr, t_title or "predicted crack onset vs the real first crack",
                  "Travel and force at which each criterion fires, for each case, against the real first crack "
                  "(the test's peak, 4,755 N at 2.51 mm)."),
            _item(act3 / "break" / "after_crack_closeup.png", sr, "largest tension at the real first crack",
                  "After post-training, at 2.51 mm: the most stressed point is on the lower inner bore, where the "
                  "printed part cracked.")]
    sections.append({"id": "act3_break", "title": "Where and when the part cracks",
                     "text": "A crack-onset check on every CalculiX increment (no fracture model): the largest tension "
                             "reaching the resin's strength, or 85 % of it, against the real first crack.",
                     "source": "talk_plots/act3_real_data/break/index.html", "items": [x for x in brk if x]})

    ramp = TEST_DIR / "ccx_ramp"
    items = [_item(ramp / "compare.png", sr, "Instron vs CalculiX ramps",
                   "J2 plasticity + NLGEOM on a 41k-node P2 mesh, two calibrations of the dogbone. With a "
                   "frictionless platen the prongs splay and the part levels off at 2.8-2.9 kN; with a sticking "
                   "platen it carries 4.7-4.9 kN at the test's peak travel (test: 4.76 kN).")]
    items += [_item(p, sr, p.stem.replace("render_", "").replace("_", " "),
                    "Plastic strain and largest principal stress on the deformed part (true scale), at first yield, "
                    "at the test's peak load and at the end of the ramp.") for p in sorted(ramp.glob("render_*.png"))]
    sections.append({"id": "ccx_ramp", "title": "Elastoplastic ramp of the test (scripts/ccx_compression_ramp.py)",
                     "text": "Friction decides the strength: the platen that holds the prong tops carries the "
                             "test's peak load; yield starts at about 0.4 kN at the nut-pocket corners.",
                     "source": "ace/README.md", "items": [x for x in items if x]})

    base = REPO_ROOT / "results" / "real_data_finetune" / "baselines"
    bj = _read_json(base / "baselines.json") or {}
    nums = ", ".join(f"{r['label']} {r['mean_rel_err'] * 100:.1f} %" for r in (bj.get("baselines") or {}).values())
    items = [_item(base / "error_vs_nodes.png", sr, "error vs nodes", f"Mean error against the test: {nums}.")]
    for d in sorted(x for x in base.iterdir() if x.is_dir()) if base.is_dir() else []:
        lab = SCORE_LABELS.get(d.name, d.name)
        items += [_item(d / "curve.png", sr, f"{lab}: force vs crosshead", None),
                  _item(d / "peeq.png", sr, f"{lab}: plastic strain at the end", None)]
    sections.append({"id": "baselines", "title": "The agents' frozen scoring ramp on reference meshes",
                     "text": "Every agent's mesh goes through one frozen CalculiX ramp (sticking platens, seating gap "
                             "and load-train compliance fitted once), scored at 0.5-2.5 mm of crosshead travel "
                             "(scripts/nl_baselines.py).",
                     "source": "scripts/nl_baselines.py", "items": [x for x in items if x]})

    gal = REPO_ROOT / "docs" / "gallery"
    for i, s in enumerate(_readme_sections(gal / "README.md", skip=("mesh quality", "wingbox"))):
        items = [_item(f, sr) for f in s["files"] if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".mp4", ".gif")
                 and not (f.suffix.lower() == ".gif" and f.with_suffix(".mp4").exists())]
        items = [x for x in items if x]
        if items:
            sections.append({"id": f"gallery_{i}", "title": s["title"], "text": s["text"],
                             "source": "docs/gallery/README.md", "items": items})
    res = REPO_ROOT / "experiments" / "mesh_systems" / "RESULTS.png"
    if res.exists():
        sections.append({"id": "mesh_systems", "title": "Six mesh systems compared",
                         "text": "Stiffness agrees across the six systems within 0.8-1.3 %; crush peak force spreads "
                                 "20-31 % and single-point peak stress 42-88 %.",
                         "source": "experiments/mesh_systems/RESULTS.md", "items": [_item(res, sr, "comparison")]})
    return {"sections": sections}


def gallery(store_root: Path) -> dict:
    stamp = max(_mtime(p) for p in (REPO_ROOT / "docs" / "gallery" / "README.md",
                                    REPO_ROOT / "talk_plots" / "act3_real_data" / "README.md",
                                    REPO_ROOT / "talk_plots" / "act3_real_data" / "videos.html"))
    return _gallery(stamp, str(Path(store_root).resolve()))


# --------------------------------------------------------------------------- API


def router(store, tasks=None):
    from fastapi import APIRouter, HTTPException
    from starlette.concurrency import run_in_threadpool

    from feacore.runstore import jsonable

    r = APIRouter()
    root = Path(store.root)

    @r.get("/api/nonlinear/capabilities")
    def caps():
        return capabilities()

    @r.get("/api/nonlinear/runs")
    async def runs():
        items = await run_in_threadpool(list_runs, root, tasks)
        return jsonable({"runs": items, "store": str(root.resolve())})

    def _placeholder(ref: str, t: dict, d: Path | None) -> dict:
        """A dashboard run that has not written anything a run is recognised by (queued, just started, failed)."""
        return {"ref": ref, "kind": RUN_TOOLS.get(t.get("tool"), "run"), "title": t.get("title"),
                "status": "running" if t.get("state") in ("queued", "running") else t.get("state"),
                "task": _task_brief(t), "curves": [], "facts": {}, "media": [], "renders": [],
                "files": _files(d, root) if d is not None and d.is_dir() else [], "abs": t.get("out_abs"),
                "source": "dashboard", "part": (t.get("args") or {}).get("part")}

    @r.get("/api/nonlinear/run")
    async def run(ref: str):
        by_out, renders = _task_maps(tasks)
        try:
            d = resolve_ref(ref, root, "dir")
        except RefError as e:
            t = next((x for x in by_out.values() if x.get("out") == ref), None)
            if t is None:
                raise HTTPException(404, str(e)) from None
            return jsonable(_placeholder(ref, t, None))
        kind = detect(d)
        key = str(d.resolve())
        if kind is None:
            t = by_out.get(key)
            if not t:
                raise HTTPException(404, f"{ref!r} is not a nonlinear run")
            return jsonable(_placeholder(ref, t, d))
        # a scoring run's videos are rendered per mesh, from its sub-folders: they belong to the run too
        mine = [t for k, ts in renders.items() if k == key or k.startswith(key + os.sep) for t in ts]
        return jsonable(await run_in_threadpool(run_record, d, kind, root, by_out.get(key), mine, True))

    @r.get("/api/nonlinear/test")
    def test():
        try:
            return jsonable(test_curve())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"cannot read the test data: {type(e).__name__}: {e}") from None

    @r.get("/api/nonlinear/meshes")
    def mesh_list():
        return jsonable({"meshes": meshes(root)})

    @r.get("/api/nonlinear/gallery")
    async def gal():
        return jsonable(await run_in_threadpool(gallery, root))

    @r.get("/api/nonlinear/file")
    def file(ref: str, download: bool = False):
        try:
            return file_response(ref, root, download)
        except RefError as e:
            raise HTTPException(404, str(e)) from None

    return r
