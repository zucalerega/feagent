"""ACE folders in the dashboard (Agents section): training runs, exported swarms, inference results and
swarm-vs-single comparisons, read from what the ACE tools write. Read-only: work is started through the
task lane (feacore/ui/tools/agents.py).

Where it looks (nothing else is ever read):

    training, inference and comparison folders   ace_runs/<name> (RUN_ROOTS) and the output folder of
                                                 every Agents task (the task lane's records)
    exported swarms                              results/trained_swarms/<name>, swarms/<name>
                                                 (SWARM_ROOTS) and every ace.export_swarm output

A folder is named by its *key*: its path relative to the repository (``ace_runs/div_s8``), or
``@runs/<path>`` inside the run store. A key is resolved, then refused unless it is one of the folders
above; a symlink that leads out of its root is refused too.

API (mounted by feacore.ui.server; every path is under /api/, so the token middleware guards it)::

    GET /api/ace/catalog                       the loops' task catalogue and suites (training forms)
    GET /api/ace/runs                          training runs, newest first, each with a summary
    GET /api/ace/run?dir=KEY                   one run: scores per generation and problem, agents, files
    GET /api/ace/agent-run?dir=KEY&run_id=ID   one agent run: checks, report pictures, the context it saw,
                                               teacher feedback, reflection, organizer, trace, solves
    GET /api/ace/file?dir=KEY&path=REL         a picture or text file inside a known folder (&swarm=true:
                                               inside an exported swarm)
    GET /api/ace/report?dir=KEY                <run>/report.html (built by the ace.report tool)
    GET /api/ace/tracker?dir=KEY               the live tracker page (ace/dashboard.html) for the run,
    GET /api/ace/tracker/data?dir=KEY          which polls ace.watch.collect(run) here
    GET /api/ace/swarms, /api/ace/swarm?dir=   exported swarms (manifest, agents, notes)
    GET /api/ace/infers, /api/ace/infer?dir=   ace.infer results (predictions, consensus, pictures)
    GET /api/ace/compares, /api/ace/compare?dir=   ace.compare outputs (compare.md, compare.png)

The tracker page fetches its data with a plain fetch (no headers). When the server has a token, the
page is served with its data URL carrying the token the page itself was requested with.
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from feacore.runstore import RunStore, jsonable
from feacore.ui.tasks import REPO_ROOT, TaskRunner, resolve_out, tail

RUN_ROOTS = ["ace_runs"]                              # repo-relative, or @runs/...
SWARM_ROOTS = ["results/trained_swarms", "swarms"]
TRAIN_TOOLS = ("ace.swarm", "ace.single")
RUN_ID = re.compile(r"^g(\d+)-(.+)-(a\d+|big|\d+)$")   # g<gen>-<problem>-<agent a<k> | big | run index>
SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]{0,200}$")
FILE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
              ".mp4": "video/mp4", ".md": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
              ".log": "text/plain; charset=utf-8", ".jsonl": "text/plain; charset=utf-8",
              ".json": "application/json"}
MAX_FILE = 50_000_000
MAX_TEXT = 400_000            # characters of a markdown or context file returned inline
ACTIVE_S = 600                # a folder written this recently, with no live task, is "active"
TASK_KEYS = ("id", "tool", "title", "state", "cancel", "created_at", "started_at", "finished_at", "error", "out")
# the served HTML pages (report, tracker) run only their own inline code and talk only to this server
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'")


# --------------------------------------------------------------------------- small helpers


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _json_line(line: str):
    try:
        return json.loads(line)
    except ValueError:
        return None


def _text(path: Path | None, limit: int = MAX_TEXT) -> str | None:
    if path is None:
        return None
    try:
        s = path.read_text(errors="replace")
    except OSError:
        return None
    return s if len(s) <= limit else s[:limit] + "\n\n[... cut: the file is longer ...]\n"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _iso(t: float | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(xs):
    xs = [x for x in xs if _num(x)]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if _num(x)]
    return statistics.median(xs) if xs else None


def _last_line(err) -> str | None:
    if not err:
        return None
    lines = [x for x in str(err).strip().splitlines() if x.strip()]
    return (lines[-1] if lines else "error")[:400]


def _gen_numbers(d: Path) -> list[int]:
    try:
        return sorted(int(g.name[3:]) for g in d.iterdir() if g.is_dir() and re.fullmatch(r"gen\d+", g.name))
    except OSError:
        return []


def _inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def _task_info(task: dict | None) -> dict | None:
    return {k: task.get(k) for k in TASK_KEYS} if task else None


class _FileCache:
    """Values computed from one file, recomputed only when its mtime or size changes. Run records
    are large (traces) and a finished run's never change, so lists and pages stay fast."""

    def __init__(self, size: int = 20000):
        self._d: OrderedDict = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, path: Path, build: Callable[[Path], Any], default=None):
        try:
            st = path.stat()
        except OSError:
            return default
        stamp, k = (st.st_mtime_ns, st.st_size), str(path)
        with self._lock:
            hit = self._d.get(k)
            if hit is not None and hit[0] == stamp:
                self._d.move_to_end(k)
                return hit[1]
        value = build(path)
        with self._lock:
            self._d[k] = (stamp, value)
            self._d.move_to_end(k)
            while len(self._d) > self._size:
                self._d.popitem(last=False)
        return value


# --------------------------------------------------------------------------- reading ACE files


def run_brief(path: Path) -> dict | None:
    """The numbers of one agent run record (gen<g>/runs/<run_id>.json), without its trace."""
    r = _json(path)
    return brief_of(r, path.stem) if isinstance(r, dict) else None


def brief_of(r: dict, name: str) -> dict:
    c = r.get("checks") or {}
    rid = str(r.get("run_id") or name)
    m = RUN_ID.match(rid)
    err = r.get("error")
    solves = r.get("solves") or []
    score = None if err else c.get("score")
    base = None if err else c.get("base_score", c.get("score"))
    gen = r.get("generation")
    return {
        "run_id": rid, "problem_id": r.get("problem_id"),
        "generation": gen if gen is not None else (int(m.group(1)) if m else None),
        "agent": m.group(3) if m and not m.group(3).isdigit() else None,
        "crashed": bool(err), "error": _last_line(err), "score": score, "base": base,
        "bonus": score - base if _num(score) and _num(base) else None,
        "dofs": c.get("dofs"), "nodes": c.get("nodes"), "order": c.get("order"),
        "rel_error": c.get("rel_error"), "field_error": c.get("field_error"), "probe_error": c.get("probe_error"),
        "accuracy": c.get("accuracy"), "efficiency": c.get("efficiency"),
        "meets_tol": c.get("meets_tol"), "valid": c.get("valid"), "failed_gate": c.get("failed_gate"),
        "turns": r.get("turns"), "stop": r.get("stop"), "n_solves": len(solves),
        "n_ok": sum(bool((s.get("result") or {}).get("ok")) for s in solves), "submitted": r.get("submitted"),
        "cited": len(r.get("cited") or []), "usd": sum(float(u.get("usd") or 0) for u in r.get("usage") or []),
        "ui_submission_id": r.get("ui_submission_id"),
        "ui_run_ids": [s["ui_run_id"] for s in solves if s.get("ui_run_id")],
    }


def costs(path: Path) -> dict:
    """cost.jsonl: total $, calls, $ by role and by generation (as ace.compare.gen_costs: a call with a
    run id g<g>-... is generation g; one without (teacher, organizer) belongs to the generation before)."""
    recs = [r for line in (_text(path, 10**9) or "").splitlines() if line.strip() and (r := _json_line(line))]
    by_role: dict[str, float] = {}
    by_gen: dict[int, float] = {}
    gen = 0
    for r in sorted(recs, key=lambda r: r.get("t") or 0):
        usd = float(r.get("usd") or 0)
        by_role[str(r.get("role"))] = by_role.get(str(r.get("role")), 0.0) + usd
        m = re.match(r"^g(\d+)-", str(r.get("run_id") or ""))
        if m:
            gen = max(gen, int(m.group(1)))
        by_gen[gen] = by_gen.get(gen, 0.0) + usd
    ts = [r["t"] for r in recs if _num(r.get("t"))]
    return {"usd": sum(by_role.values()), "calls": len(recs), "by_role": by_role, "by_gen": by_gen,
            "first_t": min(ts, default=None), "last_t": max(ts, default=None)}


NO_COSTS = {"usd": 0.0, "calls": 0, "by_role": {}, "by_gen": {}, "first_t": None, "last_t": None}


def group_stats(briefs: list[dict]) -> dict:
    """Scores of a set of agent runs (e.g. one generation on one problem); crashed runs are not scored."""
    ok = [b for b in briefs if b["score"] is not None]
    s = [b["score"] for b in ok]
    base = [b["base"] for b in ok if b["base"] is not None]
    return {"n": len(briefs), "n_scored": len(ok), "mean": _mean(s), "median": _median(s),
            "best": max(s, default=None), "min": min(s, default=None),
            "base_mean": _mean(base), "base_best": max(base, default=None),
            "within_tol": sum(b["meets_tol"] is True for b in ok),
            "failed": sum(bool(b["failed_gate"]) or b["valid"] is False for b in ok),
            "crashed": sum(b["crashed"] for b in briefs),
            "median_dofs": _median([b["dofs"] for b in ok if b["dofs"]])}


def note_changes(org: dict) -> dict[str, list]:
    """What the organizer added, edited and removed, from its before / after note lists (a kept or
    edited note keeps its id)."""
    before = {n.get("id"): n for n in org.get("before") or [] if n.get("id")}
    added, edited, kept = [], [], set()
    for n in org.get("after") or []:
        old = before.get(n.get("id"))
        if old is None:
            added.append(n)
        else:
            kept.add(n["id"])
            if (old.get("content"), old.get("section")) != (n.get("content"), n.get("section")):
                edited.append({**n, "was": old.get("content")})
    return {"added": added, "edited": edited, "removed": [n for i, n in before.items() if i not in kept]}


def trace_steps(trace: list | None, limit: int = 400, chars: int = 700) -> list[dict]:
    """The agent's steps, shortened: tool calls with their arguments, results, text and thinking."""
    out = []
    for t in (trace or [])[:limit]:
        kind = t.get("type")
        if kind == "tool_call":
            inp = dict(t["input"]) if isinstance(t.get("input"), dict) else {"input": t.get("input")}
            using = inp.pop("using", None) or []
            out.append({"type": "call", "name": t.get("name"), "turn": t.get("turn"), "using": using,
                        "input": json.dumps(inp, default=str)[:chars]})
        elif kind == "tool_result":
            res = t.get("result")
            out.append({"type": "result", "name": t.get("name"), "turn": t.get("turn"),
                        "is_error": t.get("is_error") in (True, "True", "true"),
                        "text": (res if isinstance(res, str) else json.dumps(res, default=str))[:chars]})
        elif kind in ("text", "thinking") and t.get("text"):
            out.append({"type": kind, "turn": t.get("turn"), "text": str(t["text"])[:chars]})
    if len(trace or []) > limit:
        out.append({"type": "text", "text": f"[{len(trace) - limit} more steps not shown]"})
    return out


def solve_rows(run: dict) -> list[dict]:
    """Each solve of a run, the scored one marked (the submitted one, else the last successful)."""
    solves = run.get("solves") or []
    k = run.get("submitted")
    if not (isinstance(k, int) and 0 <= k < len(solves) and (solves[k].get("result") or {}).get("ok")):
        k = next((j for j in reversed(range(len(solves))) if (solves[j].get("result") or {}).get("ok")), None)
    out = []
    for i, s in enumerate(solves):
        res, mesh = s.get("result") or {}, s.get("mesh") or {}
        out.append({"index": i + 1, "ok": bool(res.get("ok")), "qoi": res.get("qoi"), "value": res.get("value"),
                    "error": _last_line(res.get("error")), "scored": i == k, "ui_run_id": s.get("ui_run_id"),
                    "dofs": (res.get("info") or {}).get("n_dofs"),
                    "mesh": {"nodes": mesh.get("nodes"), "elements": mesh.get("elements"), "order": mesh.get("order"),
                             "h": mesh.get("h_used"), "request": mesh.get("request")}})
    return out


def _notes(pb: dict | None, cited: set, prev_ids: set | None = None) -> list[dict] | None:
    if not isinstance(pb, dict):
        return None
    return [{"id": b.get("id"), "section": b.get("section"), "content": b.get("content"), "kind": b.get("kind"),
             "source": b.get("source"), "cited": b.get("id") in cited, "helpful": b.get("helpful"),
             "harmful": b.get("harmful"), "new": prev_ids is not None and b.get("id") not in prev_ids}
            for b in pb.get("bullets") or []]


# --------------------------------------------------------------------------- the folders


class AceRuns:
    """Finds the ACE folders and reads them (one per server; see the module docstring)."""

    def __init__(self, store: RunStore, runner: TaskRunner):
        self.store = store
        self.runner = runner
        self.briefs = _FileCache()
        self.small = _FileCache(4000)        # cost logs, budget flags of task logs
        self._tracker: OrderedDict = OrderedDict()
        self._tracker_lock = threading.Lock()

    # ------------------------------------------------------------------ keys and folders
    def key_of(self, path: Path) -> str:
        p = path.resolve()
        if _inside(p, REPO_ROOT.resolve()):
            return p.relative_to(REPO_ROOT.resolve()).as_posix()
        return "@runs/" + p.relative_to(self.store.root.resolve()).as_posix()

    def _key_path(self, key: str) -> Path:
        if not isinstance(key, str) or not key or len(key) > 600 or "\\" in key or key.startswith("/") \
                or any(ord(c) < 32 for c in key):
            raise KeyError(key)
        in_store = key.startswith("@runs/")
        parts = (key[len("@runs/"):] if in_store else key).split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise KeyError(key)
        return (self.store.root if in_store else REPO_ROOT).joinpath(*parts)

    def _tasks(self) -> list[dict]:
        return self.runner.tasks.list(group="agents", limit=10_000)

    def folders(self, swarms: bool = False) -> dict[Path, dict]:
        """{resolved folder: {"key", "task"}}: the children of the roots and the output folders of the
        Agents tasks (the newest task per folder), each kept only if it stays inside its root."""
        found: dict[Path, dict] = {}
        for name in SWARM_ROOTS if swarms else RUN_ROOTS:
            root = resolve_out(name, self.store)
            real = root.resolve()
            try:
                children = sorted(root.iterdir()) if root.is_dir() else []
            except OSError:
                children = []
            for c in children:
                if c.is_dir() and SAFE_NAME.match(c.name) and _inside(c.resolve(), real):
                    found.setdefault(c.resolve(), {"key": f"{name.rstrip('/')}/{c.name}", "task": None})
        allowed = [REPO_ROOT.resolve(), self.store.root.resolve(),
                   *(resolve_out(n, self.store).resolve() for n in SWARM_ROOTS + RUN_ROOTS)]
        for t in sorted(self._tasks(), key=lambda t: t.get("created_at") or ""):   # the newest task wins
            if not t.get("out_abs") or (t.get("tool") == "ace.export_swarm") != swarms:
                continue
            p = Path(t["out_abs"]).resolve()
            if any(_inside(p, r) for r in allowed):
                key = found.get(p, {}).get("key") or self.key_of(p)
                found[p] = {"key": key, "task": t}
        return found

    def folder(self, key: str, swarms: bool = False) -> tuple[Path, dict]:
        """(folder, its entry) for a key, or KeyError unless it is one of the known folders."""
        p = self._key_path(key).resolve()
        found = self.folders(swarms)
        if p not in found:
            raise KeyError(key)
        return p, found[p]

    @staticmethod
    def file_in(d: Path, rel: str) -> Path:
        """A picture or text file inside folder d, named by a relative path; KeyError otherwise."""
        if not rel or len(rel) > 600 or "\\" in rel or rel.startswith("/") or any(ord(c) < 32 for c in rel):
            raise KeyError(rel)
        parts = rel.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise KeyError(rel)
        f = d.joinpath(*parts).resolve()
        if not _inside(f, d) or not f.is_file() or f.suffix.lower() not in FILE_TYPES:
            raise KeyError(rel)
        return f

    def tasks_for(self, d: Path) -> list[dict]:
        """Agents tasks that wrote folder d or read it (a report, an export, a comparison, inference)."""
        out = []
        for t in self._tasks():
            for v in [t.get("out_abs"), *[v for v in (t.get("args") or {}).values() if isinstance(v, str)]]:
                try:
                    if v and resolve_out(v, self.store).resolve() == d:
                        out.append(_task_info(t))
                        break
                except (OSError, ValueError):
                    continue
        return out

    def _log(self, d: Path, task: dict | None) -> Path | None:
        """The run's console output: its task's log, else run.log or <name>.log next to it (ace.watch)."""
        if task:
            try:
                p = self.runner.tasks.log_path(task["id"])
                if p.exists():
                    return p
            except KeyError:
                pass
        return next((p for p in (d / "run.log", d.parent / f"{d.name}.log") if p.exists()), None)

    # ------------------------------------------------------------------ training runs
    def briefs_of(self, d: Path) -> list[dict]:
        out = []
        for g in _gen_numbers(d):
            for f in sorted((d / f"gen{g}" / "runs").glob("*.json")):
                b = self.briefs.get(f, run_brief)
                if b is not None:
                    out.append(dict(b))
        return out

    @staticmethod
    def kind(d: Path, task: dict | None) -> str | None:
        """'swarm' (ace.run_loop), 'single' (ace.single_loop), or None (not a training run)."""
        if task and task.get("tool") in TRAIN_TOOLS:
            return "single" if task["tool"] == "ace.single" else "swarm"
        if (d / "generations.jsonl").exists() or any(d.glob("gen*/context/big.md")):
            return "single"
        if (d / "run_config.json").exists() or (d / "gen0").is_dir():
            return "swarm"
        return None

    @staticmethod
    def _status(d: Path, task: dict | None, updated: float) -> str:
        state = (task or {}).get("state")
        if state in ("queued", "running"):
            return "stopping" if state == "running" and task.get("cancel") == "stop" else state
        if state in ("failed", "interrupted", "cancelled"):
            return state
        if (d / "summary.md").exists():
            return "stopped" if (d / "STOP").exists() else "finished"
        return "active" if time.time() - updated < ACTIVE_S else "incomplete"

    @staticmethod
    def _agents(d: Path, kind: str | None, briefs: list[dict]) -> list[str]:
        if kind == "single":
            return ["big"]
        names = {b["agent"] for b in briefs if b["agent"]}
        names |= {f"a{m.group(1)}" for f in (d / "private").glob("agent*_gen0.json")
                  if (m := re.fullmatch(r"agent(\d+)_gen0", f.stem))}
        return sorted(names, key=lambda a: (len(a), a))

    def run_summary(self, d: Path, entry: dict, briefs: list[dict] | None = None) -> dict:
        task = entry.get("task")
        briefs = self.briefs_of(d) if briefs is None else briefs
        kind = self.kind(d, task)
        gens = _gen_numbers(d)
        cfg = _json(d / "run_config.json", {}) or {}
        args = (task or {}).get("args") or {}
        seen = sorted({b["problem_id"] for b in briefs if b["problem_id"]})
        train = cfg.get("train") or [p for p in (args.get("problems") or "").split(",") if p] or seen
        holdout = cfg.get("holdout") or []
        c = self.small.get(d / "cost.jsonl", costs, NO_COSTS)
        scored = [b for b in briefs if b["score"] is not None and b["generation"] is not None]
        last_gen = max((b["generation"] for b in scored), default=None)
        first_gen = min((b["generation"] for b in scored), default=None)
        files = [d, d / "cost.jsonl", d / "summary.md", *[d / f"gen{g}" for g in gens[-1:]],
                 *[d / f"gen{g}" / "runs" for g in gens[-1:]]]
        updated = max([_mtime(f) for f in files] + [c.get("last_t") or 0])
        log = self._log(d, task)
        return {
            "key": entry["key"], "name": d.name, "loop_id": d.name, "kind": kind,
            "deep": kind == "single" or (d / "gen0" / "context").is_dir() or any(d.glob("gen*/reports")),
            "status": self._status(d, task, updated),
            "budget_stop": bool(log and self.small.get(log, lambda p: "BUDGET REACHED" in tail(p, 400_000), False)),
            "generations": len(gens), "generations_scored": len({b["generation"] for b in scored}),
            "planned": args.get("generations"), "planned_agents": args.get("agents"), "train": train, "holdout": holdout,
            "agents": self._agents(d, kind, briefs), "n_runs": len(briefs), "n_crashed": sum(b["crashed"] for b in briefs),
            "usd": c["usd"], "calls": c["calls"], "mock": bool(args.get("mock")) or d.name.endswith("-mock"),
            "last": {"generation": last_gen, **group_stats([b for b in scored if b["generation"] == last_gen])}
            if scored else None,
            "first": {"generation": first_gen, **group_stats([b for b in scored if b["generation"] == first_gen])}
            if scored else None,
            "started_at": (task or {}).get("created_at") or _iso(c.get("first_t")), "updated_at": _iso(updated),
            "task": _task_info(task), "env": (task or {}).get("env") or {},
            "has": {"summary": (d / "summary.md").exists(), "scores_png": (d / "scores.png").exists(),
                    "report": (d / "report.html").exists(), "compare": (d / "compare.md").exists(),
                    "private": any((d / "private").glob("agent*_gen*.json")), "stop_file": (d / "STOP").exists()},
        }

    def list_runs(self) -> list[dict]:
        out = []
        for d, entry in self.folders().items():
            if self.kind(d, entry["task"]) is None:
                continue
            try:
                out.append(self.run_summary(d, entry))
            except Exception as e:  # noqa: BLE001 - one unreadable folder must not hide the others
                out.append({"key": entry["key"], "name": d.name, "status": "unreadable",
                            "error": f"{type(e).__name__}: {e}"})
        out.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return out

    def run_detail(self, d: Path, entry: dict) -> dict:
        briefs = self.briefs_of(d)
        summary = self.run_summary(d, entry, briefs)
        problems = list(summary["train"]) + [p for p in summary["holdout"] if p not in summary["train"]]
        problems += sorted({b["problem_id"] for b in briefs if b["problem_id"] and b["problem_id"] not in problems})
        for b in briefs:     # rank within its generation and problem (1 = best)
            group = [x["score"] for x in briefs if x["generation"] == b["generation"]
                     and x["problem_id"] == b["problem_id"] and x["score"] is not None]
            b["rank"] = 1 + sum(s > b["score"] for s in group) if b["score"] is not None else None
            b["of"] = len(group)
        personas = _json(d / "personas.json", {}) or {}
        c = self.small.get(d / "cost.jsonl", costs, NO_COSTS)
        return {
            "summary": summary,
            "problems": [{"id": p, "split": "holdout" if p in summary["holdout"] else "train"} for p in problems],
            "generations": self._generations(d, briefs, problems, c), "agents": self._agent_rows(summary, briefs, personas),
            "runs": briefs, "score_label": "base score" if any(b["bonus"] for b in briefs) else "score",
            "cost": {k: c[k] for k in ("usd", "calls", "by_role")}, "summary_md": _text(d / "summary.md"),
            "compare_md": _text(d / "compare.md"), "init": _json(d / "init.json"), "personas": personas,
            "bridge": any(b["ui_submission_id"] for b in briefs), "tasks": self.tasks_for(d),
        }

    @staticmethod
    def _generations(d: Path, briefs: list[dict], problems: list[str], c: dict) -> list[dict]:
        """Per generation: scores per problem and overall, $, phase timings, shared notes, groups."""
        phases = {r.get("gen"): {k: v for k, v in r.items() if k not in ("gen", "what")}
                  for line in (_text(d / "solver.jsonl", 10**8) or "").splitlines()
                  if (r := _json_line(line)) and r.get("what") == "phase"}
        times = {r.get("gen"): r for line in (_text(d / "generations.jsonl") or "").splitlines() if (r := _json_line(line))}
        out, cum = [], 0.0
        for g in _gen_numbers(d):
            gb = [b for b in briefs if b["generation"] == g]
            usd = c["by_gen"].get(g, 0.0)
            cum += usd
            work = d / f"gen{g}" / "work"
            started = sum(1 for _ in work.iterdir()) if work.is_dir() else 0
            out.append({
                "gen": g, "all": group_stats(gb), "usd": usd, "cum_usd": cum, "phase": phases.get(g),
                "problems": {p: group_stats([b for b in gb if b["problem_id"] == p]) for p in problems
                             if any(b["problem_id"] == p for b in gb)},
                "seconds": (times.get(g) or {}).get("seconds"), "in_progress": max(0, started - len(gb)),
                "shared_bullets": len((_json(d / f"playbook_gen{g}.json", {}) or {}).get("bullets") or []),
                "groups": _json(d / f"gen{g}" / "groups.json"),
            })
        return out

    @staticmethod
    def _agent_rows(summary: dict, briefs: list[dict], personas: dict) -> list[dict]:
        """Each persistent agent's runs by generation (the score breakdown per agent)."""
        rows = []
        for a in summary["agents"]:
            mine = sorted((b for b in briefs if b["agent"] == a), key=lambda b: (b["generation"] or 0, b["problem_id"] or ""))
            per_gen: dict[str, list] = {}
            for b in mine:
                per_gen.setdefault(str(b["generation"]), []).append(
                    {k: b[k] for k in ("run_id", "problem_id", "score", "base", "bonus", "rank", "of", "crashed",
                                       "failed_gate", "dofs", "rel_error", "field_error", "meets_tol")})
            rows.append({"agent": a, "persona": personas.get(a), "generations": per_gen,
                         "best": max((b["score"] for b in mine if b["score"] is not None), default=None),
                         "mean": _mean([b["score"] for b in mine])})
        return rows

    # ------------------------------------------------------------------ one agent run
    def agent_run(self, d: Path, run_id: str) -> dict:
        """Everything about one agent run, for the run page's agent view (what the old Streamlit agent
        tracker showed): its numbers, the run report's pictures, the context it saw and what changed in
        it, the notes as stored, the teacher's feedback, its own reflection, the organizer's changes,
        its tool calls and its solves (with the run-store ids the bridge recorded)."""
        m = RUN_ID.match(run_id or "")
        if not m or not SAFE_NAME.match(run_id):
            raise KeyError(run_id)
        g, pid, agent = int(m.group(1)), m.group(2), m.group(3)
        named = not agent.isdigit()                       # a persistent agent (a<k>, big), not a run index
        gd = d / f"gen{g}"
        rec = _json(gd / "runs" / f"{run_id}.json")
        if not isinstance(rec, dict):
            raise KeyError(run_id)
        cited = set(rec.get("cited") or [])
        k = None if not named else 0 if agent == "big" else int(agent[1:])
        prev_private = _json(d / "private" / f"agent{k}_gen{g - 1}.json") if k is not None and g > 0 else None
        prev_ids = {b.get("id") for b in (prev_private or {}).get("bullets") or []} if prev_private else None
        shared = _json(d / f"playbook_{pid}_gen{g}.json") or _json(d / f"playbook_gen{g}.json")
        return jsonable({
            "brief": brief_of(rec, run_id), "checks": rec.get("checks") or {},
            "error": rec.get("error"), "report": self._report(d, gd, run_id),
            "context": self._context(d, g, pid, agent, rec) if named else self._plain_context(rec),
            "notes": {"shared": _notes(shared, cited),
                      "private": _notes(_json(d / "private" / f"agent{k}_gen{g}.json"), cited, prev_ids)
                      if k is not None else None},
            **self._teacher(gd, pid, run_id, agent if named else None),
            "reflection": _json(gd / "reflections" / f"{run_id}.json"),
            "organizer": self._organizer(gd, agent) if named else None,
            "context_history": self._context_history(d, agent) if named else [],
            "trace": trace_steps(rec.get("trace")), "solves": solve_rows(rec),
            "meshes": [{x: mm.get(x) for x in ("index", "ok", "nodes", "elements", "order", "min_quality", "h_used", "error")}
                       for mm in rec.get("meshes") or []],
        })

    @staticmethod
    def _report(d: Path, gd: Path, run_id: str) -> dict | None:
        """The run report (--deep): its numbers and pictures, as paths inside the run folder. Its image
        paths were written relative to wherever the loop ran, so they are looked up by name here."""
        rep = _json(gd / "reports" / f"{run_id}.json")
        images = {}
        listed = rep.get("images") if isinstance(rep, dict) else None
        for name, p in (listed if isinstance(listed, dict) else {}).items():
            cand = gd / "reports" / Path(str(p)).name
            if cand.is_file():
                images[str(name)] = cand.relative_to(d).as_posix()
        if not images:
            for p in sorted((gd / "reports").glob(f"{run_id}_*.png")):
                images[p.stem[len(run_id) + 1:]] = p.relative_to(d).as_posix()
        if not isinstance(rep, dict) and not images:
            return None
        return {"numbers": (rep or {}).get("numbers") if isinstance(rep, dict) else None, "images": images}

    @staticmethod
    def _context_file(gd: Path, agent: str, pid: str) -> Path | None:
        """--deep: gen<g>/context/<agent>.md (<agent>-<problem>.md on a held-out problem)."""
        return next((p for p in (gd / "context" / f"{agent}-{pid}.md", gd / "context" / f"{agent}.md") if p.exists()), None)

    def _context(self, d: Path, g: int, pid: str, agent: str, rec: dict) -> dict | None:
        """The context the agent saw, and the diff against its previous generation's."""
        f = self._context_file(d / f"gen{g}", agent, pid)
        text = _text(f) if f else rec.get("playbook_text")
        if text is None:
            return None
        prev_g = prev = None
        for x in reversed([x for x in _gen_numbers(d) if x < g]):
            if f is not None:
                pf = self._context_file(d / f"gen{x}", agent, pid)
                if pf is not None:
                    prev_g, prev = x, _text(pf)
                    break
            else:                         # the basic loop: the context is in the run record
                pr = next(iter(sorted((d / f"gen{x}" / "runs").glob(f"g{x}-*-{agent}.json"))), None)
                old = _json(pr) if pr else None
                if isinstance(old, dict) and old.get("playbook_text") is not None:
                    prev_g, prev = x, old["playbook_text"]
                    break
        diff = None
        if prev is not None and prev != text:   # the whole context with +/- marks (the header lines dropped)
            diff = list(difflib.unified_diff(prev.splitlines(), text.splitlines(), lineterm="", n=10**6))[3:]
        return {"text": text, "prev_gen": prev_g, "diff": diff, "added": sum(x.startswith("+") for x in diff or []),
                "removed": sum(x.startswith("-") for x in diff or [])}

    @staticmethod
    def _plain_context(rec: dict) -> dict | None:
        text = rec.get("playbook_text")
        return {"text": text, "prev_gen": None, "diff": None, "added": 0, "removed": 0} if text is not None else None

    @staticmethod
    def _teacher(gd: Path, pid: str, run_id: str, agent: str | None) -> dict:
        """The deep teacher's feedback (teacher_<task>.json), or the basic loop's diagnoser
        (diagnosis_<task>.json): advice to this agent, lessons for everyone, this run's diagnosis."""
        out: dict = {"teacher": None, "teacher_task": None, "diagnosis": None}
        teach = _json(gd / f"teacher_{pid}.json")
        if isinstance(teach, dict):
            out["teacher"] = next((a for a in teach.get("per_agent") or []
                                   if a.get("run_id") == run_id or (agent and a.get("agent") == agent)), None)
            out["teacher_task"] = {k: teach.get(k) for k in ("task_summary", "setup_warnings", "highlights",
                                                             "shared_lessons", "contested_regions")}
        diag = _json(gd / f"diagnosis_{pid}.json")
        if isinstance(diag, dict):
            dg = diag.get("diagnosis") or {}
            out["diagnosis"] = {"advice": [a for a in dg.get("agent_advice") or [] if agent and a.get("agent") == agent],
                                "lessons": dg.get("lessons") or [], "divergence": dg.get("divergence"),
                                "this_run": next((x for x in dg.get("runs") or [] if x.get("run_id") == run_id), None),
                                "tool_calls": diag.get("tool_calls"), "reruns": len(diag.get("reruns") or [])}
        return out

    @staticmethod
    def _organizer(gd: Path, agent: str) -> dict | None:
        org = _json(gd / "organizer" / f"{agent}.json")
        if not isinstance(org, dict):
            return None
        return {"before": len(org.get("before") or []), "after": len(org.get("after") or []),
                "incoming": org.get("incoming") or [], "error": org.get("error"), **note_changes(org)}

    @staticmethod
    def _context_history(d: Path, agent: str) -> list[dict]:
        """Per generation: the size of the context the agent saw and the organizer's note changes after it."""
        rows = []
        for x in _gen_numbers(d):
            cf = d / f"gen{x}" / "context" / f"{agent}.md"
            org = _json(d / f"gen{x}" / "organizer" / f"{agent}.json")
            if not cf.exists() and not isinstance(org, dict):
                continue
            row: dict = {"gen": x, "context_chars": len(_text(cf) or "") if cf.exists() else None}
            if isinstance(org, dict):
                row.update({key: len(v) for key, v in note_changes(org).items()})
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ live tracker (ace.watch)
    def tracker_data(self, d: Path, entry: dict) -> dict:
        """ace.watch.collect for the run, cached until something in the folder changes."""
        log = self._log(d, entry.get("task"))
        files = [d / "cost.jsonl", d / "summary.md", *([log] if log else []), *d.glob("gen*/runs/*.json"),
                 *d.glob("gen*/work/*")]
        stamp = (max((_mtime(f) for f in files), default=0.0), len(files))
        with self._tracker_lock:
            hit = self._tracker.get(d)
            if hit is not None and hit[0] == stamp:
                return hit[1]
        from ace import watch

        data = watch.collect(d, log=log)
        if not (d / "run_config.json").exists():   # the single-agent loop writes none: its own tasks
            seen = sorted({r["problem"] for g in data["generations"] for r in g["runs"] if r.get("problem")})
            if seen:
                data.update(problems=seen, train=seen, holdout=[])
        data = jsonable(data)
        with self._tracker_lock:
            self._tracker[d] = (stamp, data)
            while len(self._tracker) > 16:
                self._tracker.popitem(last=False)
        return data

    # ------------------------------------------------------------------ exported swarms
    def swarm_summary(self, d: Path, entry: dict, runs: dict | None = None) -> dict | None:
        task = entry.get("task")
        man = _json(d / "manifest.json")
        if not isinstance(man, dict):
            if task and task.get("state") in ("queued", "running", "failed", "interrupted"):
                return {"key": entry["key"], "name": d.name, "status": task["state"], "agents": [], "tasks": [],
                        "personas": {}, "task": _task_info(task)}
            return None
        src = Path(str(man.get("source") or ""))
        source_key = None
        try:
            if runs is not None and src.is_absolute() and src.resolve() in runs:
                source_key = runs[src.resolve()]["key"]
        except OSError:
            pass
        return {"key": entry["key"], "name": d.name, "root": entry["key"].rsplit("/", 1)[0], "status": "ready",
                "run": man.get("run"), "kind": man.get("kind"), "generation": man.get("generation"),
                "tasks": man.get("tasks") or [], "agents": man.get("agents") or [],
                "personas": _json(d / "personas.json", {}) or {}, "models": man.get("models") or {},
                "git_commit": man.get("git_commit"), "exported_at": man.get("exported_at"),
                "source_key": source_key, "task": _task_info(task)}

    def list_swarms(self) -> list[dict]:
        runs = self.folders()
        out = [s for d, e in self.folders(swarms=True).items() if (s := self.swarm_summary(d, e, runs))]
        return sorted(out, key=lambda s: (s.get("exported_at") or "", s["name"]), reverse=True)

    def swarm_detail(self, d: Path, entry: dict) -> dict:
        s = self.swarm_summary(d, entry, self.folders())
        if s is None:
            raise KeyError(d.name)
        agents = []
        for a in s["agents"]:
            if not SAFE_NAME.match(str(a)):
                continue
            ctx = d / "context" / f"{a}.md"
            agents.append({"agent": a, "persona": s["personas"].get(a),
                           "notes": len((_json(d / "private" / f"{a}.json") or {}).get("bullets") or []),
                           "context": f"context/{a}.md" if ctx.is_file() else None,
                           "context_chars": len(_text(ctx) or "") if ctx.is_file() else None})
        shared = _json(d / "playbook.json") or {}
        return {**s, "agent_list": agents,
                "shared": [{"id": b.get("id"), "section": b.get("section"), "content": b.get("content")}
                           for b in shared.get("bullets") or []],
                "playbooks": {p.stem: len((_json(p) or {}).get("bullets") or []) for p in sorted((d / "playbooks").glob("*.json"))},
                "tasks_using": self.tasks_for(d)}

    # ------------------------------------------------------------------ inference results
    def infer_summary(self, d: Path, entry: dict) -> dict | None:
        task = entry.get("task")
        res = _json(d / "results.json")
        if not isinstance(res, dict):
            if not (task and task.get("tool") == "ace.infer"):
                return None
            args = task.get("args") or {}
            return {"key": entry["key"], "name": d.name, "status": task.get("state"), "problem": args.get("problem"),
                    "swarm": {"run": Path(str(args.get("swarm") or "")).name}, "n_agents": None, "n_ok": None,
                    "updated_at": task.get("created_at"), "task": _task_info(task)}
        agents = res.get("agents") or {}
        cons = res.get("consensus") or {}
        sq = (res.get("task") or {}).get("qoi")
        swarm = res.get("swarm") or {}
        return {"key": entry["key"], "name": d.name,
                "status": task.get("state") if task and task.get("state") not in ("done", None) else "done",
                "problem": res.get("problem"), "mode": res.get("mode"), "usd": res.get("usd"),
                "swarm": {"run": swarm.get("run"), "generation": swarm.get("generation"), "tasks": swarm.get("tasks")},
                "n_agents": len(agents), "n_ok": sum(1 for a in agents.values() if (a.get("prediction") or {}).get("qois")),
                "scored_quantity": sq, "consensus": (cons.get("qois") or {}).get(sq) if sq else None,
                "most_confident": cons.get("most_confident"), "updated_at": _iso(_mtime(d / "results.json")),
                "task": _task_info(task)}

    def list_infers(self) -> list[dict]:
        out = [s for d, e in self.folders().items() if (s := self.infer_summary(d, e))]
        return sorted(out, key=lambda s: s.get("updated_at") or "", reverse=True)

    def infer_detail(self, d: Path, entry: dict) -> dict:
        s = self.infer_summary(d, entry)
        if s is None:
            raise KeyError(d.name)
        res = _json(d / "results.json") or {}
        agents = []
        for a, item in sorted((res.get("agents") or {}).items()):
            if not SAFE_NAME.match(a):
                continue
            pred = item.get("prediction") or {}
            images = {str(k): f"work/{a}/{Path(str(p)).name}" for k, p in (pred.get("images") or {}).items()
                      if (d / "work" / a / Path(str(p)).name).is_file()}
            run = _json(d / "runs" / f"{a}.json") or {}
            agents.append({"agent": a, "error": _last_line(item.get("error")), "stop": item.get("stop"),
                           "turns": item.get("turns"), "prediction": {k: v for k, v in pred.items() if k != "images"},
                           "field_score": item.get("field_score"), "images": images,
                           "context": f"context/{a}.md" if (d / "context" / f"{a}.md").is_file() else None,
                           "solves": solve_rows(run) if run else [], "ui_submission_id": run.get("ui_submission_id")})
        return jsonable({**s, "results": {k: res.get(k) for k in ("problem", "task", "mode", "consensus", "usd")},
                         "agents": agents, "summary_md": _text(d / "summary.md"), "tasks": self.tasks_for(d)})

    # ------------------------------------------------------------------ comparisons
    def compare_summary(self, d: Path, entry: dict) -> dict | None:
        task = entry.get("task")
        md = _text(d / "compare.md")
        if md is None and not (task and task.get("tool") == "ace.compare"):
            return None
        m = re.search(r"swarm: `([^`]*)`; single agent: `([^`]*)`", md or "")
        return {"key": entry["key"], "name": d.name, "status": "done" if md is not None else (task or {}).get("state"),
                "swarm": m.group(1) if m else None, "single": m.group(2) if m else None,
                "png": (d / "compare.png").is_file(), "updated_at": _iso(_mtime(d / "compare.md") or _mtime(d)),
                "task": _task_info(task)}

    def list_compares(self) -> list[dict]:
        out = [s for d, e in self.folders().items() if (s := self.compare_summary(d, e))]
        return sorted(out, key=lambda s: s.get("updated_at") or "", reverse=True)


# --------------------------------------------------------------------------- catalogue


def catalog() -> dict:
    """The problems the loops can train on (ace/config.py TASKS) and the suites' defaults."""
    from ace import config
    from feacore.problems import registry

    registry.discover()
    tasks = {}
    for pid, t in config.TASKS.items():
        try:
            p = registry.get(pid)
            info = {"title": p.title, "tags": list(p.tags), "physics": p.physics, "dim": p.dim}
        except Exception:  # noqa: BLE001 - a catalogue entry whose problem does not import here
            info = {"title": pid, "tags": [], "physics": None, "dim": None}
        tasks[pid] = {"qoi": t.get("qoi"), "node_cap": t.get("node_cap"), "score": t.get("score"), **info}
    return {"tasks": tasks,
            "suites": {k: {"train": list(v[0]), "holdout": list(v[1])} for k, v in config.SUITES.items()},
            "single_default": ["cooks_membrane", "nafems_le11_axisym"],
            "defaults": {"generations": config.GENERATIONS, "agents": config.AGENTS,
                         "concurrency": config.CONCURRENCY, "budget_usd": config.BUDGET_USD}}


# --------------------------------------------------------------------------- API


def router(store: RunStore, runner: TaskRunner) -> APIRouter:
    from urllib.parse import urlencode

    ace = AceRuns(store, runner)
    r = APIRouter()

    def find(key: str, swarms: bool = False) -> tuple[Path, dict]:
        try:
            return ace.folder(key, swarms)
        except KeyError:
            raise HTTPException(404, f"no ACE folder {key!r} (known: {', '.join(SWARM_ROOTS if swarms else RUN_ROOTS)}"
                                     " and the Agents tasks' outputs)") from None

    def page(html: str, cache: str) -> HTMLResponse:
        return HTMLResponse(html, headers={"Cache-Control": cache, "Content-Security-Policy": PAGE_CSP,
                                           "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"})

    @r.get("/api/ace/catalog")
    def ace_catalog():
        return jsonable(catalog())

    @r.get("/api/ace/runs")
    def ace_runs():
        return jsonable({"runs": ace.list_runs(), "roots": RUN_ROOTS})

    @r.get("/api/ace/run")
    def ace_run(dir: str):
        d, entry = find(dir)
        if ace.kind(d, entry["task"]) is None:
            raise HTTPException(404, f"{dir!r} is not a training run")
        return jsonable(ace.run_detail(d, entry))

    @r.get("/api/ace/agent-run")
    def ace_agent_run(dir: str, run_id: str):
        d, _ = find(dir)
        try:
            return ace.agent_run(d, run_id)
        except KeyError:
            raise HTTPException(404, f"no agent run {run_id!r} in {dir!r}") from None

    @r.get("/api/ace/file")
    def ace_file(dir: str, path: str, swarm: bool = False):
        d, _ = find(dir, swarm)
        try:
            f = ace.file_in(d, path)
        except KeyError:
            raise HTTPException(404, f"no file {path!r} in {dir!r} (pictures and text files only)") from None
        if f.stat().st_size > MAX_FILE:
            raise HTTPException(413, "file too large to serve")
        return FileResponse(f, media_type=FILE_TYPES[f.suffix.lower()],
                            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    @r.get("/api/ace/report")
    def ace_report(dir: str):
        d, _ = find(dir)
        f = d / "report.html"
        if not f.is_file():
            raise HTTPException(404, "no report.html yet: build it with the Run report tool (ace.report)")
        return page(f.read_text(errors="replace"), "no-cache")

    @r.get("/api/ace/tracker")
    def ace_tracker(dir: str, request: Request):
        d, entry = find(dir)
        from ace import watch
        from feacore.ui.server import _request_token

        q = {"dir": entry["key"]}
        if token := _request_token(request):   # the page polls without headers: it carries the token
            q["token"] = token
        html = watch.PAGE.read_text()
        if 'fetch("data.json"' in html:
            html = html.replace('fetch("data.json"', "fetch(" + json.dumps("/api/ace/tracker/data?" + urlencode(q)))
            html = html.replace("/*__DATA__*/null", "null")
        else:                                   # the tracker page changed: show a snapshot instead
            data = json.dumps(ace.tracker_data(d, entry), default=str).replace("</", "<\\/")
            html = html.replace("/*__DATA__*/null", data)
        return page(html, "no-store")

    @r.get("/api/ace/tracker/data")
    def ace_tracker_data(dir: str):
        d, entry = find(dir)
        try:
            return ace.tracker_data(d, entry)
        except Exception as e:  # noqa: BLE001 - say why instead of a bare 500
            raise HTTPException(500, f"ace.watch could not read {dir!r}: {type(e).__name__}: {e}") from None

    @r.get("/api/ace/swarms")
    def ace_swarms():
        return jsonable({"swarms": ace.list_swarms(), "roots": SWARM_ROOTS})

    @r.get("/api/ace/swarm")
    def ace_swarm(dir: str):
        d, entry = find(dir, swarms=True)
        try:
            return jsonable(ace.swarm_detail(d, entry))
        except KeyError:
            raise HTTPException(404, f"{dir!r} has no manifest.json (not exported yet?)") from None

    @r.get("/api/ace/infers")
    def ace_infers():
        return jsonable({"infers": ace.list_infers()})

    @r.get("/api/ace/infer")
    def ace_infer(dir: str):
        d, entry = find(dir)
        try:
            return ace.infer_detail(d, entry)
        except KeyError:
            raise HTTPException(404, f"{dir!r} is not an inference result") from None

    @r.get("/api/ace/compares")
    def ace_compares():
        return jsonable({"compares": ace.list_compares()})

    @r.get("/api/ace/compare")
    def ace_compare(dir: str):
        d, entry = find(dir)
        s = ace.compare_summary(d, entry)
        if s is None:
            raise HTTPException(404, f"{dir!r} has no compare.md")
        return jsonable({**s, "compare_md": _text(d / "compare.md")})

    r.ace = ace     # the reader, for tests
    return r
