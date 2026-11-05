"""Long-running command-line tools started from the dashboard: the *task* lane.

A **tool** is a whitelisted command with a typed parameter list, e.g. ``ace.swarm`` =
``python -m ace.run_loop --problems ... --generations N``. Tools are registered by the modules of
``feacore.ui.tools`` (auto-discovered, one module per dashboard section). The browser only names a
tool and fills its parameters; it never sends a command line, so nothing outside the registered tools
can run.

A **task** is one run of a tool: a child process (``python <command> <argv>``, cwd = the repo root,
its own process group) and a record in the run store::

    <runs>/tasks/<task_id>/task.json     tool, args, argv, out, state, times, return code
    <runs>/tasks/<task_id>/output.log    stdout + stderr

States: queued | running | done | failed | cancelled | interrupted (the server stopped while it ran).
Tasks run on their own lane (``FEACORE_UI_TASK_WORKERS``, default 2), separate from the quick and
study lanes of ``feacore.ui.jobs``, because an agent training run can take hours.

Every task gets ``FEACORE_RUNS_DIR`` = this server's run store, so tools that record into the store
(the ACE bridge, ``scripts/gold.py render``) show up in the same UI.

API (mounted by ``feacore.ui.server``)::

    GET  /api/tools                     registered tools with their parameter schema (forms)
    POST /api/tasks                     {"tool": name, "args": {...}} -> task record (202)
    GET  /api/tasks?group=&tool=        task records, newest first
    GET  /api/tasks/{id}                one record
    GET  /api/tasks/{id}/log?tail=N     the output (plain text)
    POST /api/tasks/{id}/cancel?mode=   stop (the tool's graceful stop file, e.g. ACE's <out>/STOP) | kill
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from feacore.runstore import RunStore, new_id, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_STATES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
_SAFE_TEXT = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f]*$")


# --------------------------------------------------------------------------- tools


@dataclass
class Param:
    """One parameter of a tool.

    ``type``: str | text | int | float | bool | choice | problem | problems | path.
    ``flag``: the CLI flag (default ``--<name with dashes>``); ``""`` = positional.
    A bool becomes the bare flag when true (``flag_false`` is emitted when false, if given).
    ``problems`` is a comma-separated list of registered problem ids. A ``path`` must resolve inside
    the repo or the run store.
    """

    name: str
    type: str = "str"
    default: Any = None
    help: str = ""
    label: str | None = None
    choices: list | None = None
    required: bool = False
    flag: str | None = None
    flag_false: str | None = None
    min: float | None = None
    max: float | None = None
    advanced: bool = False   # shown folded in the form

    def cli_flag(self) -> str:
        return f"--{self.name.replace('_', '-')}" if self.flag is None else self.flag

    def schema(self) -> dict:
        d = asdict(self)
        d["label"] = self.label or self.name.replace("_", " ")
        return d


@dataclass
class Tool:
    """A whitelisted command. ``command`` is what follows the python executable, e.g.
    ``["-m", "ace.run_loop"]`` or ``["scripts/ccx_crush.py"]`` (relative to the repo root).

    ``out_param``: the parameter that names the tool's output directory. If the user leaves it
    empty it is filled with ``<out_root>/<tool short name>-<task id>`` (``out_root`` relative to the
    repo root, or ``@runs/...`` for a path inside the run store). ``stop_file``: written into the
    output directory to ask the tool to stop gracefully (ACE loops stop before the next generation).
    ``requires_env``: environment variables the tool needs unless ``unless_param`` is true (e.g. an
    API key unless ``mock``). ``build``: optional custom argv builder from the validated args.
    """

    name: str
    title: str
    group: str
    command: list[str]
    params: list[Param] = field(default_factory=list)
    description: str = ""
    out_param: str | None = None
    out_root: str | None = None
    stop_file: str | None = None
    requires_env: list[str] = field(default_factory=list)
    unless_param: str | None = None
    timeout: float | None = None          # seconds; None = no limit (training runs)
    build: Callable[[dict], list[str]] | None = None
    env: dict[str, str] = field(default_factory=dict)

    def schema(self) -> dict:
        missing = [k for k in self.requires_env if not os.environ.get(k)]
        return {"name": self.name, "title": self.title, "group": self.group, "description": self.description,
                "params": [p.schema() for p in self.params], "out_param": self.out_param,
                "can_stop": bool(self.stop_file), "requires_env": self.requires_env,
                "missing_env": missing, "unless_param": self.unless_param,
                "command": " ".join(["python", *self.command])}


TOOLS: dict[str, Tool] = {}
_DISCOVERED = False
_DISCOVERY_ERRORS: dict[str, str] = {}


def register_tool(tool: Tool) -> Tool:
    if tool.name in TOOLS and TOOLS[tool.name] is not tool:
        raise ValueError(f"duplicate tool {tool.name!r}")
    names = [p.name for p in tool.params]
    if len(names) != len(set(names)):
        raise ValueError(f"tool {tool.name!r}: duplicate parameter names")
    TOOLS[tool.name] = tool
    return tool


def discover_tools() -> dict[str, str]:
    """Import every module of ``feacore.ui.tools`` once (each registers its tools). A module that
    fails to import is reported, not raised: one broken section must not take the dashboard down."""
    global _DISCOVERED
    if _DISCOVERED:
        return dict(_DISCOVERY_ERRORS)
    import feacore.ui.tools as pkg

    for mod in pkgutil.iter_modules(pkg.__path__, prefix="feacore.ui.tools."):
        if mod.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001
            _DISCOVERY_ERRORS[mod.name] = f"{type(e).__name__}: {e}"
    _DISCOVERED = True
    return dict(_DISCOVERY_ERRORS)


def get_tool(name: str) -> Tool:
    discover_tools()
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; known: {sorted(TOOLS)}") from None


# --------------------------------------------------------------------------- validation


class ToolArgsError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _inside(path: Path, roots: list[Path]) -> bool:
    p = path.resolve()
    return any(p == r or r in p.parents for r in (r.resolve() for r in roots))


def resolve_out(value: str, store: RunStore) -> Path:
    """An output/input location: ``@runs/x`` = inside the run store, else relative to the repo root."""
    if value.startswith("@runs/"):
        return store.root / value[len("@runs/"):]
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def validate_args(tool: Tool, args: dict | None, store: RunStore) -> dict:
    """Checked, typed args (defaults filled in). Raises ToolArgsError listing every problem."""
    args = dict(args or {})
    errors: list[str] = []
    known = {p.name: p for p in tool.params}
    for k in args:
        if k not in known:
            errors.append(f"unknown parameter {k!r}")
    out: dict[str, Any] = {}
    for p in tool.params:
        v = args.get(p.name, None)
        if v is None or v == "":
            if p.required:
                errors.append(f"{p.name} is required")
            out[p.name] = p.default
            continue
        try:
            out[p.name] = _coerce(p, v, store)
        except ValueError as e:
            errors.append(f"{p.name}: {e}")
    if errors:
        raise ToolArgsError(errors)
    return out


def _coerce(p: Param, v: Any, store: RunStore) -> Any:
    t = p.type
    if t == "bool":
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("1", "true", "yes", "on"):
            return True
        if str(v).lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {v!r}")
    if t in ("int", "float"):
        try:
            x = int(v) if t == "int" and not (isinstance(v, float) and not v.is_integer()) else float(v)
            x = int(x) if t == "int" else x
        except (TypeError, ValueError):
            raise ValueError(f"not a number: {v!r}") from None
        if p.min is not None and x < p.min:
            raise ValueError(f"must be >= {p.min}")
        if p.max is not None and x > p.max:
            raise ValueError(f"must be <= {p.max}")
        return x
    s = str(v).strip()
    if not _SAFE_TEXT.match(s):
        raise ValueError("control characters are not allowed")
    if t != "text" and s.startswith("-"):
        raise ValueError("must not start with '-'")
    if t == "choice":
        if p.choices and s not in [str(c) for c in p.choices]:
            raise ValueError(f"must be one of {p.choices}")
        return s
    if t in ("problem", "problems"):
        from feacore.problems import registry

        registry.discover()
        ids = [x.strip() for x in s.split(",") if x.strip()] if t == "problems" else [s]
        unknown = [i for i in ids if i not in registry._REGISTRY]   # noqa: SLF001 - cheap membership check
        if unknown:
            raise ValueError(f"unknown problem(s) {unknown}")
        if t == "problem" and len(ids) != 1:
            raise ValueError("exactly one problem id")
        return ",".join(ids)
    if t == "path":
        path = resolve_out(s, store)
        if ".." in Path(s).parts or not _inside(path, [REPO_ROOT, store.root]):
            raise ValueError("must be a path inside the repository or the run store")
        return s
    if t in ("str", "text"):
        if len(s) > 4000:
            raise ValueError("too long")
        return s
    raise ValueError(f"unknown parameter type {t!r}")


def build_argv(tool: Tool, args: dict) -> list[str]:
    """The command line after the python executable."""
    if tool.build is not None:
        return [*tool.command, *tool.build(args)]
    argv = list(tool.command)
    positional: list[str] = []
    for p in tool.params:
        v = args.get(p.name)
        if v is None:
            continue
        flag = p.cli_flag()
        if p.type == "bool":
            if v and flag:
                argv.append(flag)
            elif not v and p.flag_false:
                argv.append(p.flag_false)
            continue
        if flag == "":
            positional.append(str(v))
        else:
            argv += [flag, str(v)]
    return argv + positional


# --------------------------------------------------------------------------- store


class TaskStore:
    def __init__(self, store: RunStore):
        self.store = store

    @property
    def root(self) -> Path:
        return self.store.root / "tasks"

    def dir(self, task_id: str) -> Path:
        if not re.fullmatch(r"task-[A-Za-z0-9_-]+", task_id):
            raise KeyError(task_id)
        return self.root / task_id

    def save(self, rec: dict) -> dict:
        d = self.dir(rec["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "task.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        tmp.replace(d / "task.json")
        return rec

    def get(self, task_id: str) -> dict:
        f = self.dir(task_id) / "task.json"
        if not f.exists():
            raise KeyError(task_id)
        return json.loads(f.read_text())

    def update(self, task_id: str, **fields) -> dict:
        rec = self.get(task_id)
        rec.update(fields)
        return self.save(rec)

    def list(self, group: str | None = None, tool: str | None = None, limit: int = 200) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for f in self.root.glob("task-*/task.json"):
            try:
                rec = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if (group and rec.get("group") != group) or (tool and rec.get("tool") != tool):
                continue
            out.append(rec)
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out[:limit]

    def log_path(self, task_id: str) -> Path:
        return self.dir(task_id) / "output.log"

    def mark_interrupted(self) -> int:
        n = 0
        for rec in self.list(limit=10_000):
            if rec.get("state") in ("queued", "running"):
                pid = rec.get("pid")
                note = "the server stopped while this task was queued or running"
                if pid and _alive(pid):
                    note += f" (process {pid} may still be running)"
                self.update(rec["id"], state="interrupted", finished_at=now_iso(), error=note)
                n += 1
        return n


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- runner


class TaskRunner:
    """Runs tasks in child processes, at most ``workers`` at a time."""

    def __init__(self, store: RunStore, workers: int | None = None):
        self.store = store
        self.tasks = TaskStore(store)
        self.workers = int(workers or os.environ.get("FEACORE_UI_TASK_WORKERS", 2))
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._queue: list[str] = []
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            for k in range(max(1, self.workers)):
                t = threading.Thread(target=self._loop, name=f"feacore-task-{k}", daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
            procs = dict(self._procs)
        for tid, p in procs.items():
            _terminate(p)
            try:
                self.tasks.update(tid, state="interrupted", finished_at=now_iso(),
                                  error="the server stopped while this task was running")
            except KeyError:
                pass

    # ------------------------------------------------------------------ submit / cancel
    def submit(self, tool_name: str, args: dict | None) -> dict:
        tool = get_tool(tool_name)
        clean = validate_args(tool, args, self.store)
        tid = new_id("task")
        out = None
        if tool.out_param:
            if not clean.get(tool.out_param):
                short = tool.name.rsplit(".", 1)[-1].replace("_", "-")
                root = tool.out_root or "@runs/task_outputs"
                clean[tool.out_param] = f"{root.rstrip('/')}/{short}-{tid.split('-', 1)[1]}"
            out = clean[tool.out_param]
        if tool.requires_env and not (tool.unless_param and clean.get(tool.unless_param)):
            missing = [k for k in tool.requires_env if not os.environ.get(k)]
            if missing:
                hint = f" (or set {tool.unless_param})" if tool.unless_param else ""
                raise ToolArgsError([f"the server has no {', '.join(missing)} in its environment{hint}"])
        argv = build_argv(tool, {**clean, **({tool.out_param: str(resolve_out(out, self.store))} if out else {})})
        rec = {"id": tid, "tool": tool.name, "title": tool.title, "group": tool.group, "args": clean,
               "argv": argv, "out": out, "out_abs": str(resolve_out(out, self.store)) if out else None,
               "state": "queued", "created_at": now_iso(), "started_at": None, "finished_at": None,
               "pid": None, "returncode": None, "error": None, "cancel": None}
        self.tasks.save(rec)
        self.start()
        with self._cv:
            self._queue.append(tid)
            self._cv.notify()
        return rec

    def cancel(self, task_id: str, mode: str = "kill") -> dict:
        rec = self.tasks.get(task_id)
        if rec["state"] not in ("queued", "running"):
            return rec
        with self._cv:
            if task_id in self._queue:
                self._queue.remove(task_id)
                return self.tasks.update(task_id, state="cancelled", finished_at=now_iso(), cancel=mode)
            proc = self._procs.get(task_id)
        tool = TOOLS.get(rec["tool"])
        if mode == "stop" and tool is not None and tool.stop_file and rec.get("out_abs"):
            d = Path(rec["out_abs"])
            d.mkdir(parents=True, exist_ok=True)
            (d / tool.stop_file).write_text(f"stop requested from the dashboard at {now_iso()}\n")
            return self.tasks.update(task_id, cancel="stop")
        rec = self.tasks.update(task_id, cancel="kill")
        if proc is not None:
            _terminate(proc)
        return rec

    def active_count(self) -> int:
        with self._lock:
            return len(self._procs) + len(self._queue)

    def queue_position(self, task_id: str) -> int | None:
        with self._lock:
            return self._queue.index(task_id) + 1 if task_id in self._queue else None

    # ------------------------------------------------------------------ worker loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._queue and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                tid = self._queue.pop(0)
            try:
                self._run(tid)
            except Exception as e:  # noqa: BLE001 - never kill the lane thread
                try:
                    self.tasks.update(tid, state="failed", finished_at=now_iso(), error=f"{type(e).__name__}: {e}")
                except KeyError:
                    pass

    def _run(self, tid: str) -> None:
        rec = self.tasks.get(tid)
        tool = get_tool(rec["tool"])
        env = os.environ.copy()
        env.update(tool.env)
        env["FEACORE_RUNS_DIR"] = str(self.store.root.resolve())
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "meshing-harness" / "src"), str(REPO_ROOT),
                                             *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
        if rec.get("out_abs"):
            Path(rec["out_abs"]).mkdir(parents=True, exist_ok=True)
        log = self.tasks.log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as logf:
            logf.write(f"=== {now_iso()} {tool.name}: python {' '.join(rec['argv'])}\n".encode())
            logf.flush()
            p = subprocess.Popen([sys.executable, *rec["argv"]], cwd=REPO_ROOT, stdout=logf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 start_new_session=True)
            with self._lock:
                self._procs[tid] = p
            self.tasks.update(tid, state="running", started_at=now_iso(), pid=p.pid)
            try:
                rc = p.wait(timeout=tool.timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _terminate(p)
                rc, timed_out = p.returncode, True
            finally:
                with self._lock:
                    self._procs.pop(tid, None)
        cur = self.tasks.get(tid)
        if cur.get("state") == "interrupted":   # stop() already recorded it
            return
        if timed_out:
            state, err = "failed", f"timed out after {tool.timeout:.0f} s"
        elif rc == 0:
            state, err = "done", None
        elif cur.get("cancel"):
            state, err = "cancelled", f"cancelled ({cur['cancel']})"
        else:
            state, err = "failed", f"exited with code {rc}; see the log"
        self.tasks.update(tid, state=state, returncode=rc, finished_at=now_iso(), error=err)


def _terminate(p: subprocess.Popen, grace: float = 10.0) -> None:
    """SIGTERM the task's process group (tools start their own children), SIGKILL after ``grace``."""
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        p.terminate()
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        p.wait()


def tail(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            data = f.read()
            data = data[data.find(b"\n") + 1:]
        else:
            data = f.read()
    return data.decode("utf-8", "replace")


# --------------------------------------------------------------------------- API


def router(runner: TaskRunner):
    from fastapi import APIRouter, Body, HTTPException
    from fastapi.responses import PlainTextResponse

    r = APIRouter()

    @r.get("/api/tools")
    def list_tools(group: str | None = None):
        errors = discover_tools()
        tools = [t.schema() for t in TOOLS.values() if not group or t.group == group]
        return {"tools": sorted(tools, key=lambda t: (t["group"], t["name"])), "import_errors": errors}

    @r.post("/api/tasks", status_code=202)
    def create_task(body: dict = Body(..., examples=[{"tool": "ace.swarm", "args": {"mock": True}}])):
        name, args = body.get("tool"), body.get("args") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            raise HTTPException(422, "body must be {\"tool\": name, \"args\": {...}}")
        try:
            return runner.submit(name, args)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ToolArgsError as e:
            raise HTTPException(422, {"detail": "invalid arguments", "errors": e.errors}) from None

    @r.get("/api/tasks")
    def list_tasks(group: str | None = None, tool: str | None = None, limit: int = 200):
        return {"tasks": runner.tasks.list(group=group, tool=tool, limit=limit), "active": runner.active_count()}

    @r.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        try:
            rec = runner.tasks.get(task_id)
        except KeyError:
            raise HTTPException(404, f"no task {task_id!r}") from None
        return {**rec, "queue_position": runner.queue_position(task_id)}

    @r.get("/api/tasks/{task_id}/log", response_class=PlainTextResponse)
    def task_log(task_id: str, tail_bytes: int = 200_000):
        try:
            path = runner.tasks.log_path(task_id)
        except KeyError:
            raise HTTPException(404, f"no task {task_id!r}") from None
        return PlainTextResponse(tail(path, max(1000, min(tail_bytes, 5_000_000))))

    @r.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, mode: str = "kill"):
        if mode not in ("stop", "kill"):
            raise HTTPException(422, "mode must be stop or kill")
        try:
            return runner.cancel(task_id, mode)
        except KeyError:
            raise HTTPException(404, f"no task {task_id!r}") from None

    return r
