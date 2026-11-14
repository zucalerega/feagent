"""HTTP API of the dashboard's Projects section (#/projects): dossier projects in the run store.

dossier (``dossier/``, ``dossier/CONTRACTS.md``) verifies one component by analysis in the context of its
system. A project is a folder of input files (YAML, CAD) plus what the pipeline derives from them (``out/``)
and the reviews people give (``reviews.yaml``). The dashboard keeps its projects in
``<run store>/dossier/<folder>/``; the examples in ``dossier/examples/`` are templates that are copied there
and never written to.

    GET    /api/dossier                              the projects with their status, and the examples
    POST   /api/dossier                              {"example", "folder"}: a new project from an example
    GET    /api/dossier/{project}                    one project: problems of the inputs, stages, verdicts,
                                                     counts, load cases with their analysis, tasks
    GET    /api/dossier/{project}/files              the input files (YAML, CAD) with sizes and times
    GET    /api/dossier/{project}/files/{path}       the text of one YAML input (a CAD file: a download)
    PUT    /api/dossier/{project}/files/{path}       save new text, then check the project (ok or problems)
    POST   /api/dossier/{project}/cad/{component}    multipart ``file``: replace the component's STEP file
    POST   /api/dossier/{project}/check              read the inputs and evaluate every number
    POST   /api/dossier/{project}/tasks              {"tool", "args"}: start dossier.run / dossier.report
                                                     for this project (409 while one of its tasks runs)
    GET    /api/dossier/{project}/params             the named numbers of the context
    GET    /api/dossier/{project}/records            the derived records by collection
    GET    /api/dossier/{project}/trace              dossier.trace.graph(project)
    GET    /api/dossier/{project}/reviews            {"reviews": {record id: {state, by, date, comment, stale}}}
    POST   /api/dossier/{project}/reviews            {"record", "state", "by", "comment"} -> the stored review
    GET    /api/dossier/{project}/report/{path}      out/report/ (index.html gets window.DOSSIER_API)
    GET    /api/dossier/{project}/out/{path}         a derived file inside out/ (pictures, JSON, logs)

``{project}`` is the folder name. A project whose inputs are wrong is not a server error: the status, the
check and a save answer 200 with ``ok: false`` and ``problems`` (each with the file, the field, the line and
the message where they can be told); the endpoints that need the evaluated inputs (params, trace, a review)
answer 422 with the same list. The pipeline itself is long-running and runs on the task lane
(feacore/ui/tools/dossier.py).

Files: an input is served or saved only when it is one of the files the project names, inside the project
folder; derived files only from ``out/``, by type. Every path is resolved with symlinks followed and must
stay inside its folder. dossier is imported on first use, so the dashboard starts without it.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from feacore.ui.tasks import REPO_ROOT, TaskRunner, ToolArgsError, tail
from feacore.ui.tools.dossier import FOLDER, GROUP, PROJECTS_DIR, STAGES, TOOL_NAMES, project_ref

EXAMPLES = REPO_ROOT / "dossier" / "examples"
YAML_SUFFIXES = (".yaml", ".yml")
CAD_SUFFIXES = (".step", ".stp")
MAX_TEXT = 2_000_000              # bytes of one YAML input saved from the browser
MAX_FILE = 50_000_000             # bytes of one derived file served
PICTURES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
            ".webp": "image/webp", ".svg": "image/svg+xml"}
TEXT = "text/plain; charset=utf-8"
OUT_TYPES = {**PICTURES, ".json": "application/json", ".txt": TEXT, ".log": TEXT, ".md": TEXT, ".csv": TEXT}
REPORT_TYPES = {**OUT_TYPES, ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".woff2": "font/woff2", ".pdf": "application/pdf"}
REVIEW_STATES = ("accepted", "rejected", "changes_requested", "open")
REVIEWED = ("decisions", "assumptions", "findings")          # the records a person reviews
SEVERITIES = ("error", "major", "minor", "info")
VERDICTS = ("pass", "fail", "inconclusive", "not_assessed")
OUTSIDE = "The dashboard works on files inside the project folder only."
# The report runs its own code and talks only to this server (as the pages of feacore/ui/ace_runs.py).
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline' 'self'; style-src 'unsafe-inline' 'self'; "
            "img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'self'; base-uri 'none'; form-action 'none'")
# An SVG opened on its own is a document: no script, nothing loaded from elsewhere.
SVG_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"
# With a token on the server, the report's own requests, pictures and links must carry it. The page was
# opened with ?token= (a link, the frame of the Projects page), or the browser holds it for this origin.
TOKEN_SHIM = """<script>(function () {
  var t = new URLSearchParams(location.search).get("token");
  try { t = t || localStorage.getItem("feacore_token"); } catch (e) {}
  if (!t) return;
  var own = function (u) { return u.origin === location.origin && u.pathname.indexOf("/api/") === 0; };
  var plain = window.fetch;
  window.fetch = function (input, init) {
    var u = new URL(typeof input === "string" ? input : input.url, location.href);
    if (!own(u)) return plain.call(this, input, init);
    init = init || {};
    var h = new Headers(init.headers || (typeof input === "string" ? undefined : input.headers));
    if (!h.has("Authorization")) h.set("Authorization", "Bearer " + t);
    init.headers = h;
    return plain.call(this, input, init);
  };
  var add = function (el, attr) {
    var u = new URL(el.getAttribute(attr), location.href);
    if (!own(u) || u.searchParams.has("token")) return;
    u.searchParams.set("token", t);
    el.setAttribute(attr, u.href);
  };
  window.addEventListener("error", function (e) {
    var el = e.target;
    if (el && el.getAttribute && el.getAttribute("src") && !el.dataset.token) { el.dataset.token = "1"; add(el, "src"); }
  }, true);
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a[href]");
    if (a && a.getAttribute("href").charAt(0) !== "#") add(a, "href");
  }, true);
})();</script>"""


class Unloadable(Exception):
    """The inputs of a project are wrong, so what was asked for cannot be made. ``check`` holds the problems."""

    def __init__(self, check: dict):
        self.check = check
        super().__init__("the project has problems")


# --------------------------------------------------------------------------- small helpers


def _dossier():
    """dossier's modules (cli, context, project, trace), imported on first use: pint and the pydantic models
    take about half a second, and a dossier that does not import must not stop the dashboard."""
    try:
        from dossier import cli, context, project, trace
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"the dossier package does not import: {type(e).__name__}: {e}") from None
    return cli, context, project, trace


def _iso(t: float | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


def _epoch(iso: str | None) -> float | None:
    """Seconds since the epoch of an ISO time in UTC (a task record's), or None."""
    try:
        return calendar.timegm(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")) if iso else None
    except ValueError:
        return None


def _yaml(path: Path) -> Any:
    """A YAML file as plain data, or None when it is missing or not YAML (the project check says why)."""
    import yaml

    try:
        return yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def clean_rel(text: Any) -> str | None:
    """A path relative to a folder as a clean posix string, or None when it is not one: absolute, with a
    backslash or a control character, or with a part that is ``..``."""
    if not isinstance(text, str) or not text or len(text) > 500 or "\\" in text or re.search(r"[\x00-\x1f]", text):
        return None
    p = PurePosixPath(text)
    if p.is_absolute() or text.startswith("~") or ".." in p.parts:
        return None
    return "/".join(x for x in p.parts if x != ".") or None


def within(root: Path, rel: str | None) -> Path | None:
    """``root / rel`` when it stays inside ``root`` with symlinks followed, else None."""
    if not rel:
        return None
    p = root / rel
    try:
        real, base = p.resolve(), root.resolve()
    except OSError:
        return None
    return p if real == base or base in real.parents else None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    """Replace a file in one step, so that a reader (a running task) never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


# --------------------------------------------------------------------------- the input files of a project


def layout(root: Path) -> dict:
    """The input files of a project, read from project.yaml without loading the project, so that a project
    with wrong inputs still lists them: ``{"doc", "files", "components", "notes"}``. A file the project
    names outside its folder is listed with ``inside: false``; it is never read or written here, and
    ``notes`` says so in the form of dossier's own problem lines."""
    doc = _yaml(root / "project.yaml")
    doc = doc if isinstance(doc, dict) else {}
    files, components, notes = [], [], []

    def entry(rel: Any, role: str, label: str, named: str, kind: str = "yaml", component: str | None = None) -> dict:
        clean = clean_rel(rel)
        p = within(root, clean)
        e = {"path": clean or str(rel), "role": role, "label": label, "kind": kind, "component": component,
             "inside": p is not None, "exists": False, "bytes": None, "modified": None, "editable": False}
        if p is None:
            notes.append(f"{named}: {rel!r} is outside the project folder. {OUTSIDE}")
            return e
        if p.is_file():
            st = p.stat()
            e.update(exists=True, bytes=st.st_size, modified=_iso(st.st_mtime))
        e["editable"] = kind == "yaml" and p.suffix.lower() in YAML_SUFFIXES and not p.is_dir()
        return e

    files.append(entry("project.yaml", "project", "Project", "project.yaml"))
    for key, default, label in (("system", "system.yaml", "System"), ("materials", "materials.yaml", "Materials"),
                                ("policy", "policy.yaml", "Factor policy"), ("terms", None, "Terms")):
        rel = doc.get(key, default)
        if isinstance(rel, str) and rel:
            files.append(entry(rel, key, label, f"project.yaml: {key}"))
    dirs = doc.get("components")
    if not isinstance(dirs, list):       # project.yaml does not say (or does not read): the usual place
        dirs = sorted(p.parent.relative_to(root).as_posix() for p in root.glob("components/*/component.yaml"))
    for i, d in enumerate(dirs):
        clean = clean_rel(d)
        if within(root, clean) is None:
            notes.append(f"project.yaml: components.{i}: {d!r} is outside the project folder. {OUTSIDE}")
            continue
        cdoc = _yaml(root / clean / "component.yaml")
        cdoc = cdoc if isinstance(cdoc, dict) else {}
        text = lambda key: cdoc.get(key) if isinstance(cdoc.get(key), str) else None  # noqa: E731
        comp = {"dir": clean, "id": text("id"), "name": text("name"), "material": text("material"), "cad": None}
        files.append(entry(f"{clean}/component.yaml", "component", "Component", "project.yaml", component=comp["id"]))
        files.append(entry(f"{clean}/requirements.yaml", "requirements", "Requirements", "project.yaml",
                           component=comp["id"]))
        flow = cdoc.get("flow") if isinstance(cdoc.get("flow"), dict) else {}
        for holder, role, label, where in ((cdoc, "cad", "CAD", "cad.file"), (flow, "flow_cad", "Flow domain CAD", "flow.cad.file")):
            cad = holder.get("cad") if isinstance(holder.get("cad"), dict) else {}
            name = cad.get("file")
            if isinstance(name, str) and name:
                e = entry(f"{clean}/{name}" if clean_rel(name) else name, role, label,
                          f"{clean}/component.yaml: {where}", kind="cad", component=comp["id"])
                files.append(e)
                if role == "cad" and e["inside"]:
                    comp["cad"] = e["path"]
        components.append(comp)
    return {"doc": doc, "files": files, "components": components, "notes": notes}


def _stat_sig(paths) -> tuple:
    out = []
    for p in sorted(set(paths)):
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def _inputs_sig(root: Path, lay: dict) -> tuple:
    """Changes when an input changes: the files the project names and every file of a component folder
    (dossier's inputs digest reads those too)."""
    paths = [root / f["path"] for f in lay["files"] if f["inside"]]
    for c in lay["components"]:
        try:
            paths += [p for p in (root / c["dir"]).iterdir() if p.is_file()]
        except OSError:
            pass
    return (*_stat_sig(paths), tuple(lay["notes"]))


# --------------------------------------------------------------------------- problems, with their place


_FIELD = re.compile(r"^(?:(?P<what>[A-Za-z_]+)\[(?P<index>\d+)\]: )?(?P<loc>\(top level\)|[^\s:]+): (?P<msg>.*)$", re.S)
_NAMED = re.compile(r"^(?P<kind>effect|parameter|duty cycle|connection|component|interface|requirement) "
                    r"(?P<id>[^\s:]+)(?: \([^)]*\))?: (?P<msg>.*)$", re.S)
_TWICE = re.compile(r"^(?:id|the name) '(?P<id>[^']+)' ")
_YAML_LINE = re.compile(r"line (\d+)")
# the files where a kind of record is written, to look there first
_HOME = {"effect": ("system",), "duty cycle": ("system",), "connection": ("system",),
         "parameter": ("system", "requirements", "materials"), "component": ("component",),
         "interface": ("component",), "requirement": ("requirements",)}


class _Texts:
    """The YAML inputs of a project as text, read once, to find the line a problem is about."""

    def __init__(self, root: Path, lay: dict):
        self.root = root
        self.files = [f for f in lay["files"] if f["editable"] and f["exists"]]
        self._text: dict[str, str] = {}

    def text(self, rel: str) -> str:
        if rel not in self._text:
            try:
                self._text[rel] = (self.root / rel).read_text(errors="replace")
            except OSError:
                self._text[rel] = ""
        return self._text[rel]

    def named(self, ident: str, roles: tuple[str, ...] = (), last: bool = False) -> tuple[str | None, int | None]:
        """(file, line) where ``ident`` is defined: ``id: X`` of a record, or the key ``X:`` of a number."""
        q = re.escape(ident)
        pats = [re.compile(rf"(?:^|[\s{{,-])id:\s*['\"]?{q}['\"]?\s*(?:[,}}#]|$)"),
                re.compile(rf"^\s*['\"]?{q}['\"]?\s*:"), re.compile(rf"[{{,]\s*['\"]?{q}['\"]?\s*:")]
        order = sorted(self.files, key=lambda f: roles.index(f["role"]) if f["role"] in roles else len(roles))
        for pat in pats:
            hits = [(f["path"], i + 1) for f in order for i, line in enumerate(self.text(f["path"]).splitlines())
                    if pat.search(line)]
            if hits:
                return hits[-1] if last else hits[0]
        return None, None

    def at(self, rel: str, path: list[str]) -> int | None:
        """The line of the deepest part of ``path`` (mapping keys and list positions) that is in the file."""
        import yaml

        try:
            node = yaml.compose(self.text(rel))
        except yaml.YAMLError:
            return None
        line = None
        for part in path:
            if isinstance(node, yaml.MappingNode):
                hit = next(((k, v) for k, v in node.value if getattr(k, "value", None) == part), None)
                if hit is None:
                    break
                line, node = hit[0].start_mark.line + 1, hit[1]
            elif isinstance(node, yaml.SequenceNode) and part.isdigit() and int(part) < len(node.value):
                node = node.value[int(part)]
                line = node.start_mark.line + 1
            else:
                break
        return line


def explain(text: str, root: Path, lay: dict, texts: _Texts | None = None) -> dict:
    """One problem line of dossier (``ProjectError.problems``) with its place: ``{"text", "file", "field",
    "line", "message"}``. ``file`` and ``line`` are None when the line does not let them be told."""
    texts = texts or _Texts(root, lay)
    shown = text.replace(str(root.resolve()) + os.sep, "").replace(str(root) + os.sep, "")
    out = {"text": shown, "file": None, "field": "", "line": None, "message": shown}
    known = {f["path"] for f in lay["files"]}
    head, sep, rest = shown.partition(": ")
    if sep and (head in known or head.endswith(YAML_SUFFIXES)):       # "<file>: [<list>[i]: ]<field>: <message>"
        out.update(file=head, message=rest)
        m = _FIELD.match(rest)
        if rest.startswith("not valid YAML"):
            if (line := _YAML_LINE.search(rest)):
                out["line"] = int(line.group(1))
        elif m and not rest.startswith("file not found"):
            parts = [m["what"], m["index"]] if m["what"] else []
            parts += [] if m["loc"] == "(top level)" else m["loc"].split(".")
            field_ = (f"{m['what']}[{m['index']}]" if m["what"] else "")
            field_ += ("." if field_ and m["loc"] != "(top level)" else "") + ("" if m["loc"] == "(top level)" else m["loc"])
            out.update(field=field_, message=m["msg"], line=texts.at(head, parts))
        return out
    if (m := _NAMED.match(shown)):                                    # "effect EF-X (IF-Y, PH-Z): <message>"
        f, line = texts.named(m["id"], _HOME[m["kind"]])
        out.update(file=f, line=line, field=f"{m['kind']} {m['id']}", message=m["msg"])
    elif (m := _TWICE.match(shown)):                                  # "id 'X' is used twice: ..."
        f, line = texts.named(m["id"], (), last=True)
        out.update(file=f, line=line, field=m["id"])
    return out


# --------------------------------------------------------------------------- the log of a run


_LOG_STAGE = re.compile(r"^== (\w+)\s*$")
_LOG_DONE = re.compile(r"^ {3}(\w+) done in [0-9.]+ s")
_LOG_FAIL = re.compile(r"^stage (\w+): (.*)$")


def run_progress(log: str) -> dict:
    """What the output of ``python -m dossier run`` says: ``{"done": [stages], "current": the stage that
    started and did not end, "failed": {"stage", "message"} | None, "problems": [input problems], "last":
    the last line}``."""
    done: list[str] = []
    current, failed, problems, last = None, None, [], ""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith("==="):
            last = line.strip()
        if (m := _LOG_STAGE.match(line)) and m.group(1) in STAGES:
            current = m.group(1)
        elif (m := _LOG_DONE.match(line)) and m.group(1) == current:
            done.append(current)
            current = None
        elif (m := _LOG_FAIL.match(line)) and m.group(1) in STAGES:
            failed = {"stage": m.group(1), "message": m.group(2).strip()}
        elif line.startswith("The project has problems:"):
            for x in lines[i + 1:]:
                if not x.startswith("  "):
                    break
                problems.append(x.strip())
    return {"done": done, "current": current, "failed": failed, "problems": problems, "last": last}


# --------------------------------------------------------------------------- projects


@dataclass
class Loaded:
    """The inputs of a project as dossier reads them: the Project, or why it does not load."""

    sig: tuple
    project: Any = None
    problems: list[str] = field(default_factory=list)     # why Project.load refused it
    digest: str | None = None                             # Project.inputs_digest(): what the stage files were made from
    ctx: Any = None                                       # the evaluated context, built on first use
    ctx_problems: list[str] | None = None                 # why the context does not evaluate


@dataclass
class Derived:
    """What the stage files of a project hold, read once per change of the files."""

    sig: tuple
    files: list[dict] = field(default_factory=list)           # one per stage file, in stage order
    collections: dict[str, list[dict]] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)     # record id -> content hash (a review keeps it)
    owner: dict[str, str] = field(default_factory=dict)       # record id -> the stage file that holds it
    warnings: list[str] = field(default_factory=list)


class Projects:
    """The dossier projects of one run store."""

    def __init__(self, store, tasks: TaskRunner | None):
        self.root = Path(store.root) / PROJECTS_DIR
        self.tasks = tasks
        self.lock = threading.RLock()        # dossier's unit registry, the caches and the writes: one at a time
        self._loaded: OrderedDict[str, Loaded] = OrderedDict()
        self._derived: OrderedDict[str, Derived] = OrderedDict()

    # ------------------------------------------------------------------ folders
    def dir(self, folder: str) -> Path:
        if not isinstance(folder, str) or not FOLDER.match(folder) or not (self.root / folder / "project.yaml").is_file():
            raise KeyError(folder)
        return self.root / folder

    def folders(self) -> tuple[list[str], list[dict]]:
        """(project folders, folders with a project.yaml under a name the dashboard does not take)."""
        good, ignored = [], []
        for d in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if d.is_dir() and (d / "project.yaml").is_file():
                if FOLDER.match(d.name):
                    good.append(d.name)
                else:
                    ignored.append({"folder": d.name, "reason": "A folder name can have letters, digits, '-' and '_' "
                                                                "only (64 at most). Rename the folder."})
        return good, ignored

    @staticmethod
    def examples() -> list[dict]:
        out = []
        for d in sorted(EXAMPLES.iterdir()) if EXAMPLES.is_dir() else []:
            doc = _yaml(d / "project.yaml")
            if d.is_dir() and isinstance(doc, dict):
                out.append({"name": d.name, "id": doc.get("id"), "title": doc.get("name") or d.name,
                            "description": _text(doc.get("description"))})
        return out

    def create(self, example: str, folder: Any) -> str:
        """A new project: a copy of an example, made by dossier's own ``new`` command."""
        names = [e["name"] for e in self.examples()]
        if example not in names:
            raise HTTPException(404, f"no example {example!r}; examples: {', '.join(names) or 'none'}")
        if not isinstance(folder, str) or not FOLDER.match(folder):
            raise HTTPException(422, "The folder name must start with a letter or a digit and can have letters, "
                                     "digits, '-' and '_' only (64 at most), for example my_valve.")
        cli = _dossier()[0]
        with self.lock:
            dst = self.root / folder
            if dst.exists():
                raise HTTPException(409, f"A project folder {folder!r} exists already. Give another name.")
            self.root.mkdir(parents=True, exist_ok=True)
            if cli.main(["new", str(dst), "--example", example]) != 0 or not (dst / "project.yaml").is_file():
                raise HTTPException(500, f"python -m dossier new did not make {folder!r} from the example {example!r}")
        return folder

    # ------------------------------------------------------------------ inputs
    def load(self, root: Path, lay: dict | None = None) -> Loaded:
        lay = lay or layout(root)
        sig = _inputs_sig(root, lay)
        with self.lock:
            hit = self._loaded.get(str(root))
            if hit is not None and hit.sig == sig:
                self._loaded.move_to_end(str(root))
                return hit
            lp = Loaded(sig)
            if lay["notes"]:                      # a file outside the project: dossier would read it, the dashboard does not
                lp.problems = list(lay["notes"])
            else:
                core = _dossier()[2]
                try:
                    lp.project = core.Project.load(root)
                    lp.digest = lp.project.inputs_digest()
                except core.ProjectError as e:
                    lp.problems = list(e.problems)
            self._loaded[str(root)] = lp
            while len(self._loaded) > 16:
                self._loaded.popitem(last=False)
            return lp

    def context(self, lp: Loaded) -> Any:
        """The evaluated context of a loaded project, or None (then ``lp.ctx_problems`` says why)."""
        with self.lock:
            if lp.ctx is None and lp.ctx_problems is None and lp.project is not None:
                _, context, core, _ = _dossier()
                try:
                    lp.ctx = context.build(lp.project)
                except core.ProjectError as e:
                    lp.ctx_problems = list(e.problems)
            return lp.ctx

    def check(self, root: Path, lay: dict | None = None) -> dict:
        """Read the inputs and evaluate every number (as ``python -m dossier check``): ok with the counts,
        or the problems."""
        lay = lay or layout(root)
        lp = self.load(root, lay)
        ctx = self.context(lp)
        lines = lp.problems if lp.project is None else (lp.ctx_problems or [])
        if lines:
            texts = _Texts(root, lay)
            return {"ok": False, "stage": "inputs" if lp.project is None else "context",
                    "problems": [explain(x, root, lay, texts) for x in lines]}
        p = lp.project
        return {"ok": True, "problems": [], "counts": {
            "elements": len(p.system.elements), "connections": len(p.system.connections),
            "phases": len(p.system.phases), "duty_cycles": len(p.system.duty_cycles),
            "components": len(p.components), "features": len(p.subject.features),
            "interfaces": len(p.subject.interfaces), "requirements": len(p.reqs()),
            "parameters": len(ctx.params), "effects": len(ctx.effects)},
            "summary": f"{len(ctx.params)} named numbers and {len(ctx.effects)} effects evaluated. The units are consistent."}

    def usable(self, root: Path) -> tuple[Any, Any]:
        """(project, context) of a project whose inputs are right; Unloadable otherwise."""
        lay = layout(root)
        lp = self.load(root, lay)
        ctx = self.context(lp)
        if lp.project is None or ctx is None:
            raise Unloadable(self.check(root, lay))
        return lp.project, ctx

    # ------------------------------------------------------------------ stage files
    @staticmethod
    def stage_files(root: Path) -> list[tuple[str, Path]]:
        """(stage, file) of every stage file, in the order of dossier.project.Project.stage_files."""
        core = _dossier()[2]
        out = [(s, root / "out" / f"{s}.json") for s in core.STAGE_FILES if (root / "out" / f"{s}.json").is_file()]
        out += [("analyze", p) for p in sorted((root / "out" / "analyses").glob("*/analysis.json"))]
        out += [("agent", p) for p in sorted((root / "agent").glob("*.json"))]
        return out

    def derived(self, root: Path) -> Derived:
        """The records of every stage file, validated as Project.collection validates them. A record that
        does not match the data model is kept as it is written, with a warning: one bad record of an
        engine does not hide the rest."""
        core = _dossier()[2]
        stage_files = self.stage_files(root)
        sig = _stat_sig(p for _, p in stage_files)
        with self.lock:
            hit = self._derived.get(str(root))
            if hit is not None and hit.sig == sig:
                self._derived.move_to_end(str(root))
                return hit
            dv = Derived(sig, collections={k: [] for k in core.COLLECTIONS})
            for stage, path in stage_files:
                rel = path.relative_to(root).as_posix()
                doc = _json(path)
                info = {"rel": rel, "stage": stage, "mtime": path.stat().st_mtime if path.exists() else None,
                        "written": None, "inputs_digest": None, "options": None, "error": None, "facts": {}}
                dv.files.append(info)
                if not isinstance(doc, dict):
                    info["error"] = "The file is not valid JSON."
                    dv.warnings.append(f"{rel}: the file is not valid JSON (a run writes it now, or it is damaged)")
                    continue
                info.update(written=doc.get("written"), inputs_digest=doc.get("inputs_digest"), options=doc.get("options"))
                if stage == "context" and isinstance(doc.get("context"), dict):
                    info["facts"] = {"parameters": len(doc["context"].get("params") or {}),
                                     "effects": len(doc["context"].get("effects") or [])}
                if stage == "geometry" and isinstance(doc.get("geometry"), dict):
                    g = doc["geometry"]
                    info["facts"] = {"features": len(g.get("features") or {}), "faces": (g.get("cad") or {}).get("n_faces"),
                                     "problems": list(g.get("problems") or []), "pictures": list(g.get("pictures") or [])}
                for key, cls in core.COLLECTIONS.items():
                    items = doc.get(key)
                    for i, raw in enumerate(items if isinstance(items, list) else []):
                        try:
                            rec = cls.model_validate(raw)
                            data, digest = rec.model_dump(mode="json", by_alias=True), core.digest_of(rec)
                        except Exception as e:  # noqa: BLE001 - pydantic.ValidationError, mostly
                            if not isinstance(raw, dict):
                                dv.warnings.append(f"{rel}: {key}[{i}] is not a record")
                                continue
                            why = " ".join(x.strip() for x in str(e).splitlines()[1:3])
                            dv.warnings.append(f"{rel}: {key}[{i}] does not match the data model: {why}")
                            data, digest = raw, core.digest_of(raw)
                        dv.collections[key].append(data)
                        if isinstance(data.get("id"), str):
                            dv.digests[data["id"]] = digest
                            dv.owner[data["id"]] = rel
            self._derived[str(root)] = dv
            while len(self._derived) > 8:
                self._derived.popitem(last=False)
            return dv

    # ------------------------------------------------------------------ reviews
    def reviews(self, root: Path, dv: Derived | None = None) -> dict[str, dict]:
        """The stored reviews: ``{record id: {state, by, date, comment, stale}}``. A review is stale when the
        record changed after the person saw it. A review of a record that is gone is stale and ``missing``."""
        dv = dv or self.derived(root)
        doc = _yaml(root / "reviews.yaml")
        stored = doc.get("reviews") if isinstance(doc, dict) else None
        out = {}
        for rid, rv in (stored.items() if isinstance(stored, dict) else []):
            if not isinstance(rv, dict):
                continue
            gone = str(rid) not in dv.digests
            item = {"state": rv.get("state") if rv.get("state") in REVIEW_STATES else "open",
                    "by": str(rv.get("by") or ""), "date": str(rv.get("date") or ""), "comment": str(rv.get("comment") or ""),
                    "stale": gone or (bool(rv.get("digest")) and rv["digest"] != dv.digests[str(rid)])}
            if gone:
                item["missing"] = True
            out[str(rid)] = item
        return out

    def set_review(self, root: Path, record: str, state: str, by: str, comment: str) -> dict:
        """Store a person's review with Project.set_review. ``open`` takes a review back: the record counts
        as not reviewed again, and reviews.yaml keeps who did that and when."""
        lay = layout(root)
        lp = self.load(root, lay)
        if lp.project is None:
            raise Unloadable(self.check(root, lay))
        with self.lock:
            try:
                rv = lp.project.set_review(record, state, by, comment)
            except KeyError:
                raise HTTPException(404, f"{record!r} is not a derived record of this project") from None
        return {"record": rv.record, "state": rv.state, "by": rv.by, "date": rv.date, "comment": rv.comment,
                "stale": False}

    # ------------------------------------------------------------------ tasks
    def tasks_of(self, folder: str, limit: int = 8) -> list[dict]:
        """The dossier tasks of a project, newest first; the newest one with what its log says."""
        if self.tasks is None:
            return []
        ref, path = project_ref(folder), str((self.root / folder).resolve())
        out = []
        for t in self.tasks.tasks.list(group=GROUP, limit=500):
            args = t.get("args") or {}
            if args.get("project") != ref and path not in (t.get("argv") or []):
                continue
            brief = {k: t.get(k) for k in ("id", "tool", "title", "state", "created_at", "started_at", "finished_at",
                                           "error", "cancel")}
            brief["args"] = ({"from": "report", "to": "report"} if t.get("tool") == "dossier.report" else
                             {k: args.get(k) for k in ("from", "to", "quick", "only", "force") if args.get(k) is not None})
            out.append(brief)
            if len(out) >= limit:
                break
        if out:
            try:
                out[0]["progress"] = run_progress(tail(self.tasks.tasks.log_path(out[0]["id"]), 400_000))
            except KeyError:
                pass
        return out

    def start(self, folder: str, tool: str, args: dict) -> dict:
        """Start a dossier tool on a project. One task per project at a time: two runs would write the same
        stage files."""
        if self.tasks is None:
            raise HTTPException(503, "this server has no task lane")
        if tool not in TOOL_NAMES:
            raise HTTPException(404, f"no project tool {tool!r}; tools: {', '.join(TOOL_NAMES)}")
        with self.lock:
            busy = next((t for t in self.tasks_of(folder, limit=50) if t["state"] in ("queued", "running")), None)
            if busy is not None:
                raise HTTPException(409, f"{busy['title']} is {busy['state']} for this project ({busy['id']}). "
                                         "Wait until it ends, or stop it.")
            try:
                return self.tasks.submit(tool, {**args, "project": project_ref(folder)})
            except ToolArgsError as e:
                raise HTTPException(422, {"detail": "invalid arguments", "errors": e.errors}) from None

    @staticmethod
    def _last_run(tasks: list[dict]) -> dict | None:
        """The newest task as the page shows it: where it is, or where it stopped and why."""
        if not tasks:
            return None
        t = tasks[0]
        prog = t.get("progress") or {"done": [], "current": None, "failed": None, "problems": [], "last": ""}
        run = {"task": t["id"], "tool": t["tool"], "title": t["title"], "state": t["state"], "args": t["args"],
               "started_at": t["started_at"], "finished_at": t["finished_at"], "done": prog["done"],
               "stage": None, "message": None, "problems": prog["problems"]}
        if t["state"] in ("queued", "running"):
            run["stage"] = prog["current"]
        elif t["state"] != "done":
            failed = prog["failed"] or {}
            run["stage"] = failed.get("stage") or prog["current"]
            if failed.get("message"):
                run["message"] = failed["message"]
            elif prog["problems"]:
                run["message"] = "The project has problems: " + " ".join(prog["problems"][:3])
            elif t["state"] == "failed":
                run["message"] = prog["last"] or t.get("error")
            else:
                run["message"] = t.get("error") or t["state"]
        return run

    # ------------------------------------------------------------------ status
    def _stages(self, root: Path, lp: Loaded, dv: Derived, run: dict | None) -> list[dict]:
        """One entry per pipeline stage: ``state`` is done | stale | not_run | partial (some load cases are
        not analysed) | running | failed (the last run stopped there; ``error`` says why)."""
        def fresh(f: dict) -> bool:        # as Project.stale: a file made from other inputs is stale
            return lp.digest is not None and f["inputs_digest"] in (None, lp.digest)

        by_stage: dict[str, list[dict]] = {}
        for f in dv.files:
            by_stage.setdefault(f["stage"], []).append(f)
        col, stamps, out = dv.collections, {}, []
        inputs_changed = "The inputs changed after this stage ran." if lp.digest else "The inputs have problems now."
        for s in STAGES:
            files = by_stage.get(s, [])
            st = {"stage": s, "state": "not_run", "file": f"out/{s}.json", "written": None, "summary": "", "error": None}
            stamps[s] = max((f["mtime"] or 0 for f in files), default=0)
            if s == "report":
                page = root / "out" / "report" / "index.html"
                st["file"] = "out/report/index.html"
                if page.is_file():
                    stamps[s] = page.stat().st_mtime
                    newer = any(f["stage"] != "agent" and (f["mtime"] or 0) > stamps[s] + 1 for f in dv.files)
                    stale = newer or any(x["state"] in ("stale", "failed", "running") for x in out)
                    st.update(state="stale" if stale else "done", written=_iso(stamps[s]),
                              summary="The stage files changed after the report was written." if stale else "")
            elif s == "analyze":
                st["file"] = "out/analyses/"
                if files:
                    cases = [c.get("id") for c in col.get("load_cases", [])]
                    seen = {f["rel"].split("/")[2] for f in files}
                    todo = [c for c in cases if c not in seen]
                    status = Counter(str(a.get("status")) for a in col.get("analyses", [])
                                     if dv.owner.get(a.get("id"), "").startswith("out/analyses/"))
                    parts = [f"{status.get('done', 0)} of {_plural(len(cases) or len(files), 'load case')} analysed"]
                    parts += [f"{n} {k.replace('_', ' ')}" for k, n in sorted(status.items()) if k != "done"]
                    parts += [f"{len(todo)} not analysed"] if todo else []
                    stale = not all(fresh(f) for f in files)
                    st.update(state="stale" if stale else "partial" if todo else "done", written=_iso(stamps[s]),
                              summary=", ".join(parts) + "." + (f" {inputs_changed}" if stale else ""))
            elif files:
                f = files[0]
                own = lambda key: [r for r in col.get(key, []) if dv.owner.get(r.get("id")) == f["rel"]]  # noqa: E731
                if s == "context":
                    text = f"{f['facts'].get('parameters', 0)} named numbers, {f['facts'].get('effects', 0)} effects"
                elif s == "geometry":
                    n = len(f["facts"].get("problems") or [])
                    text = (f"{f['facts'].get('features', 0)} features on {f['facts'].get('faces') or '?'} faces"
                            + (f", {_plural(n, 'problem')}" if n else ""))
                elif s == "validate":
                    sev = Counter(str(r.get("severity")) for r in own("findings"))
                    text = ", ".join(f"{sev[k]} {k}" for k in SEVERITIES if sev.get(k))
                    text = f"findings: {text}" if text else "no findings"
                elif s == "plan":
                    cases = col.get("load_cases", [])
                    scaled = sum(1 for c in cases if c.get("status") == "scaled")
                    text = (_plural(len(cases), "load case") + (f" ({scaled} scaled)" if scaled else "")
                            + f", {_plural(len(col.get('hand_calcs', [])), 'hand calculation')}")
                else:
                    v = Counter(str(x.get("verdict")) for x in col.get("verdicts", []))
                    text = ", ".join(f"{v[k]} {k.replace('_', ' ')}" for k in VERDICTS if v.get(k)) or "no verdicts"
                stale = f["error"] is not None or not fresh(f)
                st.update(state="stale" if stale else "done", written=f["written"] or _iso(f["mtime"]), file=f["rel"],
                          summary=f["error"] or text + "." + (f" {inputs_changed}" if stale else ""))
            out.append(st)
        if run and run.get("stage"):
            st = next(x for x in out if x["stage"] == run["stage"])
            if run["state"] in ("queued", "running"):
                st["state"] = "running"
            elif run["state"] != "done" and stamps[run["stage"]] <= (_epoch(run["finished_at"]) or time.time()) + 1:
                st.update(state="failed", error=run["message"])      # no later run made the stage good
        return out

    def status(self, folder: str, detail: bool = False) -> dict:
        root = self.dir(folder)
        lay = layout(root)
        lp = self.load(root, lay)
        self.context(lp)
        dv = self.derived(root)
        tasks = self.tasks_of(folder)
        run = self._last_run(tasks)
        doc, p = lay["doc"], lp.project
        lines = lp.problems if p is None else (lp.ctx_problems or [])
        subject_id = p.file.subject if p else doc.get("subject")
        comp = next((c for c in lay["components"] if c["id"] == subject_id), None) or {}
        col = dv.collections
        verdicts = Counter(str(x.get("verdict")) for x in col.get("verdicts", []))
        severity = Counter(str(x.get("severity")) for x in col.get("findings", []))
        reviews, progress = self.reviews(root, dv), Counter()
        for key in REVIEWED:
            for rec in col.get(key, []):
                rv = reviews.get(rec.get("id"), {})
                progress["total"] += 1
                progress["stale" if rv.get("stale") else rv.get("state", "open")] += 1
        report = root / "out" / "report" / "index.html"
        out: dict[str, Any] = {
            "folder": folder, "id": p.file.id if p else doc.get("id"),
            "name": (p.file.name if p else doc.get("name")) or folder,
            "description": _text(p.file.description if p else doc.get("description")),
            "author": (p.file.author if p else doc.get("author")) or "",
            "subject": {"id": subject_id, "name": p.subject.name if p else comp.get("name"),
                        "material": p.subject.material if p else comp.get("material"),
                        "drawing": p.subject.drawing if p else None},
            "ok": not lines, "n_problems": len(lines),
            "stages": self._stages(root, lp, dv, run),
            "stale": [f["rel"] for f in dv.files if lp.digest is None or f["inputs_digest"] not in (None, lp.digest)],
            "verdicts": {k: verdicts.get(k, 0) for k in VERDICTS},
            "findings": {k: severity.get(k, 0) for k in SEVERITIES},
            "reviews": {k: progress.get(k, 0) for k in ("total", "open", "accepted", "rejected", "changes_requested", "stale")},
            "report": {"built": report.is_file(), "written": _iso(report.stat().st_mtime) if report.is_file() else None},
            "last_run": run, "active_task": next((t["id"] for t in tasks if t["state"] in ("queued", "running")), None),
        }
        if not detail:
            return out
        texts = _Texts(root, lay)
        titles = {r.id: r.title for r in p.reqs()} if p is not None else {}
        geometry = next((f for f in dv.files if f["stage"] == "geometry"), None) or {}
        out.update(
            problems=[explain(x, root, lay, texts) for x in lines],
            root=str(root.resolve()), command=f"python -m dossier run {root.resolve()}", warnings=dv.warnings,
            tasks=[{k: v for k, v in t.items() if k != "progress"} for t in tasks],
            counts=({"elements": len(p.system.elements), "connections": len(p.system.connections),
                     "phases": len(p.system.phases), "duty_cycles": len(p.system.duty_cycles),
                     "features": len(p.subject.features), "interfaces": len(p.subject.interfaces),
                     "requirements": len(p.reqs())} if p is not None else None),
            verdict_list=[{**x, "title": titles.get(x.get("requirement"), "")} for x in col.get("verdicts", [])],
            load_cases=self._load_cases(dv, lp),
            pictures=geometry.get("facts", {}).get("pictures", []))
        return out

    @staticmethod
    def _load_cases(dv: Derived, lp: Loaded) -> list[dict]:
        """The plan, and for each load case the analysis that was made of it."""
        analyses = {a.get("load_case"): a for a in dv.collections.get("analyses", [])}
        files = {f["rel"]: f for f in dv.files}
        out = []
        for c in dv.collections.get("load_cases", []):
            item = {k: c.get(k) for k in ("id", "title", "analysis_type", "phase", "status", "scaled_from", "scale",
                                          "uses", "hand_calcs", "note", "statement")}
            item["requirements"] = sorted({x.get("requirement") for x in c.get("criteria") or [] if x.get("requirement")})
            item["loads"] = len(c.get("loads") or [])
            item["analysis"] = None
            a = analyses.get(c.get("id"))
            if a is not None:
                f = files.get(dv.owner.get(a.get("id"), ""), {})
                results = Counter(str(x.get("status")) for x in (a.get("results") or {}).values() if isinstance(x, dict))
                checks = Counter("passed" if x.get("passed") else "failed" if x.get("passed") is False else "not_evaluated"
                                 for x in a.get("checks") or [])
                item["analysis"] = {
                    "id": a.get("id"), "status": a.get("status"), "message": a.get("message") or "",
                    "runtime_s": a.get("runtime_s"), "solver": a.get("solver") or "", "results": dict(results),
                    "checks": dict(checks), "nodes": [m.get("nodes") for m in a.get("mesh") or []],
                    "run_ids": a.get("run_ids") or [], "written": f.get("written"),
                    "quick": bool((f.get("options") or {}).get("quick")),
                    "stale": lp.digest is None or f.get("inputs_digest") not in (None, lp.digest)}
            out.append(item)
        return out


# --------------------------------------------------------------------------- pages and files


def _page(html: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(html, status_code=status, headers={
        "Cache-Control": "no-store", "Content-Security-Policy": PAGE_CSP, "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff"})


def _not_built(folder: str, command: str) -> HTMLResponse:
    return _page(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>No report yet</title>
<style>body {{ font: 16px/1.55 system-ui, sans-serif; color: #0b0b0b; background: #f9f9f7; margin: 0; padding: 48px 32px; }}
main {{ max-width: 62ch; margin: 0 auto; }} h1 {{ font-size: 22px; margin: 0 0 10px; }} p {{ color: #52514e; }}
code {{ font: 13.5px ui-monospace, Menlo, Consolas, monospace; background: #f3f2ee; padding: 2px 6px; border-radius: 5px; }}</style>
</head><body><main>
<h1>The report is not built yet</h1>
<p>The project {escape(folder)} has no report. Run the verification on the Projects page of the dashboard,
or build the report from the stage files that exist.</p>
<p><code>{escape(command)}</code></p>
</main></body></html>""", status=404)


def inject(html: str, folder: str, token: bool = False) -> str:
    """The report page with ``window.DOSSIER_API`` set before ``</head>`` (dossier/CONTRACTS.md): this turns
    the page's review controls on. With a token on the request, also the script that adds the token to the
    page's own requests."""
    tag = f"<script>window.DOSSIER_API = {json.dumps(f'/api/dossier/{folder}')};</script>" + (TOKEN_SHIM if token else "")
    m = re.search(r"</head\s*>", html, re.I)
    return html[:m.start()] + tag + html[m.start():] if m else tag + html


def out_file(root: Path, rel: str, types: dict[str, str], base: str = "out") -> Path | None:
    """A derived file inside ``<project>/<base>/`` of a type the dashboard serves, or None."""
    p = within(root / base, clean_rel(rel))
    return p if p is not None and p.is_file() and p.suffix.lower() in types else None


def _send(path: Path, types: dict[str, str]) -> FileResponse:
    if path.stat().st_size > MAX_FILE:
        raise HTTPException(413, "the file is too large to serve")
    headers = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
    if path.suffix.lower() == ".svg":
        headers["Content-Security-Policy"] = SVG_CSP
    return FileResponse(path, media_type=types[path.suffix.lower()], headers=headers)


# --------------------------------------------------------------------------- API


def router(store, tasks: TaskRunner | None = None) -> APIRouter:
    projects = Projects(store, tasks)
    r = APIRouter()

    def find(folder: str) -> Path:
        try:
            return projects.dir(folder)
        except KeyError:
            raise HTTPException(404, f"no project {folder!r} in {projects.root}") from None

    def refused(e: Unloadable) -> JSONResponse:
        return JSONResponse({"detail": "The inputs of the project have problems. Correct them first.",
                             "problems": e.check["problems"], "stage": e.check.get("stage")}, status_code=422)

    # ------------------------------------------------------------------ list, create, status
    @r.get("/api/dossier")
    def listing():
        folders, ignored = projects.folders()
        items = []
        for f in folders:
            try:
                items.append(projects.status(f))
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001 - one project that cannot be read must not hide the others
                items.append({"folder": f, "name": f, "ok": False, "error": f"{type(e).__name__}: {e}"})
        return {"projects": items, "examples": projects.examples(), "ignored": ignored,
                "root": str(projects.root.resolve()), "stages": list(STAGES)}

    @r.post("/api/dossier", status_code=201)
    def create(body: dict = Body(...)):
        folder = projects.create(str(body.get("example") or ""), body.get("folder"))
        return projects.status(folder, detail=True)

    @r.get("/api/dossier/{project}")
    def detail(project: str):
        find(project)
        return projects.status(project, detail=True)

    @r.post("/api/dossier/{project}/check")
    def check(project: str):
        return projects.check(find(project))

    @r.post("/api/dossier/{project}/tasks", status_code=202)
    def start(project: str, body: dict = Body(...)):
        find(project)
        tool, args = body.get("tool"), body.get("args") or {}
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise HTTPException(422, 'the body must be {"tool": name, "args": {...}}')
        return projects.start(project, tool, {k: v for k, v in args.items() if k != "project"})

    # ------------------------------------------------------------------ input files
    def input_file(root: Path, path: str) -> tuple[dict, Path]:
        entry = next((f for f in layout(root)["files"] if f["path"] == path and f["inside"]), None)
        if entry is None:
            raise HTTPException(404, f"{path!r} is not an input file of this project")
        return entry, root / entry["path"]

    @r.get("/api/dossier/{project}/files")
    def files(project: str):
        from feacore.problems import custom

        lay = layout(find(project))
        return {"files": lay["files"], "components": lay["components"], "notes": lay["notes"],
                "max_cad_bytes": custom.max_bytes(), "max_text_bytes": MAX_TEXT}

    @r.get("/api/dossier/{project}/files/{path:path}")
    def read_file(project: str, path: str):
        entry, p = input_file(find(project), path)
        if not p.is_file():
            raise HTTPException(404, f"{path} does not exist")
        if entry["kind"] == "cad":
            return FileResponse(p, filename=p.name, media_type="application/step",
                                headers={"X-Content-Type-Options": "nosniff"})
        if not entry["editable"]:
            raise HTTPException(422, f"{path} is not a YAML file")
        data = p.read_bytes()
        if len(data) > MAX_TEXT:
            raise HTTPException(413, f"{path} is larger than {MAX_TEXT // 1_000_000} MB. Edit it on the server.")
        return {**entry, "text": data.decode("utf-8", "replace"), "sha256": _sha(data)}

    @r.put("/api/dossier/{project}/files/{path:path}")
    async def write_file(project: str, path: str, request: Request):
        """The body is the new text, or {"text": ..., "expect": sha256 of the text that was edited}. The file
        is saved as it is, also when it is wrong: the answer is the check of the project."""
        root = find(project)
        raw = await request.body()
        if len(raw) > MAX_TEXT + 200:
            raise HTTPException(413, f"the text is larger than {MAX_TEXT // 1_000_000} MB")
        expect = None
        if request.headers.get("content-type", "").split(";")[0].strip() == "application/json":
            try:
                body = json.loads(raw)
            except ValueError:
                raise HTTPException(422, "the body is not valid JSON") from None
            if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                raise HTTPException(422, 'the body must be {"text": "..."} or the text itself')
            text, expect = body["text"], body.get("expect")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(422, "the text is not UTF-8") from None
        if "\x00" in text:
            raise HTTPException(422, "the text has a NUL character")

        def save() -> dict:
            import yaml

            entry, p = input_file(root, path)
            if not entry["editable"]:
                raise HTTPException(422, f"{path} is not a YAML file of this project. Only the YAML inputs are "
                                         "edited here.")
            try:
                doc = yaml.safe_load(text)
            except yaml.YAMLError:
                doc = None                          # saved as it is: the check gives the line
            if isinstance(doc, dict):               # what the new text names must stay inside the project
                if entry["role"] == "project":
                    names = [(k, doc.get(k)) for k in ("system", "materials", "policy", "terms")]
                    comps = doc.get("components")
                    names += [("components", c) for c in (comps if isinstance(comps, list) else [])]
                    base = root
                else:
                    names = [(f"{w}cad.file", h["cad"].get("file")) for w, h in (("", doc), ("flow.", doc.get("flow")))
                             if entry["role"] == "component" and isinstance(h, dict) and isinstance(h.get("cad"), dict)]
                    base = p.parent
                bad = [f"{k}: {v!r}" for k, v in names if isinstance(v, str) and within(base, clean_rel(v)) is None]
                if bad:
                    raise HTTPException(422, f"Not saved: {path} names a file outside the project folder "
                                             f"({'; '.join(bad)}). {OUTSIDE}")
            data = text.encode("utf-8")
            with projects.lock:
                if expect is not None and p.is_file() and _sha(p.read_bytes()) != expect:
                    raise HTTPException(409, f"{path} changed on the server after you opened it. Load it again, "
                                             "then make your change again.")
                _write(p, data)
            lay = layout(root)
            saved = next(f for f in lay["files"] if f["path"] == path)
            return {**projects.check(root, lay), "saved": {**saved, "sha256": _sha(data)}}

        return await run_in_threadpool(save)

    @r.post("/api/dossier/{project}/cad/{component}")
    async def replace_cad(project: str, component: str, request: Request):
        """Multipart ``file``: the new STEP file of a component (its id, or the name of its folder)."""
        from feacore.problems import custom

        root = find(project)
        lay = layout(root)
        comp = next((c for c in lay["components"] if component in (c["id"], Path(c["dir"]).name)), None)
        if comp is None:
            known = ", ".join(str(c["id"] or c["dir"]) for c in lay["components"]) or "none"
            raise HTTPException(404, f"no component {component!r} in this project (components: {known})")
        if not comp["cad"] or Path(comp["cad"]).suffix.lower() not in CAD_SUFFIXES:
            raise HTTPException(422, f"component.yaml of {component} names no STEP file (cad.file) in its folder")
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
            custom.check_step(data)
        except custom.CadError as e:
            raise HTTPException(413 if len(data) > limit else 422, str(e)) from None

        def save() -> dict:
            target = root / comp["cad"]
            with projects.lock:
                before = {"bytes": target.stat().st_size, "sha256": _sha(target.read_bytes())} if target.is_file() else None
                _write(target, data)
            lay2 = layout(root)
            saved = next(x for x in lay2["files"] if x["path"] == comp["cad"])
            return {**projects.check(root, lay2), "component": comp["id"], "replaced": before,
                    "saved": {**saved, "sha256": _sha(data), "uploaded_as": custom.clean_name(f.filename)}}

        return await run_in_threadpool(save)

    # ------------------------------------------------------------------ numbers, records, trace
    @r.get("/api/dossier/{project}/params")
    def params(project: str):
        try:
            _, ctx = projects.usable(find(project))
        except Unloadable as e:
            return refused(e)
        used: dict[str, list[str]] = {}
        for owner, deps in [*((n, x.depends_on) for n, x in ctx.params.items()), *((x.id, x.depends_on) for x in ctx.effects)]:
            for dep in deps:
                if owner not in used.setdefault(dep, []):
                    used[dep].append(owner)
        return {"params": [{**prm.model_dump(mode="json"), "used_by": used.get(name, [])}
                           for name, prm in ctx.params.items()],
                "effects": len(ctx.effects), "duty_cycles": dict(ctx.duty_cycles)}

    @r.get("/api/dossier/{project}/records")
    def records(project: str, collection: str | None = None):
        dv = projects.derived(find(project))
        if collection is not None and collection not in dv.collections:
            raise HTTPException(404, f"no collection {collection!r}; collections: {', '.join(dv.collections)}")
        return {"collections": {k: v for k, v in dv.collections.items() if collection in (None, k)},
                "counts": {k: len(v) for k, v in dv.collections.items()},
                "files": [{k: f[k] for k in ("rel", "stage", "written")} for f in dv.files], "warnings": dv.warnings}

    @r.get("/api/dossier/{project}/trace")
    def trace_graph(project: str):
        try:
            p, ctx = projects.usable(find(project))
        except Unloadable as e:
            return refused(e)
        with projects.lock:
            try:
                return _dossier()[3].graph(p, ctx)
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001 - a stage file an engine wrote wrong: say which, not a bare 500
                raise HTTPException(500, f"dossier.trace.graph could not read the records: {type(e).__name__}: {e}") from None

    # ------------------------------------------------------------------ reviews (dossier/CONTRACTS.md)
    @r.get("/api/dossier/{project}/reviews")
    def reviews(project: str):
        return {"reviews": projects.reviews(find(project))}

    @r.post("/api/dossier/{project}/reviews")
    def review(project: str, body: dict = Body(...)):
        root = find(project)
        record, state = body.get("record"), body.get("state")
        by, comment = str(body.get("by") or "").strip(), str(body.get("comment") or "").strip()
        if not isinstance(record, str) or not record:
            raise HTTPException(422, "record: give the id of the record")
        if state not in REVIEW_STATES:
            raise HTTPException(422, f"state must be one of {', '.join(REVIEW_STATES)}")
        if state != "open" and not by:
            raise HTTPException(422, "by: give the name of the person who reviews the record")
        if len(by) > 200 or len(comment) > 4000 or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", by + comment):
            raise HTTPException(422, "by or comment: too long, or with control characters")
        try:
            return projects.set_review(root, record, state, by, comment)
        except Unloadable as e:
            return refused(e)

    # ------------------------------------------------------------------ report and derived files
    @r.get("/api/dossier/{project}/report", include_in_schema=False)
    def report_root(project: str, request: Request):
        find(project)
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"/api/dossier/{project}/report/index.html{query}", status_code=307)

    @r.get("/api/dossier/{project}/report/{path:path}")
    def report(project: str, path: str, request: Request):
        """A file of out/report/. index.html is the report page, served with window.DOSSIER_API set so that its
        review controls work. A picture that the page names by its place under out/ is found there."""
        from feacore.ui.server import _request_token

        root = find(project)
        if path in ("", "index.html"):
            page = root / "out" / "report" / "index.html"
            if not page.is_file():
                return _not_built(project, f"python -m dossier run {root.resolve()} --from report --to report")
            return _page(inject(page.read_text(errors="replace"), project, bool(_request_token(request))))
        p = out_file(root, path, REPORT_TYPES, base="out/report") or out_file(root, path, PICTURES)
        if p is None:
            raise HTTPException(404, f"no file {path!r} in the report of {project}")
        return _send(p, REPORT_TYPES)

    @r.get("/api/dossier/{project}/out/{path:path}")
    def out(project: str, path: str):
        p = out_file(find(project), path, OUT_TYPES)
        if p is None:
            raise HTTPException(404, f"no file {path!r} in out/ of {project} (pictures, JSON and text files only)")
        return _send(p, OUT_TYPES)

    @r.get("/api/dossier/{project}/{path:path}", include_in_schema=False)
    def beside_report(project: str, path: str):
        """The report page is at .../report/index.html, so a picture it names by its place on disk beside the
        report folder (../analyses/LC-X/stress.png) arrives here. Pictures of out/ only."""
        p = out_file(find(project), path, PICTURES)
        if p is None:
            raise HTTPException(404, f"no such page or picture: {path!r}")
        return _send(p, PICTURES)

    return r
