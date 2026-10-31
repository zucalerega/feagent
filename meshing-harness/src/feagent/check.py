"""Check a mesh spec: mesh (child process) -> quality gates -> renders -> report.

``run_check`` writes ``<spec out dir>/report.json`` (machine-readable, schema in
docs/report-schema.md) and ``REPORT.md`` (gate table first), and records the result in the
state file the Stop hook reads (see ``state_path`` and ``record_state``).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import Gate, MeshResult, _Base, _jsonable, overall_status

if TYPE_CHECKING:  # pragma: no cover
    from .quality import QualityReport
    from .render import RenderResult
    from .spec import MeshSpec

SCHEMA_VERSION = 1
STATE_ENV = "FEAGENT_STATE"
_ROOT_MARKERS = (".feagent", "pyproject.toml")
_RENDERS = ("groups.png", "quality.png")


@dataclass
class CheckReport(_Base):
    spec_path: str
    status: str  # "pass" | "fail"
    gates: list[Gate]  # mesher gates, then quality gates
    mesh: MeshResult  # geometry stripped (it is in <out>/manifest.json)
    quality: dict | None  # QualityReport.to_dict() without its gates; None if no mesh
    bcs: dict | None  # {"fea": {...}} | {"cfd": {...}} from the spec's bcs: section
    renders: list[RenderResult]
    created_at: str  # ISO-8601 UTC

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **super().to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> CheckReport:
        from .render import RenderResult

        d = {k: v for k, v in d.items() if k != "schema_version"}
        d["gates"] = [Gate.from_dict(g) for g in d["gates"]]
        d["mesh"] = MeshResult.from_dict(d["mesh"])
        d["renders"] = [RenderResult.from_dict(r) for r in d["renders"]]
        return cls(**d)


# --- orchestration ----------------------------------------------------------

def run_check(spec_path: str | Path, render: bool = True) -> CheckReport:
    """Mesh ``spec_path`` in a child process, assess quality, render, write and record a report.

    Raises SpecError (before meshing, recorded as state "fail") if the spec is invalid."""
    from .mesher import run_mesh_subprocess

    spec, mtime = load_checked_spec(spec_path)
    mesh = run_mesh_subprocess(spec.path)
    qrep = _assess(mesh, spec)
    out = spec.out_dir()
    renders = _renders(mesh, qrep, out) if render and qrep is not None else []
    report = _report(spec, mesh, qrep, renders)
    write_report(report, out)
    record_state(str(spec.path), report.status, spec_mtime=mtime,
                 report_created_at=report.created_at)
    return report


def load_checked_spec(spec_path: str | Path) -> tuple[MeshSpec, float | None]:
    """``load_spec`` plus the spec file's mtime, taken *before* reading it.

    Taking it first means an edit during the run leaves a recorded mtime older than the file, so
    the result reads as stale. An invalid (existing) spec is recorded as state "fail" with the
    error before the SpecError is re-raised, so an earlier "pass" cannot outlive it."""
    from .errors import SpecError
    from .spec import load_spec

    mtime = spec_mtime(Path(spec_path))
    try:
        return load_spec(spec_path), mtime
    except SpecError as e:
        if Path(spec_path).is_file():
            record_state(str(spec_path), "fail", spec_mtime=mtime, error=str(e))
        raise


def _assess(mesh: MeshResult, spec: MeshSpec) -> QualityReport | None:
    if not mesh.ok or not mesh.mesh_path:
        return None
    from .quality import assess_mesh

    return assess_mesh(mesh.mesh_path, mesh, spec.quality, spec.analysis, spec.optimize)


def _renders(mesh: MeshResult, qrep: QualityReport, out: Path) -> list[RenderResult]:
    from .meshdata import read_msh
    from .render import render_groups, render_quality

    md = read_msh(mesh.mesh_path)
    groups, quality = (out / name for name in _RENDERS)
    return [render_groups(md, groups), render_quality(md, qrep.sicn_per_tet, quality)]


def _report(spec: MeshSpec, mesh: MeshResult, qrep: QualityReport | None,
            renders: list[RenderResult]) -> CheckReport:
    gates = list(mesh.gates) + (list(qrep.gates) if qrep is not None else [])
    status = "pass" if mesh.ok and overall_status(gates) == "pass" else "fail"
    quality = None if qrep is None else {k: v for k, v in qrep.to_dict().items() if k != "gates"}
    return CheckReport(
        spec_path=str(spec.path.resolve()), status=status, gates=gates,
        mesh=replace(mesh, geometry=None), quality=quality, bcs=bcs_dict(spec),
        renders=renders, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def bcs_dict(spec: MeshSpec) -> dict | None:
    """The spec's ``bcs:`` section as plain JSON data, or None if it declares none."""
    found = {"fea": spec.bcs_fea, "cfd": spec.bcs_cfd}
    out = {kind: _jsonable(asdict(b)) for kind, b in found.items() if b is not None}
    return out or None


def write_report(r: CheckReport, out: Path) -> None:
    """Write ``report.json`` and ``REPORT.md`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(r.to_dict(), indent=1) + "\n")
    (out / "REPORT.md").write_text(report_markdown(r))


# --- REPORT.md --------------------------------------------------------------

def report_markdown(r: CheckReport) -> str:
    """Gate table first (status, id, value, threshold, message, hint), then numbers, BCs, files."""
    sections = [_md_title(r), _md_gates(r.gates), _md_mesh(r.mesh), _md_quality(r.quality),
                _md_bcs(r.bcs), _md_files(r), _md_problems(r.mesh)]
    return "\n\n".join(s for s in sections if s) + "\n"


def _md_title(r: CheckReport) -> str:
    counts = {s: sum(g.status == s for g in r.gates) for s in ("pass", "warn", "fail", "skip")}
    tally = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    return (f"# Mesh check: {Path(r.spec_path).parent.name} — {r.status.upper()}\n\n"
            f"spec: `{r.spec_path}`  \nchecked: {r.created_at}  \ngates: {tally or 'none'}")


def _cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        text = str(int(x)) if x.is_integer() and abs(x) < 1e15 else f"{x:.4g}"
    else:
        text = str(x)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _table(header: tuple[str, ...], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def gate_table(gates: list[Gate]) -> str:
    """Markdown table: status, id, value, threshold, message, hint (one row per gate)."""
    rows = [(g.status, g.id, g.value, g.threshold, g.message, g.hint) for g in gates]
    return _table(("status", "id", "value", "threshold", "message", "hint"), rows)


def _md_gates(gates: list[Gate]) -> str:
    return "## Gates\n\n" + gate_table(gates)


def _md_mesh(m: MeshResult) -> str:
    if not m.mesh_path:
        return "## Mesh\n\nno mesh was written (see the failing gates above)"
    head = (f"## Mesh\n\n{m.n_nodes} nodes, {m.n_elements} tets, order {m.order}, "
            f"size scale {m.size_scale:g}, {m.elapsed_s:g} s")
    rows = [(g.name, g.dim, len(g.tags), g.elements, g.area) for g in m.groups.values()]
    return head + "\n\n" + _table(("group", "dim", "entities", "elements", "area"), rows)


def _md_quality(q: dict | None) -> str:
    if q is None:
        return ""
    lines = [f"- tets: {q['n_tets']}, inverted: {q['inverted']}",
             _stat_line("SICN", q["sicn"]), _stat_line("gamma", q["gamma"]),
             f"- volume: CAD {q['volume_cad']:.6g}, mesh {q['volume_mesh']:.6g}, "
             f"relative error {q['volume_error']:.3g}",
             f"- components: {q['components']} (expected {q['expected_components']})"]
    lines += [_stat_line(k, q[k]) for k in ("non_ortho", "skewness") if q.get(k)]
    lines += [f"- group `{n}` area: CAD {c:.6g}, mesh {m:.6g}, relative error {e:.3g}"
              for n, (c, m, e) in q["group_area"].items()]
    return "## Quality\n\n" + "\n".join(lines)


def _stat_line(name: str, s: dict) -> str:
    return (f"- {name}: min {s['min']:.4g}, p01 {s['p01']:.4g}, median {s['median']:.4g}, "
            f"max {s['max']:.4g}")


def _md_bcs(bcs: dict | None) -> str:
    head = "## Boundary conditions (for the solver)\n\n"
    if not bcs:
        return head + ("none declared — add a `bcs:` section (bcs.fea: fixed/loads/E/nu or "
                       "bcs.cfd: inlet/outlet/velocity) naming the mesh groups.")
    lines = [f"- `{kind}.{k}`: {json.dumps(v)}" for kind, d in bcs.items() for k, v in d.items()]
    note = "Groups are exported under the same names (physical groups; NSET/ELSET in .inp)."
    return head + "\n".join(lines) + "\n\n" + note


def _md_files(r: CheckReport) -> str:
    lines = [f"- {k}: `{p}`" for k, p in r.mesh.exports.items()]
    lines += [f"- render `{x.path}`: {x.caption}" for x in r.renders]
    if r.mesh.out_dir:
        manifest = Path(r.mesh.out_dir) / "manifest.json"
        lines.append(f"- manifest (incl. geometry analysis): `{manifest}`")
    return "## Files\n\n" + "\n".join(lines) if lines else ""


def _md_problems(m: MeshResult) -> str:
    parts = [f"## Errors\n\n```\n{e}\n```" for e in m.errors]
    if m.warnings:
        parts.append("## Warnings\n\n" + "\n".join(f"- {w}" for w in m.warnings))
    return "\n\n".join(parts)


# --- text summaries (CLI and MCP share these) --------------------------------

def mesh_summary(res: MeshResult) -> str:
    """A mesh run: head line, mesher gate table, errors, mesher notes."""
    head = (f"mesh ok: {res.n_elements} tets, {res.n_nodes} nodes -> {res.mesh_path}" if res.ok
            else "mesh FAILED")
    return "\n\n".join([head, gate_table(res.gates), *res.errors, *_notes(res.warnings)])


def automesh_summary(r: CheckReport, mcp: bool = False) -> str:
    """An automesh run: spec path, status, gate table, mesher notes, report path, next step
    (worded as MCP tool calls when ``mcp``, else as CLI commands)."""
    spec = r.spec_path
    lines = [f"spec: {spec} (auto-generated starting point: add groups, bcs, refine)",
             f"status: {r.status}", gate_table(r.gates), *_notes(r.mesh.warnings)]
    if r.mesh.out_dir:
        lines.append(f"report: {Path(r.mesh.out_dir) / 'REPORT.md'}")
    nxt = (f'inspect_geometry("{spec}") for features and hazards; name the BC faces as groups by '
           f'query (find_faces), add bcs:, then check_mesh("{spec}")' if mcp else
           f"`feagent inspect {spec}` for features and hazards; name the BC faces as groups by "
           f"query (`feagent faces`), add bcs:, then `feagent check {spec}`")
    return "\n\n".join(lines + [f"next: {nxt}"])


def _notes(warnings: list[str]) -> list[str]:
    return ["mesher notes:\n" + "\n".join(f"- {w}" for w in warnings)] if warnings else []


# --- state (read by the Stop hook) ------------------------------------------
#
# {"specs": {<abs spec path>: entry}, ...other top-level keys kept (e.g. "hook_blocks")}
# entry: status ("unchecked" | "fail" | "pass"), checked_at and spec_mtime (epoch seconds; the
# spec's mtime when the recorded run read it), error (spec error text or null),
# report_created_at (created_at of the report.json this entry describes, or null),
# waiver / waiver_spec_mtime (a waiver is valid only while the spec's mtime still equals
# waiver_spec_mtime; a passing record clears it, other records keep it).

def state_path() -> Path:
    """``$FEAGENT_STATE``, else ``<root>/.feagent/state.json`` where root is the nearest ancestor
    of the cwd (itself included) that contains ``.feagent/`` or ``pyproject.toml``, else the cwd."""
    if os.environ.get(STATE_ENV):
        return Path(os.environ[STATE_ENV])
    cwd = Path.cwd().resolve()
    roots = [d for d in (cwd, *cwd.parents) if any((d / m).exists() for m in _ROOT_MARKERS)]
    return (roots[0] if roots else cwd) / ".feagent" / "state.json"


def load_state() -> dict:
    """The state file as ``{"specs": {abs_path: entry}, ...}`` (empty if missing or unreadable)."""
    try:
        data = json.loads(state_path().read_text())
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    data.setdefault("specs", {})
    return data


def record_state(spec_path: str, status: str, waiver: str | None = None, *,
                 spec_mtime: float | None = None, error: str | None = None,
                 report_created_at: str | None = None) -> dict:
    """Set and return the state entry of ``spec_path`` (format above); other entries are kept.

    ``spec_mtime`` defaults to the file's current mtime; pass the value taken before reading the
    spec to avoid recording an edit made during the run. ``waiver`` sets a waiver for that mtime;
    with no waiver a "pass" clears any earlier one and other statuses keep it."""
    p = Path(spec_path).resolve()
    mtime = _current_mtime(p) if spec_mtime is None else spec_mtime  # (the kwarg shadows the function)
    data = load_state()
    old = data["specs"].get(str(p), {})
    entry = {"status": status, "checked_at": time.time(), "spec_mtime": mtime, "error": error,
             "report_created_at": report_created_at, **_waiver_fields(old, status, waiver, mtime)}
    data["specs"][str(p)] = entry
    write_state(data)
    return entry


def _waiver_fields(old: dict, status: str, waiver: str | None, mtime: float | None) -> dict:
    if waiver is not None:
        return {"waiver": waiver, "waiver_spec_mtime": mtime}
    if status == "pass":
        return {"waiver": None, "waiver_spec_mtime": None}
    return {"waiver": old.get("waiver"), "waiver_spec_mtime": old.get("waiver_spec_mtime")}


def waive(spec_path: str, reason: str) -> dict:
    """Waive the spec's current result (status kept; "unchecked" entry if none) at its mtime."""
    key = str(Path(spec_path).resolve())
    entry = load_state()["specs"].get(key) or record_state(spec_path, "unchecked")
    entry.update(waiver=reason, waiver_spec_mtime=spec_mtime(Path(key)))
    data = load_state()
    data["specs"][key] = entry
    write_state(data)
    return entry


def write_state(data: dict) -> None:
    """Write the whole state file atomically (temp file + rename)."""
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def spec_mtime(p: Path) -> float | None:
    """The file's mtime (epoch seconds), None if it does not exist."""
    try:
        return p.stat().st_mtime
    except OSError:
        return None


_current_mtime = spec_mtime
