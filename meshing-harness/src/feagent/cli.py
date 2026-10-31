"""`feagent` command line: inspect CAD, query faces, render, mesh, auto-mesh and check specs.

Every command accepts ``--json`` (exactly one JSON document on stdout). Exit codes: 0 ok,
1 gate or mesh failure, 2 usage / spec / query / CAD error, 3 internal error. On 2 and 3 the
message and hint go to stderr and, with ``--json``, ``{"ok": false, "error", "hint"}`` to stdout.
Library imports are deferred so ``feagent --help`` stays fast and gmsh loads only when used.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .errors import FeagentError, QueryError

if TYPE_CHECKING:  # pragma: no cover
    from .meshdata import MeshData
    from .render import RenderResult
    from .types import GeometryModel

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERNAL = 0, 1, 2, 3
EXIT_CODES = ("exit codes: 0 ok; 1 a gate failed or no mesh; 2 usage, spec, query or CAD error; "
              "3 internal error (traceback on stderr)")
_SPEC_SUFFIXES = (".yaml", ".yml")
_QUERY_HINT = 'pass a JSON object, e.g. --query \'{"at": "xmin", "kind": "plane"}\''
_INTERNAL_HINT = "unexpected failure inside feagent (traceback above on stderr); please report it"


class _UsageError(Exception):
    def __init__(self, message: str, usage: str):
        super().__init__(message)
        self.message, self.usage = message, usage


class _Parser(argparse.ArgumentParser):
    """Raises _UsageError instead of exiting, so main() reports it like any error (incl. JSON)."""

    def error(self, message: str):  # type: ignore[override]
        raise _UsageError(message, self.format_usage().strip())


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``feagent`` console script; returns the process exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    as_json = "--json" in argv
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except _UsageError as e:
        return _error(as_json, e.message, e.usage, EXIT_USAGE)
    except FeagentError as e:
        return _error(as_json, e.message, e.hint or _default_hint(e), EXIT_USAGE)
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        return _error(as_json, f"internal: {type(e).__name__}: {e}", _INTERNAL_HINT, EXIT_INTERNAL)


def _default_hint(e: FeagentError) -> str:
    if not isinstance(e, QueryError):
        return ""
    from .select import QUERY_KEYS

    return f"fix the query value named above; query keys: {', '.join(QUERY_KEYS)}. {_QUERY_HINT}"


def _error(as_json: bool, message: str, hint: str, code: int) -> int:
    sys.stderr.write(f"error: {message}\n" + (f"hint: {hint}\n" if hint else ""))
    if as_json:
        _write({"ok": False, "error": message, "hint": hint}, as_json=True)
    return code


def _write(payload: Any, as_json: bool) -> None:
    sys.stdout.write((json.dumps(payload) if as_json else str(payload).rstrip("\n")) + "\n")
    sys.stdout.flush()


def _emit(args: argparse.Namespace, text: str, data: Any, code: int = EXIT_OK) -> int:
    _write(data if args.json else text, args.json)
    return code


# --- shared helpers ---------------------------------------------------------

def _is_spec(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _SPEC_SUFFIXES


def _load_model(path: str, size: float | None = None, heal: bool = False) -> GeometryModel:
    """Analyze a CAD file, or the domain a mesh spec builds (cfd external: box minus body)."""
    from .geometry import analyze, analyze_current_model

    if not _is_spec(path):
        return analyze(path, ref_size=size, heal=heal)
    from .gmsh_session import gmsh_session
    from .mesher import build_domain
    from .spec import load_spec

    spec = load_spec(path)
    with gmsh_session():
        build_domain(spec)
        return analyze_current_model(ref_size=size or spec.size.global_, source=str(spec.geometry))


def _query(text: str) -> dict:
    try:
        q = json.loads(text)
    except ValueError as e:
        raise QueryError(f"--query is not valid JSON: {e}", _QUERY_HINT) from None
    if not isinstance(q, dict):
        raise QueryError("--query must be a JSON object", _QUERY_HINT)
    return q


def _int_list(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated face tags, got {text!r}") \
            from None


def _mesh_file(path: str) -> Path:
    from .spec import load_spec

    p = load_spec(path).out_dir() / "mesh.msh" if _is_spec(path) else Path(path)
    if p.suffix.lower() != ".msh":
        raise FeagentError(f"not a gmsh mesh: {p}", "this view needs a .msh file or a mesh.yaml "
                           "(uses its out/mesh.msh); use --view faces for CAD files")
    if not p.is_file():
        raise FeagentError(f"mesh file not found: {p}", "run `feagent mesh SPEC` first")
    return p


# --- commands ---------------------------------------------------------------

def _cmd_parts(a: argparse.Namespace) -> int:
    from .parts import make_all

    paths = {name: str(p.resolve()) for name, p in make_all(a.outdir).items()}
    return _emit(a, "\n".join(paths.values()), paths)


def _cmd_inspect(a: argparse.Namespace) -> int:
    from .geometry import digest_text

    model = _load_model(a.path, a.size, a.heal)
    return _emit(a, digest_text(model, a.detail), model.to_dict())


def _cmd_faces(a: argparse.Namespace) -> int:
    from .select import describe_faces, explain_miss, select_faces

    q = _query(a.query)
    model = _load_model(a.path)
    tags = select_faces(model, q)
    if (len(tags) != a.expect) if a.expect is not None else not tags:
        raise QueryError(explain_miss(model, q, a.expect),
                         "adjust the query (see the per-key breakdown) or --expect; "
                         "`feagent render PATH --view faces` shows the face tags")
    text = f"{len(tags)} face(s) matched: {tags}\n" + describe_faces(model, tags)
    data = {"query": q, "tags": tags, "faces": [model.face(t).to_dict() for t in tags]}
    return _emit(a, text, data)


def _cmd_render(a: argparse.Namespace) -> int:
    p = Path(a.path).resolve()
    out = Path(a.output) if a.output else p.parent / f"{p.stem}_{a.view}.png"
    res = _VIEWS[a.view](a, out)
    return _emit(a, f"{res.path}\n{res.caption}", res.to_dict())


def _render_faces(a: argparse.Namespace, out: Path) -> RenderResult:
    from .render import render_faces
    from .select import select_faces

    q = _query(a.query) if a.query else None
    model = _load_model(a.path)
    picked = select_faces(model, q) if q is not None else []
    return render_faces(model, out, highlight=sorted(set(a.highlight or []) | set(picked)))


def _render_groups(a: argparse.Namespace, out: Path) -> RenderResult:
    from .meshdata import read_msh
    from .render import render_groups

    return render_groups(read_msh(_mesh_file(a.path)), out)


def _render_quality(a: argparse.Namespace, out: Path) -> RenderResult:
    from .meshdata import read_msh
    from .render import render_quality

    path = _mesh_file(a.path)
    md = read_msh(path)
    return render_quality(md, _sicn(path, md), out)


def _sicn(path: Path, md: MeshData) -> Any:
    import gmsh

    from .gmsh_session import gmsh_session
    from .gmsh_util import element_qualities

    with gmsh_session():
        gmsh.open(str(path))
        return element_qualities(md.tet_tags, ["minSICN"])["minSICN"]


_VIEWS: dict[str, Callable[[argparse.Namespace, Path], Any]] = {
    "faces": _render_faces, "groups": _render_groups, "quality": _render_quality}


def _cmd_mesh(a: argparse.Namespace) -> int:
    from .check import gate_table, load_checked_spec, record_state
    from .mesher import run_mesh_subprocess

    spec, mtime = load_checked_spec(a.spec)
    res = run_mesh_subprocess(spec.path, a.scale)
    record_state(str(spec.path), "unchecked" if res.ok else "fail", spec_mtime=mtime)
    head = (f"mesh ok: {res.n_elements} tets, {res.n_nodes} nodes -> {res.mesh_path}" if res.ok
            else "mesh FAILED")
    text = "\n\n".join([head, gate_table(res.gates), *res.errors])
    return _emit(a, text, res.to_dict(), EXIT_OK if res.ok else EXIT_FAIL)


def _cmd_check(a: argparse.Namespace) -> int:
    from .check import report_markdown, run_check

    r = run_check(a.spec, render=not a.no_render)
    return _emit(a, report_markdown(r), r.to_dict(), EXIT_OK if r.status == "pass" else EXIT_FAIL)


def _cmd_automesh(a: argparse.Namespace) -> int:
    from .check import gate_table, run_check
    from .mesher import auto_spec
    from .spec import load_spec

    part = Path(a.part).resolve()
    out = Path(a.output) if a.output else part.parent / f"{part.stem}_mesh"
    spec_path = (out / "mesh.yaml").resolve()
    auto_spec(part, "cfd" if a.cfd else "fea", spec_path=spec_path)
    r = run_check(spec_path, render=not a.no_render)
    text = (f"spec: {spec_path}\n(auto-generated starting point: add groups, bcs, refine; then "
            f"`feagent check {spec_path}`)\n\nstatus: {r.status}\n\n{gate_table(r.gates)}\n\n"
            f"report: {load_spec(spec_path).out_dir() / 'REPORT.md'}")
    return _emit(a, text, r.to_dict(), EXIT_OK if r.status == "pass" else EXIT_FAIL)


def _cmd_waive(a: argparse.Namespace) -> int:
    from .check import waive

    if not Path(a.spec).is_file():
        raise FeagentError(f"spec file not found: {a.spec}", "pass the mesh.yaml to waive")
    entry = waive(a.spec, a.reason)
    return _emit(a, f"waived {Path(a.spec).resolve()} ({entry['status']}): {a.reason}", entry)


def _cmd_hook_stop(a: argparse.Namespace) -> int:
    """Claude Code Stop hook: exit 2 (reason on stderr) while a tracked spec is not passing."""
    from .hook import main as hook_main

    return hook_main()


# --- parser -----------------------------------------------------------------

def _args_parts(p: argparse.ArgumentParser) -> None:
    p.add_argument("outdir", help="directory to write the synthetic test parts (.step) into")


def _args_inspect(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="CAD file (.step/.stp/.igs/.iges/.brep) or mesh.yaml")
    p.add_argument("--size", type=float, help="reference element size for hazards (default "
                   "spec size.global, or bbox diagonal / 40)")
    p.add_argument("--heal", action="store_true", help="heal the CAD on import")
    p.add_argument("--detail", choices=("summary", "full"), default="summary")


def _args_faces(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="CAD file or mesh.yaml")
    p.add_argument("--query", required=True, help="face query as JSON, e.g. '{\"at\": \"xmin\"}'")
    p.add_argument("--expect", type=int, help="expected number of faces (miss -> exit 2 with "
                   "an explanation)")


def _args_render(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="faces: CAD file or mesh.yaml; groups/quality: mesh.msh or "
                   "mesh.yaml (uses its out/mesh.msh)")
    p.add_argument("--view", choices=tuple(_VIEWS), default="faces")
    p.add_argument("--highlight", type=_int_list, help="face tags to highlight, e.g. 1,2")
    p.add_argument("--query", help="highlight the faces matching this JSON query")
    p.add_argument("-o", "--output", help="output PNG (default <path stem>_<view>.png)")


def _args_spec(p: argparse.ArgumentParser) -> None:
    p.add_argument("spec", help="mesh.yaml")


def _args_mesh(p: argparse.ArgumentParser) -> None:
    _args_spec(p)
    p.add_argument("--scale", type=float, default=1.0, help="global size scale (default 1)")


def _args_check(p: argparse.ArgumentParser) -> None:
    _args_spec(p)
    p.add_argument("--no-render", action="store_true", help="skip the group/quality renders")


def _args_automesh(p: argparse.ArgumentParser) -> None:
    p.add_argument("part", help="CAD file to mesh with an auto-generated spec")
    p.add_argument("--cfd", action="store_true", help="cfd analysis (default fea)")
    p.add_argument("-o", "--output", help="spec directory (default <part dir>/<stem>_mesh; the MCP "
                   "tool defaults to ./<stem>_mesh)")
    p.add_argument("--no-render", action="store_true", help="skip the group/quality renders")


def _args_waive(p: argparse.ArgumentParser) -> None:
    _args_spec(p)
    p.add_argument("--reason", required=True, help="why finishing with this spec is acceptable")


_COMMANDS: dict[str, tuple[Callable, Callable, str]] = {
    "parts": (_cmd_parts, _args_parts, "write the synthetic test parts"),
    "inspect": (_cmd_inspect, _args_inspect, "geometry digest (features, hazards, faces)"),
    "faces": (_cmd_faces, _args_faces, "select faces by geometric query"),
    "render": (_cmd_render, _args_render, "render faces, mesh groups or mesh quality to PNG"),
    "mesh": (_cmd_mesh, _args_mesh, "mesh a spec (records state 'unchecked')"),
    "check": (_cmd_check, _args_check, "mesh + quality gates + renders -> out/report.json"),
    "automesh": (_cmd_automesh, _args_automesh, "write a default spec for a part and check it"),
    "waive": (_cmd_waive, _args_waive, "accept a failing/unchecked spec with a stated reason"),
    "hook-stop": (_cmd_hook_stop, lambda p: None, "Claude Code Stop hook"),
}


def build_parser() -> argparse.ArgumentParser:
    """The ``feagent`` argument parser (one subcommand per entry of ``_COMMANDS``)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="write one JSON document to stdout")
    ap = _Parser(prog="feagent", description=__doc__.split("\n")[0], epilog=EXIT_CODES)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (func, add_args, help_text) in _COMMANDS.items():
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text,
                           epilog=EXIT_CODES)
        add_args(p)
        p.set_defaults(func=func)
    return ap


if __name__ == "__main__":
    sys.exit(main())
