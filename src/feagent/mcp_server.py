"""MCP server ``feagent`` (stdio): the meshing harness as tools for an LLM agent.

Every tool runs ``python -m feagent.cli <command> ... --json`` in a child process with a timeout
(gmsh crashes and hangs cannot take the server down) and returns compact text the model can act
on: the geometry digest, matched faces, the gate table or REPORT.md; ``render_view`` also returns
the PNG. CLI errors (exit 2/3) become tool errors carrying the CLI's message and hint.

Tool descriptions and the workflow are improver-editable files, read when the server is built:
``<context dir>/tools.yaml`` (tool name -> description; must list exactly the registered tools)
and ``<context dir>/SKILL.md`` (served as the MCP prompt ``meshing_workflow``). The context dir is
``$FEAGENT_CONTEXT_DIR`` if set, else ``agent_context/`` at the repository root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

CONTEXT_ENV = "FEAGENT_CONTEXT_DIR"
_DEFAULT_CONTEXT = Path(__file__).resolve().parents[2] / "agent_context"
TIMEOUT_S = {"inspect": 300.0, "faces": 300.0, "render": 300.0, "automesh": 1800.0,
             "mesh": 1800.0, "check": 1800.0}
_STDERR_TAIL = 2000


def context_dir() -> Path:
    """``$FEAGENT_CONTEXT_DIR`` (resolved) if set, else the repository's ``agent_context/``."""
    env = os.environ.get(CONTEXT_ENV)
    return Path(env).resolve() if env else _DEFAULT_CONTEXT


# --- running the CLI ------------------------------------------------------------

def run_cli(args: list[str], timeout: float | None = None) -> tuple[int, Any]:
    """Run ``feagent <args> --json``; return (exit code, parsed JSON) for exit 0/1.

    Raises ToolError with the CLI's error and hint for exit 2/3, a timeout or unreadable output."""
    cmd = [sys.executable, "-m", "feagent.cli", *args, "--json"]
    limit = TIMEOUT_S.get(args[0], 600.0) if timeout is None else timeout
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ToolError(f"error: `feagent {args[0]}` timed out after {limit:g} s\nhint: coarsen "
                        "size.global or simplify the spec, then retry") from None
    data = _parse(p)
    if p.returncode in (0, 1) and not (isinstance(data, dict) and data.get("ok") is False):
        return p.returncode, data
    raise ToolError(_error_text(p.returncode, data))


def _parse(p: subprocess.CompletedProcess) -> Any:
    try:
        return json.loads(p.stdout)
    except ValueError:
        raise ToolError(f"error: feagent exited {p.returncode} without a JSON result\n"
                        f"stderr: {p.stderr[-_STDERR_TAIL:]}") from None


def _error_text(code: int, data: Any) -> str:
    d = data if isinstance(data, dict) else {}
    text = f"error (exit {code}): {d.get('error', data)}"
    return text + (f"\nhint: {d['hint']}" if d.get("hint") else "")


# --- tools ------------------------------------------------------------------------

def inspect_geometry(path: str, size: float | None = None, detail: str = "summary") -> str:
    from .geometry import digest_text
    from .types import GeometryModel

    args = ["inspect", path, "--detail", detail] + ([] if size is None else ["--size", str(size)])
    _, data = run_cli(args)
    return digest_text(GeometryModel.from_dict(data), detail)


def find_faces(path: str, query: dict, expect: int | None = None) -> str:
    from .select import describe_faces
    from .types import FaceInfo

    args = ["faces", path, "--query", json.dumps(query)]
    _, data = run_cli(args + ([] if expect is None else ["--expect", str(expect)]))
    faces = {f["tag"]: FaceInfo.from_dict(f) for f in data["faces"]}
    lines = describe_faces(SimpleNamespace(face=faces.__getitem__), data["tags"])
    return f"{len(data['tags'])} face(s) matched: {data['tags']}\n{lines}"


def render_view(path: str, view: str = "faces", highlight: list[int] | None = None,
                query: dict | None = None) -> list:
    args = ["render", path, "--view", view]
    args += ["--highlight", ",".join(map(str, highlight))] if highlight else []
    _, data = run_cli(args + (["--query", json.dumps(query)] if query is not None else []))
    return [f"{data['path']}\n{data['caption']}", Image(path=data["path"])]


def automesh(path: str, cfd: bool = False) -> str:
    _, data = run_cli(["automesh", path, "--no-render"] + (["--cfd"] if cfd else []))
    return _automesh_text(data)


def mesh(spec_path: str, scale: float = 1.0) -> str:
    from .check import gate_table

    _, r = run_cli(["mesh", spec_path, "--scale", str(scale)])
    head = (f"mesh ok: {r['n_elements']} tets, {r['n_nodes']} nodes -> {r['mesh_path']}"
            if r["ok"] else "mesh FAILED")
    return "\n\n".join([head, gate_table(_gates(r["gates"])), *r["errors"]])


def check_mesh(spec_path: str) -> str:
    from .check import CheckReport, report_markdown

    _, data = run_cli(["check", spec_path])
    return report_markdown(CheckReport.from_dict(data))


def _gates(items: list[dict]) -> list:
    from .types import Gate

    return [Gate.from_dict(g) for g in items]


def _automesh_text(r: dict) -> str:
    from .check import gate_table

    spec, m = r["spec_path"], r["mesh"]
    lines = [f"spec: {spec}", f"status: {r['status']}", "", gate_table(_gates(r["gates"]))]
    if m["warnings"]:
        lines += ["", "mesher notes:", *(f"- {w}" for w in m["warnings"])]
    if m.get("out_dir"):
        lines += ["", f"report: {Path(m['out_dir']) / 'REPORT.md'}"]
    return "\n".join(lines + ["", f"next: inspect_geometry(\"{spec}\") for features and hazards; "
                              "name BC groups by query, add bcs:, then check_mesh(spec)"])


TOOLS: dict[str, Callable[..., Any]] = {
    "inspect_geometry": inspect_geometry, "find_faces": find_faces, "render_view": render_view,
    "automesh": automesh, "mesh": mesh, "check_mesh": check_mesh}


# --- server -----------------------------------------------------------------------

def load_tool_docs(ctx: Path) -> dict[str, str]:
    """``tools.yaml`` of ``ctx`` as name -> description; it must describe exactly ``TOOLS``."""
    docs = yaml.safe_load((ctx / "tools.yaml").read_text())
    if not isinstance(docs, dict) or not all(isinstance(v, str) for v in docs.values()):
        raise ValueError(f"{ctx / 'tools.yaml'}: expected a mapping of tool name -> description")
    missing, extra = sorted(set(TOOLS) - set(docs)), sorted(set(docs) - set(TOOLS))
    if missing or extra:
        raise ValueError(f"{ctx / 'tools.yaml'} must describe exactly the registered tools: "
                         f"missing {missing}, unknown {extra}")
    return {k: v.strip() for k, v in docs.items()}


def build_server(ctx: Path | None = None) -> MCPServer:
    """The ``feagent`` MCP server with tool descriptions and workflow read from ``ctx``."""
    ctx = context_dir() if ctx is None else Path(ctx)
    docs = load_tool_docs(ctx)
    skill = (ctx / "SKILL.md").read_text()
    server = MCPServer("feagent", instructions=_instructions(skill))
    for name, fn in TOOLS.items():
        server.add_tool(fn, name=name, description=docs[name], structured_output=False)
    server.prompt(name="meshing_workflow",
                  description="The meshing workflow (SKILL.md): read it before meshing.")(
        lambda: skill)
    return server


def _instructions(skill: str) -> str:
    front = skill.split("---")[1] if skill.startswith("---") else ""
    meta = yaml.safe_load(front) if front else None
    desc = meta.get("description", "") if isinstance(meta, dict) else ""
    return (f"{desc}\nRead the `meshing_workflow` prompt for the full workflow, gate->fix table "
            "and thresholds. Start with automesh.").strip()


mcp = build_server()


def main() -> None:
    """Console script ``feagent-mcp``: serve over stdio."""
    mcp.run()
