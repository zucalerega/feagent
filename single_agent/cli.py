"""Command-line access to the analyst's tools, for any agent or a human.

    python -m single_agent.cli new abc_00000065_bc0 --root ws/        # -> ws/part-xxxxxxxx/ (STEP + package.md)
    python -m single_agent.cli WS tools                               # the tool list with input schemas
    python -m single_agent.cli WS inspect_geometry
    python -m single_agent.cli WS find_faces '{"query": {"kind": "plane", "normal": [0, 0, -1]}}'
    python -m single_agent.cli WS mesh '{"h": 0.006, "order": 2}'
    python -m single_agent.cli WS solve @setup.json                   # arguments from a file
    python -m single_agent.cli WS study '{"solves": [1, 2, 3]}'
    python -m single_agent.cli WS submit_report @report.json
    python -m single_agent.cli WS status                              # meshes, solves, studies so far

Each call is one tool call: the session is reloaded from WS/session.json and saved after the call, and
the result is printed as JSON (a picture's path is in "image"). Fields of earlier solves are recomputed
when a later call (view, compare) needs them. The workspace holds no reference data: `new` copies only
the CAD and the design package (single_agent.workspace).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _args(text: str | None) -> dict:
    if not text:
        return {}
    if text.startswith("@"):
        text = Path(text[1:]).read_text()
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "new":
        ap = argparse.ArgumentParser(prog="single_agent.cli new")
        ap.add_argument("_cmd")
        ap.add_argument("problem")
        ap.add_argument("--root", default="sa_workspaces")
        a = ap.parse_args(argv)
        from single_agent import workspace
        ws = workspace.make(a.problem, a.root)
        print(json.dumps({"workspace": str(ws), "files": sorted(p.name for p in ws.iterdir())}))
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    ws, tool = Path(argv[0]), argv[1]
    if not (ws / "geometry.step").exists():
        print(json.dumps({"error": f"{ws} is not a workspace (no geometry.step)"}))
        return 2
    from single_agent.harness import TOOLS, Session
    if tool == "tools":
        print(json.dumps([{k: t[k] for k in ("name", "description", "input_schema")} for t in TOOLS], indent=1))
        return 0
    s = Session(ws)
    if tool == "status":
        st = s.state
        print(json.dumps({
            "meshes": [{k: m.get(k) for k in ("number", "ok", "nodes", "order", "min_quality", "h_used")}
                       for m in st["meshes"]],
            "solves": [{k: r.get(k) for k in ("number", "mesh", "ok", "dofs", "qois", "error")} for r in st["solves"]],
            "studies": [st_["solves"] for st_ in st["studies"]], "views": st["views"],
            "submitted": st["report"] is not None, "limits": st["limits"]}, indent=1, default=float))
        return 0
    out, err = s.call(tool, _args(argv[2] if len(argv) > 2 else None))
    print(json.dumps(out, indent=1, default=float))
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
