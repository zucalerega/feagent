"""Stage checkpoints of an agent run: what it meshed, how it set up the analysis, which solve counts.

A run record (ace/agent.py) holds every mesh with the request that produced it, the .msh file,
and every solve with its setup and the index of the mesh it used. Geometry is rebuilt from the
problem definition (deterministic), so nothing else needs saving. Older records without the
request fields fall back to the mesh tool calls in the trace.
"""

from __future__ import annotations

from pathlib import Path


def scored_solve_index(run: dict) -> int | None:
    """The solve that is scored: the submitted one, else the last successful one."""
    sub = run.get("submitted")
    solves = run.get("solves", [])
    if sub is not None and 0 <= sub < len(solves) and solves[sub]["result"].get("ok"):
        return sub
    ok = [i for i, s in enumerate(solves) if s["result"].get("ok")]
    return ok[-1] if ok else None


def mesh_index_of(run: dict, solve_index: int) -> int | None:
    s = run["solves"][solve_index]
    if "mesh_index" in s:
        return s["mesh_index"]
    path = s["mesh"].get("path", "")                       # older records: mesh_<k>.msh
    stem = Path(path).stem
    return int(stem.split("_")[1]) if stem.startswith("mesh_") else None


def mesh_request(run: dict, mesh_index: int) -> dict | None:
    """The arguments of the mesh call that produced mesh `mesh_index` (h, order, refinements)."""
    m = run["meshes"][mesh_index]
    if "request" in m:
        return dict(m["request"])
    calls = [t for t in run.get("trace", []) if t["type"] == "tool_call" and t["name"] == "mesh"]
    if mesh_index < len(calls):
        return {k: v for k, v in calls[mesh_index]["input"].items() if k != "using"}
    return None


def checkpoint(run: dict, allow_failed: bool = False) -> dict | None:
    """Everything needed to rerun a run's scored solve from any stage:
    {problem_id, solve_index, mesh_index, mesh (record), request, setup, solved}.
    If no solve succeeded: None, or with allow_failed=True the last attempted solve."""
    i = scored_solve_index(run)
    if i is None and allow_failed and run.get("solves"):
        i = len(run["solves"]) - 1
    if i is None:
        return None
    k = mesh_index_of(run, i)
    s = run["solves"][i]
    return {"problem_id": run["problem_id"], "solve_index": i, "mesh_index": k, "mesh": s["mesh"],
            "request": mesh_request(run, k) if k is not None else None, "setup": s["setup"],
            "solved": bool(s["result"].get("ok"))}
