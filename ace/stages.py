"""Layer 2 of diagnosis: the pipeline stages and an invariant per stage. Pure code, no LLM.

    geometry -> mesh -> bcs (boundary conditions and loads) -> solve -> postprocess

first_failing_stage(report) is the earliest stage whose invariant fails: the prime suspect.
Everything downstream of it is likely a symptom. The order is the logical order of the pipeline;
some checks need a solve to evaluate (e.g. mesh resolution) but still belong to their stage.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ace import checkpoint as ckpt
from ace import fea, oracles
from ace.config import MIN_QUALITY, SCORE, TASKS

STAGES = ["geometry", "mesh", "bcs", "solve", "postprocess"]

# Which oracles judge which stage (besides the direct checks in stage_report).
STAGE_ORACLES = {
    "mesh": ["mesh_convergence"],
    "bcs": ["spec_compliance", "gaming", "symmetry"],
    "solve": ["equilibrium"],
    "postprocess": ["sanity_bounds", "singularity", "known_answer"],
}
MESH_FLAGS = {"degenerate_element", "inverted_element"}      # feacore mesh failures
SOLVE_FLAGS = {"solver_failure", "non_finite"}


def _check(name: str, ok: bool, why: str) -> dict:
    return {"check": name, "ok": ok, "why": why}


def run_oracles(run: dict, target_err: float) -> dict | None:
    """Oracles for the run's scored solve (None if the run never solved)."""
    cp = ckpt.checkpoint(run, allow_failed=True)       # a failed attempt still has a setup to judge
    if cp is None:
        return None
    value = run["solves"][cp["solve_index"]]["result"].get("value")
    args = (run["problem_id"], cp["mesh"], cp["request"], cp["setup"], value, target_err)
    f = None if os.environ.get("ACE_NO_ORACLE_CACHE") == "1" else _oracle_cache_file(*args)
    if f is not None and f.exists():
        return json.loads(f.read_text())
    orc = oracles.evaluate(*args)
    if f is not None:
        _write_json(f, orc)
    return orc


# --------------------------------------------------------------------------- oracle cache
# The oracles re-solve the run and run a refinement study (4 extra solves, ~15 s in 3D), and the
# teacher's reruns ask for the same (mesh, setup) again, so results are cached on disk, keyed by
# everything evaluate() reads. Cached values come back through JSON: numpy floats as floats, tuples
# as lists, which is also what the run record on disk holds. Opt out with ACE_NO_ORACLE_CACHE=1.
ORACLE_CACHE = Path(__file__).with_name("field_refs") / "oracles"


def _mesh_hash(mesh: dict) -> str | None:
    """The mesh's fingerprint: its recorded sha256, else the hash of its .msh file (None if gone)."""
    if mesh.get("sha256"):
        return mesh["sha256"]
    path = Path(mesh.get("path") or "")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _oracle_cache_file(problem_id, mesh, request, setup, value, target_err) -> Path | None:
    h = _mesh_hash(mesh)
    if h is None:
        return None
    key = json.dumps([problem_id, h, request, setup, value, target_err, SCORE],
                     sort_keys=True, default=str)
    return ORACLE_CACHE / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-safe: {type(x).__name__}")


def _write_json(f: Path, obj) -> None:
    """Write atomically (parallel workers may write the same entry); skip anything not JSON-safe."""
    try:
        text = json.dumps(obj, default=_jsonable)
    except (TypeError, ValueError):
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, f)


def stage_report(run: dict, orc: dict | None, target_err: float) -> dict:
    """{stage: {"status": pass|fail|unknown, "checks": [...]}} plus "first_failing_stage"."""
    task = TASKS[run["problem_id"]]
    rep = {}
    # geometry: the problem's geometry builds and has named groups (deterministic per problem)
    g = fea.geometry_summary(fea.get_problem(run["problem_id"]))
    rep["geometry"] = [_check("geometry builds", bool(g["groups"]), f"{len(g['groups'])} named groups")]

    cp = ckpt.checkpoint(run, allow_failed=True)
    if cp is None:                                    # never called solve: the mesh stage stopped it
        meshes = [m for m in run.get("meshes", []) if m.get("ok")]
        if not meshes:
            rep["mesh"] = [_check("a mesh exists", False, "no mesh call succeeded")]
        elif all(m["nodes"] > task["node_cap"] for m in meshes):
            rep["mesh"] = [_check("within node cap", False, "every mesh was over the node cap, so it could not solve")]
        else:
            rep["mesh"] = [_check("a mesh exists", True, f"{len(meshes)} usable meshes")]
            rep["solve"] = [_check("solve attempted", False, "the agent never called solve")]
        return _finish(rep)

    m = cp["mesh"]
    flags = {f["id"] for f in orc.get("flags", [])} if orc else set()
    rep["mesh"] = [
        _check("within node cap", m["nodes"] <= task["node_cap"], f"{m['nodes']} / {task['node_cap']} nodes"),
        _check("element quality", (m.get("min_quality") or 0) >= MIN_QUALITY and not flags & MESH_FLAGS,
               f"min quality {m.get('min_quality')}"),
    ]
    err = run["solves"][cp["solve_index"]]["result"].get("error")
    rep["solve"] = [_check("solve succeeded", cp["solved"] and not flags & SOLVE_FLAGS,
                           err or ", ".join(sorted(flags & SOLVE_FLAGS)) or "ok")]
    get = lambda n: (orc or {}).get(n) or oracles.unavailable("oracles not computed")  # noqa: E731
    bcs_ok = all(get(n)["status"] != "fail" for n in STAGE_ORACLES["bcs"])
    for stage, names in STAGE_ORACLES.items():
        for n in names:
            o = get(n)
            if o["status"] == "unavailable":
                continue
            if n == "mesh_convergence" and (not bcs_ok or get("singularity")["status"] == "fail"):
                # Resolution is only meaningful for the right problem: with wrong boundary
                # conditions (or a real singularity, judged elsewhere) it says nothing about the mesh.
                continue
            ok, why = o["status"] == "pass", o["why"]
            rep.setdefault(stage, []).append(_check(n, ok, why))
    return _finish(rep)


def _finish(rep: dict) -> dict:
    out = {}
    for s in STAGES:
        checks = rep.get(s)
        status = "unknown" if checks is None else ("pass" if all(c["ok"] for c in checks) else "fail")
        out[s] = {"status": status, "checks": checks or []}
    out["first_failing_stage"] = first_failing_stage(out)
    return out


def first_failing_stage(report: dict) -> str | None:
    """The earliest stage whose invariant fails, or None if no stage fails."""
    return next((s for s in STAGES if report.get(s, {}).get("status") == "fail"), None)
