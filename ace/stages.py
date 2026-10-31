"""Layer 2 of diagnosis: the pipeline stages and an invariant per stage. Pure code, no LLM.

    geometry -> mesh -> bcs (boundary conditions and loads) -> solve -> postprocess

first_failing_stage(report) is the earliest stage whose invariant fails: the prime suspect.
Everything downstream of it is likely a symptom. The order is the logical order of the pipeline;
some checks need a solve to evaluate (e.g. mesh resolution) but still belong to their stage.
"""

from __future__ import annotations

from ace import checkpoint as ckpt
from ace import fea, oracles
from ace.config import MIN_QUALITY, TASKS

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
    return oracles.evaluate(run["problem_id"], cp["mesh"], cp["request"], cp["setup"], value, target_err)


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
