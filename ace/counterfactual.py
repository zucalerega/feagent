"""Layer 3 of diagnosis: counterfactual reruns. Code + solver, no LLM.

    rerun_from_stage(run, stage, change, target_err, workdir)

loads the run's checkpoint (ace.checkpoint), applies ONE typed change at `stage`, re-runs every
later stage (mesh -> solve -> oracles) and returns the oracle results and score before/after.
A hypothesis like "the mesh was too coarse at the stress raiser" becomes an experiment: refine
there, rerun, and see whether the failure goes away.

Changes come from a small whitelist (INTERVENTIONS), never free-form code. Each maps onto the
same harness functions the agent used (ace.fea.make_mesh / run_solve), so a rerun is exactly
what the agent would have got had it made that one choice differently.
"""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

from ace import checkpoint as ckpt
from ace import fea, oracles, stages
from ace.config import SCORE, TASKS

LOAD_TYPES = ("pressure", "traction", "force")

# name -> (stage it changes, argument description shown to the diagnoser)
INTERVENTIONS = {
    "refine_mesh": ("mesh", "scope: 'global' or a group name; factor > 1 makes elements that many times smaller "
                            "(global: h and all refinement sizes; group: that group's refinement size, added with "
                            "radius = h if absent)"),
    "set_local_mesh_size": ("mesh", "group: name OR point: [x, y(, z)]; size; radius (element size `size` at the "
                                    "group/point, growing to h at distance `radius`; replaces a refinement on the same target)"),
    "change_element_order": ("mesh", "order: 1 or 2"),
    "fix_constraint": ("bcs", "group: name OR point: [x, y(, z)]; type: fixed | displacement | symmetry | free; "
                              "components (for displacement); value; remove (optional list of other targets, "
                              "group names or [x, y(, z)] points, whose supports are removed). Replaces every "
                              "support on the target ('free' removes them): e.g. swap a point constraint for "
                              "a symmetry condition in one change"),
    "fix_load": ("bcs", "group; type: pressure | traction | force | none; value; remove_from (optional group whose "
                        "loads are removed). Replaces the loads on `group` ('none' removes them)"),
    "convert_units": ("bcs", "target: 'E' (Young's modulus) or 'loads' (every load value); factor to multiply by"),
}


class InterventionError(ValueError):
    pass


# --------------------------------------------------------------------------- applying one change


def _target(bc_or_args: dict):
    return ("group", bc_or_args["group"]) if bc_or_args.get("group") else ("point", tuple(bc_or_args.get("point") or ()))


def apply_mesh_change(request: dict, name: str, a: dict) -> dict:
    req = copy.deepcopy(request)
    refs = req.setdefault("refinements", []) or []
    req["refinements"] = refs
    if name == "refine_mesh":
        f = float(a["factor"])
        if f <= 0:
            raise InterventionError("factor must be > 0")
        if a.get("scope", "global") == "global":
            req["h"] = req["h"] / f
            for r in refs:
                r["size"] = r["size"] / f
        else:
            mine = [r for r in refs if r.get("group") == a["scope"]]
            if mine:
                for r in mine:
                    r["size"] = r["size"] / f
            else:
                refs.append({"group": a["scope"], "size": req["h"] / f, "radius": req["h"]})
    elif name == "set_local_mesh_size":
        if not (a.get("group") or a.get("point")):
            raise InterventionError("set_local_mesh_size needs a group or a point")
        tgt = _target(a)
        req["refinements"] = [r for r in refs if _target(r) != tgt]
        new = {"size": float(a["size"]), "radius": float(a["radius"])}
        new.update({"group": a["group"]} if a.get("group") else {"point": list(a["point"])})
        req["refinements"].append(new)
    elif name == "change_element_order":
        if int(a["order"]) not in (1, 2):
            raise InterventionError("order must be 1 or 2")
        req["order"] = int(a["order"])
    return req


def apply_bc_change(setup: dict, name: str, a: dict) -> dict:
    s = copy.deepcopy(setup)
    bcs = s.get("bcs", [])
    if name == "fix_constraint":
        if not (a.get("group") or a.get("point")):
            raise InterventionError("fix_constraint needs a group or a point")
        tgts = {_target(a)} | {("point", tuple(t)) if isinstance(t, list) else ("group", t) for t in a.get("remove") or []}
        bcs = [b for b in bcs if not (b.get("type") not in LOAD_TYPES and _target(b) in tgts)]
        if a.get("type") != "free":
            new = {k: a[k] for k in ("type", "group", "point", "components", "value") if a.get(k) is not None}
            bcs.append(new)
    elif name == "fix_load":
        drop = {a["group"]} | ({a["remove_from"]} if a.get("remove_from") else set())
        bcs = [b for b in bcs if not (b.get("type") in LOAD_TYPES and b.get("group") in drop)]
        if a.get("type", "none") != "none":
            bcs.append({"type": a["type"], "group": a["group"], "value": a["value"]})
    elif name == "convert_units":
        f = float(a["factor"])
        if a["target"] == "E":
            s["material"] = dict(s.get("material") or {}, E=float((s.get("material") or {}).get("E", 0)) * f)
        elif a["target"] == "loads":
            for b in bcs:
                if b.get("type") in LOAD_TYPES:
                    v = b.get("value")
                    b["value"] = [x * f for x in v] if isinstance(v, list) else float(v) * f
        else:
            raise InterventionError("convert_units target must be 'E' or 'loads'")
    s["bcs"] = bcs
    return s


# --------------------------------------------------------------------------- evaluating a variant


def evaluate_variant(problem_id: str, mesh: dict, request: dict | None, setup: dict, target_err: float) -> dict:
    """Solve (mesh, setup) and judge it the same way as a real run: score, oracles, stages."""
    problem = fea.get_problem(problem_id)
    qoi = TASKS[problem_id]["qoi"]
    res = fea.run_solve(problem, mesh, setup, qoi) if mesh.get("ok") else {"ok": False, "error": mesh.get("error")}
    run = {"problem_id": problem_id, "meshes": [dict(mesh, request=request)], "submitted": 0 if res["ok"] else None,
           "solves": [{"mesh": mesh, "mesh_index": 0, "setup": setup, "result": res}]}
    orc = stages.run_oracles(run, target_err)
    rep = stages.stage_report(run, orc, target_err)
    from ace import evidence
    if evidence.scored_by_test(problem_id):     # judged by the physical test, as check_run does
        err = evidence.value_error(problem_id, res["value"]) if res.get("ok") else None
    else:
        ref = fea.reference_value(problem, qoi)
        err = abs(res["value"] - ref) / abs(ref) if res.get("ok") else None
        if SCORE == "field" and res.get("ok"):
            from ace.field_error import errors_of_setup
            err = errors_of_setup(problem_id, mesh, setup)["error"]
    return {
        "solved": bool(res.get("ok")), "error": res.get("error"), "value": res.get("value"),
        "rel_error": err, "accuracy": min(1.0, target_err / max(err, 1e-300)) if err is not None else None,
        "dofs": (res.get("info") or {}).get("n_dofs"),
        "nodes": mesh.get("nodes"), "within_cap": mesh.get("ok") and mesh["nodes"] <= TASKS[problem_id]["node_cap"],
        "first_failing_stage": rep["first_failing_stage"],
        "oracles": {k: v["status"] for k, v in (orc or {}).items() if isinstance(v, dict) and "status" in v},
        "failed_checks": [f"{s}: {c['check']}: {c['why']}" for s in stages.STAGES for c in rep[s]["checks"] if not c["ok"]],
    }


def recorded_before(run: dict, cp: dict, target_err: float) -> dict | None:
    """evaluate_variant of the run's own checkpoint, read from what the loop already recorded
    (checks, oracles, stages of the scored solve) instead of solving it all again. None when the
    record cannot stand in for it: a field missing, no successful scored solve, or a different
    score mode or accuracy target than this rerun's."""
    c, orc, rep = run.get("checks"), run.get("oracles"), run.get("stages")
    if not (c and orc and rep and c.get("solved") and cp["solved"]):
        return None
    if c.get("tolerance") != target_err or ("field_error" in c) != (SCORE == "field"):
        return None
    mesh, res = cp["mesh"], run["solves"][cp["solve_index"]]["result"]
    err = c["rel_error"]                    # the same error evaluate_variant computes (point or field)
    return {
        "solved": True, "error": res.get("error"), "value": res.get("value"),
        "rel_error": err, "accuracy": min(1.0, target_err / max(err, 1e-300)),
        "dofs": (res.get("info") or {}).get("n_dofs"),
        "nodes": mesh.get("nodes"), "within_cap": mesh.get("ok") and mesh["nodes"] <= TASKS[run["problem_id"]]["node_cap"],
        "first_failing_stage": rep["first_failing_stage"],
        "oracles": {k: v["status"] for k, v in orc.items() if isinstance(v, dict) and "status" in v},
        "failed_checks": [f"{s}: {x['check']}: {x['why']}" for s in stages.STAGES for x in rep[s]["checks"] if not x["ok"]],
    }


def _stage_rank(stage: str | None) -> int:
    return len(stages.STAGES) if stage is None else stages.STAGES.index(stage)


def flipped(before: dict, after: dict) -> bool:
    """Did the change fix the failure? The first failing stage moved later (or vanished), or, for
    an accuracy failure (mesh / postprocess) that stays at the same stage, the error fell at least
    2x (+0.3 in score). With a wrong setup (bcs or earlier) the error is meaningless (e.g. a
    singular value shrinks on a coarser mesh), so only the stage moving counts there."""
    b, a = _stage_rank(before["first_failing_stage"]), _stage_rank(after["first_failing_stage"])
    if a > b:
        return True
    if a < b or before["first_failing_stage"] not in ("mesh", "postprocess"):
        return False
    if after["rel_error"] is not None and before["rel_error"] is not None:
        return after["rel_error"] <= before["rel_error"] / 2
    return False


# --------------------------------------------------------------------------- the rerun


def rerun_from_stage(run: dict, stage: str, change: dict, target_err: float, workdir: Path) -> dict:
    """change = {"name": <INTERVENTIONS key>, "args": {...}}. Returns before/after metrics, whether
    the outcome flipped, the solver wall-clock seconds, and notes (e.g. a rebuilt mesh that does
    not match the original, or a size clamped by the harness's size floor)."""
    t0 = time.perf_counter()
    name, args = change.get("name"), change.get("args") or {}
    if name not in INTERVENTIONS:
        raise InterventionError(f"unknown intervention {name!r}; allowed: {sorted(INTERVENTIONS)}")
    if INTERVENTIONS[name][0] != stage:
        raise InterventionError(f"{name} changes the {INTERVENTIONS[name][0]} stage, not {stage!r}")
    cp = ckpt.checkpoint(run, allow_failed=True)
    if cp is None:
        raise InterventionError("this run never called solve, so there is no checkpoint to rerun from")
    problem = fea.get_problem(run["problem_id"])
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    notes = []

    before = recorded_before(run, cp, target_err) or \
        evaluate_variant(run["problem_id"], cp["mesh"], cp["request"], cp["setup"], target_err)
    if stage == "mesh":
        if not cp["request"]:
            raise InterventionError("the mesh request of this run was not recorded")
        req = apply_mesh_change(cp["request"], name, args)
        mesh = fea.make_mesh(problem, workdir / f"cf_{hashlib.sha1(repr(req).encode()).hexdigest()[:8]}.msh",
                             h=req["h"], order=req.get("order", 1), refinements=req.get("refinements"))
        if mesh.get("ok") and mesh["h_used"] != req["h"] or any(
                s != r["size"] for s, r in zip(mesh.get("sizes_used", []), req.get("refinements") or [])):
            notes.append("the harness clamped a requested size to its size floor (see sizes_used)")
        setup = cp["setup"]
    else:
        req, mesh = cp["request"], _checkpoint_mesh(problem, cp, workdir, notes)
        setup = apply_bc_change(cp["setup"], name, args)
    after = evaluate_variant(run["problem_id"], mesh, req, setup, target_err)
    if not after["within_cap"] and after["solved"]:
        notes.append("the changed mesh is over the node cap: a real agent could not submit it")
    return {"run_id": run.get("run_id"), "stage": stage, "change": {"name": name, "args": args},
            "before": before, "after": after, "flipped": flipped(before, after),
            "mesh_used": {k: mesh.get(k) for k in ("nodes", "order", "h_used", "sizes_used")},
            "solver_seconds": round(time.perf_counter() - t0, 2), "notes": notes}


def _checkpoint_mesh(problem, cp: dict, workdir: Path, notes: list) -> dict:
    """The run's own mesh: the saved .msh if it still exists, else rebuilt from its request."""
    m = cp["mesh"]
    if m.get("path") and Path(m["path"]).exists():
        return m
    if not cp["request"]:
        raise InterventionError("the run's mesh file is gone and its request was not recorded")
    req = cp["request"]
    new = fea.make_mesh(problem, workdir / "checkpoint_rebuild.msh", h=req["h"], order=req.get("order", 1),
                        refinements=req.get("refinements"))
    if m.get("sha256") and new.get("sha256") != m["sha256"]:
        notes.append("the rebuilt mesh differs from the original (different code or gmsh version)")
    return new
