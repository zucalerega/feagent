"""Deterministic checks and the score. No LLM calls: these numbers are ground truth for the Reflector.

The score is the team's speed/accuracy metric (feacore.diagnostics.score.score_population):
  accuracy   = min(1, tolerance / error)   error = relative error of the scored quantity from the
                                           solve the agent submitted, vs the problem's reference
  efficiency = DOFs of the cheapest valid agent that met tolerance / own DOFs
  score      = valid * accuracy * efficiency  in [0, 1]; 1 = the cheapest setup accurate enough
It is relative to a population: here every agent on that problem in the whole run, so scores are
comparable across generations (score_population below). A run that fails a gate (no solve, over
the node cap, degenerate mesh) or has a fail-severity feacore flag is invalid and scores 0.

check_run gives one run's facts (error, DOFs, gates); score_population turns the facts of all runs
on a problem into scores. calibrate()/load_calibration() measure what a uniform mesh achieves at
the node cap; the mock agent uses that to pick plausible mesh sizes.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from ace import fea
from ace.config import AUTO_GAIN_TARGET, BEST_BONUS, FIELD_ERR_FLOOR, MIN_QUALITY, PROBE_DISCLOSED, SCORE, TASKS

DEFAULT_TOL = 0.02

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


# --------------------------------------------------------------------------- calibration


def reference_setup(problem) -> dict:
    """The problem's own (correct) setup, in the same shape the agent submits."""
    rc = problem.case("REF.msh").to_dict()
    return {
        "formulation": rc["analysis"]["formulation"],
        "thickness": rc["analysis"]["thickness"],
        "material": next(iter(rc["materials"].values())),
        "bcs": [{k: v for k, v in bc.items() if v is not None} for bc in rc["bcs"]],
        "loads": rc["loads"],
        "T_ref": rc["analysis"]["T_ref"],
    }


def _uniform_error(problem, qoi, h, order, workdir) -> tuple[int, float]:
    mesh = fea.make_mesh(problem, Path(workdir) / "uniform.msh", h=h, order=order)
    sol = fea.run_solve(problem, mesh, reference_setup(problem), qoi)
    ref = fea.reference_value(problem, qoi)
    return mesh["nodes"], abs(sol["value"] - ref) / abs(ref)


def _calibrate_order(problem, qoi, cap, order, tmp) -> dict:
    lo, hi = problem.params["h0"] / 20, problem.params["h0"] * 20   # bisection on h
    for _ in range(25):
        h = math.sqrt(lo * hi)
        n, _ = _uniform_error(problem, qoi, h, order, tmp)
        lo, hi = (h, hi) if n > cap else (lo, h)
    nodes, errs = [], []
    for f in (1.0, 1.05, 1.1, 1.15, 1.2):
        n, e = _uniform_error(problem, qoi, hi * f, order, tmp)
        nodes.append(n)
        errs.append(e)
    return {"order": order, "h": hi, "nodes": nodes, "errors": errs,
            "e_uniform": math.exp(sum(math.log(e) for e in errs) / len(errs))}


def calibrate(problem_id: str) -> dict:
    """The best uniform mesh that fits the node cap, and its error.

    For each element order, find the uniform size that just fits the cap. Point and peak errors
    on coarse meshes jump around with node placement, so average (geometric mean) over five
    slightly coarser sizes that all fit. The baseline is the better of the two orders, so an
    agent gets no credit for merely choosing quadratic elements.
    """
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    with tempfile.TemporaryDirectory() as tmp:
        per_order = [_calibrate_order(problem, task["qoi"], task["node_cap"], o, tmp) for o in (1, 2)]
    best = min(per_order, key=lambda c: c["e_uniform"])
    return {**best, "by_order": per_order}


def load_calibration(problem_ids: list[str]) -> dict:
    """Cached calibration; computes (and saves) any missing or stale entries."""
    cal = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    changed = False
    for pid in problem_ids:
        stale = cal.get(pid, {}).get("node_cap") != TASKS[pid]["node_cap"] or "by_order" not in cal.get(pid, {})
        if pid not in cal or stale:
            cal[pid] = {"node_cap": TASKS[pid]["node_cap"], **calibrate(pid)}
            changed = True
    if changed:
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2) + "\n")
    return cal


# --------------------------------------------------------------------------- per-run checks


def tolerance(problem_id: str) -> float:
    """Accuracy target: the reference's stated tolerance for the scored quantity (field mode: the
    automesher's field error divided by the target gain, see config)."""
    if SCORE == "field":
        from ace.field_error import automesher
        return automesher(problem_id)["error"] / AUTO_GAIN_TARGET
    problem = fea.get_problem(problem_id)
    qoi = TASKS[problem_id]["qoi"]
    return next((r.tolerance for r in problem.references() if r.qoi == qoi), DEFAULT_TOL)


def bc_report(problem, setup: dict) -> dict:
    """Compare the agent's setup with the reference setup (diagnostic, not a gate: other
    setups can be equivalent, and a wrong one already shows up as a large error)."""
    ref = reference_setup(problem)

    def key(bc):
        # Point BCs (e.g. pinning one node against rigid-body motion) are matched by type and
        # components only: any suitable node is equivalent, and [-0.1, 0] == [-0.1, 0.0].
        where = bc.get("group") or f"a point (components {sorted(bc.get('components') or [])})"
        return where, bc.get("type")

    ref_bcs = {key(b): b for b in ref["bcs"]}
    got_bcs = {key(b): b for b in setup.get("bcs", [])}
    sign_flips = []
    for k in ref_bcs.keys() & got_bcs.keys():
        a, b = ref_bcs[k].get("value"), got_bcs[k].get("value")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            flipped = a * b < 0
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            try:   # vectors (tractions, forces): pointing in opposite directions
                flipped = sum(float(x) * float(y) for x, y in zip(a, b)) < 0
            except (TypeError, ValueError):   # expressions: not compared
                flipped = False
        else:
            flipped = False
        if flipped:
            sign_flips.append(f"{k[1]} on {k[0]}: reversed relative to the reference")
    mat, ref_mat = setup.get("material") or {}, ref["material"]
    return {
        "missing": [f"{t} on {g}" for g, t in ref_bcs.keys() - got_bcs.keys()],
        "unexpected": [f"{t} on {g}" for g, t in got_bcs.keys() - ref_bcs.keys()],
        "sign_flips": sign_flips,
        "formulation_ok": setup.get("formulation") == ref["formulation"],
        "thickness_ok": math.isclose(float(setup.get("thickness", 1.0)), ref["thickness"], rel_tol=1e-6),
        "material_ok": all(math.isclose(float(mat.get(k, 0)), ref_mat[k], rel_tol=1e-3, abs_tol=1e-12)
                           for k in ("E", "nu", "alpha")),
    }


def check_run(problem_id: str, run: dict) -> dict:
    """The facts of one agent run (`run` is the record from ace.agent.run_agent): its scored
    solve's error and cost (DOFs), and the gates. The score itself needs the whole population
    of runs on the problem: see score_population."""
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    out = {
        "node_cap": task["node_cap"],
        "tolerance": tolerance(problem_id),
        "n_meshes": len(run["meshes"]),
        "n_solves": len(run["solves"]),
        "meshed": any(m.get("ok") for m in run["meshes"]),
        "score": None,                       # filled in by score_population
    }
    # The scored solve: the one the agent submitted, else its last successful one.
    sub = run.get("submitted")
    if sub is not None and 0 <= sub < len(run["solves"]) and run["solves"][sub]["result"].get("ok"):
        final = run["solves"][sub]
    else:
        final = next((s for s in reversed(run["solves"]) if s["result"].get("ok")), None)
    out["submitted"] = sub is not None
    if final is None:
        return {**out, "solved": False, "failed_gate": "no successful solve"}

    mesh = final["mesh"]
    ref = fea.reference_value(problem, task["qoi"])
    rel_err = abs(final["result"]["value"] - ref) / abs(ref)
    point_err = rel_err
    if SCORE == "field":                         # whole stress field vs the fine reference
        from ace.field_error import errors_of_setup
        errs = errors_of_setup(problem_id, final["mesh"], final["setup"])
        rel_err = errs["error"]
        out.update(field_error=errs["field"], probe_error=errs["points"], point_errors=errs["point_errors"])
    info = final["result"].get("info") or {}
    ncomp = 3 if problem.dim == 3 else 2
    out.update({
        "solved": True,
        "nodes": mesh["nodes"],
        "dofs": info.get("n_dofs") or mesh["nodes"] * ncomp,   # older records: nodes x components
        "order": mesh["order"],
        "min_quality": mesh["min_quality"],
        "within_cap": mesh["nodes"] <= task["node_cap"],
        "quality_ok": (mesh["min_quality"] or 0) >= MIN_QUALITY,
        "qoi_value": final["result"]["value"],
        "rel_error": rel_err,
        "point_rel_error": point_err,
        "meets_tol": rel_err <= out["tolerance"],
        "setup": bc_report(problem, final["setup"]),
    })
    gate = None if out["within_cap"] else "over node cap"
    if gate is None and not out["quality_ok"]:
        gate = f"min element quality < {MIN_QUALITY}"
    return {**out, "failed_gate": gate}


def score_population(runs: list[dict]) -> None:
    """Score every run on ONE problem against each other with feacore's score_population.
    Each run needs "checks" (check_run) and, if computed, "oracles" (its feacore flags make a run
    invalid when one is fail-severity). Writes score, accuracy, efficiency, valid, cost and
    pareto_rank into each run's checks."""
    from feacore.diagnostics.flags import Flag, Report
    from feacore.diagnostics.score import score_population as feacore_score

    from ace.oracles import baseline_fail_flags

    agents = []
    for r in runs:
        c = r["checks"]
        rep = Report()
        base = baseline_fail_flags(r["problem_id"])
        for f in (r.get("oracles") or {}).get("flags", []):
            if not (f["severity"] == "fail" and f["id"] in base):    # see oracles.baseline_fail_flags
                rep.add(Flag(**f))
        if r.get("error") or c["failed_gate"] or not c.get("solved"):
            rep.add(Flag("ace_gate", "setup", "fail", r.get("error") and "run crashed" or c["failed_gate"] or "no solve"))
        rep.metrics["n_dofs"] = c.get("dofs")
        errors = {"q": {"est_err": c["rel_error"]}} if c.get("rel_error") is not None else {}
        agents.append({"agent_id": r["run_id"], "report": rep, "errors": errors})
    tol = {"q": runs[0]["checks"]["tolerance"]} if runs else {}
    rows = feacore_score(agents, tol=tol)
    if SCORE == "field":
        _score_vs_automesher(runs, rows)
        return
    for r, row in zip(runs, rows):
        # feacore's efficiency (cheapest in-tolerance DOFs / own DOFs) exceeds 1 for a run that is
        # cheaper but OUTSIDE tolerance, which can push the score above its documented [0, 1]
        # (e.g. 0.32 accuracy x 5.8 efficiency). Cap it at 1: cheaper than the cheapest accurate
        # run earns no extra credit; its missing accuracy is already in the accuracy term.
        eff = min(1.0, row["efficiency"])
        score = row["accuracy"] * eff if row["valid"] else 0.0
        r["checks"].update(score=score, accuracy=row["accuracy"], efficiency=eff, efficiency_uncapped=row["efficiency"],
                           valid=row["valid"], cost=row["cost"], pareto_rank=row["pareto_rank"],
                           invalid_because=None if row["valid"] else [f for f in row["flags"]])


def _score_vs_automesher(runs: list[dict], rows: list[dict]) -> None:
    """Field mode: score each run against the automesher (formula in ace.config). Validity still
    comes from feacore's fail flags (rows)."""
    from ace.field_error import automesher
    auto = automesher(runs[0]["problem_id"])
    G = AUTO_GAIN_TARGET
    for r, row in zip(runs, rows):
        c = r["checks"]
        if not row["valid"] or c.get("rel_error") is None:
            c.update(score=0.0, accuracy=0.0, efficiency=0.0, gain=None, valid=False,
                     invalid_because=list(row["flags"]) or ["no solve"])
            continue
        g = auto["error"] / max(c["rel_error"], FIELD_ERR_FLOOR)
        acc = (g / G) ** 2 if g < G else g / G
        eff = auto["dofs"] / c["dofs"]
        c.update(score=acc * eff, accuracy=acc, efficiency=eff, gain=g, valid=True, invalid_because=None,
                 auto_error=auto["error"], auto_dofs=auto["dofs"], best_bonus=0.0, best_points=0.0)
    _best_bonus(runs)


def _best_bonus(runs: list[dict]) -> None:
    """Field mode: on each hidden probe point, the one valid run of a generation with the smallest
    |error| there wins 1/n_hidden of BEST_BONUS. A tie pays nobody, so copying the leader earns
    nothing: only being different AND better does."""
    by_gen: dict = {}
    for r in runs:
        if r["checks"].get("valid") and r["checks"].get("point_errors"):
            by_gen.setdefault(r.get("generation"), []).append(r)
    for grp in by_gen.values():
        hidden = range(PROBE_DISCLOSED, len(grp[0]["checks"]["point_errors"]))
        for j in hidden:
            errs = [abs(r["checks"]["point_errors"][j]) for r in grp]
            best = min(errs)
            winners = [r for r, e in zip(grp, errs) if e <= best * (1 + 1e-9) + 1e-15]
            if len(winners) == 1:
                winners[0]["checks"]["best_points"] += 1
        for r in grp:
            c = r["checks"]
            c["best_bonus"] = BEST_BONUS * c["best_points"] / len(hidden)
            c["base_score"] = c["score"]
            c["score"] += c["best_bonus"]
