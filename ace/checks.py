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

import functools
import json
import math
import tempfile
from pathlib import Path

from ace import fea
from ace.config import (AUTO_GAIN_TARGET, BEST_BONUS, DIVERSITY_BONUS, DIVERSITY_GRID, DIVERSITY_H_RATIO,
                        DIVERSITY_JACCARD, DIVERSITY_MARGIN, FIELD_ERR_FLOOR, MIN_QUALITY, PROBE_DISCLOSED,
                        SCORE, TASKS)

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
    from ace import evidence, nonlinear
    if nonlinear.is_nonlinear(problem_id):        # the frozen nonlinear ramp vs the test (ace.nonlinear)
        return nonlinear.automesher_baseline(problem_id)["mean_rel_err"] / AUTO_GAIN_TARGET
    if evidence.scored_by_test(problem_id):       # judged by its physical test (ace.evidence)
        return evidence.test_tolerance(problem_id)
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
    from ace import evidence, nonlinear
    if nonlinear.is_nonlinear(problem_id):       # scored by the frozen nonlinear ramp, not a reference
        return _check_nonlinear(problem_id, run, problem, final, out)
    # No reference value (no ground truth): the physical test is the only check (ace.evidence).
    ref = fea.reference_value(problem, task["qoi"]) if evidence.has_reference(problem_id) else None
    rel_err = abs(final["result"]["value"] - ref) / abs(ref) if ref is not None else None
    point_err = rel_err
    if SCORE == "field":                         # whole stress field vs the fine reference
        from ace.field_error import errors_of_setup
        errs = errors_of_setup(problem_id, final["mesh"], final["setup"])
        rel_err = errs["error"]
        out.update(field_error=errs["field"], probe_error=errs["points"], point_errors=errs["point_errors"])
    ev = evidence.load_active(problem_id)     # physical test data for this problem (ace.evidence)
    if ev is not None:
        out["evidence"] = evidence.predict_run(problem_id, run, ev["combined"])
        if evidence.scored_by_test(problem_id) and out["evidence"] is not None:
            # scored against the test: its error (tolerance() already widened to 2 sigma of the test)
            out["reference_rel_error"] = rel_err
            rel_err = abs(out["evidence"]["rel_err"])
            out["scored_by"] = "test"
    if rel_err is None:
        raise ValueError(f"{problem_id} has no reference value and no physical test data to score against: "
                         "give the loop its test (--evidence)")
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


def _check_nonlinear(problem_id: str, run: dict, problem, final: dict, out: dict) -> dict:
    """check_run for a nonlinear task (ace.nonlinear): the run's scored mesh through the frozen
    elastoplastic ramp, compared with the test's force-displacement curve. No field reference and
    no automesher field solve. Artifacts go to <directory of the mesh file>/nonlinear/."""
    from ace import nonlinear
    from ace.checkpoint import checkpoint
    task = TASKS[problem_id]
    cp = checkpoint(run)
    mesh = dict(cp["mesh"])
    path = Path(mesh.get("path") or "")
    if not path.is_file():                      # the .msh is gone: rebuild it from its request
        req = cp.get("request") or {}
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = fea.make_mesh(problem, path, **{k: v for k, v in req.items() if k in ("h", "order", "refinements")})
        if not mesh.get("ok"):
            return {**out, "solved": False, "failed_gate": f"could not rebuild the scored mesh: {mesh.get('error')}"}
    nl = nonlinear.run_curve(problem_id, mesh, path.parent / "nonlinear")
    ok = nl["ok"]
    nodes = nl["nodes"] or mesh["nodes"]
    out.update({
        "solved": True,
        "nonlinear": nl,
        "nodes": nodes,
        "dofs": nl["dofs"] or nodes * 3,
        "order": mesh["order"],
        "min_quality": mesh.get("min_quality"),
        "within_cap": nodes <= task["node_cap"],
        "quality_ok": (mesh.get("min_quality") or 0) >= MIN_QUALITY,
        "qoi_value": final["result"].get("value"),
        "rel_error": nl["mean_rel_err"] if ok else None,
        "point_rel_error": None,
        "meets_tol": ok and nl["mean_rel_err"] <= out["tolerance"],
        "valid": ok,
        "scored_by": "nonlinear test curve",
        "setup": bc_report(problem, final["setup"]),
    })
    gate = None if ok else f"nonlinear scoring run failed: {nl['error']}"
    if gate is None and not out["within_cap"]:
        gate = "over node cap"
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
    from ace import nonlinear
    if runs and nonlinear.is_nonlinear(runs[0]["problem_id"]):
        _score_vs_nl_automesher(runs, rows)
        return
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
    from ace import evidence
    from ace.field_error import automesher
    pid = runs[0]["problem_id"]
    auto = dict(automesher(pid))
    if evidence.scored_by_test(pid):        # the baseline's error is then its error against the test too
        auto["error"] = evidence.automesher_test_error(pid)
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
                 auto_error=auto["error"], auto_dofs=auto["dofs"], best_bonus=0.0, best_points=0.0,
                 diversity_bonus=0.0)
    _best_bonus(runs)
    _diversity_bonus(runs)


def _score_vs_nl_automesher(runs: list[dict], rows: list[dict]) -> None:
    """Nonlinear tasks: the field-mode formula against the automesher's run of the same frozen
    nonlinear ramp (ace.nonlinear.automesher_baseline), with the run's error against the test curve.
    Diversity bonus as in field mode; no be-the-best bonus (no probe points)."""
    from ace.nonlinear import automesher_baseline
    auto = automesher_baseline(runs[0]["problem_id"])
    G = AUTO_GAIN_TARGET
    for r, row in zip(runs, rows):
        c = r["checks"]
        if not row["valid"] or c.get("rel_error") is None:
            c.update(score=0.0, accuracy=0.0, efficiency=0.0, gain=None, valid=False, base_score=0.0,
                     invalid_because=list(row["flags"]) or ["no solve"])
            continue
        g = auto["mean_rel_err"] / max(c["rel_error"], FIELD_ERR_FLOOR)
        acc = (g / G) ** 2 if g < G else g / G
        eff = auto["dofs"] / c["dofs"]
        c.update(score=acc * eff, base_score=acc * eff, accuracy=acc, efficiency=eff, gain=g, valid=True,
                 invalid_because=None, auto_error=auto["mean_rel_err"], auto_dofs=auto["dofs"],
                 best_bonus=0.0, best_points=0.0, diversity_bonus=0.0)
    _diversity_bonus(runs)


def _best_bonus(runs: list[dict]) -> None:
    """Field mode: on each hidden probe point, the one valid run of a generation with the smallest
    |error| there wins 1/n_hidden of BEST_BONUS. A tie pays nobody, so copying the leader earns
    nothing: only being different AND better does."""
    from ace import evidence
    if runs and evidence.scored_by_test(runs[0]["problem_id"]):
        return      # the probe points measure the stated setup's field, not the test: no bonus
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


@functools.cache
def _diagonal(problem_id: str) -> float:
    """The part's bounding-box diagonal, from the (cached) reference's integration points (nonlinear
    tasks have no field reference: from the CAD's bounding box)."""
    from ace import nonlinear
    if nonlinear.is_nonlinear(problem_id):
        bbox = fea.geometry_summary(fea.get_problem(problem_id))["bbox"]
        return float(math.dist(bbox[:3], bbox[3:]))
    from ace.field_error import reference
    pts = reference(problem_id)["pts"]
    return float(math.dist(pts.min(axis=0), pts.max(axis=0)))


def _recipe(run: dict, grid: float) -> tuple | None:
    """(element order, global h, set of refinement targets) of the run's scored mesh, or None.
    Targets are named groups and edges as they are; boxes and points are rounded to the grid, so
    the same region moved slightly still counts as the same target."""
    from ace.checkpoint import checkpoint
    cp = checkpoint(run)
    req = cp and cp["request"]
    if not req:
        return None
    targets = set()
    for ref in req.get("refinements") or []:
        for kind in ("group", "edge", "box", "point"):
            if kind in ref:
                v = ref[kind]
                targets.add((kind, tuple(round(float(x) / grid) for x in v) if kind in ("box", "point") else v))
                break
    h = float(req.get("h") or cp["mesh"].get("h_used"))
    return cp["mesh"]["order"], h, frozenset(targets)


def _different(a: tuple, b: tuple) -> bool:
    """Structurally different mesh recipes: another order, h off by a large factor, or mostly
    different refinement targets (Jaccard distance |A Δ B| / |A ∪ B|, 0 for two empty sets)."""
    (oa, ha, ta), (ob, hb, tb) = a, b
    union = ta | tb
    jaccard = len(ta ^ tb) / len(union) if union else 0.0
    return oa != ob or max(ha, hb) / min(ha, hb) > DIVERSITY_H_RATIO or jaccard >= DIVERSITY_JACCARD


def _diversity_bonus(runs: list[dict]) -> None:
    """Field mode: +DIVERSITY_BONUS for a valid run whose recipe differs structurally from every other
    run of its generation on the problem and whose base score is within DIVERSITY_MARGIN of the best
    there. Pays for exploring a different recipe that works almost as well, which the be-the-best
    bonus alone does not. A generation with one run (the single agent's loop) gets none."""
    if not runs:
        return
    grid = DIVERSITY_GRID * _diagonal(runs[0]["problem_id"])
    by_gen: dict = {}
    for r in runs:
        rec = _recipe(r, grid) if r["checks"].get("solved") else None
        if rec is not None:
            by_gen.setdefault(r.get("generation"), []).append((r, rec))
    for grp in by_gen.values():
        valid = [(r, rec) for r, rec in grp if r["checks"].get("valid")]
        if len(grp) < 2 or not valid:
            continue
        best = max(r["checks"].get("base_score", r["checks"]["score"]) for r, _ in valid)
        for r, rec in valid:
            c = r["checks"]
            c.setdefault("base_score", c["score"])
            if (c["base_score"] >= DIVERSITY_MARGIN * best
                    and all(_different(rec, other) for o, other in grp if o is not r)):
                c["diversity_bonus"] = DIVERSITY_BONUS
                c["score"] += DIVERSITY_BONUS
