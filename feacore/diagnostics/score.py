"""Speed / accuracy score for a population of agents that set up the same problem.

Accuracy is measured on QoIs (mesh-independent functionals: strain energy, reactions,
displacements at points, stresses away from singularities), never by comparing fields
point-by-point. Each agent's error per QoI is *estimated* from its own setup (``verify``), or
measured against a reference when the problem has one (benchmarks, used to calibrate the estimators).

Per agent:
  valid       no fail-severity flag
  err[q]      estimated (or true) relative error of QoI q; singular QoIs are excluded
  accuracy    min over QoIs of min(1, tol[q] / err[q])     (worst QoI governs; 1 = all within tol)
  meets_tol   every judged QoI within its tolerance
  cost        DOFs (deterministic, hardware-independent); wall time is reported, not scored
  efficiency  cost of the cheapest valid agent that meets tolerance / own cost
  score       valid * accuracy * efficiency  in [0, 1]; 1 = cheapest setup that is accurate enough
Also: Pareto rank on (max error, cost) among valid agents, and info flags for over-resolution
(error far below tolerance: DOFs wasted) and under-resolution.
"""

from __future__ import annotations

from feacore.diagnostics.flags import THRESHOLDS as T, Flag

DEFAULT_TOL = 0.02

# ZZ underestimates the strain-energy error of P2 L2-projection recovery by a near-constant
# factor; measured on lame_cylinder_plane_strain (two meshes, effectivity 0.39 and 0.39).
# tests/test_diagnostics.py re-measures it. Single-mesh estimates divide by this.
ZZ_ENERGY_EFFECTIVITY = 0.39


def single_mesh_errors(report) -> dict:
    """Fallback when no refinement study was run: strain-energy error from ZZ (calibrated)."""
    e = report.metrics.get("zz_energy_error")
    return {} if e is None else {"strain_energy": {"est_err": e / ZZ_ENERGY_EFFECTIVITY, "source": "zz"}}


def study_errors(study: dict) -> dict:
    return {k: {"est_err": v["est_err"], "source": "gci", "kind": v["kind"]}
            for k, v in study["qois"].items()}


def reference_errors(values: dict, references: list) -> dict:
    out = {}
    for r in references:
        if r.qoi in values and r.value:
            out[r.qoi] = {"est_err": abs(values[r.qoi] - r.value) / abs(r.value), "source": "reference"}
    return out


def score_population(agents: list[dict], tol: dict | None = None) -> list[dict]:
    """``agents``: [{"agent_id", "report": Report, "errors": {qoi: {"est_err", ...}}}].
    Returns one row per agent (same order) with the fields described in the module docstring."""
    tol = tol or {}
    rows = []
    for a in agents:
        rep = a["report"]
        valid = rep.status != "fail"
        judged = {q: e["est_err"] for q, e in a.get("errors", {}).items() if e.get("est_err") is not None}
        ratios = {q: tol.get(q, DEFAULT_TOL) / max(err, 1e-300) for q, err in judged.items()}
        accuracy = min((min(1.0, r) for r in ratios.values()), default=0.0) if valid else 0.0
        meets = valid and bool(judged) and all(r >= 1.0 for r in ratios.values())
        rows.append({"agent_id": a["agent_id"], "valid": valid, "status": rep.status,
                     "flags": rep.ids(), "errors": judged, "max_err": max(judged.values(), default=None),
                     "accuracy": accuracy, "meets_tol": meets, "cost": rep.metrics.get("n_dofs"),
                     "wall_time": rep.metrics.get("wall_time"), "notes": []})
    pool = [r for r in rows if r["meets_tol"]] or [r for r in rows if r["valid"]]
    c_star = min((r["cost"] for r in pool if r["cost"]), default=None)
    for r in rows:
        r["efficiency"] = (c_star / r["cost"]) if (r["valid"] and c_star and r["cost"]) else 0.0
        r["score"] = r["accuracy"] * r["efficiency"] if r["valid"] else 0.0
        if r["meets_tol"] and r["errors"] and all(
                e < tol.get(q, DEFAULT_TOL) * T["over_resolved_factor"] for q, e in r["errors"].items()):
            r["notes"].append(Flag("over_resolved", "cost", "info",
                                   "every QoI is >10x inside tolerance: the mesh is finer than needed").to_dict())
        elif r["valid"] and not r["meets_tol"] and r["errors"]:
            worst = max(r["errors"], key=r["errors"].get)
            r["notes"].append(Flag("not_accurate_enough", "discretization", "warn",
                                   f"{worst} error {r['errors'][worst]:.2%} > tolerance "
                                   f"{tol.get(worst, DEFAULT_TOL):.2%}").to_dict())
    _pareto(rows)
    return rows


def _pareto(rows: list[dict]) -> None:
    """Pareto rank on (max_err, cost), lower is better on both; invalid agents get None."""
    live = [r for r in rows if r["valid"] and r["max_err"] is not None and r["cost"]]
    for r in rows:
        r["pareto_rank"] = None
    rank, rest = 0, live
    while rest:
        front = [a for a in rest if not any(
            b["max_err"] <= a["max_err"] and b["cost"] <= a["cost"] and
            (b["max_err"] < a["max_err"] or b["cost"] < a["cost"]) for b in rest)]
        for a in front:
            a["pareto_rank"] = rank
        rest = [a for a in rest if a not in front]
        rank += 1
