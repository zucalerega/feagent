"""Scoring: per-QoI accuracy credit, gates, efficiency, case score and population statistics.

The rationale and the alternatives considered are in docs/eval-design.md.  In short:

    S_case = G * A * (1 - w_eff * (1 - E))

* G (gates, 0/1): solver ran cleanly, geometry volume matches the CAD, load path correct
  (reaction QoIs flagged ``gate``), within budget.  A wrong-but-cheap model scores 0.
* A (accuracy, 0..1): weighted mean of per-QoI credits; credit is 1 inside the tolerance and
  decays linearly in log10(error / tolerance), reaching 0 at ``10**decades`` times the tolerance.
* E (efficiency, 0..1): 1 at or below the reference cost, 0 at the budget cap, log-linear in
  between; cost = LLM tokens + solver CPU in dollars, and wall time, combined geometrically.
* w_eff (default 0.2): efficiency can remove at most 20 % of an accurate solution's score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


# $ per million tokens (Anthropic first-party list prices, cached 2026-10-06; check current pricing).
# cache_write = 1.25 x input (5-minute TTL).  Batch API = 50 % of these.
PRICES = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


@dataclass
class ScoreConfig:
    credit_decades: float = 1.0  # zero accuracy credit at tolerance * 10**decades
    efficiency_weight: float = 0.2
    cost_ref_usd: float = 0.5  # per-case defaults; cases may override in reference.json "budget"
    cost_max_usd: float = 5.0
    time_ref_s: float = 300.0
    time_max_s: float = 3600.0
    # token prices of the model the agents run on (default: Claude Opus 5.5); see PRICES
    price_input_per_mtok: float = 4.0
    price_output_per_mtok: float = 20.0
    price_cache_read_per_mtok: float = 0.20
    price_cache_write_per_mtok: float = 5.0
    price_cpu_hour: float = 0.05  # on-demand cloud vCPU-hour, order of magnitude

    @classmethod
    def for_model(cls, model: str, **kw) -> "ScoreConfig":
        p = PRICES[model]
        return cls(price_input_per_mtok=p["input"], price_output_per_mtok=p["output"],
                   price_cache_read_per_mtok=p["cache_read"], price_cache_write_per_mtok=p["cache_write"], **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def relative_error(value: float, reference: float, floor: float = 0.0) -> float:
    den = max(abs(reference), floor)
    if den == 0:
        return abs(value - reference)
    return abs(value - reference) / den


def qoi_credit(err: float, tol: float, decades: float = 1.0) -> float:
    if not math.isfinite(err):
        return 0.0
    if err <= tol:
        return 1.0
    return max(0.0, 1.0 - math.log10(err / tol) / decades)


def efficiency(cost: float | None, ref: float, cap: float) -> float:
    if cost is None or not math.isfinite(cost):
        return 1.0  # unknown cost is not penalised; report it as missing instead
    if cost <= ref:
        return 1.0
    if cost >= cap:
        return 0.0
    return math.log(cap / cost) / math.log(cap / ref)


def cost_usd(agent_cost: dict | None, solver_cpu_s: float | None, cfg: ScoreConfig) -> float | None:
    ac = agent_cost or {}
    if "usd" in ac:
        usd = float(ac["usd"])
    else:
        usd = (ac.get("input_tokens", 0) * cfg.price_input_per_mtok
               + ac.get("output_tokens", 0) * cfg.price_output_per_mtok
               + ac.get("cache_read_tokens", 0) * cfg.price_cache_read_per_mtok
               + ac.get("cache_write_tokens", 0) * cfg.price_cache_write_per_mtok) / 1e6
    cpu = (solver_cpu_s or 0.0) + float(ac.get("solver_cpu_s_during_episode", 0.0))
    usd += cpu / 3600.0 * cfg.price_cpu_hour
    return usd


def score_qoi(spec: dict, measured: dict | None, cfg: ScoreConfig) -> dict:
    ref = None if spec.get("reference") is None else float(spec["reference"])
    tol = spec.get("tolerance", {"rel": 0.01})
    out = {"id": spec["id"], "reference": ref, "weight": float(spec.get("weight", 1.0)),
           "gate": bool(spec.get("gate", False))}
    if ref is None:  # numerical reference not computed yet (convergence study pending)
        out.update(value=None if not measured else measured.get("value"), error=None, credit=0.0, passed=False,
                   failure="reference pending")
        return out
    if measured is None or "value" not in measured:
        out.update(value=None, error=None, credit=0.0, passed=False, failure=(measured or {}).get("error"))
        return out
    val = float(measured["value"])
    if "abs" in tol:
        err = abs(val - ref)
        t = float(tol["abs"])
    else:
        err = relative_error(val, ref, float(spec.get("scale", 0.0)))
        t = float(tol["rel"])
    out.update(value=val, error=err, tolerance=t, error_type="abs" if "abs" in tol else "rel",
               credit=qoi_credit(err, t, cfg.credit_decades), passed=bool(err <= t))
    out.update({k: v for k, v in measured.items() if k not in ("value",)})
    return out


def case_score(qoi_scores: list[dict], gates: dict[str, bool], cost: float | None, wall_s: float | None,
               cfg: ScoreConfig, budget: dict | None = None) -> dict:
    budget = budget or {}
    gate_qois = {f"qoi:{q['id']}": q["passed"] for q in qoi_scores if q["gate"]}
    all_gates = {**gates, **gate_qois}
    G = 1.0 if all(all_gates.values()) else 0.0
    acc = [q for q in qoi_scores if not q["gate"]]
    wsum = sum(q["weight"] for q in acc)
    A = sum(q["weight"] * q["credit"] for q in acc) / wsum if wsum else 0.0
    E_cost = efficiency(cost, budget.get("cost_ref_usd", cfg.cost_ref_usd), budget.get("cost_max_usd", cfg.cost_max_usd))
    E_time = efficiency(wall_s, budget.get("time_ref_s", cfg.time_ref_s), budget.get("time_max_s", cfg.time_max_s))
    E = math.sqrt(E_cost * E_time)
    S = G * A * (1.0 - cfg.efficiency_weight * (1.0 - E))
    return {
        "score": S, "gates_passed": bool(G), "gates": all_gates, "accuracy": A, "efficiency": E,
        "efficiency_cost": E_cost, "efficiency_time": E_time, "cost_usd": cost, "wall_s": wall_s,
        "all_qois_within_tolerance": all(q["passed"] for q in qoi_scores) if qoi_scores else False,
        "n_qois_passed": sum(q["passed"] for q in qoi_scores), "n_qois": len(qoi_scores),
    }


# ----------------------------------------------------------------------------- population level

def pareto_front(acc: list[float], cost: list[float]) -> list[int]:
    """Indices not dominated in (maximise accuracy, minimise cost)."""
    idx = []
    for i, (a, c) in enumerate(zip(acc, cost)):
        dominated = any((a2 >= a and c2 <= c) and (a2 > a or c2 < c) for a2, c2 in zip(acc, cost))
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: cost[i])


def hypervolume_2d(acc: list[float], cost: list[float], cost_cap: float) -> float:
    """Area dominated by the population in (accuracy in [0,1], log-cost up to the cap).

    Uses log10(cost_cap / cost) as the 'cheapness' axis so one decade of savings counts the same
    everywhere; a single population number that rewards both accurate and cheap solutions.
    """
    pts = sorted(((a, math.log10(cost_cap / max(c, 1e-12))) for a, c in zip(acc, cost) if c < cost_cap),
                 key=lambda p: -p[1])
    hv, best_a = 0.0, 0.0
    for i, (a, ch) in enumerate(pts):
        best_a = max(best_a, a)
        nxt = pts[i + 1][1] if i + 1 < len(pts) else 0.0
        hv += best_a * (ch - nxt)
    return hv


def bradley_terry(results: list[dict], iters: int = 200, prior: float = 1.0) -> dict[str, float]:
    """Ratings from within-case pairwise comparisons of ``score``.

    ``results``: [{"agent": a, "case": c, "score": s}, ...].  Agents are only compared with
    agents that solved the *same* case, which removes case difficulty from the comparison
    (agents are assigned to random cases).  Ties count half.  A weak prior (virtual tie with
    an average opponent) keeps ratings finite for undefeated/winless agents.  Returns log-strengths
    (0 = average), comparable across iterations.
    """
    by_case: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        by_case.setdefault(r["case"], []).append((r["agent"], float(r["score"])))
    agents = sorted({r["agent"] for r in results})
    ix = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    W = np.full(n, prior * 0.5)
    N = np.zeros((n, n))
    for rows in by_case.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, sa = rows[i]
                b, sb = rows[j]
                ia, ib = ix[a], ix[b]
                N[ia, ib] += 1
                N[ib, ia] += 1
                if abs(sa - sb) < 1e-9:
                    W[ia] += 0.5
                    W[ib] += 0.5
                elif sa > sb:
                    W[ia] += 1
                else:
                    W[ib] += 1
    p = np.ones(n)
    for _ in range(iters):
        denom = np.array([np.sum(N[i] / (p[i] + p)) + prior / (p[i] + 1.0) for i in range(n)])
        p = W / denom
        p /= np.exp(np.mean(np.log(p)))
    return {a: float(np.log(p[ix[a]])) for a in agents}


def population_summary(results: list[dict], cost_cap: float | None = None) -> dict:
    """Population-level metrics over a list of case evaluations (each with agent, case, score...)."""
    if not results:
        return {}
    cases = sorted({r["case"] for r in results})
    solved = {c: any(r.get("all_qois_within_tolerance") for r in results if r["case"] == c) for c in cases}
    acc = [r.get("accuracy", 0.0) * (1.0 if r.get("gates_passed") else 0.0) for r in results]
    cost = [r.get("cost_usd") or float("nan") for r in results]
    finite = [i for i, c in enumerate(cost) if math.isfinite(c)]
    out = {
        "n_evaluations": len(results),
        "pass_rate": float(np.mean([bool(r.get("all_qois_within_tolerance")) and bool(r.get("gates_passed")) for r in results])),
        "gate_failure_rate": float(np.mean([not r.get("gates_passed") for r in results])),
        "mean_score": float(np.mean([r.get("score", 0.0) for r in results])),
        "median_score": float(np.median([r.get("score", 0.0) for r in results])),
        "coverage": float(np.mean(list(solved.values()))),  # cases solved by >= 1 agent
        "ratings": bradley_terry(results),
    }
    if finite:
        cap = cost_cap or max(cost[i] for i in finite) * 10
        fronts, hvs = {}, []
        for case in cases:  # trade-offs are only meaningful within a case
            idx = [i for i in finite if results[i]["case"] == case]
            a = [acc[i] for i in idx]
            c = [cost[i] for i in idx]
            fronts[case] = [results[idx[k]]["agent"] for k in pareto_front(a, c)]
            hvs.append(hypervolume_2d(a, c, cap))
        out["pareto_front"] = fronts
        out["pareto_agents"] = sorted({ag for f in fronts.values() for ag in f})
        out["hypervolume"] = float(np.mean(hvs))  # mean over cases
        out["median_cost_usd"] = float(np.median([cost[i] for i in finite]))
    return out
