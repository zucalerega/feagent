"""Attribution: how much did each playbook bullet change the score? Plain statistics, no LLM.

Every run is shown a random subset of the playbook (each bullet hidden with probability HIDE_P,
see Playbook.shown_ids). Because hiding is random, the difference in score between runs that saw
a bullet and runs that did not is a causal estimate of that bullet's effect. It does not rely on
what the agent says it used (citations are self-reported and often wrong).

The estimate is one least-squares fit:  score = cell effect + sum over bullets of (effect x shown).
A cell is a (generation, problem) pair, so differences in difficulty between problems and drift
between generations cancel out; fitting all bullets together keeps one bullet's effect from
leaking into another's. Evidence for a bullet accumulates over every generation it was in the
playbook. Confidence intervals come from a bootstrap over runs (resample runs, refit).
"""

from __future__ import annotations

import numpy as np

from ace.config import BOOTSTRAP, CI


def run_rows(runs: list[dict], checks: dict[str, dict]) -> list[dict]:
    """The fields attribution needs from each run record (crashed and gen-0 runs are skipped)."""
    rows = []
    for r in runs:
        if r.get("error") or r.get("offered") is None:
            continue
        rows.append({"run_id": r["run_id"], "gen": r["generation"], "problem": r["problem_id"],
                     "score": checks[r["run_id"]]["score"], "offered": r["offered"], "shown": r["shown"],
                     "cited": r.get("cited", [])})
    return rows


def _fit(rows: list[dict], bids: list[str]) -> np.ndarray | None:
    """Least-squares bullet effects with one fixed effect per (generation, problem) cell.
    A bullet missing from a run's playbook counts as not shown (the run did not see it)."""
    cells = sorted({(r["gen"], r["problem"]) for r in rows})
    if len(rows) <= len(cells) + len(bids):
        return None                                  # not enough runs to separate the effects
    X = np.zeros((len(rows), len(cells) + len(bids)))
    for i, r in enumerate(rows):
        X[i, cells.index((r["gen"], r["problem"]))] = 1
        for j, b in enumerate(bids):
            X[i, len(cells) + j] = b in r["shown"]
    y = np.array([r["score"] for r in rows])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None                                  # e.g. a bullet shown in every run of a cell
    return np.linalg.lstsq(X, y, rcond=None)[0][len(cells):]


def _with_interval(rows, bids, rng, n_boot):
    """Point estimates plus percentile bootstrap intervals (resample runs within each cell)."""
    est = _fit(rows, bids)
    if est is None:
        return None, None
    by_cell: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by_cell.setdefault((r["gen"], r["problem"]), []).append(i)
    boots = []
    for _ in range(n_boot):
        idx = [j for ids in by_cell.values() for j in rng.choice(ids, len(ids))]
        b = _fit([rows[j] for j in idx], bids)
        if b is not None:
            boots.append(b)
    if len(boots) < n_boot // 2:                     # resamples too often degenerate: no interval
        return est, None
    tail = (1 - CI) / 2 * 100
    boots = np.array(boots)
    return est, (np.percentile(boots, tail, axis=0), np.percentile(boots, 100 - tail, axis=0))


def estimate(rows: list[dict], train: list[str], n_boot: int = BOOTSTRAP, seed: int = 0) -> dict[str, dict]:
    """Evidence per bullet ID, pooled over all generations in `rows`:
      effect, lo, hi     effect on the score of seeing the bullet, with a CI-level interval
      n_shown, n_hidden  runs in each arm (only runs whose playbook contained the bullet)
      effect_train, effect_holdout   the same fit on train / held-out runs only (holdout = transfer)
      cite_rate          share of runs that saw the bullet and listed it in `using`
    Effects are None when there are too few runs to separate them.
    """
    rng = np.random.default_rng(seed)
    bids = sorted({b for r in rows for b in r["offered"]})
    est, ci = _with_interval(rows, bids, rng, n_boot)
    split = {}
    for name, keep in (("train", lambda r: r["problem"] in train), ("holdout", lambda r: r["problem"] not in train)):
        split[name] = _fit([r for r in rows if keep(r)], bids)
    out = {}
    for j, bid in enumerate(bids):
        offered = [r for r in rows if bid in r["offered"]]
        shown = [r for r in offered if bid in r["shown"]]
        pick = lambda a: None if a is None else float(a[j])   # noqa: E731
        out[bid] = {
            "effect": pick(est), "lo": pick(ci[0]) if ci else None, "hi": pick(ci[1]) if ci else None,
            "n_shown": len(shown), "n_hidden": len(offered) - len(shown),
            "effect_train": pick(split["train"]), "effect_holdout": pick(split["holdout"]),
            "cite_rate": float(np.mean([bid in r["cited"] for r in shown])) if shown else None,
        }
    return out


def table(evidence: dict[str, dict], playbook) -> str:
    """The evidence as a plain-text table for the Reflector and Curator prompts."""
    if not evidence:
        return "(no bullets have been tested yet)"
    f = lambda x: "  n/a" if x is None else f"{x:+.2f}"   # noqa: E731
    lines = [f"effect = mean score of runs that saw the bullet minus runs that did not "
             f"({int(CI * 100)}% interval); positive helps. Scores are log10 error ratios.",
             "id         effect  [low, high]     shown/hidden  train  holdout  cited  text"]
    for bid, e in sorted(evidence.items(), key=lambda kv: -(kv[1]["effect"] or 0)):
        b = playbook.get(bid)
        text = b.content if b else "(retired)"
        cite = "  n/a" if e["cite_rate"] is None else f"{e['cite_rate']:5.0%}"
        lines.append(f"{bid:10s} {f(e['effect'])}  [{f(e['lo'])}, {f(e['hi'])}]  {e['n_shown']:5d}/{e['n_hidden']:<6d}"
                     f"  {f(e['effect_train'])}  {f(e['effect_holdout'])}  {cite}  {text}")
    return "\n".join(lines)
