"""Token and dollar accounting. Every LLM call (real or mock) produces one usage record."""

from __future__ import annotations

import json
from pathlib import Path

from ace.config import PRICES


class BudgetExceeded(RuntimeError):
    pass


def usage_record(role: str, model: str, usage, run_id: str = "") -> dict:
    """Turn an API `usage` object into a plain dict with an estimated $ cost."""
    get = lambda k: getattr(usage, k, None) or 0  # noqa: E731  (missing/None fields count as 0)
    rec = {
        "role": role,
        "model": model,
        "run_id": run_id,
        "input": get("input_tokens"),
        "output": get("output_tokens"),
        "cache_write": get("cache_creation_input_tokens"),
        "cache_read": get("cache_read_input_tokens"),
    }
    p = PRICES.get(model)
    if p is None:  # unknown model: count tokens but flag the cost as unknown
        rec["usd"] = 0.0
        rec["price_unknown"] = True
    else:
        rec["usd"] = sum(rec[k] * p[k] for k in ("input", "output", "cache_write", "cache_read")) / 1e6
    return rec


class CostLog:
    """Append-only log (cost.jsonl) plus a running total checked against the budget."""

    def __init__(self, path: Path, budget_usd: float):
        self.path = Path(path)
        self.budget = budget_usd
        self.total = 0.0

    def add(self, records: list[dict]) -> None:
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                self.total += r["usd"]

    def over_budget(self) -> bool:
        return self.total >= self.budget

    def check(self) -> None:
        if self.over_budget():
            raise BudgetExceeded(f"spent ${self.total:.2f} of ${self.budget:.2f}")

    def by_role(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in self.path.read_text().splitlines() if self.path.exists() else []:
            r = json.loads(line)
            out[r["role"]] = out.get(r["role"], 0.0) + r["usd"]
        return out
