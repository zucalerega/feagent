"""Write problems to disk for agents (public/) and the evaluator (private/).

Layout:
  benchmarks/index.json                       public list of problems + tags
  benchmarks/<id>/public/problem.json         statement for agents
  benchmarks/<id>/public/geometry.step        CAD
  benchmarks/<id>/private/reference.json      references, expectations, reference case
  benchmarks/<id>/private/validation_*.json   reference convergence studies (scripts/validate.py)

Do not hand ``private/`` to agents under test.
"""

from __future__ import annotations

import json
from pathlib import Path

from feacore.problems.base import Problem
from feacore.problems.registry import all_problems

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"


def export_problem(problem: Problem, root: str | Path = DEFAULT_ROOT, order: int = 2) -> Path:
    out = Path(root) / problem.id
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "public" / "problem.json").write_text(json.dumps(problem.statement(), indent=2))
    problem.export_step(out / "public" / "geometry.step")
    (out / "private" / "reference.json").write_text(json.dumps(problem.private(order), indent=2))
    return out


def export_all(root: str | Path = DEFAULT_ROOT, problems: list[Problem] | None = None) -> Path:
    root = Path(root)
    problems = problems if problems is not None else all_problems()
    index = []
    for p in problems:
        export_problem(p, root)
        index.append({"id": p.id, "title": p.title, "family": p.family, "tags": list(p.tags),
                      "dim": p.dim, "physics": p.physics, "formulation": p.formulation})
    (root / "index.json").write_text(json.dumps(index, indent=2))
    return root
