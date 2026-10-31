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


def write_index(root: str | Path, problems: list[Problem]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = [{"id": p.id, "title": p.title, "family": p.family, "tags": list(p.tags), "dim": p.dim,
              "physics": p.physics, "formulation": p.formulation} for p in problems]
    path = root / "index.json"
    path.write_text(json.dumps(index, indent=2))
    return path


def export_all(root: str | Path = DEFAULT_ROOT, problems: list[Problem] | None = None
               ) -> tuple[list[Problem], dict[str, str]]:
    """Export problems (default: all registered) and write index.json for the ones that succeeded.
    Returns (exported, {problem_id: error})."""
    problems = problems if problems is not None else all_problems()
    done, failed = [], {}
    for p in problems:
        try:
            export_problem(p, root)
            done.append(p)
        except Exception as e:  # noqa: BLE001 - one broken problem must not block the rest
            failed[p.id] = f"{type(e).__name__}: {e}"
    write_index(root, done)
    return done, failed
