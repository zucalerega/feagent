"""Problem registry with auto-discovery of every module under ``feacore.problems``.

Modules register instances at import time:

    from feacore.problems.registry import register
    register(MyProblem())                     # single problem
    for w in (2, 4, 8):                       # a parametric family
        register(MyProblem(id=f"plate_hole_w{w}", width=w))
"""

from __future__ import annotations

import importlib
import pkgutil

from feacore.problems.base import Problem

_REGISTRY: dict[str, Problem] = {}
_DISCOVERED = False
_SKIP = {"base", "registry", "store"}


def register(problem: Problem) -> Problem:
    if problem.id in _REGISTRY and _REGISTRY[problem.id] is not problem:
        raise ValueError(f"duplicate problem id {problem.id!r}")
    _REGISTRY[problem.id] = problem
    return problem


def discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import feacore.problems as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, prefix="feacore.problems."):
        if mod.name.rsplit(".", 1)[-1] in _SKIP:
            continue
        importlib.import_module(mod.name)
    _DISCOVERED = True


def get(problem_id: str) -> Problem:
    discover()
    try:
        return _REGISTRY[problem_id]
    except KeyError:
        raise KeyError(f"unknown problem {problem_id!r}; known: {sorted(_REGISTRY)}") from None


def all_problems(tags: tuple[str, ...] | list[str] | None = None, family: str | None = None) -> list[Problem]:
    discover()
    out = list(_REGISTRY.values())
    if tags:
        out = [p for p in out if set(tags) <= set(p.tags)]
    if family:
        out = [p for p in out if p.family == family]
    return sorted(out, key=lambda p: p.id)
