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
_ERRORS: dict[str, str] = {}   # module -> import error (modules being edited must not break the rest)
_DISCOVERED = False
_SKIP = {"base", "registry", "store"}


def register(problem: Problem) -> Problem:
    if problem.id in _REGISTRY and _REGISTRY[problem.id] is not problem:
        raise ValueError(f"duplicate problem id {problem.id!r}")
    _REGISTRY[problem.id] = problem
    return problem


def discover(strict: bool = False) -> dict[str, str]:
    """Import every problem module once. Import errors are collected (and warned about) instead
    of raised, unless ``strict``. Returns {module: error} for modules that failed."""
    global _DISCOVERED
    if _DISCOVERED and not strict:
        return dict(_ERRORS)
    import warnings

    import feacore.problems as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, prefix="feacore.problems.", onerror=lambda name: None):
        if mod.name.rsplit(".", 1)[-1] in _SKIP:
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            _ERRORS[mod.name] = f"{type(e).__name__}: {e}"
            warnings.warn(f"problem module {mod.name} failed to import: {e}", stacklevel=2)
    _DISCOVERED = True
    return dict(_ERRORS)


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
