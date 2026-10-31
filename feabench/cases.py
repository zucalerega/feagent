"""Benchmark case layout, discovery and export.

A case directory contains::

    task/            everything an agent may see (task.json, task.md, CAD files)
    reference/       evaluator-only: reference.json (+ convergence.json, field data)

Agents must only ever receive ``task/`` (use :func:`export_task`); reference values must never
be placed in an agent's workspace or in evaluator summaries given to improvers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .paths import BENCH_DIR, REPO_ROOT

SUITE_DIRS = {
    "verification": BENCH_DIR / "verification",
    # generated from downloaded data; contains licence-restricted CAD, so it lives in git-ignored data/
    "simjeb": REPO_ROOT / "data" / "simjeb" / "cases",
}


def case_dir(case: str | Path) -> Path:
    p = Path(case)
    if p.is_absolute() or (p / "task" / "task.json").exists():
        return p
    suite, _, name = str(case).partition("/")
    if suite in SUITE_DIRS and name:
        return SUITE_DIRS[suite] / name
    return BENCH_DIR / case


def list_cases(suite: str | None = None) -> list[str]:
    out = []
    for s, d in SUITE_DIRS.items():
        if suite and s != suite:
            continue
        if d.exists():
            out += [f"{s}/{p.parent.parent.name}" for p in sorted(d.glob("*/task/task.json"))]
    return out


def load_task(case: str | Path) -> dict:
    return json.loads((case_dir(case) / "task" / "task.json").read_text())


def load_reference(case: str | Path) -> dict:
    return json.loads((case_dir(case) / "reference" / "reference.json").read_text())


def export_task(case: str | Path, dest: str | Path) -> Path:
    """Copy only the agent-visible files of a case into ``dest``."""
    src = case_dir(case) / "task"
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest
