"""Shared fixtures of the dossier tests: the example project, loaded once, and a writable copy of it."""

import shutil
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "dossier" / "examples" / "mov_valve"


@pytest.fixture(scope="session")
def example():
    """The example project, read-only (do not run stages on it: use `workproject`)."""
    from dossier.project import Project
    return Project.load(EXAMPLE)


@pytest.fixture(scope="session")
def example_ctx(example):
    from dossier import context
    return context.build(example)


@pytest.fixture()
def workproject(tmp_path):
    """A fresh copy of the example project in a temporary folder, for tests that write stage files."""
    from dossier.project import Project
    dst = tmp_path / "mov_valve"
    shutil.copytree(EXAMPLE, dst, ignore=shutil.ignore_patterns("out", "__pycache__", "reviews.yaml"))
    return Project.load(dst)
