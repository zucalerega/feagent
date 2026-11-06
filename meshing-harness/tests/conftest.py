import os
from pathlib import Path

import pytest

from feagent.parts import make_all

SRC = Path(__file__).resolve().parents[1] / "src"     # this checkout's feagent package


@pytest.fixture(scope="session", autouse=True)
def subprocess_pythonpath():
    """Child processes (`python -m feagent.cli`, `python -m feagent.mesher`, the console script)
    import this checkout's feagent, installed or not: pytest's `pythonpath` reaches only this process."""
    old = os.environ.get("PYTHONPATH")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", os.pathsep.join([str(SRC), *([old] if old else [])]))
        yield


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep every test (and CLI subprocesses it spawns) out of the real ./.feagent/state.json."""
    monkeypatch.setenv("FEAGENT_STATE", str(tmp_path / "feagent-state.json"))


@pytest.fixture(scope="session")
def parts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("parts")
    make_all(d)
    return d
