import pytest

from feagent.parts import make_all


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep every test (and CLI subprocesses it spawns) out of the real ./.feagent/state.json."""
    monkeypatch.setenv("FEAGENT_STATE", str(tmp_path / "feagent-state.json"))


@pytest.fixture(scope="session")
def parts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("parts")
    make_all(d)
    return d
