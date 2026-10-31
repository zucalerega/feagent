import pytest

from feagent.parts import make_all


@pytest.fixture(scope="session")
def parts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("parts")
    make_all(d)
    return d
