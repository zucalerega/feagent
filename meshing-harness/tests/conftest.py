import functools
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from feagent.parts import make_all

SRC = Path(__file__).resolve().parents[1] / "src"     # this checkout's feagent package

_PROBE = ("import pyvista as pv\n"
          "pl = pv.Plotter(off_screen=True, window_size=[64, 64])\n"
          "pl.add_mesh(pv.Sphere())\n"
          "pl.screenshot(return_img=True)\n"
          "pl.close()\n")


@functools.cache
def render_unavailable() -> str | None:
    """Why off-screen VTK rendering cannot work on this machine, or None if it can.

    Probed once, in a child process: a VTK that finds no X display, EGL or OSMesa does not raise,
    it segfaults, which would kill the whole pytest run (no summary)."""
    try:
        p = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "off-screen VTK rendering hung in a probe render"
    if p.returncode == 0:
        return None
    how = signal.Signals(-p.returncode).name if p.returncode < 0 else f"exit code {p.returncode}"
    return (f"no off-screen VTK rendering here (probe render: {how}; DISPLAY "
            f"{os.environ.get('DISPLAY') or 'unset'}, no usable EGL or OSMesa): run under `xvfb-run -a`")


@pytest.fixture(scope="session", autouse=True)
def subprocess_pythonpath():
    """Child processes (`python -m feagent.cli`, `python -m feagent.mesher`, the console script)
    import this checkout's feagent, installed or not: pytest's `pythonpath` reaches only this process."""
    old = os.environ.get("PYTHONPATH")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", os.pathsep.join([str(SRC), *([old] if old else [])]))
        yield


@pytest.fixture(scope="session")
def renderer():
    """For tests (and fixtures) that render, in this process or in a child (CLI, MCP tools): they
    skip, with the reason, when this machine cannot render."""
    why = render_unavailable()
    if why:
        pytest.skip(why)


@pytest.fixture(scope="session", autouse=True)
def no_render_segfault():
    """Safety net for a render not marked with `renderer`: without a working renderer, creating a
    pyvista Plotter in this process skips the test instead of segfaulting the whole run."""
    why = render_unavailable()
    if why is None:
        yield
        return
    import pyvista

    def refuse(*args, **kwargs):
        pytest.skip(why)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pyvista, "Plotter", refuse)
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
