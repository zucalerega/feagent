"""Run a rendering script with an off-screen OpenGL display and an ffmpeg on PATH.

    python -m feacore.ui.offscreen SCRIPT [ARGS...]

The dashboard's renderers (pyvista / VTK) need an OpenGL context. On a box without a screen this
launcher starts the script under ``xvfb-run`` (a virtual X server, Mesa llvmpipe software rendering)
when there is no ``DISPLAY``; with a display, or without Xvfb, it runs the script as is (VTK may still
find EGL or OSMesa). Scripts that call the ``ffmpeg`` command get the ffmpeg bundled with
imageio-ffmpeg when the system has none. The script replaces this process (``exec``), so the task
lane's stop and kill reach it and the virtual display.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

SOFTWARE_GL = {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "__GLX_VENDOR_LIBRARY_NAME": "mesa"}
SCREEN = "-screen 0 1920x1080x24"


def ffmpeg_dir() -> str | None:
    """A directory holding an ``ffmpeg`` (a link to imageio-ffmpeg's binary), when the system has none."""
    if shutil.which("ffmpeg"):
        return None
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - no bundled ffmpeg either: the script reports it
        return None
    base = Path(os.environ.get("FEACORE_RUNS_DIR") or tempfile.gettempdir()) / "cache" / "bin"
    base.mkdir(parents=True, exist_ok=True)
    link = base / "ffmpeg"
    if not link.exists():
        try:
            link.symlink_to(exe)
        except FileExistsError:
            pass
    return str(base)


def display_mode(env: dict | None = None) -> str:
    """How a renderer will get its OpenGL context: display | xvfb | none (EGL / OSMesa, if VTK has them)."""
    env = os.environ if env is None else env
    if env.get("DISPLAY"):
        return "display"
    return "xvfb" if shutil.which("xvfb-run") else "none"


def command(script: str, args: list[str], env: dict) -> list[str]:
    """The command line that runs ``script`` with an off-screen display."""
    py = [sys.executable, script, *args]
    if display_mode(env) == "xvfb":
        return ["xvfb-run", "-a", "-s", SCREEN, *py]
    return py


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    env = dict(os.environ)
    if display_mode(env) == "xvfb":         # a virtual X server has no GPU: Mesa's software renderer
        for k, v in SOFTWARE_GL.items():
            env.setdefault(k, v)
    extra = ffmpeg_dir()
    if extra:
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    cmd = command(argv[0], argv[1:], env)
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
