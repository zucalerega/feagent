"""Locate an OpenFOAM installation (ESI / OpenCFD line, v1912 .. current) and run its applications.

Private helper of the OpenFOAM backend. Discovery order:

1. ``FEACORE_OPENFOAM_BASHRC``: path to an OpenFOAM ``etc/bashrc`` (sourced in a subshell);
2. an already sourced environment (``WM_PROJECT_DIR`` set and ``simpleFoam`` on ``PATH``);
3. the usual install locations, newest first: ``/usr/lib/openfoam/openfoam*/etc/bashrc`` (the
   openfoam.com .deb / .rpm packages, e.g. ``openfoam2406``), ``/opt/openfoam*/etc/bashrc``,
   ``~/OpenFOAM/OpenFOAM-v*/etc/bashrc`` and ``/usr/share/openfoam/etc/bashrc`` (the Debian /
   Ubuntu ``openfoam`` package, v1912 on Ubuntu 24.04).

The environment is captured once (``bash -c '. bashrc; env -0'``) and passed explicitly to every
subprocess, so ``os.environ`` is never modified.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ENV_VAR = "FEACORE_OPENFOAM_BASHRC"
_CANDIDATES = (
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "/opt/openfoam*/etc/bashrc",
    "/opt/OpenFOAM-v*/etc/bashrc",
    "~/OpenFOAM/OpenFOAM-v*/etc/bashrc",
    "/usr/share/openfoam/etc/bashrc",
)
REQUIRED_APPS = ("simpleFoam", "gmshToFoam", "postProcess")


class FoamNotFound(RuntimeError):
    """No usable OpenFOAM installation."""


class FoamRunError(RuntimeError):
    """An OpenFOAM application failed."""


@dataclass
class FoamEnv:
    env: dict
    bashrc: str | None
    version: str                      # WM_PROJECT_VERSION, e.g. "v1912", "v2406"
    api: int | None                   # e.g. 1912, 2406 (None if unknown)
    apps: dict = field(default_factory=dict)

    def describe(self) -> str:
        src = self.bashrc or "the calling environment"
        return f"OpenFOAM {self.version} (api {self.api}) from {src}"


def _version_key(path: str) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", path)]
    return (1 if "/usr/share/openfoam" not in path else 0, nums)


def _source(bashrc: str) -> dict:
    out = subprocess.run(
        # 'set --': the bashrc evaluates its own arguments (file names are sourced again)
        ["bash", "--norc", "--noprofile", "-c", 'f="$1"; set --; . "$f" >/dev/null 2>&1; env -0', "_", bashrc],
        capture_output=True, timeout=60, env={"HOME": os.environ.get("HOME", "/tmp"),
                                              "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                              "USER": os.environ.get("USER", "user")})
    env = {}
    for item in out.stdout.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def _api(env: dict) -> int | None:
    for key in ("FOAM_API", "WM_PROJECT_API"):
        v = env.get(key, "")
        if v.isdigit():
            return int(v)
    m = re.search(r"(\d{4})", env.get("WM_PROJECT_VERSION", ""))
    return int(m.group(1)) if m else None


def _make(env: dict, bashrc: str | None) -> FoamEnv | None:
    if not env.get("WM_PROJECT_DIR"):
        return None
    path = env.get("PATH", "")
    apps = {a: shutil.which(a, path=path) for a in REQUIRED_APPS}
    if not all(apps.values()):
        return None
    env = dict(env)
    env.setdefault("OMP_NUM_THREADS", "1")
    return FoamEnv(env=env, bashrc=bashrc, version=env.get("WM_PROJECT_VERSION", "?"), api=_api(env), apps=apps)


@lru_cache(maxsize=1)
def find_openfoam() -> FoamEnv | None:
    """The OpenFOAM environment, or None when no installation is found."""
    explicit = os.environ.get(ENV_VAR)
    if explicit:
        fe = _make(_source(explicit), explicit) if Path(explicit).is_file() else None
        if fe is None:
            raise FoamNotFound(f"{ENV_VAR}={explicit!r} does not give a usable OpenFOAM environment "
                               f"(needs {', '.join(REQUIRED_APPS)})")
        return fe
    fe = _make(dict(os.environ), None)
    if fe is not None:
        return fe
    paths: list[str] = []
    for pat in _CANDIDATES:
        paths += glob.glob(os.path.expanduser(pat))
    for bashrc in sorted(set(paths), key=_version_key, reverse=True):
        fe = _make(_source(bashrc), bashrc)
        if fe is not None:
            return fe
    return None


def require_openfoam() -> FoamEnv:
    fe = find_openfoam()
    if fe is None:
        raise FoamNotFound(
            "OpenFOAM not found: install the ESI/OpenCFD line (v1912 or newer; see docs/solver_openfoam.md) "
            f"or point {ENV_VAR} at its etc/bashrc")
    if fe.version.isdigit() or fe.version.lower() == "dev":
        raise FoamNotFound(f"{fe.describe()} looks like the openfoam.org line, which is not supported "
                           f"(it has no simpleFoam since version 11); install the ESI line (openfoam.com)")
    return fe


def run(fe: FoamEnv, args: list[str], case_dir: str | Path, log: str | None = None,
        timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run an OpenFOAM application in ``case_dir``; the output goes to ``case_dir/log.<log>``."""
    case_dir = Path(case_dir)
    env = dict(fe.env)
    env["PWD"] = str(case_dir)
    name = log or args[0]
    logf = case_dir / f"log.{name}"
    try:
        p = subprocess.run(args, cwd=case_dir, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        logf.write_text(out)
        raise FoamRunError(f"{' '.join(args)} timed out after {timeout} s (log: {logf})") from e
    logf.write_text(p.stdout + ("\n" + p.stderr if p.stderr else ""))
    if check and (p.returncode != 0 or "FOAM FATAL" in p.stdout or "FOAM FATAL" in p.stderr):
        tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-25:])
        raise FoamRunError(f"{' '.join(args)} failed (exit {p.returncode}); last lines of {logf}:\n{tail}")
    return p


def run_monitored(fe: FoamEnv, args: list[str], case_dir: str | Path, on_line, timeout: float | None = None,
                  log: str | None = None) -> tuple[int, str, str | None]:
    """Run a solver that honours ``-opt-switch stopAtWriteNowSignal=<SIGUSR1>`` (appended here),
    feeding every output line to ``on_line(line) -> bool``; a True return (or the timeout) makes
    the solver write the current iteration and stop. Returns (exit code, output, stop reason)."""
    import signal
    import time as _time

    case_dir = Path(case_dir)
    env = dict(fe.env)
    env["PWD"] = str(case_dir)
    sig = int(signal.SIGUSR1)
    cmd = list(args) + ["-opt-switch", f"stopAtWriteNowSignal={sig}"]
    t0 = _time.monotonic()
    out: list[str] = []
    reason = None
    with subprocess.Popen(cmd, cwd=case_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            out.append(line)
            if reason is None:
                stop = on_line(line)
                if not stop and timeout is not None and _time.monotonic() - t0 > timeout:
                    stop, reason = True, "timeout"
                if stop:
                    reason = reason or "stagnated"
                    try:
                        proc.send_signal(sig)
                    except ProcessLookupError:
                        pass
        code = proc.wait()
    text = "".join(out)
    (case_dir / f"log.{log or args[0]}").write_text(text)
    return code, text, reason
