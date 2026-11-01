"""Run the fixed solver (CalculiX ``ccx``) on a deck with resource accounting.

Per-process CPU time and peak memory come from ``os.wait4`` so several runs can execute in
parallel threads without mixing up their accounting.  ccx can exit with status 0 after
printing ``*ERROR``; we therefore also scan stdout for error lines.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CCX = os.environ.get("FEABENCH_CCX", "ccx")


@dataclass
class RunResult:
    jobname: str
    workdir: str
    returncode: int | None
    wall_s: float
    cpu_s: float  # user + system time of the solver process (all threads)
    max_rss_mb: float
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: int = 0
    n_equations: int | None = None
    ccx_version: str | None = None
    threads: int = 1

    @property
    def ok(self) -> bool:
        return (
            not self.timed_out
            and self.returncode == 0
            and not self.errors
            and (Path(self.workdir) / f"{self.jobname}.frd").exists()
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def ccx_version(ccx: str = DEFAULT_CCX) -> str | None:
    exe = shutil.which(ccx)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"Version\s+([\w.]+)", out)
    return m.group(1) if m else out.strip() or None


def run_ccx(
    inp: str | Path,
    timeout: float = 600.0,
    threads: int = 1,
    ccx: str = DEFAULT_CCX,
    extra_env: dict | None = None,
) -> RunResult:
    """Run ``ccx -i <jobname>`` in the directory containing ``inp``."""
    inp = Path(inp).resolve()
    jobname = inp.stem
    workdir = inp.parent
    exe = shutil.which(ccx)
    if exe is None:
        raise FileNotFoundError(f"CalculiX executable {ccx!r} not found (set FEABENCH_CCX)")
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(threads),
        CCX_NPROC_EQUATION_SOLVER=str(threads),
        CCX_NPROC_RESULTS=str(threads),
        CCX_NPROC_STIFFNESS=str(threads),
    )
    if extra_env:
        env.update(extra_env)
    out_path = workdir / f"{jobname}.stdout"
    t0 = time.monotonic()
    timed_out = False
    with open(out_path, "w") as out:
        proc = subprocess.Popen(
            [exe, "-i", jobname],
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        status, ru = None, None
        while True:
            pid, st, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                status, ru = st, usage
                break
            if time.monotonic() - t0 > timeout:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, ru = os.wait4(proc.pid, 0)
                break
            time.sleep(0.02)
        proc.returncode = os.waitstatus_to_exitcode(status) if status is not None else None
    wall = time.monotonic() - t0
    stdout = out_path.read_text(errors="replace")
    errors = [ln.strip() for ln in stdout.splitlines() if "*ERROR" in ln.upper()]
    warnings = sum(1 for ln in stdout.splitlines() if "*WARNING" in ln.upper())
    m = re.search(r"number of equations\s*\n\s*(\d+)", stdout)
    n_eq = int(m.group(1)) if m else None
    mv = re.search(r"Version\s+([\w.]+)", stdout)
    return RunResult(
        jobname=jobname,
        workdir=str(workdir),
        returncode=proc.returncode,
        wall_s=wall,
        cpu_s=(ru.ru_utime + ru.ru_stime) if ru else float("nan"),
        max_rss_mb=(ru.ru_maxrss / 1024.0) if ru else float("nan"),
        timed_out=timed_out,
        errors=errors[:20],
        warnings=warnings,
        n_equations=n_eq,
        ccx_version=mv.group(1) if mv else None,
        threads=threads,
    )
