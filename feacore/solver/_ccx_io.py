"""CalculiX I/O: run ccx in a scratch directory, parse the ASCII .frd and the .dat totals."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class CcxError(RuntimeError):
    """CalculiX failed or produced no results."""


def fnum(v: float) -> str:
    """Float for the .inp deck: 14 significant digits, at most 20 characters (ccx truncates
    longer fields and then fails to parse them)."""
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"non-finite value {v!r} in the CalculiX deck")
    if v != 0.0 and abs(v) < 1e-90:
        v = 0.0
    s = f"{v:.14g}"
    if len(s) > 20:          # pragma: no cover - 3-digit exponents
        s = f"{v:.12g}"
    return s


def id_lines(ids, per_line: int = 16) -> str:
    """1-based ids, comma separated, 16 per line (CalculiX limit)."""
    ids = [str(int(i)) for i in ids]
    return "\n".join(", ".join(ids[i:i + per_line]) for i in range(0, len(ids), per_line))


def find_ccx(path: str | None = None) -> str:
    cand = path or os.environ.get("FEACORE_CCX") or shutil.which("ccx") or shutil.which("ccx_2.21")
    if not cand or not (Path(cand).exists() or shutil.which(cand)):
        raise CcxError("CalculiX executable not found (install calculix-ccx or set FEACORE_CCX)")
    return cand


@dataclass
class CcxRun:
    workdir: Path
    job: str
    stdout: str
    returncode: int
    wall_time: float
    warnings: list[str] = field(default_factory=list)

    @property
    def frd(self) -> Path:
        return self.workdir / f"{self.job}.frd"

    @property
    def dat(self) -> Path:
        return self.workdir / f"{self.job}.dat"


def run_ccx(inp_text: str, job: str = "job", workdir: str | Path | None = None, ccx: str | None = None,
            threads: int | None = None, timeout: float | None = None) -> CcxRun:
    import time

    exe = find_ccx(ccx)
    if workdir is None:
        base = Path(os.environ.get("FEACORE_CCX_TMP", tempfile.gettempdir())) / "ccx"
        base.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="ccx_", dir=base))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{job}.inp").write_text(inp_text)
    env = dict(os.environ)
    n = str(int(threads)) if threads else env.get("OMP_NUM_THREADS", "1")
    env.update(OMP_NUM_THREADS=n, CCX_NPROC_STIFFNESS=n, CCX_NPROC_RESULTS=n, CCX_NPROC_EQUATION_SOLVER=n)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([exe, "-i", job], cwd=workdir, env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise CcxError(f"ccx timed out after {timeout} s (workdir {workdir})") from e
    wall = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    run = CcxRun(workdir=workdir, job=job, stdout=out, returncode=p.returncode, wall_time=wall)
    errs = [ln.strip() for ln in out.splitlines() if "*ERROR" in ln]
    run.warnings = sorted({ln.strip() for ln in out.splitlines() if "*WARNING" in ln})[:20]
    if errs or p.returncode != 0 or not run.frd.exists():
        tail = "\n".join(out.splitlines()[-25:])
        raise CcxError(f"ccx failed (exit {p.returncode}) in {workdir}:\n" + ("\n".join(errs[:10]) or tail))
    return run


# ----------------------------------------------------------------------------- .frd
_FLOAT_FIX = re.compile(r"(\d)([+-]\d{2,3})$")


def _flt(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return float(_FLOAT_FIX.sub(r"\1E\2", s))


@dataclass
class FrdBlock:
    name: str
    components: list[str]
    nodes: np.ndarray        # CalculiX node ids
    values: np.ndarray       # (n, ncomp)
    step: int = 1


def parse_frd(path: str | Path) -> dict[str, FrdBlock]:
    """Nodal result blocks of the last increment in an ASCII .frd (name -> block)."""
    blocks: dict[str, FrdBlock] = {}
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith(" -4"):
            parts = ln[3:].split()
            name = parts[0]
            comps, i = [], i + 1
            while i < n and lines[i].startswith(" -5"):
                c = lines[i][3:].split()
                # pseudo components (e.g. ALL = magnitude) carry ictype 1 / pointer 0 and are not stored
                if not (len(c) >= 6 and c[0] == "ALL"):
                    comps.append(c[0])
                i += 1
            nc = len(comps)
            ids, vals = [], []
            cur: list[float] | None = None
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -1"):
                    if cur is not None:
                        vals.append(cur)
                    ids.append(int(s[3:13]))
                    body = s[13:]
                    cur = [_flt(body[k:k + 12]) for k in range(0, len(body.rstrip()), 12)]
                elif s.startswith(" -2") and cur is not None:
                    body = s[13:]
                    cur.extend(_flt(body[k:k + 12]) for k in range(0, len(body.rstrip()), 12))
                i += 1
            if cur is not None:
                vals.append(cur)
            arr = np.array([v[:nc] + [0.0] * (nc - len(v)) for v in vals], dtype=float).reshape(-1, nc)
            blocks[name] = FrdBlock(name=name, components=comps, nodes=np.array(ids, dtype=np.int64), values=arr)
        i += 1
    return blocks


# ----------------------------------------------------------------------------- .dat
def parse_dat_totals(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Totals printed by ``*NODE PRINT / *EL PRINT, TOTALS=ONLY``: (quantity, SET) -> values of the
    last increment. Quantities: 'force' (RF), 'internal energy' (ELSE), 'flux' (RFL), ..."""
    out: dict[tuple[str, str], np.ndarray] = {}
    lines = Path(path).read_text(errors="replace").splitlines()
    pat = re.compile(r"^\s*total\s+(.*?)(?:\s*\([^)]*\))?\s+for set\s+(\S+)\s+and time")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            try:
                out[(m.group(1).strip(), m.group(2).upper())] = np.array([_flt(t) for t in lines[j].split()])
            except ValueError:
                pass
    return out
