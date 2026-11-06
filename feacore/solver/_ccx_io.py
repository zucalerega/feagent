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
    errors: list[str] = field(default_factory=list)

    @property
    def frd(self) -> Path:
        return self.workdir / f"{self.job}.frd"

    @property
    def dat(self) -> Path:
        return self.workdir / f"{self.job}.dat"


def run_ccx(inp_text: str, job: str = "job", workdir: str | Path | None = None, ccx: str | None = None,
            threads: int | None = None, timeout: float | None = None, check: bool = True) -> CcxRun:
    """Run ccx on ``inp_text``. ``check=False`` returns the run even when ccx reported an error
    (``run.errors``), e.g. to keep the converged increments of a nonlinear run that diverged."""
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
    run.errors = errs + ([f"exit status {p.returncode}"] if p.returncode != 0 else [])
    if not check and run.frd.exists():
        return run
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
    for _, b in _iter_frd(path):
        blocks[b.name] = b
    return blocks


def parse_frd_series(path: str | Path) -> list[tuple[float, dict[str, FrdBlock]]]:
    """Every output increment of an ASCII .frd: [(step time, {name: block}), ...] in file order.
    A new increment starts when the result-set time changes (``100C`` header)."""
    out: list[tuple[float, dict[str, FrdBlock]]] = []
    for t, b in _iter_frd(path):
        if not out or t != out[-1][0] or b.name in out[-1][1]:
            out.append((t, {}))
        out[-1][1][b.name] = b
    return out


def _iter_frd(path: str | Path):
    """Yield (time of the result set, FrdBlock) for every nodal result block. Streams the file, so
    multi-GB .frd files of nonlinear runs are read block by block."""
    t, state = 0.0, None             # state: None | "comps" | "data"
    name, comps, ids, vals, cur = "", [], [], [], None
    with open(path, "r", errors="replace") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if state == "comps" and not ln.startswith(" -5"):
                state = "data"
            if state == "data":
                if ln.startswith(" -1"):
                    if cur is not None:
                        vals.append(cur)
                    ids.append(int(ln[3:13]))
                    body = ln[13:].rstrip()
                    cur = [_flt(body[k:k + 12]) for k in range(0, len(body), 12)]
                elif ln.startswith(" -2") and cur is not None:
                    body = ln[13:].rstrip()
                    cur.extend(_flt(body[k:k + 12]) for k in range(0, len(body), 12))
                elif ln.startswith(" -3"):
                    if cur is not None:
                        vals.append(cur)
                    nc = len(comps)
                    arr = np.array([v[:nc] + [0.0] * (nc - len(v)) for v in vals], dtype=float).reshape(-1, nc)
                    yield t, FrdBlock(name=name, components=comps, nodes=np.array(ids, dtype=np.int64), values=arr)
                    state = None
                continue
            if state == "comps":
                c = ln[3:].split()
                # pseudo components (e.g. ALL = magnitude) carry ictype 1 / pointer 0 and are not stored
                if not (len(c) >= 6 and c[0] == "ALL"):
                    comps.append(c[0])
                continue
            if ln.startswith("  100C"):
                parts = ln.split()
                t = _flt(parts[2]) if len(parts) > 2 else t
            elif ln.startswith(" -4"):
                name, comps, ids, vals, cur, state = ln[3:].split()[0], [], [], [], None, "comps"


# ----------------------------------------------------------------------------- .dat
def parse_dat_totals(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Totals printed by ``*NODE PRINT / *EL PRINT, TOTALS=ONLY``: (quantity, SET) -> values of the
    last increment. Quantities: 'force' (RF), 'internal energy' (ELSE), 'heat generation' (RFL)."""
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


_DAT_TIME = re.compile(r"for set\s+(\S+)\s+and time\s+(\S+)")


def parse_dat_series(path: str | Path) -> list[dict]:
    """Per-increment .dat output of a nonlinear run, in file order: ``{"time": t, "totals": {(quantity,
    SET): values}, "peeq": (elems (k,), ips (k,), values (k,)) or None}``. ``peeq`` comes from
    ``*EL PRINT`` PEEQ (equivalent plastic strain at the integration points). Streams the file."""
    tot_pat = re.compile(r"^\s*total\s+(.*?)(?:\s*\([^)]*\))?\s+for set\s+(\S+)\s+and time")
    out: list[dict] = []
    head, body = None, []         # current block header line and its data lines

    def close():
        if head is None or not body:
            return
        mt = tot_pat.match(head)
        if mt:
            try:
                out[-1]["totals"][(mt.group(1).strip(), mt.group(2).upper())] = np.array(
                    [_flt(v) for v in body[0].split()])
            except ValueError:
                pass
        elif "equivalent plastic strain" in head:
            a = np.array(" ".join(body).split(), dtype=float).reshape(-1, 3)
            out[-1]["peeq"] = (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2])

    with open(path, "r", errors="replace") as f:
        for ln in f:
            m = _DAT_TIME.search(ln)
            if m is not None:
                close()
                t = _flt(m.group(2))
                if not out or out[-1]["time"] != t:
                    out.append({"time": t, "totals": {}, "peeq": None})
                head, body = ln, []
            elif head is not None:
                if ln.strip():
                    body.append(ln)
                elif body:
                    close()
                    head, body = None, []
    close()
    return out


def parse_sta(path: str | Path) -> list[dict]:
    """Converged increments in the .sta file: [{"step", "inc", "attempts", "iterations", "time" (total),
    "step_time", "dt"}, ...]."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text(errors="replace").splitlines():
        f = ln.split()   # STEP INC ATT ITRS TOT-TIME STEP-TIME INC-TIME; ATT is e.g. "1U" for a cut-back attempt
        if len(f) == 7 and f[0].isdigit() and f[1].isdigit() and f[2].isdigit():
            out.append({"step": int(f[0]), "inc": int(f[1]), "attempts": int(f[2]), "iterations": int(f[3]),
                        "time": _flt(f[4]), "step_time": _flt(f[5]), "dt": _flt(f[6])})
    return out
