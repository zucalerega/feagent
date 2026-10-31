"""CalculiX ``.dat`` files: modal/buckling output parsing and the upstream regression check.

``compare_dat`` reimplements upstream ``test/datcheck.pl`` (ccx test suite) in Python:

* the reference file is split into blocks at blank lines; for each block the largest
  absolute value among tokens containing a '.' is recorded;
* the two files are walked in lockstep, line by line; lines of the *computed* file that
  contain ``time`` and participation-factor blocks are skipped;
* every numeric token is compared, and the error is measured relative to the block maximum
  (``|ref - val| / max|ref in block|``); upstream flags errors above 1e-3 (0.1 %).

Upstream's ``compare`` script additionally fails a test if the .dat is missing, has a
different number of lines than the reference, or contains ``NaN``; we do the same.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_EIG_HDR = "E I G E N V A L U E   O U T P U T"
_BUCK_HDR = "B U C K L I N G   F A C T O R   O U T P U T"


@dataclass
class Mode:
    mode: int
    eigenvalue: float
    omega: float  # rad/time
    frequency: float  # cycles/time


def read_eigen(path: str | Path) -> list[list[Mode]]:
    """Eigenvalue tables in a .dat file, one list per *FREQUENCY step."""
    tables: list[list[Mode]] = []
    lines = Path(path).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if _EIG_HDR in lines[i]:
            modes: list[Mode] = []
            i += 1
            # skip header lines until first numeric row
            while i < len(lines) and not re.match(r"^\s+\d+\s+[-+0-9.]", lines[i]):
                i += 1
            while i < len(lines) and re.match(r"^\s+\d+\s+[-+0-9.]", lines[i]):
                t = lines[i].split()
                modes.append(Mode(int(t[0]), float(t[1]), float(t[2]), float(t[3])))
                i += 1
            tables.append(modes)
        else:
            i += 1
    return tables


def read_buckling(path: str | Path) -> list[list[tuple[int, float]]]:
    """Buckling-factor tables in a .dat file, one list per *BUCKLE step."""
    tables: list[list[tuple[int, float]]] = []
    lines = Path(path).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if _BUCK_HDR in lines[i]:
            rows: list[tuple[int, float]] = []
            i += 1
            while i < len(lines) and not re.match(r"^\s+\d+\s+[-+0-9.]", lines[i]):
                i += 1
            while i < len(lines) and re.match(r"^\s+\d+\s+[-+0-9.]", lines[i]):
                t = lines[i].split()
                rows.append((int(t[0]), float(t[1])))
                i += 1
            tables.append(rows)
        else:
            i += 1
    return tables


def _num(tok: str | None) -> float:
    """Perl-like numeric conversion: non-numeric strings count as 0."""
    if tok is None:
        return 0.0
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        m = re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", tok)
        return float(m.group(0)) if m else 0.0


@dataclass
class DatComparison:
    status: str  # pass | fail | missing | missing_ref | line_count | nan
    max_rel_block_error: float = 0.0  # max |ref-val| / block max  (datcheck "error2")
    line: int | None = None
    value: float | None = None
    reference: float | None = None
    block_max: float | None = None
    n_lines: int = 0
    n_lines_ref: int = 0
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def compare_dat(dat: str | Path, ref: str | Path, tol: float = 1e-3) -> DatComparison:
    dat, ref = Path(dat), Path(ref)
    if not ref.exists():
        return DatComparison("missing_ref")
    if not dat.exists():
        return DatComparison("missing")
    ref_lines = ref.read_text(errors="replace").split("\n")
    dat_lines = dat.read_text(errors="replace").split("\n")
    # wc -l semantics: number of newline characters
    n_ref, n_dat = len(ref_lines) - 1, len(dat_lines) - 1
    if n_ref != n_dat:
        return DatComparison("line_count", n_lines=n_dat, n_lines_ref=n_ref)
    if any("NaN" in ln for ln in dat_lines):
        return DatComparison("nan", n_lines=n_dat, n_lines_ref=n_ref)

    # block maxima over tokens containing '.'
    xmax = [0.0]
    for ln in ref_lines[:n_ref]:
        if ln == "":
            xmax.append(0.0)
            continue
        for tok in ln.split():
            if "." in tok:
                v = abs(_num(tok))
                if v > xmax[-1]:
                    xmax[-1] = v

    k = 0
    participation = False
    worst = DatComparison("pass", n_lines=n_dat, n_lines_ref=n_ref)
    for j, (lr, ld) in enumerate(zip(ref_lines[:n_ref], dat_lines[:n_dat]), start=1):
        if lr == "":
            k += 1
        if "P A R T I C I P A T I O N   F A C T O R S   F O R   F R E Q U E N C Y" in lr:
            pass
        elif "P A R T I C I P A T I O N   F A C T O R S" in lr:
            participation = True
        elif "T O T A L   E F F E C T I V E   M A S S" in lr:
            participation = False
        if "time" in ld or participation:
            continue
        fr, fd = lr.split(), ld.split()
        for i, tok in enumerate(fr):
            r = _num(tok)
            if abs(r) < 1e-10:
                continue
            v = _num(fd[i] if i < len(fd) else None)
            if math.isnan(v):
                return DatComparison("nan", n_lines=n_dat, n_lines_ref=n_ref)
            err2 = abs(r - v) / xmax[k] if xmax[k] > 1e-10 else 0.0
            if err2 > worst.max_rel_block_error and (v > 1e-8 or xmax[k] > 1e-8):
                worst.max_rel_block_error = err2
                worst.line, worst.value, worst.reference, worst.block_max = j, v, r, xmax[k]
    if worst.max_rel_block_error > tol:
        worst.status = "fail"
    return worst
