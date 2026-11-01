"""Reader for CalculiX ASCII ``.frd`` result files.

The format is fixed width.  In the default "long" format (flag 1) node records are
`` -1`` + I10 node id + E12.5 values with no guaranteed separator between values
(``-4.08740E-05`` fills its 12 columns), so values must be sliced by column, not split.

Element connectivity is returned in CalculiX *input* (Abaqus) node order.  The frd file
writes 20-node hexahedra and 15-node wedges with the vertical mid-edge nodes before the
top-face mid-edge nodes; we permute those back.  Tet10 order is identical in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# frd element type code -> (name, number of nodes)
FRD_ELEMENT_TYPES = {
    1: ("hex8", 8),
    2: ("wedge6", 6),
    3: ("tet4", 4),
    4: ("hex20", 20),
    5: ("wedge15", 15),
    6: ("tet10", 10),
    7: ("tri3", 3),
    8: ("tri6", 6),
    9: ("quad4", 4),
    10: ("quad8", 8),
    11: ("line2", 2),
    12: ("line3", 3),
}

# frd order -> input order
_PERMUTATIONS = {
    "hex20": list(range(12)) + [16, 17, 18, 19, 12, 13, 14, 15],
    "wedge15": list(range(9)) + [12, 13, 14, 9, 10, 11],
}

SOLID_TYPES = ("tet4", "tet10", "hex8", "hex20", "wedge6", "wedge15")


@dataclass
class ResultBlock:
    """One result data set, e.g. DISP at one output increment."""

    name: str  # DISP, STRESS, FORC, TOSTRAIN, ENER, ...
    index: int  # running index of the output increment (0-based, in file order)
    value: float  # step time (static), frequency (modal), buckling factor
    components: list[str]
    node_ids: np.ndarray  # (n,)
    values: np.ndarray  # (n, ncomp)
    header: str = ""

    def as_map(self) -> dict[int, np.ndarray]:
        return {int(n): v for n, v in zip(self.node_ids, self.values)}


@dataclass
class FrdResults:
    node_ids: np.ndarray  # (N,)
    coords: np.ndarray  # (N, 3)
    elements: dict[str, np.ndarray] = field(default_factory=dict)  # type -> (M, k) node ids
    element_ids: dict[str, np.ndarray] = field(default_factory=dict)
    blocks: list[ResultBlock] = field(default_factory=list)

    def node_row(self) -> dict[int, int]:
        return {int(n): i for i, n in enumerate(self.node_ids)}

    def names(self) -> list[str]:
        return sorted({b.name for b in self.blocks})

    def increments(self) -> list[int]:
        return sorted({b.index for b in self.blocks})

    def get(self, name: str, index: int | None = None) -> ResultBlock:
        """Return result block ``name`` at output increment ``index`` (default: last)."""
        cands = [b for b in self.blocks if b.name == name]
        if not cands:
            raise KeyError(f"no {name!r} results in frd (have {self.names()})")
        if index is None:
            return cands[-1]
        for b in cands:
            if b.index == index:
                return b
        raise KeyError(f"no {name!r} results at increment {index}")

    def field_on_nodes(self, name: str, index: int | None = None) -> np.ndarray:
        """Field values ordered like ``self.node_ids``; nodes without data get NaN."""
        blk = self.get(name, index)
        out = np.full((len(self.node_ids), blk.values.shape[1]), np.nan)
        row = self.node_row()
        rows = np.array([row.get(int(n), -1) for n in blk.node_ids])
        ok = rows >= 0
        out[rows[ok]] = blk.values[ok]
        return out

    def solid_elements(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self.elements.items() if k in SOLID_TYPES and len(v)}


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        t = s.strip()
        # Fortran drops the 'E' for 3-digit exponents: 1.00000+100
        for i in range(len(t) - 1, 0, -1):
            if t[i] in "+-" and t[i - 1] not in "eE":
                return float(t[:i] + "E" + t[i:])
        raise


def _values(line: str, start: int, width: int = 12) -> list[float]:
    s = line.rstrip("\r\n")
    n = (len(s) - start) // width
    return [_float(s[start + i * width : start + (i + 1) * width]) for i in range(n)]


def read_frd(path: str | Path) -> FrdResults:
    path = Path(path)
    node_ids: list[int] = []
    coords: list[list[float]] = []
    elem_rows: list[tuple[int, int, list[int]]] = []
    blocks: list[ResultBlock] = []

    with open(path, "r", errors="replace") as fh:
        lines = iter(fh)
        for line in lines:
            if line.startswith("    2C"):  # nodes
                fmt = int(line.split()[-1]) if line.split()[-1].isdigit() else 1
                id_end = 13 if fmt == 1 else 8
                for ln in lines:
                    if ln.startswith(" -3"):
                        break
                    if ln.startswith(" -1"):
                        node_ids.append(int(ln[3:id_end]))
                        coords.append(_values(ln, id_end)[:3])
            elif line.startswith("    3C"):  # elements
                current: list | None = None
                for ln in lines:
                    if ln.startswith(" -3"):
                        break
                    if ln.startswith(" -1"):
                        t = ln.split()
                        current = [int(t[1]), int(t[2]), []]
                        elem_rows.append(current)  # type: ignore[arg-type]
                    elif ln.startswith(" -2") and current is not None:
                        current[2].extend(int(t) for t in ln.split()[1:])
            elif line.startswith("  100C"):
                header = line.rstrip("\n")
                toks = line[12:].split()
                try:
                    value = _float(toks[0])
                except (ValueError, IndexError):
                    value = float("nan")
                fmt = int(toks[-1]) if toks and toks[-1].isdigit() else 1
                id_end = 13 if fmt == 1 else 8
                # -4 record: name and number of components
                ln = next(lines)
                name = ln[5:13].strip() if ln.startswith(" -4") else "UNKNOWN"
                comps: list[str] = []
                ids: list[int] = []
                vals: list[list[float]] = []
                for ln in lines:
                    if ln.startswith(" -5"):
                        cname = ln[5:13].strip()
                        if cname != "ALL":
                            comps.append(cname)
                    elif ln.startswith(" -1"):
                        ids.append(int(ln[3:id_end]))
                        vals.append(_values(ln, id_end))
                    elif ln.startswith(" -2") and vals:
                        vals[-1].extend(_values(ln, id_end))
                    elif ln.startswith(" -3"):
                        break
                ncomp = len(comps) if comps else (len(vals[0]) if vals else 0)
                arr = np.array([v[:ncomp] + [np.nan] * (ncomp - len(v)) for v in vals], dtype=float)
                if arr.size == 0:
                    arr = np.zeros((0, ncomp))
                # increments: a new increment starts when the step value changes or a name repeats
                if blocks and (blocks[-1].value == value) and name not in {
                    b.name for b in blocks if b.index == blocks[-1].index
                }:
                    index = blocks[-1].index
                else:
                    index = blocks[-1].index + 1 if blocks else 0
                blocks.append(ResultBlock(name, index, value, comps, np.array(ids, dtype=np.int64), arr, header))

    res = FrdResults(
        node_ids=np.array(node_ids, dtype=np.int64),
        coords=np.array(coords, dtype=float).reshape(-1, 3),
        blocks=blocks,
    )
    by_type: dict[str, list[list[int]]] = {}
    ids_by_type: dict[str, list[int]] = {}
    for eid, code, nodes in elem_rows:
        tname, nn = FRD_ELEMENT_TYPES.get(code, (f"type{code}", len(nodes)))
        nodes = nodes[:nn]
        perm = _PERMUTATIONS.get(tname)
        if perm is not None and len(nodes) == len(perm):
            nodes = [nodes[i] for i in perm]
        by_type.setdefault(tname, []).append(nodes)
        ids_by_type.setdefault(tname, []).append(eid)
    res.elements = {k: np.array(v, dtype=np.int64) for k, v in by_type.items()}
    res.element_ids = {k: np.array(v, dtype=np.int64) for k, v in ids_by_type.items()}
    return res


def von_mises(s: np.ndarray) -> np.ndarray:
    """Von Mises from (n, 6) stress in ccx order SXX SYY SZZ SXY SYZ SZX."""
    s = np.atleast_2d(s)
    sxx, syy, szz, sxy, syz, szx = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


def principal(s: np.ndarray) -> np.ndarray:
    """Principal stresses, (n, 3) sorted descending, from (n, 6) ccx-ordered tensor components."""
    s = np.atleast_2d(s)
    t = np.zeros((len(s), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    t[:, 0, 1] = t[:, 1, 0] = s[:, 3]
    t[:, 1, 2] = t[:, 2, 1] = s[:, 4]
    t[:, 2, 0] = t[:, 0, 2] = s[:, 5]
    w = np.linalg.eigvalsh(t)
    return w[:, ::-1]
