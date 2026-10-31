"""Deterministic boundary-condition sets for a screened part.

``generate(part, n)`` returns up to ``n`` distinct BC sets, each seeded by (part id, attempt), so
re-running gives the same benchmarks. Supports and loads go on faces with enough area to carry
them; freeform (B-spline) faces take clamps and pressure only (no single normal for a directed force). Every set fully clamps at least one face (statically sufficient in 3D);
loads go on faces away from the supports so there is a load path through the part. The precheck
(rigid-body modes, no load) still runs on every set before any fine solve.

Patterns
  supports: one clamped face | two clamped faces | clamped face + roller (normal) on a planar face
  loads:    pressure | normal force | shear force | bearing force on a cylinder (perpendicular to
            its axis) | two loads
Magnitudes are fixed (1 MPa, 1 kN): linear analysis, so gold values scale exactly with the load.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

import numpy as np

from feacore.case import BC
from feacore.goldgen.parts import Face, Part

PRESSURE = 1e6       # Pa
FORCE = 1e3          # N
MIN_AREA_FRAC = 0.02  # of the part's total surface area, for a support / load face
MIN_SEPARATION = 0.25  # load-face centroid at least this x bbox diagonal from every support face


@dataclass
class BCSet:
    id: str                              # <part id>_bc<k>
    attempt: int
    bcs: list[BC]
    words: list[str] = field(default_factory=list)   # agent-facing description, one line per BC

    def to_dict(self) -> dict:
        return {"id": self.id, "attempt": self.attempt, "bcs": [asdict(b) for b in self.bcs],
                "words": self.words}


def group(face: Face) -> str:
    return f"f{face.tag}"


def _describe(f: Face) -> str:
    c = ", ".join(f"{v * 1000:.2f}" for v in f.centroid)
    if f.kind == "plane":
        n = ", ".join(f"{v:.3f}" for v in f.normal)
        return f"planar face {group(f)} (outward normal ({n}), centroid ({c}) mm, area {f.area * 1e6:.1f} mm^2)"
    if f.kind == "cylinder":
        a = ", ".join(f"{v:.3f}" for v in f.axis)
        return (f"cylindrical face {group(f)} (radius {f.radius * 1000:.2f} mm, axis ({a}), "
                f"centroid ({c}) mm, area {f.area * 1e6:.1f} mm^2)")
    return f"face {group(f)} (centroid ({c}) mm, area {f.area * 1e6:.1f} mm^2)"


def _perpendicular(v: np.ndarray, rng: random.Random) -> np.ndarray:
    a = np.eye(3)[int(np.argmin(np.abs(v)))]
    p = np.cross(v, a)
    p /= np.linalg.norm(p)
    q = np.cross(v, p)
    t = rng.uniform(0, 2 * np.pi)
    return np.cos(t) * p + np.sin(t) * q


def _round(v) -> list[float]:
    return [float(round(x, 9)) + 0.0 for x in v]


def _seed(part_id: str, attempt: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{part_id}:{attempt}".encode()).digest()[:8], "big")


def _one(part: Part, attempt: int) -> BCSet | None:
    rng = random.Random(_seed(part.id, attempt))
    total = sum(f.area for f in part.faces)
    usable = [f for f in part.faces if f.area >= MIN_AREA_FRAC * total]
    if len(usable) < 2:
        return None
    pattern = rng.choice(["one", "one", "two", "roller"])
    supports = rng.sample(usable, 2 if pattern == "two" else 1)
    bcs, words = [], []
    for f in supports:
        bcs.append(BC("fixed", group=group(f)))
        words.append(f"Clamped (all displacements zero): {_describe(f)}.")

    def far(f):
        return all(np.linalg.norm(np.subtract(f.centroid, s.centroid)) >= MIN_SEPARATION * part.diag
                   for s in supports)

    if pattern == "roller":
        # roller only on an axis-aligned plane: the solver's displacement BC acts on x / y / z
        cands = [f for f in usable if f.kind == "plane" and f not in supports and far(f)
                 and max(np.abs(f.normal)) > 0.999]
        if cands:
            f = rng.choice(cands)
            k = int(np.argmax(np.abs(f.normal)))
            bcs.append(BC("displacement", group=group(f), components=[k], value=0.0))
            words.append(f"Roller (zero {'xyz'[k]}-displacement only): {_describe(f)}.")
            supports.append(f)
    loadable = [f for f in usable if f not in supports and far(f)]
    if not loadable:
        return None
    n_loads = 2 if (rng.random() < 0.25 and len(loadable) >= 2) else 1
    for f in rng.sample(loadable, n_loads):
        kinds = {"plane": ["pressure", "normal", "shear"], "cylinder": ["bearing", "pressure"]}.get(f.kind, ["pressure"])
        kind = rng.choice(kinds)
        if kind == "pressure":
            bcs.append(BC("pressure", group=group(f), value=PRESSURE))
            words.append(f"Uniform pressure {PRESSURE / 1e6:g} MPa pushing into the part on {_describe(f)}.")
        else:
            if kind == "normal":
                d = -np.asarray(f.normal) * rng.choice([1, -1])
            elif kind == "shear":
                d = _perpendicular(np.asarray(f.normal), rng)
            else:
                d = _perpendicular(np.asarray(f.axis), rng)
            vec = _round(FORCE * d)
            bcs.append(BC("force", group=group(f), value=vec))
            words.append(f"Total force ({', '.join(f'{v:.1f}' for v in vec)}) N spread uniformly over "
                         f"{_describe(f)}.")
    return BCSet(id=f"{part.id}_bc{attempt}", attempt=attempt, bcs=bcs, words=words)


def _key(s: BCSet) -> tuple:
    return tuple(sorted((b.type, b.group, tuple(b.components or ()),
                         str(b.value)) for b in s.bcs))


def generate(part: Part, n: int = 10, max_attempts: int = 200) -> list[BCSet]:
    """Up to ``n`` distinct BC sets (fewer if the part has too few usable faces)."""
    out, seen = [], set()
    for attempt in range(max_attempts):
        s = _one(part, attempt)
        if s is None or _key(s) in seen:
            continue
        seen.add(_key(s))
        s.id = f"{part.id}_bc{len(out)}"
        out.append(s)
        if len(out) == n:
            break
    return out
