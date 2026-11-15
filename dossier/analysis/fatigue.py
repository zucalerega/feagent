"""Cyclic (fatigue) analysis: the fatigue damage of the part from the stress fields of its structural
load cases, by the elastic stress-life method.

    run(project, ctx, geo, lc, workdir, opts) -> pipeline.Analysis        # analysis type "fatigue"

Inputs
------
`lc.cycles` holds one CycleUse per duty cycle: its id, `count` (repetitions in the life of the part,
WITHOUT a life factor: the compliance stage applies the factor of the requirement) and `states` (the
structural load cases of its phases, in sequence). The part is unloaded, at its reference temperature
and with zero stress, before and after each repetition. The stress of a state is the nodal field
`fields_L<k>.npz` (`x`, `stress` = sxx, syy, szz, sxy, syz, sxz in Pa) that the structural analysis of
that load case lists in its record; a scaled load case uses the field of its base case times `scale`.
The engine checks that every load case has the same nodes on a mesh level, and stops if not.

Method (what is computed, and where it comes from)
--------------------------------------------------
1. Equivalent stress range of two stress states m and n at a node: the von Mises stress of the
   DIFFERENCE of the two tensors, d = s(m) - s(n):

       dS = sqrt( 1/2 [(d11-d22)^2 + (d22-d33)^2 + (d33-d11)^2] + 3 (d12^2 + d23^2 + d13^2) )

   The stress amplitude is S_a = dS / 2. This is the measure of ASME BPVC Section VIII Division 2,
   Part 5, 5.5.3 (fatigue assessment, elastic stress analysis and equivalent stresses). It is not the
   difference of the two von Mises stresses, which is zero when the stress turns but keeps its size.

2. Mean stress of the pair: s_m = 1/2 (I1(m) + I1(n)), with I1 = s11 + s22 + s33 (the first stress
   invariant). For one stress component this is the usual mean stress; it has a sign and it does not
   depend on shear (Sines 1959; the "effective mean stress" of Dowling, Mechanical Behavior of Materials,
   chapter on stress-based fatigue). A compressive mean stress gets no credit: s_m < 0 is taken as 0.

3. Mean stress correction to the fully reversed amplitude S_ar (`material.fatigue.mean_stress`):

       goodman   S_ar = S_a / (1 - s_m / S_u)          (no finite value when s_m >= S_u)
       swt       S_ar = sqrt((S_a + s_m) S_a)          (Smith, Watson, Topper 1970)
       none      S_ar = S_a

4. Cycles to failure, Basquin (1910): S_ar = sigma_f (2 N)^b, so N = 1/2 (S_ar / sigma_f)^(1/b).
   N is not taken below 0.5 (one load reversal): at S_ar >= sigma_f the relation has no meaning, and
   the floor keeps the damage finite (2 per cycle). Such a node is far beyond yield and fails check 9.
   With `endurance_limit` S_e: a node whose every counted cycle has S_ar <= S_e / (1 + policy.max_uncertainty)
   has zero damage; at every other node all cycles count on the Basquin line, also those below S_e
   (the "elementary" form of Miner's rule, because large cycles lower the endurance limit).

5. Cycle count: the max-min count of ASME VIII-2 Annex 5-B.5, at every node, over the stress states of
   ALL duty cycles together. Time points: each state of each duty cycle with the repetitions of its duty
   cycle, and the unloaded state with the sum of all repetitions. Repeat: take the two time points with
   the largest dS that both have repetitions left; count min(repetitions) cycles of that range; subtract
   them from both. Stop when fewer than two time points are left.
   Why not "one cycle of the largest range per repetition of each duty cycle": the system model gives
   the number of repetitions of the duty cycles but not their order. When two duty cycles load a point in
   opposite directions and alternate in time, the real cycle spans both (0 -> +s -> 0 -> -s -> 0 is one
   cycle of range 2 s, not two of range s). Counting each duty cycle alone misses this and the secondary
   cycles inside a repetition, and it errs to the unsafe side.
   What the max-min count guarantees: no order of the duty cycles gives larger ranges. For stress in one
   direction its ranges dominate those of a rainflow count (ASTM E1049) of the history in ANY order, and
   the damage rises faster than the range, so without a mean stress correction its damage is an upper
   bound. With the Goodman correction the same held for every order of 250 random histories with stresses
   up to 2/3 of S_u (a test repeats this); it is not a theorem: with the mean stress at 90 % of S_u a
   search found one history 0.6 % above it. The price of the bound: it can pair states that are not load
   reversals of the real sequence (at most one cycle per state and repetition).
   The data file and the statement also give the simple count (one cycle per repetition, each duty cycle
   alone), so the reader sees how much the order matters.

6. Damage: Miner's linear sum D = sum n_k / N_k over the counted cycles, at each node.

       damage.total@part     the largest nodal D, and where it is
       damage.total@smooth   the largest D over the nodes farther than a stated distance from sharp
                             re-entrant edges and held faces: information only (see 7)

7. Singular zones. The elastic stress has no limit at a sharp re-entrant edge (`geo["edges"]["reentrant"]`)
   and at the rim of a held face (the `fixed` and `displacement` items with faces in the `bcs` of the state
   analyses; point restraints without faces are not zones). The nodes on them come from the finest mesh
   file of the structural analysis (`f<tag>` groups; an edge is the nodes that its two faces share, as in
   the structural engine). The zone of an edge or face is everything nearer than EXCLUSION_ELEMENTS (3)
   element sizes of the finest mesh there, about two elements of the next coarser mesh.
   `damage.total@smooth` leaves the zones out. When its node is at the rim of a zone whose nodes have more
   damage (within RIM_FACTOR x the distance), the value is the singular field cut off at that distance:
   the note says so and the status is `not_converged`. When a zone cannot be found in the mesh, no result
   is converged, and when none can be found `damage.total@smooth` has no value.

8. Convergence. The damage is computed on the two finest meshes. For the node with the largest damage
   on the finest mesh, the nearest node of the coarser mesh is taken (the meshes share no nodes), and
   the stress range of the cycle with the largest damage is compared: that relative change is tested
   against `policy.max_uncertainty` (status `converged` / `not_converged`). `uncertainty` of the result is
   the uncertainty of the DAMAGE: the larger relative difference between the two meshes of (a) the
   damage at that location and (b) the largest damage in the zone. Because D ~ dS^(-1/b), a 1 % change in
   stress is about a 10 % change in damage for b = -0.1; the margin must see that, the status must not
   demand a stress that is converged to 0.5 %.
   A node in a singular zone is never `converged` (the rule of the structural engine for its peaks): it is
   `singular` when the range grows by more than GROWTH (0.1 %) from the coarser to the finest mesh, else
   `not_converged`; its `uncertainty` is empty. One mesh only: `not_converged`, no uncertainty.
   Checks follow CONTRACTS.md (passed=False blocks a pass; convergence is carried by the result's status and
   uncertainty only): the convergence check of each zone is information, passed=None (not False), when the
   change is over the limit.
   The yield and shakedown checks of both zones stay False when exceeded: they are real limits.

9. Validity. Elastic stress-life is right only where the material stays elastic. Two checks per zone:
   the largest range dS <= 2 S_y (elastic shakedown), and the largest von Mises stress of a state <= S_y
   (no yield in the first cycle). Where one fails, the damage is not valid in either direction (the real
   strain range is larger than the elastic one, and the Basquin line is not the low-cycle curve), the
   check fails, and the result cannot have the status converged. A strain-life method with a plasticity
   correction (Neuber) is needed there; the material model has no data for it.

What is NOT in the method (each is an Assumption or a Decision of the record)
----------------------------------------------------------------------------
* no fatigue strength reduction: welds, surface finish, size, environment (K_f = 1 everywhere);
* no stress between the states: a thermal transient or a pressure surge between two phases is not seen;
* no residual or assembly stress in the unloaded state;
* fatigue data and ultimate strength at one temperature for every state;
* Miner's rule has no sequence effect; failure is taken at D = 1;
* nodal stresses are used as the structural analysis recovered them.

Outputs
-------
`AN-<load case without LC->` with the two results, the checks (same nodes, counts against the context,
singular zones found, elastic range, convergence), the links to the state analyses, and two data files
in the workdir: `fatigue.json` (time points, zones, and for each result the counted cycles and the duty
cycle table at its node: range, amplitude, mean stress, cycles to failure, damage, share) and
`damage_L<k>.npz` (x, damage, largest range, smooth mask on the finest mesh).

References
----------
ASME Boiler and Pressure Vessel Code, Section VIII, Division 2, Part 5: 5.5.3 and Annex 5-B.5.
O. H. Basquin, The exponential law of endurance tests, Proc. ASTM 10 (1910).
M. A. Miner, Cumulative damage in fatigue, J. Appl. Mech. 12 (1945).
G. Sines, Behavior of metals under complex static and alternating stresses, in Metal Fatigue (1959).
K. N. Smith, P. Watson, T. H. Topper, A stress-strain function for the fatigue of metals, J. Mater. 5 (1970).
N. E. Dowling, Mechanical Behavior of Materials (Goodman relation, effective mean stress).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from dossier import __version__, units
from dossier.model import AnalysisRecord, Artifact, Check, Link, LoadCase, Origin, ResultValue
from dossier.pipeline import Analysis
from dossier.project import make_id
from dossier.records import Journal

EXCLUSION_ELEMENTS = 3.0     # element sizes of the finest mesh left out around a singular edge or a held face
HELD_BC_TYPES = ("fixed", "displacement")   # BC items of a structural analysis that hold a face
MIN_CYCLES = 0.5             # one load reversal: the smallest number of cycles to failure
GROWTH = 1e-3                # a stress range grows under refinement when it rises by more than this (relative)
RIM_FACTOR = 2.0             # the rim of a singular zone: out of the zone, within this x its distance
NODE_TOLERANCE = 1e-9        # two meshes are the same when their nodes differ by less than this x the part size
PAIRS_PER_CHUNK = 1_500_000  # nodes x pairs of time points evaluated at once (memory)
UNLOADED = "the unloaded state"
BY = "engine:fatigue"

REF_ASME_RANGE = "doc:ASME BPVC Section VIII Division 2, Part 5, paragraph 5.5.3"
REF_ASME_COUNT = "doc:ASME BPVC Section VIII Division 2, Annex 5-B, paragraph 5-B.5"
REF_BASQUIN = "doc:O. H. Basquin, The exponential law of endurance tests, Proc. ASTM 10, 1910"
REF_MINER = "doc:M. A. Miner, Cumulative damage in fatigue, J. Appl. Mech. 12, 1945"
REF_SINES = "doc:G. Sines, Behavior of metals under complex static and alternating stresses, Metal Fatigue, 1959"
REF_SWT = "doc:K. N. Smith, P. Watson, T. H. Topper, A stress-strain function for the fatigue of metals, 1970"

_FIELD_RE = re.compile(r"fields_L(\d+)\.npz$")
_FACE_RE = re.compile(r"^f(\d+)$")


class _Stop(Exception):
    """The analysis cannot give a result. `status` is the status of the record, the message says why (STE)."""

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


# =========================================================================== the stress-life relations


@dataclass
class SN:
    """The stress-life data of the material, SI."""

    sigma_f: float                   # Basquin coefficient, Pa
    b: float                         # Basquin exponent (negative)
    rule: str = "goodman"            # goodman | swt | none
    su: float = math.inf             # ultimate strength, Pa (Goodman)
    sy: float = math.inf             # yield strength, Pa (validity checks)
    endurance: float | None = None   # endurance limit, Pa
    margin: float = 0.0              # the endurance limit applies below endurance / (1 + margin)

    @property
    def power(self) -> float:
        """Damage is proportional to the stress range to this power."""
        return -1.0 / self.b

    @property
    def cut(self) -> float | None:
        return None if self.endurance is None else self.endurance / (1.0 + self.margin)


def von_mises(s) -> np.ndarray:
    """Von Mises stress of tensors (..., 6) = sxx, syy, szz, sxy, syz, sxz."""
    s = np.asarray(s, dtype=float)
    xx, yy, zz, xy, yz, xz = (s[..., k] for k in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy ** 2 + yz ** 2 + xz ** 2))


def first_invariant(s) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return s[..., 0] + s[..., 1] + s[..., 2]


def corrected_amplitude(sa, sm, sn: SN) -> np.ndarray:
    """The fully reversed stress amplitude with the same life as amplitude `sa` at mean stress `sm`.
    A compressive mean stress counts as zero. Goodman gives inf when the mean stress reaches S_u."""
    sa = np.asarray(sa, dtype=float)
    sm = np.maximum(np.asarray(sm, dtype=float), 0.0)
    if sn.rule == "none":
        return sa + 0.0 * sm
    if sn.rule == "swt":
        return np.sqrt((sa + sm) * sa)
    if sn.rule == "goodman":
        rest = 1.0 - sm / sn.su
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(rest > 0.0, sa / np.where(rest > 0.0, rest, 1.0), np.inf)
        return np.where(sa > 0.0, out, 0.0)
    raise ValueError(f"unknown mean stress rule {sn.rule!r}")


def cycles_to_failure(sar, sn: SN) -> np.ndarray:
    """Basquin: sar = sigma_f (2 N)^b. No amplitude: inf. Not below MIN_CYCLES."""
    sar = np.asarray(sar, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        n = 0.5 * np.power(np.where(sar > 0.0, sar, 1.0) / sn.sigma_f, 1.0 / sn.b)
    n = np.where(sar > 0.0, n, np.inf)
    return np.maximum(n, MIN_CYCLES)


# =========================================================================== the cycle count


@dataclass
class Point:
    """One time point of the cycle count: the unloaded state, or one state of one duty cycle."""

    label: str                    # the load case id, or UNLOADED
    duty_cycle: str | None        # None: the unloaded state
    base: str | None              # the load case whose stress field it uses (None: zero stress)
    scale: float                  # stress = scale x field of `base`
    reps: float                   # repetitions in the life of the part


@dataclass
class Nodal:
    """The result of the count at every node of one mesh."""

    damage: np.ndarray            # Miner sum of the max-min count
    simple: np.ndarray            # Miner sum of one cycle per repetition, each duty cycle alone
    max_range: np.ndarray         # largest equivalent stress range between two time points, Pa
    max_vm: np.ndarray            # largest von Mises stress of a time point, Pa
    max_sar: np.ndarray           # largest corrected amplitude of a counted cycle, Pa


def _pairs(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(n_points, 1)
    return i.astype(np.int64), j.astype(np.int64)


def _tensors(fields: dict[str, np.ndarray], points: list[Point], a: int, b: int) -> np.ndarray:
    """The stress tensor of every time point at the nodes a..b: (points, nodes, 6)."""
    out = np.zeros((len(points), b - a, 6))
    for p, pt in enumerate(points):
        if pt.base is not None:
            out[p] = pt.scale * fields[pt.base][a:b]
    return out


def _count(R: np.ndarray, I1: np.ndarray, pi: np.ndarray, pj: np.ndarray, reps, sn: SN, detail: bool = False):
    """The max-min count. R (nodes, pairs) is the range of each pair of time points, I1 (points, nodes)
    the first invariant, reps (points,) the repetitions. Returns the Miner sum (before the endurance
    cut), the largest corrected amplitude of a counted cycle, and with `detail` the cycles of each pass
    as (i, j, n, range, mean, corrected amplitude, cycles to failure), arrays over the nodes."""
    m, n_points = R.shape[0], len(reps)
    rows = np.arange(m)
    left = np.tile(np.asarray(reps, dtype=float), (m, 1))
    eps = 1e-9 * max(float(np.max(reps, initial=0.0)), 1e-300)
    damage, max_sar, steps = np.zeros(m), np.zeros(m), []
    for _ in range(max(n_points - 1, 0)):       # every pass uses up one time point at each node
        alive = left > eps
        ok = alive[:, pi] & alive[:, pj]
        if not ok.any():
            break
        Rm = np.where(ok, R, -1.0)
        k = np.argmax(Rm, axis=1)
        has = Rm[rows, k] >= 0.0
        i, j = pi[k], pj[k]
        n = np.where(has, np.minimum(left[rows, i], left[rows, j]), 0.0)
        rng = np.where(has, R[rows, k], 0.0)
        mean = 0.5 * (I1[i, rows] + I1[j, rows])
        sar = corrected_amplitude(0.5 * rng, mean, sn)
        nf = cycles_to_failure(sar, sn)
        damage += n / nf
        max_sar = np.maximum(max_sar, np.where(n > 0.0, sar, 0.0))
        left[rows, i] -= n
        left[rows, j] -= n
        if detail:
            steps.append((i, j, n, rng, mean, sar, nf))
    return damage, max_sar, steps


def _one_cycle(R, I1, pi, pj, col: dict, duty: list[tuple[str, float, list[int]]], sn: SN, detail: bool = False):
    """One cycle of the largest range for each repetition of each duty cycle, each duty cycle alone
    (its states and the unloaded state). Returns the Miner sum, the largest corrected amplitude and
    with `detail` one row (i, j, range, mean, corrected amplitude, cycles to failure) per duty cycle."""
    m = R.shape[0]
    rows = np.arange(m)
    damage, max_sar, table = np.zeros(m), np.zeros(m), []
    for _, count, pts in duty:
        cols = np.array([col[tuple(sorted(pq))] for pq in combinations([0, *pts], 2)], dtype=np.int64)
        if not len(cols) or count <= 0.0:
            table.append(None)
            continue
        c = cols[np.argmax(R[:, cols], axis=1)]
        i, j = pi[c], pj[c]
        rng = R[rows, c]
        mean = 0.5 * (I1[i, rows] + I1[j, rows])
        sar = corrected_amplitude(0.5 * rng, mean, sn)
        nf = cycles_to_failure(sar, sn)
        damage += count / nf
        max_sar = np.maximum(max_sar, sar)
        if detail:
            table.append((i, j, rng, mean, sar, nf))
    return damage, max_sar, table


def _duty(points: list[Point]) -> list[tuple[str, float, list[int]]]:
    """(duty cycle, repetitions, the indices of its time points), in the order of the load case."""
    out: dict[str, tuple[str, float, list[int]]] = {}
    for p, pt in enumerate(points):
        if pt.duty_cycle is not None:
            out.setdefault(pt.duty_cycle, (pt.duty_cycle, pt.reps, []))[2].append(p)
    return list(out.values())


def _cut(damage: np.ndarray, max_sar: np.ndarray, sn: SN) -> np.ndarray:
    """The endurance limit: no damage at a node whose every cycle is below it."""
    return damage if sn.cut is None else np.where(max_sar > sn.cut, damage, 0.0)


def evaluate(fields: dict[str, np.ndarray], points: list[Point], sn: SN) -> Nodal:
    """The count at every node. `fields[base]` is the (n, 6) stress field of a base load case."""
    n = len(next(iter(fields.values()))) if fields else 0
    pi, pj = _pairs(len(points))
    col = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(pi, pj))}
    duty, reps = _duty(points), [pt.reps for pt in points]
    out = Nodal(*(np.zeros(n) for _ in range(5)))
    step = max(1, PAIRS_PER_CHUNK // max(len(pi), 1))
    for a in range(0, n, step):
        b = min(a + step, n)
        T = _tensors(fields, points, a, b)
        I1 = first_invariant(T)
        R = von_mises(T[pi] - T[pj]).T if len(pi) else np.zeros((b - a, 0))
        d, sar, _ = _count(R, I1, pi, pj, reps, sn)
        out.damage[a:b], out.max_sar[a:b] = _cut(d, sar, sn), sar
        ds, sars, _ = _one_cycle(R, I1, pi, pj, col, duty, sn)
        out.simple[a:b] = _cut(ds, sars, sn)
        out.max_range[a:b] = R.max(axis=1) if R.shape[1] else 0.0
        out.max_vm[a:b] = von_mises(T).max(axis=0)
    return out


def count_cycles(tensors, reps, sn: SN) -> list[dict]:
    """The max-min count at ONE node, for people and tests: `tensors` (points, 6), `reps` (points,).
    Returns the counted cycles in the order of the count: a, b (indices of the two time points), count,
    range, amplitude, mean_stress, corrected_amplitude, cycles_to_failure, damage. The endurance limit is
    applied: when no cycle is over it, every damage is zero."""
    T = np.asarray(tensors, dtype=float).reshape(len(reps), 1, 6)
    pi, pj = _pairs(len(reps))
    R = von_mises(T[pi] - T[pj]).T if len(pi) else np.zeros((1, 0))
    _, sar_max, steps = _count(R, first_invariant(T), pi, pj, reps, sn, detail=True)
    live = sn.cut is None or float(sar_max[0]) > sn.cut
    out = []
    for i, j, n, rng, mean, sar, nf in steps:
        if float(n[0]) <= 0.0:
            continue
        out.append({"a": int(i[0]), "b": int(j[0]), "count": float(n[0]), "range": float(rng[0]),
                    "amplitude": 0.5 * float(rng[0]), "mean_stress": float(mean[0]),
                    "corrected_amplitude": float(sar[0]), "cycles_to_failure": float(nf[0]),
                    "damage": float(n[0] / nf[0]) if live else 0.0})
    return out


def duty_table(tensors, points: list[Point], sn: SN) -> list[dict]:
    """At ONE node, for each duty cycle alone: the largest range between its states and the unloaded
    state, and the damage of one such cycle per repetition."""
    T = np.asarray(tensors, dtype=float).reshape(len(points), 1, 6)
    pi, pj = _pairs(len(points))
    col = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(pi, pj))}
    R = von_mises(T[pi] - T[pj]).T if len(pi) else np.zeros((1, 0))
    duty = _duty(points)
    _, sar_max, table = _one_cycle(R, first_invariant(T), pi, pj, col, duty, sn, detail=True)
    live = sn.cut is None or float(sar_max[0]) > sn.cut
    out = []
    for (dc, count, _), row in zip(duty, table):
        if row is None:
            out.append({"duty_cycle": dc, "count": count, "a": None, "b": None, "range": 0.0, "amplitude": 0.0,
                        "mean_stress": 0.0, "corrected_amplitude": 0.0, "cycles_to_failure": math.inf,
                        "damage_alone": 0.0})
            continue
        i, j, rng, mean, sar, nf = row
        out.append({"duty_cycle": dc, "count": count, "a": int(i[0]), "b": int(j[0]), "range": float(rng[0]),
                    "amplitude": 0.5 * float(rng[0]), "mean_stress": float(mean[0]),
                    "corrected_amplitude": float(sar[0]), "cycles_to_failure": float(nf[0]),
                    "damage_alone": float(count / nf[0]) if live else 0.0})
    return out


# =========================================================================== inputs


def _record_id(lc: LoadCase) -> str:
    return "AN-" + (lc.id[3:] if lc.id.startswith("LC-") else lc.id)


def _stress_life(project, ctx, lc: LoadCase) -> tuple[SN, Any]:
    mat = project.material_of(lc.component if lc.component in project.components else None)
    fat = mat.fatigue
    if fat is None:
        raise _Stop("not_supported", f"The material {mat.id} has no fatigue data. The fatigue analysis needs the "
                                     f"stress-life data of the material (materials.yaml, block fatigue).")
    names = ctx.names() if ctx is not None else {}

    def pa(value, what: str) -> float:
        try:
            return float(units.si(units.to(value.quantity(names), "Pa")))
        except (units.UnitsError, ValueError) as e:
            raise _Stop("not_supported", f"The material {mat.id} gives no usable {what}: {e}") from e

    sn = SN(sigma_f=pa(fat.sigma_f, "fatigue strength coefficient sigma_f"), b=float(fat.b), rule=fat.mean_stress,
            su=pa(mat.ultimate_strength, "ultimate strength"), sy=pa(mat.yield_strength, "yield strength"),
            endurance=pa(fat.endurance_limit, "endurance limit") if fat.endurance_limit is not None else None,
            margin=float(project.policy.max_uncertainty))
    if fat.model != "basquin":
        raise _Stop("not_supported", f"The material {mat.id} has the fatigue model {fat.model}. "
                                     f"The fatigue analysis knows the Basquin model only.")
    bad = [w for w, ok in (("a fatigue strength coefficient sigma_f above zero", sn.sigma_f > 0.0),
                           ("a Basquin exponent b below zero", sn.b < 0.0),
                           ("an ultimate strength above zero", sn.su > 0.0),
                           ("a yield strength above zero", sn.sy > 0.0),
                           ("an endurance limit above zero", sn.endurance is None or sn.endurance > 0.0)) if not ok]
    if bad:
        raise _Stop("not_supported", f"The fatigue data of the material {mat.id} is not usable. "
                                     f"The analysis needs {' and '.join(bad)}.")
    return sn, mat


def _points(project, lc: LoadCase) -> tuple[list[Point], dict[str, LoadCase], dict[str, AnalysisRecord]]:
    """The time points of the count, the load cases of the plan and the analysis records by load case."""
    if not lc.cycles:
        raise _Stop("not_supported", f"The load case {lc.id} has no duty cycles. "
                                     f"The fatigue analysis needs duty cycles with their states.")
    cases = {x.id: x for x in project.collection("load_cases")}
    analyses = {a.load_case: a for a in project.collection("analyses")}
    points: list[Point] = []
    for cu in lc.cycles:
        if not math.isfinite(cu.count) or cu.count < 0.0:
            raise _Stop("not_supported", f"The duty cycle {cu.duty_cycle} has {cu.count} repetitions. "
                                         f"The number of repetitions must be zero or more.")
        for state in cu.states:
            base, scale, chain = state, 1.0, [state]
            while True:
                case = cases.get(base)
                if case is None:
                    raise _Stop("not_supported", f"The plan has no load case {base}. The duty cycle "
                                                 f"{cu.duty_cycle} needs it as a stress state.")
                if not case.scaled_from:
                    break
                scale *= float(case.scale)
                base = case.scaled_from
                if base in chain:
                    raise _Stop("failed", f"The scaled load cases {', '.join(chain)} refer to each other in a circle.")
                chain.append(base)
            rec = analyses.get(base)
            if rec is None:
                raise _Stop("not_supported", f"The load case {base} has no analysis record. "
                                             f"Do the structural analysis of {base} before the fatigue analysis.")
            if rec.status != "done":
                raise _Stop("not_supported", f"The analysis {rec.id} of load case {base} has the status "
                                             f"{rec.status.replace('_', ' ')}. The fatigue analysis needs its stress field.")
            points.append(Point(label=state, duty_cycle=cu.duty_cycle, base=base, scale=scale, reps=float(cu.count)))
    total = sum(float(cu.count) for cu in lc.cycles)
    if not points or total <= 0.0:
        raise _Stop("not_supported", f"The duty cycles of load case {lc.id} have no stress state with a repetition.")
    return [Point(label=UNLOADED, duty_cycle=None, base=None, scale=0.0, reps=total), *points], cases, analyses


@dataclass
class Level:
    """The stress fields of every base load case on one mesh."""

    level: int
    x: np.ndarray                                  # (n, 3)
    stress: dict[str, np.ndarray]                  # base load case -> (n, 6)
    files: list[dict] = field(default_factory=list)   # the files read: load case, level, path, sha256
    mismatch: float = 0.0                          # largest difference of a node coordinate between load cases, m

    @property
    def size(self) -> float:
        return float(np.linalg.norm(np.ptp(self.x, axis=0))) if len(self.x) else 0.0


def _field_files(project, rec: AnalysisRecord) -> dict[int, Path]:
    out = {}
    for a in rec.artifacts:
        hit = _FIELD_RE.search(a.path.replace("\\", "/")) if a.kind == "field" else None
        if hit:
            out[int(hit.group(1))] = project.out / a.path
    return out


def _read_field(path: Path, case: str, level: int) -> tuple[np.ndarray, np.ndarray]:
    where = f"The field file {path.name} of load case {case}"
    if not path.exists():
        raise _Stop("failed", f"{where} is not in the folder of the analysis. The analysis record lists it.")
    try:
        with np.load(path) as z:
            x, s = np.asarray(z["x"], dtype=float), np.asarray(z["stress"], dtype=float)
    except KeyError as e:
        raise _Stop("failed", f"{where} has no array {e.args[0]}. The contract asks for x and stress.") from e
    except Exception as e:  # noqa: BLE001  numpy raises several kinds for a damaged file
        raise _Stop("failed", f"{where} is not readable: {type(e).__name__}: {e}") from e
    if x.ndim != 2 or x.shape[1] != 3 or s.shape != (len(x), 6):
        raise _Stop("failed", f"{where} has x with the shape {x.shape} and stress with the shape {s.shape}. "
                              f"The contract asks for (n, 3) and (n, 6).")
    if not len(x):
        raise _Stop("failed", f"{where} has no nodes.")
    if not (np.isfinite(x).all() and np.isfinite(s).all()):
        raise _Stop("failed", f"{where} has values that are not finite numbers.")
    return x, s


def _levels(project, bases: list[str], analyses: dict[str, AnalysisRecord]) -> list[Level]:
    """The two finest mesh levels that every base load case has, coarse to fine, with the node sets checked."""
    files = {b: _field_files(project, analyses[b]) for b in bases}
    none = [b for b in bases if not files[b]]
    if none:
        raise _Stop("not_supported", f"The analysis of load case {none[0]} lists no stress field file "
                                     f"(fields_L<k>.npz, kind field). The fatigue analysis needs the stress field.")
    common = sorted(set.intersection(*(set(f) for f in files.values())))
    if not common:
        have = "; ".join(f"{b}: {', '.join('L' + str(k) for k in sorted(f))}" for b, f in files.items())
        raise _Stop("failed", f"The load cases have no mesh level in common ({have}). "
                              f"The fatigue analysis needs the stress of all load cases on the same mesh.")
    out = []
    for k in common[-2:]:
        lev = None
        for b in bases:
            x, s = _read_field(files[b][k], b, k)
            if lev is None:
                lev, first = Level(level=k, x=x, stress={}), b
            elif x.shape != lev.x.shape:
                raise _Stop("failed", f"The stress fields of the load cases {first} and {b} have different nodes on "
                                      f"mesh level {k}: {len(lev.x)} nodes and {len(x)} nodes. "
                                      f"The fatigue analysis needs the same mesh for all load cases.")
            else:
                diff = float(np.abs(x - lev.x).max())
                if diff > NODE_TOLERANCE * max(lev.size, 1e-300):
                    raise _Stop("failed", f"The stress fields of the load cases {first} and {b} have different nodes on "
                                          f"mesh level {k}: the node coordinates differ by up to {_len(diff)}. "
                                          f"The fatigue analysis needs the same mesh for all load cases.")
                lev.mismatch = max(lev.mismatch, diff)
            lev.stress[b] = s
            lev.files.append({"load_case": b, "level": k, "file": _rel(project, files[b][k]),
                              "sha256": hashlib.sha256(files[b][k].read_bytes()).hexdigest()})
        out.append(lev)
    return sorted(out, key=lambda v: (len(v.x), v.level))


# =========================================================================== singular zones


@dataclass
class Feature:
    """A place where the elastic stress has no limit: a sharp re-entrant edge or a held face."""

    kind: str                     # edge | face
    label: str                    # for people: "re-entrant edge 12"
    points: np.ndarray            # the nodes of the finest mesh on it, (k, 3)
    size: float                   # element size of the finest mesh there, m
    distance: float               # nodes nearer than this are in its zone, m
    _tree: Any = None

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree


@dataclass
class Zones:
    wanted: list[str] = field(default_factory=list)      # every singular edge and held face of the inputs
    features: list[Feature] = field(default_factory=list)    # those found in the mesh
    missing: list[str] = field(default_factory=list)     # those not found
    problem: str = ""                                    # why the mesh was of no use (for the data file)
    loose: list[str] = field(default_factory=list)       # supports without faces: not left out
    mesh: str = ""                                       # the mesh file used
    surface: tuple | None = None                         # (tree of the surface nodes, face tags per node)

    @property
    def located(self) -> bool:
        return not self.missing

    def classify(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For nodes (n, 3): in a singular zone (bool), and the index of the nearest feature (or -1)."""
        best, which = np.full(len(x), np.inf), np.full(len(x), -1, dtype=np.int64)
        for k, f in enumerate(self.features):
            d, _ = f.tree().query(x, k=1, distance_upper_bound=f.distance * (1.0 + 1e-9))
            ratio = d / f.distance
            better = ratio < best
            best[better], which[better] = ratio[better], k
        inside = best <= 1.0
        which[~inside] = -1
        return inside, which

    def face_at(self, p, geo: dict) -> str:
        """The name of the face nearest to a point, for `near`."""
        if self.surface is None:
            return ""
        tree, tags = self.surface
        d, k = tree.query(np.asarray(p, dtype=float))
        same = tree.query_ball_point(tree.data[k], r=1e-9 * max(float(np.ptp(tree.data, axis=0).max()), 1e-300))
        faces = sorted({int(tags[q]) for q in same})[:3]
        names = []
        for t in faces:
            feats = (geo.get("faces", {}).get(str(t)) or {}).get("features") or []
            names.append(f"f{t}" + (f" ({', '.join(feats[:2])})" if feats else ""))
        return ("face " if len(names) == 1 else "faces ") + " and ".join(names) if names else ""


def _spacing(points: np.ndarray) -> float:
    """The median distance between a node and its nearest neighbour."""
    if len(points) < 2:
        return 0.0
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def _surface_groups(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """The points of a gmsh mesh, the node indices of each named surface group, and the element order."""
    import meshio

    m = meshio.read(str(path))
    pts = np.asarray(m.points, dtype=float)[:, :3]
    kinds = ("triangle", "triangle6")
    groups: dict[str, list] = {}
    if m.cell_sets:                                      # gmsh 4.1
        cells = m.cells_dict
        for name, by_type in m.cell_sets_dict.items():
            if name.startswith("gmsh:"):
                continue
            for t, idx in by_type.items():
                if t in kinds and len(idx):
                    groups.setdefault(name, []).append(np.asarray(cells[t])[np.asarray(idx, dtype=np.int64)].ravel())
    else:                                                # gmsh 2.2: one physical tag per element
        phys = m.cell_data.get("gmsh:physical")
        name_of = {int(v[0]): k for k, v in (m.field_data or {}).items() if int(v[1]) == 2}
        for block, tags in zip(m.cells, phys if phys is not None else []):
            if block.type not in kinds:
                continue
            tags = np.asarray(tags)
            for t in np.unique(tags):
                if int(t) in name_of:
                    groups.setdefault(name_of[int(t)], []).append(np.asarray(block.data)[tags == t].ravel())
    order = 2 if any(c.type == "triangle6" for c in m.cells) else 1
    return pts, {k: np.unique(np.concatenate(v)) for k, v in groups.items()}, order


def _mesh_file(project, records: list[AnalysisRecord], level: int) -> Path | None:
    """The mesh file of a level, from the records of the state analyses."""
    for rec in records:
        for ml in rec.mesh:
            if ml.level == level and ml.path and (project.out / ml.path).is_file():
                return project.out / ml.path
    for rec in records:
        for a in rec.artifacts:
            if a.kind == "mesh" and a.path.endswith(".msh") and re.search(rf"L{level}(?!\d)", Path(a.path).name) \
                    and (project.out / a.path).is_file():
                return project.out / a.path
    return None


def _zones(project, geo: dict, records: list[AnalysisRecord], fine: Level) -> Zones:
    """Find the sharp re-entrant edges and the held faces in the finest mesh."""
    z = Zones()
    edges = list((geo.get("edges") or {}).get("reentrant") or [])
    held: dict[str, str] = {}                            # mesh group -> label
    for rec in records:
        for bc in rec.bcs:
            if str(bc.type).lower() not in HELD_BC_TYPES:
                continue
            if not bc.faces and not bc.features:
                z.loose.append(f"{rec.id}: {bc.type}")
                continue
            for g in (bc.faces or bc.features):
                hit = _FACE_RE.match(g)
                feats = (geo.get("faces", {}).get(hit.group(1)) or {}).get("features") or [] if hit else []
                held.setdefault(g, f"held face {g}" + (f" ({', '.join(feats[:2])})" if feats else ""))
    edge_label = [f"re-entrant edge {e.get('tag', k + 1)}" for k, e in enumerate(edges)]
    z.wanted = [*edge_label, *held.values()]
    path = _mesh_file(project, records, fine.level)
    if path is None:
        z.problem = "the records of the state analyses give no mesh file for the finest level"
        z.missing = list(z.wanted)
        return z
    try:
        pts, groups, order = _surface_groups(path)
    except Exception as e:  # noqa: BLE001  meshio raises many kinds
        z.problem = f"the mesh file {path.name} is not readable: {type(e).__name__}: {e}"
        z.missing = list(z.wanted)
        return z
    z.mesh = _rel(project, path)
    tree_x = cKDTree(fine.x)
    everywhere = _spacing(pts[np.unique(np.concatenate(list(groups.values())))]) if groups else 0.0

    def add(kind: str, label: str, nodes: np.ndarray) -> None:
        if not len(nodes):
            z.missing.append(label)
            return
        p = pts[nodes]
        size = (_spacing(p) or everywhere) * order      # quadratic elements have a node at each mid-side
        if size <= 0.0 or float(tree_x.query(p)[0].max()) > size:
            z.problem = z.problem or f"the mesh file {path.name} does not have the nodes of the stress field"
            z.missing.append(label)
            return
        z.features.append(Feature(kind=kind, label=label, points=p, size=size, distance=EXCLUSION_ELEMENTS * size))

    for e, label in zip(edges, edge_label):
        faces = [f"f{t}" for t in (e.get("faces") or [])[:2]]
        if len(faces) < 2 or any(f not in groups for f in faces):
            z.missing.append(label)
            continue
        nodes = np.intersect1d(groups[faces[0]], groups[faces[1]])
        if e.get("midpoint") is not None and e.get("length"):      # two faces can share more than one edge
            reach = 0.5 * float(e["length"]) * 1.05 + 1e-9
            near = np.linalg.norm(pts[nodes] - np.asarray(e["midpoint"], dtype=float), axis=1) <= reach
            nodes = nodes[near] if near.any() else nodes
        add("edge", f"{label} (faces {faces[0]} and {faces[1]})", nodes)
    for g, label in held.items():
        add("face", label, groups.get(g, np.zeros(0, dtype=np.int64)))
    tags, nodes = [], []
    for name, idx in groups.items():
        hit = _FACE_RE.match(name)
        if hit:
            nodes.append(idx)
            tags.append(np.full(len(idx), int(hit.group(1)), dtype=np.int64))
    if nodes:
        z.surface = (cKDTree(pts[np.concatenate(nodes)]), np.concatenate(tags))
    return z


# =========================================================================== results at a node, convergence


@dataclass
class ZoneResult:
    """One result (part or smooth) with everything the record and the data file say about it."""

    key: str
    zone: str                                    # part | smooth
    value: float | None = None
    status: str = "not_run"
    uncertainty: float | None = None
    node: int | None = None
    location: list[float] | None = None
    near: str = ""
    in_singular: bool = False
    rim: str = ""                                # smooth zone: the singular zone whose rim the node is at
    per_level: list[float] = field(default_factory=list)
    cycles: list[dict] = field(default_factory=list)       # the counted cycles at the node
    duty: list[dict] = field(default_factory=list)         # the duty cycle table at the node
    simple: float | None = None                            # the simple count, largest over the zone
    convergence: dict = field(default_factory=dict)
    elastic: dict = field(default_factory=dict)            # the two validity checks of the zone
    note: str = ""


def _ratio(a: float, b: float) -> float:
    """max / min of two damages, 1 when both are zero, inf when one is zero."""
    lo, hi = min(a, b), max(a, b)
    return 1.0 if hi <= 0.0 else (math.inf if lo <= 0.0 else hi / lo)


def _governing(nodal: Nodal, mask: np.ndarray) -> int:
    """The node of a zone with the largest damage; without damage, with the largest corrected amplitude or range."""
    idx = np.nonzero(mask)[0]
    for key in (nodal.damage, np.where(np.isfinite(nodal.max_sar), nodal.max_sar, np.inf), nodal.max_range):
        if float(np.max(key[idx])) > 0.0:
            return int(idx[np.argmax(key[idx])])
    return int(idx[0])


def _node_tensors(level: Level, points: list[Point], node: int) -> np.ndarray:
    return _tensors(level.stress, points, node, node + 1)[:, 0, :]


def _attribute(cycles: list[dict], points: list[Point], duty: list[dict]) -> None:
    """Give the damage of each counted cycle to the duty cycles of its two states, in equal parts."""
    total = sum(c["damage"] for c in cycles)
    got = {d["duty_cycle"]: 0.0 for d in duty}
    for c in cycles:
        owners = [points[k].duty_cycle for k in (c["a"], c["b"]) if points[k].duty_cycle is not None]
        for o in owners:
            got[o] += c["damage"] / len(owners)
        c["share"] = c["damage"] / total if total > 0.0 else 0.0
    for d in duty:
        d["damage"] = got[d["duty_cycle"]]
        d["share"] = got[d["duty_cycle"]] / total if total > 0.0 else 0.0


def _elastic(nodal: Nodal, mask: np.ndarray, x: np.ndarray, sn: SN) -> dict:
    """The two validity checks of a zone: range <= 2 S_y, and von Mises stress of a state <= S_y."""
    idx = np.nonzero(mask)[0]
    tol = 1.0 + 1e-9
    out = {}
    for name, values, limit in (("shakedown", nodal.max_range, 2.0 * sn.sy), ("yield", nodal.max_vm, sn.sy)):
        v = values[idx]
        k = int(idx[np.argmax(v)])
        over = int(np.count_nonzero(v > limit * tol))
        out[name] = {"value": float(values[k]), "limit": float(limit), "passed": over == 0, "nodes_over": over,
                     "nodes": int(len(idx)), "location": [float(c) for c in x[k]]}
    floor = nodal.max_sar[idx] >= sn.sigma_f
    out["floor_nodes"] = int(np.count_nonzero(floor))
    return out


def _zone_result(key: str, zone: str, levels: list[Level], nodals: list[Nodal], masks: list[np.ndarray],
                 inside: list[np.ndarray], points: list[Point], sn: SN, limit: float) -> ZoneResult:
    """The largest damage of a zone on the finest mesh, its cycles, and its change from the coarser mesh."""
    fine, nodal, mask = levels[-1], nodals[-1], masks[-1]
    r = ZoneResult(key=key, zone=zone)
    if not mask.any():
        return r
    g = _governing(nodal, mask)
    r.node, r.value, r.location = g, float(nodal.damage[g]), [float(c) for c in fine.x[g]]
    r.in_singular = bool(inside[-1][g])
    r.simple = float(nodal.simple[mask].max())
    r.elastic = _elastic(nodal, mask, fine.x, sn)
    T = _node_tensors(fine, points, g)
    r.cycles = count_cycles(T, [p.reps for p in points], sn)
    r.duty = duty_table(T, points, sn)
    _attribute(r.cycles, points, r.duty)
    r.per_level = [float(n.damage[m].max()) if m.any() else 0.0 for n, m in zip(nodals, masks)]
    if len(levels) < 2:
        return r
    # the same location on the coarser mesh: its nearest node
    coarse, nodal_c = levels[0], nodals[0]
    dist, c = cKDTree(coarse.x).query(fine.x[g])
    c = int(c)
    if r.cycles:
        lead = max(r.cycles, key=lambda q: (q["damage"], q["corrected_amplitude"], q["range"]))
        a, b = lead["a"], lead["b"]
        range_f = lead["range"]
        Tc = _node_tensors(coarse, points, c)
        range_c = float(von_mises(Tc[a] - Tc[b]))
    else:
        a = b = None
        range_f, range_c = float(nodal.max_range[g]), float(nodal_c.max_range[c])
    change = abs(range_f - range_c) / range_f if range_f > 0.0 else (0.0 if range_c <= 0.0 else math.inf)
    here = _ratio(r.value, float(nodal_c.damage[c]))
    peak = _ratio(r.per_level[-1], r.per_level[0])
    worst = max(here, peak)
    r.uncertainty = worst - 1.0 if math.isfinite(worst) else None
    r.convergence = {"levels": [coarse.level, fine.level], "pair": [a, b], "node_distance": float(dist),
                     "range_fine": range_f, "range_coarse": range_c,
                     "range_change": change if math.isfinite(change) else None,
                     "grows": range_f > range_c * (1.0 + GROWTH),
                     "damage_fine": r.value, "damage_coarse": float(nodal_c.damage[c]),
                     "damage_ratio_at_node": here if math.isfinite(here) else None,
                     "damage_ratio_of_zone": peak if math.isfinite(peak) else None,
                     "passed": bool(math.isfinite(change) and change <= limit and math.isfinite(worst))}
    return r


def _status(r: ZoneResult, zones: Zones, two_levels: bool) -> str:
    """A result in a singular zone, or cut off at the rim of one, is never converged: the stress there has
    no limit, and two meshes can hide the growth (the rule of the structural engine for its peaks)."""
    if r.value is None:
        return "not_run"
    if r.in_singular:
        return "singular" if two_levels and r.convergence.get("grows") else "not_converged"
    if r.rim or not zones.located or not two_levels:
        return "not_converged"
    if not (r.elastic["shakedown"]["passed"] and r.elastic["yield"]["passed"]):
        return "not_converged"
    return "converged" if r.convergence.get("passed") else "not_converged"


# =========================================================================== text (ASD-STE100)


def _pa(v: float) -> str:
    return units.display(units.from_si(float(v), "Pa")) if math.isfinite(v) else "no finite value"


def _len(v: float) -> str:
    return units.display(units.from_si(float(v), "m"))


def _at(p) -> str:
    return units.display(units.from_si([float(c) for c in p], "m"))


def _num(v: float, sig: int = 3) -> str:
    return units.fmt(units.from_si(float(v), ""), sig=sig) if math.isfinite(v) else "no finite number"


def _pct(v: float) -> str:
    p = 100.0 * float(v)
    return f"{_num(p, 2) if abs(p) < 100 else _num(round(p), 4)} %"


def _count_text(n: float, what: str = "cycle") -> str:
    return f"{_num(n, 6)} {what}{'' if n == 1 else 's'}"


def _list(items: list[str], limit: int = 6) -> str:
    items = list(dict.fromkeys(items))
    if len(items) > limit:
        items = [*items[:limit], f"{len(items) - limit} more"]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _rule_name(sn: SN) -> str:
    return {"goodman": "Goodman relation", "swt": "Smith-Watson-Topper relation", "none": ""}[sn.rule]


def _rel(project, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(project.out).resolve()).as_posix()
    except ValueError:
        return str(path)


def _distances(zones: Zones) -> str:
    d = [f.distance for f in zones.features]
    return _len(d[0]) if max(d) - min(d) < 1e-3 * max(d) else f"{_len(min(d))} to {_len(max(d))}"


def _cycle_sentences(c: dict, points: list[Point]) -> list[str]:
    a, b = points[c["a"]].label, points[c["b"]].label
    life = ([f"The Basquin relation gives {_num(c['cycles_to_failure'])} cycles to failure."]
            if c["cycles_to_failure"] > MIN_CYCLES else
            ["The corrected stress amplitude is over the fatigue strength coefficient.",
             f"The analysis uses {_num(MIN_CYCLES)} cycles to failure."])
    return [f"The critical cycle is between {a} and {b}, with {_count_text(c['count'], 'repetition')}.",
            f"Its stress range is {_pa(c['range'])}, its stress amplitude is {_pa(c['amplitude'])} and its mean stress "
            f"is {_pa(c['mean_stress'])}.",
            *life,
            f"This cycle gives {_pct(c['share'])} of the damage at this node."]


def _accuracy(r: ZoneResult, zones: Zones, sn: SN, limit: float) -> list[list[str]]:
    """Why the result has its status and what the reader must know about its accuracy: a paragraph about
    the mesh, one about a singular zone and one about the validity of the method (six sentences or fewer)."""
    mesh, zone, valid = [], [], []
    cv = r.convergence
    if r.status == "singular":
        zone += [f"The critical node is in the singular zone of {r.near}.",
                 f"The stress range of the critical cycle increases from {_pa(cv['range_coarse'])} to "
                 f"{_pa(cv['range_fine'])} between the two finest meshes.",
                 "The elastic stress at a sharp re-entrant edge or at the rim of a held face has no limit.",
                 "The value changes with the mesh and is not a result of the part."]
    elif not cv:
        mesh += ["The analysis has the stress fields of one mesh only.", "It gives no mesh uncertainty."]
    elif cv["range_change"] is None:
        mesh.append("The stress range of the critical cycle is zero on one of the two finest meshes.")
    else:
        mesh += [f"The stress range of the critical cycle changes by {_pct(cv['range_change'])} between the two finest "
                 f"meshes.", f"The policy limit is {_pct(limit)}."]
    if r.status != "singular" and r.in_singular:
        zone += [f"The critical node is in the singular zone of {r.near}.",
                 "The elastic stress has no limit there, and two meshes can hide an increase.",
                 "This result cannot have the status converged."]
    if cv and cv.get("damage_ratio_of_zone") is not None and r.value:
        factor = max(cv["damage_ratio_of_zone"], cv["damage_ratio_at_node"] or 1.0)
        mesh += [f"The damage is in proportion to the stress range to the power {_num(sn.power)}.",
                 f"The damage changes by a factor of {_num(factor)} between the two finest meshes."]
    e = r.elastic
    if e and not (e["shakedown"]["passed"] and e["yield"]["passed"]):
        valid += [f"The stress is over the elastic range in the {_zone_name(r.zone)} (see the checks).",
                  "The elastic stress-life method does not apply at those nodes.",
                  "The damage there can be too small or too large."]
    if not zones.located:
        valid += ["The analysis did not find all singular zones in the mesh.",
                  "Then no result can have the status converged."]
    return [p for p in (mesh, zone, valid) if p]


def _zone_name(zone: str) -> str:
    return "part" if zone == "part" else "smooth zone"


def _text(paragraphs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def _notes(part: ZoneResult, smooth: ZoneResult, zones: Zones, sn: SN, limit: float) -> None:
    part.note = _text([["This is the largest damage of all nodes, for the specified repetitions without a life factor."],
                       *_accuracy(part, zones, sn, limit)])
    if smooth.value is None:
        why = ("The analysis did not find the singular zones in the mesh." if zones.wanted else
               "The part has no node out of the singular zones.")
        smooth.note = f"{why} This result has no value."
        return
    if not zones.features:
        head = ["The geometry has no sharp re-entrant edge and the load cases hold no face.",
                "This value is the same as damage.total@part."]
    else:
        head = [f"This value leaves out the nodes nearer than {_distances(zones)} to a sharp re-entrant edge or a held "
                f"face.", "It is for information only."]
        if smooth.rim:
            head += [f"The critical node is at the rim of the singular zone of {smooth.rim}.",
                     "The value changes with the distance. It cannot have the status converged."]
    smooth.note = _text([head, *_accuracy(smooth, zones, sn, limit)])


def _statement(project, lc: LoadCase, points: list[Point], levels: list[Level], part: ZoneResult, smooth: ZoneResult,
               zones: Zones, sn: SN, limit: float) -> str:
    comp = project.components.get(lc.component)
    duty = _duty(points)
    bases = list(dict.fromkeys(p.base for p in points if p.base))
    rule = _rule_name(sn)
    paras = [[f"This analysis calculates the fatigue damage of the part {comp.name if comp else lc.component}.",
              f"It uses {_count_text(len(duty), 'duty cycle')} and the elastic stress fields of "
              f"{_count_text(len(bases), 'structural load case')}.",
              ("It calculates the damage on the two finest meshes." if len(levels) > 1 else
               "It calculates the damage on one mesh.")],
             ["A cycle is a pair of stress states.",
              "Its stress range is the von Mises stress of the difference between the two stress tensors.",
              (f"The Basquin relation gives the cycles to failure, with the {rule} for the mean stress." if rule else
               "The Basquin relation gives the cycles to failure, without a mean stress correction."),
              "The Miner rule adds the damage of all cycles at each node."],
             [f"The largest damage in the part is {_num(part.value)}, for the specified repetitions without a life "
              f"factor.",
              f"The critical node is at {_at(part.location)}" + (f", at {part.near}." if part.near else "."),
              f"This result has the status {part.status.replace('_', ' ')}."]]
    if part.cycles:
        lead = max(part.cycles, key=lambda q: (q["damage"], q["corrected_amplitude"], q["range"]))
        paras.append(_cycle_sentences(lead, points))
    else:
        paras.append(["No cycle has a stress range at this node."])
    paras += _accuracy(part, zones, sn, limit)
    if smooth.value is not None and zones.features:
        paras.append([f"Out of the singular zones, the largest damage is {_num(smooth.value)} at {_at(smooth.location)}"
                      + (f", at {smooth.near}." if smooth.near else "."),
                      f"The singular zones are the nodes nearer than {_distances(zones)} to a sharp re-entrant edge or "
                      f"a held face.",
                      "This value is for information only.",
                      f"It has the status {smooth.status.replace('_', ' ')}."])
    elif smooth.value is None:
        paras.append([smooth.note])
    if part.simple is not None:
        paras.append([f"A count of one cycle for each repetition of each duty cycle gives a largest damage of "
                      f"{_num(part.simple)}.",
                      "That count leaves out the secondary cycles of a duty cycle and the cycles between two duty cycles.",
                      "The cycle count of this analysis includes them."])
    for d in part.duty:
        if d["a"] is None:
            paras.append([f"Duty cycle {d['duty_cycle']} has no repetition or no stress state.", "It gives no damage."])
            continue
        a, b = points[d["a"]].label, points[d["b"]].label
        paras.append([f"Duty cycle {d['duty_cycle']} has {_count_text(d['count'], 'repetition')}.",
                      f"At the critical node, its largest stress range is {_pa(d['range'])}, between {a} and {b}.",
                      f"The stress amplitude is {_pa(d['amplitude'])} and the mean stress is {_pa(d['mean_stress'])}.",
                      f"One such cycle for each repetition gives {_num(d['cycles_to_failure'])} cycles to failure and "
                      f"a damage of {_num(d['damage_alone'])}.",
                      f"The cycle count gives this duty cycle {_pct(d['share'])} of the damage at the critical node."])
    return _text(paras)


def _checks(lc: LoadCase, ctx, points: list[Point], levels: list[Level], part: ZoneResult, smooth: ZoneResult,
            zones: Zones, sn: SN, limit: float, bases: list[str]) -> list[Check]:
    def cid(key: str) -> str:
        return make_id("CHK", lc.id, key)

    out = [Check(id=cid("nodes"), kind="mesh_consistency",
                 statement=f"The stress fields of {_count_text(len(bases), 'load case')} have the same nodes on each "
                           f"mesh. The largest difference of a node coordinate is {_len(max(v.mismatch for v in levels))}.",
                 passed=True, value=max(v.mismatch for v in levels),
                 limit=NODE_TOLERANCE * max(v.size for v in levels))]
    # the repetitions against the system context: a factor in the load case counts the life factor two times
    known = getattr(ctx, "duty_cycles", None) or {}
    seen: dict[str, float] = {}
    for p in points:
        if p.duty_cycle is not None:
            seen[p.duty_cycle] = p.reps
    differ = [f"{dc}: {_num(n, 6)} in the load case and {_num(known[dc], 6)} in the system context"
              for dc, n in seen.items() if dc in known and not math.isclose(n, known[dc], rel_tol=1e-9, abs_tol=1e-12)]
    unknown = [dc for dc in seen if dc not in known]
    if differ:
        text = (f"The repetitions of the load case are not those of the system context ({'; '.join(differ)}). "
                f"The analysis uses the repetitions of the load case. The compliance stage applies the life factor.")
    elif unknown:
        text = (f"The system context gives no repetitions for {_list(unknown)}. "
                f"The analysis uses the repetitions of the load case.")
    else:
        text = "The repetitions of the load case are those of the system context, without a life factor."
    out.append(Check(id=cid("repetitions"), kind="input_consistency", statement=text,
                     passed=False if differ else (None if unknown else True)))
    if zones.wanted or zones.loose:
        n_edge = sum(f.kind == "edge" for f in zones.features)
        n_face = sum(f.kind == "face" for f in zones.features)
        if zones.located:
            text = (f"The analysis found the nodes of {_count_text(n_edge, 'sharp re-entrant edge')} and "
                    f"{_count_text(n_face, 'held face')} in the finest mesh.")
        else:
            text = (f"The analysis did not find the nodes of {_list(zones.missing)} in the finest mesh. "
                    f"It cannot tell if the critical node is in a singular zone. No result has the status converged.")
        if zones.loose:
            text += (f" {_count_text(len(zones.loose), 'support')} of the load cases "
                     f"{'has' if len(zones.loose) == 1 else 'have'} no face. The smooth zone does not leave them out.")
        out.append(Check(id=cid("singular-zones"), kind="singular_zone", statement=text,
                         passed=zones.located, value=float(len(zones.features)), limit=float(len(zones.wanted))))
    for r in ((part, smooth) if (smooth.value is not None and zones.features) else (part,)):
        e, where = r.elastic, _zone_name(r.zone)
        s = e["shakedown"]
        text = (f"The largest stress range in the {where} is {_pa(s['value'])}, at {_at(s['location'])}. "
                f"The limit for elastic shakedown is {_pa(s['limit'])} (2 times the yield strength).")
        text += (" The material stays elastic after the first cycles." if s["passed"] else
                 f" {_count_text(s['nodes_over'], 'node')} of {_num(s['nodes'], 7)} "
                 f"{'is' if s['nodes_over'] == 1 else 'are'} over the limit. "
                 f"The elastic stress-life method does not apply at these nodes. "
                 f"A strain-life method with a plasticity correction is necessary there.")
        out.append(Check(id=cid(f"shakedown-{r.zone}"), kind="elastic_range", statement=text, passed=s["passed"],
                         value=s["value"], limit=s["limit"]))
        y = e["yield"]
        text = (f"The largest von Mises stress of a stress state in the {where} is {_pa(y['value'])}, at "
                f"{_at(y['location'])}. The yield strength is {_pa(y['limit'])}.")
        text += (" The material does not yield." if y["passed"] else
                 f" {_count_text(y['nodes_over'], 'node')} of {_num(y['nodes'], 7)} "
                 f"{'is' if y['nodes_over'] == 1 else 'are'} over the yield strength. "
                 f"The material yields in the first cycle there, and the mean stress of the elastic analysis is not "
                 f"correct. A strain-life method with a plasticity correction is necessary there.")
        out.append(Check(id=cid(f"yield-{r.zone}"), kind="elastic_range", statement=text, passed=y["passed"],
                         value=y["value"], limit=y["limit"]))
        if e["floor_nodes"]:
            out.append(Check(id=cid(f"basquin-{r.zone}"), kind="fatigue_data", passed=False,
                             value=float(e["floor_nodes"]), limit=0.0,
                             statement=f"At {_count_text(e['floor_nodes'], 'node')} of the {where}, the corrected stress "
                                       f"amplitude is over the fatigue strength coefficient of {_pa(sn.sigma_f)}. "
                                       f"The Basquin relation gives no life there. "
                                       f"The analysis uses {_num(MIN_CYCLES)} cycles to failure at these nodes."))
        cv = r.convergence
        if not cv:
            out.append(Check(id=cid(f"convergence-{r.zone}"), kind="error_estimate", passed=None, limit=limit,
                             statement=f"The analysis has the stress fields of one mesh only. "
                                       f"It gives no mesh uncertainty for the damage in the {where}."))
        else:
            ch = cv["range_change"]
            text = (f"At the critical node of the {where}, the stress range of the critical cycle is "
                    f"{_pa(cv['range_fine'])} on the finest mesh. It is {_pa(cv['range_coarse'])} on the coarser mesh. "
                    + (f"The change is {_pct(ch)}. " if ch is not None else "")
                    + f"The policy limit is {_pct(limit)}.")
            passed = cv["passed"]
            if not passed:
                # CONTRACTS.md: convergence is carried by the result's status and uncertainty only, never by a
                # failed check. The status not_converged and the uncertainty of the damage show it.
                passed = None
                if r.zone == "smooth":
                    text += " The damage in the smooth zone is for information only. This check does not stop a pass."
                else:
                    text += (" The change is more than the policy limit, and the status and the uncertainty of the "
                             "damage show it. This check is information only.")
            out.append(Check(id=cid(f"convergence-{r.zone}"), kind="error_estimate", statement=text,
                             passed=passed, value=ch, limit=limit))
    return out


def _journal(j: Journal, project, lc: LoadCase, mat, sn: SN, points: list[Point], cases: dict[str, LoadCase],
             part: ZoneResult, zones: Zones, limit: float, two_levels: bool) -> None:
    s = lc.id
    rule = _rule_name(sn)
    j.decide(s, "method", "Fatigue method: elastic stress-life",
             choice="The analysis uses the elastic stress-life method. It takes the stress states from the linear "
                    "structural analyses. The Basquin relation gives the cycles to failure. The Miner rule adds the "
                    "damage of the cycles.",
             rationale=f"The material record of {mat.id} gives stress-life data only (Basquin model). "
                       "The method is correct while the material stays elastic in each cycle. "
                       "Two checks of the analysis show where the stress is over this limit.",
             alternatives=[("Use the strain-life method with a plasticity correction (Neuber rule).",
                            "The material record has no strain-life data and no cyclic stress-strain curve."),
                           ("Calculate the crack growth with fracture mechanics.",
                            "That method needs an initial crack size and crack growth data. "
                            "The material record does not give them.")],
             basis="input", links=[("source", REF_BASQUIN), ("source", REF_MINER), ("uses", mat.id)])
    j.decide(s, "stress-range", "Equivalent stress range of a cycle",
             choice="The stress range between two stress states is the von Mises stress of the difference between the "
                    "two stress tensors. The stress amplitude is half of the stress range.",
             rationale="This measure is correct for stress states with different principal directions. "
                       "For one stress component, it is the usual stress range. "
                       "ASME BPVC Section VIII Division 2, paragraph 5.5.3 uses it for the elastic fatigue analysis.",
             alternatives=[("Use the difference between the von Mises stresses of the two states.",
                            "That difference is zero when the stress changes its direction but not its magnitude. "
                            "It is not conservative."),
                           ("Use the range of the largest principal stress.",
                            "It does not include the shear stress and the other principal stresses."),
                           ("Use a critical plane method.",
                            "It needs material constants that the material record does not give.")],
             links=[("source", REF_ASME_RANGE)])
    if sn.rule == "goodman":
        how = (f"The Goodman relation corrects the stress amplitude with the ultimate strength of {_pa(sn.su)}.",
               "The material record specifies the Goodman relation.")
    elif sn.rule == "swt":
        how = ("The Smith-Watson-Topper relation corrects the stress amplitude with the largest stress of the cycle.",
               "The material record specifies the Smith-Watson-Topper relation.")
    else:
        how = ("The analysis applies no mean stress correction. A tensile mean stress then has no effect on the damage.",
               "The material record specifies no mean stress correction. That is not conservative for a tensile mean "
               "stress.")
    j.decide(s, "mean-stress", "Mean stress of a cycle and its correction",
             choice="The mean stress of a cycle is the mean of the first stress invariant of its two stress states. "
                    "The first stress invariant is the sum of the three normal stresses. "
                    f"{how[0]} A compressive mean stress gets no credit: the analysis sets it to zero.",
             rationale="For one stress component, this mean stress is the usual mean stress. "
                       "It has a sign, and it does not change with the shear stress (Sines, 1959). "
                       f"{how[1]} No credit for a compressive mean stress is conservative.",
             alternatives=[("Use a von Mises stress with a sign as the mean stress.",
                            "The sign is not clear when the principal stresses have different signs."),
                           ("Give credit for a compressive mean stress.",
                            "The benefit is not sure when the surface yields or has a residual stress.")],
             basis="input" if sn.rule != "none" else "rule", needs_review=sn.rule == "none",
             links=[("source", REF_SINES), *([("source", REF_SWT)] if sn.rule == "swt" else []), ("uses", mat.id)])
    simple = f" It gives a largest damage of {_num(part.simple)} here." if part.simple is not None else ""
    j.decide(s, "cycle-count", "Cycle count: pairs of stress states, largest stress range first",
             choice="The cycle count pairs the two stress states with the largest stress range first. "
                    "The number of these cycles is the smaller number of repetitions of the two states. "
                    "The count then continues with the repetitions that stay, until it finds no more pairs. "
                    "The states of all duty cycles and the unloaded state are in one count.",
             rationale="The system model gives the repetitions of each duty cycle, but not the order of the duty cycles. "
                       "No order of the duty cycles can give larger stress ranges than this count. "
                       "It also counts the secondary cycles in a duty cycle. "
                       "ASME BPVC Section VIII Division 2, Annex 5-B, paragraph 5-B.5 gives this count.",
             alternatives=[("Count one cycle of the largest stress range for each repetition of each duty cycle.",
                            "That count leaves out the secondary cycles of a duty cycle and the cycles between two duty "
                            f"cycles. It is not conservative.{simple}"),
                           ("Count the cycles with the rainflow method on a von Mises stress with a sign.",
                            "The sign is not clear for stress states with different principal directions. "
                            "The rainflow method also needs the order of the duty cycles.")],
             links=[("source", REF_ASME_COUNT)])
    if zones.features:
        j.decide(s, "singular-zones", "Distance from the singular zones for the result damage.total@smooth",
                 choice=f"The result damage.total@smooth leaves out the nodes nearer than {_num(EXCLUSION_ELEMENTS)} "
                        "element sizes to a sharp re-entrant edge or a held face. "
                        "The element size is that of the finest mesh at the edge or the face. "
                        f"The distance is {_distances(zones)}.",
                 rationale="The elastic stress at a sharp re-entrant edge and at the rim of a held face has no limit. "
                           "The stress in the first element layers there changes with the mesh. "
                           f"{_num(EXCLUSION_ELEMENTS)} element sizes of the finest mesh are approximately 2 element sizes "
                           "of the coarser mesh. The result is for information only, because it changes with this "
                           "distance.",
                 alternatives=[("Use a distance in proportion to the wall thickness, as the structural hot-spot stress "
                                "method for welds does.", "The geometry record gives no wall thickness at an edge."),
                               ("Use the theory of critical distances.",
                                "The material record gives no threshold of the stress intensity factor.")],
                 basis="engineering_judgement", needs_review=True)
    if two_levels:
        j.decide(s, "convergence", "Mesh convergence and uncertainty of the damage",
                 choice="The convergence criterion is the change of the stress range of the critical cycle at the "
                        f"critical node between the two finest meshes. The limit is {_pct(limit)} (policy). "
                        "The uncertainty of the damage is the larger of two relative differences between the two meshes. "
                        "The first is the damage at the critical node. The second is the largest damage in the zone. "
                        "A result with its critical node in a singular zone, or at the rim of one, cannot have the "
                        "status converged.",
                 rationale="The policy limit applies to a response of the finite element analysis, which is the stress. "
                           f"The damage is in proportion to the stress range to the power {_num(sn.power)}. "
                           f"A change of the stress range of 1 % changes the damage by {_pct(1.01 ** sn.power - 1)}. "
                           "The uncertainty of the damage includes this effect. Then the margin of safety includes it "
                           "too.",
                 alternatives=[(f"Apply the policy limit of {_pct(limit)} to the damage.",
                                f"The stress range must then change by less than {_pct((1 + limit) ** (1 / sn.power) - 1)}. "
                                "The meshes of a part with stress concentrations do not get to this accuracy."),
                               ("Give the change of the stress range as the uncertainty of the damage.",
                                "The margin of safety then does not include the larger change of the damage. "
                                "It is not conservative.")],
                 basis="engineering_judgement", needs_review=True)
    if sn.endurance is not None:
        j.decide(s, "endurance-limit", "Use of the endurance limit",
                 choice=f"A node has no damage when each of its cycles is below {_pa(sn.cut)}. "
                        f"That is the endurance limit of {_pa(sn.endurance)} divided by {_num(1 + sn.margin)}. "
                        "At all other nodes, each cycle gives damage on the Basquin line, also below the endurance limit.",
                 rationale="Large cycles decrease the endurance limit of the material for the small cycles that follow. "
                           "As a result, the endurance limit is correct only for a node without a cycle above it. "
                           f"The divisor is the largest mesh uncertainty of a converged stress ({_pct(sn.margin)}, policy).",
                 alternatives=[("Give no damage to each cycle below the endurance limit.",
                                "That is not conservative when larger cycles occur at the same node."),
                               ("Ignore the endurance limit.", "The material record gives it. It is correct for a node "
                                                               "where all cycles are small.")],
                 basis="rule", links=[("uses", mat.id)])

    # ---- what the analysis takes as true
    least = (f" The count of one cycle for each repetition of each duty cycle gives {_num(part.simple)}."
             if part.simple is not None else "")
    j.assume(s, "order", "Order of the duty cycles and load reversals in a duty cycle",
             statement="The system model does not give the order of the duty cycles in the life of the part. "
                       "The cycle count pairs the stress states for the largest stress ranges. "
                       "It takes each state of a duty cycle as a possible load reversal.",
             rationale="The system model gives the number of repetitions of each duty cycle only. "
                       "A cycle between two duty cycles occurs when they alternate. "
                       "A count for one specified order cannot give larger stress ranges.",
             impact="With a specified order of the duty cycles, the damage can be smaller than this analysis gives."
                    f"{least}",
             verify_by="Compare with the operation plan of the part.", conservative=True)
    j.assume(s, "states", "Stress between the states of a duty cycle",
             statement="Between two states of a duty cycle, the stress is not larger than in these states. "
                       "The states are the steady conditions of the phases at limit loads.",
             rationale="The system model gives one set of loads for each phase and no load history between the phases.",
             impact="A fast temperature change or a pressure peak between two phases gives a larger stress range. "
                    "The damage is then larger than this analysis gives.",
             verify_by="Do a transient analysis of the transitions that change the temperature fast.", conservative=False)
    temps = _temperatures(points, cases)
    t_data = mat.properties_temperature
    data_at = (f"The fatigue data and the ultimate strength of {mat.id} apply at "
               f"{units.display(t_data.quantity())}." if t_data is not None and t_data.is_literal else
               f"The material record of {mat.id} gives no temperature for its fatigue data.")
    span = (f"The load cases have temperatures from {units.display(units.from_si(min(temps), 'K'))} to "
            f"{units.display(units.from_si(max(temps), 'K'))}." if temps else
            "The load cases give no temperature.")
    j.assume(s, "temperature", "Fatigue data at the temperatures of the load cases",
             statement=f"The analysis uses one set of fatigue data for all stress states. {data_at} {span}",
             rationale="The material record gives fatigue data for one temperature only.",
             impact="The fatigue strength and the ultimate strength change with the temperature. "
                    "A smaller fatigue strength at a temperature of the load cases gives a larger damage.",
             verify_by="Compare with fatigue data of the material at the lowest and the highest temperature.",
             conservative=None, links=[("uses", mat.id)])
    j.assume(s, "fatigue-strength", "No reduction of the fatigue strength",
             statement="The fatigue data applies to each node of the part without a reduction. "
                       "The analysis applies no fatigue strength reduction factor for welds, surface roughness, size or "
                       "environment.",
             rationale=f"The material record of {mat.id} gives one stress-life line. "
                       "The geometry record does not show the welds or the surface finish.",
             impact="A weld or a rough surface has a smaller fatigue strength than a smooth specimen. "
                    "The damage at such a location is then larger than this analysis gives.",
             verify_by="Find the welds and the surface finish in the design data. Apply the reduction factors of the "
                       "applicable standard.",
             conservative=False, links=[("uses", mat.id)])
    j.assume(s, "unloaded-state", "No stress in the unloaded state",
             statement="The part has no stress in the unloaded state at the reference temperature. "
                       "The analysis includes no residual stress and no stress from the assembly.",
             rationale="The structural load cases give the stress of the loads of each phase only.",
             impact="A residual stress or a bolt preload changes the mean stress of each cycle. "
                    "A tensile mean stress decreases the cycles to failure.",
             verify_by="Find the residual stress of the manufacture and the preload of the joints.", conservative=None)
    j.assume(s, "miner", "Failure at a damage of 1",
             statement="The damage of cycles with different stress ranges adds linearly, and the order of the cycles "
                       "has no effect. A fatigue crack starts at a damage of 1.",
             rationale="This is the Miner rule. The standards for fatigue design use it.",
             impact="Tests with cycles of different stress ranges show cracks at a damage below 1 and above 1. "
                    "The life factor of the requirement must include this scatter.",
             conservative=None, links=[("source", REF_MINER)])


def _temperatures(points: list[Point], cases: dict[str, LoadCase]) -> list[float]:
    """The temperatures (K) that the loads of the state load cases, and of the cases they use, name."""
    out, seen = [], set()
    todo = [p.label for p in points if p.duty_cycle is not None] + [p.base for p in points if p.base]
    while todo:
        cid = todo.pop()
        if cid in seen or cid not in cases:
            continue
        seen.add(cid)
        for item in cases[cid].loads:
            if item.kind in ("fluid_temperature", "temperature") and isinstance(item.si, (int, float)):
                out.append(float(item.si))
        todo += cases[cid].uses
    return out


# =========================================================================== the data file


def _clean(obj):
    """Plain JSON data: numpy types to Python, values that are not finite to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _zone_data(r: ZoneResult, points: list[Point]) -> dict:
    def name(k):
        return None if k is None else points[k].label

    return {"key": r.key, "damage": r.value, "status": r.status, "uncertainty": r.uncertainty, "node": r.node,
            "location": r.location, "near": r.near, "in_singular_zone": r.in_singular,
            "at_rim_of_singular_zone": r.rim, "per_level": r.per_level,
            "one_cycle_per_repetition": r.simple,
            "cycles": [{**c, "state_a": name(c["a"]), "state_b": name(c["b"]),
                        "duty_cycle_a": points[c["a"]].duty_cycle, "duty_cycle_b": points[c["b"]].duty_cycle}
                       for c in r.cycles],
            "duty_cycles": [{**d, "state_a": name(d["a"]), "state_b": name(d["b"])} for d in r.duty],
            "elastic_range": r.elastic, "convergence": r.convergence, "note": r.note}


# =========================================================================== run


def _empty(lc: LoadCase, status: str, message: str) -> Analysis:
    rec = AnalysisRecord(id=_record_id(lc), title=lc.title or "Fatigue analysis", load_case=lc.id,
                         analysis_type=lc.analysis_type, status=status, message=message,
                         solver=f"dossier.analysis.fatigue {__version__}", origin=Origin(by=BY, stage="analyze"))
    return Analysis(rec, Journal("solve", BY))


def run(project, ctx, geo, lc: LoadCase, workdir, opts) -> Analysis:
    """The fatigue analysis of one load case of type "fatigue". See the module text for the method."""
    t0 = time.time()
    try:
        return _run(project, ctx, geo or {}, lc, Path(workdir), opts, t0)
    except _Stop as e:
        return _empty(lc, e.status, str(e))


def _run(project, ctx, geo: dict, lc: LoadCase, workdir: Path, opts, t0: float) -> Analysis:
    log = getattr(opts, "log", None) or (lambda _msg: None)
    sn, mat = _stress_life(project, ctx, lc)
    points, cases, analyses = _points(project, lc)
    bases = list(dict.fromkeys(p.base for p in points if p.base))
    levels = _levels(project, bases, analyses)
    fine, two = levels[-1], len(levels) > 1
    limit = float(project.policy.max_uncertainty)
    records = [analyses[b] for b in bases]
    zones = _zones(project, geo, records, fine)
    log(f"  fatigue: {len(points) - 1} states of {len(_duty(points))} duty cycles, {len(fine.x)} nodes on L{fine.level}"
        + (f" and {len(levels[0].x)} on L{levels[0].level}" if two else "")
        + f", {len(zones.features)} of {len(zones.wanted)} singular zones found")

    nodals = [evaluate(v.stress, points, sn) for v in levels]
    classes = [zones.classify(v.x) for v in levels]
    inside = [c[0] for c in classes]
    everything = [np.ones(len(v.x), dtype=bool) for v in levels]
    part = _zone_result("damage.total@part", "part", levels, nodals, everything, inside, points, sn, limit)
    if zones.wanted and not zones.features:
        smooth = ZoneResult(key="damage.total@smooth", zone="smooth")      # the zones are not known: no value
    else:
        smooth = _zone_result("damage.total@smooth", "smooth", levels, nodals, [~m for m in inside], inside,
                              points, sn, limit)
    for r in (part, smooth):
        if r.value is None:
            continue
        k = int(classes[-1][1][r.node])
        r.near = zones.features[k].label if k >= 0 else zones.face_at(r.location, geo)
        if r.zone == "smooth":
            # At the rim of a singular zone whose nodes have more damage, the value is the singular field cut
            # off at the distance of the zone: it follows the distance, not the part.
            for q, f in enumerate(zones.features):
                d, _ = f.tree().query(fine.x[r.node])
                mine = classes[-1][1] == q
                if d <= RIM_FACTOR * f.distance and mine.any() and float(nodals[-1].damage[mine].max()) > r.value:
                    r.rim = f.label
                    break
        r.status = _status(r, zones, two)
        if r.in_singular:
            r.uncertainty = None                           # a value without a limit has no uncertainty
    _notes(part, smooth, zones, sn, limit)

    # ---- files
    workdir.mkdir(parents=True, exist_ok=True)
    npz = workdir / f"damage_L{fine.level}.npz"
    np.savez_compressed(npz, x=fine.x, damage=nodals[-1].damage, max_range=nodals[-1].max_range,
                        max_von_mises=nodals[-1].max_vm, one_cycle_per_repetition=nodals[-1].simple, smooth=~inside[-1])
    data = {
        "schema": 1, "tool": f"dossier.analysis.fatigue {__version__}", "load_case": lc.id, "analysis": _record_id(lc),
        "units": "SI: stress in Pa, length in m; damage and counts without a unit",
        "method": {"stress_range": "von Mises stress of the difference of two stress tensors (ASME VIII-2, 5.5.3)",
                   "mean_stress": "mean of the first stress invariant of the two states; compressive mean = 0",
                   "correction": sn.rule, "life": "Basquin: S_ar = sigma_f (2N)^b, N >= 0.5",
                   "count": "max-min, all duty cycles in one count (ASME VIII-2, Annex 5-B.5)",
                   "damage": "Miner sum; the repetitions have no life factor",
                   "attribution": "a cycle between states of two duty cycles gives half of its damage to each"},
        "material": {"id": mat.id, "sigma_f": sn.sigma_f, "b": sn.b, "mean_stress": sn.rule, "ultimate_strength": sn.su,
                     "yield_strength": sn.sy, "endurance_limit": sn.endurance, "endurance_cut": sn.cut,
                     "source": mat.fatigue.source},
        "time_points": [{"index": k, "state": p.label, "duty_cycle": p.duty_cycle, "field_of": p.base,
                         "scale": p.scale, "repetitions": p.reps} for k, p in enumerate(points)],
        "levels": [{"level": v.level, "nodes": len(v.x), "largest_damage": float(n.damage.max()),
                    "largest_range": float(n.max_range.max())} for v, n in zip(levels, nodals)],
        "fields": [f for v in levels for f in v.files],
        "singular_zones": {"element_sizes": EXCLUSION_ELEMENTS, "mesh": zones.mesh, "problem": zones.problem,
                           "not_found": zones.missing, "supports_without_faces": zones.loose,
                           "nodes_left_out": int(inside[-1].sum()), "nodes": int(len(fine.x)),
                           "zones": [{"kind": f.kind, "name": f.label, "nodes_on_it": len(f.points),
                                      "element_size": f.size, "distance": f.distance} for f in zones.features]},
        "part": _zone_data(part, points), "smooth": _zone_data(smooth, points),
    }
    jpath = workdir / "fatigue.json"
    jpath.write_text(json.dumps(_clean(data), indent=1, allow_nan=False) + "\n")

    # ---- the record
    results = {}
    for r in (part, smooth):
        results[r.key] = ResultValue(key=r.key, value=r.value, unit="", uncertainty=r.uncertainty, status=r.status,
                                     location=r.location, near=r.near, per_level=r.per_level, method="derived",
                                     note=r.note)
    j = Journal("solve", BY)
    _journal(j, project, lc, mat, sn, points, cases, part, zones, limit, two)
    states = list(dict.fromkeys(p.label for p in points if p.duty_cycle is not None))
    links = [Link(rel="uses", to=analyses[c].id, note="stress field of a state") for c in bases]
    links += [Link(rel="uses", to=analyses[c].id, note="scaled state") for c in states if c not in bases and c in analyses]
    links += [Link(rel="uses", to=mat.id, note="fatigue data")]
    links += [Link(rel="derives_from", to=dc) for dc, _, _ in _duty(points)]
    links += [Link(rel="source", to=ref) for ref in (REF_ASME_RANGE, REF_ASME_COUNT, REF_BASQUIN, REF_MINER)]
    used = {v.level for v in levels}
    rec = AnalysisRecord(
        id=_record_id(lc), title=lc.title or "Fatigue analysis", load_case=lc.id, analysis_type=lc.analysis_type,
        status="done",
        statement=_statement(project, lc, points, levels, part, smooth, zones, sn, limit),
        solver=f"dossier.analysis.fatigue {__version__}: elastic stress-life (Basquin, {sn.rule}, Miner), max-min count",
        material={"id": mat.id, "fatigue_model": "basquin", "sigma_f": sn.sigma_f, "b": sn.b, "mean_stress": sn.rule,
                  "endurance_limit": sn.endurance, "ultimate_strength": sn.su, "yield_strength": sn.sy,
                  "fatigue_source": mat.fatigue.source},
        mesh=[ml for ml in (records[0].mesh if records else []) if ml.level in used],
        results=results,
        checks=_checks(lc, ctx, points, levels, part, smooth, zones, sn, limit, bases),
        artifacts=[Artifact(path=_rel(project, jpath), kind="data",
                            caption="Fatigue analysis: the stress states, the singular zones, and the cycles and the "
                                    "duty cycles at the critical nodes."),
                   Artifact(path=_rel(project, npz), kind="data",
                            caption=f"Damage and largest stress range at each node of mesh level {fine.level}.")],
        links=links, runtime_s=round(time.time() - t0, 2), origin=Origin(by=BY, stage="analyze"))
    return Analysis(rec, j)
