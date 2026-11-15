"""Static and thermo-structural analysis of one load case with the fixed solver (feacore, scikit-fem).

    run(project, ctx, geo, lc, workdir, opts) -> pipeline.Analysis        the engine of the pipeline
    run_levels(project, ctx, geo, lc, levels, workdir, opts, ...)          the same on given meshes

What it does, in order
  1. Idealization (dossier.analysis.idealize): every load and support of the load case becomes a boundary
     condition of the solver, each with a Decision. A load without a rule gives status `not_supported`.
  2. Meshes: dossier.meshing.levels (quadratic tetrahedra, coarse to fine).
  3. One linear elastic solve on every level: `feacore.solver.solve(case, backend="skfem")`. A
     thermo-structural case first gets the temperature on the SAME mesh from
     dossier.analysis.thermal.solve_temperature and applies the thermal strain alpha (T - T_ref).
  4. Results on every level, then the convergence of every result over the levels.
  5. Checks of the analysis on itself, fields for the fatigue engine, pictures, the record.

Results (keys `<quantity>.<measure>@<location>`, SI)
  von_mises.peak@part                 always. Also `max_principal.peak`, and `@<feature>` / `@<interface>`.
                                      The peak is the largest nodal value of the solver's recovered stress.
                                      Nodes within the stated distance of a held face are left out.
  <inv>.membrane@<SCL>, <inv>.membrane_bending@<SCL>   for von_mises and max_principal, on every stress
                                      classification line (dossier.analysis.scl, 64 sample points)
  sxx | syy | szz.membrane@<SCL>      global components of the membrane tensor
  von_mises.membrane@scls, von_mises.membrane_bending@scls   the worst line (see "aggregate" below)
  displacement.max | min | mean@<part | feature | interface>   magnitude of the displacement
  radial_displacement.range@<cylindrical feature>     out-of-roundness (see "roundness" below)
  radial_displacement.max | min | mean@<cylindrical feature>   about the fixed axis of the feature
A key of `lc.result_keys` that the engine cannot give is named in the record's `message` with the reason.

Convergence rule (exact). u is the absolute uncertainty of the value v on the finest mesh.
  h is the solver's representative element size (volume / cells)^(1/3) of each level.
  p_theory is 2 for a stress and 3 for a displacement (quadratic elements, smooth solution).
  one level      no study. Status not_converged, no uncertainty.
  two levels     u = 3 |v2 - v1| x max(1, 1 / (r^p - 1)), with r = h1 / h2 and p = min(p_theory, 2).
                 (Roache's two-mesh index with the safety factor 3; never less than 3 x the change.)
  three or more  feacore.verification.gci3 classifies the finest level and two coarser levels (each at
                 least 1.3 times larger in h where the sequence allows it). The uncertainty is 3 times the
                 best error estimate that the class allows:
                   monotone, order p    the error estimate is |v_fine - v_medium| / (r^p - 1).
                                        u = max(1.25 x that with the observed p,
                                                3 x that with p limited to [0.5, p_theory])
                                        (the grid convergence index, Celik et al. 2008 / Roache)
                   monotone, p far above theory (p > 1.5 p_theory + 0.5: not in the asymptotic range)
                                        u = max(the two values above, 3 x spread). No order and no
                                        extrapolated value are given (the observed order is not used).
                   oscillatory, the changes become smaller.
                                        u = 3 x the spread (max - min) of the last three values (the last
                                        change alone can be small by chance: 80, 100, 99.9)
                   diverging, or no order   u = 3 x spread, spread = max - min of the three values. This is
                                        NOT an error estimate (error_estimated False): it is the scatter of
                                        values that do not converge.
                   changes < 1e-6       u = 1.25 x the larger change (round-off)
  line noise     A result on a stress classification line also depends on where the line cuts the elements
                 (the recovered stress scatters at the element scale). On every level the engine evaluates
                 the line four more times, moved by a quarter of the local element size in the two
                 directions across the line. u is not less than 3 x (max - min) of these five values on the
                 finest mesh. Three levels at one fixed line can agree by chance; this floor does not.
  error_estimated  True when u comes from a sequence that converges: monotone, oscillating with smaller
                 changes, two levels, or changes at round-off. Also True on a line whose last three values
                 differ by no more than the line noise (below): the noise explains the scatter, and u covers it. False for a diverging or indeterminate sequence,
                 one level, and every result without an uncertainty. The compliance stage does not conclude
                 from a not converged result without an error estimate.
  status         converged when u is an error estimate and u <= policy.max_uncertainty x max(|v|, reference),
                 else not_converged. For a result that a criterion or a hand calculation of the load case uses
                 (`lc.result_keys`, and each line of an `@scls` key), the reference is the value itself.
                 The reference is the size of the field that a small value comes from: the largest principal
                 stress (absolute) of the section tensor for a stress component and for max_principal on a
                 line, and the largest displacement of the feature for a range of the radial displacement.
                 So a component near zero in a large stress state, or a roundness near zero, can be
                 converged when nothing judges it. For every other result the reference is the value itself.
                 A value near zero that a criterion uses is then not_converged, and its note says why.
  uncertainty    `uncertainty_abs` of a result is u (in the unit of the value; the compliance stage uses it).
                 `uncertainty` is u / |v| (relative to the value; empty when v is zero). It can be above 1 for
                 a value near zero that is converged against its reference.
  peaks          A peak that is within NEAR_EDGE (2.5) local element sizes of a re-entrant edge, or within
                 two local element sizes of the border of the zone left out at a held face, is NEVER
                 converged: the stress there has no limit in linear elasticity, and three meshes can hide
                 the growth. It is `singular` when its magnitude grew at every refinement (by more than
                 0.1 %), else `not_converged`. Its uncertainty and order are empty: a value without a limit
                 has none. A peak ANYWHERE that grew at every refinement and whose sequence is diverging or
                 indeterminate is `singular` too (a list of re-entrant edges can miss an edge).
  aggregate      `@scls` is the line with the largest value. Its uncertainty covers the conservative value
                 v + u of every line. Its status is the worst status among the lines whose conservative
                 value reaches the largest value. A line without a result makes the aggregate not_converged.
                 The aggregate is for display only: the compliance stage assesses each line by itself.
  The value of a result is always the value on the finest mesh; `extrapolated` is information.

Peaks next to held faces: the rim of a rigid support is a singular line. The engine leaves out of every
peak evaluation the nodes within EXCLUSION (2) element sizes of the coarsest mesh from a held face (one
distance for all levels; a Decision with needs_review gives it).

Free part: the displacement of a free part has no datum. Displacement results of a free part are relative
to its mean rigid-body motion (the volume least-squares rigid-body fit is removed; a Decision says so).

Roundness: `radial_displacement.range@<feature>` is the largest minus the smallest radial displacement of
the nodes of a cylindrical feature AFTER the rigid-body motion of the feature (a shift across the axis and
a tilt, fitted by least squares over the feature) is removed. A rigid shift of a round bore is not
out-of-roundness. The note of the result also gives the range about the fixed axis, which includes the
rigid-body motion. A Decision with needs_review records this. The axis direction comes from `geo`; the
axis position is fitted to the nodes of the feature.

Checks in the record: load balance and restraint reactions of a free part, the resultant of every moment
traction, the loads that the solver assembled against the loads of the load case, equilibrium of the
support reactions, mesh quality, the energy-norm error estimate (Zienkiewicz-Zhu, information), the
validity of small displacements, and the temperatures of several held interfaces.

Fields: `fields_L<k>.npz` in the workdir for the two finest levels with x (n, 3), stress (n, 6: sxx, syy,
szz, sxy, syz, sxz, Pa) and u (n, 3, m; as solved, with the restraint of the model). The nodes are ALL
nodes of the quadratic solution in the solver's order, which depends on the mesh file only, so every load
case on one mesh level has the same node set in the same order.

Not done / not verified (read this before you rely on a number)
  * The recovered nodal stress scatters by 1 to 3 % at boundaries on coarse meshes (a smooth Lame cylinder
    with 2.5 elements through the wall). The peak then reads `not_converged`; this is on purpose.
  * The thermal coupling needs dossier.analysis.thermal and the solver's temperature-field load, which
    another engine supplies. The support that follows the thermal contraction is verified with a uniform
    temperature only (zero stress); a real gradient case is not verified against a closed form.
  * Stress classification lines are linearized in all six components (not the ASME choice of components).
  * `opts.record_runs` (the dashboard's run store) is not served.
  * Nonlinear effects (yield, contact at the joints, bolt preload) are not in the model.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dossier import units
from dossier.analysis import scl as scl_mod
from dossier.analysis.idealize import (REACTION_TOL, Idealizer, MeshGeom, NotSupported, Realized, Unbalanced,
                                       fmt, names)
from dossier.model import AnalysisRecord, Artifact, Check, Link, LoadCase, MeshLevel, Origin, ResultValue
from dossier.project import make_id
from dossier.records import Journal

COMPONENTS = scl_mod.COMPONENTS
P_STRESS, P_DISPLACEMENT = 2.0, 3.0
SAFETY_GCI = 1.25               # the grid convergence index with the observed order
SAFETY_TWO_LEVELS = 3.0
SAFETY_SPREAD = 3.0             # every other error estimate
MIN_RATIO = 1.3                 # the levels of a GCI are at least this far apart in h where possible
GROWTH = 1e-3                   # a peak "grew" when its magnitude rose by more than this at a refinement
EXCLUSION = 2.0                 # peak evaluation: nodes within this many coarse element sizes of a held face are left out
NEAR_EDGE = 2.5                 # a peak is at a re-entrant edge within this many local element sizes
NEAR_ZONE = 2.0                 # ... or within this many local element sizes of the border of the left-out zone
LINE_SHIFT = 0.25               # a line is also evaluated moved by this part of the local element size
SCL_OUTSIDE = 0.02              # a sample point farther outside the mesh than this part of the line: no result
EQUILIBRIUM_TOL = 1e-6
ASSEMBLY_TOL = 1e-6
ZZ_LIMIT = 0.30                 # feacore's "fail" threshold of the energy-norm error estimate
DISPLACEMENT_LIMIT = 0.05       # largest displacement / part size for a linear analysis (feacore's limit)
QUALITY_LIMIT = 0.01            # feacore's "fail" threshold of the corner mean-ratio quality
STRESS_QUANTITIES = ("von_mises", "max_principal")


class CannotServe(Exception):
    """The engine cannot give this result key. The message is the reason (STE)."""


def analysis_id(lc: LoadCase) -> str:
    return "AN-" + (lc.id[3:] if lc.id.startswith("LC-") else lc.id)


# --------------------------------------------------------------------------- convergence


def gci_triplet(h: list[float]) -> list[int]:
    """Indices (coarse to fine) of the three levels for the GCI: the finest, then each time the next coarser
    level at least MIN_RATIO larger in h. The three finest when the sequence does not allow it."""
    order_ = [int(i) for i in np.argsort(h)]                       # fine -> coarse
    chosen = [order_[0]]
    for i in order_[1:]:
        if h[i] >= MIN_RATIO * h[chosen[-1]]:
            chosen.append(i)
            if len(chosen) == 3:
                break
    if len(chosen) < 3:
        chosen = order_[:3]
    return sorted(chosen, key=lambda i: -h[i])


@dataclass
class Study:
    status: str
    uncertainty: float | None
    extrapolated: float | None = None
    order: float | None = None
    note: str = ""
    grew: bool = False                 # the magnitude rose at every refinement
    kind: str = ""                     # gci3's convergence class, or two_levels / one_level
    absolute: float | None = None      # u, the uncertainty in the unit of the value
    estimated: bool = False            # u is an error estimate of a converging sequence (ResultValue.error_estimated)


def study(h: list[float], values: list[float], p_theory: float, max_uncertainty: float,
          reference: float | None = None, noise: float | None = None) -> Study:
    """The convergence of one result over the levels (coarse to fine). The rule is in the module docstring.
    `reference` is the size that a small value is judged against (None: the value itself). `noise` is the
    change of the value on the finest mesh when its line moves by a quarter of an element (lines only)."""
    from feacore.verification.gci import gci3

    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0 or not np.all(np.isfinite(v)):
        return Study("not_converged", None, note="A mesh level has no value for this result.", kind="no_value")
    fine = float(v[-1])
    grew = bool(n >= 2 and all(abs(v[i + 1]) > abs(v[i]) * (1.0 + GROWTH) for i in range(n - 1)))
    if n == 1:
        return Study("not_converged", None, note="The analysis has one mesh level. One level gives no convergence study.",
                     kind="one_level")
    if np.max(np.abs(v)) == 0.0:
        return Study("converged", 0.0, extrapolated=0.0, note="The value is zero on every mesh level.", kind="exact",
                     absolute=0.0, estimated=True)
    size = max(abs(fine), abs(reference or 0.0), 1e-300)

    def pct(x: float) -> str:
        return f"{100 * x / size:.2g} %"

    ext = order = None
    estimated = True
    if n == 2:
        r = h[0] / h[1]
        p = min(p_theory, 2.0)
        change = abs(v[1] - v[0])
        factor = max(1.0, 1.0 / (r ** p - 1.0)) if r > 1.0 else math.inf
        u = SAFETY_TWO_LEVELS * change * factor
        estimated = math.isfinite(u)
        kind = "two_levels"
        note = (f"Two mesh levels with a size ratio of {r:.2f}. The value changes by {pct(change)} between them. "
                f"The uncertainty is {SAFETY_TWO_LEVELS:g} times this change"
                + (f", divided by {r ** p - 1.0:.2f} for the small size ratio." if factor > 1.0 else "."))
    else:
        tri = gci_triplet(list(h))
        ht, vt = [float(h[i]) for i in tri], [float(v[i]) for i in tri]
        spread = max(vt) - min(vt)
        last = abs(vt[2] - vt[1])
        try:
            g = gci3(ht, vt, p_theory=p_theory)
        except ValueError:
            g = None
        kind = g.convergence if g is not None else "indeterminate"
        if kind in ("exact", "converged"):
            u = SAFETY_GCI * max(last, abs(vt[1] - vt[0]))
            note = "The value does not change between the mesh levels (round-off only)."
        elif kind == "monotone":
            p_lim = min(max(g.p, 0.5), p_theory)
            u = max(SAFETY_GCI * last / (g.r21 ** g.p - 1.0), SAFETY_SPREAD * last / (g.r21 ** p_lim - 1.0))
            note = (f"The three values converge in one direction with the observed order {g.p:.2f} "
                    f"(theory {p_theory:g}).")
            if "p_above_theory" in g.flags:
                # not in the asymptotic range: no order and no extrapolated value are given
                u = max(u, SAFETY_SPREAD * spread)
                note += (" The observed order is far above the theory, so the meshes are not in the asymptotic range "
                         "and the engine gives no order. "
                         f"The uncertainty is {SAFETY_SPREAD:g} times the spread of the three values ({pct(spread)}).")
            else:
                order = float(g.p)
                ext = float(g.phi_ext)
                note += " The uncertainty is the larger of two grid convergence indexes."
        elif kind == "oscillatory":
            # the last change alone can be small by chance (80, 100, 99.9): the spread of the last three values
            spread = max(spread, float(np.max(v[-3:]) - np.min(v[-3:])))
            u = SAFETY_SPREAD * spread
            note = (f"The values go up and down between the mesh levels, and the changes become smaller. "
                    f"The uncertainty is {SAFETY_SPREAD:g} times the spread of the last three values ({pct(spread)}).")
        else:
            u = SAFETY_SPREAD * spread
            estimated = False
            note = (f"The changes between the mesh levels do not become smaller. The values do not converge, so the "
                    f"mesh study gives no error estimate. {SAFETY_SPREAD:g} times the spread of the three values "
                    f"({pct(spread)}) shows their scatter only.")
    if noise is not None and SAFETY_SPREAD * noise > u:
        u = SAFETY_SPREAD * noise
        note += (f"\n\nA move of the line by a quarter of an element changes the value by {pct(noise)}. The uncertainty "
                 f"is {SAFETY_SPREAD:g} times this change, which is more than the estimate from the mesh levels.")
        if not estimated and n >= 3 and float(np.max(v[-3:]) - np.min(v[-3:])) <= noise:
            # the levels scatter inside the band of the moved lines: the scatter is the line noise, and u covers it
            estimated = True
            note += (" The values of the mesh levels stay inside this band, so the line noise explains their scatter. "
                     "The engine uses this uncertainty as the error estimate.")
    status = "converged" if estimated and u <= max_uncertainty * size else "not_converged"
    if abs(fine) < 0.5 * size:
        note += (f"\n\nThe value is small against the size of its field ({size:.4g}). The uncertainty is {pct(u)} of "
                 "that size.")
    rel = float(u / abs(fine)) if abs(fine) > 0 else None
    return Study(status, rel, extrapolated=ext, order=order, note=note, grew=grew, kind=kind, absolute=float(u),
                 estimated=estimated)


# --------------------------------------------------------------------------- locations on one mesh


class Sites:
    """Where things are on one mesh: the nodes of features and interfaces, the zone at the held faces,
    and what a peak is near (feature, re-entrant edge, held face)."""

    def __init__(self, mg: MeshGeom, geo: dict, held_groups: list[str], d_excl: float):
        from scipy.spatial import cKDTree

        self.mg, self.geo = mg, geo
        self.features = geo.get("features", {}) or {}
        self.interfaces = geo.get("interfaces", {}) or {}
        self.d_excl = float(d_excl)
        self.held_groups = list(held_groups)
        n = len(mg.x)
        self.dist_held = np.full(n, np.inf)
        if held_groups:
            hn = mg.nodes(held_groups)
            self.dist_held, _ = cKDTree(mg.x[hn]).query(mg.x)
        self.eligible = self.dist_held > self.d_excl             # nodes that a peak evaluation may use
        self._face_groups = sorted((g for g in mg.md.groups if re.fullmatch(r"f\d+", g) and mg.has(g)),
                                   key=lambda g: int(g[1:]))
        self._face_of_node = None
        self._surface_tree = None
        self._edges = None

    # ---- node sets
    def groups_of(self, location: str) -> list[str]:
        """The face groups of a feature or an interface."""
        if location in self.features:
            tags = self.features[location].get("faces") or []
            groups = [f"f{t}" for t in tags] if tags else [location]
        elif location in self.interfaces:
            tags = sorted({int(t) for role in self.interfaces[location].values() for t in (role or [])})
            groups = [f"f{t}" for t in tags]
        else:
            raise CannotServe(f"The location {location} is not the part, a feature, an interface or a stress "
                              "classification line of the geometry.")
        missing = self.mg.missing(groups)
        if missing or not groups:
            raise CannotServe(f"The mesh has no face group for {names(missing) if missing else location}.")
        return groups

    def nodes(self, location: str) -> np.ndarray:
        if location == "part":
            return np.arange(len(self.mg.x))
        return self.mg.nodes(self.groups_of(location))

    # ---- what a point is near
    def _faces(self):
        if self._face_of_node is None:
            from scipy.spatial import cKDTree
            owner: dict[int, list[str]] = {}
            surface = []
            for g in self._face_groups:
                nd = self.mg.nodes([g])
                surface.append(nd)
                for i in nd:
                    owner.setdefault(int(i), []).append(g)
            self._face_of_node = owner
            self._surface_nodes = np.unique(np.concatenate(surface)) if surface else np.zeros(0, dtype=np.int64)
            self._surface_tree = cKDTree(self.mg.x[self._surface_nodes]) if len(self._surface_nodes) else None
        return self._face_of_node

    def _reentrant(self) -> list[dict]:
        if self._edges is None:
            out = []
            for e in (self.geo.get("edges", {}) or {}).get("reentrant", []) or []:
                fa = [f"f{t}" for t in e.get("faces", [])]
                if len(fa) != 2 or self.mg.missing(fa):
                    continue
                nd = np.intersect1d(self.mg.nodes([fa[0]]), self.mg.nodes([fa[1]]))
                if len(nd):
                    out.append({"tag": e.get("tag"), "faces": fa, "nodes": nd, "angle": e.get("angle_deg")})
            self._edges = out
        return self._edges

    def feature_names(self, groups: list[str]) -> list[str]:
        tags = {int(g[1:]) for g in groups}
        out = [fid for fid, f in self.features.items() if tags & {int(t) for t in (f.get("faces") or [])}]
        return out

    def describe(self, node: int) -> dict:
        """What the node is at: {"near": text, "edge": bool, "support": bool, "location": [x, y, z]}."""
        mg = self.mg
        x = mg.x[node]
        h_loc = float(mg.node_h()[node])
        owner = self._faces()
        faces = owner.get(int(node))
        on_surface = faces is not None
        if faces is None and self._surface_tree is not None:
            _, k = self._surface_tree.query(x)
            faces = owner.get(int(self._surface_nodes[k]), [])
        faces = faces or []
        feats = self.feature_names(faces)
        if feats:
            near = " and ".join(feats[:3])
        elif faces:
            near = "face " + " and ".join(faces[:3])
        else:
            near = "the volume of the part"
        if not on_surface and faces:
            near = "in the material, next to " + near
        edge, edge_text = False, ""
        best = (np.inf, None)
        for e in self._reentrant():
            d = float(np.linalg.norm(mg.x[e["nodes"]] - x, axis=1).min())
            if d < best[0]:
                best = (d, e)
        if best[1] is not None and best[0] <= NEAR_EDGE * h_loc:
            e = best[1]
            edge = True
            edge_text = (f"at the re-entrant edge {e['tag']} between {e['faces'][0]} and {e['faces'][1]}"
                         + (f" ({e['angle']:.0f} deg of material)" if isinstance(e.get("angle"), (int, float)) else ""))
        d_held = float(self.dist_held[node])
        support = bool(np.isfinite(d_held) and d_held <= self.d_excl + NEAR_ZONE * h_loc)
        text = near + (", " + edge_text if edge else "")
        if support:
            text += ", at the border of the zone that the engine leaves out at the held faces"
        return {"near": text, "edge": edge, "support": support, "location": [float(c) for c in x], "h": h_loc,
                "features": feats, "faces": faces}


# --------------------------------------------------------------------------- one mesh level


@dataclass
class LevelData:
    """Everything the engine keeps of one solve."""

    level: MeshLevel
    h: float
    values: dict[str, float] = field(default_factory=dict)
    where: dict[str, dict] = field(default_factory=dict)        # peaks: Sites.describe of the peak node
    info: dict[str, dict] = field(default_factory=dict)         # extra numbers of a result (notes)
    failed: dict[str, str] = field(default_factory=dict)        # key -> why the level has no value
    checks: dict[str, dict] = field(default_factory=dict)
    real: Realized | None = None
    warnings: list[str] = field(default_factory=list)
    solver: str = ""
    wall: float = 0.0


def parse_key(key: str) -> tuple[str, str, str]:
    m = re.fullmatch(r"([A-Za-z_0-9]+)\.([A-Za-z_0-9]+)@(.+)", key)
    if not m:
        raise CannotServe("The key is not of the form <quantity>.<measure>@<location>.")
    return m.group(1), m.group(2), m.group(3)


def p_theory_of(key: str) -> float:
    return P_DISPLACEMENT if key.split(".")[0] in ("displacement", "radial_displacement") else P_STRESS


def unit_of(key: str) -> str:
    return "m" if key.split(".")[0] in ("displacement", "radial_displacement") else "Pa"


class Evaluator:
    """The results of one solve."""

    def __init__(self, mg: MeshGeom, sites: Sites, geo: dict, scls: dict[str, dict], U: np.ndarray, S: np.ndarray,
                 sampler: Callable[[np.ndarray], np.ndarray], free: bool):
        self.mg, self.sites, self.geo, self.scls = mg, sites, geo, scls
        self.U, self.S = U, S
        self.sampler = sampler
        self.free = free
        self._sections: dict[str, Any] = {}
        self._shifted: dict[str, list] = {}
        self._inv: dict[str, np.ndarray] = {}
        self._U_rel = None
        self.used_datum = False
        self.used_roundness = False

    # ---- fields at the nodes
    def invariant(self, name: str) -> np.ndarray:
        if name not in self._inv:
            self._inv[name] = scl_mod.invariant(name, self.S)
        return self._inv[name]

    def displacement(self, result: bool = True) -> np.ndarray:
        """The displacement the results use: as solved for a held part; without the mean rigid-body
        motion for a free part. `result`: a displacement result uses it (the datum is then a decision)."""
        if not self.free:
            return self.U
        self.used_datum = self.used_datum or result
        if self._U_rel is None:
            mg = self.mg
            vol, c, J = mg.volume()
            xq = np.asarray(mg.basis.global_coordinates()).reshape(3, -1).T
            w = np.asarray(mg.basis.dx).ravel()
            uq = np.column_stack([np.asarray(mg.basis.interpolate(self.U[:, k])).ravel() for k in range(3)])
            a = (w @ uq) / vol
            b = np.linalg.solve(J, w @ np.cross(xq - c, uq))
            self._U_rel = self.U - a - np.cross(b, mg.x - c)
        return self._U_rel

    # ---- stress classification lines
    def section(self, sid: str):
        if sid not in self._sections:
            s = self.scls[sid]
            if not s.get("ok", True):
                self._sections[sid] = CannotServe(f"The geometry stage says that the line {sid} is not valid: "
                                                  f"{s.get('message') or 'no reason given'}")
            else:
                try:
                    self._sections[sid] = scl_mod.section(self.sampler, s["start"], s["end"])
                    self._shifted[sid] = self._moved_lines(np.asarray(s["start"], float), np.asarray(s["end"], float))
                except (ValueError, CannotServe) as e:
                    self._sections[sid] = CannotServe(f"The line {sid} has no result: {e}")
        sec = self._sections[sid]
        if isinstance(sec, Exception):
            raise sec
        return sec

    def _moved_lines(self, a: np.ndarray, b: np.ndarray) -> list:
        """The sections of the line moved by LINE_SHIFT of the local element size, in the two directions
        across it (four lines). A moved line that leaves the material is left out."""
        mg = self.mg
        d = (b - a) / np.linalg.norm(b - a)
        e1 = np.cross(d, np.eye(3)[int(np.argmin(np.abs(d)))])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        mid = 0.5 * (a + b)
        h = float(mg.node_h()[int(np.argmin(np.einsum("ij,ij->i", mg.x - mid, mg.x - mid)))])
        out = []
        for e in (e1, -e1, e2, -e2):
            try:
                out.append(scl_mod.section(self.sampler, a + LINE_SHIFT * h * e, b + LINE_SHIFT * h * e))
            except (ValueError, CannotServe):
                continue
        return out

    # ---- one key
    def value(self, key: str) -> tuple[float, dict | None, dict]:
        """(value, where a peak is or None, extra numbers) of one result key on this mesh."""
        quantity, measure, location = parse_key(key)
        if location in self.scls:
            if quantity not in COMPONENTS and quantity not in scl_mod.INVARIANTS:
                raise CannotServe(f"A stress classification line has no quantity {quantity}.")
            if measure not in ("membrane", "membrane_bending"):
                raise CannotServe(f"A stress classification line has no measure {measure}.")
            sec = self.section(location)
            v = sec.value(quantity, measure)
            others = [o.value(quantity, measure) for o in self._shifted.get(location, [])]
            extra = {"noise": float(max([v, *others]) - min([v, *others])), "moved": len(others)} if len(others) >= 2 else {}
            if measure == "membrane_bending" and quantity in scl_mod.INVARIANTS:
                extra["end"] = sec.governing_end(quantity)
            if quantity != "von_mises":          # a component or a principal stress can be small in a large stress state
                states = [sec.membrane] if measure == "membrane" else [sec.at_start(), sec.at_end()]
                extra["ref"] = float(max(np.abs(scl_mod.principals(t)).max() for t in states))
            return v, None, extra
        if location == "scls":
            raise CannotServe("The engine gives the measures membrane and membrane_bending of von_mises for all "
                              "lines together.")
        if quantity in STRESS_QUANTITIES:
            if measure not in ("peak", "max"):
                raise CannotServe(f"The engine gives the measure peak of {quantity} on the part, a feature or an "
                                  f"interface, and membrane or membrane_bending on a stress classification line.")
            nodes = self.sites.nodes(location)
            nodes = nodes[self.sites.eligible[nodes]]
            if not len(nodes):
                raise CannotServe(f"All nodes of {location} are in the zone that the engine leaves out at the held faces.")
            f = self.invariant(quantity)
            i = int(nodes[np.argmax(f[nodes])])
            return float(f[i]), self.sites.describe(i), {}
        if quantity == "displacement":
            if measure not in ("max", "min", "mean", "peak"):
                raise CannotServe(f"The engine gives the measures max, min and mean of the displacement, not {measure}.")
            um = np.linalg.norm(self.displacement(), axis=1)
            if measure == "mean":
                if location == "part":
                    w = np.asarray(self.mg.basis.dx).ravel()
                    uq = np.column_stack([np.asarray(self.mg.basis.interpolate(self.displacement()[:, k])).ravel()
                                          for k in range(3)])
                    return float(w @ np.linalg.norm(uq, axis=1) / w.sum()), None, {}
                q = self.mg.quad(self.sites.groups_of(location))
                uq = np.column_stack([np.asarray(q.basis.interpolate(self.displacement()[:, k])).ravel() for k in range(3)])
                return float(q.w @ np.linalg.norm(uq, axis=1) / q.area), None, {}
            nodes = self.sites.nodes(location)
            i = int(nodes[np.argmax(um[nodes]) if measure in ("max", "peak") else np.argmin(um[nodes])])
            return float(um[i]), None, {"location": [float(c) for c in self.mg.x[i]]}
        if quantity == "radial_displacement":
            return self._radial(measure, location)
        raise CannotServe(f"The structural engine has no result for the quantity {quantity}.")

    def _radial(self, measure: str, location: str) -> tuple[float, dict | None, dict]:
        f = self.sites.features.get(location)
        if f is None:
            raise CannotServe(f"The radial displacement needs a cylindrical feature, and {location} is not a feature.")
        axis = f.get("axis")
        if axis is None or f.get("kind") not in (None, "cylinder"):
            raise CannotServe(f"The feature {location} is not a cylinder with an axis in the geometry.")
        if measure not in ("range", "max", "min", "mean"):
            raise CannotServe(f"The engine gives the measures range, max, min and mean of the radial displacement, "
                              f"not {measure}.")
        mg = self.mg
        d = np.asarray(axis, dtype=float)
        d = d / np.linalg.norm(d)
        e1 = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        groups = self.sites.groups_of(location)
        q = mg.quad(groups)
        # the axis position: least-squares circle through the points of the feature, in the plane across the axis
        a, b = q.x @ e1, q.x @ e2
        A = np.column_stack([2 * a, 2 * b, np.ones_like(a)]) * np.sqrt(q.w)[:, None]
        sol, *_ = np.linalg.lstsq(A, (a * a + b * b) * np.sqrt(q.w), rcond=None)
        radius = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0)))
        r_geo = f.get("radius")
        if r_geo and abs(radius - r_geo) > 0.02 * r_geo:
            raise CannotServe(f"The nodes of {location} are not on a cylinder with the radius of the geometry "
                              f"({fmt(r_geo, 'm')}; the fit gives {fmt(radius, 'm')}).")
        c0 = sol[0] * e1 + sol[1] * e2

        def polar(x):
            r = x - c0
            pa, pb = r @ e1, r @ e2
            rr = np.maximum(np.hypot(pa, pb), 1e-300)
            return pa / rr, pb / rr, r @ d

        U = self.U                                          # the datum does not matter: the fit removes rigid motion
        nodes = mg.nodes(groups)
        cn, sn, zn = polar(mg.x[nodes])
        ur = (U[nodes] @ e1) * cn + (U[nodes] @ e2) * sn
        if measure in ("max", "min", "mean"):
            if self.free:
                raise CannotServe("The radial displacement about a fixed axis has no meaning for a free part. "
                                  "The measure range does not need a datum.")
            if measure == "mean":
                cq, sq, _ = polar(q.x)
                uq = np.column_stack([np.asarray(q.basis.interpolate(U[:, k])).ravel() for k in range(3)])
                return float(q.w @ ((uq @ e1) * cq + (uq @ e2) * sq) / q.area), None, {}
            return float(ur.max() if measure == "max" else ur.min()), None, {}
        # range: remove the rigid-body motion of the feature (shift across the axis and tilt), fitted on the faces
        cq, sq, zq = polar(q.x)
        z0 = float(q.w @ zq / q.area)
        uq = np.column_stack([np.asarray(q.basis.interpolate(U[:, k])).ravel() for k in range(3)])
        urq = (uq @ e1) * cq + (uq @ e2) * sq
        B = np.column_stack([np.ones_like(cq), cq, sq, cq * (zq - z0), sq * (zq - z0)])
        sw = np.sqrt(q.w)[:, None]
        coef, _, rank, sv = np.linalg.lstsq(B * sw, urq * sw[:, 0], rcond=None)
        fixed = float(ur.max() - ur.min())
        if rank < 5 or sv[-1] < 1e-6 * sv[0]:
            raise CannotServe(f"The faces of {location} cover too small a part of the cylinder to separate its "
                              "rigid-body motion from its change of shape.")
        rigid = coef[1] * cn + coef[2] * sn + coef[3] * cn * (zn - z0) + coef[4] * sn * (zn - z0)
        form = ur - rigid
        self.used_roundness = True
        ref = float(np.linalg.norm(self.displacement(result=False)[nodes], axis=1).max())
        return float(form.max() - form.min()), None, {"fixed_axis": fixed, "shift": float(np.hypot(coef[1], coef[2])),
                                                      "growth": float(coef[0]), "ref": ref}


# --------------------------------------------------------------------------- the engine


def run(project, ctx, geo: dict, lc: LoadCase, workdir: Path, opts):
    """Analyse one static or thermo-structural load case. Returns pipeline.Analysis(record, journal)."""
    from dossier.pipeline import Analysis
    try:
        from dossier import meshing
    except ModuleNotFoundError as e:
        if e.name != "dossier.meshing":
            raise
        j = Journal("idealize", "engine:structural")
        return Analysis(_record(lc, "not_supported", "The mesh engine dossier.meshing is not in this version."), j)
    levels = meshing.levels(project, geo, opts)
    return run_levels(project, ctx, geo, lc, levels, workdir, opts)


def _record(lc: LoadCase, status: str, message: str, **kw) -> AnalysisRecord:
    return AnalysisRecord(id=analysis_id(lc), title=lc.title, load_case=lc.id, analysis_type=lc.analysis_type,
                          status=status, message=message, origin=Origin(by="engine:structural", stage="analyze"), **kw)


def _thermal_case(project, lc: LoadCase) -> LoadCase | None:
    cases = {c.id: c for c in project.collection("load_cases")}
    for u in lc.uses:
        c = cases.get(u)
        if c is not None and c.analysis_type == "thermal_steady":
            return c
    return None


def _nodal_temperature(mg: MeshGeom, th: dict) -> np.ndarray:
    """The temperature of the thermal engine at the nodes of `mg`, in the order of `mg.x`."""
    from scipy.spatial import cKDTree
    x, T = np.asarray(th["x"], dtype=float), np.asarray(th["T"], dtype=float).ravel()
    if x.shape == mg.x.shape and np.abs(x - mg.x).max() <= 1e-9 * mg.diag:
        return T
    d, i = cKDTree(x).query(mg.x)
    if d.max() > 1e-6 * mg.diag:
        raise ValueError(f"the temperature field of the thermal engine is not on the nodes of the mesh "
                         f"(the largest distance is {d.max():.3g} m)")
    return T[i]


def run_levels(project, ctx, geo: dict, lc: LoadCase, levels: list[MeshLevel], workdir: Path, opts,
               journal: Journal | None = None, thermal_solver: Callable | None = None):
    """The analysis on given mesh levels (coarse to fine). `thermal_solver` replaces
    dossier.analysis.thermal.solve_temperature (tests)."""
    from dossier.pipeline import Analysis

    t0 = time.time()
    j = journal or Journal("idealize", "engine:structural")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    an_id = analysis_id(lc)
    log = getattr(opts, "log", None) or (lambda s: None)
    if lc.analysis_type not in ("static_structural", "thermo_structural"):
        return Analysis(_record(lc, "not_supported", f"The structural engine does not analyse the type {lc.analysis_type}."), j)
    if not levels:
        return Analysis(_record(lc, "failed", "The mesh engine gave no mesh level."), j)
    thermal = lc.analysis_type == "thermo_structural"
    th_lc = None
    if thermal:
        th_lc = _thermal_case(project, lc)
        if th_lc is None:
            return Analysis(_record(lc, "not_supported",
                                    "The thermo-structural load case lists no thermal load case in `uses`, or the plan "
                                    "does not have it. The engine needs the temperature field of a thermal load case."), j)
        if thermal_solver is None:
            try:
                from dossier.analysis import thermal as thermal_mod
                thermal_solver = thermal_mod.solve_temperature
            except (ModuleNotFoundError, AttributeError):
                return Analysis(_record(lc, "not_supported",
                                        "The thermal engine dossier.analysis.thermal is not in this version. The "
                                        "thermo-structural analysis needs its temperature field."), j)
    th_an = analysis_id(th_lc) if th_lc is not None else None
    try:
        ideal = Idealizer(project, geo, lc, j, thermal=thermal, thermal_analysis=th_an)
    except NotSupported as e:
        j.find(rule="structural.not_idealized", severity="major", category="model", subject=lc.id, key="not-idealized",
               title="The structural engine cannot make a model of the load case",
               statement=f"The load case {lc.id} has a load or a support without a model in the structural engine.",
               rationale=str(e), suggestion="Give each load the features of its role, or change the load in the system "
                                            "context to a kind that the engine has a rule for.",
               by="rule:structural.not_idealized")
        return Analysis(_record(lc, "not_supported", str(e)), j)

    comp = ideal.comp
    scls = _scls(comp, geo)
    keys, cannot = _keys(lc, scls)
    h0, d_excl = None, 0.0
    data: list[LevelData] = []
    fields: list[Artifact] = []
    artifacts: list[Artifact] = []
    picture_data = None
    n_lev = len(levels)
    for k, lev in enumerate(levels):
        mesh_path = (Path(project.out) / lev.path) if not Path(lev.path).is_absolute() else Path(lev.path)
        mg = MeshGeom(mesh_path)
        T = field_path = None
        if thermal:
            th = thermal_solver(project, ctx, geo, th_lc, lev, opts)
            T = _nodal_temperature(mg, th)
            field_path = th["path"]
        try:
            real = ideal.realize(mg, T, field_path)
        except NotSupported as e:
            return Analysis(_record(lc, "not_supported", str(e)), j)
        except Unbalanced as e:
            c = e.check
            return Analysis(_record(lc, "failed", str(e), checks=[
                Check(id=make_id("CHK", an_id, "balance"), kind=c["kind"], statement=c["statement"], passed=False,
                      value=float(c["value"]), limit=float(c["limit"]))]), j)
        res, case = _solve(ideal, real, mesh_path, lc, lev)
        (workdir / f"case_L{lev.level}.json").write_text(json.dumps(case.to_dict(), indent=1))
        artifacts.append(Artifact(path=_rel(project, workdir / f"case_L{lev.level}.json"), kind="data",
                                  caption=f"The solver input of mesh level {lev.level}."))
        U = np.column_stack([res.nodal(c)[1] for c in ("ux", "uy", "uz")])
        S = np.column_stack([res.nodal(c)[1] for c in COMPONENTS])
        X = res.nodal("ux")[0]
        if X.shape != mg.x.shape or np.abs(X - mg.x).max() > 1e-12 * max(mg.diag, 1.0):
            raise RuntimeError("the nodes of the solution are not the nodes of the mesh reader")
        if h0 is None:
            h0 = float(res.info.h_rep)
            d_excl = EXCLUSION * h0 if ideal.held else 0.0
        sites = Sites(mg, geo, real.held_groups, d_excl)
        ev = Evaluator(mg, sites, geo, scls, U, S, _sampler(res, mg), ideal.free)
        ld = LevelData(level=lev.model_copy(update={"nodes": int(res.info.n_nodes), "cells": int(res.info.n_cells),
                                                    "dofs": int(res.info.n_dofs), "h_rep": float(res.info.h_rep),
                                                    "order": int(res.info.order),
                                                    "min_quality": lev.min_quality if lev.min_quality is not None
                                                    else mg.min_quality()}),
                       h=float(res.info.h_rep), real=real, wall=float(res.info.wall_time))
        ld.warnings = [w for w in res.info.warnings if not w.startswith("info:")]
        ld.solver = next((w[6:] for w in res.info.warnings if w.startswith("info: linear_solver")), "")
        for key in keys:
            try:
                v, where, extra = ev.value(key)
                ld.values[key] = v
                if where is not None:
                    ld.where[key] = where
                if extra:
                    ld.info[key] = extra
            except CannotServe as e:
                ld.failed[key] = str(e)
        ld.info["_flags"] = {"datum": ev.used_datum, "roundness": ev.used_roundness}
        ld.checks = dict(real.checks)
        ld.checks.update(_solve_checks(res, mg, real, ideal, U, last=(k == n_lev - 1)))
        if k >= n_lev - 2:
            path = workdir / f"fields_L{lev.level}.npz"
            np.savez_compressed(path, x=mg.x, stress=S, u=U)
            fields.append(Artifact(path=_rel(project, path), kind="field",
                                   caption=f"Stress and displacement at the nodes of mesh level {lev.level} "
                                           f"({len(mg.x)} nodes)."))
        if k == n_lev - 1 and getattr(opts, "pictures", False):
            picture_data = _picture_data(mg, ideal, real, S, U)
        log(f"  {lc.id} level {lev.level}: {res.info.n_nodes} nodes, {res.info.wall_time:.1f} s")
        data.append(ld)
        del res, ev, sites, mg, U, S

    max_unc = float(project.policy.max_uncertainty)
    results, notes_missing = _results(data, keys, scls, max_unc, lc)
    for key, why in cannot.items():
        notes_missing[key] = why
    _late_decisions(j, lc, ideal, data, d_excl, h0, max_unc, results)
    checks = _checks(an_id, data, ideal)
    if picture_data is not None:
        artifacts += _pictures(project, workdir, lc, an_id, data[-1], picture_data, results, log)
    rec = _record(
        lc, "done", _message(notes_missing, data),
        statement=_statement(lc, ideal, data, results, th_an, len(scls)),
        solver=_solver_text(data[-1]),
        material={"id": ideal.material.id, "name": ideal.material.name, "E": ideal.E, "nu": ideal.nu, "rho": ideal.rho,
                  "alpha": ideal.alpha, "T_ref": ideal.T_ref},
        bcs=ideal.record_items(data[-1].real), mesh=[d.level for d in data], results=results, checks=checks,
        artifacts=[*fields, *artifacts],
        links=[Link(rel="uses", to=ideal.material.id),
               *([Link(rel="derives_from", to=th_an, note="temperature field")] if th_an else []),
               *[Link(rel="uses", to=d) for d in dict.fromkeys(ideal.decisions.values())]],
        runtime_s=round(time.time() - t0, 1))
    return Analysis(rec, j)


def _rel(project, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(project.out).resolve()))
    except ValueError:
        return str(path)


def _scls(comp, geo: dict) -> dict[str, dict]:
    """The stress classification lines: the geometry stage's check of each line where it has one."""
    out = {}
    g = geo.get("scls", {}) or {}
    for s in comp.scls:
        e = dict(g.get(s.id) or {})
        out[s.id] = {"start": list(e.get("start") or s.start), "end": list(e.get("end") or s.end),
                     "ok": bool(e.get("ok", True)), "message": e.get("message", "")}
    return out


def _keys(lc: LoadCase, scls: dict) -> tuple[list[str], dict[str, str]]:
    """The result keys the engine evaluates on every level, and the keys it cannot give (with the reason)."""
    keys = ["von_mises.peak@part"]
    for sid in scls:
        keys += [f"{q}.{m}@{sid}" for q in STRESS_QUANTITIES for m in ("membrane", "membrane_bending")]
        keys += [f"{c}.membrane@{sid}" for c in ("sxx", "syy", "szz")]
    aggregates = ["von_mises.membrane@scls", "von_mises.membrane_bending@scls"]
    cannot: dict[str, str] = {}
    for key in lc.result_keys:
        if key in keys or key in aggregates:
            continue
        try:
            q, m, loc = parse_key(key)
        except CannotServe as e:
            cannot[key] = str(e)
            continue
        if loc == "scls":
            if q in STRESS_QUANTITIES and m in ("membrane", "membrane_bending"):
                aggregates.append(key)
                keys += [f"{q}.{m}@{sid}" for sid in scls if f"{q}.{m}@{sid}" not in keys]
            else:
                cannot[key] = ("The engine gives the measures membrane and membrane_bending of von_mises and "
                               "max_principal for all lines together.")
            continue
        keys.append(key)
    return list(dict.fromkeys(keys)), cannot


def _solve(ideal: Idealizer, real: Realized, mesh_path: Path, lc: LoadCase, lev: MeshLevel):
    from feacore.case import Analysis as FeAnalysis
    from feacore.case import Case, Material
    from feacore.solver import solve
    case = Case(mesh=str(mesh_path),
                analysis=FeAnalysis(physics="elasticity", formulation="3d", order=2, T_ref=ideal.T_ref),
                materials={"material": Material(E=ideal.E, nu=ideal.nu, alpha=ideal.alpha or 0.0)},
                bcs=list(real.bcs), loads=list(real.loads),
                meta={"load_case": lc.id, "analysis": analysis_id(lc), "level": lev.level})
    return solve(case, backend="skfem"), case


def _sampler(res, mg: MeshGeom) -> Callable[[np.ndarray], np.ndarray]:
    """The six stress components at points, from the solver's `probe`. A point outside the mesh by more
    than SCL_OUTSIDE of the line length means that the line leaves the material."""
    from feacore.solver._skfem_locate import Locator
    locator = Locator(mg.mesh, mg.diag)

    def sample(pts: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(pts[-1] - pts[0])) or mg.diag
        _, _, dist = locator.locate(pts, tol_out=np.inf)
        if dist.max() > SCL_OUTSIDE * length:
            i = int(np.argmax(dist))
            raise CannotServe(f"the line leaves the material near {fmt(pts[i], 'm')} (a sample point is "
                              f"{fmt(dist.max(), 'm')} outside the mesh)")
        return np.column_stack([res.probe(c, pts) for c in COMPONENTS])

    return sample


# --------------------------------------------------------------------------- checks


def _solve_checks(res, mg: MeshGeom, real: Realized, ideal: Idealizer, U: np.ndarray, last: bool) -> dict[str, dict]:
    """The checks that need the solution of one level."""
    out: dict[str, dict] = {}
    split = res.model.split
    n_dof = len(res.f)
    fn = np.column_stack([res.f[split[k]] for k in range(3)])
    constrained = np.zeros(n_dof, dtype=bool)
    constrained[res.dirichlet_dofs] = True
    rn = np.column_stack([np.where(constrained[split[k]], res.reaction_vector[split[k]], 0.0) for k in range(3)])
    vol, c_v, _ = mg.volume()
    fs, ms = real.force_scale(), real.moment_scale(c_v)

    def about_c(force, moment_o):
        return moment_o - np.cross(c_v, force)

    # the loads the solver assembled against the loads of the load case (thermal loads have no resultant)
    F_s, M_s = fn.sum(axis=0), np.cross(mg.x, fn).sum(axis=0)
    F_i, M_i = real.force, real.moment
    if fs > 0:
        err = max(float(np.linalg.norm(F_s - F_i) / fs), float(np.linalg.norm(about_c(F_s - F_i, M_s - M_i)) / ms))
        out["assembly"] = {
            "kind": "load_resultant", "value": err, "limit": ASSEMBLY_TOL, "passed": bool(err <= ASSEMBLY_TOL),
            "statement": f"The solver assembled a total load of {fmt(F_s, 'N')}. It assembled a moment of "
                         f"{fmt(about_c(F_s, M_s), 'N*m')} about the centre of volume. The loads of the load case give "
                         f"{fmt(F_i, 'N')} and {fmt(about_c(F_i, M_i), 'N*m')}."}
    # equilibrium: the reactions of the supports against the applied loads
    R, M_R = rn.sum(axis=0), np.cross(mg.x, rn).sum(axis=0)
    r_abs = float(np.abs(rn).sum())
    # the size of the forces: the largest load, the reactions, and the nodal loads (thermal loads have no resultant)
    scale_f = max(fs, 0.01 * r_abs, 1e-3 * float(np.abs(fn).sum()), 1e-300)
    scale_m = max(ms, scale_f * 0.5 * mg.diag)
    err = max(float(np.linalg.norm(R + F_i) / scale_f), float(np.linalg.norm(about_c(R + F_i, M_R + M_i)) / scale_m))
    out["equilibrium"] = {
        "kind": "equilibrium", "value": err, "limit": EQUILIBRIUM_TOL, "passed": bool(err <= EQUILIBRIUM_TOL),
        "statement": f"The reactions of the supports give a force of {fmt(R, 'N')}. "
                     f"They give a moment of {fmt(about_c(R, M_R), 'N*m')} about the centre of volume. The applied loads give "
                     f"{fmt(F_i, 'N')} and {fmt(about_c(F_i, M_i), 'N*m')}. The sum must be zero."}
    if ideal.free and real.restraints:
        worst = max(float(np.abs(rn[r["node"]]).max()) for r in real.restraints)
        # the solver holds the node nearest to each point: that is the node of the restraint
        rel = worst / fs if fs > 0 else (0.0 if worst <= 1e-9 * max(r_abs, 1e-300) else math.inf)
        out["restraint"] = {
            "kind": "reaction", "value": rel, "limit": REACTION_TOL, "passed": bool(rel <= REACTION_TOL),
            "statement": f"The largest reaction at the three restraint nodes of the free part is {fmt(worst, 'N')}. "
                         f"The largest load is {fmt(fs, 'N')}."}
    um = float(np.linalg.norm(U, axis=1).max())
    ratio = um / mg.diag
    out["small-displacement"] = {
        "kind": "validity", "value": ratio, "limit": DISPLACEMENT_LIMIT, "passed": bool(ratio <= DISPLACEMENT_LIMIT),
        "statement": f"The largest displacement is {fmt(um, 'm')}, which is {100 * ratio:.2g} % of the size of the part "
                     f"({fmt(mg.diag, 'm')}). A linear analysis needs small displacements."}
    q = mg.min_quality()
    bad = [w for w in res.info.warnings if "Jacobian" in w or "degenerate" in w]
    out["mesh-quality"] = {
        "kind": "mesh_quality", "value": q, "limit": QUALITY_LIMIT, "passed": bool(q >= QUALITY_LIMIT and not bad),
        "lower_is_worse": True,
        "statement": f"The smallest element quality is {q:.2f} (mean ratio of the corner tetrahedron). "
                     f"A regular element has the value 1." + (" The solver reports: " + ", ".join(bad) + "." if bad else
                                     " No element has a negative Jacobian.")}
    if last:
        try:
            from feacore.diagnostics.view import SolutionView
            from feacore.diagnostics.zz import zz_estimate
            zz = zz_estimate(SolutionView.from_result(res))
            eta = float(zz["eta_rel"])
            out["error-estimate"] = {
                # information only (CONTRACTS.md): no limit, passed=None, so it does not block a pass
                "kind": "error_estimate", "value": eta, "limit": None, "passed": None,
                "statement": f"The estimated relative error of the finest mesh in the energy norm is {100 * eta:.2g} % "
                             f"(Zienkiewicz-Zhu). The cells with the largest error hold "
                             f"{100 * float(zz['concentration_top1pct']):.2g} % of it in 1 % of the cells, near "
                             f"{fmt(zz['worst_cell_at'], 'm')}. This value is information only and has no limit: the "
                             f"convergence study gives the uncertainty of each result."
                             + (f" The value is more than {100 * ZZ_LIMIT:.0f} %, the limit of the solver's own "
                                f"diagnostics. Examine the mesh near this location." if eta > ZZ_LIMIT else "")}
        except Exception as e:  # noqa: BLE001  an estimate that fails is reported, it does not stop the analysis
            out["error-estimate"] = {"kind": "error_estimate", "value": None, "limit": None, "passed": None,
                                     "statement": f"The error estimate failed: {type(e).__name__}: {e}"}
    return out


def _checks(an_id: str, data: list[LevelData], ideal: Idealizer | None) -> list[Check]:
    """One Check per kind of check: the worst level of each."""
    worst: dict[str, tuple[dict, int]] = {}
    for d in data:
        for key, c in d.checks.items():
            cur = worst.get(key)
            if cur is None or _badness(c) >= _badness(cur[0]):
                worst[key] = (c, d.level.level)
    order = ["balance", "restraint", "assembly", "equilibrium", "support-temperature", "small-displacement",
             "mesh-quality", "error-estimate"]
    keys = [k for k in order[:2] if k in worst] + [k for k in worst if k not in order] + [k for k in order[2:] if k in worst]
    out = []
    many = len(data) > 1
    for key in keys:
        c, level = worst[key]
        text = c["statement"]
        if many and key != "error-estimate":
            text += f" This is the mesh level {level}, the worst of {len(data)} levels."
        out.append(Check(id=make_id("CHK", an_id, key), kind=c["kind"], statement=text, passed=c.get("passed"),
                         value=None if c.get("value") is None else float(c["value"]),
                         limit=None if c.get("limit") is None else float(c["limit"])))
    return out


def _badness(c: dict) -> float:
    if c.get("passed") is None or c.get("value") is None:
        return math.inf
    v, lim = float(c["value"]), float(c.get("limit") or 1.0)
    if c.get("lower_is_worse"):
        return lim / max(v, 1e-300)
    return v / max(lim, 1e-300)


# --------------------------------------------------------------------------- results


def _results(data: list[LevelData], keys: list[str], scls: dict, max_unc: float,
             lc: LoadCase) -> tuple[dict[str, ResultValue], dict[str, str]]:
    h = [d.h for d in data]
    fine = data[-1]
    out: dict[str, ResultValue] = {}
    missing: dict[str, str] = {}
    judged = _judged(lc, scls)
    for key in keys:
        if key in fine.failed or key not in fine.values:
            missing[key] = fine.failed.get(key, "The finest mesh has no value.")
            continue
        have = [d for d in data if key in d.values]
        vals = [d.values[key] for d in have]
        ref = fine.info.get(key, {}).get("ref")
        if key in judged:       # a criterion or a hand calculation uses the value: judge u against the value itself
            ref = None
        st = study([d.h for d in have], vals, p_theory_of(key), max_unc, reference=ref,
                   noise=fine.info.get(key, {}).get("noise"))
        note = st.note
        if key in judged and fine.info.get(key, {}).get("ref") is not None and st.status != "converged":
            note += ("\n\nA criterion or a hand calculation uses this result. Thus the engine compares the uncertainty "
                     "with the value itself, not with the size of its field.")
        if len(have) < len(data):
            note += f" {len(data) - len(have)} coarser mesh levels have no value for this result."
        rv = ResultValue(key=key, value=float(vals[-1]), unit=unit_of(key), uncertainty=st.uncertainty,
                         uncertainty_abs=st.absolute, status=st.status, per_level=[float(v) for v in vals],
                         extrapolated=st.extrapolated, order=st.order, method="fe", error_estimated=st.estimated)
        where = fine.where.get(key)
        extra = fine.info.get(key, {})
        if st.grew and st.kind in GROWS_WITHOUT_LIMIT and _measure(key) == "peak":
            # a peak that rose at every refinement and does not converge: singular, wherever it is (the list of
            # re-entrant edges can miss an edge)
            rv.status, rv.error_estimated = "singular", False
            rv.uncertainty = rv.uncertainty_abs = rv.extrapolated = rv.order = None
            note += ("\n\nThe peak grew at each refinement and the values do not converge. "
                     "The engine gives it the status singular and no uncertainty.")
        if where is not None:
            rv.location, rv.near = where["location"], where["near"]
            if (where["edge"] or where["support"]) and rv.status != "singular":
                rv.status = "singular" if st.grew else "not_converged"
                rv.uncertainty = rv.uncertainty_abs = rv.extrapolated = None   # no limit, so no uncertainty of a limit
                rv.order, rv.error_estimated = None, False
                why = ("a sharp re-entrant edge" if where["edge"] else "the rim of a rigid support")
                note += (f"\n\nThe peak is at {why}, where the stress has no limit in linear theory. "
                         + ("The value grew at each refinement, so it is singular." if st.grew else
                            "The value did not grow at each refinement, but a finer mesh can make it grow. "
                            "The engine does not call it converged."))
            moved = _moved(have, key)
            if moved:
                note += f" {moved}"
        elif "location" in extra:
            rv.location = extra["location"]
        if "end" in extra:
            note += f" The larger value is at the {extra['end']} of the line."
        if "fixed_axis" in extra:
            note += (f" The rigid-body motion of the feature is not in this value. The range about the fixed axis is "
                     f"{fmt(extra['fixed_axis'], 'm')}. It includes a shift of the feature of {fmt(extra['shift'], 'm')}.")
        rv.note = note.strip()
        out[key] = rv
    # the worst line
    for key in [k for k in ["von_mises.membrane@scls", "von_mises.membrane_bending@scls", *lc.result_keys]
                if k.endswith("@scls")]:
        if key in out:
            continue
        try:
            q, m, _ = parse_key(key)
        except CannotServe:
            continue
        if q not in STRESS_QUANTITIES or m not in ("membrane", "membrane_bending"):
            continue
        lines = {sid: out[f"{q}.{m}@{sid}"] for sid in scls if f"{q}.{m}@{sid}" in out}
        lost = [sid for sid in scls if sid not in lines]
        if not lines:
            missing[key] = "No stress classification line has a result." if scls else \
                "The component has no stress classification line."
            continue
        out[key] = _aggregate(key, lines, lost)
    return out, missing


GROWS_WITHOUT_LIMIT = ("divergent", "oscillatory_divergent", "indeterminate")


def _measure(key: str) -> str:
    head = key.partition("@")[0]
    return head.partition(".")[2]


def _judged(lc: LoadCase, scls: dict) -> set[str]:
    """The result keys that a criterion or a hand-calculation comparison of the load case uses. A worst-line
    key (`@scls`) stands for the key of each line: the compliance stage assesses each line by itself."""
    out = set(lc.result_keys)
    for k in lc.result_keys:
        if k.endswith("@scls"):
            out |= {k[: -len("scls")] + sid for sid in scls}
    return out


def _moved(have: list[LevelData], key: str) -> str:
    if len(have) < 2 or key not in have[-2].where:
        return ""
    a, b = np.asarray(have[-1].where[key]["location"]), np.asarray(have[-2].where[key]["location"])
    d = float(np.linalg.norm(a - b))
    if d > 3.0 * max(have[-2].where[key]["h"], have[-1].where[key]["h"]):
        return (f"The peak is at different places on the two finest meshes ({fmt(d, 'm')} apart), so the values of the "
                f"levels are not the stress of one point.")
    return ""


def _aggregate(key: str, lines: dict[str, ResultValue], lost: list[str]) -> ResultValue:
    rank = {"converged": 0, "estimate": 1, "not_converged": 2, "singular": 3, "not_run": 4}
    worst_id = max(lines, key=lambda s: lines[s].value)
    w = lines[worst_id]

    def cons(r: ResultValue) -> float:
        return r.value + (r.uncertainty_abs or 0.0)

    top = max(cons(r) for r in lines.values())
    unc_abs = None if w.uncertainty_abs is None else max(top - w.value, w.uncertainty_abs)
    unc = unc_abs / abs(w.value) if unc_abs is not None and w.value != 0 else None
    govern = [s for s, r in lines.items() if cons(r) >= w.value or r.uncertainty_abs is None]
    status = max((lines[s].status for s in govern), key=lambda s: rank[s])
    note = f"The worst line is {worst_id}."
    others = [s for s in govern if s != worst_id]
    if others:
        note += f" With their uncertainty, the lines {names(others)} can reach this value."
    if lost:
        status = "not_converged" if rank[status] < rank["not_converged"] else status
        note += f" The lines {names(lost)} have no result, so the value is the worst of the other lines only."
    n = min(len(r.per_level) for r in lines.values())
    per_level = [max(r.per_level[len(r.per_level) - n + i] for r in lines.values()) for i in range(n)]
    estimated = bool(not lost and unc_abs is not None and all(lines[s].error_estimated for s in govern))
    return ResultValue(key=key, value=w.value, unit=w.unit, uncertainty=unc, uncertainty_abs=unc_abs, status=status,
                       near=worst_id, error_estimated=estimated,
                       per_level=per_level, extrapolated=w.extrapolated, order=w.order, method="fe", note=note)


def _message(missing: dict[str, str], data: list[LevelData]) -> str:
    parts = []
    if missing:
        parts.append("The engine cannot give these results: "
                     + " ".join(f"{k} ({why.rstrip('.')})." for k, why in missing.items()))
    warn = [w for w in dict.fromkeys(w for d in data for w in d.warnings)]
    if warn:
        parts.append("The solver reports: " + " ".join(w.rstrip(".") + "." for w in warn[:6]))
    return " ".join(parts)


# --------------------------------------------------------------------------- text of the record


def _solver_text(fine: LevelData) -> str:
    try:
        import skfem
        version = f"scikit-fem {skfem.__version__}"
    except Exception:  # noqa: BLE001
        version = "scikit-fem"
    return (f"feacore fixed solver, backend skfem ({version}): linear elasticity, quadratic tetrahedra, "
            f"stress recovered by L2 projection" + (f"; {fine.solver}" if fine.solver else ""))


def _statement(lc: LoadCase, ideal: Idealizer, data: list[LevelData], results: dict[str, ResultValue],
               th_an: str | None, n_scl: int) -> str:
    fine = data[-1]
    kinds: dict[str, int] = {}
    for ld in ideal.loads:
        kinds[ld.kind] = kinds.get(ld.kind, 0) + 1
    word = {"pressure": ("pressure load", "pressure loads"), "force": ("force", "forces"),
            "moment": ("moment", "moments"), "acceleration": ("acceleration", "accelerations")}
    parts = [f"{n} {word[k][n > 1]}" for k, n in kinds.items()]
    loads = (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else "".join(parts)) or "no mechanical load"
    s = [f"The engine analysed the load case {lc.id} with the finite element method.",
         f"The model is linear elastic with the material {ideal.material.id}.",
         f"The load case has {loads}."]
    if ideal.held:
        s.append(f"The seat faces of {names(h.interface for h in ideal.held)} hold the part.")
    else:
        s.append("The part is free, and three nodes stop its rigid-body motion.")
    if th_an:
        s.append(f"The temperature field comes from {th_an} on the same meshes.")
    elif ideal.thermal_effects:
        s.append(f"The load case has {len(ideal.thermal_effects)} temperature effects. The plan gives no thermal load "
                 "case for them, and the engine applies no thermal strain.")
    if ideal.other_effects:
        s.append(f"The effects {names(ideal.other_effects)} are not mechanical loads, and the engine does not apply them.")
    s.append(f"\n\nThe study has {len(data)} mesh levels of quadratic tetrahedra. The finest mesh has "
             f"{fine.level.nodes} nodes and an element size of {fmt(fine.h, 'm')}.")
    if n_scl:
        s.append(f"Each of the {n_scl} stress classification lines has {scl_mod.PANELS * scl_mod.ORDER} sample points.")
    count: dict[str, int] = {}
    for r in results.values():
        count[r.status] = count.get(r.status, 0) + 1
    order = ("converged", "not_converged", "singular")
    text = {"converged": "are converged", "not_converged": "are not converged", "singular": "are singular"}
    text1 = {"converged": "is converged", "not_converged": "is not converged", "singular": "is singular"}
    s.append(f"The analysis gives {len(results)} results: "
             + ", ".join(f"{count[k]} {(text1 if count[k] == 1 else text)[k]}" for k in order if count.get(k)) + ".")
    return " ".join(s).replace(" \n\n", "\n\n")


def _late_decisions(j: Journal, lc: LoadCase, ideal: Idealizer, data: list[LevelData], d_excl: float, h0: float,
                    max_unc: float, results: dict[str, ResultValue]) -> None:
    """The decisions that need the meshes or the results: the zone at the held faces, the convergence rule,
    the datum of a free part, the roundness."""
    if ideal.held:
        held = [h.interface for h in ideal.held]
        j.decide(
            subject=lc.id, key="peak-zone", title="Zone at the held faces without peak evaluation",
            choice=f"The engine leaves out the nodes within {fmt(d_excl, 'm')} of the held faces of {names(held)} when "
                   "it finds a peak stress.",
            rationale="The rim of a rigid support is a singular line. The stress there grows without limit when the mesh "
                      f"becomes finer. The distance is {EXCLUSION:g} element sizes of the coarsest mesh "
                      f"({fmt(h0, 'm')}), the same for all mesh levels.",
            alternatives=[("Keep all nodes.", "The peak is then an artefact of the rigid support."),
                          ("Leave out a larger zone.", "A larger zone can hide a real stress near the joint.")],
            basis="engineering_judgement", needs_review=True, links=held, by="rule:structural.peak_zone", stage="solve")
    n = len(data)
    rule = ("One mesh level gives no convergence study, so no result is converged." if n == 1 else
            f"With two mesh levels the uncertainty is {SAFETY_TWO_LEVELS:g} times the change between the levels." if n == 2
            else "With three mesh levels the uncertainty is the larger of two grid convergence indexes "
                 f"(safety factors 1.25 and 3). When the values show no order, the uncertainty is "
                 f"{SAFETY_SPREAD:g} times the spread.")
    j.decide(
        subject=lc.id, key="convergence", title="Rule for the convergence status of a result",
        choice=f"The engine calls a result converged when its discretization uncertainty is not more than "
               f"{100 * max_unc:g} %. {rule} A peak at a re-entrant edge or at the rim of a support is not converged.",
        rationale="The policy gives the limit for the uncertainty. Unstructured meshes scatter the observed order, and "
                  "earlier studies with this solver gave an index that was too small with the factor 1.25 only. "
                  "The stress at a sharp re-entrant edge has no limit in linear theory.",
        alternatives=[("Use the grid convergence index with the factor 1.25 only.",
                       "This index was too small on coarse unstructured meshes in earlier studies with this solver."),
                      ("Accept a peak at a re-entrant edge when three meshes show no growth.",
                       "Three meshes can hide the growth of a singular stress.")],
        basis="rule", links=["POLICY"], by="rule:structural.convergence", stage="solve")
    flags = [d.info.get("_flags", {}) for d in data]
    if any(f.get("datum") for f in flags):
        j.decide(
            subject=lc.id, key="displacement-datum", title="Displacement of a free part",
            choice="The engine gives the displacement of the free part relative to its mean rigid-body motion.",
            rationale="A free part has no fixed point, so its displacement has no datum. The engine removes the "
                      "rigid-body motion that fits the displacement of the volume best (least squares).",
            alternatives=[("Give the displacement relative to the three restraint nodes.",
                           "The restraint nodes are at different places on each mesh, and their position has no "
                           "physical meaning.")],
            basis="rule", by="rule:structural.displacement_datum", stage="solve")
    if any(f.get("roundness") for f in flags):
        feats = sorted({k.split("@")[1] for k in results if k.startswith("radial_displacement.range@")})
        j.decide(
            subject=lc.id, key="roundness", title="Out-of-roundness without the rigid-body motion of the feature",
            choice=f"The engine gives the range of the radial displacement of {names(feats)} after it removes the "
                   "rigid-body motion of the feature (a shift across the axis and a tilt).",
            rationale="A rigid shift of a round bore changes the radial displacement about the fixed axis, but the bore "
                      "stays round. The note of the result also gives the range about the fixed axis.",
            alternatives=[("Use the range about the fixed axis of the geometry.",
                           "That value counts a rigid shift of the feature as out-of-roundness."),
                          ("Evaluate each cross section of the feature alone.",
                           "The range over the full feature also contains a taper, so it is the larger value.")],
            basis="engineering_judgement", needs_review=True, links=feats, by="rule:structural.roundness", stage="solve")
    for d in ideal.j.decisions:
        if d.id == ideal.decisions.get("moment"):
            flat = all(a.info.get("planar", True) for a in data[-1].real.applied if a.kind == "moment")
            if not flat and not d.needs_review:
                d.needs_review = True
                d.rationale += (" The faces of a moment in this load case are not in one plane. The field is still "
                                "statically correct, but it is not a classical stress distribution.")


# --------------------------------------------------------------------------- pictures


def _picture_data(mg: MeshGeom, ideal: Idealizer, real: Realized, S: np.ndarray, U: np.ndarray) -> dict:
    return {}


def _pictures(project, workdir: Path, lc: LoadCase, an_id: str, fine: LevelData, pd: dict,
              results: dict[str, ResultValue], log) -> list[Artifact]:
    return []
