"""Quantities of interest (QoIs) evaluated on a submission's own mesh and results.

QoI specs live in a case's hidden ``reference.json``.  Kinds:

  point        {"field": "U"|"S", "component": c, "point": [x,y,z], "frame": {...}}
               value interpolated with the element shape functions at the point
  region       {"field", "component", "region": R, "stat": "mean|max|min|absmax|p99|..."} over R;
               "mean" is area-weighted over the boundary faces lying in R (mesh-independent)
  line         {"field", "component", "start", "end", "n"}: trapezoidal average along a segment
  max          {"field", "component", "exclude": [R...], "percentile": 100} over all nodes (minus exclusions)
  reaction     {"region": R, "component": "x|y|z"} or {"moment": {"point", "axis"}}: sum of reaction forces
  rotation     {"region": R, "point": c, "axis": a}: least-squares rigid rotation of R about the axis [rad]
  frequency    {"mode": n}: n-th eigenfrequency [cycles / time unit]
  buckling     {"mode": n}: n-th buckling load factor
  volume       {}: volume of the solid mesh
  field_error  {"field": "U", "points": file.npz}: relative L2 error vs a reference nodal field

Components: U -> x y z mag | r t a (cylindrical frame) | r (spherical frame)
            S -> xx yy zz xy yz zx mises tresca p1 p2 p3 | rr tt aa (cylindrical) | rr tt (spherical)
Frames: {"type": "cylindrical", "point": [...], "axis": [...]} or {"type": "spherical", "center": [...]}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ccx.dat import read_buckling, read_eigen
from .ccx.frd import FrdResults, principal, von_mises
from .fem.locate import Locator, element_volumes
from .fem.regions import select_nodes

FIELD_BLOCK = {"U": "DISP", "S": "STRESS", "E": "TOSTRAIN", "RF": "FORC"}
_S_INDEX = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "zx": 5}


class QoIError(Exception):
    pass


class ResultView:
    """Results of one solver run, with the helpers QoI extraction needs."""

    def __init__(self, frd: FrdResults, dat_path: str | Path | None = None,
                 deck_nodes: dict[int, tuple[float, float, float]] | None = None):
        self.frd = frd
        self.dat_path = Path(dat_path) if dat_path else None
        self.coords = frd.coords.copy()
        if deck_nodes:  # the .frd keeps 6 significant digits; prefer the deck's coordinates
            for i, nid in enumerate(frd.node_ids):
                xyz = deck_nodes.get(int(nid))
                if xyz is not None:
                    self.coords[i] = xyz
        rowmap = frd.node_row()
        self.elements = {}
        for t, conn in frd.solid_elements().items():
            self.elements[t] = np.vectorize(rowmap.__getitem__, otypes=[np.int64])(conn)
        self._locator = None
        self._scale = float(np.linalg.norm(self.coords.max(0) - self.coords.min(0))) if len(self.coords) else 1.0

    @property
    def locator(self) -> Locator:
        if self._locator is None:
            self._locator = Locator(self.coords, self.elements)
        return self._locator

    def nodal(self, field: str, index: int | None = None) -> np.ndarray:
        block = FIELD_BLOCK.get(field, field)
        return self.frd.field_on_nodes(block, index)

    def volume(self) -> float:
        return float(sum(v.sum() for v in element_volumes(self.coords, self.elements).values()))

    def boundary_faces(self) -> list[np.ndarray]:
        """Boundary faces as node-row arrays: tri6 for tet10, otherwise corner-only tri3/quad4."""
        if getattr(self, "_bfaces", None) is None:
            seen: dict[tuple, list] = {}
            for t, conn in self.elements.items():
                for local in _FACES[t]:
                    f = conn[:, local]
                    for row in f:
                        key = tuple(sorted(row[: (3 if len(local) in (3, 6) else 4)]))
                        seen.setdefault(key, []).append(row)
            self._bfaces = [rows[0] for rows in seen.values() if len(rows) == 1]
        return self._bfaces

    def surface_mean(self, values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        """Area-weighted mean of a nodal scalar over boundary faces whose nodes all lie in ``mask``."""
        num = area = 0.0
        for f in self.boundary_faces():
            if not mask[f].all():
                continue
            X, v = self.coords[f], values[f]
            for N, dN, w in _face_rule(len(f)):
                J = np.cross(dN[0] @ X, dN[1] @ X)
                dA = np.linalg.norm(J) * w
                num += dA * float(N @ v)
                area += dA
        return (num / area if area > 0 else float("nan")), area


# ----------------------------------------------------------------------------- faces

_TET_CORNER_FACES = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
_TET10_MID = {(0, 1): 4, (1, 2): 5, (0, 2): 6, (0, 3): 7, (1, 3): 8, (2, 3): 9}


def _tet10_face(a, b, c):
    m = lambda i, j: _TET10_MID[tuple(sorted((i, j)))]
    return [a, b, c, m(a, b), m(b, c), m(c, a)]


_HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_WEDGE_FACES = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
_FACES = {
    "tet4": [list(f) for f in _TET_CORNER_FACES],
    "tet10": [_tet10_face(*f) for f in _TET_CORNER_FACES],
    "hex8": [list(f) for f in _HEX_FACES], "hex20": [list(f) for f in _HEX_FACES],
    "wedge6": [list(f) for f in _WEDGE_FACES], "wedge15": [list(f) for f in _WEDGE_FACES],
}
_RULES: dict[int, list] = {}


def _face_rule(k: int):
    """(N, dN, weight) per quadrature point for tri3 / tri6 / quad4 faces."""
    if k in _RULES:
        return _RULES[k]
    pts = []
    if k in (3, 6):
        gp = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
        for x, y in gp:
            L = np.array([1 - x - y, x, y])
            dL = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
            if k == 3:
                N, dN = L, dL
            else:
                N = np.r_[L * (2 * L - 1), 4 * L[0] * L[1], 4 * L[1] * L[2], 4 * L[2] * L[0]]
                dN = np.zeros((2, 6))
                for d in range(2):
                    dN[d, :3] = (4 * L - 1) * dL[d]
                    for m, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
                        dN[d, 3 + m] = 4 * (dL[d, a] * L[b] + L[a] * dL[d, b])
            pts.append((N, dN, 1 / 6))
    else:
        g = 1 / np.sqrt(3)
        for x in (-g, g):
            for y in (-g, g):
                sx, sy = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
                N = (1 + sx * x) * (1 + sy * y) / 4
                dN = np.array([sx * (1 + sy * y) / 4, sy * (1 + sx * x) / 4])
                pts.append((N, dN, 1.0))
    _RULES[k] = pts
    return pts


# ----------------------------------------------------------------------------- components

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame_dirs(points: np.ndarray, frame: dict | None):
    """Unit vectors (e_r, e_t, e_a) per point for cylindrical/spherical frames."""
    if frame is None:
        raise QoIError("component requires a frame")
    P = np.atleast_2d(points)
    if frame["type"] == "cylindrical":
        a = _unit(frame["axis"])
        d = P - np.asarray(frame["point"], float)
        d = d - np.outer(d @ a, a)
        er = d / np.linalg.norm(d, axis=1, keepdims=True)
        et = np.cross(a, er)
        ea = np.tile(a, (len(P), 1))
        return er, et, ea
    if frame["type"] == "spherical":
        d = P - np.asarray(frame["center"], float)
        er = d / np.linalg.norm(d, axis=1, keepdims=True)
        k = np.where(np.abs(er[:, 2:3]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
        et = np.cross(er, k)
        et /= np.linalg.norm(et, axis=1, keepdims=True)
        return er, et, np.cross(er, et)
    raise QoIError(f"unknown frame {frame['type']}")


def _tensor(s: np.ndarray) -> np.ndarray:
    t = np.zeros((len(s), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    t[:, 0, 1] = t[:, 1, 0] = s[:, 3]
    t[:, 1, 2] = t[:, 2, 1] = s[:, 4]
    t[:, 2, 0] = t[:, 0, 2] = s[:, 5]
    return t


def component(field: str, values: np.ndarray, comp: str, points: np.ndarray | None = None,
              frame: dict | None = None) -> np.ndarray:
    v = np.atleast_2d(values)
    if field == "U" or field == "RF":
        if comp in ("x", "y", "z"):
            return v[:, "xyz".index(comp)]
        if comp == "mag":
            return np.linalg.norm(v[:, :3], axis=1)
        er, et, ea = _frame_dirs(points, frame)
        return np.einsum("nd,nd->n", v[:, :3], {"r": er, "t": et, "a": ea}[comp])
    if field in ("S", "E"):
        if comp in _S_INDEX:
            return v[:, _S_INDEX[comp]]
        if comp == "mises":
            return von_mises(v)
        if comp in ("p1", "p2", "p3"):
            return principal(v)[:, int(comp[1]) - 1]
        if comp == "tresca":
            p = principal(v)
            return p[:, 0] - p[:, 2]
        er, et, ea = _frame_dirs(points, frame)
        T = _tensor(v)
        d = {"rr": (er, er), "tt": (et, et), "aa": (ea, ea), "rt": (er, et), "ra": (er, ea), "ta": (et, ea)}[comp]
        return np.einsum("ni,nij,nj->n", d[0], T, d[1])
    raise QoIError(f"unknown field {field}")


def _stat(vals: np.ndarray, stat: str) -> float:
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        raise QoIError("no values in region")
    if stat == "mean":
        return float(vals.mean())
    if stat == "max":
        return float(vals.max())
    if stat == "min":
        return float(vals.min())
    if stat == "absmax":
        return float(vals[np.argmax(np.abs(vals))])
    if stat.startswith("p"):
        return float(np.percentile(vals, float(stat[1:])))
    raise QoIError(f"unknown stat {stat}")


# ----------------------------------------------------------------------------- evaluation

def evaluate_qoi(spec: dict, rv: ResultView, case_dir: Path | None = None) -> dict:
    kind = spec["kind"]
    idx = spec.get("increment")
    if kind == "point":
        p = np.asarray(spec["point"], float)[None]
        loc = rv.locator.locate(p)
        vals = rv.locator.interpolate(loc, rv.nodal(spec["field"], idx))
        val = component(spec["field"], vals, spec["component"], p, spec.get("frame"))[0]
        rel_dist = float(loc.distance[0]) / rv._scale
        out = {"value": float(val), "inside": bool(loc.inside[0]), "distance": float(loc.distance[0]),
               "element_quality": element_quality(rv, loc)}
        if rel_dist > spec.get("max_distance_rel", 0.01):
            raise QoIError(f"point {spec['point']} is not inside the meshed domain (distance {loc.distance[0]:.3g})")
        return out
    if kind == "region":
        m = select_nodes(rv.coords, spec["region"], rv._scale)
        if not m.any():
            raise QoIError("region selects no nodes of the submission mesh")
        stat = spec.get("stat", "mean")
        if stat == "mean" and spec.get("weighting", "area") == "area":
            full = component(spec["field"], rv.nodal(spec["field"], idx), spec["component"], rv.coords,
                             spec.get("frame"))
            val, area = rv.surface_mean(np.nan_to_num(full), m)
            if np.isfinite(val):
                return {"value": float(val), "n_nodes": int(m.sum()), "area": area}
        vals = component(spec["field"], rv.nodal(spec["field"], idx)[m], spec["component"], rv.coords[m], spec.get("frame"))
        return {"value": _stat(vals, stat), "n_nodes": int(m.sum())}
    if kind == "line":
        a, b = np.asarray(spec["start"], float), np.asarray(spec["end"], float)
        n = int(spec.get("n", 21))
        P = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        loc = rv.locator.locate(P)
        if loc.distance.max() > spec.get("max_distance_rel", 0.01) * rv._scale:
            raise QoIError("line leaves the meshed domain")
        vals = component(spec["field"], rv.locator.interpolate(loc, rv.nodal(spec["field"], idx)),
                         spec["component"], P, spec.get("frame"))
        w = np.full(n, 1.0)
        w[0] = w[-1] = 0.5
        return {"value": float(np.sum(w * vals) / np.sum(w)), "n_points": n,
                "element_quality": element_quality(rv, loc)}
    if kind == "max":
        m = np.ones(len(rv.coords), bool)
        for r in spec.get("exclude", []):
            m &= ~_within(rv.coords, r)
        if "within" in spec:
            m &= _within(rv.coords, spec["within"])
        vals = component(spec["field"], rv.nodal(spec["field"], idx)[m], spec["component"], rv.coords[m], spec.get("frame"))
        pct = spec.get("percentile", 100)
        return {"value": _stat(vals, "max" if pct >= 100 else f"p{pct}"), "n_nodes": int(m.sum())}
    if kind == "reaction":
        # ccx "RF" is the nodal internal force K u at every node, i.e. reaction + applied load:
        # reference specs therefore pick support regions that carry no applied load.  "within"
        # (solid) regions also capture reference nodes of couplings / rigid bodies on an axis.
        if "within" in spec:
            m = _within(rv.coords, spec["within"])
        elif "region" in spec:
            m = select_nodes(rv.coords, spec["region"], rv._scale)
        else:
            m = np.ones(len(rv.coords), bool)
        rf = rv.nodal("RF", idx)[m]
        rf = np.nan_to_num(rf)
        if "moment" in spec:
            c = np.asarray(spec["moment"]["point"], float)
            a = _unit(spec["moment"]["axis"])
            val = float(np.sum(np.cross(rv.coords[m] - c, rf) @ a))
        else:
            val = float(rf[:, "xyz".index(spec["component"])].sum())
        return {"value": val, "n_nodes": int(m.sum())}
    if kind == "rotation":
        m = select_nodes(rv.coords, spec["region"], rv._scale)
        if m.sum() < 3:
            raise QoIError("rotation region needs >= 3 nodes")
        a = _unit(spec["axis"])
        d = rv.coords[m] - np.asarray(spec["point"], float)
        w = np.cross(a, d - np.outer(d @ a, a))
        u = rv.nodal("U", idx)[m]
        return {"value": float(np.sum(u * w) / np.sum(w * w)), "n_nodes": int(m.sum())}
    if kind == "frequency":
        tabs = read_eigen(rv.dat_path) if rv.dat_path and rv.dat_path.exists() else []
        if not tabs:
            raise QoIError("no eigenvalue output in .dat")
        modes = tabs[spec.get("step", -1)]
        n = spec["mode"]
        if len(modes) < n:
            raise QoIError(f"only {len(modes)} modes computed, need mode {n}")
        return {"value": modes[n - 1].frequency}
    if kind == "buckling":
        tabs = read_buckling(rv.dat_path) if rv.dat_path and rv.dat_path.exists() else []
        if not tabs:
            raise QoIError("no buckling output in .dat")
        rows = tabs[spec.get("step", -1)]
        n = spec["mode"]
        if len(rows) < n:
            raise QoIError(f"only {len(rows)} buckling modes computed, need mode {n}")
        return {"value": rows[n - 1][1]}
    if kind == "volume":
        return {"value": rv.volume()}
    if kind == "field_error":
        data = np.load((case_dir / spec["points"]) if case_dir else spec["points"])
        P, ref = data["points"], data["values"]
        loc = rv.locator.locate(P)
        val = rv.locator.interpolate(loc, rv.nodal(spec["field"], idx))[:, : ref.shape[1]]
        ok = loc.distance <= spec.get("max_distance_rel", 0.02) * rv._scale
        err = np.linalg.norm(val[ok] - ref[ok]) / np.linalg.norm(ref[ok])
        return {"value": float(err), "coverage": float(ok.mean())}
    raise QoIError(f"unknown QoI kind {kind}")


def _radius_ratio(X: np.ndarray) -> float:
    """3 r_in / R_circ of a tet's corner nodes (1 = regular, ~0 = sliver)."""
    a, b, c, d = X
    V = abs(np.dot(np.cross(b - a, c - a), d - a)) / 6
    A = sum(np.linalg.norm(np.cross(q - p, r - p)) / 2 for p, q, r in [(a, b, c), (a, b, d), (a, c, d), (b, c, d)])
    M = np.array([b - a, c - a, d - a])
    rhs = 0.5 * np.array([b @ b - a @ a, c @ c - a @ a, d @ d - a @ a])
    try:
        R = np.linalg.norm(np.linalg.solve(M, rhs) - a)
    except np.linalg.LinAlgError:
        return 0.0
    return float(3 * (3 * V / A) / R) if A > 0 and R > 0 else 0.0


def element_quality(rv: "ResultView", loc) -> float | None:
    """Worst radius ratio of the tets that contain the located points (diagnostic for the evaluator)."""
    q = [_radius_ratio(rv.coords[rv.elements[t][e][:4]]) for t, e in zip(loc.etype, loc.element)
         if str(t).startswith("tet")]
    return min(q) if q else None


def _within(coords: np.ndarray, region: dict) -> np.ndarray:
    """Solid (volumetric) membership: inside a ball / box / infinite cylinder."""
    t = region["type"]
    if t == "or":
        m = np.zeros(len(coords), bool)
        for r in region["regions"]:
            m |= _within(coords, r)
        return m
    if t == "ball":
        return np.linalg.norm(coords - np.asarray(region["center"], float), axis=1) <= region["radius"]
    if t == "box":
        lo, hi = np.asarray(region["min"], float), np.asarray(region["max"], float)
        return np.all((coords >= lo) & (coords <= hi), axis=1)
    if t == "cylinder_solid":
        a = _unit(region["axis"])
        d = coords - np.asarray(region["point"], float)
        return np.linalg.norm(d - np.outer(d @ a, a), axis=1) <= region["radius"]
    return select_nodes(coords, region)
