"""Locate physical points in a solid mesh and interpolate nodal fields there.

Agents produce arbitrary meshes, so every point-wise quantity of interest is evaluated by
finding the element that contains the point and evaluating that element's shape functions:
the value a post-processor would show at that location.

* tets: local coordinates from the affine map of the four corners (vectorised); for curved
  tet10 elements a few Newton steps on the quadratic map refine them.
* hex/wedge: Newton iteration on the isoparametric map.
* points that fall slightly outside the discretised domain (e.g. on a curved CAD surface that
  the mesh facets) are assigned to the nearest candidate element with clamped local
  coordinates and flagged ``inside=False`` with the residual distance reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import shape as sf


@dataclass
class Location:
    etype: np.ndarray  # (n,) element type names
    element: np.ndarray  # (n,) row index within that type's connectivity
    xi: np.ndarray  # (n, 3) local coordinates
    inside: np.ndarray  # (n,) bool
    distance: np.ndarray  # (n,) |x(xi) - p| (0 when inside)


class Locator:
    def __init__(self, coords: np.ndarray, elements: dict[str, np.ndarray]):
        """``coords`` (N,3); ``elements`` maps type -> (M,k) connectivity as *row indices* into coords."""
        self.coords = np.asarray(coords, float)
        self.elements = {t: np.asarray(c, np.int64) for t, c in elements.items() if t in sf.N_NODES and len(c)}
        if not self.elements:
            raise ValueError("mesh has no supported solid elements")
        cents, owners = [], []
        for t, conn in self.elements.items():
            nc = sf.N_CORNERS[t]
            cents.append(self.coords[conn[:, :nc]].mean(axis=1))
            owners += [(t, i) for i in range(len(conn))]
        self._cent = np.vstack(cents)
        self._owner_type = np.array([o[0] for o in owners])
        self._owner_idx = np.array([o[1] for o in owners])
        self._tree = cKDTree(self._cent)
        # characteristic size per element (max corner distance to centroid)
        sizes = []
        for t, conn in self.elements.items():
            nc = sf.N_CORNERS[t]
            X = self.coords[conn[:, :nc]]
            sizes.append(np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=2).max(axis=1))
        self._size = np.concatenate(sizes)
        self._tet_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        diag = np.linalg.norm(self.coords.max(0) - self.coords.min(0))
        self.length_scale = diag if diag > 0 else 1.0

    # ------------------------------------------------------------------ tets (affine)
    def _tet_affine(self, t: str):
        if t not in self._tet_cache:
            conn = self.elements[t]
            X0 = self.coords[conn[:, 0]]
            M = np.stack([self.coords[conn[:, k]] - X0 for k in (1, 2, 3)], axis=2)  # columns
            det = np.linalg.det(M)
            good = np.abs(det) > 1e-300
            Minv = np.zeros_like(M)
            Minv[good] = np.linalg.inv(M[good])
            self._tet_cache[t] = (X0, Minv)
        return self._tet_cache[t]

    def _newton(self, t: str, e: int, p: np.ndarray, xi0=None, iters: int = 25) -> np.ndarray:
        X = self.coords[self.elements[t][e]]
        xi = np.array(sf.CENTER[t] if xi0 is None else xi0, float)
        for _ in range(iters):
            N = sf.shape(t, xi)[0]
            r = N @ X - p
            J = sf.dshape(t, xi)[0] @ X  # (3,3): rows d/dxi_k
            try:
                dxi = np.linalg.solve(J.T, r)
            except np.linalg.LinAlgError:
                break
            xi = xi - dxi
            if np.linalg.norm(dxi) < 1e-12:
                break
            if np.abs(xi).max() > 1e3:
                break
        return xi

    def locate(self, points: np.ndarray, k: int = 24, tol: float = 1e-6) -> Location:
        P = np.atleast_2d(np.asarray(points, float))
        n = len(P)
        etype = np.empty(n, dtype=object)
        elem = np.zeros(n, np.int64)
        XI = np.zeros((n, 3))
        ins = np.zeros(n, bool)
        dist = np.full(n, np.inf)
        best_out = np.full(n, np.inf)
        kk = min(k, len(self._cent))
        _, cand = self._tree.query(P, k=kk)
        cand = np.atleast_2d(cand)
        if cand.shape[0] != n:
            cand = cand.reshape(n, -1)
        for j in range(cand.shape[1]):
            todo = ~ins
            if not todo.any():
                break
            ids = cand[todo, j]
            pts = P[todo]
            rows = np.where(todo)[0]
            types = self._owner_type[ids]
            for t in np.unique(types):
                sel = types == t
                r = rows[sel]
                e = self._owner_idx[ids[sel]]
                p = pts[sel]
                if t in ("tet4", "tet10"):
                    X0, Minv = self._tet_affine(t)
                    xi = np.einsum("nij,nj->ni", Minv[e], p - X0[e])
                    if t == "tet10":
                        # refine on curved elements (cheap: few points are on curved boundaries)
                        curved = self._curved_tet10(e)
                        for q in np.where(curved & (sf.outside_measure(t, xi) < 0.2))[0]:
                            xi[q] = self._newton(t, e[q], p[q], xi[q], iters=8)
                else:
                    xi = np.array([self._newton(t, ee, pp) for ee, pp in zip(e, p)]).reshape(-1, 3)
                out = sf.outside_measure(t, xi)
                ok = out <= tol
                better = out < best_out[r]
                upd = ok | better
                etype[r[upd]] = t
                elem[r[upd]] = e[upd]
                XI[r[upd]] = xi[upd]
                best_out[r[upd]] = out[upd]
                ins[r[ok]] = True
        # clamp the ones that are outside and measure the residual distance
        for t in np.unique(etype[~ins].astype(str)) if (~ins).any() else []:
            sel = np.where((~ins) & (etype == t))[0]
            XI[sel] = sf.clamp(t, XI[sel])
            X = self.coords[self.elements[t][elem[sel]]]
            N = sf.shape(t, XI[sel])
            dist[sel] = np.linalg.norm(np.einsum("nk,nkd->nd", N, X) - P[sel], axis=1)
        dist[ins] = 0.0
        return Location(etype, elem, XI, ins, dist)

    def _curved_tet10(self, e: np.ndarray) -> np.ndarray:
        conn = self.elements["tet10"][e]
        X = self.coords[conn]
        dev = np.zeros(len(e))
        for m, (a, b) in enumerate(sf.TET10_EDGES):
            dev = np.maximum(dev, np.linalg.norm(X[:, 4 + m] - 0.5 * (X[:, a] + X[:, b]), axis=1))
        size = np.linalg.norm(X[:, 1] - X[:, 0], axis=1) + 1e-300
        return dev / size > 1e-8

    def interpolate(self, loc: Location, nodal: np.ndarray) -> np.ndarray:
        """Interpolate nodal values (N, c) or (N,) at located points."""
        nodal = np.asarray(nodal, float)
        squeeze = nodal.ndim == 1
        if squeeze:
            nodal = nodal[:, None]
        out = np.full((len(loc.element), nodal.shape[1]), np.nan)
        for t in np.unique(loc.etype.astype(str)):
            sel = np.where(loc.etype == t)[0]
            N = sf.shape(t, loc.xi[sel])
            conn = self.elements[t][loc.element[sel]]
            out[sel] = np.einsum("nk,nkc->nc", N, nodal[conn])
        return out[:, 0] if squeeze else out


def _quadrature(etype: str):
    """Points/weights on the reference element (collapsed Gauss-Legendre for simplices)."""
    x, w = np.polynomial.legendre.leggauss(3)
    u, wu = (x + 1) / 2, w / 2  # on [0, 1]
    if etype in ("hex8", "hex20"):
        P = np.array([[a, b, c] for a in x for b in x for c in x])
        W = np.array([wa * wb * wc for wa in w for wb in w for wc in w])
        return P, W
    if etype in ("tet4", "tet10"):
        P, W = [], []
        for a, wa in zip(u, wu):
            for b, wb in zip(u, wu):
                for c, wc in zip(u, wu):
                    P.append([a, b * (1 - a), c * (1 - a) * (1 - b)])
                    W.append(wa * wb * wc * (1 - a) ** 2 * (1 - b))
        return np.array(P), np.array(W)
    P, W = [], []  # wedge: collapsed triangle x Gauss line
    for a, wa in zip(u, wu):
        for b, wb in zip(u, wu):
            for c, wc in zip(x, w):
                P.append([a, b * (1 - a), c])
                W.append(wa * wb * (1 - a) * wc)
    return np.array(P), np.array(W)


def element_volumes(coords: np.ndarray, elements: dict[str, np.ndarray], chunk: int = 20000) -> dict[str, np.ndarray]:
    """Element volumes by Gauss quadrature of det(J) over the full isoparametric map, so curved
    (second-order) elements are measured with the geometry the solver actually integrates."""
    out = {}
    for t, conn in elements.items():
        if t not in sf.N_NODES or not len(conn):
            continue
        P, W = _quadrature(t)
        dN = sf.dshape(t, P)  # (q, 3, k)
        vols = np.empty(len(conn))
        for i in range(0, len(conn), chunk):
            X = coords[conn[i:i + chunk]]  # (m, k, 3)
            J = np.einsum("qdk,mkj->mqdj", dN, X)
            vols[i:i + chunk] = np.linalg.det(J) @ W
        out[t] = vols
    return out


def element_volumes_linear(coords: np.ndarray, elements: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Volumes from corner nodes (straight-sided approximation), via tet decomposition."""
    out = {}
    for t, conn in elements.items():
        if t not in sf.N_NODES or not len(conn):
            continue
        X = coords[conn]
        if t.startswith("tet"):
            tets = [(0, 1, 2, 3)]
        elif t.startswith("hex"):
            tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]
        else:  # wedge
            tets = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
        v = np.zeros(len(conn))
        for a, b, c, d in tets:
            v += np.einsum("ij,ij->i", np.cross(X[:, b] - X[:, a], X[:, c] - X[:, a]), X[:, d] - X[:, a]) / 6.0
        out[t] = v
    return out
