"""Code verification by the method of manufactured solutions (MMS).

Code verification asks whether the fixed solver solves its equations correctly, through the
public API only (``feacore.solver.solve`` + ``Result.probe / nodal / evaluate``):

* smooth exact solutions (manufactured, or classical exact ones such as Lame's thick cylinder)
  must converge at the theoretical rates for P_k elements: displacement / temperature L2 error
  ~ h^(k+1), recovered stress / flux ~ h^k (often better: superconvergence), strain energy
  ~ h^(2k);
* linear fields must be reproduced to round-off on distorted meshes (patch test).

Pieces
------
``to_expr``                 sympy -> string accepted by ``feacore.case.eval_expr`` (checked
                            numerically against ``sympy.lambdify``).
``Plane`` / ``Arc``         boundary geometry: outward normals and the MMS corrections that make
                            an arbitrary smooth field satisfy pressure / symmetry / convection
                            conditions exactly (zero tangential traction, zero normal
                            displacement, Robin relation with a constant ambient temperature).
``Domain`` (+ ``DOMAINS``)  gmsh geometry with named physical groups, sample points, quadrature
                            on the exact domain (for exact energies / reactions).
``elasticity_mms``          exact displacement (+ optional temperature for thermal strain) ->
``heat_mms``                body force / source, Dirichlet, traction / pressure / flux /
                            convection data, lambdified exact fields, exact strain energy.
``run_convergence``         solve on a refinement sequence, sample errors, observed orders.
``run_patch``               patch test on a distorted mesh.
``convergence_catalog``     the verification matrix used by ``tests/test_code_verification.py``
``patch_catalog``           and ``scripts/run_code_verification.py``.

Error measures: relative RMS errors over a fixed quasi-random (scrambled Halton) point cloud
covering the whole exact domain, the same points for every mesh of a study. ``probe`` evaluates
points that fall marginally outside a straight-sided discrete domain in the nearest cell, so
the measure is the RMS over the exact domain. Components compared: displacement (ux, uy[, uz])
or T; stresses (sxx, syy, sxy [, szz plane strain] [, stt axisymmetric] [, syz, sxz 3d]) or flux
(qx, qy[, qz]). Axisymmetric points are uniform in (r, z) (not r-weighted), which is the more
demanding choice near the axis.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from sympy.printing.str import StrPrinter

from feacore.case import BC, Analysis, Case, Load, Material, QoI, eval_expr

x, y, z = sp.symbols("x y z", real=True)
XYZ = (x, y, z)

DEFAULT_WORKDIR = Path("/tmp/codever")

# --------------------------------------------------------------------------- expressions


class _CaseExprPrinter(StrPrinter):
    """StrPrinter restricted to the grammar of ``feacore.case.eval_expr``."""

    _FUNCS = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "arcsin", "acos": "arccos",
        "atan": "arctan", "atan2": "arctan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
        "exp": "exp", "log": "log", "sign": "sign", "Abs": "abs",
    }

    def _print_Function(self, e):
        name = e.func.__name__
        if name not in self._FUNCS or (name == "log" and len(e.args) != 1):
            raise ValueError(f"{name}() is not available in case expressions: {e}")
        return "%s(%s)" % (self._FUNCS[name], ", ".join(self._print(a) for a in e.args))

    _print_Abs = _print_Function
    _print_exp = _print_Function
    _print_log = _print_Function
    _print_sign = _print_Function

    def _print_Exp1(self, e):
        return "e"

    def _print_Pi(self, e):
        return "pi"

    def _print_Float(self, e):
        return repr(float(e))

    def _print_Symbol(self, e):
        if e.name not in ("x", "y", "z"):
            raise ValueError(f"free symbol {e.name!r} in a case expression")
        return e.name

    def _print_Min(self, e):
        out = self._print(e.args[0])
        for a in e.args[1:]:
            out = f"minimum({out}, {self._print(a)})"
        return out

    def _print_Max(self, e):
        out = self._print(e.args[0])
        for a in e.args[1:]:
            out = f"maximum({out}, {self._print(a)})"
        return out

    def _print_Piecewise(self, e):
        *pairs, (last, cond) = e.args
        if cond != sp.true:
            raise ValueError("Piecewise needs a final True branch")
        out = self._print(last)
        for val, c in reversed(pairs):
            out = f"where({self._print(c)}, {self._print(val)}, {out})"
        return out

    def _print_Relational(self, e):
        if e.rel_op not in ("<", ">", "<=", ">="):
            raise ValueError(f"relational {e.rel_op} not allowed in case expressions")
        return f"({self._print(e.lhs)} {e.rel_op} {self._print(e.rhs)})"

    def _unsupported(self, e):
        raise ValueError(f"{e} cannot appear in a case expression")

    _print_ImaginaryUnit = _print_Infinity = _print_NegativeInfinity = _unsupported
    _print_NaN = _print_ComplexInfinity = _unsupported


def to_expr(e) -> float | str:
    """sympy expression in x, y, z -> float (if constant) or an ``eval_expr`` string."""
    e = sp.sympify(e)
    if not e.free_symbols:
        v = complex(sp.N(e, 17))
        if abs(v.imag) > 1e-14 * max(1.0, abs(v.real)):
            raise ValueError(f"complex constant {e}")
        return float(v.real)
    return _CaseExprPrinter().doprint(e)


def lambdify(e) -> Callable[[np.ndarray], np.ndarray]:
    """sympy expression -> f(points (n, 2|3)) -> (n,) float array."""
    f = sp.lambdify(XYZ, sp.sympify(e), modules="numpy")

    def g(pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        zc = pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts))
        out = f(pts[:, 0], pts[:, 1], zc)
        return np.broadcast_to(np.asarray(out, dtype=float), (len(pts),)).copy()

    return g


def eval_value(v, pts) -> np.ndarray:
    """Evaluate a case value (float or expression string) at points (n, 2|3)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    zc = pts[:, 2] if pts.shape[1] > 2 else None
    if isinstance(v, str):
        return eval_expr(v, pts[:, 0], pts[:, 1], zc)
    return np.full(len(pts), float(v))


def check_expr(e, s, pts, rtol: float = 1e-11) -> None:
    """Raise if the printed string ``s`` does not evaluate like the sympy expression ``e``."""
    ref = lambdify(e)(pts)
    got = eval_value(s, pts)
    scale = max(np.max(np.abs(ref)), 1e-300)
    if not np.all(np.isfinite(got)) or np.max(np.abs(got - ref)) > rtol * scale + 1e-300:
        raise AssertionError(f"expression printing mismatch for {e}: {s!r}")


# --------------------------------------------------------------------------- boundary geometry


@dataclass(frozen=True)
class Plane:
    """Facet group on the plane x_axis = value, outward normal sign * e_axis."""

    axis: int
    value: float
    sign: int

    def normal(self, dim: int) -> list:
        return [sp.Integer(self.sign) if i == self.axis else sp.S.Zero for i in range(dim)]

    def normal_np(self, pts: np.ndarray) -> np.ndarray:
        n = np.zeros_like(pts)
        n[:, self.axis] = self.sign
        return n


@dataclass(frozen=True)
class Arc:
    """Circular arc / cylinder r = radius about the z axis, outward normal sign * e_r."""

    radius: float
    sign: int

    def normal(self, dim: int) -> list:
        r = sp.sqrt(x**2 + y**2)
        n = [self.sign * x / r, self.sign * y / r]
        return n + [sp.S.Zero] * (dim - 2)

    def normal_np(self, pts: np.ndarray) -> np.ndarray:
        n = np.zeros_like(pts)
        r = np.hypot(pts[:, 0], pts[:, 1])
        n[:, 0] = self.sign * pts[:, 0] / r
        n[:, 1] = self.sign * pts[:, 1] / r
        return n


# --------------------------------------------------------------------------- domains


def _gauss(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w


def _tensor(*rules) -> tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(*[r[0] for r in rules], indexing="ij")
    wgrids = np.meshgrid(*[r[1] for r in rules], indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    w = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return pts, w


def _halton(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    return qmc.Halton(d=d, scramble=True, seed=seed).random(n)


@dataclass
class Domain:
    """Verification domain. ``kind``: box (2D rectangle / 3D box), annulus (quarter annulus,
    2D) or annulus3d (quarter annular cylinder). Physical groups: box 2D left/right/bottom/top,
    box 3D x0/x1/y0/y1/z0/z1, annulus inner/outer/bottom (y=0)/left (x=0), annulus3d the
    same plus z0/z1; cells "domain"."""

    name: str
    kind: str
    dim: int
    params: dict
    description: str = ""
    touches_axis: bool = False

    # ---- geometry
    @property
    def boundaries(self) -> dict:
        p = self.params
        if self.kind == "box":
            lo, hi = p["lo"], p["hi"]
            if self.dim == 2:
                return {"left": Plane(0, lo[0], -1), "right": Plane(0, hi[0], 1),
                        "bottom": Plane(1, lo[1], -1), "top": Plane(1, hi[1], 1)}
            return {f"{c}{s}": Plane(i, (lo if s == 0 else hi)[i], -1 if s == 0 else 1)
                    for i, c in enumerate("xyz") for s in (0, 1)}
        out = {"inner": Arc(p["R1"], -1), "outer": Arc(p["R2"], 1),
               "bottom": Plane(1, 0.0, -1), "left": Plane(0, 0.0, -1)}
        if self.kind == "annulus3d":
            out.update(z0=Plane(2, 0.0, -1), z1=Plane(2, p["L"], 1))
        return out

    @property
    def measure(self) -> float:
        p = self.params
        if self.kind == "box":
            return float(np.prod(np.subtract(p["hi"], p["lo"])))
        a = 0.25 * math.pi * (p["R2"] ** 2 - p["R1"] ** 2)
        return a * p["L"] if self.kind == "annulus3d" else a

    @property
    def length_scale(self) -> float:
        p = self.params
        if self.kind == "box":
            return float(np.max(np.subtract(p["hi"], p["lo"])))
        return float(p["R2"])

    def build(self) -> None:
        """Create the OCC geometry + named physical groups in the current gmsh model."""
        import gmsh

        from feacore.geometry.gmsh_tools import add_group, in_box

        occ = gmsh.model.occ
        p = self.params
        big = 1e30
        if self.kind == "box":
            lo, hi = p["lo"], p["hi"]
            if self.dim == 2:
                occ.addRectangle(lo[0], lo[1], 0, hi[0] - lo[0], hi[1] - lo[1])
            else:
                occ.addBox(lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            occ.synchronize()
            fd = self.dim - 1
            for name, pl in self.boundaries.items():
                box = [-big, -big, -big, big, big, big]
                box[pl.axis] = box[pl.axis + 3] = pl.value
                add_group(fd, in_box(fd, *box), name)
            add_group(self.dim, [t for _, t in gmsh.model.getEntities(self.dim)], "domain")
            return
        R1, R2 = p["R1"], p["R2"]
        if self.kind == "annulus":
            outer = occ.addDisk(0, 0, 0, R2, R2)
            inner = occ.addDisk(0, 0, 0, R1, R1)
            ring, _ = occ.cut([(2, outer)], [(2, inner)])
            rect = occ.addRectangle(0, 0, 0, 2 * R2, 2 * R2)
            occ.intersect(ring, [(2, rect)])
        else:
            L = p["L"]
            outer = occ.addCylinder(0, 0, 0, 0, 0, L, R2, angle=math.pi / 2)
            inner = occ.addCylinder(0, 0, 0, 0, 0, L, R1, angle=math.pi / 2)
            occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        fd = self.dim - 1
        tol = 1e-4 * R2  # OCC pads bounding boxes of curved entities
        groups = {
            "bottom": in_box(fd, -big, 0, -big, big, 0, big, tol=tol),
            "left": in_box(fd, 0, -big, -big, 0, big, big, tol=tol),
        }
        if self.kind == "annulus3d":
            groups["z0"] = in_box(fd, -big, -big, 0, big, big, 0, tol=tol)
            groups["z1"] = in_box(fd, -big, -big, p["L"], big, big, p["L"], tol=tol)
        taken = {t for v in groups.values() for t in v}
        inner_t = [t for t in in_box(fd, -big, -big, -big, R1, R1, big, tol=tol) if t not in taken]
        groups["inner"] = inner_t
        taken |= set(inner_t)
        groups["outer"] = [t for _, t in gmsh.model.getEntities(fd) if t not in taken]
        for name, tags in groups.items():
            add_group(fd, tags, name)
        add_group(self.dim, [t for _, t in gmsh.model.getEntities(self.dim)], "domain")

    def mesh(self, path: str | Path, h: float, order: int = 1, distort: float = 0.0,
             seed: int = 0) -> dict:
        """Write a gmsh mesh (order 1: tri3/tet4, order 2: tri6/tet10 with curved boundary
        nodes). ``distort`` > 0 moves interior nodes randomly by up to distort * h (P2: midside
        nodes follow their edge and get an extra interior perturbation, i.e. curved interior
        edges). Returns {"path", "n_cells", "n_nodes", "h_rep", "min_quality"}; cached on disk."""
        import gmsh

        from feacore.geometry.gmsh_tools import gmsh_session, set_uniform_size

        path = Path(path)
        side = path.with_suffix(".json")
        if path.exists() and side.exists():
            info = json.loads(side.read_text())
            if info.get("key") == [self.name, h, order, distort, seed]:
                return info
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(f"mms_{self.name}"):
            self.build()
            set_uniform_size(h)
            gmsh.model.mesh.generate(self.dim)
            if order >= 2:
                gmsh.model.mesh.setOrder(2)
            q = _distort(self.dim, h, distort, seed) if distort > 0 else None
            etypes, etags, _ = gmsh.model.mesh.getElements(self.dim)
            n_cells = int(sum(len(t) for t in etags))
            tags = np.concatenate([np.asarray(t) for t in etags])
            qual = gmsh.model.mesh.getElementQualities(tags, "minSJ")
            n_nodes = len(gmsh.model.mesh.getNodes()[0])
            gmsh.write(str(path))
        info = {"key": [self.name, h, order, distort, seed], "path": str(path), "n_cells": n_cells,
                "n_nodes": n_nodes, "h_rep": (self.measure / n_cells) ** (1.0 / self.dim),
                "min_quality": float(np.min(qual)), "distort_quality": q}
        side.write_text(json.dumps(info))
        return info

    # ---- sampling / quadrature on the exact domain
    def sample(self, n: int, seed: int = 12345) -> np.ndarray:
        p = self.params
        u = _halton(n, self.dim, seed)
        if self.kind == "box":
            lo, hi = np.asarray(p["lo"], float), np.asarray(p["hi"], float)
            return lo + u * (hi - lo)
        R1, R2 = p["R1"], p["R2"]
        r = np.sqrt(R1**2 + u[:, 0] * (R2**2 - R1**2))
        t = 0.5 * math.pi * u[:, 1]
        cols = [r * np.cos(t), r * np.sin(t)]
        if self.kind == "annulus3d":
            cols.append(p["L"] * u[:, 2])
        return np.stack(cols, axis=1)

    def quadrature(self, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre points / weights on the exact domain (no thickness / 2 pi r factors)."""
        p = self.params
        if self.kind == "box":
            return _tensor(*[_gauss(n, a, b) for a, b in zip(p["lo"], p["hi"])])
        rules = [_gauss(n, p["R1"], p["R2"]), _gauss(n, 0.0, 0.5 * math.pi)]
        if self.kind == "annulus3d":
            rules.append(_gauss(max(8, n // 2), 0.0, p["L"]))
        pol, w = _tensor(*rules)
        r, t = pol[:, 0], pol[:, 1]
        cols = [r * np.cos(t), r * np.sin(t)] + ([pol[:, 2]] if self.kind == "annulus3d" else [])
        return np.stack(cols, axis=1), w * r

    def boundary_quadrature(self, group: str, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """Gauss points / weights (length or area) on a boundary group of the exact domain."""
        p = self.params
        geom = self.boundaries[group]
        if self.kind == "box":
            rules = [_gauss(n, a, b) for a, b in zip(p["lo"], p["hi"])]
            rules[geom.axis] = (np.array([geom.value]), np.array([1.0]))
            return _tensor(*rules)
        R1, R2 = p["R1"], p["R2"]
        zr = [_gauss(max(8, n // 2), 0.0, p["L"])] if self.kind == "annulus3d" else []
        if isinstance(geom, Arc):
            R = geom.radius
            pol, w = _tensor(_gauss(n, 0.0, 0.5 * math.pi), *zr)
            cols = [R * np.cos(pol[:, 0]), R * np.sin(pol[:, 0])] + ([pol[:, 1]] if zr else [])
            return np.stack(cols, axis=1), w * R
        if group in ("bottom", "left"):
            pts, w = _tensor(_gauss(n, R1, R2), *zr)
            s = pts[:, 0]
            cols = [s, np.zeros_like(s)] if group == "bottom" else [np.zeros_like(s), s]
            if zr:
                cols.append(pts[:, 1])
            return np.stack(cols, axis=1), w
        # z0 / z1 annular faces
        pol, w = _tensor(_gauss(n, R1, R2), _gauss(n, 0.0, 0.5 * math.pi))
        r, t = pol[:, 0], pol[:, 1]
        zc = np.full_like(r, geom.value)
        return np.stack([r * np.cos(t), r * np.sin(t), zc], axis=1), w * r


def _distort(dim: int, h: float, amp: float, seed: int, max_tries: int = 6) -> float:
    """Randomly move interior nodes of the current gmsh mesh (see ``Domain.mesh``); returns the
    minimum scaled Jacobian after distortion (> 0 guaranteed by halving amp on failure)."""
    import gmsh

    rng = np.random.default_rng(seed)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3).copy()
    pos = {int(t): i for i, t in enumerate(tags)}
    interior = {int(t) for t in gmsh.model.mesh.getNodes(dim, -1, includeBoundary=False)[0]}
    etypes, etags, enodes = gmsh.model.mesh.getElements(dim)
    nv = dim + 1
    mids = {}  # midside node -> (vertex a, vertex b)
    for et, en in zip(etypes, enodes):
        npe = gmsh.model.mesh.getElementProperties(et)[3]
        conn = np.asarray(en, dtype=np.int64).reshape(-1, npe)
        if npe == nv:
            continue
        for row in conn[:200000]:
            vx = row[:nv]
            for m in row[nv:]:
                if int(m) in mids:
                    continue
                cm = coords[pos[int(m)]]
                best = None
                for a in range(nv):
                    for b in range(a + 1, nv):
                        d = np.linalg.norm(0.5 * (coords[pos[int(vx[a])]] + coords[pos[int(vx[b])]]) - cm)
                        if best is None or d < best[0]:
                            best = (d, int(vx[a]), int(vx[b]))
                mids[int(m)] = best[1:]
    vertices = [t for t in interior if t not in mids]
    all_tags = np.concatenate([np.asarray(t) for t in etags])

    def rand_ball(k):
        v = rng.normal(size=(k, 3))
        v[:, dim:] = 0.0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * rng.uniform(0, 1, size=(k, 1)) ** (1.0 / dim)

    dv = rand_ball(len(vertices))
    dm = rand_ball(len(mids))
    a = amp
    for _ in range(max_tries):
        new = coords.copy()
        for t, d in zip(vertices, dv):
            new[pos[t]] += a * h * d
        for (m, (va, vb)), d in zip(mids.items(), dm):
            new[pos[m]] = 0.5 * (new[pos[va]] + new[pos[vb]])
            if m in interior:
                new[pos[m]] += 0.4 * a * h * d
        for t in set(vertices) | set(mids):
            gmsh.model.mesh.setNode(t, new[pos[t]].tolist(), [])
        q = float(np.min(gmsh.model.mesh.getElementQualities(all_tags, "minSJ")))
        if q > 0.05:
            return q
        a *= 0.5
    raise RuntimeError("could not distort the mesh without inverting elements")


DOMAINS = {
    "square": Domain("square", "box", 2, {"lo": (0.0, 0.0), "hi": (1.0, 1.0)},
                     "unit square, unstructured triangles"),
    "annulus": Domain("annulus", "annulus", 2, {"R1": 0.5, "R2": 1.0},
                      "quarter annulus 0.5 < r < 1 (curved inner / outer arcs)"),
    "cube": Domain("cube", "box", 3, {"lo": (0.0, 0.0, 0.0), "hi": (1.0, 1.0, 1.0)},
                   "unit cube, unstructured tetrahedra"),
    "annulus3d": Domain("annulus3d", "annulus3d", 3, {"R1": 0.5, "R2": 1.0, "L": 0.25},
                        "quarter annular cylinder 0.5 < r < 1, 0 < z < 0.25"),
    "axi_ring": Domain("axi_ring", "box", 2, {"lo": (0.5, 0.0), "hi": (1.5, 1.0)},
                       "axisymmetric (r, z) in [0.5, 1.5] x [0, 1], away from the axis"),
    "axi_solid": Domain("axi_solid", "box", 2, {"lo": (0.0, 0.0), "hi": (1.0, 1.0)},
                        "axisymmetric (r, z) in [0, 1] x [0, 1], touching the axis (left = r = 0)",
                        touches_axis=True),
}


# --------------------------------------------------------------------------- elasticity


def _lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def elastic_state(formulation: str, u: Sequence, E: float, nu: float, th=0) -> dict:
    """Strains, stresses (case field names), stress tensor in the analysis coordinates, body
    force b = -div(sigma) and mechanical energy density 0.5 eps:C:eps for a displacement field
    ``u`` (axisymmetric: u = (u_r, u_z) with x = r, y = z). ``th`` = alpha * (T - T_ref)."""
    lam, mu = _lame(E, nu)
    k3 = 3 * lam + 2 * mu  # = E / (1 - 2 nu)
    S = {f: sp.S.Zero for f in ("sxx", "syy", "szz", "sxy", "syz", "sxz", "stt")}
    u = [sp.sympify(c) for c in u]
    if formulation == "3d":
        X = XYZ
        eps = [[(sp.diff(u[i], X[j]) + sp.diff(u[j], X[i])) / 2 for j in range(3)] for i in range(3)]
        tr = eps[0][0] + eps[1][1] + eps[2][2]

        def sig(t):
            return [[2 * mu * eps[i][j] + ((lam * tr - k3 * t) if i == j else 0) for j in range(3)]
                    for i in range(3)]

        s, sm = sig(th), sig(0)
        S.update(sxx=s[0][0], syy=s[1][1], szz=s[2][2], sxy=s[0][1], syz=s[1][2], sxz=s[0][2])
        body = [-sum(sp.diff(s[i][j], X[j]) for j in range(3)) for i in range(3)]
        w = sum(sm[i][j] * eps[i][j] for i in range(3) for j in range(3)) / 2
        return dict(stress=S, tensor=s, body=body, w=w)
    if formulation in ("plane_strain", "plane_stress"):
        exx, eyy = sp.diff(u[0], x), sp.diff(u[1], y)
        exy = (sp.diff(u[0], y) + sp.diff(u[1], x)) / 2

        def sig(t):
            if formulation == "plane_strain":
                tr = exx + eyy
                return (lam * tr + 2 * mu * exx - k3 * t, lam * tr + 2 * mu * eyy - k3 * t,
                        2 * mu * exy, lam * tr - k3 * t)
            c = E / (1 - nu**2)
            return (c * (exx + nu * eyy) - E * t / (1 - nu), c * (eyy + nu * exx) - E * t / (1 - nu),
                    2 * mu * exy, sp.S.Zero)

        sxx, syy, sxy, szz = sig(th)
        mxx, myy, mxy, _ = sig(0)
        S.update(sxx=sxx, syy=syy, sxy=sxy, szz=szz)
        body = [-(sp.diff(sxx, x) + sp.diff(sxy, y)), -(sp.diff(sxy, x) + sp.diff(syy, y))]
        w = (mxx * exx + myy * eyy + 2 * mxy * exy) / 2  # eps_zz = 0 (strain) / s_zz = 0 (stress)
        return dict(stress=S, tensor=[[sxx, sxy], [sxy, syy]], body=body, w=w)
    if formulation == "axisymmetric":
        r = x
        err, ezz, ett = sp.diff(u[0], x), sp.diff(u[1], y), u[0] / r
        erz = (sp.diff(u[0], y) + sp.diff(u[1], x)) / 2
        tr = err + ezz + ett

        def sig(t):
            return (lam * tr + 2 * mu * err - k3 * t, lam * tr + 2 * mu * ezz - k3 * t,
                    lam * tr + 2 * mu * ett - k3 * t, 2 * mu * erz)

        srr, szz, stt, srz = sig(th)
        mrr, mzz, mtt, mrz = sig(0)
        S.update(sxx=srr, syy=szz, sxy=srz, stt=stt)
        body = [-(sp.diff(srr, x) + sp.diff(srz, y) + (srr - stt) / r),
                -(sp.diff(srz, x) + sp.diff(szz, y) + srz / r)]
        w = (mrr * err + mzz * ezz + mtt * ett + 2 * mrz * erz) / 2
        return dict(stress=S, tensor=[[srr, srz], [srz, szz]], body=body, w=w)
    raise ValueError(f"unknown elasticity formulation {formulation!r}")


def _zero_shear(formulation, u, geom, E, nu, th):
    """Add a correction that vanishes on ``geom`` and cancels the tangential traction there,
    leaving the displacement and the normal traction on ``geom`` unchanged."""
    _, mu = _lame(E, nu)
    s = elastic_state(formulation, u, E, nu, th)["tensor"]
    u = list(u)
    if isinstance(geom, Plane):
        i, c = geom.axis, geom.value
        for j in range(len(u)):
            if j != i:
                tau = s[i][j].subs(XYZ[i], c)
                u[j] = u[j] - (XYZ[i] - c) * tau / mu
        return u
    if len(u) != 2:
        raise NotImplementedError("zero-shear correction on curved 3D faces")
    r = sp.sqrt(x**2 + y**2)
    ct, st = x / r, y / r
    tau = (s[1][1] - s[0][0]) * ct * st + s[0][1] * (ct**2 - st**2)  # s_r_theta
    R = geom.radius
    tau_arc = tau.subs({x: R * x / r, y: R * y / r}, simultaneous=True)
    g = -(r - R) * tau_arc / mu
    return [u[0] - g * st, u[1] + g * ct]


def _robin_fix(formulation, T, geom, k, hc, T_inf):
    """Correction making k dT/dn + hc (T - T_inf) = 0 on ``geom`` (T unchanged there)."""
    if isinstance(geom, Plane):
        i, c = geom.axis, geom.value
        rho = (geom.sign * k * sp.diff(T, XYZ[i]) + hc * (T - T_inf)).subs(XYZ[i], c)
        return T - (XYZ[i] - c) * geom.sign * rho / k
    r = sp.sqrt(x**2 + y**2)
    dTdr = (x * sp.diff(T, x) + y * sp.diff(T, y)) / r
    R = geom.radius
    rho = geom.sign * k * dTdr + hc * (T - T_inf)
    rho_arc = rho.subs({x: R * x / r, y: R * y / r}, simultaneous=True)
    return T - (r - R) * geom.sign * rho_arc / k


@dataclass
class MMSProblem:
    """A verification problem with a known exact solution (see ``elasticity_mms`` / ``heat_mms``)."""

    name: str
    physics: str
    formulation: str
    domain: Domain
    material: Material
    bcs: list
    loads: list
    exact_sym: dict                      # field name -> sympy expression
    u_fields: list
    s_fields: list
    thickness: float = 1.0
    T_ref: float = 0.0
    energy_exact: float | None = None    # 0.5 a(u, u) (mechanical stiffness), elasticity only
    energy_physical: float | None = None  # 0.5 int (eps - eps_th):C:(eps - eps_th), thermal cases
    reactions_exact: dict = field(default_factory=dict)  # (group, comp) -> int sigma.n (force on body)
    features: list = field(default_factory=list)
    description: str = ""
    expr_sizes: dict = field(default_factory=dict)

    @functools.cached_property
    def _exact_np(self) -> dict:
        return {f: lambdify(e) for f, e in self.exact_sym.items()}

    def exact(self, fld: str, pts) -> np.ndarray:
        return self._exact_np[fld](pts)

    def qois(self) -> list[QoI]:
        if self.physics != "elasticity":
            return []
        return [QoI(name="strain_energy", type="strain_energy")]

    def case(self, mesh_path: str | Path, order: int = 1, options: dict | None = None) -> Case:
        mat = copy.deepcopy(self.material)
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics=self.physics, formulation=self.formulation, order=order,
                              thickness=self.thickness, T_ref=self.T_ref, options=dict(options or {})),
            materials={"mat": mat},
            bcs=copy.deepcopy(self.bcs),
            loads=copy.deepcopy(self.loads),
            qois=self.qois(),
            meta={"generator": "feacore.verification.mms", "mms": self.name},
        )

    def describe(self) -> dict:
        return {"name": self.name, "physics": self.physics, "formulation": self.formulation,
                "domain": self.domain.name, "features": self.features,
                "bcs": [asdict(b) for b in self.bcs], "loads": [asdict(l) for l in self.loads],
                "exact": {k: str(v) for k, v in self.exact_sym.items()}}


def _factor(formulation: str, thickness: float, pts: np.ndarray) -> np.ndarray:
    if formulation == "axisymmetric":
        return 2 * math.pi * pts[:, 0]
    if formulation in ("plane_stress", "plane_strain"):
        return np.full(len(pts), thickness)
    return np.ones(len(pts))


def _is_zero(e, pts, scale) -> bool:
    v = lambdify(e)(pts)
    return bool(np.all(np.isfinite(v)) and np.max(np.abs(v)) <= 1e-11 * scale)


def _emit(e, pts_list, sizes, key):
    """sympy -> case value, checked numerically at the given point sets. A value that is
    constant on the first point set (the boundary it is applied on) is emitted as a float, so
    problems such as Lame's cylinder need no expression support."""
    vals = lambdify(e)(pts_list[0])
    if np.all(np.isfinite(vals)) and np.ptp(vals) <= 1e-12 * max(1.0, float(np.max(np.abs(vals)))):
        return float(np.mean(vals))
    v = to_expr(e)
    for pts in pts_list:
        if len(pts):
            check_expr(e, v, pts)
            if not np.all(np.isfinite(eval_value(v, pts))):
                raise ValueError(f"{key}: expression not finite on the domain: {v}")
    if isinstance(v, str):
        sizes[key] = len(v)
    return v


def _bc_kind(spec):
    return (spec, None) if isinstance(spec, str) else (spec[0], spec[1])


def _axis_points(dom: Domain, n=25) -> np.ndarray:
    lo, hi = dom.params["lo"], dom.params["hi"]
    return np.stack([np.zeros(n), np.linspace(lo[1], hi[1], n)], axis=1)


def elasticity_mms(name: str, formulation: str, domain: Domain, u: Sequence, bcs: dict, *,
                   E: float = 1000.0, nu: float = 0.3, alpha: float = 0.0, T=None,
                   T_ref: float = 0.0, thickness: float = 1.0, description: str = "",
                   check_tol: float = 1e-9) -> MMSProblem:
    """Manufactured elasticity problem. ``u``: exact displacement (sympy, in x, y, z; axisymmetric
    (u_r, u_z) in x = r, y = z). ``bcs``: {group: kind} with kind one of
      dirichlet   prescribed displacement, all components (expressions)
      fixed       u = 0 (checked)
      roller      prescribed normal component on a Plane (+ zero tangential traction: corrected)
      symmetry    u_n = 0 and zero tangential traction (both enforced by corrections)
      traction    traction vector sigma.n (expressions)
      pressure    p = -n.sigma.n (tangential traction removed by a correction)
      force       total force (constant traction only, checked)
      free        nothing applied; the exact traction must vanish (checked)
      none        nothing applied (e.g. the axis r = 0)
    ``T``: temperature field for thermal strain alpha (T - T_ref) (loads type "temperature")."""
    u = [sp.sympify(c) for c in u]
    ncomp = len(u)
    if ncomp != (3 if formulation == "3d" else 2):
        raise ValueError("u must have 3 components in 3d, 2 otherwise")
    Tsym = sp.sympify(T) if T is not None else None
    th = alpha * (Tsym - T_ref) if Tsym is not None else sp.S.Zero
    geoms = domain.boundaries
    # 1) corrections so the field satisfies the special BCs exactly
    for group, spec in bcs.items():
        kind, _ = _bc_kind(spec)
        geom = geoms[group]
        if kind == "symmetry":
            if not isinstance(geom, Plane):
                raise ValueError("symmetry needs a plane")
            i = geom.axis
            u[i] = u[i] - u[i].subs(XYZ[i], geom.value)
        if kind in ("symmetry", "pressure", "roller"):
            u = _zero_shear(formulation, u, geom, E, nu, th)
    st = elastic_state(formulation, u, E, nu, th)
    sig = st["tensor"]
    pts_in = domain.sample(400, seed=7)
    pts_all = [pts_in] + ([_axis_points(domain)] if domain.touches_axis else [])
    smax = max(float(np.max(np.abs(lambdify(e)(pts_in)))) for e in st["stress"].values()
               if e != 0) if any(e != 0 for e in st["stress"].values()) else 1.0
    umax = max(float(np.max(np.abs(lambdify(c)(pts_in)))) for c in u)
    sizes: dict = {}
    features = set()
    out_bcs = []
    reactions = {}
    for group, spec in bcs.items():
        kind, _ = _bc_kind(spec)
        geom = geoms[group]
        bq, bw = domain.boundary_quadrature(group, 24)
        bpts = [bq] + pts_all
        n = geom.normal(ncomp)
        t = [sum(sig[i][j] * n[j] for j in range(ncomp)) for i in range(ncomp)]
        tn = sum(t[i] * n[i] for i in range(ncomp))
        tt = [t[i] - tn * n[i] for i in range(ncomp)]
        fac = _factor(formulation, thickness, bq) * bw
        tvals = np.stack([lambdify(c)(bq) for c in t], axis=1)
        if kind in ("dirichlet", "fixed", "roller", "symmetry"):
            for c in range(ncomp):
                reactions[f"{group}:{c}"] = float(np.sum(fac * tvals[:, c]))
        if kind == "dirichlet":
            vals = [_emit(c, bpts, sizes, f"{group}.u{i}") for i, c in enumerate(u)]
            out_bcs.append(BC("displacement", group=group, components=list(range(ncomp)), value=vals))
        elif kind == "fixed":
            if not all(_is_zero(c, bq, umax) for c in u):
                raise ValueError(f"{name}: 'fixed' on {group} but u != 0 there")
            out_bcs.append(BC("fixed", group=group))
        elif kind == "roller":
            i = geom.axis
            out_bcs.append(BC("displacement", group=group, components=[i],
                              value=[_emit(u[i], bpts, sizes, f"{group}.u{i}")]))
        elif kind == "symmetry":
            out_bcs.append(BC("symmetry", group=group))
        elif kind == "traction":
            out_bcs.append(BC("traction", group=group,
                              value=[_emit(c, bpts, sizes, f"{group}.t{i}") for i, c in enumerate(t)]))
        elif kind == "pressure":
            out_bcs.append(BC("pressure", group=group, value=_emit(-tn, bpts, sizes, f"{group}.p")))
        elif kind == "force":
            if np.max(np.ptp(tvals, axis=0)) > 1e-12 * smax:
                raise ValueError(f"{name}: 'force' on {group} needs a constant traction")
            out_bcs.append(BC("force", group=group, value=[float(np.sum(fac * tvals[:, c]))
                                                           for c in range(ncomp)]))
        elif kind in ("free", "none"):
            pass
        else:
            raise ValueError(f"unknown elasticity BC kind {kind!r}")
        # consistency of the corrected field with the BC kind
        if kind in ("symmetry", "pressure", "roller"):
            if not all(_is_zero(c, bq, smax * check_tol / 1e-11) for c in tt):
                raise AssertionError(f"{name}: tangential traction on {group} not removed")
        if kind == "symmetry" and not _is_zero(u[geom.axis], bq, umax * check_tol / 1e-11):
            raise AssertionError(f"{name}: normal displacement on symmetry group {group} != 0")
        if kind == "free" and not all(_is_zero(c, bq, smax * check_tol / 1e-11) for c in t):
            raise AssertionError(f"{name}: 'free' group {group} carries a traction")
        if kind not in ("free", "none"):
            features.add(kind)
    loads = []
    body = st["body"]
    bscale = smax / domain.length_scale
    if not all(_is_zero(b, pts_in, bscale) for b in body):
        loads.append(Load("body_force", value=[_emit(b, pts_all, sizes, f"b{i}") for i, b in enumerate(body)]))
        features.add("body_force")
    if Tsym is not None:
        loads.append(Load("temperature", value=_emit(Tsym, pts_all, sizes, "T")))
        features.add("thermal")
    if any(isinstance(b.value, str) or (isinstance(b.value, list) and any(isinstance(v, str) for v in b.value))
           for b in out_bcs) or any(isinstance(l.value, str) or (isinstance(l.value, list) and
                                    any(isinstance(v, str) for v in l.value)) for l in loads):
        features.add("expressions")
    features.add(formulation)
    exact = {"ux": u[0], "uy": u[1]}
    if ncomp == 3:
        exact["uz"] = u[2]
    u_fields = list(exact)
    s_fields = {"plane_stress": ["sxx", "syy", "sxy"], "plane_strain": ["sxx", "syy", "sxy", "szz"],
                "axisymmetric": ["sxx", "syy", "sxy", "stt"],
                "3d": ["sxx", "syy", "szz", "sxy", "syz", "sxz"]}[formulation]
    for f in s_fields:
        exact[f] = st["stress"][f]
    qp, qw = domain.quadrature(40 if domain.dim == 2 else 20)
    fac = _factor(formulation, thickness, qp) * qw
    energy = float(np.sum(fac * lambdify(st["w"])(qp)))
    energy_phys = None
    if Tsym is not None:
        # physical stored energy 0.5 (eps - eps_th):C:(eps - eps_th) for isotropic C:
        # = w_mech - th tr(sigma_mech) + 0.5 th^2 I:C:I  (I:C:I = 3 (3 lam + 2 mu) in 3D /
        # plane strain / axisymmetric, 2 E / (1 - nu) for plane stress)
        mech = elastic_state(formulation, u, E, nu, 0)["stress"]
        lam, mu = _lame(E, nu)
        trm = mech["sxx"] + mech["syy"] + (mech["stt"] if formulation == "axisymmetric" else mech["szz"])
        ici = 2 * E / (1 - nu) if formulation == "plane_stress" else 3 * (3 * lam + 2 * mu)
        w_phys = st["w"] - th * trm + sp.Rational(1, 2) * ici * th**2
        energy_phys = float(np.sum(fac * lambdify(w_phys)(qp)))
    return MMSProblem(name=name, physics="elasticity", formulation=formulation, domain=domain,
                      material=Material(E=E, nu=nu, alpha=alpha), bcs=out_bcs, loads=loads,
                      exact_sym=exact, u_fields=u_fields, s_fields=s_fields, thickness=thickness,
                      T_ref=T_ref, energy_exact=energy, energy_physical=energy_phys,
                      reactions_exact=reactions, features=sorted(features),
                      description=description, expr_sizes=sizes)


# --------------------------------------------------------------------------- heat


def heat_state(formulation: str, T, k: float) -> dict:
    T = sp.sympify(T)
    if formulation == "3d":
        g = [sp.diff(T, c) for c in XYZ]
        lap = sum(sp.diff(T, c, 2) for c in XYZ)
    elif formulation == "2d":
        g = [sp.diff(T, x), sp.diff(T, y)]
        lap = sp.diff(T, x, 2) + sp.diff(T, y, 2)
    elif formulation == "axisymmetric":
        g = [sp.diff(T, x), sp.diff(T, y)]
        lap = sp.diff(T, x, 2) + sp.diff(T, x) / x + sp.diff(T, y, 2)
    else:
        raise ValueError(f"unknown heat formulation {formulation!r}")
    q = [-k * c for c in g]
    return dict(grad=g, q=q, source=-k * lap)


def heat_mms(name: str, formulation: str, domain: Domain, T, bcs: dict, *, k: float = 2.5,
             h_conv: float = 10.0, T_inf: float = 25.0, description: str = "",
             check_tol: float = 1e-9) -> MMSProblem:
    """Manufactured heat-conduction problem. ``bcs``: {group: kind}, kind one of temperature,
    flux (k dT/dn into the body), convection (h_conv, constant T_inf: the field is corrected so
    the Robin relation holds exactly), insulated (zero flux, checked), none."""
    T = sp.sympify(T)
    geoms = domain.boundaries
    for group, spec in bcs.items():
        kind, _ = _bc_kind(spec)
        if kind == "convection":
            T = _robin_fix(formulation, T, geoms[group], k, h_conv, T_inf)
    st = heat_state(formulation, T, k)
    dim = 3 if formulation == "3d" else 2
    pts_in = domain.sample(400, seed=7)
    pts_all = [pts_in] + ([_axis_points(domain)] if domain.touches_axis else [])
    qmax = max(float(np.max(np.abs(lambdify(c)(pts_in)))) for c in st["q"]) or 1.0
    sizes: dict = {}
    features = set()
    out_bcs = []
    for group, spec in bcs.items():
        kind, _ = _bc_kind(spec)
        geom = geoms[group]
        bq, _ = domain.boundary_quadrature(group, 24)
        bpts = [bq] + pts_all
        n = geom.normal(dim)
        flux_in = k * sum(st["grad"][i] * n[i] for i in range(dim))  # k dT/dn
        if kind == "temperature":
            out_bcs.append(BC("temperature", group=group, value=_emit(T, bpts, sizes, f"{group}.T")))
        elif kind == "flux":
            out_bcs.append(BC("flux", group=group, value=_emit(flux_in, bpts, sizes, f"{group}.q")))
        elif kind == "convection":
            out_bcs.append(BC("convection", group=group, h=float(h_conv), T_inf=float(T_inf)))
            res = flux_in - h_conv * (T_inf - T)
            if not _is_zero(res, bq, qmax * check_tol / 1e-11):
                raise AssertionError(f"{name}: Robin relation on {group} not satisfied")
        elif kind == "insulated":
            if not _is_zero(flux_in, bq, qmax * check_tol / 1e-11):
                raise AssertionError(f"{name}: 'insulated' group {group} carries flux")
        elif kind == "none":
            pass
        else:
            raise ValueError(f"unknown heat BC kind {kind!r}")
        if kind not in ("insulated", "none"):
            features.add(kind)
    loads = []
    if not _is_zero(st["source"], pts_in, qmax / domain.length_scale):
        loads.append(Load("source", value=_emit(st["source"], pts_all, sizes, "f")))
        features.add("source")
    if any(isinstance(b.value, str) for b in out_bcs) or any(isinstance(l.value, str) for l in loads):
        features.add("expressions")
    features.add("heat_" + formulation)
    exact = {"T": T, "qx": st["q"][0], "qy": st["q"][1]}
    s_fields = ["qx", "qy"]
    if dim == 3:
        exact["qz"] = st["q"][2]
        s_fields.append("qz")
    return MMSProblem(name=name, physics="heat", formulation=formulation, domain=domain,
                      material=Material(k=k), bcs=out_bcs, loads=loads, exact_sym=exact,
                      u_fields=["T"], s_fields=s_fields, features=sorted(features),
                      description=description, expr_sizes=sizes)
