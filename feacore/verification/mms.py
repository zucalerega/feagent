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


def rationalize(e) -> sp.Expr:
    """Floats -> exact rationals, so symbolic cancellations (e.g. removable 1/r terms on the
    axis) are exact instead of leaving 1e-16 residues."""
    return sp.nsimplify(sp.sympify(e), rational=True)


def eval_value(v, pts) -> np.ndarray:
    """Evaluate a case value (float or expression string) at points (n, 2|3)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    zc = pts[:, 2] if pts.shape[1] > 2 else None
    if isinstance(v, str):
        # errstate: eval_expr strips builtins, so a numpy RuntimeWarning inside it would raise
        # KeyError('__import__') instead of returning inf / nan
        with np.errstate(all="ignore"):
            return eval_expr(v, pts[:, 0], pts[:, 1], zc)
    return np.full(len(pts), float(v))


def check_expr(e, s, pts, rtol: float = 1e-11) -> None:
    """Raise if the printed string ``s`` does not evaluate like the sympy expression ``e``."""
    with np.errstate(all="ignore"):
        ref = lambdify(e)(pts)
    got = eval_value(s, pts)
    ok = np.isfinite(ref)  # lambdify may hit removable singularities that the string avoids
    ref, got = ref[ok], got[ok]
    if not ok.any():
        return
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
             seed: int = 0, refine: int = 0) -> dict:
        """Write a gmsh mesh (order 1: tri3/tet4, order 2: tri6/tet10 with curved boundary
        nodes): unstructured at size h, then ``refine`` uniform splittings (gmsh puts the new
        boundary nodes on the CAD curves / surfaces), so a refinement study is nested with an
        exact ratio of 2. ``distort`` > 0 moves interior nodes randomly by up to distort * h
        (P2: midside nodes follow their edge and get an extra interior perturbation, i.e. curved
        interior edges). Returns {"path", "n_cells", "n_nodes", "h_rep", "min_quality"};
        cached on disk."""
        import gmsh

        from feacore.geometry.gmsh_tools import gmsh_session, set_uniform_size

        path = Path(path)
        side = path.with_suffix(".json")
        key = [self.name, h, order, distort, seed, refine]
        if path.exists() and side.exists():
            info = json.loads(side.read_text())
            if info.get("key") == key:
                return info
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(f"mms_{self.name}"):
            self.build()
            set_uniform_size(h)
            gmsh.model.mesh.generate(self.dim)
            for _ in range(refine):
                gmsh.model.mesh.refine()
            if order >= 2:
                gmsh.model.mesh.setOrder(2)
            q = _distort(self.dim, h / 2**refine, distort, seed) if distort > 0 else None
            etypes, etags, _ = gmsh.model.mesh.getElements(self.dim)
            n_cells = int(sum(len(t) for t in etags))
            tags = np.concatenate([np.asarray(t) for t in etags])
            qual = gmsh.model.mesh.getElementQualities(tags, "minSJ")
            n_nodes = len(gmsh.model.mesh.getNodes()[0])
            gmsh.write(str(path))
        info = {"key": key, "path": str(path), "n_cells": n_cells,
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


def _ramp(geom, others) -> sp.Expr:
    """phi = 0 and d(phi)/dn = 1 on ``geom``; phi = d(phi)/dn = 0 on the parallel planes /
    concentric arcs in ``others`` (so corrections for one boundary leave the others alone)."""
    if isinstance(geom, Plane):
        s_, c = XYZ[geom.axis], geom.value
        phi = s_ - c
        for o in others:
            if isinstance(o, Plane) and o.axis == geom.axis and o.value != c:
                phi = phi * ((s_ - o.value) / (c - o.value)) ** 2
        return phi
    r, R = sp.sqrt(x**2 + y**2), geom.radius
    phi = r - R
    for o in others:
        if isinstance(o, Arc) and o.radius != R:
            phi = phi * ((r - o.radius) / (R - o.radius)) ** 2
    return phi


def _numerically_zero(e, pts, scale) -> bool:
    if e == 0:
        return True
    with np.errstate(all="ignore"):
        v = lambdify(e)(pts)
    return bool(np.all(np.isfinite(v)) and np.max(np.abs(v)) <= 1e-13 * scale)


def _zero_shear(formulation, u, geom, E, nu, th, others, pts, scale):
    """Add phi * (correction) that cancels the tangential traction on ``geom``, leaving the
    displacement and the normal traction there unchanged. On a plane the shear is restricted
    to the plane first (smaller expressions); on an arc the field itself is used as the
    extension (phi = 0 on the arc makes the restriction unnecessary)."""
    _, mu = _lame(E, nu)
    s = elastic_state(formulation, u, E, nu, th)["tensor"]
    u = list(u)
    phi = _ramp(geom, others)
    if isinstance(geom, Plane):
        i, c = geom.axis, geom.value
        for j in range(len(u)):
            if j != i:
                tau = s[i][j].subs(XYZ[i], c)
                if not _numerically_zero(tau, pts, scale):
                    u[j] = u[j] - phi * tau / mu
        return u
    r = sp.sqrt(x**2 + y**2)
    ct, st = x / r, y / r
    tau = (s[1][1] - s[0][0]) * ct * st + s[0][1] * (ct**2 - st**2)  # s_r_theta
    if not _numerically_zero(tau, pts, scale):
        g = -phi * tau / mu
        u[0], u[1] = u[0] - g * st, u[1] + g * ct
    if len(u) == 3:
        tau_rz = s[0][2] * ct + s[1][2] * st
        if not _numerically_zero(tau_rz, pts, scale):
            u[2] = u[2] - phi * tau_rz / mu
    return u


def _remove_normal(u, planes):
    """u_i -= Lagrange interpolant (in x_i) of its traces on the symmetry planes of axis i."""
    u = list(u)
    by_axis: dict = {}
    for pl in planes:
        by_axis.setdefault(pl.axis, []).append(pl.value)
    for i, cs in by_axis.items():
        s_ = XYZ[i]
        corr = 0
        for k, ck in enumerate(cs):
            L = sp.S.One
            for m, cm in enumerate(cs):
                if m != k:
                    L = L * (s_ - cm) / (ck - cm)
            corr = corr + u[i].subs(s_, ck) * L
        u[i] = u[i] - corr
    return u


def _robin_fix(formulation, T, geom, k, hc, T_inf, others, pts, scale):
    """Add phi * (correction) so that k dT/dn + hc (T - T_inf) = 0 on ``geom`` (T unchanged)."""
    phi = _ramp(geom, others)
    if isinstance(geom, Plane):
        i, c = geom.axis, geom.value
        rho = (geom.sign * k * sp.diff(T, XYZ[i]) + hc * (T - T_inf)).subs(XYZ[i], c)
    else:
        r = sp.sqrt(x**2 + y**2)
        rho = geom.sign * k * (x * sp.diff(T, x) + y * sp.diff(T, y)) / r + hc * (T - T_inf)
    if _numerically_zero(rho, pts, scale):
        return T
    return T - phi * geom.sign * rho / k


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
    layout: dict = field(default_factory=dict)          # group -> BC kind used to build it

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
    for attempt in range(3):
        v = to_expr(e)
        if all(np.all(np.isfinite(eval_value(v, pts))) for pts in pts_list if len(pts)):
            break
        # removable 1/r singularities on the axis: expand / cancel so they disappear
        e = sp.expand(e) if attempt == 0 else sp.cancel(sp.together(e))
    else:
        raise ValueError(f"{key}: expression not finite on the domain (axis?): {v}")
    for pts in pts_list:
        if len(pts):
            check_expr(e, v, pts)
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
    u = [rationalize(c) for c in u]
    ncomp = len(u)
    if ncomp != (3 if formulation == "3d" else 2):
        raise ValueError("u must have 3 components in 3d, 2 otherwise")
    E_f, nu_f, alpha_f = float(E), float(nu), float(alpha)
    E, nu, alpha, T_ref_s = (rationalize(v) for v in (E, nu, alpha, T_ref))
    Tsym = rationalize(T) if T is not None else None
    th = alpha * (Tsym - T_ref_s) if Tsym is not None else sp.S.Zero
    geoms = domain.boundaries
    # 1) corrections so the field satisfies the special BCs exactly: normal displacement on
    #    symmetry planes first, then tangential traction on symmetry / pressure / roller groups
    special = {g: geoms[g] for g, sp_ in bcs.items() if _bc_kind(sp_)[0] in ("symmetry", "pressure", "roller")}
    sym_planes = [geoms[g] for g, sp_ in bcs.items() if _bc_kind(sp_)[0] == "symmetry"]
    if any(not isinstance(pl, Plane) for pl in sym_planes):
        raise ValueError("symmetry needs a plane")
    u = _remove_normal(u, sym_planes)
    cpts = np.concatenate([domain.sample(200, seed=11)] +
                          [domain.boundary_quadrature(g, 12)[0][:, :ncomp] for g in special])
    s0 = elastic_state(formulation, u, E, nu, th)["stress"]
    with np.errstate(all="ignore"):
        sc = max([float(np.nanmax(np.abs(lambdify(e)(cpts)))) for e in s0.values() if e != 0] + [1e-300])
    for group, geom in special.items():
        u = _zero_shear(formulation, u, geom, E, nu, th, [g for n_, g in special.items() if n_ != group],
                        cpts, sc)
    if domain.touches_axis:
        u = [sp.expand(c) for c in u]
    st = elastic_state(formulation, u, E, nu, th)
    sig = st["tensor"]
    pts_in = domain.sample(400, seed=7)
    pts_all = [pts_in] + ([_axis_points(domain)] if domain.touches_axis else [])
    smax = max([float(np.max(np.abs(lambdify(e)(pts_in)))) for e in st["stress"].values() if e != 0]
               + [float(E) * float(np.max(np.abs(lambdify(th)(pts_in)))), 1e-300])
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
                      material=Material(E=E_f, nu=nu_f, alpha=alpha_f), bcs=out_bcs, loads=loads,
                      exact_sym=exact, u_fields=u_fields, s_fields=s_fields, thickness=thickness,
                      T_ref=T_ref, energy_exact=energy, energy_physical=energy_phys,
                      reactions_exact=reactions, features=sorted(features),
                      description=description, expr_sizes=sizes,
                      layout={g: _bc_kind(v)[0] for g, v in bcs.items()})


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
    T = rationalize(T)
    k_f = float(k)
    k, h_r, Tinf_r = rationalize(k), rationalize(h_conv), rationalize(T_inf)
    geoms = domain.boundaries
    special = {g: geoms[g] for g, sp_ in bcs.items() if _bc_kind(sp_)[0] == "convection"}
    dim0 = 3 if formulation == "3d" else 2
    cpts = np.concatenate([domain.sample(200, seed=11)] +
                          [domain.boundary_quadrature(g, 12)[0][:, :dim0] for g in special])
    with np.errstate(all="ignore"):
        sc = float(np.nanmax(np.abs(lambdify(T)(cpts)))) * max(k_f, float(h_conv))
    for group, geom in special.items():
        T = _robin_fix(formulation, T, geom, k, h_r, Tinf_r,
                       [g for n_, g in special.items() if n_ != group], cpts, sc)
    if domain.touches_axis:
        T = sp.expand(T)
    st = heat_state(formulation, T, k)
    dim = 3 if formulation == "3d" else 2
    pts_in = domain.sample(400, seed=7)
    pts_all = [pts_in] + ([_axis_points(domain)] if domain.touches_axis else [])
    qmax = max([float(np.max(np.abs(lambdify(c)(pts_in)))) for c in st["q"]] + [k_f * float(np.max(np.abs(lambdify(T)(pts_in))))])
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
            res = flux_in - h_r * (Tinf_r - T)
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
                      material=Material(k=k_f), bcs=out_bcs, loads=loads, exact_sym=exact,
                      u_fields=["T"], s_fields=s_fields, features=sorted(features),
                      description=description, expr_sizes=sizes)


# --------------------------------------------------------------------------- exact solutions


def lame_displacement(formulation: str, a: float, b: float, p_i: float, p_o: float,
                      E: float, nu: float) -> list:
    """Lame thick cylinder (inner radius a, outer b, pressures p_i, p_o): plane stress, plane
    strain (also axisymmetric and 3d with u_z = 0). sigma_rr = A - B/r^2, sigma_tt = A + B/r^2."""
    A = (p_i * a**2 - p_o * b**2) / (b**2 - a**2)
    B = (p_i - p_o) * a**2 * b**2 / (b**2 - a**2)
    if formulation == "plane_stress":
        c1, c2 = (1 - nu) * A / E, (1 + nu) * B / E
    else:
        c1, c2 = (1 + nu) * (1 - 2 * nu) * A / E, (1 + nu) * B / E
    if formulation == "axisymmetric":
        return [c1 * x + c2 / x, sp.S.Zero]
    r2 = x**2 + y**2
    u = [c1 * x + c2 * x / r2, c1 * y + c2 * y / r2]
    return u + [sp.S.Zero] if formulation == "3d" else u


# Smooth, non-polynomial fields without symmetries (so neither P1 nor P2 is exact and no
# error component vanishes by accident). Axisymmetric fields touching the axis are regular
# there: u_r odd in r, u_z and T even in r.
U_SQUARE = (0.01 * (sp.sin(1.7 * x + 0.4) * sp.exp(0.8 * y) + 0.5 * x * sp.cos(1.3 * y - 0.2)),
            0.01 * (sp.cos(1.1 * x + 0.3 * y + 0.5) + 0.6 * sp.sin(1.9 * y) * sp.exp(-0.7 * x)))
U_ANNULUS = (0.01 * (sp.exp(0.6 * x) * sp.sin(1.4 * y + 0.3) + 0.4 * x * y),
             0.01 * (sp.cos(1.2 * x - 0.5 * y) + 0.3 * y * sp.sin(2.1 * x)))
U_CUBE = (0.01 * sp.sin(1.2 * x + 0.3) * sp.exp(0.5 * y) * sp.cos(0.8 * z + 0.1),
          0.01 * sp.cos(0.9 * x - 0.4 * z) * sp.sin(1.3 * y + 0.2) + 0.004 * x * z,
          0.01 * sp.exp(0.4 * x - 0.6 * y) * sp.sin(1.1 * z + 0.5))
U_AXI_RING = (0.01 * sp.sin(1.3 * x + 0.2) * sp.exp(0.5 * y),
              0.01 * sp.cos(0.9 * x - 0.7 * y + 0.3))
U_AXI_SOLID = (0.01 * x * sp.exp(-0.5 * x**2) * sp.cos(1.2 * y + 0.3),
               0.01 * sp.cos(0.8 * x**2) * sp.sin(1.1 * y + 0.4))
T_THERMAL = 20 + 40 * (sp.sin(1.3 * x + 0.5) * sp.cos(0.9 * y) + 0.5 * sp.exp(0.4 * y - 0.3 * x))
T_SQUARE = 20 + 5 * sp.sin(1.4 * x + 0.3) * sp.exp(0.6 * y) + 3 * sp.cos(1.1 * y - 0.7 * x)
T_ANNULUS = 20 + 4 * sp.exp(0.7 * x) * sp.cos(1.2 * y + 0.2) + 2 * sp.sin(1.6 * x * y)
T_AXI_RING = 20 + 5 * sp.sin(1.2 * x + 0.4) * sp.cos(0.9 * y + 0.3) + 2 * sp.exp(0.5 * y)
T_AXI_SOLID = 20 + 5 * sp.exp(-0.6 * x**2) * sp.cos(1.1 * y + 0.2) + 2 * sp.cos(0.9 * x**2)
T_CUBE = (20 + 5 * sp.sin(1.2 * x + 0.3) * sp.cos(0.8 * y - 0.2 * z) * sp.exp(0.4 * z)
          + 2 * sp.cos(0.7 * x + 1.1 * y))

# Default refinement studies: unstructured base mesh at H0, then uniform splitting (ratio 2).
H0_2D, LEVELS_2D = 0.25, 4      # 2D: 4 levels, finest h ~ 1/32
H0_3D, LEVELS_3D = 0.5, {1: 4, 2: 3}  # 3D: P1 4 levels, P2 3 levels (small sizes)
# Per-problem base size: the thin quarter annular cylinder needs h0 = 0.25 (with 0.5 the P1 chord
# sag on r = 1 puts sample points ~1.7% of the diagonal outside the mesh, beyond skfem's
# probe tolerance, and P2 is pre-asymptotic).
H0_OVERRIDE = {"E-3d-annulus3d-lame": 0.25}
# Independent (non-nested) unstructured meshes, used for the robustness variant.
SIZES_2D = [0.25, 0.125, 0.0625, 0.03125]
SIZES_3D = [0.5, 1 / 3, 2 / 9, 4 / 27]

D = DOMAINS


def _conv_specs() -> dict:
    """id -> (factory, slow, note)."""
    E, nu = 1000.0, 0.3
    sq, an, cu, a3, ar, as_ = (D[k] for k in ("square", "annulus", "cube", "annulus3d", "axi_ring", "axi_solid"))
    mixed2 = {"left": "dirichlet", "bottom": "dirichlet", "right": "pressure", "top": "traction"}
    lame = {"inner": "pressure", "outer": "free", "bottom": "symmetry", "left": "symmetry"}
    specs = {
        "E-ps-square-dirichlet": (lambda: elasticity_mms(
            "E-ps-square-dirichlet", "plane_stress", sq, U_SQUARE,
            {g: "dirichlet" for g in ("left", "right", "bottom", "top")}, E=E, nu=nu, thickness=0.5,
            description="plane stress, all-Dirichlet, body force, thickness 0.5"), False),
        "E-ps-square-mixed": (lambda: elasticity_mms(
            "E-ps-square-mixed", "plane_stress", sq, U_SQUARE, mixed2, E=E, nu=nu, thickness=0.5,
            description="plane stress, Dirichlet + pressure(expr) + traction(expr), body force"), False),
        "E-pe-square-mixed": (lambda: elasticity_mms(
            "E-pe-square-mixed", "plane_strain", sq, U_SQUARE, mixed2, E=E, nu=nu,
            description="plane strain, Dirichlet + pressure(expr) + traction(expr), body force"), False),
        "E-pe-annulus-mixed": (lambda: elasticity_mms(
            "E-pe-annulus-mixed", "plane_strain", an, U_ANNULUS,
            {"inner": "dirichlet", "bottom": "symmetry", "left": "traction", "outer": "pressure"},
            E=E, nu=nu, description="plane strain, curved Dirichlet + curved pressure(expr) + "
            "symmetry + traction, body force"), False),
        "E-ps-annulus-lame": (lambda: elasticity_mms(
            "E-ps-annulus-lame", "plane_stress", an,
            lame_displacement("plane_stress", 0.5, 1.0, 10.0, 0.0, E, nu), lame, E=E, nu=nu,
            thickness=0.5, description="Lame cylinder (exact), plane stress, constant pressure "
            "on the curved inner arc, symmetry"), False),
        "E-pe-annulus-lame": (lambda: elasticity_mms(
            "E-pe-annulus-lame", "plane_strain", an,
            lame_displacement("plane_strain", 0.5, 1.0, 10.0, 0.0, E, nu), lame, E=E, nu=nu,
            description="Lame cylinder (exact), plane strain, constant pressure on the curved "
            "inner arc, symmetry"), False),
        "E-pe-square-thermal": (lambda: elasticity_mms(
            "E-pe-square-thermal", "plane_strain", sq, U_SQUARE, mixed2, E=E, nu=nu, alpha=2e-4,
            T=T_THERMAL, T_ref=20.0, description="plane strain + thermal strain (T expr), "
            "Dirichlet + pressure + traction, body force"), False),
        "E-ax-ring-mixed": (lambda: elasticity_mms(
            "E-ax-ring-mixed", "axisymmetric", ar, U_AXI_RING,
            {"left": "dirichlet", "bottom": "symmetry", "right": "traction", "top": "pressure"},
            E=E, nu=nu, description="axisymmetric away from the axis, Dirichlet + symmetry + "
            "traction + pressure, body force"), False),
        "E-ax-solid-mixed": (lambda: elasticity_mms(
            "E-ax-solid-mixed", "axisymmetric", as_, U_AXI_SOLID,
            {"left": "none", "bottom": "dirichlet", "right": "traction", "top": "pressure"},
            E=E, nu=nu, description="axisymmetric touching the axis (r = 0 natural), Dirichlet + "
            "traction + pressure, body force"), False),
        "E-ax-ring-lame": (lambda: elasticity_mms(
            "E-ax-ring-lame", "axisymmetric", ar,
            lame_displacement("axisymmetric", 0.5, 1.5, 10.0, 0.0, E, nu),
            {"left": "pressure", "right": "free", "bottom": "symmetry", "top": "symmetry"},
            E=E, nu=nu, description="Lame thick cylinder (exact), axisymmetric, u_z = 0"), False),
        "E-3d-cube-mixed": (lambda: elasticity_mms(
            "E-3d-cube-mixed", "3d", cu, U_CUBE,
            {"x0": "dirichlet", "y0": "symmetry", "z0": "dirichlet", "x1": "traction",
             "y1": "traction", "z1": "pressure"}, E=E, nu=nu,
            description="3D, Dirichlet + symmetry + traction + pressure, body force"), True),
        "E-3d-annulus3d-lame": (lambda: elasticity_mms(
            "E-3d-annulus3d-lame", "3d", a3,
            lame_displacement("3d", 0.5, 1.0, 10.0, 0.0, E, nu),
            {**lame, "z0": "symmetry", "z1": "symmetry"}, E=E, nu=nu,
            description="Lame cylinder (exact) in 3D, curved faces, plane strain via symmetry"), True),
        "H-2d-square-mixed": (lambda: heat_mms(
            "H-2d-square-mixed", "2d", sq, T_SQUARE,
            {"left": "temperature", "bottom": "temperature", "right": "flux", "top": "convection"},
            description="2D heat, temperature + flux(expr) + convection, source"), False),
        "H-2d-annulus-mixed": (lambda: heat_mms(
            "H-2d-annulus-mixed", "2d", an, T_ANNULUS,
            {"inner": "temperature", "outer": "convection", "bottom": "flux", "left": "temperature"},
            description="2D heat, curved temperature + curved convection + flux, source"), False),
        "H-2d-annulus-log": (lambda: heat_mms(
            "H-2d-annulus-log", "2d", an, 100 - 40 * sp.log(sp.sqrt(x**2 + y**2) / 0.5) / math.log(2.0),
            {"inner": "temperature", "outer": "flux", "bottom": "insulated", "left": "insulated"},
            description="radial conduction T = a + b ln r (exact), constant data, curved flux"), False),
        "H-ax-ring-mixed": (lambda: heat_mms(
            "H-ax-ring-mixed", "axisymmetric", ar, T_AXI_RING,
            {"left": "temperature", "right": "convection", "bottom": "flux", "top": "temperature"},
            description="axisymmetric heat away from the axis, source"), False),
        "H-ax-solid-mixed": (lambda: heat_mms(
            "H-ax-solid-mixed", "axisymmetric", as_, T_AXI_SOLID,
            {"left": "none", "right": "temperature", "bottom": "flux", "top": "convection"},
            description="axisymmetric heat touching the axis, source"), False),
        "H-3d-cube-mixed": (lambda: heat_mms(
            "H-3d-cube-mixed", "3d", cu, T_CUBE,
            {"x0": "temperature", "x1": "flux", "y0": "flux", "y1": "convection",
             "z0": "temperature", "z1": "flux"}, description="3D heat, source"), True),
    }
    return specs


CONVERGENCE_IDS = list(_conv_specs())


@functools.lru_cache(maxsize=None)
def convergence_problem(pid: str) -> MMSProblem:
    return _conv_specs()[pid][0]()


def convergence_catalog() -> list[dict]:
    out = []
    for pid, (_, slow) in _conv_specs().items():
        dim = 3 if "-3d-" in pid else 2
        out.append({"id": pid, "slow": slow, "dim": dim,
                    "levels": {k: default_levels(dim, k) for k in (1, 2)}})
    return out


def _patch_specs() -> dict:
    """id -> (factory, mesh h, stress scale or None)."""
    E, nu = 1000.0, 0.3
    sq, cu, ar, as_ = D["square"], D["cube"], D["axi_ring"], D["axi_solid"]
    a, b, c = 2e-3, -1e-3, 1.5e-3
    G2 = [[2e-3, 1.5e-3], [-0.5e-3, -1e-3]]
    G3 = [[2e-3, 1.5e-3, -0.7e-3], [-0.5e-3, -1e-3, 0.4e-3], [0.9e-3, 0.3e-3, 1.2e-3]]
    u2 = [1e-3 + G2[0][0] * x + G2[0][1] * y, -2e-3 + G2[1][0] * x + G2[1][1] * y]
    u3 = [1e-3 * (i + 1) + sum(G3[i][j] * XYZ[j] for j in range(3)) for i in range(3)]
    dT, alpha = 50.0, 1e-4
    specs = {}
    for f in ("plane_stress", "plane_strain"):
        tag = "ps" if f == "plane_stress" else "pe"
        specs[f"P-{tag}-const-roller"] = (lambda f=f: elasticity_mms(
            "patch", f, sq, [a * x, b * y],
            {"left": "roller", "bottom": "roller", "right": "traction", "top": "force"},
            E=E, nu=nu, thickness=0.5), 0.2, None)
        specs[f"P-{tag}-const-sym"] = (lambda f=f: elasticity_mms(
            "patch", f, sq, [a * x, b * y],
            {"left": "symmetry", "bottom": "symmetry", "right": "pressure", "top": "pressure"},
            E=E, nu=nu), 0.2, None)
        specs[f"P-{tag}-general"] = (lambda f=f: elasticity_mms(
            "patch", f, sq, u2, {"left": "dirichlet", "bottom": "dirichlet", "right": "traction",
                                 "top": "traction"}, E=E, nu=nu), 0.2, None)
        fac = 1.0 if f == "plane_stress" else 1 + nu
        specs[f"P-{tag}-thermal"] = (lambda f=f, fac=fac: elasticity_mms(
            "patch", f, sq, [fac * alpha * dT * x, fac * alpha * dT * y],
            {"left": "symmetry", "bottom": "symmetry", "right": "free", "top": "free"},
            E=E, nu=nu, alpha=alpha, T=20.0 + dT, T_ref=20.0), 0.2, E * alpha * dT)
    specs["P-3d-const-roller"] = (lambda: elasticity_mms(
        "patch", "3d", cu, [a * x, b * y, c * z],
        {"x0": "roller", "y0": "roller", "z0": "roller", "x1": "traction", "y1": "force",
         "z1": "pressure"}, E=E, nu=nu), 0.35, None)
    specs["P-3d-const-sym"] = (lambda: elasticity_mms(
        "patch", "3d", cu, [a * x, b * y, c * z],
        {"x0": "symmetry", "y0": "symmetry", "z0": "symmetry", "x1": "pressure", "y1": "pressure",
         "z1": "traction"}, E=E, nu=nu), 0.35, None)
    specs["P-3d-general"] = (lambda: elasticity_mms(
        "patch", "3d", cu, u3, {"x0": "dirichlet", "y0": "dirichlet", "z0": "dirichlet",
                                "x1": "traction", "y1": "traction", "z1": "traction"},
        E=E, nu=nu), 0.35, None)
    specs["P-3d-thermal"] = (lambda: elasticity_mms(
        "patch", "3d", cu, [alpha * dT * x, alpha * dT * y, alpha * dT * z],
        {"x0": "symmetry", "y0": "symmetry", "z0": "symmetry", "x1": "free", "y1": "free",
         "z1": "free"}, E=E, nu=nu, alpha=alpha, T=20.0 + dT, T_ref=20.0), 0.35, E * alpha * dT)
    for dom, tag in ((ar, "ring"), (as_, "solid")):
        inner = "none" if dom.touches_axis else "pressure"
        specs[f"P-ax-{tag}-const"] = (lambda dom=dom, inner=inner: elasticity_mms(
            "patch", "axisymmetric", dom, [a * x, c * y],
            {"left": inner, "bottom": "symmetry", "right": "traction", "top": "force"},
            E=E, nu=nu), 0.2, None)
        specs[f"P-ax-{tag}-general"] = (lambda dom=dom: elasticity_mms(
            "patch", "axisymmetric", dom, [a * x, c * y + 1e-3],
            {"left": "none" if dom.touches_axis else "dirichlet", "bottom": "dirichlet",
             "right": "traction", "top": "traction"}, E=E, nu=nu), 0.2, None)
        specs[f"P-ax-{tag}-thermal"] = (lambda dom=dom: elasticity_mms(
            "patch", "axisymmetric", dom, [alpha * dT * x, alpha * dT * y],
            {"left": "none" if dom.touches_axis else "free", "bottom": "symmetry", "right": "free",
             "top": "free"}, E=E, nu=nu, alpha=alpha, T=20.0 + dT, T_ref=20.0), 0.2, E * alpha * dT)
    # heat: k = 2.5, h = 10, T_inf = 48.75 makes T = 30 + 15 s satisfy the Robin condition at s = 1
    specs["P-h2d-const"] = (lambda: heat_mms(
        "patch", "2d", sq, 30 + 15 * x, {"left": "temperature", "right": "convection",
                                         "bottom": "insulated", "top": "insulated"},
        k=2.5, h_conv=10.0, T_inf=48.75), 0.2, None)
    specs["P-h2d-general"] = (lambda: heat_mms(
        "patch", "2d", sq, 30 + 15 * x - 8 * y, {"left": "temperature", "bottom": "temperature",
                                                 "right": "flux", "top": "flux"}, k=2.5), 0.2, None)
    specs["P-h3d-const"] = (lambda: heat_mms(
        "patch", "3d", cu, 30 + 15 * z, {"z0": "temperature", "z1": "convection", "x0": "insulated",
                                         "x1": "insulated", "y0": "insulated", "y1": "insulated"},
        k=2.5, h_conv=10.0, T_inf=48.75), 0.35, None)
    specs["P-h3d-general"] = (lambda: heat_mms(
        "patch", "3d", cu, 30 + 15 * x - 8 * y + 5 * z,
        {"x0": "temperature", "y0": "temperature", "z0": "temperature", "x1": "flux", "y1": "flux",
         "z1": "flux"}, k=2.5), 0.35, None)
    for dom, tag in ((ar, "ring"), (as_, "solid")):
        side = "none" if dom.touches_axis else "insulated"
        specs[f"P-hax-{tag}-const"] = (lambda dom=dom, side=side: heat_mms(
            "patch", "axisymmetric", dom, 30 + 15 * y,
            {"bottom": "temperature", "top": "convection", "left": side, "right": "insulated"},
            k=2.5, h_conv=10.0, T_inf=48.75), 0.2, None)
        specs[f"P-hax-{tag}-general"] = (lambda dom=dom: heat_mms(
            "patch", "axisymmetric", dom, 30 + 15 * y,
            {"right": "temperature", "bottom": "flux", "top": "flux",
             "left": "none" if dom.touches_axis else "insulated"}, k=2.5), 0.2, None)
    return specs


PATCH_IDS = list(_patch_specs())


@functools.lru_cache(maxsize=None)
def patch_problem(pid: str) -> MMSProblem:
    p = _patch_specs()[pid][0]()
    p.name = pid
    return p


def patch_catalog() -> list[dict]:
    return [{"id": pid, "h": h, "s_scale": ss, "dim": 3 if "3d" in pid else 2}
            for pid, (_, h, ss) in _patch_specs().items()]


# --------------------------------------------------------------------------- runners


class FeatureMissing(Exception):
    """The backend raised NotImplementedError (or is not importable): skip, don't fail."""


def _solve(case: Case, backend: str, solver: Callable | None):
    if solver is not None:
        return solver(case)
    import importlib

    from feacore.solver import BACKENDS, solve

    try:
        importlib.import_module(BACKENDS[backend])
    except ImportError as e:
        raise FeatureMissing(f"backend {backend!r} not importable: {e}") from e
    try:
        return solve(case, backend=backend)
    except NotImplementedError as e:
        raise FeatureMissing(f"{backend}: {e}") from e


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(a.T).T ** 2, axis=1)))) if a.size else 0.0


def _probe(res, fields, pts) -> np.ndarray:
    return np.stack([np.asarray(res.probe(f, pts), dtype=float).reshape(-1) for f in fields], axis=1)


def observed_orders(h: Sequence[float], e: Sequence[float]) -> list:
    out = []
    for i in range(1, len(h)):
        if e[i] is None or e[i - 1] is None or e[i] <= 0 or e[i - 1] <= 0:
            out.append(None)
        else:
            out.append(math.log(e[i - 1] / e[i]) / math.log(h[i - 1] / h[i]))
    return out


def _finest_order(h, e, floor):
    """Order of the finest pair whose errors are both above the round-off floor."""
    p = observed_orders(h, e)
    for i in range(len(p) - 1, -1, -1):
        if p[i] is not None and e[i + 1] > floor:
            return p[i], i
    return None, None


def clamped_free_exponent(formulation: str, nu: float, omega: float = math.pi / 2) -> float:
    """Leading exponent lambda of u ~ r^lambda at a wedge of angle omega with one face clamped
    and one traction-loaded (Williams 1952): kappa sin^2(lambda omega) = (kappa+1)^2/4 -
    lambda^2 sin^2(omega), kappa = 3 - 4 nu (plane strain, axisymmetric, 3D edges) or
    (3 - nu)/(1 + nu) (plane stress). lambda ~ 0.711 for nu = 0.3, omega = 90 deg."""
    from scipy.optimize import brentq

    kap = (3 - nu) / (1 + nu) if formulation == "plane_stress" else 3 - 4 * nu
    f = lambda lam: kap * math.sin(lam * omega) ** 2 - (kap + 1) ** 2 / 4 + lam**2 * math.sin(omega) ** 2  # noqa: E731
    return brentq(f, 0.5, 0.99)


def _adjacent(dom: Domain) -> list[tuple[str, str]]:
    g = list(dom.boundaries)
    if dom.kind == "box" and dom.dim == 2:
        return [("left", "bottom"), ("bottom", "right"), ("right", "top"), ("top", "left")]
    if dom.kind == "box":
        return [(a, b) for i, a in enumerate(g) for b in g[i + 1:] if a[0] != b[0]]
    ring = [("inner", "bottom"), ("bottom", "outer"), ("outer", "left"), ("left", "inner")]
    if dom.kind == "annulus3d":
        ring += [(zf, s_) for zf in ("z0", "z1") for s_ in ("inner", "outer", "bottom", "left")]
    return ring


def energy_order_theory(problem: MMSProblem, order: int) -> float:
    """2k for smooth problems. A 90-degree corner (edge in 3D) where a Dirichlet face meets a
    traction / pressure face gives the dual problem an r^lambda singularity, which caps the
    strain-energy order at k + 1 + lambda (< 2k for P2). The primal u / stress orders are not
    affected (the exact solution is smooth)."""
    if problem.physics != "elasticity":
        return 2.0 * order
    D_, N_ = {"dirichlet", "fixed"}, {"traction", "pressure", "force", "free"}
    lay = problem.layout
    dn = any((lay.get(a) in D_ and lay.get(b) in N_) or (lay.get(a) in N_ and lay.get(b) in D_)
             for a, b in _adjacent(problem.domain))
    if not dn:
        return 2.0 * order
    lam = clamped_free_exponent(problem.formulation, problem.material.nu)
    return min(2.0 * order, order + 1.0 + lam)


def default_levels(dim: int, order: int) -> tuple[float, int]:
    return (H0_3D, LEVELS_3D[order]) if dim == 3 else (H0_2D, LEVELS_2D)


def run_convergence(problem: MMSProblem, order: int, backend: str = "skfem",
                    sizes: Sequence[float] | None = None, workdir: str | Path | None = None,
                    n_points: int | None = None, seed: int = 2024, geom_order: int | None = None,
                    solver: Callable | None = None, tol: float = 0.3, verbose: bool = False,
                    options: dict | None = None, h0: float | None = None,
                    levels: int | None = None) -> dict:
    """Refinement study through the public API. Default: nested family (unstructured base mesh
    of size h0, uniformly refined ``levels - 1`` times). ``sizes``: independent unstructured
    meshes instead. Raises FeatureMissing when the backend raises NotImplementedError."""
    dom = problem.domain
    d_h0, d_lv = default_levels(dom.dim, order)
    if sizes is not None:
        plan = [(float(hh), 0) for hh in sizes]
        family = "independent"
    else:
        h0 = h0 or H0_OVERRIDE.get(problem.name, d_h0)
        plan = [(h0, i) for i in range(levels or (3 if problem.name in H0_OVERRIDE else d_lv))]
        family = "nested"
    geom_order = order if geom_order is None else geom_order
    workdir = Path(workdir or DEFAULT_WORKDIR)
    tag = f"{problem.name}_P{order}g{geom_order}_{backend}_{family}"
    case_dir = workdir / "cases" / tag
    npts = n_points or (2000 if dom.dim == 2 else 2500)
    pts = dom.sample(npts, seed)
    ex_u = np.stack([problem.exact(f, pts) for f in problem.u_fields], axis=1)
    ex_s = np.stack([problem.exact(f, pts) for f in problem.s_fields], axis=1)
    nu_, ns_ = _rms(ex_u), _rms(ex_s)
    levels = []
    for i, (h, nref) in enumerate(plan):
        minfo = dom.mesh(workdir / "meshes" / f"{dom.name}_h{h:.5g}_r{nref}_g{geom_order}.msh", h,
                         order=geom_order, refine=nref)
        case = problem.case(minfo["path"], order, options)
        case.save(case_dir / f"level{i}.json")
        t0 = time.perf_counter()
        res = _solve(case, backend, solver)
        t_solve = time.perf_counter() - t0
        uh = _probe(res, problem.u_fields, pts)
        sh = _probe(res, problem.s_fields, pts)
        lev = {"h": h / 2**nref, "h_rep": minfo["h_rep"], "n_cells": minfo["n_cells"], "n_nodes": minfo["n_nodes"],
               "mesh": minfo["path"], "case": str(case_dir / f"level{i}.json"),
               "t_solve": t_solve, "info": {k: v for k, v in asdict(res.info).items() if k != "warnings"},
               "warnings": list(res.info.warnings),
               "e_u": _rms(uh - ex_u) / nu_, "e_s": _rms(sh - ex_s) / ns_,
               "e_fields": {f: _rms(uh[:, j] - ex_u[:, j]) / nu_ for j, f in enumerate(problem.u_fields)}
               | {f: _rms(sh[:, j] - ex_s[:, j]) / ns_ for j, f in enumerate(problem.s_fields)},
               "finite": bool(np.all(np.isfinite(uh)) and np.all(np.isfinite(sh)))}
        if problem.energy_exact:
            try:
                W = float(res.evaluate(QoI(name="strain_energy", type="strain_energy")))
                lev["W"] = W
                lev["e_W"] = abs(W - problem.energy_exact) / abs(problem.energy_exact)
                if problem.energy_physical is not None:
                    lev["e_W_physical"] = abs(W - problem.energy_physical) / abs(problem.energy_physical)
            except NotImplementedError as e:
                lev["W_error"] = f"NotImplementedError: {e}"
            except Exception as e:  # noqa: BLE001
                lev["W_error"] = f"{type(e).__name__}: {e}"
        levels.append(lev)
        if verbose:
            print(f"  [{tag}] h={h:.4g} cells={minfo['n_cells']} dofs={res.info.n_dofs} "
                  f"e_u={lev['e_u']:.3e} e_s={lev['e_s']:.3e} e_W={lev.get('e_W', float('nan')):.3e} "
                  f"t={t_solve:.2f}s", flush=True)
    hs = [l["h_rep"] for l in levels]
    k = order
    theory = {"u": k + 1.0, "s": float(k), "W": energy_order_theory(problem, k)}
    floors = {"u": 1e-10, "s": 1e-10, "W": 1e-11}
    out = {"problem": problem.name, "physics": problem.physics, "formulation": problem.formulation,
           "domain": dom.name, "order": order, "geom_order": geom_order, "backend": backend,
           "family": family, "features": problem.features, "description": problem.description,
           "theory": theory,
           "levels": levels, "orders": {}, "observed": {}, "pass": {}}
    for key in ("u", "s", "W"):
        e = [l.get(f"e_{key}") for l in levels]
        if any(v is None for v in e):
            continue
        out["orders"][key] = observed_orders(hs, e)
        p, _ = _finest_order(hs, e, floors[key])
        out["observed"][key] = p
        lh, le = np.log(hs[-3:]), np.log(np.maximum(e[-3:], 1e-300))
        out.setdefault("slope3", {})[key] = float(np.polyfit(lh, le, 1)[0]) if len(hs) >= 3 else None
        out["pass"][key] = bool(p is not None and p >= theory[key] - tol)
    if problem.physics == "elasticity" and problem.energy_physical is not None:
        # thermal case: 0.5 u.K.u is judged informationally only (contract ambiguity)
        out["pass"].pop("W", None)
    finite = all(l["finite"] for l in levels)
    out["verdict"] = "pass" if finite and out["pass"] and all(out["pass"].values()) else "fail"
    return out


def run_patch(problem: MMSProblem, order: int, backend: str = "skfem", h: float = 0.2,
              distort: float = 0.3, seed: int = 3, workdir: str | Path | None = None,
              solver: Callable | None = None, s_scale: float | None = None,
              tol_u: float = 1e-9, tol_s: float = 1e-8) -> dict:
    """Patch test: the linear exact field must be reproduced on a distorted mesh (P2: curved
    interior edges) at the nodes and at interior probe points; recovered stresses / fluxes
    must be the exact constants; strain energy exact."""
    dom = problem.domain
    workdir = Path(workdir or DEFAULT_WORKDIR)
    minfo = dom.mesh(workdir / "meshes" / f"{dom.name}_h{h:.5g}_g{order}_dist{distort}_s{seed}.msh",
                     h, order=order, distort=distort, seed=seed)
    case = problem.case(minfo["path"], order)
    cpath = workdir / "cases" / f"{problem.name}_P{order}_{backend}.json"
    case.save(cpath)
    res = _solve(case, backend, solver)
    pts = dom.sample(300, seed=99)
    ex_u = np.stack([problem.exact(f, pts) for f in problem.u_fields], axis=1)
    ex_s = np.stack([problem.exact(f, pts) for f in problem.s_fields], axis=1)
    u_scale = float(np.max(np.abs(ex_u)))
    s_scale = s_scale or float(np.max(np.abs(ex_s)))
    out = {"problem": problem.name, "physics": problem.physics, "formulation": problem.formulation,
           "domain": dom.name, "order": order, "backend": backend, "features": problem.features,
           "mesh": minfo["path"], "case": str(cpath), "min_quality": minfo["min_quality"],
           "n_cells": minfo["n_cells"]}
    err_nodal = 0.0
    for f in problem.u_fields:
        coords, vals = res.nodal(f)
        err_nodal = max(err_nodal, float(np.max(np.abs(np.asarray(vals) - problem.exact(f, coords)))))
    out["err_u_nodal"] = err_nodal / u_scale
    out["err_u_probe"] = float(np.max(np.abs(_probe(res, problem.u_fields, pts) - ex_u))) / u_scale
    sh = _probe(res, problem.s_fields, pts)
    out["err_s_probe"] = float(np.max(np.abs(sh - ex_s))) / s_scale
    out["worst_s_field"] = problem.s_fields[int(np.argmax(np.max(np.abs(sh - ex_s), axis=0)))]
    checks = {"u_nodal": out["err_u_nodal"] <= tol_u, "u_probe": out["err_u_probe"] <= tol_u,
              "s_probe": out["err_s_probe"] <= tol_s}
    if problem.energy_exact and problem.energy_physical is None:
        try:
            W = float(res.evaluate(QoI(name="strain_energy", type="strain_energy")))
            out["err_W"] = abs(W - problem.energy_exact) / abs(problem.energy_exact)
            checks["W"] = out["err_W"] <= tol_s
        except NotImplementedError as e:
            out["W_error"] = f"NotImplementedError: {e}"
    out["checks"] = checks
    out["verdict"] = "pass" if all(checks.values()) else "fail"
    return out


def error_report(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


# --------------------------------------------------------------------------- self-check


def _fd_grad(f, pts, h=1e-5):
    """Central-difference gradient of f(points) -> (n,) at pts (n, d)."""
    g = np.zeros_like(pts)
    for j in range(pts.shape[1]):
        dp = np.zeros(pts.shape[1])
        dp[j] = h
        g[:, j] = (f(pts + dp) - f(pts - dp)) / (2 * h)
    return g


def _load_value(problem: MMSProblem, ltype: str):
    for l in problem.loads:
        if l.type == ltype:
            return l.value
    return None


def self_check(problem: MMSProblem, n: int = 60, seed: int = 5) -> dict:
    """Independent check of an MMS problem *as the solver sees it* (the case's expression
    strings, evaluated by ``feacore.case.eval_expr``) against its exact fields, by central finite
    differences and an independent numpy constitutive law. Returns {check: max relative
    residual}; everything should be ~1e-6 or below (finite-difference accuracy)."""
    dom = problem.domain
    pts = dom.sample(n, seed=seed)
    m = problem.material
    out: dict = {}
    ex = problem.exact
    if problem.physics == "elasticity":
        nc = 3 if problem.formulation == "3d" else 2
        uf = problem.u_fields
        G = np.stack([_fd_grad(lambda p, f=f: ex(f, p), pts) for f in uf], axis=1)  # (n, i, j)
        lam = m.E * m.nu / ((1 + m.nu) * (1 - 2 * m.nu))
        mu = m.E / (2 * (1 + m.nu))
        Tv = _load_value(problem, "temperature")
        th = m.alpha * (eval_value(Tv, pts) - problem.T_ref) if Tv is not None else np.zeros(n)
        eps = 0.5 * (G + np.transpose(G, (0, 2, 1)))
        f = problem.formulation
        if f == "3d":
            tr = np.trace(eps, axis1=1, axis2=2)
            sig = 2 * mu * eps + ((lam * tr - (3 * lam + 2 * mu) * th)[:, None, None] * np.eye(3))
            ref = {"sxx": sig[:, 0, 0], "syy": sig[:, 1, 1], "szz": sig[:, 2, 2],
                   "sxy": sig[:, 0, 1], "syz": sig[:, 1, 2], "sxz": sig[:, 0, 2]}
        elif f == "plane_strain":
            tr = eps[:, 0, 0] + eps[:, 1, 1]
            k3 = 3 * lam + 2 * mu
            ref = {"sxx": lam * tr + 2 * mu * eps[:, 0, 0] - k3 * th,
                   "syy": lam * tr + 2 * mu * eps[:, 1, 1] - k3 * th,
                   "sxy": 2 * mu * eps[:, 0, 1], "szz": lam * tr - k3 * th}
        elif f == "plane_stress":
            # independent route: 3D law with s_zz = 0 solved for eps_zz
            ezz = -(lam * (eps[:, 0, 0] + eps[:, 1, 1]) - (3 * lam + 2 * mu) * th) / (lam + 2 * mu)
            tr = eps[:, 0, 0] + eps[:, 1, 1] + ezz
            k3 = 3 * lam + 2 * mu
            ref = {"sxx": lam * tr + 2 * mu * eps[:, 0, 0] - k3 * th,
                   "syy": lam * tr + 2 * mu * eps[:, 1, 1] - k3 * th, "sxy": 2 * mu * eps[:, 0, 1]}
        else:
            r = pts[:, 0]
            ett = ex("ux", pts) / r
            tr = eps[:, 0, 0] + eps[:, 1, 1] + ett
            k3 = 3 * lam + 2 * mu
            ref = {"sxx": lam * tr + 2 * mu * eps[:, 0, 0] - k3 * th,
                   "syy": lam * tr + 2 * mu * eps[:, 1, 1] - k3 * th,
                   "sxy": 2 * mu * eps[:, 0, 1], "stt": lam * tr + 2 * mu * ett - k3 * th}
        ssc = max(max(float(np.max(np.abs(ex(fl, pts)))) for fl in problem.s_fields),
                  m.E * float(np.max(np.abs(th))), 1e-300)
        out["constitutive"] = max(float(np.max(np.abs(ref[fl] - ex(fl, pts)))) for fl in problem.s_fields) / ssc
        # equilibrium: div sigma + b = 0 with b from the case strings
        bv = _load_value(problem, "body_force")
        b = np.stack([eval_value(v, pts) for v in bv], axis=1) if bv is not None else np.zeros((n, nc))
        S = {fl: (lambda p, fl=fl: ex(fl, p)) for fl in problem.s_fields}
        zero = lambda p: np.zeros(len(p))  # noqa: E731
        comp = {(0, 0): "sxx", (1, 1): "syy", (0, 1): "sxy", (1, 0): "sxy", (2, 2): "szz",
                (1, 2): "syz", (2, 1): "syz", (0, 2): "sxz", (2, 0): "sxz"}
        sij = lambda i, j: S.get(comp[(i, j)], zero)  # noqa: E731
        div = np.zeros((n, nc))
        for i in range(nc):
            for j in range(nc):
                div[:, i] += _fd_grad(sij(i, j), pts)[:, j]
        if f == "axisymmetric":
            r = pts[:, 0]
            div[:, 0] += (ex("sxx", pts) - ex("stt", pts)) / r
            div[:, 1] += ex("sxy", pts) / r
        out["equilibrium"] = float(np.max(np.abs(div + b))) / (ssc / dom.length_scale)
        # boundary data
        worst = 0.0
        for bc in problem.bcs:
            bq, bw = dom.boundary_quadrature(bc.group, 8)
            bq = bq[:, :nc]
            nrm = dom.boundaries[bc.group].normal_np(bq)
            sig_b = np.zeros((len(bq), nc, nc))
            for i in range(nc):
                for j in range(nc):
                    sig_b[:, i, j] = sij(i, j)(bq)
            t = np.einsum("nij,nj->ni", sig_b, nrm)
            u_b = np.stack([ex(fl, bq) for fl in uf], axis=1)
            usc = max(float(np.max(np.abs(np.stack([ex(fl, pts) for fl in uf])))), 1e-300)
            if bc.type == "displacement":
                for c, v in zip(bc.components, bc.value):
                    worst = max(worst, float(np.max(np.abs(eval_value(v, bq) - u_b[:, c]))) / usc)
            elif bc.type == "fixed":
                worst = max(worst, float(np.max(np.abs(u_b))) / usc)
            elif bc.type == "symmetry":
                ax = dom.boundaries[bc.group].axis
                tt = t - np.sum(t * nrm, axis=1, keepdims=True) * nrm
                worst = max(worst, float(np.max(np.abs(u_b[:, ax]))) / usc, float(np.max(np.abs(tt))) / ssc)
            elif bc.type == "traction":
                tv = np.stack([eval_value(v, bq) for v in bc.value], axis=1)
                worst = max(worst, float(np.max(np.abs(tv - t))) / ssc)
            elif bc.type == "pressure":
                tv = -eval_value(bc.value, bq)[:, None] * nrm
                worst = max(worst, float(np.max(np.abs(tv - t))) / ssc)
            elif bc.type == "force":
                fac = _factor(f, problem.thickness, bq) * bw
                worst = max(worst, float(np.max(np.abs(np.sum(fac[:, None] * t, axis=0) - np.asarray(bc.value))))
                            / (ssc * float(np.sum(fac))))
        # free / none groups must be traction free unless they are on the axis
        named = {bc.group for bc in problem.bcs}
        for g, geom in dom.boundaries.items():
            if g in named or (dom.touches_axis and isinstance(geom, Plane) and geom.axis == 0 and geom.value == 0):
                continue
            bq, _ = dom.boundary_quadrature(g, 8)
            bq = bq[:, :nc]
            nrm = geom.normal_np(bq)
            sig_b = np.zeros((len(bq), nc, nc))
            for i in range(nc):
                for j in range(nc):
                    sig_b[:, i, j] = sij(i, j)(bq)
            worst = max(worst, float(np.max(np.abs(np.einsum("nij,nj->ni", sig_b, nrm)))) / ssc)
        out["boundary_data"] = worst
        return out
    # heat
    k = m.k
    Tf = lambda p: ex("T", p)  # noqa: E731
    g = _fd_grad(Tf, pts)
    nd = pts.shape[1]
    qsc = max(max(float(np.max(np.abs(ex(fl, pts)))) for fl in problem.s_fields), 1e-300)
    qref = -k * g
    out["flux_law"] = max(float(np.max(np.abs(qref[:, j] - ex(fl, pts)))) for j, fl in enumerate(problem.s_fields)) / qsc
    lap = np.zeros(n)
    for j in range(nd):
        lap += _fd_grad(lambda p, j=j: _fd_grad(Tf, p, 1e-4)[:, j], pts, 1e-4)[:, j]
    if problem.formulation == "axisymmetric":
        lap += g[:, 0] / pts[:, 0]
    sv = _load_value(problem, "source")
    src = eval_value(sv, pts) if sv is not None else np.zeros(n)
    out["balance"] = float(np.max(np.abs(-k * lap - src))) / (qsc / dom.length_scale)
    worst = 0.0
    Tsc = float(np.max(np.abs(Tf(pts))))
    for bc in problem.bcs:
        bq, _ = dom.boundary_quadrature(bc.group, 8)
        bq = bq[:, :nd]
        nrm = dom.boundaries[bc.group].normal_np(bq)
        dTdn = np.sum(_fd_grad(Tf, bq) * nrm, axis=1)
        if bc.type == "temperature":
            worst = max(worst, float(np.max(np.abs(eval_value(bc.value, bq) - Tf(bq)))) / Tsc)
        elif bc.type == "flux":
            worst = max(worst, float(np.max(np.abs(eval_value(bc.value, bq) - k * dTdn))) / qsc)
        elif bc.type == "convection":
            worst = max(worst, float(np.max(np.abs(k * dTdn - bc.h * (bc.T_inf - Tf(bq))))) / qsc)
    named = {bc.group for bc in problem.bcs}
    for gname, geom in dom.boundaries.items():
        if gname in named or (dom.touches_axis and isinstance(geom, Plane) and geom.axis == 0 and geom.value == 0):
            continue
        bq, _ = dom.boundary_quadrature(gname, 8)
        bq = bq[:, :nd]
        worst = max(worst, float(np.max(np.abs(k * np.sum(_fd_grad(Tf, bq) * geom.normal_np(bq), axis=1)))) / qsc)
    out["boundary_data"] = worst
    return out


# --------------------------------------------------------------------------- judging

#: round-off / output-precision floor of the relative errors per backend. CalculiX results
#: come through the ASCII .frd file (~6 significant digits).
ROUNDOFF = {"skfem": 1e-10, "ccx": 3e-6}
PATCH_TOL = {"skfem": (1e-9, 1e-8), "ccx": (5e-5, 5e-5)}


def judge(result: dict, problem: MMSProblem, floor: float | None = None, tol: float = 0.3) -> dict:
    """Observed order of the finest pair above the round-off floor vs theory - tol, per
    measure: u (displacement / T), s (recovered stress / flux), W (strain energy; not judged
    for thermal cases, where 0.5 u.K.u is not the physical energy)."""
    floor = ROUNDOFF.get(result["backend"], 1e-10) if floor is None else floor
    hs = [lv["h_rep"] for lv in result["levels"]]
    obs, ok = {}, {}
    for key, theory in result["theory"].items():
        if key == "W" and (not problem.energy_exact or problem.energy_physical is not None):
            continue
        e = [lv.get(f"e_{key}") for lv in result["levels"]]
        if any(v is None or not np.isfinite(v) for v in e):
            obs[key], ok[key] = None, False
            continue
        p, _ = _finest_order(hs, e, floor if key != "W" else floor / 10)
        obs[key] = p
        ok[key] = bool(p is not None and p >= theory - tol)
    return {"observed": obs, "pass": ok, "verdict": "pass" if ok and all(ok.values()) else "fail"}
