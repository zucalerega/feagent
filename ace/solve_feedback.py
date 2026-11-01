"""What a careful engineer looks at after a solve, when there is no reference solution.

feedback(problem, mesh, setup, result) -> compact dict:
  * every statement QoI the solve can evaluate;
  * equilibrium: support reactions per group against the applied load totals. The reactions R = K u - f
    at the constrained DOFs must balance the applied forces, sum R + sum f = 0 per component (K kills
    rigid translations), so a large residual means a setup problem, not a discretisation error;
  * max displacement and max von Mises, where they are and which named group is nearest;
  * the Zienkiewicz-Zhu estimate (feacore.diagnostics.zz): global relative energy-norm error and the
    top hotspots, i.e. clusters of cells carrying most of the estimated error energy, with the nearest
    face group, nearest re-entrant edge (geometry_info) and the local element size there.
render(problem, mesh, result, kind, view, out_png): one PNG of the mesh, von Mises or the ZZ error.
compare(problem, result_a, result_b): how far two solves on different meshes are apart (energy norm of
  the stress difference) and where: convergence evidence without a reference.

Everything returned is plain JSON (floats rounded to 3-4 significant digits).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.spatial import cKDTree

from ace import fea
from ace import field_error as fe
from ace import inspect_tools as it
from feacore.diagnostics.view import SolutionView
from feacore.diagnostics.zz import _VOIGT, _compliance_density, zz_estimate

try:  # builder G's module; without it hotspots just lack the nearest re-entrant edge
    from ace import geometry_info
except ImportError:  # pragma: no cover
    geometry_info = None

N_TOP = 5
SUPPORTS = ("fixed", "displacement", "symmetry")
LOADS = ("pressure", "traction", "force")
VIEWS = {"iso": (28, 45), "x": (0, 0), "-x": (0, 180), "y": (0, 90), "-y": (0, -90),
         "z": (90, -90), "-z": (-90, 90)}       # (elevation, azimuth) of the camera, matplotlib convention
LOG_RANGE = 3                                   # the error image spans 3 decades of error density


def _sig(x, n: int = 3):
    """Round to n significant digits; None for NaN / inf (JSON has no NaN)."""
    x = float(x)
    return float(f"{x:.{n}g}") if math.isfinite(x) else None


def _xyz(x) -> list:
    return [_sig(v, 4) for v in x]


# --------------------------------------------------------------------------- the part as the solver saw it


class _Part:
    """The solved mesh's corner nodes and cells, element sizes and named groups as point clouds
    (for "nearest group" questions). Built once per result and kept on it."""

    def __init__(self, result):
        md = result.model
        m = md.mesh
        self.dim = md.dim
        self.P = np.zeros((m.p.shape[1], 3))
        self.P[:, : self.dim] = m.p.T
        self.cells = m.t.T
        X = self.P[self.cells]
        pairs = [(a, b) for a in range(self.dim + 1) for b in range(a + 1, self.dim + 1)]
        self.size = np.mean([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in pairs], axis=0)
        self.centroids = X.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.diag = float(np.linalg.norm(np.ptp(self.P, axis=0)))
        self.groups = {}                         # name -> corner-node indices
        for name, g in md.md.groups.items():
            if g.kind == "facets":
                nodes = m.facets[:, g.index]
            elif g.kind == "edges":
                nodes = m.edges[:, g.index]
            elif g.kind == "vertices":
                nodes = g.index
            else:                                # the domain ("solid"): every point is in it
                continue
            self.groups[name] = np.unique(nodes)
        self.group_trees = {g: cKDTree(self.P[n]) for g, n in self.groups.items()}

    def local_size(self, x, r: float) -> float:
        near = self.tree.query_ball_point(x, r) or [int(self.tree.query(x)[1])]
        return float(np.mean(self.size[near]))

    def nearest_group(self, x) -> tuple:
        best = min(((t.query(x)[0], g) for g, t in self.group_trees.items()), default=(np.inf, None))
        return best[1], float(best[0])


def _part(result) -> _Part:
    if getattr(result, "_ace_part", None) is None:
        result._ace_part = _Part(result)
    return result._ace_part


# Edge kinds where the exact stress is singular, with the geometry_info key listing them (3D, 2D).
SINGULAR = {"reentrant": ("reentrant_edges", "reentrant_corners"),       # interior angle > 180 deg
            "clamp_end": ("support_boundary", "support_boundary")}       # where a supported face ends


@functools.lru_cache(maxsize=32)
def _singular(problem_id: str) -> dict:
    """{kind: ((tag, sample points (n, 3)), ...)} for the SINGULAR kinds: edges in 3D, corners in 2D.
    The tags come from geometry_info; the points are sampled from the CAD with gmsh."""
    if geometry_info is None:
        return {}
    problem = fea.get_problem(problem_id)
    try:
        info = geometry_info.describe(problem)
    except Exception:  # noqa: BLE001  geometry info is a hint here, never a failure
        return {}
    k = 0 if problem.dim == 3 else 1
    tags = {kind: [int(t) for t in info.get(keys[k]) or []] for kind, keys in SINGULAR.items()}
    if not any(tags.values()):
        return {}
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session

    def samples(t):
        if problem.dim == 3:                     # a curve: 64 points along its parametrisation
            lo, hi = gmsh.model.getParametrizationBounds(1, t)
            return gmsh.model.getValue(1, t, list(np.linspace(lo[0], hi[0], 64)))
        return gmsh.model.getValue(0, t, [])     # a corner point
    with gmsh_session(problem.id):
        problem.build_geometry()
        return {kind: tuple((t, np.asarray(samples(t), float).reshape(-1, 3)) for t in ts)
                for kind, ts in tags.items() if ts}


def _where(problem, part: _Part, x) -> dict:
    """Location of a point in words an agent can act on: coordinates, nearest named group,
    nearest re-entrant edge and nearest edge where a support ends (both stress singularities)."""
    g, d = part.nearest_group(x)
    out = {"at": _xyz(x[: part.dim]), "nearest_group": g, "group_distance": _sig(d)}
    for kind, edges in _singular(problem.id).items():
        dist, t = min((float(np.linalg.norm(s - x, axis=1).min()), t) for t, s in edges)
        out[f"nearest_{kind}_edge"] = {"tag": t, "distance": _sig(dist)}
    return out


def _clusters(problem, part: _Part, e: np.ndarray, extra=None) -> list[dict]:
    """Top N_TOP clusters (balls of 5% of the diagonal) of cells carrying most of the energy e
    (per cell), largest share first; `extra(members, centre)` adds fields to each row."""
    total = float(e.sum())
    if total <= 0:
        return []
    rows = []
    r = it.RADIUS * part.diag
    for m in sorted(it._hotspots(part.centroids, e, r), key=lambda m: -e[m].sum())[:N_TOP]:
        c = np.average(part.centroids[m], axis=0, weights=e[m])
        row = {"share": _sig(e[m].sum() / total, 2), **_where(problem, part, c),
               "local_h": _sig(part.local_size(c, r))}
        if extra is not None:
            row.update(extra(m, c))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- feedback


def _qois(problem, result) -> dict:
    out = {}
    for q in problem.qois():
        try:
            out[q.name] = _sig(result.evaluate(q), 4)
        except Exception:  # noqa: BLE001  e.g. a point outside this mesh: not evaluable here
            out[q.name] = None
    return out


def _equilibrium(result) -> dict:
    md, case = result.model, result.case
    R, f, D = result.reaction_vector, result.f, result.dirichlet_dofs
    nc = md.ncomp
    constrained = np.zeros(md.N, bool)
    constrained[D] = True

    def group_sum(vec, bc, only_constrained):
        S = md.sdofs_of(bc.group) if bc.group else np.array([md.nearest_sdof(bc.point)[0]])
        idx = [md.split[k][S] for k in range(nc)]
        return [_sig(vec[i[constrained[i]]].sum() if only_constrained else vec[i].sum(), 4) for i in idx]

    supports = {(bc.group or f"point {bc.point}"): group_sum(R, bc, True) for bc in case.bcs if bc.type in SUPPORTS}
    loads = {bc.group: group_sum(f, bc, False) for bc in case.bcs if bc.type in LOADS and bc.group}
    F = np.array([f[md.split[k]].sum() for k in range(nc)])
    Rt = np.array([R[md.split[k][constrained[md.split[k]]]].sum() for k in range(nc)])
    out = {"applied_total": [_sig(v, 4) for v in F], "reaction_total": [_sig(v, 4) for v in Rt],
           "reactions_by_support": supports, "applied_by_load_group": loads}
    if not loads and not any(l.type == "body_force" for l in case.loads):
        # Driven by prescribed displacements only (e.g. a platen ramp): no applied force to balance, and
        # the supports' reactions cancel among themselves, so residual = |~0| / |~0| would mean nothing.
        out["residual"] = None
        out["note"] = "no applied force or pressure (displacement-driven): reactions per support shown, no balance check"
        return out
    # Axisymmetric: the radial component never balances (hoop forces), only the axial one does.
    ks = [1] if md.axi else list(range(nc))
    scale = max(np.linalg.norm(F[ks]), np.linalg.norm(Rt[ks]), 1e-300)
    out["residual"] = _sig(np.linalg.norm((F + Rt)[ks]) / scale, 2)
    return out


def _extreme(problem, result, part: _Part, field: str) -> dict:
    X, v = result.nodal(field)
    i = int(np.argmax(v))
    x = np.zeros(3)
    x[: part.dim] = X[i, : part.dim]
    return {"value": _sig(v[i], 4), **_where(problem, part, x)}


def _zz(result) -> tuple[dict, np.ndarray]:
    """ZZ estimate and the per-cell error energy (eta^2)."""
    zz = zz_estimate(SolutionView.from_result(result))
    return zz, zz["eta2_cell"]


def _error_hotspots(problem, result, part: _Part, eta2: np.ndarray, local_rel: np.ndarray) -> list[dict]:
    # Per cell, eta2 / local_rel^2 = eta2 + energy^2: the cluster's own relative error follows.
    tot = np.where(local_rel > 0, eta2 / np.maximum(local_rel, 1e-300) ** 2, 0.0)

    def extra(m, c):
        return {"local_rel_error": _sig(math.sqrt(eta2[m].sum() / max(tot[m].sum(), 1e-300)), 2)}
    return _clusters(problem, part, eta2, extra)


def feedback(problem, mesh: dict, setup: dict, result) -> dict:
    """Reference-free look at one solve (see the module docstring). mesh and setup are the tool
    arguments of that solve; everything needed is already on `result`."""
    part = _part(result)
    zz, eta2 = _zz(result)
    out = {
        "dofs": int(result.info.n_dofs), "cells": int(result.info.n_cells),
        "qois": _qois(problem, result),
        "equilibrium": _equilibrium(result),
        "max_displacement": _extreme(problem, result, part, "u_mag"),
        "max_von_mises": _extreme(problem, result, part, "von_mises"),
        "error_estimate": {
            "eta_rel": _sig(zz["eta_rel"]),                       # relative energy-norm error (ZZ)
            "strain_energy_rel_error": _sig(zz["energy_error_est"]),
            "top1pct_cells_share": _sig(zz["concentration_top1pct"], 2),
            "hotspots": _error_hotspots(problem, result, part, eta2, zz["local_rel"]),
        },
    }
    notes = []
    if (res := out["equilibrium"]["residual"]) is not None and res > 1e-3:
        notes.append(f"reactions do not balance the applied loads (residual {res:.2g}): check the BCs/loads")
    hs = out["error_estimate"]["hotspots"]
    # A hotspot within one local element size (or the cluster radius) of a singular edge is that
    # singularity: refining there lowers the energy error, but the peak stress keeps growing.
    for kind, what in (("reentrant", "re-entrant edges"), ("clamp_end", "edges where a support ends")):
        near = [h for h in hs if (e := h.get(f"nearest_{kind}_edge")) and e["distance"] is not None
                and e["distance"] <= max(h["local_h"] or 0, it.RADIUS * _part(result).diag)]
        if near:
            tags = sorted({h[f"nearest_{kind}_edge"]["tag"] for h in near})
            notes.append(f"{sum(h['share'] for h in near):.0%} of the estimated error sits at {what} {tags}: "
                         "the stress there is singular (its max grows under refinement); energies, "
                         "displacements and averages still converge")
    if notes:
        out["notes"] = notes
    return out


# --------------------------------------------------------------------------- compare


def compare(problem, result_a, result_b) -> dict:
    """Energy norm of sigma_a - sigma_b over the part, relative to the finer solve's stress energy,
    integrated with the coarser mesh's quadrature (its points are where both fields are sampled),
    plus the regions where the difference lives and the change of every statement QoI."""
    coarse, fine = sorted((result_a, result_b), key=lambda r: r.info.n_cells)
    md = coarse.model
    # The solver's own quadrature of the coarse mesh. Cheaper rules (fewer points to locate in the
    # fine mesh) under-weight the steep fields at singular edges and read 20% lower.
    b = md.sb
    xq = np.asarray(b.global_coordinates())                           # (dim, nel, nqp)
    nel, nqp = xq.shape[1:]
    w = b.dx * md.wfac(xq)                                            # physical weights (thickness, 2 pi r)
    s_c = {k: np.asarray(b.interpolate(coarse.fields[k])) for k in _VOIGT}
    pts = xq.reshape(md.dim, -1).T
    s_f = {k: v.reshape(nel, nqp) for k, v in fe.stresses(fine, pts, md.dim).items()}
    E = np.repeat(md.E[:, None], nqp, axis=1)
    nu = np.repeat(md.nu[:, None], nqp, axis=1)
    d_cell = (_compliance_density({k: s_c[k] - s_f[k] for k in _VOIGT}, E, nu) * w).sum(axis=1)
    en_f = float((_compliance_density(s_f, E, nu) * w).sum())
    part_c, part_f = _part(coarse), _part(fine)
    r = it.RADIUS * part_c.diag

    def extra(m, c):
        return {"local_h_fine": _sig(part_f.local_size(c, r))}
    regions = _clusters(problem, part_c, d_cell, extra)
    for row in regions:                                               # local_h is the coarse mesh's
        row["local_h_coarse"] = row.pop("local_h")
    qa, qb = _qois(problem, coarse), _qois(problem, fine)
    changes = {n: {"coarse": qa[n], "fine": qb[n],
                   "rel_change": _sig((qb[n] - qa[n]) / abs(qb[n]), 2) if qa[n] is not None and qb[n] else None}
               for n in qa}
    return {
        "coarse": {"is": "a" if coarse is result_a else "b", "dofs": int(coarse.info.n_dofs), "cells": nel},
        "fine": {"is": "a" if fine is result_a else "b", "dofs": int(fine.info.n_dofs), "cells": int(fine.info.n_cells)},
        "stress_diff_energy_norm_rel": _sig(math.sqrt(d_cell.sum() / max(en_f, 1e-300))),
        "regions": regions,
        "qoi_change": changes,
    }


# --------------------------------------------------------------------------- render


def _boundary(part: _Part):
    """Boundary triangles of the tet mesh, oriented outward: (n, 3) node indices."""
    t = part.cells
    loc = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0))   # face nodes + the opposite node
    faces = np.vstack([t[:, f[:3]] for f in loc])
    opp = np.concatenate([t[:, f[3]] for f in loc])
    tet = np.tile(np.arange(len(t)), 4)
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    keep = first[count == 1]
    tri, opp, tet = faces[keep], opp[keep], tet[keep]
    P = part.P
    n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    flip = np.einsum("ij,ij->i", n, P[opp] - P[tri[:, 0]]) > 0          # normal points into the tet
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri, tet


def _unoccluded(pts: np.ndarray, T: np.ndarray, cam: np.ndarray, eps: float) -> np.ndarray:
    """Which points (n, 3) are seen from the camera direction `cam` (orthographic): a ray from the
    point towards the camera hits none of the triangles T (m, 3, 3) farther than eps from the point
    (Moller-Trumbore, vectorised in chunks). matplotlib draws lines and text over every surface, so
    labels and edges on hidden parts would otherwise show through."""
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = np.cross(cam, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-30
    e1, e2, pv, det, T0 = e1[ok], e2[ok], pv[ok], det[ok], T[ok, 0]
    seen = np.ones(len(pts), bool)
    for a in range(0, len(pts), 256):
        tv = pts[a:a + 256, None, :] - T0[None]                                   # (k, m, 3)
        u = np.einsum("kmj,mj->km", tv, pv) / det
        qv = np.cross(tv, e1[None])
        v = np.einsum("kmj,j->km", qv, cam) / det
        t = np.einsum("kmj,mj->km", qv, e2) / det
        hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        seen[a:a + 256] = ~hit.any(axis=1)
    return seen


def _roles(result) -> dict:
    """Named group -> role in the setup ("support" / "load"), from the solved case."""
    roles = {}
    for bc in result.case.bcs:
        if bc.group:
            roles[bc.group] = "support" if bc.type in SUPPORTS else "load"
    return roles


ROLE_COLOUR = {"support": (0.25, 0.45, 0.85, 1.0), "load": (0.9, 0.35, 0.2, 1.0)}


def _hotspot_marks(ax, hs: list, three_d: bool) -> None:
    for k, h in enumerate(hs, 1):
        x = (list(h["at"]) + [0.0])[:3]
        box = dict(boxstyle="circle,pad=0.2", fc="white", ec="red", lw=1.5, alpha=0.9)
        args = (x if three_d else x[:2])
        ax.text(*args, str(k), fontsize=10, fontweight="bold", color="red", bbox=box, zorder=20,
                ha="center", va="center")


def render(problem, mesh: dict, result, kind: str, view: str, out_png) -> str:
    """One PNG (880 x 700 px) of the solve: kind "mesh" (named groups, supports blue, loads red,
    re-entrant edges magenta), "stress" (von Mises, the max starred) or "error" (ZZ error density,
    hotspots numbered as in feedback). 3D: the visible surface from `view`; 2D ignores `view`."""
    import matplotlib
    from ace.run_report import _new_fig

    if kind not in ("mesh", "stress", "error"):
        raise ValueError(f"kind must be mesh, stress or error, not {kind!r}")
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}, not {view!r}")
    part = _part(result)
    fig = _new_fig(8.8, 7.0)
    title = f"{problem.id}: {kind}  ({result.info.n_cells} cells, {result.info.n_dofs} DOFs)"
    nv = len(part.P)
    cmap, norm, sub, hs = None, None, "", []
    if kind == "stress":
        vm = result.nodal("von_mises")[1][:nv]                      # vertex DOFs come first in skfem
        hi = float(np.percentile(vm, 99.5))
        cmap, norm = matplotlib.colormaps["viridis"], matplotlib.colors.Normalize(0, hi)
        mx = _extreme(problem, result, part, "von_mises")
        sub = f"von Mises; max {mx['value']:.4g} (star), colours capped at the 99.5th pct {hi:.3g}"
        cell_val = None
    elif kind == "error":
        zz, eta2 = _zz(result)
        vol = np.abs(SolutionView.from_result(result).cell_measure)
        dens = eta2 / np.maximum(vol, 1e-300) / max(eta2.sum() / vol.sum(), 1e-300)
        cell_val = np.log10(np.maximum(dens, 1e-30))
        top = float(np.percentile(cell_val, 99.7))
        cmap, norm = matplotlib.colormaps["inferno"], matplotlib.colors.Normalize(top - LOG_RANGE, top)
        hs = _error_hotspots(problem, result, part, eta2, zz["local_rel"])
        sub = f"ZZ error density / mean (log10); estimated rel. energy error {zz['eta_rel']:.3g}; red = hotspots"
    roles = _roles(result)
    reent = _singular(problem.id).get("reentrant", ())

    if part.dim == 3:
        tri, tet = _boundary(part)
        el, az = np.radians(VIEWS[view])
        cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        P = part.P
        n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
        area2 = np.linalg.norm(n, axis=1)
        n = n / np.maximum(area2, 1e-300)[:, None]
        front = n @ cam > 1e-3                                       # back faces are hidden anyway
        tri, tet, n, area2 = tri[front], tet[front], n[front], area2[front]
        shade = 0.45 + 0.55 * np.abs(n @ cam)                        # headlight shading shows the shape
        # face group of each surface triangle (corner nodes)
        tri_group = np.full(len(tri), None, object)
        key = {tuple(r): i for i, r in enumerate(np.sort(tri, axis=1))}
        m = result.model.mesh
        for name, g in result.model.md.groups.items():
            if g.kind == "facets" and g.gdim == 2:
                for r in np.sort(m.facets[:, g.index].T, axis=1):
                    if (i := key.get(tuple(r))) is not None:
                        tri_group[i] = name
        if kind == "mesh":
            fc = np.tile([0.82, 0.82, 0.82, 1.0], (len(tri), 1))
            for i, g in enumerate(tri_group):
                if g in roles:
                    fc[i] = ROLE_COLOUR[roles[g]]
        elif kind == "stress":
            fc = cmap(norm(vm[tri].mean(axis=1)))
        else:
            fc = cmap(norm(cell_val[tet]))
        fc[:, :3] *= shade[:, None]
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        lw = 0.25 if kind == "mesh" else 0.0
        ax.add_collection3d(Poly3DCollection(P[tri], facecolors=fc, edgecolors=(0.15, 0.15, 0.15, 0.6),
                                             linewidths=lw))
        lo, hi_ = P.min(0), P.max(0)
        ax.set_xlim(lo[0], hi_[0])
        ax.set_ylim(lo[1], hi_[1])
        ax.set_zlim(lo[2], hi_[2])
        ax.set_box_aspect(np.maximum(hi_ - lo, 1e-3 * part.diag), zoom=1.2)
        ax.view_init(*VIEWS[view])
        for a in "xyz":
            if a == view[-1]:                    # the axis along the line of sight: its ticks overprint
                getattr(ax, f"set_{a}ticks")([])
            else:
                getattr(ax, f"set_{a}label")(a, fontsize=9)
        ax.tick_params(labelsize=7)
        if kind == "mesh":
            # Label each visible face group on its visible triangle nearest to the visible part's
            # centroid (supports / loads in bold colour); re-entrant edges only where they are seen.
            eps = 1e-3 * part.diag
            seen = _unoccluded(P[tri].mean(axis=1), P[tri], cam, eps)
            total = area2[seen].sum()
            for name in {g for g in tri_group if g}:
                sel = np.nonzero((tri_group == name) & seen)[0]
                if area2[sel].sum() < 0.004 * total:
                    continue
                cs = P[tri[sel]].mean(axis=1)
                c = cs[np.argmin(np.linalg.norm(cs - np.average(cs, axis=0, weights=area2[sel]), axis=1))]
                role = roles.get(name)
                ax.text(*c, name + (f" ({role})" if role else ""), fontsize=8 if role else 7,
                        fontweight="bold" if role else "normal", ha="center", va="center", zorder=15,
                        color=ROLE_COLOUR[role] if role else "black",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))
            for t, s in reent:
                vis = _unoccluded(s, P[tri], cam, eps)
                if vis.sum() < 3:
                    continue
                ax.plot(*np.where(vis[:, None], s, np.nan).T, color="magenta", lw=2.2, zorder=12)
                ax.text(*s[np.nonzero(vis)[0][vis.sum() // 2]], f"e{t}", color="magenta", fontsize=7,
                        fontweight="bold", zorder=16)
        if kind == "stress":
            ax.scatter(*mx["at"], marker="*", s=220, c="red", edgecolors="white", depthshade=False, zorder=20)
        _hotspot_marks(ax, hs, True)
        title += f", view {view}"
    else:
        from matplotlib.tri import Triangulation
        P = part.P
        mtri = Triangulation(P[:, 0], P[:, 1], part.cells)
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        if kind == "mesh":
            ax.triplot(mtri, color="0.4", lw=0.4)
            m = result.model.mesh
            from matplotlib.collections import LineCollection
            for name, g in result.model.md.groups.items():
                if g.kind != "facets":
                    continue
                seg = P[m.facets[:, g.index].T][:, :, :2]
                role = roles.get(name)
                col = ROLE_COLOUR[role] if role else (0.3, 0.6, 0.3, 1.0)
                ax.add_collection(LineCollection(seg, colors=[col], lw=3.0 if role else 1.8, zorder=4))
                c = seg.reshape(-1, 2).mean(axis=0)
                ax.annotate(name + (f" ({role})" if role else ""), c, fontsize=8, color=col, zorder=6,
                            fontweight="bold" if role else "normal", xytext=(3, 3), textcoords="offset points")
            for t, s in reent:
                ax.plot(s[:, 0], s[:, 1], "o", ms=9, mfc="none", mec="magenta", mew=2, zorder=7)
                ax.annotate(f"p{t}", s[0, :2], color="magenta", fontsize=8, xytext=(5, -10),
                            textcoords="offset points")
        elif kind == "stress":
            ax.tripcolor(mtri, vm, shading="gouraud", cmap=cmap, norm=norm)
            ax.plot(*mx["at"][:2], "*", ms=16, color="red", mec="white")
        else:
            ax.tripcolor(mtri, facecolors=cell_val, cmap=cmap, norm=norm)
            ax.triplot(mtri, color="white", lw=0.2, alpha=0.3)
        _hotspot_marks(ax, hs, False)
        ax.set_xlabel("r" if result.model.axi else "x")
        ax.set_ylabel("z" if result.model.axi else "y")
    if cmap is not None:
        fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.7,
                     label="von Mises" if kind == "stress" else "log10 error density / mean")
    else:
        sub = ("supports blue, loads red, re-entrant edges magenta (e<tag>: refine with {'edge': tag})"
               if part.dim == 3 else "supports blue, loads red, other groups green, re-entrant corners magenta")
    fig.suptitle(title + ("\n" + sub if sub else ""), fontsize=10)
    fig.savefig(out_png, dpi=100)
    return str(out_png)
