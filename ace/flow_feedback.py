"""What a careful CFD engineer looks at after a flow solve, without a reference solution: the flow
counterpart of ace.solve_feedback (which reads elastic stress fields).

feedback(problem, mesh, setup, result) -> compact dict:
  * every statement QoI the solve can evaluate;
  * convergence: SIMPLE iterations, the last initial residuals, whether the residual target was met;
  * mass balance: the flow rate through every inlet / outlet group (positive out of the domain) and
    the imbalance (a few 1e-6 at convergence; more means the solve did not converge);
  * the fluid force (pressure + viscous) on every wall group;
  * the peak speed and the pressure range, and where they are;
  * the cell Peclet number |U| h / nu: where it is above 2, first-order upwind convection adds
    numerical diffusion of the order of the physical one;
  * reversed flow (against the mean inflow direction): recirculation zones and where they are;
  * an error indicator: at every face, the jump between the two neighbouring cells' linear
    reconstructions of the velocity (on a wall or inlet: between the cell's reconstruction and the
    boundary value). For a second-order scheme it is O(h^2 u''), so it is large where the velocity
    curves sharply relative to the cell size: thin wall layers, corners, the front of a body. It is
    reported as a global relative L2 indicator and the hotspots holding most of it, each with its
    nearest group and local cell size. It ranks where to refine; it is not an error bound.
compare(problem, result_a, result_b): the velocity difference between two solves (the finer one
  reconstructed at the coarser one's cell centres) over the domain and the regions where it lives,
  plus the change of every statement QoI: convergence evidence without a reference.
render(problem, mesh, result, kind, view, out_png): a picture of the mesh, the speed (with velocity
  arrows), the pressure or the error indicator. 2D: the cells; 3D: a section through the middle of
  the domain normal to `view` (x, y or z).

Everything returned is plain JSON (floats rounded to 3-4 significant digits).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ace import inspect_tools as it

N_TOP = 5
PECLET_UPWIND = 2.0          # cell Peclet number above which upwind convection diffuses noticeably
REVERSED = 0.01              # reversed flow: velocity against the inflow direction > 1% of the peak speed
VIEWS = ("x", "y", "z", "iso", "-x", "-y", "-z")
INLET, OUTLET, WALLS = "velocity_inlet", "pressure_outlet", ("wall", "slip", "symmetry")


def _sig(x, n: int = 3):
    """Round to n significant digits; None for NaN / inf (JSON has no NaN)."""
    x = float(x)
    return float(f"{x:.{n}g}") if math.isfinite(x) else None


def _xyz(x, dim: int) -> list:
    return [_sig(v, 4) for v in list(x)[:dim]]


# --------------------------------------------------------------------------- the domain as the solver saw it


class _Fluid:
    """Cell centres, volumes and sizes of a flow result, and its boundary groups as clouds of face
    centres (for "nearest group" questions). Built once per result and kept on it."""

    def __init__(self, result):
        m = result.mesh
        self.dim = result.dim
        self.C = np.asarray(m.C)
        self.V = np.abs(np.asarray(m.V))
        depth = result.fm.thickness if self.dim == 2 else 1.0
        self.h = (self.V / depth) ** (1.0 / self.dim)            # local cell size
        self.tree = cKDTree(self.C[:, : self.dim])
        # each cell's reach: the farthest of its face centres (anisotropic cells: wall layers, swept layers)
        n = len(self.C)
        dist = np.linalg.norm(m.Cf - self.C[m.owner], axis=1)
        self.reach = np.zeros(n)
        np.maximum.at(self.reach, m.owner, dist)
        np.maximum.at(self.reach, m.neighbour, np.linalg.norm(m.Cf[: m.n_int] - self.C[m.neighbour], axis=1))
        P = np.asarray(m.points)
        self.lo, self.hi = P.min(axis=0), P.max(axis=0)
        self.diag = float(np.linalg.norm((self.hi - self.lo)[: self.dim]))
        names, pts = [], []
        for group, patches in result.fm.group_patches.items():
            for p in patches:
                if p in result.patch:
                    names += [group] * len(result.patch[p]["Cf"])
                    pts.append(result.patch[p]["Cf"][:, : self.dim])
        self.face_group = np.array(names) if names else np.zeros(0, dtype=str)
        self.face_tree = cKDTree(np.vstack(pts)) if pts else None

    def nearest_group(self, x) -> tuple[str | None, float]:
        if self.face_tree is None:
            return None, float("nan")
        d, i = self.face_tree.query(np.asarray(x)[: self.dim])
        return str(self.face_group[int(i)]), float(d)

    def where(self, x) -> dict:
        g, d = self.nearest_group(x)
        return {"at": _xyz(x, self.dim), "nearest_group": g, "group_distance": _sig(d)}

    def local_size(self, x, r: float) -> float:
        idx = self.tree.query_ball_point(np.asarray(x)[: self.dim], r)
        if not idx:
            idx = [int(self.tree.query(np.asarray(x)[: self.dim])[1])]
        return float(np.mean(self.h[idx]))


def _fluid(result) -> _Fluid:
    f = getattr(result, "_ace_fluid", None)
    if f is None:
        f = result._ace_fluid = _Fluid(result)
    return f


def _clusters(fl: _Fluid, e: np.ndarray, extra=None) -> list[dict]:
    """Top N_TOP clusters (balls of 5% of the diagonal) of cells carrying most of e (per cell)."""
    total = float(e.sum())
    if total <= 0:
        return []
    rows = []
    r = it.RADIUS * fl.diag
    pts = fl.C[:, : fl.dim]
    for m in sorted(it._hotspots(pts, e, r), key=lambda m: -e[m].sum())[:N_TOP]:
        c = np.average(fl.C[m], axis=0, weights=e[m])
        row = {"share": _sig(e[m].sum() / total, 2), **fl.where(c), "local_h": _sig(fl.local_size(c, r))}
        if extra is not None:
            row.update(extra(m, c))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- pieces of the feedback


def _qois(problem, result) -> dict:
    out = {}
    for q in problem.qois():
        try:
            out[q.name] = _sig(result.evaluate(q), 4)
        except Exception:  # noqa: BLE001  e.g. a point outside this mesh: not evaluable here
            out[q.name] = None
    return out


def _convergence(result) -> dict:
    i = result.info
    return {"iterations": int(getattr(i, "iterations", 0)), "converged": bool(getattr(i, "converged", False)),
            "residual_target": _sig(getattr(i, "residual_tol", 0.0)),
            "last_residuals": {k: _sig(v, 2) for k, v in (getattr(i, "residuals", {}) or {}).items()}}


def _bc_types(result) -> dict:
    """{group: BC type} of the case (boundary groups without a BC are no-slip walls)."""
    out = {b.group: b.type for b in result.case.bcs if b.group}
    for g in result.fm.group_patches:
        out.setdefault(g, "wall")
    return out


def _mass_balance(result) -> dict:
    types = _bc_types(result)
    rates = {}
    for g, t in types.items():
        if t in (INLET, OUTLET):
            rates[g] = float(sum(result.patch[p]["phi"].sum() for p in result.fm.group_patches.get(g, [])
                                 if p in result.patch))
    if not rates:
        return {"flow_rates": {}, "imbalance": None, "note": "closed domain: no inlet or outlet"}
    inflow = sum(-q for q in rates.values() if q < 0)
    total = sum(float(d["phi"].sum()) for d in result.patch.values())
    return {"flow_rates": {g: _sig(q, 4) for g, q in rates.items()},
            "imbalance": _sig(abs(total) / max(inflow, 1e-300), 2)}


def _forces(result) -> dict:
    out = {}
    for g, t in _bc_types(result).items():
        if t != "wall":
            continue
        try:
            fp, fv = result.force(g)
        except Exception:  # noqa: BLE001  a group the force evaluation cannot use: skip it
            continue
        d = result.dim
        out[g] = {"total": _xyz(np.asarray(fp) + np.asarray(fv), d), "pressure": _xyz(fp, d),
                  "viscous": _xyz(fv, d)}
    return out


def _extremes(result, fl: _Fluid) -> dict:
    speed = np.linalg.norm(result.U, axis=1)
    i, lo, hi = int(np.argmax(speed)), int(np.argmin(result.p)), int(np.argmax(result.p))
    return {"max_speed": {"value": _sig(speed[i], 4), **fl.where(fl.C[i])},
            "pressure_min": {"value": _sig(result.p[lo], 4), **fl.where(fl.C[lo])},
            "pressure_max": {"value": _sig(result.p[hi], 4), **fl.where(fl.C[hi])}}


def _peclet(result, fl: _Fluid, setup: dict) -> dict:
    pe = np.linalg.norm(result.U, axis=1) * fl.h / max(result.nu, 1e-300)
    i = int(np.argmax(pe))
    return {"max": _sig(pe[i]), "median": _sig(np.median(pe)),
            "share_above_2": _sig(float(fl.V[pe > PECLET_UPWIND].sum() / fl.V.sum()), 2), **fl.where(fl.C[i])}


def _inflow_direction(result) -> np.ndarray | None:
    types = _bc_types(result)
    acc = np.zeros(3)
    for g, t in types.items():
        if t != INLET:
            continue
        for p in result.fm.group_patches.get(g, []):
            if p in result.patch:
                d = result.patch[p]
                acc += (d["U"] * np.linalg.norm(d["Sf"], axis=1)[:, None]).sum(axis=0)
    n = float(np.linalg.norm(acc))
    return acc / n if n > 0 else None


def _reversed(result, fl: _Fluid) -> dict | None:
    """Recirculation: cells whose velocity points against the mean inflow direction."""
    d = _inflow_direction(result)
    if d is None:
        return None
    speed = np.linalg.norm(result.U, axis=1)
    back = (result.U @ d) < -REVERSED * float(speed.max())
    if not back.any():
        return {"volume_share": 0.0, "zones": []}
    e = np.where(back, fl.V, 0.0)
    zones = []
    for m in sorted(it._hotspots(fl.C[:, : fl.dim], e, 2 * it.RADIUS * fl.diag), key=lambda m: -e[m].sum())[:3]:
        X = fl.C[m, : fl.dim]
        zones.append({"share": _sig(e[m].sum() / e.sum(), 2), "from": _xyz(X.min(axis=0), fl.dim),
                      "to": _xyz(X.max(axis=0), fl.dim), "nearest_group": fl.nearest_group(X.mean(axis=0))[0]})
    return {"volume_share": _sig(float(e.sum() / fl.V.sum()), 2), "zones": zones}


def _jumps(result) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: (indicator energy eta^2 = V * mean squared face jump, mean squared face jump)."""
    m = result.mesh
    C, U, G = np.asarray(m.C), np.asarray(result.U), np.asarray(result.gradU)
    n = len(C)
    own, nei = m.owner[: m.n_int], m.neighbour
    Cf = m.Cf[: m.n_int]

    def rec(c, x):
        return U[c] + np.einsum("ni,nij->nj", x - C[c], G[c])

    j2 = np.sum((rec(own, Cf) - rec(nei, Cf)) ** 2, axis=1)
    acc = np.bincount(own, j2, n) + np.bincount(nei, j2, n)
    cnt = np.bincount(own, None, n) + np.bincount(nei, None, n)
    for name, ub in result._dirichlet["U"].items():          # walls, inlets: against the boundary value
        sl = m.patch_slice(name)
        o = m.owner[sl]
        jb = np.sum((rec(o, m.Cf[sl]) - np.asarray(ub)) ** 2, axis=1)
        acc += np.bincount(o, jb, n)
        cnt += np.bincount(o, None, n)
    mean = acc / np.maximum(cnt, 1)
    return np.abs(np.asarray(m.V)) * mean, mean


def _error_estimate(result, fl: _Fluid) -> dict:
    eta2, mean = _jumps(result)
    norm2 = float(np.sum(fl.V * np.sum(np.asarray(result.U) ** 2, axis=1)))
    speed2 = np.sum(np.asarray(result.U) ** 2, axis=1)

    def extra(m, c):
        return {"local_rel": _sig(math.sqrt(float(eta2[m].sum()) / max(float((fl.V[m] * speed2[m]).sum()), 1e-300)), 2)}
    order = np.sort(eta2)[::-1]
    top = order[: max(1, len(order) // 100)].sum() / max(order.sum(), 1e-300)
    return {"rel_indicator": _sig(math.sqrt(float(eta2.sum()) / max(norm2, 1e-300))),
            "top1pct_cells_share": _sig(top, 2), "hotspots": _clusters(fl, eta2, extra)}


# --------------------------------------------------------------------------- feedback


def feedback(problem, mesh: dict, setup: dict, result) -> dict:
    """Reference-free look at one flow solve (see the module docstring)."""
    fl = _fluid(result)
    out = {
        "dofs": int(result.info.n_dofs), "cells": int(result.info.n_cells),
        "qois": _qois(problem, result),
        "convergence": _convergence(result),
        "mass_balance": _mass_balance(result),
        "wall_forces": _forces(result),
        **_extremes(result, fl),
        "cell_peclet": _peclet(result, fl, setup),
        "error_indicator": _error_estimate(result, fl),
    }
    rev = _reversed(result, fl)
    if rev is not None:
        out["reversed_flow"] = rev
    notes = []
    if not out["convergence"]["converged"]:
        notes.append("the solve stopped before reaching its residual target: the answer is not converged "
                     "(a finer or smoother mesh, or the other convection scheme, usually converges)")
    imb = out["mass_balance"].get("imbalance")
    if imb is not None and imb > 1e-3:
        notes.append(f"inflow and outflow differ by {imb:.1%}: the solution is not mass-conserving yet")
    if setup.get("convection") == "upwind" and (out["cell_peclet"]["share_above_2"] or 0) > 0.1:
        notes.append(f"upwind convection with the cell Peclet number above 2 in "
                     f"{out['cell_peclet']['share_above_2']:.0%} of the domain: numerical diffusion is of "
                     f"the order of the physical one there")
    hs = out["error_indicator"]["hotspots"]
    if hs and hs[0]["group_distance"] is not None and hs[0]["group_distance"] <= 2 * (hs[0]["local_h"] or 0):
        notes.append(f"the largest error hotspot ({hs[0]['share']:.0%}) touches group {hs[0]['nearest_group']!r}: "
                     f"the velocity changes over fewer cells than it needs there (wall layer or corner)")
    if notes:
        out["notes"] = notes
    return out


# --------------------------------------------------------------------------- compare


def compare(problem, result_a, result_b) -> dict:
    """How far two flow solves are apart: the finer one's velocity, reconstructed linearly from its
    nearest cell centre, against the coarser one's at the coarser cell centres; relative L2 over the
    domain, the regions where the difference lives, and every statement QoI's change."""
    coarse, fine = sorted((result_a, result_b), key=lambda r: r.info.n_cells)
    fc, ff = _fluid(coarse), _fluid(fine)
    x = fc.C
    _, i = ff.tree.query(x[:, : ff.dim])
    Uf = np.asarray(fine.U)[i] + np.einsum("ni,nij->nj", x - ff.C[i], np.asarray(fine.gradU)[i])
    d2 = np.sum((np.asarray(coarse.U) - Uf) ** 2, axis=1)
    e = fc.V * d2
    norm2 = float(np.sum(fc.V * np.sum(Uf ** 2, axis=1)))
    r = it.RADIUS * fc.diag

    def extra(m, c):
        return {"local_h_fine": _sig(ff.local_size(c, r))}
    regions = _clusters(fc, e, extra)
    for row in regions:                                            # local_h is the coarse mesh's
        row["local_h_coarse"] = row.pop("local_h")
    qa, qb = _qois(problem, coarse), _qois(problem, fine)
    changes = {n: {"coarse": qa[n], "fine": qb[n],
                   "rel_change": _sig((qb[n] - qa[n]) / abs(qb[n]), 2) if qa[n] is not None and qb[n] else None}
               for n in qa}
    return {
        "coarse": {"is": "a" if coarse is result_a else "b", "dofs": int(coarse.info.n_dofs),
                   "cells": int(coarse.info.n_cells)},
        "fine": {"is": "a" if fine is result_a else "b", "dofs": int(fine.info.n_dofs),
                 "cells": int(fine.info.n_cells)},
        "velocity_diff_l2_rel": _sig(math.sqrt(float(e.sum()) / max(norm2, 1e-300))),
        "regions": regions,
        "qoi_change": changes,
    }


# --------------------------------------------------------------------------- render


def _cells_2d(mesh: dict):
    """The 2D cells of the mesh file as polygons (list of (k, 2) arrays), in file order."""
    import meshio
    m = meshio.read(mesh["path"])
    polys = []
    for c in m.cells:
        if c.type in ("triangle", "quad", "triangle6", "quad8", "quad9"):
            k = 3 if c.type.startswith("triangle") else 4
            polys += [m.points[r[:k], :2] for r in np.asarray(c.data)]
    return polys


def _section(result, fl: _Fluid, axis: int, n: int = 220):
    """A grid in the plane through the domain's middle normal to `axis`: (X, Y, cell index or -1,
    the two in-plane axes)."""
    lo, hi = fl.lo, fl.hi
    ax = [a for a in range(3) if a != axis]
    span = hi[ax] - lo[ax]
    nu_, nv = (n, max(8, int(n * span[1] / max(span[0], 1e-300)))) if span[0] >= span[1] else \
        (max(8, int(n * span[0] / max(span[1], 1e-300))), n)
    u = np.linspace(lo[ax[0]], hi[ax[0]], nu_)
    v = np.linspace(lo[ax[1]], hi[ax[1]], nv)
    U, Vv = np.meshgrid(u, v)
    P = np.zeros((U.size, 3))
    P[:, ax[0]], P[:, ax[1]] = U.ravel(), Vv.ravel()
    P[:, axis] = 0.5 * (lo[axis] + hi[axis])
    d, i = fl.tree.query(P)
    i = np.where(d <= 1.05 * fl.reach[i], i, -1)                  # outside the fluid: blank
    return U, Vv, i.reshape(U.shape), ax


def render(problem, mesh: dict, result, kind: str, view: str, out_png) -> str:
    """One PNG: kind = mesh | speed | pressure | error (the error indicator), see the module docstring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fl = _fluid(result)
    kinds = ("mesh", "speed", "pressure", "error")
    if kind not in kinds:
        raise ValueError(f"kind must be one of {kinds}")
    speed = np.linalg.norm(result.U, axis=1)
    eta2, mean = _jumps(result)
    values = {"speed": speed, "pressure": np.asarray(result.p), "error": np.sqrt(mean),
              "mesh": None}[kind]
    label = {"speed": "|U| [m/s]", "pressure": "p [Pa]", "error": "velocity jump at faces [m/s]",
             "mesh": ""}[kind]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if fl.dim == 2:
        polys = _cells_2d(mesh)
        if kind == "mesh":
            ax.add_collection(PolyCollection(polys, facecolors="#e8f0fb", edgecolors="#2b4a7a", linewidths=0.25))
        else:
            # mesh-file cells and solver cells are matched by centroid (the solver may reorder them)
            cent = np.array([p.mean(axis=0) for p in polys])
            _, idx = fl.tree.query(cent)
            v = values[idx]
            norm = matplotlib.colors.LogNorm(vmin=max(v.max() * 1e-3, 1e-300), vmax=max(v.max(), 1e-300)) \
                if kind == "error" else None
            pc = PolyCollection(polys, array=v, cmap="viridis" if kind != "error" else "magma", norm=norm,
                                edgecolors="face", linewidths=0.1)
            ax.add_collection(pc)
            fig.colorbar(pc, ax=ax, label=label, shrink=0.8)
            if kind == "speed":
                _arrows(ax, fl, result, 0, 1)
        ax.autoscale()
        title = f"{problem.id}: {kind}"
    else:
        axis = {"x": 0, "-x": 0, "y": 1, "-y": 1, "z": 2, "-z": 2}.get(view, 2)
        U, Vv, idx, axs = _section(result, fl, axis)
        if kind == "mesh":
            near = np.abs(fl.C[:, axis] - 0.5 * (fl.lo[axis] + fl.hi[axis])) <= fl.reach
            ax.scatter(fl.C[near, axs[0]], fl.C[near, axs[1]], s=1.5, c="#2b4a7a")
            ax.set_title("cell centres within one cell of the section")
        else:
            img = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
            top = max(float(np.nanmax(img)), 1e-300)
            norm = matplotlib.colors.LogNorm(vmin=top * 1e-3, vmax=top) if kind == "error" else None
            pm = ax.pcolormesh(U, Vv, img, cmap="viridis" if kind != "error" else "magma", norm=norm, shading="auto")
            fig.colorbar(pm, ax=ax, label=label, shrink=0.8)
            if kind == "speed":
                _arrows(ax, fl, result, axs[0], axs[1], plane=(axis, 0.5 * (fl.lo[axis] + fl.hi[axis])))
        ax.set_xlabel("xyz"[axs[0]])
        ax.set_ylabel("xyz"[axs[1]])
        title = f"{problem.id}: {kind}, section normal to {'xyz'[axis]}"
    ax.set_aspect("equal")
    ax.set_title(title if not ax.get_title() else f"{title} ({ax.get_title()})", fontsize=10)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def _arrows(ax, fl: _Fluid, result, a: int, b: int, plane: tuple | None = None, n: int = 28) -> None:
    """In-plane velocity arrows on a coarse grid (nearest cell values)."""
    lo, hi = fl.lo, fl.hi
    span = max(hi[a] - lo[a], hi[b] - lo[b])
    step = span / n
    ga = np.arange(lo[a] + step / 2, hi[a], step)
    gb = np.arange(lo[b] + step / 2, hi[b], step)
    A, B = np.meshgrid(ga, gb)
    P = np.zeros((A.size, 3))
    P[:, a], P[:, b] = A.ravel(), B.ravel()
    if plane is not None:
        P[:, plane[0]] = plane[1]
    elif fl.dim == 2:
        P[:, 2] = fl.C[0, 2]
    d, i = fl.tree.query(P[:, : max(fl.dim, 3)] if fl.dim == 3 else P[:, :2])
    keep = d <= fl.reach[i]
    if keep.any():
        U = np.asarray(result.U)[i[keep]]
        ax.quiver(P[keep, a], P[keep, b], U[:, a], U[:, b], color="white", alpha=0.7, width=0.002)
