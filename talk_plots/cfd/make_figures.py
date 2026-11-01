#!/usr/bin/env python
"""Flow (CFD) figures for the README: feacore's flow solver on two classic benchmarks.

Regenerates talk_plots/cfd/ (PNG 200 dpi + .md caption per figure) and cfd_numbers.json:

  fig_cylinder_wake       DFG benchmark 2D-1 (Schaefer & Turek 1996), Re = 20: speed and streamlines
  fig_cylinder_vorticity  the same flow's vorticity: the shear layers that leave the cylinder
  fig_cylinder_mesh       close to the cylinder: the block-structured mesh and the recirculation bubble
  fig_cavity              lid-driven cavity, Re = 100: streamlines over speed, the main vortex and the corner eddies
  fig_cavity_ghia         the cavity's centreline velocities against Ghia, Ghia & Shin (1982)

Each case is the problem's own reference setup (feacore/problems/cfd/) on its block-structured quad mesh,
solved by the openfoam backend: steady laminar SIMPLEC, linearUpwind convection, residuals down to 1e-8.
It needs OpenFOAM and foamlib (the `cfd` extra; CLAUDE.md, FEA core, CFD). The three solves take a few
minutes. Their cell and wall values are cached in talk_plots/cfd/_cache/ (not in git); --resolve solves again.

    python talk_plots/cfd/make_figures.py [--resolve]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator  # noqa: E402

from feacore.problems.cfd import cylinder as cyl_mod  # noqa: E402
from feacore.problems.cfd.cavity import _GHIA_POINTS  # noqa: E402
from feacore.problems.registry import get  # noqa: E402

CACHE = HERE / "_cache"
CYL, CAV = "cylinder_channel_dfg_2d1", "lid_driven_cavity_re100"
SOLVES = {"cyl": (CYL, 4), "cyl_mesh": (CYL, 2), "cav": (CAV, 4)}     # (problem, k = h0 / h): study levels

# Ghia, Ghia & Shin (1982), Re = 100 columns of Table I (u on x = 0.5) and Table II (v on y = 0.5), as
# transcribed in PySPH (feacore/problems/cfd/_common.py, GHIA_COPY). The printed coordinates are rounded
# grid points j / 128 of their 129 x 129 grid.
GHIA_Y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
          0.1016, 0.0703, 0.0625, 0.0547, 0.0000]
GHIA_U = [1.0000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090,
          -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000]
GHIA_X = [1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563,
          0.0938, 0.0781, 0.0703, 0.0625, 0.0000]
GHIA_V = [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
          0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000]

# Colours from the dataviz reference palette. Speed is one blue ramp (light = slow); vorticity is blue <-> red
# around a neutral grey; text stays in ink.
INK, INK2, MESH = "#0b0b0b", "#52514e", "#8a8984"
SPEED = LinearSegmentedColormap.from_list(
    "speed", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])
VORT = LinearSegmentedColormap.from_list(
    "vorticity", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b4ae", "#e34948", "#a61f24"])
C_SOLVER, C_GHIA = "#2a78d6", "#eb6834"
SOLID = "#bdbcb6"


# --------------------------------------------------------------------------- solves (cached)


def solve(key: str) -> dict:
    """Solve one study level and keep what the figures need: cell and wall-face values, probes, QoIs."""
    from feacore.solver import solve as fea_solve

    pid, k = SOLVES[key]
    prob = get(pid)
    with tempfile.TemporaryDirectory() as tmp:
        msh = prob.mesh(Path(tmp) / f"{pid}.msh", prob.params["h0"] / k)
        case = prob.case(msh, order=2)                      # flow: order 2 = linearUpwind convection
        res = fea_solve(case, backend="openfoam")
        try:
            g = res.gradU                                    # [cell, i, j] = d U_j / d x_i
            walls = list(res.patch.values())
            out = {
                "C": res.C[:, :2], "U": res.U[:, :2], "w": g[:, 0, 1] - g[:, 1, 0],
                "bC": np.vstack([d["Cf"][:, :2] for d in walls]), "bU": np.vstack([d["U"][:, :2] for d in walls]),
                "info": json.dumps({"cells": int(res.fm.n_cells), "k": k, "h": prob.params["h0"] / k,
                                    "iterations": res.info.iterations, "converged": res.info.converged,
                                    "residuals": res.info.residuals, "wall_s": res.info.wall_time,
                                    "openfoam": res.info.openfoam}),
                "qois": json.dumps({q.name: float(res.evaluate(q)) for q in case.qois}),
            }
            if pid == CAV:
                s = np.linspace(0.0, 1.0, 257)
                out["vcl"] = res.probe("ux", np.column_stack([np.full_like(s, 0.5), s]))     # u on x = 0.5
                out["hcl"] = res.probe("uy", np.column_stack([s, np.full_like(s, 0.5)]))     # v on y = 0.5
                gy, gx = np.round(np.array(GHIA_Y) * 128) / 128, np.round(np.array(GHIA_X) * 128) / 128
                out["ghia_u"] = res.probe("ux", np.column_stack([np.full_like(gy, 0.5), gy]))
                out["ghia_v"] = res.probe("uy", np.column_stack([gx, np.full_like(gx, 0.5)]))
            else:
                p = prob.params
                xr = p["xc"] + p["D"] / 2                         # rear stagnation point
                s = np.linspace(xr + 1e-5, xr + 0.2, 2001)
                out["axis_x"], out["axis_u"] = s, res.probe("ux", np.column_stack([s, np.full_like(s, p["yc"])]))
            if key == "cyl_mesh":
                import meshio

                m = meshio.read(msh)
                out["mesh_points"] = m.points[:, :2]
                out["mesh_cells"] = np.vstack([c.data for c in m.cells if c.type == "quad"])
        finally:
            res.cleanup()
    return out


def load(key: str, resolve: bool) -> dict:
    path = CACHE / f"{key}.npz"
    if resolve or not path.exists():
        print(f"solving {key} ({SOLVES[key][0]}, h = h0 / {SOLVES[key][1]}) ...", flush=True)
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, **solve(key))
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["info"], d["qois"] = json.loads(str(d["info"])), json.loads(str(d["qois"]))
    return d


# --------------------------------------------------------------------------- numbers


def numbers(cyl: dict, cyl_mesh: dict, cav: dict) -> dict:
    prob = get(CYL)
    s = prob.force_scale
    q = cyl["qois"]
    cd, cl, dp = q["drag"] / s, q["lift"] / s, q["dp_front_rear"]
    axis = {}
    for name, d in (("k4", cyl), ("k2", cyl_mesh)):
        u = d["axis_u"]
        j = int(np.argmax(u > 0))                          # first positive u behind the cylinder
        x0 = d["axis_x"][j - 1] - u[j - 1] * (d["axis_x"][j] - d["axis_x"][j - 1]) / (u[j] - u[j - 1])
        axis[name] = float(x0 - (prob.params["xc"] + prob.params["D"] / 2))
    gu, gv = np.array(GHIA_U), np.array(GHIA_V)
    inner_u, inner_v = slice(1, -1), slice(1, -1)          # leave out the wall rows (boundary values)
    du = cav["ghia_u"][inner_u] - gu[inner_u]
    dv = cav["ghia_v"][inner_v] - gv[inner_v]
    cav_q = {name: {"solver": cav["qois"][name], "ghia": v, "converged": conv,
                    "vs_ghia": cav["qois"][name] / v - 1, "vs_converged": cav["qois"][name] / conv - 1}
             for name, _, _, v, _, _, conv in _GHIA_POINTS}
    return {
        "cylinder": {
            **cyl["info"], "c_D": cd, "c_L": cl, "dp_front_rear_Pa": dp,
            "c_D_ref": cyl_mod.CD_REF, "c_L_ref": cyl_mod.CL_REF, "dp_ref_Pa": cyl_mod.DP_REF,
            "c_D_err": cd / cyl_mod.CD_REF - 1, "c_L_err": cl / cyl_mod.CL_REF - 1, "dp_err": dp / cyl_mod.DP_REF - 1,
            "recirculation_length_m": axis["k4"], "Re": prob.reynolds, "U_mean": prob.U_mean,
        },
        "cylinder_mesh": {**cyl_mesh["info"], "recirculation_length_m": axis["k2"]},
        "cavity": {
            **cav["info"], "Re": get(CAV).reynolds, "qois": cav_q,
            "ghia_max_abs_du": float(np.max(np.abs(du))), "ghia_max_abs_dv": float(np.max(np.abs(dv))),
            "ghia_rms_du": float(np.sqrt(np.mean(du ** 2))), "ghia_rms_dv": float(np.sqrt(np.mean(dv ** 2))),
        },
    }


# --------------------------------------------------------------------------- style and output


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    })


def titles(fig, title: str, sub: str, y: float = 0.975):
    fig.suptitle(title, x=0.03, y=y, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.03, y - 0.06 * 6.75 / fig.get_figheight(), sub, ha="left", va="top", fontsize=14, color="#555555")


def save(fig, stem: str, md: str):
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    plt.close(fig)
    (HERE / f"{stem}.md").write_text(md.strip() + "\n")
    print("wrote", stem)


def grid(d: dict, box, per_m: float, inside=None):
    """Velocity (cells + wall faces, linear) and vorticity (cells, linear; nearest in the last half cell)
    sampled on a regular grid over box = (x0, x1, y0, y1); NaN where inside(X, Y) is True."""
    x0, x1, y0, y1 = box
    x = np.linspace(x0, x1, int(round((x1 - x0) * per_m)) + 1)
    y = np.linspace(y0, y1, int(round((y1 - y0) * per_m)) + 1)
    X, Y = np.meshgrid(x, y)
    UV = LinearNDInterpolator(np.vstack([d["C"], d["bC"]]), np.vstack([d["U"], d["bU"]]))(X, Y)
    W = LinearNDInterpolator(d["C"], d["w"])(X, Y)
    W = np.where(np.isnan(W), NearestNDInterpolator(d["C"], d["w"])(X, Y), W)
    if inside is not None:
        m = inside(X, Y)
        UV[m], W[m] = np.nan, np.nan
    return x, y, UV[..., 0], UV[..., 1], W


def in_cylinder(X, Y):
    p = get(CYL).params
    return (X - p["xc"]) ** 2 + (Y - p["yc"]) ** 2 < (p["D"] / 2) ** 2


def cylinder_patch(ax):
    p = get(CYL).params
    ax.add_patch(Circle((p["xc"], p["yc"]), p["D"] / 2, fc=SOLID, ec=INK2, lw=1.2, zorder=5))


def eddies(ax, x, y, U, V, fractions, lw: float, arrows: bool = True) -> np.ndarray:
    """Draw the recirculation bubble's two eddies as closed streamlines and return the bubble as a grid mask.

    The streamlines are contours of the stream function behind the cylinder, psi = int u dy from the axis
    y = yc minus int v dx along the axis (u = dpsi/dy, v = -dpsi/dx; psi = 0 at the rear stagnation point).
    The upper eddy has psi < 0 above the axis, the lower one psi > 0 below it; each gets contours at the
    given fractions of its extreme value."""
    p = get(CYL).params
    xr = p["xc"] + p["D"] / 2
    sel = x >= xr                                          # behind the cylinder: columns from the axis are free
    xs, jc = x[sel], int(np.argmin(np.abs(y - p["yc"])))
    Us, Va = np.nan_to_num(U[:, sel]), np.nan_to_num(V[jc, sel])    # masked points on the wall: no slip, 0
    psi = cumulative_trapezoid(Us, y, axis=0, initial=0.0)
    psi = psi - psi[jc] - cumulative_trapezoid(Va, xs, initial=0.0)[None, :]
    above, below = (y > y[jc])[:, None], (y < y[jc])[:, None]
    up, lo = np.where(above, psi, np.nan), np.where(below, psi, np.nan)
    bubble = np.zeros(U.shape, dtype=bool)
    bubble[:, sel] = (above & (psi < 0)) | (below & (psi > 0))
    for f, ext, sign in ((up, np.nanmin(up), -1), (lo, np.nanmax(lo), 1)):
        levels = sorted(fr * ext for fr in fractions)
        ax.contour(xs, y, f, levels=levels, colors=INK, linewidths=lw, linestyles="solid", zorder=3)
        if arrows:                                         # flow direction on the outer loop, far from the axis
            jcore, icore = np.unravel_index(np.nanargmin(f) if sign < 0 else np.nanargmax(f), f.shape)
            col = f[:, icore]
            out = min(levels, key=abs)
            rows = range(jcore, len(y)) if sign < 0 else range(jcore, -1, -1)
            j = next(r for r in rows if (col[r] > out if sign < 0 else col[r] < out))
            u, v = U[j, sel][icore], V[j, sel][icore]
            dx = 0.004 * (x[-1] - x[0]) / np.hypot(u, v)
            ax.annotate("", xy=(xs[icore] + u * dx, y[j] + v * dx), xytext=(xs[icore] - u * dx, y[j] - v * dx),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw, mutation_scale=9), zorder=4)
    return bubble


def field_axes(ax, box):
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# --------------------------------------------------------------------------- figures


def fig_cylinder_wake(d: dict, n: dict):
    box = (0.0, 1.1, 0.0, 0.41)
    x, y, U, V, _ = grid(d, box, 1000, in_cylinder)
    S = np.hypot(U, V)
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.065, 0.13, 0.83, 0.62])
    im = ax.imshow(np.ma.masked_invalid(S), extent=box, origin="lower", cmap=SPEED, vmin=0, vmax=0.45,
                   interpolation="bilinear")
    bubble = eddies(ax, x, y, U, V, (0.3, 0.75), lw=0.6, arrows=False)
    U[bubble], V[bubble] = np.nan, np.nan                 # the eddies are drawn above; streamlines stay out
    ax.streamplot(x, y, np.ma.masked_invalid(U), np.ma.masked_invalid(V), density=(3.2, 1.2), color=INK,
                  linewidth=0.6, arrowsize=0.7, zorder=3)
    cylinder_patch(ax)
    field_axes(ax, box)
    cax = fig.add_axes([0.915, 0.13, 0.012, 0.62])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("speed (m/s)")
    c = n["cylinder"]
    titles(fig, f"Flow past a cylinder at Re = {c['Re']:.0f}",
           f"DFG benchmark 2D-1, solved by feacore's OpenFOAM backend on {c['cells']:,} cells.\n"
           f"The pressure drop across the cylinder is {abs(c['dp_err']) * 100:.2f} % off the reference.")
    save(fig, "fig_cylinder_wake", f"""
# fig_cylinder_wake: "Flow past a cylinder at Re = {c['Re']:.0f}"

**Caption.** Steady laminar flow around a cylinder in a channel, the DFG benchmark 2D-1 (Schaefer & Turek 1996):
speed with streamlines, x <= 1.1 m of the 2.2 m channel. The inflow is parabolic with mean speed
{c['U_mean']:.3g} m/s, and Re = U_mean D / nu = {c['Re']:.0f}. The flow splits around the cylinder, speeds up in the
two gaps, and closes behind it in a small recirculation bubble ({c['recirculation_length_m'] * 1000:.1f} mm long).

**Key numbers** (`cfd_numbers.json`, cylinder)
- Mesh: the problem's own block-structured quad mesh at h = h0 / {c['k']} ({c['cells']:,} cells).
- Pressure drop front to back of the cylinder: {c['dp_front_rear_Pa']:.5f} Pa against the reference
  {c['dp_ref_Pa']:.5f} Pa ({c['dp_err']:+.2%}).
- Drag coefficient {c['c_D']:.4f} against {c['c_D_ref']:.4f} ({c['c_D_err']:+.2%}); lift coefficient
  {c['c_L']:.5f} against {c['c_L_ref']:.5f} ({c['c_L_err']:+.1%}).
- Solver: {c['iterations']} SIMPLEC iterations, converged = {c['converged']}, {c['wall_s']:.0f} s ({c['openfoam']}).

**Sources.** `feacore/problems/cfd/cylinder.py` (setup, mesh, FEATFLOW reference values), solved with
`feacore.solver.solve(case, backend="openfoam")`.

**Caveats**
- The drag is a wall-force integral. On this branch it converges about 0.15 % low (`docs/benchmarks_cfd.md`).
- Streamlines are drawn from the solution interpolated linearly between cell centres and wall faces. The two
  eddies behind the cylinder are contours of the stream function there (`eddies()`), the rest matplotlib
  streamlines.
""")


def fig_cylinder_vorticity(d: dict, n: dict):
    box = (0.1, 0.62, 0.0, 0.41)
    _, _, _, _, W = grid(d, box, 1500, in_cylinder)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.11, 0.11, 0.72, 0.68])
    lim = 20.0
    im = ax.imshow(np.ma.masked_invalid(W), extent=box, origin="lower", cmap=VORT,
                   norm=TwoSlopeNorm(0.0, -lim, lim), interpolation="bilinear")
    cylinder_patch(ax)
    field_axes(ax, box)
    cax = fig.add_axes([0.86, 0.11, 0.02, 0.68])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("vorticity (1/s)")
    titles(fig, "Vorticity: the shear layers", "Red turns anticlockwise, blue clockwise. They leave the cylinder\n"
           "and the channel walls, and diffuse downstream.")
    save(fig, "fig_cylinder_vorticity", f"""
# fig_cylinder_vorticity: "Vorticity: the shear layers"

**Caption.** Vorticity (dv/dx - du/dy) of the DFG 2D-1 flow, 0.1 <= x <= 0.62 m: the same solve as
`fig_cylinder_wake` ({n['cylinder']['cells']:,} cells). Clockwise vorticity (blue) leaves the top of the cylinder
and the bottom wall, anticlockwise (red) the bottom of the cylinder and the top wall. At Re = 20 the two layers
behind the cylinder stay attached to a closed, steady bubble; there is no vortex shedding.

**Caveats**
- Cell-centred Gauss gradients, interpolated linearly between cell centres. The colour scale is clipped at
  +-{lim:.0f} 1/s; the vorticity on the cylinder's surface is larger.
""")


def fig_cylinder_mesh(d: dict, n: dict):
    box = (0.135, 0.405, 0.095, 0.305)
    x, y, U, V, _ = grid(d, box, 3000, in_cylinder)
    S = np.hypot(U, V)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.11, 0.11, 0.72, 0.68])
    im = ax.imshow(np.ma.masked_invalid(S), extent=box, origin="lower", cmap=SPEED, vmin=0, vmax=0.45,
                   interpolation="bilinear", alpha=0.85)
    P, Q = d["mesh_points"], d["mesh_cells"]
    cx, cy = P[Q].mean(axis=1).T
    keep = (cx > box[0] - 0.01) & (cx < box[1] + 0.01) & (cy > box[2] - 0.01) & (cy < box[3] + 0.01)
    edges = P[Q[keep]][:, [0, 1, 2, 3, 0]]
    ax.add_collection(LineCollection([e for e in edges], colors=MESH, linewidths=0.25, alpha=0.7, zorder=2))
    p = get(CYL).params
    xr = p["xc"] + p["D"] / 2
    L = n["cylinder_mesh"]["recirculation_length_m"]
    bubble = eddies(ax, x, y, U, V, (0.12, 0.35, 0.6, 0.85), lw=0.8)
    U[bubble], V[bubble] = np.nan, np.nan                 # the eddies are drawn above; streamlines stay out
    ax.streamplot(x, y, np.ma.masked_invalid(U), np.ma.masked_invalid(V), density=1.4, color=INK, linewidth=0.7,
                  arrowsize=0.8, zorder=3)
    ax.annotate("", xy=(xr + L, p["yc"]), xytext=(xr, p["yc"]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xr + L / 2, p["yc"] - 0.045, f"recirculation\n{L * 1000:.1f} mm", ha="center", va="top", fontsize=13,
            color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    cylinder_patch(ax)
    field_axes(ax, box)
    cax = fig.add_axes([0.86, 0.11, 0.02, 0.68])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("speed (m/s)")
    m = n["cylinder_mesh"]
    titles(fig, "Close to the cylinder", f"The block-structured quad mesh ({m['cells']:,} cells) and the closed\n"
           "recirculation bubble behind the cylinder.")
    save(fig, "fig_cylinder_mesh", f"""
# fig_cylinder_mesh: "Close to the cylinder"

**Caption.** The DFG 2D-1 flow close to the cylinder, solved on the problem's block-structured quad mesh at
h = h0 / {m['k']} ({m['cells']:,} cells; grey lines): an O-grid of {int(round(64 * m['k']))} cells around the
cylinder inside a square box, graded towards the walls. Behind the cylinder the streamlines close into two
counter-rotating eddies, {m['recirculation_length_m'] * 1000:.1f} mm long on the axis ({n['cylinder']['recirculation_length_m'] * 1000:.1f} mm on
the finer h0 / {n['cylinder']['k']} mesh of `fig_cylinder_wake`).

**Key numbers** (`cfd_numbers.json`, cylinder_mesh): {m['iterations']} iterations, converged = {m['converged']}.

**Caveats**
- The recirculation length is where u turns positive on the axis y = 0.2 m behind the cylinder, read with the
  solver's own probe (`OpenFOAMResult.probe`).
- The eddies are contours of the stream function behind the cylinder (`eddies()`), with an arrowhead for the
  direction of turn; the other lines are matplotlib streamlines.
""")


def fig_cavity(d: dict, n: dict):
    box = (0.0, 1.0, 0.0, 1.0)
    x, y, U, V, _ = grid(d, box, 800)
    # the hull of cell centres and wall-face centres leaves out a few points at each corner: walls, so 0
    U, V = np.nan_to_num(U), np.nan_to_num(V)
    S = np.hypot(U, V)
    psi = cumulative_trapezoid(U, y, axis=0, initial=0.0)          # psi = int_0^y u dy (0 on the walls)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_axes([0.05, 0.1, 4.3 / 8, 4.3 / 6.5])
    im = ax.imshow(S, extent=box, origin="lower", cmap=SPEED, vmin=0, vmax=1.0, interpolation="bilinear")
    main = [-0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-3, -1e-4]
    eddy = [1e-7, 1e-6, 4e-6, 1e-5]
    ax.contour(x, y, psi, levels=main, colors=INK, linewidths=0.9, linestyles="solid", zorder=3)
    ax.contour(x, y, psi, levels=eddy, colors=INK, linewidths=0.9, linestyles="--", zorder=3)
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    ax.plot(x[i], y[j], "o", ms=7, mfc="white", mec=INK, mew=1.4, zorder=4)
    kw = dict(fontsize=13, color=INK, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.annotate(f"vortex centre\n({x[i]:.3f}, {y[j]:.3f})", xy=(x[i], y[j]), xytext=(1.05, 0.74), ha="left",
                va="center", **kw)
    ax.annotate("corner eddies (dashed)", xy=(0.06, 0.03), xytext=(0.5, -0.07), ha="center", va="top", **kw)
    ax.annotate("", xy=(0.93, 0.04), xytext=(0.72, -0.075), **kw)
    ax.annotate("", xy=(0.72, 1.04), xytext=(0.28, 1.04), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.5, 1.065, "lid, 1 m/s", ha="center", va="bottom", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    cax = fig.add_axes([0.88, 0.22, 0.022, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("speed (m/s)")
    c = n["cavity"]
    titles(fig, f"Lid-driven cavity at Re = {c['Re']:.0f}", f"Streamlines over speed in the unit square, "
           f"{c['cells']:,} cells.")
    save(fig, "fig_cavity", f"""
# fig_cavity: "Lid-driven cavity at Re = {c['Re']:.0f}"

**Caption.** A unit square of fluid driven by its lid (1 m/s, Re = {c['Re']:.0f}), the classic test of Ghia, Ghia & Shin
(1982). Streamlines are contours of the stream function psi = integral of u dy: solid for the main vortex
(psi = {', '.join(f'{v:g}' for v in main)}), dashed for the two counter-rotating corner eddies
(psi = {', '.join(f'{v:g}' for v in eddy)}). The main vortex centre sits at ({x[i]:.3f}, {y[j]:.3f}), where
psi = {psi[j, i]:.4f} m^2/s.

**Key numbers** (`cfd_numbers.json`, cavity): uniform {int(round(np.sqrt(c['cells'])))} x {int(round(np.sqrt(c['cells'])))} quad
grid, {c['iterations']} iterations, converged = {c['converged']}.

**Caveats**
- psi is integrated on an 801 x 801 sampling grid of the linearly interpolated velocity; the vortex centre is
  that grid's psi minimum (spacing 1.25 mm).
""")


def fig_cavity_ghia(d: dict, n: dict):
    s = np.linspace(0.0, 1.0, 257)
    gy, gx = np.round(np.array(GHIA_Y) * 128) / 128, np.round(np.array(GHIA_X) * 128) / 128
    c = n["cavity"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 6))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.12, top=0.74, wspace=0.32)
    a1.axvline(0, color="#e6e6e6", lw=1, zorder=0)
    a1.plot(d["vcl"], s, color=C_SOLVER, lw=2, label="OpenFOAM")
    a1.plot(GHIA_U, gy, "o", ms=8, mfc="white", mec=C_GHIA, mew=1.8, label="Ghia et al. (1982)")
    a1.set_xlabel("u on x = 0.5 (m/s)")
    a1.set_ylabel("y (m)")
    a1.set_ylim(0, 1)
    a2.axhline(0, color="#e6e6e6", lw=1, zorder=0)
    a2.plot(s, d["hcl"], color=C_SOLVER, lw=2)
    a2.plot(GHIA_X, GHIA_V, "o", ms=8, mfc="white", mec=C_GHIA, mew=1.8)
    a2.set_xlabel("x (m)")
    a2.set_ylabel("v on y = 0.5 (m/s)")
    a2.set_xlim(0, 1)
    a2.text(0.62, 0.0, "OpenFOAM", color=INK, fontsize=13, ha="left", va="center")
    a2.text(0.14, 0.075, "Ghia et al.", color=INK, fontsize=13, ha="left", va="center")
    a1.legend(loc="lower right", frameon=False, fontsize=13, handlelength=1.6)
    worst = max(c["ghia_max_abs_du"], c["ghia_max_abs_dv"])
    titles(fig, "Cavity: the classic check", f"Centreline velocities against Ghia et al.'s table: the largest gap\n"
           f"is {worst:.4f} m/s ({worst * 100:.1f} % of the lid speed).")
    rows = "\n".join(f"| `{name}` | {q['solver']:.5f} | {q['ghia']:.5f} ({q['vs_ghia']:+.1%}) | {q['converged']:.5f} "
                     f"({q['vs_converged']:+.2%}) |" for name, q in c["qois"].items())
    save(fig, "fig_cavity_ghia", f"""
# fig_cavity_ghia: "Cavity: the classic check"

**Caption.** Velocity along the two centrelines of the Re = {c['Re']:.0f} lid-driven cavity: u on the vertical line x = 0.5
(left) and v on the horizontal line y = 0.5 (right). Lines: the OpenFOAM solve of `fig_cavity`
({c['cells']:,} cells), read with the solver's own probe. Circles: the Re = 100 columns of Ghia, Ghia & Shin
(1982), Tables I and II (17 points each, at their grid points j / 128).

**Key numbers** (`cfd_numbers.json`, cavity)
- Largest gap to the 15 interior table points: {c['ghia_max_abs_du']:.4f} m/s in u, {c['ghia_max_abs_dv']:.4f} m/s in v
  (RMS {c['ghia_rms_du']:.4f} and {c['ghia_rms_dv']:.4f}), with the lid at 1 m/s.
- The four checked points of `feacore/problems/cfd/cavity.py`:

| QoI | this solve | Ghia | mesh-converged solver value |
|---|---|---|---|
{rows}

**Caveats**
- Ghia's Re = 100 values are 1.4-3.5 % smaller in magnitude than mesh-converged solutions (Richardson
  extrapolation of the solver's 32 / 64 / 128 study, GCI <= 0.25 %, in line with Sahin & Owens 2003;
  `feacore/problems/cfd/cavity.py`). Most of the gap to the table is the table's own error.
""")


def write_readme(n: dict):
    c, m, v = n["cylinder"], n["cylinder_mesh"], n["cavity"]
    worst = max(v["ghia_max_abs_du"], v["ghia_max_abs_dv"])
    (HERE / "README.md").write_text(f"""# Flow (CFD): the fixed flow solver on two classic benchmarks

`feacore` solves steady laminar flow with an OpenFOAM backend (`backend="openfoam"`, physics "flow"; see
`docs/solver_openfoam.md` and `docs/benchmarks_cfd.md`). These figures show two of its benchmarks, each solved with
the problem's own reference setup on its block-structured quad mesh. Every number is in `cfd_numbers.json`, and
each figure's `.md` has its caption, numbers and caveats.

| figure | what it shows |
|---|---|
| [fig_cylinder_wake](fig_cylinder_wake.md) ([png](fig_cylinder_wake.png)) | DFG 2D-1 cylinder, Re = {c['Re']:.0f}: speed and streamlines. Pressure drop across the cylinder {c['dp_front_rear_Pa']:.5f} Pa, {abs(c['dp_err']) * 100:.2f} % {'below' if c['dp_err'] < 0 else 'above'} the reference ({c['cells']:,} cells). |
| [fig_cylinder_vorticity](fig_cylinder_vorticity.md) ([png](fig_cylinder_vorticity.png)) | The same flow's vorticity: shear layers from the cylinder and the walls, no shedding at Re = {c['Re']:.0f}. |
| [fig_cylinder_mesh](fig_cylinder_mesh.md) ([png](fig_cylinder_mesh.png)) | Close up on the {m['cells']:,}-cell mesh: the O-grid around the cylinder and the {m['recirculation_length_m'] * 1000:.1f} mm recirculation bubble. |
| [fig_cavity](fig_cavity.md) ([png](fig_cavity.png)) | Lid-driven cavity, Re = {v['Re']:.0f}: the main vortex and the two corner eddies ({v['cells']:,} cells). |
| [fig_cavity_ghia](fig_cavity_ghia.md) ([png](fig_cavity_ghia.png)) | The cavity's centreline velocities on Ghia et al.'s table: largest gap {worst:.4f} m/s at a lid speed of 1 m/s. |

Rebuild (needs OpenFOAM and foamlib; a few minutes for the three solves, cached in `_cache/`):

```bash
python talk_plots/cfd/make_figures.py            # --resolve to solve again
```

The ACE learning loop runs on these problems too, as the `cfd` and `cfd3d` suites (`python -m ace.run_loop
--suite cfd`, PR #21).

## Supersonic flow: the swarm's first generation

[supersonic_g0.md](supersonic_g0.md): generation 0 of a 6-agent swarm run on a supersonic suite, with compressible
flow from PR #24 (the run is in progress). The target flow around a Mach 3 blunted wedge
([png](fig_supersonic_reference.png)), agent a0's first mesh with the reference bow shock over it
([png](fig_supersonic_mesh_g0.png)), and a0's run report as the teacher saw it ([`supersonic_g0/`](supersonic_g0/)).
`make_supersonic_figures.py` rebuilds them from the run folder and its field references, which are not in git.
""")
    print("wrote README.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolve", action="store_true", help="solve again instead of using talk_plots/cfd/_cache/")
    args = ap.parse_args()
    cyl, cyl_mesh, cav = (load(k, args.resolve) for k in SOLVES)
    n = numbers(cyl, cyl_mesh, cav)
    (HERE / "cfd_numbers.json").write_text(json.dumps(n, indent=1) + "\n")
    style()
    fig_cylinder_wake(cyl, n)
    fig_cylinder_vorticity(cyl, n)
    fig_cylinder_mesh(cyl_mesh, n)
    fig_cavity(cav, n)
    fig_cavity_ghia(cav, n)
    write_readme(n)


if __name__ == "__main__":
    main()
