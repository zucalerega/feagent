"""A full report of one run for the --deep loop: numbers, setup, the WHOLE mesh, where the error is, and pictures.

    build_run_report(run, out_dir)   heavy (one re-solve + images, ~1-5 s in 2D); runs in a scoring worker
    add_scores(report, run)          cheap, in the parent once the population is scored
    report_text(report)              the report as compact text for an LLM (teacher, self-reflection)

The teacher and the agent's own self-reflection read this instead of a few summary numbers. It
holds the same error map and mesh layout as the teacher's inspection tools (ace.inspect_tools) plus:

  region_grid   the whole mesh as a coarse table: the bounding box cut into cells (10 x 10 in 2D,
                6 x 6 x 4 in 3D); per cell the number of elements, their mean size, the cell's share
                of the total error energy and the worst local von Mises error in it. A text-only
                reader can see where elements are and where the error is, everywhere.
  check_points  every check point (disclosed and hidden) with reference and run von Mises.
  images        mesh (element edges, named groups coloured, check points marked), error (log error
                energy density, mesh overlaid, hotspots numbered) and stress (the run's von Mises
                next to the reference's, same colour scale). 2D: flat plots (axisymmetric: x = r,
                y = z); 3D: the outer surface from two opposite views.

The run's scored setup is re-solved ONCE on its own mesh (the solver is deterministic, so this is
the solution the agent got) and every piece is computed from that one solve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from ace import config, fea
from ace import field_error as fe
from ace import inspect_tools as it
from ace.checks import reference_setup
from ace.memory import agent_of

GRID = {2: (10, 10), 3: (6, 6, 4)}   # region_grid cells per axis (3D: the 4 goes on the shortest axis)
DPI = 100                           # images are at most 11 inches wide -> <= 1100 px
LOG_RANGE = 3                       # error image shows 3 decades of error density


# --------------------------------------------------------------------------- the one solve and what follows from it


def _field(problem_id: str, p, res) -> dict:
    """Everything pointwise, from one solve: the run's and the reference's stresses at the reference
    integration points, the error energy per point (as in inspect_tools.error_map) and von Mises."""
    mat = reference_setup(p)["material"]
    ref = fe.reference(problem_id)
    pts, sref = ref["pts"], ref["sigma"]
    srun = fe.stresses(res, pts, p.dim)
    e = it._energy_density({k: srun[k] - sref[k] for k in srun}, ref["w"], mat["E"], mat["nu"])
    vm_ref, vm_run = fe.von_mises(sref), fe.von_mises(srun)
    # Pointwise relative von Mises error. Near-zero reference stress would make it blow up, so the
    # denominator is floored at 0.2 x the median (the same floor probe_points uses).
    rel = np.abs(vm_run - vm_ref) / np.maximum(vm_ref, 0.2 * np.median(vm_ref))
    return {"pts": pts, "w": ref["w"], "e": e, "total": float(e.sum()), "vm_ref": vm_ref, "vm_run": vm_run,
            "rel": rel, "ref_energy": fe.energy_sq(sref, ref["w"], mat["E"], mat["nu"])}


def _error_map(problem_id: str, run: dict, p, res, msh: it._Mesh, f: dict, notes: list) -> dict:
    """Same output as inspect_tools.error_map, computed from the solve already done (that function
    re-solves; this one reuses `res`)."""
    pts, e, total = f["pts"], f["e"], f["total"]
    sig = it._sig
    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    r = it.RADIUS * diag
    spots = sorted(it._hotspots(pts, e, r), key=lambda m: -e[m].sum())
    rows = []
    for m in spots[: it.MAX_HOTSPOTS]:
        c = np.average(pts[m], axis=0, weights=e[m])
        group, dist = msh.nearest_group(c)
        rows.append({"centroid": [sig(x) for x in c[: p.dim]], "share": sig(e[m].sum() / total, 2),
                     "nearest_group": group, "distance": sig(dist), "local_h": sig(msh.local_size(c, r)),
                     "vm_ref": sig(f["vm_ref"][m].mean()),
                     "vm_rel_error": sig(np.abs(f["vm_run"][m] - f["vm_ref"][m]).mean() / f["vm_ref"][m].mean())})
    pp = fe.probe_points(problem_id)
    vm_p = fe.von_mises(fe.stresses(res, np.array(pp["points"]), p.dim))
    rel = (vm_p - np.array(pp["vm_ref"])) / np.array(pp["vm_ref"])
    checks = [{"i": i, "point": [sig(x) for x in pt[: p.dim]], "hidden": i >= config.PROBE_DISCLOSED,
               "vm_rel_error": sig(rel[i])} for i, pt in enumerate(pp["points"])]
    hidden = sorted((c for c in checks if c["hidden"]), key=lambda c: -abs(c["vm_rel_error"]))
    out = {"run_id": run.get("run_id"), "field_error": sig(np.sqrt(total / f["ref_energy"])),
           "probe_error": sig(np.sqrt(np.mean(rel ** 2))), "model_diagonal": sig(diag), "hotspot_radius": sig(r),
           "median_h": sig(msh.median), "n_hotspots": len(spots), "hotspots": rows, "check_points": checks,
           "worst_hidden": [c["i"] for c in hidden[:3]], "notes": notes}
    # The full check-point rows of the report (vm values too) come from the same numbers.
    out["_cp_full"] = [{"i": i, "xyz": [sig(x) for x in pt[: p.dim]], "disclosed": i < config.PROBE_DISCLOSED,
                        "vm_ref": sig(pp["vm_ref"][i]), "vm_run": sig(vm_p[i]), "rel_err": sig(rel[i])}
                       for i, pt in enumerate(pp["points"])]
    return out


def _region_grid(dim: int, msh: it._Mesh, f: dict) -> dict:
    """The bounding box cut into cells; per non-empty cell: centre, elements (by centroid), their
    mean size, share of the total error energy, worst pointwise relative von Mises error."""
    lo = np.minimum(msh.points[:, :dim].min(0), f["pts"][:, :dim].min(0))
    hi = np.maximum(msh.points[:, :dim].max(0), f["pts"][:, :dim].max(0))
    n = np.array(GRID[dim])
    if dim == 3:                                  # the 4 cells go on the shortest axis
        n = np.full(3, 6)
        n[int(np.argmin(hi - lo))] = 4
    span = np.where(hi > lo, hi - lo, 1.0)

    def cell_of(x):                               # flat cell index of each point (clipped into the box)
        ijk = np.clip(((x[:, :dim] - lo) / span * n).astype(int), 0, n - 1)
        return np.ravel_multi_index(ijk.T, n)

    ce, cp = cell_of(msh.centroids), cell_of(f["pts"])
    ncell = int(n.prod())
    n_el = np.bincount(ce, minlength=ncell)
    h_sum = np.bincount(ce, weights=msh.size, minlength=ncell)
    share = np.bincount(cp, weights=f["e"], minlength=ncell) / f["total"]
    worst = np.zeros(ncell)
    np.maximum.at(worst, cp, f["rel"])
    sig = it._sig
    cells = []
    for k in range(ncell):
        if n_el[k] == 0 and share[k] == 0:
            continue
        idx = np.array(np.unravel_index(k, n))
        centre = lo + (idx + 0.5) * span / n
        cells.append({"centre": [sig(x) for x in centre], "n_elements": int(n_el[k]),
                      "mean_h": sig(h_sum[k] / n_el[k]) if n_el[k] else None,
                      "error_share": float(share[k]), "max_rel_err": sig(worst[k], 2)})
    return {"shape": n.tolist(), "cell_size": [sig(x) for x in span / n], "cells": cells}


# --------------------------------------------------------------------------- pictures


def _group_cells(path: str, dim: int) -> dict:
    """Named lower-dimensional groups -> (kind, corner-node index array): 'seg' (n, 2) line pieces,
    'tri' (n, 3) surface triangles, 'pt' (n,) vertices. The domain group and gmsh's own sets are left out."""
    m = meshio.read(path)
    kind = "triangle" if dim == 2 else "tetra"
    out = {}
    for name, blocks in m.cell_sets_dict.items():
        if name.startswith("gmsh:") or any(t.startswith(kind) for t in blocks):
            continue
        parts = {"seg": [], "tri": [], "pt": []}
        for t, i in blocks.items():
            data = m.cells_dict[t][i]
            if t.startswith("line"):
                parts["seg"].append(data[:, :2])
            elif t.startswith("triangle"):
                parts["tri"].append(data[:, :3])
            elif t == "vertex":
                parts["pt"].append(data[:, 0])
        k = next(k for k in ("tri", "seg", "pt") if parts[k] or k == "pt")
        out[name] = (k, np.concatenate(parts[k]) if parts[k] else np.zeros(0, int))
    return out


def _colours(n: int) -> list:
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i % 10) for i in range(n)]


def _labels(axisym: bool) -> tuple[str, str]:
    return ("r", "z") if axisym else ("x", "y")


def _new_fig(w: float, h: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=DPI, layout="constrained")   # no pyplot: no global state, safe in workers
    FigureCanvasAgg(fig)
    return fig


def _fig_size(ext: np.ndarray, ncols: int, extra_w: float, max_w: float = 10.7, panel_h: float = 6.5) -> tuple[float, float]:
    """Figure size for ncols panels side by side, each with the part's aspect ratio, plus extra_w
    inches for a legend or colour bar (and room for titles)."""
    aspect = ext[1] / max(ext[0], 1e-12)
    pw = min((max_w - extra_w) / ncols, panel_h / max(aspect, 1e-3))
    return pw * ncols + extra_w, max(pw * aspect + 1.0, 3.5)


def _save(fig, path: Path) -> str:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    return str(path)


def _mark_checks(ax, checks: list, three_d: bool = False, label: bool = True, edge: str = "black") -> None:
    """Disclosed check points as filled circles, hidden ones as hollow diamonds, numbered (P<i> / H<i>)."""
    for disclosed, mk, face, name in ((True, "o", "white", "disclosed check point"),
                                      (False, "D", "none", "hidden check point")):
        cs = [c for c in checks if c["disclosed"] == disclosed]
        if not cs:
            continue
        xyz = np.array([c["xyz"] for c in cs], float)
        kw = dict(marker=mk, s=46, facecolors=face, edgecolors=edge, linewidths=1.4, label=name, zorder=6)
        if three_d:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], depthshade=False, **kw)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], **kw)
        if label:
            for c, x in zip(cs, xyz):
                txt = f"{'P' if disclosed else 'H'}{c['i']}"
                if three_d:
                    ax.text(*x, " " + txt, fontsize=7, zorder=7)
                else:
                    ax.annotate(txt, x[:2], xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=7)


def _number_hotspots(ax, hotspots: list, three_d: bool = False) -> None:
    for k, h in enumerate(hotspots, 1):
        x = h["centroid"]
        box = dict(boxstyle="circle,pad=0.25", fc="white", ec="red", lw=1.5, alpha=0.9)
        txt = f"{k}"
        if three_d:
            ax.text(*x, txt, fontsize=9, fontweight="bold", color="red", bbox=box, zorder=8)
        else:
            # The number sits beside the hotspot with an arrow to it, so it does not cover the hot colours.
            ax.annotate(txt, x[:2], xytext=(14, 14), textcoords="offset points", fontsize=9, fontweight="bold",
                        color="red", ha="center", va="center", bbox=box, zorder=8,
                        arrowprops=dict(arrowstyle="-", color="red", lw=1.2))


def _field_tri(pts: np.ndarray):
    """Triangulation of the reference integration points for smooth 2D colour plots. Delaunay fills
    holes and concave notches, so triangles much longer than the typical spacing are masked out."""
    from matplotlib.tri import Triangulation
    tri = Triangulation(pts[:, 0], pts[:, 1])
    P = pts[tri.triangles][:, :, :2]
    edge = np.max([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    tri.set_mask(edge > 3 * np.median(edge))
    return tri


def _log_density(f: dict) -> tuple[np.ndarray, float, float]:
    """log10 of the error energy density relative to the part's mean reference energy density, and
    its colour range: the top is the 99.7th percentile (a singular corner's few points would
    otherwise squash everything else into the darkest colour), LOG_RANGE decades below it."""
    mean_ref = f["ref_energy"] / f["w"].sum()
    d = np.log10(np.maximum(f["e"] / f["w"] / mean_ref, 1e-30))
    hi = float(np.percentile(d, 99.7))
    return np.clip(d, hi - LOG_RANGE, hi), hi - LOG_RANGE, hi


def _images_2d(rid: str, title: str, msh: it._Mesh, groups: dict, f: dict, em: dict, checks: list,
               axisym: bool, out_dir: Path) -> dict:
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    xl, yl = _labels(axisym)
    P = msh.points
    mtri = Triangulation(P[:, 0], P[:, 1], msh.cells)
    ext = np.ptp(P[:, :2], axis=0)
    paths = {}

    def frame(ax, t):
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.tick_params(labelsize=7)

    # mesh: element edges, named groups coloured, check points marked
    fig = _new_fig(*_fig_size(ext, 1, extra_w=2.2))
    ax = fig.add_subplot()
    ax.triplot(mtri, color="0.35", lw=0.4)
    for (name, (kind, idx)), col in zip(groups.items(), _colours(len(groups))):
        if kind == "seg" and len(idx):
            ax.add_collection(LineCollection(P[idx][:, :, :2], colors=[col], lw=3.0, label=name, zorder=4))
        elif len(idx):
            ax.scatter(P[idx, 0], P[idx, 1], s=60, color=col, marker="s", label=name, zorder=5)
    _mark_checks(ax, checks)
    frame(ax, f"{title}\nmesh: {len(msh.cells)} elements, median h {em['median_h']:g}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    paths["mesh"] = _save(fig, out_dir / f"{rid}_mesh.png")

    # error: log error-energy density, mesh overlaid faintly, hotspots numbered
    ftri = _field_tri(f["pts"])
    d, lo, hi = _log_density(f)
    fig = _new_fig(*_fig_size(ext, 1, extra_w=1.6))
    ax = fig.add_subplot()
    pc = ax.tripcolor(ftri, d, shading="gouraud", cmap="inferno", vmin=lo, vmax=hi)
    ax.triplot(mtri, color="white", lw=0.25, alpha=0.35)
    fig.colorbar(pc, ax=ax, shrink=0.8, label="log10(error energy density / mean ref. energy density)")
    _mark_checks(ax, checks, label=False, edge="cyan")
    _number_hotspots(ax, em["hotspots"])
    frame(ax, f"{title}\nwhere the error is: field error {em['field_error']:.3g}; "
              "red numbers = hotspots (largest first)")
    paths["error"] = _save(fig, out_dir / f"{rid}_error.png")

    # stress: run von Mises next to the reference, same colour scale
    vmax = float(np.percentile(f["vm_ref"], 99.5))
    fig = _new_fig(*_fig_size(ext, 2, extra_w=1.2))
    axes = fig.subplots(1, 2)
    for ax, v, t in ((axes[0], f["vm_run"], "THIS RUN"), (axes[1], f["vm_ref"], "REFERENCE")):
        pc = ax.tripcolor(ftri, v, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
        frame(ax, f"{t}: von Mises")
    fig.colorbar(pc, ax=list(axes), shrink=0.8, label="von Mises (same scale)")
    fig.suptitle(title, fontsize=9)
    paths["stress"] = _save(fig, out_dir / f"{rid}_stress.png")
    return paths


def _surface(msh: it._Mesh) -> np.ndarray:
    """Boundary triangles (corner nodes) of a tetrahedral mesh: the faces that belong to one tet only."""
    t = msh.cells
    faces = np.sort(np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]]), axis=1)
    uniq, count = np.unique(faces, axis=0, return_counts=True)
    return uniq[count == 1]


VIEWS = ((25, 45), (25, 225))        # (elevation, azimuth): two opposite corners of the part


def _images_3d(rid: str, title: str, msh: it._Mesh, groups: dict, f: dict, em: dict, checks: list,
               out_dir: Path) -> dict:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib
    P = msh.points
    surf = _surface(msh)
    tris = P[surf]
    # Surface colours come from the nearest reference integration point to each face's centre.
    near = cKDTree(f["pts"]).query(tris.mean(axis=1))[1]
    ext = np.ptp(P, axis=0)
    paths = {}

    def panel(fig, pos, view, facecolors, t, lw=0.15, edge="0.25"):
        ax = fig.add_subplot(*pos, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=facecolors, edgecolors=edge, linewidths=lw))
        lo, hi = P.min(0), P.max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(ext)
        ax.view_init(*view)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(t, fontsize=9)
        return ax

    def views_title(view):
        return f"view elev {view[0]}, azim {view[1]}"

    # mesh: surface triangles coloured by the named surface group they lie in
    fc = np.tile(np.array([0.85, 0.85, 0.85, 1.0]), (len(surf), 1))
    key = {tuple(r): k for k, r in enumerate(surf)}
    handles = []
    from matplotlib.patches import Patch
    for (name, (kind, idx)), col in zip(groups.items(), _colours(len(groups))):
        if kind == "tri" and len(idx):
            rows = [key.get(tuple(r)) for r in np.sort(idx, axis=1)]
            rows = [r for r in rows if r is not None]
            fc[rows] = matplotlib.colors.to_rgba(col)
            handles.append(Patch(color=col, label=name))
    fig = _new_fig(11, 5.5)
    for j, view in enumerate(VIEWS):
        ax = panel(fig, (1, 2, j + 1), view, fc, f"mesh, {views_title(view)}", lw=0.3, edge="0.2")
        _mark_checks(ax, checks, three_d=True, label=j == 0)
    fig.legend(handles=handles + fig.axes[0].get_legend_handles_labels()[0], fontsize=7, loc="outside lower center",
               ncol=min(8, len(handles) + 2))
    fig.suptitle(f"{title}: surface mesh ({len(msh.cells)} tets, median h {em['median_h']:g}); "
                 "groups coloured", fontsize=10)
    paths["mesh"] = _save(fig, out_dir / f"{rid}_mesh.png")

    # error on the surface
    d, lo, hi = _log_density(f)
    norm = matplotlib.colors.Normalize(lo, hi)
    cmap = matplotlib.colormaps["inferno"]
    fig = _new_fig(11, 5.5)
    for j, view in enumerate(VIEWS):
        ax = panel(fig, (1, 2, j + 1), view, cmap(norm(d[near])), f"error, {views_title(view)}", lw=0.1, edge="0.6")
        _number_hotspots(ax, em["hotspots"], three_d=True)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=fig.axes, shrink=0.6,
                 label="log10 error energy density / mean ref. density")
    fig.suptitle(f"{title}: where the error is on the surface (field error {em['field_error']:.3g}; "
                 "numbered = top hotspots, may be inside)", fontsize=10)
    paths["error"] = _save(fig, out_dir / f"{rid}_error.png")

    # stress: run vs reference von Mises, two views each, same scale
    norm = matplotlib.colors.Normalize(0, float(np.percentile(f["vm_ref"], 99.5)))
    cmap = matplotlib.colormaps["viridis"]
    fig = _new_fig(11, 9)
    for j, view in enumerate(VIEWS):
        panel(fig, (2, 2, 1 + j), view, cmap(norm(f["vm_run"][near])), f"this run: von Mises, {views_title(view)}",
              lw=0, edge="none")
        panel(fig, (2, 2, 3 + j), view, cmap(norm(f["vm_ref"][near])), f"reference: von Mises, {views_title(view)}",
              lw=0, edge="none")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=fig.axes, shrink=0.6, label="von Mises (same scale)")
    fig.suptitle(f"{title}: von Mises, this run (top) vs reference (bottom)", fontsize=10)
    paths["stress"] = _save(fig, out_dir / f"{rid}_stress.png")
    return paths


# --------------------------------------------------------------------------- the report


def build_run_report(run: dict, out_dir: Path) -> dict:
    """The full report of one run (see the module docstring); also written to out_dir/<run_id>.json.
    Never raises: on any failure returns {"run_id", "error"}."""
    t0 = time.perf_counter()
    rid = run.get("run_id") or "run"
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        problem_id = run["problem_id"]
        p = fea.get_problem(problem_id)
        workdir = out_dir / f"{rid}_work"          # only used if the run's .msh is gone and must be rebuilt
        cp, mesh, notes = it._scored_mesh(run, workdir)
        res = fe._solve(p, mesh, cp["setup"], config.TASKS[problem_id]["qoi"])
        msh = it._Mesh(mesh["path"], p.dim)
        f = _field(problem_id, p, res)
        em = _error_map(problem_id, run, p, res, msh, f, notes)
        checks = em.pop("_cp_full")
        layout = it.mesh_layout(problem_id, run, workdir)
        groups = _group_cells(mesh["path"], p.dim)
        axisym = reference_setup(p)["formulation"] == "axisymmetric"
        k = agent_of(rid)
        if p.dim == 2:
            images = _images_2d(rid, rid, msh, groups, f, em, checks, axisym, out_dir)
        else:
            images = _images_3d(rid, rid, msh, groups, f, em, checks, out_dir)
        solve = run["solves"][cp["solve_index"]]
        report = {
            "run_id": rid, "agent": None if k is None else f"a{k}", "task": problem_id,
            "dim": p.dim, "axisymmetric": axisym,
            "mesh_request": cp["request"],
            "mesh_stats": {"nodes": mesh.get("nodes"), "dofs": (solve["result"].get("info") or {}).get("n_dofs"),
                           "n_elements": int(len(msh.cells)), "min_quality": mesh.get("min_quality")},
            "setup": cp["setup"], "error_map": em, "mesh_layout": layout,
            "region_grid": _region_grid(p.dim, msh, f), "check_points": checks, "images": images,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001  a report is a bonus: never break the loop over one
        report = {"run_id": rid, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 2)}
    try:
        (out_dir / f"{rid}.json").write_text(json.dumps(report, indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        report.setdefault("error", f"could not write the report file: {e}")
    return report


def add_scores(report: dict, run: dict) -> dict:
    """Add the run's scored numbers (from run["checks"], after score_population) as report["numbers"].
    rank_in_group / group_size are read from checks["rank"] / checks["group_size"] if the caller set them."""
    c = run.get("checks") or {}
    auto_err = c.get("auto_error")
    report["numbers"] = {
        "score": c.get("score"), "base_score": c.get("base_score", c.get("score")), "bonus": c.get("best_bonus"),
        "field_error": c.get("field_error"), "probe_error": c.get("probe_error"), "gain": c.get("gain"),
        "accuracy": c.get("accuracy"), "efficiency": c.get("efficiency"),
        "meets_target": None if c.get("gain") is None else c["gain"] >= config.AUTO_GAIN_TARGET,
        "dofs": c.get("dofs"), "automesher_error": auto_err, "automesher_dofs": c.get("auto_dofs"),
        "target_error": None if auto_err is None else auto_err / config.AUTO_GAIN_TARGET,
        "rank_in_group": c.get("rank"), "group_size": c.get("group_size"),
        "valid": c.get("valid"), "invalid_because": c.get("invalid_because"),
    }
    return report


def _g(x, n: int = 3) -> str:
    return "-" if x is None else (f"{x:.{n}g}" if isinstance(x, (int, float)) else str(x))


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.3g}%"


def _xyz(v) -> str:
    return "(" + ", ".join(_g(x) for x in v) + ")"


def report_text(report: dict, max_chars: int = 6000) -> str:
    """The report as compact text: numbers, setup, mesh request, hotspots, mesh near groups, check
    points, then the region grid (worst cells first, so a cut drops the least useful rows)."""
    rid = report.get("run_id")
    if report.get("error"):
        return f"RUN {rid}: no report ({report['error']})"
    L = [f"RUN {rid} (agent {report.get('agent')}, task {report.get('task')}"
         + (", axisymmetric: x = r, y = z" if report.get("axisymmetric") else "") + ")"]
    n = report.get("numbers")
    if n:
        rank = f", rank {n['rank_in_group']}/{n['group_size']}" if n.get("rank_in_group") is not None else ""
        L.append(f"score {_g(n['score'])} (base {_g(n['base_score'])} + be-the-best bonus {_g(n['bonus'])}){rank}; "
                 f"valid {n.get('valid')}" + (f" ({n['invalid_because']})" if n.get("invalid_because") else ""))
        L.append(f"field error {_pct(n['field_error'])}, check-point error {_pct(n['probe_error'])}; "
                 f"gain {_g(n['gain'])}x the automesher (target {config.AUTO_GAIN_TARGET:g}x: "
                 f"{'met' if n['meets_target'] else 'NOT met'}; target error {_pct(n['target_error'])}); "
                 f"accuracy term {_g(n['accuracy'])}, cost term {_g(n['efficiency'])}; "
                 f"DOFs {n['dofs']} vs automesher {n['automesher_dofs']} (error {_pct(n['automesher_error'])})")
    req = report.get("mesh_request") or {}
    refs = "; ".join(f"{r.get('group') or _xyz(r.get('point') or [])} size {_g(r.get('size'))} radius {_g(r.get('radius'))}"
                     for r in req.get("refinements") or []) or "none"
    L.append(f"mesh request: h {_g(req.get('h'))}, order {req.get('order')}, refinements: {refs}")
    ms, em = report["mesh_stats"], report["error_map"]
    lay = report.get("mesh_layout") or {}
    L.append(f"mesh: {ms['nodes']} nodes, {ms['dofs']} DOFs, {ms['n_elements']} elements, min quality "
             f"{_g(ms['min_quality'])}, median h {_g(em['median_h'])}, h range {lay.get('h_range')}")
    s = report.get("setup") or {}
    L.append("setup: " + json.dumps(s, separators=(",", ":"), default=str))
    L.append(f"error map: field error {_pct(em['field_error'])}, check-point RMS {_pct(em['probe_error'])}; "
             f"{em['n_hotspots']} hotspots hold 80% of the error energy (radius {_g(em['hotspot_radius'])}); "
             "numbered as on the error image:")
    for k, h in enumerate(em["hotspots"], 1):
        L.append(f"  #{k} at {_xyz(h['centroid'])}: {_pct(h['share'])} of error energy, nearest group "
                 f"{h['nearest_group']} ({_g(h['distance'])} away), local h {_g(h['local_h'])}, "
                 f"ref von Mises {_g(h['vm_ref'])}, von Mises error {_pct(h['vm_rel_error'])}")
    if lay.get("groups"):
        L.append(f"element size near each group (within {_g(lay.get('near_radius'))}; median h {_g(lay.get('median_h'))}):")
        for g, v in lay["groups"].items():
            L.append(f"  {g}: {v['elements_touching']} elements touching (h {_g(v['h_touching'])}), "
                     f"{v['elements_near']} near (h {_g(v['h_near'])} = {_g(v['h_near_vs_median'])}x median)")
    L.append("check points (P = disclosed to the agent, H = hidden; error = run vs reference von Mises):")
    for c in report.get("check_points") or []:
        L.append(f"  {'P' if c['disclosed'] else 'H'}{c['i']} {_xyz(c['xyz'])}: ref {_g(c['vm_ref'])}, "
                 f"run {_g(c['vm_run'])}, error {c['rel_err'] * 100:+.2f}%")
    rg = report["region_grid"]
    L.append(f"region grid ({'x'.join(map(str, rg['shape']))} cells of size {_xyz(rg['cell_size'])}, empty "
             "cells dropped; worst first): centre | elements | mean h | share of error energy | worst local error")
    for c in sorted(rg["cells"], key=lambda c: -c["error_share"]):
        L.append(f"  {_xyz(c['centre'])} | {c['n_elements']} | {_g(c['mean_h'])} | {_pct(c['error_share'])} | "
                 f"{_pct(c['max_rel_err'])}")
    text = "\n".join(L)
    if len(text) > max_chars:
        cut = "\n... (cut: lower-error grid cells omitted)"
        text = text[: max_chars - len(cut)].rsplit("\n", 1)[0] + cut
    return text
