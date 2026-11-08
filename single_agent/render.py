"""A picture of the CAD part before any analysis: faces highlighted and labelled by tag.

The analyst agent reads design data that names features ("the shoulder of the -x boss"), not face
tags; this picture is how it checks that the faces it picked are the right ones. The part is
tessellated with gmsh (surface mesh only), each face's triangles are oriented outward, and the visible
surface is drawn orthographically with headlight shading, like ace.solve_feedback.render.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PALETTE = [(0.85, 0.33, 0.10), (0.00, 0.45, 0.74), (0.47, 0.67, 0.19), (0.49, 0.18, 0.56),
           (0.93, 0.69, 0.13), (0.30, 0.75, 0.93), (0.64, 0.08, 0.18), (0.20, 0.20, 0.20)]


def _tessellate(part) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    """(points (n, 3), {face tag: triangles (m, 3) oriented outward}, diagonal)."""
    import gmsh

    from feacore.geometry.gmsh_tools import gmsh_session
    with gmsh_session(part.id + "-render"):
        part.build_geometry()
        vols = [t for _, t in gmsh.model.getEntities(3)]
        lo = np.min([gmsh.model.getBoundingBox(3, v)[:3] for v in vols], axis=0)
        hi = np.max([gmsh.model.getBoundingBox(3, v)[3:] for v in vols], axis=0)
        diag = float(np.linalg.norm(hi - lo))
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 40)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
        gmsh.model.mesh.generate(2)
        tags, X, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        pos = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        pos[tags] = np.arange(len(tags))
        P = np.asarray(X).reshape(-1, 3)
        faces: dict[int, np.ndarray] = {}
        eps = 1e-3 * diag

        def inside(x) -> bool:
            return any(gmsh.model.isInside(3, v, list(map(float, x))) for v in vols)

        for _, f in gmsh.model.getEntities(2):
            _, conn = gmsh.model.mesh.getElementsByType(2, f)
            if not len(conn):
                continue
            T = pos[np.asarray(conn, dtype=np.int64).reshape(-1, 3)]
            # The triangles of one face share one orientation. Decide it on the CAD surface itself (a
            # triangle's centroid sits inside the material on a tightly curved convex face): project a few
            # large triangles' centroids onto the face, step along the CAD normal, and vote.
            n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
            votes = 0
            for i in np.argsort(-np.linalg.norm(n, axis=1))[:3]:
                uv = gmsh.model.getParametrization(2, f, list(P[T[i]].mean(axis=0)))
                x = np.asarray(gmsh.model.getValue(2, f, list(uv)))
                cad_n = np.asarray(gmsh.model.getNormal(f, list(uv)))
                out_plus, out_minus = not inside(x + eps * cad_n), not inside(x - eps * cad_n)
                if out_plus == out_minus:
                    continue                      # undecided (an edge or a thin spot): no vote
                outward = cad_n if out_plus else -cad_n
                votes += 1 if float(n[i] @ outward) > 0 else -1
            if votes < 0:
                T = T[:, [0, 2, 1]]
            faces[int(f)] = T
    return P, faces, diag


def render_geometry(part, highlight: list[int], view: str, label_all: bool, out_png) -> list[str]:
    """Write the picture; returns the highlighted faces that are visible from `view`."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ace.run_report import _new_fig
    from ace.solve_feedback import VIEWS, _unoccluded

    P, faces, diag = _tessellate(part)
    el, az = np.radians(VIEWS[view])
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    colour = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(dict.fromkeys(highlight))}
    tris, owner = [], []
    for f, T in faces.items():
        tris.append(T)
        owner.append(np.full(len(T), f))
    T = np.vstack(tris)
    owner = np.concatenate(owner)
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area2 = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area2, 1e-300)[:, None]
    front = n @ cam > 1e-3
    T, owner, n, area2 = T[front], owner[front], n[front], area2[front]
    shade = 0.45 + 0.55 * np.abs(n @ cam)
    fc = np.tile([0.80, 0.80, 0.80, 1.0], (len(T), 1))
    for f, c in colour.items():
        fc[owner == f, :3] = c
    fc[:, :3] *= shade[:, None]

    fig = _new_fig(8.8, 7.0)
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.add_collection3d(Poly3DCollection(P[T], facecolors=fc, edgecolors=(0.2, 0.2, 0.2, 0.15), linewidths=0.2))
    lo, hi = P.min(0), P.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3 * diag), zoom=1.2)
    ax.view_init(*VIEWS[view])
    for a in "xyz":
        if a == view[-1]:
            getattr(ax, f"set_{a}ticks")([])
        else:
            getattr(ax, f"set_{a}label")(a + " (m)", fontsize=9)
    ax.tick_params(labelsize=7)

    # Visibility only where a label may go: up to 40 of each candidate face's largest front triangles
    # (testing every triangle against every other is the slow part of a picture).
    cent = P[T].mean(axis=1)
    total = area2.sum()
    visible = []
    for f in sorted(set(owner.tolist())):
        hl = f in colour
        sel = np.nonzero(owner == f)[0]
        if not hl and (not label_all or area2[sel].sum() < 0.003 * total):
            continue
        sel = sel[np.argsort(-area2[sel])[:40]]
        sel = sel[_unoccluded(cent[sel], P[T], cam, 1e-3 * diag)]
        if not len(sel):
            continue
        if hl:
            visible.append(f"f{f}")
        cs = cent[sel]
        c = cs[np.argmin(np.linalg.norm(cs - np.average(cs, axis=0, weights=area2[sel]), axis=1))]
        ax.text(*c, f"f{f}", fontsize=9 if hl else 7, fontweight="bold" if hl else "normal",
                color=colour[f] if hl else "black", ha="center", va="center", zorder=15,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    hidden = [f"f{f}" for f in colour if f"f{f}" not in visible]
    title = f"{part.id}: view {view}"
    if colour:
        title += "\nhighlighted: " + ", ".join(f"f{f}" for f in colour)
        if hidden:
            title += f" (not visible from here: {', '.join(hidden)})"
    fig.suptitle(title, fontsize=10)
    fig.savefig(Path(out_png), dpi=100)
    return visible
