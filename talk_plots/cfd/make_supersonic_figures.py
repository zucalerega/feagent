#!/usr/bin/env python
"""Supersonic figures for the README: the swarm's first generation on a Mach 3 blunted wedge.

Inputs are a run folder of the supersonic suite and the field references it was scored against. Neither is in git:
the run is `ace_runs/supersonic-deep-blunt` (6 agents, compressible flow from PR #24, on the work-in-progress
branch claude/supersonic-blunt-swarm), and the references are that checkout's `ace/field_refs/`. Writes:

  fig_supersonic_mesh_g0       agent a0's generation-0 mesh, with the reference bow shock drawn over it
  fig_supersonic_reference     the target: Mach number of the reference flow, with the sonic line
  supersonic_g0/*.png          a0's run report, as the teacher and a0's self-reflection saw it: mesh, error, speed
  supersonic_g0.md             caption, the run's numbers, and the teacher's first lesson

    python talk_plots/cfd/make_supersonic_figures.py --run ace_runs/supersonic-deep-blunt --refs ace/field_refs
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures as M  # noqa: E402  (style, colours, titles, save; puts the repo root on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import meshio  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

PID, AGENT, GEN = "supersonic_blunt_wedge_2d", "a0", 0
C_SHOCK = "#eb6834"
SHOCK_MACH = 2.95          # the reference flow is Mach 3.00 ahead of the shock: this contour traces the shock


def triangles(m) -> np.ndarray:
    return np.vstack([c.data for c in m.cells if c.type == "triangle"])


def group_lines(m, name: str) -> np.ndarray:
    lines = np.vstack([c.data for c in m.cells if c.type == "line"])
    return lines[m.cell_sets_dict[name]["line"]]


def chain(segments: np.ndarray) -> list[int]:
    """Node indices of an open polyline from its unordered 2-node segments."""
    nbr: dict[int, list[int]] = {}
    for a, b in segments:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    start = next(n for n, v in nbr.items() if len(v) == 1)
    out, prev = [start], None
    while True:
        nxt = [n for n in nbr[out[-1]] if n != prev]
        if not nxt:
            return out
        prev = out[-1]
        out.append(nxt[0])


def body_polygon(m) -> np.ndarray:
    """The body as a closed polygon: the wall from flank to flank, closed by the outflow plane."""
    P = m.points[:, :2]
    return P[chain(group_lines(m, "body"))]


def draw_body(ax, poly: np.ndarray):
    ax.add_patch(PathPatch(MplPath(np.vstack([poly, poly[:1]]), closed=True), fc=M.SOLID, ec=M.INK2, lw=1.0,
                           zorder=5))


def reference_flow(refs: Path, poly: np.ndarray):
    """Triangulation of the reference flow's mesh points (nodal values), the body masked out."""
    z = np.load(refs / f"{PID}_reference_flow.npz")
    m = meshio.read(refs / f"{PID}_reference_flow.msh")
    P = m.points[:, :2]
    if len(P) != len(z["mach"]):
        raise SystemExit(f"{len(P)} mesh points but {len(z['mach'])} values in the reference flow")
    tri = mtri.Triangulation(P[:, 0], P[:, 1])
    cen = P[tri.triangles].mean(axis=1)
    tri.set_mask(MplPath(poly).contains_points(cen))
    info = json.loads(str(z["diag"]))
    return tri, z, info


def panel(fig, box):
    ax = fig.add_axes([0.13, 0.08, 0.6, 0.72])
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="the supersonic run folder (ace_runs/...)")
    ap.add_argument("--refs", type=Path, required=True, help="the field references it was scored against")
    args = ap.parse_args()
    run_id = f"g{GEN}-{PID}-{AGENT}"
    rec = json.loads((args.run / f"gen{GEN}/runs/{run_id}.json").read_text())
    c = rec["checks"]
    sub = rec["meshes"][rec["submitted"]]
    auto = json.loads((args.refs / f"{PID}_auto.json").read_text())
    teacher = json.loads((args.run / f"gen{GEN}/teacher_{PID}.json").read_text())
    lesson = next(s["content"] for s in teacher["shared_lessons"] if s["task"] == PID)
    cfg = json.loads((args.run / "run_config.json").read_text())
    agents = len(list((args.run / f"gen{GEN}/context").glob("a?.md")))

    m = meshio.read(args.run / f"gen{GEN}/work/{PID}-{AGENT}/mesh_{rec['submitted']}.msh")
    T = triangles(m)
    if len(T) != sub["elements"] or len(m.points) != sub["nodes"]:
        raise SystemExit(f"mesh file has {len(m.points)} nodes / {len(T)} triangles, the run record "
                         f"{sub['nodes']} / {sub['elements']}")
    poly = body_polygon(m)
    tri, z, info = reference_flow(args.refs, poly)
    P = m.points[:, :2]
    box = (P[:, 0].min() - 0.01, P[:, 0].max() + 0.01, P[:, 1].min() - 0.01, P[:, 1].max() + 0.01)

    M.style()
    # --- a0's mesh, with the reference bow shock over it
    fig = plt.figure(figsize=(6, 8.5))
    ax = panel(fig, box)
    ax.add_collection(LineCollection(P[T][:, [0, 1, 2, 0]], colors=M.INK2, linewidths=0.3, zorder=2))
    cs = ax.tricontour(tri, z["mach"], levels=[SHOCK_MACH], colors=C_SHOCK, linewidths=2.2, zorder=4)
    draw_body(ax, poly)
    verts = cs.get_paths()[0].vertices
    top = verts[np.argmax(verts[:, 1])]
    ax.annotate("reference\nbow shock", xy=top, xytext=(box[1] + 0.02, top[1] - 0.05), ha="left", va="top",
                fontsize=13, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=M.INK2, lw=1))
    M.titles(fig, f"Agent {AGENT}, generation {GEN}", f"Its mesh: {sub['nodes']:,} nodes, {sub['elements']:,} "
             "triangles.\nFine around the body, coarse along\nmost of the shock.")
    M.save(fig, "fig_supersonic_mesh_g0", f"""
# fig_supersonic_mesh_g0: "Agent {AGENT}, generation {GEN}"

**Caption.** The mesh agent {AGENT} submitted in generation {GEN} on `{PID}` ({sub['nodes']:,} nodes,
{sub['elements']:,} linear triangles, {c['dofs']:,} DOFs), with the reference flow's bow shock drawn over it
(orange: the Mach {SHOCK_MACH} contour of the reference solve; the freestream ahead of the shock is Mach 3.00).
It refined around the body and left most of the shock in coarse cells. See `supersonic_g0.md` for the numbers.
""")

    # --- the target: Mach number of the reference flow
    fig = plt.figure(figsize=(6, 8.5))
    ax = panel(fig, box)
    im = ax.tripcolor(tri, z["mach"], shading="gouraud", cmap=M.SPEED, vmin=0, vmax=3.0, rasterized=True)
    ax.tricontour(tri, z["mach"], levels=[1.0], colors=M.INK, linewidths=1.2, linestyles="--", zorder=4)
    draw_body(ax, poly)
    cax = fig.add_axes([0.80, 0.25, 0.03, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mach number")
    j = int(np.argmin(z["mach"]))                                  # the stagnation point, on the nose
    ax.annotate("subsonic\n(M < 1)", xy=(tri.x[j] - 0.012, tri.y[j]), xytext=(box[0] + 0.005, box[3] - 0.09),
                ha="left", va="top", fontsize=13, arrowprops=dict(arrowstyle="-", color=M.INK2, lw=1), zorder=6)
    M.titles(fig, "The target flow", f"Mach 3 at 10 deg incidence: the reference\nsolve ({info['n_cells']:,} "
             "cells). Dashed: the sonic line.")
    M.save(fig, "fig_supersonic_reference", f"""
# fig_supersonic_reference: "The target flow"

**Caption.** Inviscid (Euler) flow of a perfect gas (gamma = 1.4) at Mach 3 and 10 deg angle of attack over a
blunted wedge (nose radius 0.05 m, half-angle 10 deg): the Mach number of the reference solve the swarm is
scored against, on the problem's own structured mesh ({info['n_cells']:,} cells, h = {float(z['h']):.5f} m). A detached
bow shock stands ahead of the nose. Behind its nearly normal part the flow is subsonic (inside the dashed sonic
line M = 1), then it accelerates back to supersonic speed along both flanks.

**Key numbers**: {info['iterations']:,} iterations, converged = {info['converged']}; mass imbalance
{info['conservation']['mass_imbalance']:.1e}.

**Sources.** `{PID}_reference_flow.npz` / `.msh` from the run's `ace/field_refs/` (nodal values), rendered with
linear (Gouraud) interpolation on a Delaunay triangulation of the mesh points.
""")

    # --- the run report, as the teacher saw it
    out = HERE / "supersonic_g0"
    out.mkdir(exist_ok=True)
    for kind in ("mesh", "error", "speed"):
        shutil.copyfile(args.run / f"gen{GEN}/reports/{run_id}_{kind}.png", out / f"{run_id}_{kind}.png")
    (HERE / "supersonic_g0.md").write_text(f"""# Supersonic flow: the swarm's first generation

**Run.** `ace_runs/supersonic-deep-blunt`: {agents} Claude Sonnet agents, the `--deep` loop (Claude Opus teacher, self-reflection,
organizer), field scoring. Training tasks: {', '.join(f'`{t}`' for t in cfg['train'])}; held out:
{', '.join(f'`{t}`' for t in cfg['holdout'])}. Compressible flow and the blunted wedge come from PR #24; the run itself
is work in progress on the branch `claude/supersonic-blunt-swarm`. Generation 0 only.

**Agent {AGENT}, generation {GEN}, on `{PID}`** (`gen{GEN}/runs/{run_id}.json`, `checks`)
- Mesh: {sub['nodes']:,} nodes, {sub['elements']:,} linear triangles, {c['dofs']:,} DOFs, first-order (upwind)
  convection.
- Field error {c['field_error']:.1%} (velocity {c['velocity_error']:.1%}, pressure {c['pressure_error']:.1%}); RMS
  speed error at the check points {c['probe_error']:.1%}.
- The automesher (uniform triangles at the {c['node_cap']:,}-node cap, {auto['nodes']:,} nodes, {auto['dofs']:,} DOFs,
  second-order convection): field error {auto['error']:.1%}. So {AGENT} is {auto['error'] / c['field_error']:.2f}x as
  accurate with {auto['dofs'] / c['dofs']:.1f}x fewer DOFs: score {c['base_score']:.3f} without bonuses
  ({c['base_score'] / 0.25:.2f}x the automesher's 0.25), {c['score']:.3f} with them.

**What the teacher took from it** (`gen{GEN}/teacher_{PID}.json`, shared lesson): "{lesson.split('. ')[0]}."

**Figures**
- `fig_supersonic_mesh_g0.png`: {AGENT}'s mesh with the reference bow shock drawn over it.
- `fig_supersonic_reference.png`: the target flow (Mach number, sonic line).
- `supersonic_g0/{run_id}_mesh.png`, `_error.png`, `_speed.png`: {AGENT}'s run report, exactly as the teacher and
  {AGENT}'s self-reflection saw it. The error map is log10 of the local squared error, with the hotspots numbered
  largest first; the speed panels share one colour scale.

**Caveats**
- One run, generation 0 only: no learning is shown here. The run's generation 1 crashed in scoring and was
  restarted from its exported state (`ace_runs/supersonic-deep-blunt-2`).
- The reference is a fine solve of the problem's own mesh, not an exact solution: a detached shock has no closed
  form (`feacore/problems/cfd/blunt_wedge.py` on that branch).
""")
    print("wrote supersonic_g0.md and supersonic_g0/")


if __name__ == "__main__":
    main()
