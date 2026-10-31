"""Live agent tracker for an ACE run: follow one agent through the generations.

    .venv/bin/streamlit run ace/agent_dashboard.py

Reads a run directory (ace_runs/<name>) as the loop writes it, so it updates while the run goes:
  gen<g>/groups.json        which task each agent is on (split runs)
  gen<g>/runs/*.json        each finished run: checks (scores), trace, meshes, cited bullets
  gen<g>/work/<task>-a<k>/  the meshes an agent is making right now
  gen<g>/diagnosis_*.json   the teacher's lessons and its advice to each agent
  playbook_<task>_gen<g>.json, private/agent<k>_gen<g>.json   the context an agent saw at gen g
  cost.jsonl                every API call, with its run id
Nothing here writes to the run.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import altair as alt
import matplotlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "ace_runs"
SHORT = {"nafems_le11_3d": "LE11 3D", "nafems_le11_axisym": "LE11 axisym", "nafems_le1": "LE1",
         "nafems_le10": "LE10", "cooks_membrane": "Cook's membrane",
         "poiseuille_channel": "Poiseuille", "lid_driven_cavity_re100": "Cavity Re 100",
         "taylor_couette_2d": "Taylor-Couette", "cylinder_channel_dfg_2d1": "DFG cylinder"}
RUN_ID = re.compile(r"^g(\d+)-(.+)-a(\d+)$")

st.set_page_config(page_title="Agent tracker", page_icon=":material/monitoring:", layout="wide")


def short(pid: str) -> str:
    return SHORT.get(pid, pid)


# --------------------------------------------------------------------------- loading (cached, re-read when files change)
# The `version` / `mtime` arguments exist only as cache keys. They must NOT start with "_":
# st.cache_data skips hashing such parameters, so the cache would never invalidate.


def run_dirs() -> list[str]:
    dirs = [d for d in RUNS.iterdir() if (d / "gen0").is_dir()] if RUNS.exists() else []
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def stamp(run: str) -> float:
    """Changes whenever the loop writes something new (used as a cache key)."""
    d = RUNS / run
    files = [d / "cost.jsonl", *d.glob("gen*/groups.json"), *d.glob("gen*/runs"), *d.glob("gen*/runs/*.json")]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


@st.cache_data(max_entries=20, show_spinner=False)
def load_runs(run: str, version: float) -> pd.DataFrame:
    """One row per finished agent run. `version` (= stamp(run)) is only a cache key: no leading
    underscore, so st.cache_data hashes it and re-reads when the loop writes something new."""
    rows = []
    for f in (RUNS / run).glob("gen*/runs/*.json"):
        r = json.loads(f.read_text())
        m = RUN_ID.match(r["run_id"])
        if not m:
            continue
        c = r.get("checks") or {}
        rows.append({
            "gen": int(m.group(1)), "task": r["problem_id"], "agent": f"a{m.group(3)}", "run_id": r["run_id"],
            "score": c.get("score"), "base": c.get("base_score", c.get("score")), "bonus": c.get("best_bonus", 0.0),
            "dofs": c.get("dofs"), "field_error": c.get("field_error"), "probe_error": c.get("probe_error"),
            "error": c.get("rel_error"), "meets_target": c.get("meets_tol"), "valid": c.get("valid"),
            "turns": r.get("turns"), "cited": r.get("cited") or [], "path": str(f),
        })
    df = pd.DataFrame(rows)
    if not df.empty:   # rank within the agent's group (same generation, same task) by total score
        df["rank"] = df.groupby(["gen", "task"])["score"].rank(ascending=False, method="min").astype(int)
        df["group_size"] = df.groupby(["gen", "task"])["score"].transform("size")
    return df


@st.cache_data(max_entries=20, show_spinner=False)
def load_costs(run: str, version: float) -> pd.DataFrame:   # version: cache key, see load_runs
    f = RUNS / run / "cost.jsonl"
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
    return pd.DataFrame(rows)


@st.cache_data(max_entries=200, show_spinner=False)
def read_json(path: str, mtime: float):   # mtime: cache key (hashed), re-reads when the file changes
    return json.loads(Path(path).read_text())


def maybe_json(path: Path):
    return read_json(str(path), path.stat().st_mtime) if path.exists() else None


# --------------------------------------------------------------------------- mesh pictures


def faces_of(m) -> tuple[int, np.ndarray, np.ndarray]:
    """The mesh's dimension, the triangles to shade (quads split in two) and the element edges to draw
    (a quad's four sides, not its diagonal). Flow meshes are quad-dominant; quadratic elements are
    drawn by their corner nodes."""
    dim = 3 if any(c.type.startswith(("tetra", "hexahedron", "wedge", "pyramid")) for c in m.cells) else 2
    tri = [c.data[:, :3] for c in m.cells if c.type.startswith("triangle")]
    quad = [c.data[:, :4] for c in m.cells if c.type.startswith("quad")]
    tris = np.vstack(tri + [q[:, [0, 1, 2]] for q in quad] + [q[:, [0, 2, 3]] for q in quad]) if tri or quad \
        else np.empty((0, 3), dtype=int)
    sides = [t[:, [i, (i + 1) % 3]] for t in tri for i in range(3)] + [q[:, [i, (i + 1) % 4]] for q in quad for i in range(4)]
    edges = np.unique(np.sort(np.vstack(sides), axis=1), axis=0) if sides else np.empty((0, 2), dtype=int)
    return dim, tris, edges


@st.cache_data(max_entries=600, show_spinner=False)
def mesh_png(msh: str, mtime: float, points: tuple = (), size: tuple = (5.0, 3.6)) -> bytes:
    """The mesh: edges in 2D, the surface triangles in 3D. `points` mark the problem's scored point(s)."""
    m = meshio.read(msh)
    dim, tris, e = faces_of(m)
    fig = plt.figure(figsize=size)
    if dim == 2:
        ax = fig.add_subplot()
        ax.add_collection(LineCollection(m.points[e][:, :, :2], colors="#1d2733", linewidths=0.3))
        for pt in points:
            ax.plot(pt[0], pt[1], "o", color="#d6402b", ms=4)
        ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    else:
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(m.points[tris], facecolor="#e9edf1", edgecolor="#2b3640", linewidths=0.15))
        for pt in points:
            ax.scatter(*pt[:3], color="#d6402b", s=25, depthshade=False)
        lo, hi = m.points.min(0), m.points.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.15); ax.view_init(elev=24, azim=-58); ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()


def has_reference(task: str) -> bool:
    """Whether the task's fine stress reference is on disk. The dashboard only reads it, never builds
    it: in 3D that solve can need more memory than a laptop has. `python -m ace.field_error
    --precompute <task>` builds it (field-mode runs build it too)."""
    from ace import field_error as fe
    return (fe.CACHE / f"{task}.npz").exists()


@st.cache_data(max_entries=200, show_spinner="Solving to map the error…")
def error_map_png(run_path: str) -> bytes | None:
    """2D only: where the run's stress error is (energy density of sigma_run - sigma_ref, relative to
    the reference's mean), on a log scale. Re-solves the run's mesh (a second or two)."""
    import matplotlib.tri as mtri
    from ace import checkpoint as ckpt, fea, field_error as fe
    from ace.checks import reference_setup
    from ace.config import TASKS
    r = json.loads(Path(run_path).read_text())
    p = fea.get_problem(r["problem_id"])
    cp = ckpt.checkpoint(r)
    if p.dim != 2 or cp is None or fea.is_flow(p) or not has_reference(r["problem_id"]):
        return None
    mat, ref = reference_setup(p)["material"], fe.reference(r["problem_id"])
    s = fe.stresses(fe._solve(p, cp["mesh"], cp["setup"], TASKS[r["problem_id"]]["qoi"]), ref["pts"], 2)

    def dens(x):
        return ((1 + mat["nu"]) * (sum(x[k] ** 2 for k in fe.DIAG) + 2 * sum(x[k] ** 2 for k in fe.OFFDIAG))
                - mat["nu"] * sum(x[k] for k in fe.DIAG) ** 2) / mat["E"]
    d = dens({k: s[k] - ref["sigma"][k] for k in s}) / np.average(dens(ref["sigma"]), weights=ref["w"])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    tc = ax.tripcolor(mtri.Triangulation(ref["pts"][:, 0], ref["pts"][:, 1]), np.log10(np.maximum(d, 1e-6)),
                      shading="gouraud", cmap="magma_r", vmin=-4, vmax=1)
    fig.colorbar(tc, ax=ax, shrink=0.8, label="log10 error density")
    ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=20, show_spinner=False)
def scored_points(task: str) -> tuple:
    """Where the scored QoI is read: one point, two for a point difference (a pressure drop), none
    for a group quantity (a drag force)."""
    from ace import fea
    from ace.config import TASKS
    q = fea.official_qoi(fea.get_problem(task), TASKS[task]["qoi"])
    return tuple(tuple(x) for x in (q.point, q.point2) if x is not None)


# --------------------------------------------------------------------------- interactive 3D views (plotly)

STRESS_SCALE = "Turbo"


@st.cache_data(max_entries=300, show_spinner=False)
def surface_of(msh: str, mtime: float) -> dict:
    """Vertices, triangles and edges to draw: the boundary faces of a 3D mesh, or all faces of a
    2D mesh (at z = 0). Quadratic elements are drawn by their corner nodes."""
    m = meshio.read(msh)
    dim, tris, edges = faces_of(m)
    if dim == 3:   # keep each boundary triangle once (gmsh writes some in several groups)
        tris = np.unique(np.sort(tris, axis=1), axis=0)
    pts = m.points.copy()
    if dim == 2:
        pts[:, 2] = 0.0
    return {"pts": pts, "tris": tris, "edges": edges, "dim": dim}


def mesh_figure(pts, tris, edges=None, color=None, cmin=None, cmax=None, points=(), flat2d=False,
                height=460, show_scale=True) -> go.Figure:
    """A draggable CAD-style view: shaded surface, optional element edges, optional stress colours."""
    data = [go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        intensity=color, colorscale=STRESS_SCALE if color is not None else None, cmin=cmin, cmax=cmax,
        color=None if color is not None else "#cfd6de", showscale=color is not None and show_scale,
        colorbar=dict(title="von Mises (MPa)", thickness=12, len=0.7) if color is not None else None,
        flatshading=True, lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.8),
        hoverinfo="skip", name="surface")]
    if edges is not None and len(edges):
        seg = np.full((len(edges) * 3, 3), np.nan)
        seg[0::3], seg[1::3] = pts[edges[:, 0]], pts[edges[:, 1]]
        data.append(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                                 line=dict(color="#26313b", width=1.2), hoverinfo="skip", name="edges"))
    if points:
        q = np.array([list(pt) + [0.0] * (3 - len(pt)) for pt in points])
        data.append(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                 marker=dict(size=5, color="#d6402b"), name="scored point"))
    fig = go.Figure(data)
    cam = dict(eye=dict(x=0, y=0, z=2.0), up=dict(x=0, y=1, z=0)) if flat2d else dict(eye=dict(x=1.5, y=-1.6, z=1.1))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      scene=dict(aspectmode="data", camera=cam, xaxis=dict(visible=False),
                                 yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig


@st.cache_data(max_entries=100, show_spinner="Solving the agent's setup on its mesh…")
def run_stress(run_path: str, mtime: float) -> np.ndarray | None:
    """von Mises stress of the run's own solution at its mesh vertices (re-solves its scored setup)."""
    from ace import checkpoint as ckpt, fea, field_error as fe
    from ace.config import TASKS
    r = json.loads(Path(run_path).read_text())
    cp = ckpt.checkpoint(r)
    if cp is None:
        return None
    p = fea.get_problem(r["problem_id"])
    res = fe._solve(p, cp["mesh"], cp["setup"], TASKS[r["problem_id"]]["qoi"])
    pts = meshio.read(cp["mesh"]["path"]).points
    return np.asarray(res.probe("von_mises", pts[:, : p.dim]))


@st.cache_data(max_entries=10, show_spinner="Building the target structure…")
def target_structure(task: str, with_reference: bool) -> dict:
    """The actual part the agents mesh: a fine tessellation of the problem's CAD geometry, coloured by
    the reference von Mises stress (interpolated from the cached fine reference solve). An
    axisymmetric section is revolved into the full solid, with a 90-degree cutaway showing the section.
    Without a reference on disk (`with_reference`, also the cache key) or for a flow problem, the
    geometry is drawn uncoloured (vm None)."""
    import subprocess
    import sys
    from scipy.spatial import cKDTree
    from ace import fea, field_error as fe
    from ace.checks import reference_setup
    p = fea.get_problem(task)

    # gmsh is process-global and not safe to call from Streamlit's script threads more than once, so
    # the tessellation is made once in a separate process and kept on disk next to the references.
    path = fe.CACHE / f"{task}_target.msh"
    if not path.exists():
        fe.CACHE.mkdir(exist_ok=True)
        script = ("import gmsh, sys; from ace import fea; "
                  "from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session, set_uniform_size; "
                  "p = fea.get_problem(sys.argv[1]); "
                  "s = gmsh_session(p.id); s.__enter__(); p.build_geometry(); ensure_domain_group(p.dim); "
                  "set_uniform_size(p.params['h0'] / (1.5 if p.dim == 3 else 3)); "
                  "gmsh.model.mesh.generate(2); gmsh.write(sys.argv[2]); s.__exit__(None, None, None)")
        subprocess.run([sys.executable, "-c", script, task, str(path)], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parent.parent)
    m = meshio.read(path)
    tris = np.unique(np.sort(faces_of(m)[1], axis=1), axis=0)
    pts = m.points.copy()
    if p.dim == 2:
        pts[:, 2] = 0.0
    if fea.is_flow(p) or not with_reference:
        kind = "flow domain" if fea.is_flow(p) else "CAD part" if p.dim == 3 else "CAD section"
        return {"pts": pts, "tris": tris, "vm": None, "kind": kind, "flat": p.dim == 2}
    ref = fe.reference(task)
    vm_ref = fe.von_mises(ref["sigma"])
    tree = cKDTree(ref["pts"][:, : p.dim])

    def vm_at(x):   # inverse-distance average of the 4 nearest reference integration points
        dist, idx = tree.query(x[:, : p.dim], k=4)
        w = 1 / np.maximum(dist, 1e-12)
        return (vm_ref[idx] * w).sum(1) / w.sum(1)

    if p.dim == 2 and reference_setup(p)["formulation"] == "axisymmetric":
        # revolve the section (x = r, y = z) about the axis through 270 degrees; the two cut faces show it
        lines = np.vstack([c.data[:, [0, -1]] for c in m.cells if c.type.startswith("line")])
        th = np.linspace(0, 1.5 * np.pi, 49)
        r_, z_ = pts[:, 0], pts[:, 1]
        vm_nodes = vm_at(pts)
        P, T, C = [], [], []
        n = len(th)
        for a, b in lines:                         # each boundary segment sweeps a band of quads
            base = len(P)
            for t in th:
                P += [(r_[a] * np.cos(t), r_[a] * np.sin(t), z_[a]), (r_[b] * np.cos(t), r_[b] * np.sin(t), z_[b])]
                C += [vm_nodes[a], vm_nodes[b]]
            for j in range(n - 1):
                i0 = base + 2 * j
                T += [(i0, i0 + 1, i0 + 3), (i0, i0 + 3, i0 + 2)]
        for t in (0.0, 1.5 * np.pi):               # the two cut faces: the section itself
            base = len(P)
            P += [(r * np.cos(t), r * np.sin(t), z) for r, z in zip(r_, z_)]
            C += list(vm_nodes)
            T += [tuple(base + q for q in tri) for tri in tris]
        return {"pts": np.array(P), "tris": np.array(T), "vm": np.array(C) / 1e6, "kind": "revolved section",
                "flat": False}
    return {"pts": pts, "tris": tris, "vm": vm_at(pts) / 1e6, "kind": "CAD part" if p.dim == 3 else "CAD section",
            "flat": p.dim == 2}


def scored_mesh_path(run_path: str) -> str | None:
    from ace import checkpoint as ckpt
    cp = ckpt.checkpoint(json.loads(Path(run_path).read_text()))
    return cp["mesh"].get("path") if cp else None


# --------------------------------------------------------------------------- page


run_names = run_dirs()
with st.sidebar:
    st.header("Agent tracker")
    if not run_names:
        st.warning("No runs found in ace_runs/.")
        st.stop()
    run = st.selectbox("Run", run_names, key="run")
    live = st.toggle("Live (refresh every 15 s)", value=True, key="live")
    st.caption("Reads the run folder as the loop writes it. Finished runs appear one generation at a time; "
               "the current generation's meshes show up as agents make them.")


@st.fragment(run_every=15 if live else None)
def page() -> None:
    s = stamp(run)
    df, costs = load_runs(run, s), load_costs(run, s)
    d = RUNS / run
    gens = sorted(int(g.name[3:]) for g in d.glob("gen*") if g.name[3:].isdigit())
    current = max(gens) if gens else 0
    finished = sorted(df["gen"].unique()) if not df.empty else []
    in_progress = current not in finished
    agents = sorted({f"a{k}" for k in range(12) if (d / "private" / f"agent{k}_gen0.json").exists()}
                    | (set(df["agent"]) if not df.empty else set()), key=lambda a: int(a[1:]))
    if not agents:
        st.info("Waiting for the first generation…")
        return

    # ---- run header
    st.title(run)
    with st.container(horizontal=True):
        st.metric("Generation", f"{current}" + (" (running)" if in_progress else ""), border=True)
        st.metric("Spent", f"${costs['usd'].sum():.2f}" if not costs.empty else "$0.00", border=True)
        for task in sorted(df["task"].unique()) if not df.empty else []:
            t = df[df["task"] == task]
            last = t[t["gen"] == t["gen"].max()]["base"].mean()
            st.metric(f"{short(task)} · best base score", f"{t['base'].max():.2f}",
                      f"{last - t[t['gen'] == t['gen'].min()]['base'].mean():+.2f} mean vs gen 0",
                      border=True, chart_data=t.groupby("gen")["base"].mean().tolist(), chart_type="line")

    agent = st.segmented_control("Agent", agents, default=agents[0], key="agent", bind="query-params")
    if agent is None:
        return
    k = int(agent[1:])
    mine = df[df["agent"] == agent].sort_values("gen") if not df.empty else df

    # ---- live: what the agent is doing in the running generation
    if in_progress:
        groups = maybe_json(d / f"gen{current}" / "groups.json") or {}
        task = groups.get(agent)
        with st.container(border=True):
            st.subheader(f"Now: generation {current}" + (f" · {short(task)}" if task else ""))
            work = sorted((d / f"gen{current}" / "work").glob(f"*-{agent}/mesh_*.msh"),
                          key=lambda f: f.stat().st_mtime)
            spent = costs[costs["run_id"].astype(str).str.endswith(f"-{agent}")
                          & costs["run_id"].astype(str).str.startswith(f"g{current}-")] if not costs.empty else costs
            st.caption(f"{len(work)} mesh(es) so far · {len(spent)} model calls · ${spent['usd'].sum() if len(spent) else 0:.3f}")
            if work:
                st.image(mesh_png(str(work[-1]), work[-1].stat().st_mtime, scored_points(task) if task else ()),
                         caption=f"latest mesh: {work[-1].name}", width=420)

    if mine.empty:
        st.info(f"{agent} has no finished runs yet.")
        return

    # ---- this agent's rewards over the generations
    with st.container(horizontal=True):
        last = mine.iloc[-1]
        st.metric("Latest base score", f"{last['base']:.2f}", f"{last['base'] - mine.iloc[0]['base']:+.2f} vs first", border=True)
        st.metric("Best base score", f"{mine['base'].max():.2f}", border=True)
        st.metric("Be-the-best bonus won", f"{mine['bonus'].sum():.2f}", border=True)
        st.metric("Wins in its group", f"{int((mine['rank'] == 1).sum())} / {len(mine)}", border=True)
    with st.container(border=True):
        st.subheader("Rewards")
        pop = df.groupby(["gen", "task"], as_index=False)["base"].mean().rename(columns={"base": "group mean"})
        line = alt.Chart(mine).mark_line(point=alt.OverlayMarkDef(size=60)).encode(
            x=alt.X("gen:Q", title="generation", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("base:Q", title="base score (bonus not included)"),
            color=alt.Color("task:N", title="task", scale=alt.Scale(scheme="tableau10")),
            tooltip=["gen", "task", alt.Tooltip("base:Q", format=".3f"), alt.Tooltip("bonus:Q", format=".3f"),
                     "dofs", alt.Tooltip("error:Q", format=".2%"), "rank"])
        ref = alt.Chart(pop).mark_line(strokeDash=[4, 3], opacity=0.5).encode(
            x="gen:Q", y="group mean:Q", color=alt.Color("task:N", legend=None))
        st.altair_chart(line + ref, alt=f"{agent}'s base score per generation; dashed: mean of all agents on that task")
        st.caption("Solid: this agent. Dashed: mean of every agent on that task that generation.")

    # ---- meshes, generation by generation
    with st.container(border=True):
        st.subheader("Meshes")
        tasks = sorted(mine["task"].unique())
        pick = st.pills("Task", [short(t) for t in tasks], selection_mode="multi", default=[short(t) for t in tasks],
                        key="task_filter")
        shown = mine[mine["task"].map(short).isin(pick or [])]
        cols = st.columns(4)
        for i, (_, r) in enumerate(shown.iterrows()):
            path = scored_mesh_path(r["path"])
            with cols[i % 4]:
                if path and Path(path).exists():
                    st.image(mesh_png(path, Path(path).stat().st_mtime, scored_points(r["task"]), (4.0, 3.0)),
                             caption=f"g{r['gen']} · {short(r['task'])} · base {r['base']:.2f}"
                                     + (f" +{r['bonus']:.2f}" if r["bonus"] else "")
                                     + f" · {int(r['dofs']):,} DOFs · {r['error']:.2%}")

    # ---- one generation in detail: what it saw, what it did, what it was told
    with st.container(border=True):
        st.subheader("One generation in detail")
        g = st.select_slider("Generation", options=list(mine["gen"]), value=int(mine["gen"].max()), key="detail_gen")
        r = mine[mine["gen"] == g].iloc[0]
        run_rec = json.loads(Path(r["path"]).read_text())
        # the agent's mesh next to the structure it is trying to capture, both draggable
        path = scored_mesh_path(r["path"])
        tgt = target_structure(r["task"], has_reference(r["task"]))
        stressed = tgt["vm"] is not None           # a stress reference on disk: colouring and error map
        lo, hi = (float(np.percentile(tgt["vm"], 1)), float(np.percentile(tgt["vm"], 99))) if stressed else (None, None)
        colour = stressed and st.toggle("Colour by stress (agent's solution vs reference)", key="colour_stress")
        mcol, tcol = st.columns(2)
        with mcol:
            with st.container(border=True):
                st.markdown(f"**{agent}'s scored mesh** · drag to rotate, scroll to zoom")
                if path and Path(path).exists():
                    surf = surface_of(path, Path(path).stat().st_mtime)
                    vm = None
                    if colour:
                        full = run_stress(r["path"], Path(r["path"]).stat().st_mtime)
                        vm = None if full is None else full / 1e6
                    st.plotly_chart(mesh_figure(surf["pts"], surf["tris"], surf["edges"], vm, lo, hi,
                                                points=scored_points(r["task"]), flat2d=surf["dim"] == 2),
                                    key=f"mesh_{r['run_id']}")
        with tcol:
            with st.container(border=True):
                st.markdown(f"**Target: the actual structure** ({tgt['kind']}" + (", reference stress" if stressed else "") + ")")
                st.plotly_chart(mesh_figure(tgt["pts"], tgt["tris"], None, tgt["vm"], lo, hi, flat2d=tgt["flat"]),
                                key=f"target_{r['task']}")
                if not stressed and tgt["kind"] != "flow domain":
                    st.caption(f"No stress reference on disk for this task, so no stress colours or error map. "
                               f"The dashboard does not build one (in 3D it can need more memory than a laptop has); "
                               f"`python -m ace.field_error --precompute {r['task']}` does.")
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(f"**{short(r['task'])}** · rank {r['rank']} of {r['group_size']} · `{r['run_id']}`")
            st.table(pd.DataFrame({"": {
                "base score": f"{r['base']:.3f}", "bonus": f"{r['bonus']:.3f}",
                "DOFs": f"{int(r['dofs']):,}" if pd.notna(r["dofs"]) else "-",
                "field error": f"{r['field_error']:.2%}" if pd.notna(r["field_error"]) else "-",
                "check-point error": f"{r['probe_error']:.2%}" if pd.notna(r["probe_error"]) else "-",
                "scored error": f"{r['error']:.2%}" if pd.notna(r["error"]) else "-",
                # field mode's target is 2x the automesher; otherwise it is the task's tolerance
                ("meets 2x target" if pd.notna(r["field_error"]) else "within tolerance"):
                    "yes" if r["meets_target"] else "no"}}))
            if stressed and st.toggle("Show where the error is (2D tasks)", key="errmap"):
                png = error_map_png(r["path"])
                if png:
                    st.image(png, caption="error density; dark = large")
                else:
                    st.caption("Error maps are drawn for 2D tasks only.")
        with right:
            view = st.segmented_control("Show", ["Context it saw", "Teacher's advice", "What it did"],
                                        default="Context it saw", key="detail_view")
            cited = set(r["cited"])
            if view == "Context it saw":
                shared = maybe_json(d / f"playbook_{r['task']}_gen{g}.json") or maybe_json(d / f"playbook_gen{g}.json")
                private = maybe_json(d / "private" / f"agent{k}_gen{g}.json")
                prev = maybe_json(d / "private" / f"agent{k}_gen{g - 1}.json") if g > 0 else None
                prev_ids = {b["id"] for b in (prev or {}).get("bullets", [])}
                st.markdown(f"**Shared playbook for {short(r['task'])}** ({len((shared or {}).get('bullets', []))} bullets)")
                for b in (shared or {}).get("bullets", []):
                    st.markdown(f"{':green-badge[cited]' if b['id'] in cited else ''} `{b['id']}` {b['content']}")
                st.markdown(f"**{agent}'s own notes** ({len((private or {}).get('bullets', []))}; kept across tasks)")
                for b in (private or {}).get("bullets", []):
                    tags = (":blue-badge[new]" if b["id"] not in prev_ids else "") + \
                           (" :green-badge[cited]" if b["id"] in cited else "") + \
                           (" :violet-badge[teacher advice]" if b.get("evidence_type") == "advice" else "")
                    st.markdown(f"{tags} `{b['id']}` {b['content']}")
            elif view == "Teacher's advice":
                diag = maybe_json(d / f"gen{g}" / f"diagnosis_{r['task']}.json")
                dg = (diag or {}).get("diagnosis") or {}
                adv = [a for a in dg.get("agent_advice", []) if a.get("agent") == agent]
                st.markdown(f"**To {agent}** (goes into its own notes for generation {g + 1})")
                for a in adv or [{"content": "(no advice addressed to this agent)"}]:
                    st.markdown(f"- {a['content']}")
                st.markdown("**To everyone on this task** (candidate lessons)")
                for les in dg.get("lessons", []):
                    st.markdown(f"- {les['content']} :gray-badge[{les.get('evidence_type', '')}]")
                mine_diag = [x for x in dg.get("runs", []) if x.get("run_id") == r["run_id"]]
                if mine_diag:
                    st.markdown("**Diagnosis of this run**")
                    st.markdown(f"{mine_diag[0].get('cause', '')}")
                st.caption(f"{(diag or {}).get('tool_calls', 0)} teacher tool calls, {len((diag or {}).get('reruns', []))} reruns")
            else:
                for t in run_rec.get("trace", []):
                    if t["type"] == "tool_call" and t.get("name") in ("mesh", "solve", "submit"):
                        args = {a: v for a, v in (t.get("input") or {}).items() if a not in ("using",)}
                        used = t.get("input", {}).get("using") or []
                        st.markdown(f"**{t['name']}**" + (f" · using {', '.join(used)}" if used else ""))
                        st.code(json.dumps(args, indent=None)[:600], language="json", wrap_lines=True)
                    elif t["type"] in ("text", "thinking") and t.get("text"):
                        st.caption(("thinking: " if t["type"] == "thinking" else "") + t["text"][:500])

    # ---- all of this agent's runs as a table
    with st.expander("All runs of this agent"):
        st.dataframe(mine[["gen", "task", "base", "bonus", "rank", "dofs", "field_error", "probe_error", "turns", "run_id"]],
                     hide_index=True, alt=f"All runs of {agent}",
                     column_config={"field_error": st.column_config.NumberColumn(format="percent"),
                                    "probe_error": st.column_config.NumberColumn(format="percent"),
                                    "base": st.column_config.NumberColumn(format="%.3f"),
                                    "bonus": st.column_config.NumberColumn(format="%.3f")})


page()
