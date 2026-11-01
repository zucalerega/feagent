"""Crush-to-snap video of a nonlinear crush.py run, in the original 3-panel look (CPU, llvmpipe).

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \
    python experiments/physical/render_crush.py RUN_DIR --out OUT.mp4 [--label "mesh system"] [--workers 4]
        [--min-s 11] [--hold-s 4] [--fps 30] [--max-frames N] [--only 0,-1] [--layout panels|single]
        [--size 1920x1080] [--clock CLOCK.json] [--no-lift]

RUN_DIR is a crush.py run dir: result_NNNN.vtu (points in m; point fields displacement [m], von_mises [Pa]; cell
fields PEEQ, failed, state 0 intact / 1 failed / 2 detached fragment), force_displacement.csv, run.json. A run that
continues another one (run.json continued_from) is followed back, so the video starts at zero load. Works on any
linear or quadratic cell type pyvista reads (C3D10 tetra10, C3D4, C3D8 hexahedron).
Panels: outside (von Mises up to the UTS; failed cells red; detached fragments grey; platens), cut in half (von Mises),
cut in half (displacement + yellow arrows). True scale (no magnification), interpolated between increments so the
loading part lasts >= --min-s, slow 70 degree orbit, a held ending on the last (separated) state. Broken cells are
drawn red in all three panels, also as an x-ray overlay (visible through intact cells and platens). In the held ending of a separated run the piece attached to the top platen (face-connected
intact cells) and the platen are lifted by LIFT_FRAC of the height, so the crack faces show (labelled "not simulated";
--no-lift turns it off). Per-part camera (PART_VIEW): abc_00000003 cracks at its bottom ring, so its camera sits low.
--layout single: the outside view only, with large captions (tiles of snap_compare.py); --clock CLOCK.json
({"fps": 30, "travel_mm": [per frame]}) puts the frames on a shared platen-travel time base (snap_compare.py writes it).
--only i,j renders just those frame indices (negative = from the end) as PNGs, for a look before a full render.
Frames split over processes with WORKER=i NWORK=n (or --workers n); frames already on disk are kept (resumable)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k, _v in (("LIBGL_ALWAYS_SOFTWARE", "1"), ("GALLIUM_DRIVER", "llvmpipe"), ("__GLX_VENDOR_LIBRARY_NAME", "mesa")):
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True
REPO = Path(__file__).resolve().parents[2]
TITLE_S, ORBIT, LIFT_S, LIFT_FRAC = 1.0, 70.0, 1.5, 0.06
AX = {"x": 0, "y": 1, "z": 2}
PART_NAMES = {"abc_00000003": "closed tube with stubs", "abc_00000065": "shaft clamp"}
BG = dict(color="#1d2330", top="#3a4356")
# camera elevation change (deg) from pyvista's iso view per part, and the platen opacity
PART_VIEW = {"abc_00000003": {"elev": -24.0}, "abc_00000065": {"elev": 0.0}}
PLATEN_OPACITY = 0.28
FACES = {10: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],                       # VTK_TETRA
         12: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]}  # VTK_HEXAHEDRON
MATERIAL_CAPTION = "Formlabs Alpha Fast Resin, 2-min cure, calibrated from dogbone test: E 1.88 GPa, UTS 50.5 MPa"


def _step(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[-1])


def chain(run: Path) -> list[Path]:
    """Run dirs from the oldest ancestor to RUN_DIR (run.json continued_from)."""
    out = [run]
    while True:
        meta = run / "run.json"
        if not meta.exists():
            break
        parent = json.loads(meta.read_text()).get("continued_from")
        if not parent:
            break
        if not Path(parent).exists():      # the chain was copied to another machine: look next to this run
            parent = run.parent / Path(parent).name
            if not parent.exists():
                break
        run = Path(parent); out.insert(0, run)
    return out


def series(run: Path):
    """[(step, travel_mm, force_N, n_failed, vtu path)] over the whole chain, newest run wins on a step."""
    runs = chain(run)
    rows, vtus = {}, {}
    for i, r in enumerate(runs):
        cut = None
        if i + 1 < len(runs):
            m = json.loads((runs[i + 1] / "run.json").read_text()) if (runs[i + 1] / "run.json").exists() else {}
            cut = m.get("continued_at_step")
        for row in csv.DictReader(open(r / "force_displacement.csv")):
            st = int(float(row["step"]))
            if cut is not None and st > int(cut):
                continue
            rows[st] = (float(row.get("platen_travel_mm") or row.get("travel_mm")), float(row["force_N"]),
                        int(float(row.get("n_failed", 0))))
        for f in r.glob("result_*.vtu"):
            if cut is None or _step(f) <= int(cut):
                vtus[_step(f)] = f
    steps = sorted(s for s in rows if s in vtus)
    return [(s, *rows[s], vtus[s]) for s in steps], runs


def field(g, *names):
    for n in names:
        for d in (g.point_data, g.cell_data):
            if n in d:
                return np.asarray(d[n])
    raise KeyError(names)


def load(path):
    g = pv.read(path)
    g = g.linear_copy() if hasattr(g, "linear_copy") else g
    u = field(g, "displacement", "U", "u")
    vm = field(g, "von_mises", "von Mises", "S_Mises", "mises")
    st = np.asarray(g.cell_data["state"]) if "state" in g.cell_data else \
        (np.asarray(g.cell_data["failed"]).astype(int) if "failed" in g.cell_data else np.zeros(g.n_cells, int))
    return g.points.copy() * 1e3, u * 1e3, vm / 1e6 if vm.max() > 1e4 else vm, st.astype(int), g


def platens(lo, hi, axis):
    pad, th = 0.15 * (hi - lo), 0.04 * (hi[axis] - lo[axis])

    def slab(at, down):
        b0, b1 = lo - pad, hi + pad
        b0[axis], b1[axis] = (at - th, at) if down else (at, at + th)
        return pv.Box(bounds=(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]))
    return slab(lo[axis], True), slab(hi[axis], False)


def upper_piece(g0, pts, state, axis):
    """Cells of the intact pieces (face-connected) that touch the top platen: they rise in the held ending."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cd = g0.cells_dict
    ct = next(iter(cd))
    conn = np.asarray(cd[ct])
    if len(cd) != 1 or ct not in FACES or len(conn) != g0.n_cells:
        return np.zeros(g0.n_cells, bool)
    ids = np.where(state == 0)[0]
    fl = FACES[ct]
    faces = np.sort(np.concatenate([conn[ids][:, list(f)] for f in fl]), axis=1)
    owner = np.tile(ids, len(fl))
    _, inv = np.unique(faces, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    same = inv[order][1:] == inv[order][:-1]
    a, b = owner[order][:-1][same], owner[order][1:][same]
    n = g0.n_cells
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, lab = connected_components(graph, directed=False)
    top = pts[:, axis] >= pts[:, axis].max() - 1e-3 * np.ptp(pts[:, axis])
    touch = top[conn].any(axis=1) & (state == 0)
    return np.isin(lab, np.unique(lab[touch])) & (state == 0)


def cut_azimuth(axis: int, c: int, frac: float, orbit: float) -> float:
    """Camera azimuth (deg, from pyvista's iso view) of panel c at orbit fraction frac. Panel 0 orbits 0 -> orbit.
    The cut panels swing through face-on of the cut plane (normal along (axis + 1) % 3) when it is horizontal,
    so the cut stays visible to the last (held) frame."""
    side = (axis + 1) % 3
    if c == 0:
        return orbit * frac
    if side == 2:                                     # cut face looks up: any azimuth shows it
        return -30 + orbit * frac
    face_on = (0.0 if side == 0 else 90.0) - 45.0     # iso camera sits at 45 deg in the xy plane
    return face_on + orbit / 2 - orbit * frac


def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def frame_plan(n_inc, fps, min_s, hold_s, lift=True):
    """(increment a, increment b, t, lift fraction) per frame: the loading part lasts >= min_s seconds."""
    n_load = max(int(min_s * fps), n_inc)
    plan = [(0, 0, 0.0, 0.0)] * int(TITLE_S * fps)
    for f in range(n_load):
        x = f / (n_load - 1) * (n_inc - 1)
        a = int(np.floor(x)); b = min(a + 1, n_inc - 1)
        plan.append((a, b, x - a, 0.0))
    n_hold = int(hold_s * fps)
    return plan + [(n_inc - 1, n_inc - 1, 0.0, _ease(k / (LIFT_S * fps)) if lift else 0.0) for k in range(n_hold)]


def clock_plan(travel, clock, fps, lift=True):
    """Frames on a shared platen-travel clock: per clock travel, the bracketing increments of this run; past the
    run's last increment its final state is held (and lifted)."""
    tr = np.maximum.accumulate(travel)
    plan, n_inc, done = [], len(tr), None
    for f, T in enumerate(clock):
        a = int(np.searchsorted(tr, T, side="right") - 1)
        a = min(max(a, 0), n_inc - 1)
        if a >= n_inc - 1:
            done = f if done is None else done
            plan.append((n_inc - 1, n_inc - 1, 0.0, _ease((f - done) / (LIFT_S * fps)) if lift else 0.0))
            continue
        b = a + 1
        t = 0.0 if tr[b] <= tr[a] else float(np.clip((T - tr[a]) / (tr[b] - tr[a]), 0, 1))
        plan.append((a, b, t, 0.0))
    return plan


def _surf(cur, mask):
    return cur.extract_cells(np.where(mask)[0]).extract_surface(algorithm="dataset_surface") if mask.any() else None


def xray(pl, mesh, color, opacity=0.9):
    """Draw mesh in a layer-1 renderer over the current subplot (no depth test against the part), so broken cells
    inside the wall or behind the platen stay visible. Returns (overlay, base) to share the camera after framing."""
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
    base = pl.renderer
    ov = vtkRenderer(); ov.SetLayer(1); ov.SetViewport(base.GetViewport()); ov.InteractiveOff()
    mp = vtkPolyDataMapper(); mp.SetInputData(mesh); mp.ScalarVisibilityOff()
    ac = vtkActor(); ac.SetMapper(mp)
    pr = ac.GetProperty(); pr.SetColor(pv.Color(color).float_rgb); pr.SetOpacity(opacity); pr.SetAmbient(0.35)
    ov.AddActor(ac)
    pl.ren_win.SetNumberOfLayers(2); pl.ren_win.AddRenderer(ov)
    return ov, base


def scene(pl, g0, pts, u, vm, state, ids_half, plats, umax, arrows_idx, uts, panels=3, upper=None, lift_vec=None,
          bar_scale=1.0):
    cur = g0.copy(); cur.points = pts + u
    cur.point_data["vm"] = vm; cur.point_data["umag"] = np.linalg.norm(u, axis=1) * 1e3
    lifted = cur.copy(); lifted.points = cur.points + (lift_vec if lift_vec is not None else 0.0)
    sb = dict(color="white", title_font_size=int(14 * bar_scale), label_font_size=int(12 * bar_scale), position_x=0.2,
              position_y=0.02, width=0.6, height=0.11, fmt="%.0f", n_labels=5)
    in_half = np.zeros(cur.n_cells, bool); in_half[ids_half] = True
    up = upper if (upper is not None and lift_vec is not None) else np.zeros(cur.n_cells, bool)
    overlays = []
    for c in range(panels):
        pl.subplot(0, c)
        sel = (state == 0) & (in_half if c else True)
        first = True
        for grid, mask in ((cur, sel & ~up), (lifted, sel & up)):
            body = _surf(grid, mask)
            if body is None or not body.n_points:
                continue
            if c < 2:
                pl.add_mesh(body, scalars="vm", clim=[0, uts], cmap="turbo", show_scalar_bar=(c == 0 and first),
                            scalar_bar_args={**sb, "title": f"von Mises (MPa), colour limit = UTS {uts:.1f}; red = broken"})
            else:
                pl.add_mesh(body, scalars="umag", clim=[0, umax], cmap="viridis", show_scalar_bar=first,
                            scalar_bar_args={**sb, "title": "displacement (um)"})
            first = False
        bad = (state == 1) & (in_half if c else True)
        if bad.any():
            pl.add_mesh(_surf(cur, bad), color="#ff2a2a")
            overlays.append(xray(pl, _surf(cur, bad), "#ff2a2a", 0.85))
        frag = (state == 2) & (in_half if c else True)
        if frag.any():
            pl.add_mesh(_surf(cur, frag), color="#b9bec8", opacity=0.6)
        if c == 0:
            for p in plats:
                pl.add_mesh(p, color="#8a8f98", opacity=PLATEN_OPACITY)
        if c == 2 and arrows_idx.size:
            a = pv.PolyData(pts[arrows_idx] + u[arrows_idx]); a["v"] = u[arrows_idx]
            n = np.linalg.norm(a["v"], axis=1).max() or 1.0
            pl.add_mesh(a.glyph(orient="v", scale="v", factor=0.08 * np.ptp(pts, axis=0).max() / n,
                                geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04)), color="#ffe14d")
    for bar in pl.scalar_bars.values():
        bar.SetTextPositionToPrecedeScalarBar()
    return overlays


def render(run: Path, out: Path, label: str, fps: int, min_s: float, hold_s: float, zoom: float,
           wid: int = 0, nwork: int = 1, max_frames: int | None = None, only: list[int] | None = None,
           layout: str = "panels", size=(1920, 1080), clock: list[float] | None = None, lift: bool = True):
    rows, runs = series(run)
    if not rows:
        raise SystemExit(f"{run}: no increments with a .vtu")
    meta = {}
    for r in runs:
        if (r / "run.json").exists():
            meta = {**meta, **json.loads((r / "run.json").read_text())}
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    part = meta.get("part") or summ.get("part") or next((p for p in PART_NAMES if p in str(run)), "part")
    axis = AX[meta.get("axis") or {"abc_00000003": "z", "abc_00000065": "y"}.get(part, "z")]
    mat = json.loads((REPO / "experiments" / "material" / "alpha_fast_dogbone1.json").read_text())
    uts, peeq = float(mat["uts_eng"]), float(mat["failure_peeq"])
    crit = (meta.get("failure") or {}).get("criterion") or "peeq"
    rule = {"peeq": f"a cell breaks at PEEQ {peeq:g}",
            "peeq_tension": f"a cell breaks at a tensile PEEQ of {peeq:g} (no break in compression)",
            "principal": f"a cell breaks at a max principal stress of {uts:g} MPa"}.get(crit, crit)
    platen = {"bonded": "bonded platens", "sliding": "frictionless platens", "sliding_top": "frictionless top platen",
              "friction": "friction platen (mu {})".format((meta.get("platen_contact") or {}).get("mu", "?"))
              }.get(meta.get("platen") or "bonded", "bonded platens")
    files = [r[4] for r in rows]
    pts, _, _, _, g0 = load(files[0])
    # increment 0 = unloaded state (the CSV starts at the first converged increment)
    zero = (pts, np.zeros_like(pts), np.zeros(len(pts)), np.zeros(g0.n_cells, int), g0)
    travel = np.r_[0.0, [r[1] for r in rows]]; force = np.r_[0.0, [r[2] for r in rows]]
    n_inc = len(files) + 1
    lo, hi, mid = pts.min(0), pts.max(0), pts.mean(0)
    side = (axis + 1) % 3
    ids_half = np.where(g0.cell_centers().points[:, side] * 1e3 < mid[side])[0]
    on_cut = np.where(np.abs(pts[:, side] - mid[side]) < 0.04 * np.ptp(pts[:, side]))[0]
    arrows_idx = np.random.default_rng(0).choice(on_cut, size=min(300, on_cut.size), replace=False)
    last = load(files[-1]); umax = float(np.linalg.norm(last[1], axis=1).max() * 1e3) or 1.0
    separated = bool(summ.get("separated"))
    lift = lift and separated
    upper = upper_piece(g0, pts, last[3], axis) if lift else None
    if upper is not None and not upper.any():
        lift, upper = False, None          # nothing intact is attached to the top
    lift_mm = LIFT_FRAC * float(hi[axis] - lo[axis])
    peak_i = int(np.argmax(force))
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    cache = {}
    get = lambda i: zero if i == 0 else cache.setdefault(i, load(files[i - 1]))  # noqa: E731
    plan = clock_plan(travel, clock, fps, lift) if clock is not None else frame_plan(n_inc, fps, min_s, hold_s, lift)
    if max_frames:
        plan = plan[:max_frames]
    todo = range(len(plan)) if only is None else sorted({i % len(plan) for i in only})
    nodes = meta.get("n_nodes") or summ.get("n_nodes") or g0.n_points
    ctype = {12: "hexahedron", 10: "tetra", 24: "tetra10", 25: "hexahedron20"}.get(int(pv.read(files[0]).celltypes[0]), "")
    el = PART_VIEW.get(part, {}).get("elev", 0.0)
    span = hi - lo; span[axis] += lift_mm if lift else 0.0
    bounds = list(np.ravel(np.c_[lo - 0.1 * span, lo + 1.1 * span]))
    single = layout == "single"
    W, H = size
    k = H / 1080                      # caption scale
    for f in todo:
        if f % nwork != wid or (frames / f"f{f:04d}.png").exists():
            continue
        a, b, t, lf = plan[f]
        if len(cache) > 6:
            cache.clear()
        A, B = get(a), get(b)
        u = (1 - t) * A[1] + t * B[1]; vm = (1 - t) * A[2] + t * B[2]; state = B[3] if t > 0.5 else A[3]
        tr = (1 - t) * travel[a] + t * travel[b]; fo = (1 - t) * force[a] + t * force[b]
        fo = fo if abs(fo) >= 0.5 else 0.0          # no "-0 N" after separation
        shift = np.zeros(3); shift[axis] = -tr
        lv = None
        if lf > 0:
            lv = np.zeros(3); lv[axis] = lf * lift_mm
        bot, top = platens(lo, hi, axis); top.points = top.points + shift + (lv if lv is not None else 0.0)
        panels = 1 if single else 3
        if single:
            pl = pv.Plotter(off_screen=True, shape=(2, 1), row_weights=[0.82, 0.18], window_size=(W, H), border=False)
        else:
            pl = pv.Plotter(off_screen=True, shape=(2, 3), row_weights=[0.86, 0.14], groups=[(1, np.s_[:])],
                            window_size=(W, H), border=False)
        overlays = scene(pl, g0, pts, u, vm, state, ids_half, (bot, top), umax, arrows_idx, uts, panels,
              upper if lv is not None else None, lv, bar_scale=max(k, 0.7))
        labels = ("outside", "inside: cut in half, von Mises", "inside: movement (arrows = displacement)")
        orbit_f = f / len(plan)
        for c in range(panels):   # deterministic camera: same framing every frame, orbit by frame index
            pl.subplot(0, c); pl.set_background(BG["color"], top=BG["top"], all_renderers=False)
            if not single:
                pl.add_text(labels[c], font_size=12, position="upper_left", color="white")
            pl.camera_position = "iso"; pl.reset_camera(bounds=bounds)
            pl.camera.Azimuth(cut_azimuth(axis, c, orbit_f, ORBIT)); pl.camera.Elevation(el)
            pl.camera.OrthogonalizeViewUp(); pl.camera.zoom(zoom)
        for ov, base in overlays:   # x-ray layers share their panel's (final) camera
            ov.SetActiveCamera(base.GetActiveCamera())
        pl.subplot(1, 0); pl.set_background(BG["color"], all_renderers=False)
        n_broken = int((state == 1).sum()); n_frag = int((state == 2).sum())
        phase = "separated" if (a == n_inc - 1 and separated) else \
            ("after peak" if a >= peak_i and force[a] < force[peak_i] else "loading")
        if single:
            pl.add_text(label, font_size=int(15 * k), position=(int(10 * k), int(H * 0.18 * 0.55)), color="white")
            pl.add_text(f"travel {tr:6.3f} mm   force {fo:6.0f} N   broken {n_broken}"
                        + (f" (+{n_frag} detached)" if n_frag else "") + f"   [{phase}]"
                        + (f"\npeak {force[peak_i]:.0f} N at {travel[peak_i]:.3f} mm; {nodes:,} nodes, {ctype}"
                           + (f"; upper piece lifted {lf * lift_mm:.1f} mm (not simulated)" if lf > 0 else "")),
                        font_size=int(10 * k), position=(int(10 * k), int(6 * k)), color="white")
        else:
            pl.add_text(f"{part} ({PART_NAMES.get(part, '')})   -   mesh system: {label}   -   {nodes:,} nodes, "
                        f"{g0.n_cells:,} {ctype} cells   -   nonlinear crush (CalculiX)", font_size=12, position=(8, 110),
                        color="white")
            pl.add_text(f"platen travel {tr:6.3f} mm    force {fo:8.0f} N (peak {force[peak_i]:.0f} N at "
                        f"{travel[peak_i]:.3f} mm)    broken elements {n_broken}" + (f", detached {n_frag}" if n_frag else "")
                        + f"    [{phase}]\nmaterial: {MATERIAL_CAPTION}\nJ2 plasticity from the test curve, {rule}; "
                        f"NLGEOM; {platen}, displacement control; true scale; "
                        f"red = broken (drawn through the part), grey = detached" + (f"\nheld ending: the upper piece and its platen are lifted "
                        f"{lf * lift_mm:.1f} mm to show the crack (not simulated)" if lf > 0 else ""),
                        font_size=10, position=(8, 4), color="white")
        pl.screenshot(str(frames / f"f{f:04d}.png")); pl.close()
    if only is not None or (nwork > 1 and wid != 0):
        return None
    while len(list(frames.glob("f*.png"))) < len(plan):
        time.sleep(3)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%04d.png"),
                    "-frames:v", str(len(plan)), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
                   check=True)
    print(f"wrote {out}: {n_inc} increments ({len(runs)} run dirs), {len(plan)} frames, {len(plan) / fps:.1f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-s", type=float, default=11.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--zoom", type=float, default=1.1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="render only the first N frames (preview)")
    ap.add_argument("--only", default=None, help="comma-separated frame indices to render as PNGs (no video)")
    ap.add_argument("--layout", default="panels", choices=["panels", "single"])
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--clock", default=None, help="JSON {fps, travel_mm: [...]}: shared platen-travel time base")
    ap.add_argument("--no-lift", action="store_true", help="hold the separated state without lifting the upper piece")
    a = ap.parse_args(argv)
    run, out = Path(a.run_dir), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = a.label
    if label is None:
        rj = run / "run.json"
        label = json.loads(rj.read_text()).get("system", "?") if rj.exists() else "?"
    if a.workers > 1 and "WORKER" not in os.environ:
        procs = [subprocess.Popen([sys.executable, __file__] + (argv if argv is not None else sys.argv[1:]),
                                  env={**os.environ, "WORKER": str(i), "NWORK": str(a.workers)})
                 for i in range(a.workers)]
        rc = [p.wait() for p in procs]
        if any(rc):
            raise SystemExit(f"workers failed: {rc}")
        return
    clock, fps = None, a.fps
    if a.clock:
        cj = json.loads(Path(a.clock).read_text())
        clock, fps = [float(v) for v in cj["travel_mm"]], int(cj.get("fps", a.fps))
    w, h = (int(v) for v in a.size.lower().split("x"))
    render(run, out, label, fps, a.min_s, a.hold_s, a.zoom, int(os.environ.get("WORKER", "0")),
           int(os.environ.get("NWORK", "1")), a.max_frames,
           [int(v) for v in a.only.split(",")] if a.only else None, a.layout, (w, h), clock, not a.no_lift)


if __name__ == "__main__":
    main()
