"""Crush-to-snap video of a nonlinear crush.py run, in the original 3-panel look (CPU, llvmpipe).

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \
    python experiments/physical/render_crush.py RUN_DIR --out OUT.mp4 [--label "mesh system"] [--workers 4]
        [--min-s 11] [--hold-s 3] [--fps 30] [--max-frames N]

RUN_DIR is a crush.py run dir: result_NNNN.vtu (points in m; point fields displacement [m], von_mises [Pa]; cell
fields PEEQ, failed, state 0 intact / 1 failed / 2 detached fragment), force_displacement.csv, run.json. A run that
continues another one (run.json continued_from) is followed back, so the video starts at zero load. Works on any
linear or quadratic cell type pyvista reads (C3D10 tetra10, C3D4, C3D8 hexahedron).
Panels: outside (von Mises up to the UTS; failed cells red; detached fragments grey; platens), cut in half (von Mises),
cut in half (displacement + yellow arrows). True scale (no magnification), interpolated between increments so the
loading part lasts >= --min-s, slow 70 degree orbit, a held ending on the last (separated) state.
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
TITLE_S, ORBIT = 1.0, 70.0
AX = {"x": 0, "y": 1, "z": 2}
PART_NAMES = {"abc_00000003": "closed tube with stubs", "abc_00000065": "shaft clamp"}
BG = dict(color="#1d2330", top="#3a4356")
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
        if not parent or not Path(parent).exists():
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


def frame_plan(n_inc, fps, min_s, hold_s):
    """(increment a, increment b, t) per frame: the loading part lasts >= min_s seconds."""
    n_load = max(int(min_s * fps), n_inc)
    plan = [(0, 0, 0.0)] * int(TITLE_S * fps)
    for f in range(n_load):
        x = f / (n_load - 1) * (n_inc - 1)
        a = int(np.floor(x)); b = min(a + 1, n_inc - 1)
        plan.append((a, b, x - a))
    return plan + [(n_inc - 1, n_inc - 1, 0.0)] * int(hold_s * fps)


def scene(pl, g0, pts, u, vm, state, ids_half, plats, umax, arrows_idx, uts):
    cur = g0.copy(); cur.points = pts + u
    cur.point_data["vm"] = vm; cur.point_data["umag"] = np.linalg.norm(u, axis=1) * 1e3
    sb = dict(color="white", title_font_size=14, label_font_size=12, position_x=0.2, position_y=0.02, width=0.6, height=0.11,
              fmt="%.0f", n_labels=5)
    in_half = np.zeros(cur.n_cells, bool); in_half[ids_half] = True
    for c in range(3):
        pl.subplot(0, c)
        sel = (state == 0) & (in_half if c else True)
        body = cur.extract_cells(np.where(sel)[0]).extract_surface(algorithm="dataset_surface") if sel.any() else None
        if body is not None and body.n_points:
            if c < 2:
                pl.add_mesh(body, scalars="vm", clim=[0, uts], cmap="turbo", show_scalar_bar=(c == 0),
                            scalar_bar_args={**sb, "title": f"von Mises (MPa), colour limit = UTS {uts:.1f}; red = broken"})
            else:
                pl.add_mesh(body, scalars="umag", clim=[0, umax], cmap="viridis",
                            scalar_bar_args={**sb, "title": "displacement (um)"})
        bad = (state == 1) & (in_half if c else True)
        if bad.any() and c < 2:
            pl.add_mesh(cur.extract_cells(np.where(bad)[0]).extract_surface(algorithm="dataset_surface"), color="#ff2a2a")
        frag = (state == 2) & (in_half if c else True)
        if frag.any() and c < 2:
            pl.add_mesh(cur.extract_cells(np.where(frag)[0]).extract_surface(algorithm="dataset_surface"),
                        color="#b9bec8", opacity=0.6)
        if c == 0:
            for p in plats:
                pl.add_mesh(p, color="#8a8f98", opacity=0.55)
        if c == 2 and arrows_idx.size:
            a = pv.PolyData(pts[arrows_idx] + u[arrows_idx]); a["v"] = u[arrows_idx]
            n = np.linalg.norm(a["v"], axis=1).max() or 1.0
            pl.add_mesh(a.glyph(orient="v", scale="v", factor=0.08 * np.ptp(pts, axis=0).max() / n,
                                geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04)), color="#ffe14d")
    for bar in pl.scalar_bars.values():
        bar.SetTextPositionToPrecedeScalarBar()


def render(run: Path, out: Path, label: str, fps: int, min_s: float, hold_s: float, zoom: float,
           wid: int = 0, nwork: int = 1, max_frames: int | None = None):
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
    peak_i = int(np.argmax(force))
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    cache = {}
    get = lambda i: zero if i == 0 else cache.setdefault(i, load(files[i - 1]))  # noqa: E731
    plan = frame_plan(n_inc, fps, min_s, hold_s)
    if max_frames:
        plan = plan[:max_frames]
    nodes = meta.get("n_nodes") or summ.get("n_nodes") or g0.n_points
    ctype = {12: "hexahedron", 10: "tetra", 24: "tetra10", 25: "hexahedron20"}.get(int(pv.read(files[0]).celltypes[0]), "")
    bounds = list(np.ravel(np.c_[lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)]))
    for f, (a, b, t) in enumerate(plan):
        if f % nwork != wid or (frames / f"f{f:04d}.png").exists():
            continue
        if len(cache) > 6:
            cache.clear()
        A, B = get(a), get(b)
        u = (1 - t) * A[1] + t * B[1]; vm = (1 - t) * A[2] + t * B[2]; state = B[3] if t > 0.5 else A[3]
        tr = (1 - t) * travel[a] + t * travel[b]; fo = (1 - t) * force[a] + t * force[b]
        shift = np.zeros(3); shift[axis] = -tr
        bot, top = platens(lo, hi, axis); top.points = top.points + shift
        pl = pv.Plotter(off_screen=True, shape=(2, 3), row_weights=[0.86, 0.14], groups=[(1, np.s_[:])],
                        window_size=(1920, 1080), border=False)
        scene(pl, g0, pts, u, vm, state, ids_half, (bot, top), umax, arrows_idx, uts)
        labels = ("outside", "inside: cut in half, von Mises", "inside: movement (arrows = displacement)")
        for c in range(3):   # deterministic camera: same framing every frame, orbit by frame index
            pl.subplot(0, c); pl.set_background(BG["color"], top=BG["top"])
            pl.add_text(labels[c], font_size=12, position="upper_left", color="white")
            pl.camera_position = "iso"; pl.reset_camera(bounds=bounds)
            pl.camera.Azimuth(cut_azimuth(axis, c, f / len(plan), ORBIT)); pl.camera.zoom(zoom)
        pl.subplot(1, 0); pl.set_background(BG["color"], top=BG["color"])
        n_broken = int((state == 1).sum()); n_frag = int((state == 2).sum())
        phase = "separated" if (a == n_inc - 1 and summ.get("separated")) else \
            ("after peak" if a >= peak_i and force[a] < force[peak_i] else "loading")
        pl.add_text(f"{part} ({PART_NAMES.get(part, '')})   -   mesh system: {label}   -   {nodes:,} nodes, "
                    f"{g0.n_cells:,} {ctype} cells   -   nonlinear crush (CalculiX)", font_size=12, position=(8, 110),
                    color="white")
        pl.add_text(f"platen travel {tr:6.3f} mm    force {fo:8.0f} N (peak {force[peak_i]:.0f} N at "
                    f"{travel[peak_i]:.3f} mm)    broken elements {n_broken}" + (f", detached {n_frag}" if n_frag else "")
                    + f"    [{phase}]\nmaterial: {MATERIAL_CAPTION}\nJ2 plasticity from the test curve, a cell breaks at "
                    f"PEEQ {peeq:g}; NLGEOM; bonded platens, displacement control; true scale (no magnification); "
                    f"red = broken, grey = detached", font_size=10, position=(8, 8), color="white")
        pl.screenshot(str(frames / f"f{f:04d}.png")); pl.close()
    if nwork > 1 and wid != 0:
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
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=1.1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="render only the first N frames (preview)")
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
    render(run, out, label, a.fps, a.min_s, a.hold_s, a.zoom, int(os.environ.get("WORKER", "0")),
           int(os.environ.get("NWORK", "1")), a.max_frames)


if __name__ == "__main__":
    main()
