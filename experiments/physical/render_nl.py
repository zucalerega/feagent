"""Crush-to-snap video in the original 3-panel look, from a nonlinear increment series (CPU, llvmpipe).

usage: render_nl.py RUN_DIR AXIS(x|y|z) OUT.mp4 TITLE NOTE
RUN_DIR holds result_NNNN.vtu (points in m; point fields: displacement (vector), von Mises; cell fields:
PEEQ, failed) and force_displacement.csv (step, travel_mm, force_N, n_failed).
Panels: outside (von Mises up to the UTS, broken elements red, platens), cut in half (von Mises),
cut in half (displacement magnitude + arrows). True scale, >= 10 s, interpolated, slow orbit."""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

UTS, FPS, MIN_S, HOLD_S, TITLE_S, ORBIT = 50.5, 30, 11.0, 2.0, 1.0, 70.0
AX = {"x": 0, "y": 1, "z": 2}


def series(run):
    files = sorted(run.glob("result_*.vtu"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    rows = list(csv.DictReader(open(run / "force_displacement.csv")))
    return files, rows


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
    failed = field(g, "failed", "FAILED") if any(k in g.cell_data for k in ("failed", "FAILED")) else np.zeros(g.n_cells)
    return g.points.copy() * 1e3, u * 1e3, vm / 1e6 if vm.max() > 1e4 else vm, failed.astype(bool), g


def platens(lo, hi, axis):
    pad, th = 0.15 * (hi - lo), 0.04 * (hi[axis] - lo[axis])
    def slab(at, down):
        b0, b1 = lo - pad, hi + pad
        b0[axis], b1[axis] = (at - th, at) if down else (at, at + th)
        return pv.Box(bounds=(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]))
    return slab(lo[axis], True), slab(hi[axis], False)


def half_ids(centers, axis, mid):
    side = (axis + 1) % 3
    return np.where(centers[:, side] < mid[side])[0]


def frame_plan(n_inc):
    """(increment a, increment b, t) per frame: interpolate so the loading part lasts >= MIN_S seconds."""
    n_load = max(int(MIN_S * FPS), n_inc)
    plan = [(0, 0, 0.0)] * int(TITLE_S * FPS)
    for f in range(n_load):
        x = f / (n_load - 1) * (n_inc - 1)
        a = int(np.floor(x)); b = min(a + 1, n_inc - 1)
        plan.append((a, b, x - a))
    return plan + [(n_inc - 1, n_inc - 1, 0.0)] * int(HOLD_S * FPS)


def scene(pl, g0, pts, u, vm, failed, ids_half, axis, plats, umax, arrows_idx):
    cur = g0.copy(); cur.points = pts + u
    cur.point_data["vm"] = vm; cur.point_data["umag"] = np.linalg.norm(u, axis=1) * 1e3
    ok = np.where(~failed)[0]; bad = np.where(failed)[0]
    sb = dict(color="white", title_font_size=14, label_font_size=12, position_x=0.2, position_y=0.05, width=0.6)
    for c in range(3):
        pl.subplot(0, c); pl.set_background("#1d2330", top="#3a4356")
        part = cur if c == 0 else cur.extract_cells(np.intersect1d(ids_half, np.arange(cur.n_cells)))
        keep = np.where(~failed[part.cell_data["vtkOriginalCellIds"]])[0] if c else ok
        body = part.extract_cells(keep).extract_surface(algorithm="dataset_surface") if keep.size else None
        if body is not None and body.n_points:
            if c < 2:
                pl.add_mesh(body, scalars="vm", clim=[0, UTS], cmap="turbo", show_scalar_bar=(c == 0),
                            scalar_bar_args={**sb, "title": "von Mises (MPa)   red = broken"})
            else:
                pl.add_mesh(body, scalars="umag", clim=[0, umax], cmap="viridis",
                            scalar_bar_args={**sb, "title": "displacement (um)"})
        if bad.size and c == 0:
            pl.add_mesh(cur.extract_cells(bad).extract_surface(algorithm="dataset_surface"), color="#ff2a2a")
        if c == 0:
            for p in plats:
                pl.add_mesh(p, color="#8a8f98", opacity=0.55)
        if c == 2 and arrows_idx.size:
            a = pv.PolyData(pts[arrows_idx] + u[arrows_idx]); a["v"] = u[arrows_idx]
            n = np.linalg.norm(a["v"], axis=1).max() or 1.0
            pl.add_mesh(a.glyph(orient="v", scale="v", factor=0.08 * np.ptp(pts, axis=0).max() / n,
                                geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04)), color="#ffe14d")


def render(run, axis, out, title, note):
    files, rows = series(run)
    pts, u0, _, _, g0 = load(files[0])
    lo, hi, mid = pts.min(0), pts.max(0), pts.mean(0)
    ids_half = half_ids(g0.cell_centers().points * 1e3, axis, mid)
    g0.cell_data["vtkOriginalCellIds"] = np.arange(g0.n_cells)
    side = (axis + 1) % 3
    on_cut = np.where(np.abs(pts[:, side] - mid[side]) < 0.04 * np.ptp(pts[:, side]))[0]
    arrows_idx = np.random.default_rng(0).choice(on_cut, size=min(300, on_cut.size), replace=False)
    last = load(files[-1]); umax = float(np.linalg.norm(last[1], axis=1).max() * 1e3) or 1.0
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    cache, cams = {}, None
    get = lambda i: cache.setdefault(i, load(files[i]))
    for f, (a, b, t) in enumerate(frame_plan(len(files))):
        A, B = get(a), get(b)
        u = (1 - t) * A[1] + t * B[1]; vm = (1 - t) * A[2] + t * B[2]; failed = B[3] if t > 0.5 else A[3]
        travel = (1 - t) * float(rows[a]["travel_mm"]) + t * float(rows[b]["travel_mm"])
        force = (1 - t) * float(rows[a]["force_N"]) + t * float(rows[b]["force_N"])
        shift = np.zeros(3); shift[axis] = -travel
        bot, top = platens(lo, hi, axis); top.points = top.points + shift
        pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1920, 1080), border=False)
        scene(pl, g0, pts, u, vm, failed, ids_half, axis, (bot, top), umax, arrows_idx)
        labels = ("outside", "inside: cut in half, von Mises", "inside: movement (arrows = displacement)")
        for c in range(3):
            pl.subplot(0, c); pl.add_text(labels[c], font_size=12, position="upper_left", color="white")
            if cams is None:
                pl.camera_position = "iso"; pl.camera.Azimuth(-30 if c else 0); pl.reset_camera(); pl.camera.zoom(1.1)
            else:
                pl.camera_position = cams[c]
            pl.camera.Azimuth(ORBIT / len(frame_plan(len(files))))
        cams = [pl.renderers[c].camera_position for c in range(3)]
        pl.subplot(0, 0)
        pl.add_text(f"{title}\nplaten travel {travel:6.2f} mm   force {force:8.0f} N   broken elements {int(failed.sum())}"
                    f"\n{note}", font_size=12, position="lower_left", color="white")
        pl.screenshot(str(frames / f"f{f:04d}.png")); pl.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out}: {len(files)} increments")


if __name__ == "__main__":
    run, ax, out, title, note = sys.argv[1:6]
    render(Path(run), AX[ax], Path(out), title, note)
