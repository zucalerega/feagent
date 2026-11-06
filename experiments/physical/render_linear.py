"""Load-ramp video of a linear platen compression run (render_comp.py look), rendered on the CPU (llvmpipe).

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \
    python experiments/physical/render_linear.py RUN_DIR --f-full N --mag M --out OUT.mp4 [--label "mesh system"]
        [--ramp-s 10] [--hold-s 3] [--fps 30] [--workers 1]

RUN_DIR is a linear_ramp.py output (result.vtu for the F_REF solve + summary.json). Linear elasticity: every field
scales with F / F_REF. Three panels: outside (von Mises), cut in half (von Mises), cut in half (displacement + yellow
arrows); grey platens; dark gradient background; slow 70 degree orbit; caption strip across the bottom.
Colour limit of von Mises = the dogbone UTS. Frames can be split over processes: WORKER=i NWORK=n (or --workers n,
which starts them); frame k goes to worker k % n, the last one to finish encodes."""
from __future__ import annotations

import argparse
import json
import os
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
ORBIT_DEG = 70.0
AX = {"x": 0, "y": 1, "z": 2}
MATERIAL_CAPTION = ("Formlabs Alpha Fast Resin, 2-min cure, calibrated from dogbone test: E 1.88 GPa, UTS 50.5 MPa")
PART_NAMES = {"abc_00000003": "closed tube with stubs", "abc_00000065": "shaft clamp"}
BG = dict(color="#1d2330", top="#3a4356")


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


def load(run: Path):
    g = pv.read(run / "result.vtu")
    g = g.linear_copy() if hasattr(g, "linear_copy") else g
    g.points = g.points * 1e3                      # m -> mm for display
    g.point_data["u_mm"] = np.asarray(g.point_data["displacement"]) * 1e3
    g.point_data["vm_mpa"] = np.asarray(g.point_data["von_mises"]) / 1e6
    return g


def platens(g, axis: int):
    lo, hi = np.array(g.bounds[0::2]), np.array(g.bounds[1::2])
    pad, th = 0.15 * (hi - lo), 0.04 * (hi[axis] - lo[axis])

    def slab(at, outward):
        b0, b1 = lo - pad, hi + pad
        b0[axis], b1[axis] = (at - th, at) if outward < 0 else (at, at + th)
        return pv.Box(bounds=(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]))
    return slab(lo[axis], -1), slab(hi[axis], +1)


def build(g, axis, mag):
    side = (axis + 1) % 3                          # cut plane contains the load axis
    surf = g.extract_surface(algorithm="dataset_surface")
    keep = np.where(g.cell_centers().points[:, side] < g.center[side])[0]   # cells on one side: a crinkle cut
    half = g.extract_cells(keep).extract_surface(algorithm="dataset_surface")
    on_cut = np.where(np.abs(half.points[:, side] - g.center[side]) < 0.04 * np.ptp(g.points[:, side]))[0]
    idx = np.random.default_rng(0).choice(on_cut, size=min(300, on_cut.size), replace=False)
    return dict(base=g.points.copy(), surf=surf, sl=half, sl2=half.copy(), s_base=surf.points.copy(),
                c_base=half.points.copy(), arrows_src=half.extract_points(idx, adjacent_cells=False), mag=mag)


def update(st, scale):
    m = st["mag"] * scale
    for mesh, b in ((st["surf"], st["s_base"]), (st["sl"], st["c_base"]), (st["sl2"], st["c_base"])):
        mesh.points = b + m * np.asarray(mesh.point_data["u_mm"])
        mesh.point_data["vm"] = np.asarray(mesh.point_data["vm_mpa"]) * scale
        mesh.point_data["umag_um"] = np.linalg.norm(mesh.point_data["u_mm"], axis=1) * 1e3 * scale


def arrows(st, scale):
    src = st["arrows_src"]
    pts = src.points + st["mag"] * scale * np.asarray(src.point_data["u_mm"])
    a = pv.PolyData(pts); a["v"] = np.asarray(src.point_data["u_mm"]) * scale
    length = 0.08 * max(np.ptp(st["base"], axis=0))
    norm = np.linalg.norm(a["v"], axis=1).max() or 1.0
    return a.glyph(orient="v", scale="v", factor=length / norm, geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04))


def plotter(st, g, axis, u_max_um, uts, zoom):
    pl = pv.Plotter(off_screen=True, shape=(2, 3), row_weights=[0.86, 0.14], groups=[(1, np.s_[:])],
                    window_size=(1920, 1080), border=False)
    bot, top = platens(g, axis)
    sb = dict(color="white", title_font_size=14, label_font_size=12, n_labels=5, position_y=0.02, height=0.11,
              width=0.6, position_x=0.2, fmt="%.0f")
    pl.subplot(0, 0)
    pl.add_mesh(st["surf"], scalars="vm", clim=[0, uts], cmap="turbo", smooth_shading=True,
                scalar_bar_args={**sb, "title": f"von Mises (MPa), colour limit = UTS {uts:.1f}"})
    pl.add_mesh(bot, color="#8a8f98", opacity=0.55)
    top_actor = pl.add_mesh(top.copy(), color="#8a8f98", opacity=0.55)
    pl.add_text("outside", font_size=12, position="upper_left", color="white")
    pl.subplot(0, 1)
    pl.add_mesh(st["sl"], scalars="vm", clim=[0, uts], cmap="turbo", show_scalar_bar=False)
    pl.add_text("inside: cut in half, von Mises", font_size=12, position="upper_left", color="white")
    pl.subplot(0, 2)
    pl.add_mesh(st["sl2"], scalars="umag_um", clim=[0, u_max_um], cmap="viridis",
                scalar_bar_args={**sb, "title": "displacement (um, true size)", "fmt": "%.0f"})
    pl.add_text("inside: movement (arrows = displacement)", font_size=12, position="upper_left", color="white")
    for bar in pl.scalar_bars.values():          # labels under the bar, so they do not collide with the title
        bar.SetTextPositionToPrecedeScalarBar()
    cams = []
    for c in range(3):
        pl.subplot(0, c); pl.set_background(BG["color"], top=BG["top"])
        pl.camera_position = "iso"; pl.reset_camera(); pl.camera.zoom(zoom)
        cams.append(pl.camera_position)
    pl.subplot(1, 0); pl.set_background(BG["color"], top=BG["color"])
    return pl, top_actor, cams


def render(run: Path, f_full: float, mag: float, out: Path, label: str, ramp_s: float, hold_s: float, fps: int,
           zoom: float, wid: int = 0, nwork: int = 1):
    s = json.loads((run / "summary.json").read_text())
    axis, f_ref = AX[s["axis"]], float(s["force_N"])
    mat = json.loads((REPO / "experiments" / "material" / "alpha_fast_dogbone1.json").read_text())
    uts = float(mat["uts_eng"])                    # MPa
    g = load(run); st = build(g, axis, mag); update(st, 0.0)
    u_max_um = float(np.linalg.norm(g.point_data["u_mm"], axis=1).max()) * 1e3 * f_full / f_ref
    pl, top_actor, cams = plotter(st, g, axis, u_max_um, uts, zoom)
    _, top0 = platens(g, axis); top_base = top0.points.copy()
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    n = int(round((ramp_s + hold_s) * fps))
    arrow_actor = text_actor = None
    for k in range(n):
        if k % nwork != wid or (frames / f"f{k:04d}.png").exists():
            continue
        lam = min(1.0, k / fps / ramp_s); scale = lam * f_full / f_ref
        update(st, scale)
        shift = np.zeros(3); shift[axis] = mag * s["mean_u_axis_top_m"] * 1e3 * scale
        top_actor.mapper.dataset.points = top_base + shift
        pl.subplot(0, 2)
        if arrow_actor is not None:
            pl.remove_actor(arrow_actor)
        arrow_actor = pl.add_mesh(arrows(st, max(scale, 1e-9)), color="#ffe14d")
        pl.subplot(1, 0)
        for t in text_actor or []:
            pl.remove_actor(t)
        sv = 1e-6 * scale
        text_actor = [pl.add_text(
            f"{s['part']} ({PART_NAMES.get(s['part'], '')})   -   mesh system: {label}   -   {s['n_dofs']:,} DOFs "
            f"({s['n_cells']:,} {s['cell_type']} cells)", font_size=12, position=(8, 100), color="white"),
            pl.add_text(
            f"platen compression along {s['axis']}:  F = {lam * f_full:6.0f} N of {f_full:.0f} N    "
            f"von Mises 99.9 % (volume) {s['robust_von_mises_Pa'] * sv:5.1f} MPa, peak {s['peak_von_mises_Pa'] * sv:5.1f} MPa"
            f"    top platen moves {abs(s['mean_u_axis_top_m']) * 1e6 * scale:6.1f} um\n"
            f"material: {MATERIAL_CAPTION}; nu 0.35 (assumed); linear elastic\n"
            f"shape drawn with deformation magnified x{mag:g} (displacement colours are true size); "
            f"solved by feacore/skfem on CPU", font_size=10, position=(8, 10), color="white")]
        for c in range(3):
            pl.subplot(0, c); pl.camera_position = cams[c]; pl.camera.Azimuth(cut_azimuth(axis, c, k / n, ORBIT_DEG))
        pl.screenshot(str(frames / f"f{k:04d}.png"))
    pl.close()
    if nwork > 1 and wid != 0:
        return None
    while len(list(frames.glob("f*.png"))) < n:
        time.sleep(3)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out} ({n} frames, {n / fps:.1f} s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--f-full", type=float, required=True)
    ap.add_argument("--mag", type=float, required=True, help="deformation magnification of the shape")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="mesh system name for the title (default: summary system)")
    ap.add_argument("--ramp-s", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom", type=float, default=float(os.environ.get("ZOOM", "1.25")))
    ap.add_argument("--workers", type=int, default=1, help="start this many frame workers (WORKER/NWORK)")
    a = ap.parse_args(argv)
    run, out = Path(a.run_dir), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = a.label or json.loads((run / "summary.json").read_text())["system"]
    if a.workers > 1 and "WORKER" not in os.environ:
        procs = [subprocess.Popen([sys.executable, __file__] + (argv if argv is not None else sys.argv[1:]),
                                  env={**os.environ, "WORKER": str(i), "NWORK": str(a.workers)})
                 for i in range(a.workers)]
        rc = [p.wait() for p in procs]
        if any(rc):
            raise SystemExit(f"workers failed: {rc}")
        return
    render(run, a.f_full, a.mag, out, label, a.ramp_s, a.hold_s, a.fps, a.zoom,
           int(os.environ.get("WORKER", "0")), int(os.environ.get("NWORK", "1")))


if __name__ == "__main__":
    main()
