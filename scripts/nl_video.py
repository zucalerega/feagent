#!/usr/bin/env python3
"""Videos of the real-data fine-tuning (problem abc_00000065_compression, scored by ace.nonlinear).

    python scripts/nl_video.py run <run_dir> <out.mp4>          # one 2 s block per generation: the best run's
                                                                # mesh, its PEEQ at the peak, every agent's curve
    python scripts/nl_video.py compare <out.mp4> [--run <run_dir>]
                                    # automesher | swarm bc0 mesh (before fine-tuning) | best fine-tuned mesh
                                    # (uniform 4 mm as a placeholder without --run), animated through the load
    python scripts/nl_video.py ramp <case_dir> <out.mp4>        # one run_curve directory through its increments

Every 3D view uses the same fixed camera (ace.nonlinear.NL_CAMERA) and PEEQ colour range (PEEQ_CLIM), so
frames line up. Light background. Encoded with the ffmpeg bundled in imageio-ffmpeg (no system
ffmpeg needed): 30 fps, h264, yuv420p. The baselines come from scripts/nl_baselines.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PID = "abc_00000065_compression"
BASE = ROOT / "results" / "real_data_finetune" / "baselines"
FPS = 30
W, H = 1920, 1080           # frame size in pixels (a multiple of 8 both ways: no rescaling by ffmpeg)
DPI = 100
CMAP = "YlOrRd"             # PEEQ colour map; elastic material (PEEQ ~ 0) is drawn grey
ELASTIC = "#d5d8dd"
MESH_FACE, MESH_EDGE = "#e9edf2", "#3a4250"
MEAS_COLOR = "black"
SIM_COLOR = "#2f6fb0"
AGENT_COLORS = ["#2f6fb0", "#d9822b", "#3a9a5b", "#b23a48", "#7b5ea7", "#8c6d31"]


# --------------------------------------------------------------------------- encoding

class Video:
    """Frames (H, W, 3 uint8) -> mp4 through imageio-ffmpeg's bundled ffmpeg."""

    def __init__(self, path: Path):
        import imageio_ffmpeg
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.gen = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
                                               pix_fmt_out="yuv420p", macro_block_size=8, quality=None,
                                               output_params=["-crf", "20", "-movflags", "+faststart"])
        self.gen.send(None)
        self.n = 0

    def add(self, frame: np.ndarray, repeat: int = 1):
        b = np.ascontiguousarray(frame[:, :, :3]).tobytes()
        for _ in range(repeat):
            self.gen.send(b)
        self.n += repeat

    def close(self):
        self.gen.close()
        print(f"wrote {self.path} ({self.n} frames, {self.n / FPS:.1f} s)")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    return fig


def grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def image_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax


def pixels(rect) -> tuple[int, int]:
    """Pixel size of a figure-fraction rectangle: the size to render a 3D view at."""
    return int(rect[2] * W), int(rect[3] * H)


# --------------------------------------------------------------------------- one scored run

class Series:
    """A run_curve directory: its curve (summary.json) and its increments (series/inc_*.vtu) restricted to the
    outer surface of the corner-node tets (straight edges: the element edges as the solver sees them)."""

    def __init__(self, case_dir: Path, label: str, summary: dict | None = None):
        import pyvista as pv
        self.dir = Path(case_dir)
        self.label = label
        self.s = summary or json.loads((self.dir / "summary.json").read_text())
        sd = resolve(self.s.get("series_dir"), self.dir) or self.dir / "series"
        files = sorted(sd.glob("inc_*.vtu"))
        if not files:
            raise SystemExit(f"no series (inc_*.vtu) in {sd}: run_curve(..., series=True) writes it")
        g0 = pv.read(files[0])
        surf = g0.linear_copy().extract_surface()        # tetra10 -> corner tets -> outer triangles
        self.ids = np.asarray(surf["vtkOriginalPointIds"])
        self.X = np.asarray(surf.points).copy()
        self.surf = pv.PolyData(self.X.copy(), surf.faces)
        self.U, self.P = [], []
        for f in files:
            g = g0 if f == files[0] else pv.read(f)
            self.U.append(np.asarray(g.point_data["displacement"])[self.ids])
            self.P.append(np.asarray(g.point_data["PEEQ"])[self.ids])
        # increment k of the series (inc_000 = unloaded) <-> point k of the curve (run_curve starts it at
        # the origin)
        n = min(len(files), len(self.s["force_N"]))
        self.U, self.P = self.U[:n], self.P[:n]
        self.u_part, self.u_cross, self.F = (np.asarray(self.s[k][:n], float)
                                             for k in ("u_part_mm", "u_cross_mm", "force_N"))

    def caption(self) -> str:
        return f"{self.label}\n{self.s['nodes']:,} nodes   error {self.s['mean_rel_err'] * 100:.1f} %"

    def state(self, u_part: float):
        """Surface points, PEEQ and the curve point at part displacement u_part (mm), interpolated between
        the bracketing increments; past the last converged increment the run stays there."""
        k = int(np.clip(np.searchsorted(self.u_part, u_part) - 1, 0, len(self.u_part) - 2))
        a = float(np.clip((u_part - self.u_part[k]) / max(self.u_part[k + 1] - self.u_part[k], 1e-12), 0, 1))
        U = (1 - a) * self.U[k] + a * self.U[k + 1]
        P = (1 - a) * self.P[k] + a * self.P[k + 1]
        xy = ((1 - a) * self.u_cross[k] + a * self.u_cross[k + 1], (1 - a) * self.F[k] + a * self.F[k + 1])
        return self.X + U, P, xy


def resolve(p, base: Path) -> Path | None:
    """A path stored in a summary: absolute, or relative to the repo or to the summary's directory."""
    if not p:
        return None
    p = Path(p)
    for c in (p, ROOT / p, base / p):
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------- 3D views

class PartView:
    """An off-screen pyvista view of one surface with the fixed camera: the mesh (edges) or the deformed
    shape coloured by PEEQ. render() -> RGB image of the panel's pixel size."""

    def __init__(self, surf, size: tuple[int, int], peeq_max: float | None = None, zoom: float = 1.3):
        import pyvista as pv
        from ace.nonlinear import NL_CAMERA
        self.p = pv.Plotter(off_screen=True, window_size=list(size))
        self.p.set_background("white")
        self.mesh = surf.copy()
        if peeq_max is None:
            self.p.add_mesh(self.mesh, color=MESH_FACE, show_edges=True, edge_color=MESH_EDGE, line_width=0.6)
        else:
            self.mesh["PEEQ"] = np.zeros(self.mesh.n_points)
            self.p.add_mesh(self.mesh, scalars="PEEQ", cmap=CMAP, clim=(peeq_max * 1e-3, peeq_max),
                            below_color=ELASTIC, show_scalar_bar=False, smooth_shading=False)
        self.camera = NL_CAMERA
        self.p.camera_position = self.camera
        self.p.camera.zoom(zoom)          # once (zoom narrows the view angle cumulatively): NL_CAMERA frames
                                          # the part loosely, fill the panel more

    def render(self, points=None, peeq=None) -> np.ndarray:
        if points is not None:
            self.mesh.points = points
        if peeq is not None:
            self.mesh["PEEQ"] = peeq
        self.p.camera_position = self.camera
        self.p.render()
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def snapshot(surf, size, zoom: float = 1.3) -> np.ndarray:
    """One still of the undeformed mesh with its element edges."""
    v = PartView(surf, size, zoom=zoom)
    img = v.render()
    v.close()
    return img


def peeq_scale(given: float | None) -> float:
    """Top of the PEEQ colour scale: the fixed range of ace.nonlinear (same in every video and picture)."""
    from ace.nonlinear import PEEQ_CLIM
    return given or PEEQ_CLIM[1]


def colorbar(fig, rect, peeq_max: float):
    import matplotlib as mpl
    cax = fig.add_axes(rect)
    cmap = mpl.colormaps[CMAP].with_extremes(under=ELASTIC)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, peeq_max), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("equivalent plastic strain PEEQ (grey: elastic)", fontsize=11)
    cb.ax.tick_params(labelsize=10)


# --------------------------------------------------------------------------- curves

def curve_axes(fig, rect, meas: dict, xmax: float, fmax: float):
    """Force vs crosshead travel with the measured loading branch in black and its checkpoints."""
    ax = fig.add_axes(rect)
    ax.plot(meas["curve_mm"], meas["curve_N"], color=MEAS_COLOR, lw=2.2, label="measured (Instron)")
    ax.plot(meas["checkpoints_mm"], meas["force_N"], "o", mfc="white", mec=MEAS_COLOR, ms=6, zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("crosshead travel (mm)", fontsize=11)
    ax.set_ylabel("force (N)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def limits(meas: dict, series: list[Series]) -> tuple[float, float]:
    xmax = max([max(meas["curve_mm"])] + [s.u_cross.max() for s in series]) * 1.04
    fmax = max([max(meas["curve_N"])] + [s.F.max() for s in series]) * 1.08
    return xmax, fmax


# --------------------------------------------------------------------------- animation (compare, ramp)

def animate(series: list[Series], out: Path, title: str, peeq_max: float | None = None,
            seconds: float = 5.0, hold_start: float = 1.0, hold_end: float = 2.5):
    """Columns of [mesh, deformed part coloured by PEEQ, force-displacement curve with a moving point],
    advancing together in part displacement. One column (ramp) is laid out as a row instead."""
    from ace.nonlinear import measured
    meas = measured()
    pmax = peeq_scale(peeq_max)
    xmax, fmax = limits(meas, series)
    fig = figure()
    fig.text(0.012, 0.965, title, fontsize=20, weight="bold", va="center")
    status = fig.text(0.012, 0.925, "", fontsize=14, va="center", color="#333333")
    colorbar(fig, [0.70, 0.955, 0.27, 0.016], pmax)
    cols = []
    n = len(series)
    for j, s in enumerate(series):
        if n == 1:      # a row: mesh | PEEQ | curve
            r_mesh, r_peeq, r_curve = [0.01, 0.10, 0.31, 0.74], [0.33, 0.10, 0.31, 0.74], [0.69, 0.14, 0.29, 0.66]
            fig.text(0.012, 0.87, s.caption().replace("\n", "     "), fontsize=14, va="center")
            fig.text(0.165, 0.82, "mesh", fontsize=13, ha="center")
            fig.text(0.485, 0.82, "PEEQ on the deformed part", fontsize=13, ha="center")
        else:           # a column: mesh / PEEQ / curve
            x0, w = j / n, 1 / n
            r_mesh, r_peeq = [x0 + 0.005, 0.35, w / 2 - 0.005, 0.46], [x0 + w / 2, 0.35, w / 2 - 0.005, 0.46]
            r_curve = [x0 + 0.055, 0.065, w - 0.08, 0.23]
            fig.text(x0 + w / 2, 0.875, s.caption(), fontsize=14, ha="center", va="center", linespacing=1.4)
            fig.text(x0 + w / 4, 0.82, "mesh", fontsize=12, ha="center", color="#555555")
            fig.text(x0 + 3 * w / 4, 0.82, "PEEQ (deformed)", fontsize=12, ha="center", color="#555555")
        zoom = 1.3 if n == 1 else 1.0
        image_axes(fig, r_mesh).imshow(snapshot(s.surf, pixels(r_mesh), zoom))
        view = PartView(s.surf, pixels(r_peeq), pmax, zoom)
        im = image_axes(fig, r_peeq).imshow(view.render(s.X, s.P[0]))
        ax = curve_axes(fig, r_curve, meas, xmax, fmax)
        ax.plot(s.u_cross, s.F, color=SIM_COLOR, lw=1.6, label="simulation")
        dot, = ax.plot([0], [0], "o", color="#c0392b", ms=9, zorder=6)
        if j == 0:
            ax.legend(fontsize=9, frameon=False, loc="lower right")
        cols.append((s, view, im, dot))

    umax = max(s.u_part[-1] for s in series)
    nf = int(seconds * FPS)
    video = Video(out)
    for i in range(nf + 1):
        u = umax * i / nf
        for s, view, im, dot in cols:
            pts, P, (x, y) = s.state(u)
            im.set_data(view.render(pts, P))
            dot.set_data([x], [y])
        status.set_text(f"part shortening {u:.2f} mm   (platen pushes the prong tops down; deformation true scale)")
        frame = grab(fig)
        video.add(frame, int(hold_start * FPS) if i == 0 else int(hold_end * FPS) if i == nf else 1)
    video.close()
    for _, view, _, _ in cols:
        view.close()


# --------------------------------------------------------------------------- run: one block per generation

def run_records(run_dir: Path) -> dict[int, list[dict]]:
    """{generation: [run record with a nonlinear check, ...]} of the compression problem."""
    gens: dict[int, list[dict]] = {}
    for f in sorted(run_dir.glob("gen*/runs/*.json")):
        r = json.loads(f.read_text())
        nl = (r.get("checks") or {}).get("nonlinear")
        if r.get("problem_id") == PID and nl and nl.get("ok"):
            r["_agent"] = re.sub(r".*-", "", r["run_id"])
            gens.setdefault(int(r["generation"]), []).append(r)
    return dict(sorted(gens.items()))


def series_of(record: dict, run_dir: Path) -> Series:
    """The scored run's Series: its nonlinear dict says where the curve and the increments are."""
    nl = record["checks"]["nonlinear"]
    d = resolve(nl.get("series_dir"), run_dir)
    case_dir = d.parent if d else resolve(Path(nl["curve_csv"]).parent, run_dir)
    return Series(case_dir, f"g{record['generation']} {record['_agent']}", summary=nl)


def best_record(records: list[dict]) -> dict:
    return max(records, key=lambda r: r["checks"].get("score") or 0.0)


def video_run(run_dir: Path, out: Path, hold: float = 2.0, peeq_max: float | None = None):
    from ace.nonlinear import curve_plot
    import matplotlib.image as mpimg
    gens = run_records(run_dir)
    if not gens:
        raise SystemExit(f"no scored {PID} runs with a nonlinear check under {run_dir}")
    best = {g: series_of(best_record(rs), run_dir) for g, rs in gens.items()}
    pmax = peeq_scale(peeq_max)
    r_mesh, r_peeq, r_curve = [0.005, 0.07, 0.315, 0.76], [0.325, 0.07, 0.315, 0.76], [0.645, 0.05, 0.35, 0.80]
    video = Video(out)
    with tempfile.TemporaryDirectory() as tmp:
        for g, rs in gens.items():
            b, s = best_record(rs), best[g]
            c = b["checks"]
            fig = figure()
            fig.text(0.012, 0.955, f"Generation {g}", fontsize=24, weight="bold", va="center")
            fig.text(0.20, 0.955, f"best: {b['_agent']}   error {c['nonlinear']['mean_rel_err'] * 100:.1f} %   "
                     f"{c['nonlinear']['nodes']:,} nodes   score {c.get('score') or 0:.2f}",
                     fontsize=18, va="center")
            fig.text(0.16, 0.86, "best mesh", fontsize=14, ha="center")
            fig.text(0.48, 0.86, "PEEQ at the peak (deformed, true scale)", fontsize=14, ha="center")
            colorbar(fig, [0.70, 0.94, 0.27, 0.016], pmax)
            image_axes(fig, r_mesh).imshow(snapshot(s.surf, pixels(r_mesh)))
            v = PartView(s.surf, pixels(r_peeq), pmax)
            image_axes(fig, r_peeq).imshow(v.render(s.X + s.U[-1], s.P[-1]))
            v.close()
            curves = [{"label": f"{r['_agent']}: {r['checks']['nonlinear']['mean_rel_err'] * 100:.1f} %, "
                                f"{r['checks']['nonlinear']['nodes']:,} nodes, score {r['checks'].get('score') or 0:.2f}",
                       "u_cross_mm": r["checks"]["nonlinear"]["u_cross_mm"],
                       "force_N": r["checks"]["nonlinear"]["force_N"]} for r in sorted(rs, key=lambda r: r["_agent"])]
            png = curve_plot(curves, str(Path(tmp) / f"g{g}.png"), title=f"generation {g}: every agent vs the test")
            image_axes(fig, r_curve).imshow(mpimg.imread(png))
            video.add(grab(fig), int(hold * FPS))
            import matplotlib.pyplot as plt
            plt.close(fig)
    video.close()


# --------------------------------------------------------------------------- compare

def baseline(name: str, label: str) -> Series:
    d = BASE / name
    if not (d / "summary.json").exists():
        raise SystemExit(f"no {d}/summary.json: run scripts/nl_baselines.py first")
    return Series(d, label)


def video_compare(out: Path, run_dir: Path | None, peeq_max: float | None = None):
    cols = [baseline("auto", "automesher (gmsh defaults)"), baseline("swarm_bc0", "swarm bc0 mesh (before fine-tuning)")]
    if run_dir:
        recs = [r for rs in run_records(run_dir).values() for r in rs]
        if not recs:
            raise SystemExit(f"no scored runs under {run_dir}")
        b = best_record(recs)
        s = series_of(b, run_dir)
        s.label = f"best fine-tuned mesh (g{b['generation']} {b['_agent']})"
        cols.append(s)
    else:
        cols.append(baseline("h4", "uniform 4 mm (placeholder)"))
    animate(cols, out, "C-bracket compression: three meshes against the real test", peeq_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="one block per generation of a fine-tuning run")
    a.add_argument("run_dir", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--hold", type=float, default=2.0, help="seconds per generation")
    b = sub.add_parser("compare", help="automesher | swarm bc0 | best fine-tuned, through the load")
    b.add_argument("out", type=Path)
    b.add_argument("--run", type=Path, help="fine-tuning run dir (else the uniform 4 mm mesh stands in)")
    c = sub.add_parser("ramp", help="one run_curve directory through its increments")
    c.add_argument("case_dir", type=Path)
    c.add_argument("out", type=Path)
    for p in (a, b, c):
        p.add_argument("--peeq-max", type=float, help="top of the PEEQ colour scale (default: ace.nonlinear.PEEQ_CLIM)")
    args = ap.parse_args()
    if args.cmd == "run":
        video_run(args.run_dir, args.out, args.hold, args.peeq_max)
    elif args.cmd == "compare":
        video_compare(args.out, args.run, args.peeq_max)
    else:
        s = Series(args.case_dir, args.case_dir.name)
        animate([s], args.out, f"Nonlinear ramp: {args.case_dir.name}", args.peeq_max)


if __name__ == "__main__":
    main()
