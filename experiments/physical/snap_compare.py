"""Snap comparison: the crush runs of several mesh systems on one part, side by side on a shared platen-travel clock.

    python experiments/physical/snap_compare.py --part abc_00000003 --run SYSTEM:RUN_DIR[:LABEL] --run ... \
        --out OUT_DIR [--workers 4] [--base-s 8] [--crack-s 4] [--hold-s 4] [--ffmpeg ffmpeg]

Every tile shows the same platen travel in every frame (render_crush.py --layout single --clock). The clock runs at a
constant rate in travel, except inside each run's snap window (peak force -> its last increment), where it
steps through that run's own increments (at most --crack-s seconds per run), so each snap plays in slow motion while
the other tiles, at the same travel, barely move. After the last run's end the clock holds (--hold-s; separated runs
lift their upper piece, as in render_crush.py). Writes OUT_DIR/<part>_snap_clock.json, the tiles
OUT_DIR/tiles/<part>_<system>.mp4 and OUT_DIR/<part>_snap_compare.mp4 (1920x1080, a title bar over a grid)."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TITLE_S = 1.0


def run_travel(run: Path):
    """(travel per increment incl. 0, the peak-force increment (start of the snap window), separated?) over the chain."""
    sys.path.insert(0, str(HERE))
    import render_crush as RC
    rows, _ = RC.series(run)
    tr = np.r_[0.0, [r[1] for r in rows]]
    force = np.r_[0.0, [r[2] for r in rows]]
    first = int(np.argmax(force)) if len(rows) > 1 else None
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    return tr, first, bool(summ.get("separated"))


def clock(runs: dict, fps: int, base_s: float, crack_s: float, hold_s: float) -> np.ndarray:
    """Platen travel per video frame, shared by all tiles."""
    T_end = max(float(tr[-1]) for tr, _, _ in runs.values())
    wins = []                                    # (lo, hi, seconds per knot)
    knots = {0.0, T_end}
    for tr, first, _ in runs.values():
        if first is None:
            continue
        k = np.unique(np.maximum.accumulate(tr)[max(first - 1, 0):])
        knots.update(k.tolist())
        wins.append((float(k[0]), float(k[-1]), min(3.0 / fps, crack_s / max(len(k) - 1, 1))))
    x = np.array(sorted(knots))
    in_win = sum(hi - lo for lo, hi, _ in wins)
    rate = max(T_end - in_win, 1e-9) / base_s            # mm per second outside the crack windows
    t = [0.0]
    for p, q in zip(x[:-1], x[1:]):
        dts = [spk for lo, hi, spk in wins if lo <= p and q <= hi]
        dt = (q - p) / rate
        if dts:
            dt = max(min(dts), min(dt, 3.0 / fps))
        t.append(t[-1] + dt)
    t = np.array(t)
    n = int(math.ceil(t[-1] * fps)) + 1
    T = np.interp(np.arange(n) / fps, t, x)
    return np.r_[np.zeros(int(TITLE_S * fps)), T, np.full(int(hold_s * fps), T_end)]


def grid_shape(n):
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    return 3, 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", required=True)
    ap.add_argument("--run", action="append", required=True, help="SYSTEM:RUN_DIR[:LABEL]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--base-s", type=float, default=8.0)
    ap.add_argument("--crack-s", type=float, default=4.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--rerender", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    specs = []
    for r in a.run:
        s, d, *lab = r.split(":", 2)
        specs.append((s, Path(d), lab[0] if lab else s))
    runs = {s: run_travel(d) for s, d, _ in specs}
    T = clock(runs, a.fps, a.base_s, a.crack_s, a.hold_s)
    ck = out / f"{a.part}_snap_clock.json"
    old = json.loads(ck.read_text()) if ck.exists() else None
    new = {"fps": a.fps, "travel_mm": [round(float(v), 7) for v in T], "systems": [s for s, _, _ in specs]}
    changed = old != new
    ck.write_text(json.dumps(new))
    rows, cols = grid_shape(len(specs))
    tw, th = 1920 // cols, (1080 - 60) // rows
    tw -= tw % 2; th -= th % 2
    tiles = []
    for s, d, lab in specs:
        tile = out / "tiles" / f"{a.part}_{s}.mp4"
        frames = tile.parent / (tile.stem + "_frames")
        if a.rerender or changed or not tile.exists():
            if frames.exists():
                for f in frames.glob("f*.png"):
                    f.unlink()
            subprocess.run([sys.executable, str(HERE / "render_crush.py"), str(d), "--out", str(tile), "--label", lab,
                            "--layout", "single", "--size", f"{tw}x{th}", "--clock", str(ck), "--zoom", "1.25",
                            "--workers", str(a.workers)], check=True, env={**os.environ, "LP_NUM_THREADS": "1"})
        tiles.append(tile)
    inputs = sum((["-i", str(t)] for t in tiles), [])
    n = len(tiles)
    lay = "|".join(f"{(i % cols) * tw}_{(i // cols) * th}" for i in range(n))
    pad = "".join(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];" for i in range(n))
    if n == 1:
        stack = "[v0]null[g]"
    else:
        stack = "".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={lay}:fill=0x1d2330[g]"
    names = {"abc_00000003": "closed tube with stubs", "abc_00000065": "shaft clamp"}
    title = (f"{a.part} ({names.get(a.part, '')}): crush to snap, {n} mesh systems on one platen-travel clock "
             f"(slow motion around each snap)")
    tf = out / f"{a.part}_title.txt"
    tf.write_text(title)
    vf = (pad + stack + f";[g]pad=1920:1080:0:60:color=0x12161f,drawtext=fontfile={FONT}:textfile={tf}:"
          f"fontcolor=white:fontsize=26:x=24:y=16[o]")
    dst = out / f"{a.part}_snap_compare.mp4"
    subprocess.run([a.ffmpeg, "-hide_banner", "-v", "error", "-y", *inputs, "-filter_complex", vf, "-map", "[o]",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(a.fps), str(dst)], check=True)
    print(f"wrote {dst}: {n} systems, {len(T)} frames, {len(T) / a.fps:.1f} s")


if __name__ == "__main__":
    main()
