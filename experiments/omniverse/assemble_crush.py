"""RTX frames (render_crush_rtx.py) -> captioned H.264 mp4 (1920x1080, yuv420p, 30 fps).

    python experiments/omniverse/assemble_crush.py --frames FR --meta crush_meta.json --out OUT.mp4 [--ffmpeg ffmpeg]

Per frame (numbers from crush_meta.json "frames", which follow the increments of the run):
  * caption bar at the bottom: part, mesh system, mesh size; platen travel, force (peak), broken / detached cells and the
    phase; the model line; the lift note in the held ending;
  * force vs platen travel inset (right of the bar) with the current point;
  * von Mises colour key (0 .. UTS, turbo) with the red / grey keys, top left.
Frames are composited with Pillow and piped to ffmpeg."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CREDIT = "NVIDIA Omniverse RTX (Isaac Sim 6.2, RTX Real-Time) on DGX Spark"
CMAP_LO = 0.08                   # build_crush_usd.CMAP_LO: turbo from here (its darkest end is lost on a dark set)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def key_image(uts: float) -> Image.Image:
    from matplotlib import colormaps
    w, h = 300, 330
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 14, fill=(12, 15, 22, 170))
    f, fb = font(FONT, 17), font(BOLD, 19)
    d.text((16, 12), "von Mises (MPa)", font=fb, fill=(240, 240, 240))
    x0, y0, bw, bh = 20, 46, 26, 190
    lut = (colormaps["turbo"](CMAP_LO + (1 - CMAP_LO) * np.linspace(1, 0, bh))[:, :3] * 255).astype(np.uint8)
    im.paste(Image.fromarray(np.repeat(lut[:, None, :], bw, axis=1)), (x0, y0))
    for v in np.linspace(0, uts, 6):
        y = y0 + bh - 1 - int(round(v / uts * (bh - 1)))
        d.line((x0 + bw, y, x0 + bw + 6, y), fill=(230, 230, 230), width=2)
        d.text((x0 + bw + 12, y - 10), f"{v:.0f}", font=f, fill=(230, 230, 230))
    d.text((x0 + bw + 60, y0 - 2), "= UTS", font=f, fill=(200, 200, 200))
    y = y0 + bh + 18
    for col, txt in (((242, 20, 14), "broken cell"), ((158, 163, 173), "detached fragment")):
        d.rectangle((x0, y, x0 + bw, y + 18), fill=col)
        d.text((x0 + bw + 12, y - 1), txt, font=f, fill=(230, 230, 230))
        y += 30
    return im


def inset(tr, fo, sep, w=420, h=150):
    """Force vs travel axes; returns (image, mapper(x_mm, F_N) -> pixel)."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(FONT, 14)
    L, R_, T, B = 56, 12, 10, 28
    xmax = max(tr) * 1.04 or 1.0
    ymax = max(fo) * 1.1 or 1.0

    def m(x, y):
        return L + (w - L - R_) * x / xmax, h - B - (h - B - T) * y / ymax
    d.line((L, T, L, h - B), fill=(200, 200, 200), width=1)
    d.line((L, h - B, w - R_, h - B), fill=(200, 200, 200), width=1)
    d.text((4, T - 2), f"{ymax / 1e3:.0f} kN", font=f, fill=(200, 200, 200))
    d.text((L + 8, T - 2), "force vs platen travel", font=f, fill=(200, 200, 200))
    d.text((4, h - B - 16), "0", font=f, fill=(200, 200, 200))
    d.text((w - R_ - 110, h - B + 6), f"travel {xmax:.2f} mm", font=f, fill=(200, 200, 200))
    d.line([m(x, y) for x, y in zip(tr, fo)], fill=(110, 120, 140), width=2)
    if sep is not None:
        x, _ = m(sep, 0)
        d.line((x, T, x, h - B), fill=(242, 60, 50), width=1)
    return im, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--still", default=None, help="also write this frame index as <out>.png ... (comma list)")
    a = ap.parse_args(argv)
    meta = json.loads(Path(a.meta).read_text())
    fr = Path(a.frames)
    frames = meta["frames"]
    fps = meta.get("fps", 30)
    W, H = meta.get("width", 1920), meta.get("height", 1080)
    pt = meta["per_tc"]
    tr, fo = pt["travel_mm"], pt["force_N"]
    sep = meta.get("travel_at_separation_mm")
    key = key_image(meta["uts_MPa"])
    ins, mapper = inset(tr, fo, sep if meta.get("separated") else None)
    bar_h = 132
    fb, f, fs = font(BOLD, 23), font(FONT, 22), font(FONT, 17)
    pk, pkx = meta["peak_force_N"], meta["travel_at_peak_mm"]
    head = (f"{meta['part']} ({meta['part_name']})   ·   mesh system: {meta['label']}   ·   "
            f"{meta['n_nodes']:,} nodes / {meta['n_cells']:,} {meta['cell_type']}")
    model = (f"CalculiX crush: J2 plasticity from the Alpha Fast resin dogbone, a cell breaks at PEEQ {meta['failure_peeq']:g}; "
             f"NLGEOM; bonded platens; true scale   ·   {CREDIT}")
    stills = {int(v) % len(frames) for v in a.still.split(",")} if a.still else set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([a.ffmpeg, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    peak_seen = 0
    for i, f_ in enumerate(frames):
        p = fr / f"frame_{i:05d}.png"
        im = Image.open(p).convert("RGBA") if p.exists() else Image.new("RGBA", (W, H), (20, 22, 28, 255))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle((0, H - bar_h, W, H), fill=(8, 10, 15, 200))
        ov.alpha_composite(key, (24, 24))
        ix, iy = W - ins.width - 24, H - bar_h - ins.height - 12
        d.rounded_rectangle((ix - 8, iy - 6, ix + ins.width + 6, iy + ins.height + 4), 12, fill=(12, 15, 22, 150))
        ov.alpha_composite(ins, (ix, iy))
        x, y = mapper(f_["travel_mm"], f_["force_N"])
        d.ellipse((ix + x - 6, iy + y - 6, ix + x + 6, iy + y + 6), fill=(255, 225, 77))
        inc = f_["inc"]
        peak_seen = max(peak_seen, f_["force_N"])
        last = inc == meta["n_increments"] - 1
        phase = "separated" if (last and meta["separated"]) else ("after peak" if f_["travel_mm"] >= pkx and
                                                                   f_["force_N"] < pk * 0.999 else "loading")
        fN = f_["force_N"] if abs(f_["force_N"]) >= 0.5 else 0.0
        line2 = (f"platen travel {f_['travel_mm']:.3f} mm    force {fN:,.0f} N  (peak {pk:,.0f} N at {pkx:.3f} mm)"
                 f"    broken cells {f_['broken']}" + (f", detached {f_['detached']}" if f_["detached"] else "")
                 + f"    [{phase}]")
        d.text((24, H - bar_h + 12), head, font=fb, fill=(245, 245, 245))
        d.text((24, H - bar_h + 50), line2, font=f, fill=(255, 236, 160))
        note = model
        if f_["lift_mm"] > 0:
            note = (f"held ending: the upper piece and its platen are lifted {f_['lift_mm']:.1f} mm to show the crack "
                    f"(not simulated)   ·   {CREDIT}")
        d.text((24, H - bar_h + 90), note, font=fs, fill=(200, 205, 215))
        im.alpha_composite(ov)
        rgb = im.convert("RGB")
        if i in stills:
            rgb.save(out.with_name(f"{out.stem}_f{i:04d}.png"))
        ff.stdin.write(rgb.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / fps:.1f} s")


if __name__ == "__main__":
    main()
