"""Digitize the force-displacement curve of a Bluehill report image (red curve, gray gridlines).

usage: digitize_bluehill_plot.py IMAGE OUT.csv CHECK.png [SOURCE_NAME]

The curve is followed column by column: where a column holds more than one red run (the legend's red
sample line, labels), the run nearest the previous point is kept, so legend pixels never enter the data.
"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
src, out_csv, out_png = sys.argv[1:4]
source = sys.argv[4] if len(sys.argv) > 4 else Path(src).name
im = np.asarray(Image.open(src).convert("RGB")).astype(int)
H, W, _ = im.shape
r, g, b = im[..., 0], im[..., 1], im[..., 2]
red = (r > 170) & (r - g > 50) & (r - b > 50)
# axis calibration from the gridline centres (gray columns / rows), least squares
def centres(mask_counts, thresh):
    idx = np.where(mask_counts > thresh)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    return [float(np.mean(gr)) for gr in groups]
gray = (abs(r - g) < 12) & (abs(g - b) < 12) & (r > 150) & (r < 235)
xc = [c for c in centres(gray.sum(0), 0.35 * H) if c > 300]            # 1..9 mm (0 is the dark axis)
yc = [c for c in centres(gray.sum(1), 0.35 * W) if 130 < c < 1050]     # 5000..500 N
xs_mm = np.arange(1, len(xc) + 1); ys_N = 500.0 * np.arange(len(yc), 0, -1)
ax, bx = np.polyfit(xs_mm, xc, 1)       # px = ax * mm + bx
ay, by = np.polyfit(ys_N, yc, 1)        # px = ay * N + by
print(f"x: {ax:.3f} px/mm, x0 at {bx:.2f} px (residual max {np.abs(np.polyval([ax, bx], xs_mm) - xc).max():.2f} px); "
      f"y: {ay:.5f} px/N, y0 at {by:.2f} px (residual max {np.abs(np.polyval([ay, by], ys_N) - yc).max():.2f} px)")
pts, prev = [], None          # prev: pixel row of the last point kept
for col in range(int(bx) + 1, W - 30):
    rows = np.where(red[:, col])[0]
    if rows.size == 0:
        continue
    runs = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
    if prev is None:
        run = max(runs, key=len)
    else:                     # the run nearest the curve so far (0 when it spans the previous row)
        run = min(runs, key=lambda r: max(r.min() - prev, prev - r.max(), 0))
    top, bot = run.min(), run.max()
    prev = bot if bot - top > 25 else 0.5 * (top + bot)
    mm = (col - bx) / ax
    if bot - top > 25:            # a drop (or the final break line): load falls at nearly constant displacement
        pts.append((mm, (top + 3 - by) / ay)); pts.append((mm, (bot - 3 - by) / ay))
    else:
        pts.append((mm, (0.5 * (top + bot) - by) / ay))
pts = np.array(pts)
pts[:, 1] = np.clip(pts[:, 1], 0, None)
i = int(np.argmax(pts[:, 1]))
print(f"{len(pts)} points; max force {pts[i, 1]:.0f} N at {pts[i, 0]:.3f} mm; last point {pts[-1, 0]:.3f} mm "
      "(compare with the report's table)")
near = pts[np.abs(pts[:, 0] - pts[-1, 0]) < 0.02]
print(f"top of the final break line: {near[:, 1].max():.0f} N at {pts[-1, 0]:.3f} mm")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"# Digitized from the plot in {source} (Bluehill report printed to PDF; the raw data was not exported)."])
    w.writerow(["# Resolution: 1 px = %.4f mm and %.1f N; the line is ~6 px wide, so about +-0.01 mm and +-15 N." % (1 / ax, 1 / abs(ay))])
    w.writerow(["Displacement", "Force"])
    w.writerow(["(mm)", "(N)"])
    for x, y in pts:
        w.writerow([f"{x:.4f}", f"{y:.1f}"])
# overlay for checking
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(im.astype(np.uint8)); axs[0].plot(pts[:, 0] * ax + bx, pts[:, 1] * ay + by, "b-", lw=0.6); axs[0].set_title("digitized (blue) over the report image")
axs[1].plot(pts[:, 0], pts[:, 1], "b-", lw=0.8); axs[1].set_xlabel("displacement [mm]"); axs[1].set_ylabel("force [N]"); axs[1].grid(True)
axs[1].set_title("digitized curve")
fig.tight_layout(); fig.savefig(out_png, dpi=90)
