#!/usr/bin/env python3
"""Pictures of the C-bracket's CalculiX ramp (scripts/ccx_compression_ramp.py): the part's outer surface on
the deformed shape (true scale), coloured by equivalent plastic strain (PEEQ) and by the largest principal
stress, at first yield, at the test's peak load (or the model's largest force) and at the end of the ramp.
Optionally the same increments as .vtu files for ParaView.

    python scripts/render_ccx_ramp.py                         # every finished run in runs/ccx_ramp
    python scripts/render_ccx_ramp.py --runs corrected_stick --vtu

Reads <work>/<run>/job.inp + job.frd (streamed) and <work>/<run>.json; writes
physical_tests/abc_00000065_compression/ccx_ramp/render_<run>.png and, with --vtu, <work>/<run>/vtu/*.vtu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "physical_tests" / "abc_00000065_compression" / "ccx_ramp"
TEST_PEAK_N = 4755.2          # digitized peak of the Instron curve (the report says 4.80 kN)


def read_inp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates (n, 3; node id = row + 1) and C3D10 connectivity (0-based) of a feacore ccx deck."""
    nodes, conn, mode, buf = {}, [], None, []
    with open(path) as f:
        for ln in f:
            if ln.startswith("*"):
                u = ln.upper().replace(" ", "")
                mode = ("n" if u.startswith("*NODE,") or u.strip() == "*NODE" else
                        "e" if u.startswith("*ELEMENT") and "C3D10" in u else None)
                buf = []
                continue
            if mode == "n":
                a = ln.split(",")
                nodes[int(a[0])] = [float(x) for x in a[1:4]]
            elif mode == "e":
                buf += [int(x) for x in ln.replace(",", " ").split()]
                if len(buf) >= 11:
                    conn.append(buf[1:11])
                    buf = []
    X = np.zeros((max(nodes), 3))
    for k, v in nodes.items():
        X[k - 1] = v
    return X, np.array(conn) - 1


def boundary_triangles(conn: np.ndarray) -> np.ndarray:
    """Outer faces of the tets as corner-node triangles (faces that belong to one tet only)."""
    c = conn[:, :4]
    f = np.vstack([c[:, [0, 1, 2]], c[:, [0, 1, 3]], c[:, [0, 2, 3]], c[:, [1, 2, 3]]])
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def frd_increments(path: Path, times: list[float]) -> dict[float, dict]:
    """The nodal blocks (DISP, STRESS, PE) of the output increment nearest each wanted time, in one
    streamed pass over the .frd (nonlinear .frd files are hundreds of MB)."""
    from feacore.solver._ccx_io import _iter_frd
    best: dict[float, tuple[float, dict]] = {}
    for t, b in _iter_frd(path):
        if b.name not in ("DISP", "STRESS", "PE"):
            continue
        for want in times:
            bt, data = best.get(want, (None, None))
            if bt == t:
                data[b.name] = b
            elif bt is None or abs(t - want) < abs(bt - want):
                best[want] = (t, {b.name: b})
    return {t: data for t, data in best.values()}


def nodal(block, n: int, ncols: int) -> np.ndarray:
    out = np.zeros((n, ncols))
    idx = block.nodes - 1
    ok = (idx >= 0) & (idx < n)
    out[idx[ok]] = block.values[ok][:, :ncols]
    return out


def principal_max(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = S.T
    T = np.stack([np.stack([sxx, sxy, szx], -1), np.stack([sxy, syy, syz], -1), np.stack([szx, syz, szz], -1)], -2)
    return np.linalg.eigvalsh(T)[:, -1]


def render(run: str, work: Path, vtu: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r = json.loads((work / f"{run}.json").read_text())
    umax = r["umax_mm"]
    hist = r["history"]
    F = np.array([h["F_N"] for h in hist])
    k_peak = int(np.argmin(np.abs(F - TEST_PEAK_N))) if F.max() >= 0.9 * TEST_PEAK_N else int(np.argmax(F))
    stages = [("first yield", r["first_yield"]["u_mm"]) if "first_yield" in r else None,
              ("test's peak load" if F.max() >= 0.9 * TEST_PEAK_N else "model's largest force", hist[k_peak]["u_mm"]),
              ("end of the ramp", hist[-1]["u_mm"])]
    stages = [s for s in stages if s]
    X, conn = read_inp(work / run / "job.inp")
    tri = boundary_triangles(conn)
    incs = frd_increments(work / run / "job.frd", [u / umax for _, u in stages])
    times = sorted(incs)
    fig = plt.figure(figsize=(15, 4.6 * len(stages)), dpi=110)
    for row, (label, u) in enumerate(stages):
        t = min(times, key=lambda h: abs(h - u / umax))
        b = incs[t]
        n = len(X)
        U = nodal(b["DISP"], n, 3)
        peeq = nodal(b["PE"], n, 1)[:, 0] if "PE" in b else np.zeros(n)
        s1 = principal_max(nodal(b["STRESS"], n, 6)) / 1e6
        Y = (X + U) * 1000.0                                           # deformed, true scale, mm
        Fu = float(np.interp(t * umax, [h["u_mm"] for h in hist], F))
        if vtu:
            import meshio
            d = work / run / "vtu"
            d.mkdir(exist_ok=True)
            meshio.Mesh(X, [("tetra10", conn)], point_data={"displacement": U, "peeq": peeq, "s1_MPa": s1}).write(
                str(d / f"{run}_u{t * umax:.2f}mm.vtu"))
        for col, (name, v, vmin, vmax, cmap) in enumerate((("equivalent plastic strain", peeq, 0.0, 0.10, "inferno"),
                                                           ("largest principal stress [MPa]", s1, -20.0, 60.0, "RdBu_r"))):
            for view, azim in enumerate((-58, 122)):
                ax = fig.add_subplot(len(stages), 4, 4 * row + 2 * col + view + 1, projection="3d")
                P = Y[tri][:, :, [0, 2, 1]]                            # plot x, z, y: y (the load axis) up
                val = v[tri].mean(axis=1)
                nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
                light = 0.55 + 0.45 * np.abs(nrm @ np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]))
                rgba = matplotlib.colormaps[cmap](colors.Normalize(vmin, vmax, clip=True)(val))
                rgba[:, :3] *= light[:, None]
                ax.add_collection3d(Poly3DCollection(P, facecolors=rgba, edgecolors="none"))
                lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
                ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
                ax.set_box_aspect(hi - lo)
                ax.view_init(elev=18, azim=azim)
                ax.set_axis_off()
                if view == 0:
                    ax.set_title(f"{name}\n{label}: travel {t * umax:.2f} mm, {Fu / 1000:.2f} kN", fontsize=8.5)
                    sm = cm.ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=cmap)
                    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.0, location="left")
    fig.suptitle(f"abc_00000065 compression, CalculiX J2 + NLGEOM ({run}: E {r['material']['E']:.0f} MPa, "
                 f"{r['platen']} platen); deformed shape at true scale, two opposite views", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"render_{run}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=ROOT / "runs" / "ccx_ramp")
    ap.add_argument("--runs", default="", help="comma-separated run names (default: every finished run)")
    ap.add_argument("--vtu", action="store_true", help="also write the rendered increments as .vtu (ParaView)")
    a = ap.parse_args(argv)
    runs = [x for x in a.runs.split(",") if x] or sorted(
        p.stem for p in a.work.glob("*_s*.json") if p.stem.rsplit("_", 1)[-1] in ("slide", "stick"))
    for run in runs:
        print(render(run, a.work, a.vtu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
