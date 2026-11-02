#!/usr/bin/env python3
"""Act 3: where and when the part is predicted to crack, against the real test's first crack.

A criterion-based crack ONSET prediction (no fracture model, no crack growth) on the CalculiX ramps of the story
cases of scripts/nl_story_video.py (first / before / stage2 / stage3 / after). Per increment of each case's series:
max nodal PEEQ and max principal stress s1; the first crosshead travel (interpolated within the step) at which
  (a) PEEQ >= failure_peeq    (plastic strain at break of the resin's dogbone test)
  (b) s1   >= uts_eng         (the dogbone's tensile strength)
Each is evaluated raw (the single highest node) and robust (the value reached by at least ROBUST_K nodes within
ROBUST_R_MM: one spiky node at a sharp corner does not count). The real first crack is the test's peak: the
first load drop of the Instron curve.

    python scripts/nl_break.py analyze --run ~/feagent/ace_runs/nl_ft    # break.json, table (md/png/pdf)
    python scripts/nl_break.py videos  --run ~/feagent/ace_runs/nl_ft    # before_after_break.mp4, after_break.mp4,
                                                     # three_way_break.mp4, first_vs_after_break.mp4 (--only NAME ...)
    python scripts/nl_break.py still   --run ~/feagent/ace_runs/nl_ft    # after_crack_closeup.png/.pdf
    python scripts/nl_break.py page                                      # index.html
    python scripts/nl_break.py all     --run ~/feagent/ace_runs/nl_ft

Outputs: talk_plots/act3_real_data/break/. Reuses the cases, renderer and video layouts of nl_story_video.py
(its own outputs are not touched).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nl_story_video as sv  # noqa: E402
import nl_video as nv  # noqa: E402

OUT = ROOT / "talk_plots" / "act3_real_data" / "break"
MATERIAL = ROOT / "experiments" / "material" / "alpha_fast_dogbone1.json"
ROBUST_K, ROBUST_R_MM = 3, 3.0      # robust value: reached by >= 3 nodes within 3 mm (about 2x the node spacing:
                                    # at 1.5 mm the coarser regions, e.g. the bore, had < 3 nodes and were skipped)
FACE_TOL_MM = 2.0                   # faces within 2 mm of a site (as ace.nonlinear._hotspots)
CRACK = "#d62728"                   # predicted crack marker
REPORT_PEAK = (2.56, 4800.0)        # the machine report's peak (mm, N); the digitized curve peaks a bit earlier
ORDER = ("first", "before", "stage2", "stage3", "after")
CRIT = {"peeq": "PEEQ >= failure_peeq", "s1": "s1 >= UTS"}
# what the bare face tags are (CAD faces of abc_00000065; teammates' notes in cload_compression.json)
# (checked against the faces' extents: f8 is the right prong's outer side at x = 23.7 mm, where the hex nut pocket
#  f38-f42 opens; f14 / f16 the prongs' inner sides at x = -/+12.9 mm; f12 / f13 the flat sides at z = -/+8.1 mm)
FACE_NAMES = {"f5": "inner bore", "f6": "outer side", "f8": "prong outer side", "f12": "flat side",
              "f13": "flat side", "f14": "prong inner side", "f16": "prong inner side", "f17": "prong top",
              "f34": "prong top", "f38": "nut pocket", "f39": "nut pocket", "f40": "nut pocket",
              "f41": "nut pocket", "f42": "nut pocket", "f55": "prong hole"}
POTENTIAL = 0.85                    # 'potential crack': s1 at 85 % of the dogbone strength (chosen after seeing the test)
POT = "#e67e00"                     # potential crack marker (amber)
# Where the printed part actually cracked first: the team's observation of the broken part (no photo in the repo):
# the lower inner bore (face f5), at the 'after' case's most stressed spot at the real crack.
REAL_SITE = {"xyz_mm": [-14.56, -13.22, -4.31], "faces": ["f5"], "name": "lower inner bore", "source": "observed"}


def site_match(xyz_mm, faces) -> bool:
    """A site matches the observed crack when it is on the bore (f5) in its lower half (y < 0)."""
    return "f5" in faces and xyz_mm[1] < 0


def material() -> dict:
    m = json.loads(MATERIAL.read_text())
    return {"failure_peeq": float(m["failure_peeq"]), "uts_MPa": float(m["uts_eng"])}


def real_crack() -> dict:
    """The test's first crack: the peak of the loading branch, right before the first load drop."""
    from ace.nonlinear import measured
    m = measured()
    i = int(np.argmax(m["curve_N"]))
    return {"travel_mm": float(m["curve_mm"][i]), "force_N": float(m["curve_N"][i]),
            "report_travel_mm": REPORT_PEAK[0], "report_force_N": REPORT_PEAK[1]}


# --------------------------------------------------------------------------- one case's full fields

class Fields:
    """Every node of a case's series: X, displacement, PEEQ, s1 (MPa) per increment, and the CAD face groups
    of its mesh (to name the faces at a site)."""

    def __init__(self, c: sv.Case, checkout: Path):
        import pyvista as pv
        sd = nv.resolve(c.s.s.get("series_dir"), c.s.dir) or c.s.dir / "series"
        files = sorted(Path(sd).glob("inc_*.vtu"))[: len(c.s.F)]
        self.U, self.P, self.S1, self.S3, self.Pip = [], [], [], [], []
        for f in files:
            g = pv.read(f)
            self.U.append(np.asarray(g.point_data["displacement"]))
            self.P.append(np.asarray(g.point_data["PEEQ"]))
            self.S1.append(np.asarray(g.point_data["s1"]) / 1e6)
            self.S3.append(np.asarray(g.point_data["s3"]) / 1e6)
            self.Pip.append(float(np.max(g.cell_data["PEEQ_ip_max"])) if "PEEQ_ip_max" in g.cell_data else 0.0)
        self.X = np.asarray(g.points)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.X)
        self.faces = self._faces(c, checkout)

    def _faces(self, c, checkout: Path) -> dict:
        from feacore.solver._ccx_mesh import read_gmsh
        p = c.s.s.get("mesh_path")
        mp = next((q for q in (nv.resolve(p, c.s.dir), checkout / str(p), c.s.dir / "mesh_straight.msh",
                               c.s.dir / "mesh.msh") if q and Path(q).exists()), None)
        if mp is None:
            return {}
        gm = read_gmsh(mp)
        out = {}
        for g, blocks in gm.groups.items():
            if g.startswith("f") and g[1:].isdigit() and gm.group_dim.get(g) == 2:
                ids = np.unique(np.concatenate([conn.ravel() for _, conn in blocks]))
                out[g] = gm.points[ids]
        return out

    def faces_near(self, xyz: np.ndarray) -> list[str]:
        near = [g for g, P in self.faces.items() if np.min(np.linalg.norm(P - xyz, axis=1)) < FACE_TOL_MM * 1e-3]
        return sorted(near, key=lambda g: int(g[1:]))

    def robust(self, v: np.ndarray, top: int = 300) -> tuple[float, int]:
        """(value reached by >= ROBUST_K nodes within ROBUST_R_MM of one node, that node), searched around the
        `top` highest nodes."""
        best, at = -np.inf, 0
        for i in np.argsort(-v)[:top]:
            nb = self.tree.query_ball_point(self.X[i], ROBUST_R_MM * 1e-3)
            if len(nb) >= ROBUST_K:
                r = float(np.sort(v[nb])[-ROBUST_K])
                if r > best:
                    best, at = r, int(i)
        return best, at


def series_of_max(fd: Fields, field: str, robust: bool) -> tuple[np.ndarray, np.ndarray]:
    vals = fd.P if field == "peeq" else fd.S1
    out, where = [], []
    for v in vals:
        if robust:
            r, i = fd.robust(v)
        else:
            i = int(np.argmax(v))
            r = float(v[i])
        out.append(r)
        where.append(i)
    return np.array(out), np.array(where)


def onset(c: sv.Case, fd: Fields, field: str, thr: float, robust: bool, x_real: float) -> dict:
    """First crossing of thr by the case's max field, interpolated linearly within the step."""
    v, where = series_of_max(fd, field, robust)
    xc, F, up = c.xc, c.s.F, c.s.u_part
    d = {"criterion": CRIT[field], "robust": robust, "threshold": thr,
         "max_series": [round(float(a), 5) for a in v],
         "at_real_crack": float(np.interp(x_real, xc, v)), "end_value": float(v[-1]), "end_travel_mm": float(xc[-1])}
    kr = min(int(np.searchsorted(xc, x_real)), len(v) - 1)      # first increment at/after the real crack
    ir = int(where[kr])
    d["real_site"] = {"increment": kr, "node": ir, "value": float(v[kr]), "xyz_mm": (fd.X[ir] * 1e3).round(2).tolist(),
                      "faces": fd.faces_near(fd.X[ir])}
    k = next((k for k in range(1, len(v)) if v[k] >= thr), None)
    if k is None:
        i = int(where[-1])
        d.update(predicted=False, node=i, xyz_mm=(fd.X[i] * 1e3).round(2).tolist(), faces=fd.faces_near(fd.X[i]))
        d["site_match"] = site_match(d["real_site"]["xyz_mm"], d["real_site"]["faces"])
        return d
    a = float(np.clip((thr - v[k - 1]) / max(v[k] - v[k - 1], 1e-12), 0, 1))
    i = int(where[k])
    d["state_at_site"] = {"peeq": float(fd.P[k][i]), "s1_MPa": float(fd.S1[k][i]), "s3_MPa": float(fd.S3[k][i])}
    d.update(predicted=True, increment=k, travel_mm=float((1 - a) * xc[k - 1] + a * xc[k]),
             force_N=float((1 - a) * F[k - 1] + a * F[k]), u_part_mm=float((1 - a) * up[k - 1] + a * up[k]),
             node=i, xyz_mm=(fd.X[i] * 1e3).round(2).tolist(), faces=fd.faces_near(fd.X[i]))
    d["before_real"] = d["travel_mm"] < x_real
    d["site_match"] = site_match(d["xyz_mm"], d["faces"])
    return d


# --------------------------------------------------------------------------- analysis

def load(run_dir: Path):
    """The story's cases (nl_story_video.load_cases), with 'after' = the run's most accurate mesh."""
    return sv.load_cases(run_dir, "accurate")


def analyze(run_dir: Path) -> dict:
    cases, pick = load(run_dir)
    mat, real = material(), real_crack()
    checkout = run_dir.resolve().parents[1]
    res = {"material": mat, "real_crack": real, "real_site": REAL_SITE, "potential_fraction": POTENTIAL,
           "robust": {"k": ROBUST_K, "r_mm": ROBUST_R_MM},
           "after_pick": pick, "cases": {}}
    for key in ORDER:
        c = cases[key]
        fd = Fields(c, checkout)
        r = {"label": c.label, "nodes": c.nodes, "mean_rel_err": c.err, "peeq_ip_max_end": fd.Pip[-1],
             "peeq_ip_max_series": [round(p, 5) for p in fd.Pip]}
        for field, thr in (("peeq", mat["failure_peeq"]), ("s1", mat["uts_MPa"])):
            for rob in (False, True):
                r[f"{field}_{'robust' if rob else 'raw'}"] = onset(c, fd, field, thr, rob, real["travel_mm"])
        for rob in (False, True):
            r[f"s1_{'robust' if rob else 'raw'}_85"] = onset(c, fd, "s1", POTENTIAL * mat["uts_MPa"], rob,
                                                              real["travel_mm"])
        res["cases"][key] = r
        print(key, json.dumps({k: (v.get("travel_mm"), v.get("force_N"), v["faces"], round(v["at_real_crack"], 4))
                               for k, v in r.items() if isinstance(v, dict)}))
    res["mesh_sensitivity"] = sensitivity(mat, real["travel_mm"], checkout, res["cases"]["after"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "break.json").write_text(json.dumps(res, indent=1))
    table(res)
    return res


class _Lite:
    """A baseline run_curve directory as the minimal Case that Fields/onset need."""

    def __init__(self, d: Path, label: str):
        self.s = nv.Series(d, label)
        self.xc = np.maximum.accumulate(self.s.u_cross)
        self.label, self.nodes = label, int(self.s.s["nodes"])


SENS = (("auto", "automesher (gmsh defaults)"), ("swarm_bc0", "pre-trained swarm mesh"),
        ("h4", "uniform 4 mm"), ("h25", "uniform 2.5 mm"), ("h20", "uniform 2 mm"))


def sensitivity(mat: dict, x_real: float, checkout: Path, after: dict) -> list[dict]:
    """The same criteria on the baseline meshes (learned conditions, same ramp): how the onset moves with
    the mesh."""
    out = []
    for name, label in SENS:
        d = nv.BASE / name
        if not (d / "summary.json").exists() or not list((d / "series").glob("inc_*.vtu")):
            continue
        c = _Lite(d, label)
        fd = Fields(c, checkout)
        r = {"name": name, "label": label, "nodes": c.nodes, "mean_rel_err": float(c.s.s["mean_rel_err"])}
        for field, thr in (("peeq", mat["failure_peeq"]), ("s1", mat["uts_MPa"])):
            for rob in (False, True):
                o = onset(c, fd, field, thr, rob, x_real)
                r[f"{field}_{'robust' if rob else 'raw'}"] = {k: o.get(k) for k in (
                    "predicted", "travel_mm", "force_N", "faces", "at_real_crack", "end_value", "end_travel_mm")}
        o = onset(c, fd, "s1", POTENTIAL * mat["uts_MPa"], False, x_real)
        r["s1_raw_85"] = {k: o.get(k) for k in ("predicted", "travel_mm", "force_N", "xyz_mm", "faces", "site_match")}
        out.append(r)
        print("sens", name, c.nodes, {k: (v["travel_mm"], round(v.get("at_real_crack", -1), 4)) for k, v in r.items()
                                      if isinstance(v, dict)})
    return out


def faces_txt(fs: list[str]) -> str:
    names = sorted({FACE_NAMES[f] for f in fs if f in FACE_NAMES})
    return " / ".join(fs) + (f" ({', '.join(names)})" if names else "")


def site_name(xyz_mm, faces) -> str:
    """A short name for a site: 'lower inner bore', 'nut pocket corner', 'prong inner side', ..."""
    if site_match(xyz_mm, faces):
        return "lower inner bore"
    if "f5" in faces:
        return "upper inner bore"
    if "f8" in faces and any(f in faces for f in ("f38", "f39", "f41", "f42")):
        return "nut-pocket corner"
    names = [FACE_NAMES[f] for f in faces if f in FACE_NAMES and FACE_NAMES[f] != "flat side"]
    return names[0] if names else ("flat side" if faces else "interior")


def site_of(d: dict) -> tuple[list, list]:
    """(xyz, faces) of a criterion's site: where it is first reached, else the most stressed point at the real crack."""
    return (d["xyz_mm"], d["faces"]) if d["predicted"] else (d["real_site"]["xyz_mm"], d["real_site"]["faces"])


def cell(d: dict, what: str = "break") -> str:
    if not d["predicted"]:
        unit = " MPa" if d["criterion"].startswith("s1") else ""
        return f"no {what} by {d['end_travel_mm']:.2f} mm ({d['at_real_crack']:.3g}{unit} at the real crack)"
    return f"{d['travel_mm']:.2f} mm, {d['force_N']:,.0f} N"


def match_txt(r: dict) -> str:
    """Does the stress criterion's site (raw) match the observed crack (lower inner bore)?"""
    out = []
    for k, tag in (("s1_raw_85", "85 %"), ("s1_raw", "100 %")):
        d = r[k]
        xyz, fs = site_of(d)
        yes = "yes" if site_match(xyz, fs) else "no"
        out.append(f"{tag}: {yes} ({site_name(xyz, fs)}{'' if d['predicted'] else ', most stressed'})")
    return "; ".join(out)


COLS = [("peeq_raw", "PEEQ, raw"), ("peeq_robust", "PEEQ, robust"), ("s1_raw", "s1 >= 100 %, raw"),
        ("s1_robust", "s1 >= 100 %, robust"), ("s1_raw_85", "potential: s1 >= 85 %, raw")]


def table(res: dict):
    """break_table.md, and the same as a figure (break_table.png/pdf)."""
    real, mat = res["real_crack"], res["material"]
    cols = COLS
    rows = []
    for key in ORDER:
        r = res["cases"][key]
        rows.append([f"{key}: {r['label']} ({r['nodes']:,} nodes, {sv.pct(r['mean_rel_err'])} off)"]
                    + [cell(r[k], "potential crack" if k.endswith("_85") else "break") for k, _ in cols]
                    + [match_txt(r)])
    md = [f"# Predicted crack onset vs the real first crack",
          "",
          f"Real first crack (test): first load drop at the peak, {real['force_N']:,.0f} N at "
          f"{real['travel_mm']:.2f} mm crosshead (digitized curve; the machine report says "
          f"{real['report_force_N'] / 1e3:.2f} kN at {real['report_travel_mm']:.2f} mm). Where: the lower inner bore "
          f"(face f5, near {tuple(REAL_SITE['xyz_mm'])} mm), observed by the team on the broken part (no photo in the repo).",
          f"Criteria: PEEQ >= {mat['failure_peeq']} (dogbone plastic strain at break), s1 >= {mat['uts_MPa']} MPa "
          f"(dogbone UTS: 'predicted crack'), s1 >= {POTENTIAL:.0%} of it = {POTENTIAL * mat['uts_MPa']:.1f} MPa "
          f"('potential crack': a margin for print defects and layer lines; the threshold was chosen after seeing this "
          f"test). Raw: the single highest node; robust: reached by >= {ROBUST_K} nodes within {ROBUST_R_MM} mm. "
          f"Travel = each case's own crosshead mapping.",
          "",
          "| case | " + " | ".join(h for _, h in cols) + " | site matches the real crack (observed: lower inner bore)? |",
          "|---" * (len(cols) + 2) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Sites (first node over the threshold, else the most stressed at the end; undeformed xyz mm, faces "
           "within 2 mm):", ""]
    for key in ORDER:
        for k, h in cols + [("s1_robust_85", "potential: s1 >= 85 %, robust")]:
            d = res["cases"][key][k]
            md.append(f"- {key}, {h}: {'site' if d['predicted'] else 'max at the end'} {d['xyz_mm']} mm, "
                      f"{faces_txt(d['faces'])} ({site_name(d['xyz_mm'], d['faces'])}); value at the real crack "
                      f"travel {d['at_real_crack']:.3g}"
                      + ("" if d["predicted"] else f"; most stressed at the real crack {d['real_site']['xyz_mm']} mm, "
                         f"{faces_txt(d['real_site']['faces'])}"))
    md += ["", "Potential crack (s1 >= 85 %), robust: " + "; ".join(
        f"{k} {cell(res['cases'][k]['s1_robust_85'], 'potential crack')}" for k in ORDER), ""]
    md += ["", "Mesh sensitivity (learned conditions, same ramp; onset travel by each criterion; value at the "
           "real crack travel in brackets):", "",
           "| mesh | nodes | PEEQ raw | PEEQ robust | s1 raw (MPa) | s1 robust (MPa) | s1 >= 85 %, raw (site) |",
           "|---|---|---|---|---|---|---|"]
    for r in res.get("mesh_sensitivity", []):
        p85 = r.get("s1_raw_85", {})
        p85t = (f"{p85['travel_mm']:.2f} mm ({site_name(p85['xyz_mm'], p85['faces'])})" if p85.get("predicted")
                else "none")
        md.append(f"| {r['label']} | {r['nodes']:,} | " + " | ".join(
            (f"{r[k]['travel_mm']:.2f} mm" if r[k]["predicted"] else "none") + f" [{r[k]['at_real_crack']:.3g}]"
            for k, _ in cols[:4]) + f" | {p85t} |")
    (OUT / "break_table.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    fig, ax = plt.subplots(figsize=(17, 5.4))
    ax.set_axis_off()
    head = ["case"] + [textwrap.fill(h, 16) for _, h in cols] + ["site matches the real\ncrack (observed)?"]
    body = [[textwrap.fill(r[0], 34)] + [textwrap.fill(x, 19) for x in r[1:-1]] + [textwrap.fill(r[-1], 30)]
            for r in rows]
    rt = f"{real['travel_mm']:.2f} mm, {real['force_N']:,.0f} N"
    body.append(["real test: first crack"] + [rt] * len(cols) + ["lower inner bore (f5)"])
    t = ax.table(cellText=body, colLabels=head, loc="center", cellLoc="center",
                 colWidths=[0.21] + [0.12] * len(cols) + [0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 3.6)
    for (i, j), cl in t.get_celld().items():
        cl.set_edgecolor("#cccccc")
        if i == 0 or i == len(body):
            cl.set_text_props(weight="bold")
        if 0 < i < len(body) and 0 < j <= len(cols):
            d = res["cases"][ORDER[i - 1]][cols[j - 1][0]]
            if d["predicted"]:
                cl.set_text_props(color=POT if cols[j - 1][0].endswith("_85") else CRACK)
    ax.set_title(f"Predicted crack onset (crosshead travel, force) vs the real first crack; PEEQ >= "
                 f"{mat['failure_peeq']}, s1 >= {mat['uts_MPa']:.1f} MPa (100 %) or {POTENTIAL * mat['uts_MPa']:.1f} "
                 f"MPa (85 %, potential)", fontsize=12)
    fig.text(0.5, 0.01, "Real crack site observed by the team on the broken part (no photo in the repo). Potential crack: "
             "tension at 85 % of the resin's dogbone strength (a margin for print defects and layer lines; the "
             "threshold was chosen after seeing this test).", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"break_table.{ext}", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- videos

VIDEO_CRIT = "s1_raw"      # the marker in the videos: max principal stress reaches the dogbone strength


def project(view, xyz) -> tuple[float, float, bool]:
    """Pixel (column, row) of a world point in a PartView's image, and whether it is in front (not hidden
    behind the part itself: compared with the z-buffer in a small window, so a node on a silhouette edge counts
    as visible)."""
    ren = view.p.renderer
    ren.SetWorldPoint(*map(float, xyz), 1.0)
    ren.WorldToDisplay()
    dx, dy, _ = ren.GetDisplayPoint()
    h = view.p.window_size[1]
    col, row = dx, h - dy
    try:
        zb = view.p.get_image_depth()                     # negative distance from the camera, NaN on background
        r0, c0 = int(np.clip(row, 0, zb.shape[0] - 1)), int(np.clip(col, 0, zb.shape[1] - 1))
        z = zb[max(r0 - 4, 0): r0 + 5, max(c0 - 4, 0): c0 + 5]
        cam = view.p.camera                               # depth image = distance along the view axis
        ax_ = np.array(cam.focal_point) - np.array(cam.position)
        depth = -float(np.dot(np.asarray(xyz) - np.array(cam.position), ax_ / np.linalg.norm(ax_)))
        visible = bool(np.isnan(z).any() or (depth >= z - 1.5e-3).any())
    except Exception:  # noqa: BLE001  no depth buffer: assume visible
        visible = True
    return col, row, visible


def _halo(lw: float):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw + 3, foreground="white")]


class Marks:
    """The crack markers of one case, on its part and its graph:
    - amber ring 'potential crack' where s1 first reaches 85 % of the dogbone strength (amber dot on the graph);
    - red ring + X 'predicted crack' where s1 first reaches the strength (red X on the graph); it replaces the amber
      ring when both are at the same spot; if the strength is never reached the panel says 'no break predicted'
      (and, if not even 85 % is reached, a dashed red ring marks the most stressed point from the real crack on);
    - black ring 'real crack (observed)' at the lower inner bore, from the real crack's travel on."""

    def __init__(self, fig, c, view, im, graph, mag: float, d: dict, p: dict, real: dict, uts: float):
        from matplotlib.patches import Circle
        self.c, self.view, self.mag, self.d, self.p, self.real = c, view, mag, d, p, real
        self.ax = ax = im.axes
        ax.set_autoscale_on(False)
        X = c.s.X
        near = lambda xyz: int(np.argmin(np.linalg.norm(X - np.array(xyz) * 1e-3, axis=1)))   # nearest surface pt
        self.j = near(d["xyz_mm"] if d["predicted"] else d["real_site"]["xyz_mm"])
        self.jp = near(p["xyz_mm"]) if p["predicted"] else self.j
        self.jo = near(REAL_SITE["xyz_mm"])
        self.same = p["predicted"] and np.linalg.norm(X[self.jp] - X[self.j]) < 3e-3
        self.x_on = d["travel_mm"] if d["predicted"] else real["travel_mm"]
        self.match = site_match(*site_of(d)) or (p["predicted"] and site_match(p["xyz_mm"], p["faces"]))
        self.ring = Circle((0, 0), 24, fill=False, ec=CRACK, lw=3.5, ls="-" if d["predicted"] else (0, (4, 3)),
                           visible=False, zorder=10, path_effects=_halo(3.5))
        self.pring = Circle((0, 0), 24, fill=False, ec=POT, lw=3.5, visible=False, zorder=10, path_effects=_halo(3.5))
        self.oring = Circle((0, 0), 38, fill=False, ec="black", lw=3.0, visible=False, zorder=9, path_effects=_halo(3))
        for r in (self.oring, self.pring, self.ring):
            ax.add_patch(r)
        self.x, = ax.plot([], [], "x", ms=16, mew=3.5, color=CRACK, zorder=11)
        rs = d["at_real_crack"]
        self.lab = "predicted crack" if d["predicted"] else f"most stressed: {rs:.0f} MPa ({rs / uts:.0%} of strength)"
        box = lambda ec: dict(boxstyle="round,pad=0.25", fc="white", ec=ec, alpha=0.9)
        self.label = ax.annotate(self.lab, (0, 0), xytext=(26, 26), textcoords="offset points", fontsize=15,
                                 weight="bold", color=CRACK, visible=False, zorder=12, bbox=box(CRACK))
        self.plabel = ax.annotate("potential crack", (0, 0), xytext=(26, 26), textcoords="offset points", fontsize=15,
                                  weight="bold", color=POT, visible=False, zorder=12, bbox=box(POT))
        olab = "real crack here (observed):\nlower inner bore" if self.match else "real crack (observed):\nlower inner bore"
        self.olabel = ax.annotate(olab, (0, 0), xytext=(-34, -34), textcoords="offset points", fontsize=13,
                                  weight="bold", color="black", ha="right", va="top", visible=False, zorder=12,
                                  bbox=box("black"))
        b, gb = ax.get_position(), graph.ax.get_position()
        self.one_line = gb.x0 > b.x1 - 0.01          # graph beside the part (single video): status above it
        self.narrow = b.width < 0.4                   # three columns: smaller status in three lines beside the part
        kw = dict(fontsize=12 if self.narrow else 15, weight="bold", ha="left", va="top", zorder=12, linespacing=1.3)
        if self.one_line:
            self.status = fig.text(gb.x0 + gb.width / 2, gb.y1 + 0.03, "", fontsize=16, weight="bold",
                                   color=CRACK, ha="center", va="bottom", zorder=12)
            self.pstatus = graph.ax.text(0.03, 0.80, "", transform=graph.ax.transAxes, color=POT,
                                         **{**kw, "fontsize": 13})
            self.ostatus = graph.ax.text(0.03, 0.69, "", transform=graph.ax.transAxes, color="black",
                                         **{**kw, "fontsize": 13})
        else:                                         # graph below: the empty upper left of the part panel
            x0, y1 = b.x0 + 0.008, b.y1 - 0.03
            self.status = fig.text(x0, y1, "", color=CRACK, **kw)
            self.pstatus = fig.text(x0, y1 - 0.105, "", color=POT, **kw)
            self.ostatus = fig.text(x0, y1 - 0.18, "", color="black", **{**kw, "fontsize": 13})
        gkw = dict(textcoords="offset points", fontsize=12, weight="bold", visible=False)
        self.gx, = graph.ax.plot([], [], "X", ms=15, color=CRACK, mec="white", mew=1.2, zorder=9)
        self.gl = graph.ax.annotate("predicted crack", (0, 0), xytext=(10, -24), color=CRACK, **gkw)
        self.gp, = graph.ax.plot([], [], "o", ms=12, color=POT, mec="white", mew=1.2, zorder=9)
        self.gpl = graph.ax.annotate("potential crack", (0, 0), xytext=(-12, 38), ha="right", color=POT,
                                     arrowprops=dict(arrowstyle="-", color=POT, lw=1.2), **gkw)
        if p["predicted"] and abs(p["travel_mm"] - real["travel_mm"]) < 0.25:     # next to the real crack's label
            self.gpl.set_position((0, -50))
            self.gpl.set_horizontalalignment("center")

    def _place(self, x, j):
        pts, _, _ = self.c.at(x, self.mag)
        return project(self.view, pts[j])

    def _keep_inside(self, ann):
        """Shift a ring's label left when its box would run past the right edge of the part panel (three
        columns: it would cross the divider)."""
        fig = self.ax.figure
        r = fig.canvas.get_renderer()
        ann.set_position((26, 26))
        over = ann.get_window_extent(r).x1 + 8 - self.ax.get_window_extent(r).x1       # + 8 px: the box's pad
        if over > 0:
            ann.set_position((26 - over * 72 / fig.dpi, 26))

    def update(self, x: float, phase: str):
        d, p, real = self.d, self.p, self.real
        on = x >= self.x_on - 1e-9
        pon = p["predicted"] and x >= p["travel_mm"] - 1e-9
        oon = x >= real["travel_mm"] - 1e-9
        red_ring = on and (d["predicted"] or not p["predicted"])      # no-break dashed ring only without amber
        amber_ring = pon and not (d["predicted"] and on and self.same)
        if red_ring:
            col, row, vis = self._place(x, self.j)
            self.ring.center = (col, row)
            self.ring.set_alpha(1.0 if vis else 0.45)
            self.x.set_data(([col], [row]) if d["predicted"] else ([], []))
            self.label.xy = (col, row)
            self.label.set_text(self.lab if vis else f"{self.lab}\n(on the hidden side)")
            self._keep_inside(self.label)
        else:
            self.x.set_data([], [])
        self.ring.set_visible(red_ring)
        self.label.set_visible(red_ring)
        if amber_ring:
            col, row, vis = self._place(x, self.jp)
            self.pring.center = (col, row)
            self.pring.set_alpha(1.0 if vis else 0.45)
            self.plabel.xy = (col, row)
            self.plabel.set_text("potential crack" if vis else "potential crack\n(on the hidden side)")
            self._keep_inside(self.plabel)
        self.pring.set_visible(amber_ring)
        self.plabel.set_visible(amber_ring)
        if oon:
            col, row, _ = self._place(x, self.jo)
            self.oring.center = (col, row)
            self.olabel.xy = (col, row)
        self.oring.set_visible(oon and SHOW_REAL)
        self.olabel.set_visible(oon and SHOW_REAL)
        # graph
        if d["predicted"]:
            self.gx.set_data([d["travel_mm"]], [d["force_N"] if on else np.nan])
            self.gl.xy = (d["travel_mm"], d["force_N"])
            self.gl.set_visible(on)
        if p["predicted"]:
            self.gp.set_data([p["travel_mm"]], [p["force_N"] if pon else np.nan])
            self.gpl.xy = (p["travel_mm"], p["force_N"])
            self.gpl.set_visible(pon)
        # status text
        sep = "\n" if self.narrow else " "
        if d["predicted"]:
            early = real["travel_mm"] - d["travel_mm"]
            when = f"{early:.2f} mm before the real crack" if early > 0.02 else "at the real crack"
            t = (f"predicted crack at {d['travel_mm']:.2f} mm, {d['force_N']:,.0f} N" if self.one_line else
                 f"predicted crack\nat {d['travel_mm']:.2f} mm,{sep}{d['force_N']:,.0f} N")
            if SHOW_REAL:
                t += f" ({when})" if self.one_line else f"\n({early:.2f} mm early)"
        else:
            t = (f"no break predicted (up to {d['end_travel_mm']:.2f} mm)" if self.one_line else
                 f"no break{sep}predicted\n(up to {d['end_travel_mm']:.2f} mm)")
        self.status.set_text(t if on else "")
        if p["predicted"]:
            pt = (f"potential crack (85 % of strength)\nat {p['travel_mm']:.2f} mm, {p['force_N']:,.0f} N" if self.one_line
                  else f"potential crack\nat {p['travel_mm']:.2f} mm,{sep}{p['force_N']:,.0f} N")
            self.pstatus.set_text(pt if pon else "")
        if self.one_line:
            ot = ("same spot as the real crack\n(observed: lower inner bore)" if self.match else
                  "the real crack (observed) is\nat the lower inner bore")
        else:
            ot = ("same spot as the real\ncrack (observed):\nlower inner bore" if self.match else
                  "wrong place: the real\ncrack (observed) is at\nthe lower inner bore")
        self.ostatus.set_text(ot if oon and SHOW_REAL else "")


def real_marker(graph, real: dict):
    ax = graph.ax
    ax.plot([real["travel_mm"]], [real["force_N"]], "D", ms=11, color="black", mec="white", mew=1.2, zorder=8)
    ax.annotate("real first crack (test)", (real["travel_mm"], real["force_N"]), xytext=(-14, 10),
                textcoords="offset points", ha="right", fontsize=12, weight="bold", color="black")


SHOW_REAL = False      # the videos show predictions only (the user shows the real crack separately); True brings back
                       # the black 'real crack (observed)' ring, the 'real first crack (test)' diamond and the comparisons


POT_NOTE = ("Amber: potential crack: tension at 85 % of the resin's dogbone strength (a margin for print defects and "
            "layer lines; the threshold was chosen after seeing this test).")
OBS_NOTE = ("Black ring: where the printed part cracked first, the lower inner bore, as observed by the team on the "
            "broken part (no photo; the ring is drawn at the simulation's most stressed bore point).")


def footnote(fig, res: dict, single: bool):
    """The criteria in small print: under the graph (single video), or in the free right edge of the right panel
    ."""
    import textwrap
    mat = res["material"]
    pe = res["cases"]["before"]["peeq_raw"]["travel_mm"], res["cases"]["after"]["peeq_raw"]["travel_mm"]
    when = f"{pe[1]:.1f} mm" if single else f"{pe[0]:.1f} mm (before) / {pe[1]:.1f} mm (after)"
    txt = (f"Red: the largest tension (max principal stress) reaches the resin's strength "
           f"({mat['uts_MPa']:.1f} MPa, dogbone test): a crack-onset criterion, no crack growth.  The dogbone's "
           f"strain at break (PEEQ {mat['failure_peeq']}) is passed at {when}, but only at the nut-pocket corners, "
           f"where the resin is squeezed from all sides.")
    g = dict(fontsize=10.5, color="#555555")
    if single:
        fig.text(0.775, 0.004, textwrap.fill(f"{txt}  {POT_NOTE}  {OBS_NOTE}", 100), ha="center", va="bottom", **g)
    else:
        fig.text(0.995, 0.43, textwrap.fill(f"{txt}  {POT_NOTE}  {OBS_NOTE}", 38), ha="right", va="bottom", **g)


def videos(run_dir: Path, mag: float = 1.0, only: list[str] | None = None) -> dict:
    """before_after_break.mp4, after_break.mp4, three_way_break.mp4 and first_vs_after_break.mp4: nl_story_video's
    before_after / after_only / three_way / first_vs_after videos with the crack markers, written to OUT (its own
    outputs are not touched). only: just these (names without '_break' or with it)."""
    res = json.loads((OUT / "break.json").read_text())
    cases, _ = load(run_dir)
    real, uts = res["real_crack"], res["material"]["uts_MPa"]
    orig_run, orig_out = sv.run_video, sv.OUT

    def wrapped(name, fig, panels, graphs, texts, mag_):
        for g in graphs:
            if SHOW_REAL:
                real_marker(g, real)
        marks = [Marks(fig, c, view, im, next(g for g in graphs if any(it[0] is c for it in g.items)), mag_,
                       res["cases"][c.key][VIDEO_CRIT], res["cases"][c.key][VIDEO_CRIT + "_85"], real, uts)
                 for c, view, im in panels]
        # no small print in the videos (the user's call: too much text for a slide); the criteria and
        # caveats are on the break page (index.html) and in break_table.md

        def texts2(x, phase):
            texts(x, phase)
            for m in marks:
                m.update(x, phase)
        return orig_run(name if name.endswith("_break") else f"{name}_break", fig, panels, graphs, texts2, mag_)

    sv.run_video, sv.OUT = wrapped, OUT
    try:
        fmax = sv.fmax_of(list(cases.values()), sv.cases_meas())
        jobs = {"before_after_break": lambda: sv.video_before_after(cases, mag, zoom=1.45),
                "after_break": lambda: sv.video_single(cases["after"], "after_break", AFTER_HEAD, mag, 1.5, fmax),
                "three_way_break": lambda: sv.video_three_way(cases, mag, zoom=1.35),         # zooms as nl_story_video
                "first_vs_after_break": lambda: sv.video_first_vs_after(cases, mag, zoom=1.3)}
        f = OUT / "videos.json"
        info = json.loads(f.read_text()) if f.exists() else {}
        for name, job in jobs.items():
            if not only or name in only or name.removesuffix("_break") in only:
                info[name] = job()
    finally:
        sv.run_video, sv.OUT = orig_run, orig_out
    f.write_text(json.dumps(info, indent=1))
    return info


AFTER_HEAD = "After post-training on the real test: most accurate mesh"   # as nl_story_video's after_only


# --------------------------------------------------------------------------- still close-up

POCKET = ("f38", "f39", "f41", "f42")       # the nut-pocket corner faces


def _state_at(c, files: list[Path], x: float):
    """Full-mesh grid with displacement and s1 (MPa) at crosshead travel x, interpolated between increments."""
    import pyvista as pv
    k = int(np.clip(np.searchsorted(c.xc, x), 1, len(c.xc) - 1))
    a = float(np.clip((x - c.xc[k - 1]) / max(c.xc[k] - c.xc[k - 1], 1e-12), 0, 1))
    g0, g1 = pv.read(files[k - 1]), pv.read(files[k])
    g = g1.copy()
    for f in ("displacement", "s1", "PEEQ"):
        g.point_data[f] = (1 - a) * np.asarray(g0.point_data[f]) + a * np.asarray(g1.point_data[f])
    g.point_data["s1"] = g.point_data["s1"] / 1e6
    return g, float(np.interp(x, c.xc, c.s.F))


def still(run_dir: Path):
    """after_crack_closeup.png/.pdf: the 'after' case at the real first crack, coloured by the largest tension
    s1: the whole part with the two most stressed sites ringed, and a close-up of each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pyvista as pv
    from matplotlib.patches import Circle
    res = json.loads((OUT / "break.json").read_text())
    cases, _ = load(run_dir)
    c = cases["after"]
    r = res["cases"]["after"]
    real, uts = res["real_crack"], res["material"]["uts_MPa"]
    sd = nv.resolve(c.s.s.get("series_dir"), c.s.dir) or c.s.dir / "series"
    files = sorted(Path(sd).glob("inc_*.vtu"))
    g, F = _state_at(c, files, real["travel_mm"])
    fd = Fields(c, run_dir.resolve().parents[1])
    s1 = np.asarray(g.point_data["s1"])
    near_pocket = np.zeros(len(s1), bool)
    for f in POCKET:
        if f in fd.faces:
            dist, _ = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(fd.faces[f]).query(fd.X)
            near_pocket |= dist < FACE_TOL_MM * 1e-3
    i_all = int(np.argmax(s1))
    i_pk = int(np.flatnonzero(near_pocket)[np.argmax(s1[near_pocket])]) if near_pocket.any() else i_all
    sites = [("A", i_all), ("B", i_pk)] if i_pk != i_all else [("A", i_all)]

    surf = g.linear_copy().extract_surface(algorithm="dataset_surface") if "algorithm" in \
        pv.UnstructuredGrid.extract_surface.__code__.co_varnames else g.linear_copy().extract_surface()
    surf = surf.warp_by_vector("displacement", factor=1.0).compute_normals(cell_normals=False, auto_orient_normals=True)
    ids = np.asarray(surf["vtkOriginalPointIds"])
    cmap = mpl.colormaps["YlOrRd"].with_extremes(under=nv.ELASTIC, over="#4a0010")
    clim = (1.0, uts)

    def render(size, camera, rings=()):
        pl = pv.Plotter(off_screen=True, window_size=list(size))
        pl.set_background("white")
        pl.add_mesh(surf, scalars="s1", cmap=cmap, clim=clim, below_color=nv.ELASTIC, above_color="#4a0010",
                    show_scalar_bar=False)
        pl.camera_position = camera
        img = pl.screenshot(return_img=True)
        pts = []
        for _, i in rings:
            j = int(np.argmin(np.linalg.norm(surf.points - (g.points[i] + g.point_data["displacement"][i]), axis=1)))
            pts.append((j, project(type("V", (), {"p": pl})(), surf.points[j])))
        pl.close()
        return img, pts

    def closeup(i):
        x = g.points[i] + g.point_data["displacement"][i]
        j = int(np.argmin(np.linalg.norm(surf.points - x, axis=1)))
        n = np.asarray(surf.point_data["Normals"][j])
        d = n + np.array([0.0, 0.25, 1.1 if x[2] >= -0.004 else -1.1])
        d /= np.linalg.norm(d)
        return [tuple(x + 0.045 * d), tuple(x), (0.0, 1.0, 0.0)]

    from ace.nonlinear import NL_CAMERA
    fig = plt.figure(figsize=(16, 7.6), dpi=120, facecolor="white")
    fig.text(0.012, 0.965, f"After post-training (most accurate mesh, {c.nodes:,} nodes) at the real first crack: "
             f"{real['travel_mm']:.2f} mm crosshead, simulated {F:,.0f} N (test {real['force_N']:,.0f} N)",
             fontsize=15, weight="bold", va="center")
    fig.text(0.012, 0.893, f"Colour: largest tension s1 (MPa, true scale deformation); grey: no tension; dark red: over the "
             f"resin's strength {uts:.1f} MPa.\nStress criterion: no break predicted; A is at {s1[i_all] / uts:.0%} of the "
             f"strength, just over the {POTENTIAL:.0%} 'potential crack' tier ({POTENTIAL * uts:.1f} MPa).\nA is where the "
             f"printed part cracked first (observed by the team: the lower inner bore; no photo in the repo).",
             fontsize=11, color="#444444", va="center", linespacing=1.4)
    ax = fig.add_axes([0.0, 0.08, 0.40, 0.80])
    ax.set_axis_off()
    img, pts = render((760, 760), NL_CAMERA, sites)
    ax.imshow(img)
    ax.set_autoscale_on(False)
    obs = lambda i: site_match((fd.X[i] * 1e3).tolist(), fd.faces_near(fd.X[i]))
    for (tag, i), (_, (col, row, vis)) in zip(sites, pts):
        ec = "black" if obs(i) else CRACK
        ax.add_patch(Circle((col, row), 28, fill=False, ec=ec, lw=3, ls="-" if vis else (0, (4, 3))))
        ax.annotate(f"{tag}: real crack (observed)" if obs(i) else tag, (col, row), xytext=(22, 22),
                    textcoords="offset points", fontsize=16 if not obs(i) else 13, weight="bold", color=ec,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ec, alpha=0.85) if obs(i) else None)
    for k, (tag, i) in enumerate(sites):
        a2 = fig.add_axes([0.41 + k * 0.295, 0.13, 0.285, 0.70])
        a2.set_axis_off()
        im2, p2 = render((640, 640), closeup(i), [(tag, i)])
        a2.imshow(im2)
        a2.set_autoscale_on(False)
        col, row, _ = p2[0][1]
        a2.add_patch(Circle((col, row), 34, fill=False, ec="black" if obs(i) else CRACK, lw=3))
        fs = fd.faces_near(fd.X[i])
        head = (f"{tag}: real crack site (observed): {site_name((fd.X[i] * 1e3).tolist(), fs)}" if obs(i) else
                f"{tag}: {site_name((fd.X[i] * 1e3).tolist(), fs)} (no crack observed here)")
        a2.set_title(f"{head}\ns1 = {s1[i]:.1f} MPa ({s1[i] / uts:.0%} of strength), {faces_txt(fs)}",
                     fontsize=11.5, color="#222222", weight="bold" if obs(i) else "normal")
        a2.text(0.5, -0.03, f"xyz = ({', '.join(f'{v:.1f}' for v in fd.X[i] * 1e3)}) mm (undeformed)",
                transform=a2.transAxes, ha="center", va="top", fontsize=10, color="#555555")
    cax = fig.add_axes([0.06, 0.07, 0.28, 0.025])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*clim), cmap=cmap), cax=cax,
                      orientation="horizontal", extend="both")
    cb.set_label("largest tension s1 (MPa)", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"after_crack_closeup.{ext}", dpi=120)
    plt.close(fig)
    info = {"travel_mm": real["travel_mm"], "force_sim_N": F,
            "sites": {tag: {"s1_MPa": float(s1[i]), "xyz_mm": (fd.X[i] * 1e3).round(2).tolist(),
                            "faces": fd.faces_near(fd.X[i]), "observed_crack_site": bool(obs(i))} for tag, i in sites}}
    (OUT / "closeup.json").write_text(json.dumps(info, indent=1))
    print("closeup", json.dumps(info))


# --------------------------------------------------------------------------- page

def page():
    """index.html: the videos, the close-up, the table and the caveats."""
    import html as h
    res = json.loads((OUT / "break.json").read_text())
    vids = json.loads((OUT / "videos.json").read_text()) if (OUT / "videos.json").exists() else {}
    cu = json.loads((OUT / "closeup.json").read_text()) if (OUT / "closeup.json").exists() else {}
    real, mat, cs = res["real_crack"], res["material"], res["cases"]
    b, a = cs["before"], cs["after"]
    pot = POTENTIAL * mat["uts_MPa"]
    cols = [("peeq_raw", "PEEQ &ge; %.3g, raw" % mat["failure_peeq"]), ("peeq_robust", "PEEQ, robust"),
            ("s1_raw", "predicted crack: s1 &ge; %.1f MPa, raw" % mat["uts_MPa"]), ("s1_robust", "s1 &ge; 100 %, robust"),
            ("s1_raw_85", "potential crack: s1 &ge; %.1f MPa (85 %%), raw" % pot), ("s1_robust_85", "s1 &ge; 85 %, robust")]

    def td(d):
        cls = "pot" if d["threshold"] < mat["uts_MPa"] - 1e-6 else "hit"
        xyz, fs = site_of(d)
        where = f'{h.escape(site_name(xyz, fs))}: {h.escape(" / ".join(fs))}'
        if d["predicted"]:
            return (f'<td class="{cls}">{d["travel_mm"]:.2f} mm, {d["force_N"]:,.0f} N<br><small>{where}</small></td>')
        unit = " MPa" if d["criterion"].startswith("s1") else ""
        return (f'<td>none by {d["end_travel_mm"]:.2f} mm<br><small>at the real crack: {d["at_real_crack"]:.3g}{unit}, '
                f'{where}</small></td>')

    def sens_td(d):
        t = f"{d['travel_mm']:.2f} mm" if d["predicted"] else "none"
        return f"<td>{t} <small>[{d['at_real_crack']:.3g}]</small></td>"

    def sens85(r):
        d = r.get("s1_raw_85", {})
        if not d.get("predicted"):
            return "<td>none</td>"
        return f"<td>{d['travel_mm']:.2f} mm <small>({h.escape(site_name(d['xyz_mm'], d['faces']))})</small></td>"
    rows = "".join(f"<tr><th>{k}<br><small>{h.escape(cs[k]['label'])}, {cs[k]['nodes']:,} nodes, "
                   f"{sv.pct(cs[k]['mean_rel_err'])} off</small></th>" + "".join(td(cs[k][c]) for c, _ in cols)
                   + f"<td>{h.escape(match_txt(cs[k]))}</td></tr>" for k in ORDER)
    rows += (f'<tr class="real"><th>real test</th><td colspan="{len(cols)}">first load drop at the peak: '
             f'{real["force_N"]:,.0f} N at {real["travel_mm"]:.2f} mm (machine report: {real["report_force_N"] / 1e3:.2f} kN '
             f'at {real["report_travel_mm"]:.2f} mm)</td><td>lower inner bore, f5 (observed)</td></tr>')
    sens = "".join(f"<tr><th>{h.escape(r['label'])}</th><td>{r['nodes']:,}</td><td>{sv.pct(r['mean_rel_err'])}</td>"
                   + "".join(sens_td(r[c]) for c, _ in cols[:4]) + sens85(r) + "</tr>"
                   for r in res.get("mesh_sensitivity", []))
    sa, sb = cu.get("sites", {}).get("A", {}), cu.get("sites", {}).get("B", {})
    ps = a["peeq_robust"].get("state_at_site", {})
    ap, bp = a["s1_raw_85"], b["s1_raw_85"]
    vid = lambda n, cap: (f'<section><video src="{n}.mp4" controls loop muted autoplay playsinline></video>'
                          f'<p class="cap">{cap}</p><p class="meta">{n}.mp4'
                          + (f' &middot; {vids[n]["seconds"]:.1f} s &middot; {vids[n]["mb"]:.1f} MB' if n in vids else "")
                          + "</p></section>")
    potnote = (f"Potential crack: tension at {POTENTIAL:.0%} of the resin's dogbone strength ({pot:.1f} MPa; a margin for "
               f"print defects and layer lines; the threshold was chosen after seeing this test).")
    obsnote = ("The real crack site (lower inner bore, face f5) is the team's observation of the broken part; there is no "
               "photo in the repo, and which side and the exact point were not recorded: the black ring is drawn at the "
               "simulation's most stressed bore point, so the match is a region match, not a point match.")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicted Crack Onset</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1f23; --muted: #5b6270; --line: #e3e6ea; --red: #c62828; --amber: #b86200; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #16181c; --fg: #e9ebee; --muted: #a2a9b4;
  --line: #2c3036; --red: #ff6b6b; --amber: #ffb04a; }} }}
:root[data-theme="dark"] {{ --bg: #16181c; --fg: #e9ebee; --muted: #a2a9b4; --line: #2c3036; --red: #ff6b6b; --amber: #ffb04a; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; margin: 0 auto;
       max-width: 1200px; padding: 24px 16px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; }} h2 {{ font-size: 20px; margin: 40px 0 8px; }}
.lead, .meta, .cap {{ color: var(--muted); }} .meta {{ font-size: 13px; }} .cap small, .fine {{ font-size: 13px; }}
video, img {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
small {{ color: var(--muted); }} td.hit {{ color: var(--red); font-weight: 600; }}
td.pot {{ color: var(--amber); font-weight: 600; }} tr.real {{ font-weight: 600; }}
ul {{ padding-left: 20px; }}
</style></head><body>
<h1>Where and when does the part crack?</h1>
<p class="lead">Part abc_00000065 crushed in an Instron. Real first crack: the first load drop, at the peak,
{real["force_N"]:,.0f} N at {real["travel_mm"]:.2f} mm crosshead, at the <b>lower inner bore</b> (observed by the team on the
broken part). The simulation has no fracture model, so this is a crack-<em>onset</em> criterion checked on each CalculiX
increment: the largest tension s1 reaching the resin's strength ({mat["uts_MPa"]:.1f} MPa: predicted crack) or
{POTENTIAL:.0%} of it ({pot:.1f} MPa: potential crack), or the plastic strain PEEQ reaching the dogbone's strain at break
({mat["failure_peeq"]}).</p>
<h2>Answer</h2>
<ul>
<li><b>After post-training (most accurate mesh): no break predicted</b> by the full-strength criterion up to
{a["s1_raw"]["end_travel_mm"]:.2f} mm, but a <b>potential crack at {ap["travel_mm"]:.2f} mm, {ap["force_N"]:,.0f} N</b>
(real crack: {real["travel_mm"]:.2f} mm, {real["force_N"]:,.0f} N), on the {h.escape(site_name(ap["xyz_mm"], ap["faces"]))}
({h.escape(" / ".join(ap["faces"]))}): <b>the place where the part actually cracked</b>. There the tension is
{a["s1_raw"]["at_real_crack"]:.0f} MPa at the real crack, {a["s1_raw"]["at_real_crack"] / mat["uts_MPa"]:.0%} of the
strength.</li>
<li><b>Before post-training: crack predicted at {b["s1_raw"]["travel_mm"]:.2f} mm, {b["s1_raw"]["force_N"]:,.0f} N</b>
(potential crack already at {bp["travel_mm"]:.2f} mm), {real["travel_mm"] - b["s1_raw"]["travel_mm"]:.2f} mm and
{real["force_N"] - b["s1_raw"]["force_N"]:,.0f} N before the real crack, and <b>in the wrong place</b>: on the upper inner
edge of a prong ({h.escape(faces_txt(b["s1_raw"]["faces"]))}), where the frictionless platens let the prong tops slide
inward (about 6 mm each by the end) and bend the prongs. The real crack is at the lower inner bore.</li>
<li>The strain criterion fires far too early in every case (1.0-1.6 mm raw), always at the nut-pocket corners
(f8 with f38/f39 or f41/f42). There the resin is squeezed from all sides (after: s1 = {ps.get("s1_MPa", float("nan")):.0f} MPa,
s3 = {ps.get("s3_MPa", float("nan")):.0f} MPa when PEEQ passes {mat["failure_peeq"]}), so a tensile break strain does not
apply; and PEEQ at a sharp corner keeps growing as the mesh is refined.</li>
</ul>
<h2>Before vs after post-training</h2>
{vid("before_after_break", "Red ring and X: where and when the largest tension reaches the resin's strength (predicted crack). "
     "Amber ring: where it first reaches 85 % of it (potential crack). Black ring: where the real part cracked first "
     "(observed: lower inner bore). Black diamond on the graphs: the real first crack (test peak). Part colour: plastic "
     "strain PEEQ, as in the original video. <small>" + potnote + " " + obsnote + "</small>")}
<h2>After post-training</h2>
{vid("after_break", "The most accurate mesh: it follows the test to 2 % and stays below the full strength everywhere; it "
     "reaches the potential-crack tier at the lower inner bore just as the real part cracks there. <small>" + potnote
     + "</small>")}
<h2>First simulation, before and after post-training</h2>
{vid("three_way_break", "Red ring and X: where and when the largest tension reaches the resin's strength (predicted crack); "
     "amber ring: where it first reaches 85 % of it (potential crack).<br>The teammates' first nonlinear simulation | "
     "before | after post-training, true scale; part colour: plastic strain PEEQ.")}
<h2>First simulation vs after post-training</h2>
{vid("first_vs_after_break", "Red ring and X: where and when the largest tension reaches the resin's strength (predicted "
     "crack); amber ring: where it first reaches 85 % of it (potential crack).<br>The teammates' first nonlinear "
     "simulation (frictionless platens) vs after post-training, true scale; part colour: plastic strain PEEQ.")}
<h2>Close-up at the real first crack (after post-training)</h2>
<img src="after_crack_closeup.png" alt="after post-training, largest tension at the real first crack: the lower inner bore (real crack site) and the nut pocket">
<p class="cap">Largest tension s1 at {real["travel_mm"]:.2f} mm. A: the most stressed point,
{sa.get("s1_MPa", float("nan")):.1f} MPa on {h.escape(faces_txt(sa.get("faces", [])))}: where the printed part cracked
first (observed). B: the most stressed point near the nut-pocket corners, {sb.get("s1_MPa", float("nan")):.1f} MPa (no crack
observed there). For comparison with a photo of the broken part. PDF:
<a href="after_crack_closeup.pdf">after_crack_closeup.pdf</a>. <small>{obsnote}</small></p>
<h2>Predicted vs real, all cases</h2>
<div class="tw"><table><tr><th>case</th>{"".join(f"<th>{t}</th>" for _, t in cols)}
<th>site matches the real crack (observed: lower inner bore)?</th></tr>{rows}</table></div>
<p class="cap">Travel uses each case's own crosshead mapping (before: crosshead = part shortening). Raw: the single highest
node; robust: reached by at least {ROBUST_K} nodes within {ROBUST_R_MM:g} mm (about twice the node spacing, so a single
spiky node does not count). Red: a break is predicted; amber: a potential crack. A site matches when it lies on the bore
(f5) below its centre. <small>{potnote}</small>
Also as <a href="break_table.png">PNG</a>, <a href="break_table.pdf">PDF</a>, <a href="break_table.md">Markdown</a>;
data in <a href="break.json">break.json</a>.</p>
<h2>How much the mesh moves it</h2>
<div class="tw"><table><tr><th>mesh (learned conditions)</th><th>nodes</th><th>force error</th>
{"".join(f"<th>{t}</th>" for _, t in cols[:4])}<th>potential crack (85 %), raw</th></tr>{sens}</table></div>
<p class="cap">Onset travel by each criterion; in brackets the value at the real crack travel (PEEQ, or s1 in MPa).
At the real crack every mesh puts the largest tension at 42-49 MPa (83-98 % of the strength); only the pre-trained swarm
mesh (2.58 mm) and the uniform 2 mm mesh (2.71 mm) cross it, just after the real crack. Where the 85 % tier is first
reached moves with the mesh: the lower inner bore on the post-trained, 4 mm and 2.5 mm meshes, a nut-pocket corner on the
swarm and 2 mm meshes, the outer side on the coarse default mesh.</p>
<h2>Caveats</h2>
<ul>
<li>Onset only: a criterion read off a continuum simulation; no crack, no crack growth, no load drop is modelled.</li>
<li>The {POTENTIAL:.0%} tier was chosen after seeing this test, so its timing match is not an independent prediction.</li>
<li>One dogbone gives both thresholds; its strain is from the crosshead (includes grip compliance), and the 50.5 MPa
is an engineering stress while s1 is a true (Cauchy) stress: the true strength is a few % higher, which would push
the stress-criterion onset later still.</li>
<li>Nodal s1 and PEEQ are recovered from integration points; at sharp re-entrant corners both depend on the mesh
(PEEQ strongly), which is why the robust values and the mesh table are given.</li>
<li>{obsnote}</li>
<li>The printed part may break earlier than the bulk resin (layer lines, voids, the sharp pocket corners); the test
cracked at {real["force_N"]:,.0f} N where the best simulation is at {a["s1_raw"]["at_real_crack"] / mat["uts_MPa"]:.0%} of the
strength, which is within that kind of scatter.</li>
</ul>
</body></html>
"""
    (OUT / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["analyze", "videos", "still", "page", "all"])
    ap.add_argument("--run", type=Path, default=Path.home() / "feagent" / "ace_runs" / "nl_ft")
    ap.add_argument("--mag", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", help="videos: just these (before_after after three_way first_vs_after)")
    a = ap.parse_args()
    run = a.run.expanduser()
    if a.cmd in ("analyze", "all"):
        analyze(run)
    if a.cmd in ("videos", "all"):
        videos(run, a.mag, a.only)
    if a.cmd in ("still", "all"):
        still(run)
    if a.cmd in ("page", "all"):
        page()


if __name__ == "__main__":
    main()
