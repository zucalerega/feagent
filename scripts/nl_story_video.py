#!/usr/bin/env python3
"""Act 3 story videos: the simulation against the real compression test, before and after post-training.

Each video shows the part deforming through the nonlinear CalculiX ramp (coloured by equivalent plastic
strain PEEQ, fixed camera and colour range, scripts/nl_video.py) next to its force vs crosshead-travel
curve filling in over the real Instron curve (black).

Cases (all: part abc_00000065_compression, J2 resin, NLGEOM):
  first    the teammates' first nonlinear simulation (scripts/ccx_compression_ramp.py, run team_slide): the
           problem's reference mesh at h = 2 mm, base fixed, frictionless platen to 3 mm, 30 adaptive
           increments, crosshead = part shortening; re-solved here with its increments by `first`
  before   what we started with: frictionless platens, automesher mesh, no seating / load-train correction
           (crosshead = part shortening); solved here by `before`
  stage2   learned test conditions (the frozen spec of ace.nonlinear) on the automesher mesh (baselines/auto)
  stage3   learned conditions on the pre-trained swarm mesh (baselines/swarm_bc0)
  after    learned conditions on the fine-tuning run's best mesh (highest base score over every
           generation, see best_scoring)

    python scripts/nl_story_video.py before                    # solve the 'before' case (needs ccx)
    python scripts/nl_story_video.py first                     # solve the 'first' case (needs ccx)
    python scripts/nl_story_video.py videos --run ~/feagent/ace_runs/nl_ft [--mag 1]
                       # before_after.mp4, four_stages.mp4, before_only.mp4, after_only.mp4, three_way.mp4,
                       # first_vs_after.mp4, key frames, videos.html

Outputs go to talk_plots/act3_real_data/videos/ (and videos.html next to it).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nl_video as nv  # noqa: E402  (Video, Series, PartView, ... are reused from there)

OUT = ROOT / "talk_plots" / "act3_real_data" / "videos"
BEFORE_DIR = OUT / "before_run"            # run_curve directory of the 'before' case (series/ only on the box)
AUTO_MESH = nv.BASE / "auto" / "mesh.msh"   # the automesher mesh, shared by 'before' and 'stage2'
BEFORE_U_MAX_MM = 2.6                      # part shortening = crosshead travel: past the last checkpoint (2.5)
BEFORE_INCREMENTS = 26                     # 0.1 mm steps
FIRST_DIR = OUT / "first_run"              # run_curve directory of the 'first' case (series/, mesh only on the box)
# the committed team_slide run (summary.json: 41,186 nodes) was meshed at h = 2 mm, not the script's default 3 mm
# (16,405 nodes); its other settings are the ramp script's defaults
FIRST_H_MM = 2.0                           # element size of the problem's reference mesh
FIRST_U_MAX_MM, FIRST_INCREMENTS = 3.0, 30  # platen travel and increments

X_END = 2.55          # the animation drives the crosshead from 0 to here (mm); the test peaks at 2.51 mm
X_LIM = 2.75          # x axis limit of the graphs
LIVE_FROM = 0.3       # the live error is shown from this crosshead travel on (near 0 the test force is ~0)
KEY_MID_MM = 2.0      # the "middle" key frame: well past where the before curve crosses the test (~1.3 mm)
RAMP_S, HOLD_START_S, HOLD_END_S = 7.0, 0.5, 2.5

COLORS = {"first": "#b8860b", "before": "#c0392b", "stage2": "#d9822b", "stage3": "#7b5ea7", "after": "#1f6fb5"}
LABELS = {"first": "first nonlinear simulation",
          "before": "what we started with", "stage2": "learned test conditions, default mesh",
          "stage3": "learned conditions, pre-trained mesh", "after": "trained on the real test: most accurate mesh"}
BEFORE_SETUP = "frictionless platens, no seating correction"   # stated next to every 'before' label
FIRST_SETUP = "teammates' setup: frictionless platens"         # stated next to every 'first' label


# --------------------------------------------------------------------------- the 'before' case

def frictionless_bcs(u_m: float) -> list:
    """The starting setup: platens that only push in y (the faces slide freely), base held in y only, and
    two point constraints on the base bottom to remove the rigid-body motion (as the hand runs)."""
    from feacore.case import BC
    y0 = -0.043956                                   # y of the base bottom face f33 (m)

    def slide(group, v):
        return BC(type="displacement", group=group, components=[1], value=v)

    return [slide("f33", 0.0),
            BC(type="displacement", point=[0.0, y0, 0.0], components=[0, 2], value=0.0),
            BC(type="displacement", point=[0.02, y0, 0.0], components=[2], value=0.0),
            slide("f17", -u_m), slide("f34", -u_m)]


@contextlib.contextmanager
def before_setup():
    """Within this block ace.nonlinear.run_curve solves the 'before' case: its frozen case with the
    frictionless BCs, ramped to BEFORE_U_MAX_MM, and no crosshead correction (delta = C = 0, so
    crosshead = part shortening). ace/nonlinear.py itself is not changed; the patch is undone after."""
    from ace import nonlinear as nl
    s = copy.deepcopy(nl.spec())
    s.update(u_part_max_mm=BEFORE_U_MAX_MM, increments=BEFORE_INCREMENTS, delta_mm=0.0, compliance_mm_per_N=0.0)
    orig_spec, orig_build = nl.spec, nl.build_case

    def build(problem, mesh, adaptive=False):
        case = orig_build(problem, mesh, adaptive)            # the frozen case (spec patched below) ...
        case.bcs = frictionless_bcs(BEFORE_U_MAX_MM * 1e-3)   # ... with the starting BCs
        # only the platen reactions are needed; with these BCs CalculiX's .dat has no energy total for
        # the official strain-energy QoI, which would stop write_series
        case.qois = [q for q in case.qois if q.name.startswith("Ry_")]
        return case

    nl.measured()                    # cache the test data with the real spec first
    nl.spec, nl.build_case = (lambda: s), build
    try:
        yield
    finally:
        nl.spec, nl.build_case = orig_spec, orig_build


def solve_before(threads: int = 4) -> dict:
    from ace import nonlinear as nl
    with before_setup():
        out = nl.run_curve(nv.PID, {"path": str(AUTO_MESH), "order": 2}, BEFORE_DIR, threads=threads)
    if not out["ok"]:
        raise SystemExit(f"the 'before' run failed: {out['error']}")
    print(f"before: {out['nodes']} nodes, mean error {out['mean_rel_err']:.2%}, converged {out['converged']}, "
          f"reached {out['u_cross_mm'][-1]} mm, {out['wall_s']} s")
    return out


# --------------------------------------------------------------------------- the 'first' case

@contextlib.contextmanager
def first_setup():
    """Within this block ace.nonlinear.run_curve solves the teammates' first nonlinear simulation exactly as
    scripts/ccx_compression_ramp.py set it up (run team_slide: its ramp_case with the team material and the
    frictionless platen), ramped to FIRST_U_MAX_MM, with no crosshead correction (crosshead = part
    shortening, as the teammates compared it). ace/nonlinear.py itself is not changed; the patch is undone after."""
    from ace import nonlinear as nl
    from feacore.case import QoI
    import ccx_compression_ramp as ramp
    s = copy.deepcopy(nl.spec())
    s.update(u_part_max_mm=FIRST_U_MAX_MM, increments=FIRST_INCREMENTS, delta_mm=0.0, compliance_mm_per_N=0.0)
    orig_spec, orig_build = nl.spec, nl.build_case

    def build(problem, mesh, adaptive=False):
        # ramp_case is adaptive already (as the team run): run_curve's first, fixed-increment try is the same run
        case = ramp.ramp_case(Path(mesh["path"]), ramp.team_material(), "slide", FIRST_U_MAX_MM, FIRST_INCREMENTS)
        # run_curve reads the platen force as the reactions of f17 and f34 (together the group prong_tops);
        # the problem's other QoIs are not needed (and the strain energy one could stop write_series)
        case.qois = [QoI(name=f"Ry_{g}", type="reaction", group=g, component=1) for g in ("f17", "f34")]
        return case

    nl.measured()                    # cache the test data with the real spec first
    nl.spec, nl.build_case = (lambda: s), build
    try:
        yield
    finally:
        nl.spec, nl.build_case = orig_spec, orig_build


def solve_first(threads: int = 8) -> dict:
    from ace import nonlinear as nl
    from feacore.problems.registry import get
    FIRST_DIR.mkdir(parents=True, exist_ok=True)
    mesh = FIRST_DIR / f"mesh_h{FIRST_H_MM:g}.msh"        # the problem's reference mesh, as the ramp script
    if not mesh.exists():
        get(nv.PID).mesh(mesh, FIRST_H_MM / 1000.0, geom_order=2)
    with first_setup():
        out = nl.run_curve(nv.PID, {"path": str(mesh), "order": 2}, FIRST_DIR, threads=threads)
    if not out["ok"]:
        raise SystemExit(f"the 'first' run failed: {out['error']}")
    print(f"first: {out['nodes']} nodes, mean error {out['mean_rel_err']:.2%}, converged {out['converged']}, "
          f"reached {out['u_cross_mm'][-1]} mm, {out['wall_s']} s")
    return out


def check_first() -> dict:
    """The re-run against the committed curve of the teammates' run (curve_team_slide.csv): force difference
    over the shared travel range, relative to the committed force (from 0.1 mm on, where it is not ~0)."""
    import ccx_compression_ramp as ramp
    s = json.loads((FIRST_DIR / "summary.json").read_text())
    u0, F0 = np.loadtxt(ramp.OUT_DIR / "curve_team_slide.csv", delimiter=",", skiprows=2, usecols=(0, 1)).T
    u, F = np.asarray(s["u_part_mm"]), np.asarray(s["force_N"])
    k = (u0 >= 0.1) & (u0 <= min(u[-1], u0[-1]))
    rel = np.abs(np.interp(u0[k], u, F) - F0[k]) / F0[k]
    out = {"points": int(k.sum()), "max_rel_diff": float(rel.max()), "mean_rel_diff": float(rel.mean()),
           "F_max_N": float(F.max()), "F_max_committed_N": float(F0.max())}
    print("first vs curve_team_slide.csv:", json.dumps(out))
    return out


# --------------------------------------------------------------------------- picking the cases

def best_scoring(run_dir: Path) -> dict:
    """The fine-tuning run's record with the highest base score (accuracy vs the test, node-capped) among every
    generation's scored runs: what the training optimised. Re-run after the run ends."""
    recs = [r for rs in nv.run_records(run_dir).values() for r in rs]
    if not recs:
        raise SystemExit(f"no scored runs under {run_dir}")
    return max(recs, key=lambda r: r["checks"].get("base_score") or 0.0)


def most_accurate(run_dir: Path) -> dict:
    """The fine-tuning run's record with the lowest force error against the test (checks.nonlinear.mean_rel_err)
    among every generation's scored runs: the user's choice for the 'after' case. Ties go to the earliest."""
    recs = [r for rs in nv.run_records(run_dir).values() for r in rs]
    if not recs:
        raise SystemExit(f"no scored runs under {run_dir}")
    return min(recs, key=lambda r: (r["checks"]["nonlinear"]["mean_rel_err"], r["generation"]))


PICKS = {"accurate": most_accurate, "score": best_scoring}   # how the 'after' mesh is chosen (--pick)


class Case:
    """One case of the story: its Series (curve + increments) plus how it is labelled and coloured."""

    def __init__(self, key: str, series: nv.Series, note: str = ""):
        self.key, self.s, self.note = key, series, note
        self.color, self.label = COLORS[key], LABELS[key]
        self.err = float(series.s["mean_rel_err"])
        self.nodes = int(series.s["nodes"])
        self.sub = f"{self.nodes:,} nodes"                    # line under the label
        self.xc = np.maximum.accumulate(series.u_cross)      # crosshead travel, kept non-decreasing for interp

    def at(self, x: float, mag: float):
        """Surface points (displacement x mag), PEEQ and force at crosshead travel x (mm), interpolated
        linearly between increments; past the last increment the run stays there."""
        u_part = float(np.interp(x, self.xc, self.s.u_part))
        pts, P, _ = self.s.state(u_part)
        return self.s.X + mag * (pts - self.s.X), P, float(np.interp(x, self.xc, self.s.F))

    def filled(self, x: float):
        """The part of the curve drawn by crosshead travel x."""
        k = self.xc < x
        return np.r_[self.xc[k], min(x, self.xc[-1])], np.r_[self.s.F[k], np.interp(x, self.xc, self.s.F)]


def load_cases(run_dir: Path, pick: str = "accurate") -> tuple[dict[str, Case], dict]:
    rec = PICKS[pick](run_dir)
    nl = rec["checks"]["nonlinear"]
    checkout = run_dir.resolve().parents[1]                  # series_dir is relative to the run's checkout
    after_dir = (checkout / nl["series_dir"]).parent
    cases = {"first": Case("first", nv.Series(FIRST_DIR, LABELS["first"])),
             "before": Case("before", nv.Series(BEFORE_DIR, LABELS["before"])),
             "stage2": Case("stage2", nv.Series(nv.BASE / "auto", LABELS["stage2"])),
             "stage3": Case("stage3", nv.Series(nv.BASE / "swarm_bc0", LABELS["stage3"])),
             "after": Case("after", nv.Series(after_dir, LABELS["after"], summary=nl))}
    cases["after"].sub = f"generation {rec['generation']}, {cases['after'].nodes:,} nodes"
    cases["before"].sub = f"{BEFORE_SETUP}, default mesh, {cases['before'].nodes:,} nodes"
    cases["first"].sub = f"{FIRST_SETUP}, {cases['first'].nodes:,}-node mesh"
    pick = {"run_id": rec["run_id"], "generation": rec["generation"], "agent": rec["_agent"],
            "base_score": rec["checks"]["base_score"], "score": rec["checks"]["score"],
            "mean_rel_err": nl["mean_rel_err"], "nodes": nl["nodes"], "dir": str(after_dir)}
    return cases, pick


# --------------------------------------------------------------------------- drawing helpers

def pct(e: float) -> str:
    return f"{e * 100:.1f} %" if e < 0.1 else f"{e * 100:.0f} %"


class Graph:
    """A force vs crosshead-travel axes over the measured curve, with one filling-in line per case and a
    moving dot at its head. The scoring checkpoints are not drawn (a 3-minute talk: they only confuse)."""

    def __init__(self, fig, rect, cases: list[Case], fmax: float, meas: dict, font: float = 12,
                 legend_loc="upper left", gaps: bool = True):
        self.ax = ax = nv.curve_axes(fig, rect, meas, X_LIM, fmax)
        ax.lines[0].set_label("real test (Instron)")
        ax.lines[1].set_visible(False)                       # nv.curve_axes' checkpoint markers: hidden
        ax.lines[1].set_label("_nolegend_")
        ax.set_xlabel("crosshead travel (mm)", fontsize=font)
        ax.set_ylabel("force (N)", fontsize=font)
        ax.tick_params(labelsize=font - 1)
        ax.grid(alpha=0.25)
        self.meas, self.cases = meas, cases
        self.ck = np.array(meas["checkpoints_mm"])
        self.items = []
        for c in cases:
            label = c.label if "simulation" in c.label else f"simulation: {c.label}"
            line, = ax.plot([], [], color=c.color, lw=3.0, label=label, zorder=4)
            head, = ax.plot([], [], "o", color=c.color, ms=10, zorder=7)
            self.items.append((c, line, head))
        self.legend = ax.legend(fontsize=font - 2, frameon=False, loc=legend_loc)

    def update(self, x: float):
        for c, line, head in self.items:
            line.set_data(*c.filled(x))
            xh = min(x, c.xc[-1])
            head.set_data([xh], [np.interp(xh, c.xc, c.s.F)])


def live_text(c: Case, x: float, meas: dict, short: bool = False, two_lines: bool = False) -> str:
    """'right now' comparison of the simulation and the test at crosshead travel x."""
    if x < LIVE_FROM:
        return "loading ..."
    xh = min(x, c.xc[-1])
    fs = np.interp(xh, c.xc, c.s.F)
    fm = np.interp(xh, meas["curve_mm"], meas["curve_N"])
    e = abs(fs - fm) / fm
    if short:
        return f"right now: {pct(e)} off"
    side = "too low" if fs < fm else "too high"
    if two_lines:
        return f"right now: simulation {fs:,.0f} N, test {fm:,.0f} N\n{pct(e)} {side}"
    return f"right now: {fs:,.0f} N vs {fm:,.0f} N in the test ({pct(e)} {side})"


def fmax_of(cases: list[Case], meas: dict) -> float:
    return max([max(meas["curve_N"])] + [c.s.F[c.xc <= X_LIM].max() for c in cases]) * 1.1


def subtitle(fig, mag: float, y: float = 0.905):
    scale = "true scale" if mag == 1 else f"deformation x {mag:g}"
    fig.text(0.012, y, f"black: the real compression test     colour: our simulation     part coloured by "
             f"permanent (plastic) strain, {scale}", fontsize=14, color="#444444", va="center")


# --------------------------------------------------------------------------- the animation loop

def timeline():
    """[(crosshead travel, repeat count, phase)]: a start hold, the ramp, an end hold."""
    n = int(RAMP_S * nv.FPS)
    out = [(0.0, int(HOLD_START_S * nv.FPS), "start")]
    out += [(X_END * i / n, 1, "ramp") for i in range(1, n + 1)]
    out.append((X_END, int(HOLD_END_S * nv.FPS), "end"))
    return out


def run_video(name: str, fig, panels: list, graphs: list, texts, mag: float) -> dict:
    """Animate: panels [(Case, PartView, image)], graphs [Graph], texts(x, phase) updates the labels.
    Writes OUT/<name>.mp4 and the key frames <name>_{start,middle,end}.png."""
    import matplotlib.image as mpimg
    video = nv.Video(OUT / f"{name}.mp4")
    keys = {}
    steps = timeline()
    mid = int(np.argmin([abs(x - KEY_MID_MM) for x, _, _ in steps]))
    for i, (x, rep, phase) in enumerate(steps):
        for c, view, im in panels:
            pts, P, _ = c.at(x, mag)
            im.set_data(view.render(pts, P))
        for g in graphs:
            g.update(x)
        texts(x, phase)
        frame = nv.grab(fig)
        video.add(frame, rep)
        for tag, j in (("start", 0), ("middle", mid), ("end", len(steps) - 1)):
            if i == j:
                keys[tag] = OUT / f"{name}_{tag}.png"
                mpimg.imsave(keys[tag], np.ascontiguousarray(frame[:, :, :3]))
    video.close()
    for _, view, _ in panels:
        view.close()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {"mp4": str(video.path), "frames": video.n, "seconds": video.n / nv.FPS,
            "mb": video.path.stat().st_size / 1e6, "keys": {k: str(v) for k, v in keys.items()}}


def part_panel(fig, rect, c: Case, zoom: float):
    view = nv.PartView(c.s.surf, nv.pixels(rect), nv.peeq_scale(None), zoom)
    im = nv.image_axes(fig, rect).imshow(view.render(c.s.X, c.s.P[0]))
    return c, view, im


# --------------------------------------------------------------------------- the four videos

def video_before_after(cases: dict[str, Case], mag: float, zoom: float) -> dict:
    return video_pair(cases, "before_after", ("before", "after"), ("before post-training", "after post-training"),
                      "Post-training on the real test: simulation vs experiment",
                      lambda head, c: f"{head}: {pct(c.err)} off the real test", mag, zoom)


def video_first_vs_after(cases: dict[str, Case], mag: float, zoom: float) -> dict:
    """The teammates' first nonlinear simulation (many nodes, far off) vs after post-training (few, close)."""
    return video_pair(cases, "first_vs_after", ("first", "after"), ("first nonlinear simulation", "after post-training"),
                      "Our first nonlinear simulation vs after post-training",
                      lambda head, c: f"{c.nodes:,} nodes, {pct(c.err)} off the real test", mag, zoom)


def video_pair(cases: dict[str, Case], name: str, keys, heads, title: str, end_text, mag: float, zoom: float) -> dict:
    """Two columns, each a case (keys) under its heading (heads): the part, a live comparison line and its own
    graph. end_text(head, case) is the line held at the end."""
    from ace.nonlinear import measured
    meas = measured()
    pair = [cases[k] for k in keys]
    fmax = fmax_of(pair, meas)
    fig = nv.figure()
    fig.text(0.012, 0.955, title, fontsize=28, weight="bold", va="center")
    subtitle(fig, mag)
    nv.colorbar(fig, [0.80, 0.948, 0.18, 0.014], nv.peeq_scale(None))
    panels, graphs, live = [], [], []
    for j, c in enumerate(pair):
        x0 = j * 0.5
        fig.text(x0 + 0.25, 0.85, heads[j], fontsize=26, weight="bold", color=c.color, ha="center", va="center")
        sub = c.sub if c.label == heads[j] else f"{c.label}  ({c.sub})"      # the heading already names it
        fig.text(x0 + 0.25, 0.81, sub, fontsize=14, color="#555555", ha="center", va="center")
        panels.append(part_panel(fig, [x0 + 0.01, 0.37, 0.48, 0.42], c, zoom))
        live.append(fig.text(x0 + 0.25, 0.345, "", fontsize=18, weight="bold", color=c.color, ha="center",
                             va="center"))
        graphs.append(Graph(fig, [x0 + 0.06, 0.065, 0.42, 0.24], [c], fmax, meas, font=12))
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([0.5, 0.5], [0.04, 0.88], color="#cccccc", lw=1))      # divider between the columns

    def texts(x, phase):
        for j, (c, t) in enumerate(zip(pair, live)):
            t.set_text(end_text(heads[j], c) if phase == "end" else live_text(c, x, meas))
    return run_video(name, fig, panels, graphs, texts, mag)


def video_three_way(cases: dict[str, Case], mag: float, zoom: float) -> dict:
    """Three columns: first nonlinear simulation | before | after post-training, each with its part, a live
    comparison and its own graph over the test (one shared graph would stack the first two curves, which
    both level off near 3 kN)."""
    from ace.nonlinear import measured
    meas = measured()
    trio = [cases[k] for k in ("first", "before", "after")]
    heads = ("first nonlinear simulation", "before post-training", "after post-training")
    subs = (trio[0].sub, f"{BEFORE_SETUP}, default mesh, {trio[1].nodes:,} nodes",
            f"learned test conditions, most accurate mesh, {trio[2].nodes:,} nodes")
    fmax = fmax_of(trio, meas)
    fig = nv.figure()
    fig.text(0.012, 0.955, "Three simulations of the real compression test", fontsize=28, weight="bold",
             va="center")
    subtitle(fig, mag)
    nv.colorbar(fig, [0.80, 0.948, 0.18, 0.014], nv.peeq_scale(None))
    panels, graphs, live = [], [], []
    w = 1 / 3
    for j, c in enumerate(trio):
        x0 = j * w
        fig.text(x0 + w / 2, 0.855, heads[j], fontsize=22, weight="bold", color=c.color, ha="center", va="center")
        fig.text(x0 + w / 2, 0.815, subs[j], fontsize=12, color="#555555", ha="center", va="center")
        panels.append(part_panel(fig, [x0 + 0.005, 0.42, w - 0.01, 0.38], c, zoom))
        live.append(fig.text(x0 + w / 2, 0.375, "", fontsize=15, weight="bold", color=c.color, ha="center",
                             va="center", linespacing=1.3))
        graphs.append(Graph(fig, [x0 + 0.05, 0.07, w - 0.07, 0.24], [c], fmax, meas, font=11))
    from matplotlib.lines import Line2D
    for xd in (w, 2 * w):                                                       # dividers between the columns
        fig.add_artist(Line2D([xd, xd], [0.04, 0.88], color="#cccccc", lw=1))

    def texts(x, phase):
        for c, t in zip(trio, live):
            t.set_text(f"{c.nodes:,} nodes\n{pct(c.err)} off the real test" if phase == "end"
                       else live_text(c, x, meas, two_lines=True))
    return run_video("three_way", fig, panels, graphs, texts, mag)


def video_four_stages(cases: dict[str, Case], mag: float, zoom: float) -> dict:
    from ace.nonlinear import measured
    meas = measured()
    order = [cases[k] for k in ("before", "stage2", "stage3", "after")]
    fmax = fmax_of(order, meas)
    fig = nv.figure()
    fig.text(0.012, 0.955, "Four steps from our first simulation to the real test", fontsize=28, weight="bold",
             va="center")
    subtitle(fig, mag)
    nv.colorbar(fig, [0.80, 0.948, 0.18, 0.014], nv.peeq_scale(None))
    panels, live = [], []
    for j, c in enumerate(order):
        x0, w = j / 4, 1 / 4
        head = textwrap.fill(f"{j + 1}. {c.label}", 30)          # two lines: four labels fit side by side
        fig.text(x0 + w / 2, 0.855, head, fontsize=15, weight="bold", color=c.color, ha="center", va="center",
                 linespacing=1.3)
        sub = f"{BEFORE_SETUP}, {c.nodes:,} nodes" if c.key == "before" else c.sub
        fig.text(x0 + w / 2, 0.805, sub, fontsize=12, color="#555555", ha="center", va="center")
        panels.append(part_panel(fig, [x0 + 0.005, 0.42, w - 0.01, 0.39], c, zoom))
        live.append(fig.text(x0 + w / 2, 0.40, "", fontsize=17, weight="bold", color=c.color, ha="center",
                             va="center"))
    g = Graph(fig, [0.06, 0.07, 0.56, 0.28], order, fmax, meas, font=12, gaps=False)
    g.legend.remove()
    handles = g.ax.lines[:2] + [it[1] for it in g.items]
    leg = g.ax.legend(handles, [h.get_label() for h in handles], fontsize=12, frameon=False,
                      loc="center left", bbox_to_anchor=(1.03, 0.5))

    def texts(x, phase):
        for j, (c, t) in enumerate(zip(order, live)):
            t.set_text(f"mean error {pct(c.err)}" if phase == "end" else live_text(c, x, meas, short=True))
            leg.get_texts()[2 + j].set_text(f"{j + 1}. {c.label}" + (f": {pct(c.err)} off" if phase == "end" else ""))
    return run_video("four_stages", fig, panels, [g], texts, mag)


def video_single(c: Case, name: str, head: str, mag: float, zoom: float, fmax: float) -> dict:
    from ace.nonlinear import measured
    meas = measured()
    fig = nv.figure()
    fig.text(0.012, 0.955, head, fontsize=28, weight="bold", color=c.color, va="center")
    subtitle(fig, mag)
    fig.text(0.012, 0.855, f"{c.label}  ({c.sub})", fontsize=17, color="#444444", va="center")
    panels = [part_panel(fig, [0.0, 0.07, 0.53, 0.76], c, zoom)]
    nv.colorbar(fig, [0.12, 0.06, 0.30, 0.016], nv.peeq_scale(None))
    live = fig.text(0.775, 0.82, "", fontsize=17, weight="bold", color=c.color, ha="center", va="center")
    g = Graph(fig, [0.59, 0.12, 0.38, 0.62], [c], fmax, meas, font=14)
    short = head.split(":")[0].lower()

    def texts(x, phase):
        live.set_text(f"{short}: {pct(c.err)} off the real test" if phase == "end"
                      else live_text(c, x, meas, two_lines=True))
    return run_video(name, fig, panels, [g], texts, mag)


# --------------------------------------------------------------------------- page

def write_page(cases: dict[str, Case], pick: dict, info: dict, mag: float):
    f, b, s2, s3, a = (cases[k] for k in ("first", "before", "stage2", "stage3", "after"))
    scale = "Deformation at true scale" if mag == 1 else f"Deformation magnified x {mag:g} (same for every case)"
    captions = {
        "before_after": (f"Before post-training the simulation is {pct(b.err)} off the real test; "
                         f"after post-training it is {pct(a.err)} off.",
                         f"Before: frictionless platens, default mesh ({b.nodes:,} nodes), no seating. After: learned "
                         f"test conditions + the fine-tuning run's most accurate mesh ({a.sub}, agent {pick['agent']}). "
                         f"{scale}."),
        "four_stages": (f"Error vs the real test: {pct(b.err)} &rarr; {pct(s2.err)} &rarr; {pct(s3.err)} &rarr; "
                        f"{pct(a.err)}.",
                        f"Learned test conditions do most of the work; the meshes ({s2.nodes:,} / {s3.nodes:,} / "
                        f"{a.nodes:,} nodes) take it the rest of the way. {scale}."),
        "before_only": (f"What we started with: {pct(b.err)} off the real test.",
                        f"Frictionless platens, default mesh ({b.nodes:,} nodes), no seating or machine compliance; "
                        f"it levels off near {b.s.F[b.xc <= 2.5].max():,.0f} N while the test reaches "
                        f"{max(cases_meas()['curve_N']):,.0f} N. {scale}."),
        "after_only": (f"After post-training: {pct(a.err)} off the real test.",
                       f"Learned test conditions + the fine-tuning run's most accurate mesh ({a.sub}, agent "
                       f"{pick['agent']}). {scale}."),
        "three_way": (f"First nonlinear simulation {pct(f.err)} off the real test, before post-training "
                      f"{pct(b.err)}, after post-training {pct(a.err)}.",
                      f"First: the teammates' setup (frictionless platens, fixed base, {f.nodes:,}-node mesh). "
                      f"Before: frictionless platens, default mesh ({b.nodes:,} nodes). After: learned test conditions "
                      f"+ most accurate mesh ({a.nodes:,} nodes). {scale}."),
        "first_vs_after": (f"{f.nodes:,} nodes, {pct(f.err)} off vs {a.nodes:,} nodes, {pct(a.err)} off: "
                           f"most of the gain is the learned test conditions; the post-trained mesh adds the rest.",
                           f"First: the teammates' first nonlinear simulation (frictionless platens, no seating or "
                           f"machine compliance). After: learned test conditions + most accurate mesh ({a.sub}). "
                           f"{scale}."),
    }
    blocks = []
    for name, (l1, l2) in captions.items():
        v = info[name]
        frames = "".join(f'<figure><img src="videos/{Path(v["keys"][t]).name}" alt="{name} {t}" loading="lazy">'
                         f'<figcaption>{t}</figcaption></figure>' for t in ("start", "middle", "end"))
        blocks.append(f"""
<section>
  <h2>{name.replace('_', ' ')}</h2>
  <video src="videos/{name}.mp4" controls loop muted autoplay playsinline></video>
  <p class="cap">{l1}<br>{l2}</p>
  <div class="keys">{frames}</div>
  <p class="meta">videos/{name}.mp4 &middot; {v['seconds']:.1f} s &middot; {v['mb']:.1f} MB</p>
</section>""")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Act 3 Videos</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1f23; --muted: #5b6270; --line: #e3e6ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #16181c; --fg: #e9ebee; --muted: #a2a9b4; --line: #2c3036; }} }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; margin: 0 auto;
       max-width: 1200px; padding: 24px 16px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; }} h2 {{ font-size: 20px; margin: 40px 0 8px; text-transform: capitalize; }}
.lead, .meta {{ color: var(--muted); }} .meta {{ font-size: 13px; }}
video {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
.keys {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
.keys img {{ width: 100%; border: 1px solid var(--line); border-radius: 4px; }}
figure {{ margin: 0; }} figcaption {{ font-size: 13px; color: var(--muted); text-align: center; }}
@media (max-width: 640px) {{ .keys {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>Act 3: simulation vs the real compression test</h1>
<p class="lead">Part abc_00000065 compressed in an Instron; force vs crosshead travel. Error = mean relative force
error at the 5 checkpoints (0.5 to 2.5 mm). Colour on the part: plastic strain PEEQ, same range in every video.</p>
{''.join(blocks)}
</body></html>
"""
    (OUT.parent / "videos.html").write_text(html)


def cases_meas() -> dict:
    from ace.nonlinear import measured
    return measured()


def make_videos(run_dir: Path, mag: float, only: list[str] | None, pick_by: str = "accurate"):
    cases, pick = load_cases(run_dir, pick_by)
    for c in cases.values():
        print(f"{c.key:7s} {c.nodes:6d} nodes  mean error {c.err:.4%}  ends at {c.xc[-1]:.3f} mm  "
              f"converged {c.s.s.get('converged')}")
    print("after =", json.dumps(pick))
    fmax = fmax_of(list(cases.values()), cases_meas())
    jobs = {"before_after": lambda: video_before_after(cases, mag, zoom=1.45),
            "four_stages": lambda: video_four_stages(cases, mag, zoom=1.35),
            "before_only": lambda: video_single(cases["before"], "before_only", "Before post-training: our first simulation",
                                                mag, 1.5, fmax),
            "after_only": lambda: video_single(cases["after"], "after_only", "After post-training on the real test: most accurate mesh",
                                               mag, 1.5, fmax),
            "three_way": lambda: video_three_way(cases, mag, zoom=1.35),
            "first_vs_after": lambda: video_first_vs_after(cases, mag, zoom=1.3)}   # 1.45 clips the base
    info_file = OUT / "videos.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    for name, job in jobs.items():
        if not only or name in only:
            info[name] = job()
    info["cases"] = {k: {"nodes": c.nodes, "mean_rel_err": c.err, "rel_err": c.s.s["rel_err"], "label": c.label}
                     for k, c in cases.items()}
    info["after_pick"], info["magnification"] = pick, mag
    info_file.write_text(json.dumps(info, indent=1))
    if all(k in info for k in jobs):
        write_page(cases, pick, info, mag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("before", help="solve the 'before' case (needs ccx)")
    b.add_argument("--threads", type=int, default=4)
    f = sub.add_parser("first", help="solve the 'first' case (needs ccx) and check it against the committed curve")
    f.add_argument("--threads", type=int, default=8)
    v = sub.add_parser("videos", help="all videos, key frames and videos.html")
    v.add_argument("--run", type=Path, required=True, help="fine-tuning run dir (the 'after' mesh)")
    v.add_argument("--mag", type=float, default=1.0, help="deformation magnification, the same for every case")
    v.add_argument("--only", nargs="*", help="just these videos (before_after four_stages before_only after_only three_way "
                                           "first_vs_after)")
    v.add_argument("--pick", choices=sorted(PICKS), default="accurate",
                   help="the 'after' mesh: most accurate (default, the user's choice) or best base score")
    a = ap.parse_args()
    if a.cmd == "before":
        solve_before(a.threads)
    elif a.cmd == "first":
        solve_first(a.threads)
        check_first()
    else:
        make_videos(a.run, a.mag, a.only, a.pick)


if __name__ == "__main__":
    main()
