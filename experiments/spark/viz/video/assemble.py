"""PNG frames (render_twin.py) -> wingbox_twin.mp4: H.264, yuv420p, 30 fps, captions from summary.json.

    python3 assemble.py --frames frames --summary ../summary.json --out wingbox_twin.mp4 --ffmpeg /path/ffmpeg [--dry-run]

Host side, standard library only. Reads <frames>/manifest.json (or, for --dry-run before any render,
<frames>/../plan/plan.json). One segment per shot (drawtext captions + short fades), then a concat with -c copy.
Captions go through drawtext textfile= with expansion=none, so no escaping of ':', '%' or quotes is needed."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TITLE = "FEAgent virtual twin — wingbox stage 1"
CREDIT = "rendered with NVIDIA Omniverse RTX (Isaac Sim 6.2) on DGX Spark GB10"
GALLERY_TITLE = "Gold benchmark parts (ABC dataset): harness meshes vs certified answers"
GALLERY_SUB = ("FEAgent mesh harness (automesh, tet10) on gold parts; every certified QoI checked against its band\n"
               "von Mises stress on the deformed shape")
CARD_S = 3.0


def captions(s: dict, shots: list) -> dict:
    """Shot name -> list of (role, text). Every number comes from summary.json."""
    g = lambda k, d=None: s.get(k, d)  # noqa: E731
    dofs = f"{g('n_dofs', 0):,}"
    solve = f"{g('solve_time_s', 0):.1f} s"
    pct = g("tip_vs_beam_percent")
    pct = f"{pct:+.1f} %" if pct is not None else "n/a"
    vmax = g("vm_color_range_MPa", [0, 0])[1]
    clip = g("vm_clip_percentile")
    peak = g("vm_peak_singular_MPa", g("max_von_mises_MPa", 0))
    conv = g("vm_converged_away_from_root_MPa")
    eq = g("equilibrium_rel_error")
    eq = f"{eq:.0e}" if isinstance(eq, (int, float)) else str(eq)
    parts = g("parts", [])
    nwarn = sum(1 for p in parts if p.get("gate_status") == "warn")
    nfail = sum(1 for p in parts if p.get("gate_status") == "fail")
    npass = sum(1 for p in parts if p.get("gate_status") == "pass")
    verdict = str(g("verdict", "n/a")).upper()
    scale = g("deformation_scale", 1)
    c = {
        "a_orbit": [("title", TITLE),
                    ("sub", f"LC1: 20 kN tip lift · {dofs} DOFs · verdict {verdict}"),
                    ("cap", f"von Mises stress · colour max {vmax:.1f} MPa (p{clip:g})" if clip else
                     f"von Mises stress · colour max {vmax:.1f} MPa"),
                    ("credit", CREDIT)],
        "b_deform": [("cap", f"Deformation ×{scale:g} · tip deflection {g('tip_deflection_mm'):.3f} mm "
                             f"(extrapolated {g('tip_deflection_extrapolated_mm', g('tip_deflection_mm')):.3f})"),
                     ("cap2", f"beam theory {g('beam_theory_mm'):.3f} mm ({pct}) · GCI {g('gci_percent'):.3f} % "
                              f"· equilibrium {eq}")],
        "c_root": [("label", f"{peak:.1f} MPa peak: root-corner singularity\nnot a design value (grows with refinement)"),
                   ("cap", f"Clamped root corner · the {peak:.1f} MPa peak is a stress singularity"),
                   ("cap2", (f"converged field away from the root ≈ {conv:.1f} MPa · " if conv else "")
                    + f"colour max {vmax:.1f} MPa")],
        "d_umag": [("cap", f"Displacement magnitude |u| · max {g('max_displacement_mm'):.2f} mm "
                           f"· shown ×{scale:g}")],
        "e_gates": [("cap", f"Mesh-quality gates · {nwarn} warn, {nfail} fail, {npass} pass "
                            f"of {len(parts)} parts"),
                    ("cap2", "warn = thin_walls on the solved mesh (~1 element through the wall, gate needs 2); "
                             "every part spec passes")],
        "f_closing": [("bigtitle", TITLE),
                      ("numbers", f"Verdict {verdict} · {dofs} DOFs · cuDSS on NVIDIA GB10 (DGX Spark) {solve}\n"
                                  f"tip {g('tip_deflection_mm'):.3f} mm (extrap. "
                                  f"{g('tip_deflection_extrapolated_mm', g('tip_deflection_mm')):.3f}) vs beam theory "
                                  f"{g('beam_theory_mm'):.3f} mm ({pct}) · GCI {g('gci_percent'):.3f} % "
                                  f"· equilibrium {eq}\n"
                                  f"von Mises colour max {vmax:.1f} MPa (p{clip:g}); {peak:.1f} MPa root-corner peak "
                                  f"is a singularity"),
                      ("credit_c", CREDIT)],
    }
    for sh in shots:
        if sh["name"].startswith("g_"):
            vr, clip_g, sc = sh.get("vm_range"), sh.get("clip"), sh.get("scale")
            line2 = (f"von Mises colour 0–{vr[1]:.3g} MPa" + (f" (p{clip_g:g})" if clip_g else "")) if vr else "von Mises"
            if sc:
                line2 += f" · deformation ×{sc:g}"
            c[sh["name"]] = [("cap", sh.get("title") or sh["name"]), ("cap2", line2), ("credit", CREDIT)]
    c["_gallery_card"] = [("cardtitle", GALLERY_TITLE), ("cardsub", GALLERY_SUB), ("credit_c", CREDIT)]
    return c


STYLE = {  # role -> drawtext options (x, y, size, font, box)
    "title": ("40", "40", 46, BOLD, True, "0.4", "5.6"),
    "sub": ("40", "112", 28, FONT, True, "0.6", "5.6"),
    "cap": ("40", "h-150", 32, BOLD, True, None, None),
    "cap2": ("40", "h-96", 26, FONT, True, None, None),
    "credit": ("w-tw-36", "h-th-28", 20, FONT, False, None, None),
    "credit_c": ("(w-tw)/2", "h-th-40", 24, FONT, False, "1.2", None),
    "bigtitle": ("(w-tw)/2", "h*0.11", 60, BOLD, False, "0.5", None),
    "numbers": ("(w-tw)/2", "h*0.76", 26, FONT, True, "0.9", None),
    "cardtitle": ("(w-tw)/2", "h*0.38", 46, BOLD, False, "0.3", None),
    "cardsub": ("(w-tw)/2", "h*0.52", 28, FONT, False, "0.7", None),
}


def drawtext(textfile: Path, role: str, dur: float, anchor=None, static=False) -> str:
    if role == "label":
        x, y = anchor
        o = (f"x={x + 18}:y={y - 40}", 28, BOLD, True, None, None)
        pos, size, font, box, t_in, t_out = o
    else:
        x, y, size, font, box, t_in, t_out = STYLE[role]
        pos = f"x={x}:y={y}"
    t_in = float(t_in) if t_in else 0.25
    t_out = float(t_out) if t_out else dur - 0.25
    alpha = (f"if(lt(t,{t_in}),0,if(lt(t,{t_in + 0.4}),(t-{t_in})/0.4,"
             f"if(lt(t,{t_out}),1,if(lt(t,{t_out + 0.4}),({t_out + 0.4}-t)/0.4,0))))")
    opts = [f"fontfile={font}", f"textfile={textfile}", "expansion=none", pos, f"fontsize={size}",
            "fontcolor=white", "line_spacing=10"] + ([] if static else [f"alpha='{alpha}'"])
    if box:
        opts += ["box=1", "boxcolor=0x0b1416@0.72", "boxborderw=14"]
    else:
        opts += ["shadowcolor=0x000000@0.8", "shadowx=2", "shadowy=2"]
    return "drawtext=" + ":".join(opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    man_path = a.frames / "manifest.json"
    if not man_path.exists():
        plan = a.frames.parent / "plan" / "plan.json"
        if not a.dry_run or not plan.exists():
            sys.exit(f"assemble: {man_path} not found (render first)")
        man_path = plan
    man = json.loads(man_path.read_text())
    fps = man.get("fps", 30)
    summary = json.loads(a.summary.read_text())
    work = a.work or a.out.with_name("work")
    caps = captions(summary, man["shots"])
    segs = []
    cmds = []
    items = []
    for sh in man["shots"]:
        if sh["name"].startswith("g_") and not any(i.get("card") for i in items):
            items.append({"name": "g_card", "card": True, "n_frames": int(CARD_S * fps)})
        items.append(sh)
    for k, sh in enumerate(items):
        name, n = sh["name"], sh["n_frames"]
        dur = n / fps
        filt = []
        for j, (role, text) in enumerate(caps.get("_gallery_card" if sh.get("card") else name, [])):
            tf = work / f"{name}.{j}.{role}.txt"
            if a.dry_run:
                print(f"# caption {tf.name}: {text!r}")
            else:
                work.mkdir(parents=True, exist_ok=True)
                tf.write_text(text)
            anchor = None
            if role == "label":
                lab = sh.get("label") or {}
                anchor = lab.get("anchor_px")
                if not anchor:
                    continue
            filt.append(drawtext(tf, role, dur, anchor))
        fade = [f"fade=t=in:st=0:d=0.25", f"fade=t=out:st={dur - 0.25:.3f}:d=0.25"]
        vf = ",".join(["format=rgb24"] + filt + fade + ["format=yuv420p"])
        seg = work / f"{k:02d}_{name}.mp4"
        if sh.get("card"):
            src = ["-f", "lavfi", "-i", f"color=c=0x0b1416:s=1920x1080:r={fps}:d={dur}"]
        else:
            src = ["-framerate", str(fps), "-start_number", "0", "-i", str(a.frames / sh.get("dir", name) / "frame_%05d.png")]
        cmd = [a.ffmpeg, "-hide_banner", "-v", "error", "-y", *src, "-frames:v", str(n), "-vf", vf,
               "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-r", str(fps),
               "-an", str(seg)]
        cmds.append(cmd)
        segs.append(seg)
    listing = work / "segments.txt"
    final = [a.ffmpeg, "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(a.out)]
    for cmd in cmds:
        print("+ " + shlex.join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)
    print(f"+ write {listing}: " + " ".join(s.name for s in segs))
    if not a.dry_run:
        listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    print("+ " + shlex.join(final), flush=True)
    if not a.dry_run:
        subprocess.run(final, check=True)
        total = sum(sh["n_frames"] for sh in items) / fps
        print(f"assemble: wrote {a.out} ({total:.1f} s, {len(segs)} shots)")


if __name__ == "__main__":
    main()
