"""Nonlinear section tools (dashboard #/nonlinear): CalculiX analyses past yield of the printed parts, and the
renderers that turn a finished run into stills and videos.

  nonlinear.score          the agents' frozen scoring ramp (ace.nonlinear) on reference meshes, vs the real test
  nonlinear.ramp           elastoplastic ramp of the C-bracket compression test (scripts/ccx_compression_ramp.py)
  nonlinear.crush_mesh     a contract mesh for crush runs (mesh_systems/ace_sizing, straight-edged C3D10)
  nonlinear.imperfection   the same mesh moved by its first buckling mode (the neck-study recipe)
  nonlinear.crush          crush to fracture: J2 + NLGEOM + element erosion (experiments/physical/crush.py)
  nonlinear.render_ramp    stills of a ramp: plastic strain and largest principal stress (matplotlib)
  nonlinear.render_crush   video of a crush run (pyvista, off-screen display, ffmpeg)
  nonlinear.render_score   video of a scoring ramp (pyvista, imageio-ffmpeg)

Defaults are small so they finish on a 4-core box; each title says how long a run takes there. Every output
goes to the run store under <runs>/nonlinear/ (feacore.ui.nonlinear_api lists it). Renderers that need an
OpenGL context run through feacore.ui.offscreen (xvfb-run when there is no display).
"""

from __future__ import annotations

import re
from pathlib import Path

from feacore.ui.tasks import Param, Tool, ToolArgsError, register_tool

GROUP = "nonlinear"
PARTS = ["abc_00000065", "abc_00000003"]
STEP = {p: f"benchmarks/gold/{p}_bc0/public/geometry.step" for p in PARTS}
AXIS = {"abc_00000065": "y", "abc_00000003": "z"}
OFFSCREEN = ["-m", "feacore.ui.offscreen"]
SCORE_MESHES = {"auto": "the automesher (gmsh default sizes, 2.9k nodes)",
                "swarm_bc0": "the trained swarm's bc0 mesh (6.4k nodes)",
                "h4": "uniform 4 mm (9.5k nodes)", "h25": "uniform 2.5 mm (24k nodes)", "h20": "uniform 2 mm (41k nodes)"}


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _out(args: dict, key: str = "out") -> Path:
    return Path(args[key])


# --------------------------------------------------------------------------- the agents' scoring ramp


def _score_argv(a: dict) -> list[str]:
    only = [m for m in SCORE_MESHES if a.get(m)]
    if not only:
        raise ToolArgsError(["pick at least one mesh"])
    return ["--only", *only, "--threads", str(a["threads"]), "--out", str(_out(a))]


register_tool(Tool(
    name="nonlinear.score", group=GROUP, command=[*OFFSCREEN, "scripts/nl_baselines.py"],
    title="Score meshes against the real test (the agents' frozen ramp)",
    description="The nonlinear check the agents are scored with (ace.nonlinear): one frozen elastoplastic CalculiX "
                "ramp per mesh (J2 resin from the dogbone, sticking platens, NLGEOM, to 2.4 mm), its force mapped to "
                "the crosshead (seating gap + load-train compliance) and compared with the Instron curve at 0.5-2.5 mm. "
                "Automesher + uniform 4 mm take about 1 min here; 2.5 mm a few minutes, 2 mm about 10 min.",
    params=[Param("auto", "bool", True, label="automesher", help=SCORE_MESHES["auto"]),
            Param("swarm_bc0", "bool", False, label="swarm mesh", help=SCORE_MESHES["swarm_bc0"]),
            Param("h4", "bool", True, label="uniform 4 mm", help=SCORE_MESHES["h4"]),
            Param("h25", "bool", False, label="uniform 2.5 mm", help=SCORE_MESHES["h25"] + "; a few minutes",
                  advanced=True),
            Param("h20", "bool", False, label="uniform 2 mm", help=SCORE_MESHES["h20"] + "; about 10 minutes",
                  advanced=True),
            Param("threads", "int", 4, help="CalculiX threads", min=1, max=64, advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/score", build=_score_argv))


# --------------------------------------------------------------------------- elastoplastic ramp of the test


CURVED_MAX_H = 3.0     # curved quadratic tets coarser than this fail CalculiX (nonpositive Jacobian, midside check)


def _ramp_argv(a: dict) -> list[str]:
    if not a["straight"] and a["h"] > CURVED_MAX_H:
        raise ToolArgsError([f"curved tets need h <= {CURVED_MAX_H:g} mm (coarser ones fail CalculiX's checks): "
                             "lower h or tick straight edges"])
    n_mat = len([m for m in a["materials"].split(",") if m])
    threads = max(1, a["threads"] // n_mat)          # the materials run in parallel and share the threads
    argv = ["--h", _num(a["h"]), "--umax", _num(a["umax"]), "--increments", str(a["increments"]),
            "--materials", a["materials"], "--platen", a["platen"], "--threads", str(threads),
            "--work", str(_out(a)), "--out", str(_out(a))]
    return argv + (["--straight"] if a["straight"] else []) + (["--vtu"] if a.get("vtu") else [])


register_tool(Tool(
    name="nonlinear.ramp", group=GROUP, command=["scripts/ccx_compression_ramp.py"],
    title="Elastoplastic ramp of the C-bracket compression test",
    description="The printed C-bracket (abc_00000065_compression) pushed past yield on CalculiX: J2 plasticity from "
                "the resin's dogbone test, NLGEOM, the prong tops moved by a rigid platen, the bottom held, compared "
                "with the Instron curve. The default (5 mm, straight edges, 6.5k nodes) takes about 30 s here and "
                "reads stiff; curved 3 mm tets (16k nodes) about 1.5 min; the committed runs used a 41k-node mesh "
                "(about 10 min per run on 4 cores).",
    params=[Param("h", "float", 5.0, label="element size (mm)",
                  help="straight 5 mm: 6.5k nodes; curved needs <= 3 mm", min=1.5, max=12.0),
            Param("straight", "bool", True, label="straight edges",
                  help="straight-edged quadratic tets: any size runs, but the faceted bore reads a few % stiff"),
            Param("umax", "float", 3.0, label="platen travel (mm)", help="the test peaks at 2.5 mm of crosshead travel",
                  min=0.2, max=8.0),
            Param("increments", "int", 15, help="equal increments (cut back when needed)", min=4, max=200),
            Param("materials", "choice", "team", choices=["team", "corrected", "team,corrected"],
                  help="team: the dogbone as measured (E 1880 MPa apparent); corrected: datasheet E with the grip "
                       "slack removed; both: two runs side by side"),
            Param("platen", "choice", "stick", choices=["stick", "slide"],
                  help="stick: the platen holds the prong tops (friction); slide: frictionless, the prongs splay"),
            Param("threads", "int", 4, help="CPU threads in total (two materials share them)", min=1, max=64,
                  advanced=True),
            Param("vtu", "bool", False, help="also keep the per-increment .vtu series (ParaView)", advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/ramp", build=_ramp_argv))


# --------------------------------------------------------------------------- crush to fracture


def _mesh_argv(a: dict) -> list[str]:
    part = a["part"]
    argv = [STEP[part], AXIS[part], str(_out(a)), "--purpose", "nonlinear", "--h", _num(a["h"] / 1000.0),
            "--straight"]
    return argv + ([] if a.get("hotspots", True) else ["--no-hotspots"])


register_tool(Tool(
    name="nonlinear.crush_mesh", group=GROUP, command=["experiments/mesh_systems/ace_sizing/build.py"],
    title="Mesh a part for a crush run",
    description="A contract mesh (groups bottom, top, domain; experiments/mesh_systems/CONTRACT.md) made the ACE "
                "agents' way: a global size, refinement at the platens and at the two worst stress spots of a "
                "coarse linear solve, quality gates. Quadratic tets with straight edges, so CalculiX accepts any "
                "size (the faceted bore can fail the volume gate). 8 mm gives about 6k nodes in seconds; the study "
                "meshes had about 40k nodes.",
    params=[Param("part", "choice", "abc_00000065", choices=PARTS,
                  help="abc_00000065: the C-bracket (shaft clamp), crushed along y; abc_00000003: the tube, along z"),
            Param("h", "float", 8.0, label="element size (mm)", help="8 mm: ~6k nodes (a crush takes minutes); "
                  "3.5 mm: ~30k nodes (hours)", min=2.0, max=15.0),
            Param("hotspots", "bool", True, help="refine at the two worst von Mises spots of a coarse linear solve",
                  advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/meshes", build=_mesh_argv))


def _imperfection_argv(a: dict) -> list[str]:
    return ["--part", a["part"], "--mesh-dir", a["mesh_dir"], "--out", str(_out(a)), "--platen", a["platen"],
            "--amp-mm", _num(a["amp_mm"]), "--mode", str(a["mode"]), "--modes", str(a["modes"]),
            "--threads", str(a["threads"])]


register_tool(Tool(
    name="nonlinear.imperfection", group=GROUP, command=["experiments/physical/imperfection.py"],
    title="Buckling imperfection of a crush mesh (neck-study recipe)",
    description="Linear buckling (CalculiX *BUCKLE) of a crush mesh, then the mesh moved by its first mode, scaled "
                "to a small amplitude: with tension-only erosion it makes the clamp break at the arm necks, as the "
                "printed part did (experiments/physical/CRUSH_RUNS.md). Seconds on a coarse mesh.",
    params=[Param("part", "choice", "abc_00000065", choices=PARTS),
            Param("mesh_dir", "path", required=True, label="mesh directory", help="a crush mesh (mesh.msh with "
                  "groups bottom, top, domain)"),
            Param("platen", "choice", "sliding", choices=["sliding", "sliding_top", "bonded"],
                  help="platens of the buckling load case (the study used sliding)"),
            Param("amp_mm", "float", 0.1, label="amplitude (mm)", help="largest node move", min=0.001, max=2.0),
            Param("mode", "int", 1, help="which buckling mode (1 = lowest)", min=1, max=10, advanced=True),
            Param("modes", "int", 4, help="modes to compute", min=1, max=20, advanced=True),
            Param("threads", "int", 4, min=1, max=64, advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/meshes", build=_imperfection_argv))


_SWITCH = re.compile(r"^\d*\.?\d+:\d*\.?\d+$")


def _crush_argv(a: dict) -> list[str]:
    errors = []
    if not a.get("mesh_dir") and not a.get("resume"):
        errors.append("mesh_dir is required (or resume a run)")
    if a.get("compression_switch") and not _SWITCH.match(a["compression_switch"]):
        errors.append("compression_switch must look like 0.5:5 (force ratio : factor)")
    if a.get("compression_switch") and a["failure"] == "peeq":
        errors.append("compression_switch needs failure peeq_tension or principal")
    if a["platen"] == "friction" and a["part"] != "abc_00000065":
        errors.append("the friction platen is implemented for abc_00000065 (load along y) only")
    if errors:
        raise ToolArgsError(errors)
    out = _out(a)
    argv = ["--system", a["system"], "--part", a["part"], "--out", str(out.parent), "--run-id", out.name,
            "--threads", str(a["threads"]), "--platen", a["platen"], "--failure", a["failure"],
            "--release", a["release"]]
    if a.get("mesh_dir") and not a.get("resume"):
        argv += ["--mesh-dir", a["mesh_dir"]]
    if a.get("compression_factor") is not None:
        argv += ["--compression-factor", _num(a["compression_factor"])]
    if a.get("compression_switch"):
        argv += ["--compression-switch", a["compression_switch"]]
    if a["platen"] == "friction":
        argv += ["--mu", _num(a["mu"])]
    if a.get("resume"):
        argv += ["--resume", a["resume"]]
    return argv


register_tool(Tool(
    name="nonlinear.crush", group=GROUP, command=["experiments/physical/crush.py"],
    title="Crush to fracture (long run)",
    description="CalculiX J2 plasticity + NLGEOM + element erosion: the platen moves in small increments (0.05 mm "
                "through yield, 0.02 mm around failure), failed cells are removed between blocks, until the part "
                "separates or the force drops below 1 % of its peak. About 6 min for an 8 mm mesh (7k nodes) on "
                "4 cores; the 40k-node study runs took 2-6 hours. Kill keeps a checkpoint: resume continues it.",
    params=[Param("part", "choice", "abc_00000065", choices=PARTS),
            Param("mesh_dir", "path", label="mesh directory", help="a crush mesh: build one with 'Mesh a part for a "
                  "crush run' (not needed to resume)"),
            Param("platen", "choice", "bonded", choices=["bonded", "sliding", "sliding_top", "friction"],
                  help="bonded: the platens hold the faces (stuck tips); sliding: frictionless; friction: a rigid "
                       "contact platen with Coulomb friction (slow)"),
            Param("failure", "choice", "peeq", choices=["peeq", "peeq_tension", "principal"],
                  help="erosion: peeq = plastic strain at break in any stress state; peeq_tension = only in "
                       "tension (the neck-study recipe); principal = tension above the dogbone's strength"),
            Param("compression_factor", "float", None, label="compression factor",
                  help="peeq_tension / principal: cells in compression fail at this x the break strain (default 5; "
                       "0 = never; the neck recipe uses 0 with the switch 0.5:5)", min=0.0, max=100.0),
            Param("compression_switch", "str", "", label="compression switch",
                  help="R:F, e.g. 0.5:5: after the first failure, once the force is below R x peak, the factor "
                       "becomes F", advanced=True),
            Param("release", "choice", "adaptive", choices=["adaptive", "fixed", "fixed-held"],
                  help="failed cells removed per block: adaptive cap (default) or a fixed 8", advanced=True),
            Param("mu", "float", 0.3, help="friction coefficient (platen friction only)", min=0.0, max=2.0,
                  advanced=True),
            Param("threads", "int", 4, help="CalculiX threads", min=1, max=64),
            Param("system", "str", "dashboard", help="a label for the mesh (run.json)", advanced=True),
            Param("resume", "path", help="a stopped crush run to continue into a new run dir", advanced=True),
            Param("out", "path", help="run directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/crush", build=_crush_argv))


# --------------------------------------------------------------------------- renderers


def _render_ramp_argv(a: dict) -> list[str]:
    argv = ["--work", a["run"], "--out", str(_out(a))]
    if a.get("runs"):
        argv += ["--runs", a["runs"]]
    return argv + (["--vtu"] if a.get("vtu") else [])


register_tool(Tool(
    name="nonlinear.render_ramp", group=GROUP, command=["scripts/render_ccx_ramp.py"],
    title="Stills of a ramp: plastic strain and tension",
    description="The deformed part at true scale, coloured by plastic strain (PEEQ) and the largest principal "
                "stress, at first yield, at the test's peak load and at the end of the ramp, from two sides. "
                "Matplotlib only; under a minute.",
    params=[Param("run", "path", required=True, help="a ramp's output directory"),
            Param("runs", "str", "", help="run names, e.g. team_stick (default: every finished run)", advanced=True),
            Param("vtu", "bool", False, help="also write the rendered increments as .vtu", advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/renders", build=_render_ramp_argv))


def _render_crush_argv(a: dict) -> list[str]:
    argv = [a["run"], "--out", str(_out(a) / "crush.mp4"), "--layout", a["layout"], "--size", a["size"],
            "--min-s", _num(a["min_s"]), "--hold-s", _num(a["hold_s"]), "--fps", str(a["fps"]),
            "--workers", str(a["workers"])]
    if a.get("label"):
        argv += ["--label", a["label"]]
    if a.get("max_frames"):
        argv += ["--max-frames", str(a["max_frames"])]
    if a.get("only"):
        if not re.fullmatch(r"-?\d+(,-?\d+)*", a["only"]):
            raise ToolArgsError(["only: comma-separated frame indices, e.g. 0,-1"])
        argv += ["--only", a["only"]]
    return argv + (["--no-lift"] if a.get("no_lift") else [])


register_tool(Tool(
    name="nonlinear.render_crush", group=GROUP, command=[*OFFSCREEN, "experiments/physical/render_crush.py"],
    title="Video of a crush run",
    description="The original 3-panel crush video: outside (von Mises up to the strength, broken cells red), cut in "
                "half, and movement arrows; true scale, a slow orbit and a held ending. Needs pyvista with an "
                "off-screen display (xvfb-run here) and ffmpeg. About 2-3 min for a 7k-node run on 3 workers.",
    params=[Param("run", "path", required=True, help="a crush run directory"),
            Param("label", "str", "", help="caption for the mesh (default: run.json system)"),
            Param("layout", "choice", "panels", choices=["panels", "single"],
                  help="panels: three views; single: the outside view only, large captions"),
            Param("size", "choice", "1920x1080", choices=["1920x1080", "1280x720"],
                  help="1280x720 renders faster but crops the long captions"),
            Param("min_s", "float", 8.0, label="loading (s)", help="seconds for the loading part", min=2.0, max=60.0),
            Param("hold_s", "float", 3.0, label="hold (s)", help="seconds held on the last state", min=0.0, max=20.0),
            Param("workers", "int", 3, help="render processes", min=1, max=8),
            Param("fps", "int", 30, min=10, max=60, advanced=True),
            Param("max_frames", "int", None, help="render only the first N frames (a preview)", min=1, max=100000,
                  advanced=True),
            Param("only", "str", "", help="just these frame indices, as PNG stills (e.g. 0,-1)", advanced=True),
            Param("no_lift", "bool", False, help="do not lift the upper piece in the held ending", advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/renders", build=_render_crush_argv))


def _render_score_argv(a: dict) -> list[str]:
    argv = [a["run"], str(_out(a) / "ramp.mp4")]
    return argv + (["--peeq-max", _num(a["peeq_max"])] if a.get("peeq_max") else [])


register_tool(Tool(
    name="nonlinear.render_score", group=GROUP, command=[*OFFSCREEN, "scripts/nl_video.py", "ramp"],
    title="Video of a scoring ramp",
    description="One scored mesh through its increments: the mesh, plastic strain on the deformed part (fixed camera "
                "and colour range) and its force-travel curve filling in over the Instron curve. Needs the run's "
                "series/ (written by 'Score meshes'); pyvista off-screen and imageio-ffmpeg. About a minute.",
    params=[Param("run", "path", required=True, help="one scored mesh's directory (it holds summary.json and series/)"),
            Param("peeq_max", "float", None, label="PEEQ scale top", help="default 0.05", min=1e-4, max=1.0,
                  advanced=True),
            Param("out", "path", help="output directory (default: a new folder under the run store)", advanced=True)],
    out_param="out", out_root="@runs/nonlinear/renders", build=_render_score_argv))
