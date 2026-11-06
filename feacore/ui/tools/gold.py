"""Gold page tools (pages/gold.js): regenerate a part's study meshes, verify and render sets on a stored
mesh level, import generator archives, build nonlinear gold sets. The commands are
``feacore.goldgen.local`` (regenerate, import, nonlinear) and ``scripts/gold.py`` (verify, render); the
task runs with this server's FEACORE_RUNS_DIR and FEACORE_GOLD_DIR, so its output lands where the
Gold page reads it (regenerated meshes in <runs>/gold_regen/, rendered runs in the run store)."""

from __future__ import annotations

import os
import re

from feacore.ui.tasks import Param, Tool, ToolArgsError, register_tool

CPUS = os.cpu_count() or 4
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def ids(text: str | None) -> list[str]:
    """Set or part ids from a space- or comma-separated field (each checked: never a flag)."""
    out = [x for x in re.split(r"[\s,]+", text or "") if x]
    bad = [x for x in out if not _ID.match(x)]
    if bad:
        raise ToolArgsError([f"not a set or part id: {', '.join(bad)}"])
    return out


register_tool(Tool(
    name="gold.regenerate", title="Regenerate gold study meshes", group="gold",
    command=["-m", "feacore.goldgen.local", "regenerate"],
    description="Meshes a gold part again from its STEP with the generator's settings (the stored L0 base "
                "size, nested splits, each level's curving scheme, one gmsh thread) into this run store, and "
                "compares every level's sha256 and cell count with index.json.",
    params=[Param("part", required=True, flag="", help="part id, e.g. abc_00000003"),
            Param("levels", default="0", help="comma-separated, e.g. 0,1. L0 takes seconds, L1 under a minute, "
                                              "L2 (~200k cells) several minutes and ~2 GB"),
            Param("threads", "int", 1, min=1, max=CPUS, help="gmsh threads (the generator used 1)"),
            Param("install", "bool", False, help="copy a level into the gold root's _meshes/ when its sha256 "
                                                 "matches index.json (then it is the stored mesh)"),
            Param("source", "path", advanced=True, help="the part's original STEP (the ABC file index.json "
                                                        "names), meshed at the part's scale as the generator "
                                                        "did; default: the public geometry.step"),
            Param("h0_frac", "float", advanced=True, min=0.01, max=2.0,
                  help="base size / bounding-box diagonal (default: from the stored L0)")],
    timeout=6 * 3600))

register_tool(Tool(
    name="gold.verify", title="Verify gold sets on a stored mesh level", group="gold",
    command=["scripts/gold.py"],
    build=lambda a: ["verify", *ids(a.get("ids")), "--level", str(a["level"])],
    description="Re-solves each set on the stored mesh of that level with the fixed solver and compares every "
                "QoI with the generator's value (scripts/gold.py verify). Needs the stored meshes.",
    params=[Param("ids", help="set or part ids, space or comma separated (default: every set)"),
            Param("level", "int", 0, min=0, max=3, help="L0 ~1 s per set, L1 ~20 s and 3 GB, L2 needs a GPU")],
    timeout=6 * 3600))

register_tool(Tool(
    name="gold.render", title="Render gold solutions for the viewer", group="gold",
    command=["scripts/gold.py"],
    build=lambda a: ["render", *ids(a.get("ids")), "--level", str(a["level"]),
                     *(["--png", a["png_dir"]] if a.get("png") else [])],
    description="Re-solves sets on a stored mesh level and records each solve in this run store (source gold): "
                "the Gold page links them to the field viewer (scripts/gold.py render).",
    params=[Param("ids", help="set or part ids, space or comma separated (default: every set)"),
            Param("level", "int", 1, min=0, max=3),
            Param("png", "bool", False, help="also write a PNG per set and a contact sheet (needs pyvista)"),
            Param("png_dir", "path", advanced=True, help="where the PNGs go (default: the task's folder)")],
    out_param="png_dir", out_root="@runs/gold_png", timeout=6 * 3600))

register_tool(Tool(
    name="gold.import", title="Import a gold generator archive", group="gold",
    command=["-m", "feacore.goldgen.local", "import"],
    description="Checks every BC set of a goldgen archive and adds it, with its meshes, to the gold root "
                "(scripts/gold.py import). Upload the archive on the Gold page first, or give a path in the "
                "repository or the run store.",
    params=[Param("archive", "path", required=True, flag="",
                  help="e.g. @runs/gold_archives/goldgen-l2_0001.tgz, or an extracted directory"),
            Param("no_meshes", "bool", False, flag="--no-meshes", help="statements and references only")],
    timeout=2 * 3600))

register_tool(Tool(
    name="gold.nonlinear", title="Nonlinear gold sets (CalculiX, CPU)", group="gold",
    command=["-m", "feacore.goldgen.local", "nonlinear"],
    description="scripts/gold_nonlinear.py on this gold root: S355 steel with linear hardening, loads scaled "
                "past first yield, a CalculiX load ramp on each mesh level (CPU), GCI per QoI. Certifying a QoI "
                "needs three levels; L2 (~1M DOFs) needs a large machine.",
    params=[Param("sets", required=True, flag="", help="linear gold set ids, comma separated"),
            Param("levels", default="0,1,2", help="mesh levels, e.g. 0,1 for a quick look (nothing certified)"),
            Param("meshes", "choice", "stored", choices=["stored", "regenerated"],
                  help="stored: the certified meshes; regenerated: the ones made on this machine"),
            Param("threads", "int", min(4, CPUS), min=1, max=CPUS, help="CalculiX threads"),
            Param("increments", "int", 10, min=2, max=200, advanced=True, help="load increments per ramp"),
            Param("out", "path", advanced=True, help="output folder (default: in the run store)")],
    out_param="out", out_root="@runs/gold_nonlinear", timeout=48 * 3600))
