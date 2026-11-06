"""Your CAD page tools (pages/cad.js): the reference study of an uploaded part, and trained agents on a
problem made from it. The commands are ``feacore.problems.custom reference`` and ``ace.infer``; the
task's FEACORE_RUNS_DIR is this server's run store, whose cad/ holds the uploads, so both find the
problem the page made."""

from __future__ import annotations

import os

from feacore.ui.tasks import Param, Tool, register_tool

CPUS = os.cpu_count() or 4


register_tool(Tool(
    name="cad.reference", title="Reference study of an uploaded part (CPU)", group="cad",
    command=["-m", "feacore.problems.custom", "reference"],
    description="Nested curved tet10 meshes of the part (the gold generator's pipeline, the finest level capped), "
                "every BC set solved on every level with the fixed solver, and each QoI certified with the gold "
                "rules: the QoIs that converge become the set's references. Each solve is recorded for the viewer.",
    params=[Param("upload", required=True, flag="", help="upload id (cad_...)"),
            Param("sets", help="comma-separated problem ids (default: every BC set of the upload)"),
            Param("levels", "int", 3, min=3, max=4, help="nested levels (each split has 8x the cells)"),
            Param("max_cells", "int", 30_000, min=500, max=2_000_000,
                  help="cap on the finest level's cells (holes and fillets set a floor): ~40k cells are ~175k DOFs, "
                       "about 6 min and 6 GB per solve on a 4-core CPU"),
            Param("h0_frac", "float", 0.237, min=0.02, max=2.0, advanced=True,
                  help="base size / bounding-box diagonal before the cap enlarges it (the gold generator's value)"),
            Param("threads", "int", 1, min=1, max=CPUS, advanced=True, help="gmsh threads")],
    timeout=12 * 3600))

register_tool(Tool(
    name="cad.infer", title="Run trained agents on a problem", group="cad",
    command=["-m", "ace.infer"],
    description="An exported swarm's agents set up and solve the problem, each in its own process, with no "
                "learning (python -m ace.infer). Without a reference value the result is the agents' consensus "
                "and each run's own ZZ error estimate.",
    params=[Param("problem", "problem", required=True),
            Param("swarm", "path", "results/trained_swarms/div_s2", help="an exported swarm (ace.export_swarm)"),
            Param("agents", default="all", help="comma-separated agents, e.g. a0,a1, or all"),
            Param("mock", "bool", True, help="canned LLM answers, no API calls (a free end-to-end check)"),
            Param("out", "path", advanced=True, help="output folder (default: in the run store)")],
    out_param="out", out_root="@runs/cad_agents", requires_env=["ANTHROPIC_API_KEY"], unless_param="mock",
    timeout=6 * 3600))
