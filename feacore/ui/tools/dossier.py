"""Projects section tools (dashboard #/projects): the dossier pipeline as tasks.

  dossier.run      python -m dossier run PROJECT [--from S] [--to S] [--quick] [--force] [--only LC ...]
  dossier.report   python -m dossier run PROJECT --from report --to report

A project is a folder ``<runs>/dossier/<folder>/`` (the Projects page makes one from an example;
feacore/ui/dossier_api.py). The ``project`` parameter is ``@runs/dossier/<folder>``: the task runner turns
``@runs/`` into this server's run store, and ``project_dir`` below refuses every other place. The examples
that ship with dossier (``dossier/examples/``) are templates: a run never writes into them.

A run renders pictures with pyvista / VTK unless it is quick, so both tools start through
feacore.ui.offscreen (a virtual display on a server without a screen). The pipeline has no graceful stop:
Kill ends a run, and the stage files already written stay.

This module does not import dossier (the dashboard must start without it): STAGES is a copy of
``dossier.pipeline.STAGES``, and tests/test_ui_dossier.py checks that the two are the same.
"""

from __future__ import annotations

import re
from pathlib import Path

from feacore.ui.tasks import REPO_ROOT, Param, Tool, ToolArgsError, register_tool

GROUP = "dossier"
PROJECTS_DIR = "dossier"                                     # <runs>/dossier/<folder>/
FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")     # a project folder: a simple slug
STAGES = ("context", "geometry", "validate", "plan", "analyze", "assess", "report")
_RECORD_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")        # dossier.model.ID_RE


def project_ref(folder: str) -> str:
    """The ``project`` argument for a project folder of the dashboard."""
    return f"@runs/{PROJECTS_DIR}/{folder}"


def project_dir(args: dict) -> Path:
    """The project folder of a task, checked: ``<a run store>/dossier/<folder>`` with a project.yaml.
    The task runner has already resolved ``@runs/`` and refused a path outside the repository and the
    run store; this adds the rest, so that no other folder (the examples least of all) is ever run."""
    raw = str(args.get("project") or "")
    p = Path(raw)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    if p.parent.name != PROJECTS_DIR or not FOLDER.match(p.name) or (REPO_ROOT / "dossier") in (p, *p.parents):
        raise ToolArgsError([f"project: give {project_ref('<project folder>')} (a project of this dashboard), "
                             f"not {raw!r}"])
    if not (p / "project.yaml").is_file():
        raise ToolArgsError([f"project: there is no project {p.name!r} in {p.parent}"])
    return p


def load_case_ids(text: str | None) -> list[str]:
    """Load case ids from a space- or comma-separated field (each one checked: never a flag)."""
    ids = [x for x in re.split(r"[\s,]+", text or "") if x]
    bad = [x for x in ids if not _RECORD_ID.match(x)]
    if bad:
        raise ToolArgsError([f"only: not a load case id: {', '.join(bad)}"])
    return ids


def _run_argv(a: dict) -> list[str]:
    first, last = a.get("from") or STAGES[0], a.get("to") or STAGES[-1]
    if STAGES.index(first) > STAGES.index(last):
        raise ToolArgsError([f"from: the stage {first} comes after the stage {last}"])
    argv = ["run", str(project_dir(a)), "--from", first, "--to", last]
    for name in ("quick", "force", "no_pictures", "record_runs"):
        if a.get(name):
            argv.append(f"--{name.replace('_', '-')}")
    for name in ("levels", "node_cap"):
        if a.get(name) is not None:
            argv += [f"--{name.replace('_', '-')}", str(a[name])]
    only = load_case_ids(a.get("only"))
    return argv + (["--only", *only] if only else [])      # last: --only takes every word that follows


PROJECT = Param("project", "path", required=True,
                help="the project folder: @runs/dossier/<folder> (the Projects page fills it in)")

RUN = register_tool(Tool(
    name="dossier.run", title="Run verification", group=GROUP, command=["-m", "dossier"],
    description="The verification pipeline of one project (python -m dossier run): it evaluates the system "
                "context, reads the CAD, checks the requirements against the context, selects the load cases, "
                "analyses them, computes the margins and writes the report. A quick run uses coarse meshes "
                "and two mesh levels and makes no pictures. A stage that fails stops the run, and the log "
                "gives the reason.",
    params=[
        PROJECT,
        Param("quick", "bool", True, help="a first look: coarse meshes, two mesh levels, no pictures"),
        Param("from", "choice", STAGES[0], choices=list(STAGES), label="from stage",
              help="the first stage to run; the stages before it must have run"),
        Param("to", "choice", STAGES[-1], choices=list(STAGES), label="to stage", help="the last stage to run"),
        Param("only", help="load case ids to analyse, e.g. LC-FIRE LC-PROOF (default: all)"),
        Param("force", "bool", False, help="analyse a load case again even when its analysis is current"),
        Param("levels", "int", None, min=1, max=6, advanced=True, label="mesh levels",
              help="meshes of each convergence study (default 3, quick 2)"),
        Param("node_cap", "int", None, min=1000, max=2_000_000, advanced=True, label="node cap",
              help="nodes of the finest mesh (default 120 000, quick 25 000)"),
        Param("no_pictures", "bool", False, advanced=True, label="no pictures", help="a full run without pictures"),
        Param("record_runs", "bool", False, advanced=True, label="record solves",
              help="also write the solves to this dashboard's run store (the 3D field viewer)"),
    ],
    build=_run_argv, display=True, timeout=24 * 3600))

REPORT = register_tool(Tool(
    name="dossier.report", title="Build report", group=GROUP, command=["-m", "dossier"],
    description="Writes the report of one project again from the stage files it has (python -m dossier run "
                "--from report --to report). Use it after a review, or when only the report is stale.",
    params=[PROJECT],
    build=lambda a: ["run", str(project_dir(a)), "--from", "report", "--to", "report"],
    display=True, timeout=3600))

TOOL_NAMES = (RUN.name, REPORT.name)
