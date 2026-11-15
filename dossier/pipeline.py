"""The pipeline: the stages in order, and the contract of each engine.

    context    dossier.context.build(project) -> Context                         every number evaluated
    geometry   dossier.geometry.build(project, options) -> (dict, Journal)        features -> CAD faces, measured
    validate   dossier.system.check.check(project, ctx, geo) -> Journal           is the system model sound?
               dossier.requirements.validate.validate(project, ctx, geo) -> Journal   are the requirements right?
    plan       dossier.loadcases.plan(project, ctx, geo) -> Plan                  load cases + hand calculations
    analyze    per load case, by analysis type:
               dossier.analysis.structural.run(...) | thermal.run(...) | fatigue.run(...) | flow.run(...)
               -> Analysis(record, journal); then dossier.handcalc.compare(...) adds the hand-calculation checks
    assess     dossier.compliance.assess(project, ctx) -> Assessment               margins and verdicts
    report     dossier.report.build(project, options) -> dict of paths             the linked STE report

Every engine is a function of the project, the context and the stage files before it; it returns records
and a Journal (its decisions, assumptions and findings). Only this module writes stage files. A stage
whose engine is missing or fails raises StageError with the reason; `run` stops there.

An engine never guesses silently: a choice is a Decision (with its rationale and the alternatives), a
thing taken as true is an Assumption (with its impact), a problem in the inputs is a Finding.
"""

from __future__ import annotations

import importlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from dossier import context as context_mod
from dossier.model import AnalysisRecord, ComplianceItem, HandCalcResult, LoadCase, Origin, RequirementVerdict
from dossier.project import Project, ProjectError
from dossier.records import Journal

STAGES = ("context", "geometry", "validate", "plan", "analyze", "assess", "report")

# analysis type -> the module whose run() analyses a load case of that type
ANALYSIS_ENGINES = {
    "static_structural": "dossier.analysis.structural",
    "thermo_structural": "dossier.analysis.structural",
    "thermal_steady": "dossier.analysis.thermal",
    "fatigue": "dossier.analysis.fatigue",
    "flow": "dossier.analysis.flow",
}


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"stage {stage}: {message}")


@dataclass
class Options:
    """Options of a run. `quick` is for tests and first looks: coarse meshes, two levels, no pictures."""

    quick: bool = False
    pictures: bool = True
    levels: int = 3                 # meshes of a convergence study
    node_cap: int = 120_000         # nodes of the finest mesh
    only: list[str] = field(default_factory=list)   # load case ids to analyse (empty: all)
    force: bool = False             # analyse a load case again even when its analysis file is current
    record_runs: bool = False       # also write solves to the dashboard's run store (the 3D viewer)
    log: Callable[[str], None] = print

    def __post_init__(self):
        if self.quick:
            self.levels = min(self.levels, 2)
            self.node_cap = min(self.node_cap, 25_000)
            self.pictures = False


@dataclass
class Plan:
    load_cases: list[LoadCase]
    hand_calcs: list[HandCalcResult]
    journal: Journal


@dataclass
class Analysis:
    record: AnalysisRecord
    journal: Journal


@dataclass
class Assessment:
    items: list[ComplianceItem]
    verdicts: list[RequirementVerdict]
    journal: Journal


def _engine(module: str, name: str, stage: str):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and module.startswith(e.name):
            raise StageError(stage, f"the engine {module} is not installed in this version") from e
        raise
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise StageError(stage, f"{module} has no {name}()") from e


def _journal_payload(j: Journal) -> dict:
    return {"findings": j.findings, "decisions": j.decisions, "assumptions": j.assumptions}


# --------------------------------------------------------------------------- stages


def stage_context(project: Project, opts: Options) -> dict:
    ctx = context_mod.run_stage(project)
    return {"parameters": len(ctx.params), "effects": len(ctx.effects)}


def load_geometry(project: Project) -> dict:
    doc = project.read_stage("geometry")
    if doc is None:
        raise StageError("geometry", "run the geometry stage first")
    return doc["geometry"]


def stage_geometry(project: Project, opts: Options) -> dict:
    build = _engine("dossier.geometry", "build", "geometry")
    geo, journal = build(project, opts)
    # The meshes belong to the geometry stage: every analysis engine uses the same levels, and the decision
    # for their sizes and the findings of the mesh checks are records a person reviews.
    meshes: list = []
    try:
        levels = _engine("dossier.meshing", "levels_with_journal", "geometry")
        meshes, mesh_journal = levels(project, geo, opts)
        journal.extend(mesh_journal)
    except StageError:
        pass
    geo["meshes"] = [m.model_dump(mode="json") for m in meshes]
    project.write_stage("geometry", geometry=geo, **_journal_payload(journal))
    return {"features": len(geo.get("features", {})), "problems": len(geo.get("problems", [])),
            "findings": len(journal.findings)}


def stage_validate(project: Project, opts: Options) -> dict:
    ctx = context_mod.load(project)
    geo = load_geometry(project)
    journal = Journal("validate", "engine:validate")
    journal.extend(_engine("dossier.system.check", "check", "validate")(project, ctx, geo))
    journal.extend(_engine("dossier.requirements.validate", "validate", "validate")(project, ctx, geo))
    project.write_stage("validate", **_journal_payload(journal))
    try:
        svg = _engine("dossier.system.diagram", "render", "validate")(project, ctx)
        (project.out / "system.svg").write_text(svg)
    except StageError:
        pass
    by = {}
    for f in journal.findings:
        by[f.severity] = by.get(f.severity, 0) + 1
    return {"findings": by}


def stage_plan(project: Project, opts: Options) -> dict:
    ctx = context_mod.load(project)
    geo = load_geometry(project)
    plan: Plan = _engine("dossier.loadcases", "plan", "plan")(project, ctx, geo)
    project.write_stage("plan", load_cases=plan.load_cases, hand_calcs=plan.hand_calcs, **_journal_payload(plan.journal))
    return {"load_cases": [f"{lc.id} ({lc.analysis_type}, {lc.status})" for lc in plan.load_cases],
            "hand_calcs": len(plan.hand_calcs)}


def analysis_order(load_cases: list[LoadCase]) -> list[LoadCase]:
    """Load cases in an order where every case comes after the cases it uses or scales from."""
    by_id = {lc.id: lc for lc in load_cases}
    done: list[str] = []

    def visit(lc: LoadCase, chain: tuple[str, ...] = ()):
        if lc.id in done:
            return
        if lc.id in chain:
            raise StageError("analyze", f"load cases use each other in a circle: {' -> '.join((*chain, lc.id))}")
        for dep in [*lc.uses, *([lc.scaled_from] if lc.scaled_from else [])]:
            if dep in by_id:
                visit(by_id[dep], (*chain, lc.id))
        done.append(lc.id)

    for lc in load_cases:
        visit(lc)
    return [by_id[i] for i in done]


def analyze_one(project: Project, ctx, geo: dict, lc: LoadCase, opts: Options) -> Analysis:
    """Analyse one load case with the engine of its type. A failure of the engine comes back as a
    record with status "failed" and the reason, so one bad load case does not stop the others."""
    workdir = project.analysis_dir(lc.id)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if lc.status == "not_supported":
        rec = AnalysisRecord(id=f"AN-{lc.id[3:] if lc.id.startswith('LC-') else lc.id}", title=lc.title, load_case=lc.id,
                             analysis_type=lc.analysis_type, status="not_supported", message=lc.note,
                             origin=Origin(by="engine:pipeline", stage="analyze"))
        return Analysis(rec, Journal("solve", "engine:pipeline"))
    module = "dossier.analysis.scaled" if lc.scaled_from else ANALYSIS_ENGINES[lc.analysis_type]
    try:
        run = _engine(module, "run", "analyze")
        out: Analysis = run(project, ctx, geo, lc, workdir, opts)
    except StageError as e:
        rec = AnalysisRecord(id=f"AN-{lc.id[3:] if lc.id.startswith('LC-') else lc.id}", title=lc.title, load_case=lc.id,
                             analysis_type=lc.analysis_type, status="not_supported", message=str(e),
                             origin=Origin(by="engine:pipeline", stage="analyze"))
        return Analysis(rec, Journal("solve", "engine:pipeline"))
    except Exception as e:  # noqa: BLE001  the reason goes into the record and the log
        (workdir / "error.log").write_text(traceback.format_exc())
        rec = AnalysisRecord(id=f"AN-{lc.id[3:] if lc.id.startswith('LC-') else lc.id}", title=lc.title, load_case=lc.id,
                             analysis_type=lc.analysis_type, status="failed",
                             message=f"{type(e).__name__}: {e} (see analyses/{lc.id}/error.log)",
                             origin=Origin(by="engine:pipeline", stage="analyze"))
        return Analysis(rec, Journal("solve", "engine:pipeline"))
    if out.record.runtime_s is None:
        out.record.runtime_s = round(time.time() - t0, 1)
    return out


def stage_analyze(project: Project, opts: Options) -> dict:
    ctx = context_mod.load(project)
    geo = load_geometry(project)
    load_cases = project.collection("load_cases")
    if not load_cases:
        raise StageError("analyze", "the plan has no load cases; run the plan stage first")
    hand_calcs = project.collection("hand_calcs")
    digest = project.inputs_digest()
    summary = {}
    for lc in analysis_order(load_cases):
        if opts.only and lc.id not in opts.only:
            continue
        path = project.analysis_dir(lc.id) / "analysis.json"
        old = project.read_stage("analyze", path)
        if old and not opts.force and old.get("inputs_digest") == digest and old.get("options") == _opt_key(opts) \
                and old.get("analyses") and old["analyses"][0].get("status") == "done":
            summary[lc.id] = "current"
            continue
        opts.log(f"analyze {lc.id}: {lc.title} [{lc.analysis_type}{', scaled from ' + lc.scaled_from if lc.scaled_from else ''}]")
        out = analyze_one(project, ctx, geo, lc, opts)
        if out.record.status == "done":
            try:
                compare = _engine("dossier.handcalc", "compare", "analyze")
                for hc in hand_calcs:
                    if hc.load_case == lc.id:
                        out.record.checks += compare(hc, out.record, project)
            except StageError:
                pass
        project.write_stage("analyze", path, options=_opt_key(opts), analyses=[out.record], **_journal_payload(out.journal))
        summary[lc.id] = out.record.status + (f": {out.record.message}" if out.record.message else "")
        opts.log(f"  -> {summary[lc.id]}")
    return summary


def _opt_key(opts: Options) -> dict:
    return {"quick": opts.quick, "levels": opts.levels, "node_cap": opts.node_cap}


def stage_assess(project: Project, opts: Options) -> dict:
    ctx = context_mod.load(project)
    out = _engine("dossier.compliance", "assess", "assess")(project, ctx)
    project.write_stage("assess", compliance=out.items, verdicts=out.verdicts, **_journal_payload(out.journal))
    return {v.requirement: v.verdict + (f" (margin {v.margin:+.2f})" if v.margin is not None else "") for v in out.verdicts}


def stage_report(project: Project, opts: Options) -> dict:
    return _engine("dossier.report", "build", "report")(project, opts)


_RUNNERS = {"context": stage_context, "geometry": stage_geometry, "validate": stage_validate, "plan": stage_plan,
            "analyze": stage_analyze, "assess": stage_assess, "report": stage_report}


def run(project: Project | str, first: str = "context", last: str = "report", opts: Options | None = None) -> dict:
    """Run the stages from `first` to `last`. Returns {stage: summary}. Raises StageError / ProjectError."""
    opts = opts or Options()
    if not isinstance(project, Project):
        project = Project.load(project)
    i0, i1 = STAGES.index(first), STAGES.index(last)
    out: dict[str, Any] = {}
    for stage in STAGES[i0:i1 + 1]:
        t0 = time.time()
        opts.log(f"== {stage}")
        try:
            out[stage] = _RUNNERS[stage](project, opts)
        except (StageError, ProjectError):
            raise
        except Exception as e:  # noqa: BLE001
            raise StageError(stage, f"{type(e).__name__}: {e}\n{traceback.format_exc()}") from e
        opts.log(f"   {stage} done in {time.time() - t0:.1f} s: {out[stage]}")
    return out
