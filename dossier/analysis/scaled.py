"""A proportional load case: its results are a multiple of the results of another analysis. No solve.

    run(project, ctx, geo, lc, workdir, opts) -> pipeline.Analysis     for a load case with `scaled_from`

The plan stage marks a load case as scaled when every mechanical load is the same multiple (`lc.scale`) of
the loads of the base load case, with the same supports and no thermal load. A linear static analysis is
then proportional: stress and displacement results are `scale` times the base results. The von Mises
stress and the largest principal stress scale the same way because the scale is positive.

What the record holds
  results     every stress and displacement result of the base analysis times `scale`: value, per_level,
              extrapolated value and the absolute uncertainty are scaled; status, uncertainty (relative),
              observed order, location and `near` carry over; `method` is "scaled". A base result of another quantity is not copied; the
              message names it.
  checks      one check that the base analysis is done and that its own checks passed. A base check that
              did not pass fails this check too (a failed check is never hidden).
  mesh, solver, material   those of the base analysis. `bcs` is empty: the model is the model of the base.
  artifacts   none. The stress field of a scaled load case is the field of the base analysis times the
              scale (the fatigue engine reads it that way).

The engine fails with a clear message (status "failed", no result) when the base analysis file is missing,
is not done, was made from other input files, is not a static structural analysis, or when the scale is
not a positive number.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from dossier.model import AnalysisRecord, Check, Link, LoadCase, Origin, ResultValue
from dossier.project import make_id
from dossier.records import Journal

# quantities that are proportional to the loads of a linear static analysis
LINEAR = ("von_mises", "max_principal", "sxx", "syy", "szz", "sxy", "syz", "sxz", "displacement",
          "radial_displacement")


def _analysis_id(lc: LoadCase) -> str:
    return "AN-" + (lc.id[3:] if lc.id.startswith("LC-") else lc.id)


def _record(lc: LoadCase, status: str, message: str, **kw) -> AnalysisRecord:
    return AnalysisRecord(id=_analysis_id(lc), title=lc.title, load_case=lc.id, analysis_type=lc.analysis_type,
                          status=status, message=message, origin=Origin(by="engine:scaled", stage="analyze"), **kw)


def scale_result(r: ResultValue, scale: float, base_id: str) -> ResultValue:
    return r.model_copy(update={
        "value": None if r.value is None else r.value * scale,
        "per_level": [v * scale for v in r.per_level],
        "extrapolated": None if r.extrapolated is None else r.extrapolated * scale,
        "uncertainty_abs": None if r.uncertainty_abs is None else r.uncertainty_abs * abs(scale),
        "method": "scaled",
        "note": (f"{scale:.6g} times the value of {base_id}. " + (r.note or "")).strip()})


def run(project, ctx, geo: dict, lc: LoadCase, workdir: Path, opts):
    """The results of a scaled load case from the analysis of its base load case."""
    from dossier.pipeline import Analysis

    t0 = time.time()
    j = Journal("solve", "engine:scaled")

    def fail(message: str) -> "Analysis":
        return Analysis(_record(lc, "failed", message), j)

    if not lc.scaled_from:
        return fail(f"The load case {lc.id} has no base load case. The scaled engine needs `scaled_from`.")
    scale = float(lc.scale)
    if not (math.isfinite(scale) and scale > 0.0):
        return fail(f"The scale of the load case {lc.id} is {lc.scale}. The scale must be a positive number.")
    path = project.analysis_dir(lc.scaled_from) / "analysis.json"
    doc = project.read_stage("analyze", path)
    if doc is None or not doc.get("analyses"):
        return fail(f"The base analysis of {lc.scaled_from} is not in the project. Analyse {lc.scaled_from} first.")
    base = AnalysisRecord.model_validate(doc["analyses"][0])
    if base.status != "done":
        return fail(f"The base analysis {base.id} has the status {base.status}, so it gives no results to scale."
                    + (f" Its message is: {base.message}" if base.message else ""))
    if doc.get("inputs_digest") not in (None, project.inputs_digest()):
        return fail(f"The base analysis {base.id} comes from other input files. Analyse {lc.scaled_from} again first.")
    if base.analysis_type != "static_structural":
        return fail(f"The base analysis {base.id} has the type {base.analysis_type}. Only the results of a static "
                    "structural analysis are in proportion to the loads.")

    results, left_out = {}, []
    for key, r in base.results.items():
        if key.split(".")[0] in LINEAR and r.value is not None:
            results[key] = scale_result(r, scale, base.id)
        else:
            left_out.append(key)
    missing = [k for k in lc.result_keys if k not in results]
    bad = [c for c in base.checks if c.passed is False]
    check = Check(
        id=make_id("CHK", _analysis_id(lc), "base"), kind="scaling",
        statement=(f"The base analysis {base.id} is done. {len(base.checks) - len(bad)} of its {len(base.checks)} "
                   "checks did not fail."
                   + (f" These checks failed: {', '.join(c.id for c in bad)}." if bad else "")),
        passed=not bad, value=float(len(bad)), limit=0.0, link=base.id)
    message = []
    if left_out:
        message.append("The engine does not scale these results of the base analysis: " + ", ".join(left_out) + ".")
    if missing:
        message.append("The base analysis does not give these results of the load case: " + ", ".join(missing) + ".")
    statement = (f"The engine did not solve the load case {lc.id}. Its loads are {scale:.6g} times the loads of "
                 f"{lc.scaled_from}, with the same supports. A linear static analysis gives results in proportion "
                 f"to the loads. Each result is {scale:.6g} times the result of {base.id}. The convergence status and "
                 f"the relative uncertainty are those of {base.id}. The model and the meshes are those of {base.id}.")
    rec = _record(lc, "done", " ".join(message), statement=statement, solver=f"{base.solver}; results scaled by {scale:.6g}",
                  material=dict(base.material), mesh=list(base.mesh), results=results, checks=[check],
                  links=[Link(rel="derives_from", to=base.id, note=f"scale {scale:.6g}")],
                  runtime_s=round(time.time() - t0, 3))
    return Analysis(rec, j)
