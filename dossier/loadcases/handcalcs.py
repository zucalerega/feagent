"""The hand calculations of the plan: every spec of the component, for every load case it applies to.

`dossier.handcalc` decides whether a spec applies (`applies(spec, load_case)`: the spec's `when` against
the load case) and evaluates it (`evaluate(spec, project, ctx, geo, load_case)`). This module calls both for
each load case of the plan, lists the result ids in `LoadCase.hand_calcs`, and adds the result keys of the
comparisons to `LoadCase.result_keys`, so that the analysis gives the numbers the hand calculation is
compared with.

A hand calculation is a check of the analysis, not a part of it. A hand calculation that fails must not
stop the plan: the failure becomes a finding (category `model`, major) with the error text, and the load
case stays in the plan without that check. A spec that applies to no load case becomes a minor finding,
because the check its author wanted does not take place.
"""

from __future__ import annotations

from typing import Any

from dossier.model import HandCalcResult, HandCalcSpec, LoadCase
from dossier.project import Project
from dossier.records import Journal

RULE = "plan.hand_calc"


def _error(e: BaseException) -> str:
    text = " ".join(str(e).split())
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def _when(spec: HandCalcSpec) -> str:
    when = spec.when
    return "; ".join(x for x in (
        f"analysis: {', '.join(when.analysis)}" if when.analysis else "",
        f"needs: {', '.join(when.needs)}" if when.needs else "",
        f"phases: {', '.join(when.phases)}" if when.phases else "") if x) or "no condition"


def apply(project: Project, ctx, geo: dict | None, cases: list[LoadCase], journal: Journal,
          engine: Any = None) -> list[HandCalcResult]:
    """Evaluate the hand calculations of every load case. `engine` is `dossier.handcalc` (or a stand-in
    with the same `applies` and `evaluate`); it is imported here when it is not given."""
    comp = project.subject
    if not comp.hand_calcs:
        return []
    if engine is None:
        try:
            from dossier import handcalc as engine          # noqa: PLC0415  the engine of another stage
            missing = [name for name in ("applies", "evaluate") if not callable(getattr(engine, name, None))]
            if missing:
                raise ImportError(f"dossier.handcalc has no {' and no '.join(name + '()' for name in missing)}")
        except ImportError as e:
            journal.find(
                rule=RULE, severity="major", category="model", subject=comp.id, key="hand-calc-engine",
                title="The plan has no hand calculations",
                statement=f"The plan cannot evaluate the {len(comp.hand_calcs)} hand calculations of {comp.id}.",
                rationale=f"The import of the hand-calculation engine dossier.handcalc stopped with this error: "
                          f"{_error(e)}",
                suggestion="Install the hand-calculation engine. Then make the plan again.",
                values={"error": _error(e)})
            return []

    results: list[HandCalcResult] = []
    seen: set[str] = set()          # result ids
    applied: set[str] = set()       # specs that apply to a load case
    broken: set[str] = set()        # specs whose `when` cannot be evaluated
    for lc in cases:
        for spec in comp.hand_calcs:
            if spec.id in broken:
                continue
            try:
                if not engine.applies(spec, lc):
                    continue
            except Exception as e:  # noqa: BLE001  one bad spec must not stop the plan
                broken.add(spec.id)
                journal.find(
                    rule=RULE, severity="major", category="model", subject=spec.id, key="when-failed",
                    title=f"The condition of the hand calculation {spec.id} gives an error",
                    statement=f"The plan cannot find the load cases of the hand calculation {spec.id} ({spec.title}).",
                    rationale=f"The `when` condition of the spec is: {_when(spec)}. Its evaluation for the load case "
                              f"{lc.id} stopped with this error: {_error(e)}",
                    suggestion=f"Correct the `when` condition of {spec.id} in the component file. Then make the "
                               f"plan again.",
                    values={"error": _error(e), "load_case": lc.id})
                continue
            applied.add(spec.id)
            try:
                res = engine.evaluate(spec, project, ctx, geo, lc)
                if not isinstance(res, HandCalcResult):
                    raise TypeError(f"evaluate() gave {type(res).__name__}, not a HandCalcResult")
                if res.id in seen:
                    raise ValueError(f"the result id {res.id} is in the plan already; a result id must name "
                                     f"the spec and the load case")
            except Exception as e:  # noqa: BLE001  one bad hand calculation must not stop the plan
                journal.find(
                    rule=RULE, severity="major", category="model", subject=spec.id, key=f"{lc.id}-failed",
                    title=f"The hand calculation {spec.id} gives no result for {lc.id}",
                    statement=f"The hand calculation {spec.id} ({spec.title}) gives no result for the load case "
                              f"{lc.id}.",
                    rationale=f"The evaluation stopped with this error: {_error(e)}",
                    suggestion=f"Correct the inputs or the `when` condition of {spec.id} in the component file. "
                               f"Then make the plan again.",
                    values={"error": _error(e), "load_case": lc.id, "model": spec.model},
                    links=[("about", lc.id)])
                continue
            if res.load_case is None:
                res = res.model_copy(update={"load_case": lc.id})
            seen.add(res.id)
            results.append(res)
            lc.hand_calcs.append(res.id)
            for key in [c.result_key for c in res.comparisons] + [c.with_ for c in spec.compare]:
                if key and key not in lc.result_keys:
                    lc.result_keys.append(key)

    for spec in comp.hand_calcs:
        if spec.id in applied or spec.id in broken:
            continue
        journal.find(
            rule=RULE, severity="minor", category="model", subject=spec.id, key="not-used",
            title=f"The hand calculation {spec.id} applies to no load case",
            statement=f"The hand calculation {spec.id} ({spec.title}) applies to no load case of the plan.",
            rationale=f"The `when` condition of the spec is: {_when(spec)}. No load case of the plan agrees with "
                      f"it, thus this check of the analysis does not take place.",
            suggestion=f"Change the `when` condition of {spec.id}, or remove the spec.",
            values={"when": spec.when.model_dump(mode="json")})
    return results
