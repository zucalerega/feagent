"""The hand-calculation engine: a specification of a project -> a record, and a record against an analysis.

    applies(spec, load_case)                      does the `when` of the specification agree with the load case?
    evaluate(spec, project, ctx, geo, load_case)  the HandCalcResult: inputs with their sources, steps, outputs,
                                                  assumptions, warnings, and the comparisons the analysis must serve
    compare(hc, analysis, project)                one Check per comparison: analysis against hand calculation
    compared(hc, analysis, project)               the same comparisons with `fe_value`, `ratio` and `agree` filled

Inputs. Each input of the specification is a dossier.model.Value: a literal ("22 mm") or an expression of
project parameters, of the material (E, nu, alpha, k_th, rho, Sy, Su) and of the load case through
``load("INT-ID", "pressure")`` and ``feat("FT-ID", "radius")``. The engine checks the dimension of each input
against the model and converts it to SI. An input that the specification does not give takes the model's
default name from the project context (nu, E, k_th, ...), or for `fatigue.sigma_f` and `fatigue.b` the fatigue
data of the subject's material. Every input is recorded with where its number comes from, and the record
links to the parameters, the effects and the features that it reads.

Comparisons. A comparison has a band for the ratio analysis / hand calculation (from the specification, or
the project's `policy.handcalc_band`), or an absolute tolerance. `compare` never calls a comparison a pass
when it cannot be made: no such result, a result without a value, a hand value of zero, or an analysis
result that is itself a hand calculation give `passed = None`.
"""

from __future__ import annotations

import math
from typing import Any

from dossier import units
from dossier.handcalc.base import HandCalcError, Input, Model, get, shown
from dossier.model import (AnalysisRecord, CalcInput, CalcOutput, Check, Comparison, HandCalcResult, HandCalcSpec,
                           Link, LoadCase, Origin)
from dossier.project import make_id

ORIGIN_BY = "engine:handcalc"


# --------------------------------------------------------------------------- applies


def applies(spec: HandCalcSpec, load_case: LoadCase) -> bool:
    """The `when` of a specification against a load case. `analysis`: the analysis type of the load case
    must be in the list (empty: any type). `phases`: the phase of the load case must be in the list (empty:
    any). `needs`: every "kind@INTERFACE" must be a load of the load case with a value ("kind" alone: on any
    interface; a support has no value, thus "support@INTERFACE" asks that the interface holds the part)."""
    when = spec.when
    if when.analysis and load_case.analysis_type not in when.analysis:
        return False
    if when.phases and load_case.phase not in when.phases:
        return False
    for need in when.needs:
        kind, _, interface = need.partition("@")
        kind, interface = kind.strip(), interface.strip()
        if kind == "support":
            ok = (not interface and bool(load_case.supports)) or interface in load_case.supports or any(
                x.kind == "support" and (not interface or x.interface == interface) for x in load_case.loads)
        else:
            ok = any(x.kind == kind and x.si is not None and (not interface or x.interface == interface)
                     for x in load_case.loads)
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- evaluate


class _Reader:
    """The load() and feat() callables of the input expressions, with a note of what each call reads."""

    def __init__(self, ctx, geo: dict | None, load_case: LoadCase | None):
        loads = list(load_case.loads) if load_case is not None else []
        inner = ctx.functions(loads, geo)
        self.effects: list[str] = []
        self.features: list[str] = []

        def load(interface: str, kind: str):
            if load_case is None:
                raise units.UnitsError(f'load("{interface}", "{kind}"): this hand calculation has no load case')
            q = inner["load"](interface, kind)
            self.effects += [x.effect for x in loads if x.interface == interface and x.kind == kind
                             and x.si is not None]
            return q

        def feat(feature: str, prop: str):
            if geo is None:
                raise units.UnitsError(f'feat("{feature}", "{prop}"): there is no geometry; the geometry stage '
                                       f'must run before this hand calculation')
            q = inner["feat"](feature, prop)
            self.features.append(feature)
            return q

        self.functions = {"load": load, "feat": feat}

    def take(self) -> tuple[list[str], list[str]]:
        """What the calls read since the last take: (effect ids, feature ids), each once, in order."""
        out = (list(dict.fromkeys(self.effects)), list(dict.fromkeys(self.features)))
        self.effects, self.features = [], []
        return out


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    return text if text.endswith((".", "!", "?")) else text + "."


def _parameter_source(name: str, ctx) -> str:
    """Where a parameter of the context comes from, in words."""
    p = ctx.params[name]
    origin = "the system" if p.origin == "system" else p.origin
    text = f"The parameter {name} comes from {origin} (basis: {p.basis})."
    if p.source:
        text += " The source is: " + _sentence(p.source)
    elif p.expr:
        text += f" Its expression is {p.expr}."
    return text


def _fatigue_default(name: str, project) -> tuple[Any, str, str] | None:
    """(quantity, source text, material id) of `fatigue.sigma_f` or `fatigue.b` of the subject's material."""
    try:
        mat = project.material_of()
    except Exception:  # noqa: BLE001  no project, or no material: there is no default
        return None
    fat = getattr(mat, "fatigue", None)
    if fat is None or not name.startswith("fatigue."):
        return None
    attr = name.split(".", 1)[1]
    if attr == "sigma_f":
        q = fat.sigma_f.quantity()
    elif attr == "b":
        q = units.Q(float(fat.b))
    else:
        return None
    text = f"The fatigue data of the material {mat.id} give this value."
    if fat.source:
        text += " The source is: " + _sentence(fat.source)
    return q, text, mat.id


def _is_offset(q) -> bool:
    """True for a quantity in a unit with an offset (degC, degF): such a value is a temperature, not a difference."""
    try:
        return float(units.Q(0.0, q.units).to_base_units().magnitude) != 0.0
    except Exception:  # noqa: BLE001
        return False


def _to_si(q, unit: str, what: str, difference: bool = False) -> float:
    """A scalar quantity as a number in `unit` (SI), with errors that name the input."""
    if units.is_vector(q):
        raise HandCalcError(f"{what} must be one number, but its expression gives the vector {units.fmt(q)}. "
                            f"Take one component with [0], [1] or [2], or the size with norm().")
    if not units.compatible(q, unit):
        raise HandCalcError(f"{what} must have the unit {unit or '1 (a pure number)'}, but its value is "
                            f"{units.fmt(q)}.")
    if difference and _is_offset(q):
        raise HandCalcError(f"{what} is a temperature difference, but its value {units.fmt(q)} is a temperature. "
                            f"Give the difference in K.")
    v = float(units.to(q, unit).magnitude)
    if not math.isfinite(v):
        raise HandCalcError(f"{what} is not a finite number.")
    return v


def _read_input(inp: Input, spec: HandCalcSpec, model: Model, names: dict, reader: _Reader, ctx, project,
                load_case: LoadCase | None) -> tuple[Any, str, list[str]] | None:
    """(quantity, source text, ids of the records it reads) of one input, or None for an optional input
    that has no value."""
    what = f"The input {inp.name} of the hand calculation {spec.id}"
    value = spec.inputs.get(inp.name)
    if value is not None:
        try:
            q = value.quantity(names, reader.functions)
        except (units.UnitsError, ValueError) as e:
            raise HandCalcError(f"{what} has no value: {_sentence(e)}") from e
        effects, features = reader.take()
        if value.is_literal:
            return q, f"The inputs of {spec.id} give this value.", []
        expr = value.expr if isinstance(value.expr, str) else str(value.expr)
        read = sorted(n for n in value.names() if n in ctx.params)
        text = f"The expression is {expr}."
        if expr.strip() in ctx.params:
            text += " " + _parameter_source(expr.strip(), ctx)
        if effects:
            where = f"The load case {load_case.id}" if load_case is not None else "The load case"
            text += (f" {where} gives the effect {effects[0]}." if len(effects) == 1
                     else f" {where} gives the effects {', '.join(effects)}.")
        if features:
            text += (f" The geometry gives the feature {features[0]}." if len(features) == 1
                     else f" The geometry gives the features {', '.join(features)}.")
        return q, text, [*read, *effects, *features]
    if inp.default and inp.default in names:
        return (names[inp.default], f"The inputs of {spec.id} give no value. The model takes the parameter "
                f"{inp.default}. {_parameter_source(inp.default, ctx)}", [inp.default])
    if inp.default and inp.default.startswith("fatigue."):
        got = _fatigue_default(inp.default, project)
        if got is not None:
            q, text, mat_id = got
            return q, f"The inputs of {spec.id} give no value. {text}", [mat_id]
    if inp.optional:
        return None
    hint = (f" The project context has no parameter {inp.default}, which is the default of the model."
            if inp.default else "")
    raise HandCalcError(f"{what} has no value.{hint} Give {inp.name} ({inp.description}, unit "
                        f"{inp.unit or '1'}) in the inputs of {spec.id}.")


def evaluate(spec: HandCalcSpec, project, ctx, geo: dict | None, load_case: LoadCase | None) -> HandCalcResult:
    """Evaluate one hand-calculation specification, for one load case or without a load case.

    The record id is make_id("HC", <spec id without HC->, <load case id>), or the spec id when there is no
    load case. Raises HandCalcError (a ValueError) with a sentence that names the specification, the input
    and the cause: an unknown model or input, a missing input, a wrong dimension, an impossible number."""
    model = get(spec.model)
    known = [x.name for x in model.inputs]
    extra = sorted(set(spec.inputs) - set(known))
    if extra:
        raise HandCalcError(f"The hand calculation {spec.id} gives the input {', '.join(extra)}, which the model "
                            f"{model.name} does not have. The inputs of the model are: {', '.join(known)}.")
    names = ctx.names()
    reader = _Reader(ctx, geo, load_case)
    values: dict[str, float] = {}
    inputs: list[CalcInput] = []
    links: list[Link] = []

    def link(to: str, note: str) -> None:
        if to != spec.id and not any(l.to == to for l in links):
            links.append(Link(rel="uses", to=to, note=note))

    for inp in model.inputs:
        got = _read_input(inp, spec, model, names, reader, ctx, project, load_case)
        if got is None:
            continue
        q, source, reads = got
        v = _to_si(q, inp.unit, f"The input {inp.name} ({inp.description}) of the hand calculation {spec.id}",
                   inp.difference)
        values[inp.name] = v
        inputs.append(CalcInput(name=inp.name, si=v, unit=inp.unit, display=units.display(q), source=source))
        for r in reads:
            link(r, f"input {inp.name}")
    try:
        calc = model.calculate(values)
    except HandCalcError as e:
        raise HandCalcError(f"The hand calculation {spec.id} is not possible. {e}") from e

    outputs = [CalcOutput(name=o.name, si=calc.values[o.name], unit=o.unit, display=calc.show(o.name),
                          description=o.description) for o in model.outputs]
    comparisons: list[Comparison] = []
    for cmp in spec.compare:
        try:
            out = model.output(cmp.output)
        except HandCalcError as e:
            raise HandCalcError(f"The hand calculation {spec.id} compares an output that does not exist. {e}") from e
        band = (float(cmp.band[0]), float(cmp.band[1])) if cmp.band is not None else None
        tolerance = None
        if cmp.tolerance is not None:
            what = f"The tolerance of the output {cmp.output} of the hand calculation {spec.id}"
            try:
                tq = cmp.tolerance.quantity(names, reader.functions)
            except (units.UnitsError, ValueError) as e:
                raise HandCalcError(f"{what} has no value: {_sentence(e)}") from e
            reader.take()
            tolerance = _to_si(tq, out.unit, what, difference=True)
            if tolerance < 0.0:
                raise HandCalcError(f"{what} must be zero or more. It is {units.fmt(tq)}.")
        elif band is None:
            band = (float(project.policy.handcalc_band[0]), float(project.policy.handcalc_band[1]))
        if band is not None and not band[0] <= band[1]:
            raise HandCalcError(f"The band of the output {cmp.output} of the hand calculation {spec.id} must be "
                                f"[low, high]. It is [{band[0]:g}, {band[1]:g}].")
        comparisons.append(Comparison(output=cmp.output, result_key=cmp.with_, hand_value=calc.values[cmp.output],
                                      band=band, tolerance_si=tolerance, why=cmp.why))

    if load_case is not None:
        rid = make_id("HC", spec.id.removeprefix("HC-"), load_case.id)
        title = f"{spec.title}, load case {load_case.id}"
    else:
        rid, title = spec.id, spec.title
    return HandCalcResult(
        id=rid, title=title, links=links, origin=Origin(by=ORIGIN_BY, stage="plan"),
        spec=spec.id, model=model.name, load_case=load_case.id if load_case is not None else None,
        reference=model.reference, inputs=inputs, outputs=outputs, steps=list(calc.steps),
        assumptions=list(model.assumptions), warnings=list(calc.warnings), comparisons=comparisons)


# --------------------------------------------------------------------------- compare


def _number(x: float) -> str:
    return units.display(units.Q(float(x)))


def _one(hc: HandCalcResult, cmp: Comparison, analysis: AnalysisRecord, policy_band: tuple[float, float],
         check_id: str) -> tuple[Comparison, Check]:
    out = next((o for o in hc.outputs if o.name == cmp.output), None)
    unit = out.unit if out is not None else ""
    hand = cmp.hand_value
    hand_text = shown(hand, unit)
    said = f"The hand calculation {hc.id} gives {hand_text} for {cmp.output}."
    key = cmp.result_key

    def undecided(statement: str, fe: float | None = None) -> tuple[Comparison, Check]:
        return (cmp.model_copy(update={"fe_value": fe, "ratio": None, "agree": None}),
                Check(id=check_id, kind="hand_calc", statement=statement, passed=None, link=hc.id))

    rv = analysis.results.get(key)
    if rv is None:
        return undecided(f"The analysis {analysis.id} gives no result {key}. {said} The comparison is not possible.")
    if rv.value is None or not math.isfinite(rv.value):
        return undecided(f"The analysis {analysis.id} has no value for the result {key} (status: {rv.status}). "
                         f"{said} The comparison is not possible.")
    fe = float(rv.value)
    if rv.method == "hand_calc":
        return undecided(f"The result {key} of the analysis {analysis.id} comes from a hand calculation. {said} "
                         f"The comparison is not an independent check.", fe)
    fe_unit = rv.unit or unit
    if rv.unit and unit and not units.compatible(units.from_si(1.0, rv.unit), unit):
        return undecided(f"The analysis {analysis.id} gives the result {key} in {rv.unit}. The hand calculation "
                         f"{hc.id} gives {cmp.output} in {unit}. The comparison is not possible.", fe)
    fe_text = shown(fe, fe_unit)
    gives = f"The analysis {analysis.id} gives {fe_text} for {key}."
    status = "" if rv.status in ("converged", "estimate") else f" The analysis result has the status {rv.status}."

    if cmp.tolerance_si is not None:
        diff = abs(fe - hand)
        ok = diff <= cmp.tolerance_si
        statement = (f"{gives} {said} The difference is {shown(diff, unit)}. The accepted difference is "
                     f"{shown(cmp.tolerance_si, unit)}. The two results {'agree' if ok else 'do not agree'}.{status}")
        return (cmp.model_copy(update={"fe_value": fe, "ratio": None, "agree": ok}),
                Check(id=check_id, kind="hand_calc", statement=statement, passed=ok, value=diff,
                      limit=cmp.tolerance_si, link=hc.id))

    lo, hi = cmp.band if cmp.band is not None else policy_band
    if hand == 0.0:
        return undecided(f"{gives} The hand calculation {hc.id} gives zero for {cmp.output}. A ratio is not "
                         f"possible.", fe)
    ratio = fe / hand
    ok = lo <= ratio <= hi
    statement = (f"{gives} {said} The ratio of the analysis result to the hand calculation is {_number(ratio)}. "
                 f"The accepted band is {lo:g} to {hi:g}. The two results {'agree' if ok else 'do not agree'}.{status}")
    return (cmp.model_copy(update={"fe_value": fe, "ratio": ratio, "band": (lo, hi), "agree": ok}),
            Check(id=check_id, kind="hand_calc", statement=statement, passed=ok, value=ratio,
                  limit=hi if ratio >= 1.0 else lo, link=hc.id))


def _all(hc: HandCalcResult, analysis: AnalysisRecord, project) -> list[tuple[Comparison, Check]]:
    band = tuple(project.policy.handcalc_band) if project is not None else (0.8, 1.25)
    twice = {c.output for c in hc.comparisons if sum(1 for d in hc.comparisons if d.output == c.output) > 1}
    return [_one(hc, c, analysis, (float(band[0]), float(band[1])),
                 make_id("CHK", analysis.id, hc.id, c.output, c.result_key if c.output in twice else ""))
            for c in hc.comparisons]


def compare(hc: HandCalcResult, analysis: AnalysisRecord, project) -> list[Check]:
    """One Check (kind "hand_calc", link = the hand calculation) per comparison of the hand calculation.

    `value` is the ratio analysis / hand calculation, and `limit` the end of the band on its side; with a
    tolerance, `value` is the absolute difference and `limit` the tolerance. `passed` is None when the
    comparison cannot be made. The id is make_id("CHK", analysis id, hand calculation id, output); when two
    comparisons of one hand calculation use the same output, the result key is added."""
    return [check for _, check in _all(hc, analysis, project)]


def compared(hc: HandCalcResult, analysis: AnalysisRecord, project) -> list[Comparison]:
    """The comparisons of a hand calculation with `fe_value`, `ratio` and `agree` filled from an analysis
    (copies: the record does not change). For the report's table of hand calculation against analysis."""
    return [cmp for cmp, _ in _all(hc, analysis, project)]
