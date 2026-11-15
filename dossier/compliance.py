"""Margins of safety and verdicts: the assess stage.

    assess(project, ctx) -> dossier.pipeline.Assessment(items, verdicts, journal)

The inputs are the stage files: the load cases and hand calculations of the plan, the analysis records and
the findings. The output is one ComplianceItem for each criterion of each requirement in each load case
that serves it, and one RequirementVerdict for each requirement of the component under verification. An
engineer signs the verdicts, so a pass must be earned and every other verdict gives its reason.

What gets a verdict
    * A requirement whose verification is not "analysis": not_assessed, the statement names the method.
    * A requirement with no criterion: not_assessed, with the finding FND-<requirement>-NOT-VERIFIABLE.
    * Every other requirement: one item for each criterion and each load case whose `criteria` hold
      CriterionRef(requirement, index). A criterion that no load case serves counts as not assessed.

The response
    * The analysis of a load case is the last AnalysisRecord of the stage files with that `load_case`.
    * The result is `results[criterion.key()]`. For the location `scls` it is the aggregate key
      `<quantity>.<measure>@scls`. When the analysis has no such value, each line (`...@<SCL id>`, for
      the lines of the component and every `...@SCL-*` key) is assessed as a result of its own and the
      worst line governs: first by verdict (fail, inconclusive, pass), then by margin. A line of the
      component without a result makes a pass inconclusive. `result_key` then names the line.
    * `value` and `unit` are SI. A result in another unit is converted with `dossier.units`.

The numerical uncertainty (the contract with the analysis engines)
    dU = ResultValue.uncertainty_abs (in the unit of the value) when it is given,
         else ResultValue.uncertainty x |value| (`uncertainty` is relative to the value),
         else unknown: dU = 0, and the reasons say that the analysis gave no uncertainty.
    conservative_value = value + dU for "<=", value - dU for ">=" (the unsafe side).
    The note of a result is not read. ComplianceItem.uncertainty is always dU / |value|, so that
    conservative_value = value x (1 +- uncertainty) for every item, also for a temperature.

The factor and the limit
    factor   a number in the criterion, or a name: a parameter of the namespace (`ctx.params`, for example
             `yield_factor` of a requirement), else a factor of the policy. It must be a pure number > 0.
    limit    `allowable` names a material property (yield_strength -> Sy, ultimate_strength -> Su), or
             `limit` is a Value, evaluated with `ctx.names()`. It must be one number > 0.
    The result and the limit must have the same dimension. `factor_source` and `limit_source` say where the
    numbers come from; the item links to the parameters it reads.

The margin of safety (`margin_of_safety`)
    "<="   MS = limit / (factor x response) - 1       a response of zero or less has no finite margin
    ">="   MS = (response / factor) / limit - 1
    `margin` is the margin of the conservative value. It is None when the margin has no finite value.

The verdict of an item, in this order
    1. The factor or the limit has no valid value, or the result and the limit differ in dimension:
       not_assessed.
    2. No analysis record, a record that is not "done", or no value for the result: not_assessed, with the
       message of the record. Exception: a hand calculation of the same load case that has a comparison
       with the same result key gives an estimate: inconclusive, with the margin of the estimate.
    3. By the status of the result:
       converged      pass when the margin of the conservative value is >= 0; fail when the margin is
                      negative also with the uncertainty on the safe side; else inconclusive
       singular       inconclusive (the reasons say when the criterion fails on every mesh of the study)
       not_converged  with a stated uncertainty dU and k = policy.unconverged_margin_factor > 0: pass when the
                      margin is >= 0 with k x dU on the unsafe side; fail when the margin is negative with
                      k x dU on the safe side; else inconclusive. The stated uncertainty of a mesh sequence that
                      does not converge is already a conservative bound; k covers the weaker basis. `margin` and
                      `conservative_value` stay those of 1 x dU. With no uncertainty, or with k = 0: inconclusive.
                      The journal records this rule once (DEC-<policy>-UNCONVERGED-MARGIN).
       estimate       inconclusive
    4. A pass or a fail becomes inconclusive when the evidence is in doubt: a check with `passed is False`
       in the analysis or in an analysis it rests on (`uses`, `scaled_from`, the states of its duty
       cycles); such an analysis that is missing or not "done"; a plan file or an analysis file that is
       older than the input files (`Project.stale`). A check with `passed is None` is a note.

Caveats: the findings of severity error or major whose subject (or an "about" link) is a record the item
rests on. These are the requirement and its parameters; the load case, its phase and its effects, and
those of the load cases it uses; every parameter that the effects, the limit and the factor depend on,
with the requirement or material that states it; the material; the location of the result. A caveat does
not change the verdict. The statement names it.

The verdict of a requirement: fail if an item fails, else inconclusive if an item is inconclusive, else
not_assessed if an item or a criterion has no verdict, else pass. `governing` is the item with the
smallest margin among the items that decide, and `margin` is its margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dossier import units
from dossier.context import MATERIAL_NAMES
from dossier.model import (AnalysisRecord, ComplianceItem, Criterion, Finding, HandCalcResult, Link, LoadCase,
                           Origin, Requirement, RequirementVerdict, ResultValue)
from dossier.pipeline import Assessment
from dossier.project import make_id
from dossier.records import Journal

BY = "engine:compliance"
ESTIMATE_REASON = "The value is an estimate from a hand calculation. No analysis confirms it."
_STRESS = ("von_mises", "max_principal", "sxx", "syy", "szz", "sxy", "syz", "sxz")
_HOW_SINGULAR = "An engineer can use section stresses, model the real edge radius, or do an elastic-plastic analysis."
_HOW_SINGULAR_OTHER = "An engineer can model the real edge radius, or use a result that does not include the edge."
_HOW_REFINE = "An engineer can do the analysis again with a finer mesh or with more mesh levels."
_RANK = {"fail": 0, "inconclusive": 1, "not_assessed": 2, "pass": 3}
_TOL = 1e-9          # a margin this near to zero is zero: the rounding of the arithmetic
_MATERIAL_ATTR = {attr: name for name, attr in MATERIAL_NAMES.items()}     # yield_strength -> Sy


def margin_of_safety(response: float, limit: float, factor: float = 1.0, comparator: str = "<=") -> float:
    """MS = limit / (factor x response) - 1 for "<="; MS = (response / factor) / limit - 1 for ">=".
    MS >= 0 meets the criterion. The limit and the factor must be above zero. For "<=", a response of zero
    or less meets every such limit: the margin is +inf."""
    if not (limit > 0 and factor > 0):
        raise ValueError(f"the margin of safety needs a limit and a factor above zero (limit {limit}, factor {factor})")
    if comparator == "<=":
        return math.inf if response <= 0 else limit / (factor * response) - 1.0
    return (response / factor) / limit - 1.0


# --------------------------------------------------------------------------- words and numbers

_QUANTITY = {   # quantity -> (adjective, noun)
    "von_mises": ("von Mises", "stress"), "max_principal": ("maximum principal", "stress"),
    "sxx": ("sxx", "stress"), "syy": ("syy", "stress"), "szz": ("szz", "stress"),
    "displacement": ("", "displacement"), "radial_displacement": ("radial", "displacement"),
    "temperature": ("", "temperature"), "heat_flow": ("", "heat flow"), "pressure_drop": ("", "pressure drop"),
    "damage": ("fatigue", "damage"),
}
_MEASURE = {
    "peak": "peak {a} {n}", "max": "maximum {a} {n}", "min": "minimum {a} {n}", "mean": "mean {a} {n}",
    "range": "range of the {a} {n}", "membrane": "{a} membrane {n}", "membrane_bending": "{a} membrane plus bending {n}",
    "total": "total {a} {n}",
}


def _what(key: str) -> str:
    """A result key in words: von_mises.peak@part -> "peak von Mises stress in the part"."""
    head, _, where = key.partition("@")
    quantity, _, measure = head.partition(".")
    if quantity not in _QUANTITY or measure not in _MEASURE:
        return f"result {key}"
    adj, noun = _QUANTITY[quantity]
    form = "largest {a} {n}" if measure == "max" and adj.startswith("maximum") else _MEASURE[measure]
    name = " ".join(form.format(a=adj, n=noun).split())
    if where == "part":
        return f"{name} in the part"
    return f"{name} on the worst stress classification line" if where == "scls" else f"{name} on {where}"


def _show(value: float, unit: str, sig: int = 4) -> str:
    """An SI number for people: 71300000 Pa -> "71.3 MPa"."""
    return units.display(units.from_si(value, unit), sig)


def _show_delta(value: float, unit: str) -> str:
    """An uncertainty for people, with two digits (display() gives a temperature five)."""
    return units.fmt(units.from_si(value, unit), "K", 2) if unit == "K" else _show(value, unit, 2)


def _pct(u: float) -> str:
    return units.fmt(units.Q(100.0 * u), None, 2) + " %"


def _ms(m: float) -> str:
    """A margin of safety with its sign: +1.34, -0.008. Small margins get a third decimal."""
    if abs(m) >= 0.0995:
        return f"{m:+.2f}"
    return f"{m:+.3f}" if abs(m) >= 0.0005 or m == 0 else f"{m:+.1e}"


def _fac(f: float) -> str:
    s = units.fmt(units.Q(f), None, 4)
    return s if "." in s or "e" in s else s + ".0"


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _in_unit(unit: str) -> str:
    return f"in {unit}" if unit else "a pure number"


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)


def _to_si(value: float, unit: str) -> tuple[float, str]:
    """A number in `unit` -> (the number in SI, the SI unit)."""
    try:
        q = units.Q(float(value), unit or "")
        return float(units.si(q)), units.si_unit(q)
    except Exception as e:  # noqa: BLE001  pint raises several kinds
        raise units.UnitsError(f"unknown unit {unit!r}: {e}") from e


# --------------------------------------------------------------------------- the inputs of the stage


class _Inputs:
    """The stage files, read once."""

    def __init__(self, project, ctx):
        self.project, self.ctx = project, ctx
        self.cases: dict[str, LoadCase] = {lc.id: lc for lc in project.collection("load_cases")}
        self.analyses: dict[str, AnalysisRecord] = {a.load_case: a for a in project.collection("analyses")}
        self.hand_calcs: list[HandCalcResult] = project.collection("hand_calcs")
        self.findings: dict[str, Finding] = {f.id: f for f in project.collection("findings")}
        self.req_ids = {r.id for reqs in project.requirements.values() for r in reqs}
        self.stale = set(project.stale())
        self.unconverged = False        # a result with the status not converged and an uncertainty was assessed
        self.known = (set(project.inputs()) | set(ctx.params) | set(self.cases) | set(self.findings)
                      | {a.id for a in self.analyses.values()} | {h.id for h in self.hand_calcs})

    def is_stale(self, path) -> bool:
        return str(path.relative_to(self.project.root)) in self.stale

    def links(self, *links: Link) -> list[Link]:
        """Links to records that exist (a link to a missing record is a hole in the trace)."""
        return [l for l in links if l.to in self.known]

    def major(self, subjects: set[str]) -> list[str]:
        """The findings of severity error or major about one of `subjects`."""
        return [f.id for f in self.findings.values() if f.severity in ("error", "major")
                and (f.subject in subjects or any(l.rel == "about" and l.to in subjects for l in f.links))]

    def param_source(self, name: str) -> str:
        p = self.ctx.params[name]
        if p.origin == "system":
            return f"system parameter {name}"
        return f"parameter {name} of {p.origin}" if p.origin in self.req_ids else f"property {name} of {p.origin}"


@dataclass
class _Factor:
    value: float
    source: str
    names: set[str] = field(default_factory=set)       # the parameters it reads
    links: list[Link] = field(default_factory=list)


@dataclass
class _Limit:
    value: float                # SI
    unit: str                   # SI unit
    source: str
    word: str                   # allowable | limit | lower limit
    names: set[str] = field(default_factory=set)
    links: list[Link] = field(default_factory=list)


def _factor(crit: Criterion, label: str, req: Requirement, inp: _Inputs) -> _Factor | str:
    """The factor of safety of a criterion, or the problem in one sentence."""
    out = None
    if isinstance(crit.factor, str):
        name = crit.factor.strip()
        try:
            out = _Factor(float(name), f"criterion {label} of {req.id}")
        except ValueError:
            policy = inp.project.policy
            if name in inp.ctx.params:
                p = inp.ctx.params[name]
                if isinstance(p.si, list) or p.unit:
                    return f"Parameter {name} is {p.display}. A factor of safety must be a pure number."
                out = _Factor(float(p.si), inp.param_source(name), {name},
                              inp.links(Link(rel="uses", to=name, note="factor of safety")))
            elif name in policy.factors:
                out = _Factor(float(policy.factors[name].value), f"factor {name} of the policy {policy.id}", set(),
                              inp.links(Link(rel="uses", to=policy.id, note=f"factor of safety: {name}")))
            else:
                return f"The factor name {name} is not a parameter of the project and not a factor of the policy."
    else:
        given = "factor" in crit.model_fields_set
        out = _Factor(float(crit.factor), f"criterion {label} of {req.id}" if given else "the criterion gives no factor")
    if not (math.isfinite(out.value) and out.value > 0):
        return f"The factor of safety is {out.value:g} ({out.source}). A factor of safety must be above zero."
    return out


def _limit(crit: Criterion, label: str, req: Requirement, inp: _Inputs) -> _Limit | str:
    """The limit or allowable of a criterion in SI, or the problem in one sentence."""
    ctx = inp.ctx
    if crit.allowable is not None:
        attr = crit.allowable.strip()
        name = _MATERIAL_ATTR.get(attr, attr)
        p = ctx.params.get(name)
        if p is None or p.origin not in inp.project.materials or isinstance(p.si, list):
            return f"The allowable name {attr} is not a material property of the project."
        out = _Limit(float(p.si), p.unit, f"{attr.replace('_', ' ')} of {p.origin}", "allowable", {name},
                     inp.links(Link(rel="uses", to=name, note="allowable")))
    else:
        try:
            q = crit.limit.quantity(ctx.names())
        except (units.UnitsError, ValueError) as e:
            return f'The limit of criterion {label} has no value. The error is: "{e}"'
        if units.is_vector(q):
            return f"The limit of criterion {label} is a vector. A limit must be one number."
        names = {n for n in crit.limit.names() if n in ctx.params}
        expr = crit.limit.expr.strip() if isinstance(crit.limit.expr, str) else None
        if crit.limit.is_literal:
            source = f"criterion {label} of {req.id}"
        else:
            source = inp.param_source(expr) if expr in ctx.params else f"expression {expr} of criterion {label}"
        out = _Limit(float(units.si(q)), units.si_unit(q), source, "limit" if crit.comparator == "<=" else "lower limit",
                     names, inp.links(*(Link(rel="uses", to=n, note="limit") for n in sorted(names))))
    if not (math.isfinite(out.value) and out.value > 0):
        return (f"The {out.word} is {_show(out.value, out.unit)} ({out.source}). "
                f"The margin of safety needs a limit above zero.")
    return out


# --------------------------------------------------------------------------- one response against the limit


@dataclass
class _Eval:
    """One response against the limit, before the checks of the analysis."""

    key: str
    verdict: str = "not_assessed"
    cause: str = ""                       # why the verdict is not a pass, as a clause for "because ..."
    reasons: list[str] = field(default_factory=list)
    value: float | None = None            # SI
    unit: str = ""
    du: float | None = None               # absolute numerical uncertainty, SI. None: the analysis gave none
    rel: float | None = None              # the same relative to the value
    du_given_abs: bool = False
    status: str = ""                      # the status of the result
    finest: bool = False                  # the value is the value on the finest mesh of the study
    conservative: float | None = None
    margin: float | None = None           # of the conservative value; math.inf: no finite value
    nominal: float | None = None          # of the value without its uncertainty
    k: float | None = None                # not converged with an uncertainty: the factor on it (policy)
    k_margin: float | None = None         # the margin with k x the uncertainty: unsafe side, safe side for a fail

    @property
    def rank(self) -> tuple:
        return (_RANK[self.verdict], math.inf if self.margin is None else self.margin)

    def uncertainty_text(self) -> str:
        if not self.du:
            return ""
        return _show_delta(self.du, self.unit) if self.du_given_abs or self.rel is None else _pct(self.rel)


def _times(k: float) -> str:
    """A factor on the uncertainty in words: 2.0 -> "2 times", 1.5 -> "1.5 times"."""
    return units.fmt(units.Q(k), None, 4) + " times"


def _evaluate(rv: ResultValue, key: str, crit: Criterion, limit: _Limit, factor: _Factor, an_id: str,
              max_uncertainty: float, k: float = 0.0) -> _Eval:
    """The conservative value, the margins and the verdict that one result supports on its own."""
    ev = _Eval(key=key)
    if not _finite(rv.value) or rv.status == "not_run":
        ev.cause = f"analysis {an_id} gives no value for the result {key}"
        ev.reasons = [f"Analysis {an_id} gives no value for the result {key}."]
        if rv.note:
            ev.reasons.append(f'The note of the result is: "{rv.note}"')
        return ev
    try:
        value, ev.unit = _to_si(rv.value, rv.unit)
        scale = _to_si(1.0, rv.unit)[0] - _to_si(0.0, rv.unit)[0]      # of a difference: 1 for K and for degC
    except units.UnitsError:
        ev.cause = "the unit of the result is not a known unit"
        ev.reasons = [f"The result {key} has the unit {rv.unit}. The unit system does not know this unit."]
        return ev
    ev.value = value
    if not units.compatible(units.from_si(1.0, limit.unit), ev.unit):
        ev.cause = "the result and the limit do not have the same dimension"
        ev.reasons = [f"The result {key} is {_in_unit(ev.unit)}.", f"The {limit.word} is {_in_unit(limit.unit)}.",
                      "The margin of safety needs two values of the same dimension."]
        return ev

    if _finite(rv.uncertainty_abs):
        ev.du, ev.du_given_abs = abs(rv.uncertainty_abs) * scale, True
        ev.rel = ev.du / abs(value) if value else None
    elif _finite(rv.uncertainty):
        ev.rel = abs(rv.uncertainty)
        ev.du = ev.rel * abs(value)
    du = ev.du or 0.0
    unsafe = 1.0 if crit.comparator == "<=" else -1.0

    def ms(x: float) -> float:
        return margin_of_safety(x, limit.value, factor.value, crit.comparator)

    ev.conservative = value + unsafe * du
    ev.nominal, ev.margin, safe = ms(value), ms(ev.conservative), ms(value - unsafe * du)
    no_uncertainty = "The analysis gives no uncertainty for this result."
    levels = [_to_si(v, rv.unit)[0] for v in rv.per_level if _finite(v)]
    ev.status = rv.status
    ev.finest = bool(levels) and math.isclose(levels[-1], value, rel_tol=1e-9, abs_tol=0.0)
    note = [f'The note of the result is: "{rv.note}"'] if rv.note else []
    at_edge = _HOW_SINGULAR if key.partition(".")[0] in _STRESS else _HOW_SINGULAR_OTHER

    if rv.status == "converged":
        ev.reasons.append("The result has the status converged.")
        if ev.du is None:
            ev.reasons += [no_uncertainty, "The assessment uses the value as it is."]
        else:
            ev.reasons.append(f"The value with its uncertainty on the unsafe side is {_show(ev.conservative, ev.unit)}.")
        if ev.margin >= -_TOL:
            ev.verdict = "pass"
            if du and math.isfinite(ev.nominal):
                ev.reasons.append(f"The margin of safety is {_ms(ev.nominal)} without the uncertainty.")
        elif safe < -_TOL:
            ev.verdict, ev.cause = "fail", "the margin of safety is negative"
            if du:
                ev.reasons += [f"The margin of safety is {_ms(safe)} with the uncertainty on the safe side.",
                               "The criterion fails for each value in the uncertainty band."]
            else:
                ev.reasons.append("The margin of safety is negative.")
        else:
            ev.verdict, ev.cause = "inconclusive", "the margin is smaller than the numerical uncertainty"
            ev.reasons += [f"The margin of safety is {_ms(ev.nominal)} without the uncertainty and {_ms(ev.margin)} "
                           f"with the uncertainty on the unsafe side.",
                           "The margin is smaller than the numerical uncertainty.", _HOW_REFINE]
    elif rv.status == "singular":
        ev.verdict, ev.cause = "inconclusive", "the result is a singular peak"
        ev.reasons += ["The result has the status singular.",
                       "The value increases with each mesh refinement and has no upper bound."]
        if levels and ms(min(levels) if unsafe > 0 else max(levels)) < -_TOL:
            side = "the peak is above" if unsafe > 0 else "the value is below"
            ev.reasons += [f"The value on the coarsest mesh is {_show(levels[0], ev.unit)}.",
                           f"With the factor of safety, {side} the limit on every mesh of the study."]
        ev.reasons.append(at_edge)
    elif rv.status == "not_converged":
        ev.verdict, ev.cause = "inconclusive", "the result has the status not converged"
        ev.reasons.append("The result has the status not converged.")
        if _finite(rv.uncertainty) and not ev.du_given_abs:
            ev.reasons.append(f"The uncertainty of the result is {_pct(abs(rv.uncertainty))}, "
                              f"and the policy limit is {_pct(max_uncertainty)}.")
        elif ev.du is not None:
            ev.reasons.append(f"The uncertainty of the result is {ev.uncertainty_text()}.")
        if ev.du is not None and k > 0 and _finite(k):
            ev.k = k
            unsafe_k, safe_k = ms(value + unsafe * k * du), ms(value - unsafe * k * du)
            times = _times(k)
            if unsafe_k >= -_TOL:
                ev.verdict, ev.cause, ev.k_margin = "pass", "", unsafe_k
                ev.reasons += [f"The margin of safety is {_ms(unsafe_k)} with {times} this uncertainty on the unsafe side.",
                               f"The policy accepts this margin for a result with the status not converged."]
            elif safe_k < -_TOL:
                ev.verdict, ev.cause, ev.k_margin = "fail", "the margin of safety is negative", safe_k
                ev.reasons += [f"The margin of safety is {_ms(safe_k)} with {times} this uncertainty on the safe side.",
                               "The criterion fails for each value in this wider band."]
            else:
                ev.k_margin = unsafe_k
                ev.cause = f"the margin is smaller than {times} the uncertainty of a result with the status not converged"
                ev.reasons += [f"The margin of safety is {_ms(unsafe_k)} with {times} this uncertainty on the unsafe side "
                               f"and {_ms(safe_k)} on the safe side.",
                               f"The margin is smaller than {times} the uncertainty.", _HOW_REFINE]
        elif ev.du is not None:
            ev.reasons.append(_HOW_REFINE)
        else:                   # no mesh study, or a value that has no limit: the note of the analysis says which
            ev.reasons += [no_uncertainty, *note, _HOW_REFINE if len(levels) < 2 else at_edge]
    else:                                                   # estimate
        ev.verdict, ev.cause = "inconclusive", "the value is an estimate"
        ev.reasons += ["The analysis gives this value as an estimate.", *note, "An estimate does not confirm a criterion."]
    if rv.near:
        ev.reasons.append(f'The location of the {"peak" if rv.status == "singular" else "value"} is "{rv.near}".')
    return ev


def _response(crit: Criterion, lc: LoadCase, an: AnalysisRecord, limit: _Limit, factor: _Factor,
              inp: _Inputs) -> _Eval:
    """The result of the criterion in the analysis: the key itself, or for `scls` the worst line."""
    key = crit.key()
    max_u, k = inp.project.policy.max_uncertainty, inp.project.policy.unconverged_margin_factor
    ev = None
    if key in an.results:
        ev = _evaluate(an.results[key], key, crit, limit, factor, an.id, max_u, k)
        inp.unconverged |= ev.k is not None
        if ev.value is not None:
            return ev
    if crit.location == "scls":
        prefix = key[: -len("scls")]
        comp = inp.project.components.get(lc.component, inp.project.subject)
        lines = [s.id for s in comp.scls]
        lines += [k[len(prefix):] for k in an.results if k.startswith(prefix + "SCL-") and k[len(prefix):] not in lines]
        evals = [_evaluate(an.results[prefix + line], prefix + line, crit, limit, factor, an.id, max_u, k)
                 for line in lines if prefix + line in an.results]
        inp.unconverged |= any(e.k is not None for e in evals)
        usable = [e for e in evals if e.margin is not None]
        if usable:
            worst = min(usable, key=lambda e: e.rank)
            worst.reasons += [f"Analysis {an.id} has no result for all stress classification lines together.",
                              f"The assessment uses the worst of {len(usable)} lines: {worst.key.partition('@')[2]}."]
            missing = [s.id for s in comp.scls if prefix + s.id not in {e.key for e in usable}]
            if missing:
                worst.reasons.append(f"Analysis {an.id} has no result for {_join(missing)}.")
                if worst.verdict == "pass":
                    worst.verdict, worst.cause = "inconclusive", "one or more stress classification lines have no result"
            return worst
        if evals and ev is None:
            return evals[0]
    if ev is None:
        ev = _Eval(key=key, cause=f"analysis {an.id} has no result for {key}",
                   reasons=[f"Analysis {an.id} has no result with the key {key}."])
    return ev


def _estimate(crit: Criterion, lc: LoadCase, limit: _Limit, factor: _Factor,
              inp: _Inputs) -> tuple[_Eval, HandCalcResult, str] | None:
    """A hand calculation of the load case that gives the result of the criterion: an estimate."""
    key = crit.key()
    for hc in inp.hand_calcs:
        if hc.load_case != lc.id:
            continue
        for cmp in hc.comparisons:
            out = next((o for o in hc.outputs if o.name == cmp.output), None)
            if cmp.result_key != key or out is None or not _finite(cmp.hand_value):
                continue
            try:
                value, unit = _to_si(cmp.hand_value, out.unit)
            except units.UnitsError:
                continue
            if not units.compatible(units.from_si(1.0, limit.unit), unit):
                continue
            ev = _Eval(key=key, verdict="inconclusive", cause="the value is an estimate from a hand calculation",
                       value=value, unit=unit, conservative=value)
            ev.nominal = ev.margin = margin_of_safety(value, limit.value, factor.value, crit.comparator)
            return ev, hc, cmp.output
    return None


# --------------------------------------------------------------------------- what the result rests on


def _upstream(lc: LoadCase, cases: dict[str, LoadCase], scaled: bool = True) -> list[str]:
    """The load cases whose analyses feed the analysis of `lc`: the cases it uses, the states of its duty
    cycles and the case it is scaled from, and theirs in turn. With `scaled` False, a scaled case that has
    loads of its own stands for itself (the loads that feed it are its own)."""
    out: list[str] = []
    todo = [lc]
    while todo:
        c = todo.pop(0)
        deps = [*c.uses, *(s for cy in c.cycles for s in cy.states)]
        if c.scaled_from and (scaled or not c.loads):
            deps.append(c.scaled_from)
        for i in deps:
            if i != lc.id and i not in out:
                out.append(i)
                if i in cases:
                    todo.append(cases[i])
    return out


def _doubts(lc: LoadCase, an: AnalysisRecord, inp: _Inputs) -> tuple[list[tuple[str, list[str]]], list[str], list[Link]]:
    """What puts the evidence of a load case in doubt: (blockers, notes, links). A blocker is (cause,
    reasons): it turns a pass or a fail into inconclusive. A note does not change the verdict. The links
    go to the records of the failed checks (a hand calculation that does not agree)."""
    blockers: list[tuple[str, list[str]]] = []
    notes: list[str] = []
    links: list[Link] = []
    project = inp.project
    if inp.is_stale(project.stage_path("plan")):
        blockers.append(("the input files changed after the plan stage",
                         ["The input files changed after the plan stage wrote the load cases.",
                          "Do the plan stage and the analyses again."]))
    checked: list[str] = []
    failed = n_checks = 0
    for case_id, rec in [(lc.id, an)] + [(i, inp.analyses.get(i)) for i in _upstream(lc, inp.cases)]:
        if rec is None or rec.status != "done":
            blockers.append((f"load case {case_id} has no complete analysis",
                             [f"Load case {lc.id} uses the results of load case {case_id}.",
                              f"Load case {case_id} has no complete analysis."]))
            continue
        checked.append(rec.id)
        n_checks += len(rec.checks)
        if inp.is_stale(project.analysis_dir(case_id) / "analysis.json"):
            blockers.append(("the input files changed after the analysis",
                             [f"The input files changed after analysis {rec.id}.", "Do the analysis again."]))
        for c in rec.checks:
            says = [f'The check says: "{c.statement}"'] if c.statement else []
            if c.passed is False:
                failed += 1
                blockers.append(("a check of the analysis failed", [f"Check {c.id} of analysis {rec.id} failed.", *says]))
                links += inp.links(Link(rel="evidence", to=c.link or "", note=f"check {c.id} failed"))
            elif c.passed is None:
                notes.append(f"Check {c.id} of analysis {rec.id} has no result.")
    if n_checks and not failed:
        notes.insert(0, f"No check of {'analysis' if len(checked) == 1 else 'the analyses'} {_join(checked)} failed.")
    if not an.checks:
        notes.append(f"The record of analysis {an.id} has no check.")
    return blockers, notes, links


def _caveats(req: Requirement, crit: Criterion, lc: LoadCase, limit, factor, inp: _Inputs) -> list[str]:
    """The findings of severity error or major about a record the item rests on (see the module text)."""
    ctx, project = inp.ctx, inp.project
    comp = project.components.get(lc.component, project.subject)
    duty_cycles = {d.id: d for d in project.system.duty_cycles}
    subjects = {req.id, *req.parameters}
    effects: set[str] = set()
    todo: list[str] = [n for x in (limit, factor) if not isinstance(x, str) for n in x.names]
    for c in [lc] + [inp.cases[i] for i in _upstream(lc, inp.cases, scaled=False) if i in inp.cases]:
        subjects |= {c.id, *([c.phase] if c.phase else []), *c.supports}
        if c.id in inp.analyses:
            subjects.add(inp.analyses[c.id].id)
        effects |= {x.effect for x in c.loads}
        subjects |= {i for x in c.loads for i in (x.interface, *x.features) if i}
        for cy in c.cycles:                         # the count of a duty cycle feeds the damage
            subjects.add(cy.duty_cycle)
            if cy.duty_cycle in duty_cycles:
                todo += duty_cycles[cy.duty_cycle].count.names()
    subjects |= effects
    todo += [n for e in ctx.effects if e.id in effects for n in e.depends_on]
    names: set[str] = set()
    reached: dict[str, set[str]] = {}               # another requirement -> its parameters that the item reads
    while todo:                                     # every parameter the loads, the limit and the factor depend on
        n = todo.pop()
        if n in names or n not in ctx.params:
            continue
        names.add(n)
        todo += ctx.params[n].depends_on
        origin = ctx.params[n].origin
        if origin in project.materials:
            subjects.add(origin)
        elif origin in inp.req_ids and origin != req.id:
            reached.setdefault(origin, set()).add(n)
    subjects |= names
    if lc.analysis_type != "flow":
        subjects.add(comp.material)
    if crit.location == "scls":
        subjects |= {s.id for s in comp.scls}
    elif crit.location != "part":
        subjects.add(crit.location)
        subjects |= {f for it in comp.interfaces if it.id == crit.location for fs in it.features.values() for f in fs}

    out = []
    for f in inp.findings.values():
        if f.severity not in ("error", "major"):
            continue
        if f.subject in subjects or any(l.rel == "about" and l.to in subjects for l in f.links):
            out.append(f.id)
        elif f.subject in reached:      # on a requirement that states a number the item reads: is it about that number?
            named = {l.to for l in f.links} & set(project.requirement(f.subject).parameters)
            if not named or named & reached[f.subject]:
                out.append(f.id)
    return out


# --------------------------------------------------------------------------- one compliance item


def _verdict_sentence(verdict: str, cause: str, k: float | None = None) -> str:
    if k is not None and verdict == "pass":
        return f"The result has the status not converged, but the criterion passes with {_times(k)} its uncertainty."
    if k is not None and verdict == "fail":
        return f"The result has the status not converged, but the criterion fails also with {_times(k)} its uncertainty."
    if verdict == "pass":
        return "The criterion passes."
    if verdict == "fail":
        return "The criterion fails."
    return f"The verdict is inconclusive because {cause}." if verdict == "inconclusive" \
        else f"The assessment gives no verdict because {cause}."


def _item(req: Requirement, k: int, crit: Criterion, lc: LoadCase, item_id: str, inp: _Inputs,
          estimates: list) -> tuple[ComplianceItem, str]:
    """One criterion in one load case: (the item, the cause of its verdict as a clause)."""
    label = crit.id or str(k)
    key = crit.key()
    an = inp.analyses.get(lc.id)
    factor = _factor(crit, label, req, inp)
    limit = _limit(crit, label, req, inp)
    item = ComplianceItem(id=item_id, title=f"{req.title}: criterion {label} in load case {lc.id}",
                          origin=Origin(by=BY, stage="assess"), requirement=req.id, criterion=k, load_case=lc.id,
                          analysis=an.id if an is not None else None, result_key=key, comparator=crit.comparator,
                          caveats=_caveats(req, crit, lc, limit, factor, inp))
    for x in (factor, limit):
        if not isinstance(x, str):
            item.links += x.links
    if isinstance(factor, _Factor):
        item.factor, item.factor_source = factor.value, factor.source
    if isinstance(limit, _Limit):
        item.limit, item.limit_source, item.unit = limit.value, limit.source, limit.unit

    ev: _Eval | None = None
    hc: HandCalcResult | None = None
    verdict, cause, reasons = "not_assessed", "", []
    problems = [x for x in (factor, limit) if isinstance(x, str)]
    if problems:
        cause = "the factor of safety has no valid value" if isinstance(factor, str) else "the limit has no valid value"
        reasons = problems
    else:
        if an is None:
            cause = f"load case {lc.id} has no analysis"
            reasons = [f"Load case {lc.id} has no analysis record."]
            if lc.status == "not_supported" and lc.note:
                reasons.append(f'The note of the load case is: "{lc.note}"')
        elif an.status != "done":
            cause = f"analysis {an.id} has the status {an.status.replace('_', ' ')}"
            reasons = [f"Analysis {an.id} of load case {lc.id} has the status {an.status.replace('_', ' ')}."]
        else:
            ev = _response(crit, lc, an, limit, factor, inp)
            verdict, cause, reasons = ev.verdict, ev.cause, list(ev.reasons)
        if an is not None and an.message and (ev is None or ev.value is None):
            reasons.append(f'The message of the analysis is: "{an.message}"')

        if ev is None or ev.value is None:              # no analysis value: is there a hand calculation?
            found = _estimate(crit, lc, limit, factor, inp)
            if found is not None:
                ev, hc, output = found
                verdict, cause = ev.verdict, ev.cause
                why = reasons
                reasons = [ESTIMATE_REASON, f"Hand calculation {hc.id} gives the value as its output {output}.", *why]
                reasons += [f'The hand calculation says about the limits of its formula: "{w}"' for w in hc.warnings]
                if inp.is_stale(inp.project.stage_path("plan")):
                    reasons.append("The input files changed after the plan stage wrote the hand calculations.")
                item.analysis = None
                item.links += inp.links(Link(rel="evidence", to=hc.id, note="estimate from a hand calculation"))
                estimates.append((item_id, key, lc, hc, output, why))
        elif ev.margin is not None:                     # an analysis value with a margin: is the evidence sound?
            blockers, notes, links = _doubts(lc, an, inp)
            for _, rs in blockers:
                reasons += rs
            if blockers and verdict in ("pass", "fail"):
                verdict, cause = "inconclusive", blockers[0][0]
            reasons += notes
            item.links += [l for l in links if l not in item.links]

    if ev is not None and ev.value is not None:
        item.result_key, item.value, item.unit = ev.key, ev.value, ev.unit
        if ev.margin is not None:
            item.conservative_value = ev.conservative
            item.margin = ev.margin if math.isfinite(ev.margin) else None
            item.uncertainty = ev.rel
        else:
            item.limit = None       # the limit is in another dimension: the reasons give it with its unit
    item.verdict, item.reasons = verdict, reasons
    item.statement = _statement(key, lc, ev, hc, limit, factor, verdict, cause, item.caveats)
    return item, cause


def _statement(key: str, lc: LoadCase, ev: _Eval | None, hc: HandCalcResult | None, limit, factor, verdict: str,
               cause: str, caveats: list[str]) -> str:
    """The item in words: the value, the limit, the factor, the margin, the verdict, the caveats."""
    s = []
    if ev is not None and ev.value is not None:
        shown = _show(ev.value, ev.unit)
        if hc is not None:
            s.append(f"Hand calculation {hc.id} gives an estimate of {shown} for the {_what(ev.key)}.")
        else:
            unc = ev.uncertainty_text()
            mesh = " on the finest mesh" if ev.status == "singular" and ev.finest else ""
            s.append(f"The {_what(ev.key)} is {shown}{mesh}{' with an uncertainty of ' + unc if unc else ''}.")
    else:
        s.append(f"Load case {lc.id} gives no value for the {_what(key)}.")
    if isinstance(limit, _Limit):
        s.append(f"The {limit.word} is {_show(limit.value, limit.unit)} ({limit.source}).")
    if isinstance(factor, _Factor):
        s.append(f"The factor of safety is {_fac(factor.value)} ({factor.source}).")
    if ev is not None and ev.margin is not None:
        if math.isinf(ev.margin):
            s.append("The value is not above zero, so the margin of safety has no upper bound.")
        else:
            extra = ""
            if ev.k is not None and ev.k_margin is not None and math.isfinite(ev.k_margin):
                side = "safe" if ev.verdict == "fail" else "unsafe"
                extra = f" and {_ms(ev.k_margin)} with {_times(ev.k)} the uncertainty on the {side} side"
            s.append(f"The margin of safety {'of the estimate ' if hc is not None else ''}is {_ms(ev.margin)}"
                     f"{' with the uncertainty on the unsafe side' if ev.du else ''}{extra}.")
    decided_by_k = ev is not None and hc is None and ev.k is not None and verdict == ev.verdict
    s.append(_verdict_sentence(verdict, cause, ev.k if decided_by_k else None))
    if caveats:
        s.append(f"Finding {caveats[0]} applies to the inputs of this item." if len(caveats) == 1
                 else f"The findings {_join(caveats)} apply to the inputs of this item.")
    return " ".join(s)


# --------------------------------------------------------------------------- the verdict of a requirement


def _findings_sentence(ids: list[str], of: str) -> list[str]:
    if not ids:
        return []
    return [f"Finding {ids[0]} applies to {of}." if len(ids) == 1 else f"The findings {_join(ids)} apply to {of}."]


def _no_verdict(req: Requirement, inp: _Inputs) -> RequirementVerdict:
    """A requirement that analysis does not assess: another verification method, or no criterion."""
    on_req = inp.major({req.id, *req.parameters})
    if req.verification != "analysis":
        s = [f"The verification method of requirement {req.id} is {req.verification}.",
             "The analysis gives no verdict for this requirement."]
    else:
        s = [f"Requirement {req.id} has no measurable criterion.", "The analysis cannot give a verdict for it."]
        cause = make_id("FND", req.id, "not-verifiable")
        if cause in inp.findings:
            s.append(f"Finding {cause} gives the cause.")
            on_req = [i for i in on_req if i != cause]
    return RequirementVerdict(requirement=req.id, verdict="not_assessed",
                              statement=" ".join(s + _findings_sentence(on_req, "this requirement")))


def _roll_up(req: Requirement, items: list[ComplianceItem], causes: dict[str, str],
             unserved: list[str]) -> RequirementVerdict:
    """The verdict of a requirement from its items. `unserved`: the criteria that no load case serves."""
    count = {v: sum(1 for i in items if i.verdict == v) for v in _RANK}
    if count["fail"]:
        verdict = "fail"
    elif count["inconclusive"]:
        verdict = "inconclusive"
    else:
        verdict = "not_assessed" if count["not_assessed"] or unserved or not items else "pass"
    deciders = [i for i in items if i.verdict == verdict]
    gov = min(deciders, key=lambda i: math.inf if i.margin is None else i.margin) if deciders else None

    s = [{"pass": f"Requirement {req.id} passes.", "fail": f"Requirement {req.id} fails.",
          "inconclusive": f"The verdict for requirement {req.id} is inconclusive.",
          "not_assessed": f"The assessment gives no verdict for requirement {req.id}."}[verdict]]
    if len(items) > 1:
        words = {"pass": ("passes", "pass"), "fail": ("fails", "fail"), "inconclusive": ("is inconclusive", "are inconclusive"),
                 "not_assessed": ("has no verdict", "have no verdict")}
        parts = [f"{count[v]} {words[v][count[v] > 1]}" for v in ("pass", "fail", "inconclusive", "not_assessed") if count[v]]
        s.append(f"Of the {len(items)} compliance items, {_join(parts)}.")
    if gov is not None:
        margin = f" with a margin of safety of {_ms(gov.margin)}" if gov.margin is not None else ""
        s.append(f"Compliance item {gov.id} decides the verdict{margin}." if len(items) > 1 or margin
                 else f"The requirement has one compliance item, {gov.id}.")
        if verdict != "pass" and causes.get(gov.id):
            s.append(f"The cause is that {causes[gov.id]}.")
    if unserved:
        s.append(f"No load case serves criterion {unserved[0]}." if len(unserved) == 1
                 else f"No load case serves the criteria {_join(unserved)}.")
    caveats = list(dict.fromkeys(c for i in items for c in i.caveats))
    s += _findings_sentence(caveats, "the inputs of this requirement")
    return RequirementVerdict(requirement=req.id, verdict=verdict, margin=gov.margin if gov is not None else None,
                              governing=gov.id if gov is not None else None, statement=" ".join(s),
                              items=[i.id for i in items])


# --------------------------------------------------------------------------- the stage


def assess(project, ctx) -> Assessment:
    """Margins and verdicts for every requirement of the component under verification."""
    inp = _Inputs(project, ctx)
    items: list[ComplianceItem] = []
    verdicts: list[RequirementVerdict] = []
    estimates: list = []
    causes: dict[str, str] = {}
    for req in project.reqs():
        if req.verification != "analysis" or not req.criteria:
            verdicts.append(_no_verdict(req, inp))
            continue
        mine: list[ComplianceItem] = []
        unserved: list[str] = []
        for k, crit in enumerate(req.criteria):
            served = [lc for lc in inp.cases.values()
                      if any(c.requirement == req.id and c.criterion == k for c in lc.criteria)]
            if not served:
                unserved.append(crit.id or str(k))
            for lc in served:
                item_id = make_id("CMPL", req.id, crit.id or str(k), lc.id)
                if item_id in causes:                   # two criteria with the same label
                    item_id = make_id("CMPL", req.id, f"{crit.id}-{k}", lc.id)
                item, causes[item_id] = _item(req, k, crit, lc, item_id, inp, estimates)
                mine.append(item)
        items += mine
        verdicts.append(_roll_up(req, mine, causes, unserved))
    return Assessment(items, verdicts, _journal(project, inp, items, estimates))


def _journal(project, inp: _Inputs, items: list[ComplianceItem], estimates: list) -> Journal:
    """The assumption and the decisions behind the margins."""
    j = Journal("assess", BY)
    subject = project.subject.id
    thermal = [i for i in items if i.load_case in inp.cases and inp.cases[i.load_case].analysis_type == "thermo_structural"]
    if thermal:
        cases = list(dict.fromkeys(i.load_case for i in thermal))
        j.assume(subject, "factor-on-thermal-stress",
                 "Factor of safety on the total stress of a thermo-structural load case",
                 statement="The factor of safety applies to the total stress of a thermo-structural load case. "
                           "The total stress includes the thermal stress.",
                 rationale="The analysis gives one stress field for the mechanical loads and the temperature field together. "
                           "The requirements give one factor for each criterion and no different rule for thermal stress.",
                 impact="Thermal stress does not increase with the mechanical loads. "
                        "Where thermal stress adds to the mechanical stress, the margin is too small. This is the safe side. "
                        "Where thermal stress relieves the mechanical stress, the margin is too large. This is the unsafe side.",
                 verify_by="Compare the stress of each load case with the stress of the same phase without the "
                           "temperature field.",
                 conservative=None,
                 links=[("about", c) for c in cases] + [("about", r) for r in dict.fromkeys(i.requirement for i in thermal)])
    if items:
        limit = _pct(project.policy.max_uncertainty)
        j.decide(subject, "uncertainty-in-margin", "Numerical uncertainty in the margin of safety",
                 choice="The margin of safety uses each result with its numerical uncertainty on the unsafe side. "
                        "The verdict is pass only when this margin is zero or more. "
                        "The verdict is fail only when the margin is negative with the uncertainty on the safe side.",
                 rationale="The analysis gives an uncertainty for each result, as an absolute number or relative to the value. "
                           "For an upper limit, the unsafe value is the result plus its uncertainty. "
                           "For a lower limit, the unsafe value is the result minus its uncertainty. "
                           "A pass or a fail is then correct for each value in the uncertainty band. "
                           f"The policy permits an uncertainty of {limit} for a result with the status converged. "
                           "When the analysis gives no uncertainty, the assessment uses the value as it is and says so.",
                 alternatives=[("Use each result without its uncertainty.",
                                "A pass can then come from a discretization error that is larger than the margin."),
                               ("Use the Richardson extrapolation of each result.",
                                "The extrapolation is not reliable when the order of convergence of the mesh study is irregular.")],
                 links=[("uses", project.policy.id)])
    k = project.policy.unconverged_margin_factor
    if items and inp.unconverged and k > 0:
        pol = project.policy
        times = _times(k)
        j.decide(pol.id, "unconverged-margin", "Margin of a result with the status not converged",
                 choice=f"A result with the status not converged and a stated uncertainty can give a pass or a fail. "
                        f"The margin of safety then uses {times} the uncertainty. "
                        f"The verdict is pass when this margin is zero or more with the uncertainty on the unsafe side. "
                        f"The verdict is fail when this margin is negative with the uncertainty on the safe side. "
                        "Else the verdict is inconclusive. A result with no stated uncertainty stays inconclusive.",
                 rationale=f"Policy {pol.id} gives the factor unconverged_margin_factor of {_fac(k)}. "
                           "The mesh study states the uncertainty of each result. "
                           f"For a mesh sequence that does not reach the policy limit of {_pct(pol.max_uncertainty)}, "
                           "this uncertainty is already a conservative bound. "
                           f"The factor of {_fac(k)} covers the weaker basis of this bound. "
                           "A margin that holds with this factor is much larger than a probable discretization error.",
                 alternatives=[("Refuse each result with an uncertainty over the policy limit.",
                                "Then a result with a very large margin gives no verdict, and the engineer gets no information.")],
                 links=[("uses", pol.id)])
    for item_id, key, lc, hc, output, why in estimates:
        j.decide(item_id, "estimate", "Estimate from a hand calculation in place of an analysis",
                 choice=f"Use the output {output} of hand calculation {hc.id} as an estimate of {key} for load case {lc.id}.",
                 rationale=" ".join([why[0] if why else f"Load case {lc.id} has no analysis result for {key}.",
                                     "The hand calculation gives the same quantity for the loads of this load case.",
                                     "An estimate does not confirm a criterion, so the verdict is inconclusive."]),
                 alternatives=[("Give no value for this criterion.",
                                "The estimate shows the engineer how large the margin is.")],
                 needs_review=True, links=[("evidence", hc.id), ("about", lc.id)])
    return j
