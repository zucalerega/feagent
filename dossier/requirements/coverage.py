"""Rules on what the requirements cover, and on requirements that disagree with each other.

Coverage: loads come from the system context, so a phase in which the part has loads needs a requirement of
kind strength that says what the part must withstand there. A phase whose loads are the same pattern as the
loads of a covered phase, with a scale of 1 or less, needs none (`dossier.context.proportional`): the rule
records this as an assumption, not as a finding, because the criteria of the covered phase are then taken as
right for the smaller phase.

When does `FND-<PHASE>-UNCOVERED` exist (the plan links to it)? For each phase with a mechanical or thermal
effect on the part that no strength requirement applies to and that is not in proportion (scale <= 1) to a
phase with a strength requirement. It is major when the part has a load there: a mechanical effect that is
not zero, a heat flux, or a temperature that differs from the reference temperature of the component by more
than `policy.thermal_delta`. It is info when the effects of the phase load nothing.

Conflicts: two criteria conflict when they judge the same result (the same key and comparator) in the same
phase against different limits on the response. A yield criterion and an ultimate criterion on the same
result (two different allowables) are two checks, not a conflict.
"""

from __future__ import annotations

from dossier import units
from dossier.context import MECHANICAL, THERMAL, proportional
from dossier.model import PARAM_ROLES, Criterion, Value
from dossier.requirements.base import (QUANTITY_NEEDS, QUANTITY_UNIT, ROLE_WORDS, TEMPERATURE_KINDS, Env, cap, count,
                                       criterion_key, criterion_name, fit, join, number, quote, rule, shown, some)

LOADS = (*MECHANICAL, *THERMAL)

# --------------------------------------------------------------------------- phases


def _temperatures(env: Env) -> tuple[float, float]:
    """(the reference temperature of the component, the thermal limit of the policy), in K."""
    t_ref = float(units.si(units.to(env.comp.reference_temperature.quantity(env.names), "K")))
    return t_ref, float(units.si(env.policy.thermal_delta.quantity(env.names)))


def _loading(env: Env, phase: str) -> tuple[list, list, list]:
    """(mechanical effects, thermal effects, the effects that load the part) of one phase."""
    t_ref, t_delta = _temperatures(env)
    mech = env.effects(phase, MECHANICAL)
    therm = env.effects(phase, THERMAL)
    real = [e for e in mech if env.magnitude(e) > 0.0]
    real += [e for e in therm if (e.kind == "heat_flux" and env.magnitude(e) > 0.0)
             or (e.kind in TEMPERATURE_KINDS and isinstance(e.si, (int, float)) and abs(e.si - t_ref) > t_delta)]
    return mech, therm, real


def _effects_sentences(env: Env, phase, mech: list, therm: list) -> list[str]:
    parts = [count(len(x), word) for x, word in ((mech, "mechanical effect"), (therm, "thermal effect")) if x]
    effs = [*mech, *therm]
    listed = join(f"{e.id} ({e.kind}, {env.size(e)})" for e in effs[:3])
    return [f"The system context gives {join(parts)} on the part in {phase.id}, a phase of category {phase.category}.",
            (f"The first 3 effects are {listed}." if len(effs) > 3 else
             f"The {'effect is' if len(effs) == 1 else 'effects are'} {listed}.")]


@rule("req.phase_uncovered", "major", "coverage",
      "A requirement of kind strength covers each phase in which the part has loads.")
def phase_uncovered(env: Env) -> None:
    """Key `uncovered`: the plan links to FND-<phase>-UNCOVERED (see the module text for when it exists)."""
    covered: dict[str, list[str]] = {}
    for r in env.reqs:
        if r.kind == "strength":
            for ph in env.phases_of(r):
                covered.setdefault(ph, []).append(r.id)
    covered_ids = [p.id for p in env.phases if p.id in covered]
    t_ref, t_delta = _temperatures(env)
    for ph in env.phases:
        if ph.id in covered:
            continue
        mech, therm, real = _loading(env, ph.id)
        if not mech and not therm:
            continue
        under: tuple[float, str] | None = None            # the covered phase with the nearest larger loads
        over: tuple[float, str] | None = None             # the covered phase with the nearest smaller loads
        for c in covered_ids:
            s = proportional(env.loads(c), env.loads(ph.id))
            if s is None:
                continue
            if s <= 1.0 + 1e-9:
                under = (s, c) if under is None or s > under[0] else under
            else:
                over = (s, c) if over is None or s < over[0] else over
        if under is not None:
            _covered_by_scale(env, ph, under[0], under[1], covered[under[1]])
            continue
        rationale = _effects_sentences(env, ph, mech, therm)
        values = {"category": ph.category, "mechanical_effects": len(mech), "thermal_effects": len(therm),
                  "loads": {e.id: e.si for e in (*mech, *therm)}}
        links = [("evidence", e.id) for e in (*mech, *therm)]
        name = f"the phase {ph.id} ({quote(ph.name)})"
        if not real:                                       # the effects are there, and they load nothing
            zero = "The phase has no mechanical effect" if not mech else "The mechanical effects of the phase are zero"
            rationale.append(f"{zero}, and its temperatures are within {shown(t_delta, 'K')} of the reference temperature "
                             f"{shown(t_ref, 'K')}." if therm else f"{zero}.")
            env.find(subject=ph.id, key="uncovered", severity="info", title=f"No strength requirement for the phase {ph.id}",
                     statement=[f"No requirement of kind strength applies to {name}.",
                                "The effects of this phase do not load the part."],
                     rationale=rationale,
                     suggestion=["No change is necessary.", f"Make sure that the system context has all loads of {ph.id}."],
                     values=values, links=links)
            continue
        rationale.append(f"The requirements of kind strength apply to {some(covered_ids, 6, 'phase')}." if covered_ids
                         else "The part has no requirement of kind strength.")
        if over is not None:
            rationale.append(f"The loads of {ph.id} are {number(over[0])} times the loads of {over[1]}, thus {over[1]} "
                             f"does not cover them.")
        elif covered_ids:
            rationale.append(f"The loads of {ph.id} are not in proportion to the loads of one of these phases.")
        env.find(subject=ph.id, key="uncovered", title=f"No strength requirement for the phase {ph.id}",
                 statement=[f"No requirement of kind strength applies to {name}.", "The part has loads in this phase."],
                 rationale=rationale,
                 suggestion=f"Add a requirement of kind strength that applies to {ph.id}, with criteria for its loads.",
                 values=values, links=links)


def _covered_by_scale(env: Env, ph, scale: float, other: str, reqs: list[str]) -> None:
    one = len(reqs) == 1
    env.assume(subject=ph.id, key="covered-by-scale", title=f"{ph.id} needs no strength requirement of its own",
               statement=f"The strength {'requirement' if one else 'requirements'} {join(reqs)} of the phase {other} also "
                         f"{'covers' if one else 'cover'} the phase {ph.id}.",
               rationale=[f"No requirement of kind strength applies to {ph.id}.",
                          f"The loads of {ph.id} are {number(scale)} times the loads of {other}, on the same interfaces "
                          f"and with the same supports.",
                          f"{join(reqs)} {'applies' if one else 'apply'} to {other}.",
                          "A linear static analysis gives stresses in proportion to the loads."],
               impact=[f"If the part does not respond linearly, the stresses of {ph.id} are not {number(scale)} times the "
                       f"stresses of {other}.",
                       f"If {ph.id} needs different criteria, it needs a strength requirement of its own."],
               verify_by=f"Make sure that the criteria of {join(reqs)} are correct for {ph.id}.",
               conservative=None, links=[("evidence", other), *(("evidence", r) for r in reqs)])


# --------------------------------------------------------------------------- duty cycles


@rule("req.duty_cycle_uncovered", "minor", "coverage", "A requirement of kind life names each duty cycle that loads the part.")
def duty_cycle_uncovered(env: Env) -> None:
    named: dict[str, list[str]] = {}
    for r in env.reqs:
        if r.kind == "life":
            for d in r.applies.duty_cycles:
                named.setdefault(d, []).append(r.id)
    for dc in env.project.system.duty_cycles:
        if dc.id in named:
            continue
        sequence = list(dict.fromkeys(dc.sequence))
        loaded = [ph for ph in sequence if env.effects(ph, LOADS)]
        if not loaded:
            continue
        n = env.ctx.duty_cycles.get(dc.id)
        rationale = [f"The duty cycle {dc.id} repeats {number(n)} times in the life of the part." if n is not None
                     else f"The system context gives no count for the duty cycle {dc.id}."]
        if dc.source:
            rationale.append(f"The source of the count is {quote(dc.source)}.")
        rationale.append(f"Its phase is {sequence[0]}, and the part has loads in this phase." if len(sequence) == 1 else
                         fit(f"Its phases are {join(sequence)}, and the part has loads in {join(loaded)}.",
                             f"It has {count(len(sequence), 'phase')}, and the part has loads in {len(loaded)} of them."))
        rationale.append(f"The requirements of kind life name {some(named, 6, 'duty cycle')}." if named
                         else "The part has no requirement of kind life that names a duty cycle.")
        env.find(subject=dc.id, key="uncovered", title=f"No life requirement for the duty cycle {dc.id}",
                 statement=f"No requirement of kind life names the duty cycle {dc.id} ({quote(dc.name)}).",
                 rationale=rationale,
                 suggestion=[f"Add {dc.id} to the duty cycles of a life requirement.",
                             "If the duty cycle causes no fatigue damage, record the reason."],
                 values={"count": n, "phases": sequence},
                 links=[*(("evidence", ph) for ph in sequence),
                        *(("evidence", x) for x in sorted(dc.count.names()) if x in env.ctx.params)])


# --------------------------------------------------------------------------- where a requirement applies

_NEEDS_WORDS = {THERMAL: "thermal effect", ("mass_flow",): "mass flow", LOADS: "mechanical or thermal effect"}


@rule("req.phase_without_effect", "minor", "coverage",
      "The part has the effects that a requirement needs in each phase that the requirement applies to.")
def phase_without_effect(env: Env) -> None:
    for r in env.reqs:
        phases = env.phases_of(r)
        if not phases:
            continue
        empty = [ph for ph in phases if not [e for e in env.effects(ph) if e.kind not in ("support", "signal")]]
        if empty:
            these = "this phase" if len(empty) == 1 else "these phases"
            listed = all(ph in r.applies.phases for ph in empty)
            env.find(subject=r.id, key="no-effect", title=f"{r.id} applies to a phase without effects on the part",
                     statement=f"{r.id} applies to {some(empty, 4, 'phase')}, but the part has no effect in {these}.",
                     rationale=[(f"{r.id} lists {these} in its phases." if listed
                                 else f"{r.id} applies to the phases of category {r.applies.category}."),
                                f"The system context gives no load, no temperature and no flow on the part in {these}."],
                     suggestion=([f"Remove {these} from the phases of {r.id}.",
                                  "If the part has loads there, add the effects to the system context."] if listed else
                                 [f"No change is necessary if the part has no load in {these}.",
                                  "If it has loads there, add the effects to the system context."]),
                     values={"phases": empty}, links=[("evidence", ph) for ph in empty])
        for i, c in enumerate(r.criteria):
            needs = QUANTITY_NEEDS.get(c.quantity)
            if not needs:
                continue
            lacking = [ph for ph in phases if ph not in empty and not env.effects(ph, needs)]
            if not lacking:
                continue
            word = _NEEDS_WORDS.get(tuple(needs), "effect")
            has = list(dict.fromkeys(e.kind for e in env.effects(lacking[0]) if e.kind != "signal"))
            env.find(subject=r.id, key=f"criterion-{criterion_key(r, i)}-no-effect",
                     title=f"A criterion of {r.id} applies to a phase without the effect it needs",
                     statement=f"{cap(criterion_name(r, i))} of {r.id} needs a {word} in {some(lacking, 4, 'phase')}, "
                               f"but the system context gives none.",
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                f"The quantity {c.quantity} needs an effect of kind {join(needs, 'or')}.",
                                f"In {lacking[0]}, the system context gives these kinds on the part: {join(has)}."],
                     suggestion=[f"Remove {some(lacking, 4, 'phase')} from the phases of {r.id}.",
                                 "If the effect exists, add it to the system context."],
                     values={"criterion": i, "phases": lacking, "needs": list(needs)},
                     links=[("evidence", ph) for ph in lacking])


@rule("req.no_phase", "major", "coverage", "A requirement that has criteria applies to a phase or to a duty cycle.")
def no_phase(env: Env) -> None:
    for r in env.reqs:
        if not r.criteria or env.phases_of(r) or r.applies.duty_cycles:
            continue
        n = len(r.criteria)
        env.find(subject=r.id, key="applies-nowhere", title=f"{r.id} has criteria but applies to no phase",
                 statement=f"{r.id} has {count(n, 'criterion', 'criteria')} but applies to no phase and to no duty cycle.",
                 rationale=[f"The {'criterion' if n == 1 else 'criteria'} of {r.id} {'asks' if n == 1 else 'ask'} for "
                            f"{some((c.key() for c in r.criteria), 3)}.",
                            "The requirement gives no phase, no category of phases and no duty cycle.",
                            f"A load case is a phase at limit loads, thus the plan has no load case for {r.id}."],
                 suggestion=f"Give {r.id} the phases, or the category of phases, that its criteria apply to.",
                 values={"criteria": [c.key() for c in r.criteria]})


# --------------------------------------------------------------------------- conflicts


@rule("req.role_conflict", "major", "conflict", "Parameters with the same role have the same value.")
def role_conflict(env: Env) -> None:
    for role in PARAM_ROLES:
        got = env.with_role(role)
        values = [g[3].si for g in got]
        if len(got) < 2 or max(values) - min(values) <= 1e-9 * max(abs(max(values)), abs(min(values))):
            continue
        words = ROLE_WORDS[role]
        first = got[0][0]
        reqs = list(dict.fromkeys(g[0].id for g in got))
        distinct = len({round(v / max(abs(max(values)), abs(min(values))), 9) for v in values})
        rationale = [f"{r.id} gives {rp.display} (parameter {name})." for r, name, _, rp in got[:4]]
        rationale.append(f"The role {role} says that each of these parameters is the {words}.")
        env.find(subject=first.id, key=f"conflict-{role}", title=f"Different values for the {words}",
                 statement=(f"{join(reqs)} give {count(distinct, 'different value')} for the {words}." if len(reqs) > 1
                            else f"{first.id} gives {count(distinct, 'different value')} for the {words}."),
                 rationale=rationale,
                 suggestion=["Make the values agree.",
                             "If the parameters are different quantities, remove the role from one and give it a check."],
                 values={name: rp.si for _, name, _, rp in got},
                 links=[*(("about", name) for _, name, _, _ in got), *(("about", x) for x in reqs[1:])])


def _limit(env: Env, c: Criterion) -> tuple[float, float, float, str] | None:
    """(the limit in SI, the factor, the limit on the response itself, the SI unit) of a criterion, or None
    when the criterion has no value (the quality rules say why)."""
    unit = QUANTITY_UNIT.get(c.quantity)
    try:
        if c.allowable is not None:
            v = getattr(env.material, c.allowable, None)
            if not isinstance(v, Value):
                return None
            q = v.quantity()
        else:
            q = c.limit.quantity(env.names)
    except (units.UnitsError, ValueError, TypeError):
        return None
    if unit is None or units.is_vector(q) or not units.compatible(q, unit):
        return None
    f = c.factor
    if isinstance(f, str):
        try:
            f = float(f)
        except ValueError:
            p = env.ctx.params.get(f) if f in env.req_params else None
            if p is not None and p.unit == "" and not isinstance(p.si, list):
                f = float(p.si)
            elif f in env.policy.factors:
                f = float(env.policy.factors[f].value)
            else:
                return None
    if not f > 0:
        return None
    limit = float(units.si(q))
    return limit, float(f), (limit / f if c.comparator == "<=" else limit * f), unit


@rule("req.criteria_conflict", "minor", "conflict",
      "Two criteria for the same result in the same phase have the same limit.")
def criteria_conflict(env: Env) -> None:
    order = {p.id: k for k, p in enumerate(env.phases)}
    items = []
    for r in env.reqs:
        for i, c in enumerate(r.criteria):
            got = _limit(env, c)
            if got is not None:
                scope = set(r.applies.duty_cycles) if c.quantity == "damage" else set(env.phases_of(r))
                items.append((r, i, c, got, scope))
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ra, ia, ca, (la, fa, ea, unit), sa = items[a]
            rb, ib, cb, (lb, fb, eb, _), sb = items[b]
            if ca.key() != cb.key() or ca.comparator != cb.comparator:
                continue
            if ca.allowable and cb.allowable and ca.allowable != cb.allowable:
                continue                                   # yield and ultimate on one result: two checks, no conflict
            common = sorted(sa & sb, key=lambda x: order.get(x, 0))
            if (ra is not rb and not common) or abs(ea - eb) <= 1e-9 * max(abs(ea), abs(eb)):
                continue
            where = f" in {some(common, 3, 'phase')}" if common else ""

            def line(r, i, c, limit, factor, effective) -> str:
                detail = f" ({shown(limit, unit)} with the factor {number(factor)})" if factor != 1.0 else ""
                return f"{cap(criterion_name(r, i))} of {r.id} gives {c.key()} {c.comparator} {shown(effective, unit)}{detail}."

            a_is_strict = ea < eb if ca.comparator == "<=" else ea > eb
            strict, loose = ((ra, ia), (rb, ib)) if a_is_strict else ((rb, ib), (ra, ia))
            env.find(subject=ra.id, key=f"criterion-{criterion_key(ra, ia)}-conflict-{rb.id}-{criterion_key(rb, ib)}",
                     title=f"Two limits for the result {ca.key()}",
                     statement=(f"{ra.id} and {rb.id} judge the result {ca.key()}{where} against different limits."
                                if ra is not rb else
                                f"Two criteria of {ra.id} judge the result {ca.key()} against different limits."),
                     rationale=[line(ra, ia, ca, la, fa, ea), line(rb, ib, cb, lb, fb, eb),
                                f"The limit of {criterion_name(*strict)} of {strict[0].id} is the stricter limit.",
                                f"{cap(criterion_name(*loose))} of {loose[0].id} decides nothing{where}."],
                     suggestion=["Keep one of the two criteria, or give the two criteria different phases.",
                                 "If the part needs the two limits, record the reason."],
                     values={"first": ea, "second": eb, "unit": unit, "phases": common},
                     links=[("about", rb.id)] if ra is not rb else [])
