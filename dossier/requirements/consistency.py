"""Rules on the numbers of the requirements against the system context and against the policy.

A number a requirement states is a claim. The role of a parameter says what the number must cover in the
system context (dossier.model.PARAM_ROLES); an explicit `check` says it in an expression. A finding gives the
chain of the evidence: the requirement value, the context value with its phase and its effect, and how the
context value comes from its parameters.
"""

from __future__ import annotations

import math

from dossier import units
from dossier.requirements.base import (INSTRUCTION, POLICY_FACTOR, ROLE_WORDS, TEMPERATURE_KINDS, Env, article, bound, cap,
                                       count,
                                       criterion_name, derivation, fit, join, names_in, number, percent, quote, rule,
                                       shown, some)

# --------------------------------------------------------------------------- shared wording


def _inputs(env: Env, names: list[str]) -> dict:
    return {n: env.ctx.params[n].si for n in names}


def _evidence(e, names: list[str]) -> list[tuple]:
    return [("evidence", e.id), ("evidence", e.phase), *(("evidence", n) for n in names)]


def _not_checked(env: Env, r, name: str, rp, role: str, missing: str, fix: str) -> None:
    """A claim that the validation cannot compare, because the system context has no value for it."""
    words = ROLE_WORDS[role]
    env.find(subject=r.id, key=env.role_key(r, role, name, f"{role}-not-checked"), severity="info",
             title=f"{cap(words)} of {r.id} has no value in the system context to compare with",
             statement=f"The validation did not compare the {words} of {r.id} with the system context.",
             rationale=[f"The requirement {r.id} gives {article(words)} of {rp.display} (parameter {name}).", missing],
             suggestion=fix, values={"requirement": rp.si, "unit": rp.unit}, links=[("about", name)])


def _used_by(env: Env, name: str) -> str:
    """One sentence: what reads a requirement parameter (so the reader sees what a wrong value changes)."""
    crit = env.criteria_using(name)
    if crit:
        r, i = crit[0]
        more = f" {count(len(crit) - 1, 'more criterion uses', 'more criteria use')} it also." if len(crit) > 1 else ""
        return f"{cap(criterion_name(r, i))} of {r.id} uses the parameter {name}.{more}"
    readers = env.readers(name)
    if readers:
        return f"The loads of {some(readers, 4, 'phase')} read the parameter {name}."
    return f"No criterion and no load of the part reads the parameter {name}."


# --------------------------------------------------------------------------- roles against the context


@rule("req.meop_envelope", "major", "consistency",
      "The MEOP of a requirement is not less than the largest pressure on the part in the operation phases.")
def meop_envelope(env: Env) -> None:
    for r, name, _, rp in env.with_role("meop"):
        best = env.ctx.envelope(env.project, None, "pressure", env.operation)
        if best is None:
            _not_checked(env, r, name, rp, "meop", "The system context gives no pressure on the part in an operation phase.",
                         "Add the pressures of the operation phases to the system context.")
            continue
        ctx_si, e = best
        lines, names = env.context_lines(e)
        rationale = [f"The requirement {r.id} gives an MEOP of {rp.display} (parameter {name}).", *lines]
        values = {"requirement": rp.si, "context": ctx_si, "unit": "Pa", "difference": ctx_si - rp.si,
                  "inputs": _inputs(env, names)}
        links = [("about", name), *_evidence(e, names)]
        if rp.si < ctx_si * (1.0 - 1e-9):
            readers = env.readers(name)
            suggestion = [f"Increase the MEOP of {r.id} to {bound(ctx_si, 'Pa', up=True)} or more."]
            if readers:
                suggestion.append(fit(f"The loads of {join(readers)} read the parameter {name} and change with it.",
                                      f"The loads of {len(readers)} phases read the parameter {name} and change with it.",
                                      limit=INSTRUCTION))
            env.find(subject=r.id, key=env.role_key(r, "meop", name, "meop-low"),
                     title=f"MEOP of {r.id} is lower than the pressure of the system context",
                     statement=f"The MEOP of {r.id} is {percent((ctx_si - rp.si) / ctx_si)} lower than the largest "
                               f"pressure on the part in the operation phases.",
                     rationale=rationale, suggestion=suggestion, values=values, links=links)
        elif ctx_si > 0 and rp.si > 1.25 * ctx_si:
            env.find(subject=r.id, key=env.role_key(r, "meop", name, "meop-high"), severity="info",
                     title=f"MEOP of {r.id} is much higher than the pressure of the system context",
                     statement=f"The MEOP of {r.id} is {percent((rp.si - ctx_si) / ctx_si)} higher than the largest "
                               f"pressure on the part in the operation phases.",
                     rationale=rationale,
                     suggestion=["No change is necessary for the verification.",
                                 "Make sure that the system context has the largest pressure of the operation phases."],
                     values=values, links=links)


@rule("req.temperature_range", "major", "consistency",
      "The fluid temperature range of a requirement includes the fluid temperatures of the operation phases.")
def temperature_range(env: Env) -> None:
    temps = [e for e in env.effects(kinds=TEMPERATURE_KINDS)
             if env.wetted(e) and e.phase in env.operation and isinstance(e.si, (int, float))]
    for role, low in (("min_temperature", True), ("max_temperature", False)):
        words = ROLE_WORDS[role]
        for r, name, _, rp in env.with_role(role):
            if not temps:
                _not_checked(env, r, name, rp, role,
                             "The system context gives no fluid temperature on the wetted surfaces of the part in an "
                             "operation phase.", "Add the fluid temperatures of the operation phases to the system context.")
                continue
            e = min(temps, key=lambda x: x.si) if low else max(temps, key=lambda x: x.si)
            slack = 1e-9 * max(abs(e.si), 1.0)
            if (rp.si <= e.si + slack) if low else (rp.si >= e.si - slack):
                continue
            lines, names = env.context_lines(e)
            rationale = [f"The requirement {r.id} gives {article(words)} of {rp.display} (parameter {name}).", *lines]
            same = [x for x in temps if x is not e and abs(x.si - e.si) <= slack]
            if same and len(rationale) < 6:
                rationale.append(f"{count(len(same), 'other effect gives', 'other effects give')} the same temperature "
                                 f"on the wetted surfaces.")
            gap = shown(abs(rp.si - e.si), "K")
            env.find(subject=r.id, key=env.role_key(r, role, name, role.replace("_", "-")),
                     title=f"{cap(words)} of {r.id} is {'higher' if low else 'lower'} than the fluid temperature of the "
                           f"system context",
                     statement=(f"The {words} of {r.id} is {gap} higher than the lowest fluid temperature of the "
                                f"operation phases." if low else
                                f"The {words} of {r.id} is {gap} lower than the highest fluid temperature of the "
                                f"operation phases."),
                     rationale=rationale,
                     suggestion=(f"Decrease the {words} of {r.id} to {bound(e.si, 'K', up=False)} or less." if low else
                                 f"Increase the {words} of {r.id} to {bound(e.si, 'K', up=True)} or more."),
                     values={"requirement": rp.si, "context": e.si, "unit": "K", "difference": rp.si - e.si,
                             "inputs": _inputs(env, names)},
                     links=[("about", name), *_evidence(e, names)])


@rule("req.rated_flow", "major", "consistency",
      "The rated mass flow of a requirement agrees with the mass flow of the system context within 1 %.")
def rated_flow(env: Env) -> None:
    for r, name, _, rp in env.with_role("rated_flow"):
        scope = env.phases_of(r) or env.operation
        flows = [e for e in env.effects(kinds=("mass_flow",)) if e.phase in scope and isinstance(e.si, (int, float))]
        if not flows:
            _not_checked(env, r, name, rp, "rated_flow",
                         f"The system context gives no mass flow through the part in {some(scope, 4, 'phase')}.",
                         "Add the mass flow to the system context, or correct the phases of the requirement.")
            continue
        if any(abs(rp.si - e.si) <= 0.01 * abs(e.si) for e in flows):
            continue
        e = max(flows, key=lambda x: abs(x.si))
        lines, names = env.context_lines(e)
        rationale = [f"The requirement {r.id} gives a rated mass flow of {rp.display} (parameter {name}).", *lines]
        if len(rationale) < 6:
            rationale.append("The two values must agree within 1 %, because the flow analysis uses the context value.")
        env.find(subject=r.id, key=env.role_key(r, "rated_flow", name, "rated-flow"),
                 title=f"Rated mass flow of {r.id} does not agree with the system context",
                 statement=f"The rated mass flow of {r.id} is {percent((rp.si - e.si) / e.si)} "
                           f"{'higher' if rp.si > e.si else 'lower'} than the mass flow of the system context.",
                 rationale=rationale, suggestion=f"Change the rated mass flow of {r.id} to {shown(e.si, 'kg/s')}.",
                 values={"requirement": rp.si, "context": e.si, "unit": "kg/s", "difference": rp.si - e.si,
                         "inputs": _inputs(env, names)},
                 links=[("about", name), *_evidence(e, names)])


@rule("req.design_cycles", "major", "consistency",
      "The number of design cycles is not less than the count of the duty cycles of its requirement (a parameter "
      "without a check).")
def design_cycles(env: Env) -> None:
    by_id = {dc.id: dc for dc in env.project.system.duty_cycles}
    for r, name, p, rp in env.with_role("design_cycles"):
        ids = [d for d in r.applies.duty_cycles if d in env.ctx.duty_cycles]
        if p.check is not None or not ids:
            continue                                   # an explicit check says what the number must cover
        counts = [env.ctx.duty_cycles[d] for d in ids]
        total = float(sum(counts))
        if rp.si >= total * (1.0 - 1e-9):
            continue
        by_source: dict[str, list[str]] = {}
        for d in ids:
            if by_id[d].source:
                by_source.setdefault(by_id[d].source, []).append(d)
        rationale = [f"The requirement {r.id} gives {number(rp.si)} design cycles (parameter {name}).",
                     fit(f"{r.id} names {count(len(ids), 'duty cycle')}: {join(ids)}.",
                         f"{r.id} names {count(len(ids), 'duty cycle')}."),
                     (f"The system context gives {' + '.join(number(c) for c in counts)} = {number(total)} repetitions of "
                      f"these duty cycles." if len(ids) > 1 else
                      f"The system context gives {number(total)} repetitions of this duty cycle.")]
        if by_source:
            rationale.append("The sources of the counts are "
                             + join(f"{quote(s)} ({', '.join(ds)})" for s, ds in list(by_source.items())[:3]) + ".")
        rationale.append(f"The parameter {name} has no check, thus the validation compares it with the duty cycles.")
        env.find(subject=r.id, key=env.role_key(r, "design_cycles", name, "design-cycles"),
                 title=f"Design cycles of {r.id} are fewer than the cycles of the system context",
                 statement=f"The number of design cycles of {r.id} is {percent((total - rp.si) / total)} lower than the "
                           f"count of its duty cycles.",
                 rationale=rationale,
                 suggestion=[f"Increase the number of design cycles of {r.id} to {number(math.ceil(total - 1e-9))} or more.",
                             f"If a different number is correct, give the parameter {name} a check with the reason."],
                 values={"requirement": rp.si, "context": total, "unit": "", "difference": total - rp.si,
                         "duty_cycles": dict(zip(ids, counts))},
                 links=[("about", name), *(("evidence", d) for d in ids)])


# --------------------------------------------------------------------------- explicit checks


@rule("req.param_check", "major", "consistency", "Each parameter obeys its explicit check against the system context.")
def param_check(env: Env) -> None:
    for r in env.reqs:
        for name, p in r.parameters.items():
            ck = p.check
            rp = env.ctx.params.get(name)
            if ck is None or rp is None:
                continue
            key = f"check-{name}"
            written = f"{name} {ck.relation} {ck.against}"
            reads = names_in(ck.against, env.ctx)
            links = [("about", name), *(("evidence", n) for n in reads if n != name)]

            def broken(statement: list[str] | str, why: str) -> None:
                env.find(subject=r.id, key=key, severity="error", title=f"Check of {name} in {r.id} has no value",
                         statement=statement, rationale=[f"The requirement {r.id} gives {name} = {rp.display}.",
                                                         f"The check is {written}.", why],
                         suggestion="Correct the expression of the check. It can read the parameters of the project.",
                         values={"requirement": rp.si, "unit": rp.unit, "against": ck.against}, links=links)

            try:
                against = env.ctx.evaluate(ck.against)
            except units.UnitsError as e:
                broken(f"The validation cannot calculate the check of the parameter {name} of {r.id}.",
                       f"The error is {quote(e)}.")
                continue
            if units.is_vector(against) or isinstance(rp.si, list):
                broken(f"The check of the parameter {name} of {r.id} compares a vector.", "A check compares two numbers.")
                continue
            if not units.compatible(against, rp.unit):
                broken([f"The check of the parameter {name} of {r.id} compares {rp.display} with {units.display(against)}.",
                        "The two values have different units."],
                       "A check compares two values with the same unit.")
                continue
            a, v = float(units.si(against)), float(rp.si)
            tol = abs(ck.tolerance) * abs(a)
            slack = tol + 1e-9 * max(abs(a), abs(v))
            if {">=": v >= a - slack, "<=": v <= a + slack, "==": abs(v - a) <= slack}[ck.relation]:
                continue
            total = shown(a, rp.unit)
            chain, names = derivation(env.ctx, ck.against, total, "The check")
            gap = f"{percent((v - a) / a)} " if a else ""
            rationale = [f"The requirement {r.id} gives {name} = {rp.display}.",
                         f"The check is {written}" + (f", with a tolerance of {percent(ck.tolerance)}." if ck.tolerance else "."),
                         *(chain or [f"The check gives {total}."])]
            if ck.why and len(rationale) < 6:
                rationale.append(f"The reason of the check is {quote(ck.why)}.")
            env.find(subject=r.id, key=key, title=f"Parameter {name} of {r.id} does not obey its check",
                     statement={">=": f"The parameter {name} of {r.id} is {gap}lower than the minimum that its check gives.",
                                "<=": f"The parameter {name} of {r.id} is {gap}higher than the maximum that its check gives.",
                                "==": f"The parameter {name} of {r.id} is not equal to the value that its check gives.",
                                }[ck.relation],
                     rationale=rationale,
                     suggestion={">=": f"Increase the parameter {name} of {r.id} to {bound(a, rp.unit, up=True)} or more.",
                                 "<=": f"Decrease the parameter {name} of {r.id} to {bound(a, rp.unit, up=False)} or less.",
                                 "==": f"Change the parameter {name} of {r.id} to {shown(a, rp.unit, 6)}.",
                                 }[ck.relation],
                     values={"requirement": v, "context": a, "unit": rp.unit, "difference": a - v, "relation": ck.relation,
                             "tolerance": ck.tolerance, "inputs": _inputs(env, names)},
                     links=[("about", name), *(("evidence", n) for n in names if n != name)])


# --------------------------------------------------------------------------- factors against the policy


@rule("req.policy_factor", "major", "policy", "Each factor of a requirement is not less than the factor of the policy.")
def policy_factor(env: Env) -> None:
    policy = env.policy
    for role, key in POLICY_FACTOR.items():
        words = ROLE_WORDS[role]
        for r, name, _, rp in env.with_role(role):
            given = [f"The requirement {r.id} gives {article(words)} of {rp.display} (parameter {name})."]
            links = [("about", name), ("evidence", policy.id)]
            f = policy.factors.get(key)
            if f is None:
                env.find(subject=r.id, key=env.role_key(r, role, name, f"{role}-not-checked"), severity="info",
                         title=f"No policy factor for the {words} of {r.id}",
                         statement=[f"The policy {policy.id} has no factor {key}.",
                                    f"The validation did not compare the {words} of {r.id}."],
                         rationale=[*given, f"The factors of the policy are: {some(policy.factors, 8) or 'none'}."],
                         suggestion=f"Add the factor {key} to the policy, with its source.",
                         values={"requirement": rp.si}, links=links)
                continue
            slack = 1e-9 * max(abs(f.value), abs(rp.si))
            if abs(rp.si - f.value) <= slack:
                continue
            rationale = [*given, f"The policy {policy.id} gives a minimum of {number(f.value)} for the factor {key}."]
            if f.note:
                rationale.append(f"The note of this policy factor is {quote(f.note)}.")
            where = f.source or policy.source
            rationale.append(f"The source of the policy is {quote(where)}." if where else "The policy gives no source.")
            rationale.append(_used_by(env, name))
            values = {"requirement": rp.si, "policy": f.value, "unit": "", "difference": f.value - rp.si}
            if rp.si < f.value:
                env.find(subject=r.id, key=env.role_key(r, role, name, role.replace("_", "-")),
                         title=f"{cap(words)} of {r.id} is lower than the policy",
                         statement=f"The {words} of {r.id} is lower than the minimum of the policy.",
                         rationale=rationale,
                         suggestion=[f"Increase the {words} of {r.id} to {number(f.value)} or more.",
                                     "If the lower factor is correct, record the approved deviation from the policy."],
                         values=values, links=links)
            else:
                env.find(subject=r.id, key=env.role_key(r, role, name, f"{role}-high"), severity="info",
                         title=f"{cap(words)} of {r.id} is higher than the policy",
                         statement=f"The {words} of {r.id} is higher than the minimum of the policy.",
                         rationale=rationale,
                         suggestion=["No change is necessary for the verification.",
                                     "Make sure that the specification asks for the higher factor."],
                         values=values, links=links)
