"""Rules on the quality of each requirement: can analysis verify it as it is written and modelled?

A requirement is the source text, its parameters (numbers it states) and its criteria (what analysis
compares). The rules here read one requirement at a time; they do not need the system context, except for
the names an expression may read.
"""

from __future__ import annotations

from dossier import units
from dossier.model import PARAM_ROLES, Requirement, Value
from dossier.requirements import wording
from dossier.requirements.base import (FIXED_LOCATIONS, MEASURES, QUANTITY_UNIT, ROLE_UNIT, ROLE_WORDS, SECTION_MEASURES,
                                       UNIT_WORDS, Env, cap, count, criterion_key, criterion_name, join, nearest, number,
                                       quote, rule, some)

# --------------------------------------------------------------------------- verifiable by analysis


@rule("req.not_verifiable", "major", "quality", "A requirement that analysis verifies has a criterion.")
def not_verifiable(env: Env) -> None:
    """Key `not-verifiable`: the plan and the compliance stage link to FND-<requirement>-NOT-VERIFIABLE."""
    for r in env.reqs:
        if r.verification != "analysis" or r.criteria:
            continue
        vague = [v.word for v in wording.vague(r.text)]
        values = {"criteria": 0, "parameters": len(r.parameters), "vague_words": vague}
        if r.kind == "definition":                      # a definition has nothing to verify: the method is the defect
            env.find(subject=r.id, key="not-verifiable", severity="minor", title=f"Analysis cannot verify the definition {r.id}",
                     statement=f"{r.id} is a definition, but its verification method is analysis and it has no criterion.",
                     rationale=[f"The kind of {r.id} is definition.",
                                "A definition gives a name or a number, and no result that analysis can compare with a limit.",
                                f"The text is {quote(r.text)}."],
                     suggestion=f"Change the verification method of {r.id} to review.", values=values)
            continue
        rationale = [f"The verification method of {r.id} is analysis.",
                     "The requirement has no criterion: no result quantity, no location and no limit."]
        if r.parameters:
            rationale.append(f"It gives {count(len(r.parameters), 'parameter')} ({some(r.parameters, 4)}), "
                             f"but a parameter is not a limit for a result.")
        rationale.append(f"The text is {quote(r.text)}.")
        if vague:
            rationale.append(f"The text uses {join(quote(w, True) for w in vague)}, which gives no measure.")
        env.find(subject=r.id, key="not-verifiable", title=f"Analysis cannot verify {r.id}",
                 statement=f"Analysis cannot verify {r.id}, because the requirement has no criterion.",
                 rationale=rationale,
                 suggestion=[f"Add a criterion to {r.id}: a quantity, a measure, a location, a limit and the phases.",
                             f"If analysis is not the correct method, change the verification method of {r.id}."],
                 values=values)


# --------------------------------------------------------------------------- the text: obligation, singular, vague


@rule("req.no_shall", "minor", "quality", 'The text of a requirement that is an obligation uses "shall".')
def no_shall(env: Env) -> None:
    for r in env.reqs:
        if r.kind == "definition" or wording.obligations(r.text):
            continue
        other = wording.modals(r.text)
        rationale = [f"The kind of {r.id} is {r.kind}, thus the requirement is an obligation of the part.",
                     f"The text is {quote(r.text)}."]
        if other:
            rationale.append(f"The text uses {join(quote(w) for w in other)} for the obligation.")
        env.find(subject=r.id, key="no-shall", title=f'No "shall" in {r.id}',
                 statement=f'The text of {r.id} has no "shall".', rationale=rationale,
                 suggestion=[f'Write the obligation of {r.id} with "shall".',
                             "If the text is a definition, change the kind of the requirement to definition."],
                 values={"shall": 0})


@rule("req.not_singular", "minor", "quality", 'The text of a requirement has one "shall" only.')
def not_singular(env: Env) -> None:
    for r in env.reqs:
        n = wording.obligations(r.text)
        if n < 2:
            continue
        env.find(subject=r.id, key="not-singular", title=f"{r.id} is not one obligation",
                 statement=[f'The text of {r.id} has {n} "shall".', "A requirement must give one obligation."],
                 rationale=[f"The text is {quote(r.text)}.",
                            'Each "shall" is an obligation with its own verification and its own verdict.'],
                 suggestion=f"Divide {r.id} into {n} requirements with one obligation each.",
                 values={"shall": n})


_VAGUE_WHY = {      # kind -> (one word, two words or more)
    "measure": ("gives no level that a verification can show", "give no level that a verification can show"),
    "escape": ("lets the reader decide the level", "let the reader decide the level"),
    "open": ("shows that a value is not final", "show that a value is not final"),
    "list": ("makes the list open", "make the list open"),
    "ambiguous": ("permits two readings", "permit two readings"),
    "approximate": ("comes before a number that has no tolerance", "come before numbers that have no tolerance"),
}
_VAGUE_FIX = {
    "measure": "Replace {words} with a measurable limit: a quantity, a value and a unit.",
    "escape": "Replace {words} with a measurable limit: a quantity, a value and a unit.",
    "open": "Replace {words} with the final value.",
    "list": "Replace {words} with the complete list.",
    "ambiguous": 'Replace {words} with "and" or with "or".',
    "approximate": "Give the number after {words} with a tolerance.",
}


@rule("req.vague_word", "minor", "quality",
      'The text of a requirement has no word without a measure ("sufficiently", "adequate", "as required", "TBD").')
def vague_word(env: Env) -> None:
    for r in env.reqs:
        hits = wording.vague(r.text)
        if not hits:
            continue
        kinds = list(dict.fromkeys(h.kind for h in hits))
        # An open value is always major. Other words are major when no criterion gives the measure in their place.
        major = "open" in kinds or (not r.criteria and any(k != "approximate" for k in kinds))
        quoted = join(quote(h.word, True) for h in hits)
        rationale = [f"The text is {quote(r.text)}."]
        for k in kinds[:3]:
            ws = [quote(h.word, True) for h in hits if h.kind == k]
            rationale.append(f"{join(ws)} {_VAGUE_WHY[k][len(ws) > 1]}.")
        if r.criteria:
            rationale.append(f"{r.id} has {count(len(r.criteria), 'criterion', 'criteria')} that "
                             f"{'gives' if len(r.criteria) == 1 else 'give'} a measure, but the text does not.")
        else:
            rationale.append(f"{r.id} has no criterion that gives a measure in the place of the text.")
        fixes = list(dict.fromkeys(_VAGUE_FIX[k] for k in kinds))
        suggestion = []
        for fix in fixes[:3]:
            ws = [quote(h.word, True) for h in hits if _VAGUE_FIX[h.kind] == fix]
            suggestion.append(fix.format(words=join(ws)))
        one = len(hits) == 1 and " " not in hits[0].word
        env.find(subject=r.id, key="vague", title=f"{'Word' if one else 'Words'} without a measure in {r.id}",
                 statement=[f"The text of {r.id} uses {quoted}.",
                            f"{'This word gives' if one else 'These words give'} no limit that a verification can measure."],
                 rationale=rationale, suggestion=suggestion, severity="major" if major else "minor",
                 values={"words": [h.word for h in hits], "kinds": kinds, "criteria": len(r.criteria)})


# --------------------------------------------------------------------------- criteria that cannot be evaluated


def _criteria(env: Env):
    for r in env.reqs:
        for i, c in enumerate(r.criteria):
            yield r, i, c


def _of(req: Requirement, i: int) -> str:
    return f"{criterion_name(req, i)} of {req.id}"


@rule("req.criterion_vocabulary", "error", "quality",
      "Each criterion names a result quantity and a measure that the analysis gives.")
def criterion_vocabulary(env: Env) -> None:
    for r, i, c in _criteria(env):
        for what, value, known in (("quantity", c.quantity, tuple(QUANTITY_UNIT)), ("measure", c.measure, MEASURES)):
            if value in known:
                continue
            near = nearest(value, known)
            env.find(subject=r.id, key=f"{criterion_key(r, i)}-{what}", title=f"Unknown {what} in a criterion of {r.id}",
                     statement=[f"{cap(_of(r, i))} asks for the {what} {quote(value)}.",
                                f"The analysis gives no result with this {what}."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                f"The analysis knows these {'quantities' if what == 'quantity' else 'measures'}: "
                                f"{join(known)}."],
                     suggestion=(f"Change the {what} of the criterion to {near}, if that is the correct {what}." if near
                                 else f"Change the {what} of the criterion to one that the analysis knows."),
                     values={"criterion": i, what: value, "known": list(known)})


def _location_kind(env: Env, location: str) -> str | None:
    comp = env.comp
    if location in FIXED_LOCATIONS:
        return location
    if location in comp.features:
        return "feature"
    if any(it.id == location for it in comp.interfaces):
        return "interface"
    if any(s.id == location for s in comp.scls):
        return "scl"
    return None


def _axis_known(env: Env, features: list[str]) -> bool | None:
    """Does one of the features have an axis? None when the geometry stage does not say."""
    measured = env.geo.get("features") if isinstance(env.geo.get("features"), dict) else None
    if measured is None:
        return None
    seen = [measured[f] for f in features if isinstance(measured.get(f), dict) and "axis" in measured[f]]
    if len(seen) < len(features):
        return None
    return any(f["axis"] is not None for f in seen)


@rule("req.criterion_location", "error", "quality",
      "Each criterion names a location that the component has and that can give the result.")
def criterion_location(env: Env) -> None:
    comp = env.comp
    ids = [*comp.features, *(it.id for it in comp.interfaces), *(s.id for s in comp.scls)]
    for r, i, c in _criteria(env):
        key = criterion_key(r, i)
        kind = _location_kind(env, c.location)
        if kind is None:
            near = nearest(c.location, [*FIXED_LOCATIONS, *ids])
            env.find(subject=r.id, key=f"{key}-location", title=f"Unknown location in a criterion of {r.id}",
                     statement=[f"{cap(_of(r, i))} has the location {quote(c.location)}.",
                                f"The component {comp.id} has no feature, interface or stress classification line "
                                f"with this id."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                "A location is part, scls, or the id of a feature, an interface or a stress "
                                "classification line.",
                                f"The component {comp.id} has {count(len(comp.features), 'feature')}, "
                                f"{count(len(comp.interfaces), 'interface')} and "
                                f"{count(len(comp.scls), 'stress classification line')}."],
                     suggestion=(f"Change the location of the criterion to {near}, if that is the correct location." if near
                                 else f"Change the location of the criterion to an id of the component {comp.id}."),
                     values={"criterion": i, "location": c.location})
            continue
        if kind == "scls" and not comp.scls:
            env.find(subject=r.id, key=f"{key}-location", title=f"No stress classification line for a criterion of {r.id}",
                     statement=[f"{cap(_of(r, i))} has the location scls.",
                                f"The component {comp.id} has no stress classification line."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                "The location scls is the worst result of all stress classification lines."],
                     suggestion=f"Add a stress classification line to the component {comp.id}, or change the location.",
                     values={"criterion": i, "location": c.location, "scls": 0})
            continue
        if c.measure in SECTION_MEASURES and kind not in ("scls", "scl"):
            env.find(subject=r.id, key=f"{key}-location", title=f"Section stress away from a line in {r.id}",
                     statement=[f"{cap(_of(r, i))} asks for the measure {c.measure} at {c.location}.",
                                "This measure is a section stress, which only a stress classification line gives."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                f"The location {c.location} is {'the full part' if kind == 'part' else 'a ' + kind}, "
                                f"not a stress classification line."],
                     suggestion=["Change the location to scls or to the id of a stress classification line.",
                                 "If a surface stress is correct, change the measure to peak."],
                     values={"criterion": i, "location": c.location, "measure": c.measure})
            continue
        if c.quantity == "radial_displacement":
            feats = ([c.location] if kind == "feature" else
                     [f for it in comp.interfaces if it.id == c.location for fs in it.features.values() for f in fs
                      if f in comp.features] if kind == "interface" else [])
            axis = _axis_known(env, feats) if feats else False
            if axis is False:
                what = {"part": "the full part", "scls": "a set of stress classification lines",
                        "scl": "a stress classification line"}.get(kind, f"a {kind} without an axis")
                env.find(subject=r.id, key=f"{key}-location", title=f"No axis for a radial displacement in {r.id}",
                         statement=[f"{cap(_of(r, i))} asks for radial_displacement at {c.location}.",
                                    "This location has no axis."],
                         rationale=[f"The result key of the criterion is {c.key()}.",
                                    "The quantity radial_displacement is the displacement from the axis of a "
                                    "cylindrical feature.",
                                    f"The location {c.location} is {what}."],
                         suggestion="Change the location of the criterion to a cylindrical feature.",
                         values={"criterion": i, "location": c.location})


def _material_values(env: Env) -> dict[str, Value]:
    """The properties of the material that are quantities: name -> value."""
    m = env.material
    return {name: getattr(m, name) for name in type(m).model_fields if isinstance(getattr(m, name), Value)}


def _unit_words(unit: str) -> str:
    return f"in {unit}" if unit else "without a unit"


@rule("req.criterion_allowable", "error", "quality",
      "Each allowable is a property of the material of the part, with the unit of the result quantity.")
def criterion_allowable(env: Env) -> None:
    m = env.material
    have = _material_values(env)
    for r, i, c in _criteria(env):
        if c.allowable is None:
            continue
        key = f"{criterion_key(r, i)}-allowable"
        if c.allowable not in have:
            given = c.allowable in type(m).model_fields
            near = None if given else nearest(c.allowable, have)
            env.find(subject=r.id, key=key, title=f"Allowable of a criterion of {r.id} is not a material property",
                     statement=[f"{cap(_of(r, i))} has the allowable {quote(c.allowable)}.",
                                (f"The material {m.id} gives no value for this property." if given and getattr(m, c.allowable) is None
                                 else f"The material {m.id} has no property with this name that is a quantity.")],
                     rationale=[f"The material of the part is {m.id} ({quote(m.name)}).",
                                f"Its properties with a value are: {join(have)}."],
                     suggestion=(f"Change the allowable to {near}, if that is the correct property." if near
                                 else f"Give the property in the material file, or change the allowable of the criterion."),
                     values={"criterion": i, "allowable": c.allowable}, links=[("evidence", m.id)])
            continue
        unit = QUANTITY_UNIT.get(c.quantity)
        q = have[c.allowable].quantity()
        if unit is not None and not units.compatible(q, unit):
            env.find(subject=r.id, key=key, title=f"Allowable of a criterion of {r.id} has the wrong unit",
                     statement=[f"The allowable {c.allowable} of {_of(r, i)} is {units.display(q)}.",
                                f"The quantity {c.quantity} needs a limit {_unit_words(unit)}."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                f"The material {m.id} gives {c.allowable} = {units.display(q)}."],
                     suggestion="Change the allowable or the quantity of the criterion.",
                     values={"criterion": i, "allowable": c.allowable, "unit": units.si_unit(q), "needs": unit},
                     links=[("evidence", m.id)])


@rule("req.criterion_limit", "error", "quality", "Each limit gives one number with the unit of the result quantity.")
def criterion_limit(env: Env) -> None:
    for r, i, c in _criteria(env):
        if c.limit is None:
            continue
        key = f"{criterion_key(r, i)}-limit"
        written = quote(c.limit.expr) if not c.limit.is_literal else None
        reads = [n for n in c.limit.names() if n in env.ctx.params]
        try:
            q = c.limit.quantity(env.names)
        except (units.UnitsError, ValueError, TypeError) as e:
            env.find(subject=r.id, key=key, title=f"Limit of a criterion of {r.id} has no value",
                     statement=f"The validation cannot calculate the limit of {_of(r, i)}.",
                     rationale=[f"The limit is the expression {written}." if written else "The limit is a literal value.",
                                f"The error is {quote(e)}."],
                     suggestion="Correct the expression of the limit. It can read the parameters of the project.",
                     values={"criterion": i, "limit": str(c.limit.expr)}, links=[("about", n) for n in reads])
            continue
        if units.is_vector(q):
            env.find(subject=r.id, key=key, title=f"Limit of a criterion of {r.id} is a vector",
                     statement=[f"The limit of {_of(r, i)} is the vector {units.display(q)}.", "A limit must be one number."],
                     rationale=[f"The limit is the expression {written}." if written else "The limit is a literal value.",
                                f"The result key of the criterion is {c.key()}."],
                     suggestion="Give one number as the limit, for example one component of the vector.",
                     values={"criterion": i}, links=[("about", n) for n in reads])
            continue
        unit = QUANTITY_UNIT.get(c.quantity)
        if unit is not None and not units.compatible(q, unit):
            env.find(subject=r.id, key=key, title=f"Limit of a criterion of {r.id} has the wrong unit",
                     statement=[f"The limit of {_of(r, i)} is {units.display(q)}.",
                                f"The quantity {c.quantity} needs a limit {_unit_words(unit)}."],
                     rationale=[f"The result key of the criterion is {c.key()}.",
                                f"The limit is the expression {written}." if written else "The limit is a literal value."],
                     suggestion="Change the limit or the quantity of the criterion.",
                     values={"criterion": i, "limit": units.si(q), "unit": units.si_unit(q), "needs": unit},
                     links=[("about", n) for n in reads])


@rule("req.criterion_factor", "error", "quality",
      "Each factor of a criterion is a number, a requirement parameter or a factor of the policy.")
def criterion_factor(env: Env) -> None:
    policy = env.policy
    for r, i, c in _criteria(env):
        key = f"{criterion_key(r, i)}-factor"
        f = c.factor
        if isinstance(f, str):
            try:
                f = float(f)                                   # "1.5" written as text is a number
            except ValueError:
                pass
        if isinstance(f, (int, float)):
            if not f > 0:
                env.find(subject=r.id, key=key, title=f"Factor of a criterion of {r.id} is not positive",
                         statement=[f"The factor of {_of(r, i)} is {number(f)}.", "A factor must be more than zero."],
                         rationale=["The margin of safety divides the allowable by the factor and by the result.",
                                    f"The result key of the criterion is {c.key()}."],
                         suggestion="Give a factor that is 1.0 or more.", values={"criterion": i, "factor": float(f)})
            continue
        if f in env.req_params:
            p = env.ctx.params[f]
            owner = env.req_params[f]
            if isinstance(p.si, list) or p.unit != "":
                env.find(subject=r.id, key=key, title=f"Factor of a criterion of {r.id} has a unit",
                         statement=[f"The factor {f} of {_of(r, i)} is {p.display}.",
                                    "A factor must be a number without a unit."],
                         rationale=[f"The requirement {owner} gives the parameter {f} = {p.display}."],
                         suggestion="Name a parameter that is a factor, or give the factor as a number.",
                         values={"criterion": i, "factor": f, "unit": p.unit}, links=[("about", f)])
            elif not p.si > 0:
                env.find(subject=r.id, key=key, title=f"Factor of a criterion of {r.id} is not positive",
                         statement=[f"The factor {f} of {_of(r, i)} is {p.display}.", "A factor must be more than zero."],
                         rationale=[f"The requirement {owner} gives the parameter {f} = {p.display}."],
                         suggestion=f"Correct the parameter {f} of {owner}.",
                         values={"criterion": i, "factor": p.si}, links=[("about", f)])
            continue
        if f in policy.factors:
            if not policy.factors[f].value > 0:
                env.find(subject=r.id, key=key, title=f"Factor of a criterion of {r.id} is not positive",
                         statement=[f"The policy factor {f} of {_of(r, i)} is {number(policy.factors[f].value)}.",
                                    "A factor must be more than zero."],
                         rationale=[f"The policy {policy.id} gives the factor {f}."],
                         suggestion=f"Correct the factor {f} of the policy.",
                         values={"criterion": i, "factor": policy.factors[f].value}, links=[("evidence", policy.id)])
            continue
        candidates = [n for n in env.req_params if env.ctx.params[n].unit == "" and not isinstance(env.ctx.params[n].si, list)]
        near = nearest(f, [*candidates, *policy.factors])
        where = ("a parameter of the system context" if f in env.ctx.params and env.ctx.params[f].origin == "system"
                 else "a property of the material" if f in env.ctx.params else None)
        env.find(subject=r.id, key=key, title=f"Unknown factor in a criterion of {r.id}",
                 statement=[f"{cap(_of(r, i))} has the factor {quote(f)}.",
                            (f"This name is {where}, not a requirement parameter or a policy factor." if where
                             else "No requirement parameter and no policy factor has this name.")],
                 rationale=[f"The requirement parameters without a unit are: {some(candidates, 8) or 'none'}.",
                            f"The factors of the policy {policy.id} are: {some(policy.factors, 8) or 'none'}."],
                 suggestion=(f"Change the factor to {near}, if that is the correct factor." if near
                             else "Give the factor as a number, or name a requirement parameter or a policy factor."),
                 values={"criterion": i, "factor": f}, links=[("evidence", policy.id)])


# --------------------------------------------------------------------------- parameters: text and roles


@rule("req.number_in_text", "minor", "quality", "The text of a requirement gives the value of each of its parameters.")
def number_in_text(env: Env) -> None:
    for r in env.reqs:
        for name, p in r.parameters.items():
            rp = env.ctx.params.get(name)
            if rp is None or not p.is_literal or isinstance(p.value, list) or isinstance(rp.si, list):
                continue                                       # a derived number or a vector: the text need not give it
            if wording.gives(r.text, float(p.value), float(rp.si), rp.unit) is not False:
                continue
            found = list(dict.fromkeys(n.shown for n in wording.numbers(r.text)))
            env.find(subject=r.id, key=f"text-{name}", title=f"Value of {name} is not in the text of {r.id}",
                     statement=f"The text of {r.id} does not give the value of its parameter {name}.",
                     rationale=[f"The parameter {name} of {r.id} is {rp.display}.",
                                (f"The text gives {'this number' if len(found) == 1 else 'these numbers'}: "
                                 f"{some((quote(x) for x in found), 8)}." if found else "The text gives no number."),
                                f"The text is {quote(r.text)}."],
                     suggestion=[f"Make the text of {r.id} and the parameter {name} agree.",
                                 "The text is the source, thus correct the parameter if the text is correct."],
                     values={"parameter": rp.si, "unit": rp.unit, "text_numbers": found}, links=[("about", name)])


@rule("req.parameter_role", "error", "quality",
      "Each parameter role is a role of the validation, and the unit of the parameter agrees with the role.")
def parameter_role(env: Env) -> None:
    for r in env.reqs:
        for name, p in r.parameters.items():
            rp = env.ctx.params.get(name)
            if not p.role or rp is None:
                continue
            if p.role not in PARAM_ROLES:
                near = nearest(p.role, PARAM_ROLES)
                env.find(subject=r.id, key=f"role-{name}", title=f"Unknown role of {name} in {r.id}", severity="minor",
                         statement=[f"The parameter {name} of {r.id} has the role {quote(p.role)}.",
                                    "The validation has no such role, thus no rule compares the parameter with the "
                                    "system context."],
                         rationale=[f"The roles are: {join(PARAM_ROLES)}."],
                         suggestion=[(f"Change the role to {near}, if that is the correct role." if near
                                      else "Change the role to one of the roles of the validation."),
                                     "If no role applies, remove the role and give the parameter a check."],
                         values={"role": p.role}, links=[("about", name)])
                continue
            unit = ROLE_UNIT[p.role]
            if isinstance(rp.si, list) or rp.unit != unit:
                env.find(subject=r.id, key=f"role-{name}", title=f"Unit of {name} does not agree with its role",
                         statement=[f"The parameter {name} of {r.id} has the role {p.role} and the value {rp.display}.",
                                    f"The role {p.role} needs {UNIT_WORDS.get(unit, 'a value in ' + unit)}."],
                         rationale=[f"The role {p.role} is the {ROLE_WORDS[p.role]} of the requirement.",
                                    "The validation cannot compare the parameter with the system context."],
                         suggestion=f"Correct the unit or the role of the parameter {name}.",
                         values={"role": p.role, "unit": rp.unit, "needs": unit}, links=[("about", name)])
