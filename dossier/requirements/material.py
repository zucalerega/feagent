"""Rules on the material of the part against the system context: temperatures and media.

The subject of these records is the material, so the compliance stage shows a major finding here as a caveat
of every margin (every margin rests on the material).
"""

from __future__ import annotations

import re

from dossier import units
from dossier.requirements.base import TEMPERATURE_KINDS, Env, quote, rule, shown, some

_COLD = re.compile(r"\b(cryogenic|low(?:er)?[ -]temperatures?|cold|sub-?zero)\b", re.I)
_HOT = re.compile(r"\b(elevated|high(?:er)?[ -]temperatures?|hot)\b", re.I)
_CONSERVATIVE = re.compile(r"\bconservative\b", re.I)
# a sentence with one of these words does not claim "conservative": it denies it, doubts it or assumes it
_DOUBT = re.compile(r"\b(?:not|no|never|non|cannot|unknown|uncertain|unclear|may|might|could|if|whether|unless|except|"
                    r"assum\w*|tbc|tbd)\b|n't|\?|\bun-?conservative\b|\bnon-?conservative\b", re.I)


def _temperatures(env: Env) -> list:
    """Every temperature that the system context gives on the part (fluids, walls, the ambient air)."""
    return [e for e in env.effects(kinds=TEMPERATURE_KINDS) if isinstance(e.si, (int, float))]


@rule("req.material_temperature", "major", "material",
      "The temperatures of the system context are in the service temperature range of the material.")
def material_temperature(env: Env) -> None:
    m = env.material
    temps = _temperatures(env)
    if not temps:
        return
    cold, hot = min(temps, key=lambda e: e.si), max(temps, key=lambda e: e.si)
    if m.service_temperature is None:
        env.find(subject=m.id, key="no-service-range", severity="minor", title=f"No service temperature range for {m.id}",
                 statement=[f"The material file gives no service temperature range for {m.id}.",
                            "The validation did not compare the temperatures of the system context with the material."],
                 rationale=[f"The system context gives temperatures from {cold.display} to {hot.display} on the part.",
                            f"The lowest temperature is the effect {cold.id} in {cold.phase}.",
                            f"The highest temperature is the effect {hot.id} in {hot.phase}."],
                 suggestion=f"Add the service temperature range of {m.id} to the material file, with its source.",
                 values={"context_low": cold.si, "context_high": hot.si, "unit": "K"},
                 links=[("evidence", cold.id), ("evidence", hot.id)])
        return
    lo, hi = (float(units.si(units.to(v.quantity(), "K"))) for v in m.service_temperature)
    for e, low in ((cold, True), (hot, False)):
        if (e.si >= lo - 1e-9 * max(abs(lo), 1.0)) if low else (e.si <= hi + 1e-9 * max(abs(hi), 1.0)):
            continue
        lines, names = env.context_lines(e, most=4)
        rationale = [f"The material file gives a service temperature range of {shown(lo, 'K')} to {shown(hi, 'K')} for {m.id}."]
        if m.source:
            rationale.append(f"The source of the material data is {quote(m.source)}.")
        rationale += lines
        gap = shown(lo - e.si if low else e.si - hi, "K")
        env.find(subject=m.id, key="too-cold" if low else "too-hot",
                 title=f"{m.id} is {'colder' if low else 'hotter'} than its service temperature range",
                 statement=f"The system context gives a temperature that is {gap} {'lower' if low else 'higher'} than the "
                           f"service temperature range of the material {m.id}.",
                 rationale=rationale[:6],
                 suggestion=[f"Use a material with a service temperature range that includes {e.display}.",
                             "If the range in the material file is wrong, correct it and give the source."],
                 values={"material_low": lo, "material_high": hi, "context": e.si, "unit": "K",
                         "inputs": {n: env.ctx.params[n].si for n in names}},
                 links=[("evidence", e.id), ("evidence", e.phase), *(("evidence", n) for n in names)])


@rule("req.material_compatibility", "major", "material",
      "The material is compatible with each medium of the fluid connections of the part.")
def material_compatibility(env: Env) -> None:
    m = env.material
    media: dict[str, list[str]] = {}
    for c in env.project.system.connections:
        if c.kind == "fluid" and c.medium.strip() and env.element in (c.from_, c.to):
            media.setdefault(c.medium.strip(), []).append(c.id)
    lower = {k.lower(): v for k, v in m.compatibility.items()}
    for medium, conns in media.items():
        rating = m.compatibility.get(medium) or lower.get(medium.lower())
        if rating == "ok":
            continue
        one = len(conns) == 1
        brings = (f"The {'connection' if one else 'connections'} {some(conns, 4)} "
                  f"{'brings' if one else 'bring'} {medium} to the part.")
        where = f"The source of the material data is {quote(m.source)}." if m.source else "The material file gives no source."
        common = dict(subject=m.id, key=f"medium-{medium}", values={"medium": medium, "rating": rating or "not rated"},
                      links=[("evidence", c) for c in conns])
        if rating == "incompatible":
            env.find(title=f"{m.id} is not compatible with {medium}",
                     statement=f"The material {m.id} of the part is not compatible with the medium {medium}.",
                     rationale=[brings, f"The material file rates {m.id} with {medium} as {quote(rating)}.", where],
                     suggestion=[f"Change the material of the part to one that is compatible with {medium}.",
                                 "If the rating is wrong, correct the material file and give the source."], **common)
        elif rating == "conditional":
            rationale = [brings, f"The material file rates {m.id} with {medium} as {quote(rating)}.", where]
            if m.note:
                rationale.append(f"The note of the material is {quote(m.note)}.")
            env.find(severity="minor", title=f"{m.id} is compatible with {medium} only with conditions",
                     statement=f"The material {m.id} of the part is compatible with the medium {medium} only with conditions.",
                     rationale=rationale,
                     suggestion=f"Record the conditions for {m.id} with {medium}, and make sure that the part obeys them.",
                     **common)
        else:
            rated = list(m.compatibility)
            env.find(severity="minor", title=f"No compatibility rating of {m.id} with {medium}",
                     statement=f"The material file gives no compatibility rating of {m.id} with the medium {medium}.",
                     rationale=[brings, (f"The material file rates {m.id} for {some(rated, 6, 'medium', 'media')}." if rated
                                         else f"The material file rates {m.id} for no medium.")],
                     suggestion=f"Add the rating of {m.id} with {medium} to the material file, with its source.", **common)


def _note_says_conservative(note: str, cold: bool, hot: bool) -> bool:
    """Does one sentence of the material note say, without doubt, that the properties are conservative on the
    side that matters (`cold`: the context is colder than the properties temperature; `hot`: hotter)? A
    sentence that names no side speaks for both. This reads words, not meaning: the record says so."""
    for sentence in re.split(r"(?<=[.!?;])\s+", note or ""):
        if not _CONSERVATIVE.search(sentence) or _DOUBT.search(sentence):
            continue
        says_cold, says_hot = bool(_COLD.search(sentence)), bool(_HOT.search(sentence))
        if not (says_cold or says_hot) or ((says_cold or not cold) and (says_hot or not hot)):
            return True
    return False


@rule("req.material_properties_temperature", "info", "material",
      "The material properties apply at the temperatures of the system context (if not, an assumption says so).",
      makes="assumption")
def material_properties_temperature(env: Env) -> None:
    m = env.material
    temps = _temperatures(env)
    if not temps:
        return
    cold, hot = min(temps, key=lambda e: e.si), max(temps, key=lambda e: e.si)
    delta = float(units.si(env.policy.thermal_delta.quantity(env.names)))
    span = f"from {cold.display} to {hot.display}" if hot.si - cold.si > 1e-9 else f"of {cold.display}"
    verify = (f"Compare the properties in the specification of {m.id} at {cold.display} and at {hot.display} with the "
              f"values in the material file.")
    if m.properties_temperature is None:
        env.assume(subject=m.id, key="properties-temperature", title=f"Properties of {m.id} at an unknown temperature",
                   statement=f"The properties of {m.id} in the material file apply at the temperatures of the system "
                             f"context, {span}.",
                   rationale=[f"The material file gives one set of properties for {m.id} and no temperature for them.",
                              f"The lowest temperature of the system context is {cold.display} (effect {cold.id}, {cold.phase}).",
                              f"The highest temperature is {hot.display} (effect {hot.id}, {hot.phase}).",
                              "The validation does not know if these properties are conservative at these temperatures."],
                   impact=["If the strength at a temperature of the system context is lower, the margins of safety are too high.",
                           "If the stiffness or the expansion is different, the thermal stresses are different."],
                   verify_by=verify, conservative=None, links=[("evidence", cold.id), ("evidence", hot.id)])
        return
    t0 = float(units.si(units.to(m.properties_temperature.quantity(), "K")))
    far = max((cold, hot), key=lambda e: abs(e.si - t0))
    if abs(far.si - t0) <= delta:
        return
    below, above = cold.si < t0 - delta, hot.si > t0 + delta
    says = _note_says_conservative(m.note, below, above)
    rationale = [f"The material file gives the properties of {m.id} at {shown(t0, 'K')} only.",
                 f"The system context gives {far.display} in the phase {far.phase} (effect {far.id}).",
                 f"The difference is {shown(abs(far.si - t0), 'K')}, and the policy limit is {shown(delta, 'K')}."]
    if m.note:
        rationale.append(f"The note of the material is {quote(m.note)}.")
    if says:                                            # the claim is the note's, and the record says so
        rationale += ['The note says "conservative" for these temperatures, thus this record says the same.',
                      "The validation did not check the note."]
        verify = [verify, "Make sure that the note applies to each property that the analysis uses."]
    elif m.note:
        rationale.append("The note does not say that these properties are conservative at these temperatures.")
    else:
        rationale.append("The material file has no note, thus the validation does not know if this is conservative.")
    env.assume(subject=m.id, key="properties-temperature",
               title=f"Properties of {m.id} at {shown(t0, 'K')} for temperatures {span}",
               statement=f"The properties of {m.id} at {shown(t0, 'K')} apply at all temperatures of the system context, "
                         f"{span}.",
               rationale=rationale,
               impact=[f"If the strength at a temperature of the system context is lower than at {shown(t0, 'K')}, the margins "
                       f"of safety are too high.",
                       "If the stiffness or the expansion is different, the thermal stresses are different."],
               verify_by=verify, conservative=True if says else None,
               links=[("evidence", far.id), ("evidence", far.phase), ("evidence", env.policy.id)])
