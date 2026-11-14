"""Small helpers for the text of the planner's records.

The text follows ASD-STE100 Simplified Technical English (dossier/CONTRACTS.md, "Text"): one topic per
sentence, 25 words or fewer in a descriptive sentence and 20 or fewer in an instruction, active voice,
simple tenses, the same word for the same thing. The helpers keep lists and counts regular
("the interfaces INT-A, INT-B and INT-C", "1 load", "3 loads") and choose a shorter wording when a list
makes a sentence too long (`fit`). Words the planner always uses: load case, phase, part, interface,
requirement, criterion, system context, plan, policy, thermal strain, temperature field, steady state.
"""

from __future__ import annotations

import re
from typing import Iterable

from dossier import units
from dossier.model import LoadItem, Phase

ANALYSIS_WORDS = {
    "static_structural": "static structural analysis",
    "thermo_structural": "thermo-structural analysis",
    "thermal_steady": "steady-state thermal analysis",
    "flow": "flow analysis",
    "fatigue": "fatigue analysis",
}

# (singular, plural) of each effect kind as a load
KIND_WORDS = {
    "pressure": ("pressure", "pressures"), "force": ("force", "forces"), "moment": ("moment", "moments"),
    "acceleration": ("acceleration", "accelerations"), "displacement": ("displacement", "displacements"),
    "fluid_temperature": ("fluid temperature", "fluid temperatures"), "temperature": ("temperature", "temperatures"),
    "heat_flux": ("heat flux", "heat fluxes"), "mass_flow": ("mass flow", "mass flows"),
    "support": ("support", "supports"),
}

DESCRIPTIVE = 25      # words in a descriptive sentence
INSTRUCTION = 20      # words in an instruction

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def words(text: str) -> int:
    return len(text.split())


def fit(*candidates: str, limit: int = DESCRIPTIVE) -> str:
    """The first wording whose sentences all have `limit` words or fewer; the last one when none has."""
    for c in candidates:
        if all(words(s) <= limit for s in sentences(c)):
            return c
    return candidates[-1]


def unique(items: Iterable) -> list:
    return list(dict.fromkeys(items))


def join(items: Iterable[str]) -> str:
    """A, B and C."""
    items = [str(x) for x in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def number(x: float, sig: int = 4) -> str:
    """A pure number for people: 1.667, 65, 0.6667."""
    return units.fmt(units.Q(float(x)), None, sig)


def count(n: float, singular: str, plural: str | None = None) -> str:
    """1 load, 3 loads, 2.5 cycles."""
    return f"{number(n)} {singular if float(n) == 1.0 else (plural or singular + 's')}"


def shown(si: float | list[float], unit: str) -> str:
    """An SI value in the unit people read (dossier.units.display)."""
    return units.display(units.from_si(si, unit))


def cap(text: str) -> str:
    """The first letter in upper case; the rest as it is (ids keep their case)."""
    return text[:1].upper() + text[1:]


def the(noun: str, ids: Iterable[str], plural: str | None = None) -> str:
    """the interface INT-A | the interfaces INT-A and INT-B."""
    ids = [str(x) for x in ids]
    return f"the {noun if len(ids) == 1 else (plural or noun + 's')} {join(ids)}"


def phase_ref(phase: Phase, name_words: int = 6) -> str:
    """PH-FIRE (Steady firing): the id and the name the engineer gave, when the name is short."""
    return f"{phase.id} ({phase.name})" if 0 < words(phase.name) <= name_words else phase.id


def interfaces_of(items: Iterable[LoadItem], most: int = 8) -> str:
    """Where loads act: "the interface INT-A", "the interfaces INT-A and INT-B", "12 interfaces"; a load
    without an interface acts on the whole part."""
    items = list(items)
    ids = unique(x.interface for x in items if x.interface)
    body = any(not x.interface for x in items)
    if not ids:
        return "the whole part"
    text = f"{len(ids)} interfaces" if len(ids) > most else the("interface", ids)
    return f"{text} and the whole part" if body else text


def kinds_of(items: Iterable[LoadItem]) -> str:
    """1 mass flow, 3 pressures and 3 fluid temperatures."""
    n: dict[str, int] = {}
    for x in items:
        n[x.kind] = n.get(x.kind, 0) + 1
    return join(count(k, *KIND_WORDS.get(kind, (kind, kind + "s"))) for kind, k in n.items())


def kind_word(kind: str) -> str:
    return KIND_WORDS.get(kind, (kind,))[0]


def holds(supports: list[str]) -> str:
    """The sentence that says what holds the part."""
    if not supports:
        return "No interface holds the part in this phase."
    if len(supports) == 1:
        return f"The interface {supports[0]} holds the part."
    return f"The interfaces {join(supports)} hold the part."


def criteria_of(by_requirement: dict[str, list[str]]) -> list[str]:
    """Wordings of "which criteria of which requirements", from the full one to the shortest.
    `by_requirement`: requirement id -> the labels of its criteria."""
    n = sum(len(v) for v in by_requirement.values())
    full = join(f"the {'criterion' if len(labels) == 1 else 'criteria'} {join(labels)} of {rid}"
                for rid, labels in by_requirement.items())
    short = f"{count(n, 'criterion', 'criteria')} of {the('requirement', by_requirement)}"
    shortest = f"{count(n, 'criterion', 'criteria')} of {count(len(by_requirement), 'requirement')}"
    return [full, short, shortest]
