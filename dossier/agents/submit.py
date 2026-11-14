"""What a role submits, and the check that turns a submission into records.

A submission is one JSON document in a reduced form that a language model can fill reliably (`Submission`:
`findings`, `decisions`, `assumptions`). A language model can invent things, so nothing of a submission
becomes a record before `check` has passed it.

    check(project, role, data, by="")    -> Result      checks only, writes nothing
    submit(project, role, data, by="")   -> Result      checks, and writes <project>/agent/<role>.json when accepted
    schema(role)                         -> dict        the JSON schema of the submission

The whole submission is REJECTED (Result.errors, nothing is written) when
    * a `subject` or an `evidence` id is not a record or an input of the project (project.inputs(),
      project.records(), the names of the context), or the id of a new record of the same submission;
    * a required text is empty, a field is missing or unknown, a `key` is not a short slug;
    * a `severity`, a `category` or a `confidence` is not in the vocabulary;
    * a finding duplicates an existing finding: the same subject and a similar statement. "Similar" is
      `similarity(a, b) >= SIMILAR`: the share of the significant words (and numbers) of the shorter
      statement that the other statement also has. The findings of an earlier submission of the same role
      do not count, because the new submission replaces them;
    * the id of a new record is the id of a record that another stage file holds, or two records of the
      submission have the same subject and key.

The submission is ACCEPTED WITH A WARNING (Result.warnings) when
    * a number with a unit in a rationale (for a finding also in its statement) matches no number of the
      context, the inputs or the records within 1 % (compared in SI through dossier.units): a number that the
      role calculated itself, or invented;
    * a text names an id with a prefix of the project (REQ-, PH-, EF-, LC-, ...) that the dossier does not have;
    * `dossier.ste.check` reports issues in a text;
    * a decision gives no alternative.

Accepted records are full Finding / Decision / Assumption records:
    id            FND-A-<subject>-<key>, DEC-A-..., ASM-A-...  (make_id; the "A" keeps them apart from the
                  ids that rules make, and `check` refuses an id that another file already uses)
    origin        by = "agent:<role>" or "agent:<role> (<model name>)"; stage = validate | plan | solve by role
    links         one `evidence` link per evidence id
    decisions     basis "engineering_judgement", needs_review = True
    findings      rule "agent.<role>"; `values` holds the numbers of the text that the dossier confirms (SI)
Where the reviewer sees the confidence and the warnings: in the last sentences of the record's `rationale`
("The agent gives this finding a medium confidence. The submission check does not find 11 MPa in the
dossier."). The stage file also holds them as data under `agent.records[<id>]`, with the submission as it
came in under `submission`.

A role never gives a review, never changes an input file and never changes a record of a rule: this module
writes one file, <project>/agent/<role>.json, and a second submission of the same role replaces it.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from dossier import context as context_mod
from dossier import model, units
from dossier.agents import roles as roles_mod
from dossier.model import Alternative, Assumption, Decision, Finding, Link, Origin
from dossier.project import Project, make_id

# =========================================================================== the submission (schema)


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


_SUBJECT = "The id of the one record that the proposal is about. It must be an id of the dossier."
_KEY = ("A short slug that names the proposal: lower-case letters, digits and hyphens, for example "
        "'surge-estimate'. The record id is made from the subject and the key.")
_EVIDENCE = "The ids of every record that the claim rests on. Each must be an id of the dossier."
_CONFIDENCE = ("high: the dossier shows it directly. medium: engineering judgement that a person must confirm. "
               "low: a question worth an answer.")


class FindingIn(_In):
    """A problem in the dossier."""

    subject: str = Field(description=_SUBJECT)
    key: str = Field(description=_KEY)
    title: str = Field(description="A short title, a few words.")
    severity: model.Severity = Field(description="error | major | minor | info")
    category: model.FindingCategory = Field(
        description="consistency | coverage | quality | policy | conflict | material | context | model")
    statement: str = Field(description="What is wrong, in one to three sentences of Simplified Technical English.")
    rationale: str = Field(description="The evidence: the record ids and the numbers with their units, as the "
                                       "dossier shows them.")
    suggestion: str = Field(default="", description="The correction that you propose.")
    evidence: list[str] = Field(description=_EVIDENCE)
    confidence: Literal["high", "medium", "low"] = Field(description=_CONFIDENCE)


class AlternativeIn(_In):
    option: str = Field(description="Another choice.")
    why_not: str = Field(description="Why you do not propose it.")


class DecisionIn(_In):
    """A choice that the role proposes and that a person must make."""

    subject: str = Field(description=_SUBJECT)
    key: str = Field(description=_KEY)
    title: str = Field(description="A short title, a few words.")
    choice: str = Field(description="What you propose to do.")
    rationale: str = Field(description="Why, with the record ids and the numbers.")
    alternatives: list[AlternativeIn] = Field(description="The other choices and why you do not propose them.")
    evidence: list[str] = Field(description=_EVIDENCE)
    confidence: Literal["high", "medium", "low"] = Field(description=_CONFIDENCE)


class AssumptionIn(_In):
    """Something that the dossier takes as true without proof."""

    subject: str = Field(description=_SUBJECT)
    key: str = Field(description=_KEY)
    title: str = Field(description="A short title, a few words.")
    statement: str = Field(description="What the dossier takes as true.")
    rationale: str = Field(description="Why that is acceptable, or why the dossier does not show it.")
    impact: str = Field(description="What changes if the assumption is wrong.")
    verify_by: str = Field(default="", description="How a person can check it.")
    conservative: bool | None = Field(default=None, description="true: errs to the safe side. false: does not. "
                                                                "null: not known.")
    evidence: list[str] = Field(description=_EVIDENCE)
    confidence: Literal["high", "medium", "low"] = Field(description=_CONFIDENCE)


class Submission(_In):
    """What a judgement role submits: proposals for a person to review. Every list can be empty."""

    summary: str = Field(default="", description="One or two sentences: what you reviewed and what you found.")
    findings: list[FindingIn] = Field(default_factory=list)
    decisions: list[DecisionIn] = Field(default_factory=list)
    assumptions: list[AssumptionIn] = Field(default_factory=list)


KINDS = {"findings": (FindingIn, "FND", "finding"), "decisions": (DecisionIn, "DEC", "decision"),
         "assumptions": (AssumptionIn, "ASM", "assumption")}
# the texts that must not be empty, per kind
REQUIRED_TEXT = {"findings": ("subject", "key", "title", "statement", "rationale"),
                 "decisions": ("subject", "key", "title", "choice", "rationale"),
                 "assumptions": ("subject", "key", "title", "statement", "rationale", "impact")}
# the texts that the number check reads (a suggestion, a choice or an impact can hold a new number)
NUMBER_TEXT = {"findings": ("statement", "rationale"), "decisions": ("rationale",), "assumptions": ("rationale",)}
# the texts that the STE check reads
STE_TEXT = {"findings": ("title", "statement", "rationale", "suggestion"),
            "decisions": ("title", "choice", "rationale"),
            "assumptions": ("title", "statement", "rationale", "impact", "verify_by")}
KEY_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
KEY_MAX = 48


def schema(role: "str | roles_mod.Role | None" = None) -> dict:
    """The JSON schema of the submission (the same form for every role; the title names the role)."""
    s = Submission.model_json_schema()
    if role is not None:
        r = roles_mod.get(role)
        s["title"] = f"Submission of the role {r.name}"
        s["description"] = (f"{r.title}: proposals for a person to review. Every list can be empty. "
                            f"The records get origin.by = agent:{r.name} and origin.stage = {r.stage}.")
    return s


# =========================================================================== numbers with units in a text

# Unit symbols that count as a unit after a number. A fixed list: pint also reads "in", "at", "a" and "as"
# as units, and those are English words.
_ATOMS = ("GPa", "MPa", "kPa", "mbar", "bar", "psi", "Pa", "MN", "kN", "mN", "N", "min", "mm", "cm", "km", "µm",
          "μm", "um", "nm", "ms", "mg", "kg", "g0", "g", "kW", "MW", "mW", "W", "kJ", "MJ", "J", "kHz", "Hz",
          "degC", "°C", "K", "m", "s", "h")
_A = "(?:" + "|".join(re.escape(a) for a in _ATOMS) + r")(?:\^-?\d+|\*\*-?\d+|[²³])?"
_UNIT = rf"(?:1/)?{_A}(?:[*·/]\(?{_A}(?:[*·]{_A})*\)?)*"
QUANTITY_RE = re.compile(
    rf"(?<![\w.,])(?P<num>[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)[    ]?(?P<unit>%|{_UNIT})(?![\w°µμ%])")
_NORMAL = (("−", "-"), ("·", "*"), ("²", "^2"), ("³", "^3"), ("µ", "u"), ("μ", "u"), ("°C", "degC"))


@dataclass(frozen=True)
class Quoted:
    text: str        # as written: "10.72 MPa"
    si: float        # magnitude in SI
    unit: str        # SI unit (dossier.units.si_unit): "Pa", "" for a percentage
    alt: tuple[float, str] | None = None    # a second reading: "25 g" is 25 grams, or 25 times standard gravity


def _parse(num: str, unit: str) -> tuple[float, str] | None:
    s = f"{num} {unit}"
    for a, b in _NORMAL:
        s = s.replace(a, b)
    try:
        q = units.parse(s)
        v = units.si(q)
    except Exception:  # noqa: BLE001  pint raises several kinds; a text that is no quantity is not checked
        return None
    if isinstance(v, list) or not math.isfinite(v):
        return None
    return float(v), units.si_unit(q)


def quantities(text: str) -> list[Quoted]:
    """The numbers with a unit in a text ("10.72 MPa", "15570 W/(m²·K)", "2 %"), in SI."""
    out = []
    for m in QUANTITY_RE.finditer(text or ""):
        got = _parse(m["num"], m["unit"])
        if got is not None:
            alt = _parse(m["num"], "g0") if m["unit"] == "g" else None
            out.append(Quoted(text=" ".join(m.group(0).split()), si=got[0], unit=got[1], alt=alt))
    return out


@lru_cache(maxsize=None)
def _dim(unit: str | None):
    """The dimension of an SI unit text (so that "N*m" and "J" are the same); None when it is not known."""
    if unit is None:
        return None
    try:
        return units.ureg.parse_units(unit).dimensionality if unit else units.ureg.parse_units("").dimensionality
    except Exception:  # noqa: BLE001
        return None


class Numbers:
    """Every number of the dossier in SI: the context, the inputs, the records and the geometry stage.
    Each number has the SI unit where the dossier gives one (None: the unit is not known, the number
    matches by magnitude)."""

    def __init__(self) -> None:
        self.items: list[tuple[float, str | None, str]] = []      # (|value|, SI unit, where it is)

    def add(self, value: Any, unit: str | None, where: str, vector: bool = False) -> None:
        """One number, or a list of numbers; `vector`: the list is a vector, so its magnitude is a number too."""
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (list, tuple)):
            comps = [float(v) for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
            for v in comps:
                self.add(v, unit, where)
            if vector and len(comps) == 3:
                self.add(math.sqrt(sum(c * c for c in comps)), unit, where)
            return
        if isinstance(value, (int, float)) and math.isfinite(value):
            self.items.append((abs(float(value)), unit, where))

    def add_text(self, text: Any, where: str) -> None:
        if isinstance(text, str) and text:
            for q in quantities(text):
                self.add(q.si, q.unit, where)

    def match(self, si: float, unit: str | None, rtol: float = 0.01) -> tuple[float, str | None, str] | None:
        """The number of the dossier that is nearest to `si` and within `rtol` of it (same dimension, or
        a number whose unit the dossier does not give). The sign does not count."""
        d, x, best = _dim(unit), abs(si), None
        for v, u, where in self.items:
            if u is not None and _dim(u) != d:
                continue
            err = abs(x - v) / v if v else (0.0 if x == 0.0 else math.inf)
            if err <= rtol and (best is None or err < best[0]):
                best = (err, (v, u, where))
        return best[1] if best else None

    def nearest(self, si: float, unit: str | None) -> tuple[float, str | None, str] | None:
        """The number of the same dimension that is nearest (for the text of a warning)."""
        d, x, best = _dim(unit), abs(si), None
        for v, u, where in self.items:
            if u is None or _dim(u) != d or v == 0.0 or x == 0.0:
                continue
            err = abs(math.log(x / v))
            if best is None or err < best[0]:
                best = (err, (v, u, where))
        return best[1] if best else None


# keys of the stage files (JSON): which unit a number under the key has
_SIBLING_UNIT = ("si", "value", "per_level", "extrapolated", "conservative_value", "limit")   # the "unit" next to it
_KEY_UNIT = {"h_si": "W/(m^2*K)", "at_si": "m", "location": "m", "h": "m", "h_rep": "m", "uncertainty": "",
             "order": "", "factor": "", "margin": "", "ratio": "", "band": "", "scale": "", "count": "",
             "min_quality": "", "runtime_s": "s"}
_SKIP_KEY = ("nodes", "cells", "dofs", "level", "criterion", "schema")
_GEO_UNIT = {"area": "m^2", "radius": "m", "diameter": "m", "length": "m", "volume": "m^3", "mass": "kg", "bbox": "m"}


def _walk_json(obj: Any, pool: Numbers, where: str, key: str = "", unit: str | None = None) -> None:
    if isinstance(obj, dict):
        here = obj.get("id") if isinstance(obj.get("id"), str) else where
        sibling = obj.get("unit") if isinstance(obj.get("unit"), str) else None
        for k, v in obj.items():
            if k in _SKIP_KEY:
                continue
            u = sibling if (k in _SIBLING_UNIT and sibling is not None) else _KEY_UNIT.get(k)
            _walk_json(v, pool, here, k, u)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            pool.add(list(obj), unit, where, vector=key in ("si", "value"))
        else:
            for v in obj:
                _walk_json(v, pool, where, key, unit)
    elif isinstance(obj, str):
        pool.add_text(obj, where)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        pool.add(obj, unit, where)


def _walk_input(obj: Any, pool: Numbers, where: str, names: dict) -> None:
    if isinstance(obj, model.Value):
        try:
            q = obj.quantity(names)
            pool.add(units.si(q), units.si_unit(q), where, vector=True)
        except Exception:  # noqa: BLE001  an expression of a load case (load(), feat()) has no value here
            pass
        for f in ("title", "note", "source"):
            pool.add_text(getattr(obj, f, ""), where)
        if getattr(obj, "check", None) is not None:
            _walk_input(obj.check, pool, where, names)
    elif isinstance(obj, model.Scl):
        pool.add(list(obj.start), "m", obj.id)
        pool.add(list(obj.end), "m", obj.id)
        pool.add_text(obj.description, obj.id)
    elif isinstance(obj, (model.FeatureDef, model.Cad)):
        pool.add_text(getattr(obj, "description", ""), where)
    elif isinstance(obj, BaseModel):
        here = getattr(obj, "id", None) or where
        for name in type(obj).model_fields:
            if name != "position":
                _walk_input(getattr(obj, name), pool, here if isinstance(here, str) else where, names)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            named = isinstance(k, str) and isinstance(v, (model.Value, model.FeatureDef))
            _walk_input(v, pool, (k if k in names or isinstance(v, model.FeatureDef) else f"{where}.{k}") if named
                        else where, names)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_input(v, pool, where, names)
    elif isinstance(obj, str):
        pool.add_text(obj, where)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        pool.add(obj, "", where)


def known_numbers(project: Project, ctx: context_mod.Context | None = None, skip_role: str | None = None) -> Numbers:
    """Every number that a role can quote from the dossier. `skip_role`: leave out the records that this
    role proposed before (its new submission replaces them)."""
    ctx = ctx or context_mod.load(project)
    pool = Numbers()
    for name, p in ctx.params.items():
        pool.add(p.si, p.unit, name, vector=True)
    for e in ctx.effects:
        pool.add(e.si, e.unit, e.id, vector=True)
        pool.add(e.h_si, "W/(m^2*K)", e.id)
        pool.add(e.at_si, "m", e.id)
    for dc, n in ctx.duty_cycles.items():
        pool.add(n, "", dc)
    names = ctx.names()
    _walk_input(project.system, pool, project.system.id, names)
    for comp in project.components.values():
        _walk_input(comp, pool, comp.id, names)
        for r in project.requirements.get(comp.id, []):
            _walk_input(r, pool, r.id, names)
    for m in project.materials.values():
        _walk_input(m, pool, m.id, names)
    _walk_input(project.policy, pool, project.policy.id, names)
    for f in project.stage_files():
        doc = json.loads(f.read_text())
        for key in (*KINDS, "load_cases", "hand_calcs", "analyses", "compliance", "verdicts"):
            for rec in doc.get(key, []):
                if skip_role and roles_mod.role_of((rec.get("origin") or {}).get("by", "")) == skip_role:
                    continue
                _walk_json(rec, pool, rec.get("id") or rec.get("requirement") or f.stem)
        geo = doc.get("geometry") if f.stem == "geometry" else None
        if isinstance(geo, dict):
            cad = geo.get("cad") or {}
            for k in ("volume", "mass", "bbox"):
                v = cad.get(k)
                pool.add([c for row in v for c in row] if k == "bbox" and v else v, _GEO_UNIT[k], "geometry")
            for fid, feat in (geo.get("features") or {}).items():
                for k in ("area", "radius", "diameter", "length"):
                    pool.add(feat.get(k), _GEO_UNIT[k], fid)
            for sid, scl in (geo.get("scls") or {}).items():
                pool.add(scl.get("length"), "m", sid)
            for edge in (geo.get("edges") or {}).get("reentrant", []) or []:
                pool.add(edge.get("length"), "m", "geometry")
    return pool


def _show(si: float, unit: str | None) -> str:
    try:
        return units.display(units.from_si(si, unit or ""))
    except Exception:  # noqa: BLE001
        return f"{si:g} {unit or ''}".strip()


# =========================================================================== similar statements

SIMILAR = 0.6
_STOP = frozenset("""a an and are as at be by can does do for from gives give has have in is it its must no not of on
or than that the there this to when which with requirement requirements system context value values dossier
record""".split())


def _words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", (text or "").lower())
    return {t for t in tokens if t not in _STOP and (len(t) > 2 or t[0].isdigit())}


def similarity(a: str, b: str) -> float:
    """How much two statements say the same, 0 to 1: the share of the significant words and numbers of the
    shorter statement that the other statement also has. Two findings on the same subject with a
    similarity of SIMILAR (0.6) or more are the same finding."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# =========================================================================== the STE check (another engine)


def ste_issues(text: str, kind: str = "descriptive", terms: Any = None) -> list[str] | None:
    """The issues that dossier.ste.check reports for a text, as short texts. None: the STE checker is not
    installed in this version (or it failed), so the text was not checked."""
    try:
        from dossier import ste
        check_fn = ste.check
    except Exception:  # noqa: BLE001  the checker belongs to another engine; it can be missing
        return None
    try:
        issues = check_fn(text, kind=kind, terms=terms)
    except Exception:  # noqa: BLE001
        return None
    out = []
    for i in issues or []:
        if isinstance(i, str):
            out.append(i)
            continue
        msg = next((str(getattr(i, a)) for a in ("message", "msg", "text", "description") if getattr(i, a, None)), "")
        rule = next((str(getattr(i, a)) for a in ("rule", "code", "id") if getattr(i, a, None)), "")
        out.append(f"{rule}: {msg}" if rule and msg else (msg or rule or str(i)))
    return out


# =========================================================================== the result


@dataclass
class Result:
    """What `check` and `submit` return. `ok` is false when the submission is rejected."""

    role: str
    by: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    confidence: dict[str, str] = field(default_factory=dict)             # record id -> confidence
    record_warnings: dict[str, list[str]] = field(default_factory=dict)  # record id -> its warnings
    summary: str = ""
    path: Path | None = None            # the stage file, when `submit` wrote it

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def records(self) -> list:
        return [*self.findings, *self.decisions, *self.assumptions]

    def text(self) -> str:
        """The result for the shell, and for the model when its submission is rejected."""
        if not self.ok:
            lines = [f"REJECTED: the submission of {self.role} has {len(self.errors)} error(s). Nothing was written."]
            lines += [f"  {i}. {e}" for i, e in enumerate(self.errors, 1)]
            return "\n".join(lines)
        n = (len(self.findings), len(self.decisions), len(self.assumptions))
        head = "WRITTEN" if self.path else "ACCEPTED (check only, nothing written)"
        lines = [f"{head}: {n[0]} finding(s), {n[1]} decision(s), {n[2]} assumption(s) of {self.role}"
                 + (f" to {self.path}" if self.path else "")]
        lines += [f"  {r.id}  [{self.confidence.get(r.id, '')}]  {r.title}" for r in self.records]
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s) (the records hold them for the reviewer):")
            lines += [f"  - {w}" for w in self.warnings]
        lines += [f"note: {x}" for x in self.notes]
        return "\n".join(lines)


# =========================================================================== the check

_LITERAL = {"severity": roles_mod.SEVERITIES, "category": roles_mod.CATEGORIES, "confidence": roles_mod.CONFIDENCES}


def _loc(loc: tuple) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "(the document)"


def _structure_errors(data: Any) -> list[str]:
    """Missing and unknown fields, wrong types and words outside the vocabulary, each with its place."""
    try:
        Submission.model_validate(data)
        return []
    except pydantic.ValidationError as e:
        out = []
        for err in e.errors():
            loc, kind, name = _loc(err["loc"]), err["type"], (err["loc"][-1] if err["loc"] else "")
            if kind == "literal_error" or (kind.startswith("literal") and name in _LITERAL):
                vocab = " | ".join(_LITERAL.get(name, ())) or err.get("ctx", {}).get("expected", "")
                out.append(f"{loc}: {err.get('input')!r} is not in the vocabulary ({vocab})")
            elif kind == "missing":
                out.append(f"{loc}: the field is missing"
                           + (f" ({' | '.join(_LITERAL[name])})" if name in _LITERAL else ""))
            elif kind == "extra_forbidden":
                out.append(f"{loc}: unknown field")
            else:
                out.append(f"{loc}: {err['msg']}")
        return out


def _near(x: str, known: set[str]) -> str:
    near = difflib.get_close_matches(x, known, n=3, cutoff=0.6)
    return f" (nearest ids: {', '.join(near)})" if near else ""


def _id_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![\w-])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+(?![\w-])", text or "")


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _tail(kind: str, confidence: str, numbers: list[str], ids: list[str], ste_count: int, no_alternative: bool) -> str:
    """The sentences that the tool adds to a rationale: the confidence of the agent and the warnings."""
    out = [f"The agent gives this {kind} a {confidence} confidence."]
    if numbers:
        out.append(f"The submission check does not find {_join(numbers)} in the dossier.")
    if ids:
        out.append(f"The submission check does not find the id {_join(ids)} in the dossier.")
    if ste_count:
        out.append(f"The STE check reports {ste_count} issue{'s' if ste_count != 1 else ''} in the text of the agent.")
    if no_alternative:
        out.append("The agent gives no alternative.")
    return " ".join(out)


def check(project: Project, role: "str | roles_mod.Role", data: Any, by: str = "") -> Result:
    """Check a submission against the project. Returns the records that it gives (nothing is written)."""
    r = roles_mod.get(role)
    by = " ".join((by or "").replace("(", " ").replace(")", " ").split())[:80]
    res = Result(role=r.name, by=by)
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    if not isinstance(data, dict):
        res.errors.append("the submission must be one JSON object with the lists `findings`, `decisions` and "
                          "`assumptions`")
        return res
    res.errors += _structure_errors(data)
    res.summary = data.get("summary") if isinstance(data.get("summary"), str) else ""

    ctx = context_mod.load(project)
    others = {i: rec for i, rec in project.records().items() if roles_mod.role_of(rec) != r.name}
    known = set(project.inputs()) | set(ctx.params) | set(others)
    existing = [f for f in others.values() if isinstance(f, Finding)]
    prefixes = {i.split("-", 1)[0] for i in known if "-" in i} | {"FND", "DEC", "ASM", "LC", "AN", "REQ", "PH", "EF"}

    # ---- pass 1: the ids of the new records, so that a record can name another record of the submission
    new_ids: dict[tuple[str, int], str] = {}
    seen: dict[str, str] = {}
    for kind, (_, prefix, _) in KINDS.items():
        items = data.get(kind)
        for i, item in enumerate(items if isinstance(items, list) else []):
            if not isinstance(item, dict):
                continue
            where, subject, key = f"{kind}[{i}]", item.get("subject"), item.get("key")
            if isinstance(key, str) and key.strip() and (not KEY_RE.match(key.strip()) or len(key.strip()) > KEY_MAX):
                res.errors.append(f"{where}.key: {key!r} is not a short slug (lower-case letters, digits and hyphens, "
                                  f"at most {KEY_MAX} characters)")
                continue
            if not (isinstance(subject, str) and subject.strip() and isinstance(key, str) and key.strip()):
                continue
            rid = make_id(prefix, "A", subject.strip(), key.strip())
            if rid in seen:
                res.errors.append(f"{where}: the subject {subject!r} and the key {key!r} give the id {rid}, which "
                                  f"{seen[rid]} of this submission also has; use another key")
            elif rid in others:
                res.errors.append(f"{where}: the id {rid} is the id of a record that {others[rid].origin.by} made; "
                                  f"use another key")
            else:
                seen[rid] = where
                new_ids[(kind, i)] = rid
    known_all = known | set(new_ids.values())

    # ---- pass 2: texts, ids, duplicates
    statements: list[tuple[str, str, str]] = []          # (where, subject, statement) of the new findings
    for kind in KINDS:
        items = data.get(kind)
        for i, item in enumerate(items if isinstance(items, list) else []):
            if not isinstance(item, dict):
                continue
            where = f"{kind}[{i}]"
            for name in REQUIRED_TEXT[kind]:
                if isinstance(item.get(name), str) and not item[name].strip():
                    res.errors.append(f"{where}.{name}: the text is empty")
            for j, alt in enumerate(item.get("alternatives") or [] if kind == "decisions" else []):
                for name in ("option", "why_not"):
                    if isinstance(alt, dict) and isinstance(alt.get(name), str) and not alt[name].strip():
                        res.errors.append(f"{where}.alternatives[{j}].{name}: the text is empty")
            subject = item.get("subject")
            if isinstance(subject, str) and subject.strip():
                subject = subject.strip()
                if subject not in known_all:
                    res.errors.append(f"{where}.subject: {subject!r} is not a record or an input of the project"
                                      f"{_near(subject, known_all)}")
                elif subject == new_ids.get((kind, i)):
                    res.errors.append(f"{where}.subject: a record cannot be about itself")
            for j, e in enumerate(item.get("evidence") if isinstance(item.get("evidence"), list) else []):
                if isinstance(e, str) and e.strip() not in known_all:
                    res.errors.append(f"{where}.evidence[{j}]: {e!r} is not a record or an input of the project"
                                      f"{_near(e.strip(), known_all)}")
            if kind == "findings" and isinstance(subject, str) and isinstance(item.get("statement"), str) \
                    and item["statement"].strip():
                st = item["statement"]
                for old in existing:
                    s = similarity(st, old.statement)
                    if old.subject == subject and s >= SIMILAR:
                        res.errors.append(
                            f"{where}: the finding duplicates the existing finding {old.id} ({old.origin.by}): the same "
                            f"subject {subject} and a similar statement ({s:.0%} of the significant words of the "
                            f"shorter statement are in the other). Do not repeat an existing finding")
                for w2, s2, st2 in statements:
                    s = similarity(st, st2)
                    if s2 == subject and s >= SIMILAR:
                        res.errors.append(f"{where}: the finding duplicates {w2} of this submission: the same subject "
                                          f"{subject} and a similar statement ({s:.0%} of the significant words)")
                statements.append((where, subject, st))
    if res.errors:
        return res

    # ---- accepted: warnings and records
    sub = Submission.model_validate(data)
    pool = known_numbers(project, ctx, skip_role=r.name)
    origin = Origin(by=roles_mod.origin_by(r, by), stage=r.stage)
    terms = getattr(project, "terms", None) or None
    ste_ran = True
    for kind, (_, _, word) in KINDS.items():
        for i, item in enumerate(getattr(sub, kind)):
            rid = new_ids[(kind, i)]
            warns: list[str] = []
            values: dict[str, float] = {}
            missing_numbers: list[str] = []
            for name in NUMBER_TEXT[kind]:
                for q in quantities(getattr(item, name)):
                    hit = pool.match(q.si, q.unit) or (pool.match(*q.alt) if q.alt else None)
                    if hit is not None:
                        values[q.text] = hit[0]
                    elif q.text not in missing_numbers:
                        missing_numbers.append(q.text)
                        near = pool.nearest(q.si, q.unit)
                        warns.append(f"{rid}.{name}: the text gives {q.text}; no number of the context or the records "
                                     f"is within 1 % of it"
                                     + (f" (the nearest is {_show(near[0], near[1])}, {near[2]})" if near else "")
                                     + ": a number that the role calculated or invented")
            missing_ids: list[str] = []
            for name in STE_TEXT[kind]:
                for tok in _id_tokens(getattr(item, name)):
                    if tok.split("-", 1)[0] in prefixes and tok not in known_all and tok not in missing_ids \
                            and not any(k.startswith(tok + "-") for k in known_all):
                        missing_ids.append(tok)
                        warns.append(f"{rid}.{name}: the text names {tok}, which is not a record or an input of the "
                                     f"project{_near(tok, known_all)}")
            ste_count = 0
            for name in STE_TEXT[kind]:
                text = getattr(item, name)
                if not text.strip():
                    continue
                issues = ste_issues(text, "descriptive", terms)
                if issues is None:
                    ste_ran = False
                    break
                if issues:
                    ste_count += len(issues)
                    shown = "; ".join(issues[:3]) + (f"; and {len(issues) - 3} more" if len(issues) > 3 else "")
                    warns.append(f"{rid}.{name}: the STE check reports {len(issues)} issue(s): {shown}")
            no_alt = kind == "decisions" and not item.alternatives
            if no_alt:
                warns.append(f"{rid}: the decision gives no alternative")
            rationale = item.rationale.strip() + " " + _tail(word, item.confidence, missing_numbers, missing_ids,
                                                              ste_count, no_alt)
            evidence = list(dict.fromkeys(e.strip() for e in item.evidence))
            links = [Link(rel="evidence", to=e) for e in evidence if e != item.subject.strip()]
            common = dict(id=rid, title=item.title.strip(), subject=item.subject.strip(), rationale=rationale,
                          links=links, origin=origin)
            if kind == "findings":
                res.findings.append(Finding(rule=f"agent.{r.name}", severity=item.severity, category=item.category,
                                            statement=item.statement.strip(), suggestion=item.suggestion.strip(),
                                            values=values, **common))
            elif kind == "decisions":
                res.decisions.append(Decision(stage=r.stage, choice=item.choice.strip(), basis="engineering_judgement",
                                              needs_review=True,
                                              alternatives=[Alternative(option=a.option.strip(), why_not=a.why_not.strip())
                                                            for a in item.alternatives], **common))
            else:
                res.assumptions.append(Assumption(stage=r.stage, statement=item.statement.strip(),
                                                  impact=item.impact.strip(), verify_by=item.verify_by.strip(),
                                                  conservative=item.conservative, **common))
            res.confidence[rid] = item.confidence
            if warns:
                res.record_warnings[rid] = warns
                res.warnings += warns
    if not ste_ran:
        res.notes.append("the STE checker (dossier.ste) is not installed in this version: the texts were not checked "
                         "against the STE rules")
    return res


def agent_path(project: Project, role: "str | roles_mod.Role") -> Path:
    """The stage file of a role: <project>/agent/<role>.json (Project.stage_files reads it)."""
    return project.root / "agent" / f"{roles_mod.get(role).name}.json"


def submit(project: Project, role: "str | roles_mod.Role", data: Any, by: str = "") -> Result:
    """Check a submission and, when it is accepted, write its records to <project>/agent/<role>.json.
    A second submission of the same role replaces the file. A rejected submission writes nothing."""
    res = check(project, role, data, by)
    if not res.ok:
        return res
    raw = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    info = {"role": res.role, "by": res.by, "origin": roles_mod.origin_by(res.role, res.by), "summary": res.summary,
            "records": {x.id: {"confidence": res.confidence.get(x.id, ""), "warnings": res.record_warnings.get(x.id, [])}
                        for x in res.records},
            "notes": res.notes}
    res.path = project.write_stage(res.role, agent_path(project, res.role), findings=res.findings,
                                   decisions=res.decisions, assumptions=res.assumptions, agent=info, submission=raw)
    return res
