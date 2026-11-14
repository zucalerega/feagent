"""What the rules of the requirement validation share.

    Rule, rule(...)    the registry: an id `req.<name>`, one line that says what the rule makes sure of, the
                       default severity and the category of its findings, and the function
    Env                the project, the context and the lookups a rule needs. `Env.find` and `Env.assume`
                       write a record with its id, its origin (`rule:<id>`) and its links in one way.
    vocabulary         the result keys of the analysis (CONTRACTS.md) and the parameter roles
    text helpers       quote, join, count, shown, bound, percent, fit, derivation

Text of the records: ASD-STE100 (CONTRACTS.md, "Text"). A sentence has 25 words or fewer (an instruction 20),
a text field has six sentences or fewer, the voice is active, and every number has its unit and its source.
Words an engineer wrote (a requirement text, a description, a source, a note) are in double quotes, as they
are. The part under verification is always "the part".

Links of a finding (compliance reads them): the subject is the record the finding questions. A link `about`
names another record that the finding questions (the parameter, the second requirement of a conflict), and
the compliance stage shows the finding as a caveat of every result that rests on that record. A link
`evidence` names a record that gives a number of the rationale (an effect, a phase, a context parameter);
the finding does not question it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from dossier import units
from dossier.context import MECHANICAL, THERMAL, Context, ResolvedEffect, ResolvedParam
from dossier.model import (DEFAULT_ROLE, PARAM_ROLES, Assumption, Finding, LoadItem, ReqParameter, Requirement)
from dossier.project import Project, make_id
from dossier.records import Journal

# --------------------------------------------------------------------------- the registry


@dataclass(frozen=True)
class Rule:
    id: str                              # req.<name>
    description: str                     # one line: what the rule makes sure of (the report prints it)
    severity: str                        # the default severity of its findings
    category: str                        # the category of its findings
    fn: Callable[["Env"], None]
    makes: str = "finding"               # "assumption" for a rule that records an assumption, not a finding

    @property
    def name(self) -> str:
        return self.id.split(".", 1)[1]


REGISTRY: list[Rule] = []


def rule(id: str, severity: str, category: str, description: str, makes: str = "finding"):
    """Register a rule. The modules of the package register their rules in the order they are imported."""
    def register(fn: Callable[["Env"], None]) -> Callable[["Env"], None]:
        if any(r.id == id for r in REGISTRY):
            raise ValueError(f"rule {id} is registered twice")
        REGISTRY.append(Rule(id=id, description=description, severity=severity, category=category, fn=fn, makes=makes))
        return fn
    return register


# --------------------------------------------------------------------------- vocabulary

# Result keys of the analysis, CONTRACTS.md "Result keys": quantity -> the SI unit of its values
QUANTITY_UNIT = {"von_mises": "Pa", "max_principal": "Pa", "sxx": "Pa", "syy": "Pa", "szz": "Pa",
                 "displacement": "m", "radial_displacement": "m", "temperature": "K", "heat_flow": "W",
                 "pressure_drop": "Pa", "damage": ""}
MEASURES = ("peak", "max", "min", "mean", "range", "membrane", "membrane_bending", "total")
SECTION_MEASURES = ("membrane", "membrane_bending")          # section stresses: on a stress classification line
FIXED_LOCATIONS = ("part", "scls")
# the effect kinds a result quantity needs in its phase (a quantity that is not here needs no phase effect)
QUANTITY_NEEDS = {"temperature": THERMAL, "heat_flow": THERMAL, "pressure_drop": ("mass_flow",),
                  **{q: (*MECHANICAL, *THERMAL) for q in ("von_mises", "max_principal", "sxx", "syy", "szz",
                                                          "displacement", "radial_displacement")}}

# Parameter roles (dossier.model.PARAM_ROLES): the SI unit a parameter with the role has, and its name in a sentence
ROLE_UNIT = {"meop": "Pa", "min_temperature": "K", "max_temperature": "K", "rated_flow": "kg/s", "design_cycles": "",
             "proof_factor": "", "burst_factor": "", "yield_factor": "", "ultimate_factor": "", "life_factor": ""}
ROLE_WORDS = {"meop": "MEOP", "min_temperature": "lowest fluid temperature", "max_temperature": "highest fluid temperature",
              "rated_flow": "rated mass flow", "design_cycles": "number of design cycles", "proof_factor": "proof factor",
              "burst_factor": "burst factor", "yield_factor": "yield factor", "ultimate_factor": "ultimate factor",
              "life_factor": "life factor"}
UNIT_WORDS = {"Pa": "a pressure", "K": "a temperature", "kg/s": "a mass flow", "": "a number without a unit",
              "m": "a length", "W": "a power"}
# factor roles -> the factor of the policy they must not be lower than
POLICY_FACTOR = {"proof_factor": "proof", "burst_factor": "burst", "yield_factor": "yield", "ultimate_factor": "ultimate",
                 "life_factor": "fatigue_life"}
assert set(ROLE_UNIT) == set(PARAM_ROLES) == set(ROLE_WORDS), "the roles of dossier.model changed"

TEMPERATURE_KINDS = ("fluid_temperature", "temperature")

# --------------------------------------------------------------------------- text helpers

DESCRIPTIVE = 25      # words in a descriptive sentence
INSTRUCTION = 20      # words in an instruction
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_NAME = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


def quote(text: Any, keep_dot: bool = False) -> str:
    """Words an engineer wrote, in double quotes, as they are (on one line; the last full stop goes after
    the closing quote, where the sentence ends)."""
    s = " ".join(str(text).split()).replace('"', "'")
    if not keep_dot:
        s = s.rstrip(".").rstrip()
    return f'"{s}"'


def join(items: Iterable[Any], word: str = "and") -> str:
    """A, B and C."""
    items = [str(x) for x in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def some(items: Iterable[Any], most: int = 6, noun: str = "more") -> str:
    """join() of a list that can be long: the first `most` items, then "and 3 more"."""
    items = [str(x) for x in items]
    if len(items) <= most:
        return join(items)
    return ", ".join(items[:most]) + f" and {len(items) - most} {noun}"


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def number(x: float, sig: int = 4) -> str:
    """A pure number for people: 1.667, 65, 0.6667."""
    return units.fmt(units.Q(float(x)), None, sig)


def count(n: float, one: str, many: str | None = None) -> str:
    """1 effect, 3 effects."""
    return f"{number(n)} {one if float(n) == 1.0 else (many or one + 's')}"


def shown(si: float, unit: str, sig: int = 4) -> str:
    """An SI value in the unit people read (dossier.units.display)."""
    return units.display(units.from_si(si, unit), sig)


def bound(si: float, unit: str, up: bool, sig: int = 4) -> str:
    """shown(), rounded to the safe side: up for a minimum ("increase to 10.73 MPa or more"), down for a
    maximum. The plain display rounds 10.7239 MPa to 10.72 MPa, which is still too low."""
    text = shown(si, unit, sig)
    try:
        q = units.from_si(si, unit)
        back = units.parse(text)                                  # the display, read back: its unit, its rounding
        exact = float(q.to(back.units).magnitude)
        rounded = float(back.magnitude)
        slack = 1e-12 * max(abs(exact), 1e-300)
        if (up and rounded >= exact - slack) or (not up and rounded <= exact + slack) or exact == 0.0:
            return text
        e = math.floor(math.log10(abs(exact)))
        step = 10.0 ** (e - (sig - 1))
        safe = (math.ceil(exact / step) if up else math.floor(exact / step)) * step
        body = f"{safe:.{max(sig - 1 - e, 0)}f}"
        body = body.rstrip("0").rstrip(".") if "." in body else body
        label = text.split(" ", 1)[1] if " " in text else ""
        return f"{body} {label}".strip()
    except Exception:  # noqa: BLE001  a unit the display writes in a way that cannot be read back
        return text


def percent(fraction: float) -> str:
    """0.0213 -> "2.1 %"."""
    return f"{number(100.0 * abs(fraction), 2)} %"


def words(text: str) -> int:
    return len(text.split())


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def fit(*candidates: str, limit: int = DESCRIPTIVE) -> str:
    """The first wording whose sentences all have `limit` words or fewer; the last one when none has."""
    for c in candidates:
        if all(words(s) <= limit for s in sentences(c)):
            return c
    return candidates[-1]


def text(parts: str | Iterable[str]) -> str:
    """Sentences -> one text field (one paragraph, no line breaks)."""
    if isinstance(parts, str):
        parts = [parts]
    return " ".join(" ".join(str(p).split()) for p in parts if p and str(p).strip())


def criterion_name(req: Requirement, index: int) -> str:
    """How a sentence names a criterion: the criterion "yield" | criterion 2 (people count from 1)."""
    label = req.criteria[index].id
    same = [c.id for c in req.criteria].count(label)
    return f"the criterion {quote(label)}" if label and same == 1 else f"criterion {index + 1}"


def criterion_key(req: Requirement, index: int) -> str:
    """The part of a finding id that names a criterion: its label when it has one of its own, else c<number>."""
    label = req.criteria[index].id
    return label if label and [c.id for c in req.criteria].count(label) == 1 else f"c{index + 1}"


def nearest(word: str, known: Iterable[str]) -> str | None:
    import difflib
    known = list(known)
    by_lower = {k.lower(): k for k in known}
    hit = difflib.get_close_matches(str(word).lower(), list(by_lower), n=1, cutoff=0.6)
    return by_lower[hit[0]] if hit else None


def names_in(expr: str | None, ctx: Context) -> list[str]:
    """The parameters an expression reads, in the order of the text, each once."""
    out: list[str] = []
    for m in _NAME.finditer(expr or ""):
        if m.group(0) in ctx.params and m.group(0) not in out:
            out.append(m.group(0))
    return out


def substitute(expr: str, ctx: Context) -> str:
    """The expression with each parameter replaced by its value: "p_pump_max + dp_surge" -> "10.4 MPa + 0.3239 MPa"."""
    def value(m: re.Match) -> str:
        p = ctx.params.get(m.group(0))
        if p is None:
            return m.group(0)
        power = expr[m.end():].lstrip().startswith("**")
        return f"({p.display})" if power or p.display.startswith("-") else p.display
    return _NAME.sub(value, expr)


def sources(ctx: Context, names: Iterable[str]) -> str:
    """One sentence that says where the inputs of a derivation come from ("" when there is no input)."""
    by_source: dict[str, list[str]] = {}
    without: list[str] = []
    for n in names:
        p = ctx.params[n]
        if p.source:
            by_source.setdefault(p.source, []).append(n)
        else:
            without.append(n)
    if by_source:
        items = list(by_source.items())
        if len(items) == 1 and len(items[0][1]) == 1:
            return f"The source of {items[0][1][0]} is {quote(items[0][0])}."
        return "The sources of the inputs are " + join(f"{quote(s)} ({', '.join(ns)})" for s, ns in items[:3]) + "."
    if without:
        return f"The system context gives no source for {some(without, 4)}."
    return ""


def derivation(ctx: Context, expr: str | None, total: str, what: str = "The effect", most: int = 3) -> tuple[list[str], list[str]]:
    """How a context value comes from its parameters: (sentences, the parameters they show).

    `expr` is the expression of the value (of an effect, of a check), `total` its result for people, `what`
    the subject of the first sentence. The first sentence shows the expression with its numbers
    ("The effect reads p_max_inlet = p_pump_max + dp_surge = 10.4 MPa + 0.3239 MPa = 10.72 MPa."), the next
    one the sources of its inputs, then one sentence for each input that is derived, up to `most` sentences.
    The trace graph has the full tree (a parameter links to the parameters it reads)."""
    expr = (expr or "").strip()
    names = names_in(expr, ctx)
    if not names:
        return [], []
    lines: list[str] = []
    shown_names: list[str] = []

    def formula(head: str, e: str, result: str) -> str:
        filled = substitute(e, ctx)
        long = f"{head} {e} = {filled} = {result}." if filled != e else f"{head} {e} = {result}."
        return fit(long, f"{head} {e} = {result}.")

    if expr in ctx.params:                                        # the value is one parameter
        p = ctx.params[expr]
        shown_names.append(expr)
        if p.expr:
            lines.append(formula(f"{what} reads {expr} =", p.expr, p.display))
            level = names_in(p.expr, ctx)
        else:
            lines.append(f"{what} reads the parameter {expr} = {p.display}.")
            level = []
        leaves = [n for n in (level or [expr]) if not ctx.params[n].expr]
    else:
        lines.append(formula(f"{what} is", expr, total))
        level = names
        leaves = [n for n in level if not ctx.params[n].expr]
    shown_names += [n for n in level if n not in shown_names]
    where = sources(ctx, leaves)
    if where and len(lines) < most:
        lines.append(where)
    for n in level:
        p = ctx.params[n]
        if p.expr and len(lines) < most:
            lines.append(formula(f"The parameter {n} is", p.expr, p.display))
    return lines, shown_names


# --------------------------------------------------------------------------- the environment of a rule


class Env:
    """The project and its context, with the lookups the rules share. `rule` is the rule that runs."""

    def __init__(self, project: Project, ctx: Context, geo: dict | None, journal: Journal):
        self.project = project
        self.ctx = ctx
        self.geo: dict = geo if isinstance(geo, dict) else {}
        self.journal = journal
        self.rule: Rule | None = None
        self.comp = project.subject
        self.element = self.comp.element
        self.material = project.material_of()
        self.policy = project.policy
        self.reqs: list[Requirement] = list(project.reqs())
        self.phases = list(project.system.phases)
        self.operation = [p.id for p in self.phases if p.category == "operation"]
        self.names = ctx.names()
        # every requirement parameter of the project: name -> the requirement that states it
        self.req_params = {name: r.id for rs in project.requirements.values() for r in rs for name in r.parameters}
        self._loads: dict[str, list[LoadItem]] = {}
        self._ids: set[str] = set()

    # ---- the system context, seen from the part
    def effects(self, phase: str | None = None, kinds: Iterable[str] | None = None) -> list[ResolvedEffect]:
        """The effects on the part, in one phase or in all, of some kinds or of all."""
        kinds = None if kinds is None else tuple(kinds)
        return [e for e in self.ctx.effects_on(self.element, phase) if kinds is None or e.kind in kinds]

    @staticmethod
    def magnitude(e: ResolvedEffect) -> float:
        if e.si is None:
            return 0.0
        return float(sum(c * c for c in e.si) ** 0.5) if isinstance(e.si, list) else abs(float(e.si))

    def size(self, e: ResolvedEffect) -> str:
        """The magnitude of an effect for people: 411.9 N for the vector [0, 411.9, 0] N."""
        return shown(self.magnitude(e), e.unit) if isinstance(e.si, list) else e.display

    def loads(self, phase: str) -> list[LoadItem]:
        if phase not in self._loads:
            self._loads[phase] = self.ctx.loads(self.project, None, phase)
        return self._loads[phase]

    def wetted(self, e: ResolvedEffect) -> bool:
        return (e.role or DEFAULT_ROLE.get(e.kind, "")) == "wetted"

    def context_lines(self, e: ResolvedEffect, most: int = 5) -> tuple[list[str], list[str]]:
        """Where a scalar context value is and how it comes from its parameters: (sentences, the parameters
        they show). The sentences are the middle of a rationale: the value with its phase, effect and
        connection; what the engineer wrote about the effect; the derivation."""
        lines = [f"The system context gives {e.display} in the phase {e.phase} (effect {e.id}, connection {e.connection})."]
        if e.description:
            lines.append(f"The description of the effect is {quote(e.description)}.")
        chain, names = derivation(self.ctx, e.expr, e.display, "The effect", most=max(most - len(lines), 1))
        if not chain:
            lines.append(f"The effect gives this value as a number, with the source {quote(e.source)}." if e.source
                         else "The effect gives this value as a number, without a source.")
        return lines + chain, names

    def phases_of(self, req: Requirement) -> list[str]:
        return self.project.phases_of(req)

    # ---- requirement parameters
    def with_role(self, role: str) -> list[tuple[Requirement, str, ReqParameter, ResolvedParam]]:
        """The parameters of the requirements that have a role and the unit of the role (one number)."""
        out = []
        for r in self.reqs:
            for name, p in r.parameters.items():
                rp = self.ctx.params.get(name)
                if p.role == role and rp is not None and not isinstance(rp.si, list) and rp.unit == ROLE_UNIT.get(role):
                    out.append((r, name, p, rp))
        return out

    def role_key(self, req: Requirement, role: str, name: str, key: str) -> str:
        """The key of a finding about a role: the key alone, or with the parameter name when the requirement
        gives the role to two parameters."""
        twice = sum(1 for p in req.parameters.values() if p.role == role) > 1
        return f"{key}-{name}" if twice else key

    def dependents(self, name: str) -> set[str]:
        """The name and every parameter that reads it, directly or through other parameters."""
        out = {name}
        grew = True
        while grew:
            grew = False
            for n, p in self.ctx.params.items():
                if n not in out and out.intersection(p.depends_on):
                    out.add(n)
                    grew = True
        return out

    def readers(self, name: str) -> list[str]:
        """The phases in which a load on the part reads the parameter, in the order of the system."""
        names = self.dependents(name)
        hit = {e.phase for e in self.effects() if names.intersection(e.depends_on)}
        return [p.id for p in self.phases if p.id in hit]

    def criteria_using(self, name: str) -> list[tuple[Requirement, int]]:
        """The criteria whose factor or limit reads the parameter."""
        out = []
        for r in self.reqs:
            for i, c in enumerate(r.criteria):
                reads = c.limit.names() if c.limit is not None else set()
                if c.factor == name or name in reads:
                    out.append((r, i))
        return out

    # ---- records
    def _key(self, prefix: str, subject: str, key: str) -> str:
        key = re.sub(r"[^A-Za-z0-9]+", "-", str(key)).strip("-") or "x"
        base, n = key, 1
        while make_id(prefix, subject, key) in self._ids:
            n += 1
            key = f"{base}-{n}"
        self._ids.add(make_id(prefix, subject, key))
        return key

    @staticmethod
    def _links(links: Iterable[Any], subject: str) -> list[tuple]:
        out: list[tuple] = []
        for x in links or ():
            x = ("evidence", x) if isinstance(x, str) else tuple(x)
            if x[1] and x[1] != subject and x[:2] not in [y[:2] for y in out]:
                out.append(x)
        return out

    def find(self, subject: str, key: str, title: str, statement: str | Iterable[str], rationale: str | Iterable[str],
             suggestion: str | Iterable[str] = "", values: dict | None = None, links: Iterable[Any] = (),
             severity: str | None = None) -> Finding:
        """A finding of the rule that runs. `links`: (rel, id) pairs; a bare id is evidence. The subject is
        linked by its own field (Finding.refs), so it is not repeated in the links."""
        r = self.rule
        return self.journal.find(rule=r.id, severity=severity or r.severity, category=r.category, subject=subject,
                                 key=self._key("FND", subject, key), title=title, statement=text(statement),
                                 rationale=text(rationale), suggestion=text(suggestion), values=values or {},
                                 links=self._links(links, subject))

    def assume(self, subject: str, key: str, title: str, statement: str | Iterable[str], rationale: str | Iterable[str],
               impact: str | Iterable[str], verify_by: str | Iterable[str] = "", conservative: bool | None = None,
               links: Iterable[Any] = ()) -> Assumption:
        return self.journal.assume(subject=subject, key=self._key("ASM", subject, key), title=title,
                                   statement=text(statement), rationale=text(rationale), impact=text(impact),
                                   verify_by=text(verify_by), conservative=conservative,
                                   links=self._links(links, subject), by=f"rule:{self.rule.id}")
