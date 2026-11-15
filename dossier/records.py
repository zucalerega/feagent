"""Helpers every stage uses to write its decisions, assumptions and findings the same way.

    j = Journal(stage="plan", by="rule:loadcases.proportional")
    j.decide(subject="LC-PH-BURST", key="scaled", title="...", choice="...", rationale="...",
             alternatives=[("Solve the burst case on its own mesh study.", "The result is the same ...")])

Ids are stable (make_id from the subject and a key), so a person's review of a record survives a re-run.
Text is ASD-STE100 Simplified Technical English: short sentences, one topic each, active voice, simple
tenses, no "should / may / could", numbers with units.
"""

from __future__ import annotations

from typing import Any, Iterable

from dossier.model import Alternative, Assumption, Decision, Finding, Link, Origin
from dossier.project import make_id


def link(rel: str, to: str, note: str = "") -> Link:
    return Link(rel=rel, to=to, note=note)


def _links(links: Iterable[Any]) -> list[Link]:
    out = []
    for x in links or ():
        if isinstance(x, Link):
            out.append(x)
        elif isinstance(x, str):
            out.append(Link(rel="uses", to=x))
        else:
            out.append(Link(rel=x[0], to=x[1], note=x[2] if len(x) > 2 else ""))
    return out


class Journal:
    """Collects the records a stage (or one engine of it) makes. `by` names the rule, engine, agent or
    person; each call may override it."""

    def __init__(self, stage: str, by: str):
        self.stage = stage
        self.by = by
        self.decisions: list[Decision] = []
        self.assumptions: list[Assumption] = []
        self.findings: list[Finding] = []

    def _origin(self, by: str | None) -> Origin:
        return Origin(by=by or self.by, stage=self.stage)

    def decide(self, subject: str, key: str, title: str, choice: str, rationale: str,
               alternatives: Iterable[tuple[str, str]] = (), basis: str = "rule", needs_review: bool = False,
               links: Iterable[Any] = (), by: str | None = None, stage: str | None = None) -> Decision:
        d = Decision(id=make_id("DEC", subject, key), title=title, stage=stage or self.stage, subject=subject,
                     choice=choice, rationale=rationale,
                     alternatives=[Alternative(option=o, why_not=w) for o, w in alternatives],
                     basis=basis, needs_review=needs_review, links=_links(links), origin=self._origin(by))
        self.decisions.append(d)
        return d

    def assume(self, subject: str, key: str, title: str, statement: str, rationale: str, impact: str,
               verify_by: str = "", conservative: bool | None = None, links: Iterable[Any] = (),
               by: str | None = None, stage: str | None = None) -> Assumption:
        a = Assumption(id=make_id("ASM", subject, key), title=title, stage=stage or self.stage, subject=subject,
                       statement=statement, rationale=rationale, impact=impact, verify_by=verify_by,
                       conservative=conservative, links=_links(links), origin=self._origin(by))
        self.assumptions.append(a)
        return a

    def find(self, rule: str, severity: str, category: str, subject: str, key: str, title: str, statement: str,
             rationale: str, suggestion: str = "", values: dict | None = None, links: Iterable[Any] = (),
             by: str | None = None) -> Finding:
        f = Finding(id=make_id("FND", subject, key), title=title, rule=rule, severity=severity, category=category,
                    subject=subject, statement=statement, rationale=rationale, suggestion=suggestion,
                    values=values or {}, links=_links(links), origin=self._origin(by or f"rule:{rule}"))
        self.findings.append(f)
        return f

    def extend(self, other: "Journal") -> "Journal":
        self.decisions += other.decisions
        self.assumptions += other.assumptions
        self.findings += other.findings
        return self
