"""Requirement validation: are the requirements right, given the system context?

    validate(project, ctx, geo) -> Journal      findings, and assumptions where a rule takes something as true

TABLE
"""

from __future__ import annotations

from typing import Iterable

from dossier.context import Context
from dossier.model import Assumption, Finding
from dossier.project import Project
from dossier.records import Journal
from dossier.requirements import base
from dossier.requirements import quality, consistency, coverage, material  # noqa: F401  they register the rules, in this order
from dossier.requirements.base import Env, Rule

RULES: tuple[Rule, ...] = tuple(base.REGISTRY)

# When True, a rule that raises stops the validation (tests). When False, the failure is an error finding of
# that rule and the other rules still run: one failed check must not hide the findings of the others.
STRICT = False


def validate(project: Project, ctx: Context, geo: dict | None = None) -> Journal:
    """Run every rule on the requirements of the component under verification."""
    journal = Journal("validate", "engine:requirements.validate")
    env = Env(project, ctx, geo, journal)
    env.strict = STRICT
    for r in RULES:
        env.rule = r
        try:
            r.fn(env)
        except Exception as e:  # noqa: BLE001
            if STRICT:
                raise
            env.find(subject=env.comp.id, key=f"{r.name}-failed", severity="error",
                     title=f"The rule {r.id} did not complete",
                     statement=f"The rule {r.id} stopped with an error, thus its check is not complete.",
                     rationale=[f"The rule makes sure of this: {base.quote(r.description)}.",
                                f"The error is {base.quote(f'{type(e).__name__}: {e}')}."],
                     suggestion="Report this error. Do the check of this rule by hand until the rule runs.",
                     values={"rule": r.id})
    return journal


def summary(findings: Iterable[Finding], assumptions: Iterable[Assumption] = ()) -> list[dict]:
    """One row for each rule, in the order they run: what the rule makes sure of and what it made.
    {"rule", "description", "severity", "category", "makes", "findings": [ids], "assumptions": [ids]}.
    A rule with no id in either list ran and found nothing."""
    findings, assumptions = list(findings), list(assumptions)
    return [{"rule": r.id, "description": r.description, "severity": r.severity, "category": r.category, "makes": r.makes,
             "findings": [f.id for f in findings if f.rule == r.id],
             "assumptions": [a.id for a in assumptions if a.origin.by == f"rule:{r.id}"]} for r in RULES]
