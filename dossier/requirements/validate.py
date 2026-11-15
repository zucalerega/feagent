"""Requirement validation: are the requirements right, given the system context?

    validate(project, ctx, geo) -> Journal      findings, and assumptions where a rule takes something as true

Rules (id, default severity, category, what the rule makes sure of). Rule req.material_properties_temperature
makes an assumption, not a finding. Severity may differ for one finding: the text of a finding says why.

    req.not_verifiable                major  quality     A requirement that analysis verifies has a criterion.
    req.no_shall                      minor  quality     The text of a requirement that is an obligation uses "shall".
    req.not_singular                  minor  quality     The text of a requirement has one "shall" only.
    req.vague_word                    minor  quality     The text of a requirement has no word without a measure ("sufficiently", "adequate", "as required", "TBD").
    req.criterion_vocabulary          error  quality     Each criterion names a result quantity and a measure that the analysis gives.
    req.criterion_location            error  quality     Each criterion names a location that the component has and that can give the result.
    req.criterion_allowable           error  quality     Each allowable is a property of the material of the part, with the unit of the result quantity.
    req.criterion_limit               error  quality     Each limit gives one number with the unit of the result quantity.
    req.criterion_factor              error  quality     Each factor of a criterion is a number, a requirement parameter or a factor of the policy.
    req.number_in_text                minor  quality     The text of a requirement gives the value of each of its parameters.
    req.parameter_role                error  quality     Each parameter role is a role of the validation, and the unit of the parameter agrees with the role.
    req.meop_envelope                 major  consistency The MEOP of a requirement is not less than the largest pressure on the part in the operation phases.
    req.temperature_range             major  consistency The fluid temperature range of a requirement includes the fluid temperatures of the operation phases.
    req.rated_flow                    major  consistency The rated mass flow of a requirement agrees with the mass flow of the system context within 1 %.
    req.design_cycles                 major  consistency The number of design cycles is not less than the count of the duty cycles of its requirement (a parameter without a check).
    req.param_check                   major  consistency Each parameter obeys its explicit check against the system context.
    req.policy_factor                 major  policy      Each factor of a requirement is not less than the factor of the policy.
    req.phase_uncovered               major  coverage    A requirement of kind strength covers each phase in which the part has loads.
    req.duty_cycle_uncovered          minor  coverage    A requirement of kind life names each duty cycle that loads the part.
    req.phase_without_effect          minor  coverage    The part has the effects that a requirement needs in each phase that the requirement applies to.
    req.no_phase                      major  coverage    A requirement that has criteria applies to a phase or to a duty cycle.
    req.role_conflict                 major  conflict    Parameters with the same role have the same value.
    req.criteria_conflict             minor  conflict    Two criteria for the same result in the same phase have the same limit.
    req.material_temperature          major  material    The temperatures of the system context are in the service temperature range of the material.
    req.material_compatibility        major  material    The material is compatible with each medium of the fluid connections of the part.
    req.material_properties_temperatureinfo   material    The material properties apply at the temperatures of the system context (if not, an assumption says so).
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
