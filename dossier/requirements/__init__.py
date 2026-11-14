"""The requirements of the component under verification: are they right, given the system context?

    dossier.requirements.validate.validate(project, ctx, geo) -> Journal     the engine of the validate stage
    dossier.requirements.validate.RULES                                      the rules, in the order they run
    dossier.requirements.validate.summary(findings, assumptions)             one row for each rule: what it made

    validate.py      the entry point, the table of rules (its module text)
    base.py          the registry, the lookups every rule uses, the helpers for the text of a record
    wording.py       what the text of a requirement says: "shall", words without a measure, numbers
    quality.py       rules: can analysis verify each requirement?
    consistency.py   rules: do the numbers of the requirements agree with the system context and the policy?
    coverage.py      rules: do the requirements cover the phases and duty cycles? do two of them conflict?
    material.py      rules: is the material right for the temperatures and the media of the system context?

This package does not import `validate` by name: `dossier.requirements.validate` is always the module.
"""
