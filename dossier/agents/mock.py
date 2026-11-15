"""--mock: a canned submission in place of a language model.

For tests and for the dashboard without an API key: `python -m dossier.agents run PROJECT ROLE --mock` runs
the whole path (brief, submission, check, records) with no call to a model. The canned submission is made
from the project, so that every id and every number it names is real and the check accepts it; it is the
same on every run. Every text of it starts with MARK, so that nobody takes a canned record for a judgement:

    "Canned example of the mock mode. ..."

    canned(project, role)        the canned submission of a role, as a dict
    MockLLM(project, role)       stands in for ace.llm.LLM: structured() returns the canned submission
"""

from __future__ import annotations

from typing import Any

from dossier import context as context_mod
from dossier.agents import roles as roles_mod
from dossier.project import Project

MARK = "Canned example of the mock mode."
BAD_ID = "REQ-DOES-NOT-EXIST"       # what a faulty mock answer names (tests of the correction round)


def canned(project: Project, role: "str | roles_mod.Role") -> dict:
    """The canned submission of a role: one finding, one decision, one assumption about records that the
    project has. Deterministic. The texts say that a language model did not write them."""
    r = roles_mod.get(role)
    ctx = context_mod.build(project)
    comp, mat = project.subject, project.material_of()
    phase = project.system.phases[0].id if project.system.phases else comp.id
    load_cases = project.collection("load_cases")
    analyses = project.collection("analyses")

    # what the finding is about, and one real number to quote, by role
    req = next((q for q in project.reqs() if q.parameters), None) or next(iter(project.reqs()), None)
    if r.name == "plan-reviewer" and load_cases:
        lc = load_cases[0]
        subject, evidence = lc.id, [lc.id] + ([lc.phase] if lc.phase else [])
        fact = f"The load case {lc.id} has {len(lc.loads)} loads and the analysis type {lc.analysis_type}."
    elif r.name == "analysis-checker" and analyses:
        a = analyses[0]
        subject, evidence = a.id, [a.id, a.load_case]
        fact = f"The analysis {a.id} of the load case {a.load_case} has the status {a.status}."
    elif r.name == "requirements-reviewer" and req is not None and req.parameters:
        name = next(iter(req.parameters))
        subject, evidence = req.id, [req.id, name]
        fact = f"The requirement {req.id} gives {ctx.params[name].display} for the number {name}."
    elif r.name == "plan-reviewer":
        subject, evidence = phase, [phase]
        fact = f"The phase {phase} is the first phase of the system context."
    else:
        subject, evidence = comp.id, [comp.id, mat.id]
        fact = f"The component {comp.id} has the material {mat.id}."
    stage = {"plan-reviewer": "plan", "analysis-checker": "analyze"}.get(r.name)
    if stage == "plan" and not load_cases:
        fact += " The plan stage has not run."
    if stage == "analyze" and not analyses:
        fact += " No analysis has run."

    return {
        "summary": f"{MARK} The role {r.name} did not review the dossier. A language model did not write these records.",
        "findings": [{
            "subject": subject, "key": "canned-example",
            "title": "Canned example of a finding",
            "severity": "info", "category": "quality",
            "statement": f"{MARK} A language model did not write this finding of the role {r.name}.",
            "rationale": f"{MARK} {fact} The mock mode quotes this record to show a citation.",
            "suggestion": f"{MARK} Reject this record. Run the role with a language model for a real review.",
            "evidence": evidence, "confidence": "low",
        }],
        "decisions": [{
            "subject": phase, "key": "canned-example",
            "title": "Canned example of a decision",
            "choice": f"{MARK} Keep the phase {phase} in the verification.",
            "rationale": f"{MARK} The phase {phase} is the first phase of the system context. A language model did "
                         f"not make this decision.",
            "alternatives": [{"option": f"{MARK} Remove the phase {phase} from the verification.",
                              "why_not": f"{MARK} The mock mode does not judge the phases."}],
            "evidence": [phase], "confidence": "low",
        }],
        "assumptions": [{
            "subject": comp.id, "key": "canned-example",
            "title": "Canned example of an assumption",
            "statement": f"{MARK} The material data of the component {comp.id} are correct.",
            "rationale": f"{MARK} The material {mat.id} gives a yield strength of {ctx.params['Sy'].display}. The mock "
                         f"mode did not check this number.",
            "impact": f"{MARK} No result changes. This record is an example.",
            "verify_by": f"{MARK} Compare the material data with the material specification.",
            "conservative": None, "evidence": [mat.id], "confidence": "low",
        }],
    }


class MockLLM:
    """Stands in for ace.llm.LLM in a mock run. `faults`: that many first answers name a record that does
    not exist, so the check rejects them and the loop asks for a correction (for tests)."""

    model = "mock"

    def __init__(self, project: Project, role: "str | roles_mod.Role", faults: int = 0):
        self.project = project
        self.role = roles_mod.get(role)
        self.faults = faults
        self.calls: list[dict] = []           # what each call got: {"role", "system", "user"}

    def structured(self, role: str, system: str, user: Any, schema, records: list | None = None):
        """The signature of ace.llm.LLM.structured: (parsed submission, usage records)."""
        records = [] if records is None else records
        records.append({"role": role, "model": self.model, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
                        "usd": 0.0})
        data = canned(self.project, self.role)
        if len(self.calls) < self.faults:
            data["findings"][0]["evidence"].append(BAD_ID)
        self.calls.append({"role": role, "system": system, "user": user})
        return schema.model_validate(data), records
