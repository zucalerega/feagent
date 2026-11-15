"""dossier.agents: the judgement roles of a dossier.

The engines of dossier are rules and solvers. A rule cannot notice that a requirement can be read in two
ways, that a load path is missing from the system context, that a support is the wrong idealization or that
an assumption is not conservative. The roles here are language-model reviewers that read the dossier and
PROPOSE findings, decisions and assumptions. A proposal is a record like any other: a person reviews it.

Roles (dossier/agents/roles.py)
    requirements-reviewer   the specification against the system context (records of stage `validate`)
    plan-reviewer           the load-case plan and the hand calculations (stage `plan`)
    analysis-checker        each analysis, as an independent checker (stage `solve`)

How a role works, from the shell or from Python
    python -m dossier.agents roles
    python -m dossier.agents brief  PROJECT ROLE [--out FILE]      what the role reads: instructions + dossier
    python -m dossier.agents schema ROLE                           the JSON schema of what it submits
    python -m dossier.agents submit PROJECT ROLE FILE.json [--by MODEL_NAME] [--check]
    python -m dossier.agents run    PROJECT ROLE [--mock] [--model sonnet|opus]

    from dossier import agents
    text = agents.build_brief(project, "requirements-reviewer")
    result = agents.submit.submit(project, "requirements-reviewer", data, by="claude-opus-5-5")   # .ok .errors .warnings
    result = agents.check(project, "requirements-reviewer", data)                                 # the same, writes nothing
    out = agents.run(project, "plan-reviewer", mock=True)

Two ways to run a role. `run` makes one structured call through ace.llm (needs ANTHROPIC_API_KEY; `--mock`
gives a canned submission with no call). Without a key, an agent that a person drives (a Claude Code
subagent, as in the pilots of single_agent) reads `brief`, writes the JSON document and calls `submit`:
everything a role needs works from the shell.

What keeps a role honest (dossier/agents/submit.py)
    * It proposes records only. It gives no review, changes no input file and no record of a rule: the one
      file it writes is <project>/agent/<role>.json, which Project reads with the other stage files.
    * `submit` rejects the whole submission when it names a record that does not exist, leaves a required
      text empty, uses a word outside the vocabulary, or repeats an existing finding.
    * `submit` warns when a number with a unit is not a number of the dossier (within 1 %, in SI), when a
      text names an id that the dossier does not have, and when the STE check reports issues.
    * Each record carries origin.by = "agent:<role>" (plus " (<model>)" when the model is known), and the
      confidence of the agent and the warnings in the last sentences of its rationale. Decisions are marked
      `needs_review`.

Modules: roles (who looks for what, the instruction texts), brief (the dossier as Markdown), submit (the
submission schema, the check, the records), loop (one run through the API), mock (the canned submission).
"""

# The names here must not be the name of a submodule (brief, loop, mock, roles, submit): `agents.submit` is the
# module, and its function is `agents.submit.submit`.
from dossier.agents import brief, loop, mock, roles, submit  # noqa: F401  (the submodules, by name)
from dossier.agents.brief import build as build_brief
from dossier.agents.loop import NoKey, RunResult, run
from dossier.agents.roles import ROLES, Role, instructions, model_of, role_of
from dossier.agents.submit import Result, Submission, check, schema

__all__ = ["ROLES", "Role", "Result", "RunResult", "Submission", "NoKey", "brief", "build_brief", "check",
           "instructions", "loop", "mock", "model_of", "role_of", "roles", "run", "schema", "submit"]
