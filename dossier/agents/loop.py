"""One run of a role: build the brief, make one structured call, check the answer, write the records.

    run(project, role, mock=False, model="opus")  -> RunResult

    1. the system prompt is the role's instructions (dossier.agents.roles.instructions, the form without
       shell commands); the user message is the dossier part of the brief (dossier.agents.brief.dossier),
       for the analysis checker followed by the picture files (PNG) that the analyses list;
    2. one call through ace.llm (`LLM.structured`: the output is constrained to the JSON schema of
       dossier.agents.submit.Submission and validated with pydantic);
    3. dossier.agents.submit.submit checks the answer and writes <project>/agent/<role>.json;
    4. when the check rejects the answer, the error list goes back to the model once, with its submission,
       for a corrected one. A second rejection ends the run with nothing written.

The model configuration is added to ace's tables from here (`configure`), the way single_agent.agent does
it: ace.llm reads its model, token cap, effort and timeout per role name from dictionaries in ace.config
and ace.llm, and this module registers the role "dossier-<role>" there. Nothing in ace/ is edited. Effort
is `high` (a judgement role; Claude Opus 5.5 defaults to `medium`), the answer is capped at 32 000 tokens
(thinking counts toward the cap; the call streams, so a long answer does not time out).

`mock=True` uses dossier.agents.mock.MockLLM: no call, a canned submission that says so in its text.
Without ANTHROPIC_API_KEY and without `mock`, `run` raises NoKey: it needs the key, or `--mock`, or an agent
that a person drives with `python -m dossier.agents brief` and `submit` (no key needed: that is how the
pilots of this repository run, see single_agent/README.md).

The brief that the model read is kept beside the records: <project>/agent/<role>.brief.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dossier.agents import brief as brief_mod
from dossier.agents import roles as roles_mod
from dossier.agents import submit as submit_mod
from dossier.project import Project

MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
DEFAULT_MODEL = "opus"
EFFORT = "high"
MAX_TOKENS = 32_000
TIMEOUT = 600               # seconds of silence of the stream before the call gives up
MAX_PICTURES = 8            # picture files sent with the brief of the analysis checker
MAX_PICTURE_BYTES = 3_000_000

TASK = "Review the dossier in your role. Then give your submission."


class NoKey(RuntimeError):
    """`run` without --mock and without an API key."""

    def __init__(self, project: Any = "PROJECT", role: str = "ROLE"):
        super().__init__(
            "`run` makes a call to the Claude API and needs ANTHROPIC_API_KEY in the environment. Without a key:\n"
            f"  python -m dossier.agents run {project} {role} --mock      a canned submission (no model, for tests)\n"
            "or let an agent that a person drives do the review:\n"
            f"  python -m dossier.agents brief {project} {role} --out brief.md\n"
            f"  python -m dossier.agents submit {project} {role} submission.json --by MODEL_NAME")


@dataclass
class RunResult:
    role: str
    model: str                               # the model id, or "mock"
    result: submit_mod.Result | None = None  # the check of the last submission
    attempts: int = 0                        # submissions that the model gave (1, or 2 after a correction)
    usage: list[dict] = field(default_factory=list)
    brief_chars: int = 0
    brief_path: Path | None = None
    error: str = ""                          # why there is no accepted submission

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    @property
    def usd(self) -> float:
        return sum(u.get("usd", 0.0) for u in self.usage)

    def text(self) -> str:
        head = (f"{self.role} [{self.model}]: brief of {self.brief_chars} characters, {self.attempts} submission(s)"
                + (f", {sum(u.get('input', 0) + u.get('cache_read', 0) + u.get('cache_write', 0) for u in self.usage)} "
                   f"tokens in, {sum(u.get('output', 0) for u in self.usage)} out, about ${self.usd:.2f}"
                   if self.model != "mock" else ""))
        parts = [head]
        if self.error:
            parts.append(self.error)
        if self.result is not None:
            parts.append(self.result.text())
        return "\n".join(parts)


def has_key() -> bool:
    """A credential that the Anthropic SDK reads from the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def model_id(model: str) -> str:
    """`opus` / `sonnet`, or a full model id."""
    if model in MODELS:
        return MODELS[model]
    if model.startswith("claude-"):
        return model
    raise ValueError(f"unknown model {model!r}; use {' or '.join(MODELS)} (or a full model id)")


def llm_role(role: "str | roles_mod.Role") -> str:
    """The role name under which ace.llm knows a dossier role (its usage records carry it)."""
    return f"dossier-{roles_mod.get(role).name}"


def configure(role: "str | roles_mod.Role", model: str = DEFAULT_MODEL) -> str:
    """Register the role for ace.llm: its model, token cap, effort and timeout. Returns the model id.
    (ace.llm reads these per role from shared dictionaries; this adds entries, it edits nothing in ace/.)"""
    from ace import config as ace_config
    from ace import llm as ace_llm
    name, mid = llm_role(role), model_id(model)
    ace_config.MODELS[name] = mid
    ace_config.MAX_TOKENS[name] = MAX_TOKENS
    ace_config.EFFORT[name] = EFFORT
    ace_llm.TIMEOUT[name] = TIMEOUT
    return mid


def _picture_blocks(project: Project, role: roles_mod.Role) -> list[dict]:
    """The PNG pictures that the brief names, as image blocks with their captions (the analysis checker)."""
    if role.name != "analysis-checker":
        return []
    from ace.llm import image_block, text_block
    out: list[dict] = []
    for path, caption in brief_mod.pictures(project, role):
        if path.suffix.lower() != ".png" or path.stat().st_size > MAX_PICTURE_BYTES:
            continue
        out += [text_block(f"Picture file `{path}`" + (f": {caption}" if caption else "")), image_block(path)]
        if len(out) >= 2 * MAX_PICTURES:
            break
    return out


def correction(sub: Any, result: submit_mod.Result) -> str:
    """What goes back to the model when the check rejects its submission."""
    raw = sub.model_dump(mode="json") if hasattr(sub, "model_dump") else sub
    return ("The tool checked your submission and rejected it. It wrote nothing.\n\n"
            "Your submission:\n```json\n" + json.dumps(raw, indent=1) + "\n```\n\n"
            "The errors:\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(result.errors, 1)) + "\n\n"
            "Give the complete submission again with every error corrected. Keep what was right. Name only ids "
            "that the dossier has.")


def run(project: "Project | str | Path", role: "str | roles_mod.Role", mock: bool = False, model: str = DEFAULT_MODEL,
        llm: Any = None, pictures: bool = True, log: Callable[[str], None] | None = None) -> RunResult:
    """One run of a role (see the module docstring). `llm`: an object with ace.llm.LLM's `structured`
    (tests); by default the mock or ace.llm.LLM. Raises NoKey when a real call has no API key."""
    if not isinstance(project, Project):
        project = Project.load(project)
    r = roles_mod.get(role)
    log = log or (lambda _msg: None)
    if llm is None and mock:
        from dossier.agents.mock import MockLLM
        llm = MockLLM(project, r)
    if llm is None:
        if not has_key():
            raise NoKey(project.root, r.name)
        name = configure(r, model)
        from ace.llm import LLM
        llm = LLM()
    elif getattr(llm, "model", None):
        name = str(llm.model)                # a mock, or a stand-in of a test: it names itself
    else:
        name = configure(r, model)           # an ace.llm.LLM that the caller made
    out = RunResult(role=r.name, model=name)

    system = roles_mod.instructions(r, project=str(project.root), shell=False)
    data = brief_mod.dossier(project, r, reserve=len(system))
    out.brief_chars = len(system) + len(data)
    out.brief_path = submit_mod.agent_path(project, r).with_suffix(".brief.md")
    out.brief_path.parent.mkdir(parents=True, exist_ok=True)
    out.brief_path.write_text(f"# Brief of the role `{r.name}` (what the model read)\n\n{system}\n{data}")
    log(f"{r.name}: brief of {out.brief_chars} characters -> {out.brief_path}")

    blocks = _picture_blocks(project, r) if (pictures and name != "mock") else []
    user: Any = f"{data}\n{TASK}" if not blocks else [{"type": "text", "text": data}, *blocks,
                                                       {"type": "text", "text": TASK}]
    for attempt in (1, 2):
        sub, _ = llm.structured(llm_role(r), system, user, submit_mod.Submission, out.usage)
        if sub is None:
            out.error = ("the model gave no answer that fits the schema of the submission (a refusal, or an answer "
                         "cut off at the token cap)")
            return out
        out.attempts = attempt
        out.result = submit_mod.submit(project, r, sub, by=name)
        if out.result.ok:
            log(f"{r.name}: accepted on submission {attempt}")
            return out
        log(f"{r.name}: submission {attempt} rejected with {len(out.result.errors)} error(s)")
        if attempt == 1:
            note = correction(sub, out.result)
            user = f"{data}\n{TASK}\n\n{note}" if isinstance(user, str) else [*user, {"type": "text", "text": note}]
    out.error = "the check rejected the submission and the corrected submission. Nothing was written."
    return out
