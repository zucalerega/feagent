"""The only place that talks to the Claude API. Two kinds of call:

  agent_turn  one step of the generator's tool-use loop
  structured  one Reflector/Curator call whose JSON output is validated against a pydantic model

Every call returns usage records (see ace.cost) so the loop can track spend.
The API key is read by the SDK from ANTHROPIC_API_KEY; this code never touches it.
With mock=True, ace.mock.MockLLM answers instead and nothing is sent anywhere.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pydantic

from ace.config import EFFORT, MAX_TOKENS, MODELS
from ace.cost import usage_record


def make_llm(mock: bool):
    if mock:
        from ace.mock import MockLLM
        return MockLLM()
    return LLM()


def text_block(s: str) -> dict:
    """A text content block, for a user message built from several parts (text and images)."""
    return {"type": "text", "text": s}


def image_block(path) -> dict:
    """A PNG file as an image content block (base64, sent inline with the message)."""
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _cached_system(text: str) -> list[dict]:
    # The system prompt (base instructions + playbook) is identical for every run in a
    # generation, so mark it for prompt caching: later calls read it at ~10% of the price.
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


class LLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def agent_turn(self, system: str, messages: list, tools: list[dict], run_id: str = "",
                   role: str = "generator"):
        """One step of a tool-use loop for `role` (the generator, or the diagnoser)."""
        resp = self.client.beta.messages.create(
            model=MODELS[role],
            max_tokens=MAX_TOKENS[role],
            system=_cached_system(system),
            messages=messages,
            tools=tools,
            cache_control={"type": "ephemeral"},   # also cache the growing conversation
            # "summarized" returns a readable summary of the reasoning (the default returns
            # empty thinking text). Same cost; it is what the trace and the Reflector see.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": EFFORT[role]},
            # If a safety classifier declines, re-run on a fallback model instead of failing.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        return resp, usage_record(role, resp.model, resp.usage, run_id)

    def structured(self, role: str, system: str, user: str | list[dict], schema: type[pydantic.BaseModel],
                   records: list | None = None):
        """Returns (parsed model or None, usage records). Retries once, then gives up.

        `user` is the user message: a string, or a list of content blocks (text_block /
        image_block) when the model should also see images (the deep teacher, self-reflection).

        The API constrains the output to the JSON schema; we still validate with pydantic
        ourselves (rather than messages.parse) so a failed attempt's tokens are still counted.
        """
        import anthropic
        fmt = {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}
        records = [] if records is None else records   # caller's list survives an exception
        for _attempt in range(2):
            resp = self.client.messages.create(
                model=MODELS[role],
                max_tokens=MAX_TOKENS[role],
                system=_cached_system(system),
                messages=[{"role": "user", "content": user}],
                output_config={"format": fmt, "effort": EFFORT[role]},
            )
            records.append(usage_record(role, resp.model, resp.usage))
            text = "".join(b.text for b in resp.content if b.type == "text")
            try:
                return schema.model_validate_json(text), records
            except pydantic.ValidationError:
                continue  # truncated (max_tokens), refused, or malformed: try once more
        return None, records
