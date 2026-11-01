"""The only place that talks to the Claude API. Two kinds of call:

  agent_turn  one step of the generator's tool-use loop
  structured  one Reflector/Curator call whose JSON output is validated against a pydantic model

Every call returns usage records (see ace.cost) so the loop can track spend.
The API key is read by the SDK from ANTHROPIC_API_KEY; this code never touches it.
With mock=True, ace.mock.MockLLM answers instead and nothing is sent anywhere.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import pydantic

from ace.config import EFFORT, FAST, MAX_TOKENS, MODELS
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


# Seconds before a structured call gives up. The --deep roles are on the generation's critical path
# and their answers are a few thousand tokens, so a stalled call should fail fast (the loop then
# skips that one update) rather than hold the whole generation for 10 minutes.
# Structured calls stream, so for them this is the longest SILENCE allowed (the API sends events,
# including keep-alive pings, while it works), not a cap on the whole call.
TIMEOUT = {"teacher": 420, "reflect": 240, "organizer": 240,   # teacher: Opus, high effort, up to 32k tokens
           # agent_turn of the single-agent baseline: without an explicit timeout the SDK refuses a
           # non-streaming call whose max_tokens (32k) could take over 10 minutes
           "big_generator": 1200}


def _cached_system(text: str) -> list[dict]:
    # The system prompt (base instructions + playbook) is identical for every run in a
    # generation, so mark it for prompt caching: later calls read it at ~10% of the price.
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# Beta flag the Claude API docs give for fast mode (with speed="fast" on the beta endpoint).
FAST_BETA = "fast-mode-2026-02-01"


class LLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self._fast_ok = True               # set False (once, logged) if the API rejects fast mode
        self._lock = threading.Lock()

    def _request(self, role: str, stream: bool, **kw):
        """Send one Messages request for `role` and return the final message.

        stream=True streams it and assembles the message (get_final_message): the answer is the same,
        but a long answer no longer has to arrive within one HTTP response time limit.
        If FAST[role] is on, the request asks for fast mode; if the API rejects that (400), fast mode
        is switched off for the rest of the run, logged once, and the request is resent normally."""
        import anthropic
        if FAST.get(role) and self._fast_ok:
            try:
                return self._send(self.client.beta.messages, stream, speed="fast", betas=[FAST_BETA], **kw)
            except anthropic.BadRequestError as e:
                with self._lock:
                    if self._fast_ok:
                        self._fast_ok = False
                        print(f"  fast mode rejected for {role}, continuing at normal speed: {e}")
        return self._send(self.client.messages, stream, **kw)

    @staticmethod
    def _send(api, stream: bool, **kw):
        if not stream:
            return api.create(**kw)
        with api.stream(**kw) as s:
            return s.get_final_message()

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
            **({"timeout": TIMEOUT[role]} if role in TIMEOUT else {}),
        )
        return resp, usage_record(role, resp.model, resp.usage, run_id)

    def structured(self, role: str, system: str, user: str | list[dict], schema: type[pydantic.BaseModel],
                   records: list | None = None):
        """Returns (parsed model or None, usage records). Retries once, then gives up.

        `user` is the user message: a string, or a list of content blocks (text_block /
        image_block) when the model should also see images (the deep teacher, self-reflection).

        The API constrains the output to the JSON schema; we still validate with pydantic
        ourselves (rather than messages.parse) so a failed attempt's tokens are still counted.
        The call streams (see _request), so a long answer (the teacher's, up to 32k tokens) is fine.
        """
        import anthropic
        fmt = {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}
        records = [] if records is None else records   # caller's list survives an exception
        for _attempt in range(2):
            resp = self._request(
                role, stream=True,
                model=MODELS[role],
                max_tokens=MAX_TOKENS[role],
                system=_cached_system(system),
                messages=[{"role": "user", "content": user}],
                output_config={"format": fmt, "effort": EFFORT[role]},
                timeout=TIMEOUT.get(role, 600),
            )
            records.append(usage_record(role, resp.model, resp.usage))
            text = "".join(b.text for b in resp.content if b.type == "text")
            try:
                return schema.model_validate_json(text), records
            except pydantic.ValidationError:
                continue  # truncated (max_tokens), refused, or malformed: try once more
        return None, records

    def warm_cache(self, role: str, system: str, user: list[dict], schema: type[pydantic.BaseModel],
                   records: list | None = None) -> None:
        """Write the prompt cache for calls that will share this system prompt, schema and the start
        of `user` (up to its block marked cache_control), by sending them with max_tokens=1. Parallel
        calls cannot read a cache that is still being written, so the deep teacher sends this first
        and then all its per-agent calls at once: they read the long shared prefix at ~5% of the
        price. The 1-token answer is thrown away. A failure only costs the cache, so it is reported
        and swallowed. It uses the same speed as the calls it warms: switching speed invalidates the
        cache of the system prompt and messages."""
        import anthropic
        try:
            fmt = {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}
            resp = self._request(
                role, stream=False, model=MODELS[role], max_tokens=1, system=_cached_system(system),
                messages=[{"role": "user", "content": user}],
                output_config={"format": fmt, "effort": EFFORT[role]}, timeout=120)
        except Exception as e:      # noqa: BLE001
            print(f"  warm_cache({role}) failed, calls go uncached: {e}")
            return
        if records is not None:
            records.append(usage_record(role, resp.model, resp.usage))
