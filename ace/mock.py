"""--mock: canned LLM responses so the whole loop runs end to end for free (and in tests).

The mock generator follows a fixed script (inspect -> mesh -> solve -> finish) with seeded
randomness (element order, mesh size, occasionally a wrong load sign) so runs differ. It reads
the playbook in its system prompt: a bullet mentioning quadratic/order 2 makes it use order 2,
which is enough to see scores move across generations. It uses the reference setup to solve,
which a real agent never sees; that is fine because nothing here is a real result.
"""

from __future__ import annotations

import json
import random
import re
import zlib
from types import SimpleNamespace as NS

from ace.checks import load_calibration, reference_setup
from ace.config import MODELS
from ace.curator import CuratorOutput, Decision
from ace.cost import usage_record
from ace.reflector import BulletVote, Lesson, Reflection

BULLET_ID = re.compile(r"\[([a-z]+-\d{4})\]")


def _usage(inp: int, out: int):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _response(content: list, stop: str, role: str):
    return NS(content=content, stop_reason=stop, usage=_usage(1500, 200), model=MODELS[role])


class MockLLM:
    # ---------------------------------------------------------------- generator
    def agent_turn(self, system: str, messages: list, tools: list, run_id: str = ""):
        from ace import fea   # local import: only needed by the mock generator

        problem_id = messages[0]["content"].splitlines()[0].removeprefix("Problem id: ").strip()
        turn = sum(1 for m in messages if m["role"] == "assistant")
        rng = random.Random(zlib.crc32(f"{run_id}-{turn}".encode()))
        playbook_ids = BULLET_ID.findall(system)
        knows_p2 = re.search(r"quadratic|order 2|order=2", system, re.I) is not None
        last = json.loads(messages[-1]["content"][-1]["content"]) if turn else {}

        using = playbook_ids[:2]

        def tool(name, args):
            return _response([NS(type="thinking", thinking=f"mock: next I call {name}"),
                              NS(type="tool_use", id=f"t{turn}", name=name, input={**args, "using": using})],
                             "tool_use", "generator")

        if turn == 0:
            return tool("inspect_geometry", {}), self._rec(run_id)
        if turn == 1 or (last.get("within_cap") is False):
            if last.get("within_cap") is False:          # over the cap: same order, coarser
                return tool("mesh", {"h": last["h_used"] * 1.3, "order": last["order"]}), self._rec(run_id)
            cal = load_calibration([problem_id])[problem_id]
            order = 2 if knows_p2 else rng.choice([1, 2])
            h = cal["by_order"][order - 1]["h"] * rng.uniform(0.85, 1.3)   # near the uniform size that fits
            return tool("mesh", {"h": h, "order": order}), self._rec(run_id)
        if "value" not in last and "error" not in last:   # meshed and within cap: solve
            setup = reference_setup(fea.get_problem(problem_id))
            if rng.random() < 0.2:                       # sometimes get a load direction wrong
                for bc in setup["bcs"]:
                    if bc["type"] == "traction":
                        bc["value"] = [-v for v in bc["value"]]
            return tool("solve", setup), self._rec(run_id)
        if "solve" in last and last.get("ok"):
            return tool("submit", {"solve": last["solve"]}), self._rec(run_id)
        return _response([NS(type="text", text="Done.")], "end_turn", "generator"), self._rec(run_id)

    def _rec(self, run_id):
        return usage_record("generator", MODELS["generator"], _usage(1500, 200), run_id)

    # ---------------------------------------------------------------- reflector / curator
    def structured(self, role: str, system: str, user: str, schema, records: list | None = None):
        rec = [] if records is None else records
        rec.append(usage_record(role, MODELS[role], _usage(6000, 800)))
        if schema is Reflection:
            cited = sorted(set(BULLET_ID.findall(user)))
            return Reflection(
                divergence="mock: the best run used quadratic elements",
                root_causes=[],
                lessons=[
                    Lesson(section="meshing", scope="general", evidence="mock",
                           content="When the scored quantity is a point stress, use quadratic (order 2) "
                                   "elements, because they resolve stress gradients with fewer nodes."),
                    Lesson(section="bcs", scope="one_off", evidence="mock",
                           content="When solving this exact problem, double-check the load sign."),
                ],
                bullet_votes=[BulletVote(bullet_id=b, vote="helpful", reason="mock") for b in cited],
            ), rec
        if schema is CuratorOutput:
            # Every candidate duplicates the first existing bullet; with none yet, the first
            # candidate is added and the rest duplicate it.
            n = len(re.findall(r"^\d+\. \(", user, re.M))
            existing = BULLET_ID.findall(user)
            return CuratorOutput(decisions=[
                Decision(lesson_index=i, action="DUPLICATE", duplicate_of=existing[0], content="") if existing
                else Decision(lesson_index=i, action="ADD", content="") if i == 0
                else Decision(lesson_index=i, action="DUPLICATE", duplicate_of="#0", content="")
                for i in range(n)
            ]), rec
        raise ValueError(f"mock has no canned answer for {schema}")
