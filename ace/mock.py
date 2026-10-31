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

from ace.checks import reference_setup
from ace.config import MODELS
from ace.curator import CuratorOutput, Decision
from ace.cost import usage_record
from ace.reflector import BulletVote, Lesson, Reflection

BULLET_ID = re.compile(r"\[((?:a\d+-)?[a-z]+-\d{4})\]")


def _usage(inp: int, out: int):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _response(content: list, stop: str, role: str):
    return NS(content=content, stop_reason=stop, usage=_usage(1500, 200), model=MODELS[role])


class MockLLM:
    # ---------------------------------------------------------------- generator
    def agent_turn(self, system: str, messages: list, tools: list, run_id: str = "", role: str = "generator"):
        if role == "diagnoser":
            return self._diagnoser_turn(messages)
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
            order = 2 if knows_p2 else rng.choice([1, 2])
            h = fea.get_problem(problem_id).params["h0"] * rng.uniform(0.6, 2.0)   # around the problem's own size
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

    def _rec(self, run_id, role="generator"):
        return usage_record(role, MODELS[role], _usage(1500, 200), run_id)

    # ---------------------------------------------------------------- diagnoser
    def _diagnoser_turn(self, messages: list):
        """Scripted session: list_runs -> stage report of the worst run -> diff best/worst ->
        one rerun of the worst run -> submit a diagnosis that CLAIMS the rerun confirmed the
        cause (so tests can check that code downgrades the claim when the rerun did not flip)."""
        turn = sum(1 for m in messages if m["role"] == "assistant")
        rec = self._rec("diagnoser", "diagnoser")

        def tool(name, args):
            return _response([NS(type="tool_use", id=f"d{turn}", name=name, input=args)], "tool_use", "diagnoser"), rec

        if turn == 0:
            return tool("list_runs", {})
        listing = json.loads(messages[2]["content"][0]["content"])["runs"]      # the list_runs result
        best, worst = listing[0], listing[-1]
        stage = "bcs" if worst["first_failing_stage"] == "bcs" else "mesh"
        change = {"name": "refine_mesh", "args": {"scope": "global", "factor": 1.5}} if stage == "mesh" else \
                 {"name": "convert_units", "args": {"target": "E", "factor": 1.0}}
        if turn == 1:
            return tool("get_stage_report", {"run_id": worst["run_id"]})
        if turn == 2:
            return tool("diff_runs", {"run_a": best["run_id"], "run_b": worst["run_id"]})
        if turn == 3:
            return tool("rerun_from_stage", {"run_id": worst["run_id"], "stage": stage, "change": change})
        cited = sorted({c for r in listing for c in r["cited"]})
        return tool("submit_diagnosis", {
            "divergence": "mock: the worst run diverged at its mesh",
            "runs": [{"run_id": worst["run_id"], "diagnosis_status": "confirmed", "stage": stage,
                      "category": "mesh_too_coarse" if stage == "mesh" else "units", "cause": "mock cause",
                      "evidence": "mock rerun", "hypotheses_tested": [
                          {"stage": stage, "intervention": change, "result": "confirmed"}]}],
            "lessons": [
                {"section": "meshing", "scope": "general", "evidence": "mock", "evidence_type": "counterfactual",
                 "confidence": "high", "run_ids": [worst["run_id"]],
                 "content": "When the scored quantity is a point stress, use quadratic (order 2) elements, "
                            "because they resolve stress gradients with fewer nodes."},
                {"section": "tool_use", "scope": "general", "evidence": "mock", "evidence_type": "contrast",
                 "confidence": "medium", "run_ids": [best["run_id"]],
                 "content": "When a mesh is over the node cap, scale every size by the hint's factor, because it lands under the cap."},
                {"section": "bcs", "scope": "general", "evidence": "mock", "evidence_type": "oracle_only",
                 "confidence": "low", "content": "When in doubt, check the supports, because oracles said so."}],
            "bullet_votes": [{"bullet_id": c, "vote": "helpful", "reason": "mock"} for c in cited]})


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


def fake_rerun(flip: bool = True):
    """A stand-in for counterfactual.rerun_from_stage (mock mode): no solver, a fixed outcome."""
    def rerun(run, stage, change, target_err, workdir):
        before = {"rel_error": 0.2, "accuracy": 0.1, "dofs": 1000, "first_failing_stage": stage, "solved": True, "within_cap": True,
                  "nodes": 100, "failed_checks": [f"{stage}: mock"]}
        after = dict(before, rel_error=0.01 if flip else 0.2, accuracy=1.0 if flip else 0.1,
                     first_failing_stage=None if flip else stage, failed_checks=[] if flip else before["failed_checks"])
        return {"run_id": run["run_id"], "stage": stage, "change": change, "before": before, "after": after,
                "flipped": flip, "solver_seconds": 0.0, "notes": ["mock rerun (no solver)"], "mesh_used": {}}
    return rerun
