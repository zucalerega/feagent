"""--mock: canned model responses, so the whole loop runs for free (and in tests).

The mock analyst follows a fixed script through the real harness: inspect_geometry -> find_faces -> mesh ->
solve -> refine_mesh -> solve [-> refine_mesh -> solve] -> study -> submit_report. It cheats where a model
would think: it solves the REFERENCE setup (single_agent.oracle.reference_setup; a real agent never sees
it), except that on some runs (seeded by the run id) it halves one load, so the scorecard's wrong-setup path
is exercised too. It reads its rulebook: when a learned rule mentions "three meshes" it studies three meshes
instead of two, and when one mentions "extrapolat" it reports extrapolated values; so the canned reflection
below (which adds such rules) visibly moves the scores across generations. None of it is a result.
"""

from __future__ import annotations

import json
import zlib
from types import SimpleNamespace as NS

from single_agent import workspace
from single_agent.agent import ROLE_AGENT


def _usage(inp: int, out: int):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _results(messages: list) -> list[dict]:
    """Every tool result so far, as (name unknown) JSON dicts in order."""
    out = []
    for m in messages:
        if m["role"] != "user" or isinstance(m["content"], str):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b["content"]
                txt = c if isinstance(c, str) else next(x["text"] for x in c if x["type"] == "text")
                try:
                    out.append(json.loads(txt))
                except (json.JSONDecodeError, TypeError):
                    out.append({})
    return out


def _calls(messages: list) -> list[str]:
    return [b.name for m in messages if m["role"] == "assistant" for b in m["content"] if b.type == "tool_use"]


class MockLLM:
    def agent_turn(self, system: str, messages: list, tools: list, run_id: str = "", role: str = ROLE_AGENT):
        from ace.config import MODELS
        from ace.cost import usage_record
        task = messages[0]["content"]
        label = task.split("Workspace: ", 1)[1].split(".", 1)[0]
        pid = workspace.problem_for_label(label)
        learned = system.split("# Learned rules", 1)[1].lower() if "# Learned rules" in system else ""
        three = "three meshes" in learned
        extrap = "extrapolat" in learned
        done, res = _calls(messages), _results(messages)
        seed = zlib.crc32(run_id.encode())
        n_levels = 3 if three else 2
        script = ["inspect_geometry", "find_faces", "mesh", "solve"] + ["refine_mesh", "solve"] * (n_levels - 1) \
            + ["study", "submit_report"]
        step = len(done)
        name = script[min(step, len(script) - 1)]
        args: dict = {}
        if name == "find_faces":
            args = {"query": {"kind": "plane"}}
        elif name == "mesh":
            args = {"h": 0.009, "order": 2}
        elif name == "refine_mesh":
            meshes = [r["mesh"] for r in res if "mesh" in r and "nodes" in r]
            args = {"mesh": meshes[-1], "factor": 1.35}
        elif name == "solve":
            from single_agent.oracle import reference_setup
            setup = reference_setup(pid)
            if seed % 3 == 0:          # a misread load on a third of the runs
                b = next((b for b in setup["bcs"] if b["type"] in ("force", "pressure")), None)
                if b is not None:
                    b["value"] = [0.5 * v for v in b["value"]] if isinstance(b["value"], list) else 0.5 * b["value"]
            meshes = [r["mesh"] for r in res if "mesh" in r and "nodes" in r]
            args = {"mesh": meshes[-1], **setup}
        elif name == "study":
            args = {"solves": [r["solve"] for r in res if r.get("ok") and "solve" in r and "dofs" in r]}
        elif name == "submit_report":
            solves = [r for r in res if r.get("ok") and "solve" in r and "dofs" in r]
            study = next((r for r in reversed(res) if "refinement_ratios" in r), {})
            key = workspace.load_key(pid)
            answers = []
            for q in key["questions"]:
                row = (study.get("qois") or {}).get(q["qoi"], {})
                v = (solves[-1]["qois"] if solves else {}).get(q["qoi"])
                if extrap and row.get("extrapolated") and row.get("convergence") == "monotone":
                    v = row["extrapolated"]
                status = "singular" if q["qoi"] == "vm_max" else "converged"
                U = row.get("gci_fine") or (2 * row["change"] if row.get("change") else None)
                answers.append({"id": q["id"], "quantity": q["qoi"], "value": v, "uncertainty": U, "status": status})
            args = {"idealization": [{"design_item": "as read", "faces": [], "model": "reference (mock)"}],
                    "final_solve": solves[-1]["solve"] if solves else 1, "study_solves": [s["solve"] for s in solves],
                    "answers": answers, "rules_used": ["D1"], "summary": "mock analysis"}
        block = NS(type="tool_use", id=f"tu_{step}", name=name, input=args)
        text = NS(type="text", text=f"(mock) step {step}: {name}")
        resp = NS(content=[text, block], stop_reason="tool_use", usage=_usage(3000, 300), model=MODELS[role])
        return resp, usage_record(role, resp.model, resp.usage, run_id)

    def structured(self, role: str, system: str, user, schema, records: list | None = None):
        from ace.config import MODELS
        from ace.cost import usage_record
        from single_agent.reflect import Diagnosis, Reflection, RuleOp
        records = [] if records is None else records
        records.append(usage_record(role, MODELS[role], _usage(8000, 800)))
        ops = []
        if "L001" not in user:
            ops.append(RuleOp(op="add", section="studies", reason="mock: the study had only two meshes",
                              text="When a question needs a converged value, run three meshes of one setup before "
                                   "reporting, because two meshes give no observed order."))
        elif "L002" not in user:
            ops.append(RuleOp(op="add", section="studies", reason="mock: finest values were reported",
                              text="When the study converges monotonically near the theoretical order, report the "
                                   "extrapolated value, because the finest mesh still carries the discretization error."))
        else:
            ops.append(RuleOp(op="vote", id="L001", vote="helpful", reason="mock"))
        return Reflection(diagnoses=[Diagnosis(task="mock", went_well=["it submitted"], went_wrong=[],
                                               root_causes=[])], ops=ops), records
