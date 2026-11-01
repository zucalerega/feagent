"""--mock: canned LLM responses so the whole loop runs end to end for free (and in tests).

The mock generator follows a fixed script (inspect -> mesh -> solve -> view -> finer mesh ->
solve -> compare -> [nonlinear_check of the finer mesh, when the task offers it] -> submit the finer
solve) with seeded
randomness (element order, mesh size, occasionally a wrong load sign) so runs differ. It reads
the playbook in its system prompt: a bullet mentioning quadratic/order 2 makes it use order 2,
which is enough to see scores move across generations. It uses the reference setup to solve,
which a real agent never sees; that is fine because nothing here is a real result.
Structured outputs (reflector, curator, teacher, ...) are canned per schema; a schema without a canned
answer gets a minimal valid instance built from its JSON schema (_from_schema).
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

BULLET_ID = re.compile(r"\[((?:[ap]\d+-)?[a-z]+-\d{4})\]")


def _usage(inp: int, out: int):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def _result_json(block: dict) -> dict:
    """A tool_result's JSON: its content is the JSON text, or a list [text, image] (the view tool)."""
    c = block["content"]
    return json.loads(c if isinstance(c, str) else next(b["text"] for b in c if b["type"] == "text"))


def _response(content: list, stop: str, role: str):
    return NS(content=content, stop_reason=stop, usage=_usage(1500, 200), model=MODELS[role])


def _problem_id(task_text: str) -> str:
    """The problem id from the first line of a task prompt; in requirements mode that line holds a
    neutral task label instead (ace.requirements.label_for)."""
    shown = task_text.splitlines()[0].removeprefix("Problem id: ").strip()
    if shown.startswith("task-"):
        from ace.requirements import problem_for_label
        return problem_for_label(shown)
    return shown


class MockLLM:
    # ---------------------------------------------------------------- generator
    def agent_turn(self, system: str, messages: list, tools: list, run_id: str = "", role: str = "generator"):
        if role == "diagnoser":
            return self._diagnoser_turn(messages, {t["name"] for t in tools})
        if role == "big_generator":
            return self._big_turn(system, messages, tools, run_id)
        from ace import fea   # local import: only needed by the mock generator

        problem_id = _problem_id(messages[0]["content"])
        turn = sum(1 for m in messages if m["role"] == "assistant")
        rng = random.Random(zlib.crc32(f"{run_id}-{turn}".encode()))
        playbook_ids = BULLET_ID.findall(system)
        knows_p2 = re.search(r"quadratic|order 2|order=2", system, re.I) is not None
        last = _result_json(messages[-1]["content"][-1]) if turn else {}
        calls = [b for m in messages if m["role"] == "assistant" for b in m["content"] if b.type == "tool_use"]
        prev = calls[-1].name if calls else None
        first_mesh = next((c.input for c in calls if c.name == "mesh"), None)
        offered = {t["name"] for t in tools}

        using = playbook_ids[:2]

        def tool(name, args):
            return _response([NS(type="thinking", thinking=f"mock: next I call {name}"),
                              NS(type="tool_use", id=f"t{turn}", name=name, input={**args, "using": using})],
                             "tool_use", "generator")

        problem = fea.get_problem(problem_id)
        flow = fea.is_flow(problem)
        if turn == 0:
            return tool("inspect_geometry", {}), self._rec(run_id)
        if turn == 1 or (last.get("within_cap") is False):
            if last.get("within_cap") is False:          # over the cap: same element choice, coarser
                same = {"quads": last.get("quads", False)} if flow else {"order": last["order"]}
                return tool("mesh", {"h": last["h_used"] * 1.3, **same}), self._rec(run_id)
            h = self._h0(problem_id) * rng.uniform(0.6, 2.0)   # around the problem's own size
            if flow:                                     # finite volumes: no element order
                return tool("mesh", {"h": h, "quads": problem.dim == 2 and rng.random() < 0.5}), self._rec(run_id)
            order = 2 if knows_p2 else rng.choice([1, 2])
            return tool("mesh", {"h": h, "order": order}), self._rec(run_id)
        if prev == "view":                               # second, finer mesh for a convergence check
            return tool("mesh", {"h": first_mesh["h"] * 0.8, "order": first_mesh["order"]}), self._rec(run_id)
        if prev == "compare" and "nonlinear_check" in offered:
            # a nonlinear (test-scored) task: one check of the finer mesh, the latest one (mesh numbers count
            # every mesh call from 1); without CalculiX it fails with a clear error and the run goes on
            return tool("nonlinear_check", {"mesh": sum(c.name == "mesh" for c in calls)}), self._rec(run_id)
        if prev in ("compare", "nonlinear_check"):
            # the finer solve (compare reached means solves 1 and 2 both succeeded, even if compare failed)
            solve_b = next(c.input for c in reversed(calls) if c.name == "compare")["solve_b"]
            return tool("submit", {"solve": solve_b}), self._rec(run_id)
        if prev == "mesh" and last.get("ok"):            # meshed and within cap: solve
            setup = reference_setup(problem)
            if flow:   # a second-order lesson in the playbook -> the second-order convection scheme
                setup["convection"] = "linearUpwind" if knows_p2 else rng.choice(["upwind", "linearUpwind"])
            if rng.random() < 0.2:                       # sometimes get a load / inflow direction wrong
                for bc in setup["bcs"]:
                    if bc["type"] in ("traction", "velocity_inlet") and isinstance(bc.get("value"), list) \
                            and all(isinstance(v, (int, float)) for v in bc["value"]):
                        bc["value"] = [-v for v in bc["value"]]
            return tool("solve", setup), self._rec(run_id)
        if prev == "solve" and last.get("ok"):
            if flow:                                     # flow tasks have no view / compare: submit
                return tool("submit", {"solve": last["solve"]}), self._rec(run_id)
            if last["solve"] == 1:
                return tool("view", {"solve": 1, "kind": "stress", "view": "iso"}), self._rec(run_id)
            return tool("compare", {"solve_a": 1, "solve_b": last["solve"]}), self._rec(run_id)
        return _response([NS(type="text", text="Done.")], "end_turn", "generator"), self._rec(run_id)

    @staticmethod
    def _h0(problem_id: str) -> float:
        """The problem's own mesh size; gold CAD parts have none, so 1/15 of the bounding-box diagonal."""
        import math

        from ace import fea
        p = fea.get_problem(problem_id)
        if "h0" in p.params:
            return p.params["h0"]
        b = fea.geometry_summary(p)["bbox"]
        return math.dist(b[:3], b[3:]) / 15

    def _big_turn(self, system: str, messages: list, tools: list, run_id: str):
        """The single-agent baseline (ace/single_loop.py): every task in one conversation, each tool
        call carrying a `task`. Works the tasks in order: replays the conversation so far on the first
        unsubmitted task as a one-task conversation, lets the generator script above pick the next
        call, and tags it with that task."""
        tasks = re.findall(r"^Problem id: (\S+)", messages[0]["content"], re.M)
        results = {r["tool_use_id"]: r for m in messages[1:] if m["role"] == "user" for r in m["content"]}
        calls = [b for m in messages if m["role"] == "assistant" for b in m["content"] if b.type == "tool_use"]
        done = {c.input["task"] for c in calls if c.name == "submit" and not results[c.id]["is_error"]}
        task = next((t for t in tasks if t not in done), None)
        rec = self._rec(run_id, "big_generator")
        if task is None:
            return _response([NS(type="text", text="Done.")], "end_turn", "big_generator"), rec
        sub = [{"role": "user", "content": f"Problem id: {task}"}]
        for c in calls:
            if c.input.get("task") == task:
                sub += [{"role": "assistant", "content": [c]}, {"role": "user", "content": [results[c.id]]}]
        resp, _ = self.agent_turn(system, sub, tools, run_id=f"{run_id}-{task}")
        turn = sum(1 for m in messages if m["role"] == "assistant")
        for b in resp.content:
            if b.type == "tool_use":            # unique ids across tasks; the task argument
                b.id, b.input = f"b{turn}", {**b.input, "task": task}
        return resp, rec

    def _rec(self, run_id, role="generator"):
        return usage_record(role, MODELS[role], _usage(1500, 200), run_id)

    @staticmethod
    def _from_schema(schema, text: str):
        """A minimal valid instance of a pydantic schema with no canned answer (e.g. one a new role
        adds), built from its JSON schema: required fields only, one item per list, the first enum
        value; a string field named "agent" gets the first agent id (a0, a1, ...) in the prompt."""
        js = schema.model_json_schema()
        defs = js.get("$defs", {})
        agent = (re.findall(r"\ba\d+\b", text) or ["a0"])[0]

        def value(s: dict, name: str = ""):
            if "$ref" in s:
                return value(defs[s["$ref"].rsplit("/", 1)[-1]], name)
            if "const" in s:
                return s["const"]
            if "enum" in s:
                return s["enum"][0]
            if "anyOf" in s:                  # Optional[X] and unions: the first non-null option
                return value(next((o for o in s["anyOf"] if o.get("type") != "null"), {"type": "null"}), name)
            if "allOf" in s:
                return value(s["allOf"][0], name)
            if "default" in s:
                return s["default"]
            t = s.get("type")
            if t == "object":
                return {k: value(v, k) for k, v in s.get("properties", {}).items() if k in s.get("required", [])}
            if t == "array":
                return [value(s.get("items", {}), name)] * max(1, s.get("minItems", 1))
            if t == "string":
                return agent if name == "agent" else f"mock {name}"
            if t in ("integer", "number"):
                return s.get("minimum", 0)
            if t == "boolean":
                return False
            return None
        return schema.model_validate(value(js))

    # ---------------------------------------------------------------- diagnoser
    def _diagnoser_turn(self, messages: list, offered: set):
        """Scripted session: list_runs -> stage report of the worst run -> diff best/worst ->
        one rerun of the worst run (if reruns are offered) -> its error map (if offered, field
        mode) -> submit. With a rerun, the diagnosis CLAIMS the rerun confirmed the cause (so
        tests can check that code downgrades the claim when the rerun did not flip); without one
        (inspect mode) it claims only contrast evidence."""
        from ace import config
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
        rerun = "rerun_from_stage" in offered
        steps = [("get_stage_report", {"run_id": worst["run_id"]}),
                 ("diff_runs", {"run_a": best["run_id"], "run_b": worst["run_id"]})]
        if rerun:
            steps.append(("rerun_from_stage", {"run_id": worst["run_id"], "stage": stage, "change": change}))
        if "get_error_map" in offered and config.SCORE == "field":
            steps.append(("get_error_map", {"run_id": worst["run_id"]}))
        if turn <= len(steps):
            return tool(*steps[turn - 1])
        cited = sorted({c for r in listing for c in r["cited"]})
        agents = {r["run_id"]: r["run_id"].rsplit("-", 1)[-1] for r in (best, worst)}
        return tool("submit_diagnosis", {
            "divergence": "mock: the worst run diverged at its mesh",
            "runs": [{"run_id": worst["run_id"], "diagnosis_status": "confirmed" if rerun else "contrast_only",
                      "stage": stage, "category": "mesh_too_coarse" if stage == "mesh" else "units",
                      "cause": "mock cause", "evidence": "mock rerun" if rerun else "mock error map",
                      "hypotheses_tested": [{"stage": stage, "intervention": change, "result": "confirmed"}] if rerun
                      else []}],
            "agent_advice": [{"agent": a, "section": "meshing", "run_ids": [rid],
                              "content": "When a mesh is refined, refine where the error is, because the score "
                                         "covers the whole field.",
                              "ideas_to_try": ["refine along the loaded boundary at half the global size",
                                               "try a coarser global size with one local refinement"]}
                             for rid, a in agents.items()],
            "lessons": [
                {"section": "meshing", "scope": "general", "evidence": "mock",
                 "evidence_type": "counterfactual" if rerun else "contrast",
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
        rec.append(usage_record(role, MODELS.get(role, MODELS["teacher"]), _usage(6000, 800)))
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
        text = user if isinstance(user, str) else "\n".join(b["text"] for b in user if b["type"] == "text")
        if schema.__name__ in ("RequirementsDraft", "SetupProposal"):
            return getattr(self, "_" + schema.__name__)(text, schema), rec
        if schema.__name__ in ("AgentFeedback", "CrossRunFeedback", "SelfReflection", "OrganizerOutput"):
            return getattr(self, "_" + schema.__name__)(text, schema), rec
        if schema.__name__ == "AssignedFeedback":             # AgentFeedback plus the assignment
            return self._AgentFeedback(text, schema), rec
        return self._from_schema(schema, text), rec

    # ---------------------------------------------------------------- ace.requirements roles

    def _RequirementsDraft(self, text: str, schema):
        """The stored requirements of the problem if it has them; otherwise its statement with the
        support sentences dropped (enough to exercise the code, not a real translation)."""
        from ace import requirements
        pid = re.search(r"^PROBLEM (\S+)", text, re.M).group(1)
        if requirements.path_for(pid).exists():
            d = requirements.load(pid)
            return schema.model_validate({
                "requirements": d["requirements"],
                "group_aliases": [{"group": g, "alias": a} for g, a in (d.get("group_aliases") or {}).items()],
                "mapping": d.get("mapping") or []})
        statement = requirements.raw_problem(pid).description()
        kept = [ln for ln in statement.splitlines() if not ln.lower().startswith("support")]
        return schema.model_validate({"requirements": "\n".join(kept), "group_aliases": [], "mapping": []})

    def _SetupProposal(self, text: str, schema):
        """A reader that gets it right: the reference setup under the aliases it was shown."""
        from ace import requirements
        pid = _problem_id(text)
        setup = reference_setup(requirements.view_problem(pid, requirements.aliases_for(pid)))
        return schema.model_validate({"reasoning": "mock: the reference setup",
                                      **{k: setup[k] for k in ("formulation", "thickness", "bcs", "loads", "T_ref")},
                                      "material": {k: setup["material"][k] for k in ("E", "nu", "alpha")}})

    # ---------------------------------------------------------------- --deep roles
    # Deterministic answers that name the run ids / agents in the prompt, so the code downstream
    # (memory updates, output files) is exercised. Notes mention order 2 so mock scores can move.

    @staticmethod
    def _note(section="meshing", kind="fix", content="mock: use quadratic (order 2) elements"):
        return {"section": section, "kind": kind, "content": content}

    def warm_cache(self, *args, **kwargs):
        pass                                        # no cache to warm in mock mode

    def _AgentFeedback(self, text: str, schema):
        rid = re.search(r"^FOR RUN (\S+)", text, re.M).group(1)
        task = re.search(r"^TASK (\S+)", text, re.M).group(1)
        call = {"h": 0.1, "order": 2, "refinements": [{"group": None, "point": [0.0, 0.0], "size": 0.02,
                                                       "radius": 0.2}]}
        role = re.search(r"^YOUR ROLE: (\w+)", text, re.M)
        assignment = role and {"agent": rid.rsplit("-", 1)[-1], "role": role.group(1),
                               "experiment": f"mock {role.group(1)} experiment for {rid}", "why": "mock"}
        return schema.model_validate({
            "assignment": assignment,
            "agent": rid.rsplit("-", 1)[-1], "run_id": rid, "diagnosis": f"mock: {rid} error near the corner",
            "step_critique": [{"step": "mesh call 1", "what_it_did": "uniform mesh", "verdict": "mistake",
                               "correction": "mock: refine the corner to h/4"}],
            "promised_items_check": [], "strengths": ["mock strength"], "mistakes": ["mock mistake"],
            "next_mesh_call": call,
            "notes": [{**self._note(content=f"mock teacher note {i} for {rid}: refine the hotspot to h/4, order 2"),
                       "task": task} for i in range(8)]})

    def _CrossRunFeedback(self, text: str, schema):
        run_ids = list(dict.fromkeys(re.findall(r"^=== RUN (\S+) ", text, re.M)))
        return schema.model_validate({
            "task_summary": "mock: the best run used quadratic elements", "setup_warnings": [],
            "contested_regions": [{"region": "mock corner", "why_contested": "mock", "which_agents_differ": [],
                                   "what_to_do": "mock: refine there"}],
            "highlights": [{"agent": rid.rsplit("-", 1)[-1], "run_id": rid, "what": "mock idea", "why_it_works": "mock",
                            "how_to_copy": "mock"} for rid in run_ids[:1]],
            "shared_lessons": [self._note(kind="idea", content=f"mock shared lesson {i}: order 2 near hotspots")
                               for i in range(6)],
            "bullet_ratings": [{"bullet_id": b, "rating": "useful", "why": "mock"}
                               for b in sorted(set(BULLET_ID.findall(text)))]})

    def _SelfReflection(self, text: str, schema):
        rid = re.search(r"^MY RUN (\S+)", text, re.M).group(1)
        return schema.model_validate({
            "what_happened": f"mock: in {rid} I meshed uniformly",
            "what_i_learned": [self._note(kind="idea", content=f"mock self note {i} from {rid}: order 2")
                               for i in range(5)],
            "plan_next": "mock: refine the hotspot", "next_mesh_call": {"h": 0.1, "order": 2, "refinements": []}})

    def _OrganizerOutput(self, text: str, schema):
        # Keep every current note (with its id) and add every incoming one as new: the notes are
        # the one-line JSON objects organizer.build_user writes.
        notes = []
        for line in text.splitlines():
            if line.startswith("{"):
                n = json.loads(line)
                notes.append({"id": n.get("id"), "section": n["section"], "kind": n.get("kind") or "idea",
                              "source": n.get("source") or "mock", "content": n["content"]})
        return schema.model_validate({"notes": notes})


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
