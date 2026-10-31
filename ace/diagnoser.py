"""The diagnoser: an LLM agent that diagnoses the runs on one problem by EXPERIMENT, not by reading.

Design principle: the LLM proposes hypotheses; code and the solver provide proof.
  - Layer 1 (ace/oracles.py) and layer 2 (ace/stages.py) are computed before the session and are
    ground truth the diagnoser may not contradict.
  - Layer 3 (ace/counterfactual.py) is a tool: the diagnoser tests "the cause was X" by rerunning
    the run from the suspect stage with ONE typed change and seeing whether the failure flips.
  - After the session, code checks every claim against the reruns that actually happened
    (verify): a hypothesis marked "confirmed" without a flipped rerun is downgraded, and so is a
    lesson that claims counterfactual evidence it does not have.

One session per problem per generation, limited to DIAGNOSER_MAX_TOOL_CALLS tool calls and
DIAGNOSER_MAX_RERUNS reruns. The output (Diagnosis) extends the Reflector's schema, so the
Curator consumes it the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import BaseModel, Field

from ace import counterfactual as cf
from ace.config import (DIAGNOSER_MAX_RERUNS, DIAGNOSER_MAX_TOOL_CALLS, DIAGNOSER_MIN_RERUNS, MAX_SOLVES, SCORE, TASKS, TRACE_FIELD_CHARS,
                        TRACE_PAGE)
from ace.reflector import BulletVote, Lesson, Section  # noqa: F401  (Section re-exported for the schema)

Stage = Literal["geometry", "mesh", "bcs", "solve", "postprocess"]
Category = Literal["missing_constraint", "wrong_load_location", "load_magnitude_or_sign", "units",
                   "mesh_too_coarse", "singularity", "over_constraint", "gaming", "solver", "other", "none"]


# --------------------------------------------------------------------------- output schema


class Intervention(BaseModel):
    name: Literal["refine_mesh", "set_local_mesh_size", "change_element_order", "fix_constraint", "fix_load",
                  "convert_units"]
    args: dict = Field(default_factory=dict)


class Hypothesis(BaseModel):
    stage: Stage
    intervention: Intervention
    result: Literal["confirmed", "rejected", "not_tested"]
    before: dict = Field(default_factory=dict)     # filled in by code from the actual rerun
    after: dict = Field(default_factory=dict)
    note: str = ""


class RunDiagnosis(BaseModel):
    run_id: str
    diagnosis_status: Literal["confirmed", "contrast_only", "unexplained"]
    stage: Stage | None = Field(description="the root-cause stage, or null if the run did not fail")
    category: Category
    cause: str
    evidence: str = Field(description="run_id + step, oracle names, or rerun results")
    hypotheses_tested: list[Hypothesis] = Field(default_factory=list)
    gaming: str | None = Field(default=None, description="how the run made the problem easier, if it did")


class DiagnosisLesson(Lesson):
    evidence_type: Literal["counterfactual", "contrast", "oracle_only"]
    confidence: Literal["high", "medium", "low"]
    run_ids: list[str] = Field(default_factory=list, description="runs whose evidence supports the lesson")


class AgentAdvice(BaseModel):
    agent: str = Field(description='the agent, as at the end of its run_id, e.g. "a3"')
    section: Section
    content: str = Field(description='advice for THIS agent from its own run: "When X, do Y, because Z.", at most 40 words')
    run_ids: list[str] = Field(default_factory=list, description="the run(s) of this agent it is based on")


class Diagnosis(BaseModel):
    divergence: str = Field(description="where the best and worst runs first diverged")
    runs: list[RunDiagnosis]
    lessons: list[DiagnosisLesson] = Field(description="1 to 3 overall lessons, for every agent on this problem")
    agent_advice: list[AgentAdvice] = Field(default_factory=list,
                                            description="one piece of advice for EACH agent whose run you saw")
    bullet_votes: list[BulletVote] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list, description="leave empty; filled in by code")


# --------------------------------------------------------------------------- the procedure


SYSTEM = f"""\
You are the diagnoser in a loop that improves an FEA setup agent. Several agent runs attempted the
same problem. Find out WHY the weaker runs did worse, by experiment, and turn confirmed causes into
short lessons for the agent's playbook.

The pipeline has stages: geometry -> mesh -> bcs (boundary conditions and loads) -> solve ->
postprocess. Each run already has physics-oracle results and a stage report computed by code
(ground truth). The score is the team's speed/accuracy metric: accuracy = min(1, tolerance /
error) times efficiency = DOFs of the cheapest run on this problem that met the tolerance / this
run's DOFs. 1 = the cheapest mesh that is accurate enough; a run that is accurate but uses many
more DOFs than needed, or cheap but outside the tolerance, scores lower; invalid runs score 0.
Each agent had at most {MAX_SOLVES} solves and a node cap (a safety limit).

Procedure:
1. Start from the code evidence: oracle failures and each run's first failing stage
   (list_runs, get_stage_report, get_oracles). Never contradict an oracle result.
2. Contrast the best and worst runs with diff_runs; find where their choices first diverged.
3. For each failed or clearly weaker run, form at most 2 hypotheses, each naming a stage and ONE
   typed intervention that would fix it.
4. Test the most important hypothesis with rerun_from_stage. If the outcome flips (the first
   failing stage moves later, or for an accuracy failure the error falls 2x or more) it is
   CONFIRMED; otherwise REJECTED, and test the second hypothesis if you have reruns left.
   Reruns report DOFs too: for a run that was accurate but expensive, a cheaper mesh that stays
   within tolerance is also a finding (score depends on both).
5. If nothing is confirmed, mark the run "unexplained" (or "contrast_only" if the best-vs-worst
   contrast still points clearly at a cause). Do not invent a cause.
6. Look for gaming: runs that scored or passed by making the problem easier (a load left out,
   supports added).
7. Write 1-3 lessons only from confirmed causes (evidence_type "counterfactual") or strong
   contrast evidence (evidence_type "contrast", confidence "medium"). Form: "When X, do Y,
   because Z.", at most 40 words, with a scope (general / problem_class / one_off). List the
   supporting run_ids. Never put a reference value or numeric answer of this problem in a lesson.
8. Write agent_advice: for EACH agent (the a<k> at the end of its run_id), one piece of advice
   specific to its own run: what it should change, or keep doing, to beat the other agents next
   time. Agents keep this advice across problems, so make it about how the agent works, not
   about this problem's numbers. Lessons (step 7) are the overall advice for everyone.
9. Vote helpful / harmful / neutral on each playbook bullet the runs cited, with a reason.

Limits: at most {DIAGNOSER_MAX_TOOL_CALLS} tool calls and {DIAGNOSER_MAX_RERUNS} reruns per session. Run at
least {DIAGNOSER_MIN_RERUNS} rerun before submitting: even when every run looks fine, test the change most
likely to beat the best run (e.g. refining where its error is, or a cheaper mesh).
When you are done, or a limit is reached, call submit_diagnosis with your result. Report the
results of reruns exactly as the tool returned them; code checks every claim against the reruns
that actually ran.
"""

if SCORE == "field":   # field mode: the score is against the automesher (ace.config, ace.checks)
    SYSTEM = SYSTEM.replace(
        "The score is the team's speed/accuracy metric: accuracy = min(1, tolerance /\nerror) times efficiency = DOFs of the cheapest run on this problem that met the tolerance / this\nrun's DOFs. 1 = the cheapest mesh that is accurate enough; a run that is accurate but uses many\nmore DOFs than needed, or cheap but outside the tolerance, scores lower; invalid runs score 0.",
        "rel_error = max(field error, probe error): the energy-norm error of the run's whole stress field\n"
        "vs a fine reference, and the RMS von Mises error at 12 check points (4 told to the agent, 8\n"
        "hidden; get_oracles/list_runs show field_error and probe_error separately). The score\n"
        "is against an automesher (gmsh default sizes, quadratic): gain g = automesher error / rel_error,\n"
        "accuracy A = (g/2)^2 below the 2x target (tolerance = automesher error / 2), g/2 above it;\n"
        "efficiency = automesher DOFs / run DOFs; score = A x efficiency (1 = 2x the automesher's accuracy\n"
        "at its DOFs; invalid runs score 0), plus a bonus of up to 0.25 for having the smallest error\n"
        "on the hidden check points among the runs (a tie pays nobody). Where the mesh puts its elements is\n"
        "what matters.")
    assert "automesher" in SYSTEM


def _intervention_help() -> str:
    return "; ".join(f"{k} [{v[0]}]: {v[1]}" for k, v in cf.INTERVENTIONS.items())


def tool_specs() -> list[dict]:
    run_id = {"type": "string"}
    return [
        {"name": "list_runs", "description": "All runs on this problem, best to worst: score, error, first failing "
                                             "stage, failed checks, mesh size, solves, cited bullets.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "read_trace", "description": f"A run's steps (what it said and thought, each tool call and result), "
                                              f"{TRACE_PAGE} per page; long fields are cut. Use next_start to page.",
         "input_schema": {"type": "object", "properties": {"run_id": run_id, "start": {"type": "integer"}},
                          "required": ["run_id"]}},
        {"name": "get_oracles", "description": "A run's physics-oracle results (pass / fail / unavailable, values, why).",
         "input_schema": {"type": "object", "properties": {"run_id": run_id}, "required": ["run_id"]}},
        {"name": "get_stage_report", "description": "A run's stage-by-stage invariant checks and first failing stage.",
         "input_schema": {"type": "object", "properties": {"run_id": run_id}, "required": ["run_id"]}},
        {"name": "diff_runs", "description": "Where two runs' choices first diverged, and their final mesh and setup "
                                             "side by side.",
         "input_schema": {"type": "object", "properties": {"run_a": run_id, "run_b": run_id},
                          "required": ["run_a", "run_b"]}},
        {"name": "rerun_from_stage", "description": "Counterfactual experiment: rerun a run's scored solve from `stage` "
                                                    "with ONE change and compare before/after. Interventions: "
                                                    + _intervention_help(),
         "input_schema": {"type": "object", "properties": {
             "run_id": run_id, "stage": {"type": "string", "enum": ["mesh", "bcs"]},
             "change": {"type": "object", "properties": {"name": {"type": "string", "enum": list(cf.INTERVENTIONS)},
                                                         "args": {"type": "object"}}, "required": ["name", "args"]}},
             "required": ["run_id", "stage", "change"]}},
        {"name": "get_playbook_bullets", "description": "The text of playbook bullets by ID.",
         "input_schema": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                          "required": ["ids"]}},
        # The plain pydantic schema (not the strict structured-output form, which would forbid the
        # free-form `args` / `before` / `after` objects); pydantic validates the submission anyway.
        {"name": "submit_diagnosis", "description": "Submit the final diagnosis (ends the session).",
         "input_schema": Diagnosis.model_json_schema()},
    ]


# --------------------------------------------------------------------------- the tools


def _cut(x, n: int = TRACE_FIELD_CHARS) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


class DiagnoserTools:
    """Executes the diagnoser's tool calls over the runs of one problem, and logs every rerun so
    the final diagnosis can be checked against what actually happened."""

    def __init__(self, problem_id: str, runs: list[dict], playbook, target_err: float, workdir: Path,
                 rerun_fn=None):
        self.problem_id = problem_id
        self.runs = {r["run_id"]: r for r in runs}
        self.playbook = playbook
        self.target_err = target_err
        self.workdir = Path(workdir)
        self.rerun_fn = rerun_fn or cf.rerun_from_stage      # mock mode passes a fake
        self.calls = 0
        self.reruns: list[dict] = []                            # every rerun that actually ran
        self.solver_seconds = 0.0

    def _run(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise KeyError(f"unknown run {run_id!r}; runs: {sorted(self.runs)}")
        return self.runs[run_id]

    def call(self, name: str, a: dict) -> tuple[dict, bool]:
        if name != "submit_diagnosis":
            self.calls += 1
            if self.calls > DIAGNOSER_MAX_TOOL_CALLS:
                return {"error": f"tool-call limit ({DIAGNOSER_MAX_TOOL_CALLS}) reached: call submit_diagnosis now "
                                 "with what you have"}, True
        fn = getattr(self, f"_{name}", None)
        if fn is None or name not in {s["name"] for s in tool_specs()}:   # the model named a tool that doesn't exist
            return {"error": f"unknown tool {name!r}; available: {sorted(s['name'] for s in tool_specs())}"}, True
        try:
            return fn(**a), False
        except (KeyError, TypeError, cf.InterventionError) as e:
            return {"error": f"{type(e).__name__}: {e}"}, True

    def _list_runs(self) -> dict:
        rows = []
        for r in sorted(self.runs.values(), key=lambda r: -r["checks"]["score"]):
            c, st = r["checks"], r.get("stages") or {}
            failed = [f"{s}: {x['check']}" for s in ("geometry", "mesh", "bcs", "solve", "postprocess")
                      for x in st.get(s, {}).get("checks", []) if not x["ok"]]
            rows.append({"run_id": r["run_id"], "score": round(c["score"], 3), "accuracy": c.get("accuracy"),
                         "efficiency": c.get("efficiency"), "dofs": c.get("dofs"), "rel_error": c.get("rel_error"),
                         **({"field_error": c.get("field_error"), "probe_error": c.get("probe_error")} if SCORE == "field" else {}),
                         "first_failing_stage": st.get("first_failing_stage"), "failed_checks": failed,
                         "nodes": c.get("nodes"), "node_cap": c.get("node_cap"), "order": c.get("order"),
                         "n_meshes": c.get("n_meshes"), "n_solves": c.get("n_solves"), "cited": r.get("cited", [])})
        return {"problem": self.problem_id, "qoi": TASKS[self.problem_id]["qoi"], "tolerance": self.target_err,
                "runs": rows}

    def _read_trace(self, run_id: str, start: int = 0) -> dict:
        trace = self._run(run_id)["trace"]
        steps = []
        for t in trace[start:start + TRACE_PAGE]:
            body = t.get("text") if t["type"] in ("text", "thinking") else (t.get("input") if t["type"] == "tool_call" else t.get("result"))
            steps.append({"turn": t.get("turn"), "type": t["type"], "name": t.get("name"), "content": _cut(body)})
        nxt = start + TRACE_PAGE
        return {"run_id": run_id, "steps": steps, "total_steps": len(trace), "next_start": nxt if nxt < len(trace) else None}

    def _get_oracles(self, run_id: str) -> dict:
        orc = self._run(run_id).get("oracles") or {}
        return {k: {"status": v["status"], "why": v["why"], "values": _cut(v.get("values"), 400)}
                for k, v in orc.items() if isinstance(v, dict) and "status" in v}

    def _get_stage_report(self, run_id: str) -> dict:
        return self._run(run_id).get("stages") or {"error": "no stage report for this run"}

    def _diff_runs(self, run_a: str, run_b: str) -> dict:
        def decisions(r):
            return [(t["name"], {k: v for k, v in t["input"].items() if k != "using"})
                    for t in r["trace"] if t["type"] == "tool_call" and t["name"] in ("mesh", "solve", "submit")]

        def final(r):
            from ace import checkpoint as ckpt
            cp = ckpt.checkpoint(r, allow_failed=True)
            return {"mesh_request": cp and cp["request"], "setup": cp and cp["setup"],
                    "nodes": r["checks"].get("nodes"), "dofs": r["checks"].get("dofs"), "score": r["checks"]["score"]}
        da, db = decisions(self._run(run_a)), decisions(self._run(run_b))
        first = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), None)
        if first is None and len(da) != len(db):
            first = min(len(da), len(db))
        return {"first_divergent_decision": first,
                "at_divergence": {run_a: _cut(da[first]) if first is not None and first < len(da) else None,
                                  run_b: _cut(db[first]) if first is not None and first < len(db) else None},
                "decisions": {run_a: len(da), run_b: len(db)},
                "final": {run_a: final(self._run(run_a)), run_b: final(self._run(run_b))}}

    def _rerun_from_stage(self, run_id: str, stage: str, change: dict) -> dict:
        if len(self.reruns) >= DIAGNOSER_MAX_RERUNS:
            return {"error": f"rerun limit ({DIAGNOSER_MAX_RERUNS}) reached"}
        r = self.rerun_fn(self._run(run_id), stage, change, self.target_err, self.workdir / f"rerun{len(self.reruns)}")
        self.reruns.append(r)
        self.solver_seconds += r.get("solver_seconds", 0.0)
        keep = ("solved", "rel_error", "accuracy", "dofs", "nodes", "within_cap", "first_failing_stage", "failed_checks")
        return {"flipped": r["flipped"], "before": {k: r["before"].get(k) for k in keep},
                "after": {k: r["after"].get(k) for k in keep}, "notes": r.get("notes", []),
                "reruns_left": DIAGNOSER_MAX_RERUNS - len(self.reruns)}

    def _get_playbook_bullets(self, ids: list[str]) -> dict:
        return {i: (b.content if (b := self.playbook.get(i)) else "(not in the playbook)") for i in ids}


# --------------------------------------------------------------------------- code checks the claims


def verify(diag: Diagnosis, reruns: list[dict]) -> Diagnosis:
    """Overwrite every hypothesis's result and metrics with what the logged reruns show, and
    downgrade claims the evidence does not support."""
    notes = []

    def matching(run_id, h):
        return [r for r in reruns if r["run_id"] == run_id and r["stage"] == h.stage
                and r["change"]["name"] == h.intervention.name and r["change"]["args"] == h.intervention.args]

    confirmed_runs = set()
    for rd in diag.runs:
        for h in rd.hypotheses_tested:
            m = matching(rd.run_id, h)
            actual = "not_tested" if not m else ("confirmed" if m[-1]["flipped"] else "rejected")
            if actual != h.result:
                notes.append(f"{rd.run_id}: hypothesis {h.intervention.name} claimed {h.result}, reruns show {actual}")
                h.result = actual
            if m:
                h.before = {k: m[-1]["before"].get(k) for k in ("rel_error", "accuracy", "dofs", "first_failing_stage")}
                h.after = {k: m[-1]["after"].get(k) for k in ("rel_error", "accuracy", "dofs", "first_failing_stage", "within_cap")}
            if h.result == "confirmed":
                confirmed_runs.add(rd.run_id)
        if rd.diagnosis_status == "confirmed" and rd.run_id not in confirmed_runs:
            notes.append(f"{rd.run_id}: status confirmed without a confirmed rerun, downgraded to contrast_only")
            rd.diagnosis_status = "contrast_only"
    for les in diag.lessons:
        if les.evidence_type == "counterfactual" and not (set(les.run_ids) & confirmed_runs):
            notes.append(f"lesson '{les.content[:60]}...' claimed counterfactual evidence without a confirmed rerun, "
                         "downgraded to contrast")
            les.evidence_type = "contrast"
    diag.lessons = diag.lessons[:3]
    diag.verification_notes = notes
    return diag


# --------------------------------------------------------------------------- the session


def diagnose_problem(problem_id: str, runs: list[dict], playbook, llm, target_err: float, workdir: Path,
                     rerun_fn=None, usage: list | None = None) -> dict:
    """Run one diagnoser session. Returns {"diagnosis": Diagnosis | None, "log": [...],
    "reruns": [...], "solver_seconds": float}; token usage goes into `usage` as it happens."""
    tools = DiagnoserTools(problem_id, runs, playbook, target_err, workdir, rerun_fn)
    specs = tool_specs()
    usage = [] if usage is None else usage
    messages: list = [{"role": "user", "content":
                       f"Diagnose the {len(runs)} runs on problem {problem_id} (scored quantity "
                       f"{TASKS[problem_id]['qoi']}, node cap {TASKS[problem_id]['node_cap']}). Start with list_runs."}]
    log, diag, bad_submits, refused = [], None, 0, False
    # Room for the tool-call limit, plus a few turns to submit after hitting it.
    for _turn in range(DIAGNOSER_MAX_TOOL_CALLS + 4):
        resp, rec = llm.agent_turn(SYSTEM, messages, specs, run_id=f"diagnose-{problem_id}", role="diagnoser")
        usage.append(rec)
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            messages.append({"role": "user", "content": "Call submit_diagnosis with your result."})
            continue
        results = []
        for b in calls:
            if (b.name == "submit_diagnosis" and not refused and len(tools.reruns) < DIAGNOSER_MIN_RERUNS
                    and tools.calls < DIAGNOSER_MAX_TOOL_CALLS):
                # Refused once: lessons should rest on at least one experiment, not only on reading.
                refused = True
                out, err = {"error": f"run at least {DIAGNOSER_MIN_RERUNS} rerun_from_stage before submitting: "
                                     "test the change most likely to beat the best run, then submit"}, True
            elif b.name == "submit_diagnosis":
                try:
                    diag = Diagnosis.model_validate(b.input)
                    out, err = {"submitted": True}, False
                except pydantic.ValidationError as e:
                    bad_submits += 1
                    out, err = {"error": f"invalid diagnosis: {e.errors()[:3]}"}, True
            else:
                out, err = tools.call(b.name, b.input or {})
            log.append({"tool": b.name, "input": b.input, "is_error": err, "output": _cut(out, 2000)})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str),
                            "is_error": err})
        messages.append({"role": "user", "content": results})
        if diag is not None or bad_submits >= 3:
            break
    if diag is not None:
        diag = verify(diag, tools.reruns)
    return {"diagnosis": diag, "log": log, "reruns": tools.reruns, "solver_seconds": round(tools.solver_seconds, 2),
            "tool_calls": tools.calls}
