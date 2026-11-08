"""One analysis by one agent: a tool-use conversation over the harness, in a fresh workspace.

    run_problem(problem_id, root, rulebook_text, llm, run_id) -> run record (JSON-able)

The record holds the trace (text, reasoning summaries, tool calls and results), the session state (meshes,
solves, studies, the report), the token usage and the wall time. Scoring is single_agent.evaluate's job.
The model behind the agent is whatever `configure` registered for the role "sa_agent" in ace.config (the
loop sets it per arm); ace.llm.LLM makes the calls (adaptive thinking with summaries, effort, prompt caching,
the server-side refusal fallback), and --mock swaps in single_agent.mock.MockLLM.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from single_agent import config, prompts, workspace
from single_agent.harness import TOOLS, Session

ROLE_AGENT, ROLE_REFLECT = "sa_agent", "sa_reflect"


def configure(model: str, reflect_model: str | None = None) -> None:
    """Register the arm's models for ace.llm (its dicts are shared module state, per process)."""
    from ace import config as ace_config
    from ace import llm as ace_llm
    for role, key, m in ((ROLE_AGENT, "agent", model), (ROLE_REFLECT, "reflect", reflect_model or model)):
        ace_config.MODELS[role] = m
        ace_config.MAX_TOKENS[role] = config.MAX_TOKENS[key]
        ace_config.EFFORT[role] = config.EFFORT[key]
        ace_llm.TIMEOUT[role] = config.TIMEOUT[key]


def _to_trace(block, turn: int) -> dict | None:
    from ace.agent import _block_to_trace
    t = _block_to_trace(block)
    return dict(t, turn=turn) if t else None


def run_problem(problem_id: str, root: str | Path, rulebook_text: str, llm, run_id: str,
                limits: dict | None = None, max_turns: int | None = None, generation: int | None = None) -> dict:
    from ace.agent import tool_result_content
    max_turns = max_turns or config.MAX_TURNS
    ws = workspace.make(problem_id, root)
    session = Session(ws, limits=limits)
    package = (ws / "package.md").read_text()
    n_q = len(workspace.questions(problem_id))
    system = prompts.system_prompt(rulebook_text)
    messages: list = [{"role": "user", "content": prompts.task_message(package, ws.name, session.limits, max_turns,
                                                                       n_q)}]
    trace: list[dict] = []
    usage: list[dict] = []
    stop, error, nudges = "max_turns", None, 0
    t0 = time.time()
    try:
        for turn in range(max_turns):
            resp, rec = llm.agent_turn(system, messages, TOOLS, run_id=run_id, role=ROLE_AGENT)
            usage.append(rec)
            messages.append({"role": "assistant", "content": resp.content})   # unchanged: thinking blocks too
            trace += [t for b in resp.content if (t := _to_trace(b, turn))]
            calls = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not calls:
                if session.finished or nudges >= config.MAX_NUDGES or turn == max_turns - 1:
                    stop = resp.stop_reason
                    break
                nudges += 1
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": "not run: the turn was cut off",
                     "is_error": True} for b in calls] + [{"type": "text", "text": prompts.NUDGE}]})
                trace.append({"type": "nudge", "turn": turn, "stop_reason": resp.stop_reason})
                continue
            results = []
            for b in calls:
                out, is_err = session.call(b.name, b.input or {})
                trace.append({"type": "tool_result", "id": b.id, "name": b.name, "result": out, "is_error": is_err,
                              "turn": turn})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": tool_result_content(out),
                                "is_error": is_err})
            left = max_turns - turn - 1
            if left in (8, 3) and not session.finished:
                results.append({"type": "text", "text": prompts.turns_warning(left)})
            messages.append({"role": "user", "content": results})
            if session.finished:
                stop = "submitted"
                break
    except Exception as e:  # noqa: BLE001  a crash ends this analysis, keeps what was done
        error = f"{type(e).__name__}: {e}"
        stop = "error"
        trace.append({"type": "error", "text": traceback.format_exc()[-3000:]})
    return {
        "problem_id": problem_id, "label": ws.name, "run_id": run_id, "generation": generation,
        "workspace": str(ws), "stop": stop, "turns": len(usage), "nudges": nudges, "error": error,
        "trace": trace, "session": session.state, "usage": usage, "wall_s": round(time.time() - t0, 1),
    }
