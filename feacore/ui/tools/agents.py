"""Agents section: the ACE loops (``ace/``) as dashboard tools (contract: feacore/ui/tasks.py).

    ace.swarm         python -m ace.run_loop        multi-agent (swarm) training
    ace.single        python -m ace.single_loop     the single-agent baseline
    ace.compare       python -m ace.compare         swarm vs single agent: compare.md, compare.png
    ace.report        python -m ace.report          one run as a self-contained report.html
    ace.export_swarm  python -m ace.export_swarm    what a trained swarm learned, for inference
    ace.infer         python -m ace.infer           a trained swarm on any registered problem

Outputs go to ace_runs/<tool>-<task id> (exported swarms: swarms/...) unless the form names one. The task
runner sets FEACORE_RUNS_DIR, so the loops' run-store bridge (ace/ui_bridge.py) records every solve and
agent run for the live #/swarm view. The training loops stop gracefully through <out>/STOP (before the
next generation). Real runs need ANTHROPIC_API_KEY in the server's environment; mock runs are free.

Settings ACE reads from its environment rather than its command line become environment variables of
the task (Tool.build_env): ACE_SCORE=field (field scoring; --deep and the single-agent baseline always
use it) and ACE_HARNESS=1 (the feagent mesh harness in the agent's 3D tool results).
"""

from __future__ import annotations

import os
import re

from feacore.ui.tasks import Param, Tool, ToolArgsError, register_tool

GROUP = "agents"
KEY = ["ANTHROPIC_API_KEY"]
# Without a key only mock runs work, so the forms start in the state that runs.
MOCK_DEFAULT = not os.environ.get("ANTHROPIC_API_KEY")
_AGENT_LIST = re.compile(r"^a\d+(,a\d+)*$")


def _ids(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _check_tasks(value: str | None, what: str) -> None:
    """The loops can only train on problems in their task catalogue (ace/config.py TASKS): each needs a
    scored quantity and a node cap. Any registered problem works for inference (ace.infer)."""
    from ace import config

    unknown = [p for p in _ids(value) if p not in config.TASKS]
    if unknown:
        raise ToolArgsError([f"{what}: {', '.join(unknown)} not in the loop's task catalogue (ace/config.py "
                             f"TASKS); ace.infer runs a trained swarm on any problem"])


def _holdout(value: str | None) -> str | None:
    """None = the suite's held-out problems, "" = none ('none' in the form), else the checked list."""
    if value is None or value == "":
        return None
    if value.strip().lower() == "none":
        return ""
    _check_tasks(value, "holdout")
    return ",".join(_ids(value))


def _agent_list(value: str | None, what: str, allow_all: bool = False) -> None:
    if value and not (_AGENT_LIST.match(value.replace(" ", "")) or (allow_all and value == "all")):
        raise ToolArgsError([f"{what}: agent names like a0,a3{' or all' if allow_all else ''}"])


def _flags(args: dict, names: list[str]) -> list[str]:
    """``--name value`` for every set argument (the bare flag for a true bool), in this order."""
    argv: list[str] = []
    for n in names:
        v = args.get(n)
        if v is None or v is False or v == "":
            continue
        flag = "--" + n.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    return argv


# --------------------------------------------------------------------------- swarm training (run_loop)

# run_loop's own defaults for the choice settings: left off the command line, so it reads as typed by hand
SWARM_CLI_DEFAULTS = {"suite": "fea", "memory": "shared+private", "teacher": "rerun", "shared_scope": "all",
                      "shared_bar": "any", "statement": "bcs"}


def swarm_argv(a: dict) -> list[str]:
    _check_tasks(a.get("problems"), "problems")
    hold = _holdout(a.get("holdout"))
    _agent_list(a.get("init_agents"), "init agents")
    _agent_list(a.get("personas"), "personas", allow_all=True)
    deep = bool(a.get("deep"))
    if deep and a.get("memory") == "shared":
        raise ToolArgsError(["deep needs memory shared+private (persistent agents with their own notes)"])
    a = {k: v for k, v in a.items() if SWARM_CLI_DEFAULTS.get(k) != v}
    argv = _flags(a, ["mock", "suite", "problems"])
    if hold is not None:
        argv += ["--holdout", hold]
    argv += _flags(a, ["generations", "agents", "concurrency", "budget_usd", "memory", "runs_per_problem",
                       "top_n", "playbook", "teacher", "shared_scope", "shared_bar", "personas", "persona_count",
                       "statement", "evidence", "evidence_score", "init_from", "init_agents", "pre_teach", "deep"])
    if deep or a.get("split"):                   # --deep needs --split: set it rather than refuse
        argv.append("--split")
    if a.get("peers") in ("on", "off"):
        argv.append("--peers" if a["peers"] == "on" else "--no-peers")
    return argv + ["--out", str(a["out"])]


def ace_env(a: dict) -> dict[str, str]:
    env = {}
    if a.get("score") == "field" or a.get("deep"):   # --deep refuses to run without field scoring
        env["ACE_SCORE"] = "field"
    if a.get("harness"):
        env["ACE_HARNESS"] = "1"
    return env


SWARM = register_tool(Tool(
    name="ace.swarm", title="Swarm training", group=GROUP, command=["-m", "ace.run_loop"],
    description="Train a population of agents (ACE, ace/README.md). Each generation every agent meshes and "
                "solves its task, the runs are scored, and a teacher, the agents' own reflections and an "
                "organizer rewrite the agents' notes for the next generation.",
    params=[
        Param("mock", type="bool", default=MOCK_DEFAULT, label="mock LLM",
              help="canned LLM responses, no API calls: free, for trying the loop"),
        Param("problems", type="problems", label="train problems",
              help="taught on; comma-separated; empty = the suite's"),
        Param("holdout", label="held-out problems",
              help="evaluated every generation, never taught; 'none' = none; empty = the suite's"),
        Param("generations", type="int", default=4, min=1, max=200, help="generation 0 starts from empty notes"),
        Param("agents", type="int", default=6, min=1, max=64, help="persistent agents (shared+private memory)"),
        Param("concurrency", type="int", default=4, min=1, max=64, help="agent runs at a time"),
        Param("budget_usd", type="float", default=10.0, min=0, max=10000, label="budget ($)",
              help="checked before each agent run and after each teacher call"),
        Param("deep", type="bool", default=False, label="deep teaching",
              help="run reports, deep teacher, self-reflection, organizer; implies split and field scoring"),
        Param("score", type="choice", default="point", choices=["point", "field"], label="scoring",
              help="point: the scored quantity vs its reference; field: the whole stress field vs a fine "
                   "reference (needs it built: python -m ace.field_error --precompute <problem>)"),
        Param("split", type="bool", default=False,
              help="each generation, split the agents into one group per train problem"),
        Param("persona_count", type="int", min=0, max=64, label="personas (first N agents)",
              help="give agents a0..a(N-1) a fixed exploration style", advanced=True),
        Param("personas", label="personas (by name)", help="all, or agents like a0,a3,a5", advanced=True),
        Param("init_from", type="path", label="start from swarm",
              help="an exported swarm, e.g. results/trained_swarms/div_s8 (post-training)", advanced=True),
        Param("init_agents", label="its agents", help="best first, e.g. a6,a3,a1,a7; empty = all", advanced=True),
        Param("pre_teach", type="bool", default=False, label="pre-teach",
              help="deep, one train problem: the teacher adapts the notes to the task before generation 0",
              advanced=True),
        Param("suite", type="choice", default="fea", choices=["fea", "cfd", "cfd3d"],
              help="default problems: elasticity, 2D or 3D laminar flow (OpenFOAM)", advanced=True),
        Param("memory", type="choice", default="shared+private", choices=["shared+private", "shared"],
              help="shared playbook plus each agent's own notes, or one shared playbook", advanced=True),
        Param("runs_per_problem", type="int", min=1, max=64, help="shared memory only (anonymous runs)",
              advanced=True),
        Param("teacher", type="choice", default="rerun", choices=["rerun", "inspect", "both"],
              help="counterfactual reruns, a detailed inspection (field scoring), or both", advanced=True),
        Param("peers", type="choice", default="auto", choices=["auto", "on", "off"],
              help="show agents what the others did on their task (auto: on with split)", advanced=True),
        Param("shared_scope", type="choice", default="all", choices=["all", "task"],
              help="with split: one shared playbook, or one per task", advanced=True),
        Param("shared_bar", type="choice", default="any", choices=["any", "counterfactual"],
              help="which teacher lessons may enter the shared playbook", advanced=True),
        Param("top_n", type="int", min=1, max=500, help="shared bullets shown to an agent (default 30)",
              advanced=True),
        Param("playbook", type="path", help="start from this playbook JSON", advanced=True),
        Param("statement", type="choice", default="bcs", choices=["bcs", "requirements"],
              help="the problem statement with its BCs, or engineering requirements only", advanced=True),
        Param("evidence", type="path", help="physical test data: a test spec JSON or a folder of them",
              advanced=True),
        Param("evidence_score", type="bool", default=False, help="score against the test data", advanced=True),
        Param("harness", type="bool", default=False, label="mesh harness",
              help="the feagent mesh harness in the agent's 3D tool results (ACE_HARNESS=1)", advanced=True),
        Param("out", type="path", label="output folder", help="default ace_runs/swarm-<task id>", advanced=True),
    ],
    out_param="out", out_root="ace_runs", stop_file="STOP", requires_env=KEY, unless_param="mock",
    build=swarm_argv, build_env=ace_env))


# --------------------------------------------------------------------------- single-agent baseline


def single_argv(a: dict) -> list[str]:
    _check_tasks(a.get("problems"), "problems")
    return _flags(a, ["mock", "problems", "generations", "budget_usd", "swarm_dir", "swarm_agents",
                      "solves_per_task", "turns", "concurrency", "gen_seconds", "evidence", "evidence_score"]) \
        + ["--out", str(a["out"])]


SINGLE = register_tool(Tool(
    name="ace.single", title="Single-agent training", group=GROUP, command=["-m", "ace.single_loop"],
    description="The swarm's baseline: one agent does every task each generation in one conversation, with "
                "the whole swarm's solve and turn budget, and learns through the same teacher, reflection and "
                "organizer. Always field-scored (ACE_SCORE=field).",
    params=[
        Param("mock", type="bool", default=MOCK_DEFAULT, label="mock LLM",
              help="canned LLM responses, no API calls: free"),
        Param("problems", type="problems", label="tasks",
              help="all solid or all flow; empty = cooks_membrane,nafems_le11_axisym"),
        Param("generations", type="int", default=4, min=1, max=200),
        Param("budget_usd", type="float", default=10.0, min=0, max=10000, label="budget ($)"),
        Param("swarm_dir", type="path", label="swarm run to match",
              help="a --deep swarm run: its time per generation sets this agent's; compared at the end"),
        Param("swarm_agents", type="int", min=1, max=64, help="size of the swarm to match (default 8)",
              advanced=True),
        Param("solves_per_task", type="int", min=1, max=500, advanced=True),
        Param("turns", type="int", min=1, max=10000, help="model calls per generation", advanced=True),
        Param("concurrency", type="int", min=1, max=64, help="scoring processes", advanced=True),
        Param("gen_seconds", type="float", min=1, max=86400, help="time budget per generation without a swarm run",
              advanced=True),
        Param("evidence", type="path", help="physical test data (as the swarm)", advanced=True),
        Param("evidence_score", type="bool", default=False, advanced=True),
        Param("harness", type="bool", default=False, label="mesh harness", help="ACE_HARNESS=1", advanced=True),
        Param("out", type="path", label="output folder", help="default ace_runs/single-<task id>", advanced=True),
    ],
    out_param="out", out_root="ace_runs", stop_file="STOP", requires_env=KEY, unless_param="mock",
    build=single_argv, build_env=lambda a: {"ACE_SCORE": "field", **ace_env({"harness": a.get("harness")})}))


# --------------------------------------------------------------------------- evaluation and inference


COMPARE = register_tool(Tool(
    name="ace.compare", title="Compare swarm and single agent", group=GROUP, command=["-m", "ace.compare"],
    description="Base score per task per generation of a --deep swarm run and a single-agent run, against "
                "generations, cumulative $ and wall time (compare.md, compare.png).",
    params=[
        Param("swarm_dir", type="path", required=True, flag="", label="swarm run"),
        Param("single_dir", type="path", required=True, flag="", label="single-agent run"),
        Param("out", type="path", label="output folder", help="default ace_runs/compare-<task id>", advanced=True),
    ],
    out_param="out", out_root="ace_runs", timeout=1800))

REPORT = register_tool(Tool(
    name="ace.report", title="Run report", group=GROUP, command=["-m", "ace.report"],
    description="The whole run as one self-contained page: every stage, prompt and number (<run>/report.html).",
    params=[Param("run_dir", type="path", required=True, flag="", label="run folder")],
    timeout=1800))

EXPORT = register_tool(Tool(
    name="ace.export_swarm", title="Export a trained swarm", group=GROUP, command=["-m", "ace.export_swarm"],
    description="Copy what a finished --deep (or single-agent) run learned into a small folder: the shared "
                "playbook, each agent's notes, personas and last feedback, and the context each agent gets on "
                "a new problem. No meshes, no traces.",
    params=[
        Param("run_dir", type="path", required=True, flag="", label="run folder"),
        Param("out_dir", type="path", flag="", label="export to", help="default swarms/<run name>"),
    ],
    out_param="out_dir", out_root="swarms", timeout=600))


def infer_argv(a: dict) -> list[str]:
    _agent_list(a.get("agents"), "agents", allow_all=True)
    return _flags(a, ["swarm", "problem", "agents", "mock"]) + ["--out", str(a["out"])]


INFER = register_tool(Tool(
    name="ace.infer", title="Run a trained swarm", group=GROUP, command=["-m", "ace.infer"],
    description="Run an exported swarm's agents on any registered problem (benchmarks, gold parts, your own "
                "CAD), one process per agent, with the context training gave them. Nothing learns. Writes "
                "each agent's prediction and the consensus between them (results.json, summary.md).",
    params=[
        Param("swarm", type="path", required=True, label="exported swarm",
              help="e.g. results/trained_swarms/div_s8"),
        Param("problem", type="problem", required=True),
        Param("agents", default="all", help="all, or agents like a0,a3"),
        Param("mock", type="bool", default=MOCK_DEFAULT, label="mock LLM", help="canned LLM responses: free"),
        Param("out", type="path", label="output folder", help="default ace_runs/infer-<task id>", advanced=True),
    ],
    out_param="out", out_root="ace_runs", requires_env=KEY, unless_param="mock", build=infer_argv))
