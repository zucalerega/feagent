"""Dashboard tools for the single analyst agent (``single_agent/``, ``single_agent/README.md``): the training
loop of one arm (Opus or Sonnet), the non-LLM oracle baseline, and the comparison of arms. Group
``single_agent`` (shown on the Agents page, Train tab); outputs under ``sa_runs/``.

The loop has no graceful stop: Kill ends it, and ``--budget-usd`` bounds what a real run spends."""

from __future__ import annotations

import os

from feacore.ui.tasks import Param, Tool, ToolArgsError, register_tool

GROUP = "single_agent"
KEY = ["ANTHROPIC_API_KEY"]
MOCK_DEFAULT = not os.environ.get("ANTHROPIC_API_KEY")   # without a key only mock runs work


def _ids(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _packaged(value: str | None, what: str) -> list[str]:
    """The ids, each one a problem with a design package (single_agent/design/)."""
    from single_agent import workspace

    known = workspace.packaged()
    ids = _ids(value)
    unknown = [p for p in ids if p not in known]
    if unknown:
        raise ToolArgsError([f"{what}: {', '.join(unknown)} has no design package; packaged: {', '.join(known)}"])
    return ids


def _flags(args: dict, names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        v = args.get(n)
        if v is not None and v != "":
            out += [f"--{n.replace('_', '-')}", str(v)]
    return out


def _loop_argv(a: dict) -> list[str]:
    argv = ["--model", a["model"], "--reflector", a["reflector"], "--rules", a["rules"]]
    if a.get("problems"):
        argv += ["--problems", ",".join(_packaged(a["problems"], "problems"))]
    hold = (a.get("holdout") or "").strip()
    if hold.lower() == "none":
        argv += ["--holdout", ""]
    elif hold:
        argv += ["--holdout", ",".join(_packaged(hold, "holdout"))]
    argv += _flags(a, ["generations", "holdout_every", "rulebook", "concurrency", "budget_usd", "node_cap",
                       "max_solves", "max_turns", "out"])
    if a.get("no_learn"):
        argv.append("--no-learn")
    if a.get("mock"):
        argv.append("--mock")
    return argv


LOOP = register_tool(Tool(
    name="single_agent.loop", title="Single analyst agent", group=GROUP, command=["-m", "single_agent.loop"],
    description="One agent (one arm: Opus or Sonnet) does the whole analysis layer from a design package: decides "
                "the supports and loads, meshes the part, runs its own convergence study and answers the review "
                "questions. A deterministic scorecard grades every answer, and the arm learns rules from its "
                "training scorecards (single_agent/README.md).",
    params=[
        Param("model", type="choice", choices=["sonnet", "opus"], default="sonnet", help="the arm"),
        Param("mock", type="bool", default=MOCK_DEFAULT, help="canned model responses: free, real harness and scoring"),
        Param("generations", type="int", default=4, min=1, max=30),
        Param("budget_usd", type="float", default=40.0, min=0, label="budget ($)",
              help="the run stops before a generation that would pass it"),
        Param("problems", help="training problems, comma-separated (default: the four abc_00000003 sets)"),
        Param("holdout", help="holdout problems, comma-separated; 'none' for no holdout (default: three sets)"),
        Param("rules", type="choice", choices=["handbook", "none"], default="handbook",
              help="start from the general handbook, or from nothing"),
        Param("reflector", type="choice", choices=["same", "opus", "sonnet"], default="same", advanced=True,
              help="model of the learning step"),
        Param("no_learn", type="bool", default=False, advanced=True, flag="--no-learn",
              help="handbook only, no learned rules"),
        Param("rulebook", type="path", advanced=True, help="start from another run's rulebook JSON"),
        Param("holdout_every", type="int", advanced=True, min=0, help="0: holdout in the first and last generation"),
        Param("concurrency", type="int", advanced=True, min=1, max=8),
        Param("node_cap", type="int", advanced=True, min=1000),
        Param("max_solves", type="int", advanced=True, min=1),
        Param("max_turns", type="int", advanced=True, min=1),
        Param("out", type="path", advanced=True, help="output folder (default: sa_runs/loop-<task>)"),
    ],
    out_param="out", out_root="sa_runs", requires_env=KEY, unless_param="mock", build=_loop_argv,
))


def _oracle_argv(a: dict) -> list[str]:
    return [*_packaged(a.get("problems"), "problems"), *_flags(a, ["h", "levels", "factor", "out"])]


ORACLE = register_tool(Tool(
    name="single_agent.oracle", title="Oracle baseline (no LLM)", group=GROUP, command=["-m", "single_agent.oracle"],
    description="The non-LLM baseline: each problem's reference setup and a scripted convergence study, graded by "
                "the same scorecard as the agents.",
    params=[
        Param("problems", help="comma-separated (default: every packaged problem)"),
        Param("levels", type="int", default=3, min=2, max=5, help="meshes in the scripted study"),
        Param("h", type="float", default=0.006, min=0.0005, advanced=True, help="coarsest mesh size (m)"),
        Param("factor", type="float", default=1.4, min=1.1, max=3, advanced=True, help="refinement factor"),
        Param("out", type="path", advanced=True, help="output folder (default: sa_runs/oracle-<task>)"),
    ],
    out_param="out", out_root="sa_runs", build=_oracle_argv,
))


def _compare_argv(a: dict) -> list[str]:
    runs = [a[k] for k in ("run1", "run2", "run3") if a.get(k)]
    return [*runs, *_flags(a, ["oracle", "out"])]


COMPARE = register_tool(Tool(
    name="single_agent.compare", title="Compare analyst arms", group=GROUP, command=["-m", "single_agent.compare"],
    description="Score tables of two or three arms (and the oracle) by generation, on the training and the holdout "
                "problems.",
    params=[
        Param("run1", type="path", required=True, help="a single_agent.loop output folder"),
        Param("run2", type="path", help="another arm's output folder"),
        Param("run3", type="path", advanced=True),
        Param("oracle", type="path", help="a single_agent.oracle output folder"),
        Param("out", type="path", advanced=True, help="output folder (default: sa_runs/compare-<task>)"),
    ],
    out_param="out", out_root="sa_runs", build=_compare_argv,
))
