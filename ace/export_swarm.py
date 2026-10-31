"""Export a trained swarm: copy a finished --deep run's final learned state into a small folder.

    python -m ace.export_swarm ace_runs/<run> <out_dir>

A run folder holds every generation's meshes, traces and reports (gigabytes). What an agent has
*learned* is much smaller: the shared playbook, its private notes, its persona, and the last
feedback it got (teacher, own reflection, its last run in numbers). That is everything
ace.process_brief.deep_context needs to rebuild the agent's context, so it is all that
ace.infer needs to run the trained agents on a new problem.

Works for swarm runs (ace.run_loop --deep, agents a0, a1, ...) and single-agent runs
(ace.single_loop, agent "big").

Output:
  manifest.json          run name, final generation, tasks trained on, agents, models, git commit
  playbook.json          the final shared playbook (per-task scope: playbooks/<task>.json instead)
  private/<agent>.json   each agent's final private notes
  personas.json          {agent: persona}, if the run used personas
  state.json             what the context is built from besides the notes: each agent's last teacher
                         feedback, own reflection and last-run summary; each task's last teacher
                         feedback (setup warnings, contested regions); the last highlights
  context/<agent>.md     the context each agent gets on a NEW problem (exactly what ace.infer uses)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from ace import config, process_brief
from ace.memory import agent_of
from ace.playbook import Playbook

BIG = "big"     # the single-agent baseline's name (ace.single_loop.AGENT)


def _gen_of(path: Path) -> int:
    return int(re.search(r"_gen(\d+)\.json$", path.name).group(1))


def final_generation(run: Path) -> int:
    """The last generation whose notes were saved. The loop saves the notes a generation starts
    with, and after the last learning step saves the next generation's, so this is the most
    trained state, even when that generation's runs did not finish."""
    gens = [_gen_of(p) for p in (run / "private").glob("agent*_gen*.json")]
    if not gens:
        raise SystemExit(f"{run}: no private/agent*_gen*.json: not a shared+private (--deep) run?")
    return max(gens)


def is_single(run: Path) -> bool:
    return any(run.glob(f"gen*/context/{BIG}.md"))


def _gen_dirs_desc(run: Path, below: int) -> list[Path]:
    """gen<g> folders with g < below, newest first: where the last feedback before `below` is."""
    dirs = [d for d in run.glob("gen*") if d.is_dir() and re.fullmatch(r"gen\d+", d.name)]
    return sorted((d for d in dirs if int(d.name[3:]) < below), key=lambda d: -int(d.name[3:]))


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect_state(run: Path, g: int, single: bool) -> dict:
    """Rebuild, from the files the loop wrote, the in-memory state run_loop (or single_loop) holds
    when it builds generation g's contexts: for each agent its LAST teacher feedback, its own last
    reflection and a summary of its last taught run (newest generation first, as the loop's
    state is updated in place each generation); each task's last teacher feedback; and the
    highlights of the last generation that was taught."""
    from ace.run_loop import _run_summary     # the loop's own summary, so the context matches

    agents: dict[str, dict] = {}
    teacher: dict[str, dict] = {}
    gdirs = _gen_dirs_desc(run, g)
    taught = next((d for d in gdirs if any(d.glob("teacher_*.json"))), None)
    # the highlights the next context shows are those of the newest taught generation only
    highlights = [{**h, "task": tf.stem.removeprefix("teacher_")}
                  for tf in (sorted(taught.glob("teacher_*.json")) if taught else [])
                  for h in json.loads(tf.read_text()).get("highlights") or []]
    for gdir in gdirs:
        for tf in sorted(gdir.glob("teacher_*.json")):
            task = tf.stem.removeprefix("teacher_")
            fb = json.loads(tf.read_text())
            teacher.setdefault(task, fb)          # newest first: keep the first one seen
            for af in fb.get("per_agent") or []:
                k = agent_of(af.get("run_id") or "")
                a = BIG if single else (f"a{k}" if k is not None else af.get("agent"))
                # single agent: one state per task (single_loop keys its state by task)
                slot = agents.setdefault(a, {}).setdefault(task, {}) if single else agents.setdefault(a, {})
                if "feedback" in slot:
                    continue                      # a newer generation already gave this one
                run_id = af.get("run_id")
                slot["feedback"] = {**af, "agent": a, "task": task}
                refl = _read(gdir / "reflections" / f"{run_id}.json")
                if refl is not None:
                    slot["reflection"] = {**refl, "run_id": run_id, "task": task}
                rec = _read(gdir / "runs" / f"{run_id}.json")
                if rec is not None:
                    slot["last_run"] = _run_summary(rec, _read(gdir / "reports" / f"{run_id}.json"))
    return {"agents": agents, "teacher": teacher, "highlights": highlights}


def load_notes(swarm: Path) -> tuple[Playbook | dict[str, Playbook], dict[str, Playbook]]:
    """(shared playbook, or {task: playbook} for per-task scope; {agent: private notes})."""
    if (swarm / "playbook.json").exists():
        shared = Playbook.load(swarm / "playbook.json")
    else:
        shared = {p.stem: Playbook.load(p) for p in sorted((swarm / "playbooks").glob("*.json"))}
    private = {p.stem: Playbook.load(p) for p in sorted((swarm / "private").glob("*.json"))}
    return shared, private


def agent_context(swarm: Path, agent: str, problem: str | None) -> str:
    """The context block agent `agent` of an exported swarm sees on `problem`, built as training
    builds it. On a task it trained on, it also gets that task's setup warnings and contested
    regions; on any other problem it is the held-out context (task None), as training gives an
    agent on a held-out problem. No peers' runs: at inference nobody ran before it."""
    manifest = json.loads((swarm / "manifest.json").read_text())
    state = json.loads((swarm / "state.json").read_text())
    shared, private = load_notes(swarm)
    trained = problem in manifest["tasks"]
    task = problem if trained else None
    if isinstance(shared, dict):                  # per-task scope: that task's playbook, else all of them
        shared = shared.get(problem) or Playbook(bullets=[b for pb in shared.values() for b in pb.bullets])
    notes = private[agent]
    me = state["agents"].get(agent, {})
    if manifest["kind"] == "single":
        # single_context over this one problem, with its own last feedback when it trained on it
        label = problem or "(the new problem)"
        d = (me.get(problem) or {}) if trained else {}
        task_feedback = state["teacher"].get(problem) if trained else None
        return process_brief.single_context([label], shared, notes, {label: {**d, "task_feedback": task_feedback}})
    personas = _read(swarm / "personas.json") or {}
    return process_brief.deep_context(
        agent, task, shared, notes, me.get("feedback"), me.get("reflection"), state["highlights"], "",
        last_run=me.get("last_run"), task_feedback=state["teacher"].get(task) if task else None,
        persona=personas.get(agent))


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(run: Path, out: Path) -> dict:
    g = final_generation(run)
    single = is_single(run)
    out.mkdir(parents=True, exist_ok=True)

    # the shared playbook: one file, or one per task (--shared-scope task)
    if (run / f"playbook_gen{g}.json").exists():
        shutil.copy(run / f"playbook_gen{g}.json", out / "playbook.json")
    else:
        (out / "playbooks").mkdir(exist_ok=True)
        for p in run.glob(f"playbook_*_gen{g}.json"):
            shutil.copy(p, out / "playbooks" / (p.name.removeprefix("playbook_").removesuffix(f"_gen{g}.json") + ".json"))

    # private notes, renamed from agent<k> to the agent's name (a<k>, or big)
    (out / "private").mkdir(exist_ok=True)
    agents = []
    for p in sorted(run.glob(f"private/agent*_gen{g}.json"), key=lambda p: int(re.search(r"agent(\d+)_", p.name).group(1))):
        k = int(re.search(r"agent(\d+)_", p.name).group(1))
        a = BIG if single else f"a{k}"
        shutil.copy(p, out / "private" / f"{a}.json")
        agents.append(a)
    if (run / "personas.json").exists():
        shutil.copy(run / "personas.json", out / "personas.json")

    state = collect_state(run, g, single)
    (out / "state.json").write_text(json.dumps(state, indent=1, default=str))
    tasks = sorted(state["teacher"]) or sorted({t for d in run.glob("gen*") for t in
                                                (_read(d / "groups.json") or {}).values()})
    manifest = {"run": run.resolve().name, "source": str(run.resolve()), "kind": "single" if single else "swarm",
                "generation": g, "tasks": tasks, "agents": agents, "models": dict(config.MODELS),
                "git_commit": git_commit(), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # what each agent will read on a new problem, for a human to read
    (out / "context").mkdir(exist_ok=True)
    for a in agents:
        (out / "context" / f"{a}.md").write_text(agent_context(out, a, None))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a finished run folder, e.g. ace_runs/scale_s8")
    ap.add_argument("out_dir", type=Path, help="where to write the exported swarm")
    args = ap.parse_args(argv)
    m = export(args.run_dir, args.out_dir)
    print(f"exported {m['run']} ({m['kind']}, generation {m['generation']}): agents {', '.join(m['agents'])}; "
          f"trained on {', '.join(m['tasks'])} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
