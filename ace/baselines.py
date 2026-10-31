"""Single-agent baselines for the ACE swarm: is the collective actually useful?

Three ways to spend effort *without* the collective, all on exactly the swarm's setup: the same
problems and held-out set, models, tools, limits, diagnoser and memory code. Every baseline is
`ace.run_loop --memory shared+private --agents 1`, so prompts, memory updates and scoring go
through the swarm's own code paths. A lone agent simply has no peers to learn from and nothing
is ever promoted.

  single     one agent, one attempt per problem per generation. The swarm member's loop, alone:
             1/N of the swarm's runs and roughly 1/N of its cost.
  matched    one agent, N attempts per problem per generation. As many runs per generation as an
             N-agent swarm, learning only from its own attempts: same compute, no collective.
  isolated   N independent single agents (separate loops, no shared notes, no promotion). The
             swarm's agents and runs minus every channel between them; also gives the spread
             between individual agents.

Run the swarm with ace.run_loop (default --memory shared+private --agents 6), the baselines
here, then put them on one scale with ace.compare:

    python -m ace.run_loop --out ace_runs/swarm --generations 4 --budget-usd 10
    python -m ace.baselines matched  --swarm-agents 6 --out ace_runs/matched  --generations 4 --budget-usd 10
    python -m ace.baselines isolated --swarm-agents 6 --out ace_runs/isolated --generations 4 --budget-usd 10 --jobs 3
    python -m ace.baselines single   --out ace_runs/single --generations 4 --budget-usd 2
    python -m ace.compare --exp swarm=ace_runs/swarm --exp matched=ace_runs/matched \\
        --exp isolated=ace_runs/isolated --exp single=ace_runs/single

Keep --problems, --holdout, --generations and --top-n identical to the swarm run (ace.compare
checks the setup and refuses mismatched experiments).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ace import config

# run_loop flags passed through unchanged (None = run_loop's own default)
PASSTHROUGH = ("problems", "holdout", "generations", "concurrency", "top_n", "playbook", "ui_runs_dir")


def loop_argv(args, out: Path, attempts: int, budget: float) -> list[str]:
    argv = ["--memory", "shared+private", "--agents", "1", "--attempts", str(attempts),
            "--out", str(out), "--budget-usd", f"{budget:.6g}"]
    for k in PASSTHROUGH:
        v = getattr(args, k)
        if v is not None:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    if args.mock:
        argv.append("--mock")
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["single", "matched", "isolated"])
    ap.add_argument("--swarm-agents", type=int, default=config.AGENTS,
                    help="agents in the swarm being matched (matched: attempts; isolated: loops)")
    ap.add_argument("--budget-usd", type=float, default=config.BUDGET_USD,
                    help="total for this baseline (isolated: split evenly across the loops)")
    ap.add_argument("--jobs", type=int, default=1, help="isolated: how many loops run at the same time")
    ap.add_argument("--out", type=Path, help="output dir (default ace_runs/<timestamp>-<kind>)")
    ap.add_argument("--mock", action="store_true", help="canned LLM responses, no API calls")
    ap.add_argument("--problems")
    ap.add_argument("--holdout")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--top-n", type=int)
    ap.add_argument("--playbook", type=Path)
    ap.add_argument("--ui-runs-dir", type=Path)
    args = ap.parse_args(argv)

    n = args.swarm_agents
    out = args.out or Path("ace_runs") / (time.strftime("%Y%m%d-%H%M%S") + f"-{args.kind}" + ("-mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"kind": args.kind, "swarm_agents": n, "budget_usd": args.budget_usd, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if args.kind in ("single", "matched"):
        from ace.run_loop import main as run_loop

        attempts = 1 if args.kind == "single" else n
        meta.update(loops=["."], attempts=attempts)
        (out / "baseline.json").write_text(json.dumps(meta, indent=1))
        run_loop(loop_argv(args, out, attempts, args.budget_usd))
    else:
        loops = [out / f"agent{k}" for k in range(n)]
        meta.update(loops=[p.name for p in loops], attempts=1)
        (out / "baseline.json").write_text(json.dumps(meta, indent=1))

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            p for p in (str(repo / "src"), str(repo), os.environ.get("PYTHONPATH", "")) if p)}

        def run_one(loop: Path) -> int:
            # A separate process per loop: gmsh and the loop's process pool stay independent.
            cmd = [sys.executable, "-m", "ace.run_loop", *loop_argv(args, loop, 1, args.budget_usd / n)]
            with open(loop.with_suffix(".log"), "w") as log:
                return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode

        with ThreadPoolExecutor(max(1, args.jobs)) as ex:
            codes = list(ex.map(run_one, loops))
        meta["exit_codes"] = codes
        for loop, code in zip(loops, codes):
            print(f"{loop}: exit {code} (log: {loop.with_suffix('.log')})")

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "baseline.json").write_text(json.dumps(meta, indent=1))
    print(f"baseline output: {out}\ncompare with: python -m ace.compare --exp swarm=<swarm dir> --exp {args.kind}={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
