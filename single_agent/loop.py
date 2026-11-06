"""The single-agent loop: analyse -> score -> learn general rules -> analyse again.

    python -m single_agent.loop --mock --generations 2                        # free, end to end
    python -m single_agent.loop --model sonnet --generations 4 --budget-usd 40 --out sa_runs/sonnet
    python -m single_agent.loop --model opus   --generations 4 --budget-usd 60 --out sa_runs/opus
    python -m single_agent.compare sa_runs/sonnet sa_runs/opus

One ARM = one model doing everything (the analyst and, by default, its own learning step). Each generation:
  1. every training problem (and, in the first and last generation, every holdout problem) is analysed in a
     fresh workspace by the agent with the current rulebook (single_agent.agent), problems in parallel;
  2. each analysis is scored (single_agent.evaluate);
  3. the learning step reads the training analyses and their scorecards and updates the learned rules
     (single_agent.reflect, single_agent.rulebook). Holdout problems are never reflected on.

Ablations: --rules none (no handbook: the model's own judgement plus what it learns), --no-learn (the
handbook only, no learning step), --reflector opus (a Sonnet analyst with an Opus teacher).

Outputs in --out:
  config.json, cost.jsonl, generations.jsonl (one line per generation), summary.md
  rulebook_gen<g>.json (the rulebook generation g used), diff_gen<g>.md (what learning changed after g)
  gen<g>/runs/<problem>.json (run records), gen<g>/cards/<problem>.json (scorecards)
  gen<g>/reflection.json, gen<g>/reflection_prompt.md
  work/gen<g>/<label>/ (the workspaces: meshes, pictures, session.json)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from single_agent import config, workspace
from single_agent.rulebook import Rulebook, diff_markdown


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=config.HERE).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _worker(job: dict) -> dict:
    """One analysis and its scorecard, in a child process (gmsh is process-global)."""
    from single_agent.agent import configure, run_problem
    from single_agent.evaluate import evaluate
    configure(job["model"], job["reflect_model"])
    if job["mock"]:
        from single_agent.mock import MockLLM
        llm = MockLLM()
    else:
        from ace.llm import LLM
        llm = LLM()
    out = Path(job["out"])
    try:
        run = run_problem(job["problem"], out / "work" / f"gen{job['gen']}", job["rulebook_text"], llm,
                          run_id=f"g{job['gen']}-{job['problem']}", limits=job["limits"], max_turns=job["max_turns"],
                          generation=job["gen"])
    except Exception as e:  # noqa: BLE001
        run = {"problem_id": job["problem"], "run_id": f"g{job['gen']}-{job['problem']}", "generation": job["gen"],
               "error": f"{type(e).__name__}: {e}", "trace": [{"type": "error", "text": traceback.format_exc()}],
               "session": {}, "usage": [], "turns": 0}
    try:
        card = evaluate(job["problem"], run)
    except Exception as e:  # noqa: BLE001
        card = {"problem_id": job["problem"], "score": 0.0, "failed": f"evaluation crashed: {type(e).__name__}: {e}",
                "parts": {k: 0.0 for k in config.WEIGHTS}, "trace": traceback.format_exc()}
    card["split"] = job["split"]
    gdir = out / f"gen{job['gen']}"
    (gdir / "runs").mkdir(parents=True, exist_ok=True)
    (gdir / "cards").mkdir(parents=True, exist_ok=True)
    (gdir / "runs" / f"{job['problem']}.json").write_text(json.dumps(run, default=float))
    (gdir / "cards" / f"{job['problem']}.json").write_text(json.dumps(card, indent=1, default=float))
    return {"problem": job["problem"], "card": card, "usage": run.get("usage") or []}


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def generation_row(gen: int, cards: list[dict], spent: float) -> dict:
    row = {"generation": gen, "usd_total": round(spent, 4)}
    for split in ("train", "holdout"):
        cs = [c for c in cards if c.get("split") == split]
        if not cs:
            continue
        row[split] = {"n": len(cs), "score": _mean([c.get("score", 0.0) for c in cs]),
                      **{k: _mean([c["parts"].get(k, 0.0) for c in cs]) for k in config.WEIGHTS},
                      "setup_equivalent": sum(bool((c.get("setup") or {}).get("equivalent")) for c in cs),
                      "submitted": sum(bool(c.get("submitted")) for c in cs)}
    return row


def write_summary(out: Path, args, rows: list[dict], cards_by_gen: dict) -> None:
    from single_agent.evaluate import summary_line
    lines = [f"# Single-agent loop: {args.model} ({config.ARMS.get(args.model, args.model)})", "",
             f"rules: {args.rules}, learning: {'off' if args.no_learn else 'on'}, reflector: {args.reflector}, "
             f"train {args.problems}, holdout {args.holdout}", "",
             "| gen | split | n | score | setup | answers | calibration | study | setups right | submitted | $ total |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for split in ("train", "holdout"):
            if split in r:
                s = r[split]
                lines.append(f"| {r['generation']} | {split} | {s['n']} | {s['score']:.3f} | {s['setup']:.2f} | "
                             f"{s['answers']:.2f} | {s['calibration']:.2f} | {s['study']:.2f} | "
                             f"{s['setup_equivalent']}/{s['n']} | {s['submitted']}/{s['n']} | {r['usd_total']:.2f} |")
    for g, cards in sorted(cards_by_gen.items()):
        lines += ["", f"## Generation {g}", ""]
        lines += [f"- [{c.get('split')}] {summary_line(c) if 'cost' in c else c.get('failed')}" for c in cards]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sonnet", help="arm: opus | sonnet | a model id")
    ap.add_argument("--reflector", default="same", help="model of the learning step: same | opus | sonnet | id")
    ap.add_argument("--problems", default=",".join(config.TRAIN), help="training problems (comma-separated)")
    ap.add_argument("--holdout", default=",".join(config.HOLDOUT), help="holdout problems ('' for none)")
    ap.add_argument("--generations", type=int, default=config.GENERATIONS)
    ap.add_argument("--holdout-every", type=int, default=config.HOLDOUT_EVERY)
    ap.add_argument("--rules", choices=("handbook", "none"), default="handbook")
    ap.add_argument("--rulebook", help="start from this rulebook JSON (e.g. another run's rulebook_gen3.json)")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    ap.add_argument("--budget-usd", type=float, default=config.BUDGET_USD)
    ap.add_argument("--node-cap", type=int, default=config.NODE_CAP)
    ap.add_argument("--max-solves", type=int, default=config.MAX_SOLVES)
    ap.add_argument("--max-turns", type=int, default=config.MAX_TURNS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", action="store_true", help="canned model responses: free, for testing the plumbing")
    args = ap.parse_args(argv)

    from ace.cost import CostLog
    model = config.ARMS.get(args.model, args.model)
    reflect_model = model if args.reflector == "same" else config.ARMS.get(args.reflector, args.reflector)
    train = [p for p in args.problems.split(",") if p]
    holdout = [p for p in args.holdout.split(",") if p]
    known = set(workspace.packaged())
    missing = [p for p in train + holdout if p not in known]
    if missing:
        raise SystemExit(f"no design package for {missing}; packaged: {sorted(known)}")
    out = Path(args.out or f"sa_runs/{args.model}{'-mock' if args.mock else ''}-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    limits = {"node_cap": args.node_cap, "max_solves": args.max_solves, "max_views": config.MAX_VIEWS}
    (out / "config.json").write_text(json.dumps({
        "model": model, "reflect_model": reflect_model, "arm": args.model, "rules": args.rules,
        "learning": not args.no_learn, "train": train, "holdout": holdout, "generations": args.generations,
        "limits": limits, "max_turns": args.max_turns, "effort": config.EFFORT, "mock": args.mock,
        "git_commit": _git_commit(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    rb = Rulebook.load(args.rulebook) if args.rulebook else Rulebook.initial(args.rules)
    cost = CostLog(out / "cost.jsonl", args.budget_usd)
    rows, cards_by_gen = [], {}
    if args.mock:
        from single_agent.mock import MockLLM
        llm = MockLLM()
    else:
        from ace.llm import LLM
        llm = LLM()
    from single_agent.agent import configure
    configure(model, reflect_model)

    ctx = mp.get_context("spawn")
    for gen in range(args.generations):
        if cost.over_budget():
            print(f"budget reached (${cost.total:.2f}); stopping before generation {gen}")
            break
        rb.save(out / f"rulebook_gen{gen}.json")
        last = gen == args.generations - 1
        eval_holdout = gen == 0 or last or (args.holdout_every and gen % args.holdout_every == 0)
        jobs = [{"problem": p, "split": "train"} for p in train] + \
               ([{"problem": p, "split": "holdout"} for p in holdout] if eval_holdout else [])
        common = {"gen": gen, "model": model, "reflect_model": reflect_model, "mock": args.mock, "out": str(out),
                  "rulebook_text": rb.render(), "limits": limits, "max_turns": args.max_turns}
        print(f"== generation {gen}: {len(jobs)} analyses ({sum(j['split'] == 'train' for j in jobs)} train), "
              f"{len(rb.learned)} learned rules, ${cost.total:.2f} spent", flush=True)
        cards = []
        with ProcessPoolExecutor(max_workers=args.concurrency, mp_context=ctx) as pool:
            pending, queue = set(), list(jobs)
            while queue or pending:
                while queue and len(pending) < args.concurrency and not cost.over_budget():
                    pending.add(pool.submit(_worker, {**common, **queue.pop(0)}))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        r = f.result()
                    except Exception as e:  # noqa: BLE001  a worker that died
                        print(f"   worker failed: {type(e).__name__}: {e}")
                        continue
                    cost.add(r["usage"])
                    cards.append(r["card"])
                    c = r["card"]
                    print(f"   [{c.get('split')}] {r['problem']}: score {c.get('score', 0):.3f} parts {c.get('parts')} "
                          f"{c.get('failed') or ''} (${cost.total:.2f})", flush=True)
            if queue:
                print(f"   budget reached: {len(queue)} analyses not started")
        cards_by_gen[gen] = cards
        rows.append(generation_row(gen, cards, cost.total))
        with (out / "generations.jsonl").open("a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        write_summary(out, args, rows, cards_by_gen)
        if args.no_learn or last or cost.over_budget():
            continue
        # ---- learning step on the training analyses
        from single_agent.reflect import reflect
        items = []
        for c in cards:
            if c.get("split") != "train":
                continue
            p = out / f"gen{gen}" / "runs" / f"{c['problem_id']}.json"
            items.append((json.loads(p.read_text()), c, workspace.package_text(c["problem_id"])))
        if not items:
            continue
        try:
            refl, recs, prompt = reflect(llm, items, rb)
        except Exception as e:  # noqa: BLE001  a failed learning step skips this update, the loop goes on
            print(f"   learning step failed: {type(e).__name__}: {e}")
            continue
        cost.add(recs)
        (out / f"gen{gen}" / "reflection_prompt.md").write_text(prompt)
        if refl is None:
            print("   learning step returned nothing usable; rulebook unchanged")
            continue
        (out / f"gen{gen}" / "reflection.json").write_text(refl.model_dump_json(indent=1))
        before = Rulebook(rb.handbook, [dict(r) for r in rb.learned])
        log = rb.apply([op.model_dump() for op in refl.ops], gen)
        (out / f"diff_gen{gen}.md").write_text(diff_markdown(before, rb, log, gen))
        print(f"   learning: {sum(e['result'] == 'added' for e in log)} added, "
              f"{sum(e['result'].startswith('rejected') for e in log)} rejected, {len(rb.learned)} learned rules", flush=True)
    rb.save(out / "rulebook_final.json")
    print(f"done: {out} (${cost.total:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
