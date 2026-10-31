"""Swarm vs single-agent baseline: base score per task per generation, against generations, cumulative
$ and wall time.

    python -m ace.compare ace_runs/deep2 ace_runs/single1            # writes compare.md + compare.png
    python -m ace.compare ace_runs/deep2 ace_runs/single1 --out reports/compare1

Reads only what both loops write in the same format: gen<g>/runs/*.json (each run's checks) and
cost.jsonl (every API call, with a timestamp). The base score is the field-mode score without the
be-the-best bonus, comparable between the two (the bonus needs peers). Crashed runs are left out.

$ and wall time per generation come from cost.jsonl for both runs, the same way: a call with a run
id "g<g>-..." starts or continues generation g, and the calls without one (teacher, reflection,
organizer) belong to the generation before them. Wall time of generation g = from its first call to
the first call of generation g+1 (the last generation: to its last call), so it is approximate (it
misses the scoring before the first call of the run and after the last call).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RUN_GEN = re.compile(r"^g(\d+)-")


def base_scores(run_dir: Path) -> dict[tuple[int, str], list[float]]:
    """{(generation, task): [base score of each run that did not crash]}."""
    out: dict = {}
    for f in Path(run_dir).glob("gen*/runs/*.json"):
        r = json.loads(f.read_text())
        c = r.get("checks") or {}
        if r.get("error") or c.get("score") is None:
            continue
        out.setdefault((r["generation"], r["problem_id"]), []).append(c.get("base_score", c["score"]))
    return out


def gen_costs(run_dir: Path) -> dict[int, dict]:
    """{generation: {"usd", "seconds", "cum_usd", "cum_seconds"}} from cost.jsonl (see the module doc)."""
    path = Path(run_dir) / "cost.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    per: dict[int, dict] = {}
    gen = 0
    for r in sorted(recs, key=lambda r: r["t"]):
        m = RUN_GEN.match(r.get("run_id") or "")
        if m:
            gen = max(gen, int(m.group(1)))
        d = per.setdefault(gen, {"usd": 0.0, "start": r["t"], "end": r["t"]})
        d["usd"] += r["usd"]
        d["end"] = r["t"]
    cum_usd = cum_s = 0.0
    gens = sorted(per)
    for i, g in enumerate(gens):
        d = per[g]
        d["seconds"] = (per[gens[i + 1]]["start"] if i + 1 < len(gens) else d["end"]) - d["start"]
        cum_usd += d["usd"]
        cum_s += d["seconds"]
        d.update(cum_usd=cum_usd, cum_seconds=cum_s)
    return per


def comparison_rows(swarm_dir: Path, single_dir: Path) -> list[dict]:
    sw, si = base_scores(swarm_dir), base_scores(single_dir)
    cs, ci = gen_costs(swarm_dir), gen_costs(single_dir)
    rows = []
    for gen, task in sorted(set(sw) | set(si)):
        s, b = sw.get((gen, task)), si.get((gen, task))
        rows.append({"gen": gen, "task": task,
                     "swarm_best": max(s) if s else None, "swarm_mean": statistics.mean(s) if s else None,
                     "swarm_n": len(s or []), "single": max(b) if b else None,
                     "swarm_cum_usd": cs.get(gen, {}).get("cum_usd"), "single_cum_usd": ci.get(gen, {}).get("cum_usd"),
                     "swarm_cum_min": (cs[gen]["cum_seconds"] / 60) if gen in cs else None,
                     "single_cum_min": (ci[gen]["cum_seconds"] / 60) if gen in ci else None})
    return rows


def table(rows: list[dict], swarm_dir: Path, single_dir: Path) -> str:
    f = lambda x, p=".2f": "-" if x is None else format(x, p)   # noqa: E731
    lines = ["# Swarm vs single agent", "", f"swarm: `{swarm_dir}`; single agent: `{single_dir}`", "",
             "Base score = field-mode score without the be-the-best bonus. Swarm mean / best are over its runs on",
             "that task in that generation (n runs); the single agent has one run per task. Compare the single",
             "agent with the swarm MEAN: the swarm best is picked by the hidden ground-truth score, an oracle the",
             "single agent does not have. Cumulative $ and minutes include learning; minutes are approximate",
             "(from API call timestamps).", "",
             "| gen | task | swarm mean | swarm best (oracle pick) | n | single | swarm cum $ | single cum $ | swarm cum min | single cum min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['gen']} | {r['task']} | {f(r['swarm_mean'])} | {f(r['swarm_best'])} | {r['swarm_n']} | "
                     f"{f(r['single'])} | {f(r['swarm_cum_usd'])} | {f(r['single_cum_usd'])} | "
                     f"{f(r['swarm_cum_min'], '.0f')} | {f(r['single_cum_min'], '.0f')} |")
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], path: Path) -> None:
    """One column per task; rows: base score against generation, cumulative $ and cumulative minutes.
    Swarm in blue (mean solid: the fair comparison; best dotted: picked with the hidden score), single
    agent in orange; one y-scale per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swarm, single, ink = "#2a78d6", "#eb6834", "#333333"
    tasks = sorted({r["task"] for r in rows})
    xs = [("gen", "gen", "generation"), ("swarm_cum_usd", "single_cum_usd", "cumulative $"),
          ("swarm_cum_min", "single_cum_min", "cumulative wall time (min)")]
    fig, axes = plt.subplots(len(xs), len(tasks), figsize=(4.2 * len(tasks), 3.2 * len(xs)), dpi=150, squeeze=False)
    top = max([1.05] + [r[k] * 1.05 for r in rows for k in ("swarm_best", "single") if r[k] is not None])
    for j, task in enumerate(tasks):
        pts = [r for r in rows if r["task"] == task]
        for i, (xs_key, xi_key, xlabel) in enumerate(xs):
            ax = axes[i][j]
            for y, xk, color, ls, label in (("swarm_mean", xs_key, swarm, "-", "swarm mean"),
                                            ("swarm_best", xs_key, swarm, ":", "swarm best (oracle pick)"),
                                            ("single", xi_key, single, "-", "single agent")):
                p = [(r[xk], r[y]) for r in pts if r[xk] is not None and r[y] is not None]
                if p:
                    ax.plot(*zip(*p), color=color, ls=ls, lw=2, marker="o", ms=4, label=label)
            ax.set_ylim(0, top)
            ax.set_xlabel(xlabel, color=ink)
            if j == 0:
                ax.set_ylabel("base score", color=ink)
            if i == 0:
                ax.set_title(task, fontsize=10, color=ink)
            ax.grid(axis="y", color="#e5e5e5", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def write_comparison(swarm_dir: Path, single_dir: Path, out: Path) -> str:
    """compare.md + compare.png in `out`; returns the table."""
    out.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(swarm_dir, single_dir)
    text = table(rows, swarm_dir, single_dir)
    (out / "compare.md").write_text(text)
    if rows:
        plot(rows, out / "compare.png")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("swarm_dir", type=Path, help="a run_loop --deep output dir")
    ap.add_argument("single_dir", type=Path, help="an ace.single_loop output dir")
    ap.add_argument("--out", type=Path, help="where to write compare.md / compare.png (default: single_dir)")
    args = ap.parse_args(argv)
    print(write_comparison(args.swarm_dir, args.single_dir, args.out or args.single_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
