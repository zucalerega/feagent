"""Compare ACE experiments on one scale: the swarm against single-agent baselines.

    python -m ace.compare --exp swarm=ace_runs/swarm --exp matched=ace_runs/matched \\
        --exp isolated=ace_runs/isolated --exp single=ace_runs/single [--out dir]

The first --exp is the reference (normally the swarm); every difference is reference - other,
so a positive difference means the reference did better.

Why re-score: the loop's score is relative. Efficiency compares a run with the cheapest
in-tolerance run *in the same output directory*, so scores of separate runs are on different
scales. Here every experiment's runs on a problem are pooled and re-scored together with the
loop's own ace.checks.score_population. Absolute numbers then depend on which experiments are
pooled; differences between experiments are what to read. The share of runs within tolerance
and the median relative error do not depend on pooling and are reported too.

Before anything is compared, the experiments' run_config.json files are checked: problems,
held-out set, models, limits, diagnoser settings and the loop's code must match (see
ace.run_loop.SETUP_KEYS). Only the treatment (memory / agents / attempts), the run size and
the budget may differ. --allow-mismatch overrides the check.

Reported per experiment:
  * learning curves: mean score per generation on training and held-out problems, against
    cumulative $ (compute) and generation;
  * the last generation's held-out score (each problem weighted equally) with a bootstrap 90 %
    interval, and the difference to the reference with its interval and P(difference > 0);
  * held-out score at matched spend: each experiment's last generation within the cheapest
    experiment's total spend.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ace.checks import score_population
from ace.run_loop import SETUP_KEYS

N_BOOT = 5000


# --------------------------------------------------------------------------- loading


def loop_dirs(path: Path) -> list[Path]:
    """A baseline dir (baseline.json) lists its loops; a plain run_loop output dir is one loop."""
    meta = path / "baseline.json"
    if meta.exists():
        return [(path / p).resolve() for p in json.loads(meta.read_text())["loops"]]
    return [path]


def load_experiment(label: str, path: Path) -> dict:
    exp = {"label": label, "path": str(path), "loops": [], "runs": []}
    for i, loop in enumerate(loop_dirs(Path(path))):
        cfg = loop / "run_config.json"
        prog = loop / "progress.json"
        exp["loops"].append({
            "dir": str(loop),
            "config": json.loads(cfg.read_text()) if cfg.exists() else None,
            "progress": json.loads(prog.read_text()) if prog.exists() else None,
        })
        for f in sorted(loop.glob("gen*/runs/*.json")):
            r = json.loads(f.read_text())
            r["_exp"], r["_loop"], r["_orig_run_id"] = label, i, r["run_id"]
            exp["runs"].append(r)
    if not exp["runs"]:
        raise SystemExit(f"{label}: no runs found under {path}")
    return exp


# --------------------------------------------------------------------------- setup check


def setup_of(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in SETUP_KEYS}


def check_setups(exps: list[dict]) -> tuple[list[str], list[str]]:
    """Returns (differences, warnings). Differences are setup keys that do not match the first
    experiment's first loop; warnings note loops without a run_config.json (older runs)."""
    ref, ref_name, diffs, warns = None, None, [], []
    for e in exps:
        for i, lp in enumerate(e["loops"]):
            name = f"{e['label']}" + (f"[{i}]" if len(e["loops"]) > 1 else "")
            if lp["config"] is None:
                warns.append(f"{name}: no run_config.json (run made before ace.compare existed); setup not verified")
                continue
            s = setup_of(lp["config"])
            if ref is None:
                ref, ref_name = s, name
                continue
            for k in SETUP_KEYS:
                if s[k] != ref[k]:
                    diffs.append(f"{name}.{k} = {s[k]!r} but {ref_name}.{k} = {ref[k]!r}")
    return diffs, warns


# --------------------------------------------------------------------------- scoring


def rescore(exps: list[dict]) -> None:
    """Pool every experiment's runs per problem and score them against each other (in place on
    copies held by the experiments; the files on disk are not touched)."""
    by_problem: dict[str, list[dict]] = {}
    for e in exps:
        for r in e["runs"]:
            r["run_id"] = f"{r['_exp']}/{r['_loop']}/{r['_orig_run_id']}"   # unique across experiments
            by_problem.setdefault(r["problem_id"], []).append(r)
    for runs in by_problem.values():
        score_population(runs)


def split_of(exps: list[dict]) -> tuple[list[str], list[str]]:
    for e in exps:
        for lp in e["loops"]:
            if lp["config"]:
                return lp["config"]["problems"], lp["config"]["holdout"]
    probs = sorted({r["problem_id"] for e in exps for r in e["runs"]})
    return probs, []


def macro_mean(runs: list[dict], problems: list[str], key: str = "score") -> float | None:
    """Mean over problems of the mean over runs (each problem weighs the same). Crashed runs
    (API or code errors) say nothing about the agent and are left out, as in run_loop."""
    per = []
    for p in problems:
        vals = [float(r["checks"].get(key) or 0.0) for r in runs if r["problem_id"] == p and not r.get("error")]
        if vals:
            per.append(statistics.mean(vals))
    return statistics.mean(per) if per else None


def boot_means(runs: list[dict], problems: list[str], rng) -> np.ndarray:
    """Bootstrap distribution of macro_mean, resampling runs within each problem."""
    groups = [np.array([float(r["checks"].get("score") or 0.0) for r in runs if r["problem_id"] == p and not r.get("error")])
              for p in problems]
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.full(N_BOOT, np.nan)
    draws = [g[rng.integers(0, len(g), size=(N_BOOT, len(g)))].mean(axis=1) for g in groups]
    return np.mean(draws, axis=0)


def cumulative_cost(exp: dict) -> dict[int, dict]:
    """Cumulative $ and agent runs at the end of each generation, summed over the experiment's loops.
    Uses progress.json (all LLM spend incl. diagnoser and curator); older runs without it fall back
    to the agents' own token cost recorded in each run (diagnoser/curator spend then missing)."""
    gens = sorted({r["generation"] for r in exp["runs"]})
    out = {g: {"usd": 0.0, "runs": 0, "usd_source": "progress.json"} for g in gens}
    for i, lp in enumerate(exp["loops"]):
        mine = [r for r in exp["runs"] if r["_loop"] == i]
        prog = {p["gen"]: p for p in (lp["progress"] or {}).get("generations", [])}
        run_usd = 0.0
        n = 0
        for g in gens:
            gr = [r for r in mine if r["generation"] == g]
            n += len(gr)
            run_usd += sum(u.get("usd", 0.0) for r in gr for u in r.get("usage", []))
            if g in prog and prog[g].get("usd") is not None:
                out[g]["usd"] += float(prog[g]["usd"])
            else:
                out[g]["usd"] += run_usd
                out[g]["usd_source"] = "agent tokens only"
            out[g]["runs"] += n
    return out


# --------------------------------------------------------------------------- report


def compare(exps: list[dict], out: Path, allow_mismatch: bool = False) -> dict:
    diffs, warns = check_setups(exps)
    if diffs and not allow_mismatch:
        raise SystemExit("setups differ; these experiments are not comparable:\n  " + "\n  ".join(diffs)
                         + "\n(use --allow-mismatch to compare anyway)")
    rescore(exps)
    train, holdout = split_of(exps)
    rng = np.random.default_rng(0)
    res = {"experiments": [], "setup_differences": diffs, "warnings": warns, "train": train, "holdout": holdout}

    ref_final = None
    for e in exps:
        cost = cumulative_cost(e)
        curve = []
        for g in sorted(cost):
            gr = [r for r in e["runs"] if r["generation"] == g]
            curve.append({
                "gen": g, "runs": len(gr), "cum_runs": cost[g]["runs"], "cum_usd": cost[g]["usd"],
                "usd_source": cost[g]["usd_source"],
                "train": macro_mean(gr, train), "holdout": macro_mean(gr, holdout),
                "train_within_tol": macro_mean(gr, train, "meets_tol"),
                "holdout_within_tol": macro_mean(gr, holdout, "meets_tol"),
                "train_best": max((float(r["checks"].get("score") or 0) for r in gr
                                   if r["problem_id"] in train and not r.get("error")), default=None),
            })
        last = max(cost)
        fin = [r for r in e["runs"] if r["generation"] == last]
        boots = boot_means(fin, holdout, rng)
        errs = [r["checks"]["rel_error"] for r in fin if r["problem_id"] in holdout and r["checks"].get("rel_error") is not None]
        entry = {
            "label": e["label"], "path": e["path"], "loops": len(e["loops"]),
            "treatment": [{k: (lp["config"] or {}).get(k) for k in ("memory", "agents", "attempts", "runs_per_problem")}
                          for lp in e["loops"]][:1],
            "curve": curve, "last_gen": last,
            "final_holdout": macro_mean(fin, holdout), "final_holdout_ci": [float(np.nanpercentile(boots, 5)),
                                                                            float(np.nanpercentile(boots, 95))],
            "final_train": macro_mean(fin, train),
            "final_holdout_median_rel_error": statistics.median(errs) if errs else None,
            "per_problem": {p: macro_mean(fin, [p]) for p in train + holdout},
            "total_usd": cost[last]["usd"], "total_runs": cost[last]["runs"],
            "_boots": boots,
        }
        if ref_final is None:
            ref_final = entry
        else:
            d = ref_final["_boots"] - boots
            entry["diff_vs_ref"] = {"mean": (ref_final["final_holdout"] or 0) - (entry["final_holdout"] or 0),
                                    "ci": [float(np.nanpercentile(d, 5)), float(np.nanpercentile(d, 95))],
                                    "p_ref_better": float(np.nanmean(d > 0))}
        res["experiments"].append(entry)

    # held-out score at matched spend: everyone's last generation within the cheapest total spend
    cap = min(x["total_usd"] for x in res["experiments"])
    res["matched_spend_usd"] = cap
    for x in res["experiments"]:
        within = [c for c in x["curve"] if c["cum_usd"] <= cap * (1 + 1e-9)] or x["curve"][:1]
        x["at_matched_spend"] = {"gen": within[-1]["gen"], "holdout": within[-1]["holdout"],
                                 "train": within[-1]["train"], "cum_usd": within[-1]["cum_usd"]}

    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({**res, "experiments": [
        {k: v for k, v in x.items() if not k.startswith("_")} for x in res["experiments"]]}, indent=1, default=float))
    (out / "compare.md").write_text(render_md(res))
    plot(res, out / "compare.png")
    return res


def _f(v, fmt=".3f"):
    return "-" if v is None else format(v, fmt)


def render_md(res: dict) -> str:
    exps = res["experiments"]
    ref = exps[0]["label"]
    L = ["# Swarm vs baselines", "",
         f"Reference: **{ref}**. Scores are re-scored on the pooled runs of all experiments (see ace/compare.py);",
         "compare differences, not absolute values. Held-out problems are never reflected on, so their score",
         f"is the transfer test. Train: {', '.join(res['train'])}. Held out: {', '.join(res['holdout']) or '-'}.", ""]
    if res["warnings"]:
        L += ["Warnings:", *[f"- {w}" for w in res["warnings"]], ""]
    if res["setup_differences"]:
        L += ["**Setup differences (compared anyway with --allow-mismatch):**", *[f"- {d}" for d in res["setup_differences"]], ""]
    L += ["## Final generation", "",
          "| experiment | loops | last gen | runs | total $ | held-out score [90% CI] | train score | held-out median rel. error | "
          f"{ref} - this [90% CI] | P({ref} better) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in exps:
        d = x.get("diff_vs_ref")
        L.append(f"| {x['label']} | {x['loops']} | {x['last_gen']} | {x['total_runs']} | {x['total_usd']:.2f} | "
                 f"{_f(x['final_holdout'])} [{_f(x['final_holdout_ci'][0])}, {_f(x['final_holdout_ci'][1])}] | "
                 f"{_f(x['final_train'])} | {_f(x['final_holdout_median_rel_error'], '.2e')} | "
                 + (f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p_ref_better']:.2f} |" if d else "- | - |"))
    L += ["", f"## Held-out score at matched spend (${res['matched_spend_usd']:.2f}, the cheapest experiment's total)", "",
          "| experiment | generation reached | cumulative $ | held-out score | train score |", "|---|---|---|---|---|"]
    for x in exps:
        m = x["at_matched_spend"]
        L.append(f"| {x['label']} | {m['gen']} | {m['cum_usd']:.2f} | {_f(m['holdout'])} | {_f(m['train'])} |")
    L += ["", "## Learning curves", "",
          "| experiment | gen | runs | cumulative runs | cumulative $ | train score | held-out score | train within tol | held-out within tol |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in exps:
        for c in x["curve"]:
            L.append(f"| {x['label']} | {c['gen']} | {c['runs']} | {c['cum_runs']} | {c['cum_usd']:.2f} | {_f(c['train'])} | "
                     f"{_f(c['holdout'])} | {_f(c['train_within_tol'], '.2f')} | {_f(c['holdout_within_tol'], '.2f')} |")
    L += ["", "## Final generation per problem", "", "| problem | " + " | ".join(x["label"] for x in exps) + " |",
          "|---" * (len(exps) + 1) + "|"]
    for p in res["train"] + res["holdout"]:
        L.append(f"| {p}{' (held out)' if p in res['holdout'] else ''} | " + " | ".join(_f(x["per_problem"].get(p)) for x in exps) + " |")
    srcs = {c["usd_source"] for x in exps for c in x["curve"]}
    if "agent tokens only" in srcs:
        L += ["", "Note: some runs predate progress.json; their $ counts the agents' tokens only (no diagnoser/curator)."]
    return "\n".join(L) + "\n"


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    for ax, split in zip(axes, ("holdout", "train")):
        for i, x in enumerate(res["experiments"]):
            pts = [c for c in x["curve"] if c[split] is not None]
            if not pts:
                continue
            ax.plot([c["cum_usd"] for c in pts], [c[split] for c in pts], marker="o", lw=2, ms=6,
                    color=colors[i % len(colors)], label=x["label"])
            for c in pts:
                ax.annotate(str(c["gen"]), (c["cum_usd"], c[split]), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color="#555555")
        ax.set_title("held-out problems (transfer)" if split == "holdout" else "training problems", fontsize=10)
        ax.set_xlabel("cumulative spend ($); labels = generation")
        ax.grid(color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean score (pooled re-scoring)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", required=True, metavar="LABEL=DIR",
                    help="an experiment: a run_loop output dir or an ace.baselines dir; the first is the reference")
    ap.add_argument("--out", type=Path, help="output dir (default ace_runs/compare-<timestamp>)")
    ap.add_argument("--allow-mismatch", action="store_true", help="compare even if the setups differ")
    args = ap.parse_args(argv)
    exps = []
    for spec in args.exp:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--exp needs LABEL=DIR, got {spec!r}")
        exps.append(load_experiment(label, Path(path)))
    out = args.out or Path("ace_runs") / f"compare-{time.strftime('%Y%m%d-%H%M%S')}"
    compare(exps, out, args.allow_mismatch)
    print((out / "compare.md").read_text())
    print(f"written: {out}/compare.md, compare.json, compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
