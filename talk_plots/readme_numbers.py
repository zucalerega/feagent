"""Numbers for talk_plots/README.md that no act script computes: best vs average agent for every team size
(both gold studies, both tasks) and the API spend per experiment. Reads ace_runs/ only; no API calls.

    .venv/bin/python talk_plots/readme_numbers.py      # writes talk_plots/readme_numbers.json and prints the tables
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "ace_runs")
TASKS = ["abc_00000065_bc0", "abc_00000003_bc7"]
STUDIES = {  # study -> (team size -> run folder, generations averaged over)
    "afternoon (div, diversity on)": ({1: "div_s1", 2: "div_s2", 4: "div_s4", 8: "div_s8"}, range(1, 8)),
    "morning (scale)": ({1: "scale_s1", 2: "scale_s2", 4: "scale_s4", 8: "scale_s8"}, range(1, 5)),
}
GROUPS = {
    "Act 1: field-mode benchmarks (deep1, cook_auto, split1, le11)": ["deep1", "cook_auto", "split1", "le11"],
    "Older point-mode benchmark runs (train1, cook_field)": ["train1", "cook_field"],
    "Act 2 afternoon: div_s1/s2/s4/s8": ["div_s1", "div_s2", "div_s4", "div_s8"],
    "Act 2 morning: scale_s1/s1_opus/s2/s4/s8": ["scale_s1", "scale_s1_opus", "scale_s2", "scale_s4", "scale_s8"],
    "First gold attempt (gold_swarm, gold_single)": ["gold_swarm", "gold_single"],
    "Aborted / buggy starts (div_s1_bug, div_s2_bug, *_aborted_gen0)": [
        "div_s1_bug", "div_s2_bug", "split1_aborted_gen0", "train1_aborted_gen0"],
    "Development and smoke tests (mock2, mockd, mockm, run1, smoke1, smoke2)": [
        "mock2", "mockd", "mockm", "run1", "smoke1", "smoke2"],
}


def scores(run):
    """task -> gen -> list of base scores (invalid runs count 0)."""
    out = {}
    for f in glob.glob(f"{RUNS}/{run}/gen*/runs/*.json"):
        d = json.load(open(f))
        c = d["checks"]
        out.setdefault(d["problem_id"], {}).setdefault(int(d["generation"]), []).append(
            c["base_score"] if c.get("valid") else 0.0)
    return out


def spend(run):
    roles = {}
    for line in open(f"{RUNS}/{run}/cost.jsonl"):
        x = json.loads(line)
        roles[x["role"]] = roles.get(x["role"], 0.0) + x["usd"]
    return roles


res = {"best_vs_mean": [], "spend": {}}
for study, (teams, gens) in STUDIES.items():
    for task in TASKS:
        for n, run in teams.items():
            s = scores(run)[task]
            g = [k for k in gens if k in s]
            best = sum(max(s[k]) for k in g) / len(g)
            mean = sum(sum(s[k]) / len(s[k]) for k in g) / len(g)
            res["best_vs_mean"].append(dict(study=study, task=task, agents=n, run=run, gens=g,
                                            runs_per_gen=len(s[g[0]]), best=best, mean=mean))
for f in sorted(glob.glob(f"{RUNS}/*/cost.jsonl")):
    run = f.split("/")[-2]
    res["spend"][run] = spend(run)
listed = {r for v in GROUPS.values() for r in v}
assert listed == set(res["spend"]), set(res["spend"]) ^ listed
res["spend_groups"] = {k: sum(sum(res["spend"][r].values()) for r in v) for k, v in GROUPS.items()}
res["spend_total"] = sum(res["spend_groups"].values())

json.dump(res, open(os.path.join(os.path.dirname(__file__), "readme_numbers.json"), "w"), indent=1)
for r in res["best_vs_mean"]:
    print(f"{r['study']:30s} {r['task']} {r['agents']} agents ({r['run']}, gens {r['gens'][0]}-{r['gens'][-1]}, "
          f"{r['runs_per_gen']}/gen): best {r['best']:.3f} mean {r['mean']:.3f}")
for k, v in res["spend_groups"].items():
    print(f"${v:7.2f}  {k}")
print(f"${res['spend_total']:7.2f}  total")
