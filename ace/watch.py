"""Live tracker for an ACE run: how every agent is doing, generation by generation.

    python -m ace.watch ace_runs/<dir>                  # serve http://localhost:8765, refreshes itself
    python -m ace.watch ace_runs/<dir> --export out.html   # one static, shareable snapshot

Read-only: it only reads what run_loop writes (run records, playbooks, reflections, cost log,
the run log, and mesh files in each run's work folder for agents still running). The page is
ace/dashboard.html; this module builds the JSON it shows.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ace import config

PAGE = Path(__file__).with_name("dashboard.html")


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


@lru_cache(maxsize=None)
def _reference(problem_id: str) -> float:
    from ace import fea
    p = fea.get_problem(problem_id)
    return fea.reference_value(p, config.TASKS[problem_id]["qoi"])


def _profile(run: dict, c: dict) -> dict:
    """Strengths and weaknesses of one finished run, each 0 (weak) .. 1 (strong)."""
    meshes = [m for m in run.get("meshes", []) if m.get("ok")]
    over = sum(m["nodes"] > c["node_cap"] for m in meshes)
    s = c.get("setup") or {}
    setup_issues = len(s.get("missing", [])) + len(s.get("unexpected", [])) + len(s.get("sign_flips", [])) \
        + sum(s.get(k) is False for k in ("formulation_ok", "thickness_ok", "material_ok"))
    floor, ceil = config.SCORE_FLOOR, config.SCORE_CEIL
    return {
        "accuracy": max(0.0, min(1.0, (c["score"] - floor) / (ceil - floor))),
        "budget use": min(1.0, c["nodes"] / c["node_cap"]) if c.get("nodes") else 0.0,
        "mesh aim": 1 - over / len(meshes) if meshes else 0.0,       # meshes that fit the cap
        # Same BCs as the reference setup. An equivalent setup written differently scores 0.5 here.
        "ref. setup": 1.0 if c.get("solved") and setup_issues == 0 else (0.5 if c.get("solved") else 0.0),
        "self-check": min(1.0, (c.get("n_solves", 0) - 1) / 2) if c.get("n_solves") else 0.0,
    }


def _finished(run: dict) -> dict:
    c = run["checks"]
    ref = _reference(run["problem_id"])
    errs = [abs(s["result"]["value"] - ref) / abs(ref) if s["result"].get("ok") else None for s in run["solves"]]
    issues = []
    s = c.get("setup") or {}
    issues += [f"missing {x}" for x in s.get("missing", [])] + [f"unexpected {x}" for x in s.get("unexpected", [])]
    issues += s.get("sign_flips", [])
    return {
        "run_id": run["run_id"], "problem": run["problem_id"], "state": "error" if run.get("error") else "done",
        "score": None if run.get("error") else c["score"], "rel_error": c.get("rel_error"), "e_uniform": c.get("e_uniform"),
        "gate": c.get("failed_gate"), "nodes": c.get("nodes"), "cap": c["node_cap"], "order": c.get("order"),
        "turns": run["turns"], "meshes": [m.get("nodes") for m in run["meshes"]],
        "solve_errors": errs, "submitted": run.get("submitted"), "cited": run.get("cited", []),
        "usd": sum(u.get("usd", 0) for u in run.get("usage", [])), "issues": issues,
        "profile": None if run.get("error") else _profile(run, c),
    }


def collect(run_dir: Path, log: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    log = log or next((p for p in (run_dir / "run.log", run_dir.parent / f"{run_dir.name}.log") if p.exists()), None)
    log_text = log.read_text(errors="replace") if log else ""
    done_in_log = set(re.findall(r"run (g\d+-\S+) done", log_text))
    problems = config.TRAIN + config.HOLDOUT
    gens = []
    g = 0
    while (gdir := run_dir / f"gen{g}").exists():
        recs = {p.stem: _json(p) for p in sorted((gdir / "runs").glob("*.json"))}
        runs = [_finished(r) for r in recs.values() if r]
        known = {r["run_id"] for r in runs}
        # Agents of a generation still in progress: one work folder per started run.
        for wd in sorted((gdir / "work").glob("*")) if (gdir / "work").exists() else []:
            pid, i = wd.name.rsplit("-", 1)
            rid = f"g{g}-{pid}-{i}"
            if rid not in known and pid in config.TASKS:
                runs.append({"run_id": rid, "problem": pid, "state": "finished, checking" if rid in done_in_log else "running",
                             "meshes": [None] * len(list(wd.glob("mesh_*.msh"))), "cap": config.TASKS[pid]["node_cap"]})
        for pid in config.TASKS:          # rank within each problem (1 = best)
            ranked = sorted((r for r in runs if r["problem"] == pid and r.get("score") is not None), key=lambda r: -r["score"])
            for k, r in enumerate(ranked, 1):
                r["rank"], r["of"] = k, len(ranked)
        refl = {}
        for p in gdir.glob("reflection_*.json"):
            if not p.name.startswith("reflection_prompt_"):
                o = _json(p) or {}
                refl[p.stem.removeprefix("reflection_")] = {"divergence": o.get("divergence"), "lessons": o.get("lessons", [])}
        pb = _json(run_dir / f"playbook_gen{g}.json", {"bullets": []})
        gens.append({"gen": g, "runs": sorted(runs, key=lambda r: r["run_id"]), "playbook": pb["bullets"],
                     "reflections": refl, "changes": _json(gdir / "merge_changes.json")})
        g += 1

    cost = [json.loads(x) for x in (run_dir / "cost.jsonl").read_text().splitlines()] if (run_dir / "cost.jsonl").exists() else []
    by_role: dict[str, float] = {}
    for c in cost:
        by_role[c["role"]] = by_role.get(c["role"], 0) + c["usd"]
    finished = "# ACE run summary" in log_text or (run_dir / "summary.md").exists()
    budget_stop = "BUDGET REACHED" in log_text
    phase = "finished" if finished else ("stopped by budget" if budget_stop else
                                          "reflecting" if gens and all(r["state"] != "running" for r in gens[-1]["runs"]) and gens[-1]["runs"] else "agents running")
    return {
        "run": run_dir.name, "phase": phase, "problems": problems, "train": config.TRAIN, "holdout": config.HOLDOUT,
        "tasks": config.TASKS, "models": config.MODELS, "max_solves": config.MAX_SOLVES,
        "score_floor": config.SCORE_FLOOR, "score_ceil": config.SCORE_CEIL,
        "generations": gens, "usd": sum(by_role.values()), "usd_by_role": by_role, "calls": len(cost),
    }


def export(run_dir: Path, out: Path) -> Path:
    data = json.dumps(collect(run_dir), default=str).replace("</", "<\\/")
    out.write_text(PAGE.read_text().replace("/*__DATA__*/null", data))
    return out


def serve(run_dir: Path, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                body, ctype = json.dumps(collect(run_dir), default=str).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):   # keep the terminal quiet
            pass

    print(f"watching {run_dir}\nopen http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--export", type=Path, help="write a static HTML snapshot instead of serving")
    a = ap.parse_args()
    if a.export:
        print(export(a.run_dir, a.export))
    else:
        serve(a.run_dir, a.port)
