"""Scan every local run for equal-budget wins (no API calls, ~1-2 min).

    .venv/bin/python talk_plots/equal_budget/gallery_scan.py

1. Setup check. A run counts only if its scored setup is physically the same problem as the reference
   setup (ace/checks.py reference_setup). Text comparison is not enough (traction 1/16 on Cook's 16-long
   edge == total force 1; symmetry == zero normal displacement), so each distinct setup is SOLVED on the
   problem's automesher mesh next to the reference setup, and the two stress fields are compared at the
   reference integration points (relative energy-norm difference). Equivalent setups agree to round-off;
   wrong ones (deep1 generation-0 LE11: temperature sqrt(x**2+y**2)+y instead of x+y) differ by percent.
2. Table of every valid, correct-setup run scored by field error (checks.rel_error = the scored error,
   max(field, probe), checks.dofs), from ace_runs/*/gen*/runs/*.json.
3. Pairs: within one training run and one problem, a later-generation run with <= the DOFs and
   >= 2x lower error than an earlier-generation run; and every run vs the automesher refined to the
   same DOFs (log-log interpolation in talk_plots/act1_agentic_works/automesher_sweep.json; the
   figures regenerate that mesh exactly).

Writes gallery_data/setup_check.json, gallery_data/runs.csv, gallery_data/pairs.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / "gallery_data"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from ace import checkpoint as ckpt, fea  # noqa: E402
from ace import field_error as fe  # noqa: E402
from ace.checks import reference_setup  # noqa: E402
from ace.config import TASKS  # noqa: E402

SAME_SETUP_TOL = 1e-6        # relative energy-norm difference of the two stress fields
SWEEP = json.loads((ROOT / "talk_plots/act1_agentic_works/automesher_sweep.json").read_text())


def automesher_file(pid: str, path: Path) -> dict:
    """The scored automesher mesh (ace/field_error.py automesher), written to path."""
    import gmsh
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    p = fea.get_problem(pid)
    with gmsh_session(p.id):
        p.build_geometry()
        ensure_domain_group(p.dim)
        gmsh.model.mesh.generate(p.dim)
        gmsh.model.mesh.setOrder(2)
        if "gold" in p.tags or "physical" in p.tags:
            fe._unfold_tet10()
        gmsh.write(str(path))
    return {"path": str(path), "order": 2}


def setup_difference(pid: str, setups: list[dict]) -> list[float]:
    """Relative energy-norm difference between each setup's stress field and the reference setup's,
    all solved on the same (automesher) mesh."""
    p = fea.get_problem(pid)
    ref = fe.reference(pid)
    mat = reference_setup(p)["material"]
    qoi = TASKS[pid]["qoi"]
    with tempfile.TemporaryDirectory() as tmp:
        mesh = automesher_file(pid, Path(tmp) / "auto.msh")
        s0 = fe.stresses(fe._solve(p, mesh, reference_setup(p), qoi), ref["pts"], p.dim)
        n0 = fe.energy_sq(s0, ref["w"], mat["E"], mat["nu"])
        out = []
        for s in setups:
            si = fe.stresses(fe._solve(p, mesh, s, qoi), ref["pts"], p.dim)
            out.append(float(np.sqrt(fe.energy_sq({k: si[k] - s0[k] for k in s0}, ref["w"], mat["E"], mat["nu"]) / n0)))
    return out


def load_runs() -> list[dict]:
    rows = []
    for f in sorted(Path("ace_runs").glob("*/gen*/runs/*.json")):
        r = json.loads(f.read_text())
        c = r.get("checks") or {}
        if "field_error" not in c:            # scored by a single QoI (older runs): not comparable
            continue
        cp = ckpt.checkpoint(r)
        rows.append({"run_dir": f.parts[1], "gen": int(f.parent.parent.name[3:]), "run_id": r["run_id"],
                     "problem": r["problem_id"], "file": str(f), "valid": bool(c.get("valid")),
                     "error": c.get("rel_error"), "field_error": c.get("field_error"), "dofs": c.get("dofs"),
                     "auto_error": c.get("auto_error"), "auto_dofs": c.get("auto_dofs"),
                     "setup": json.dumps(cp["setup"], sort_keys=True) if cp else None})
    return rows


def auto_error_at(pid: str, dofs: float) -> float | None:
    """The automesher's error refined to `dofs` (log-log interpolation in the sweep), None outside it."""
    if pid not in SWEEP:
        return None
    r = sorted(SWEEP[pid]["rows"], key=lambda d: d["dofs"])
    x, y = np.log([d["dofs"] for d in r]), np.log([d["error"] for d in r])
    if not x[0] <= np.log(dofs) <= x[-1]:
        return None
    return float(np.exp(np.interp(np.log(dofs), x, y)))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rows = load_runs()

    # 1. setup check, one solve per distinct setup
    check = {}
    by_problem = defaultdict(set)
    for d in rows:
        if d["setup"]:
            by_problem[d["problem"]].add(d["setup"])
    for pid, setups in sorted(by_problem.items()):
        setups = sorted(setups)
        diff = setup_difference(pid, [json.loads(s) for s in setups])
        check[pid] = [{"setup": json.loads(s), "rel_difference_vs_reference": e, "correct": e < SAME_SETUP_TOL,
                       "n_runs": sum(d["setup"] == s for d in rows)} for s, e in zip(setups, diff)]
        for s, e in zip(setups, diff):
            print(f"{pid:22s} diff {e:.2e}  correct {e < SAME_SETUP_TOL}  {s[:110]}")
    (DATA / "setup_check.json").write_text(json.dumps(check, indent=1))
    correct = {(pid, json.dumps(c["setup"], sort_keys=True)) for pid, cs in check.items() for c in cs if c["correct"]}
    for d in rows:
        d["setup_correct"] = (d["problem"], d["setup"]) in correct
        d["auto_error_equal_dofs"] = auto_error_at(d["problem"], d["dofs"]) if d["dofs"] else None

    with open(DATA / "runs.csv", "w", newline="") as f:
        keys = [k for k in rows[0] if k != "setup"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 3. pairs
    ok = [d for d in rows if d["valid"] and d["setup_correct"] and d["error"] is not None]
    later_vs_earlier = []
    groups = defaultdict(list)
    for d in ok:
        groups[(d["run_dir"], d["problem"])].append(d)
    for (rd, pid), rs in groups.items():
        for a in rs:
            for b in rs:
                if b["gen"] > a["gen"] and b["dofs"] <= a["dofs"] and b["error"] * 2 <= a["error"]:
                    later_vs_earlier.append({"run_dir": rd, "problem": pid, "earlier": a["run_id"], "later": b["run_id"],
                                             "earlier_error": a["error"], "later_error": b["error"],
                                             "earlier_dofs": a["dofs"], "later_dofs": b["dofs"],
                                             "ratio": a["error"] / b["error"]})
    vs_auto = [{"run_dir": d["run_dir"], "problem": d["problem"], "run_id": d["run_id"], "gen": d["gen"],
                "error": d["error"], "dofs": d["dofs"], "auto_error_equal_dofs": d["auto_error_equal_dofs"],
                "ratio": d["auto_error_equal_dofs"] / d["error"]}
               for d in ok if d["auto_error_equal_dofs"] and d["auto_error_equal_dofs"] >= 2 * d["error"]]
    later_vs_earlier.sort(key=lambda p: -p["ratio"])
    vs_auto.sort(key=lambda p: -p["ratio"])
    (DATA / "pairs.json").write_text(json.dumps({"later_vs_earlier": later_vs_earlier, "vs_equal_dof_automesher": vs_auto},
                                                indent=1))
    excluded = [d for d in rows if d["valid"] and not d["setup_correct"]]
    print(f"\n{len(rows)} field-scored runs, {len(ok)} valid with a correct setup, "
          f"{len(excluded)} valid but wrong setup: " + str(sorted({(d['run_dir'], d['problem'], d['gen']) for d in excluded})))
    print(f"later-vs-earlier pairs: {len(later_vs_earlier)}; vs equal-DOF automesher: {len(vs_auto)}")
    seen = set()
    for p in later_vs_earlier:
        k = (p["run_dir"], p["problem"])
        if k not in seen:
            seen.add(k)
            print("  best", p)
    for p in vs_auto[:12]:
        print("  auto", p)


if __name__ == "__main__":
    main()
