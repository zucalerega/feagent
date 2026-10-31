"""Run the reference convergence study of every registered problem and write a validation report.

    python scripts/validate.py                       # all problems, skfem, order 2
    python scripts/validate.py --tags nafems --order 1 --jobs 4
    python scripts/validate.py --problems nafems_le1 lame_cylinder_plane_strain --backend ccx

Each study runs in its own process (gmsh is process-global). Results go to
benchmarks/<id>/private/validation_<backend>_p<order>.json and a markdown summary to
docs/validation_report.md (or --report).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_one(pid: str, backend: str, order: int, out_root: str, nested: bool = False) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from feacore.problems.registry import get
    from feacore.verification.study import convergence_study, save_study

    t0 = time.perf_counter()
    try:
        study = convergence_study(get(pid), order=order, backend=backend, nested=nested)
        study["wall_time"] = time.perf_counter() - t0
        tag = "_nested" if nested else ""
        save_study(study, Path(out_root) / pid / "private" / f"validation_{backend}_p{order}{tag}.json")
        return study
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        return {"problem": pid, "backend": backend, "order": order, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "passed": False, "qois": [],
                "wall_time": time.perf_counter() - t0}


def _fmt(x, spec=".4g"):
    return "" if x is None else format(x, spec)


def _pct(x):
    return "" if x is None else f"{100 * x:.2f}"


def report_markdown(studies: list[dict], backend: str, order: int) -> str:
    lines = [f"# Validation report: backend `{backend}`, order {order}", "",
             "Reference convergence studies (`scripts/validate.py`). GCI per Celik et al. (2008) on the three "
             "finest meshes; p = observed order. **pass** = extrapolated (or finest) value within the reference "
             "tolerance; for QoIs expected to diverge, **pass** = divergence detected.", "",
             "| problem | QoI | reference | finest | extrapolated | p (theory) | GCI % | error % | convergence | flags | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_pass = n_judged = 0
    for s in sorted(studies, key=lambda s: s["problem"]):
        if "error" in s:
            lines.append(f"| {s['problem']} | ERROR | | | | | | | | `{s['error'][:120]}` | FAIL |")
            n_judged += 1
            continue
        for q in s["qois"]:
            g = q.get("gci") or {}
            ref = q.get("reference")
            ext = g.get("phi_ext")
            err = q.get("err_ext") if q.get("err_ext") is not None else q.get("err_fine")
            verdict = {True: "pass", False: "**FAIL**", None: "-"}[q.get("passed")]
            if q.get("passed") is not None:
                n_judged += 1
                n_pass += bool(q["passed"])
            p = g.get("p")
            lines.append(
                f"| {s['problem']} | {q['name']} | {_fmt(ref['value']) if ref else ''} | {_fmt(q['values'][-1])} "
                f"| {_fmt(ext)} | {_fmt(p, '.2f')} ({_fmt(q.get('p_theory'), '.0f')}) | {_pct(g.get('gci_fine'))} "
                f"| {_pct(err)} | {g.get('convergence', '')} | {', '.join(g.get('flags', []))} | {verdict} |")
    lines += ["", f"**{n_pass} / {n_judged} judged QoIs pass.**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--backend", default="skfem")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any judged QoI fails")
    ap.add_argument("--nested", action="store_true", help="nested (split) refinement instead of remeshing")
    args = ap.parse_args(argv)

    from feacore.problems.registry import all_problems, discover

    errors = discover()
    for mod, err in errors.items():
        print(f"warning: {mod} failed to import: {err}", file=sys.stderr)
    ids = args.problems or [p.id for p in all_problems(tags=args.tags)]
    print(f"validating {len(ids)} problems with {args.backend} p{args.order} on {args.jobs} processes")
    studies = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_run_one, pid, args.backend, args.order, args.out, args.nested): pid for pid in ids}
        for f in as_completed(futs):
            s = f.result()
            studies.append(s)
            status = "ERROR " + s["error"] if "error" in s else ("pass" if s.get("passed") else
                                                                 "no verdict" if s.get("passed") is None else "FAIL")
            print(f"  {s['problem']:40s} {s.get('wall_time', 0):7.1f}s  {status}")
    tag = "_nested" if args.nested else ""
    report = Path(args.report or ROOT / "docs" / f"validation_report_{args.backend}_p{args.order}{tag}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_markdown(studies, args.backend, args.order))
    (report.with_suffix(".json")).write_text(json.dumps(
        [{k: s.get(k) for k in ("problem", "passed", "error", "wall_time")} for s in studies], indent=2))
    print(f"report: {report}")
    failed = [s["problem"] for s in studies if s.get("passed") is False]
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
