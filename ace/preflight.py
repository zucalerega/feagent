"""Pre-flight check before a real (paid) ACE run: can this box solve every task of the suite?

    python -m ace.preflight --suite cfd            # free: no API calls
    python -m ace.preflight --suite cfd3d --ping   # + one real generator call (about $0.01)

Checks, in order (exit code 1 if any fails):

1. ANTHROPIC_API_KEY is set (its value is never printed);
2. flow tasks: OpenFOAM (ESI line, v1912+) and foamlib are found;
3. every task's planted good mesh (``planted.GOOD_MESH``) meshes under the node cap, and the
   problem's reference setup on it solves to within the task's tolerance. If this fails, every
   agent's solve would fail too, while its API calls are still billed.
4. with ACE_SCORE=field: every task's field reference, check points and automesher build (they are
   cached, so the loop then starts at once), and the good mesh's field error is shown against the
   automesher's (flow tasks: ace.flow_field).

``--ping`` sends one generator call (``config.MODELS["generator"]``, with the task's real tool list and
the same thinking / effort / fallback settings as the loop) and asks for a one-word answer. It checks
the key, model access and that the API accepts the tool schemas.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from ace import config, fea, planted
from ace.checks import reference_setup, tolerance


def check_tasks(problem_ids: list[str]) -> bool:
    ok = True
    print(f"{'task':28s} {'nodes/cap':>12s} {'value':>12s} {'reference':>12s} {'error':>8s} {'tol':>5s} {'time':>6s}")
    with tempfile.TemporaryDirectory(prefix="ace_preflight_") as tmp:
        for pid in problem_ids:
            problem, cap = fea.get_problem(pid), config.TASKS[pid]["node_cap"]
            req = planted.GOOD_MESH.get(pid)
            if req is None:
                print(f"{pid:28s} no planted.GOOD_MESH entry: skipped")
                continue
            t0 = time.perf_counter()
            m = fea.mesh_from_request(problem, Path(tmp) / f"{pid}.msh", req)
            if not m["ok"]:
                print(f"{pid:28s} MESH FAILED: {m['error']}")
                ok = False
                continue
            sol = fea.run_solve(problem, m, reference_setup(problem), config.TASKS[pid]["qoi"])
            dt = time.perf_counter() - t0
            if not sol["ok"]:
                print(f"{pid:28s} SOLVE FAILED after {dt:.0f} s: {sol['error']}")
                ok = False
                continue
            # the scored QoI against its reference within the reference's own tolerance, in any scoring
            # mode (field mode's target is checked separately, check_field)
            qoi = config.TASKS[pid]["qoi"]
            ref = fea.reference_value(problem, qoi)
            tol = next((r.tolerance for r in problem.references() if r.qoi == qoi), tolerance(pid))
            err = abs(sol["value"] - ref) / abs(ref)
            good = err <= tol and m["nodes"] <= cap
            ok &= good
            print(f"{pid:28s} {m['nodes']:>6d}/{cap:<5d} {sol['value']:12.6g} {ref:12.6g} {err:8.2%} {tol:5.0%} "
                  f"{dt:5.0f}s{'' if good else '  FAIL'}")
    return ok


def check_field(problem_ids: list[str]) -> bool:
    """Field mode (ACE_SCORE=field): build each task's field reference, check points and automesher
    (cached in ace/field_refs/, or ACE_FIELD_REFS), then score the planted good mesh against them.
    The good mesh should beat the automesher (gain > 1); the reference's own error is shown."""
    from ace import field_error
    from ace.config import AUTO_GAIN_TARGET
    ok = True
    print(f"\nfield mode: {'task':26s} {'ref. check':>10s} {'auto error':>10s} {'auto DOFs':>9s} {'good error':>10s} "
          f"{'gain':>6s} {'time':>6s}")
    with tempfile.TemporaryDirectory(prefix="ace_preflight_field_") as tmp:
        for pid in problem_ids:
            req = planted.GOOD_MESH.get(pid)
            problem = fea.get_problem(pid)
            t0 = time.perf_counter()
            try:
                auto = field_error.automesher(pid)
                field_error.probe_points(pid)
                rc = field_error.reference_check(pid)
                m = fea.mesh_from_request(problem, Path(tmp) / f"{pid}.msh", req) if req else None
                good = field_error.errors_of_setup(pid, m, reference_setup(problem)) if m and m["ok"] else None
            except Exception as e:  # noqa: BLE001  any failure here would fail every run's scoring
                print(f"field mode: {pid:26s} FAILED: {type(e).__name__}: {e}")
                ok = False
                continue
            ref = "exact" if rc.get("exact") else f"{rc.get('coarse_vs_fine', float('nan')):.2%}"
            gain = auto["error"] / good["error"] if good else None
            print(f"field mode: {pid:26s} {ref:>10s} {auto['error']:10.2%} {auto['dofs']:9d} "
                  f"{(good or {}).get('error', float('nan')):10.2%} {gain or float('nan'):6.2f} "
                  f"{time.perf_counter() - t0:5.0f}s" + ("" if gain is None or gain > 1 else "  (good mesh below the automesher)"))
    print(f"(gain = automesher error / good mesh error; the target is {AUTO_GAIN_TARGET:g}x; ref. check = the "
          f"reference's coarser level vs its finer one)")
    return ok


def check_flow_backend() -> bool:
    from feacore.solver._foam_env import FoamNotFound, require_openfoam
    from feacore.solver._foam_io import FOAMLIB_ERROR

    try:
        fe = require_openfoam()
    except FoamNotFound as e:
        print(f"OpenFOAM: NOT FOUND: {e}")
        return False
    print(f"OpenFOAM: {fe.describe()}")
    if FOAMLIB_ERROR is not None:
        print(f"foamlib: MISSING ({FOAMLIB_ERROR}); pip install -e '.[cfd]'")
        return False
    return True


def ping(problem_id: str) -> bool:
    """One real generator call with the task's tools: key, model access, tool schemas."""
    from ace.agent import system_prompt, tools_for
    from ace.llm import LLM

    problem = fea.get_problem(problem_id)
    try:
        resp, rec = LLM().agent_turn(system_prompt("", flow=fea.is_flow(problem)),
                                     [{"role": "user", "content": "Pre-flight check: reply with the single word "
                                                                  "OK and do not call any tool."}],
                                     tools_for(problem_id), run_id="preflight")
    except Exception as e:  # noqa: BLE001  report whatever the SDK raised
        print(f"API ping: FAILED: {type(e).__name__}: {e}")
        return False
    text = " ".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
    print(f"API ping: {resp.model} answered {text[:40]!r}, {rec['input'] + rec['cache_write'] + rec['cache_read']} "
          f"input / {rec['output']} output tokens, ${rec['usd']:.4f}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=sorted(config.SUITES), default="cfd")
    ap.add_argument("--problems", default=None, help="comma-separated task ids instead of the suite's")
    ap.add_argument("--ping", action="store_true", help="also make one real generator API call (about $0.01)")
    args = ap.parse_args(argv)
    train, holdout = config.SUITES[args.suite]
    pids = [p for p in args.problems.split(",") if p] if args.problems else train + holdout
    ok = True
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"ANTHROPIC_API_KEY: {'set' if key else 'NOT SET (needed for a real run; --mock runs without it)'}")
    ok &= key or not args.ping
    if any(fea.is_flow(fea.get_problem(p)) for p in pids) and not check_flow_backend():
        return 1
    ok &= check_tasks(pids)
    if config.SCORE == "field":
        ok &= check_field(pids)
    if args.ping and key:
        ok &= ping(pids[0])
    print("pre-flight " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
