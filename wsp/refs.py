"""Build the field-mode references and automesher baselines for the student's problems (cached in
ace/field_refs/, gitignored), in parallel processes.

    python -m wsp.refs                 # every train and test problem
    python -m wsp.refs --ids a,b --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _one(pid: str) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from wsp import problems  # noqa: F401  registers the training variants in this worker
    from ace import field_error
    problems.ensure_tasks([pid])
    t = time.perf_counter()
    try:
        field_error.reference(pid)
        field_error.coarse_reference(pid)
        field_error.probe_points(pid)
        auto = field_error.automesher(pid)
        chk = field_error.reference_check(pid)
        return {"id": pid, "ok": True, "seconds": round(time.perf_counter() - t, 1),
                "auto_error": auto["error"], "auto_dofs": auto["dofs"], "ref_self_error": chk["coarse_vs_fine"]}
    except Exception as e:  # noqa: BLE001
        return {"id": pid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    from wsp import problems
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    ids = [x for x in a.ids.split(",") if x] or problems.train_ids() + problems.test_ids()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_one, pid) for pid in ids]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(sorted(rows, key=lambda r: r["id"]), fh, indent=1)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
