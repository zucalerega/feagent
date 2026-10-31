#!/usr/bin/env python
"""Generate surrogate training / test data with the fixed solver (parallel over processes).

Each sample: sample family parameters (``feacore.surrogate.family``), mesh with ``Problem.mesh`` at
a fixed relative size, solve with backend "skfem" (order 2), store graph + features + FEM targets
(nondimensional) as ``.npz`` next to the ``.msh`` and the case ``.json``.

Examples:
  # in-distribution samples (train / val / test are split later by train_surrogate.py)
  python scripts/make_surrogate_dataset.py --out /data/surrogate/ds --split id --n 400 --workers 3
  # out-of-distribution test samples, n per category
  python scripts/make_surrogate_dataset.py --out /data/surrogate/ds --split ood --n 10 --workers 3

Keep datasets outside the repo (or under the gitignored models/ dir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker: workers are the parallelism
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _make_one(task: dict) -> dict:
    import numpy as np

    from feacore.solver import solve
    from feacore.surrogate.data import fem_targets, save_sample
    from feacore.surrogate.family import SurrogatePlate, sample_params
    from feacore.surrogate.graph import build_graph

    cat, idx = task["category"], task["index"]
    out = Path(task["dir"])
    stem = out / f"{cat}_{idx:05d}"
    if task.get("skip_existing") and stem.with_suffix(".npz").exists():
        return {"name": stem.name, "skipped": True}
    try:
        rng = np.random.default_rng(np.random.SeedSequence([task["seed"], task["cat_id"], idx]))
        params, h = sample_params(rng, cat, task["h_rel"])
        prob = SurrogatePlate(id=stem.name, **params)
        t0 = time.perf_counter()
        prob.mesh(stem.with_suffix(".msh"), h, geom_order=2)
        t_mesh = time.perf_counter() - t0
        case = prob.case(stem.with_suffix(".msh").name, order=2)
        case.base_dir = str(out)
        case.save(stem.with_suffix(".json"))
        t0 = time.perf_counter()
        res = solve(case, backend="skfem")
        t_solve = time.perf_counter() - t0
        graph = build_graph(case)
        y = fem_targets(res, graph)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("non-finite FEM targets")
        meta = {"category": cat, "index": idx, "params": params, "h": h, "h_rel": task["h_rel"],
                "t_mesh": t_mesh, "t_solve": t_solve, "fem_wall_time": res.info.wall_time,
                "fem_n_dofs": res.info.n_dofs, "fem_qois": res.evaluate_all(), "fem_warnings": res.info.warnings}
        save_sample(stem.with_suffix(".npz"), graph, y, meta)
        return {"name": stem.name, "n_nodes": graph.n_nodes, "t_mesh": t_mesh, "t_solve": t_solve}
    except Exception as e:  # noqa: BLE001 - report and continue
        for suf in (".npz",):
            stem.with_suffix(suf).unlink(missing_ok=True)
        return {"name": stem.name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}


def _refeaturize_one(path: str) -> dict:
    """Rebuild graph + features of an existing sample from its case/mesh; keep the FEM targets."""
    import numpy as np

    from feacore.case import Case
    from feacore.surrogate.data import load_sample, save_sample
    from feacore.surrogate.graph import build_graph

    p = Path(path)
    try:
        with np.load(p, allow_pickle=False) as z:
            y, meta, old_pts = z["targets"], json.loads(str(z["meta"])), z["points"]
        graph = build_graph(Case.load(p.with_suffix(".json")))
        if graph.points.shape != old_pts.shape or not np.allclose(graph.points, old_pts):
            raise ValueError("mesh nodes changed since the targets were computed")
        save_sample(p, graph, y, meta)
        load_sample(p)
        return {"name": p.stem}
    except Exception as e:  # noqa: BLE001
        return {"name": p.stem, "error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    from feacore.surrogate.family import DEFAULT_H_REL, ID_RANGES, OOD_CATEGORIES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root (outside the repo)")
    ap.add_argument("--split", choices=("id", "ood"), default="id")
    ap.add_argument("--n", type=int, default=100, help="samples (ood: per category)")
    ap.add_argument("--categories", nargs="*", default=None, help="ood categories (default: all)")
    ap.add_argument("--start", type=int, default=0, help="first sample index (to extend a dataset)")
    ap.add_argument("--h-rel", type=float, default=DEFAULT_H_REL, help="mesh size / max(W, H)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--refeaturize", action="store_true",
                    help="recompute graph features of every existing sample under --out (no FEM re-solve)")
    args = ap.parse_args(argv)

    root = Path(args.out)
    if args.refeaturize:
        paths = [str(p) for p in sorted(root.rglob("*.npz"))]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(_refeaturize_one, paths, chunksize=16))
        bad = [r for r in res if "error" in r]
        for r in bad[:10]:
            print(f"FAILED {r['name']}: {r['error']}")
        print(f"refeaturized {len(res) - len(bad)} / {len(res)} samples under {root}")
        return 0 if not bad else 1
    out = root / args.split
    out.mkdir(parents=True, exist_ok=True)
    cats = ["id"] if args.split == "id" else (args.categories or list(OOD_CATEGORIES))
    all_cats = ["id"] + list(OOD_CATEGORIES)
    tasks = [{"category": c, "cat_id": all_cats.index(c), "index": i, "seed": args.seed, "dir": str(out),
              "h_rel": args.h_rel, "skip_existing": args.skip_existing}
             for c in cats for i in range(args.start, args.start + args.n)]
    t0 = time.perf_counter()
    ok, failed = [], []
    ctx = get_context("spawn")  # gmsh is process-global: never fork a process that used it
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_make_one, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            (failed if "error" in r else ok).append(r)
            if "error" in r:
                print(f"[{k}/{len(tasks)}] FAILED {r['name']}: {r['error']}", flush=True)
            elif k % 25 == 0 or k == len(tasks):
                print(f"[{k}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
    elapsed = time.perf_counter() - t0
    done = [r for r in ok if not r.get("skipped")]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest["family"] = "feacore.surrogate.family.SurrogatePlate"
    manifest["id_ranges"] = ID_RANGES
    manifest["ood_categories"] = OOD_CATEGORIES
    manifest["runs"].append({
        "split": args.split, "categories": cats, "n": args.n, "start": args.start, "h_rel": args.h_rel,
        "seed": args.seed, "workers": args.workers, "ok": len(ok), "failed": len(failed),
        "elapsed_s": elapsed, "argv": sys.argv[1:],
        "mean_nodes": (sum(r["n_nodes"] for r in done) / len(done)) if done else None,
        "mean_t_solve": (sum(r["t_solve"] for r in done) / len(done)) if done else None,
        "failures": [{k: r[k] for k in ("name", "error")} for r in failed],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"done: {len(ok)} ok, {len(failed)} failed in {elapsed:.0f}s -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
