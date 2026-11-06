"""A hand-written meshing rule built from the teacher's lessons: the no-model baseline.

    python -m wsp.heuristic [--split test] > recipes.jsonl     # then: python -m wsp.evaluate --recipes ...
    python -m wsp.heuristic --tune                             # pick the constants on the TRAIN split

It reads only what the student reads (the geometry in u) and applies the teacher's rules: quadratic
elements; a coarse far field (about one element across the part's thinnest section, at most H_MAX);
every small arc (fillet, notch root, hole) refined to a fraction of its radius, and every re-entrant
corner and end of a clamped support to the size floor, each grading linearly back to h at a fixed slope.
If a student only matches this rule, it has learned the rule; if it beats it, it learned more.
"""

from __future__ import annotations

import argparse
import itertools
import json

# Tuned on the training split (python -m wsp.heuristic --tune, 54 settings): 3.86x mean on the 36 training parts.
DEFAULT = {"h_max": 400.0, "wall_frac": 1.0, "arc_frac": 1 / 3, "slope": 0.5, "small_arc": 0.5, "clamp_ends": False}


def recipe(pid: str, c: dict = DEFAULT) -> dict:
    from wsp import task
    geo = task.geometry(pid)
    g = geo.raw
    sf = g.get("smallest_features") or {}
    wall = geo.u(sf["thinnest_wall"]["thickness"]) if "thinnest_wall" in sf else c["h_max"]
    h = float(min(c["h_max"], max(4 * geo.floor_u, c["wall_frac"] * wall)))
    refs = []
    for tag, cur in sorted(geo.curves.items()):
        r = cur.get("radius")
        if r and geo.u(r) < c["small_arc"] * h:
            size = max(geo.floor_u, c["arc_frac"] * geo.u(r))
            refs.append({"curve": tag, "size": size, "radius": max(size, (h - size) / c["slope"])})
    singular = list(g.get("reentrant_corners") or [])
    # the end of a support is singular only where a clamp meets a free edge (not a symmetry or roller
    # support): the statement says which, in words the student reads too
    text = task.statement(pid).lower()
    if c["clamp_ends"] and any(w in text for w in ("clamp", "welded", "fully fixed", "built-in", "encastre")):
        singular += [k for k in (g.get("support_boundary") or []) if k not in singular]
    for k in singular:
        size = geo.floor_u
        refs.append({"corner": k, "size": size, "radius": max(size, (h - size) / c["slope"])})
    return task.canonical({"order": 2, "h": h, "refinements": refs[: task.MAX_REFINEMENTS]})


def tune(workers: int = 3) -> tuple[dict, float]:
    """Grid-search the constants on the training split (never on the test split)."""
    import statistics

    from wsp import problems, task
    pids = problems.train_ids()
    best = (DEFAULT, -1.0)
    for h_max, arc, slope, ends in itertools.product((150.0, 250.0, 400.0), (0.2, 1 / 3, 0.5), (0.35, 0.5, 0.7),
                                                     (True, False)):
        c = {**DEFAULT, "h_max": h_max, "arc_frac": arc, "slope": slope, "clamp_ends": ends}
        rs = task.score_many([(p, recipe(p, c), None) for p in pids], workers=workers)
        m = statistics.mean(r["x_auto"] for r in rs)
        print(json.dumps({**c, "train_mean_x": round(m, 3)}), flush=True)
        if m > best[1]:
            best = (c, m)
    return best


def main(argv=None) -> int:
    from wsp import problems
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--constants", default=None, help="JSON dict overriding DEFAULT")
    a = ap.parse_args(argv)
    if a.tune:
        c, m = tune()
        print(json.dumps({"best": c, "train_mean_x": m}))
        return 0
    c = {**DEFAULT, **(json.loads(a.constants) if a.constants else {})}
    pids = problems.test_ids() if a.split == "test" else problems.train_ids()
    for p in pids:
        print(json.dumps({"pid": p, "spec": recipe(p, c)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
