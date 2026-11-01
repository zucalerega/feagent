#!/usr/bin/env python3
"""Fixed layouts for the version-ranking test: does HotSpot 6.0 or 7.0 order the same layouts differently?

Standard library only. Deterministic: the seed is fixed here, so anyone regenerates the same file.
Legality follows the door's published rules (agents/challenge.json): 500 um lattice on die centres, die plus halo at least
500 um from every frame edge, halo rings may touch but not overlap, die bodies at least 100 um apart along x or y.

Strata, per fixture:
  reference   the published layout, snapped to the lattice, as in the door's reference file
  near-NN     12 layouts: every die within 5 mm per coordinate of its reference position
  any-NN      12 layouts: dies placed anywhere legal in the frame

The references are the door's pinned snapped reference files, written out here, so nothing depends on an outside file.
snap() uses Python's round, ties to even. The Ascend fixture follows the corresponding geometry rules; the door doesn't expose it yet.

Usage: python rank_layouts.py <out.json>
"""
import json
import random
import sys

SEED = 20261101
FRAME, LATTICE, MARGIN, GAP = 45.0, 0.5, 0.5, 0.1
NEAR_MM = 5.0
PER_STRATUM = 12
EPS = 1e-9

# name, width, height, power, halo (mm), may rotate; from the door's challenge file
MICRO = [("cpu", 8.25, 9.0, 150, 0.18, True)] * 4 + [("dram", 8.75, 8.75, 20, 0.135, False)] * 4
ASCEND = [("compute", 14.5, 31.4, 256, 0.09, True), ("io", 10.5, 16.0, 14, 0.045, True)] + [("hbm", 7.75, 11.87, 20, 0.045, True)] * 4


def size(die, rotated):
    return (die[2], die[1]) if rotated else (die[1], die[2])


def legal(dies, centers, rotated):
    boxes = []
    for die, c, r in zip(dies, centers, rotated):
        w, h = size(die, r)
        halo = die[4]
        x0, x1, y0, y1 = c[0] - w / 2, c[0] + w / 2, c[1] - h / 2, c[1] + h / 2
        if min(x0 - halo, y0 - halo) < MARGIN - EPS or max(x1 + halo, y1 + halo) > FRAME - MARGIN + EPS:
            return False
        boxes.append((x0, x1, y0, y1, halo))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            gx = max(a[0] - b[1], b[0] - a[1])
            gy = max(a[2] - b[3], b[2] - a[3])
            if max(gx, gy) < GAP - EPS:  # die bodies
                return False
            if max(gx, gy) < a[4] + b[4] - EPS:  # halos may touch, not overlap
                return False
    return True


def snap(v):
    return round(v / LATTICE) * LATTICE


def random_any(dies, rng):
    while True:
        centers, rotated = [], []
        for k, die in enumerate(dies):
            placed = False
            for _ in range(600):
                r = die[5] and rng.random() < 0.5
                w, h = size(die, r)
                halo = die[4]
                lo_x, hi_x = MARGIN + halo + w / 2, FRAME - MARGIN - halo - w / 2
                lo_y, hi_y = MARGIN + halo + h / 2, FRAME - MARGIN - halo - h / 2
                x = snap(rng.uniform(lo_x, hi_x))
                y = snap(rng.uniform(lo_y, hi_y))
                if legal(dies[: k + 1], centers + [[x, y]], rotated + [r]):
                    centers.append([x, y])
                    rotated.append(r)
                    placed = True
                    break
            if not placed:
                break
        if len(centers) == len(dies):
            return centers, rotated


def random_near(dies, ref_centers, ref_rotated, rng):
    steps = int(NEAR_MM / LATTICE)
    while True:
        centers = [[c[0] + rng.randint(-steps, steps) * LATTICE, c[1] + rng.randint(-steps, steps) * LATTICE] for c in ref_centers]
        if legal(dies, centers, ref_rotated):
            return centers, list(ref_rotated)


def fixture(name, dies, ref_centers, ref_rotated, rng):
    assert legal(dies, ref_centers, ref_rotated), (name, "reference is not legal under these rules")
    layouts = [{"id": "reference", "stratum": "reference", "centers_mm": ref_centers, "rotated": ref_rotated}]
    seen = {json.dumps([ref_centers, ref_rotated])}
    for stratum, make in (("near", lambda: random_near(dies, ref_centers, ref_rotated, rng)), ("any", lambda: random_any(dies, rng))):
        n = 0
        while n < PER_STRATUM:
            c, r = make()
            key = json.dumps([c, r])
            if key in seen:
                continue
            seen.add(key)
            n += 1
            layouts.append({"id": f"{stratum}-{n:02d}", "stratum": stratum, "centers_mm": c, "rotated": r})
    return {
        "dies": [{"die": i, "name": d[0], "w_mm": d[1], "h_mm": d[2], "power_w": d[3], "halo_mm": d[4]} for i, d in enumerate(dies)],
        "layouts": layouts,
    }


def main():
    out = sys.argv[1]
    rng = random.Random(SEED)
    micro_ref = [[18.0, 17.5], [27.0, 17.5], [27.0, 27.5], [18.0, 27.5], [8.5, 17.5], [36.5, 17.5], [36.5, 27.5], [8.5, 27.5]]
    ascend_ref = [[17.0, 22.5], [38.5, 22.5], [7.5, 11.0], [21.5, 11.0], [7.5, 34.0], [21.5, 34.0]]
    ascend_rot = [True, False, True, True, True, True]
    data = {
        "schema": "sentilab-rank-layouts-v1",
        "seed": SEED,
        "frame_mm": FRAME,
        "rules": {"lattice_mm": LATTICE, "margin_mm": MARGIN, "min_die_gap_mm": GAP, "near_mm": NEAR_MM, "per_stratum": PER_STRATUM},
        "fixtures": {
            "micro150-tap25d-fig5a": fixture("micro150", MICRO, micro_ref, [False] * 8, rng),
            "ascend910-tap25d-fig6a": fixture("ascend910", ASCEND, ascend_ref, ascend_rot, rng),
        },
    }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
