#!/usr/bin/env python
"""Where a mesh of abc_00000065 is refined: element edge lengths at the arm necks and at the tip nut pockets.

    python experiments/mesh_systems/neck_tip_sizes.py LABEL=MESH_DIR [...] [--json OUT.json]

Per mesh: nodes / cells as the solver builds the model (tet10 -> C3D10, hexahedron -> C3D8), the quality gates
of its report.json (fail / warn listed), and corner-edge lengths of the cells whose centroid is
  neck  - in the ring wall between bore (r 19.67 mm) and outer arc (r 27.77 mm), -15 mm < y < +12 mm;
  tip   - in the nut-pocket / bolt-hole block of the arm tips, y > 19.5 mm, 12.9 mm < |x| < 23.8 mm;
min / median edge [mm] and cells across the 8.1 mm wall (8.1 / median edge).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TET_E = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
HEX_E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def gates(d: Path) -> dict:
    r = json.loads((d / "report.json").read_text())
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("gates", "mesh_gates") and isinstance(v, list):
                    found.extend(v)
                walk(v)
    walk(r)
    st = {}
    for g in found:
        if isinstance(g, dict) and "id" in g:
            st.setdefault(g["id"], (g.get("status"), g.get("value")))
    bad = {k: (s, v) for k, (s, v) in st.items() if s not in ("pass", None)}
    return {"n_gates": len(st), "not_pass": {k: f"{s} {v:.3g}" if isinstance(v, float) else f"{s}" for k, (s, v) in bad.items()}}


def sizes(d: Path) -> dict:
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    gm = read_gmsh(d / "mesh.msh")
    order = 2 if any(ct == "tetra10" for ct, _ in gm.groups["domain"]) else 1
    m = build_model(gm, 3, order)
    X = m.X * 1e3
    nc = m.n_corners
    c = m.conn[:, :nc]
    cen = X[c].mean(axis=1)
    edges = HEX_E if nc == 8 else TET_E
    L = np.stack([np.linalg.norm(X[c[:, a]] - X[c[:, b]], axis=1) for a, b in edges], axis=1)
    r = np.hypot(cen[:, 0], cen[:, 1])
    neck = (r > 19.67) & (r < 27.77) & (cen[:, 1] > -15) & (cen[:, 1] < 12)
    tip = (cen[:, 1] > 19.5) & (np.abs(cen[:, 0]) > 12.9) & (np.abs(cen[:, 0]) < 23.8)
    out = {"nodes": int(m.n_nodes), "cells": int(m.n_elems), "cell_type": m.cell_type, "dofs": 3 * int(m.n_nodes)}
    for name, sel in (("neck", neck), ("tip", tip)):
        e = L[sel]
        out[name] = {"cells": int(sel.sum()), "min_edge_mm": round(float(e.min()), 2),
                     "median_edge_mm": round(float(np.median(e)), 2)}
    out["neck"]["across_wall"] = round(8.1 / out["neck"]["median_edge_mm"], 1)
    return out


def main() -> int:
    args = [x for x in sys.argv[1:] if "=" in x]
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    res = {}
    for a in args:
        lab, d = a.split("=", 1)
        d = Path(d)
        res[lab] = {**sizes(d), "gates": gates(d), "dir": str(d)}
        print(lab, json.dumps(res[lab]), flush=True)
    if js:
        Path(js).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
