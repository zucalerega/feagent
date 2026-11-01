#!/usr/bin/env python
"""Sanity solve of a linear netgen mesh with feacore/skfem (CPU, PARDISO): fixed bottom, 100 N
compression on top along AXIS. Usage: sanity.py OUT_DIR AXIS [MATERIAL_JSON]"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from feacore.adapters.mesh_harness import case_from_mesh_harness
from feacore.solver import solve

out, ax = Path(sys.argv[1]), "xyz".index(sys.argv[2])
mat = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "experiments/material/alpha_fast_dogbone1.json"))
f = [0.0, 0.0, 0.0]
f[ax] = -100.0
case = case_from_mesh_harness(out, fea={"fixed": ["bottom"], "loads": [{"group": "top", "force": f}],
                                        "E": mat["E"] * 1e6, "nu": mat["nu"]})
res = solve(case, backend="skfem", linear_solver="pardiso")
s = res.summary()
print(json.dumps({k: s[k] for k in s if k != "fields"}, indent=1, default=float)[:3000])
