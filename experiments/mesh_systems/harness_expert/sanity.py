#!/usr/bin/env python
"""Sanity solve of a harness_expert LINEAR mesh: fixed bottom, 100 N compressive force on top (the spec's
bcs, set by build.py), Alpha Fast E/nu. Prints reaction, top mean displacement, peak von Mises.
usage: sanity.py OUT_DIR"""
import json
import sys
import time

from feacore.adapters.mesh_harness import case_from_mesh_harness
from feacore.solver import solve

out = sys.argv[1]
case = case_from_mesh_harness(out)
t0 = time.time()
res = solve(case, "skfem", linear_solver="pardiso")
vals = res.evaluate_all()
print(json.dumps({"solve_s": round(time.time() - t0, 1), **vals}, default=float, indent=1))
