"""Compression between platens: harness automesh -> query groups -> gates -> feacore solve (CPU PARDISO) -> vtu."""
import json, os, subprocess, sys, time
from pathlib import Path
import yaml
B = 1.0
def plane(n, axis, at, tol=5e-4):
    lo = [-B, -B, -B]; hi = [B, B, B]; lo[axis] = at - tol; hi[axis] = at + tol
    return {"kind": "plane", "normal": n, "box": [lo, hi]}
SETUP = {  # part: (axis index, bottom query, top query, top count); 'at' is too tight for metre parts (harness bug)
    "abc_00000003": (2, plane([0, 0, -1], 2, 0.0), plane([0, 0, 1], 2, 0.06125), 1),
    "abc_00000065": (1, plane([0, -1, 0], 1, -0.04396), plane([0, 1, 0], 1, 0.03384), 2),
}
E, NU, F0 = float(os.environ.get("COMP_E_PA", 2.23e9)), 0.35, float(os.environ.get("COMP_FORCE_N", 100.0))   # override: COMP_E_PA, COMP_FORCE_N
#  # Formlabs Alpha Fast Resin, default post-cure (TDS 2026-09-28); nu assumed
def sh(*a): return subprocess.run(a, check=False, capture_output=True, text=True)
def run(part, root):
    axis, bot, top, ntop = SETUP[part]
    d = root / part; step = root / f"{part}.step"
    sh("feagent", "automesh", str(step), "-o", str(d), "--no-render", "--json")
    spec = yaml.safe_load((d / "mesh.yaml").read_text())
    spec["groups"] = {"bottom": {"select": bot, "expect": 1}, "top": {"select": top, "expect": ntop}}
    force = [0.0, 0.0, 0.0]; force[axis] = -F0
    spec["bcs"] = {"fea": {"fixed": ["bottom"], "loads": [{"group": "top", "force": force}], "E": E, "nu": NU}}
    spec["element"] = {**spec.get("element", {}), "order": 2}
    (d / "mesh.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    chk = sh("feagent", "check", str(d / "mesh.yaml"), "--no-render", "--json")
    rep = json.loads(chk.stdout) if chk.stdout.strip() else {"error": chk.stderr[-800:]}
    sh(sys.executable, "-m", "feacore.adapters.mesh_harness", str(d / "out"), "--out", str(d / "case.json"))
    from feacore.case import Case
    from feacore.solver import solve
    t = time.time(); res = solve(Case.load(d / "case.json"), backend="skfem", linear_solver="pardiso"); ts = time.time() - t
    res.write_vtu(str(d / "result.vtu"))
    gates = {g["id"]: g["status"] for g in rep.get("gates", [])}
    s = {"part": part, "axis": "xyz"[axis], "force_N": F0, "E": E, "nu": NU, "solve_s": round(ts, 2),
         "status": rep.get("status"), "gates": gates, "qois": {k: float(v) for k, v in res.evaluate_all().items()}}
    (d / "summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s, indent=1))
if __name__ == "__main__":
    root = Path(sys.argv[1])
    for p in sys.argv[2:]: run(p, root)
