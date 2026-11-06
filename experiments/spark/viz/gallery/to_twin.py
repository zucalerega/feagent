"""<set dir>: r.vtu (SI: m, Pa) -> r_mm.vtu (mm, MPa) + run_manifest.json -> export_twin.py -> <set>.usdc."""
import json
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np

d, s = Path(sys.argv[1]), sys.argv[2]
m = meshio.read(d / "r.vtu")
m.points = np.asarray(m.points, float) * 1e3
for k in list(m.point_data):
    v = np.asarray(m.point_data[k], float)
    if k in ("displacement", "ux", "uy", "uz", "u_mag"):
        m.point_data[k] = v * 1e3
    elif "mises" in k or k.startswith(("s_", "sigma", "stress")):
        m.point_data[k] = v * 1e-6
meshio.write(d / "r_mm.vtu", m)
r = json.loads((d / "r.json").read_text())
pr = json.loads((d / "pilot_result.json").read_text())
info = r.get("info", {})
dofs = int(info.get("n_dofs") or pr["n_dofs"])
cmp_ = pr.get("compare", [])
worst = max(abs(c["rel_err"]) for c in cmp_) * 100
nin = sum(1 for c in cmp_ if c["in_band"])
caption = (f"{s}: harness {dofs:,} DOFs, worst certified-QoI error {worst:.2f} % "
           f"(in band {nin}/{len(cmp_)}), cuDSS on GB10")
man = {"title": caption, "placeholder": None, "assembly": "Part", "result_vtu": "r_mm.vtu", "result_json": "r.json",
       "vm_clip_percentile": 99.5,
       "parts": [{"name": s, "owner": "mesh harness (feagent automesh)", "gate": "pass",
                  "report": str(Path.home() / "gold-catalog/pilot" / s.rsplit("_bc", 1)[0] / "out/report.json")}],
       "summary": {"solver": "cuDSS on NVIDIA GB10 (DGX Spark)", "verdict": "pass" if nin == len(cmp_) else "fail",
                   "gallery_caption": caption, "worst_qoi_err_percent": round(worst, 3),
                   "qoi_in_band": f"{nin}/{len(cmp_)}", "units_note": "converted from SI (m, Pa) to mm, MPa"}}
(d / "run_manifest.json").write_text(json.dumps(man, indent=1))
subprocess.run([sys.executable, "-W", "ignore", str(Path.home() / "feagent-viz/export_twin.py"), str(d / "run_manifest.json"),
                "--out", str(d), "--name", s, "--no-usdz"], check=True)
print("[to_twin]", caption)
