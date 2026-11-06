"""Peak force, travel at peak, first failure and crack height fraction (hf: 0 = base, 1 = platen) of crush.py runs
of abc_00000065; hf_neck is the first increment with failed cells in the neck wall (bore r 19.67 mm to ring r 27.77 mm,
-15 < y < 12 mm). The system label is the run dir's grandparent (RUN/systems/<id>/crush/<run-id>).

    python neck_hf.py RUN_DIR [RUN_DIR ...]      # prints one JSON line per run, then "JSON {all}"
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
import meshio
out = {}
for d in map(Path, sys.argv[1:]):
    rows = list(csv.DictReader(open(d / "force_displacement.csv")))
    F = np.array([float(r["force_N"]) for r in rows]); T = np.array([float(r["platen_travel_mm"]) for r in rows])
    nf = np.array([int(r["n_failed"]) for r in rows]); ip = int(F.argmax())
    files = sorted(d.glob("result_*.vtu"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    first = int(np.argmax(nf > 0)) if (nf > 0).any() else None
    res = {"peak_kN": F[ip] / 1e3, "travel_at_peak_mm": T[ip], "first_fail_travel_mm": T[first] if first is not None else None,
           "n_inc": len(rows), "past_peak": bool(F[-1] < 0.9 * F[ip])}
    if first is not None:
        hf_first = None; hf_neck = None
        for f in files[max(first - 2, 0):]:
            m = meshio.read(f)
            pts = m.points * (1e3 if np.ptp(m.points[:, 1]) < 1 else 1)
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
            fl = None; cents = []
            for i, cb in enumerate(m.cells):
                if cb.data.shape[1] < 4: continue
                c = pts[cb.data[:, :4 if cb.data.shape[1] in (4, 10) else 8]].mean(1)
                key = next((k for k in m.cell_data if k.lower() == "failed"), None)
                v = np.asarray(m.cell_data[key][i]).ravel() > 0.5
                cents.append(c[v])
            c = np.concatenate(cents) if cents else np.zeros((0, 3))
            if len(c) == 0: continue
            if hf_first is None: hf_first = float((c[:, 1].mean() - ymin) / (ymax - ymin))
            r = np.hypot(c[:, 0], c[:, 1]); nk = (r > 19.67) & (r < 27.77) & (c[:, 1] > -15) & (c[:, 1] < 12)
            if nk.any():
                hf_neck = float((c[nk, 1].mean() - ymin) / (ymax - ymin)); res["neck_first_file"] = f.name; break
        res["hf_first"] = hf_first; res["hf_neck"] = hf_neck
    out[d.parts[-3].replace("_n40k_imp", "")] = res
    print(d.parts[-3], json.dumps(res), flush=True)
print("JSON", json.dumps(out))
