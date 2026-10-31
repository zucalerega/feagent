"""Alpha Fast Resin (2-min cure) material from dogbone1.pdf (Instron, crosshead displacement).
Assumes ASTM D638 Type I (13.0 x 3.2 mm section, 115 mm grip separation), per physical_tests/README.md.
Curve digitized from the PDF plot; table points read by eye (+-15 N, +-0.05 mm)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent            # outputs land next to this script
A, L = 13.0 * 3.2, 115.0                        # mm^2, mm
d = np.array([0, .25, .5, .75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 6.86, 7.5, 8.0, 8.45])
F = np.array([0, 180, 340, 490, 640, 760, 880, 1070, 1250, 1420, 1580, 1720, 1840, 1940, 2010, 2070, 2090, 2100,
              2085, 2065, 2040])
d_break, F_break = 8.52, 2040.0
s_eng, e_eng = F / A, d / L                     # MPa, -
E = float(np.polyfit(e_eng[:3], s_eng[:3], 1)[0])   # apparent modulus (crosshead: includes grip/frame compliance)
i_max = int(np.argmax(F))                        # beyond max force the dogbone necks: true-stress conversion invalid
s_true, e_true = s_eng[: i_max + 1] * (1 + e_eng[: i_max + 1]), np.log1p(e_eng[: i_max + 1])
e_pl = np.maximum(e_true - s_true / E, 0.0)
keep = np.concatenate([[True], np.diff(e_pl) > 1e-5])
table = [[float(s), float(p)] for s, p in zip(s_true[keep], e_pl[keep]) if p > 0 or s == s_true[keep][1]]
e_f_true = float(np.log1p(d_break / L)); peeq_f = e_f_true - s_true[-1] / E
mat = {"name": "Formlabs Alpha Fast Resin, 2 min cure (dogbone1.pdf)", "units": "MPa, mm/mm",
       "E": round(E, 1), "nu": 0.35, "nu_note": "not measured; assumed",
       "plastic_table_true_stress_vs_plastic_strain": table, "uts_eng": round(float(s_eng.max()), 2),
       "failure_peeq": round(float(peeq_f), 4), "failure_true_strain": round(e_f_true, 4),
       "caveats": ["specimen geometry assumed (ASTM D638 Type I); not in the PDF",
                   "strain from crosshead over 115 mm: includes grip and frame compliance (E is apparent, low)",
                   "post-necking curve not used for hardening; failure = plastic strain at break (approximate)"]}
json.dump(mat, open(OUT / "alpha_fast_dogbone1.json", "w"), indent=1)
# 1D check: rebuild the engineering curve from the table
sp, pp = np.array(table)[:, 0], np.array(table)[:, 1]
st = np.linspace(0, sp.max(), 200); ep = np.interp(st, sp, pp, left=0); et = st / E + ep
se, ee = st / np.exp(et), np.expm1(et)
plt.figure(figsize=(7, 4.5)); plt.plot(e_eng * 100, s_eng, "o", label="dogbone1 (digitized)")
plt.plot(ee * 100, se, "-", label=f"model: E={E:.0f} MPa + plastic table")
plt.axvline(d_break / L * 100, ls="--", c="r", label=f"break (model failure at PEEQ {peeq_f:.3f})")
plt.xlabel("engineering strain (%)"); plt.ylabel("engineering stress (MPa)"); plt.legend(); plt.grid(alpha=.3)
plt.title("Alpha Fast Resin (2 min cure): dogbone vs material model"); plt.tight_layout()
plt.savefig(OUT / "alpha_fast_dogbone1_fit.png", dpi=130)
print(json.dumps({k: mat[k] for k in ("E", "uts_eng", "failure_peeq", "failure_true_strain")}), len(table), "table rows")
