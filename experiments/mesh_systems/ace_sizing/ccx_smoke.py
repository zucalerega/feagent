#!/usr/bin/env python
"""CalculiX smoke check: ccx_smoke.py OUT_DIR AXIS CCX_BIN  (tiny linear-elastic static step on OUT_DIR/mesh.inp).

mesh.inp carries harness-style surface elements (CPS6) for the bottom/top groups; CalculiX treats CPS6 as
plane-stress elements, so they are stripped for the deck (the NSETs are what the BCs use).
"""
import re, subprocess, sys
from pathlib import Path

out, axis, ccx = Path(sys.argv[1]), "xyz".index(sys.argv[2]) + 1, sys.argv[3]
text = out.joinpath("mesh.inp").read_text()
blocks = re.split(r"(?m)^(?=\*)", text)
keep = []
for b in blocks:
    if re.match(r"\*ELEMENT,\s*type=CPS", b, re.I): continue
    if re.match(r"\*ELSET,\s*ELSET=(bottom|top)\s*$", b.split("\n")[0], re.I): continue
    keep.append(b)
d = out / "ccx"; d.mkdir(exist_ok=True)
(d / "mesh_vol.inp").write_text("".join(keep))
(d / "smoke.inp").write_text(f"""*INCLUDE, INPUT=mesh_vol.inp
*MATERIAL, NAME=RESIN
*ELASTIC
1.8798E9, 0.35
*SOLID SECTION, ELSET=domain, MATERIAL=RESIN
*BOUNDARY
bottom, 1, 3, 0.
*STEP
*STATIC
*BOUNDARY
top, {axis}, {axis}, -1.E-5
*NODE PRINT, NSET=top
U
*NODE FILE
U
*END STEP
""")
r = subprocess.run([ccx, "smoke"], cwd=d, capture_output=True, text=True)
log = r.stdout + r.stderr
(d / "ccx.log").write_text(log)
print("exit", r.returncode, "| ERROR" if "*ERROR" in log else "| no errors", "| converged" if "Job finished" in log else "| NOT finished")
