#!/usr/bin/env python
"""CalculiX smoke check of mesh.inp: linear-elastic static step, fixed `bottom`, top moved 1e-5 m
along -AXIS. Prints OK + reaction when ccx finishes cleanly.   ccx_smoke.py OUT_DIR AXIS"""
import os, re, subprocess, sys
from pathlib import Path
CCX = os.path.expanduser("~/micromamba-local/envs/ccx/bin/ccx")
d, axis = Path(sys.argv[1]), "xyz".index(sys.argv[2])
w = d / "ccx_smoke"; w.mkdir(exist_ok=True)


def strip_surface(text):
    """Drop the surface-element blocks (CPS3/CPS4/S3/S4 ...: ccx would treat them as 2D elements) and the
    surface ELSETs bottom/top that point at them; keeps nodes, volume elements, NSETs and ELSET domain."""
    out, skip = [], False
    for line in text.splitlines():
        if line.startswith("*"):
            u = line.upper()
            skip = (u.startswith("*ELEMENT") and re.search(r"TYPE=(CPS|S\d|CPE|CAX|T3D|B3\d)", u.replace(" ", ""))) \
                or u.replace(" ", "").startswith(("*ELSET,ELSET=BOTTOM", "*ELSET,ELSET=TOP"))
        if not skip:
            out.append(line)
    return "\n".join(out) + "\n"


(w / "mesh.inp").write_text(strip_surface((d / "mesh.inp").read_text()))
(w / "job.inp").write_text(f"""*INCLUDE, INPUT=mesh.inp
*MATERIAL, NAME=RESIN
*ELASTIC
1879.8E6, 0.35
*SOLID SECTION, ELSET=domain, MATERIAL=RESIN
*BOUNDARY
bottom, 1, 3
*STEP
*STATIC
*BOUNDARY
top, {axis + 1}, {axis + 1}, -1.0E-5
*NODE FILE
U
*EL FILE
S
*NODE PRINT, NSET=bottom, TOTALS=ONLY
RF
*END STEP
""")
env = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
p = subprocess.run([CCX, "job"], cwd=w, env=env, capture_output=True, text=True)
log = p.stdout + p.stderr
(w / "ccx.log").write_text(log)
dat = (w / "job.dat").read_text() if (w / "job.dat").exists() else ""
errs = [l for l in log.splitlines() if "*ERROR" in l or "ERROR" in l]
print("returncode", p.returncode, "| errors:", errs[:3], "| 'Job finished':", "Job finished" in log)
print([l.strip() for l in dat.splitlines() if l.strip()][-2:])
