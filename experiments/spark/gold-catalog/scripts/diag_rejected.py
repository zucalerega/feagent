"""Why did gold.py import reject a part's STEP? Re-import each rejected part's STEP as GoldProblem does and
report the worst face area / centroid deviation (gold.py's limit: 1e-6 relative)."""
import json, re, sys, tarfile, tempfile
from pathlib import Path
import numpy as np

A = Path("/media/dell/jaimesThumbdrive/gold_benchmarks/archives")
rej = {}
for log in sorted((Path.home() / "gold-catalog/import").glob("goldgen-*.log")):
    for line in log.read_text().splitlines():
        m = re.match(r"\s*SKIP (abc_\d+)_bc\d+: STEP", line)
        if m:
            rej.setdefault(m.group(1), log.stem)
out = {}
import gmsh
from feacore.geometry.gmsh_tools import gmsh_session
from feacore.problems.gold import GoldProblem
for part, arch in rej.items():
    with tempfile.TemporaryDirectory(dir=Path.home() / "gold-catalog") as tmp, tarfile.open(A / f"{arch}.tgz") as tf:
        mem = [m for m in tf.getmembers() if re.match(rf"out/{part}_bc\d+/", m.name)]
        sid = sorted({m.name.split("/")[1] for m in mem})[0]
        tf.extractall(tmp, members=[m for m in mem if m.name.startswith(f"out/{sid}/")], filter="data")
        p = GoldProblem(Path(tmp) / "out" / sid)
        faces = {f"f{f['tag']}": f for f in p.part["faces"]}
        diag = float(np.linalg.norm(np.subtract(p.part["bbox"][3:], p.part["bbox"][:3])))
        with gmsh_session(part):
            p.build_geometry()
            got = {}
            for dim, pt in gmsh.model.getPhysicalGroups(2):
                (tag,) = gmsh.model.getEntitiesForPhysicalGroup(dim, pt)
                got[gmsh.model.getPhysicalName(dim, pt)] = (gmsh.model.occ.getMass(2, int(tag)), gmsh.model.occ.getCenterOfMass(2, int(tag)))
        ea = {g: abs(got[g][0] / f["area"] - 1) for g, f in faces.items() if g in got}
        ec = {g: float(np.linalg.norm(np.subtract(got[g][1], f["centroid"]))) / diag for g, f in faces.items() if g in got}
        wa, wc = max(ea, key=ea.get), max(ec, key=ec.get)
        out[part] = {"archive": arch, "faces": len(faces), "step_faces": len(got), "same_names": set(got) == set(faces),
                     "worst_area_rel": [wa, ea[wa], faces[wa]["kind"]], "worst_centroid_rel_diag": [wc, ec[wc], faces[wc]["kind"]],
                     "n_faces_area_gt_1e-6": sum(v > 1e-6 for v in ea.values()), "n_faces_area_gt_1e-3": sum(v > 1e-3 for v in ea.values()),
                     "n_faces_centroid_gt_1e-6": sum(v > 1e-6 for v in ec.values()), "n_faces_centroid_gt_1e-3": sum(v > 1e-3 for v in ec.values())}
        print(part, json.dumps(out[part]), flush=True)
(Path.home() / "gold-catalog/import/rejected_parts.json").write_text(json.dumps(out, indent=1))
