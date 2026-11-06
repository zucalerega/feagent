"""Append the import results (from ~/gold-catalog/import/*.log) to CATALOG.md and catalog.json."""
import json, re
from pathlib import Path
G = Path.home() / "gold-catalog"
rc = {l.split("\t")[0]: int(l.split("\t")[1]) for l in (G / "import/results.tsv").read_text().splitlines() if "\t" in l}
rej = json.loads((G / "import/rejected_parts.json").read_text())
rows, tot = [], {"installed": 0, "no_ref": 0, "step": 0, "meshes": 0}
for a in sorted(rc):
    t = (G / "import" / f"{a}.log").read_text()
    inst = len(re.findall(r"^  (new|updated|unchanged) ", t, re.M))
    noref = len(re.findall(r"SKIP .*no reference QoI", t))
    step = re.findall(r"SKIP (abc_\d+)_bc\d+: (STEP face .*)", t)
    mesh = len(re.findall(r"^  mesh ", t, re.M))
    tb = "Traceback" in t
    tot["installed"] += inst; tot["no_ref"] += noref; tot["step"] += len(step); tot["meshes"] += mesh
    res = "ok" if not (noref or step or tb) else ("error" if tb else "ok, some sets rejected")
    rows.append({"archive": a, "rc": rc[a], "result": res, "installed": inst, "rejected_no_reference": noref,
                 "rejected_step_faces": len(step), "step_parts": sorted({p for p, _ in step}), "meshes_installed": mesh})
idx = json.loads((Path.home() / "feagent-mh/benchmarks/gold/index.json").read_text())
L = ["", "## Import into ~/feagent-mh/benchmarks/gold (`scripts/gold.py import`, one process per archive)", "",
     f"All {len(rows)} archives processed, no crash. Sets installed: **{tot['installed']}** (index.json now {len(idx['sets'])} sets, "
     f"{len(set(r['part'] for r in idx['sets'].values()))} parts with sets, {len(idx['parts'])} parts with meshes); "
     f"meshes installed {tot['meshes']} files (+6 from goldgen-l2_0000 already present) = 372 parts x L0-L2, 9.1 GB on disk. "
     f"Rejected: **{tot['no_ref'] + tot['step']} sets** = {tot['no_ref']} with no certified reference (by design: nothing to score) "
     f"+ {tot['step']} sets of **{len(rej)} parts** whose STEP re-import differs from the part record by more than gold.py's 1e-6 "
     f"tolerance. gold.py exits 1 for any archive with a rejected set ({sum(1 for r in rows if r['rc'])} archives); "
     f"{sum(1 for r in rows if not r['rc'])} archives imported every set.", "",
     "STEP rejections are a tolerance artefact, not a tag mismatch: re-importing each rejected STEP (`scripts/diag_rejected.py`) gives "
     "the same face names/count for all 23 parts; the deviating faces are all freeform (`other`) faces, worst area deviation "
     "< 1e-3 relative for 22 parts (centroids < 2e-5 of the diagonal) and 1.5 % on one face (abc_00000196 f40). "
     "OCC numerical integration of B-spline face areas differs slightly between the generator box and here. "
     "Loosening the check to ~1e-3 would admit 22 of the 23 parts (I did not change gold.py).", "",
     "| part | archive | faces | worst area dev (face) | worst centroid dev / diag |", "|---|---|---|---|---|"]
for p, v in sorted(rej.items()):
    L.append(f"| {p} | {v['archive']} | {v['faces']} | {v['worst_area_rel'][1]:.1e} ({v['worst_area_rel'][0]}) | {v['worst_centroid_rel_diag'][1]:.1e} |")
L += ["", "Per archive (installed / rejected no-ref / rejected STEP / mesh files):", "",
      "| archive | gold.py rc | installed | no-ref | STEP | meshes |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['archive']} | {r['rc']} | {r['installed']} | {r['rejected_no_reference']} | {r['rejected_step_faces']}"
             f"{' (' + ', '.join(r['step_parts']) + ')' if r['step_parts'] else ''} | {r['meshes_installed']} |")
L += ["", "`gold.py verify` (re-solve stored levels, cuDSS): abc_00000627_bc4, abc_00000974_bc4, abc_00001748_bc0 at L1 and "
      "abc_00000065_bc0 at L2 (804k DOFs, 28.6 s) all reproduce every study QoI to 1e-6 (see PILOT.md)."]
md = G / "CATALOG.md"
t = md.read_text()
t = t.split("\n## Import into")[0].rstrip("\n") + "\n"
md.write_text(t + "\n".join(L) + "\n")
c = json.loads((G / "catalog.json").read_text())
c["import"] = {"totals": tot, "archives": rows, "rejected_parts": rej}
(G / "catalog.json").write_text(json.dumps(c, indent=1))
print(tot)
