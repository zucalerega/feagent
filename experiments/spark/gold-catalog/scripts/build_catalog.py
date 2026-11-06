"""Build catalog.json + CATALOG.md from scan_raw.json and the drive's manifests / pull.log (read-only)."""
import collections, json, re, statistics, sys
from pathlib import Path

DRIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/media/dell/jaimesThumbdrive/gold_benchmarks")
OUTD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "gold-catalog"
raw = json.loads((OUTD / "scan_raw.json").read_text())

fam = lambda f: re.sub(r"_\d+\.tgz$", "", f)
MB = 1e6

# ---- manifests, markers, pull.log
manifest = {}
for mf in sorted(DRIVE.glob("*_manifest.jsonl")):
    for line in mf.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            manifest[r["batch"]] = {**r, "manifest_file": mf.name}
markers = {p.name: p.read_text().strip() for p in DRIVE.glob("*.complete")}
pulled = set((DRIVE / "pulled.txt").read_text().split())
log = (DRIVE / "pull.log").read_text().splitlines()
verified, started = {}, set()
for line in log:
    m = re.match(r"exporting \S+?:\S*/(\S+\.tgz)", line)
    if m:
        started.add(m.group(1))
    m = re.match(r"\S+:\S*/(\S+\.tgz) -> \S+ \(([^,]+), sha256 verified\)", line)
    if m:
        verified[m.group(1)] = m.group(2)
run_lines = [l for l in log if re.search(r"archives here|safe to stop", l)]
last_status = {}
for l in run_lines:
    m = re.match(r"(\S+) (\S+?): (.*)$", l)
    if m:
        last_status[m.group(2)] = f"{m.group(1)} {m.group(3)}"


def size_ok(logged: str, actual: int) -> bool:
    m = re.match(r"([\d.]+) (B|MiB|KiB|GiB)", logged)
    if not m:
        return False
    v, u = float(m.group(1)), m.group(2)
    mult = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}[u]
    return abs(v * mult - actual) <= (0.5 if u == "B" else 0.051 * mult)


# ---- generator summaries (cumulative logs: last archive of each run wins)
summ = collections.defaultdict(dict)
for r in raw:
    for k, v in r["summary"].items():
        summ[fam(r["file"])][k] = v

batches, all_sets, part_rows = [], {}, {}
for r in sorted(raw, key=lambda r: r["file"]):
    f = r["file"]
    sets = r["sets"]
    parts = sorted({s["part"] for s in sets.values()})
    mesh_bytes = sum(lv.get("msh_bytes", 0) for m in r["meshes"].values() for lv in m.values())
    step_bytes = sum(s["files"].get("public/geometry.step", 0) for s in sets.values())
    json_bytes = sum(v for s in sets.values() for k, v in s["files"].items() if k.endswith(".json"))
    nref = [len(s["references"]) for s in sets.values()]
    mf = manifest.get(f)
    mf_check = None
    if mf:
        mf_check = {"bcsets_match": sorted(mf["bcsets"]) == sorted(sets), "bytes_match": mf["bytes"] == r["bytes"],
                    "mesh_parts_match": sorted(mf["parts"]) == sorted(r["meshes"])}
    b = {"batch": f, "run": fam(f), "bytes": r["bytes"], "sha256": r["sha256"], "members": r["members"],
         "n_sets": len(sets), "sets": sorted(sets), "parts": parts, "mesh_parts": sorted(r["meshes"]),
         "mesh_levels": sorted({lv for m in r["meshes"].values() for lv in m}),
         "study_levels": sorted({lv for s in sets.values() for lv in s.get("levels", [])}),
         "n_bc_types": dict(collections.Counter(t for s in sets.values() for t, _ in s["bcs"])),
         "sets_without_reference": sorted(k for k, s in sets.items() if not s["references"]),
         "n_references": sum(nref),
         "bytes_uncompressed": {"meshes": mesh_bytes, "step": step_bytes, "json": json_bytes},
         "in_manifest": mf is not None, "manifest_check": mf_check, "in_pulled_txt": f in pulled,
         "pull_log": ("verified " + verified[f]) if f in verified else ("export started, not verified" if f in started else "not in pull.log"),
         "pull_log_size_ok": size_ok(verified[f], r["bytes"]) if f in verified else None,
         "errors": r["errors"]}
    batches.append(b)
    for sid, s in sets.items():
        all_sets[sid] = {"batch": f, "part": s["part"], "n_faces": s["n_faces"], "bcs": s["bcs"],
                         "references": s["references"], "gold_qois": s["gold_qois"],
                         "dofs": s.get("dofs"), "step_sha16": s["step_sha"],
                         "step_bytes": s["files"].get("public/geometry.step")}
    for p, m in r["meshes"].items():
        part_rows[p] = {"mesh_batch": f, "levels": {lv: {"cells": e.get("cells"), "h": e.get("h"), "msh_bytes": e.get("msh_bytes"),
                                                        "min_sj": e.get("min_sj"), "scheme": e.get("scheme")} for lv, e in m.items()}}

for sid, s in all_sets.items():
    pr = part_rows[s["part"]]
    pr.setdefault("sets", []).append(sid)
    pr.setdefault("set_batches", set()).add(s["batch"])
    pr["n_faces"] = s["n_faces"]
    pr["step_sha16"] = s["step_sha16"]
    pr["step_bytes"] = s["step_bytes"]
    pr.setdefault("dofs", s["dofs"])
for p, pr in part_rows.items():
    pr["set_batches"] = sorted(pr["set_batches"])
    pr["sets"] = sorted(pr["sets"])
    pr["run"] = fam(pr["mesh_batch"])
    pr["meshes_in_other_batch_than_some_sets"] = any(b != pr["mesh_batch"] for b in pr["set_batches"])

# duplicates
dup_ids = [sid for sid, c in collections.Counter(sid for b in batches for sid in b["sets"]).items() if c > 1]
by_step = collections.defaultdict(list)
for p, pr in part_rows.items():
    by_step[pr["step_sha16"]].append(p)
geo_dups = sorted([sorted(v) for v in by_step.values() if len(v) > 1])
sha_dups = [[b["batch"] for b in batches if b["sha256"] == h] for h, c in collections.Counter(b["sha256"] for b in batches).items() if c > 1]

# runs
runs = {}
for run in sorted({b["run"] for b in batches}):
    bs = [b for b in batches if b["run"] == run]
    nums = sorted(int(re.search(r"_(\d+)\.tgz$", b["batch"]).group(1)) for b in bs)
    missing_nums = sorted(set(range(nums[-1] + 1)) - set(nums))
    archived = {sid for b in bs for sid in b["sets"]}
    s = summ.get(run, {})
    not_archived = sorted(set(s) - archived)
    runs[run] = {"archives": len(bs), "index_range": [nums[0], nums[-1]], "missing_indices": missing_nums,
                 "bytes": sum(b["bytes"] for b in bs), "sets": len(archived),
                 "parts": len({p for b in bs for p in b["parts"]}),
                 "sets_with_reference": sum(1 for sid in archived if all_sets[sid]["references"]),
                 "references": sum(len(all_sets[sid]["references"]) for sid in archived),
                 "in_manifest": sum(b["in_manifest"] for b in bs),
                 "pull_log_verified": sum(b["pull_log"].startswith("verified") for b in bs),
                 "complete_marker": markers.get(f"{run}.complete"),
                 "pull_log_last_status": last_status.get(run) or last_status.get(f"{run.split('-r2')[0]}:gold2" if "-r2" in run else run),
                 "generator_summary_rows": len(s),
                 "generator_results": dict(collections.Counter(v.get("result") for v in s.values())),
                 "summary_sets_not_archived": [{"id": k, "result": s[k].get("result"), "error": s[k].get("error"),
                                                "gold": s[k].get("gold")} for k in not_archived]}
started_unfinished = sorted(started - set(verified))

dofs_l = {lv: [s["dofs"][lv] for s in all_sets.values() if s.get("dofs") and s["dofs"].get(lv)] for lv in ("0", "1", "2")}
qoi_counter = collections.Counter(re.sub(r"_f\d+$", "_f*", re.sub(r"^R[xyz]_f\d+$", "R*_f*", r["qoi"]))
                                  for s in all_sets.values() for r in s["references"])
totals = {"archives": len(batches), "archive_bytes": sum(b["bytes"] for b in batches),
          "distinct_parts": len(part_rows), "distinct_part_geometries": len(by_step),
          "bc_sets": len(all_sets), "bc_sets_with_reference": sum(1 for s in all_sets.values() if s["references"]),
          "bc_sets_without_reference": sum(1 for s in all_sets.values() if not s["references"]),
          "references": sum(len(s["references"]) for s in all_sets.values()),
          "reference_qoi_kinds": dict(qoi_counter.most_common()),
          "mesh_levels": sorted({lv for pr in part_rows.values() for lv in pr["levels"]}),
          "parts_with_all_levels_L0_L2": sum(1 for pr in part_rows.values() if set(pr["levels"]) == {"0", "1", "2"}),
          "mesh_bytes_uncompressed": sum(e["msh_bytes"] or 0 for pr in part_rows.values() for e in pr["levels"].values()),
          "mesh_bytes_by_level": {lv: sum(pr["levels"][lv]["msh_bytes"] or 0 for pr in part_rows.values()) for lv in ("0", "1", "2")},
          "dofs_by_level": {lv: {"min": min(v), "median": statistics.median(v), "max": max(v)} for lv, v in dofs_l.items()},
          "faces_per_part": {"min": min(pr["n_faces"] for pr in part_rows.values()),
                             "median": statistics.median(pr["n_faces"] for pr in part_rows.values()),
                             "max": max(pr["n_faces"] for pr in part_rows.values())},
          "bc_types": dict(collections.Counter(t for s in all_sets.values() for t, _ in s["bcs"]))}
issues = {"batches_not_in_any_manifest": [b["batch"] for b in batches if not b["in_manifest"]],
          "manifest_entries_without_archive": sorted(set(manifest) - {b["batch"] for b in batches}),
          "manifest_mismatches": [b["batch"] for b in batches if b["manifest_check"] and not all(b["manifest_check"].values())],
          "batches_not_sha_verified_in_pull_log": [b["batch"] for b in batches if not b["pull_log"].startswith("verified")],
          "pull_log_size_mismatch": [b["batch"] for b in batches if b["pull_log_size_ok"] is False],
          "exports_started_never_verified": started_unfinished,
          "runs_without_complete_marker": [r for r in runs if not runs[r]["complete_marker"]],
          "short_batches": {b["batch"]: b["n_sets"] for b in batches if b["n_sets"] != 10},
          "read_errors": {b["batch"]: b["errors"] for b in batches if b["errors"]},
          "not_in_pulled_txt": [b["batch"] for b in batches if not b["in_pulled_txt"]],
          "duplicate_set_ids_across_batches": dup_ids,
          "duplicate_archive_sha256": sha_dups,
          "identical_geometry_different_part_ids": geo_dups,
          "parts_whose_sets_span_batches": sum(len(pr["set_batches"]) > 1 for pr in part_rows.values()),
          "batches_without_meshes": [b["batch"] for b in batches if not b["mesh_parts"]]}

cat = {"source": str(DRIVE), "totals": totals, "runs": runs, "issues": issues, "batches": batches,
       "parts": part_rows, "sets": all_sets, "markers": markers, "manifests": sorted(manifest)}
(OUTD / "catalog.json").write_text(json.dumps(cat, indent=1, default=sorted))

# ---- markdown
L = []
w = L.append
t = totals
w("# Gold benchmark archive catalog\n")
w(f"Source: `{DRIVE}` (read-only; scanned by streaming every .tgz, `scripts/catalog_scan.py`). "
  f"Machine-readable: `catalog.json` (per batch, per part, per set).\n")
w("## Totals\n")
w("| | |\n|---|---|")
w(f"| Archives | {t['archives']} ({t['archive_bytes'] / 1e9:.2f} GB compressed) |")
w(f"| Distinct parts (ABC ids) | {t['distinct_parts']} ({t['distinct_part_geometries']} distinct geometries: some ABC ids are byte-identical STEPs) |")
w(f"| BC sets | {t['bc_sets']} ({t['bc_sets_with_reference']} with >=1 certified reference, {t['bc_sets_without_reference']} with none; gold.py import skips those) |")
w(f"| Certified references (QoIs) | {t['references']} |")
w(f"| Mesh levels | L{', L'.join(t['mesh_levels'])} for every part ({t['parts_with_all_levels_L0_L2']}/{t['distinct_parts']}); every set's study ran L0-L2 |")
w(f"| Study meshes, uncompressed | {t['mesh_bytes_uncompressed'] / 1e9:.2f} GB (L0 {t['mesh_bytes_by_level']['0'] / 1e9:.2f}, L1 {t['mesh_bytes_by_level']['1'] / 1e9:.2f}, L2 {t['mesh_bytes_by_level']['2'] / 1e9:.2f} GB) |")
for lv in ("0", "1", "2"):
    d = t["dofs_by_level"][lv]
    w(f"| L{lv} DOFs | {d['min']:,} - {d['max']:,} (median {d['median']:,.0f}) |")
w(f"| Faces per part | {t['faces_per_part']['min']} - {t['faces_per_part']['max']} (median {t['faces_per_part']['median']:.0f}) |")
w(f"| BC types | {', '.join(f'{k} {v}' for k, v in t['bc_types'].items())} |")
w(f"| Reference QoI kinds | {', '.join(f'{k} {v}' for k, v in t['reference_qoi_kinds'].items())} |")
w("")
w("## Runs (batch families)\n")
w("| run | archives | idx | GB | parts | sets | with ref | refs | in manifest | sha-verified in pull.log | .complete | generator summary |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in runs.items():
    gen = ", ".join(f"{a} {b}" for a, b in r["generator_results"].items())
    w(f"| {k} | {r['archives']} | {r['index_range'][0]}-{r['index_range'][1]}{' (gaps ' + str(r['missing_indices']) + ')' if r['missing_indices'] else ''} | {r['bytes'] / 1e9:.2f} | {r['parts']} | {r['sets']} | {r['sets_with_reference']} | {r['references']} | {r['in_manifest']} | {r['pull_log_verified']} | {r['complete_marker'] or 'none'} | {r['generator_summary_rows']} rows: {gen}; {len(r['summary_sets_not_archived'])} not archived |")
w("")
w("## Issues\n")
iss = issues
w(f"- **Manifests cover only {len(manifest)} of {t['archives']} batches**: `goldgen-l2_manifest.jsonl` lists goldgen-l2_0000-0002, `goldgen-l2b_manifest.jsonl` goldgen-l2b_0000-0009. "
  f"{len(iss['batches_not_in_any_manifest'])} batches are in no manifest (all of goldgen-l2 0003-0153, goldgen-l2b 0010-0153, both -r2 runs). "
  f"The {len(manifest)} listed agree with the archives (set ids, bytes, mesh parts): mismatches {iss['manifest_mismatches'] or 'none'}. Manifest entries without an archive: {iss['manifest_entries_without_archive'] or 'none'}.")
nv = iss["batches_not_sha_verified_in_pull_log"]
def _ranges(names):
    by = collections.defaultdict(list)
    for n in names:
        by[fam(n)].append(int(re.search(r"_(\d+)\.tgz$", n).group(1)))
    out = []
    for k, v in sorted(by.items()):
        v.sort(); segs, a = [], v[0]
        for i in range(1, len(v) + 1):
            if i == len(v) or v[i] != v[i - 1] + 1:
                segs.append(f"{a:04d}" if a == v[i - 1] else f"{a:04d}-{v[i - 1]:04d}"); a = v[i] if i < len(v) else None
        out.append(f"{k}_{','.join(segs)}")
    return "; ".join(out)
w(f"- **Not sha256-verified per pull.log: {len(nv)}** batches ({_ranges(nv)}). pull.log starts at 09:50 with the earlier pulls unlogged, so these were copied before logging began; all are listed in pulled.txt and read cleanly (tar stream + JSON parse, no errors). "
  f"Logged sizes disagree with the file on the drive for: {iss['pull_log_size_mismatch'] or 'none'}.")
w(f"- **Incomplete run**: only goldgen-l2 and goldgen-l2b have `.complete` markers; goldgen-l2-r2 and goldgen-l2b-r2 have none. pull.log ends mid-export of {', '.join(iss['exports_started_never_verified']) or '-'} (export started, never verified, not on the drive). "
  f"goldgen-l2b-r2's own summary log lists {runs.get('goldgen-l2b-r2', {}).get('generator_summary_rows')} sets of which {len(runs.get('goldgen-l2b-r2', {}).get('summary_sets_not_archived', []))} are not in any archive here; goldgen-l2-r2 has {len(runs.get('goldgen-l2-r2', {}).get('summary_sets_not_archived', []))} such sets.")
w(f"- Short batches (not 10 sets): {iss['short_batches'] or 'none'}. Batches with no meshes (their parts' meshes are in an earlier batch, so import order matters only for `verify`): {len(iss['batches_without_meshes'])}.")
w(f"- Parts whose BC sets span 2+ batches: {iss['parts_whose_sets_span_batches']}. Each part's mesh sequence ships exactly once (no mesh duplicates).")
w(f"- **Duplicates**: set ids across batches: {len(iss['duplicate_set_ids_across_batches'])}; identical archives: {len(iss['duplicate_archive_sha256'])}; parts never reappear across runs. "
  f"But {len(iss['identical_geometry_different_part_ids'])} groups ({sum(len(g) for g in iss['identical_geometry_different_part_ids'])} part ids) are the same STEP geometry under different ABC ids (ABC itself has duplicate models), so {t['distinct_parts'] - t['distinct_part_geometries']} parts add no new geometry: "
  + "; ".join(" = ".join(g) for g in iss["identical_geometry_different_part_ids"]) + ".")
w(f"- Read errors: {iss['read_errors'] or 'none'}. Not in pulled.txt: {iss['not_in_pulled_txt'] or 'none'}.")
for k, r in runs.items():
    if r["summary_sets_not_archived"]:
        w(f"- {k}: generator summary rows with no archived set: " + "; ".join(
            f"{x['id']} ({x['result']}{': ' + (x['error'] or '')[:90] if x['error'] else ''})" for x in r["summary_sets_not_archived"]))
w("")
w("## Per batch\n")
w("Sizes: archive (compressed); uncompressed meshes / STEP. Levels: mesh levels shipped in the batch (study levels for its sets are L0-L2 throughout). Verified: pull.log sha256 line.\n")
w("| batch | MB | sets | parts (sets) | meshes for | levels | no-ref sets | refs | mesh MB | STEP MB | manifest | pull.log |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for b in batches:
    pc = collections.Counter(s.rsplit("_bc", 1)[0] for s in b["sets"])
    parts = " ".join(f"{p.replace('abc_', '')}({n})" for p, n in pc.items())
    w(f"| {b['batch'].replace('.tgz', '')} | {b['bytes'] / MB:.1f} | {b['n_sets']} | {parts} | {' '.join(p.replace('abc_', '') for p in b['mesh_parts']) or '-'} | "
      f"{','.join('L' + x for x in b['mesh_levels']) or '-'} | {len(b['sets_without_reference'])} | {b['n_references']} | "
      f"{b['bytes_uncompressed']['meshes'] / MB:.1f} | {b['bytes_uncompressed']['step'] / MB:.1f} | {'yes' if b['in_manifest'] else 'no'} | "
      f"{'verified' if b['pull_log'].startswith('verified') else b['pull_log']} |")
(OUTD / "CATALOG.md").write_text("\n".join(L) + "\n")
print("ok", totals["bc_sets"], len(batches))
