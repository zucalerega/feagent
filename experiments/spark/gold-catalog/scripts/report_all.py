"""Aggregate ~/gold-runs/*/*/status.json -> results.csv (one row per certified QoI), cases.csv, RESULTS.md."""
from __future__ import annotations

import collections, csv, json, re, statistics, sys
from pathlib import Path

HOME = Path.home()
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HOME / "gold-runs"
GOLD = HOME / "feagent-mh" / "benchmarks" / "gold"
idx = json.loads((GOLD / "index.json").read_text())


def qtype(q: str) -> str:
    if re.match(r"^R[xyz]_f\d+$", q):
        return "reaction"
    if q.startswith("u_mean_"):
        return "u_mean_face"
    return q


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (k - lo)


cases, rows = [], []
for st in sorted(RUNS.glob("abc_*/abc_*/status.json")):
    s = json.loads(st.read_text())
    res = s.get("result") or {}
    mesh = s.get("mesh") or {}
    study = json.loads((GOLD / s["set"] / "private" / "study.json").read_text())
    ref = json.loads((GOLD / s["set"] / "private" / "reference.json").read_text())
    gd = {r["level"]: r["n_dofs"] for r in study["runs"]}
    lv = {r["level"]: r["qois"] for r in study["runs"]}
    ls = res.get("linear_solver") or []
    solver = re.search(r"linear_solver=(\w+)", ls[0]).group(1) if ls else None
    cmp = res.get("compare") or []
    c = {"set": s["set"], "part": s["part"], "status": s["status"], "stage": s.get("stage"),
         "mesh_status": mesh.get("status"), "failed_gates": " ".join(mesh.get("failed_gates") or []),
         "warn_gates": " ".join(mesh.get("warn_gates") or []), "fixes": " | ".join(mesh.get("fixes") or []),
         "faces": idx["parts"].get(s["part"], {}).get("faces"),
         "n_nodes": (mesh.get("mesh") or {}).get("n_nodes"), "harness_dofs": res.get("n_dofs"),
         "gold_L0_dofs": gd.get(0), "gold_L1_dofs": gd.get(1), "gold_L2_dofs": gd.get(2),
         "mesh_stage_s": mesh.get("mesh_stage_s"), "solve_wall_s": s.get("solve_wall_s"),
         "solver_s": res.get("solver_wall_s"), "queue_wait_s": s.get("queue_wait_s"), "linear_solver": solver,
         "peak_rss_gb": res.get("peak_rss_gb"), "est_gb": s.get("est_gb"), "mapping_ok": s.get("mapping_ok"),
         "n_refs": len(ref["references"]), "n_in_band": sum(x["in_band"] for x in cmp),
         "error": (s.get("error") or res.get("error") or "")[:300].replace("\n", " ")}
    cases.append(c)
    l2 = lv.get(2, {})
    r_scale = max([abs(v) for k, v in l2.items() if re.match(r"^R[xyz]_f\d+$", k)] or [0.0])
    for r in ref["references"]:
        x = next((y for y in cmp if y["qoi"] == r["qoi"]), {})
        e = x.get("rel_err")
        row = {"set": s["set"], "part": s["part"], "status": s["status"], "qoi": r["qoi"], "qoi_type": qtype(r["qoi"]),
               "gold": r["value"], "tol": r["tolerance"], "harness": x.get("harness"), "rel_err": e,
               "in_band": x.get("in_band") if e is not None else None, "harness_dofs": res.get("n_dofs"),
               "gold_L1_dofs": gd.get(1), "gold_L2_dofs": gd.get(2)}
        isr = qtype(r["qoi"]) == "reaction"
        row["near_zero_reaction"] = bool(isr and r_scale > 0 and abs(r["value"]) < 1e-3 * r_scale)
        row["err_over_max_reaction"] = (x["harness"] - r["value"]) / r_scale if isr and r_scale and x.get("harness") is not None else None
        for k in (0, 1, 2):
            v = lv.get(k, {}).get(r["qoi"])
            row[f"L{k}_rel_err"] = (v / r["value"] - 1) if v is not None and r["value"] else None
        rows.append(row)

with open(RUNS / "cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cases[0]))
    w.writeheader(); w.writerows(cases)
with open(RUNS / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ markdown
runs = json.loads((RUNS / "runs.json").read_text()) if (RUNS / "runs.json").is_file() else []
L = []
w = L.append
n = len(cases)
sc = collections.Counter(c["status"] for c in cases)
solved = [c for c in cases if c["harness_dofs"]]
w("# Harness vs gold: full benchmark\n")
w(f"Cases (part x BC set): {n} of {len(idx['sets'])} imported sets, {len({c['part'] for c in cases})} parts. "
  "Pipeline per part: `feagent.mesher.auto_spec` sizes + one `ids` group per gold face any set uses + `feagent check` "
  "(gate hints followed automatically, at most 3 fixes) -> per set `feacore.adapters.mesh_harness` case with the gold BCs "
  "and the gold QoI definitions -> skfem P2 solve with cuDSS -> every certified reference vs the harness value.\n")
w("## Status\n")
w("| status | cases | share |\n|---|---|---|")
for k, v in sc.most_common():
    w(f"| {k} | {v} | {v / n:.1%} |")
w("")
w("`gate_fail` = the harness mesh failed a quality gate after the automatic fixes but was still solved and scored (shown separately below).\n")
fails = [c for c in cases if c["status"] not in ("ok", "gate_fail")]
if fails:
    w("Failures by category and cause:\n")
    w("| status | stage | cause | cases |\n|---|---|---|---|")
    cause = collections.Counter((c["status"], c["stage"], re.sub(r"[-\d.e]+", "#", c["error"])[:110]) for c in fails)
    for (a, b, e), v in cause.most_common(25):
        w(f"| {a} | {b} | {e} | {v} |")
    w("")
gf = collections.Counter(g for c in cases for g in c["failed_gates"].split())
if gf:
    w(f"Failed gates after fixes (cases): {', '.join(f'{k} {v}' for k, v in gf.most_common())}. "
      f"Parts that needed fixes: {len({c['part'] for c in cases if c['fixes']})} "
      f"({collections.Counter(re.sub(r'[ ].*', '', x.strip()) for c in {c['part']: c for c in cases}.values() for x in c['fixes'].split('|') if x.strip()).most_common()}).\n")


def table(sel, title):
    w(f"### {title}\n")
    w("| QoI type | refs | scored | within certified band | median abs err | p90 abs err | max abs err | gold L1 within band | gold L1 median err |")
    w("|---|---|---|---|---|---|---|---|---|")
    groups = collections.defaultdict(list)
    for r in sel:
        groups[r["qoi_type"]].append(r)
    groups["ALL"] = list(sel)
    for k in sorted(groups, key=lambda k: (k == "ALL", k)):
        rs = groups[k]
        sc_ = [r for r in rs if r["rel_err"] is not None]
        e = [abs(r["rel_err"]) for r in sc_]
        l1 = [r for r in rs if r["L1_rel_err"] is not None]
        if not sc_:
            continue
        w(f"| {k} | {len(rs)} | {len(sc_)} | {sum(r['in_band'] for r in sc_) / len(sc_):.1%} | {statistics.median(e):.2%} | "
          f"{pct(e, 0.9):.2%} | {max(e):.1%} | {sum(abs(r['L1_rel_err']) <= r['tol'] for r in l1) / max(1, len(l1)):.1%} | "
          f"{statistics.median(abs(r['L1_rel_err']) for r in l1):.2%} |")
    w("")


w("## Accuracy vs the certified references\n")
w("Relative error = harness / gold - 1. Band = the reference's certified tolerance (max(limit, 2 x GCI)). "
  "`gold L1` columns: the generator's own uniform L1 mesh on the same QoIs, for scale.\n")
table([r for r in rows if r["status"] in ("ok", "gate_fail")], "All solved cases")
nz = [r for r in rows if r["near_zero_reaction"]]
table([r for r in rows if r["status"] in ("ok", "gate_fail") and not r["near_zero_reaction"]],
      f"All solved cases, without the {len(nz)} near-zero reaction components")
if nz:
    e = [abs(r["err_over_max_reaction"]) for r in nz if r["err_over_max_reaction"] is not None]
    w(f"Near-zero reaction components: certified reactions smaller than 0.1 % of the set's largest reaction (e.g. a ~1 N "
      f"transverse component next to ~9 kN). The generator certified them because three noisy values happened to be monotone, "
      f"but relative to the load path they are numerical noise: {len(nz)} references, {sum(bool(r['in_band']) for r in nz)} inside their "
      f"relative band; measured against the set's largest reaction their harness error is median {statistics.median(e) if e else float('nan'):.2e}, "
      f"max {max(e) if e else float('nan'):.2e}.\n")
table([r for r in rows if r["status"] == "ok"], "Cases whose mesh passed every gate")
table([r for r in rows if r["status"] == "gate_fail"], "Cases whose mesh failed a gate (solved anyway)")
cs = [c for c in solved]
if cs:
    allin = sum(c["n_in_band"] == c["n_refs"] for c in cs)
    w(f"Cases with every certified QoI inside its band: {allin}/{len(cs)} ({allin / len(cs):.1%}).\n")
    r2 = [c["harness_dofs"] / c["gold_L2_dofs"] for c in cs if c["gold_L2_dofs"]]
    r1 = [c["harness_dofs"] / c["gold_L1_dofs"] for c in cs if c["gold_L1_dofs"]]
    w("## DOFs\n")
    w(f"Harness DOFs: median {statistics.median(c['harness_dofs'] for c in cs):,.0f} (min {min(c['harness_dofs'] for c in cs):,}, "
      f"max {max(c['harness_dofs'] for c in cs):,}). Ratio to gold L1: median {statistics.median(r1):.2f} "
      f"(p10 {pct(r1, 0.1):.2f}, p90 {pct(r1, 0.9):.2f}); to gold L2: median {statistics.median(r2):.2f} "
      f"(p10 {pct(r2, 0.1):.2f}, p90 {pct(r2, 0.9):.2f}).\n")
    worst = sorted((r for r in rows if r["rel_err"] is not None), key=lambda r: -abs(r["rel_err"]))[:15]
    w("## Largest errors\n")
    w("| set | QoI | harness | gold | rel err | band | gold L1 err | harness DOFs | L2 DOFs |\n|---|---|---|---|---|---|---|---|---|")
    for r in worst:
        l1 = f"{r['L1_rel_err']:+.1%}" if r["L1_rel_err"] is not None else "-"
        w(f"| {r['set']} | {r['qoi']} | {r['harness']:.4g} | {r['gold']:.4g} | {r['rel_err']:+.1%} | {r['tol']:.1%} | {l1} | {r['harness_dofs']:,} | {r['gold_L2_dofs']:,} |")
    w("")
    w("## Resources and throughput\n")
    lsc = collections.Counter(c["linear_solver"] for c in cs)
    w(f"Linear solver actually used: {dict(lsc)}. Mesh stage per part: median {statistics.median(c['mesh_stage_s'] for c in {c['part']: c for c in cs}.values() if c['mesh_stage_s']):.0f} s, "
      f"max {max(c['mesh_stage_s'] or 0 for c in cs):.0f} s. Solve process wall per case: median {statistics.median(c['solve_wall_s'] for c in cs if c['solve_wall_s']):.1f} s, "
      f"max {max(c['solve_wall_s'] or 0 for c in cs):.0f} s. Peak host RSS per solve: median {statistics.median(c['peak_rss_gb'] for c in cs if c['peak_rss_gb']):.1f} GB, "
      f"max {max(c['peak_rss_gb'] or 0 for c in cs):.1f} GB (GPU-side memory on the unified-memory GB10 is not in RSS).\n")
for r in runs:
    w(f"- run `{r.get('label')}` started {r.get('started')}: {r.get('cases')} cases on {r.get('parts')} parts in {r.get('wall_s', 0) / 60:.1f} min "
      f"({r.get('cases_per_min')} cases/min; {r.get('mesh_workers')} mesh workers, <= {r.get('max_solves')} concurrent solves); "
      f"GPU utilisation (nvidia-smi, 5 s samples) mean {r.get('gpu_util_mean_all')} % overall, {r.get('gpu_util_mean_while_solving')} % while a solve ran.")
w("")
w("Files: `results.csv` (one row per certified reference QoI: harness, gold, rel_err, tol, in_band, DOFs, and the generator's L0-L2 errors), "
  "`cases.csv` (one row per case: status, gates, fixes, DOFs, timings, solver, memory, error), per case `<part>/<set>/status.json`, "
  "per part `<part>/mesh.yaml`, `<part>/out/report.json`, `<part>/mesh_status.json`.")
(RUNS / "RESULTS.md").write_text("\n".join(L) + "\n")
print(f"{n} cases, {len(rows)} QoI rows; status {dict(sc)}")
