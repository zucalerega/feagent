"""Tables for PILOT.md from ~/gold-catalog/pilot/<part>/ (mesh_status.json, auto/out/report.json, <set>/result.json)."""
import json, re
from pathlib import Path

HOME = Path.home()
P = HOME / "gold-catalog" / "pilot"
GOLD = HOME / "feagent-mh" / "benchmarks" / "gold"
SETS = ["abc_00000627_bc4", "abc_00000974_bc4", "abc_00001748_bc0", "abc_00000330_bc1", "abc_00000264_bc9", "abc_00000223_bc8"]
tsv = {l.split("\t")[0]: l.split("\t") for l in (P / "pilot.tsv").read_text().splitlines() if "\t" in l}
tsv.setdefault("abc_00000264_bc9", ["", "0", "0", "0", "104", "1", "23"])   # run by hand (see PILOT.md): mesh stage, case, solve s
L = []
w = L.append
summ = []
for sid in SETS:
    part = sid.rsplit("_bc", 1)[0]
    d = P / part
    ref = json.loads((GOLD / sid / "private" / "reference.json").read_text())
    study = json.loads((GOLD / sid / "private" / "study.json").read_text())
    ms = json.loads((d / "mesh_status.json").read_text()) if (d / "mesh_status.json").is_file() else {}
    auto = json.loads((d / "auto" / "out" / "report.json").read_text()) if (d / "auto" / "out" / "report.json").is_file() else None
    res = json.loads((d / sid / "result.json").read_text()) if (d / sid / "result.json").is_file() else {}
    mp = json.loads((d / sid / "mapping.json").read_text()) if (d / sid / "mapping.json").is_file() else {}
    gd = {r["level"]: r["n_dofs"] for r in study["runs"]}
    lv = {r["level"]: r["qois"] for r in study["runs"]}
    w(f"### {sid} ({len(ref['part']['faces'])} faces, STEP {(GOLD / sid / 'public' / 'geometry.step').stat().st_size / 1e3:.0f} kB)\n")
    w("BCs: " + "; ".join(ref["bcset"]["words"]).replace("|", "/") + "\n")
    w("Group mapping (`ids: [tag]`, expect 1; harness mesh group vs gold face): " + ", ".join(
        f"{g} area {v.get('area_rel_err', float('nan')):+.1e}, centroid {v.get('centroid_err_rel_diag', float('nan')):.1e} diag"
        + ("" if v.get("ok") else " **MISMATCH**") for g, v in mp.items()) + "\n")
    if auto:
        fa = [g["id"] for g in auto["gates"] if g["status"] == "fail"]
        w(f"`feagent automesh`: status {auto['status']}{' (failed: ' + ', '.join(fa) + ')' if fa else ''}, "
          f"{auto['mesh'].get('n_nodes')} nodes / {auto['mesh'].get('n_elements')} tets, {ms.get('automesh_s')} s.")
    if ms.get("gates"):
        gl = ", ".join(f"{k} {v[0]}" + (f" ({v[1]:.3g} vs {v[2]})" if v[0] != "pass" and v[1] is not None else "") for k, v in ms["gates"].items())
        w(f"`feagent check` (spec = automesh sizes + groups + bcs): **{ms.get('gate_status')}**, {ms['mesh'].get('n_nodes')} nodes, "
          f"{ms['mesh'].get('n_elements')} tets, {ms.get('check_s')} s. Gates: {gl}.")
    else:
        w(f"`feagent check`: {ms.get('status')} {ms.get('error', '')}")
    if res.get("status") == "ok":
        w(f"Solve: {res['n_dofs']:,} DOFs (gold L0 {gd.get(0):,}, L1 {gd.get(1):,}, L2 {gd.get(2):,}; harness/L2 = {res['n_dofs'] / gd[2]:.2f}), "
          f"{res['linear_solver'][0].replace('info: ', '')}, solve() {res['solver_wall_s']:.1f} s, process peak RSS {res['peak_rss_gb']:.1f} GB.\n")
    else:
        w(f"Solve: {res.get('status')} {res.get('error', '')}\n")
    w("| QoI | harness | gold reference | rel err | certified band | inside? | gold L1 err | gold L2 err |")
    w("|---|---|---|---|---|---|---|---|")
    n_in = 0
    for c in res.get("compare", []):
        l1 = lv.get(1, {}).get(c["qoi"]); l2 = lv.get(2, {}).get(c["qoi"])
        n_in += c["in_band"]
        w(f"| {c['qoi']} | {c['harness']:.5g} | {c['gold']:.5g} | {c['rel_err']:+.2%} | ±{c['tol']:.2%} | {'yes' if c['in_band'] else '**no**'} | "
          f"{(l1 / c['gold'] - 1):+.2%} | {(l2 / c['gold'] - 1):+.2%} |")
    sing = [k for k, v in study["qois"].items() if v.get("kind") == "singular"]
    w(f"\nSingular (no reference, not scored): {', '.join(sing) or '-'}.\n")
    summ.append((sid, len(ref["part"]["faces"]), ms.get("gate_status"), ms.get("failed_gates"), res.get("n_dofs"), gd.get(1), gd.get(2),
                 n_in, len(res.get("compare", [])), (ms.get("automesh_s") or 0) + (ms.get("check_s") or 0), res.get("total_s") or 0,
                 max((abs(c["rel_err"]) for c in res.get("compare", [])), default=None)))
H = ["## Summary\n", "| set | faces | gates | failed | harness DOFs | gold L1 / L2 DOFs | in band | worst abs err | automesh + check s | solve s |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for s in summ:
    dofs = f"{s[4]:,}" if s[4] else "not solved"
    worst = f"{s[11]:.2%}" if s[11] is not None else "-"
    H.append(f"| {s[0]} | {s[1]} | {s[2]} | {', '.join(s[3] or []) or '-'} | {dofs} | {s[5]:,} / {s[6]:,} | {s[7]}/{s[8]} | {worst} | {s[9]:.0f} | {s[10]:.0f} |")
(P / "pilot_tables.md").write_text("\n".join(H) + "\n\n## Per case\n\n" + "\n".join(L) + "\n")
print("ok")
