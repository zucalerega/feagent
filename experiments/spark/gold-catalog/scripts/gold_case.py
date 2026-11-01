"""Harness-vs-gold benchmark of one gold part / BC set (run with PYTHONPATH=<mh>/meshing-harness/src:<mh>).

    gold_case.py mesh  PART WORK [--sets ID ...]    automesh the part's STEP, write WORK/mesh.yaml (auto sizes
                                                    + ids-query groups for every gold face the sets use + the
                                                    first set's bcs), feagent check -> WORK/out/report.json
    gold_case.py case  SET WORK                     adapter -> WORK/<SET>/case.json (gold BCs and gold QoI
                                                    definitions on the harness mesh) + group mapping check
    gold_case.py solve SET WORK                     solve with cuDSS -> WORK/<SET>/result.json, compare with refs
"""
from __future__ import annotations

import argparse, json, os, resource, shutil, subprocess, sys, time
from pathlib import Path

GOLD = Path(os.environ.get("FEACORE_GOLD_DIR", Path.home() / "feagent-mh" / "benchmarks" / "gold"))
PY = sys.executable


def ref_of(sid: str) -> dict:
    return json.loads((GOLD / sid / "private" / "reference.json").read_text())


def study_of(sid: str) -> dict:
    return json.loads((GOLD / sid / "private" / "study.json").read_text())


def sets_of(part: str) -> list[str]:
    idx = json.loads((GOLD / "index.json").read_text())
    return sorted(k for k, v in idx["sets"].items() if v["part"] == part)


def set_groups(ref: dict) -> list[str]:
    g = [b["group"] for b in ref["reference_case"]["bcs"] if b.get("group")]
    g += [q["group"] for q in ref["reference_case"]["qois"] if q.get("group")]
    return list(dict.fromkeys(g))


def fea_of(ref: dict) -> dict:
    fea = {"E": 210e9, "nu": 0.3, "fixed": [], "loads": [], "pressure": [], "displacement": []}
    mats = ref["reference_case"]["materials"]
    (m,) = mats.values()
    fea["E"], fea["nu"] = float(m["E"]), float(m["nu"])
    for b in ref["reference_case"]["bcs"]:
        t, g = b["type"], b["group"]
        if t == "fixed":
            fea["fixed"].append(g)
        elif t == "force":
            fea["loads"].append({"group": g, "force": [float(x) for x in b["value"]]})
        elif t == "pressure":
            fea["pressure"].append({"group": g, "value": float(b["value"])})
        elif t == "displacement":
            fea["displacement"].append({"group": g, "components": list(b["components"]), "value": float(b["value"] or 0.0)})
        else:
            raise ValueError(f"unmapped gold BC type {t}")
    return {k: v for k, v in fea.items() if v != []}


def run(cmd, log, cwd=None, env_extra=None, timeout=None):
    env = dict(os.environ, **(env_extra or {}))
    t0 = time.perf_counter()
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n"); f.flush()
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, time.perf_counter() - t0


def cmd_mesh(a) -> int:
    import yaml
    work = Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
    sets = a.sets or sets_of(a.part)
    refs = {s: ref_of(s) for s in sets}
    step = work / "part.step"
    shutil.copyfile(GOLD / sets[0] / "public" / "geometry.step", step)
    log = work / "harness.log"
    st = {"part": a.part, "sets": sets}
    env = {"FEAGENT_STATE": str(work / "feagent_state.json"), "MPLBACKEND": "Agg"}
    rc, t = run([PY, "-m", "feagent.cli", "automesh", step, "-o", work / "auto", "--no-render"], log, env_extra=env)
    st["automesh_rc"], st["automesh_s"] = rc, round(t, 2)
    auto = work / "auto" / "mesh.yaml"
    if not auto.is_file():
        st["status"] = "mesh_fail"; st["error"] = f"automesh rc={rc}, no spec"
        (work / "mesh_status.json").write_text(json.dumps(st, indent=1)); return 1
    spec = yaml.safe_load(auto.read_text())
    spec["geometry"] = "part.step"
    spec["name"] = a.part
    groups = list(dict.fromkeys(g for r in refs.values() for g in set_groups(r)))
    spec["groups"] = {g: {"select": {"ids": [int(g[1:])]}, "expect": 1} for g in groups}
    spec["bcs"] = {"fea": fea_of(refs[sets[0]])}
    spec["export"] = ["msh"]
    (work / "mesh.yaml").write_text("# gold benchmark: auto spec sizes + gold faces by OCC tag (ids queries)\n"
                                    + yaml.safe_dump(spec, sort_keys=False))
    rc, t = run([PY, "-m", "feagent.cli", "check", work / "mesh.yaml", "--no-render"], log, env_extra=env)
    st["check_rc"], st["check_s"] = rc, round(t, 2)
    rep = work / "out" / "report.json"
    if not rep.is_file() or not (work / "out" / "mesh.msh").is_file():
        st["status"] = "mesh_fail"; st["error"] = f"check rc={rc}, no report/mesh"
    else:
        r = json.loads(rep.read_text())
        st["gate_status"] = r["status"]
        st["gates"] = {g["id"]: [g["status"], g.get("value"), g.get("threshold")] for g in r["gates"]}
        st["failed_gates"] = [g["id"] for g in r["gates"] if g["status"] == "fail"]
        st["warn_gates"] = [g["id"] for g in r["gates"] if g["status"] == "warn"]
        st["mesh"] = {k: r["mesh"].get(k) for k in ("n_nodes", "n_elements", "n_tets", "order", "nodes", "elements", "tets") if k in r["mesh"]}
        st["status"] = "ok" if r["status"] == "pass" else "gate_fail"
    st["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)
    (work / "mesh_status.json").write_text(json.dumps(st, indent=1))
    return 0 if st["status"] == "ok" else 1


def group_geometry(msh: Path, groups: list[str]) -> dict:
    import meshio, numpy as np
    m = meshio.read(msh)
    out = {}
    for g in groups:
        if g not in m.field_data:
            out[g] = None; continue
        tag = m.field_data[g][0]
        A, C = 0.0, np.zeros(3)
        for block, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
            if block.type not in ("triangle", "triangle6"):
                continue
            c = block.data[np.asarray(phys) == tag][:, :3]
            if block.type == "triangle6":       # split in four through the midside nodes
                c6 = block.data[np.asarray(phys) == tag]
                c = np.concatenate([c6[:, [0, 3, 5]], c6[:, [3, 1, 4]], c6[:, [5, 4, 2]], c6[:, [3, 4, 5]]])
            P = m.points[c]
            a = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
            A += a.sum(); C += (a[:, None] * P.mean(1)).sum(0)
        out[g] = (A, (C / A).tolist() if A > 0 else None)
    return out


def cmd_case(a) -> int:
    import numpy as np
    from feacore.adapters.mesh_harness import case_from_mesh_harness
    from feacore.case import QoI
    work = Path(a.work).resolve(); d = work / a.set; d.mkdir(parents=True, exist_ok=True)
    ref = ref_of(a.set)
    fea = fea_of(ref)
    case = case_from_mesh_harness(work / "out", fea=fea, mesh=work / "out" / "mesh.msh")
    case.qois = [QoI(**{k: v for k, v in q.items() if k in ("name", "type", "field", "point", "group", "component")})
                 for q in ref["reference_case"]["qois"]]
    case.meta.update(problem=a.set, setup="harness_vs_gold")
    case.save(d / "case.json")
    faces = {f"f{f['tag']}": f for f in ref["part"]["faces"]}
    diag = float(np.linalg.norm(np.subtract(ref["part"]["bbox"][3:], ref["part"]["bbox"][:3])))
    geo = group_geometry(work / "out" / "mesh.msh", set_groups(ref))
    mp = {}
    for g, v in geo.items():
        if v is None:
            mp[g] = {"ok": False, "why": "group missing in mesh"}; continue
        A, c = v
        ea = A / faces[g]["area"] - 1
        ec = float(np.linalg.norm(np.subtract(c, faces[g]["centroid"]))) / diag
        mp[g] = {"area_rel_err": ea, "centroid_err_rel_diag": ec, "ok": abs(ea) < 0.02 and ec < 0.01}
    (d / "mapping.json").write_text(json.dumps(mp, indent=1))
    return 0 if all(v["ok"] for v in mp.values()) else 3


def cmd_solve(a) -> int:
    from feacore.case import Case
    from feacore.solver import solve
    work = Path(a.work).resolve(); d = work / a.set
    ref = ref_of(a.set); study = study_of(a.set)
    case = Case.load(d / "case.json")
    t0 = time.perf_counter()
    out = {"set": a.set, "part": ref["part"]["id"] if "id" in ref["part"] else None}
    try:
        res = solve(case, backend="skfem", linear_solver=a.linear_solver)
        vals = res.evaluate_all()
        out.update(status="ok", n_dofs=res.info.n_dofs, n_nodes=res.info.n_nodes, n_cells=res.info.n_cells,
                   solver_wall_s=round(res.info.wall_time, 3),
                   linear_solver=[w for w in res.info.warnings if w.startswith("info: linear_solver")],
                   warnings=[w for w in res.info.warnings if not w.startswith("info: ")][:20],
                   qois={k: (float(v) if v == v else None) for k, v in vals.items()})
    except Exception as e:  # noqa: BLE001
        out.update(status="solve_fail", error=f"{type(e).__name__}: {e}"[:2000])
    out["total_s"] = round(time.perf_counter() - t0, 3)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    gold_dofs = {r["level"]: r["n_dofs"] for r in study["runs"]}
    out["gold_dofs"] = gold_dofs
    cmp = []
    for r in ref["references"]:
        v = out.get("qois", {}).get(r["qoi"])
        e = (v / r["value"] - 1) if v is not None and r["value"] != 0 else None
        cmp.append({"qoi": r["qoi"], "harness": v, "gold": r["value"], "tol": r["tolerance"], "rel_err": e,
                    "in_band": (abs(e) <= r["tolerance"]) if e is not None else False})
    out["compare"] = cmp
    (d / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("set", "status", "n_dofs", "total_s", "linear_solver", "error")}))
    for c in cmp:
        e = "n/a" if c["rel_err"] is None else f"{c['rel_err']:+.3%}"
        print(f"  {c['qoi']:14s} harness {c['harness']} gold {c['gold']:.6g} err {e} tol {c['tol']:.2%} {'IN' if c['in_band'] else 'OUT'}")
    return 0 if out["status"] == "ok" else 1


def _hint_refines(hint: str) -> list[dict]:
    """The `refine: [...]` snippet a sicn/inverted gate hint proposes, parsed (empty if none)."""
    import re, yaml
    m = re.search(r"refine: (\[.*?\}\])", hint or "")
    if not m:
        return []
    try:
        v = yaml.safe_load(m.group(1))
        return v if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return []


def cmd_mesh2(a) -> int:
    """Full-run mesh stage: one mesh per part for all its BC sets (the mesh does not depend on the BCs).
    Spec = feagent.mesher.auto_spec (what automesh writes, without automesh's own extra mesh) + ids
    groups for every gold face any set uses + the first set's bcs. On failing gates, follows the gate
    hints like an agent would (up to 3 fixes): sicn/inverted -> the hint's refine snippet, then
    size.global x 0.7; mesh_generated -> algorithm mesh3d 1, then heal; connectivity/geometry -> heal."""
    import yaml
    from feagent.mesher import auto_spec
    t_start = time.perf_counter()
    work = Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
    sets = sets_of(a.part)
    refs = {s: ref_of(s) for s in sets}
    step = work / "part.step"
    shutil.copyfile(GOLD / sets[0] / "public" / "geometry.step", step)
    spec = auto_spec(step)
    spec.update(geometry="part.step", name=a.part, export=["msh"])
    groups = list(dict.fromkeys(g for r in refs.values() for g in set_groups(r)))
    spec["groups"] = {g: {"select": {"ids": [int(g[1:])]}, "expect": 1} for g in groups}
    spec["bcs"] = {"fea": fea_of(refs[sets[0]])}
    log = work / "harness.log"
    env = {"FEAGENT_STATE": str(work / "feagent_state.json"), "MPLBACKEND": "Agg"}
    st = {"part": a.part, "sets": sets, "groups": groups, "attempts": []}
    for attempt in range(4):
        (work / "mesh.yaml").write_text("# gold benchmark: auto_spec sizes + gold faces by OCC tag (ids queries)\n"
                                        + yaml.safe_dump(spec, sort_keys=False))
        rep = work / "out" / "report.json"
        rep.unlink(missing_ok=True)
        rc, t = run([PY, "-m", "feagent.cli", "check", work / "mesh.yaml", "--no-render"], log, env_extra=env)
        att = {"rc": rc, "check_s": round(t, 2)}
        if not rep.is_file():
            att["error"] = f"check rc={rc}, no report"
            st["attempts"].append(att)
            break
        r = json.loads(rep.read_text())
        shutil.copyfile(rep, work / "out" / f"report_attempt{attempt}.json")
        att["status"] = r["status"]
        att["failed"] = [g["id"] for g in r["gates"] if g["status"] == "fail"]
        att["mesh"] = {k: r.get("mesh", {}).get(k) for k in ("n_nodes", "n_elements", "elapsed_s")}
        att["min_sicn"] = next((g.get("value") for g in r["gates"] if g["id"] == "sicn"), None)
        st["attempts"].append(att)
        if r["status"] == "pass" or attempt == 3 or time.perf_counter() - t_start > 900:
            break
        if a.max_dofs and 3 * int(att["mesh"].get("n_nodes") or 0) > a.max_dofs:
            st["stopped"] = f"mesh already {3 * att['mesh']['n_nodes']:,} DOFs > solvable {a.max_dofs:,}: no refining fixes"
            break
        gates = {g["id"]: g for g in r["gates"]}
        fix = None
        bad = set(att["failed"])
        if bad & {"sicn", "inverted"}:
            hint = (gates.get("sicn") if "sicn" in bad else gates.get("inverted"))["hint"]
            new = [x for x in _hint_refines(hint) if x not in spec.get("refine", [])]
            done = st.get("fixes", [])
            if new and not any(f.startswith("refine") for f in done):
                spec.setdefault("refine", []).extend(new); fix = f"refine {new}"
            elif not any(f.startswith("size.min") for f in done):
                spec["size"]["min"] = float(f"{spec['size']['min'] * 3:.3g}"); fix = f"size.min -> {spec['size']['min']}"
            else:
                spec["size"]["global"] = float(f"{spec['size']['global'] * 0.7:.3g}"); fix = f"size.global -> {spec['size']['global']}"
        elif "mesh_generated" in bad:
            if spec.get("algorithm", {}).get("mesh3d") != 1:
                spec["algorithm"] = {"mesh3d": 1}; fix = "algorithm.mesh3d 1"
            elif not spec.get("heal"):
                spec["heal"] = True; fix = "heal true"
        elif bad & {"connectivity", "geometry"} and not spec.get("heal"):
            spec["heal"] = True; fix = "heal true"
        elif "area_fidelity" in bad:
            g = gates["area_fidelity"].get("message", "")
            import re
            m = re.search(r"group '([^']+)'", g)
            if m:
                spec.setdefault("refine", []).append({"group": m.group(1), "size": float(f"{spec['size']['global'] / 2:.3g}")})
                fix = f"refine group {m.group(1)}"
        elif "volume_fidelity" in bad:
            spec["size"]["curvature"] = int(spec["size"].get("curvature", 12) * 2); fix = f"size.curvature {spec['size']['curvature']}"
        if fix is None:
            break
        st.setdefault("fixes", []).append(fix)
    last = st["attempts"][-1]
    rep = work / "out" / "report.json"
    if "status" not in last or not (work / "out" / "mesh.msh").is_file():
        st["status"] = "mesh_fail"; st["error"] = last.get("error", "no mesh")
    else:
        r = json.loads(rep.read_text())
        st["gate_status"] = r["status"]
        st["gates"] = {g["id"]: [g["status"], g.get("value"), g.get("threshold")] for g in r["gates"]}
        st["failed_gates"] = [g["id"] for g in r["gates"] if g["status"] == "fail"]
        st["warn_gates"] = [g["id"] for g in r["gates"] if g["status"] == "warn"]
        st["mesh"] = {k: r.get("mesh", {}).get(k) for k in ("n_nodes", "n_elements", "order", "elapsed_s")}
        st["status"] = "ok" if r["status"] == "pass" else ("gate_fail" if st["gates"].get("mesh_generated", ["fail"])[0] == "pass" else "mesh_fail")
    st["mesh_stage_s"] = round(time.perf_counter() - t_start, 2)
    st["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)
    (work / "mesh_status.json").write_text(json.dumps(st, indent=1))
    return 0 if st["status"] == "ok" else 1


def cmd_cases(a) -> int:
    """case.json + mapping.json for every BC set of a meshed part (one process, one mesh read)."""
    work = Path(a.work).resolve()
    bad = 0
    for sid in (a.sets or sets_of(a.part)):
        try:
            rc = cmd_case(argparse.Namespace(set=sid, work=str(work)))
        except Exception as e:  # noqa: BLE001
            (work / sid).mkdir(parents=True, exist_ok=True)
            (work / sid / "case_error.txt").write_text(f"{type(e).__name__}: {e}")
            rc = 2
        bad += rc != 0
        print(sid, rc, flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("mesh"); s.add_argument("part"); s.add_argument("work"); s.add_argument("--sets", nargs="*")
    s.set_defaults(func=cmd_mesh)
    s = sub.add_parser("mesh2"); s.add_argument("part"); s.add_argument("work"); s.add_argument("--max-dofs", type=int, default=0)
    s.set_defaults(func=cmd_mesh2)
    s = sub.add_parser("cases"); s.add_argument("part"); s.add_argument("work"); s.add_argument("--sets", nargs="*")
    s.set_defaults(func=cmd_cases)
    s = sub.add_parser("case"); s.add_argument("set"); s.add_argument("work"); s.set_defaults(func=cmd_case)
    s = sub.add_parser("solve"); s.add_argument("set"); s.add_argument("work")
    s.add_argument("--linear-solver", default="cudss"); s.set_defaults(func=cmd_solve)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
