"""Nonlinear (elastoplastic) gold benchmarks from the linear gold sets: same part, supports, load
directions and nested meshes; S355 steel with linear hardening; loads scaled past first yield.

    python scripts/gold_nonlinear.py GOLD_DIR SET_ID [SET_ID ...] --out OUT [--threads 8] [--levels 0 1 2]
    python scripts/gold_nonlinear.py GOLD_DIR --pick 5 --max-dofs 300000 --out OUT   # pilot: small sets

GOLD_DIR is an unpacked gold folder (out/<set>/ + out/_meshes/<part>/L<k>.msh). For each set:
1. a linear CalculiX solve on the middle level gives the peak von Mises away from singular corners
   (same rule as scripts/ccx_plastic_ramp.py); every load is scaled by k so that peak = ratio x sy, with
   ratio drawn per set in [1.5, 2.5] from a hash of the set id (re-runs give the same benchmark). k is
   part of the problem: the statement gives the scaled loads.
2. the scaled case runs load-stepped (fixed increments, adaptive retry on divergence) on each level,
   NLGEOM when the linear displacement at full load exceeds 1 % of the part's size.
3. per QoI (the set's own QoIs at full load, the loaded-face displacements at half load, and the
   plastic volume fraction at full load) a GCI over the levels; QoIs that converge within the gold
   tolerance become references, as in feacore.goldgen.

Meshes: the gold L<k>.msh themselves (curved tet10), run with the ccx option accept_curved; a linear ccx solve
on the middle level is compared with skfem's values on the same mesh first (linear_ccx_vs_skfem).

OUT/<set>_nl/ gets public/problem.json + geometry.step, private/reference.json + study.json, and
runs.json (timings, increments, load factor reached per level).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from feacore.case import BC, Case, QoI
from feacore.goldgen.bcgen import BCSet
from feacore.goldgen.run import GOLD_TOL, _fmt, classify, qois_for
from feacore.problems.base import Reference
from feacore.solver import solve
from feacore.verification.gci import analyze
from feacore.verification.study import default_expected_order

_spec = importlib.util.spec_from_file_location("ccx_plastic_ramp", Path(__file__).with_name("ccx_plastic_ramp.py"))
_ramp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ramp)
E, NU, SY = _ramp.E, _ramp.NU, _ramp.SY
TANGENT = 1.5e9          # tangent modulus of the linear hardening (Pa)
RATIO = (1.5, 2.5)
NLGEOM_FRAC = 0.01       # NLGEOM when linear u_max at full load > this x bbox diagonal


def ratio_for(set_id: str) -> float:
    u = int(hashlib.sha256(set_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return RATIO[0] + u * (RATIO[1] - RATIO[0])


def load_set(gold: Path, sid: str):
    ref = json.loads((gold / "out" / sid / "private" / "reference.json").read_text())
    study = json.loads((gold / "out" / sid / "private" / "study.json").read_text())
    b = ref["bcset"]
    bcset = BCSet(id=b["id"], attempt=b["attempt"], bcs=[BC(**x) for x in b["bcs"]], words=b.get("words", []))
    return ref, study, bcset


def level_case(ref: dict, mesh: Path) -> Case:
    c = Case.from_dict(copy.deepcopy(ref["reference_case"]))
    c.mesh, c.base_dir = str(mesh.resolve()), None
    # the gold meshes are curved (interior edges too, from the high-order optimizer); CalculiX's C3D10
    # rule is not exact on them, so step 1 checks ccx against the certified skfem values on L0
    c.analysis.options = {"accept_curved": True}
    return c


def extra_qois(bcset: BCSet) -> list[QoI]:
    """Loaded-face mean displacement at half load is evaluated on the half-load increment."""
    return [q for q, _ in qois_for(bcset) if q.name.startswith("u_mean_")]


def plastic_fraction(r) -> float:
    inc = r.increments[-1]
    vol = _ramp.nodal_volumes(r.model, r.cell_measures())
    return float(vol[inc.PEEQ > 1e-9].sum() / vol.sum())


def solve_nl(case: Case, k: float, n: int, nlgeom: bool, threads: int, wd: Path):
    c = _ramp.scaled_case(case, k, n, TANGENT)
    c.analysis.options["nlgeom"] = nlgeom
    t0 = time.time()
    r = solve(c, backend="ccx", threads=threads, workdir=wd)
    mode = "fixed"
    if not r.converged:
        c.analysis.options["adaptive"] = True
        r = solve(c, backend="ccx", threads=threads, workdir=wd.with_name(wd.name + "_adaptive"))
        mode = "adaptive"
    return r, mode, time.time() - t0, c


def at_lambda(r, lam: float) -> int | None:
    for i, inc in enumerate(r.increments):
        if abs(inc.lam - lam) < 1e-6:
            return i
    return None


def run_set(gold: Path, sid: str, out: Path, levels: list[int], threads: int, n: int, log) -> dict:
    ref, study, bcset = load_set(gold, sid)
    part = sid.rsplit("_bc", 1)[0]
    h_rep = {r["level"]: r["h_rep"] for r in study["runs"]}
    dofs = {r["level"]: r["n_dofs"] for r in study["runs"]}
    d = out / f"{sid}_nl"
    wdir = d / "work"
    wdir.mkdir(parents=True, exist_ok=True)

    # 1. load scale from a linear ccx solve on the coarsest level
    lv = levels[1] if len(levels) > 1 else levels[0]   # the middle level: L0 can be too coarse to compare
    base = level_case(ref, gold / "out" / "_meshes" / part / f"L{lv}.msh")
    for m in base.materials.values():
        m.E, m.nu = E, NU
    t0 = time.time()
    lin = solve(base, backend="ccx", threads=threads, workdir=wdir / "linear")
    X, vm = lin.nodal("von_mises")
    sing, excl = _ramp.singular_mask(X, vm)
    peak = float(vm[~excl].max())
    ratio = ratio_for(sid)
    k = ratio * SY / peak
    diag = float(np.linalg.norm(X.max(0) - X.min(0)))
    u_lin = k * float(lin.evaluate(QoI("u_max", "max", field="u_mag")))
    nlgeom = u_lin > NLGEOM_FRAC * diag
    lin_check = {}
    sk = next(r for r in study["runs"] if r["level"] == lv)["qois"]
    for q, _ in qois_for(bcset):
        if q.name in sk and abs(sk[q.name]) > 0 and not q.name.startswith("vm_max"):
            lin_check[q.name] = float(lin.evaluate(q)) / sk[q.name] - 1.0
    worst = max((abs(v) for v in lin_check.values()), default=0.0)
    log(f"{sid}: ccx vs skfem on L{lv} (linear): worst {worst:.2%}: "
        + ", ".join(f"{n} {v:+.2%}" for n, v in lin_check.items()))
    log(f"{sid}: linear peak off-singular {peak / 1e6:.1f} MPa -> k = {k:.4g} (ratio {ratio:.2f}), "
        f"u_lin {u_lin * 1e3:.3f} mm of {diag * 1e3:.0f} mm, nlgeom {nlgeom}, {time.time() - t0:.0f}s")

    # 2. nonlinear ramp per level
    qmain = [q for q, _ in qois_for(bcset)]
    qhalf = extra_qois(bcset)
    runs = []
    for L in levels:
        case = level_case(ref, gold / "out" / "_meshes" / part / f"L{L}.msh")
        r, mode, wall, nlcase = solve_nl(case, k, n, nlgeom, threads, wdir / f"L{L}")
        lam = r.increments[-1].lam
        row = {"level": L, "h_rep": h_rep[L], "n_dofs": dofs[L], "mode": mode, "wall_s": wall,
               "converged": bool(r.converged), "lambda_reached": lam, "increments": len(r.increments),
               "iterations": [inc.iterations for inc in r.increments], "qois": {}}
        if r.converged:
            row["qois"] = {q.name: float(r.evaluate(q)) for q in qmain}
            row["qois"]["plastic_vol_frac"] = plastic_fraction(r)
            ih = at_lambda(r, 0.5)
            if ih is not None:
                r.use_increment(ih)
                row["qois"].update({f"{q.name}@0.5": float(r.evaluate(q)) for q in qhalf})
        runs.append(row)
        log(f"  L{L}: {dofs[L]} DOFs, {mode}, lambda {lam:.3f}, {len(r.increments)} incs, {wall:.0f}s")
        if not r.converged:
            break

    (d / "runs.json").write_text(json.dumps({"runs": runs, "k": k, "nlgeom": nlgeom, "lin_check": lin_check}, indent=2))

    # 3. certification
    refs, st = [], {}
    ok = [r for r in runs if r["converged"]]
    names = list(ok[0]["qois"]) if ok else []
    for name in names:
        vals = [r["qois"].get(name) for r in ok]
        h = [r["h_rep"] for r in ok]
        if len(ok) < 3 or any(v is None for v in vals):
            st[name] = {"values": vals, "h": h, "kind": "insufficient_levels", "gold": False}
            continue
        base_name = name.split("@")[0]
        fam = next((f for q, f in qois_for(bcset) if q.name == base_name), "stress")   # plastic_vol_frac: 3 %
        qq = next((q for q in qmain if q.name == base_name), QoI(name, "integral", field="u_mag"))
        p_th = default_expected_order(qq, 2)
        a = analyze(h, vals, p_theory=p_th)
        g = a["gci"]
        kind = classify(g, p_th)
        tol = GOLD_TOL[fam]
        is_gold = kind in ("converged", "reduced_order") and g.get("gci_fine") is not None and g["gci_fine"] <= tol
        st[name] = {"values": vals, "h": h, "kind": kind, "p": g.get("p"), "p_theory": p_th,
                    "phi_ext": g.get("phi_ext"), "gci_fine": g.get("gci_fine"), "gold": bool(is_gold)}
        if is_gold:
            value = g["phi_ext"] if g.get("phi_ext") is not None else vals[-1]
            refs.append(Reference(name, float(value), "converged_numerical",
                                  source=f"gold_nonlinear: {len(ok)} nested tet10 meshes, CalculiX J2 ramp, GCI",
                                  tolerance=max(tol, 2 * g["gci_fine"]),
                                  notes=f"{kind}; p = {_fmt(g.get('p'), '.2f')}; GCI_fine = {_fmt(g.get('gci_fine'), '.2%')}"))

    # 4. write the benchmark
    pub, prv = d / "public", d / "private"
    pub.mkdir(exist_ok=True), prv.mkdir(exist_ok=True)
    src_pub = gold / "out" / sid / "public"
    shutil.copy(src_pub / "geometry.step", pub / "geometry.step")
    prob = json.loads((src_pub / "problem.json").read_text())
    scaled = _ramp.scaled_case(level_case(ref, gold / "out" / "_meshes" / part / f"L{levels[0]}.msh"), k, n, TANGENT)
    load_lines = [f"{b.type} on {b.group}: {b.value}" for b in scaled.bcs if b.type in ("force", "pressure", "traction")]
    prob.update({
        "id": f"{sid}_nl", "title": f"Nonlinear gold benchmark {sid}_nl", "tags": ["gold", "3d", "abc", "nonlinear"],
        "description": prob["description"].replace("Static linear-elastic analysis", "Static elastoplastic analysis")
        .replace("Material: isotropic linear elastic steel, E = 210 GPa, Poisson's ratio 0.3.",
                 f"Material: S355 steel, E = 210 GPa, nu = 0.3, von Mises plasticity, yield 355 MPa, linear "
                 f"isotropic hardening with tangent modulus {TANGENT / 1e9:g} GPa. Loads ramp linearly from 0."
                 f"{' Large displacements (geometric nonlinearity).' if nlgeom else ' Small displacements.'}")
        + f"\nLoads are {k:.6g} x the linear benchmark's, i.e. " + "; ".join(load_lines)
        + "\nAlso report: plastic_vol_frac = volume fraction with plastic strain > 0 at full load; "
          "u_mean_<face>@0.5 = the loaded face's mean displacement at half load.",
        "load_scale": k, "nlgeom": nlgeom})
    (pub / "problem.json").write_text(json.dumps(prob, indent=2))
    (prv / "reference.json").write_text(json.dumps({
        "id": f"{sid}_nl", "references": [asdict(x) for x in refs], "linear_set": sid, "load_scale": k,
        "linear_ccx_vs_skfem": lin_check,
        "ratio": ratio, "nlgeom": nlgeom, "reference_case": nlcase.to_dict() if runs else None}, indent=2))
    (prv / "study.json").write_text(json.dumps({"runs": runs, "qois": st}, indent=2))
    shutil.rmtree(wdir, ignore_errors=True)   # ccx work files (large); runs.json keeps the record
    s = {"id": f"{sid}_nl", "levels_converged": len(ok), "gold": [x.qoi for x in refs], "n_qois": len(st),
         "wall_s": sum(r["wall_s"] for r in runs), "nlgeom": nlgeom}
    log(f"  -> {len(refs)}/{len(st)} QoIs certified: {', '.join(s['gold']) or '-'}")
    return s


def pick(gold: Path, n: int, max_dofs: int) -> list[str]:
    rows = []
    for p in sorted((gold / "out").glob("*_bc*/private/study.json")):
        st = json.loads(p.read_text())
        big = max(r["n_dofs"] for r in st["runs"])
        if big <= max_dofs and len(st["runs"]) >= 3:
            rows.append((big, p.parent.parent.name))
    rows.sort()
    step = max(1, len(rows) // n)
    return [sid for _, sid in rows[::step][:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--max-dofs", type=int, default=300_000)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--increments", type=int, default=10)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    gold, out = Path(a.gold), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = a.sets or pick(gold, a.pick, a.max_dofs)
    i, m = map(int, a.shard.split("/"))
    sets = sets[i::m]
    logf = open(out / "run.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for sid in sets:
        if (out / f"{sid}_nl" / "private" / "study.json").exists():
            continue
        try:
            s = run_set(gold, sid, out, a.levels, a.threads, a.increments, log)
        except Exception as e:  # noqa: BLE001 - one bad set must not stop the batch
            s = {"id": f"{sid}_nl", "error": f"{type(e).__name__}: {e}"}
            log(f"{sid}: ERROR {s['error']}\n" + traceback.format_exc())
        with open(out / "summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
