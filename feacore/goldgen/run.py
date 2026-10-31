"""Solve every BC set of a part on its mesh sequence and certify gold values.

Per BC set: precheck on the coarsest mesh (invalid sets are recorded, not solved), solve on every
level, then for each QoI a GCI analysis over the three finest levels (plus a power-law fit with
four or more). A QoI becomes a gold reference only if it converges (``verify.classify``:
converged or reduced_order) and its fine-grid GCI is within ``GOLD_TOL``. Diverging QoIs (peak
stresses at clamp edges / corners) are recorded as singular: they have no mesh-independent value.

Output per BC set, in the benchmark-store layout (feacore.problems.store):
  <out>/<id>/public/problem.json, geometry.step       for agents
  <out>/<id>/private/reference.json, study.json        evaluator only
and one line per solve in <out>/solves.jsonl (DOFs, seconds) for the cost model.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from feacore.case import Analysis, Case, Material, QoI, Region
from feacore.diagnostics import diagnose_result, precheck
from feacore.diagnostics.verify import classify
from feacore.goldgen.bcgen import BCSet
from feacore.goldgen.parts import Part, export_scaled_step
from feacore.problems.base import Reference
from feacore.verification.gci import analyze
from feacore.verification.study import default_expected_order

STEEL = Material(E=210e9, nu=0.3)
GOLD_TOL = {"strain_energy": 0.01, "displacement": 0.02, "stress": 0.03, "reaction": 0.02}


def _fmt(v, spec: str) -> str:
    """Format a number that may be None (e.g. no observed order when values converge exactly)."""
    return "-" if v is None else format(v, spec)


def qois_for(bcset: BCSet) -> list[tuple[QoI, str]]:
    """(QoI, family) pairs: global energy, displacements, stresses, and support reactions when
    the load is shared between supports (a single support's reaction equals the load exactly)."""
    q = [(QoI("strain_energy", "strain_energy"), "strain_energy"),
         (QoI("u_max", "max", field="u_mag"), "displacement"),
         (QoI("vm_mean", "mean", field="von_mises"), "stress"),
         (QoI("vm_max", "max", field="von_mises"), "stress")]
    loads = [b for b in bcset.bcs if b.type in ("pressure", "force", "traction")]
    for b in loads:
        q.append((QoI(f"u_mean_{b.group}", "mean", field="u_mag", group=b.group), "displacement"))
    supports = [b for b in bcset.bcs if b.type in ("fixed", "displacement")]
    if len(supports) > 1:
        for b in supports:
            for k in (b.components or [0, 1, 2]):
                q.append((QoI(f"R{'xyz'[k]}_{b.group}", "reaction", group=b.group, component=k), "reaction"))
    return q


def make_case(mesh: str, bcset: BCSet, linear_solver: str | None = None) -> Case:
    opts = {"linear_solver": linear_solver} if linear_solver else {}
    return Case(mesh=mesh, analysis=Analysis(physics="elasticity", formulation="3d", order=2, options=opts),
                materials={"steel": STEEL}, bcs=list(bcset.bcs), regions=[Region("solid", "steel")],
                qois=[q for q, _ in qois_for(bcset)],
                meta={"problem": bcset.id, "setup": "gold_reference"})


def statement(part: Part, bcset: BCSet) -> dict:
    lines = [
        f"Static linear-elastic analysis of the part in geometry.step (third-party CAD {part.source}, "
        f"rescaled to SI metres; bounding-box diagonal {part.diag * 1000:.1f} mm).",
        "Material: isotropic linear elastic steel, E = 210 GPa, Poisson's ratio 0.3. No gravity.",
        "Faces are named f<tag> by their OCC face tag in this STEP file; positions below are in mm.",
        *bcset.words,
        "All other faces are free.",
        "Report: strain_energy (J); u_max = maximum displacement magnitude (m); vm_mean = volume-average "
        "von Mises stress (Pa); u_mean_<face> = mean displacement magnitude over each loaded face (m); "
        "with several supports, the support reactions R<x|y|z>_<face> (N).",
    ]
    return {"id": bcset.id, "title": f"Gold benchmark {bcset.id}", "family": part.id,
            "tags": ["gold", "3d", "abc"], "dim": 3, "physics": "elasticity", "formulation": "3d",
            "units": "SI (m, N, Pa)", "description": "\n".join(lines),
            "qois": [asdict(q) for q, _ in qois_for(bcset)], "geometry_file": "geometry.step"}


def certify(bcset: BCSet, runs: list[dict]) -> tuple[list[Reference], dict]:
    """GCI per QoI over the levels; references for the QoIs that qualify."""
    h = [r["h_rep"] for r in runs]
    refs, study = [], {}
    for q, family in qois_for(bcset):
        vals = [r["qois"][q.name] for r in runs]
        if len(runs) < 3:                   # a convergence study needs three levels: record, certify nothing
            study[q.name] = {"values": vals, "h": h, "kind": "insufficient_levels", "gold": False}
            continue
        p_th = default_expected_order(q, 2)
        a = analyze(h, vals, p_theory=p_th)
        g = a["gci"]
        kind = classify(g, p_th)
        entry = {"values": vals, "h": h, "kind": kind, "p": g.get("p"), "p_theory": p_th,
                 "phi_ext": g.get("phi_ext"), "gci_fine": g.get("gci_fine"), "fit": a.get("fit")}
        tol = GOLD_TOL[family]
        if kind in ("converged", "reduced_order") and g.get("gci_fine") is not None and g["gci_fine"] <= tol:
            value = g["phi_ext"] if g.get("phi_ext") is not None else vals[-1]
            refs.append(Reference(q.name, float(value), "converged_numerical",
                                  source=f"feacore goldgen: {len(runs)} refined tet10 meshes, GCI (Celik 2008)",
                                  tolerance=max(tol, 2 * g["gci_fine"]),
                                  notes=f"{kind}; p = {_fmt(g.get('p'), '.2f')} (theory {p_th:g}); "
                                        f"GCI_fine = {_fmt(g.get('gci_fine'), '.2%')}"))
            entry["gold"] = True
        else:
            entry["gold"] = False
        study[q.name] = entry
    return refs, study


def run_bcset(step, part: Part, bcset: BCSet, meshes: list[dict], out: Path,
              linear_solver: str | None = None, log=None) -> dict:
    from feacore.solver.skfem_backend import solve

    d = out / bcset.id
    pre = precheck(make_case(meshes[0]["path"], bcset, linear_solver))
    summary = {"id": bcset.id, "precheck": pre.status, "flags": pre.ids()}
    if pre.status == "fail":
        summary["result"] = "invalid_setup"
        return summary
    runs = []
    for k, m in enumerate(meshes):
        t0 = time.perf_counter()
        res = solve(make_case(m["path"], bcset, linear_solver))
        vals = res.evaluate_all()
        dt = time.perf_counter() - t0
        row = {"id": bcset.id, "level": k, "h": m["h"], "h_rep": res.info.h_rep, "n_dofs": res.info.n_dofs,
               "solve_s": res.info.wall_time, "total_s": dt, "qois": vals}
        if k == len(meshes) - 1:
            rep = diagnose_result(res)
            row["diagnostics"] = {"status": rep.status, "flags": rep.ids(),
                                  "zz_eta_rel": rep.metrics.get("zz_eta_rel")}
        runs.append(row)
        if log:
            log.write(json.dumps({k_: v for k_, v in row.items() if k_ != "qois"}) + "\n")
            log.flush()
    refs, study = certify(bcset, runs)
    (d / "public").mkdir(parents=True, exist_ok=True)
    (d / "private").mkdir(parents=True, exist_ok=True)
    (d / "public" / "problem.json").write_text(json.dumps(statement(part, bcset), indent=2))
    export_scaled_step(step, part, d / "public" / "geometry.step")
    (d / "private" / "reference.json").write_text(json.dumps({
        "id": bcset.id, "references": [asdict(r) for r in refs], "bcset": bcset.to_dict(),
        "reference_case": make_case("MESH.msh", bcset).to_dict(), "part": json.loads(part.to_json())}, indent=2))
    (d / "private" / "study.json").write_text(json.dumps({"runs": runs, "qois": study}, indent=2, default=float))
    summary.update(result="ok", gold=[r.qoi for r in refs],
                   singular=[k for k, v in study.items() if v["kind"] == "singular"],
                   dofs=[r["n_dofs"] for r in runs], seconds=[round(r["total_s"], 2) for r in runs])
    return summary
