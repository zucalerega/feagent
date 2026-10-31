"""Reference convergence study of a Problem: mesh sequence -> solve -> GCI -> compare to references."""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from feacore.case import DISPLACEMENT_LIKE, QoI
from feacore.problems.base import Problem
from feacore.solver import solve
from feacore.verification.gci import analyze


def default_expected_order(qoi: QoI, order: int) -> float | None:
    """A-priori order in h for smooth problems (P_k elements)."""
    if qoi.type == "strain_energy" or qoi.type == "reaction":
        return 2.0 * order
    if qoi.type in ("integral", "mean"):
        return float(order + 1)
    if qoi.field in DISPLACEMENT_LIKE:
        return float(order + 1)
    if qoi.field is not None:
        return float(order)
    return None


def convergence_study(problem: Problem, sizes: list[float] | None = None, order: int = 2,
                      backend: str = "skfem", geom_order: int | None = None,
                      workdir: str | Path | None = None, verbose: bool = False) -> dict:
    sizes = sorted(sizes or problem.study_sizes(order), reverse=True)  # coarse -> fine
    geom_order = order if geom_order is None else geom_order
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix=f"study_{problem.id}_")
        workdir = tmp.name
    workdir = Path(workdir)
    qois = problem.qois()
    runs = []
    try:
        for i, h in enumerate(sizes):
            t0 = time.perf_counter()
            mesh_path = problem.mesh(workdir / f"{problem.id}_m{i}.msh", h, geom_order=geom_order)
            t_mesh = time.perf_counter() - t0
            res = solve(problem.case(mesh_path, order=order), backend=backend)
            values = res.evaluate_all()
            runs.append({"h_nominal": h, "mesh_time": t_mesh, "info": asdict(res.info), "qois": values})
            if verbose:
                print(f"[{problem.id}] h={h:.4g} cells={res.info.n_cells} dofs={res.info.n_dofs} "
                      f"t={res.info.wall_time:.2f}s " + " ".join(f"{k}={v:.6g}" for k, v in values.items()))
    finally:
        if tmp is not None:
            tmp.cleanup()

    refs = {r.qoi: r for r in problem.references()}
    expect = problem.expectations()
    h_rep = [r["info"]["h_rep"] for r in runs]
    qoi_results = []
    for q in qois:
        vals = [r["qois"][q.name] for r in runs]
        p_th = problem.expected_order(q, order)
        p_th = default_expected_order(q, order) if p_th is None else p_th
        entry = {"name": q.name, "type": q.type, "field": q.field, "values": vals,
                 "p_theory": p_th, "expectation": expect.get(q.name, "converge")}
        if len(runs) >= 3:
            entry.update(analyze(h_rep, vals, p_theory=p_th))
        ref = refs.get(q.name)
        entry["reference"] = asdict(ref) if ref else None
        entry.update(_judge(entry, ref))
        qoi_results.append(entry)

    judged = [e["passed"] for e in qoi_results if e["passed"] is not None]
    return {
        "problem": problem.id,
        "backend": backend,
        "order": order,
        "geom_order": geom_order,
        "sizes": sizes,
        "h_rep": h_rep,
        "runs": runs,
        "qois": qoi_results,
        "passed": all(judged) if judged else None,
    }


def _judge(entry: dict, ref) -> dict:
    """Pass/fail of one QoI. The extrapolated value is used only when the extrapolation is
    trustworthy: monotone convergence with an observed order near theory. Otherwise (order far
    below / above theory, oscillation) the finest-mesh value is judged instead."""
    g = entry.get("gci")
    out = {"err_fine": None, "err_ext": None, "ref_in_gci_band": None, "extrapolation_reliable": None,
           "passed": None}
    if entry["expectation"] == "diverge":
        if g is not None:
            out["passed"] = g["convergence"] in ("divergent", "oscillatory_divergent") or \
                "p_far_below_theory" in g["flags"]
        return out
    if g is not None:
        out["extrapolation_reliable"] = bool(
            g.get("phi_ext") is not None and g["convergence"] in ("monotone", "exact", "converged")
            and not {"p_far_below_theory", "p_above_theory"} & set(g["flags"]))
    if ref is None:
        return out

    def rel(v):
        return (v - ref.value) / ref.value if ref.value != 0 else v - ref.value

    fine = entry["values"][-1]
    out["err_fine"] = rel(fine)
    if g is not None and g.get("phi_ext") is not None:
        out["err_ext"] = rel(g["phi_ext"])
        band = (g["gci_fine"] or 0.0) * abs(fine)
        out["ref_in_gci_band"] = bool(abs(fine - ref.value) <= band)
    best = out["err_ext"] if out["extrapolation_reliable"] else out["err_fine"]
    out["passed"] = bool(abs(best) <= ref.tolerance) if not math.isnan(best) else False
    return out


def save_study(study: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=2, default=float))
    return path
