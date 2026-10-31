"""Converged numerical reference values for the parts whose handbook curve fits proved inaccurate.

For the stepped shaft and the U-notched bar the converged FE peak stress is 8-20 % above the
Peterson / Pilkey curve fits (see docs/test_parts.md). The judged reference is therefore the
converged FE value, cross-checked by a second, independent solver:

  value        Richardson-extrapolated value of the reference convergence study (skfem, P2,
               isoparametric, meshes h0, h0/2, h0/4) when the three meshes converge monotonically
               with an observed order in [1, 3.5], else the finest-mesh value
  unc          relative uncertainty: GCI_fine (monotone) or the last relative change
  ccx_fine     CalculiX (independent element formulation and stress recovery) on the finest mesh
  skfem_fine   skfem on that same mesh

Regenerate (about 2 min on one core) with::

    OMP_NUM_THREADS=1 python -m feacore.problems.parts._numerical_refs

which re-runs the studies and the CalculiX cross-check and prints the dict below.
"""

from __future__ import annotations

# Generated 2026-10-31 (skfem P2 + MKL PARDISO; CalculiX 2.21 on the finest mesh). value = Richardson
# extrapolation when monotone with 0.5 p_th <= p <= 1.5 p_th + 0.5 (p_th = 2), else the finest mesh.
# Notched bars: CalculiX models plane stress as a 3D slab of the given 5 mm thickness, so its value
# includes a through-thickness effect at the notch root (+1.4 to +2.1 %); it is not a plane-stress
# cross-check there. Shafts (axisymmetric): CalculiX and skfem agree to < 0.1 %.
REFS: dict[str, dict[str, dict]] = {
    "stepped_shaft_axi_Dd1p5_rd0p05": {"peak_stress_fillet": {"value": 268.96e6, "unc": 0.0019, "ccx_fine": 269.20e6, "skfem_fine": 269.37e6}},
    "stepped_shaft_axi_Dd1p5_rd0p1": {"peak_stress_fillet": {"value": 211.30e6, "unc": 0.0013, "ccx_fine": 211.39e6, "skfem_fine": 211.52e6}},
    "stepped_shaft_axi_Dd1p5_rd0p2": {"peak_stress_fillet": {"value": 170.24e6, "unc": 0.0015, "ccx_fine": 170.31e6, "skfem_fine": 170.45e6}},
    "stepped_shaft_axi_Dd2_rd0p05": {"peak_stress_fillet": {"value": 289.50e6, "unc": 0.0044, "ccx_fine": 289.30e6, "skfem_fine": 289.50e6}},
    "stepped_shaft_axi_Dd2_rd0p1": {"peak_stress_fillet": {"value": 223.01e6, "unc": 0.0028, "ccx_fine": 222.86e6, "skfem_fine": 223.01e6}},
    "stepped_shaft_axi_Dd2_rd0p2": {"peak_stress_fillet": {"value": 175.33e6, "unc": 0.0016, "ccx_fine": 175.43e6, "skfem_fine": 175.55e6}},
    "notched_bar_t5_r2": {"peak_stress_notch": {"value": 347.43e6, "unc": 0.00066, "ccx_fine": 354.72e6, "skfem_fine": 347.61e6}},
    "notched_bar_t5_r4": {"peak_stress_notch": {"value": 264.24e6, "unc": 0.00011, "ccx_fine": 267.91e6, "skfem_fine": 264.24e6}},
    "notched_bar_t10_r2": {"peak_stress_notch": {"value": 476.13e6, "unc": 0.0016, "ccx_fine": 486.72e6, "skfem_fine": 476.75e6}},
    "notched_bar_t10_r4": {"peak_stress_notch": {"value": 354.64e6, "unc": 0.00038, "ccx_fine": 359.55e6, "skfem_fine": 354.75e6}},
}

SOURCE = (
    "feacore reference convergence study: scikit-fem P2 isoparametric (3 meshes, h0/h = 1, 2, 4, "
    "local refinement at the stress raiser), Richardson extrapolation (Celik et al. 2008); "
    "cross-checked with CalculiX on the finest mesh."
)


def numerical_reference(problem_id: str, qoi: str) -> dict | None:
    return REFS.get(problem_id, {}).get(qoi)


def _regenerate(ids: list[str] | None = None) -> dict:  # pragma: no cover - maintenance tool
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from feacore.problems.registry import all_problems
    from feacore.solver import solve
    from feacore.verification.study import convergence_study

    targets = {"stepped_shaft": "peak_stress_fillet", "notched_bar": "peak_stress_notch"}
    out: dict = {}
    for p in all_problems(tags=["part"]):
        key = targets.get(p.family)
        if key is None or p.dim != 2 or "singular_trap" in p.tags or (ids and p.id not in ids):
            continue
        st = convergence_study(p, order=2, backend="skfem")
        q = next(q for q in st["qois"] if q["name"] == key)
        g = q["gci"]
        fine = q["values"][-1]
        p_ok = g.get("p") is not None and 1.0 <= g["p"] <= 3.5
        mono = g["convergence"] == "monotone" and g["phi_ext"] is not None and p_ok
        val = g["phi_ext"] if mono else fine
        unc = g["gci_fine"] if mono else abs(q["values"][-1] - q["values"][-2]) / abs(fine)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mesh = p.mesh(f"{tmp}/m.msh", st["sizes"][-1])
            sk = solve(p.case(mesh), backend="skfem").evaluate_all()[key]
            try:
                cx = solve(p.case(mesh), backend="ccx").evaluate_all()[key]
            except Exception:  # noqa: BLE001 - CalculiX optional
                cx = None
        out[p.id] = {key: {"value": float(f"{val:.5g}"), "unc": float(f"{unc:.2g}"),
                           "ccx_fine": None if cx is None else float(f"{cx:.5g}"),
                           "skfem_fine": float(f"{sk:.5g}")}}
    return out


if __name__ == "__main__":  # pragma: no cover
    import pprint

    pprint.pprint(_regenerate(), width=110, sort_dicts=True)
