"""Interpretation layer on top of ``feacore.verification``: plain-word flag explanations,
per-QoI verdicts, submission analysis (an agent's own mesh sequence) and the multi-agent
consensus view. Pure numpy; no FastAPI or gmsh here."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from feacore.case import STRESS_LIKE, QoI
from feacore.verification.gci import analyze

# --------------------------------------------------------------------------- flags in plain words

# severity: info < warning < serious < critical
FLAG_INFO = {
    "divergent": ("critical", "Diverging under refinement",
                  "Each refinement moves the value further away instead of closer. The quantity has no "
                  "mesh-independent value as set up: typically a stress singularity (sharp re-entrant "
                  "corner, point load, point support) or a setup error such as a wrong boundary condition."),
    "singular_suspected": ("critical", "Singularity suspected",
                           "A stress-type quantity that grows or converges far slower than theory as the mesh is "
                           "refined. This is the signature of a geometric or load singularity: the 'peak' value is "
                           "set by the mesh, not by the physics. Report it at a distance from the corner, or "
                           "model the fillet / distributed load instead."),
    "oscillatory": ("warning", "Oscillating",
                    "The value goes up and down between successive meshes. Richardson extrapolation is less "
                    "reliable; the error band is only indicative."),
    "not_asymptotic": ("warning", "Not yet in the asymptotic range",
                       "The three meshes do not show a consistent convergence rate (GCI ratio far from 1). Refine "
                       "further before trusting the extrapolated value and its error band."),
    "p_far_below_theory": ("serious", "Order far below theory",
                           "Observed convergence order is less than half the theoretical order for these elements. "
                           "Points to reduced smoothness (singularity, sharp corner, contact-like constraint), or to "
                           "meshes that are not refined uniformly."),
    "p_below_theory": ("info", "Order below theory",
                       "Converging, but more slowly than the element order predicts (common for stress peaks "
                       "near a notch)."),
    "p_above_theory": ("warning", "Order above theory",
                       "Converging faster than theory allows; usually error cancellation or meshes too close to "
                       "each other. The GCI band may be optimistic."),
    "order_undefined": ("warning", "Order undefined",
                        "The observed order could not be computed from these three values."),
    "refinement_ratio_below_1.3": ("warning", "Meshes too similar",
                                   "Refinement ratio below 1.3: differences between meshes are small and can be "
                                   "dominated by noise rather than discretization error."),
    "phi_near_zero": ("info", "Value near zero",
                      "The quantity is close to zero, so relative errors are not meaningful."),
    "two_solutions_identical": ("warning", "Two meshes give identical values",
                                "Two of the three values coincide; the convergence rate cannot be estimated."),
    "outlier_vs_consensus": ("serious", "Disagrees with the other agents",
                             "This answer lies outside the population consensus by more than its own error band "
                             "and the reference tolerance. Likely a setup error (wrong load, units, support or "
                             "QoI definition)."),
    "insufficient_levels": ("info", "Fewer than three meshes",
                            "Richardson extrapolation and the GCI need at least three meshes."),
    "converged_plateau": ("info", "Converged to a plateau",
                          "Successive meshes change the value by less than 0.1%: too little to measure a convergence "
                          "rate (round-off, geometry and quadrature errors dominate). Treat the value as "
                          "mesh-converged; the uncertainty shown is the fine-medium difference instead of a GCI."),
    "expected_divergent": ("info", "Expected to diverge",
                           "The problem is designed so that this quantity does not converge (a singular point). "
                           "Divergence here is the correct answer."),
}

VERDICTS = {
    "converged": ("good", "Converged"),
    "exact": ("good", "Exact on every mesh"),
    "plateau": ("good", "Converged (plateau)"),
    "converging": ("warning", "Converging, not asymptotic"),
    "slow": ("serious", "Converging slowly"),
    "oscillating": ("warning", "Oscillating"),
    "diverging": ("critical", "Diverging"),
    "expected_divergence": ("good", "Diverges, as expected"),
    "unexpected_convergence": ("serious", "Converges, but should diverge"),
    "insufficient": ("info", "Need three meshes"),
    "undetermined": ("warning", "Undetermined"),
}


PLATEAU_REL = 1e-3  # successive relative changes below this: converged, rate not measurable


def _plateau(values: list[float]) -> float | None:
    """|fine - medium| when the last two successive relative changes are all < PLATEAU_REL."""
    v = [float(x) for x in values if x is not None]
    if len(v) < 3:
        return None
    scale = max(abs(x) for x in v[-3:])
    if scale == 0:
        return None
    changes = [abs(v[-1] - v[-2]), abs(v[-2] - v[-3])]
    return changes[0] if max(changes) < PLATEAU_REL * scale else None


def explain(flag: str) -> dict:
    sev, title, text = FLAG_INFO.get(flag, ("info", flag.replace("_", " "), ""))
    return {"flag": flag, "severity": sev, "title": title, "text": text}


def is_stress_like(qtype: str | None, field: str | None) -> bool:
    return field in STRESS_LIKE and qtype in ("point", "max", "min", "max_abs")


def interpret_entry(entry: dict) -> dict:
    """Verdict, derived flags and explanations for one QoI entry (a ``convergence_study``
    ``qois[i]`` dict or an ``analyze_submission`` entry)."""
    g = entry.get("gci")
    expectation = entry.get("expectation", "converge")
    flags = list(g.get("flags", [])) if g else []
    n = len([v for v in entry.get("values", []) if v is not None])
    if not g:
        if n < 3:
            flags.append("insufficient_levels")
        verdict = "insufficient" if n < 3 else "undetermined"
    else:
        conv = g.get("convergence")
        stress = is_stress_like(entry.get("type"), entry.get("field"))
        plateau = _plateau(entry.get("values", [])) if conv not in ("monotone", "exact") or g.get("p") is None else None
        if plateau is not None:
            flags = [f for f in flags if f not in ("divergent", "oscillatory", "order_undefined")]
            flags.append("converged_plateau")
        elif stress and (conv in ("divergent", "oscillatory_divergent") or "p_far_below_theory" in flags):
            flags.append("singular_suspected")
        if expectation == "diverge":
            diverged = conv in ("divergent", "oscillatory_divergent") or "p_far_below_theory" in flags
            verdict = "expected_divergence" if diverged else "unexpected_convergence"
            flags.append("expected_divergent")
        elif conv == "exact":
            verdict = "exact"
        elif plateau is not None:
            verdict = "plateau"
        elif conv in ("divergent", "oscillatory_divergent"):
            verdict = "diverging"
        elif conv == "oscillatory":
            verdict = "oscillating"
        elif conv == "monotone":
            if g.get("p") is None:
                verdict = "undetermined"
            elif "p_far_below_theory" in flags:
                verdict = "slow"
            elif "not_asymptotic" in flags:
                verdict = "converging"
            else:
                verdict = "converged"
        else:
            verdict = "undetermined"
    flags = list(dict.fromkeys(flags))
    status, label = VERDICTS[verdict]
    explanations = [explain(f) for f in flags]
    if verdict == "expected_divergence":  # the singularity is the designed answer, not an alarm
        for e in explanations:
            if e["flag"] in ("divergent", "singular_suspected", "p_far_below_theory"):
                e["severity"] = "info"
    out = {"verdict": verdict, "status": status, "label": label, "flags": flags,
           "explanations": explanations}
    vals = [v for v in entry.get("values", []) if v is not None]
    fine = vals[-1] if vals else None
    out["fine"] = fine
    if g and g.get("gci_fine") is not None and fine is not None:
        out["band_abs"] = abs(g["gci_fine"] * fine)
    elif verdict == "plateau":
        out["band_abs"] = _plateau(entry.get("values", []))
    else:
        out["band_abs"] = None
    out["phi_ext"] = g.get("phi_ext") if g else None
    out["p"] = g.get("p") if g else None
    out["p_theory"] = entry.get("p_theory")
    out["summary"] = _summary_sentence(entry, out)
    return out


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}g}"


def _summary_sentence(entry: dict, out: dict) -> str:
    g = entry.get("gci") or {}
    v = out["verdict"]
    if v == "insufficient":
        return "Fewer than three meshes: no Richardson / GCI estimate."
    if v in ("converged", "converging", "slow"):
        s = (f"Observed order p = {_fmt(out['p'], 3)} (theory {_fmt(out['p_theory'], 2)}); "
             f"extrapolated value {_fmt(out['phi_ext'], 6)}")
        if g.get("gci_fine") is not None:
            s += f"; fine-mesh uncertainty (GCI) {100 * g['gci_fine']:.2g}%"
        return s + "."
    if v == "exact":
        return "The value is identical on all meshes (the discretization represents it exactly)."
    if v == "plateau":
        return (f"Converged: the last meshes differ by less than {100 * PLATEAU_REL:.1g}% "
                f"(fine-mesh value {_fmt(out['fine'], 6)}); the convergence rate is below measurement resolution.")
    if v == "diverging":
        return "Successive differences grow: the value does not converge under refinement."
    if v == "oscillating":
        return f"Values oscillate between meshes; extrapolated estimate {_fmt(out['phi_ext'], 6)} is indicative only."
    if v == "expected_divergence":
        return "Diverges under refinement, as expected for this singular quantity."
    if v == "unexpected_convergence":
        return "This quantity should be singular but the values converge: check the setup."
    return "Convergence could not be classified."


def interpret_study(study: dict) -> dict:
    """{qoi name: interpret_entry(...)} for a ``convergence_study`` result dict."""
    return {q["name"]: interpret_entry(q) for q in study.get("qois", [])}


# --------------------------------------------------------------------------- submissions


def _qoi_defs(problem_id: str | None) -> tuple[dict, object]:
    if not problem_id:
        return {}, None
    try:
        from feacore.problems.registry import get

        p = get(problem_id)
        return {q.name: q for q in p.qois()}, p
    except Exception:  # noqa: BLE001 - unknown problem: analyse without definitions
        return {}, None


def expected_order(qdef: QoI | None, order: int | None, problem=None) -> float | None:
    if qdef is None or order is None:
        return None
    from feacore.verification.study import default_expected_order

    p = None
    if problem is not None:
        try:
            p = problem.expected_order(qdef, order)
        except Exception:  # noqa: BLE001
            p = None
    return default_expected_order(qdef, order) if p is None else p


def analyze_submission(problem_id: str | None, levels: list[dict], order: int | None = None) -> dict:
    """Per-QoI Richardson/GCI analysis of an agent's runs (``levels``: h + qois)."""
    defs, problem = _qoi_defs(problem_id)
    expect = {}
    if problem is not None:
        try:
            expect = problem.expectations()
        except Exception:  # noqa: BLE001
            expect = {}
    lv = sorted(levels, key=lambda l: -float(l["h"]))  # coarse -> fine
    orders = {l.get("order") for l in lv if l.get("order") is not None}
    if order is None and len(orders) == 1:
        order = orders.pop()
    names = list(defs) + [n for l in lv for n in l["qois"] if n not in defs]
    names = list(dict.fromkeys(names))
    out = []
    for name in names:
        pts = [(float(l["h"]), l["qois"].get(name), l.get("run_id")) for l in lv]
        pts = [(h, float(v), r) for h, v, r in pts if v is not None and math.isfinite(float(v))]
        if not pts:
            continue
        q = defs.get(name)
        entry = {
            "name": name,
            "type": q.type if q else None,
            "field": q.field if q else None,
            "h": [p[0] for p in pts],
            "values": [p[1] for p in pts],
            "run_ids": [p[2] for p in pts],
            "p_theory": expected_order(q, order, problem),
            "expectation": expect.get(name, "converge"),
            "gci": None,
            "fit": None,
        }
        hs = np.array(entry["h"])
        if len(pts) >= 3 and len(np.unique(hs)) == len(hs):
            try:
                entry.update(analyze(hs, np.array(entry["values"]), p_theory=entry["p_theory"]))
            except Exception as e:  # noqa: BLE001
                entry["analysis_error"] = f"{type(e).__name__}: {e}"
        entry["interpretation"] = interpret_entry(entry)
        out.append(entry)
    return {"order": order, "n_levels": len(lv), "qois": out}


# --------------------------------------------------------------------------- consensus


def _best_estimate(interp: dict) -> float | None:
    """Extrapolated value when the convergence is trustworthy, else the fine-mesh value."""
    if interp.get("verdict") in ("converged", "converging", "exact") and interp.get("phi_ext") is not None:
        return interp["phi_ext"]
    return interp.get("fine")


def consensus(problem_id: str, submissions: list[dict], references: list[dict] | None = None,
              reference_studies: list[dict] | None = None, qoi_names: list[str] | None = None) -> dict:
    """Multi-agent view: per QoI, every submission's fine value +- GCI band, extrapolated value,
    flags; robust population consensus (median / MAD) and outliers; reference values."""
    refs = {r["qoi"]: r for r in (references or [])}
    names = list(qoi_names or [])
    for s in submissions:
        for e in (s.get("analysis") or {}).get("qois", []):
            if e["name"] not in names:
                names.append(e["name"])
    qois = []
    for name in names:
        ref = refs.get(name)
        tol = ref.get("tolerance", 0.02) if ref else 0.02
        rows = []
        for s in submissions:
            e = next((e for e in (s.get("analysis") or {}).get("qois", []) if e["name"] == name), None)
            if e is None:
                continue
            it = interpret_entry(e)  # recomputed: stored records follow the current rules
            g = e.get("gci") or {}
            row = {
                "submission_id": s.get("submission_id"),
                "agent_id": s.get("agent_id"),
                "created_at": s.get("created_at"),
                "n_levels": len(e.get("values", [])),
                "h_fine": e["h"][-1] if e.get("h") else None,
                "fine": it.get("fine"),
                "band_abs": it.get("band_abs"),
                "phi_ext": it.get("phi_ext"),
                "p": it.get("p"),
                "p_theory": e.get("p_theory"),
                "convergence": g.get("convergence"),
                "gci_fine": g.get("gci_fine"),
                "verdict": it.get("verdict"),
                "status": it.get("status"),
                "label": it.get("label"),
                "flags": list(it.get("flags", [])),
                "best": _best_estimate(it),
                "values": e.get("values"),
                "h": e.get("h"),
                "run_ids": e.get("run_ids"),
            }
            if ref is not None and ref.get("value"):
                rv = ref["value"]
                row["err_ref_fine"] = (row["fine"] - rv) / rv if row["fine"] is not None else None
                row["err_ref_best"] = (row["best"] - rv) / rv if row["best"] is not None else None
                row["within_tolerance"] = (abs(row["err_ref_best"]) <= tol) if row["err_ref_best"] is not None else None
            rows.append(row)
        best = np.array([r["best"] for r in rows if r["best"] is not None], dtype=float)
        cons = None
        if best.size:
            med = float(np.median(best))
            mad = float(np.median(np.abs(best - med)))
            sigma = 1.4826 * mad
            cons = {"n": int(best.size), "median": med, "mad": mad, "sigma_robust": sigma,
                    "min": float(best.min()), "max": float(best.max()),
                    "spread_rel": (float(best.max() - best.min()) / abs(med)) if med else None}
            if best.size >= 3:
                for r in rows:
                    if r["best"] is None:
                        continue
                    thr = max(3 * sigma, r["band_abs"] or 0.0, tol * abs(med))
                    r["deviation"] = r["best"] - med
                    if abs(r["best"] - med) > thr:
                        r["flags"].append("outlier_vs_consensus")
                cons["n_outliers"] = sum("outlier_vs_consensus" in r["flags"] for r in rows)
            in_band = [r for r in rows if r["band_abs"] is not None and r["fine"] is not None
                       and abs(r["fine"] - med) <= r["band_abs"]]
            cons["n_band_contains_median"] = len(in_band)
            if ref is not None and ref.get("value"):
                cons["err_ref"] = (med - ref["value"]) / ref["value"]
        for r in rows:
            r["explanations"] = [explain(f) for f in r["flags"]]
        studies = []
        for st in reference_studies or []:
            res = st.get("result") or st
            e = next((q for q in res.get("qois", []) if q["name"] == name), None)
            if e is None:
                continue
            it = interpret_entry(e)
            studies.append({"study_id": st.get("study_id"), "backend": res.get("backend"),
                            "order": res.get("order"), "source": st.get("source", "validation"),
                            "fine": it["fine"], "band_abs": it["band_abs"], "phi_ext": it["phi_ext"],
                            "p": it["p"], "verdict": it["verdict"], "label": it["label"],
                            "status": it["status"], "flags": it["flags"]})
        qois.append({"name": name, "reference": ref, "tolerance": tol, "consensus": cons,
                     "submissions": rows, "reference_studies": studies})
    agents = sorted({s.get("agent_id") or "?" for s in submissions})
    return {"problem_id": problem_id, "n_submissions": len(submissions), "agents": agents, "qois": qois}


def reference_dicts(problem) -> list[dict]:
    try:
        return [asdict(r) for r in problem.references()]
    except Exception:  # noqa: BLE001
        return []
