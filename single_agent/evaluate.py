"""The scorecard of one analysis: deterministic, no LLM, the only module that reads the gold data.

Four parts, each in [0, 1], then a weighted composite (config.WEIGHTS):

  setup        Did the agent turn the design data into the right model? Its final setup and the reference
               setup are solved on the agent's own final mesh and compared on the global and face QoIs:
               within config.EQUIV_TOL they are equivalent (1.0), whatever the wording (a force or the same
               force per area, a bonded face as fixed or as all components zero). Otherwise 0.5 x the share of
               reference items (each support and load) the agent got right, as a diagnostic match by face,
               kind and value.
  answers      Each review question: a converging quantity earns 1 within the reference's tolerance and
               tol / error beyond it (reactions by magnitude when the key says sign_free); a singular quantity
               (expect "diverge") earns 1 when the agent reported it singular or not converged, 0 when it
               claimed a converged value.
  calibration  Converging answers whose stated uncertainty covers the reference (|value - ref| <= U |value|,
               plus a quarter of the reference's tolerance for the reference's own uncertainty) while U <= 10 %
               (a huge band covers anything and says nothing).
  study        A convergence study behind the final setup: 1.0 for 3+ meshes with that setup analysed with the
               study tool, 0.7 for 3+ meshes without it, 0.4 for 2, 0 for 1.

Also reported, outside the composite: the discretization error of the agent's final mesh with the
reference setup (meshing skill alone, whatever the setup), its gain over uniform refinement at the same
DOFs (from the gold study's nested meshes), whether reported values come from the agent's own solves,
and the cost (DOFs, solves, turns, tokens, $).
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from single_agent import config, workspace
from single_agent.harness import build_case, canonical_setup, evaluate_qois, quiet

SUPPORT_TYPES = ("fixed", "displacement", "symmetry")
LOAD_TYPES = ("pressure", "traction", "force")


# --------------------------------------------------------------------------- helpers


def _rel(a, b) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1e-300)


def _faces(group: str | None, groups: dict) -> frozenset:
    if not group:
        return frozenset()
    if group in groups:
        return frozenset(groups[group])
    return frozenset([group])


@functools.lru_cache(maxsize=16)
def _face_info(problem_id: str) -> dict:
    """{f<tag>: {area, outward normal (flat faces) or None}}, computed from the CAD by ace.geometry_info on the
    part's STEP, independently of the gold part data (which stated both boss shoulders of abc_00000003 inward
    until `scripts/gold.py restate`)."""
    from ace import geometry_info
    from feacore.problems import registry
    from single_agent.part import BlindPart, label_for, register
    part = register(BlindPart(registry.get(problem_id).step_path, "eval-" + label_for(problem_id)))
    with quiet():
        d = geometry_info.describe(part)
    return {f"f{f['tag']}": {"area": f.get("area"), "normal": f.get("normal")} for f in d["faces"]}


def canonical_items(setup: dict, groups: dict, faces: dict) -> list[dict]:
    """Supports and loads as comparable items: {kind: support|prescribed|load, faces, comps | force | pressure}.
    A support's components are a set of axes ("n" = the face normal, for symmetry); a load on a flat face is
    also expressed as its total force, so a pressure, a traction and a force with the same resultant match."""
    items = []
    for b in setup.get("bcs", []):
        fs = _faces(b.get("group"), groups)
        t, v = b.get("type"), b.get("value")
        if t == "fixed":
            items.append({"kind": "support", "faces": fs, "comps": frozenset({0, 1, 2}), "src": b})
        elif t == "displacement":
            vals = v if isinstance(v, list) else [v] * len(b.get("components") or [])
            zero = all(isinstance(x, (int, float)) and abs(x) < 1e-15 for x in vals)
            items.append({"kind": "support" if zero else "prescribed", "faces": fs,
                          "comps": frozenset(b.get("components") or []), "src": b})
        elif t == "symmetry":
            items.append({"kind": "support", "faces": fs, "comps": frozenset({"n"}), "src": b})
        elif t in LOAD_TYPES:
            it = {"kind": "load", "faces": fs, "type": t, "src": b}
            area = sum(faces.get(f, {}).get("area") or 0.0 for f in fs)
            normals = [faces.get(f, {}).get("normal") for f in fs]
            flat = all(n is not None for n in normals) and len({tuple(np.round(n, 6)) for n in normals}) == 1
            try:
                if t == "pressure":
                    it["pressure"] = float(v)
                    if flat and area:
                        it["force"] = list(-float(v) * area * np.asarray(normals[0], float))
                elif t == "traction":
                    it["force"] = list(np.asarray(v, float) * area)
                else:
                    it["force"] = list(np.asarray(v, float))
            except (TypeError, ValueError):
                pass                                # an expression: compared only by the equivalence solve
            items.append(it)
    return items


def _axis_of(normal) -> int | None:
    if normal is None:
        return None
    n = np.abs(np.asarray(normal, float))
    return int(np.argmax(n)) if n.max() > 0.999 else None


def match_items(ref_items: list[dict], got_items: list[dict], faces: dict) -> dict:
    """Each reference item: match / value differs / type differs / faces differ / missing; plus extra items."""
    rows, used = [], set()

    def comps_of(it):
        c = set(it["comps"])
        if "n" in c:
            ax = _axis_of(faces.get(next(iter(it["faces"]), ""), {}).get("normal"))
            c = (c - {"n"}) | ({ax} if ax is not None else {"n"})
        return c

    for r in ref_items:
        cands = [(i, g) for i, g in enumerate(got_items) if g["kind"] == r["kind"] and g["faces"] & r["faces"]]
        status, detail = "missing", ""
        for i, g in cands:
            if g["faces"] != r["faces"]:
                status, detail = "faces differ", f"applied on {sorted(g['faces'])}"
                continue
            if r["kind"] == "support":
                ok = comps_of(g) == comps_of(r)
                status, detail = ("match", "") if ok else ("value differs", f"components {sorted(map(str, comps_of(g)))}")
            elif r["kind"] == "load":
                if "force" in r and "force" in g:
                    ok = _rel_vec(g["force"], r["force"]) <= 0.01
                    detail = "" if ok else f"resultant {np.round(g['force'], 1).tolist()} N vs {np.round(r['force'], 1).tolist()} N"
                elif "pressure" in r and "pressure" in g:
                    ok = _rel(g["pressure"], r["pressure"]) <= 0.01
                    detail = "" if ok else f"pressure {g['pressure']:.4g} vs {r['pressure']:.4g} Pa"
                else:
                    ok, detail = False, f"{g['type']} where a {r['type']} was intended"
                status = "match" if ok else "value differs"
            else:
                status = "match"
            used.add(i)
            if status == "match":
                break
        rows.append({"reference": _describe(r), "status": status, **({"detail": detail} if detail else {})})
    extra = [_describe(g) for i, g in enumerate(got_items) if i not in used
             and not any(g["kind"] == r["kind"] and g["faces"] & r["faces"] for r in ref_items)]
    return {"items": rows, "extra": extra,
            "matched": sum(r["status"] == "match" for r in rows), "of": len(rows)}


def _rel_vec(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _describe(it: dict) -> str:
    b = it["src"]
    where = "+".join(sorted(it["faces"])) or f"point {b.get('point')}"
    val = b.get("value")
    if it["kind"] in ("support", "prescribed"):
        if b["type"] == "fixed":
            return f"fixed on {where}"
        if b["type"] == "symmetry":
            return f"symmetry on {where}"
        return f"displacement {b.get('components')} = {val} on {where}"
    return f"{b['type']} {val} on {where}"


# --------------------------------------------------------------------------- the parts


def _solve(mesh: dict, setup: dict, qois: list[dict]):
    from feacore.solver import solve
    case = build_case(mesh, {**setup, "qois": qois})
    with quiet():
        res = solve(case, backend="skfem")
    return res, evaluate_qois(res, qois)


def _uniform_error(problem, qoi: str, dofs: int) -> float | None:
    """Relative error of the gold study's uniform (nested) meshes at `dofs`, interpolated log-log between
    levels (None outside the levels' DOF range or when the QoI is not a reference)."""
    ref = next((r for r in problem.references() if r.qoi == qoi), None)
    runs = problem.study().get("runs") or []
    pts = [(r["n_dofs"], abs(r["qois"][qoi] - ref.value) / abs(ref.value)) for r in runs
           if ref is not None and qoi in r.get("qois", {})]
    pts = sorted(p for p in pts if p[1] > 0)
    if len(pts) < 2 or not pts[0][0] <= dofs <= pts[-1][0]:
        return None
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    return float(np.exp(np.interp(math.log(dofs), x, y)))


def evaluate(problem_id: str, run: dict) -> dict:
    """The scorecard of one run record (single_agent.agent.run_problem)."""
    from feacore.problems import registry
    p = registry.get(problem_id)
    key = workspace.load_key(problem_id)
    refs = {r.qoi: r for r in p.references()}
    state = run.get("session") or {}
    report = state.get("report")
    solves, meshes = state.get("solves", []), state.get("meshes", [])
    card: dict = {"problem_id": problem_id, "submitted": report is not None,
                  "parts": {"setup": 0.0, "answers": 0.0, "calibration": 0.0, "study": 0.0}}
    if key.get("suspect"):
        card["reference_suspect"] = key["suspect"]

    final = None
    if report and isinstance(report.get("final_solve"), int) and 1 <= report["final_solve"] <= len(solves) \
            and solves[report["final_solve"] - 1].get("ok"):
        final = solves[report["final_solve"] - 1]
    if final is None:
        final = next((s for s in reversed(solves) if s.get("ok")), None)
        card["final_from"] = "last successful solve" if final else None
    card["cost"] = _cost(run, solves)
    if final is None:
        card.update(failed="no successful solve", score=0.0)
        card["answers"] = _answers(key, report, refs, solves, state)[0]
        return card

    mesh = meshes[final["mesh"] - 1]
    groups = (mesh.get("request") or {}).get("groups") or {}
    card["final_solve"] = final["number"]
    card["final_mesh"] = {k: mesh.get(k) for k in ("number", "nodes", "order", "min_quality", "h_used")}
    card["final_mesh"]["dofs"] = final.get("dofs")

    # ---- setup: reference vs agent on the agent's own final mesh
    ref_case = p.case(mesh["path"], order=mesh["order"]).to_dict()
    ref_setup = {"material": next(iter(ref_case["materials"].values())), "bcs": ref_case["bcs"],
                 "loads": ref_case["loads"]}
    cmp_qois = [q for q in ref_case["qois"]
                if q["type"] in ("strain_energy", "mean", "max") and q["name"] != "vm_max"]
    all_ref_qois = ref_case["qois"]
    faces = _face_info(problem_id)
    try:
        ref_res, ref_vals = _solve(mesh, ref_setup, all_ref_qois)
    except Exception as e:  # noqa: BLE001
        card.update(failed=f"reference setup could not be solved on the agent's mesh: {type(e).__name__}: {e}",
                    score=0.0)
        return card
    setup_card = {"material_ok": all(math.isclose(float((final["setup"].get("material") or {}).get(k, 0)),
                                                  float(ref_setup["material"][k]), rel_tol=1e-3)
                                     for k in ("E", "nu"))}
    agent_setup = {k: final["setup"].get(k) for k in ("material", "bcs", "loads")}
    try:
        _, got_vals = _solve(mesh, agent_setup, cmp_qois)
        diffs = {q["name"]: _rel(got_vals.get(q["name"]), ref_vals[q["name"]]) for q in cmp_qois}
        worst = max((d for d in diffs.values() if d is not None), default=math.inf)
        setup_card.update(qoi_differences={k: _r(v) for k, v in diffs.items()}, max_difference=_r(worst),
                          equivalent=bool(worst <= config.EQUIV_TOL))
    except Exception as e:  # noqa: BLE001  e.g. an unsupported part: the agent's setup does not solve
        setup_card.update(equivalent=False, solve_error=f"{type(e).__name__}: {e}")
    m = match_items(canonical_items(ref_setup, {}, faces), canonical_items(agent_setup, groups, faces), faces)
    setup_card.update(match=m)
    card["setup"] = setup_card
    card["parts"]["setup"] = 1.0 if setup_card["equivalent"] else 0.5 * m["matched"] / max(m["of"], 1)

    # ---- meshing: the agent's final mesh with the reference setup, against the gold references
    disc = {}
    for name, r in refs.items():
        e = _rel(ref_vals.get(name), r.value)
        u = _uniform_error(p, name, final.get("dofs") or 0)
        disc[name] = {"error": _r(e), "tolerance": _r(r.tolerance), "uniform_error_same_dofs": _r(u),
                      "gain_vs_uniform": _r(u / e) if u and e else None}
    card["meshing"] = {"dofs": final.get("dofs"), "by_qoi": disc,
                       "within_tolerance": sum(1 for d in disc.values() if d["error"] is not None
                                               and d["error"] <= d["tolerance"]),
                       "of": len(disc)}

    # ---- answers and calibration
    rows, ans_score, cal_score = _answers(key, report, refs, solves, state, final)
    card["answers"] = rows
    card["parts"]["answers"] = ans_score
    card["parts"]["calibration"] = cal_score

    # ---- study
    card["study"] = _study(state, final)
    card["parts"]["study"] = card["study"]["score"]

    card["score"] = round(sum(config.WEIGHTS[k] * v for k, v in card["parts"].items()), 4)
    card["parts"] = {k: round(v, 4) for k, v in card["parts"].items()}
    return card


def _r(x, n: int = 4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(f"{x:.{n}g}")


def _answers(key: dict, report: dict | None, refs: dict, solves: list, state: dict, final: dict | None = None):
    given = {str(a.get("id", "")).strip().upper(): a for a in (report or {}).get("answers") or []}
    rows, credits, honest = [], [], []
    final_key = canonical_setup(final["setup"]) if final else None
    for q in key["questions"]:
        a = given.get(q["id"].upper())
        row = {"id": q["id"], "qoi": q["qoi"], "expect": q["expect"]}
        if q["expect"] == "diverge":
            status = (a or {}).get("status")
            row.update(status=status, credit=1.0 if status in ("singular", "not_converged") else 0.0)
            row["verdict"] = ("recognised as mesh-dependent" if row["credit"] else
                              "missing" if a is None else "claimed a converged value for a singular quantity")
            credits.append(row["credit"])
            rows.append(row)
            continue
        ref = refs[q["qoi"]]
        row.update(reference=ref.value, tolerance=ref.tolerance)
        v = (a or {}).get("value")
        if a is None or not isinstance(v, (int, float)):
            row.update(credit=0.0, verdict="missing" if a is None else "no value")
            credits.append(0.0)
            honest.append(False)
            rows.append(row)
            continue
        sign_free = bool(q.get("sign_free"))
        err = _rel(abs(v), abs(ref.value)) if sign_free else _rel(v, ref.value)
        credit = 1.0 if err <= ref.tolerance else ref.tolerance / err
        U = a.get("uncertainty")
        U = abs(float(U)) if isinstance(U, (int, float)) else None
        dist = abs(abs(v) - abs(ref.value)) if sign_free else abs(v - ref.value)
        # The reference is itself uncertain (its tolerance is max(family limit, 2 x its GCI)): a quarter of the
        # tolerance is allowed on top of the agent's band, so a sharp, correct band is not failed by the
        # reference's own scatter.
        covered = U is not None and dist <= U * abs(v) + 0.25 * ref.tolerance * abs(ref.value)
        row.update(value=v, error=_r(err), credit=_r(credit), uncertainty=U, covered=covered,
                   honest=bool(covered and U <= 0.10), status=a.get("status"),
                   supported=_supported(a, v, U, solves, state, final_key))
        if sign_free and v * ref.value < 0:
            row["note"] = "opposite sign to the reference convention (scored by magnitude)"
        row["verdict"] = ("within tolerance" if err <= ref.tolerance else f"{err:.1%} off (tolerance {ref.tolerance:.1%})")
        credits.append(row["credit"])
        honest.append(row["honest"])
        rows.append(row)
    ans = sum(credits) / len(credits) if credits else 0.0
    cal = sum(honest) / len(honest) if honest else 0.0
    return rows, ans, cal


def _supported(a: dict, v: float, U, solves: list, state: dict, final_key) -> bool:
    """Does the reported value come from the agent's own analysis? It must be within max(2 %, 2 U) of the
    named QoI in one of its solves with the final setup, or of an extrapolated value in one of its studies."""
    name = a.get("quantity")
    cands = []
    for s in solves:
        if s.get("ok") and (final_key is None or canonical_setup(s["setup"]) == final_key):
            vals = s.get("qois") or {}
            cands += [vals[name]] if name in vals else list(vals.values())
    for st in state.get("studies", []):
        for qn, row in (st.get("qois") or {}).items():
            if name in (None, qn):
                cands += [x for x in (row.get("extrapolated"), (row.get("power_fit") or {}).get("limit")) if x is not None]
    tol = max(0.02, 2 * (U or 0.0))
    return any(c is not None and abs(abs(v) - abs(c)) <= tol * max(abs(c), 1e-300) for c in cands)


def _study(state: dict, final: dict) -> dict:
    fk = canonical_setup(final["setup"])
    same = [s for s in state.get("solves", []) if s.get("ok") and canonical_setup(s["setup"]) == fk]
    meshes = sorted({s["mesh"] for s in same})
    nums = {s["number"] for s in same}
    studied = [st["solves"] for st in state.get("studies", []) if len(st["solves"]) >= 3 and set(st["solves"]) <= nums]
    n = len(meshes)
    score = 1.0 if n >= 3 and studied else 0.7 if n >= 3 else 0.4 if n == 2 else 0.0
    h = sorted((s["h_rep"] for s in same), reverse=True)
    return {"meshes_with_final_setup": n, "studies_on_final_setup": studied, "score": score,
            "h_rep_ratios": [_r(h[i] / h[i + 1], 3) for i in range(len(h) - 1)]}


def _cost(run: dict, solves: list) -> dict:
    usage = run.get("usage") or []
    ok = [s for s in solves if s.get("ok")]
    return {"solves": len(solves), "solved_dofs_total": sum(s.get("dofs") or 0 for s in ok),
            "max_dofs": max((s.get("dofs") or 0 for s in ok), default=0),
            "turns": run.get("turns"), "tokens_in": sum(u.get("input", 0) + u.get("cache_read", 0)
                                                        + u.get("cache_write", 0) for u in usage),
            "tokens_out": sum(u.get("output", 0) for u in usage),
            "usd": round(sum(u.get("usd", 0.0) for u in usage), 4),
            "wall_s": run.get("wall_s")}


def summary_line(card: dict) -> str:
    p = card["parts"]
    s = card.get("setup", {})
    return (f"{card['problem_id']}: score {card.get('score', 0):.2f} | setup {'equivalent' if s.get('equivalent') else 'WRONG'}"
            f" ({s.get('match', {}).get('matched', 0)}/{s.get('match', {}).get('of', 0)} items) | answers {p['answers']:.2f}"
            f" | calibration {p['calibration']:.2f} | study {p['study']:.2f} | ${card['cost']['usd']:.2f}")


if __name__ == "__main__":   # python -m single_agent.evaluate RUN.json [PROBLEM_ID]
    import sys
    rec = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(evaluate(sys.argv[2] if len(sys.argv) > 2 else rec["problem_id"], rec), indent=1, default=float))
