"""Score an agent's reply against a NAFEMS benchmark: setup check, solve, reference comparison.

    python scripts/compare_nafems.py REPLY [--problem nafems_le10] [--root benchmarks] [--json OUT]

REPLY is the agent's raw reply (case JSON, optionally in a ```json fence) or a case .json file.
The mesh path in the case is resolved against benchmarks/<problem>/public/.

Report:
  setup      each reference BC matched / missing in the agent's case, plus extra agent BCs
  solve      DOFs, wall time, diagnostics status and flags (feacore.diagnostics)
  reference  per reference QoI: agent value, NAFEMS value, relative error, pass (within tolerance)
  verdict    pass = valid reply, setup matches, solution not failed, every reference QoI within tolerance
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def parse_reply(text: str) -> dict:
    """{"invalid": reason} or the case dict."""
    s = text.strip()
    if s.startswith("INVALID_CONFIG"):
        return {"invalid": s.split(":", 1)[1].strip() if ":" in s else ""}
    m = FENCE.search(s)
    body = m.group(1) if m else s[s.find("{"): s.rfind("}") + 1]
    return json.loads(body)


def _norm(bc) -> tuple:
    value = bc.value
    if isinstance(value, list):
        value = tuple(float(v) for v in value)
    elif value is not None:
        value = float(value)
    return (bc.type, bc.group, tuple(sorted(bc.components)) if bc.components else None, value)


def compare_setup(agent_case, ref_case) -> dict:
    """BC-by-BC comparison on (type, group, components, value). A "displacement" of every
    component to 0 and "fixed" are treated as the same BC."""
    def canon(bc):
        t = _norm(bc)
        if bc.type == "displacement" and t[2] == tuple(range(agent_case.dim)) and t[3] in (0.0, (0.0,) * agent_case.dim):
            return ("fixed", bc.group, None, None)
        if bc.type == "fixed":
            return ("fixed", bc.group, None, None)
        if t[3] is not None and isinstance(t[3], tuple) and len(set(t[3])) == 1 and t[2] and len(t[3]) == len(t[2]):
            return (t[0], t[1], t[2], t[3][0])   # [0, 0] on two components == 0
        return t

    ref = [canon(b) for b in ref_case.bcs]
    got = [canon(b) for b in agent_case.bcs]
    rows = [{"reference": list(r), "matched": r in got} for r in ref]
    extra = [list(g) for g in got if g not in ref]
    return {"bcs": rows, "extra": extra, "matches": all(r["matched"] for r in rows) and not extra}


def compare(reply: str, problem_id: str, root: Path) -> dict:
    from feacore.case import Case, CaseError
    from feacore.diagnostics import diagnose
    from feacore.problems.registry import get

    problem = get(problem_id)
    public = root / problem_id / "public"
    out = {"problem": problem_id, "verdict": "fail"}
    try:
        d = parse_reply(reply)
    except (json.JSONDecodeError, ValueError) as e:
        out["error"] = f"reply is neither INVALID_CONFIG nor parseable JSON: {e}"
        return out
    if "invalid" in d:
        out["error"] = f"agent declared INVALID_CONFIG ({d['invalid']}) but {problem_id} is a valid setup"
        return out
    try:
        case = Case.from_dict(d, base_dir=public).check()
    except CaseError as e:
        out["error"] = str(e)
        return out

    out["setup"] = compare_setup(case, problem.case(case.mesh_path(), order=case.analysis.order))
    try:
        rep, res = diagnose(case)
    except Exception as e:  # noqa: BLE001 - mesh / group errors are the agent's fault, report them
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["solve"] = {"status": rep.status, "flags": [f"{f.severity}:{f.id}" for f in rep.flags],
                    "n_dofs": rep.metrics.get("n_dofs"), "wall_time": rep.metrics.get("wall_time"),
                    "zz_eta_rel": rep.metrics.get("zz_eta_rel")}
    if res is None:
        return out

    # evaluate the reference QoIs on the agent's solution (by the reference definition, so a
    # misnamed or misplaced agent QoI cannot hide the comparison)
    agent_qois = {q.name: res.evaluate(q) for q in case.qois}
    ref_qois = {q.name: q for q in problem.qois()}
    rows = []
    for r in problem.references():
        v = res.evaluate(ref_qois[r.qoi])
        err = abs(v - r.value) / abs(r.value)
        rows.append({"qoi": r.qoi, "agent_value": v, "agent_reported": agent_qois.get(r.qoi),
                     "reference": r.value, "rel_error": err, "tolerance": r.tolerance,
                     "pass": err <= r.tolerance, "source": r.source})
    out["reference"] = rows
    ok = out["setup"]["matches"] and rep.status != "fail" and all(r["pass"] for r in rows)
    out["verdict"] = "pass" if ok else "fail"
    return out


def render(out: dict) -> str:
    lines = [f"{out['problem']}: {out['verdict'].upper()}"]
    if "error" in out:
        lines.append(f"  error: {out['error']}")
    if "setup" in out:
        s = out["setup"]
        lines.append(f"  setup: {'matches NAFEMS' if s['matches'] else 'DIFFERS from NAFEMS'}")
        for r in s["bcs"]:
            lines.append(f"    [{'x' if r['matched'] else ' '}] {r['reference']}")
        for e in s["extra"]:
            lines.append(f"    [+] extra: {e}")
    if "solve" in out:
        s = out["solve"]
        lines.append(f"  solve: {s['status']}, {s['n_dofs']} dofs, {s['wall_time']:.2f}s, "
                     f"ZZ error {s['zz_eta_rel']:.1%}" if s.get("wall_time") is not None else
                     f"  solve: {s['status']} (not solved)")
        if s["flags"]:
            lines.append(f"    flags: {', '.join(s['flags'])}")
    for r in out.get("reference", []):
        lines.append(f"  {r['qoi']}: {r['agent_value'] / 1e6:.4g} MPa vs NAFEMS {r['reference'] / 1e6:.4g} MPa "
                     f"-> error {r['rel_error']:.2%} (tol {r['tolerance']:.0%}) {'PASS' if r['pass'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reply")
    ap.add_argument("--problem", default="nafems_le10")
    ap.add_argument("--root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args(argv)
    out = compare(Path(args.reply).read_text(), args.problem, Path(args.root))
    print(render(out))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    return 0 if out["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
