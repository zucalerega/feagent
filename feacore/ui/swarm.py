"""Read model for the ACE agent swarm (``GET /api/swarm``), built from run-store records written by the
ACE bridge. Contract (see docs/ui.md):

* each ACE solve = a run record: ``source="ace"``, ``agent_id="ace:<ace_run_id>"``, ``problem_id``,
  ``meta = {source: "ace", loop_id, generation, ace_run_id, solve_index, mesh: {h, order, refinements, ...}}``
* each finished ACE agent run = a submission: ``agent_id="ace:<ace_run_id>"``, ``run_ids`` = its successful
  solves, ``meta = {source: "ace", loop_id, generation, ace_run_id, submitted_run_id, score, gates: [...],
  setup_diagnostics: [...], cited_bullets: [...], tokens?}``
* ``score = log10(e_uniform / e_agent)``: 0 = as good as the best uniform mesh with the same node budget,
  +1 = 10x better; a failed gate scores -2. An agent run with no successful solve is a submission with
  ``run_ids = []`` and score -2 (gate "no successful solve"); a crashed run has score None and ``meta.error``.
* ``meta.mesh.refinements`` is the agent's list of refinement dicts; ``solve_index`` counts from 0;
  ``tokens`` = {input, output, cache_write, cache_read, total}. ACE ids are unique only within a loop.
"""

from __future__ import annotations

import math
import time
from datetime import datetime

ACTIVE_WINDOW_S = 180.0  # records newer than this mark the loop as running (the page polls faster)


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _is_ace(rec: dict) -> bool:
    return rec.get("source") == "ace" or (rec.get("meta") or {}).get("source") == "ace"


def _gen(meta: dict):
    g = meta.get("generation")
    try:
        return int(g)
    except (TypeError, ValueError):
        return g


def _ace_id(rec: dict) -> str | None:
    m = rec.get("meta") or {}
    if m.get("ace_run_id") is not None:
        return str(m["ace_run_id"])
    a = rec.get("agent_id") or ""
    return a[4:] if a.startswith("ace:") else (a or None)


def _names(items) -> list[str]:
    """Gates / diagnostics may be strings or dicts: keep a short readable name for each."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append(str(it.get("name") or it.get("gate") or it.get("id") or it.get("flag") or it.get("code") or it))
        else:
            out.append(str(it))
    return out


def _tokens(t):
    if isinstance(t, dict):
        vals = [v for v in t.values() if isinstance(v, (int, float))]
        return int(t.get("total", sum(vals))) if vals else None
    return int(t) if isinstance(t, (int, float)) else None


def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mesh_summary(mesh: dict | None, info: dict | None = None) -> dict:
    """The agent's mesh call (h, order, refinements = list of dicts) as display fields."""
    mesh = dict(mesh or {})
    refs = mesh.get("refinements")
    if isinstance(refs, list):
        n_ref = len(refs)
    elif isinstance(refs, (int, float)):
        n_ref, refs = int(refs), None
    else:
        n_ref, refs = 0, None
    return {"h": _num(mesh.get("h")), "order": mesh.get("order", (info or {}).get("order")),
            "n_refinements": n_ref, "refinements": refs if isinstance(refs, list) else None}


def _scored_qoi(run: dict | None, meta: dict) -> str | None:
    """Name of the QoI the agent was scored on: meta.qoi if the bridge records it, else the QoI
    declared in the scored run's case, else the run's only QoI value."""
    if meta.get("qoi"):
        return str(meta["qoi"])
    case_qois = ((run or {}).get("case") or {}).get("qois") or []
    if case_qois and isinstance(case_qois[0], dict) and case_qois[0].get("name"):
        return case_qois[0]["name"]
    q = (run or {}).get("qois") or {}
    return next(iter(q)) if len(q) == 1 else None


def _median(v: list[float]):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def swarm_view(store, registry=None, loop_id: str | None = None, generation=None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    subs = [s for s in store.list_submissions() if _is_ace(s)]
    runs = [r for r in store.list_runs() if _is_ace(r)]
    runs_by_id = {r["run_id"]: r for r in runs}

    # ---- loops overview
    loops: dict[str, dict] = {}
    for rec, kind in [(s, "sub") for s in subs] + [(r, "run") for r in runs]:
        m = rec.get("meta") or {}
        lid = str(m.get("loop_id") or "default")
        lp = loops.setdefault(lid, {"loop_id": lid, "generations": set(), "n_runs": 0, "n_submissions": 0,
                                    "problems": set(), "last_update": None, "first_seen": None})
        g = _gen(m)
        if g is not None:
            lp["generations"].add(g)
        lp["n_runs" if kind == "run" else "n_submissions"] += 1
        if rec.get("problem_id"):
            lp["problems"].add(rec["problem_id"])
        t = _ts(rec.get("finished_at") or rec.get("created_at"))
        if t is not None:
            lp["last_update"] = max(lp["last_update"] or t, t)
            lp["first_seen"] = min(lp["first_seen"] or t, t)
    loop_list = []
    for lp in loops.values():
        gens = sorted(lp["generations"], key=lambda g: (isinstance(g, str), g))
        loop_list.append({**lp, "generations": gens, "problems": sorted(lp["problems"]),
                          "active": bool(lp["last_update"] and now - lp["last_update"] < ACTIVE_WINDOW_S)})
    loop_list.sort(key=lambda l: -(l["last_update"] or 0))
    out = {"loops": loop_list, "loop_id": None, "generation": None, "generations": [], "problems": [],
           "active": False, "server_time": now, "n_ace_runs": len(runs), "n_ace_submissions": len(subs)}
    if not loop_list:
        return out

    lid = loop_id if loop_id in loops else loop_list[0]["loop_id"]
    lp = next(l for l in loop_list if l["loop_id"] == lid)
    gens = lp["generations"]
    if generation is not None:
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            pass
    gen = generation if generation in gens else (gens[-1] if gens else None)
    out.update(loop_id=lid, generation=gen, active=lp["active"])

    in_loop = lambda rec: str((rec.get("meta") or {}).get("loop_id") or "default") == lid  # noqa: E731
    loop_subs = [s for s in subs if in_loop(s)]
    loop_runs = [r for r in runs if in_loop(r)]

    # ---- score distribution per generation (all problems)
    for g in gens:
        rows = []
        for s in loop_subs:
            m = s.get("meta") or {}
            if _gen(m) != g:
                continue
            rows.append({"score": _num(m.get("score")), "gated": bool(m.get("gates")), "problem_id": s.get("problem_id"),
                         "ace_run_id": _ace_id(s), "crashed": bool(m.get("error"))})
        scores = [r["score"] for r in rows if r["score"] is not None]
        out["generations"].append({
            "generation": g, "n_submissions": len(rows), "n_gated": sum(r["gated"] for r in rows),
            "n_crashed": sum(r["crashed"] for r in rows),
            "median": _median(scores), "mean": (sum(scores) / len(scores)) if scores else None,
            "best": max(scores) if scores else None, "worst": min(scores) if scores else None,
            "points": rows,
            "n_running": len({_ace_id(r) for r in loop_runs if _gen(r.get("meta") or {}) == g}
                             - {_ace_id(s) for s in loop_subs if _gen(s.get("meta") or {}) == g}),
        })

    # ---- per-problem leaderboards for the selected generation
    g_subs = [s for s in loop_subs if _gen(s.get("meta") or {}) == gen]
    g_runs = [r for r in loop_runs if _gen(r.get("meta") or {}) == gen]
    attempts: dict[tuple, list[dict]] = {}   # keyed by (loop_id, ace_run_id): ids are unique only per loop
    for r in g_runs:
        attempts.setdefault((lid, _ace_id(r) or "?"), []).append(r)
    pids = sorted({s.get("problem_id") for s in g_subs} | {r.get("problem_id") for r in g_runs} - {None})
    for pid in pids:
        title, refs = pid, {}
        if registry is not None:
            try:
                p = registry.get(pid)
                title = p.title or pid
                refs = {r.qoi: {"qoi": r.qoi, "value": r.value, "tolerance": r.tolerance, "kind": r.kind}
                        for r in p.references()}
            except Exception:  # noqa: BLE001 - unknown problem: show the raw records
                pass
        rows = []
        submitted_keys = set()
        for s in (s for s in g_subs if s.get("problem_id") == pid):
            m = s.get("meta") or {}
            key = (lid, _ace_id(s))
            submitted_keys.add(key)
            run = runs_by_id.get(m.get("submitted_run_id")) or store_get_run(store, m.get("submitted_run_id"))
            tries = sorted((r for r in attempts.get(key, []) if r.get("problem_id") == pid),
                           key=lambda r: r.get("created_at") or "")
            shown = run or (tries[-1] if tries else None)   # no successful solve: show the last attempt
            name = _scored_qoi(run, m)
            q = (run or {}).get("qois") or {}
            val = _num(q.get(name)) if name else None
            ref = refs.get(name) if name else None
            err = (val - ref["value"]) / ref["value"] if (ref and val is not None and ref["value"]) else None
            info = (shown or {}).get("info") or {}
            crashed = bool(m.get("error"))
            scored_index = ((run or {}).get("meta") or {}).get("solve_index")
            rows.append({
                "status": "crashed" if crashed else "finished", "loop_id": lid, "ace_run_id": key[1],
                "agent_id": s.get("agent_id"), "submission_id": s.get("submission_id"),
                "submitted_run_id": m.get("submitted_run_id"), "scored_solve": (scored_index + 1) if isinstance(scored_index, int) else None,
                "score": _num(m.get("score")), "gates": _names(m.get("gates")), "error": m.get("error"),
                "setup_diagnostics": _names(m.get("setup_diagnostics")), "n_cited_bullets": len(m.get("cited_bullets") or []),
                "tokens": _tokens(m.get("tokens")), "qoi": name, "value": val, "reference": ref["value"] if ref else None,
                "rel_error": err, "n_dofs": info.get("n_dofs"), "n_nodes": info.get("n_nodes"), "h_rep": info.get("h_rep"),
                "mesh": _mesh_summary(((shown or {}).get("meta") or {}).get("mesh"), info),
                "n_solves": len(s.get("run_ids") or []), "n_attempts": len(tries),
                "run_ids": s.get("run_ids") or [], "created_at": s.get("created_at"),
            })
        for key, rs in attempts.items():
            rs = sorted((r for r in rs if r.get("problem_id") == pid), key=lambda r: r.get("created_at") or "")
            if not rs or key in submitted_keys:
                continue
            last = rs[-1]
            info = last.get("info") or {}
            rows.append({
                "status": "running", "loop_id": lid, "ace_run_id": key[1], "agent_id": last.get("agent_id"),
                "submission_id": None, "submitted_run_id": None, "scored_solve": None, "score": None, "gates": [],
                "error": None, "setup_diagnostics": [], "n_cited_bullets": 0, "tokens": None, "qoi": None, "value": None,
                "reference": None, "rel_error": None, "n_dofs": info.get("n_dofs"), "n_nodes": info.get("n_nodes"),
                "h_rep": info.get("h_rep"), "mesh": _mesh_summary((last.get("meta") or {}).get("mesh"), info),
                "n_solves": sum(r.get("status") == "succeeded" for r in rs), "n_attempts": len(rs),
                "run_ids": [r["run_id"] for r in rs if r.get("status") == "succeeded"],
                "last_run_id": last["run_id"], "last_status": last.get("status"), "created_at": last.get("created_at"),
            })
        order = {"finished": 0, "crashed": 1, "running": 2}
        rows.sort(key=lambda r: (order[r["status"]], -(r["score"] if r["score"] is not None else -1e9)))
        scores = [r["score"] for r in rows if r["score"] is not None]
        names = [r["qoi"] for r in rows if r["qoi"]]
        main_qoi = max(set(names), key=names.count) if names else None
        out["problems"].append({"problem_id": pid, "title": title, "qoi": main_qoi,
                                "reference": refs.get(main_qoi) if main_qoi else None,
                                "n_agents": len(rows), "best": max(scores) if scores else None,
                                "median": _median(scores), "rows": rows})
    return out


def store_get_run(store, run_id):
    if not run_id:
        return None
    try:
        return store.get_run(run_id)
    except KeyError:
        return None
