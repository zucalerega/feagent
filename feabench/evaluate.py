"""Evaluate one submission: run the fixed solver on the agent's deck, extract QoIs, score.

Submission contract (until the agent MCP is defined): a directory containing ``model.inp``
(a CalculiX deck; may ``*INCLUDE`` other files in the same directory) written in the unit
system stated in the task.  Optionally ``submission.json`` with self-reported values, e.g.
``{"qois": {"<id>": value}, "notes": "..."}``, which is recorded but never used for scoring.

The harness appends output requests (U, RF, S, E) before each step's ``*END STEP``; ccx
accumulates output cards within a step, so the agent's own requests are unaffected.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .cases import case_dir, load_reference, load_task
from .ccx.frd import read_frd
from .ccx.inp import ensure_output_requests, read_deck_materials, read_deck_nodes, scan_deck
from .ccx.runner import run_ccx
from .qoi import QoIError, ResultView, evaluate_qoi
from .scoring import ScoreConfig, case_score, cost_usd, score_qoi

EXPECTED_PROCEDURE = {"static": "static", "frequency": "frequency", "buckle": "buckle"}


def _find_deck(sub: Path) -> Path:
    if (sub / "model.inp").exists():
        return sub / "model.inp"
    decks = sorted(sub.glob("*.inp"))
    if len(decks) == 1:
        return decks[0]
    raise FileNotFoundError(f"{sub}: expected model.inp (found {[d.name for d in decks]})")


def evaluate_submission(case: str | Path, submission: str | Path, out: str | Path | None = None,
                        timeout: float = 900.0, threads: int = 1, agent_cost: dict | None = None,
                        cfg: ScoreConfig | None = None, keep_workdir: bool = False) -> dict:
    cfg = cfg or ScoreConfig()
    cdir = case_dir(case)
    task, ref = load_task(case), load_reference(case)
    sub = Path(submission)
    deck = _find_deck(sub)
    work = Path(tempfile.mkdtemp(prefix="feabench-eval-"))
    try:
        shutil.copytree(sub, work, dirs_exist_ok=True)
        model = work / "model.inp"
        model.write_text(ensure_output_requests(deck.read_text(errors="replace")))
        if deck.name != "model.inp" and (work / deck.name).exists():
            (work / deck.name).unlink()
        info = scan_deck(model)
        gates: dict[str, bool] = {}
        expected = EXPECTED_PROCEDURE.get(task.get("analysis", {}).get("type", "static"))
        gates["analysis_type"] = expected in info.procedures
        material_check = check_materials(task, read_deck_materials(model))
        if material_check is not None:
            gates["material_properties"] = material_check["ok"]
        rr = run_ccx(model, timeout=timeout, threads=threads)
        gates["solver_ok"] = rr.ok
        qoi_measured: dict[str, dict] = {}
        mesh_stats: dict = {}
        if rr.ok:
            frd = read_frd(work / "model.frd")
            rv = ResultView(frd, work / "model.dat", deck_nodes=read_deck_nodes(model))
            mesh_stats = {"n_nodes": int(len(frd.node_ids)),
                          "elements": {k: int(len(v)) for k, v in frd.elements.items()},
                          "n_equations": rr.n_equations}
            if "volume" in ref:
                try:
                    vol = rv.volume()
                    mesh_stats["volume"] = vol
                    tol = ref.get("volume_tolerance", 0.02)
                    gates["geometry_volume"] = abs(vol - ref["volume"]) <= tol * abs(ref["volume"])
                except Exception as exc:  # pragma: no cover - degenerate meshes
                    mesh_stats["volume_error"] = str(exc)
                    gates["geometry_volume"] = False
            for spec in ref["qois"]:
                try:
                    qoi_measured[spec["id"]] = evaluate_qoi(spec, rv, cdir / "reference")
                except (QoIError, KeyError, ValueError, IndexError) as exc:
                    qoi_measured[spec["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
        scores = [score_qoi(spec, qoi_measured.get(spec["id"]), cfg) for spec in ref["qois"]]
        agent_wall = (agent_cost or {}).get("wall_s")
        wall = (agent_wall or 0.0) + rr.wall_s if agent_cost else None
        cost = cost_usd(agent_cost, rr.cpu_s, cfg) if agent_cost else None
        summary = case_score(scores, gates, cost, wall, cfg, ref.get("budget"))
        summary.update(case=str(case), solver_cpu_s=rr.cpu_s, solver_wall_s=rr.wall_s,
                       n_equations=rr.n_equations)
        self_reported = None
        if (sub / "submission.json").exists():
            try:
                self_reported = json.loads((sub / "submission.json").read_text())
            except json.JSONDecodeError:
                self_reported = {"error": "unparseable submission.json"}
        report = {
            "case": str(case),
            "submission": str(sub),
            "summary": summary,
            "gates": gates,
            "qois": scores,
            "solver": rr.to_dict() | {"workdir": str(work) if keep_workdir else None},
            "deck": info.to_dict() | {"keywords": None},
            "mesh": mesh_stats,
            "materials": material_check,
            "agent_cost": agent_cost,
            "self_reported": self_reported,
            "score_config": cfg.to_dict(),
        }
        if out:
            Path(out).write_text(json.dumps(report, indent=1, default=float))
        return report
    finally:
        if not keep_workdir:
            shutil.rmtree(work, ignore_errors=True)


def check_materials(task: dict, deck_mats: dict[str, dict], rtol: float = 1e-3) -> dict | None:
    """Single-material tasks: every *ELASTIC must equal the specified E, nu (and *DENSITY when the
    analysis needs mass).  Catches unit slips (Pa vs MPa, kg vs tonne) with a precise message."""
    spec = task.get("materials") or []
    if len(spec) != 1:
        return None
    m = spec[0]
    needs_density = task.get("analysis", {}).get("type") == "frequency" or any(
        ld.get("type") == "gravity" for ld in task.get("loads", []))
    problems = []
    if not deck_mats:
        problems.append("no *MATERIAL in deck")
    for name, d in deck_mats.items():
        for key in ("E", "nu"):
            if key not in d:
                problems.append(f"{name}: missing {key}")
            elif abs(d[key] - m[key]) > rtol * abs(m[key]):
                problems.append(f"{name}: {key} = {d[key]:g}, task specifies {m[key]:g}")
        if needs_density and "density" in m:
            if "density" not in d:
                problems.append(f"{name}: missing *DENSITY (needed for this analysis)")
            elif abs(d["density"] - m["density"]) > rtol * abs(m["density"]):
                ratio = d["density"] / m["density"] if m["density"] else float("nan")
                problems.append(f"{name}: density = {d['density']:g}, task specifies {m['density']:g} "
                                f"(ratio {ratio:.3g}; units are tonne/mm^3 for mm-N-s)")
    return {"ok": not problems, "problems": problems, "deck": deck_mats}
