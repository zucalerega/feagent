"""Disk caches of the diagnosis layers (run errors, oracles), the probe-point file naming, and the
counterfactual "before" read from the run record: each must give exactly what recomputing gives."""

import json
import shutil

import pytest

from ace import checkpoint as ckpt
from ace import config, fea, field_error, oracles, stages
from ace import counterfactual as cf
from ace.checks import check_run, reference_setup, tolerance

PID = "cooks_membrane"


@pytest.fixture(scope="module", autouse=True)
def private_caches(tmp_path_factory):
    """Point the error and oracle caches at a temporary directory, so these tests never write into
    the shared ace/field_refs caches that real runs read."""
    d = tmp_path_factory.mktemp("caches")
    old = field_error.ERRORS_CACHE, stages.ORACLE_CACHE
    field_error.ERRORS_CACHE, stages.ORACLE_CACHE = d / "errors", d / "oracles"
    yield d
    field_error.ERRORS_CACHE, stages.ORACLE_CACHE = old


@pytest.fixture(scope="module")
def run(tmp_path_factory, private_caches):
    """A small Cook's membrane run with the fields the loop adds (checks, oracles, stages)."""
    p = fea.get_problem(PID)
    req = {"h": p.params["h0"], "order": 2, "refinements": []}
    mesh = fea.make_mesh(p, tmp_path_factory.mktemp("cook") / "mesh_0.msh", h=req["h"], order=2)
    setup = reference_setup(p)
    res = fea.run_solve(p, mesh, setup, config.TASKS[PID]["qoi"])
    r = {"problem_id": PID, "run_id": "cache-test", "meshes": [dict(mesh, request=req)], "submitted": 0,
         "solves": [{"mesh": mesh, "mesh_index": 0, "setup": setup, "result": res}]}
    tol = tolerance(PID)
    r["checks"] = check_run(PID, r)
    r["oracles"] = stages.run_oracles(r, tol)
    r["stages"] = stages.stage_report(r, r["oracles"], tol)
    return r


def _boom(*a, **k):
    raise AssertionError("a cache hit must not recompute")


def test_error_cache_hit_equals_fresh_solve(run, monkeypatch):
    s = run["solves"][0]
    monkeypatch.setenv("ACE_NO_ERROR_CACHE", "1")
    fresh = field_error.errors_of_setup(PID, s["mesh"], s["setup"])
    monkeypatch.delenv("ACE_NO_ERROR_CACHE")
    f = field_error._errors_cache_file(PID, s["mesh"], s["setup"])
    f.unlink(missing_ok=True)
    cold = field_error.errors_of_setup(PID, s["mesh"], s["setup"])
    assert f.exists() and cold == fresh
    monkeypatch.setattr(field_error, "_solve", _boom)
    assert field_error.errors_of_setup(PID, s["mesh"], s["setup"]) == fresh
    f.unlink()


def test_error_cache_key_changes_with_setup_and_mesh(run):
    s = run["solves"][0]
    k = field_error._errors_cache_file(PID, s["mesh"], s["setup"])
    other = dict(s["setup"], material=dict(s["setup"]["material"], E=2 * s["setup"]["material"]["E"]))
    assert field_error._errors_cache_file(PID, s["mesh"], other) != k
    assert field_error._errors_cache_file(PID, dict(s["mesh"], sha256="0" * 64), s["setup"]) != k


def test_oracle_cache_round_trips(run, tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "ORACLE_CACHE", tmp_path)
    tol = tolerance(PID)
    cold = stages.run_oracles(run, tol)
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(oracles, "evaluate", _boom)
    warm = stages.run_oracles(run, tol)
    assert warm == json.loads(json.dumps(cold))                 # numpy floats come back as equal floats
    assert stages.stage_report(run, warm, tol) == stages.stage_report(run, cold, tol)
    assert json.dumps(warm, default=str) == json.dumps(cold, default=str)   # the run record on disk


def test_oracle_cache_opt_out(run, tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "ORACLE_CACHE", tmp_path)
    monkeypatch.setenv("ACE_NO_ORACLE_CACHE", "1")
    stages.run_oracles(run, tolerance(PID))
    assert not list(tmp_path.iterdir())


def test_probe_point_files_by_count(tmp_path, monkeypatch):
    for name in (f"{PID}.npz", f"{PID}_coarse.npz", f"{PID}_points.json"):
        shutil.copy(field_error.CACHE / name, tmp_path / name)
    monkeypatch.setattr(field_error, "CACHE", tmp_path)
    monkeypatch.setattr(config, "PROBE_POINTS", 12)
    assert field_error.probe_points(PID) == json.loads((tmp_path / f"{PID}_points.json").read_text())
    monkeypatch.setattr(config, "PROBE_POINTS", 5)
    assert len(field_error.probe_points(PID)["points"]) == 5
    assert (tmp_path / f"{PID}_points5.json").exists()


def test_recorded_before_equals_recomputed(run, tmp_path):
    cp = ckpt.checkpoint(run, allow_failed=True)
    tol = tolerance(PID)
    assert cf.recorded_before(run, cp, tol) == cf.evaluate_variant(PID, cp["mesh"], cp["request"], cp["setup"], tol)
    assert cf.recorded_before(run, cp, tol * 2) is None          # another accuracy target: recompute
    assert cf.recorded_before({k: v for k, v in run.items() if k != "oracles"}, cp, tol) is None
