"""HTTP API of the UI / solver service (feacore.ui) and the run store, with real solves.

Real jobs run in child processes (as in production); tests that need a registered problem or a
working fixed solver skip cleanly when those are missing.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.runstore import RunStore, check_id, jsonable  # noqa: E402
from feacore.ui.analysis import analyze_submission, interpret_entry  # noqa: E402
from feacore.ui.server import create_app  # noqa: E402
from feacore.ui.viewdata import decode_array, mesh_triangulation  # noqa: E402

PREFERRED = ("timoshenko_cantilever", "nafems_le1", "kirsch_exact_traction")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="module")
def client(runs_dir):
    app = create_app(runs_dir=runs_dir)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def problem():
    """A small registered 2D elasticity problem with default study sizes (or skip)."""
    from feacore.problems.registry import all_problems, get

    for pid in PREFERRED:
        try:
            return get(pid)
        except KeyError:
            pass
    for p in all_problems():
        if p.dim == 2 and p.physics == "elasticity":
            try:
                p.study_sizes(2)
                return p
            except Exception:  # noqa: BLE001
                continue
    pytest.skip("no 2D elasticity problem registered")


@pytest.fixture(scope="module")
def solver_ok():
    from feacore.ui.worker import check_backends

    b = check_backends()
    if not b.get("skfem", {}).get("ok"):
        pytest.skip(f"skfem backend not available: {b.get('skfem')}")
    return b


@pytest.fixture(scope="module")
def case_and_mesh(problem, tmp_path_factory):
    """Reference case + a coarse mesh generated with gmsh in this (test) process."""
    d = tmp_path_factory.mktemp("case")
    h = max(problem.study_sizes(2)) * 2
    mesh = problem.mesh(d / "part.msh", h, geom_order=2)
    case = problem.case(mesh, order=2).to_dict()
    case["mesh"] = "part.msh"
    return case, mesh


def _post_solve(client, case: dict, mesh_path, **data):
    files = {"case": ("case.json", json.dumps(case), "application/json"),
             "mesh": (mesh_path.name, mesh_path.read_bytes(), "application/octet-stream")}
    return client.post("/api/solve", data={k: str(v) for k, v in data.items()}, files=files)


def _wait(client, url, key="status", done=("succeeded", "failed", "interrupted"), timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(url).json()
        if rec.get(key) in done:
            return rec
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {url}: {rec.get(key)}")


# --------------------------------------------------------------------------- unit: store, analysis, viewdata


def test_runstore_roundtrip(tmp_path):
    store = RunStore(tmp_path / "rs")
    rec = store.create_run(backend="skfem", case={"mesh": "m.msh"}, problem_id="p", agent_id="a")
    assert (store.root / ".gitignore").read_text().strip().endswith("*")
    rid = rec["run_id"]
    store.update_run(rid, status="succeeded", qois={"q": 1.5}, info={"h_rep": 0.1, "order": 2})
    assert store.get_run(rid)["qois"] == {"q": 1.5}
    assert [r["run_id"] for r in store.list_runs(agent_id="a")] == [rid]
    assert store.list_runs(agent_id="nobody") == []
    with pytest.raises(KeyError):
        check_id("../etc")
    with pytest.raises(KeyError):
        store.get_run("run-does-not-exist")
    assert jsonable({"a": float("nan"), "b": np.float32(2.0), "c": np.arange(2)}) == {"a": None, "b": 2.0, "c": [0, 1]}


def test_submission_from_levels(tmp_path):
    store = RunStore(tmp_path / "rs")
    # synthetic second-order convergence towards 1.0 (unit test data, never shown in the UI)
    levels = [{"h": h, "qois": {"q": 1.0 + 0.3 * h**2}, "order": 2} for h in (0.4, 0.2, 0.1)]
    sub = store.create_submission("no_such_problem", "agent-x", levels=levels)
    e = sub["analysis"]["qois"][0]
    assert e["gci"]["convergence"] == "monotone"
    assert math.isclose(e["gci"]["p"], 2.0, rel_tol=1e-6)
    assert math.isclose(e["gci"]["phi_ext"], 1.0, rel_tol=1e-9)
    assert e["interpretation"]["verdict"] == "converged"
    assert store.get_submission(sub["submission_id"])["agent_id"] == "agent-x"


def test_interpretation_flags():
    # stress that grows under refinement -> divergent + singular_suspected
    e = {"name": "vm", "type": "max", "field": "von_mises", "values": [1.0, 1.4, 2.0], "p_theory": 2.0,
         "gci": {"convergence": "divergent", "flags": ["divergent"], "p": None, "phi_ext": None, "gci_fine": None}}
    it = interpret_entry(e)
    assert it["verdict"] == "diverging" and "singular_suspected" in it["flags"]
    assert all(x["text"] for x in it["explanations"])
    # the same, expected by the problem: reported as the correct answer, not an alarm
    it2 = interpret_entry({**e, "expectation": "diverge"})
    assert it2["verdict"] == "expected_divergence" and it2["status"] == "good"
    # changes below 0.1 %: converged plateau, not "divergent"
    e3 = {"name": "u", "type": "point", "field": "ux", "values": [1.00002, 1.00001, 1.000015], "p_theory": 3.0,
          "gci": {"convergence": "oscillatory_divergent", "flags": ["divergent"], "p": None, "phi_ext": None, "gci_fine": None}}
    it3 = interpret_entry(e3)
    assert it3["verdict"] == "plateau" and "divergent" not in it3["flags"]
    assert analyze_submission(None, [{"h": 0.1, "qois": {"u": 1.0}}])["qois"][0]["interpretation"]["verdict"] == "insufficient"
    # gci3's noise floor ("converged") is a good state
    e4 = {"name": "u", "type": "point", "field": "ux", "values": [1.0, 1.0 + 1e-8, 1.0], "p_theory": 3.0,
          "gci": {"convergence": "converged", "flags": ["converged_within_noise"], "p": None, "phi_ext": 1.0,
                  "gci_fine": 1.25e-8}}
    it4 = interpret_entry(e4)
    assert it4["status"] == "good" and it4["extrapolation_reliable"] is True
    # order far above theory: converged, but the extrapolation is flagged as unreliable
    e5 = {"name": "s", "type": "point", "field": "sxx", "values": [0.9, 0.99, 0.999], "p_theory": 2.0,
          "extrapolation_reliable": False,
          "gci": {"convergence": "monotone", "flags": ["p_above_theory"], "p": 3.3, "phi_ext": 1.0, "gci_fine": 1e-3}}
    it5 = interpret_entry(e5)
    assert it5["verdict"] == "converged" and "extrapolation_unreliable" in it5["flags"]
    assert "unreliable" in it5["summary"]


def test_boundary_triangulation_tet4():
    # two tets sharing a face: 6 boundary faces, closed and outward oriented
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    tri = mesh_triangulation(pts, cells, "tetra")
    assert tri["triangles"].shape == (6, 3)
    P = pts[tri["node_ids"]]
    a, b, c = (P[tri["triangles"][:, k]] for k in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert vol > 0  # outward orientation gives the positive enclosed volume
    assert math.isclose(vol, 1 / 6 + abs(np.linalg.det(pts[[2, 3, 4]] - pts[1])) / 6)


# --------------------------------------------------------------------------- HTTP: static, health, problems


def test_frontend_served(client, tmp_path, monkeypatch):
    r = client.get("/")
    assert r.status_code == 200 and "__IMPORTMAP__" not in r.text
    assert '"three": "/vendor/three.module.min.js"' in r.text  # same-origin import map
    for path in ("/static/js/app.js", "/static/js/plot.js", "/static/js/mesh3d.js", "/static/css/app.css"):
        assert client.get(path).status_code == 200, path
    vend = tmp_path / "vendor"  # a vendored copy is served as is (no network in tests)
    vend.mkdir()
    (vend / "three.module.min.js").write_text("export const REVISION = '170';")
    monkeypatch.setenv("FEACORE_UI_VENDOR_DIR", str(vend))
    r = client.get("/vendor/three.module.min.js")
    assert r.status_code == 200 and "REVISION" in r.text
    assert client.get("/vendor/not-a-file.js").status_code == 404


def test_health(client):
    r = client.get("/api/health", params={"wait": True})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and "skfem" in d["backends"]
    assert d["cpu"]["count"] >= 1 and "gpus" in d["gpu"]
    assert set(d["jobs"]["workers"]) == {"quick", "study"}


def test_problems(client, problem):
    d = client.get("/api/problems").json()
    ids = [p["id"] for p in d["problems"]]
    assert problem.id in ids and d["total"] >= len(ids)
    tag = problem.tags[0]
    filtered = client.get("/api/problems", params={"tag": tag}).json()["problems"]
    assert filtered and all(tag in p["tags"] for p in filtered)
    det = client.get(f"/api/problems/{problem.id}").json()
    assert det["statement"]["id"] == problem.id and det["statement"]["description"]
    assert isinstance(det["evaluator"]["references"], list)
    assert client.get("/api/problems/definitely_not_a_problem").status_code == 404


def test_geometry_preview(client, problem):
    t0 = time.time()
    r = client.get(f"/api/problems/{problem.id}/geometry")
    assert r.status_code == 200, r.text
    g = r.json()
    tris = decode_array(g["triangles"])
    pts = decode_array(g["points"])
    assert tris.shape[1] == 3 and tris.max() < len(pts) and g["n_triangles"] == len(tris) > 0
    assert any(gr["dim"] < problem.dim for gr in g["groups"])
    first = time.time() - t0
    t1 = time.time()
    assert client.get(f"/api/problems/{problem.id}/geometry").status_code == 200
    assert time.time() - t1 < max(1.0, first)  # cached


# --------------------------------------------------------------------------- HTTP: solve service


def test_solve_validation_errors(client, case_and_mesh):
    case, mesh = case_and_mesh
    bad = {**case, "analysis": {**case["analysis"], "physics": "acoustics"}}
    r = _post_solve(client, bad, mesh)
    assert r.status_code == 422 and r.json()["errors"]
    r = client.post("/api/solve", data={"case": json.dumps(case)})
    assert r.status_code == 422 and "mesh" in r.json()["detail"]
    r = _post_solve(client, case, mesh, backend="nope")
    assert r.status_code == 422


def test_solve_real(client, case_and_mesh, problem, solver_ok):
    case, mesh = case_and_mesh
    r = _post_solve(client, case, mesh, agent_id="test-agent", problem_id=problem.id)
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["status"] == "succeeded" and rec["agent_id"] == "test-agent"
    assert set(rec["qois"]) == {q.name for q in problem.qois()}
    assert all(v is not None and math.isfinite(v) for v in rec["qois"].values())
    assert rec["info"]["backend"] == "skfem" and rec["info"]["n_dofs"] > 0
    rid = rec["run_id"]
    assert client.get(f"/api/runs/{rid}").json()["qois"] == rec["qois"]
    assert rid in [x["run_id"] for x in client.get("/api/runs", params={"agent_id": "test-agent"}).json()["runs"]]
    # viewer data
    avail = rec["fields"]["available"]
    assert "von_mises" in avail and "ux" in avail
    f = client.get(f"/api/runs/{rid}/field", params={"name": "von_mises"}).json()
    pts, tri, vals = decode_array(f["points"]), decode_array(f["triangles"]), decode_array(f["values"])
    assert pts.shape == (f["n_points"], 3) and tri.shape == (f["n_triangles"], 3) and vals.shape == (f["n_points"],)
    assert tri.max() < len(pts) and np.isfinite(vals).all()
    assert decode_array(f["displacement"]).shape == pts.shape
    assert f["range"][0] <= f["range"][1]
    f2 = client.get(f"/api/runs/{rid}/field", params={"name": "ux", "geometry": False, "encoding": "list"}).json()
    assert "points" not in f2 and len(f2["values"]["data"]) == f["n_points"]
    assert client.get(f"/api/runs/{rid}/field", params={"name": "nope"}).status_code == 404
    assert client.get(f"/api/runs/{rid}/mesh").status_code == 200
    assert client.get("/api/runs/run-missing").status_code == 404


def test_solve_reports_solver_errors(client, case_and_mesh, solver_ok):
    case, mesh = case_and_mesh
    bad = json.loads(json.dumps(case))
    bad["bcs"][0]["group"] = "no_such_group"
    r = _post_solve(client, bad, mesh, agent_id="test-agent")
    assert r.status_code == 422
    rec = r.json()
    assert rec["status"] == "failed" and rec["error"]


def test_solve_nowait(client, case_and_mesh, solver_ok):
    case, mesh = case_and_mesh
    r = _post_solve(client, case, mesh, wait="false", fields="false")
    assert r.status_code == 202
    rec = _wait(client, f"/api/runs/{r.json()['run_id']}")
    assert rec["status"] == "succeeded" and rec["fields"] is None


def test_import_run(client):
    r = client.post("/api/runs", json={"problem_id": "x", "agent_id": "ext", "backend": "skfem",
                                       "qois": {"q": 2.5}, "info": {"h_rep": 0.1, "order": 2}})
    assert r.status_code == 201
    assert r.json()["status"] == "succeeded" and r.json()["source"] == "import"
    assert client.post("/api/runs", json={"agent_id": "ext"}).status_code == 422


# --------------------------------------------------------------------------- HTTP: studies, submissions, consensus


@pytest.fixture(scope="module")
def finished_study(client, problem, solver_ok):
    h0 = max(problem.study_sizes(2))
    r = client.post("/api/studies", json={"problem_id": problem.id, "backend": "skfem", "order": 2,
                                          "sizes": [2 * h0, h0, h0 / 2]})
    assert r.status_code == 202, r.text
    sid = r.json()["study_id"]
    st = _wait(client, f"/api/studies/{sid}")
    assert st["status"] == "succeeded", st.get("error")
    return st


def test_study(client, finished_study, problem):
    st = finished_study
    assert len(st["run_ids"]) == 3 and len(st["runs"]) == 3
    assert st["progress"]["done"] == 3
    res = st["result"]
    assert {q["name"] for q in res["qois"]} == {q.name for q in problem.qois()}
    for q in res["qois"]:
        assert len(q["values"]) == 3 and q["gci"] is not None
        it = st["interpretation"][q["name"]]
        assert it["verdict"] and it["label"] and it["summary"]
    assert all(r["fields"] for r in st["runs"])
    listed = client.get("/api/studies", params={"problem_id": problem.id}).json()["studies"]
    assert st["study_id"] in [s["study_id"] for s in listed]
    assert client.get("/api/studies/study-missing").status_code == 404
    assert client.post("/api/studies", json={"problem_id": "nope"}).status_code == 404
    assert client.post("/api/studies", json={"problem_id": problem.id, "sizes": [0.1, -1]}).status_code == 422


def test_submissions_and_consensus(client, finished_study, problem):
    st = finished_study
    r = client.post("/api/submissions", json={"problem_id": problem.id, "agent_id": "agent-a", "run_ids": st["run_ids"]})
    assert r.status_code == 201, r.text
    sub = r.json()
    assert len(sub["levels"]) == 3 and sub["analysis"]["qois"]
    # a second "agent" posts levels solved elsewhere (re-using the real values of the study)
    levels = [{"h": run["info"]["h_rep"], "qois": run["qois"], "order": 2} for run in st["runs"]]
    r2 = client.post("/api/submissions", json={"problem_id": problem.id, "agent_id": "agent-b", "levels": levels})
    assert r2.status_code == 201
    assert client.post("/api/submissions", json={"problem_id": problem.id, "agent_id": "c"}).status_code == 422
    assert client.post("/api/submissions", json={"problem_id": problem.id, "agent_id": "c",
                                                 "run_ids": ["run-missing"]}).status_code == 404
    subs = client.get("/api/submissions", params={"problem_id": problem.id}).json()["submissions"]
    assert {s["agent_id"] for s in subs} >= {"agent-a", "agent-b"}
    assert client.get(f"/api/submissions/{sub['submission_id']}").json()["agent_id"] == "agent-a"
    c = client.get(f"/api/consensus/{problem.id}").json()
    assert c["n_submissions"] >= 2 and set(c["agents"]) >= {"agent-a", "agent-b"}
    q0 = c["qois"][0]
    assert len(q0["submissions"]) >= 2 and q0["consensus"]["median"] is not None
    assert q0["reference_studies"]  # the study above is the fixed-solver reference row
    assert client.get("/api/consensus/definitely_not_a_problem").status_code == 404
