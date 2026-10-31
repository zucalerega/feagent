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
    app = create_app(runs_dir=runs_dir, token="")  # no auth, whatever FEACORE_UI_TOKEN says
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
    # point-stress scatter (< 2 %, no trend) on unstructured meshes is not a singularity
    e6 = {"name": "s", "type": "point", "field": "syy", "values": [-5.380, -5.430, -5.395], "p_theory": 2.0,
          "gci": {"convergence": "oscillatory_divergent", "flags": ["divergent"], "p": None, "phi_ext": None, "gci_fine": None}}
    it6 = interpret_entry(e6)
    assert it6["verdict"] == "scatter" and "singular_suspected" not in it6["flags"] and it6["status"] == "warning"
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
    assert isinstance(d["active_jobs"], int) and d["active_jobs"] >= 0 and d["auth_required"] is False
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
    assert client.get("/api/health").json()["active_jobs"] >= 1  # queued or running right now
    rec = _wait(client, f"/api/runs/{r.json()['run_id']}")
    assert rec["status"] == "succeeded" and rec["fields"] is None
    t0 = time.time()
    while client.get("/api/health").json()["active_jobs"] and time.time() - t0 < 30:
        time.sleep(0.2)  # the record is written just before the worker process exits
    assert client.get("/api/health").json()["active_jobs"] == 0


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


# --------------------------------------------------------------------------- HTTP: benchmark batches


def _wait_batch(client, bid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = client.get(f"/api/batches/{bid}").json()
        if b["status"] not in ("queued", "running"):
            return b
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for batch {bid}")


def test_batch_run_and_downloads(client, problem, solver_ok):
    import csv
    import io

    from feacore.ui.analysis import CSV_COLUMNS, OUTCOMES

    r = client.post("/api/batches", json={"problem_ids": [problem.id], "backend": "skfem", "order": 2,
                                          "levels": 3, "ratio": 2.0, "name": "test batch"})
    assert r.status_code == 202, r.text
    b = r.json()
    assert b["n_problems"] == 1 and b["items"][0]["problem_id"] == problem.id
    b = _wait_batch(client, b["batch_id"])
    assert b["status"] == "succeeded" and b["n_done"] == 1
    item = b["items"][0]
    assert item["finest_run_id"] and item["elapsed"] > 0 and item["progress"]["done"] == 3
    rows = b["rows"]
    assert {r["qoi"] for r in rows} == {q.name for q in problem.qois()}
    for row in rows:
        assert row["outcome"] in OUTCOMES and row["study_id"] == item["study_id"]
        assert row["n_meshes"] == 3 and row["finest"] is not None
    assert sum(b["outcomes"].values()) == len(rows)
    # downloads
    c = client.get(f"/api/batches/{b['batch_id']}/results.csv")
    assert c.status_code == 200 and "attachment" in c.headers["content-disposition"]
    parsed = list(csv.DictReader(io.StringIO(c.text)))
    assert list(parsed[0].keys()) == CSV_COLUMNS and len(parsed) == len(rows)
    j = client.get(f"/api/batches/{b['batch_id']}/results.json").json()
    assert len(j["rows"]) == len(rows) and j["settings"]["levels"] == 3
    assert client.get(f"/api/batches/{b['batch_id']}/results.xml").status_code == 404
    listed = client.get("/api/batches").json()
    assert b["batch_id"] in [x["batch_id"] for x in listed["batches"]] and listed["study_workers"] >= 1
    # the study belongs to the batch
    st = client.get(f"/api/studies/{item['study_id']}").json()
    assert st["batch_id"] == b["batch_id"] and st["elapsed"] > 0


def test_batch_validation_and_cancel(client, problem, solver_ok):
    assert client.post("/api/batches", json={"problem_ids": []}).status_code == 422
    assert client.post("/api/batches", json={"problem_ids": ["no_such_problem"]}).status_code == 404
    assert client.post("/api/batches", json={"problem_ids": [problem.id], "backend": "nope"}).status_code == 422
    assert client.get("/api/batches/batch-missing").status_code == 404
    r = client.post("/api/batches", json={"problem_ids": [problem.id, problem.id], "levels": 3})
    assert r.status_code == 202 and r.json()["n_problems"] == 1  # duplicates collapse
    r = client.post("/api/batches", json={"problem_ids": [problem.id], "levels": 3})
    r2 = client.post("/api/batches", json={"problem_ids": [problem.id], "levels": 4})
    bid = r2.json()["batch_id"]
    c = client.post(f"/api/batches/{bid}/cancel").json()
    assert c["cancelled"] == 1
    b = _wait_batch(client, bid)
    assert b["counts"]["cancelled"] == 1 and b["items"][0]["status"] == "cancelled"
    _wait_batch(client, r.json()["batch_id"])


# --------------------------------------------------------------------------- shared-token auth


def test_token_auth(tmp_path):
    app = create_app(runs_dir=tmp_path / "runs", token="s3cret-token")
    with TestClient(app) as c:
        assert c.get("/").status_code == 200  # the page shell stays open (it accepts ?token=)
        assert c.get("/static/js/app.js").status_code == 200
        r = c.get("/api/problems")
        assert r.status_code == 401 and "token" in r.json()["detail"]
        assert r.headers["www-authenticate"] == "Bearer"
        assert c.get("/api/problems", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
        assert c.get("/api/problems", headers={"X-Feacore-Token": "s3cret-token"}).status_code == 200
        assert c.get("/api/problems", params={"token": "s3cret-token"}).status_code == 200
        assert c.get("/api/problems", headers={"Authorization": "Bearer s3cret-tokeN"}).status_code == 401
        assert c.get("/api/problems", params={"token": ""}).status_code == 401
        assert c.post("/api/runs", json={"qois": {"q": 1.0}}).status_code == 401
        assert c.post("/api/runs", json={"qois": {"q": 1.0}},
                      headers={"Authorization": "Bearer s3cret-token"}).status_code == 201
        a = c.get("/api/auth").json()
        assert a["required"] is True and a["valid"] is False
        # health stays reachable for deploy scripts, but reveals only the job count without a token
        h = c.get("/api/health")
        assert h.status_code == 200 and h.json() == {"ok": True, "active_jobs": 0, "auth_required": True}
        full = c.get("/api/health", headers={"Authorization": "Bearer s3cret-token"}).json()
        assert full["active_jobs"] == 0 and "hostname" in full and "store" in full
        assert c.get("/api/auth", params={"token": "s3cret-token"}).json()["valid"] is True


def test_no_token_configured(client):
    a = client.get("/api/auth").json()
    assert a["required"] is False and a["valid"] is True


# --------------------------------------------------------------------------- ACE swarm (synthetic records per the bridge contract)


def _ace_solve(store, loop_id, gen, ace_id, pid, idx, qoi_value, n_dofs, status="succeeded", h=0.1):
    meta = {"source": "ace", "loop_id": loop_id, "generation": gen, "ace_run_id": ace_id, "solve_index": idx,
            "mesh": {"h": h, "order": 2, "refinements": idx}}
    rec = store.create_run(backend="skfem", problem_id=pid, agent_id=f"ace:{ace_id}", source="ace",
                           status="running", meta=meta)
    if status == "succeeded":
        store.update_run(rec["run_id"], status="succeeded", qois={"syy_D": qoi_value},
                         info={"n_dofs": n_dofs, "n_nodes": n_dofs // 2, "h_rep": h, "order": 2})
    else:
        store.update_run(rec["run_id"], status="failed", error="synthetic failure")
    return rec["run_id"]


def _ace_submit(store, loop_id, gen, ace_id, pid, run_ids, score, gates=()):
    return store.create_submission(pid, f"ace:{ace_id}", run_ids=run_ids, meta={
        "source": "ace", "loop_id": loop_id, "generation": gen, "ace_run_id": ace_id,
        "submitted_run_id": run_ids[-1], "score": score, "gates": list(gates),
        "setup_diagnostics": [{"name": "coarse_mesh"}], "cited_bullets": ["b1", "b2"], "tokens": {"input": 900, "output": 100}})


def test_swarm_empty(tmp_path):
    app = create_app(runs_dir=tmp_path / "runs", token="")
    with TestClient(app) as c:
        d = c.get("/api/swarm").json()
        assert d["loops"] == [] and d["problems"] == [] and d["loop_id"] is None and d["active"] is False
        assert d["store_root"]


def test_swarm_leaderboards(tmp_path):
    store = RunStore(tmp_path / "runs")
    pid = "nafems_le1"  # registered: the leaderboard shows its reference; unknown ids still work
    # loop A, generation 0: two finished agents (one gated) + one still running
    r1 = [_ace_solve(store, "loopA", 0, "a1", pid, i, 9.20e7 + i * 2e5, 5000 * (i + 1)) for i in range(3)]
    _ace_submit(store, "loopA", 0, "a1", pid, r1, score=0.4)
    r2 = [_ace_solve(store, "loopA", 0, "a2", pid, 0, 8.0e7, 4000)]
    _ace_solve(store, "loopA", 0, "a2", pid, 1, 0, 0, status="failed")
    _ace_submit(store, "loopA", 0, "a2", pid, r2, score=-2.0, gates=["dof_budget"])
    _ace_solve(store, "loopA", 0, "a3", pid, 0, 9.1e7, 3000)
    # generation 1: one better agent; a second loop with an unknown problem
    r4 = [_ace_solve(store, "loopA", 1, "a4", pid, i, 9.26e7, 8000) for i in range(3)]
    _ace_submit(store, "loopA", 1, "a4", pid, r4, score=1.1)
    rb = [_ace_solve(store, "loopB", 0, "b1", "some_unknown_problem", 0, 1.0, 100)]
    _ace_submit(store, "loopB", 0, "b1", "some_unknown_problem", rb, score=0.0)
    store.create_run(backend="skfem", problem_id=pid, agent_id="not-ace", source="api", status="succeeded")

    app = create_app(runs_dir=tmp_path / "runs", token="")
    with TestClient(app) as c:
        d = c.get("/api/swarm", params={"loop_id": "loopA", "generation": 0}).json()
        assert {l["loop_id"] for l in d["loops"]} == {"loopA", "loopB"}
        assert d["score_scheme"] == "log"            # older log-score records still render
        la = next(l for l in d["loops"] if l["loop_id"] == "loopA")
        assert la["generations"] == [0, 1] and la["n_submissions"] == 3 and la["active"] is True
        assert d["loop_id"] == "loopA" and d["generation"] == 0
        gens = {g["generation"]: g for g in d["generations"]}
        assert gens[0]["n_submissions"] == 2 and gens[0]["n_gated"] == 1 and gens[0]["n_running"] == 1
        assert gens[0]["best"] == 0.4 and gens[1]["median"] == 1.1
        [prob] = d["problems"]
        assert prob["problem_id"] == pid and prob["reference"]["qoi"] == "syy_D"
        rows = prob["rows"]
        assert [r["ace_run_id"] for r in rows] == ["a1", "a2", "a3"]  # by score, running agents last
        top = rows[0]
        assert top["status"] == "finished" and top["score"] == 0.4 and top["n_solves"] == 3 and top["n_attempts"] == 3
        assert top["submitted_run_id"] == r1[-1] and top["value"] == pytest.approx(9.24e7)
        assert top["rel_error"] == pytest.approx((9.24e7 - 9.27e7) / 9.27e7) and top["n_dofs"] == 15000
        assert top["scored_solve"] == 3 and top["mesh"]["n_refinements"] == 2
        assert top["tokens"] == 1000 and top["setup_diagnostics"] == ["coarse_mesh"] and top["n_cited_bullets"] == 2
        assert rows[1]["gates"] == ["dof_budget"] and rows[1]["n_attempts"] == 2 and rows[1]["n_solves"] == 1
        assert rows[2]["status"] == "running" and rows[2]["score"] is None
        # default selection: latest generation of the most recently active loop
        d1 = c.get("/api/swarm", params={"loop_id": "loopA"}).json()
        assert d1["generation"] == 1 and d1["problems"][0]["rows"][0]["score"] == 1.1
        db = c.get("/api/swarm", params={"loop_id": "loopB"}).json()
        assert db["problems"][0]["title"] == "some_unknown_problem" and db["problems"][0]["reference"] is None
        # ACE submissions also feed the regular consensus view
        cons = c.get(f"/api/consensus/{pid}").json()
        assert {"ace:a1", "ace:a2", "ace:a4"} <= set(cons["agents"])


def test_swarm_bridge_record_shapes(tmp_path):
    """Shapes the real ACE bridge writes: refinements as a list of dicts, the scored QoI named in the
    case (not references()[0]), runs with no successful solve, crashed runs, ids reused across loops."""
    from feacore.runstore import write_json, new_id, now_iso, SUBMISSION_SCHEMA

    store = RunStore(tmp_path / "runs")
    pid = "lame_cylinder_plane_strain"            # references: ur_bore first, hoop_bore second
    refine = [{"point": [0.0, 0.1], "size": 0.002, "radius": 0.01}, {"point": [0.1, 0.0], "size": 0.002, "radius": 0.01}]

    def solve(loop, aid, idx, value, ok=True):
        meta = {"source": "ace", "loop_id": loop, "generation": 0, "ace_run_id": aid, "solve_index": idx,
                "mesh": {"h": 0.01, "order": 2, "refinements": refine}}
        rec = store.create_run(backend="skfem", problem_id=pid, agent_id=f"ace:{aid}", source="ace", status="running",
                               meta=meta, case={"mesh": "m.msh", "qois": [{"name": "hoop_bore", "type": "point", "field": "stt"}]})
        if ok:
            store.update_run(rec["run_id"], status="succeeded", qois={"hoop_bore": value}, info={"n_dofs": 900, "order": 2, "h_rep": 0.01 / (idx + 1)})
        else:
            store.update_run(rec["run_id"], status="failed", error="no_such_group")
        return rec["run_id"]

    def submit(loop, aid, run_ids, scored, score, gates=(), error=None):
        meta = {"source": "ace", "loop_id": loop, "generation": 0, "ace_run_id": aid, "submitted_run_id": scored,
                "score": score, "gates": list(gates), "setup_diagnostics": [], "cited_bullets": [],
                "tokens": {"input": 10, "output": 5, "cache_write": 0, "cache_read": 0, "total": 15}}
        if error:
            meta["error"] = error
        if run_ids:
            return store.create_submission(pid, f"ace:{aid}", run_ids=run_ids, meta=meta)
        sid = new_id("sub")                        # what the bridge writes when there is no successful solve
        write_json(store.submissions_dir / f"{sid}.json", {
            "schema": SUBMISSION_SCHEMA, "submission_id": sid, "created_at": now_iso(), "problem_id": pid,
            "agent_id": f"ace:{aid}", "run_ids": [], "levels": [], "order": None, "notes": "", "meta": meta,
            "analysis": {"order": None, "n_levels": 0, "qois": []}})

    good = [solve("L1", "g0-lame-0", i, 1.66e8 + i * 3e5) for i in range(2)]
    submit("L1", "g0-lame-0", good, good[0], 0.7)          # ACE scored the first solve, not the last
    failed = solve("L1", "g0-lame-1", 0, None, ok=False)
    submit("L1", "g0-lame-1", [], None, -2.0, gates=["no successful solve"])
    submit("L1", "g0-lame-2", [], None, None, error="APIError: overloaded")
    other = [solve("L2", "g0-lame-0", 0, 1.0e8)]           # same ace id, another loop
    submit("L2", "g0-lame-0", other, other[0], -0.5)

    app = create_app(runs_dir=tmp_path / "runs", token="")
    with TestClient(app) as c:
        d = c.get("/api/swarm", params={"loop_id": "L1"}).json()
        [prob] = d["problems"]
        assert prob["qoi"] == "hoop_bore" and prob["reference"]["qoi"] == "hoop_bore"
        rows = {r["ace_run_id"]: r for r in prob["rows"]}
        assert set(rows) == {"g0-lame-0", "g0-lame-1", "g0-lame-2"}          # nothing from loop L2
        a = rows["g0-lame-0"]
        assert a["status"] == "finished" and a["qoi"] == "hoop_bore" and a["value"] == pytest.approx(1.66e8)
        assert a["reference"] == pytest.approx(prob["reference"]["value"]) and a["rel_error"] is not None
        assert a["scored_solve"] == 1 and a["n_solves"] == 2 and a["tokens"] == 15
        assert a["mesh"]["n_refinements"] == 2 and a["mesh"]["refinements"] == refine
        b = rows["g0-lame-1"]
        assert b["status"] == "finished" and b["score"] == -2.0 and b["gates"] == ["no successful solve"]
        assert b["n_solves"] == 0 and b["n_attempts"] == 1 and b["value"] is None
        cr = rows["g0-lame-2"]
        assert cr["status"] == "crashed" and cr["score"] is None and cr["error"] == "APIError: overloaded"
        g0 = d["generations"][0]
        assert g0["n_submissions"] == 3 and g0["n_crashed"] == 1 and g0["n_running"] == 0 and g0["n_gated"] == 1
        d2 = c.get("/api/swarm", params={"loop_id": "L2"}).json()
        assert [r["score"] for r in d2["problems"][0]["rows"]] == [-0.5]
        assert c.get(f"/api/consensus/{pid}").status_code == 200   # empty / crashed submissions don't break it


def test_swarm_speed_accuracy_scores(tmp_path):
    """New-style ACE records: 0-1 speed/accuracy score with score_parts, null = pending (not scored
    yet) or crashed, invalid runs score 0 with a gate, diagnosis text, persistent agent ids, and
    scores rewritten in place by the bridge each generation."""
    from feacore.runstore import write_json

    store = RunStore(tmp_path / "runs")
    pid = "nafems_le1"

    def run_and_submit(gen, aid, score, parts=None, gates=(), error=None, diagnosis=None, agent=None):
        rid = _ace_solve(store, "L-new", gen, aid, pid, 0, 9.25e7, 6000)
        meta = {"source": "ace", "loop_id": "L-new", "generation": gen, "ace_run_id": aid, "submitted_run_id": rid,
                "score": score, "gates": list(gates), "setup_diagnostics": [], "cited_bullets": [],
                "tokens": {"input": 1, "output": 1, "cache_write": 0, "cache_read": 0, "total": 2}}
        if parts is not None:
            meta["score_parts"] = parts
        if error:
            meta["error"] = error
        if diagnosis:
            meta["diagnosis"] = diagnosis
        if agent:
            meta["agent"] = agent
        return store.create_submission(pid, f"ace:{aid}", run_ids=[rid], meta=meta)

    parts = lambda acc, eff, ok=True: {"accuracy": acc, "efficiency": eff, "valid": ok, "rel_error": 0.004,  # noqa: E731
                                       "tolerance": 0.02, "meets_tol": ok, "dofs": 6000, "pareto_rank": 1, "population": 5}
    run_and_submit(0, "g0-le1-a1", 0.8, parts(0.9, 0.89), diagnosis="good mesh near D", agent="a1")
    run_and_submit(0, "g0-le1-a2", 0.0, parts(0.2, 0.5, ok=False), gates=["invalid: unconverged"])   # agent from the id
    run_and_submit(0, "g0-le1-a3", None, error="APIError: overloaded", agent="a3")
    run_and_submit(1, "g1-le1-a1", None, agent="a1")                                       # finished, not scored yet
    late = run_and_submit(1, "g1-le1-a2", 0.95, parts(0.97, 0.98), diagnosis={"verdict": "refined at the hole"})

    app = create_app(runs_dir=tmp_path / "runs", token="")
    with TestClient(app) as c:
        d = c.get("/api/swarm", params={"loop_id": "L-new", "generation": 1}).json()
        assert d["score_scheme"] == "speed_accuracy"
        rows = d["problems"][0]["rows"]
        assert [(r["agent"], r["status"]) for r in rows] == [("a2", "finished"), ("a1", "pending")]
        top = rows[0]
        assert top["score"] == 0.95 and top["score_parts"]["accuracy"] == 0.97 and top["score_parts"]["meets_tol"] is True
        assert top["diagnosis"] == "refined at the hole" and rows[1]["score"] is None
        gens = {g["generation"]: g for g in d["generations"]}
        assert gens[1]["n_pending"] == 1 and gens[1]["n_within_tolerance"] == 1 and gens[1]["median"] == 0.95
        assert gens[0]["n_invalid"] == 1 and gens[0]["n_crashed"] == 1 and gens[0]["n_pending"] == 0
        agents = {a["agent"]: a["generations"] for a in d["agents"]}
        assert list(agents) == ["a1", "a2", "a3"]
        assert agents["a1"]["0"]["mean"] == 0.8 and agents["a1"]["1"]["mean"] is None and agents["a1"]["1"]["n_pending"] == 1
        assert agents["a2"]["0"]["mean"] == 0.0 and agents["a2"]["1"]["mean"] == 0.95
        d0 = c.get("/api/swarm", params={"loop_id": "L-new", "generation": 0}).json()
        r0 = {r["agent"]: r for r in d0["problems"][0]["rows"]}
        assert r0["a1"]["diagnosis"] == "good mesh near D" and r0["a2"]["gates"] == ["invalid: unconverged"]
        assert r0["a3"]["status"] == "crashed" and r0["a2"]["score_parts"]["valid"] is False
        # the bridge rewrites scores in place every generation: the next read shows the new value
        path = store.submissions_dir / f"{late['submission_id']}.json"
        rec = json.loads(path.read_text())
        rec["meta"]["score"] = 0.5
        write_json(path, rec)
        d1 = c.get("/api/swarm", params={"loop_id": "L-new", "generation": 1}).json()
        assert d1["problems"][0]["rows"][0]["score"] == 0.5
