"""The dashboard's Gold and Your CAD sections (feacore/ui/gold_api.py, feacore/ui/cad_api.py,
feacore/problems/custom.py, feacore/goldgen/local.py, feacore/ui/tools/{gold,cad}.py).

Gold: the listing of a temporary gold root with the local status of its study meshes, path safety,
regenerating a level as a task and solving a set on it for the viewer. Your CAD: refused uploads,
a STEP upload that becomes registered problems which solve, are found by fresh processes and by the
ACE agents, and a reference study that certifies them. Meshes stay small (coarse levels, a box).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "benchmarks" / "gold"
PART = "abc_00000065"
SETS = ("abc_00000065_bc0", "abc_00000065_bc1")
BRACKET = ROOT / "physical_tests" / "parts" / "mounting_bracket_vertical.step"
ENV_KEYS = ("FEACORE_GOLD_DIR", "FEACORE_CAD_DIR", "FEACORE_CAD_MAX_BYTES")
made: dict = {}     # ids created by the upload test, used by the later ones


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A gold root with the two abc_00000065 sets (L0 present and listed, L1 present with other bytes,
    L2 missing) and an empty run store. The environment and the registry are restored afterwards."""
    tmp = tmp_path_factory.mktemp("goldcad")
    gold = tmp / "gold"
    for sid in SETS:
        shutil.copytree(GOLD / sid, gold / sid)
    index = json.loads((GOLD / "index.json").read_text())
    index["sets"] = {k: v for k, v in index["sets"].items() if k in SETS}
    index["parts"] = {PART: index["parts"][PART]}
    d = gold / "_meshes" / PART
    d.mkdir(parents=True)
    (d / "L0.msh").write_bytes(b"stored L0")
    index["parts"][PART]["meshes"]["L0"]["sha256"] = hashlib.sha256(b"stored L0").hexdigest()
    (d / "L1.msh").write_bytes(b"other bytes than the index says")
    (gold / "index.json").write_text(json.dumps(index))
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    os.environ["FEACORE_GOLD_DIR"] = str(gold)
    yield {"gold": gold, "runs": tmp / "runs", "index": index, "tmp": tmp}
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    from feacore.problems import custom

    custom.refresh()      # the temporary problems leave the registry with their root


@pytest.fixture(scope="module")
def client(env):
    from feacore.ui.server import create_app

    app = create_app(runs_dir=env["runs"], token="")
    with TestClient(app) as c:
        yield c


def _wait(client, url, key, done, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(url).json()
        if rec.get(key) in done:
            return rec
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {url}: {rec.get(key)}")


def _task(client, tool, args, timeout=600):
    r = client.post("/api/tasks", json={"tool": tool, "args": args})
    assert r.status_code == 202, r.text
    rec = _wait(client, f"/api/tasks/{r.json()['id']}", "state", ("done", "failed", "cancelled"), timeout)
    log = client.get(f"/api/tasks/{rec['id']}/log").text
    assert rec["state"] == "done", log[-3000:]
    return rec, log


def _step(path: Path, build) -> bytes:
    """A STEP file written by gmsh from ``build(occ)``."""
    import gmsh

    from feacore.geometry.gmsh_tools import gmsh_session

    with gmsh_session("t"):
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    return path.read_bytes()


def _upload(client, name: str, data: bytes, units: str = "si"):
    return client.post("/api/cad/uploads", files={"file": (name, data, "application/step")}, data={"units": units})


# --------------------------------------------------------------------------- gold


def test_gold_listing_and_mesh_status(client, env):
    d = client.get("/api/gold").json()
    assert d["n_sets"] == 2 and d["n_parts"] == 1 and d["n_meshes"] == 3 and d["n_present"] == 2 and d["n_ok"] == 1
    (p,) = d["parts"]
    assert {m["level"]: m["status"] for m in p["meshes"]} == {"L0": "ok", "L1": "mismatch", "L2": "missing"}
    s = {x["id"]: x for x in p["sets"]}["abc_00000065_bc0"]
    row = env["index"]["sets"]["abc_00000065_bc0"]
    assert s["bcs"] == ["clamp f12", "pressure 2.1 MPa f7"]
    assert [r["qoi"] for r in s["references"]] == row["references"] and s["singular"] == row["singular"]
    assert s["dofs"] == row["dofs"] and s["seconds"] == row["seconds"] and s["scored_qoi"] == "strain_energy"
    assert len(s["levels"]) == 3 and s["levels"][0]["qois"]["strain_energy"] > 0


def test_gold_downloads_are_path_safe(client, env):
    r = client.get(f"/api/gold/meshes/{PART}/L0")
    assert r.status_code == 200 and r.content == b"stored L0"
    for url in (f"/api/gold/meshes/{PART}/L2", "/api/gold/meshes/..%2F..%2Fetc/L0", f"/api/gold/meshes/{PART}/..%2Findex",
                f"/api/gold/meshes/{PART}/L0%2F..%2F..%2Findex", "/api/gold/meshes/..%2F..%2F..%2Fetc%2Fpasswd/L0",
                "/api/gold/regenerated/..%2Fx/L0", "/api/cad/uploads/..%2F..%2Fetc", "/api/cad/uploads/cad_nope_000000/geometry"):
        assert client.get(url).status_code == 404, url
    assert client.delete("/api/cad/problems/..%2F..%2Fx").status_code == 404
    assert client.put("/api/gold/archives/..%2Fevil.tgz", content=b"x").status_code in (404, 422)
    assert client.put("/api/gold/archives/x.tgz", content=b"not a tar").status_code == 422
    tar = env["tmp"] / "a.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(GOLD / "index.json", arcname="index.json")
    r = client.put("/api/gold/archives/goldgen-test_0000.tar", content=tar.read_bytes())
    assert r.status_code == 201 and r.json()["path"] == "@runs/gold_archives/goldgen-test_0000.tar"
    assert [a["name"] for a in client.get("/api/gold/archives").json()["archives"]] == ["goldgen-test_0000.tar"]
    for ids in ("--evil", "abc,-x", "../x"):    # tool arguments never become flags or paths
        assert client.post("/api/tasks", json={"tool": "gold.verify", "args": {"ids": ids}}).status_code == 422


def test_tools_are_registered(client):
    tools = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}
    assert {"gold.regenerate", "gold.verify", "gold.render", "gold.import", "gold.nonlinear"} <= set(tools)
    assert {"cad.reference", "cad.infer"} <= set(tools)
    assert tools["cad.infer"]["requires_env"] == ["ANTHROPIC_API_KEY"] and tools["cad.infer"]["unless_param"] == "mock"


def test_regenerate_installs_only_byte_identical_meshes(env, tmp_path, monkeypatch):
    """--install copies a regenerated level into _meshes/ only when its sha256 is the index's."""
    import feacore.goldgen.mesh as gm
    from feacore.goldgen import local

    def fake(step, part, k, out, *a, **kw):
        Path(out).write_bytes(b"stored L0" if k == 0 else b"different")
        return {"h": 0.01, "cells": 7, "scheme": "curved-opt", "min_sj": 0.5, "seconds": 0.1}

    monkeypatch.setattr(gm, "mesh_level_safe", fake)
    root = tmp_path / "gold"
    shutil.copytree(env["gold"], root)
    (root / "_meshes" / PART / "L0.msh").unlink()
    summary = local.regenerate(PART, [0, 1], root, out=tmp_path / "regen", install=True, log=lambda m: None)
    assert summary["levels"]["L0"]["match"] and summary["levels"]["L0"]["installed"]
    assert (root / "_meshes" / PART / "L0.msh").read_bytes() == b"stored L0"
    assert not summary["levels"]["L1"]["match"] and not summary["levels"]["L1"]["installed"]
    assert (root / "_meshes" / PART / "L1.msh").read_bytes() == b"other bytes than the index says"
    assert summary["h0_frac"] == 0.237       # the generator's argument, recovered from the stored L0 size


def test_regenerate_task_and_solve_a_level_for_the_viewer(client, env):
    _, log = _task(client, "gold.regenerate", {"part": PART, "levels": "0"})
    assert "L0:" in log and "sha256" in log
    p = client.get("/api/gold").json()["parts"][0]
    rg = next(m for m in p["meshes"] if m["level"] == "L0")["regenerated"]
    stored = env["index"]["parts"][PART]["meshes"]["L0"]
    assert rg["available"] and rg["stored_sha256"] == stored["sha256"]
    assert abs(rg["cells"] - stored["cells"]) <= 0.05 * stored["cells"]     # the same plan, not the same bytes
    assert client.post("/api/gold/solve", json={"set_id": SETS[0], "level": 2}).status_code == 404   # not here
    r = client.post("/api/gold/solve", json={"set_id": SETS[0], "level": 0, "source": "regenerated"})
    assert r.status_code == 202, r.text
    run = _wait(client, f"/api/runs/{r.json()['run_id']}", "status", ("succeeded", "failed"))
    assert run["status"] == "succeeded", run.get("error")
    assert run["fields"] and run["source"] == "gold" and run["meta"]["mesh_source"] == "regenerated"
    gold_l0 = next(x for x in p["sets"] if x["id"] == SETS[0])["levels"][0]["qois"]
    assert math.isclose(run["qois"]["strain_energy"], gold_l0["strain_energy"], rel_tol=0.05)
    assert client.get(f"/api/runs/{run['run_id']}/field").status_code == 200       # the viewer's data
    listed = next(x for x in client.get("/api/gold").json()["parts"][0]["sets"] if x["id"] == SETS[0])["runs"]
    assert run["run_id"] in [x["run_id"] for x in listed]


# --------------------------------------------------------------------------- Your CAD


def test_uploads_that_cannot_be_used_are_refused(client, env, monkeypatch):
    tmp = env["tmp"]
    two = _step(tmp / "two.step", lambda occ: (occ.addBox(0, 0, 0, 10, 10, 10), occ.addBox(20, 0, 0, 10, 10, 10)))
    shell = _step(tmp / "shell.step", lambda occ: occ.addRectangle(0, 0, 0, 10, 10))
    bracket = BRACKET.read_bytes()
    cases = [("x.step", b"", "empty"), ("x.step", b"hello world", "not a STEP file"),
             ("x.step", bracket[:-200], "incomplete"), ("two.step", two, "2 solids"), ("shell.step", shell, "no solid")]
    for name, data, needle in cases:
        r = _upload(client, name, data)
        assert r.status_code == 422 and needle in r.json()["detail"], (needle, r.text)
    r = _upload(client, "bracket.step", bracket, units="feet")
    assert r.status_code == 422 and "units" in r.json()["detail"]
    monkeypatch.setenv("FEACORE_CAD_MAX_BYTES", "1000")
    r = _upload(client, "bracket.step", bracket)
    assert r.status_code in (413, 422) and "limit" in r.json()["detail"]
    monkeypatch.delenv("FEACORE_CAD_MAX_BYTES")
    cad = env["runs"] / "cad"
    assert not cad.exists() or not any(cad.iterdir())               # nothing half-made is left behind
    assert client.get("/api/cad").json()["uploads"] == []


def test_upload_becomes_registered_problems(client):
    r = _upload(client, BRACKET.name, BRACKET.read_bytes())
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["n_faces"] == 30 and math.isclose(up["diag_mm"], 143.27, rel_tol=1e-3) and not up["existing"]
    assert sum(f["kind"] == "plane" for f in up["faces"]) >= 10        # flat B-spline faces count as planes
    again = _upload(client, BRACKET.name, BRACKET.read_bytes())
    assert again.status_code == 200 and again.json()["existing"] and again.json()["id"] == up["id"]
    uid = up["id"]
    auto = client.post(f"/api/cad/uploads/{uid}/sets", json={"mode": "auto", "count": 2})
    assert auto.status_code == 201 and auto.json()["problem_ids"] == [f"{uid}_bc0", f"{uid}_bc1"]
    big = sorted(up["faces"], key=lambda f: -f["area"])
    spec = {"mode": "manual", "fixed": [f"f{big[0]['tag']}"], "material": {"E": 70e9, "nu": 0.33}, "node_cap": 30000,
            "loads": [{"group": f"f{big[2]['tag']}", "type": "force", "value": [0, 0, -2000]}], "label": "alu"}
    bad = [({**spec, "fixed": []}, "clamp at least one face"), ({**spec, "loads": []}, "at least one load"),
           ({**spec, "fixed": ["f9999"]}, "not a face of this part"),
           ({**spec, "loads": [{"group": f"f{big[0]['tag']}", "type": "pressure", "value": 1e6}]}, "two supports or loads"),
           ({**spec, "material": {"E": -1, "nu": 0.3}}, "must be positive")]
    for s, needle in bad:
        r = client.post(f"/api/cad/uploads/{uid}/sets", json=s)
        assert r.status_code == 422 and needle in r.json()["detail"], (needle, r.text)
    manual = client.post(f"/api/cad/uploads/{uid}/sets", json=spec)
    assert manual.status_code == 201, manual.text
    (pid,) = manual.json()["problem_ids"]
    assert pid == f"{uid}_m0"
    listed = {p["id"]: p for p in client.get("/api/problems?tag=custom").json()["problems"]}
    assert {f"{uid}_bc0", f"{uid}_bc1", pid} <= set(listed) and listed[pid]["family"] == uid
    det = client.get(f"/api/problems/{pid}").json()
    assert "E = 70 GPa" in det["statement"]["description"] and det["evaluator"]["references"] == []
    assert len(det["evaluator"]["study_sizes"]) == 3
    geo = client.get(f"/api/problems/{pid}/geometry")     # built by a worker process: it finds the problem too
    assert geo.status_code == 200 and geo.json()["n_triangles"] > 0
    view = next(u for u in client.get("/api/cad").json()["uploads"] if u["id"] == uid)
    s = next(x for x in view["sets"] if x["id"] == pid)
    assert s["scored_qoi"] == "strain_energy" and s["node_cap"] == 30000 and s["references"] == [] and s["registered"]
    made.update(uid=uid, pid=pid)


def test_custom_problem_solves_on_a_coarse_mesh(client, tmp_path):
    from feacore.problems import registry

    if "pid" not in made:
        pytest.skip("needs the upload test")
    p = registry.get(made["pid"])
    mesh = p.mesh(tmp_path / "coarse.msh", p.study_sizes()[0] * 2)      # the gold meshing pipeline, coarse
    case = p.case("coarse.msh").to_dict()
    r = client.post("/api/solve", data={"problem_id": p.id, "fields": "false"},
                    files={"case": ("case.json", json.dumps(case), "application/json"),
                           "mesh": ("coarse.msh", mesh.read_bytes(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    q = r.json()["qois"]
    assert q["strain_energy"] > 0 and q["u_max"] > 0 and all(math.isfinite(v) for v in q.values())


def _fresh(code: str, env: dict) -> dict:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_fresh_processes_and_the_agents_find_the_problem(env):
    if "pid" not in made:
        pytest.skip("needs the upload test")
    pid = made["pid"]
    base = {k: v for k, v in os.environ.items() if k not in ("FEACORE_CAD_DIR", "FEACORE_RUNS_DIR")}
    base["PYTHONPATH"] = str(ROOT)
    code = ("import json; from feacore.problems import registry; from ace import config, fea; "
            "from ace.infer import task_entry; from ace.agent import task_prompt; "
            f"p = fea.get_problem({pid!r}); print(json.dumps({{'tags': list(p.tags), 'task': config.TASKS[{pid!r}], "
            f"'entry': task_entry({pid!r}), 'prompt': task_prompt({pid!r}), 'n': len(registry.all_problems(tags=['custom']))}}))")
    got = _fresh(code, {**base, "FEACORE_RUNS_DIR": str(env["runs"])})     # what the dashboard's tasks get
    assert "custom" in got["tags"] and got["task"] == {"qoi": "strain_energy", "node_cap": 30000} == got["entry"]
    assert f"Problem id: {pid}" in got["prompt"] and "SCORED QUANTITY: strain_energy" in got["prompt"]
    assert "at most 30000 nodes" in got["prompt"] and got["n"] == 3
    only_dir = _fresh(f"import json; from feacore.problems import registry; print(json.dumps(registry.get({pid!r}).family))",
                      {**base, "FEACORE_CAD_DIR": str(env["runs"] / "cad")})
    assert only_dir == made["uid"]


def test_reference_study_certifies_a_small_part(client, env):
    """A box (fast): a generated BC set, the CPU reference study as a task, references certified,
    the solves recorded for the viewer, and the agents' scored quantity then has a reference."""
    box = _step(env["tmp"] / "box.step", lambda occ: occ.addBox(0, 0, 0, 60, 30, 20))
    r = _upload(client, "box.step", box)
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    (pid,) = client.post(f"/api/cad/uploads/{uid}/sets", json={"mode": "auto", "count": 1}).json()["problem_ids"]
    _, log = _task(client, "cad.reference", {"upload": uid, "max_cells": 3000}, timeout=900)
    assert f"{pid}: certified" in log
    s = next(x for u in client.get("/api/cad").json()["uploads"] if u["id"] == uid for x in u["sets"])
    assert len(s["levels"]) == 3 and s["references"], log[-2000:]
    assert s["scored_qoi"] in {x["qoi"] for x in s["references"]}
    runs = [x for x in client.get(f"/api/runs?problem_id={pid}").json()["runs"] if (x["agent_id"] or "").startswith("reference:")]
    assert len(runs) == 3 and all(x["fields"] for x in runs)
    from feacore.problems import registry

    p = registry.get(pid)
    assert p.references() and p.mesh_file(2) is not None and len(p.study_sizes()) == 3
    from feacore.problems.custom import ace_tasks

    assert ace_tasks()[pid]["qoi"] == s["scored_qoi"]


def test_adopting_a_finished_study_certifies_it(client, env):
    """A finished P2 study of the problem (here a stored record whose QoIs converge at the theoretical
    order) goes through the gold certification and becomes the reference at once."""
    from feacore.problems import registry
    from feacore.runstore import RunStore
    from feacore.verification.study import default_expected_order

    if "pid" not in made:
        pytest.skip("needs the upload test")
    pid = made["pid"]
    assert client.post(f"/api/cad/problems/{pid}/adopt", json={"study_id": "study-nope"}).status_code == 404
    store, hs = RunStore(env["runs"]), [0.02, 0.01, 0.005]
    runs = [{"h_nominal": h, "info": {"h_rep": h, "n_dofs": 1000 * 8**i, "wall_time": 1.0},
             "qois": {q.name: 1.0 + 0.02 * (h / hs[0]) ** default_expected_order(q, 2) for q in registry.get(pid).qois()}}
            for i, h in enumerate(hs)]

    def study(order=2, status="succeeded", problem=pid):
        sid = store.create_study(problem, order=order, sizes=hs)["study_id"]
        store.update_study(sid, status=status, result={"runs": runs})
        return sid

    for sid, needle in ((study(order=1), "P2"), (study(status="running"), "not succeeded"),
                        (study(problem="cooks_membrane"), "cooks_membrane")):
        r = client.post(f"/api/cad/problems/{pid}/adopt", json={"study_id": sid})
        assert r.status_code == 422 and needle in r.text, r.text
    r = client.post(f"/api/cad/problems/{pid}/adopt", json={"study_id": study()})
    assert r.status_code == 200, r.text
    certified = {x["qoi"] for x in r.json()["references"]}
    assert "strain_energy" in certified
    assert {x.qoi for x in registry.get(pid).references()} == certified     # the registered problem sees it now
