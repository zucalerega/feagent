"""The dashboard's Projects section (feacore/ui/dossier_api.py, feacore/ui/tools/dossier.py).

A project is made from the shipped example in a temporary run store; nothing is written to dossier/examples.
Only the context stage runs (fast); the report is a small fake page.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.ui.server import create_app  # noqa: E402
from feacore.ui.tasks import TOOLS  # noqa: E402
from feacore.ui.tools import dossier as tools  # noqa: E402

PROJECT = "t_valve"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    app = create_app(runs_dir=runs, token="")
    with TestClient(app) as c:
        r = c.post("/api/dossier", json={"example": "mov_valve", "folder": PROJECT})
        assert r.status_code == 201, r.text
        yield {"c": c, "root": runs / "dossier" / PROJECT}


def test_list_and_create(env):
    c = env["c"]
    d = c.get("/api/dossier").json()
    assert [p["folder"] for p in d["projects"]] == [PROJECT]
    assert any(e["name"] == "mov_valve" for e in d["examples"])
    assert c.post("/api/dossier", json={"example": "mov_valve", "folder": PROJECT}).status_code == 409
    for bad in ("../x", "a/b", "", ".hid"):
        assert c.post("/api/dossier", json={"example": "mov_valve", "folder": bad}).status_code == 422
    assert c.post("/api/dossier", json={"example": "nope", "folder": "zz"}).status_code == 404


def test_status_check_params(env):
    c = env["c"]
    s = c.get(f"/api/dossier/{PROJECT}")
    assert s.status_code == 200 and s.json()["folder"] == PROJECT
    assert c.get("/api/dossier/missing").status_code == 404
    chk = c.post(f"/api/dossier/{PROJECT}/check").json()
    assert chk["ok"] is True, chk
    p = c.get(f"/api/dossier/{PROJECT}/params").json()
    assert p and len(str(p)) > 100


def test_files_read_write_and_problems(env):
    c = env["c"]
    files = c.get(f"/api/dossier/{PROJECT}/files").json()["files"]
    paths = [f["path"] for f in files]
    assert "system.yaml" in paths and "project.yaml" in paths
    r = c.get(f"/api/dossier/{PROJECT}/files/system.yaml")
    assert r.status_code == 200
    text = r.json()["text"]
    assert "effects" in text
    # a unit error: a pressure in metres; saved as it is, the problems come back
    broken = text.replace("MPa", "m", 1)
    assert broken != text
    out = c.put(f"/api/dossier/{PROJECT}/files/system.yaml", content=broken.encode()).json()
    assert out["ok"] is False and out["problems"]
    assert c.get(f"/api/dossier/{PROJECT}/files/system.yaml").status_code == 200
    out = c.put(f"/api/dossier/{PROJECT}/files/system.yaml", content=text.encode()).json()
    assert out["ok"] is True, out
    # refused: traversal, other file types, files that are not inputs
    for bad in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "out/x.yaml", "components/mov_body/geometry.step"):
        assert c.put(f"/api/dossier/{PROJECT}/files/{bad}", content=b"x: 1").status_code in (400, 403, 404, 405, 415, 422)
        if not bad.endswith(".step"):      # a CAD file can be downloaded, never written as text
            assert c.get(f"/api/dossier/{PROJECT}/files/{bad}").status_code in (400, 403, 404, 405, 415, 422)


def test_records_trace_reviews(env):
    c = env["c"]
    assert c.post(f"/api/dossier/{PROJECT}/tasks", json={"tool": "dossier.run", "args": {"to": "context", "quick": True}}
                  ).status_code == 202
    import time
    for _ in range(120):
        t = c.get(f"/api/dossier/{PROJECT}").json()
        if (env["root"] / "out" / "context.json").exists() or any(
                x.get("state") in ("done", "failed") for x in t.get("tasks", [])):
            break
        time.sleep(0.5)
    rec = c.get(f"/api/dossier/{PROJECT}/records").json()
    assert rec
    assert c.get(f"/api/dossier/{PROJECT}/trace").status_code == 200
    assert c.get(f"/api/dossier/{PROJECT}/reviews").status_code == 200
    # a review of a record that does not exist is refused; a missing name is refused
    assert c.post(f"/api/dossier/{PROJECT}/reviews", json={"record": "X", "state": "weird"}).status_code == 422
    assert c.post(f"/api/dossier/{PROJECT}/reviews", json={"record": "X", "state": "accepted"}).status_code == 422


def test_report_serving(env):
    c = env["c"]
    page = c.get(f"/api/dossier/{PROJECT}/report/index.html")
    assert page.status_code == 404 and "not built" in page.text.lower()
    rep = env["root"] / "out" / "report"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "index.html").write_text("<html><head><title>r</title></head><body>hi</body></html>")
    page = c.get(f"/api/dossier/{PROJECT}/report/index.html").text
    assert f'window.DOSSIER_API = "/api/dossier/{PROJECT}"' in page and page.index("DOSSIER_API") < page.index("</head>")
    for bad in ("../../project.yaml", "..%2f..%2fproject.yaml", "%2e%2e/project.yaml"):
        assert c.get(f"/api/dossier/{PROJECT}/report/{bad}").status_code in (400, 403, 404, 405)
        assert c.get(f"/api/dossier/{PROJECT}/out/{bad}").status_code in (400, 403, 404, 405)


def test_tools():
    assert tools.RUN.name in TOOLS and tools.REPORT.name in TOOLS
    from dossier.pipeline import STAGES
    assert tuple(STAGES) == tools.STAGES
    with pytest.raises(Exception):
        tools._run_argv({"project": "/etc"})
    with pytest.raises(Exception):
        tools.load_case_ids("--force")
    assert tools.load_case_ids("LC-A, LC-B") == ["LC-A", "LC-B"]


def test_reassess_argv(env):
    """The Re-assess button runs the stages assess to report of the project, nothing analysed again."""
    ref = tools.project_ref(PROJECT)
    runs = env["root"].parent.parent
    argv = tools._run_argv({"project": str(runs / "dossier" / PROJECT), "from": "assess", "to": "report", "quick": False})
    assert argv[0] == "run" and argv[2:6] == ["--from", "assess", "--to", "report"]
    assert "--quick" not in argv and "--force" not in argv and ref.endswith(PROJECT)
    with pytest.raises(Exception):      # the stages must be in order
        tools._run_argv({"project": str(runs / "dossier" / PROJECT), "from": "report", "to": "assess"})


def test_offscreen_runs_directly_without_xvfb_or_off_linux(monkeypatch):
    """No xvfb-run on macOS and Windows: the script runs as it is. Only Linux without a display starts Xvfb."""
    import sys

    from feacore.ui import offscreen

    monkeypatch.setattr(offscreen.shutil, "which", lambda name: "/usr/bin/xvfb-run" if name == "xvfb-run" else None)
    monkeypatch.setattr(sys, "platform", "linux")
    assert offscreen.display_mode({}) == "xvfb"
    assert offscreen.command("s.py", ["a"], {})[0] == "xvfb-run"
    assert offscreen.display_mode({"DISPLAY": ":0"}) == "display"
    assert offscreen.command("s.py", ["a"], {"DISPLAY": ":0"})[0] == sys.executable
    for plat in ("darwin", "win32"):
        monkeypatch.setattr(sys, "platform", plat)
        assert offscreen.display_mode({}) == "none"
        assert offscreen.command("s.py", ["a"], {}) == [sys.executable, "s.py", "a"]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(offscreen.shutil, "which", lambda name: None)
    assert offscreen.display_mode({}) == "none"
