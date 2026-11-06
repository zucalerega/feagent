"""The dashboard's Nonlinear and Test data sections: the nonlinear tools (argv they build), the run listing and
curves of small fake run directories, uploads of machine exports read with the loop's own reader, path checks
on every file the pages serve, and one real short CalculiX run through the task lane."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.runstore import RunStore  # noqa: E402
from feacore.ui import tasks as T  # noqa: E402
from feacore.ui.media import REPO_ROOT, RefError, resolve_ref  # noqa: E402
from feacore.ui.server import create_app  # noqa: E402

COMPRESSION_CSV = REPO_ROOT / "physical_tests" / "abc_00000065_compression" / "cload_compression.csv"
CRUSH_COLS = ["step", "platen_travel_mm", "force_N", "n_failed", "n_detached", "max_peeq", "iterations", "block"]


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="module")
def client(store_root):
    app = create_app(runs_dir=store_root, token="")
    with TestClient(app) as c:
        yield c


def _args(name: str, args: dict, store_root: Path) -> list[str]:
    """The argv a tool would run with (without starting it)."""
    tool = T.get_tool(name)
    clean = T.validate_args(tool, args, RunStore(store_root))
    if tool.out_param and not clean.get(tool.out_param):
        clean[tool.out_param] = str(store_root / "out" / "x-1")
    return T.build_argv(tool, clean)


# --------------------------------------------------------------------------- tools


def test_the_sections_tools_are_registered(client):
    d = client.get("/api/tools").json()
    assert not d["import_errors"]
    names = {t["name"] for t in d["tools"]}
    assert {"nonlinear.score", "nonlinear.ramp", "nonlinear.crush_mesh", "nonlinear.imperfection", "nonlinear.crush",
            "nonlinear.render_ramp", "nonlinear.render_crush", "nonlinear.render_score", "testdata.summarize"} <= names
    groups = {t["name"]: t["group"] for t in d["tools"]}
    assert groups["nonlinear.crush"] == "nonlinear" and groups["testdata.summarize"] == "testdata"


def test_score_and_ramp_argv(store_root):
    argv = _args("nonlinear.score", {}, store_root)
    assert argv[:3] == ["-m", "feacore.ui.offscreen", "scripts/nl_baselines.py"]
    assert argv[3:] == ["--only", "auto", "h4", "--threads", "4", "--out", str(store_root / "out" / "x-1")]
    with pytest.raises(T.ToolArgsError, match="at least one mesh"):
        _args("nonlinear.score", {"auto": False, "h4": False}, store_root)

    argv = _args("nonlinear.ramp", {"materials": "team,corrected"}, store_root)
    assert argv[0] == "scripts/ccx_compression_ramp.py" and "--straight" in argv
    assert argv[argv.index("--threads") + 1] == "2"             # the two materials share 4 threads
    assert argv[argv.index("--work") + 1] == argv[argv.index("--out") + 1]
    with pytest.raises(T.ToolArgsError, match="curved tets need h"):
        _args("nonlinear.ramp", {"straight": False, "h": 5}, store_root)
    assert "--straight" not in _args("nonlinear.ramp", {"straight": False, "h": 3}, store_root)


def test_crush_argv_and_its_checks(store_root):
    out = store_root / "nonlinear" / "crush" / "crush-1"
    argv = _args("nonlinear.crush", {"mesh_dir": "experiments/physical", "failure": "peeq_tension",
                                     "compression_factor": 0, "compression_switch": "0.5:5", "out": str(out)}, store_root)
    assert argv[0] == "experiments/physical/crush.py"
    assert argv[argv.index("--out") + 1] == str(out.parent) and argv[argv.index("--run-id") + 1] == "crush-1"
    assert argv[argv.index("--compression-factor") + 1] == "0" and "--compression-switch" in argv
    assert argv[argv.index("--mesh-dir") + 1] == "experiments/physical" and "--resume" not in argv
    with pytest.raises(T.ToolArgsError, match="mesh_dir is required"):
        _args("nonlinear.crush", {}, store_root)
    with pytest.raises(T.ToolArgsError, match="0.5:5"):
        _args("nonlinear.crush", {"mesh_dir": "experiments", "failure": "peeq_tension", "compression_switch": "half"},
              store_root)
    with pytest.raises(T.ToolArgsError, match="needs failure"):
        _args("nonlinear.crush", {"mesh_dir": "experiments", "compression_switch": "0.5:5"}, store_root)
    resumed = _args("nonlinear.crush", {"resume": "experiments/physical"}, store_root)
    assert "--resume" in resumed and "--mesh-dir" not in resumed
    with pytest.raises(T.ToolArgsError):          # paths stay inside the repo or the run store
        _args("nonlinear.crush", {"mesh_dir": "/etc"}, store_root)


def test_mesh_and_renderer_argv(store_root):
    argv = _args("nonlinear.crush_mesh", {"part": "abc_00000003", "h": 6}, store_root)
    assert argv[:4] == ["experiments/mesh_systems/ace_sizing/build.py",
                        "benchmarks/gold/abc_00000003_bc0/public/geometry.step", "z", str(store_root / "out" / "x-1")]
    assert argv[argv.index("--h") + 1] == "0.006" and "--straight" in argv
    run = str(store_root / "nonlinear" / "crush" / "fake")
    argv = _args("nonlinear.render_crush", {"run": run, "only": "0,-1"}, store_root)
    assert argv[:3] == ["-m", "feacore.ui.offscreen", "experiments/physical/render_crush.py"] and argv[3] == run
    assert argv[argv.index("--out") + 1].endswith("/crush.mp4") and argv[argv.index("--only") + 1] == "0,-1"
    with pytest.raises(T.ToolArgsError, match="frame indices"):
        _args("nonlinear.render_crush", {"run": run, "only": "0;rm"}, store_root)
    argv = _args("nonlinear.render_score", {"run": run}, store_root)
    assert argv[:5] == ["-m", "feacore.ui.offscreen", "scripts/nl_video.py", "ramp", run]
    assert _args("testdata.summarize", {"spec": "physical_tests/abc_00000065_compression/cload_compression.json",
                                        "predict": True, "h": 4}, store_root)[-3:] == ["--predict", "--h", "0.004"]


# --------------------------------------------------------------------------- runs and curves


def _fake_crush(d: Path, finished: bool = True) -> None:
    d.mkdir(parents=True)
    u = np.linspace(0.05, 2.0, 40)
    F = np.where(u < 1.7, 2300 * u, 2300 * 1.7 * np.exp(-(u - 1.7) * 8))
    nf = np.where(u > 1.2, np.round((u - 1.2) * 100), 0).astype(int)
    with open(d / "force_displacement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CRUSH_COLS)
        for i, (a, b, c) in enumerate(zip(u, F, nf)):
            w.writerow([i + 1, f"{a:.6g}", f"{b:.6g}", c, 0, 0.01, 2, i // 4])
        f.write("41,2.05")                           # a row the solver is still writing
    (d / "run.json").write_text(json.dumps({"system": "test_mesh", "part": "abc_00000065", "platen": "bonded",
                                            "failure": {"criterion": "peeq"}, "release": "adaptive", "threads": 2}))
    (d / "progress.log").write_text("12:00:00 start\n12:00:01 step 1 block 0 travel 0.05 mm force 115 N\n")
    if finished:
        (d / "summary.json").write_text(json.dumps({"stop_reason": "separated", "separated": True, "n_nodes": 1234,
                                                    "first_failure": {"height_fraction": 0.62}}))


def _fake_ramp(d: Path) -> None:
    d.mkdir(parents=True)
    hist = [{"u_mm": 0.2 * k, "F_N": 3000 * (1 - math.exp(-0.2 * k / 1.2)), "max_peeq": 0.0,
             "plastic_volume_fraction": 0.0, "max_von_mises_MPa": 10.0, "iterations": 3} for k in range(1, 16)]
    (d / "team_stick.json").write_text(json.dumps({"name": "team_stick", "material": {"name": "team", "E": 1879.8},
                                                   "platen": "stick", "n_nodes": 6493, "history": hist,
                                                   "converged": True}))


def _fake_score(d: Path) -> None:
    d.mkdir(parents=True)
    u = [0.0, 0.5, 1.0, 1.5, 2.0, 2.4]
    F = [0.0, 1500.0, 2700.0, 3600.0, 4300.0, 4700.0]
    uc = [0.0] + [a + 0.2335 + 2.898e-5 * b for a, b in zip(u[1:], F[1:])]
    (d / "summary.json").write_text(json.dumps({"ok": True, "mesh_sha256": "x", "nodes": 9506, "dofs": 28518,
                                                "u_part_mm": u, "u_cross_mm": uc, "force_N": F,
                                                "mean_rel_err": 0.05, "series_dir": None}))
    (d / "curve.csv").write_text("u_part_mm,u_cross_mm,force_N\n" + "".join(f"{a},{b},{c}\n" for a, b, c in zip(u, uc, F)))


def test_run_listing_and_curves(client, store_root):
    nl = store_root / "nonlinear"
    _fake_crush(nl / "crush" / "fake1")
    _fake_crush(nl / "test_mesh" / "abc_00000065" / "run-live", finished=False)     # the CLI's default layout
    _fake_ramp(nl / "ramp" / "r1")
    _fake_score(nl / "score" / "s1" / "h4")
    (nl / "crush" / "fake1" / "media").mkdir()
    shutil.copy(REPO_ROOT / "talk_plots" / "act3_real_data" / "fig1_measured_curve.png",
                nl / "crush" / "fake1" / "media" / "still.png")

    runs = {r["ref"]: r for r in client.get("/api/nonlinear/runs").json()["runs"]}
    assert runs["@runs/nonlinear/crush/fake1"]["kind"] == "crush"
    assert runs["@runs/nonlinear/crush/fake1"]["status"] == "finished"
    assert runs["@runs/nonlinear/test_mesh/abc_00000065/run-live"]["status"] == "running"   # fresh progress.log
    assert runs["@runs/nonlinear/ramp/r1"]["kind"] == "ramp"
    assert runs["@runs/nonlinear/score/s1"]["kind"] == "score" and runs["@runs/nonlinear/score/s1"]["n_curves"] == 1
    assert "physical_tests/abc_00000065_compression/ccx_ramp" in runs          # committed runs are listed too
    assert abs(runs["@runs/nonlinear/crush/fake1"]["peak_N"] - 3910) < 20

    d = client.get("/api/nonlinear/run", params={"ref": "@runs/nonlinear/crush/fake1"}).json()
    (c,) = d["curves"]
    assert len(c["u"]) == len(c["F"]) == 41 and c["u"][0] == 0.0                # the origin, then 40 full rows
    assert 0 < c["error"] < 1 and c["first_failure"][0] > 1.2
    assert d["facts"]["n_failed"] == 80 and d["facts"]["separated"] is True
    assert [m["name"] for m in d["media"]] == ["media/still.png"]
    assert {f["name"] for f in d["files"]} >= {"force_displacement.csv", "run.json", "summary.json"}
    img = client.get("/api/nonlinear/file", params={"ref": d["media"][0]["ref"]})
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    ramp = client.get("/api/nonlinear/run", params={"ref": "@runs/nonlinear/ramp/r1"}).json()
    assert ramp["curves"][0]["name"] == "team_stick" and len(ramp["curves"][0]["u"]) == 16
    score = client.get("/api/nonlinear/run", params={"ref": "@runs/nonlinear/score/s1"}).json()
    assert score["curves"][0]["error"] == 0.05 and len(score["curves"][0]["u_cross"]) == 6

    test = client.get("/api/nonlinear/test").json()
    assert test["checkpoints_mm"] == [0.5, 1.0, 1.5, 2.0, 2.5] and abs(test["peak_N"] - 4755.2) < 1
    assert test["delta_mm"] == pytest.approx(0.2335)


def test_gallery_has_captions_and_the_committed_videos(client):
    g = client.get("/api/nonlinear/gallery").json()
    items = [it for s in g["sections"] for it in s["items"]]
    videos = [it for it in items if it["kind"] == "video"]
    assert any(v["ref"] == "talk_plots/act3_real_data/videos/before_after.mp4" and v["caption"] for v in videos)
    assert any(v["rtx"] for v in videos)                       # the Omniverse grid: committed media only
    assert all(it["poster"] for it in videos if it["ref"].startswith("talk_plots/"))
    fig1 = next(it for it in items if it["ref"].endswith("fig1_measured_curve.png"))
    assert fig1["title"].startswith("Real test") and "Instron" in fig1["caption"]
    r = client.get("/api/nonlinear/file", params={"ref": videos[0]["ref"]}, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_meshes_and_capabilities(client, store_root):
    d = store_root / "nonlinear" / "meshes" / "m1"
    d.mkdir(parents=True)
    (d / "mesh.msh").write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
    (d / "report.json").write_text(json.dumps({"part": "abc_00000065", "system": "ace_sizing", "n_nodes": 7227,
                                               "n_elements": 3600, "knobs": {"h": 0.008, "straight_edges": True}}))
    (m,) = client.get("/api/nonlinear/meshes").json()["meshes"]
    assert m["ref"] == "@runs/nonlinear/meshes/m1" and m["h_mm"] == 8.0 and m["straight"] is True
    caps = client.get("/api/nonlinear/capabilities").json()
    assert set(caps) >= {"ccx", "ffmpeg", "display", "pyvista", "notes"}


# --------------------------------------------------------------------------- test data


def _bluehill(path: Path, compliance: float, peak: float = 300.0, cycles: int = 3) -> Path:
    """A Bluehill-like export: semicolons, decimal commas, units in the header, kN, a title block."""
    rng = np.random.default_rng(1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Specimen label", "pull 1"])
        w.writerow([])
        w.writerow(["Time (s)", "Load (kN)", "Extension (Video) (mm)"])
        t = 0.0
        for c in range(cycles):
            for L in np.concatenate([np.linspace(0, peak, 80), np.linspace(peak, 0, 80)[1:]]):
                t += 0.5
                d = compliance * L + 5e-6 + rng.normal(0, 2e-7)
                w.writerow([f"{t:.2f}".replace(".", ","), f"{L / 1000:.6f}".replace(".", ","),
                            f"{d * 1000:.6f}".replace(".", ",")])
    return path


def test_upload_parse_save_list_and_the_loop_reads_it(client, store_root, tmp_path):
    from ace import evidence as EV

    raw = COMPRESSION_CSV.read_bytes()
    form = {"problem_id": "abc_00000065_compression", "until_peak": "true", "window_lo": "0.22", "window_hi": "0.38",
            "sign": "-1"}
    p = client.post("/api/testdata/parse", files={"file": ("cload_compression.csv", raw, "text/csv")}, data=form)
    assert p.status_code == 200, p.text
    pj = p.json()
    assert pj["data"] == {"load": "Force", "load_unit": "n", "displacement": "Displacement", "displacement_unit": "mm"}
    assert pj["sniff"]["header_row"] == 3 and pj["sniff"]["units_row"] == 4
    assert abs(pj["stats"]["peak_N"] - 4755.2) < 1 and pj["loop"]["ok"] and len(pj["curve"]) > 500

    s = client.post("/api/testdata", files={"file": ("cload_compression.csv", raw, "text/csv")},
                    data={**form, "test_id": "cbracket re-upload", "specimen": "print 1", "part": "abc_00000065",
                          "description": "same export as the committed test", "notes": "for the test suite"})
    assert s.status_code == 201, s.text
    rec = s.json()
    assert rec["uploaded"] and rec["dir_ref"].startswith("@runs/testdata/cbracket-re-upload-")
    d = Path(rec["dir_abs"])
    assert sorted(x.name for x in d.iterdir()) == ["cload_compression.csv", "summary.json", "test.json"]
    assert EV.discover([d]) == [d / "test.json"]                 # only the spec looks like a spec to the loop
    t = EV.process_test(EV.read_spec(d / "test.json"))
    assert t["problem_id"] == "abc_00000065_compression" and t["sign"] == -1.0 and t["fit_window"] == [0.22, 0.38]
    assert json.loads((d / "test.json").read_text())["lab"]["specimen"] == "print 1"
    assert f"--evidence {d}" in rec["commands"]["loop"]

    listed = client.get("/api/testdata").json()
    ids = [x["id"] for x in listed["tests"]]
    assert rec["id"] in ids and "cload-compression-1" in ids
    assert {"abc_00000003_pull", "abc_00000065_compression"} <= {x["id"] for x in listed["problems"]}
    assert [x["id"] for x in listed["parts"]["parts"]] == ["abc_00000003_pull", "abc_00000065_pull",
                                                           "mounting_bracket_vertical"]
    item = client.get("/api/testdata/item", params={"id": "cload-compression-1"}).json()
    assert item["extra_curves"][0]["name"] == "dogbone1.csv" and item["loop"]["ok"]

    assert client.delete("/api/testdata/item", params={"id": "cload-compression-1"}).status_code == 404
    assert client.delete("/api/testdata/item", params={"id": rec["id"]}).status_code == 200
    assert not d.exists()


def test_other_export_formats_and_bad_files(client, tmp_path):
    f = _bluehill(tmp_path / "pull.csv", 1.578e-7)
    p = client.post("/api/testdata/parse", files={"file": ("pull.csv", f.read_bytes(), "text/csv")},
                    data={"problem_id": "abc_00000003_pull"})
    assert p.status_code == 200, p.text
    pj = p.json()
    assert pj["sniff"]["delimiter"] == "semicolon" and pj["sniff"]["decimal_comma"]
    assert pj["data"]["load"] == "Load (kN)" and pj["data"]["load_unit"] == "kn"
    assert pj["data"]["displacement"] == "Extension (Video) (mm)" and pj["data"]["displacement_unit"] == "mm"
    assert abs(pj["loop"]["compliance"] / 1.578e-7 - 1) < 0.01 and pj["stats"]["loadings"] == 3

    bad = client.post("/api/testdata/parse", files={"file": ("x.csv", b"1,2\n3,4\n5,6\n7,8\n", "text/csv")})
    assert bad.status_code == 422 and "header" in bad.json()["detail"]
    assert client.post("/api/testdata/parse", files={"file": ("x.xlsx", b"PK", "application/zip")}).status_code == 422
    nop = client.post("/api/testdata", files={"file": ("pull.csv", f.read_bytes(), "text/csv")})
    assert nop.status_code == 422 and "problem_id" in nop.text
    wrong = client.post("/api/testdata/parse", files={"file": ("pull.csv", f.read_bytes(), "text/csv")},
                        data={"load": "Time (s)", "displacement": "Time (s)"})
    assert wrong.status_code == 422 and "different columns" in wrong.json()["detail"]


# --------------------------------------------------------------------------- path checks


@pytest.mark.parametrize("ref", ["../etc/passwd", "@runs/../../etc/passwd", "/etc/passwd", "README.md",
                                 "@runs/tasks/task-x/task.json", "physical_tests/../README.md",
                                 "@runs/nonlinear/../tasks", "physical_tests\\..\\README.md", "~/x", "",
                                 "feacore/ui/server.py", "@runs/nonlinear/link/passwd"])
def test_file_refs_outside_the_allowed_directories_are_refused(client, store_root, ref):
    link = store_root / "nonlinear" / "link"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink("/etc", link)                       # a symlink out of the store resolves outside: refused
    for url in ("/api/nonlinear/file", "/api/testdata/file"):
        assert client.get(url, params={"ref": ref}).status_code in (404, 422), (url, ref)
    assert client.get("/api/nonlinear/run", params={"ref": ref}).status_code in (404, 422)


def test_unknown_file_types_and_test_ids_are_refused(client, store_root):
    d = store_root / "nonlinear" / "crush" / "evil"
    d.mkdir(parents=True, exist_ok=True)
    (d / "page.html").write_text("<script>alert(1)</script>")
    assert client.get("/api/nonlinear/file", params={"ref": "@runs/nonlinear/crush/evil/page.html"}).status_code == 404
    assert client.get("/api/nonlinear/file", params={"ref": "experiments/physical/crush.py"}).status_code == 404
    assert client.get("/api/testdata/item", params={"id": "../../etc"}).status_code == 404
    assert client.delete("/api/testdata/item", params={"id": "../testdata"}).status_code == 404
    with pytest.raises(RefError):
        resolve_ref("@runs/nonlinear/../../x", store_root)


# --------------------------------------------------------------------------- one real CalculiX run


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")
def test_a_real_scoring_ramp_runs_through_the_task_lane(client, store_root):
    """The agents' frozen scoring ramp on the automesher's mesh (2.9k nodes, about 20 s): the task writes a
    scoring run that the listing finds, with the committed error against the test (7.8 %)."""
    r = client.post("/api/tasks", json={"tool": "nonlinear.score", "args": {"auto": True, "h4": False, "threads": 2}})
    assert r.status_code == 202, r.text
    task = r.json()
    t0 = time.time()
    while time.time() - t0 < 300:
        rec = client.get(f"/api/tasks/{task['id']}").json()
        if rec["state"] not in ("queued", "running"):
            break
        time.sleep(1)
    assert rec["state"] == "done", client.get(f"/api/tasks/{task['id']}/log").text[-3000:]
    run = client.get("/api/nonlinear/run", params={"ref": task["out"]}).json()
    assert run["kind"] == "score" and run["source"] == "dashboard" and run["task"]["id"] == task["id"]
    (c,) = run["curves"]
    assert c["name"] == "auto" and c["error"] == pytest.approx(0.0784, abs=0.003)
    assert c["has_series"] and len(c["F"]) > 10
