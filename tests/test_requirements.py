"""Requirements mode (ace/requirements.py): agents get engineering requirements, not boundary conditions.
Everything runs without an API key: the LLM roles use ace.mock."""

import json
import shutil

import pytest

from ace import config, fea
from ace import requirements as R
from ace.checks import reference_setup
from ace.config import TASKS

# Requirements are written for elasticity problems only; run_loop refuses flow tasks in requirements mode.
ELASTIC = [pid for pid in TASKS if not fea.is_flow(fea.get_problem(pid))]


@pytest.fixture
def requirements_mode(monkeypatch):
    monkeypatch.setattr(config, "STATEMENT", "requirements")
    monkeypatch.setenv("ACE_STATEMENT", "requirements")


def test_every_loop_problem_has_clean_current_requirements():
    for pid in ELASTIC:
        assert R.lint(pid, R.load(pid)) == [], pid


def test_numbers_in_reads_units_fractions_and_exponents():
    nums = R.numbers_in("E = 210 GPa, nu 1/3, alpha 2.3e-4 per degC or 2.3 x 10^-4, p = 100 MPa, "
                        "E = 2e+11 Pa, t = 0.1 m, on edge_x0")
    for v in (2.1e11, 1 / 3, 2.3e-4, 1e8, 2e11, 0.1):
        assert any(abs(n - v) <= 1e-9 * v for n in nums), v
    assert 0.0 not in nums                     # the 0 of the identifier edge_x0 is not a number


def test_lint_catches_bc_language_benchmark_names_group_names_and_missing_numbers():
    good = R.load("nafems_le1")
    bad = dict(good, requirements=good["requirements"].replace("10 MPa", "a uniform tension")
               + "\nEdge AB has zero x-displacement and the rim is fixed (see NAFEMS LE1); load edge_BC.")
    issues = "\n".join(R.lint("nafems_le1", bad))
    for expected in ("prescribes a displacement", "support-type word", "names the benchmark",
                     "names the CAD group 'edge_BC'", "missing: pressure value on edge_BC"):
        assert expected in issues, expected


def test_lint_requires_neutral_aliases_for_telltale_group_names():
    good = R.load("cooks_membrane")
    assert "gives its support or load away" in "\n".join(R.lint("cooks_membrane", dict(good, group_aliases={})))
    issues = "\n".join(R.lint("cooks_membrane", dict(good, group_aliases={"clamped": "fixed_edge", "loaded": "top"})))
    assert "still gives a support or load away" in issues and "name of another group" in issues


def test_aliases_rename_groups_everywhere_without_changing_the_physics(requirements_mode, tmp_path):
    p = fea.get_problem("cooks_membrane")
    names = {g["name"] for g in fea.geometry_summary(p)["groups"]}
    assert {"left_edge", "right_edge"} <= names and not names & {"clamped", "loaded"}
    assert {bc["group"] for bc in reference_setup(p)["bcs"]} == {"left_edge", "right_edge"}
    raw = R.raw_problem("cooks_membrane")
    a = fea.run_solve(raw, fea.make_mesh(raw, tmp_path / "a.msh", h=4.0, order=2), reference_setup(raw), "uy_C")
    b = fea.run_solve(p, fea.make_mesh(p, tmp_path / "b.msh", h=4.0, order=2), reference_setup(p), "uy_C")
    assert a["ok"] and b["ok"] and a["value"] == b["value"]


def test_task_prompt_gives_requirements_and_hides_the_problem_id(requirements_mode):
    from ace.agent import task_prompt
    for pid in ELASTIC:
        prompt = task_prompt(pid)
        assert R.agent_text(pid) in prompt and R.label_for(pid) in prompt
        assert pid not in prompt
        statement = R.raw_problem(pid).description()     # its support lines: "Supports: ..." or "Clamped ..."
        supports = [ln for ln in statement.splitlines() if ln.startswith(("Supports:", "Clamped", "Roller"))]
        assert supports and not any(ln in prompt for ln in supports), pid


def test_task_prompt_default_mode_is_the_statement():
    from ace.agent import task_prompt
    assert config.STATEMENT == "bcs"
    prompt = task_prompt("cooks_membrane")
    assert prompt.startswith("Problem id: cooks_membrane") and R.raw_problem("cooks_membrane").description() in prompt


def test_equivalence_accepts_rewritten_setups_and_rejects_wrong_ones(tmp_path):
    p = R.view_problem("cooks_membrane", R.aliases_for("cooks_membrane"))
    mesh = R.round_trip_mesh(p, tmp_path)
    ref = reference_setup(p)
    assert R.equivalence("cooks_membrane", p, ref, mesh)["equivalent"]
    # the same load written as force per area: equivalent, though the setup reads differently
    as_traction = json.loads(json.dumps(ref))
    as_traction["bcs"] = [bc if bc["type"] != "force" else {"type": "traction", "group": bc["group"], "value": [0.0, 1 / 16]}
                          for bc in as_traction["bcs"]]
    res = R.equivalence("cooks_membrane", p, as_traction, mesh)
    assert res["equivalent"] and res["setup_report"]["unexpected"] == ["traction on right_edge"]
    flipped = json.loads(json.dumps(ref))
    for bc in flipped["bcs"]:
        if bc["type"] == "force":
            bc["value"] = [0.0, -1.0]
    res = R.equivalence("cooks_membrane", p, flipped, mesh)
    assert not res["equivalent"] and res["setup_report"]["sign_flips"]
    unsupported = dict(ref, bcs=[bc for bc in ref["bcs"] if bc["type"] != "fixed"])
    assert not R.equivalence("cooks_membrane", p, unsupported, mesh)["equivalent"]


def test_generate_mock_round_trip(tmp_path, monkeypatch):
    shutil.copy(R.path_for("cooks_membrane"), tmp_path / "cooks_membrane.json")
    monkeypatch.setattr(R, "STORE", tmp_path)
    assert R.main(["generate", "--problems", "cooks_membrane", "--mock", "--trials", "1", "--force"]) == 0
    data = json.loads((tmp_path / "cooks_membrane.json").read_text())
    assert data["validation"]["lint"] == [] and data["validation"]["round_trip"]["recovered"] == 1
    assert data["source"]["fingerprint"] == R.fingerprint("cooks_membrane")


def test_loop_refuses_problems_without_requirements(tmp_path, monkeypatch):
    from ace.run_loop import main
    monkeypatch.setattr(config, "STATEMENT", "bcs")
    monkeypatch.setenv("ACE_STATEMENT", "bcs")
    monkeypatch.setattr(R, "STORE", tmp_path / "empty")
    with pytest.raises(SystemExit, match="no requirements file"):
        main(["--mock", "--statement", "requirements", "--memory", "shared", "--problems", "cooks_membrane",
              "--holdout", "", "--generations", "1", "--out", str(tmp_path / "run")])


def test_loop_end_to_end_in_requirements_mode_mock(tmp_path, monkeypatch):
    from ace.run_loop import main
    monkeypatch.setattr(config, "STATEMENT", "bcs")       # restored after the test (main sets both)
    monkeypatch.setenv("ACE_STATEMENT", "bcs")
    main(["--mock", "--statement", "requirements", "--memory", "shared", "--problems", "cooks_membrane",
          "--holdout", "nafems_le1", "--runs-per-problem", "1", "--generations", "1", "--concurrency", "1",
          "--out", str(tmp_path)])
    record = json.loads((tmp_path / "gen0" / "runs" / "g0-cooks_membrane-0.json").read_text())
    groups = {bc["group"] for s in record["solves"] for bc in s["setup"]["bcs"]}
    assert groups == {"left_edge", "right_edge"}              # the agent only ever saw the neutral names
    assert record["checks"]["setup"]["missing"] == [] and "score" in record["checks"]
    assert "engineering requirements" in (tmp_path / "summary.md").read_text()


def test_inspect_geometry_leaves_supports_and_loads_to_the_agent(monkeypatch):
    from ace import geometry_info
    hints = geometry_info.describe(fea.get_problem("cooks_membrane"))["hints"]
    assert "Supported: clamped" in hints                      # bcs mode: read off the statement
    monkeypatch.setattr(config, "STATEMENT", "requirements")
    info = geometry_info.describe(fea.get_problem("cooks_membrane"))
    assert "Supported" not in info["hints"] and "Loaded" not in info["hints"]
    assert not info["support_boundary"] and not info["load_boundary"]
