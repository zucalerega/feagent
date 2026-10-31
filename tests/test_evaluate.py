"""End-to-end: reference pipeline passes; deliberately broken submissions are caught by the right gate."""

import json
import re
import shutil

import pytest

from feabench.cases import export_task, list_cases, load_reference
from feabench.evaluate import evaluate_submission
from feabench.paths import VERIFICATION_DIR

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")
CASE = "verification/lame_cylinder"


@pytest.fixture(scope="module")
def good_deck(tmp_path_factory):
    from feabench.suites.verification import run_reference

    out = tmp_path_factory.mktemp("ref")
    rep = run_reference("lame_cylinder", 2, out)
    return rep, (out / "lame_cylinder_L2" / "model.inp").read_text()


def _submit(tmp_path, text, name="sub"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.inp").write_text(text)
    return evaluate_submission(CASE, d)


def test_cases_are_listed_and_export_hides_references(tmp_path):
    cases = list_cases("verification")
    assert CASE in cases and len(cases) >= 10
    dest = export_task(CASE, tmp_path / "agent")
    names = {p.name for p in dest.rglob("*")}
    assert {"task.json", "task.md", "geometry.step"} <= names
    assert not any("reference" in n for n in names)
    for case in list_cases("verification"):  # no reference value leaks into an agent's files
        exported = export_task(case, tmp_path / case.replace("/", "_"))
        text = "".join((exported / f).read_text() for f in ("task.json", "task.md"))
        for q in load_reference(case)["qois"]:
            r = q["reference"]
            if r == round(r):  # round numbers (20 MPa, -100000 N) also occur as legitimate inputs
                continue
            for fmt in (".4g", ".5g", ".6g"):
                assert f"{r:{fmt}}" not in text, (case, q["id"])


def test_reference_pipeline_passes(good_deck):
    rep, _ = good_deck
    s = rep["summary"]
    assert s["gates_passed"] and s["all_qois_within_tolerance"] and s["score"] == pytest.approx(1.0)
    assert rep["gates"]["material_properties"]


def test_unit_error_in_modulus_fails_material_gate(tmp_path, good_deck):
    _, deck = good_deck
    bad = deck.replace("*ELASTIC\n210000, 0.3", "*ELASTIC\n2.1e+11, 0.3")
    assert bad != deck
    rep = _submit(tmp_path, bad)
    assert rep["gates"]["material_properties"] is False and rep["summary"]["score"] == 0.0
    assert any("task specifies 210000" in p for p in rep["materials"]["problems"])


def test_wrong_pressure_magnitude_fails_load_path_gate(tmp_path, good_deck):
    _, deck = good_deck
    bad = re.sub(r"(, P\d), 100\b", r"\1, 10", deck)
    assert bad != deck
    rep = _submit(tmp_path, bad)
    gates = rep["summary"]["gates"]
    assert gates["qoi:reaction_x"] is False and rep["summary"]["score"] == 0.0
    assert rep["summary"]["accuracy"] == pytest.approx(0.0)  # 90 % errors give no credit


def test_missing_symmetry_bc_is_detected(tmp_path, good_deck):
    _, deck = good_deck
    bad = deck.replace("N_SYM_X, 1, 1, 0.0\n", "")
    assert bad != deck
    rep = _submit(tmp_path, bad)
    # without ux=0 on x=0 the model is singular or the reaction vanishes: either way it fails
    assert rep["summary"]["score"] == 0.0


def test_scaled_geometry_fails_volume_gate(tmp_path, good_deck):
    _, deck = good_deck
    lines, out, in_nodes = deck.splitlines(), [], False
    for ln in lines:  # mm -> m mistake
        if ln.upper().startswith("*NODE,") or ln.upper() == "*NODE":
            in_nodes = True
        elif ln.startswith("*"):
            in_nodes = False
        elif in_nodes:
            i, x, y, z = ln.split(",")
            ln = f"{i}, {float(x) / 1000:.12g}, {float(y) / 1000:.12g}, {float(z) / 1000:.12g}"
        out.append(ln)
    rep = _submit(tmp_path, "\n".join(out) + "\n")
    assert rep["gates"].get("geometry_volume") is False and rep["summary"]["score"] == 0.0


def test_all_reference_values_present():
    for d in sorted(VERIFICATION_DIR.glob("*/reference/reference.json")):
        ref = json.loads(d.read_text())
        missing = [q["id"] for q in ref["qois"] if q.get("reference") is None]
        assert not missing, f"{d.parent.parent.name}: numerical references not computed: {missing}"
