"""Physical test data in the loop (ace/evidence.py): a machine export becomes a measured compliance
that every run is compared with, that agents and teachers read, and that can replace the reference
in the score. Everything runs without an API key (mock LLM)."""

import csv
import json
import math

import numpy as np
import pytest

from ace import config
from ace import evidence as EV
from ace import fea


def write_export(path, compliance, cycles=3, peak=300.0, noise=2e-7, seating=10e-6, disp_unit="mm",
                 load_unit="N", delim=",", decimal_comma=False, units_in_header=False):
    """A Bluehill-like export: a title block, a header row, a units row, then Time, Load, Extension.
    The first loading has a seating toe (grips settling); later cycles start from the set it leaves."""
    rng = np.random.default_rng(0)
    rows, t = [], 0.0
    for c in range(cycles):
        for L in np.concatenate([np.linspace(0, peak, 80), np.linspace(peak, 0, 80)[1:]]):
            toe = seating * (1 - math.exp(-L / 20)) if c == 0 else seating
            d = compliance * L + toe + rng.normal(0, noise)
            t += 0.5
            rows.append((t, L / {"N": 1, "kN": 1e3}[load_unit], d / {"mm": 1e-3, "m": 1.0}[disp_unit]))
    fmt = (lambda v: f"{v:.6f}".replace(".", ",")) if decimal_comma else (lambda v: f"{v:.6f}")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=delim, quoting=csv.QUOTE_ALL)
        w.writerow(["Specimen label", "printed part 1"])
        w.writerow([])
        if units_in_header:
            w.writerow(["Time (s)", f"Load ({load_unit})", f"Extension (Video) ({disp_unit})"])
        else:
            w.writerow(["Time", "Load", "Extension (Video)"])
            w.writerow(["(s)", f"({load_unit})", f"({disp_unit})"])
        for r in rows:
            w.writerow([fmt(v) for v in r])
    return path


def spec_file(tmp_path, problem_id, data_name, **extra):
    spec = {"test_id": "t1", "problem_id": problem_id, "description": "printed part, wedge grips",
            "data": {"file": data_name, "load": "Load", "displacement": "Extension (Video)"}, **extra}
    p = tmp_path / "t1.json"
    p.write_text(json.dumps(spec))
    return p


C_TRUE = 1.578e-7     # m per N: what the pull test of abc_00000003_pull should show


def test_reads_an_export_and_fits_the_cycles(tmp_path):
    write_export(tmp_path / "raw.csv", C_TRUE)
    t = EV.process_test(EV.read_spec(spec_file(tmp_path, "abc_00000003_pull", "raw.csv")))
    assert abs(t["compliance"] / C_TRUE - 1) < 0.005
    assert t["cycles_dropped"] == [1] and t["cycles_used"] == [2, 3]          # the seating toe is left out
    assert t["nominal_load"] == 300.0 and abs(t["value_at_nominal_load"] / (C_TRUE * 300) - 1) < 0.005
    assert 0.06 < t["u_rel_total"] < 0.09          # instrument 1 %, material 5 % (no modulus given), geometry 5 %
    assert t["measurement"]["name"] == "ext_marks" and not t["nonlinearity_significant"]


@pytest.mark.parametrize("kw", [dict(delim=";", decimal_comma=True, load_unit="kN", units_in_header=True),
                                dict(disp_unit="m", delim="\t")])
def test_reads_other_export_formats(tmp_path, kw):
    write_export(tmp_path / "raw.csv", C_TRUE, **kw)
    t = EV.process_test(EV.read_spec(spec_file(tmp_path, "abc_00000003_pull", "raw.csv")))
    assert abs(t["compliance"] / C_TRUE - 1) < 0.005


def test_strain_channel_with_a_gauge_length(tmp_path):
    write_export(tmp_path / "raw.csv", C_TRUE / 0.054, disp_unit="m")      # "extension" column holds strain
    spec = spec_file(tmp_path, "abc_00000003_pull", "raw.csv")
    s = json.loads(spec.read_text())
    s["data"] = {"file": "raw.csv", "load": "Load", "strain": "Extension (Video)", "strain_unit": "",
                 "gauge_length_mm": 54}
    spec.write_text(json.dumps(s))
    assert abs(EV.process_test(EV.read_spec(spec))["compliance"] / C_TRUE - 1) < 0.005


def test_measured_modulus_rescales_predictions():
    ev = {"compliance": 2.0, "nominal_load": 10.0, "u_rel_total": 0.05, "e_scale": 10e9 / 8e9}
    c = EV.compare_value(16.0, ev)          # 1.6 per N with E = 10 GPa -> 2.0 per N with the measured 8 GPa
    assert abs(c["predicted_compliance"] - 2.0) < 1e-12 and c["consistent"]
    assert not EV.compare_value(10.0, ev)["consistent"]      # 1.25 per N: -37.5 %, 7.5 sigma


def test_pull_test_problem_builds_and_solves(tmp_path):
    from ace.checks import reference_setup
    from ace.field_error import _solve
    from feacore.case import QoI
    p = fea.get_problem("abc_00000003_pull")
    names = {g["name"] for g in fea.geometry_summary(p)["groups"]}
    assert {"stub_neg_grip", "stub_pos_grip", "stub_neg_end", "stub_pos_end", "solid"} <= names
    mesh = fea.make_mesh(p, tmp_path / "m.msh", h=0.005, order=2)
    res = _solve(p, mesh, reference_setup(p), "ext_marks")
    ref = fea.reference_value(p, "ext_marks")
    assert abs(res.evaluate_all()["ext_marks"] / ref - 1) < 0.12           # coarse mesh: about -9 %
    reaction = res.evaluate_qoi(QoI(name="r", type="reaction", group="stub_neg_grip", component=0))
    assert abs(reaction + 300.0) < 1e-6 * 300


def test_loop_uses_test_data_that_arrives_mid_run(tmp_path, monkeypatch):
    """Generation 0 runs without data; the test arrives before generation 1 (after the population
    has settled). The settled consensus is compared with it, earlier runs are re-scored against it,
    and generation 1's agents get a test_check in every solve."""
    from ace.run_loop import main
    for k, v in (("ACE_STATEMENT", "bcs"), (EV.ACTIVE_ENV, ""), (EV.SCORE_ENV, "0")):
        monkeypatch.setenv(k, v)                      # main sets them; restored after the test
    monkeypatch.setattr(config, "STATEMENT", "bcs")
    data_dir = tmp_path / "lab"
    data_dir.mkdir()
    # Cook's membrane: uy_C = 23.96 per unit load in the FEA; the "real panel" is 10 % softer.
    write_export(data_dir / "raw.csv", 1.10 * 23.96, peak=1.0, noise=1e-3, seating=0.0, disp_unit="m")
    spec = spec_file(data_dir, "cooks_membrane", "raw.csv", model_load=1.0)
    out = tmp_path / "run"
    original = EV.refresh

    def refresh(out_dir, sources, all_runs, generation):    # the lab sends its file during generation 0
        if generation == 1:
            (out_dir / "evidence" / "lab").mkdir(parents=True, exist_ok=True)
            for f in data_dir.iterdir():
                (out_dir / "evidence" / "lab" / f.name).write_bytes(f.read_bytes())
        return original(out_dir, sources, all_runs, generation)

    monkeypatch.setattr(EV, "refresh", refresh)
    main(["--mock", "--memory", "shared", "--problems", "cooks_membrane", "--holdout", "", "--runs-per-problem", "2",
          "--generations", "2", "--concurrency", "1", "--evidence-score", "--out", str(out)])
    active = json.loads((out / "evidence_active" / "cooks_membrane.json").read_text())
    assert active["since_generation"] == 1
    cons = active["consensus"]
    assert cons["generation"] == 0 and -0.15 < cons["rel_err"] < -0.03     # settled FEA vs the softer part
    g0 = json.loads((out / "gen0" / "runs" / "g0-cooks_membrane-0.json").read_text())
    assert g0["checks"]["evidence"]["rel_err"] < 0 and "reference_rel_error" in g0["checks"]
    g1 = json.loads((out / "gen1" / "runs" / "g1-cooks_membrane-0.json").read_text())
    assert any("test_check" in s for s in g1["solves"]) and g1["checks"]["evidence"] is not None
    assert g1["checks"]["tolerance"] >= 2 * active["combined"]["u_rel_total"]   # scored against the test
    monkeypatch.setenv(EV.ACTIVE_ENV, str(out / "evidence_active"))
    section = EV.teacher_section("cooks_membrane", [g0, g1])
    assert "PHYSICAL TEST DATA" in section and "Each run's prediction against the test" in section
    assert spec.exists()


def test_summarize_cli(tmp_path, capsys):
    write_export(tmp_path / "raw.csv", C_TRUE)
    assert EV.main(["summarize", str(spec_file(tmp_path, "abc_00000003_pull", "raw.csv"))]) == 0
    assert '"compliance"' in capsys.readouterr().out
