"""The core of dossier: units, the data model, the project store and the resolved context."""

import math

import pytest

from dossier import context, model, units
from dossier.project import Project, ProjectError, make_id
from dossier.records import Journal


def test_units_parse_convert_and_display():
    q = units.parse("10.4 MPa")
    assert units.si(q) == pytest.approx(10.4e6) and units.si_unit(q) == "Pa"
    assert units.display(units.parse("0.025 m")) == "25 mm"
    assert units.si(units.parse("20 degC")) == pytest.approx(293.15)
    assert units.si(units.parse("8 g0")) == pytest.approx(78.4532)
    assert units.si_unit(units.parse("90 N*m")) == "N*m" and units.si_unit(units.parse("2 kJ")) == "J"


def test_a_degree_in_a_derived_unit_is_a_temperature_difference():
    # pint reads "1/degC" as one over an absolute temperature, which is wrong by a factor of 274
    assert units.si(units.parse("1.3e-5 1/degC")) == pytest.approx(1.3e-5)
    assert units.si(units.parse("16 um/m/degC")) == pytest.approx(16e-6)
    assert units.si(units.parse("15 W/(m*degC)")) == pytest.approx(15.0)
    assert units.si(units.parse("12 W/(m^2*degC)")) == pytest.approx(12.0)
    assert units.si(units.parse("7.2e-6 1/degF")) == pytest.approx(7.2e-6 * 1.8)
    assert units.si(units.parse("20 degC")) == pytest.approx(293.15)          # a plain temperature stays absolute
    assert units.si(units.parse("13e-6 1/K")) == pytest.approx(13e-6)


def test_expressions_check_dimensions_and_reject_code():
    n = {"p": units.parse("10 MPa"), "l": units.parse("2 m"), "f": units.parse({"value": [0, 3, 4], "unit": "N"})}
    assert units.si(units.evaluate("p * pi * (25 * mm)**2 / 4", n)) == pytest.approx(10e6 * math.pi * 0.025**2 / 4)
    assert units.si(units.evaluate("norm(f)", n)) == pytest.approx(5.0)
    assert units.si(units.evaluate("f[1] * l", n)) == pytest.approx(6.0)
    for bad in ("p + l", "unknown_name + 1", "__import__('os')", "p.real", "exp(p)"):
        with pytest.raises(units.UnitsError):
            units.evaluate(bad, n)


def test_value_reads_literals_and_expressions():
    assert model.Value.model_validate("10.4 MPa").quantity().to("Pa").magnitude == pytest.approx(10.4e6)
    v = model.Value.model_validate({"expr": ["-a * b", 0, 0], "unit": "N"})
    q = v.quantity({"a": units.parse("2 MPa"), "b": units.parse("100 mm^2")})
    assert units.si(q) == pytest.approx([-200.0, 0.0, 0.0]) and v.names() == {"a", "b"}
    with pytest.raises(ValueError):
        model.Value.model_validate({"value": 1.0, "expr": "x"})
    with pytest.raises(units.UnitsError):                      # a pressure cannot be shown in newtons
        model.Value.model_validate({"expr": "a", "unit": "N"}).quantity({"a": units.parse("2 MPa")})


def test_example_loads_and_resolves(example, example_ctx):
    ctx = example_ctx
    assert example.subject.id == "CMP-MOV-BODY" and len(example.reqs()) == 10
    # derived numbers: velocity, closure surge, the largest inlet pressure
    assert ctx.params["v_lox"].si == pytest.approx(5.3 / (1141 * math.pi * 0.025**2 / 4))
    assert ctx.params["dp_surge"].si == pytest.approx(1141 * 1.5 * ctx.params["v_lox"].si / 0.05)
    assert ctx.params["p_max_inlet"].si == pytest.approx(10.4e6 + ctx.params["dp_surge"].si)
    assert ctx.params["p_max_inlet"].depends_on == ["dp_surge", "p_pump_max"]
    # requirement parameters and material properties are names too
    assert ctx.params["meop"].origin == "REQ-MOV-001" and ctx.params["Sy"].origin == "MAT-316L"
    # the planted defect: the requirement's MEOP is under the largest operating pressure
    ops = [p.id for p in example.system.phases if p.category == "operation"]
    worst, effect = ctx.envelope(example, None, "pressure", ops)
    assert worst > ctx.params["meop"].si and effect.id == "EF-IN-P-SHUTDOWN"


def test_loads_map_effects_to_interfaces(example, example_ctx):
    loads = {x.effect: x for x in example_ctx.loads(example, None, "PH-PROOF")}
    p = loads["EF-IN-P-PROOF"]
    assert p.interface == "INT-INLET" and p.role == "wetted" and "FT-CAVITY-WALL" in p.features
    assert p.si == pytest.approx(1.5 * 10.5e6) and p.unit == "Pa"
    f = loads["EF-ACT-F-PROOF"]
    assert f.interface == "INT-ACT" and f.features == ["FT-PAD-TOP"] and f.si[2] == pytest.approx(15.75e6 * math.pi * 0.008**2)
    fire = {x.effect: x for x in example_ctx.loads(example, None, "PH-FIRE")}
    assert fire["EF-IN-HOLD"].kind == "support" and fire["EF-ENV-T"].features == ["*"] and fire["EF-ENV-T"].h_si == 12.0
    fn = example_ctx.functions(list(loads.values()))
    assert units.si(fn["load"]("INT-INLET", "pressure")) == pytest.approx(15.75e6)
    with pytest.raises(units.UnitsError):
        fn["load"]("INT-INLET", "moment")


def test_project_errors_name_the_file_and_the_field(tmp_path, workproject):
    sysfile = workproject.root / "system.yaml"
    sysfile.write_text(sysfile.read_text().replace("p_pump_max:     {value: 10.4 MPa", "p_pump_max:     {value: 10.4 MPa, bogus: 1"))
    with pytest.raises(ProjectError) as e:
        Project.load(workproject.root)
    assert "system.yaml" in str(e.value) and "bogus" in str(e.value)


def test_unit_error_in_an_effect_is_reported(workproject):
    sysfile = workproject.root / "system.yaml"
    sysfile.write_text(sysfile.read_text().replace("value: {expr: p_deadhead}", "value: {expr: l_line}", 1))
    with pytest.raises(ProjectError) as e:
        context.build(Project.load(workproject.root))
    assert "EF-IN-P-PRESTART" in str(e.value) and "pressure must be in Pa" in str(e.value)


def test_journal_ids_are_stable_and_decisions_need_a_reason():
    assert make_id("DEC", "LC-FIRE", "support") == "DEC-LC-FIRE-SUPPORT"
    j = Journal("plan", "rule:test")
    d = j.decide("LC-FIRE", "support", "Support", "Hold the inlet flange.", "The pump holds the valve.",
                 alternatives=[("Hold the bolt holes.", "The flange face carries the load.")])
    assert d.id == "DEC-LC-FIRE-SUPPORT" and d.origin.by == "rule:test" and d.refs()[-1].to == "LC-FIRE"
    with pytest.raises(ValueError):
        j.decide("LC-FIRE", "x", "t", "c", "  ")


def test_stage_files_and_reviews(workproject):
    j = Journal("plan", "rule:test")
    d = j.decide("LC-FIRE", "support", "Support", "Hold the inlet flange.", "The pump holds the valve.")
    workproject.write_stage("plan", decisions=j.decisions)
    assert [x.id for x in workproject.collection("decisions")] == [d.id]
    rv = workproject.set_review(d.id, "accepted", "A. Engineer", "Checked against the ICD.")
    assert workproject.review_of(d)[0].state == "accepted" and not workproject.review_of(d)[1]
    d2 = d.model_copy(update={"choice": "Hold the bolt holes."})
    assert workproject.review_of(d2) == (rv, True)          # the record changed after the review


# ---- python -m dossier ui: thin start of the dashboard (no server is started here)

def test_ui_subcommand_builds_the_dashboard_call(monkeypatch, capsys):
    import feacore.ui.__main__ as dash
    from dossier import cli

    calls = []
    monkeypatch.setattr(dash, "main", lambda argv=None: calls.append(list(argv)))
    assert cli.main(["ui"]) == 0
    assert calls[-1] == ["--host", "127.0.0.1", "--port", "8000"]
    out = capsys.readouterr().out
    assert "http://localhost:8000/#/projects" in out and "Ctrl+C" in out
    assert cli.main(["ui", "--port", "8123", "--runs-dir", "/tmp/r", "--host", "0.0.0.0"]) == 0
    assert calls[-1] == ["--host", "0.0.0.0", "--port", "8123", "--runs-dir", "/tmp/r"]
    assert "http://localhost:8123/#/projects" in capsys.readouterr().out


def test_ui_subcommand_stops_quietly_on_ctrl_c(monkeypatch):
    import feacore.ui.__main__ as dash
    from dossier import cli

    def interrupted(argv=None):
        raise KeyboardInterrupt
    monkeypatch.setattr(dash, "main", interrupted)
    assert cli.main(["ui", "--port", "8001"]) == 0
