"""Tests of dossier.system: the system model check and the system diagram."""

import copy
import re
import xml.etree.ElementTree as ET

import pytest

from dossier import context
from dossier.model import Parameter, Value
from dossier.system import check as syscheck
from dossier.system import diagram

SVG = "{http://www.w3.org/2000/svg}"


def _run(project):
    ctx = context.build(project)
    return syscheck.check(project, ctx, None)


def _ids(journal):
    return {f.id for f in journal.findings}


def _copy(example):
    return copy.deepcopy(example)


@pytest.fixture(scope="module")
def example_journal(example, example_ctx):
    return syscheck.check(example, example_ctx, None)


# ----------------------------------------------------------------------------- the example

def test_example_assumptions(example, example_journal):
    ids = {a.id for a in example_journal.assumptions}
    assumed = {n for n, p in example.system.parameters.items() if p.basis == "assumed"}
    assert ids == {f"ASM-{n.upper().replace('_', '-')}-ASSUMED" for n in assumed}
    assert "ASM-DP-VALVE-EST-ASSUMED" in ids
    one = next(a for a in example_journal.assumptions if a.id == "ASM-DP-VALVE-EST-ASSUMED")
    assert "EF-OUT-P-FIRE" in one.impact


def test_example_findings(example_journal):
    by_id = {f.id: f for f in example_journal.findings}
    assert "FND-H-PURGE-NO-JUSTIFICATION" in by_id
    assert by_id["FND-H-PURGE-NO-JUSTIFICATION"].severity == "major"
    for ph in ("PROOF", "BURST", "LEAK"):
        assert by_id[f"FND-PH-{ph}-FREE"].severity == "info"
    majors = {i for i, f in by_id.items() if f.severity in ("major", "error")}
    assert majors == {"FND-H-PURGE-NO-JUSTIFICATION"}


def test_example_check_is_deterministic(example, example_ctx, example_journal):
    again = syscheck.check(example, example_ctx, None)
    assert [f.id for f in again.findings] == [f.id for f in example_journal.findings]
    assert [a.id for a in again.assumptions] == [a.id for a in example_journal.assumptions]


# ----------------------------------------------------------------------------- triggered checks

def test_assumed_without_note(example):
    p = _copy(example)
    p.system.parameters["h_chill"].note = ""
    assert "FND-H-CHILL-NO-JUSTIFICATION" in _ids(_run(p))


def test_missing_source(example):
    p = _copy(example)
    name = next(n for n, v in p.system.parameters.items() if v.basis in ("specified", "measured", "reference") and v.source)
    p.system.parameters[name].source = ""
    j = _run(p)
    hit = [f for f in j.findings if f.rule == "system.parameter_source" and name in f.statement]
    assert hit and hit[0].severity == "minor"


def test_unused_parameter(example):
    p = _copy(example)
    p.system.parameters["zz_nothing_reads_this"] = Parameter(value=1.0, unit="m", basis="specified", source="test")
    j = _run(p)
    hit = [f for f in j.findings if f.rule == "system.parameter_unused" and "zz_nothing_reads_this" in f.statement]
    assert hit and hit[0].severity == "info"


def test_fluid_temperature_without_h(example):
    p = _copy(example)
    eff = next(e for c in p.system.connections for e in c.effects if e.kind == "fluid_temperature" and e.h is not None)
    eff.h = None
    f = next(f for f in _run(p).findings if f.rule == "system.heat_transfer" and f.id.startswith("FND-" + eff.id))
    assert f.severity == "major"


def test_zero_vector_effect(example):
    p = _copy(example)
    eff = next(e for c in p.system.connections for e in c.effects if e.kind == "force")
    eff.value = Value(value=[0.0, 0.0, 0.0], unit="N")
    f = next(f for f in _run(p).findings if f.rule == "system.effect_value" and eff.id in f.id)
    assert f.severity == "minor"


def test_negative_pressure(example):
    p = _copy(example)
    eff = next(e for c in p.system.connections for e in c.effects if e.kind == "pressure")
    eff.value = Value(value=-5.0, unit="MPa")
    f = next(f for f in _run(p).findings if f.rule == "system.effect_value" and eff.id in f.id)
    assert f.severity == "major"


def test_phase_without_effect(example):
    p = _copy(example)
    from dossier.model import Phase
    p.system.phases.append(Phase(id="PH-IDLE", name="Idle phase", category="storage"))
    f = next(f for f in _run(p).findings if f.id.startswith("FND-PH-IDLE"))
    assert f.severity == "info"


def test_no_support_in_operation(example):
    p = _copy(example)
    for c in p.system.connections:
        c.effects = [e for e in c.effects if e.kind != "support"]
    f = [f for f in _run(p).findings if f.rule == "system.subject_support"]
    assert f and f[0].severity == "major"


def test_duty_cycle_count_not_positive(example):
    p = _copy(example)
    if not p.system.duty_cycles:
        pytest.skip("the example has no duty cycle")
    p.system.duty_cycles[0].count.value = 0.0
    p.system.duty_cycles[0].count.expr = None
    assert [f for f in _run(p).findings if f.rule == "system.duty_cycle"]


def test_element_without_connection(example):
    p = _copy(example)
    from dossier.model import Element
    p.system.elements.append(Element(id="EL-LONELY", name="Lonely element", kind="equipment", group="Test"))
    f = next(f for f in _run(p).findings if f.rule == "system.element_connection")
    assert f.severity == "minor" and "EL-LONELY" in f.statement


def test_effect_without_interface(example):
    p = _copy(example)
    comp = p.subject
    comp.interfaces = [i for i in comp.interfaces if i.connection != next(
        c.id for c in p.system.connections if any(e.kind == "pressure" for e in c.effects) and c.to == "EL-MOV" or c.from_ == "EL-MOV")]
    f = [f for f in _run(p).findings if f.rule == "system.effect_interface"]
    assert f and all(x.severity == "major" for x in f)


# ----------------------------------------------------------------------------- the diagram

def test_svg_parses_with_a_group_for_every_record(example, example_ctx):
    svg = diagram.render(example, example_ctx)
    root = ET.fromstring(svg)
    assert root.tag == SVG + "svg"
    assert root.get("viewBox") and root.get("width") is None and root.get("height") is None
    groups = {g.get("id"): g for g in root.iter(SVG + "g") if g.get("id")}
    for el in example.system.elements:
        assert el.id in groups and "dz-element" in groups[el.id].get("class")
        assert groups[el.id].get("data-kind") == el.kind
        assert groups[el.id].find(SVG + "title") is not None
    for c in example.system.connections:
        assert c.id in groups and "dz-connection" in groups[c.id].get("class")
        assert groups[c.id].get("data-kind") == c.kind
        assert groups[c.id].find(SVG + "title") is not None
    subject = groups[next(e.id for e in example.system.elements if e.component == example.file.subject)]
    assert "dz-subject" in subject.get("class")
    assert "var(--dz-" in svg
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")


def test_svg_is_deterministic(example, example_ctx):
    assert diagram.render(example, example_ctx) == diagram.render(example, example_ctx)


def _overlap(a, b):
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


def test_no_overlapping_boxes(example, example_ctx):
    lay = diagram.layout(example, example_ctx)
    assert len(lay.boxes) == len(example.system.elements)
    boxes = list(lay.boxes.values())
    for i, a in enumerate(boxes):
        assert a.x >= 0 and a.y >= 0 and a.x + a.w <= lay.width and a.y + a.h <= lay.height
        for b in boxes[i + 1:]:
            assert not _overlap(a, b), (a.id, b.id)
    assert all(e.label_ok for e in lay.edges)


def test_layout_of_a_large_system(example):
    from dossier.model import Connection, Element
    p = _copy(example)
    ids = [e.id for e in p.system.elements]
    for i in range(24):
        eid = f"EL-X{i}"
        p.system.elements.append(Element(id=eid, name=f"Extra element {i}", kind="equipment", group=f"Group {i % 4}"))
        p.system.connections.append(Connection.model_validate(
            {"id": f"CN-X{i}", "name": f"Extra {i}", "from": ids[i % len(ids)], "to": eid, "kind": "fluid", "medium": "LOX"}))
    ctx = context.build(p)
    lay = diagram.layout(p, ctx)
    assert len(lay.boxes) >= 35
    boxes = list(lay.boxes.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not _overlap(a, b), (a.id, b.id)
    ET.fromstring(diagram.render(p, ctx))


# ----------------------------------------------------------------------------- the effects table

def test_effects_table_counts(example, example_ctx):
    rows = diagram.effects_table(example, example_ctx)
    subject = example.subject
    expected = 0
    for c in example.system.connections:
        for e in c.effects:
            acts = e.acts_on or c.to
            if acts != next(el.id for el in example.system.elements if el.component == subject.id):
                continue
            expected += len(e.phase) if isinstance(e.phase, list) else 1
    assert len(rows) == expected > 0
    for r in rows:
        for key in ("phase", "phase_name", "connection", "interface", "kind", "display", "expr", "basis", "source", "effect"):
            assert key in r
    assert len({(r["effect"], r["phase"]) for r in rows}) == len(rows)
    assert {r["phase"] for r in rows} <= {p.id for p in example.system.phases}


def test_effects_table_matches_diagram_counts(example, example_ctx):
    rows = diagram.effects_table(example, example_ctx)
    root = ET.fromstring(diagram.render(example, example_ctx))
    total = 0
    for g in root.iter(SVG + "g"):
        if "dz-connection" in (g.get("class") or "") and g.get("data-effects"):
            total += sum(int(x.split(":")[1]) for x in g.get("data-effects").split())
    assert total == len(rows)
