"""dossier.compliance: margins of safety and verdicts.

Every test builds a small set of stage records by hand (load cases, analyses, hand calculations, findings),
runs `compliance.assess` on a copy of the example project and compares with numbers worked out by hand in
the comments. Example data used: Sy = 170 MPa, Su = 485 MPa (MAT-316L); yield_factor 1.1, ultimate_factor
1.4, life_factor 4 (requirements); policy factor `yield` 1.25, max_uncertainty 5 %.
"""

import json
import math
import re

import pytest

from dossier import compliance, context, pipeline
from dossier.model import (AnalysisRecord, CalcOutput, Check, Comparison, CriterionRef, CycleUse, Finding,
                           HandCalcResult, Link, LoadCase, ReqParameter, ResultValue, Value)
from dossier.project import Project, make_id

MPA = 1e6
PROOF = ("REQ-MOV-002", 0)          # von_mises.membrane_bending@scls <= Sy, factor 1.0
BURST = ("REQ-MOV-003", 0)          # von_mises.membrane@scls <= Su, factor 1.0
OP_YIELD = ("REQ-MOV-004", 0)       # von_mises.peak@part <= Sy, factor yield_factor = 1.1
OP_ULT = ("REQ-MOV-004", 1)         # von_mises.membrane_bending@scls <= Su, factor ultimate_factor = 1.4
LIFE = ("REQ-MOV-006", 0)           # damage.total@part <= 1.0, factor life_factor = 4
TEMP = ("REQ-MOV-007", 0)           # temperature.min@INT-ACT >= t_act_min = 233 K
DROP = ("REQ-MOV-008", 0)           # pressure_drop.total@part <= dp_max = 60 kPa
ROUND = ("REQ-MOV-010", 0)          # radial_displacement.range@FT-SEAT-OUT-WALL <= 20 um
MB = "von_mises.membrane_bending@"
LINES = ("SCL-SHELL", "SCL-NOZZLE-OUT", "SCL-FLOOR", "SCL-ROOF", "SCL-NECK-ROOT")


# --------------------------------------------------------------------------- builders


def case(id, refs, phase=None, analysis_type="static_structural", **kw):
    return LoadCase(id=id, title=id, analysis_type=analysis_type, component="CMP-MOV-BODY", phase=phase,
                    criteria=[CriterionRef(requirement=r, criterion=k) for r, k in refs], **kw)


def res(key, value, unit="Pa", u=0.02, status="converged", **kw):
    kw.setdefault("error_estimated", u is not None)      # a mesh study that converges, unless a test says not
    return ResultValue(key=key, value=value, unit=unit, uncertainty=u, status=status, **kw)


def analysis(lc_id, results=(), status="done", analysis_type="static_structural", **kw):
    """An analysis record. A worst-line result `...@scls` with no line results of its own stands for the same
    result on each of the five lines (the compliance stage assesses each line by itself)."""
    out = {r.key: r for r in results}
    for r in results:
        prefix = r.key[: -len("scls")]
        if r.key.endswith("@scls") and not any(k.startswith(prefix + "SCL-") for k in out):
            out.update({prefix + line: r.model_copy(update={"key": prefix + line}) for line in LINES})
    return AnalysisRecord(id="AN-" + lc_id[3:], load_case=lc_id, analysis_type=analysis_type, status=status,
                          results=out, **kw)


def check(id, passed, kind="equilibrium", statement="The reactions balance the applied loads."):
    return Check(id=id, kind=kind, statement=statement, passed=passed)


def finding(subject, key, severity="major", **kw):
    return Finding(id=make_id("FND", subject, key), title=key, rule="test." + key, severity=severity,
                   category="consistency", subject=subject, statement="The value is too low.",
                   rationale="The system context gives a larger value.", **kw)


def assess(project, load_cases, analyses=(), hand_calcs=(), findings=()):
    """Write the stage files and run the assess engine: (assessment, items by id, verdicts by requirement)."""
    project.write_stage("plan", load_cases=list(load_cases), hand_calcs=list(hand_calcs))
    if findings:
        project.write_stage("validate", findings=list(findings))
    for a in analyses:
        project.write_stage("analyze", project.analysis_dir(a.load_case) / "analysis.json", analyses=[a])
    out = compliance.assess(project, context.build(project))
    return out, {i.id: i for i in out.items}, {v.requirement: v for v in out.verdicts}


def one(project, ref, result, lc_id="LC-X", **kw):
    """One criterion, one load case, one result: the item."""
    out, items, _ = assess(project, [case(lc_id, [ref])], [analysis(lc_id, [result], **kw)])
    assert len(out.items) == 1
    return out.items[0]


# ASD-STE100 rules a machine can check (CONTRACTS.md, "Text").
_BANNED = re.compile(r"\b(should|may|might|could|would|shall|etc|has been|have been|had been)\b|and/or|e\.g\.|i\.e\.|n't\b", re.I)
_PASSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+(not\s+)?\w+ed\b", re.I)
_ING = re.compile(r"\b\w+ing\b", re.I)


def ste_problems(text, sentences=6):
    """Sentences over 25 words, modal verbs, passive forms, -ing forms, paragraphs over six sentences."""
    plain = re.sub(r'"[^"]*"', "QUOTE", text)           # text another engine or a person wrote is quoted as it is
    parts = [s for s in re.split(r"(?<=[.!?])\s+", plain.strip()) if s]
    out = [("paragraph", text)] if len(parts) > sentences else []
    for s in parts:
        if len(s.split()) > 25:
            out.append(("long", s))
        if _BANNED.search(s) or _PASSIVE.search(s):
            out.append(("form", s))
        if [w for w in _ING.findall(s) if w.lower() not in ("bending", "finding")]:     # technical names, not verbs
            out.append(("ing", s))
    return out


def texts(out):
    """Every text the engine wrote: (where, text, the largest number of sentences)."""
    for i in out.items:
        yield i.id, i.statement, 6
        for r in i.reasons:
            yield i.id, r, 2
    for v in out.verdicts:
        yield v.requirement, v.statement, 6
    for d in out.journal.decisions:
        yield d.id, d.choice, 6
        yield d.id, d.rationale, 6
        for a in d.alternatives:
            yield d.id, a.option, 1
            yield d.id, a.why_not, 1
    for a in out.journal.assumptions:
        for t in (a.statement, a.rationale, a.impact, a.verify_by):
            yield a.id, t, 6


# --------------------------------------------------------------------------- the formula


def test_margin_formulas_by_hand():
    ms = compliance.margin_of_safety
    assert ms(71.3e6, 170e6, 1.0, "<=") == pytest.approx(1.384292, abs=1e-6)       # 170 / 71.3 = 2.384292
    assert ms(100e6, 170e6, 1.1, "<=") == pytest.approx(0.545455, abs=1e-6)        # 170 / 110 = 1.545455
    assert ms(0.1, 1.0, 4.0, "<=") == pytest.approx(1.5)                           # 1 / (4 x 0.1) = 2.5
    assert ms(251.3, 233.0, 1.0, ">=") == pytest.approx(0.078541, abs=1e-6)        # 251.3 / 233 = 1.078541
    assert ms(500.0, 200.0, 2.0, ">=") == pytest.approx(0.25)                      # (500 / 2) / 200 = 1.25
    assert ms(170e6, 170e6, 1.0, "<=") == 0.0 and ms(233.0, 233.0, 1.0, ">=") == 0.0
    assert ms(0.0, 1.0, 4.0, "<=") == math.inf and ms(-5.0, 10.0, 1.0, "<=") == math.inf
    assert ms(-5.0, 10.0, 1.0, ">=") == pytest.approx(-1.5)
    for bad in ((1.0, 0.0, 1.0), (1.0, -2.0, 1.0), (1.0, 2.0, 0.0)):
        with pytest.raises(ValueError):
            ms(*bad)


# --------------------------------------------------------------------------- converged results


def test_pass_is_a_converged_result_with_a_positive_conservative_margin(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 71.3 * MPA, u=0.02), "LC-PROOF",
               checks=[check("CHK-EQUILIBRIUM", True)])
    # conservative value 71.3 x 1.02 = 72.726 MPa; MS = 170 / 72.726 - 1 = 1.33754
    assert item.id == "CMPL-REQ-MOV-002-YIELD-LC-PROOF" and item.verdict == "pass"
    assert item.conservative_value == pytest.approx(72.726e6)
    assert item.margin == pytest.approx(1.33754, abs=1e-5)
    assert (item.value, item.unit, item.uncertainty, item.comparator) == (71.3e6, "Pa", 0.02, "<=")
    assert (item.limit, item.limit_source) == (170e6, "yield strength of MAT-316L")
    assert (item.factor, item.factor_source) == (1.0, "criterion yield of REQ-MOV-002")
    assert (item.requirement, item.criterion, item.load_case, item.analysis) == ("REQ-MOV-002", 0, "LC-PROOF", "AN-PROOF")
    assert item.result_key == MB + "SCL-SHELL" and item.origin.by == "engine:compliance" and item.origin.stage == "assess"
    assert item.statement == (
        "The von Mises membrane plus bending stress on SCL-SHELL, the line that governs, is 71.3 MPa "
        "with an uncertainty of 2 %. The allowable is 170 MPa (yield strength of MAT-316L). "
        "The factor of safety is 1.0 (criterion yield of REQ-MOV-002). "
        "The margin of safety is +1.34 with the uncertainty on the unsafe side. The criterion passes.")
    assert item.reasons == ["The result has the status converged.",
                            "The value with its uncertainty on the unsafe side is 72.73 MPa.",
                            "The margin of safety is +1.38 without the uncertainty.",        # 170 / 71.3 - 1 = 1.384
                            "The assessment uses each of the 5 stress classification lines by itself. "
                            "Line SCL-SHELL governs.",
                            "No check of analysis AN-PROOF failed."]
    assert Link(rel="uses", to="Sy", note="allowable") in item.links
    assert {(l.rel, l.to) for l in item.refs()} >= {("verifies", "REQ-MOV-002"), ("uses", "LC-PROOF"), ("evidence", "AN-PROOF")}


def test_a_margin_of_zero_passes(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 170 * MPA, u=0.0))
    assert item.verdict == "pass" and item.margin == pytest.approx(0.0, abs=1e-12)
    assert "The margin of safety is +0.000." in item.statement


def test_fail_needs_a_negative_margin_with_the_uncertainty_on_the_safe_side(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 180 * MPA, u=0.02))
    # unsafe side: 180 x 1.02 = 183.6 MPa, MS = 170 / 183.6 - 1 = -0.07407
    # safe side:   180 x 0.98 = 176.4 MPa, MS = 170 / 176.4 - 1 = -0.03628, still negative: fail
    assert item.verdict == "fail" and item.margin == pytest.approx(-0.07407, abs=1e-5)
    assert item.conservative_value == pytest.approx(183.6e6)
    assert "The margin of safety is -0.036 with the uncertainty on the safe side." in item.reasons
    assert item.statement.endswith("The margin of safety is -0.074 with the uncertainty on the unsafe side. The criterion fails.")


def test_inconclusive_when_the_margin_is_smaller_than_the_uncertainty(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 168 * MPA, u=0.02))
    # value:       MS = 170 / 168 - 1 = +0.01190
    # unsafe side: 168 x 1.02 = 171.36 MPa, MS = 170 / 171.36 - 1 = -0.00794
    # safe side:   168 x 0.98 = 164.64 MPa, MS = +0.03256: not a fail either
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.00794, abs=1e-5)
    assert "The margin of safety is +0.012 without the uncertainty and -0.008 with the uncertainty on the unsafe side." in item.reasons
    assert "The margin is smaller than the numerical uncertainty." in item.reasons
    assert item.statement.endswith("The verdict is inconclusive because the margin is smaller than the numerical uncertainty.")


def test_converged_result_without_an_uncertainty_is_used_as_it_is(workproject):
    # only a hand calculation or a derived result can decide without an uncertainty
    item = one(workproject, PROOF, res(MB + "scls", 150 * MPA, u=None, method="derived"))
    # no uncertainty: MS = 170 / 150 - 1 = 0.13333
    assert item.verdict == "pass" and item.margin == pytest.approx(0.13333, abs=1e-5)
    assert item.uncertainty is None and item.conservative_value == 150e6
    assert item.reasons[:3] == ["The result has the status converged.", "The analysis gives no uncertainty for this result.",
                                "The assessment uses the value as it is."]
    assert "The margin of safety is +0.13. The criterion passes." in item.statement and "uncertainty" not in item.statement
    assert one(workproject, PROOF, res(MB + "scls", 171 * MPA, u=None, method="derived")).verdict == "fail"     # 170 / 171 - 1 = -0.0058


def test_absolute_uncertainty_wins_over_the_relative_one(workproject):
    rv = ResultValue(key=MB + "scls", value=100 * MPA, unit="Pa", uncertainty=0.5, uncertainty_abs=5 * MPA, status="converged")
    item = one(workproject, PROOF, rv)
    # 100 + 5 = 105 MPa (not 100 x 1.5); MS = 170 / 105 - 1 = 0.61905; relative to the value: 5 / 100
    assert item.conservative_value == pytest.approx(105e6) and item.margin == pytest.approx(0.61905, abs=1e-5)
    assert item.uncertainty == pytest.approx(0.05) and "is 100 MPa with an uncertainty of 5 MPa." in item.statement


def test_a_result_in_another_unit_is_converted_to_si(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 71.3, unit="MPa", u=0.02))
    assert (item.value, item.unit) == (pytest.approx(71.3e6), "Pa") and item.margin == pytest.approx(1.33754, abs=1e-5)


# --------------------------------------------------------------------------- results that cannot give a pass


def test_singular_peak_is_inconclusive_and_says_when_every_mesh_fails(workproject):
    peak = res("von_mises.peak@part", 412 * MPA, u=None, status="singular", per_level=[310 * MPA, 365 * MPA, 412 * MPA],
               near="the edge between FT-SEAT-OUT-SHOULDER and FT-OUTLET-BORE")
    item = one(workproject, OP_YIELD, peak, "LC-FIRE")
    # finest mesh:   MS = 170 / (1.1 x 412) - 1 = 170 / 453.2 - 1 = -0.62489
    # coarsest mesh: MS = 170 / (1.1 x 310) - 1 = 170 / 341 - 1 = -0.50147: the criterion fails on every mesh
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.62489, abs=1e-5)
    assert item.factor == 1.1 and item.factor_source == "parameter yield_factor of REQ-MOV-004"
    assert item.reasons == [
        "The result has the status singular.",
        "The value increases with each mesh refinement and has no upper bound.",
        "The value on the coarsest mesh is 310 MPa.",
        "With the factor of safety, the peak is above the limit on every mesh of the study.",
        "An engineer can use section stresses, model the real edge radius, or do an elastic-plastic analysis.",
        'The location of the peak is "the edge between FT-SEAT-OUT-SHOULDER and FT-OUTLET-BORE".',
        "The record of analysis AN-FIRE has no check."]
    assert item.statement == (
        "The peak von Mises stress in the part is 412 MPa on the finest mesh. "
        "The allowable is 170 MPa (yield strength of MAT-316L). "
        "The factor of safety is 1.1 (parameter yield_factor of REQ-MOV-004). The margin of safety is -0.62. "
        "The verdict is inconclusive because the result is a singular peak.")


def test_singular_peak_under_the_limit_is_still_inconclusive(workproject):
    peak = res("von_mises.peak@part", 150 * MPA, u=None, status="singular", per_level=[120 * MPA, 135 * MPA, 150 * MPA])
    item = one(workproject, OP_YIELD, peak)
    # finest mesh: MS = 170 / (1.1 x 150) - 1 = +0.03030: positive, but the value has no limit under refinement
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(0.03030, abs=1e-5)
    assert not any("every mesh" in r for r in item.reasons)
    peak = res("von_mises.peak@part", 190 * MPA, u=None, status="singular", per_level=[120 * MPA, 150 * MPA, 190 * MPA])
    item = one(workproject, OP_YIELD, peak)
    # finest: 170 / 209 - 1 = -0.18660; coarsest: 170 / 132 - 1 = +0.28788: not above the limit on every mesh
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.18660, abs=1e-5)
    assert not any("every mesh" in r for r in item.reasons)
    # a singular damage: section stresses are no help, the advice is another one
    item = one(workproject, LIFE, res("damage.total@part", 0.05, unit="", u=None, status="singular", per_level=[0.02, 0.05]))
    # MS = 1 / (4 x 0.05) - 1 = +4.0
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(4.0)
    assert "An engineer can model the real edge radius, or use a result that does not include the edge." in item.reasons


REFINE = "An engineer can do the analysis again with a finer mesh or with more mesh levels."


def test_not_converged_passes_when_the_margin_holds_with_k_times_the_uncertainty(workproject):
    # the burst case of the first full run: 116.3 MPa, uncertainty 6.9 % (policy limit 5 %), Su = 485 MPa, factor 1.0
    item = one(workproject, BURST, res("von_mises.membrane@scls", 116.3 * MPA, u=0.069, status="not_converged",
                                       near="SCL-SHELL"), "LC-BURST", checks=[check("CHK-EQUILIBRIUM", True)])
    # 1 x: 116.3 x 1.069 = 124.3247 MPa; MS = 485 / 124.3247 - 1 = +2.90108   (margin and conservative value)
    # 2 x: 116.3 x 1.138 = 132.3494 MPa; MS = 485 / 132.3494 - 1 = +2.66455   >= 0: pass, and the margin shown
    assert workproject.policy.unconverged_margin_factor == 2.0
    assert item.verdict == "pass" and item.margin == pytest.approx(2.66455, abs=1e-5)
    assert item.conservative_value == pytest.approx(124.3247e6) and item.uncertainty == pytest.approx(0.069)
    assert item.reasons[:5] == ["The result has the status not converged.",
                                "The uncertainty of the result is 6.9 %, and the policy limit is 5 %.",
                                "The margin of safety is +2.66 with 2 times this uncertainty on the unsafe side.",
                                "The policy accepts this margin for a result with the status not converged.",
                                'The location of the value is "SCL-SHELL".']
    assert item.statement.endswith(
        "The margin of safety is +2.90 with the uncertainty on the unsafe side. The margin that decides the verdict is "
        "+2.66, with 2 times the uncertainty on the unsafe side. "
        "The result has the status not converged, but the criterion passes with 2 times its uncertainty.")


def test_not_converged_fails_when_the_margin_is_negative_with_k_times_the_uncertainty_on_the_safe_side(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 300 * MPA, u=0.10, status="not_converged"))
    # 1 x unsafe: 330 MPa; MS = 170 / 330 - 1 = -0.48485 (the margin)
    # 2 x safe:   300 x 0.8 = 240 MPa; MS = 170 / 240 - 1 = -0.29167 < 0: fail, and the margin shown
    assert item.verdict == "fail" and item.margin == pytest.approx(-0.29167, abs=1e-5)
    assert "The margin of safety is -0.29 with 2 times this uncertainty on the safe side." in item.reasons
    assert "The criterion fails for each value in this wider band." in item.reasons
    assert item.statement.endswith("The margin that decides the verdict is -0.29, with 2 times the uncertainty on the safe side. "
                                   "The result has the status not converged, but the criterion fails also with 2 times its uncertainty.")


def test_not_converged_is_inconclusive_when_the_sign_changes_inside_k_times_the_uncertainty(workproject):
    item = one(workproject, OP_YIELD, res("von_mises.peak@part", 140 * MPA, u=0.06, status="not_converged",
                                          near="FT-NECK-ROOT-FILLET"))
    # 1 x unsafe: 148.4 MPa; MS = 170 / (1.1 x 148.4) - 1 = 170 / 163.24 - 1 = +0.04141 (a pass if it were converged)
    # 2 x unsafe: 156.8 MPa; MS = 170 / 172.48 - 1 = -0.01438
    # 2 x safe:   123.2 MPa; MS = 170 / 135.52 - 1 = +0.25443: neither a pass nor a fail
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(0.04141, abs=1e-5)
    assert item.reasons[:5] == ["The result has the status not converged.",
                                "The uncertainty of the result is 6 %, and the policy limit is 5 %.",
                                "The margin of safety is -0.014 with 2 times this uncertainty on the unsafe side and +0.25 on the safe side.",
                                "The margin is smaller than 2 times the uncertainty.", REFINE]
    assert item.statement.endswith("The verdict is inconclusive because the margin is smaller than 2 times the uncertainty "
                                   "of a result with the status not converged.")


def test_not_converged_with_an_absolute_and_a_relative_uncertainty(workproject):
    rv = ResultValue(key=MB + "scls", value=92 * MPA, unit="Pa", uncertainty=0.0697, uncertainty_abs=6.4 * MPA,
                     status="not_converged", error_estimated=True)
    item = one(workproject, PROOF, rv)
    # the absolute one is used: 1 x: 98.4 MPa, MS = 170 / 98.4 - 1 = +0.72764; 2 x: 104.8 MPa, MS = +0.62214: pass
    assert item.verdict == "pass" and item.margin == pytest.approx(0.62214, abs=1e-5)
    assert item.reasons[1:3] == ["The uncertainty of the result is 6.4 MPa (7 %), and the policy limit is 5 %.",
                                 "The margin of safety is +0.62 with 2 times this uncertainty on the unsafe side."]


def test_factor_zero_gives_the_old_rule(workproject):
    workproject.policy.unconverged_margin_factor = 0.0
    out, items, _ = assess(workproject, [case("LC-BURST", [BURST])],
                           [analysis("LC-BURST", [res("von_mises.membrane@scls", 116.3 * MPA, u=0.069, status="not_converged",
                                                      near="SCL-SHELL")])])
    item = out.items[0]
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(2.90108, abs=1e-5)
    assert item.reasons[:4] == ["The result has the status not converged.",
                                "The uncertainty of the result is 6.9 %, and the policy limit is 5 %.", REFINE,
                                'The location of the value is "SCL-SHELL".']
    assert item.statement.endswith("The margin of safety is +2.90 with the uncertainty on the unsafe side. "
                                   "The verdict is inconclusive because the result has the status not converged.")
    assert not any(d.id.endswith("UNCONVERGED-MARGIN") for d in out.journal.decisions)


def test_not_converged_temperature_with_an_absolute_uncertainty(workproject):
    th = dict(lc_id="LC-TH-FIRE", analysis_type="thermal_steady")

    def t(value, du):
        return temperature(value, uncertainty_abs=du).model_copy(update={"status": "not_converged",
                                                                         "error_estimated": True})

    item = one(workproject, TEMP, t(251.3, 4.0), **th)
    # absolute, in K: 1 x: 247.3 K, MS = 247.3 / 233 - 1 = +0.06137; 2 x: 243.3 K, MS = +0.04421: pass (shown)
    assert item.verdict == "pass" and item.margin == pytest.approx(0.04421, abs=1e-5)
    assert item.conservative_value == pytest.approx(247.3)
    assert item.reasons[1:3] == ["The uncertainty of the result is 4 K.",
                                 "The margin of safety is +0.044 with 2 times this uncertainty on the unsafe side."]
    item = one(workproject, TEMP, t(220.0, 4.0), **th)
    # 2 x safe: 228 K, MS = 228 / 233 - 1 = -0.02146: fail, the margin shown (1 x unsafe: 216 / 233 - 1 = -0.07296)
    assert item.verdict == "fail" and item.margin == pytest.approx(-0.02146, abs=1e-5)
    assert "The margin of safety is -0.021 with 2 times this uncertainty on the safe side." in item.reasons
    item = one(workproject, TEMP, t(236.0, 3.0), **th)
    # 1 x: 233 K, MS = 0 (a pass if it were converged); 2 x unsafe: 230 K, MS = -0.01288; 2 x safe 242 K, +0.03863
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(0.0, abs=1e-12)


def test_not_converged_without_an_uncertainty_stays_inconclusive(workproject):
    # no uncertainty: the note of the analysis gives the cause. One mesh level: a mesh study is the remedy
    one_level = res("von_mises.peak@part", 100 * MPA, u=None, status="not_converged", per_level=[100 * MPA],
                    note="The analysis has one mesh level.")
    item = one(workproject, OP_YIELD, one_level)
    # MS = 170 / (1.1 x 100) - 1 = +0.54545
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(0.54545, abs=1e-5)
    assert item.reasons[:4] == ["The result has the status not converged.", "The analysis gives no uncertainty for this result.",
                                'The note of the result is: "The analysis has one mesh level."',
                                "An engineer can do the analysis again with a finer mesh or with more mesh levels."]
    # several levels and no uncertainty: the value has no limit (a peak at an edge that did not grow each time)
    at_edge = res("von_mises.peak@part", 100 * MPA, u=None, status="not_converged", per_level=[90 * MPA, 104 * MPA, 100 * MPA],
                  note="The peak is at a sharp re-entrant edge, where the stress has no limit in linear theory.")
    item = one(workproject, OP_YIELD, at_edge)
    assert item.reasons[2:4] == ['The note of the result is: "The peak is at a sharp re-entrant edge, where the stress has no '
                                 'limit in linear theory."',
                                 "An engineer can use section stresses, model the real edge radius, or do an elastic-plastic analysis."]


def test_a_failed_check_still_blocks_a_pass_or_a_fail_of_a_result_that_is_not_converged(workproject):
    eq = check("CHK-EQUILIBRIUM", False, statement="The reactions are 12 % smaller than the applied loads.")
    item = one(workproject, BURST, res("von_mises.membrane@scls", 116.3 * MPA, u=0.069, status="not_converged"), checks=[eq])
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(2.90108, abs=1e-5)       # a pass with k = 2
    assert "Check CHK-EQUILIBRIUM of analysis AN-X failed." in item.reasons
    assert item.statement.endswith("The verdict is inconclusive because a check of the analysis failed.")
    item = one(workproject, PROOF, res(MB + "scls", 300 * MPA, u=0.10, status="not_converged"), checks=[eq])
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.48485, abs=1e-5)      # a fail with k = 2


def test_the_rule_for_results_that_are_not_converged_is_one_decision_of_the_policy(workproject):
    cases = [case("LC-BURST", [BURST]), case("LC-PROOF", [PROOF])]
    ans = [analysis("LC-BURST", [res("von_mises.membrane@scls", 116.3 * MPA, u=0.069, status="not_converged")]),
           analysis("LC-PROOF", [res(MB + "scls", 300 * MPA, u=0.10, status="not_converged")])]
    out, _, _ = assess(workproject, cases, ans)
    dec = [d for d in out.journal.decisions if d.subject == workproject.policy.id]
    assert [d.id for d in dec] == ["DEC-POLICY-UNCONVERGED-MARGIN"]
    assert "2 times the uncertainty" in dec[0].choice and "conservative bound" in dec[0].rationale
    assert "5 %" in dec[0].rationale and dec[0].alternatives[0].option.startswith("Refuse each result")
    assert ("uses", "POLICY") in {(l.rel, l.to) for l in dec[0].refs()}
    # only converged results: no such decision
    out, _, _ = assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)])])
    assert [d.id for d in out.journal.decisions] == ["DEC-CMP-MOV-BODY-UNCERTAINTY-IN-MARGIN"]


def test_estimate_status_is_inconclusive(workproject):
    item = one(workproject, DROP, res("pressure_drop.total@part", 30e3, u=None, status="estimate", method="hand_calc",
                                      note="The value comes from a loss coefficient."))
    # MS = 60 / 30 - 1 = +1.0
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(1.0)
    assert item.reasons[:3] == ["The analysis gives this value as an estimate.",
                                'The note of the result is: "The value comes from a loss coefficient."',
                                "An estimate does not confirm a criterion."]
    assert item.statement.endswith("The verdict is inconclusive because the value is an estimate.")


def test_failed_check_makes_the_verdict_inconclusive(workproject):
    good = res(MB + "scls", 71.3 * MPA)
    eq = check("CHK-EQUILIBRIUM", False, statement="The reactions are 12 % smaller than the applied loads.")
    item = one(workproject, PROOF, good, checks=[check("CHK-MESH", True, "mesh_quality"), eq])
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(1.33754, abs=1e-5)      # the margin stays
    assert "Check CHK-EQUILIBRIUM of analysis AN-X failed." in item.reasons
    assert 'The check says: "The reactions are 12 % smaller than the applied loads."' in item.reasons
    assert not any(r.startswith("No check of") for r in item.reasons)
    assert item.statement.endswith("The verdict is inconclusive because a check of the analysis failed.")
    # a fail from an analysis with a failed check is not a reliable fail
    item = one(workproject, PROOF, res(MB + "scls", 300 * MPA), checks=[eq])
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(170 / 306 - 1)          # -0.44444
    # a check that has no result is a note, not a blocker
    item = one(workproject, PROOF, good, checks=[check("CHK-EQUILIBRIUM", True), check("CHK-HC-FLOOR", None, "hand_calc")])
    assert item.verdict == "pass" and "Check CHK-HC-FLOOR of analysis AN-X has no result." in item.reasons


def test_hand_calculation_that_does_not_agree_makes_the_verdict_inconclusive(workproject):
    hc = HandCalcResult(id="HCR-SHELL-HOOP-PROOF", spec="HC-SHELL-HOOP", model="thick_cylinder_pressure", load_case="LC-PROOF")
    bad = Check(id="CHK-PROOF-HC-SHELL-HOOP", kind="hand_calc", passed=False, value=1.9, limit=1.5, link=hc.id,
                statement="The ratio of the analysis result to the hand calculation is 1.9. The accepted band is 0.6 to 1.5.")
    out, items, verdicts = assess(workproject, [case("LC-PROOF", [PROOF])],
                                  [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)], checks=[check("CHK-PROOF-EQ", True), bad])],
                                  hand_calcs=[hc])
    item = out.items[0]
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(1.33754, abs=1e-5)
    assert "Check CHK-PROOF-HC-SHELL-HOOP of analysis AN-PROOF failed." in item.reasons
    assert Link(rel="evidence", to=hc.id, note="check CHK-PROOF-HC-SHELL-HOOP failed") in item.links
    assert verdicts["REQ-MOV-002"].statement == (
        "The verdict for requirement REQ-MOV-002 is inconclusive. "
        "Compliance item CMPL-REQ-MOV-002-YIELD-LC-PROOF decides the verdict with a margin of safety of +1.34. "
        "The cause is that a check of the analysis failed.")


def test_failed_error_estimate_check_blocks_a_pass_like_any_failed_check(workproject):
    # the brief: any check with passed False makes the verdict inconclusive and is named
    est = Check(id="CHK-TH-FIRE-ERROR-ESTIMATE", kind="error_estimate", passed=False, value=0.31, limit=0.05,
                statement="The largest relative uncertainty of a result is 31 % (heat_flow.total@INT-ENV).")
    item = one(workproject, TEMP, temperature(251.3, uncertainty_abs=4.0), "LC-TH-FIRE", analysis_type="thermal_steady",
               checks=[check("CHK-TH-FIRE-HEAT-BALANCE", True, "heat_balance"), est])
    assert item.verdict == "inconclusive"
    assert item.margin == pytest.approx(0.06137, abs=1e-5)       # the margin itself is unchanged
    assert "Check CHK-TH-FIRE-ERROR-ESTIMATE of analysis AN-TH-FIRE failed." in item.reasons
    assert "a check of the analysis failed" in item.statement


def test_failed_check_in_an_analysis_the_result_rests_on(workproject):
    cases = [case("LC-PROOF", [PROOF]), case("LC-BURST", [BURST], scaled_from="LC-PROOF", scale=2.5 / 1.5, status="scaled")]
    burst = analysis("LC-BURST", [res("von_mises.membrane@scls", 96 * MPA)])
    proof = analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)], checks=[check("CHK-EQUILIBRIUM", False)])
    _, items, _ = assess(workproject, cases, [proof, burst])
    item = items["CMPL-REQ-MOV-003-ULTIMATE-LC-BURST"]
    # 96 x 1.02 = 97.92 MPa; MS = 485 / 97.92 - 1 = 3.95302: a pass, but the source analysis has a failed check
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(3.95302, abs=1e-5)
    assert "Check CHK-EQUILIBRIUM of analysis AN-PROOF failed." in item.reasons
    # the same when the analysis it rests on is missing
    cases[1] = case("LC-BURST", [BURST], uses=["LC-GONE"])
    _, items, _ = assess(workproject, cases, [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)]), burst])
    item = items["CMPL-REQ-MOV-003-ULTIMATE-LC-BURST"]
    assert item.verdict == "inconclusive" and "Load case LC-GONE has no complete analysis." in item.reasons
    assert items["CMPL-REQ-MOV-002-YIELD-LC-PROOF"].verdict == "pass"


def test_stage_files_older_than_the_inputs_cannot_give_a_pass(workproject):
    cases, ans = [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)])]
    assert assess(workproject, cases, ans)[0].items[0].verdict == "pass"
    policy = workproject.root / "policy.yaml"
    policy.write_text(policy.read_text() + "\n# a change after the analysis\n")
    project = Project.load(workproject.root)
    item = compliance.assess(project, context.build(project)).items[0]
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(1.33754, abs=1e-5)
    assert "The input files changed after analysis AN-PROOF." in item.reasons
    assert "The input files changed after the plan stage wrote the load cases." in item.reasons


# --------------------------------------------------------------------------- no analysis value


def test_missing_analysis_is_not_assessed_with_the_reason(workproject):
    cases = [case("LC-PROOF", [PROOF]), case("LC-BURST", [BURST]), case("LC-FIRE", [OP_YIELD])]
    ans = [analysis("LC-BURST", status="failed", message="LinAlgError: the stiffness matrix is singular (see analyses/LC-BURST/error.log)"),
           analysis("LC-FIRE", [res(MB + "scls", 90 * MPA)], message="The peak evaluation did not run.")]
    _, items, verdicts = assess(workproject, cases, ans)
    none, failed, nokey = (items[i] for i in ("CMPL-REQ-MOV-002-YIELD-LC-PROOF", "CMPL-REQ-MOV-003-ULTIMATE-LC-BURST",
                                              "CMPL-REQ-MOV-004-YIELD-LC-FIRE"))
    for item in (none, failed, nokey):
        assert item.verdict == "not_assessed" and item.value is None and item.margin is None and item.conservative_value is None
    assert none.analysis is None and none.reasons == ["Load case LC-PROOF has no analysis record."]
    assert none.statement == (
        "Load case LC-PROOF gives no value for the von Mises membrane plus bending stress on the worst stress "
        "classification line. The allowable is 170 MPa (yield strength of MAT-316L). "
        "The factor of safety is 1.0 (criterion yield of REQ-MOV-002). "
        "The assessment gives no verdict because load case LC-PROOF has no analysis.")
    assert failed.analysis == "AN-BURST" and failed.reasons == [
        "Analysis AN-BURST of load case LC-BURST has the status failed.",
        'The message of the analysis is: "LinAlgError: the stiffness matrix is singular (see analyses/LC-BURST/error.log)"']
    assert nokey.reasons == ["Analysis AN-FIRE has no result with the key von_mises.peak@part.",
                             'The message of the analysis is: "The peak evaluation did not run."']
    assert nokey.limit == 170e6 and nokey.factor == 1.1         # what the criterion asks is still in the item
    assert {v.verdict for v in verdicts.values()} == {"not_assessed"}
    # a result without a value, or one that did not run, is no result
    for rv in (res("von_mises.peak@part", None, status="not_run", note="The mesh study did not finish."),
               res("von_mises.peak@part", float("nan"))):
        item = one(workproject, OP_YIELD, rv)
        assert item.verdict == "not_assessed" and item.reasons[0] == "Analysis AN-X gives no value for the result von_mises.peak@part."


def hand_calc(lc_id, value=14.4e3, unit="Pa", key="pressure_drop.total@part", **kw):
    return HandCalcResult(id=f"HC-PRESSURE-DROP-{lc_id}", spec="HC-PRESSURE-DROP", model="pressure_drop_minor_loss",
                          load_case=lc_id, outputs=[CalcOutput(name="dp_total", si=value, unit=unit, display="14.4 kPa")],
                          comparisons=[Comparison(output="dp_total", result_key=key, hand_value=value)], **kw)


def test_hand_calculation_is_an_estimate_when_there_is_no_analysis(workproject):
    cases = [case("LC-FLOW-FIRE", [DROP], "PH-FIRE", "flow")]
    ans = [analysis("LC-FLOW-FIRE", status="not_supported", analysis_type="flow", message="OpenFOAM is not on this computer.")]
    hc = hand_calc("LC-FLOW-FIRE", warnings=["The loss coefficient K is an estimate."])
    out, items, verdicts = assess(workproject, cases, ans, [hc])
    item = items["CMPL-REQ-MOV-008-PRESSURE-DROP-LC-FLOW-FIRE"]
    # the estimate is 14.4 kPa: MS = 60 / 14.4 - 1 = 3.16667
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(3.16667, abs=1e-5)
    assert (item.value, item.unit, item.uncertainty, item.conservative_value) == (14.4e3, "Pa", None, 14.4e3)
    assert item.limit == pytest.approx(60e3) and item.limit_source == "parameter dp_max of REQ-MOV-008"
    assert item.factor_source == "the criterion gives no factor"
    assert item.analysis is None and Link(rel="evidence", to=hc.id, note="estimate from a hand calculation") in item.links
    assert item.reasons == [
        "The value is an estimate from a hand calculation. No analysis confirms it.",
        "Hand calculation HC-PRESSURE-DROP-LC-FLOW-FIRE gives the value as its output dp_total.",
        "Analysis AN-FLOW-FIRE of load case LC-FLOW-FIRE has the status not supported.",
        'The message of the analysis is: "OpenFOAM is not on this computer."',
        'The hand calculation says about the limits of its formula: "The loss coefficient K is an estimate."']
    assert item.statement == (
        "Hand calculation HC-PRESSURE-DROP-LC-FLOW-FIRE gives an estimate of 14.4 kPa for the total pressure drop in the part. "
        "The limit is 60 kPa (parameter dp_max of REQ-MOV-008). The factor of safety is 1.0 (the criterion gives no factor). "
        "The margin of safety of the estimate is +3.17. "
        "The verdict is inconclusive because the value is an estimate from a hand calculation.")
    assert verdicts["REQ-MOV-008"].verdict == "inconclusive" and verdicts["REQ-MOV-008"].margin == pytest.approx(3.16667, abs=1e-5)
    dec = next(d for d in out.journal.decisions if d.id == make_id("DEC", item.id, "estimate"))
    assert dec.subject == item.id and dec.needs_review and dec.stage == "assess" and dec.alternatives
    assert {(l.rel, l.to) for l in dec.refs()} >= {("evidence", hc.id), ("about", "LC-FLOW-FIRE"), ("about", item.id)}
    # also when the analysis is done but has no such result, and when there is no record
    for ans in ([analysis("LC-FLOW-FIRE", [res("pressure_drop.mean@part", 1.0)], analysis_type="flow")], []):
        (workproject.analysis_dir("LC-FLOW-FIRE") / "analysis.json").unlink(missing_ok=True)
        item = assess(workproject, cases, ans, [hc])[0].items[0]
        assert item.verdict == "inconclusive" and item.margin == pytest.approx(3.16667, abs=1e-5)


def test_hand_calculation_of_another_case_key_or_dimension_is_no_estimate(workproject):
    cases = [case("LC-FLOW-FIRE", [DROP], "PH-FIRE", "flow")]
    for hc in (hand_calc("LC-FLOW-CHILL"), hand_calc("LC-FLOW-FIRE", key="pressure_drop.mean@part"),
               hand_calc("LC-FLOW-FIRE", unit="m")):
        out, _, _ = assess(workproject, cases, [], [hc])
        assert out.items[0].verdict == "not_assessed" and out.items[0].value is None
        assert not [d for d in out.journal.decisions if d.id.endswith("-ESTIMATE")]
    # an analysis value wins over the hand calculation
    ans = [analysis("LC-FLOW-FIRE", [res("pressure_drop.total@part", 20e3)], analysis_type="flow")]
    item = assess(workproject, cases, ans, [hand_calc("LC-FLOW-FIRE")])[0].items[0]
    assert item.verdict == "pass" and item.margin == pytest.approx(60 / 20.4 - 1)      # 20 x 1.02 = 20.4 kPa: +1.94118


def test_unit_mismatch_is_not_assessed(workproject):
    key = "radial_displacement.range@FT-SEAT-OUT-WALL"
    item = one(workproject, ROUND, res(key, 12.3 * MPA, unit="Pa"))                    # a stress where a length belongs
    assert item.verdict == "not_assessed" and item.margin is None and item.conservative_value is None
    assert (item.value, item.unit, item.limit) == (12.3e6, "Pa", None)
    assert item.reasons[:3] == [f"The result {key} is in Pa.", "The limit is in m.",
                                "The margin of safety needs two values of the same dimension."]
    assert item.statement == (
        "The range of the radial displacement on FT-SEAT-OUT-WALL is 12.3 MPa. "
        "The limit is 20 µm (parameter roundness_max of REQ-MOV-010). "
        "The factor of safety is 1.0 (the criterion gives no factor). "
        "The assessment gives no verdict because the result and the limit do not have the same dimension.")
    # a result without a unit is a pure number: also a mismatch for a length
    item = one(workproject, ROUND, res(key, 12.3e-6, unit=""))
    assert item.verdict == "not_assessed" and item.reasons[0] == f"The result {key} is a pure number."
    item = one(workproject, ROUND, res(key, 12.3e-6, unit="furlongs_per_fortnight"))
    assert item.verdict == "not_assessed" and "does not know this unit" in item.reasons[0]
    # the same result in the right unit: 12.3 um x 1.02 = 12.546 um; MS = 20 / 12.546 - 1 = 0.59413
    item = one(workproject, ROUND, res(key, 12.3e-6, unit="m"))
    assert item.verdict == "pass" and item.margin == pytest.approx(0.59413, abs=1e-5)
    assert item.statement.startswith("The range of the radial displacement on FT-SEAT-OUT-WALL is 12.3 µm with an uncertainty of 2 %.")


# --------------------------------------------------------------------------- lower limits, pure numbers


def temperature(value, **kw):
    return ResultValue(key="temperature.min@INT-ACT", value=value, unit="K", status="converged", **kw)


def test_temperature_lower_limit(workproject):
    th = dict(lc_id="LC-TH-FIRE", analysis_type="thermal_steady")
    item = one(workproject, TEMP, temperature(251.3, uncertainty=0.01), **th)
    # relative: 251.3 x 0.99 = 248.787 K; MS = 248.787 / 233 - 1 = 0.06776
    assert item.verdict == "pass" and item.comparator == ">=" and item.conservative_value == pytest.approx(248.787)
    assert item.margin == pytest.approx(0.06776, abs=1e-5) and item.limit == 233.0
    item = one(workproject, TEMP, temperature(251.3, uncertainty=0.01, uncertainty_abs=4.0), **th)
    # absolute, in K: 251.3 - 4 = 247.3 K; MS = 247.3 / 233 - 1 = 0.06137; relative to the value 4 / 251.3 = 0.01592
    assert item.verdict == "pass" and item.conservative_value == pytest.approx(247.3)
    assert item.margin == pytest.approx(0.06137, abs=1e-5) and item.uncertainty == pytest.approx(0.01592, abs=1e-5)
    assert item.statement == (
        "The minimum temperature on INT-ACT is 251.3 K with an uncertainty of 4 K. "
        "The lower limit is 233 K (parameter t_act_min of REQ-MOV-007). "
        "The factor of safety is 1.0 (the criterion gives no factor). "
        "The margin of safety is +0.061 with the uncertainty on the unsafe side. The criterion passes.")
    assert "The margin of safety is +0.079 without the uncertainty." in item.reasons       # 251.3 / 233 - 1 = 0.07854
    item = one(workproject, TEMP, temperature(235.0, uncertainty_abs=4.0), **th)
    # 235 - 4 = 231 K: MS = 231 / 233 - 1 = -0.00858; safe side 239 K: MS = +0.02575: inconclusive
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.00858, abs=1e-5)
    item = one(workproject, TEMP, temperature(225.0, uncertainty_abs=4.0), **th)
    # 225 - 4 = 221 K: MS = 221 / 233 - 1 = -0.05150; safe side 229 K: MS = -0.01717: fail
    assert item.verdict == "fail" and item.margin == pytest.approx(-0.05150, abs=1e-5)
    assert "The margin of safety is -0.017 with the uncertainty on the safe side." in item.reasons


def test_damage_with_a_life_factor(workproject):
    item = one(workproject, LIFE, res("damage.total@part", 0.1, unit="", u=0.2), "LC-FATIGUE", analysis_type="fatigue")
    # 0.1 x 1.2 = 0.12; MS = 1 / (4 x 0.12) - 1 = 1 / 0.48 - 1 = 1.08333
    assert item.verdict == "pass" and item.margin == pytest.approx(1.08333, abs=1e-5) and item.unit == ""
    assert (item.factor, item.factor_source) == (4.0, "parameter life_factor of REQ-MOV-006")
    assert (item.limit, item.limit_source) == (1.0, "criterion damage of REQ-MOV-006")
    assert item.statement == (
        "The total fatigue damage in the part is 0.1 with an uncertainty of 20 %. "
        "The limit is 1 (criterion damage of REQ-MOV-006). "
        "The factor of safety is 4.0 (parameter life_factor of REQ-MOV-006). "
        "The margin of safety is +1.08 with the uncertainty on the unsafe side. The criterion passes.")
    item = one(workproject, LIFE, res("damage.total@part", 0.3, unit="", u=0.1))
    # 0.33: MS = 1 / 1.32 - 1 = -0.24242; safe side 0.27: MS = 1 / 1.08 - 1 = -0.07407: fail
    assert item.verdict == "fail" and item.margin == pytest.approx(-0.24242, abs=1e-5)
    item = one(workproject, LIFE, res("damage.total@part", 0.24, unit="", u=0.1))
    # value: 1 / 0.96 - 1 = +0.04167; 0.264: MS = 1 / 1.056 - 1 = -0.05303: inconclusive
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(-0.05303, abs=1e-5)


def test_zero_response_passes_with_no_finite_margin(workproject):
    out, items, verdicts = assess(workproject, [case("LC-FATIGUE", [LIFE], analysis_type="fatigue")],
                                  [analysis("LC-FATIGUE", [res("damage.total@part", 0.0, unit="", u=0.2)], analysis_type="fatigue")])
    item = out.items[0]
    assert item.verdict == "pass" and item.margin is None and item.conservative_value == 0.0
    assert "The value is not above zero, so the margin of safety has no upper bound. The criterion passes." in item.statement
    assert verdicts["REQ-MOV-006"].verdict == "pass" and verdicts["REQ-MOV-006"].margin is None
    assert verdicts["REQ-MOV-006"].governing == item.id


# --------------------------------------------------------------------------- stress classification lines


def lines(values, **kw):
    return [res(MB + line, v * MPA, **kw) for line, v in zip(LINES, values) if v is not None]


def test_scls_aggregate_key_is_never_used_for_a_verdict(workproject):
    ans = [analysis("LC-PROOF", [res(MB + "scls", 100 * MPA), *lines([96, 60, 150, 30, 20])])]
    item = assess(workproject, [case("LC-PROOF", [PROOF])], ans)[0].items[0]
    # the lines govern, not the aggregate (100 MPa): SCL-FLOOR 150 x 1.02 = 153 MPa; MS = 170 / 153 - 1 = 0.11111
    assert item.result_key == MB + "SCL-FLOOR" and item.margin == pytest.approx(0.11111, abs=1e-5)
    assert item.verdict == "pass"


def test_scls_two_lines_the_aggregate_of_the_engine_does_not_pass(workproject):
    """Line A 100 +/- 1 MPa converged, line B 95 +/- 20 MPa not converged, limit 133 MPa. The aggregate of the
    structural engine (100 MPa, u 15 MPa, not converged) passes with k = 2 (130 MPa); line B alone does not:
    95 + 2 x 20 = 135 MPa > 133 MPa."""
    from dossier.analysis.structural import _aggregate

    a = ResultValue(key=MB + "SCL-SHELL", value=100 * MPA, unit="Pa", uncertainty_abs=1 * MPA, uncertainty=0.01,
                    status="converged", per_level=[99 * MPA, 100 * MPA], error_estimated=True)
    b = ResultValue(key=MB + "SCL-FLOOR", value=95 * MPA, unit="Pa", uncertainty_abs=20 * MPA, uncertainty=20 / 95,
                    status="not_converged", per_level=[80 * MPA, 95 * MPA], error_estimated=True)
    agg = _aggregate(MB + "scls", {"SCL-SHELL": a, "SCL-FLOOR": b}, [])
    assert agg.value == 100 * MPA and agg.uncertainty_abs == pytest.approx(15 * MPA)      # 115 MPa covers line B
    req = workproject.requirement("REQ-MOV-002")
    req.criteria[0].allowable, req.criteria[0].limit = None, Value(value=133, unit="MPa")
    workproject.subject.scls = [s for s in workproject.subject.scls if s.id in ("SCL-SHELL", "SCL-FLOOR")]
    out, _, _ = assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [agg, a, b])])
    item = out.items[0]
    # line A: 101 MPa, MS = 133 / 101 - 1 = +0.31683: pass. Line B: 1 x: 115 MPa, MS = +0.15652;
    # 2 x unsafe: 135 MPa, MS = -0.01481; 2 x safe: 55 MPa, MS = +1.41818: inconclusive. B governs.
    assert item.verdict == "inconclusive" and item.result_key == MB + "SCL-FLOOR"
    assert item.margin == pytest.approx(0.15652, abs=1e-5)
    assert "on SCL-FLOOR, the line that governs," in item.statement


def test_scls_a_lost_line_stops_a_pass_also_with_an_aggregate_key(workproject):
    # the structural engine wrote an aggregate from SCL-SHELL alone; the four other lines have no result
    agg = res(MB + "scls", 100 * MPA, u=0.01)
    one_line = res(MB + "SCL-SHELL", 100 * MPA, u=0.01)
    out, _, _ = assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [agg, one_line])])
    item = out.items[0]
    # SCL-SHELL alone: 101 MPa, MS = 170 / 101 - 1 = +0.68317, a pass of one line only
    assert item.verdict == "inconclusive" and item.result_key == MB + "SCL-SHELL"
    assert item.margin == pytest.approx(0.68317, abs=1e-5)
    assert "Analysis AN-PROOF has no result for SCL-NOZZLE-OUT, SCL-FLOOR, SCL-ROOF and SCL-NECK-ROOT on the finest mesh." \
        in item.reasons
    assert item.statement.endswith("The verdict is inconclusive because one or more stress classification lines have no result.")


def test_scls_falls_back_to_the_worst_line(workproject):
    def run(results):
        return assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", results)])[0].items[0]

    item = run(lines([96, 60, 150, 30, 20]))
    # the worst line is SCL-FLOOR: 150 x 1.02 = 153 MPa; MS = 170 / 153 - 1 = 0.11111
    assert item.verdict == "pass" and item.result_key == MB + "SCL-FLOOR" and item.value == 150e6
    assert item.margin == pytest.approx(0.11111, abs=1e-5)
    assert "The assessment uses each of the 5 stress classification lines by itself. Line SCL-FLOOR governs." in item.reasons
    assert item.statement.startswith("The von Mises membrane plus bending stress on SCL-FLOOR, the line that governs, is 150 MPa "
                                     "with an uncertainty of 2 %.")
    # an aggregate key without a value is a missing key
    assert run([res(MB + "scls", None, status="not_run"), *lines([96, 60, 150, 30, 20])]).result_key == MB + "SCL-FLOOR"

    item = run(lines([96, 60, 150, None, 20]))
    # a line of the component has no result: the pass of the other lines is not a pass of every line
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(0.11111, abs=1e-5)
    assert "Analysis AN-PROOF has no result for SCL-ROOF on the finest mesh." in item.reasons
    assert item.statement.endswith("The verdict is inconclusive because one or more stress classification lines have no result.")

    item = run([*lines([None, 60, 140, 30, 20]), res(MB + "SCL-SHELL", 150 * MPA, u=0.09, status="not_converged")])
    # a line that is not converged can hide a larger value: it governs before the passing lines
    # 150 x 1.09 = 163.5 MPa; MS = 170 / 163.5 - 1 = 0.03976; 2 x: 177 MPa, MS = -0.03955; 2 x safe 123 MPa: +0.38211
    assert item.verdict == "inconclusive" and item.result_key == MB + "SCL-SHELL"
    assert item.margin == pytest.approx(0.03976, abs=1e-5)
    item = run([*lines([None, 60, 150, 30, 20]), res(MB + "SCL-SHELL", 96 * MPA, u=0.09, status="not_converged")])
    # 96 MPa with 2 x 9 %: 113.28 MPa, MS = +0.50071: a pass; the smallest margin governs (SCL-FLOOR, +0.11111)
    assert item.verdict == "pass" and item.result_key == MB + "SCL-FLOOR"

    item = run(lines([96, 60, 200, None, 20]))
    # a line that fails decides, also with a line missing: 200 x 1.02 = 204 MPa; MS = 170 / 204 - 1 = -0.16667
    assert item.verdict == "fail" and item.result_key == MB + "SCL-FLOOR" and item.margin == pytest.approx(-0.16667, abs=1e-5)

    item = run([res("von_mises.membrane@SCL-SHELL", 96 * MPA)])            # another measure is not this result
    assert item.verdict == "not_assessed" and item.reasons[0] == f"Analysis AN-PROOF has no result with the key {MB}scls."


# --------------------------------------------------------------------------- factor and limit


def test_factor_from_a_requirement_parameter_the_policy_or_a_number(workproject):
    crit = workproject.requirement("REQ-MOV-004").criteria[0]
    peak = res("von_mises.peak@part", 100 * MPA)                # conservative value 102 MPa

    item = one(workproject, OP_YIELD, peak)
    # yield_factor = 1.1 (REQ-MOV-004): MS = 170 / (1.1 x 102) - 1 = 170 / 112.2 - 1 = 0.51515
    assert (item.factor, item.factor_source) == (1.1, "parameter yield_factor of REQ-MOV-004")
    assert item.margin == pytest.approx(0.51515, abs=1e-5)
    assert Link(rel="uses", to="yield_factor", note="factor of safety") in item.links

    crit.factor = "yield"
    item = one(workproject, OP_YIELD, peak)
    # policy factor yield = 1.25: MS = 170 / (1.25 x 102) - 1 = 170 / 127.5 - 1 = 0.33333
    assert (item.factor, item.factor_source) == (1.25, "factor yield of the policy POLICY")
    assert item.margin == pytest.approx(0.33333, abs=1e-5)
    assert Link(rel="uses", to="POLICY", note="factor of safety: yield") in item.links
    assert "The factor of safety is 1.25 (factor yield of the policy POLICY)." in item.statement

    crit.factor = 1.5
    item = one(workproject, OP_YIELD, peak)
    # MS = 170 / (1.5 x 102) - 1 = 170 / 153 - 1 = 0.11111
    assert (item.factor, item.factor_source) == (1.5, "criterion yield of REQ-MOV-004")
    assert item.margin == pytest.approx(0.11111, abs=1e-5)

    crit.factor = "2.0"                                          # a number written as text
    assert one(workproject, OP_YIELD, peak).margin == pytest.approx(170 / 204 - 1)         # -0.16667

    for bad, reason in (("safety_margin", "The factor name safety_margin is not a parameter of the project and not a factor of the policy."),
                        ("meop", "Parameter meop is 10.5 MPa. A factor of safety must be a pure number."),
                        (0.0, "The factor of safety is 0 (criterion yield of REQ-MOV-004). A factor of safety must be above zero.")):
        crit.factor = bad
        item = one(workproject, OP_YIELD, peak)
        assert item.verdict == "not_assessed" and item.margin is None and item.reasons == [reason]
        assert item.statement.endswith("The assessment gives no verdict because the factor of safety has no valid value.")


def test_limit_from_an_allowable_a_parameter_an_expression_or_a_number(workproject):
    crit = workproject.requirement("REQ-MOV-004").criteria[0]
    crit.factor = 1.0
    peak = res("von_mises.peak@part", 100 * MPA, u=0.0)

    crit.allowable = "ultimate_strength"
    item = one(workproject, OP_YIELD, peak)
    assert (item.limit, item.limit_source) == (485e6, "ultimate strength of MAT-316L") and item.margin == pytest.approx(3.85)

    crit.allowable, crit.limit = None, Value.model_validate({"expr": "Sy / 2", "unit": "MPa"})
    item = one(workproject, OP_YIELD, peak)
    # 85 MPa: MS = 85 / 100 - 1 = -0.15
    assert (item.limit, item.limit_source) == (85e6, "expression Sy / 2 of criterion yield")
    assert item.margin == pytest.approx(-0.15) and Link(rel="uses", to="Sy", note="limit") in item.links
    assert "The limit is 85 MPa (expression Sy / 2 of criterion yield)." in item.statement

    crit.limit = Value.model_validate("120 MPa")
    item = one(workproject, OP_YIELD, peak)
    assert (item.limit, item.limit_source) == (120e6, "criterion yield of REQ-MOV-004") and item.margin == pytest.approx(0.2)

    crit.limit = Value.model_validate({"expr": "p_pump_max"})
    assert one(workproject, OP_YIELD, peak).limit_source == "system parameter p_pump_max"

    for bad, text in (({"expr": "no_such_name * 2"}, "has no value"), ({"expr": ["Sy", 0, 0]}, "is a vector"),
                      ("0 MPa", "needs a limit above zero"), ("-5 MPa", "needs a limit above zero")):
        crit.limit = Value.model_validate(bad)
        item = one(workproject, OP_YIELD, peak)
        assert item.verdict == "not_assessed" and item.margin is None and text in item.reasons[0]
        assert item.statement.endswith("The assessment gives no verdict because the limit has no valid value.")

    crit.limit, crit.allowable = None, "hardness"
    item = one(workproject, OP_YIELD, peak)
    assert item.verdict == "not_assessed" and item.reasons == ["The allowable name hardness is not a material property of the project."]


# --------------------------------------------------------------------------- caveats


def test_caveat_walk_reaches_the_requirement_that_states_a_load_parameter(workproject, example_ctx):
    def loaded(id, refs, phase, **kw):
        return case(id, refs, phase, loads=example_ctx.loads(workproject, None, phase), **kw)

    cases = [loaded("LC-PROOF", [PROOF], "PH-PROOF"),
             loaded("LC-BURST", [BURST], "PH-BURST", scaled_from="LC-PROOF", scale=2.5 / 1.5, status="scaled"),
             loaded("LC-FIRE", [OP_YIELD, OP_ULT], "PH-FIRE", analysis_type="thermo_structural"),
             case("LC-FATIGUE", [LIFE], analysis_type="fatigue", uses=["LC-PROOF", "LC-FIRE"],
                  cycles=[CycleUse(duty_cycle="DC-PROOF", count=2, states=["LC-PROOF"])])]
    ans = [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)]), analysis("LC-BURST", [res("von_mises.membrane@scls", 96 * MPA)]),
           analysis("LC-FIRE", [res("von_mises.peak@part", 100 * MPA), res(MB + "scls", 90 * MPA)], analysis_type="thermo_structural"),
           analysis("LC-FATIGUE", [res("damage.total@part", 0.1, unit="")], analysis_type="fatigue")]
    # the validation engine writes a finding on a requirement, with an "about" link to the parameter
    meop = finding("REQ-MOV-001", "meop-low", links=[Link(rel="about", to="meop"), Link(rel="evidence", to="EF-IN-P-SHUTDOWN")])
    fnd = [meop,                                                                # the MEOP of REQ-MOV-001 is too low
           finding("REQ-MOV-001", "no-shall", "minor"),                         # a minor finding is not a caveat
           finding("REQ-MOV-004", "yield-factor", links=[Link(rel="about", to="yield_factor")]),
           finding("REQ-MOV-003", "burst-factor", "error", links=[Link(rel="about", to="burst_factor")]),
           finding("REQ-MOV-006", "design-cycles", links=[Link(rel="about", to="design_cycles")]),
           finding("PH-TRANSPORT", "uncovered"),                                # a phase with no load case here
           finding("MAT-316L", "too-cold"),                                     # the material of the part
           finding("SCL-ROOF", "outside"),                                      # a stress classification line
           finding("INT-ENV", "overlap", "error"),                              # an interface that firing loads
           finding("SYS-OXFEED", "count", links=[Link(rel="about", to="n_proof")])]   # the count of the proof cycle
    out, items, verdicts = assess(workproject, cases, ans, findings=fnd)
    proof, burst = items["CMPL-REQ-MOV-002-YIELD-LC-PROOF"], items["CMPL-REQ-MOV-003-ULTIMATE-LC-BURST"]
    fire_y, fire_u = items["CMPL-REQ-MOV-004-YIELD-LC-FIRE"], items["CMPL-REQ-MOV-004-ULTIMATE-LC-FIRE"]
    life = items["CMPL-REQ-MOV-006-DAMAGE-LC-FATIGUE"]
    mat, scl, env = "FND-MAT-316L-TOO-COLD", "FND-SCL-ROOF-OUTSIDE", "FND-INT-ENV-OVERLAP"
    # the proof pressure is proof_factor * meop: the finding on REQ-MOV-001 reaches the proof and burst items
    assert proof.caveats == [meop.id, mat, scl]
    assert burst.caveats == [meop.id, "FND-REQ-MOV-003-BURST-FACTOR", mat, scl]
    # the loads of firing come from the system context: no requirement parameter feeds them
    assert fire_y.caveats == ["FND-REQ-MOV-004-YIELD-FACTOR", mat, env]
    assert fire_u.caveats == ["FND-REQ-MOV-004-YIELD-FACTOR", mat, scl, env]
    # the fatigue case uses the proof and firing cases, and the count of the proof duty cycle
    assert life.caveats == [meop.id, "FND-REQ-MOV-006-DESIGN-CYCLES", mat, env, "FND-SYS-OXFEED-COUNT"]
    # a caveat does not change the verdict; the statement names it; the record links to it
    assert proof.verdict == "pass" and proof.margin == pytest.approx(1.33754, abs=1e-5)
    assert proof.statement.endswith(
        "The criterion passes. The findings FND-REQ-MOV-001-MEOP-LOW, FND-MAT-316L-TOO-COLD and FND-SCL-ROOF-OUTSIDE "
        "apply to the inputs of this item.")
    assert Link(rel="about", to=meop.id, note="open finding") in proof.refs()
    assert verdicts["REQ-MOV-002"].verdict == "pass" and meop.id in verdicts["REQ-MOV-002"].statement
    # requirements that analysis does not assess name their findings too
    assert verdicts["REQ-MOV-001"].statement == (
        "The verification method of requirement REQ-MOV-001 is review. The analysis gives no verdict for this requirement. "
        "Finding FND-REQ-MOV-001-MEOP-LOW applies to this requirement.")


def test_caveat_on_another_requirement_is_about_the_number_the_item_reads(workproject, example_ctx):
    workproject.requirement("REQ-MOV-001").parameters["p_rated"] = ReqParameter.model_validate("9 MPa")   # no load reads it
    cases = [case("LC-PROOF", [PROOF], "PH-PROOF", loads=example_ctx.loads(workproject, None, "PH-PROOF"))]
    ans = [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)])]

    def run(*fnd):
        return assess(workproject, cases, ans, findings=fnd)[0].items[0]

    def about(name):
        return [Link(rel="about", to=name)]

    assert run(finding("REQ-MOV-001", "meop-low", links=about("meop"))).caveats == ["FND-REQ-MOV-001-MEOP-LOW"]
    assert run(finding("REQ-MOV-001", "rated", links=[Link(rel="evidence", to="p_rated")])).caveats == []   # another number
    assert run(finding("REQ-MOV-001", "text")).caveats == ["FND-REQ-MOV-001-TEXT"]     # names no number: it counts
    assert run(finding("REQ-MOV-005", "range")).caveats == []                          # states no number of these loads
    # the effects of the load case and the parameters they depend on (a_bore = pi * d_bore**2 / 4)
    assert run(finding("EF-IN-P-PROOF", "basis")).caveats == ["FND-EF-IN-P-PROOF-BASIS"]
    assert run(finding("EF-IN-P-FIRE", "basis")).caveats == []                          # an effect of another phase
    assert run(finding("SYS-OXFEED", "area", links=about("d_bore"))).caveats == ["FND-SYS-OXFEED-AREA"]
    assert run(finding("SYS-OXFEED", "closure", links=about("t_close"))).caveats == []  # the proof loads do not read it
    assert run(finding("FT-CAVITY-FLOOR", "faces", "error")).caveats == ["FND-FT-CAVITY-FLOOR-FACES"]   # a loaded feature
    assert run(finding("FT-NECK", "faces", "error")).caveats == []                      # no load of the case acts on it
    item = run(finding("EF-IN-P-PROOF", "basis"))
    assert item.verdict == "pass"
    assert item.statement.endswith("The criterion passes. Finding FND-EF-IN-P-PROOF-BASIS applies to the inputs of this item.")


# --------------------------------------------------------------------------- the verdict of a requirement


def operation(fire_peak, fire_scls, shut_peak, shut_scls):
    cases = [case("LC-FIRE", [OP_YIELD, OP_ULT], "PH-FIRE", "thermo_structural"),
             case("LC-SHUTDOWN", [OP_YIELD, OP_ULT], "PH-SHUTDOWN", "thermo_structural")]
    ans = [analysis("LC-FIRE", [r for r in (fire_peak, fire_scls) if r], analysis_type="thermo_structural"),
           analysis("LC-SHUTDOWN", [r for r in (shut_peak, shut_scls) if r], analysis_type="thermo_structural")]
    return cases, ans


def test_requirement_roll_up(workproject):
    peak, scls = "von_mises.peak@part", MB + "scls"
    ids = [f"CMPL-REQ-MOV-004-{c}-{lc}" for c in ("YIELD", "ULTIMATE") for lc in ("LC-FIRE", "LC-SHUTDOWN")]

    # all pass: the smallest margin governs
    _, items, verdicts = assess(workproject, *operation(res(peak, 100 * MPA), res(scls, 200 * MPA), res(peak, 120 * MPA), res(scls, 150 * MPA)))
    v = verdicts["REQ-MOV-004"]
    # yield: 170 / (1.1 x 102) - 1 = 0.51515 and 170 / (1.1 x 122.4) - 1 = 0.26263
    # ultimate: 485 / (1.4 x 204) - 1 = 0.69818 and 485 / (1.4 x 153) - 1 = 1.26424
    assert [items[i].margin for i in ids] == pytest.approx([0.51515, 0.26263, 0.69818, 1.26424], abs=1e-5)
    assert v.verdict == "pass" and v.governing == "CMPL-REQ-MOV-004-YIELD-LC-SHUTDOWN" and v.margin == pytest.approx(0.26263, abs=1e-5)
    assert v.items == ids
    assert v.statement == ("Requirement REQ-MOV-004 passes. Of the 4 compliance items, 4 pass. "
                           "Compliance item CMPL-REQ-MOV-004-YIELD-LC-SHUTDOWN decides the verdict with a margin of safety of +0.26.")

    # one item without a verdict: the requirement has no verdict, though the others pass
    _, items, verdicts = assess(workproject, *operation(res(peak, 100 * MPA), res(scls, 200 * MPA), res(peak, 120 * MPA), None))
    v = verdicts["REQ-MOV-004"]
    assert v.verdict == "not_assessed" and v.governing == "CMPL-REQ-MOV-004-ULTIMATE-LC-SHUTDOWN" and v.margin is None
    assert v.statement == ("The assessment gives no verdict for requirement REQ-MOV-004. "
                           "Of the 4 compliance items, 3 pass and 1 has no verdict. "
                           "Compliance item CMPL-REQ-MOV-004-ULTIMATE-LC-SHUTDOWN decides the verdict. "
                           "The cause is that analysis AN-SHUTDOWN has no result for von_mises.membrane_bending@scls.")

    # an inconclusive item wins over not assessed and over pass; the smallest inconclusive margin governs
    sing = res(peak, 412 * MPA, u=None, status="singular", per_level=[310 * MPA, 412 * MPA])
    notc = res(peak, 140 * MPA, u=0.06, status="not_converged")
    _, items, verdicts = assess(workproject, *operation(sing, res(scls, 200 * MPA), notc, None))
    v = verdicts["REQ-MOV-004"]
    # singular: 170 / 453.2 - 1 = -0.62489; not converged: 170 / 163.24 - 1 = +0.04141 (-0.01438 with 2 x 6 %)
    assert v.verdict == "inconclusive" and v.governing == "CMPL-REQ-MOV-004-YIELD-LC-FIRE"
    assert v.margin == pytest.approx(-0.62489, abs=1e-5)
    assert v.statement == ("The verdict for requirement REQ-MOV-004 is inconclusive. "
                           "Of the 4 compliance items, 1 passes, 2 are inconclusive and 1 has no verdict. "
                           "Compliance item CMPL-REQ-MOV-004-YIELD-LC-FIRE decides the verdict with a margin of safety of -0.62. "
                           "The cause is that the result is a singular peak.")

    # a fail wins over everything; the governing item is the fail, not the smaller inconclusive margin
    _, items, verdicts = assess(workproject, *operation(sing, res(scls, 200 * MPA), notc, res(scls, 400 * MPA)))
    v = verdicts["REQ-MOV-004"]
    # ultimate in shutdown: 485 / (1.4 x 408) - 1 = -0.15091; safe side 392 MPa: 485 / 548.8 - 1 = -0.11625: fail
    assert v.verdict == "fail" and v.governing == "CMPL-REQ-MOV-004-ULTIMATE-LC-SHUTDOWN"
    assert v.margin == pytest.approx(-0.15091, abs=1e-5)
    assert v.statement == ("Requirement REQ-MOV-004 fails. Of the 4 compliance items, 1 passes, 1 fails and 2 are inconclusive. "
                           "Compliance item CMPL-REQ-MOV-004-ULTIMATE-LC-SHUTDOWN decides the verdict with a margin of safety of -0.15. "
                           "The cause is that the margin of safety is negative.")


def test_criterion_that_no_load_case_serves_is_not_assessed(workproject):
    cases = [case("LC-FIRE", [OP_YIELD], "PH-FIRE", "thermo_structural")]         # the ultimate criterion has no load case
    out, items, verdicts = assess(workproject, cases, [analysis("LC-FIRE", [res("von_mises.peak@part", 100 * MPA)])])
    v = verdicts["REQ-MOV-004"]
    assert [i.verdict for i in out.items] == ["pass"] and v.verdict == "not_assessed" and v.governing is None
    assert v.statement == ("The assessment gives no verdict for requirement REQ-MOV-004. No load case serves criterion ultimate.")
    assert verdicts["REQ-MOV-002"].verdict == "not_assessed" and verdicts["REQ-MOV-002"].items == []
    assert verdicts["REQ-MOV-002"].statement == ("The assessment gives no verdict for requirement REQ-MOV-002. "
                                                 "No load case serves criterion yield.")


def test_requirements_that_analysis_does_not_assess(workproject):
    nv = finding("REQ-MOV-009", "not-verifiable", "major")
    out, items, verdicts = assess(workproject, [case("LC-PROOF", [PROOF])], [], findings=[nv, finding("t_fluid_min", "lox", "major")])
    assert [v.requirement for v in out.verdicts] == [r.id for r in workproject.reqs()]          # one verdict each, in order
    for rid in ("REQ-MOV-001", "REQ-MOV-005", "REQ-MOV-009"):
        v = verdicts[rid]
        assert (v.verdict, v.margin, v.governing, v.items) == ("not_assessed", None, None, [])
    assert verdicts["REQ-MOV-001"].statement == ("The verification method of requirement REQ-MOV-001 is review. "
                                                 "The analysis gives no verdict for this requirement.")
    assert verdicts["REQ-MOV-005"].statement.endswith("Finding FND-T-FLUID-MIN-LOX applies to this requirement.")
    assert verdicts["REQ-MOV-009"].statement == ("Requirement REQ-MOV-009 has no measurable criterion. "
                                                 "The analysis cannot give a verdict for it. "
                                                 "Finding FND-REQ-MOV-009-NOT-VERIFIABLE gives the cause.")
    # without the finding of the validation, the statement stands on its own
    workproject.stage_path("validate").unlink()
    v = {x.requirement: x for x in compliance.assess(workproject, context.build(workproject)).verdicts}["REQ-MOV-009"]
    assert v.statement == "Requirement REQ-MOV-009 has no measurable criterion. The analysis cannot give a verdict for it."
    # a test requirement with a criterion and a load case still gets no verdict from analysis
    workproject.requirement("REQ-MOV-002").verification = "test"
    out, _, verdicts = assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)])])
    assert out.items == [] and verdicts["REQ-MOV-002"].verdict == "not_assessed"
    assert verdicts["REQ-MOV-002"].statement.startswith("The verification method of requirement REQ-MOV-002 is test.")


# --------------------------------------------------------------------------- journal, text, stage file


def test_journal_records_the_assumption_and_the_decisions(workproject):
    cases, ans = operation(res("von_mises.peak@part", 100 * MPA), res(MB + "scls", 200 * MPA), None, None)
    out, _, _ = assess(workproject, cases, ans)
    j = out.journal
    assert [a.id for a in j.assumptions] == ["ASM-CMP-MOV-BODY-FACTOR-ON-THERMAL-STRESS"]
    asm = j.assumptions[0]
    assert asm.stage == "assess" and asm.origin.by == "engine:compliance" and asm.subject == "CMP-MOV-BODY"
    assert asm.statement == ("The factor of safety applies to the total stress of a thermo-structural load case. "
                             "The total stress includes the thermal stress.")
    assert "This is the safe side." in asm.impact and "This is the unsafe side." in asm.impact and asm.conservative is None
    assert {(l.rel, l.to) for l in asm.refs()} >= {("about", "LC-FIRE"), ("about", "LC-SHUTDOWN"), ("about", "REQ-MOV-004")}
    assert [d.id for d in j.decisions] == ["DEC-CMP-MOV-BODY-UNCERTAINTY-IN-MARGIN"]
    dec = j.decisions[0]
    assert dec.choice.startswith("The margin of safety uses each result with its numerical uncertainty on the unsafe side.")
    assert "5 %" in dec.rationale and len(dec.alternatives) == 2 and not j.findings
    # no thermo-structural load case: no assumption about thermal stress
    out, _, _ = assess(workproject, [case("LC-PROOF", [PROOF])], [analysis("LC-PROOF", [res(MB + "scls", 71.3 * MPA)])])
    assert not out.journal.assumptions and len(out.journal.decisions) == 1
    # nothing to assess: nothing to decide
    out, _, _ = assess(workproject, [], [])
    assert not out.items and not out.journal.decisions and not out.journal.assumptions


def everything(project, ctx):
    """One assessment that goes through most branches."""
    def loaded(id, refs, phase, analysis_type="static_structural", **kw):
        return case(id, refs, phase, analysis_type, loads=ctx.loads(project, None, phase), **kw)

    cases = [loaded("LC-PROOF", [PROOF], "PH-PROOF"),
             loaded("LC-BURST", [BURST], "PH-BURST", scaled_from="LC-PROOF", scale=2.5 / 1.5, status="scaled"),
             loaded("LC-TH-FIRE", [TEMP], "PH-FIRE", "thermal_steady"),
             loaded("LC-FIRE", [OP_YIELD, OP_ULT], "PH-FIRE", "thermo_structural", uses=["LC-TH-FIRE"]),
             loaded("LC-PRESTART", [OP_YIELD, OP_ULT, ROUND], "PH-PRESTART", "thermo_structural"),
             loaded("LC-SHUTDOWN", [OP_YIELD, OP_ULT], "PH-SHUTDOWN", "thermo_structural"),
             loaded("LC-CHILL", [OP_YIELD, OP_ULT], "PH-CHILL", "thermo_structural"),
             loaded("LC-FLOW-FIRE", [DROP], "PH-FIRE", "flow"),
             case("LC-FATIGUE", [LIFE], analysis_type="fatigue", uses=["LC-PROOF", "LC-FIRE"])]
    ts = dict(analysis_type="thermo_structural")
    ans = [analysis("LC-PROOF", [res(MB + "scls", 168 * MPA)], checks=[check("CHK-EQUILIBRIUM", True), check("CHK-HC-FLOOR", None, "hand_calc")]),
           analysis("LC-BURST", lines([96, 60, 150, None, 20])),
           analysis("LC-TH-FIRE", [temperature(251.3, uncertainty=0.016, uncertainty_abs=4.0)], analysis_type="thermal_steady"),
           analysis("LC-FIRE", [res("von_mises.peak@part", 412 * MPA, u=None, status="singular", per_level=[310 * MPA, 412 * MPA],
                                    near="the edge between FT-SEAT-OUT-SHOULDER and FT-OUTLET-BORE"), res(MB + "scls", 400 * MPA)], **ts),
           analysis("LC-PRESTART", [res("von_mises.peak@part", 100 * MPA, u=0.08, status="not_converged"), res(MB + "scls", 90 * MPA),
                                    res("radial_displacement.range@FT-SEAT-OUT-WALL", 12.3 * MPA)],
                    checks=[check("CHK-EQUILIBRIUM", False, statement="The reactions are 12 % smaller than the applied loads.")], **ts),
           analysis("LC-SHUTDOWN", status="failed", message="ValueError: the mesh has no group FT-NECK", **ts),
           analysis("LC-FLOW-FIRE", status="not_supported", analysis_type="flow", message="OpenFOAM is not on this computer."),
           analysis("LC-FATIGUE", [res("damage.total@part", 0.0, unit="", u=None, method="derived")],
                    analysis_type="fatigue")]
    fnd = [finding("REQ-MOV-001", "meop"), finding("REQ-MOV-004", "yield-factor"), finding("REQ-MOV-009", "not-verifiable")]
    return assess(project, cases, ans, [hand_calc("LC-FLOW-FIRE", warnings=["The loss coefficient K is an estimate."])], fnd)


def test_text_follows_the_ste_rules(workproject, example_ctx):
    out, items, verdicts = everything(workproject, example_ctx)
    assert {i.verdict for i in out.items} == {"pass", "fail", "inconclusive", "not_assessed"}
    assert {v.verdict for v in out.verdicts} == {"pass", "fail", "inconclusive", "not_assessed"}
    problems = [(where, kind, s) for where, text, n in texts(out) for kind, s in ste_problems(text, n)]
    assert not problems, problems
    for i in out.items:                                          # every verdict has its reason in words
        assert i.statement and i.reasons and i.title
    for d in out.journal.decisions:
        assert not ste_problems(d.title) and d.rationale
    # the checker itself finds what it must find
    assert ste_problems("The result is converged.") and ste_problems("The margin should be positive.")
    assert ste_problems("The engine is using the value.") and ste_problems(" ".join(["word"] * 26) + ".")
    assert not ste_problems('The check says: "The solver was stopped and should be run again."')


def test_stage_file_round_trip(workproject, example_ctx):
    out, items, verdicts = everything(workproject, example_ctx)
    summary = pipeline.stage_assess(workproject, pipeline.Options())
    assert summary["REQ-MOV-002"] == "inconclusive (margin -0.01)" and summary["REQ-MOV-009"] == "not_assessed"
    assert summary["REQ-MOV-006"] == "pass"                      # a damage of zero: no finite margin to print

    def no_constant(c):                                          # NaN and Infinity are not JSON
        raise AssertionError(f"{c} in the stage file")

    json.loads(workproject.stage_path("assess").read_text(), parse_constant=no_constant)
    assert workproject.collection("compliance") == out.items and workproject.collection("verdicts") == out.verdicts
    records = workproject.records()
    assert {i.id for i in out.items} <= set(records) and len({i.id for i in out.items}) == len(out.items)
    missing = [(i.id, l.rel, l.to) for i in out.items for l in i.refs()
               if l.to not in records and l.to not in workproject.inputs() and l.to not in example_ctx.params]
    assert not missing, missing                                  # every link of an item points to a record
    # the stage is repeatable: the same records, the same ids
    again = compliance.assess(workproject, context.build(workproject))
    assert again.items == out.items and again.verdicts == out.verdicts
    assert [d.id for d in again.journal.decisions] == [d.id for d in out.journal.decisions]


# --------------------------------------------------------------------------- review fixes: honest verdicts


def test_converged_fe_result_without_an_uncertainty_is_inconclusive(workproject):
    item = one(workproject, PROOF, res(MB + "scls", 150 * MPA, u=None))           # method fe
    assert item.verdict == "inconclusive"
    assert "The method of the result is fe. Only a hand calculation or a derived result can decide a criterion " \
           "without an uncertainty." in item.reasons
    assert one(workproject, PROOF, res(MB + "scls", 150 * MPA, u=None, method="hand_calc")).verdict == "pass"


def test_not_converged_without_an_error_estimate_is_inconclusive(workproject):
    # 116.3 MPa, u 6.9 %: with an error estimate it passes with 2 x u (MS +2.66); as a spread of values that do
    # not converge it is no bound, and the verdict is inconclusive
    item = one(workproject, BURST, res("von_mises.membrane@scls", 116.3 * MPA, u=0.069, status="not_converged",
                                       error_estimated=False))
    assert item.verdict == "inconclusive" and item.margin == pytest.approx(2.90108, abs=1e-5)
    assert "The result has no error estimate. The values of the mesh study do not converge." in item.reasons
    assert item.statement.endswith("The verdict is inconclusive because the result has no error estimate.")


def test_the_margin_shown_is_the_margin_that_decides(workproject):
    _, items, verdicts = assess(workproject, [case("LC-BURST", [BURST])],
                                [analysis("LC-BURST", [res("von_mises.membrane@scls", 116.3 * MPA, u=0.069,
                                                           status="not_converged")])])
    item = next(iter(items.values()))
    # 2 x: 116.3 x 1.138 = 132.3494 MPa; MS = 485 / 132.3494 - 1 = +2.66455 decides; 1 x gives +2.90108
    assert item.verdict == "pass" and item.margin == pytest.approx(2.66455, abs=1e-5)
    assert verdicts["REQ-MOV-003"].margin == pytest.approx(2.66455, abs=1e-5)
    assert "The margin of safety is +2.90 with the uncertainty on the unsafe side." in item.statement
    assert "The margin that decides the verdict is +2.66" in item.statement
