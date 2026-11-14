"""The fatigue engine (dossier/analysis/fatigue.py).

Every formula is checked against a hand calculation that is written out in the test. The stress fields
are written by the test, per the contract of the structural engine (fields_L<k>.npz with x and stress),
so each answer follows from a few numbers.

The test material (not the example's): yield 400 MPa, ultimate 600 MPa, sigma_f = 800 MPa, b = -0.2.
With these, the Basquin relation S_ar = sigma_f (2N)^b gives  N = 1/2 (S_ar / 800 MPa)^-5,  so the
damage of n cycles is  D = n / N = 2 n (S_ar / 800 MPa)^5.  A zero-to-200 MPa uniaxial cycle has
S_a = 100 MPa and a mean stress of 100 MPa:
    no correction   S_ar = 100 MPa          N = 1/2 * 8^5 = 16384            D(1000) = 0.06103515625
    Goodman         S_ar = 100 / (1 - 100/600) = 120 MPa   N = 6584.362...   D(1000) = 2000 * 0.15^5 = 0.151875
    SWT             S_ar = sqrt(200 * 100) = 141.42 MPa                      D(1000) = 0.345266983
"""

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from dossier import context, model, pipeline, units
from dossier.analysis import fatigue
from dossier.pipeline import Options
from dossier.project import Project

EXAMPLE = Path(__file__).resolve().parents[2] / "dossier" / "examples" / "mov_valve"
SIGMA_F, SU, SY, POWER = 800.0, 600.0, 400.0, 5         # MPa, MPa, MPa, -1/b


def pa(mpa: float) -> float:
    return units.si(units.Q(float(mpa), "MPa"))


def uniaxial(*sxx_mpa: float) -> np.ndarray:
    """One node per value, with sxx only."""
    s = np.zeros((len(sxx_mpa), 6))
    s[:, 0] = [pa(v) for v in sxx_mpa]
    return s


def goodman(sa: float, sm: float) -> float:
    return sa / (1.0 - max(sm, 0.0) / SU)


def damage(n: float, sar_mpa: float) -> float:
    """The hand formula of the test material: D = 2 n (S_ar / 800 MPa)^5."""
    return 2.0 * n * (sar_mpa / SIGMA_F) ** POWER


class Bench:
    """A copy of the example project in which the test writes the stress fields of the structural analyses."""

    def __init__(self, root: Path, mean_stress: str = "goodman", endurance: str | None = None, fatigue_data: bool = True,
                 sy: str = "400 MPa", su: str = "600 MPa", sigma_f: str = "800 MPa", b: float = -0.2):
        dst = root / "mov_valve"
        shutil.copytree(EXAMPLE, dst, ignore=shutil.ignore_patterns("out", "__pycache__", "reviews.yaml"))
        self.project = Project.load(dst)
        mat = self.project.material_of()
        mat.yield_strength, mat.ultimate_strength = model.Value.model_validate(sy), model.Value.model_validate(su)
        mat.fatigue = model.Fatigue(sigma_f=sigma_f, b=b, mean_stress=mean_stress, endurance_limit=endurance,
                                    source="Test data.") if fatigue_data else None
        self.ctx = context.build(self.project)
        self.cases: dict[str, model.LoadCase] = {}
        self.geo: dict = {"edges": {"reentrant": []}, "faces": {}, "features": {}}

    def state(self, lc_id: str, stress=None, x=None, scaled_from: str | None = None, scale: float = 1.0, bcs=(),
              mesh=(), status: str = "done", record: bool = True, loads=()):
        """A structural load case. `stress` is an (n, 6) array (mesh level 1) or {level: array}; `x` the same
        for the node coordinates (default: nodes 10 mm apart on the x axis)."""
        self.cases[lc_id] = model.LoadCase(
            id=lc_id, analysis_type="static_structural", component=self.project.subject.id, loads=list(loads),
            scaled_from=scaled_from, scale=scale, status="scaled" if scaled_from else "selected")
        if scaled_from or not record:
            return
        d = self.project.analysis_dir(lc_id)
        d.mkdir(parents=True, exist_ok=True)
        fields = stress if isinstance(stress, dict) else {1: stress}
        arts = []
        for k, s in fields.items():
            s = np.asarray(s, dtype=float)
            xk = x[k] if isinstance(x, dict) else x
            xk = np.asarray(xk, dtype=float) if xk is not None else np.array([[0.01 * i, 0.0, 0.0] for i in range(len(s))])
            np.savez(d / f"fields_L{k}.npz", x=xk, stress=s, u=np.zeros_like(xk))
            arts.append(model.Artifact(path=f"analyses/{lc_id}/fields_L{k}.npz", kind="field"))
        rec = model.AnalysisRecord(id="AN-" + lc_id[3:], load_case=lc_id, analysis_type="static_structural",
                                   status=status, artifacts=arts, bcs=list(bcs), mesh=list(mesh))
        self.project.write_stage("analyze", d / "analysis.json", analyses=[rec])

    def run(self, cycles: list[tuple[str, float, list[str]]], lc_id: str = "LC-FATIGUE"):
        """Run the engine for duty cycles given as (duty cycle id, repetitions, state load case ids)."""
        lc = model.LoadCase(id=lc_id, title="Fatigue analysis of the test", analysis_type="fatigue",
                            component=self.project.subject.id,
                            cycles=[model.CycleUse(duty_cycle=d, count=n, states=s) for d, n, s in cycles],
                            uses=sorted({s for _, _, st in cycles for s in st}))
        self.project.write_stage("plan", load_cases=[*self.cases.values(), lc])
        self.lc = lc
        out = fatigue.run(self.project, self.ctx, self.geo, lc, self.project.analysis_dir(lc.id),
                          Options(quick=True, log=lambda _msg: None))
        self.data = None
        path = self.project.analysis_dir(lc.id) / "fatigue.json"
        if out.record.status == "done":
            self.data = json.loads(path.read_text())
        return out


@pytest.fixture()
def bench(tmp_path):
    return Bench(tmp_path)


def part(out) -> model.ResultValue:
    return out.record.results["damage.total@part"]


def smooth(out) -> model.ResultValue:
    return out.record.results["damage.total@smooth"]


def check(out, key: str) -> model.Check:
    return next(c for c in out.record.checks if c.id == f"CHK-LC-FATIGUE-{key}")


# =========================================================================== the relations, one at a time


def test_von_mises_is_the_invariant_of_the_tensor():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(50, 6)) * 1e8
    for row in s:                                   # against the principal stresses of the 3 x 3 tensor
        xx, yy, zz, xy, yz, xz = row
        p1, p2, p3 = np.linalg.eigvalsh(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
        hand = math.sqrt(0.5 * ((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2))
        assert float(fatigue.von_mises(row)) == pytest.approx(hand, rel=1e-12)
    assert float(fatigue.von_mises([200.0, 0, 0, 0, 0, 0])) == pytest.approx(200.0)             # uniaxial
    assert float(fatigue.von_mises([0, 0, 0, 0, 100.0, 0])) == pytest.approx(100.0 * math.sqrt(3))  # pure shear
    assert float(fatigue.von_mises([-70.0, -70.0, -70.0, 0, 0, 0])) == pytest.approx(0.0)       # pressure only


def test_basquin_zero_to_maximum_by_hand(tmp_path):
    """sxx cycles 0 -> 200 MPa, 1000 times, no mean stress correction: S_a = 100 MPa = sigma_f / 8,
    2N = 8^5 = 32768, N = 16384, D = 1000 / 16384."""
    b = Bench(tmp_path, mean_stress="none")
    b.state("LC-PROOF", uniaxial(200.0, 50.0))
    out = b.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    assert out.record.status == "done" and out.record.id == "AN-FATIGUE"
    assert part(out).value == pytest.approx(1000 / 16384, rel=1e-12) == pytest.approx(0.06103515625)
    assert part(out).location == [0.0, 0.0, 0.0] and part(out).unit == "" and part(out).method == "derived"
    (cyc,) = b.data["part"]["cycles"]
    assert (cyc["state_a"], cyc["state_b"], cyc["count"]) == ("the unloaded state", "LC-PROOF", 1000)
    assert cyc["range"] == pytest.approx(pa(200)) and cyc["amplitude"] == pytest.approx(pa(100))
    assert cyc["mean_stress"] == pytest.approx(pa(100)) and cyc["corrected_amplitude"] == pytest.approx(pa(100))
    assert cyc["cycles_to_failure"] == pytest.approx(16384) and cyc["share"] == pytest.approx(1.0)
    # the second node has a quarter of the stress: 4^5 = 1024 times less damage
    z = np.load(b.project.analysis_dir("LC-FATIGUE") / "damage_L1.npz")
    assert z["damage"] == pytest.approx([1000 / 16384, 1000 / 16384 / 1024])


def test_goodman_swt_and_no_credit_for_compression_by_hand(tmp_path):
    """0 -> 200 MPa: S_a = 100 MPa, mean 100 MPa. Goodman: 100 / (1 - 100/600) = 120 MPa, D = 2000 * 0.15^5.
    SWT: sqrt(200 * 100) MPa. 0 -> -200 MPa: the mean of -100 MPa counts as zero, S_ar = 100 MPa, N = 16384."""
    hand = {"goodman": 2000 * 0.15 ** 5, "swt": 2000 * (math.sqrt(200 * 100) / 800) ** 5, "none": 1000 / 16384}
    assert hand["goodman"] == pytest.approx(0.151875) and hand["swt"] == pytest.approx(0.345266983)
    for rule, expect in hand.items():
        b = Bench(tmp_path / rule, mean_stress=rule)
        b.state("LC-PROOF", uniaxial(200.0))
        b.state("LC-LEAK", uniaxial(-200.0))
        out = b.run([("DC-PROOF", 1000, ["LC-PROOF"])])
        assert part(out).value == pytest.approx(expect, rel=1e-12)
        if rule == "goodman":
            (cyc,) = b.data["part"]["cycles"]
            assert cyc["corrected_amplitude"] == pytest.approx(pa(120))
            assert cyc["cycles_to_failure"] == pytest.approx(6584.362139917695)
        out = b.run([("DC-LEAK", 1000, ["LC-LEAK"])])              # compression
        (cyc,) = b.data["part"]["cycles"]
        assert cyc["mean_stress"] == pytest.approx(pa(-100)) and cyc["corrected_amplitude"] == pytest.approx(pa(100))
        assert part(out).value == pytest.approx(1000 / 16384, rel=1e-12)


def test_mean_stress_is_the_first_invariant_not_a_von_mises_stress(bench):
    """Equibiaxial 0 -> (100, 100) MPa: the von Mises range is 100 MPa (S_a = 50 MPa) and the mean stress is
    1/2 (100 + 100) = 100 MPa. Pure shear 0 -> 100 MPa: range 100 sqrt(3) MPa and no mean stress."""
    s = np.zeros((2, 6))
    s[0, 0] = s[0, 1] = pa(100)
    s[1, 3] = pa(100)
    bench.state("LC-PROOF", s)
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    biaxial = damage(1000, goodman(50, 100))
    shear = damage(1000, 50 * math.sqrt(3))
    z = np.load(bench.project.analysis_dir("LC-FATIGUE") / "damage_L1.npz")
    assert z["damage"] == pytest.approx([biaxial, shear], rel=1e-12)
    assert part(out).value == pytest.approx(shear) and part(out).location == [0.01, 0.0, 0.0]


def test_miner_sum_over_two_duty_cycles(bench):
    """1000 cycles 0 -> 200 MPa: D = 0.151875. 20000 cycles 0 -> 100 MPa: S_a = 50, mean 50,
    S_ar = 50 / (1 - 50/600) = 600/11 MPa = 3/44 sigma_f, D = 40000 (3/44)^5 = 0.058939016. Sum 0.210814016."""
    bench.state("LC-PROOF", uniaxial(200.0))
    bench.state("LC-LEAK", uniaxial(100.0))
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"]), ("DC-LEAK", 20000, ["LC-LEAK"])])
    d1, d2 = 2000 * 0.15 ** 5, 40000 * (3 / 44) ** 5
    assert d2 == pytest.approx(0.058939016, rel=1e-8)
    assert part(out).value == pytest.approx(d1 + d2, rel=1e-12) == pytest.approx(0.210814016, rel=1e-8)
    rows = {r["duty_cycle"]: r for r in bench.data["part"]["duty_cycles"]}
    assert rows["DC-PROOF"]["damage"] == pytest.approx(d1) and rows["DC-LEAK"]["damage"] == pytest.approx(d2)
    assert rows["DC-PROOF"]["damage_alone"] == pytest.approx(d1) and rows["DC-LEAK"]["range"] == pytest.approx(pa(100))
    assert rows["DC-PROOF"]["share"] == pytest.approx(d1 / (d1 + d2)) and rows["DC-LEAK"]["share"] == pytest.approx(d2 / (d1 + d2))
    # the states load the node in the same direction: the simple count gives the same number
    assert bench.data["part"]["one_cycle_per_repetition"] == pytest.approx(d1 + d2)
    assert "Duty cycle DC-LEAK has 20000 repetitions." in out.record.statement


def test_cycles_between_two_duty_cycles_are_counted(bench):
    """Two duty cycles load the node in opposite directions: 0 -> +200 MPa 1000 times, 0 -> -200 MPa 400 times.
    Their order is not known. When they alternate the stress goes +200 -> -200 MPa, a cycle of range 400 MPa
    with no mean stress: 400 cycles with S_ar = 200 MPa, D = 800 * 0.25^5 = 0.78125. The other 600 tension
    cycles: 1200 * 0.15^5 = 0.091125. Each duty cycle alone gives only 0.151875 + 400/16384 = 0.17629."""
    bench.state("LC-PROOF", uniaxial(200.0))
    bench.state("LC-LEAK", uniaxial(-200.0))
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"]), ("DC-LEAK", 400, ["LC-LEAK"])])
    assert part(out).value == pytest.approx(0.78125 + 0.091125, rel=1e-12)
    first, second = bench.data["part"]["cycles"]
    assert {first["state_a"], first["state_b"]} == {"LC-PROOF", "LC-LEAK"} and first["count"] == 400
    assert first["range"] == pytest.approx(pa(400)) and first["mean_stress"] == pytest.approx(0.0)
    assert {second["state_a"], second["state_b"]} == {"the unloaded state", "LC-PROOF"} and second["count"] == 600
    assert bench.data["part"]["one_cycle_per_repetition"] == pytest.approx(0.151875 + 400 / 16384)
    rows = {r["duty_cycle"]: r for r in bench.data["part"]["duty_cycles"]}       # the cycle between them: half each
    assert rows["DC-LEAK"]["damage"] == pytest.approx(0.78125 / 2)
    assert rows["DC-PROOF"]["damage"] == pytest.approx(0.78125 / 2 + 0.091125)
    assert "0.176" in out.record.statement       # the reader sees what the simple count gives


def test_secondary_cycle_in_a_duty_cycle(bench):
    """One duty cycle 0 -> +180 -> -20 -> +160 -> 0 MPa. A rainflow count of this loop gives two cycles:
    180 <-> -20 (range 200, mean 80) and 160 <-> 0 (range 160, mean 80). The max-min count gives the same.
    Goodman: 100 / (1 - 80/600) = 1500/13 MPa = 15/104 sigma_f, and 80 / (13/15) = 1200/13 MPa = 3/26 sigma_f."""
    bench.state("LC-CHILL", uniaxial(180.0))
    bench.state("LC-PRESTART", uniaxial(-20.0))
    bench.state("LC-FIRE", uniaxial(160.0))
    out = bench.run([("DC-RUN", 1000, ["LC-CHILL", "LC-PRESTART", "LC-FIRE"])])
    main, second = 2000 * (15 / 104) ** 5, 2000 * (3 / 26) ** 5
    assert (main, second) == pytest.approx((0.1248302, 0.04090437), rel=1e-6)
    assert part(out).value == pytest.approx(main + second, rel=1e-12)
    assert [c["range"] for c in bench.data["part"]["cycles"]] == pytest.approx([pa(200), pa(160)])
    assert bench.data["part"]["one_cycle_per_repetition"] == pytest.approx(main)     # the simple count misses one


def test_max_min_count_against_a_rainflow_count_by_hand():
    """The loop 0, 10, 2, 8, 1, 9, 0 (ASTM E1049 rainflow, started at the largest peak): cycles 2-8 (range 6),
    1-9 (range 8), 0-10 (range 10). The max-min count pairs 10-0, 9-1, 8-2: the same three ranges."""
    sn = fatigue.SN(sigma_f=pa(800), b=-0.2, rule="none")
    values = [0.0, 10.0, 2.0, 8.0, 1.0, 9.0]
    cycles = fatigue.count_cycles(uniaxial(*values), [3.0] * 6, sn)
    assert [(values[c["a"]], values[c["b"]]) for c in cycles] == [(0.0, 10.0), (1.0, 9.0), (2.0, 8.0)]
    assert [c["range"] for c in cycles] == pytest.approx([pa(10), pa(8), pa(6)]) and {c["count"] for c in cycles} == {3.0}


def test_non_proportional_states_use_the_tensor_difference(bench):
    """A duty cycle 0 -> A -> B -> 0 with A: sxx = 200 MPa, B: syy = 200 MPa. Both states have a von Mises
    stress of 200 MPa, so the difference of the von Mises stresses of A and B is zero. The stress difference
    A - B is (200, -200, 0) MPa with a von Mises stress of sqrt(1/2 (400^2 + 200^2 + 200^2)) = 346.41 MPa.
    S_a = 173.2 MPa, mean 1/2 (200 + 200) = 200 MPa, Goodman: 173.2 / (1 - 200/600) = 259.81 MPa.
    100 repetitions: D = 200 (259.81/800)^5 = 0.7225027. A range of 200 MPa would give 0.0151875."""
    a, b_ = np.zeros((1, 6)), np.zeros((1, 6))
    a[0, 0] = b_[0, 1] = pa(200)
    bench.state("LC-CHILL", a)
    bench.state("LC-FIRE", b_)
    out = bench.run([("DC-RUN", 100, ["LC-CHILL", "LC-FIRE"])])
    rng = math.sqrt(0.5 * (400.0 ** 2 + 200.0 ** 2 + 200.0 ** 2))
    hand = damage(100, goodman(rng / 2, 200.0))
    assert hand == pytest.approx(0.7225027, rel=1e-6)
    assert part(out).value == pytest.approx(hand, rel=1e-12)
    (cyc,) = bench.data["part"]["cycles"]
    assert {cyc["state_a"], cyc["state_b"]} == {"LC-CHILL", "LC-FIRE"}
    assert cyc["range"] == pytest.approx(pa(rng)) and cyc["mean_stress"] == pytest.approx(pa(200))
    assert part(out).value > 40 * damage(100, goodman(100, 100))      # what a range of 200 MPa would give


def test_scaled_state_uses_the_field_of_its_base_case(bench):
    """LC-LEAK is LC-PROOF times 2/3 and has no field. LC-PROOF: 0 -> 300 MPa, 10 times: S_ar = 150 / 0.75 =
    200 MPa, D = 20 * 0.25^5 = 0.01953125. LC-LEAK: 0 -> 200 MPa, 1000 times: D = 0.151875."""
    bench.state("LC-PROOF", uniaxial(300.0))
    bench.state("LC-LEAK", scaled_from="LC-PROOF", scale=2 / 3)
    out = bench.run([("DC-PROOF", 10, ["LC-PROOF"]), ("DC-LEAK", 1000, ["LC-LEAK"])])
    assert part(out).value == pytest.approx(0.01953125 + 0.151875, rel=1e-12)
    leak = next(p for p in bench.data["time_points"] if p["state"] == "LC-LEAK")
    assert leak["field_of"] == "LC-PROOF" and leak["scale"] == pytest.approx(2 / 3) and leak["repetitions"] == 1000
    assert [f["load_case"] for f in bench.data["fields"]] == ["LC-PROOF"]      # one field file, with its hash
    assert len(bench.data["fields"][0]["sha256"]) == 64
    assert {(l.rel, l.to) for l in out.record.links} >= {("uses", "AN-PROOF"), ("uses", "MAT-316L"),
                                                         ("derives_from", "DC-PROOF"), ("derives_from", "DC-LEAK")}


def test_scaling_of_a_scaled_case_and_a_circle(bench):
    bench.state("LC-BURST", uniaxial(300.0))
    bench.state("LC-PROOF", scaled_from="LC-BURST", scale=0.5)
    bench.state("LC-LEAK", scaled_from="LC-PROOF", scale=4 / 3)          # 300 * 0.5 * 4/3 = 200 MPa
    out = bench.run([("DC-LEAK", 1000, ["LC-LEAK"])])
    assert part(out).value == pytest.approx(0.151875, rel=1e-12)
    bench.cases["LC-BURST"] = model.LoadCase(id="LC-BURST", analysis_type="static_structural", component="CMP-MOV-BODY",
                                             scaled_from="LC-LEAK", scale=2.0, status="scaled")
    out = bench.run([("DC-LEAK", 1000, ["LC-LEAK"])])
    assert out.record.status == "failed" and "circle" in out.record.message


def test_cycle_floor_at_one_reversal(bench):
    """An elastic stress amplitude over sigma_f has no Basquin life: the engine uses N = 0.5, D = 2 per cycle,
    and the checks say that the stress is far out of the elastic range."""
    bench.state("LC-PROOF", uniaxial(2000.0))
    out = bench.run([("DC-PROOF", 3, ["LC-PROOF"])])
    assert part(out).value == pytest.approx(6.0) and part(out).status == "not_converged"
    assert check(out, "YIELD-PART").passed is False and "800 MPa" in check(out, "YIELD-PART").statement
    assert "0.5 cycles to failure" in out.record.statement


def test_endurance_limit_applies_only_where_every_cycle_is_below_it(tmp_path):
    """Endurance limit 105 MPa, policy uncertainty 5 %: the cut is 100 MPa. Fully reversed cycles (no mean):
    node 0 has amplitudes 150 and 90 MPa: both count, the small one on the Basquin line.
    node 1 has amplitudes 99 and 90 MPa: no damage. node 2 has 102 MPa, in the 5 % band under the limit: it counts."""
    b = Bench(tmp_path, endurance="105 MPa")
    b.state("LC-PROOF", uniaxial(150.0, 99.0, 102.0))
    b.state("LC-LEAK", uniaxial(-150.0, -99.0, -102.0))
    b.state("LC-CHILL", uniaxial(90.0, 90.0, 0.0))
    b.state("LC-FIRE", uniaxial(-90.0, -90.0, 0.0))
    out = b.run([("DC-PROOF", 100, ["LC-PROOF", "LC-LEAK"]), ("DC-RUN", 1000, ["LC-CHILL", "LC-FIRE"])])
    z = np.load(b.project.analysis_dir("LC-FATIGUE") / "damage_L1.npz")
    assert z["damage"] == pytest.approx([damage(100, 150) + damage(1000, 90), 0.0, damage(100, 102)], rel=1e-12)
    assert part(out).value == pytest.approx(damage(100, 150) + damage(1000, 90))
    assert any(d.id == "DEC-LC-FATIGUE-ENDURANCE-LIMIT" and "100 MPa" in d.choice for d in out.journal.decisions)


def test_vectorized_count_against_a_plain_loop(monkeypatch):
    """Random stress states at 60 nodes, three duty cycles: the engine's count at every node against a
    count written with plain loops, with the von Mises stress from the principal stresses."""
    rng = np.random.default_rng(7)
    sn = fatigue.SN(sigma_f=pa(800), b=-0.2, rule="goodman", su=pa(600))
    names, reps = ["A", "B", "C", "D", "E", "F"], [65.0, 65.0, 65.0, 2.0, 200.0, 200.0]
    fields = {k: rng.normal(size=(60, 6)) * pa(60) for k in names}
    duty = ["DC-RUN", "DC-RUN", "DC-RUN", "DC-PROOF", "DC-LEAK", "DC-LEAK"]
    points = [fatigue.Point(fatigue.UNLOADED, None, None, 0.0, 467.0 - 65.0 * 2 - 200.0)]      # 65 + 2 + 200 = 267
    points += [fatigue.Point(k, d, k, 1.0, n) for k, d, n in zip(names, duty, reps)]

    def vm(t):
        p = np.linalg.eigvalsh(np.array([[t[0], t[3], t[5]], [t[3], t[1], t[4]], [t[5], t[4], t[2]]]))
        return math.sqrt(0.5 * ((p[0] - p[1]) ** 2 + (p[1] - p[2]) ** 2 + (p[2] - p[0]) ** 2))

    def plain(node):
        t = [np.zeros(6)] + [fields[k][node] for k in names]
        left, total = [p.reps for p in points], 0.0
        while True:
            pairs = [(vm(t[i] - t[j]), i, j) for i in range(7) for j in range(i + 1, 7) if left[i] > 0 and left[j] > 0]
            if not pairs:
                return total
            r, i, j = max(pairs, key=lambda q: q[0])
            n = min(left[i], left[j])
            mean = max(0.0, 0.5 * (t[i][:3].sum() + t[j][:3].sum()))
            sar = 0.5 * r / (1.0 - mean / pa(600)) if mean < pa(600) else math.inf
            total += n / max(0.5, 0.5 * (sar / pa(800)) ** -5.0)
            left[i] -= n
            left[j] -= n

    hand = np.array([plain(k) for k in range(60)])
    assert fatigue.evaluate(fields, points, sn).damage == pytest.approx(hand, rel=1e-10)
    monkeypatch.setattr(fatigue, "PAIRS_PER_CHUNK", 100)           # 21 pairs: 4 nodes at a time
    assert fatigue.evaluate(fields, points, sn).damage == pytest.approx(hand, rel=1e-10)


# =========================================================================== the limits of the method


def test_stress_over_the_elastic_range_fails_a_check_and_cannot_converge(tmp_path):
    """The example's material: yield 170 MPa. Node 0: 0 -> 200 MPa, over the yield strength but under the
    shakedown limit of 340 MPa. Node 1: 0 -> 400 MPa, over both. The two meshes agree, so only the
    validity checks stand between the result and the status converged."""
    b = Bench(tmp_path, sy="170 MPa", su="485 MPa", sigma_f="1000 MPa", b=-0.1)
    b.state("LC-PROOF", {1: uniaxial(200.0, 100.0), 2: uniaxial(200.0, 100.0)})
    out = b.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert check(out, "SHAKEDOWN-PART").passed is True and check(out, "YIELD-PART").passed is False
    y = check(out, "YIELD-PART")
    assert y.value == pytest.approx(pa(200)) and y.limit == pytest.approx(pa(170))
    assert "1 node of 2 is over the yield strength" in y.statement and "strain-life method" in y.statement
    assert part(out).status == "not_converged" and smooth(out).status == "not_converged"
    assert "elastic stress-life method does not apply" in part(out).note

    b = Bench(tmp_path / "b", sy="170 MPa", su="485 MPa", sigma_f="1000 MPa", b=-0.1)
    b.state("LC-PROOF", {1: uniaxial(200.0, 400.0), 2: uniaxial(200.0, 400.0)})
    out = b.run([("DC-PROOF", 2, ["LC-PROOF"])])
    s = check(out, "SHAKEDOWN-PART")
    assert s.passed is False and s.value == pytest.approx(pa(400)) and s.limit == pytest.approx(pa(340))
    assert "does not apply at these nodes" in s.statement and "plasticity correction" in s.statement
    assert part(out).status == "not_converged"

    b = Bench(tmp_path / "c", sy="170 MPa", su="485 MPa", sigma_f="1000 MPa", b=-0.1)      # all elastic: it converges
    b.state("LC-PROOF", {1: uniaxial(150.0, 100.0), 2: uniaxial(150.0, 100.0)})
    out = b.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert check(out, "SHAKEDOWN-PART").passed and check(out, "YIELD-PART").passed
    assert part(out).status == "converged" and part(out).uncertainty == pytest.approx(0.0)
    # by hand, with the example's Basquin line: S_a = 75, Goodman 75 / (1 - 75/485) = 88.72 MPa, 2N = (88.72/1000)^-10
    sar = 75.0 / (1.0 - 75.0 / 485.0)
    assert part(out).value == pytest.approx(2 / (0.5 * (sar / 1000.0) ** -10.0), rel=1e-10)


def test_convergence_between_the_two_finest_meshes(bench):
    """The fine mesh gives 200 MPa where the coarse mesh gives 192.3 MPa (4 % more): the stress range changes
    by 3.85 %, under the policy's 5 %, so the result is converged. The damage changes far more:
    Goodman 120 MPa against 96.15 / (1 - 96.15/600) = 114.50 MPa, to the power 5: a factor of 1.264."""
    coarse = 200.0 / 1.04
    bench.state("LC-PROOF", {1: uniaxial(coarse, 20.0), 2: uniaxial(200.0, 20.0)})
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    r = part(out)
    factor = (120.0 / goodman(coarse / 2, coarse / 2)) ** 5
    assert factor == pytest.approx(1.264, abs=1e-3)
    assert r.status == "converged" and r.value == pytest.approx(0.151875)
    assert r.per_level == pytest.approx([0.151875 / factor, 0.151875])
    assert r.uncertainty == pytest.approx(factor - 1.0)             # the uncertainty of the damage, not of the stress
    c = check(out, "CONVERGENCE-PART")
    assert c.passed is True and c.value == pytest.approx(1 - 1 / 1.04) and c.limit == pytest.approx(0.05)
    assert "factor of 1.26" in r.note and "power 5" in r.note

    bench.state("LC-PROOF", {1: uniaxial(200.0 / 1.10, 20.0), 2: uniaxial(200.0, 20.0)})      # 10 % more: not converged
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    assert part(out).status == "not_converged" and check(out, "CONVERGENCE-PART").passed is False
    assert part(out).uncertainty == pytest.approx((120.0 / goodman(100 / 1.1, 100 / 1.1)) ** 5 - 1.0)


def test_convergence_compares_the_nearest_node_of_the_coarser_mesh(bench):
    """The meshes share no nodes. The critical node of the fine mesh is at x = 30 mm; the coarse mesh has
    nodes at 0, 28 and 60 mm. The comparison takes the node at 28 mm (180 MPa), not the coarse maximum."""
    x = {1: [[0.0, 0, 0], [0.028, 0, 0], [0.06, 0, 0]], 2: [[0.0, 0, 0], [0.01, 0, 0], [0.03, 0, 0], [0.05, 0, 0]]}
    bench.state("LC-PROOF", {1: uniaxial(100.0, 180.0, 190.0), 2: uniaxial(100.0, 150.0, 200.0, 120.0)}, x=x)
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    cv = bench.data["part"]["convergence"]
    assert part(out).location == [0.03, 0.0, 0.0]
    assert cv["node_distance"] == pytest.approx(0.002) and cv["range_coarse"] == pytest.approx(pa(180))
    assert cv["range_change"] == pytest.approx(0.10) and part(out).status == "not_converged"
    # the uncertainty is the larger of: the damage at the node (180 MPa), the largest damage of the mesh (190 MPa)
    assert part(out).uncertainty == pytest.approx((120.0 / goodman(90, 90)) ** 5 - 1.0)
    assert part(out).per_level == pytest.approx([damage(1000, goodman(95, 95)), 0.151875])


def test_one_mesh_gives_no_uncertainty_and_is_not_converged(bench):
    bench.state("LC-PROOF", uniaxial(200.0))
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    assert part(out).status == "not_converged" and part(out).uncertainty is None and len(part(out).per_level) == 1
    assert check(out, "CONVERGENCE-PART").passed is None and "one mesh only" in part(out).note


# =========================================================================== singular zones


def _hand_mesh(path: Path) -> np.ndarray:
    """Two faces of linear triangles that meet at a sharp edge, and a held strip, 1 mm node spacing:
    f1: the plane z = 0, x and y from 0 to 10 mm.   f2: the plane x = 0, z from 0 to 10 mm.
    f3: the plane x = 10 mm, z from 0 to 3 mm.      The edge of f1 and f2 is the line x = 0, z = 0."""
    import meshio

    mm = np.arange(11) * 1e-3
    pts, index = [], {}

    def node(x, y, z):
        key = (round(x * 1e6), round(y * 1e6), round(z * 1e6))
        if key not in index:
            index[key] = len(pts)
            pts.append([x, y, z])
        return index[key]

    def strip(at, n):
        tris = []
        for a in range(n):
            for j in range(10):
                q = [node(*at(a, j)), node(*at(a + 1, j)), node(*at(a + 1, j + 1)), node(*at(a, j + 1))]
                tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        return np.array(tris)

    f1 = strip(lambda a, j: (mm[a], mm[j], 0.0), 10)
    f2 = strip(lambda a, j: (0.0, mm[j], mm[a]), 10)
    f3 = strip(lambda a, j: (mm[10], mm[j], mm[a]), 3)
    cells = [("triangle", f1), ("triangle", f2), ("triangle", f3)]
    tags = [np.full(len(f), k + 1) for k, f in enumerate((f1, f2, f3))]
    m = meshio.Mesh(np.array(pts), cells, cell_data={"gmsh:physical": tags, "gmsh:geometrical": tags},
                    field_data={"f1": np.array([1, 2]), "f2": np.array([2, 2]), "f3": np.array([3, 2])})
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), m, file_format="gmsh22", binary=False)
    return np.array(pts)


def _hand_stress(x: np.ndarray, edge_factor: float = 1.0) -> np.ndarray:
    """sxx in MPa: 200 / sqrt(1 + r) from the edge (r in mm) plus 180 / sqrt(1 + s) from the held strip
    (s = 10 mm - x). At the edge the first term is `edge_factor` x 200 MPa: a peak that the mesh changes."""
    r = np.hypot(x[:, 0], x[:, 2]) * 1e3
    s = 10.0 - x[:, 0] * 1e3
    first = np.where(r < 0.5, edge_factor * 200.0, 200.0 / np.sqrt(1.0 + r))
    return uniaxial(*(first + 180.0 / np.sqrt(1.0 + s)))


def test_singular_zones_are_left_out_of_the_smooth_result(bench, monkeypatch):
    """Zone = 2.5 element sizes of 1 mm (nodes at 0, 1, 2 mm are in it, nodes at 3 mm are not).
    Part: the largest stress is at the edge, 200 + 180/sqrt(11) = 254.3 MPa, and it grows under refinement.
    Without the zones of the edge and of the held strip, the largest stress is at x = 3 mm on f1:
    200/sqrt(4) + 180/sqrt(8) = 163.64 MPa. At the held strip it would be 200/sqrt(11) + 180 = 240.3 MPa."""
    monkeypatch.setattr(fatigue, "EXCLUSION_ELEMENTS", 2.5)
    x = _hand_mesh(bench.project.out / "meshes" / "hand_L2.msh")
    level = model.MeshLevel(level=2, h=1e-3, nodes=len(x), cells=0, order=1, path="meshes/hand_L2.msh")
    held = model.BCItem(type="fixed", faces=["f3"], description="The strip is held.")
    free = model.BCItem(type="displacement", value={"points": [[0, 0, 0]]}, description="Three nodes stop the motion.")
    bench.state("LC-PROOF", {1: _hand_stress(x, edge_factor=0.8), 2: _hand_stress(x)}, x={1: x, 2: x},
                bcs=[held, free], mesh=[level])
    bench.geo = {"edges": {"reentrant": [{"tag": 7, "faces": [1, 2], "angle_deg": 270.0, "length": 0.010,
                                          "midpoint": [0.0, 0.005, 0.0]}]},
                 "faces": {"1": {"features": ["FT-PAD-TOP"]}, "2": {"features": []}, "3": {"features": ["FT-NECK"]}}}
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    p, s = part(out), smooth(out)

    edge = 200.0 + 180.0 / math.sqrt(11.0)
    assert p.value == pytest.approx(damage(1000, goodman(edge / 2, edge / 2)), rel=1e-10)
    assert p.status == "singular" and p.near == "re-entrant edge 7 (faces f1 and f2)"
    assert p.location[0] == 0.0 and p.location[2] == 0.0
    assert "no limit" in p.note and "increases from" in p.note

    best = 200.0 / math.sqrt(4.0) + 180.0 / math.sqrt(8.0)
    assert best == pytest.approx(163.64, abs=0.01)
    assert s.value == pytest.approx(damage(1000, goodman(best / 2, best / 2)), rel=1e-10)
    assert s.location == pytest.approx([0.003, 0.0, 0.0]) and s.near == "face f1 (FT-PAD-TOP)"
    assert s.status == "converged" and s.uncertainty == pytest.approx(0.0)       # the two meshes agree out of the zones
    assert "information only" in s.note and "2.5 mm" in s.note and "rim of the singular zone" in s.note
    assert s.value < 0.2 * p.value

    zones = bench.data["singular_zones"]
    assert [(z["kind"], z["nodes_on_it"]) for z in zones["zones"]] == [("edge", 11), ("face", 44)]
    assert [z["distance"] for z in zones["zones"]] == pytest.approx([0.0025, 0.0025])
    assert zones["mesh"] == "meshes/hand_L2.msh" and zones["supports_without_faces"] == ["AN-PROOF: displacement"]
    # left out: the lines x = 0, 1, 2 mm of f1 and z = 1, 2 mm of f2 (55 nodes), the held strip and x = 8, 9, 10 mm
    assert zones["nodes_left_out"] == 5 * 11 + 3 * 11 + 3 * 11
    c = check(out, "SINGULAR-ZONES")
    assert c.passed is True and "1 sharp re-entrant edge and 1 held face" in c.statement and "no face" in c.statement
    assert any(d.id == "DEC-LC-FATIGUE-SINGULAR-ZONES" and d.needs_review for d in out.journal.decisions)
    npz = np.load(bench.project.analysis_dir("LC-FATIGUE") / "damage_L2.npz")
    assert int((~npz["smooth"]).sum()) == zones["nodes_left_out"]


def test_singular_zones_that_cannot_be_found_block_the_status(bench):
    """The geometry names a re-entrant edge, but the state analysis gives no mesh file: the engine cannot
    tell where the edge is. No smooth value, and the part result cannot be converged."""
    bench.state("LC-PROOF", {1: uniaxial(200.0, 50.0), 2: uniaxial(200.0, 50.0)})
    bench.geo = {"edges": {"reentrant": [{"tag": 7, "faces": [1, 2], "length": 0.01, "midpoint": [0, 0.005, 0]}]}}
    out = bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    assert part(out).value == pytest.approx(0.151875) and part(out).status == "not_converged"
    assert smooth(out).value is None and smooth(out).status == "not_run"
    c = check(out, "SINGULAR-ZONES")
    assert c.passed is False and "re-entrant edge 7" in c.statement
    assert "did not find" in part(out).note and bench.data["singular_zones"]["problem"]


def test_l_bracket_meshed_with_gmsh(tmp_path):
    """A real quadratic mesh (gmsh 4.1 file, f<tag> groups) of an L-shaped bracket: the inner corner is a
    sharp re-entrant edge, one end face is held. The stress is 300 MPa (1 mm / r)^0.45 with r the distance
    from the corner line, limited at r = h/2: a peak that grows when the mesh gets finer."""
    gmsh = pytest.importorskip("gmsh")
    import meshio

    b = Bench(tmp_path)
    corner = np.array([0.010, 0.010])                        # the corner line: x = 10 mm, z = 10 mm, along y
    faces, xs = {}, {}
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        box1 = gmsh.model.occ.addBox(0, 0, 0, 0.020, 0.010, 0.010)
        box2 = gmsh.model.occ.addBox(0, 0, 0.010, 0.010, 0.010, 0.010)
        gmsh.model.occ.fuse([(3, box1)], [(3, box2)])
        gmsh.model.occ.synchronize()
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.addPhysicalGroup(2, [tag], name=f"f{tag}")
            faces[tuple(round(c, 4) for c in gmsh.model.occ.getCenterOfMass(2, tag))] = tag
        gmsh.model.addPhysicalGroup(2, [faces[(0.015, 0.005, 0.01)]], name="FT-STEP")      # a face in two groups
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        for level, h in ((1, 0.0030), (2, 0.0020)):
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeMin", h)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.model.mesh.generate(3)
            path = b.project.out / "meshes" / f"bracket_L{level}.msh"
            path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(path))
            xs[level] = np.asarray(meshio.read(str(path)).points, dtype=float)
    finally:
        gmsh.finalize()

    def field(x, h):
        r = np.hypot(x[:, 0] - corner[0], x[:, 2] - corner[1])
        return uniaxial(*(300.0 * (1e-3 / np.maximum(r, h / 2)) ** 0.45))

    step, wall, end = faces[(0.015, 0.005, 0.01)], faces[(0.01, 0.005, 0.015)], faces[(0.02, 0.005, 0.005)]
    levels = [model.MeshLevel(level=k, h=h, nodes=len(xs[k]), cells=0, path=f"meshes/bracket_L{k}.msh")
              for k, h in ((1, 0.003), (2, 0.002))]
    b.state("LC-PROOF", {1: field(xs[1], 0.003), 2: field(xs[2], 0.002)}, x=xs, mesh=levels,
            bcs=[model.BCItem(type="displacement", faces=[f"f{end}"], features=["FT-END"])])
    b.geo = {"edges": {"reentrant": [{"tag": 21, "faces": [step, wall], "angle_deg": 270.0, "length": 0.010,
                                      "midpoint": [0.010, 0.005, 0.010]}]},
             "faces": {str(step): {"features": ["FT-STEP"]}}}
    out = b.run([("DC-PROOF", 100, ["LC-PROOF"])])
    p, s = part(out), smooth(out)

    # the part result is the corner peak of the fine mesh: 300 (1 mm / 1 mm)^0.45 = 300 MPa, 244 MPa on the coarse
    assert p.value == pytest.approx(damage(100, goodman(150, 150)), rel=1e-9) and p.status == "singular"
    assert p.near == f"re-entrant edge 21 (faces f{step} and f{wall})"
    assert math.hypot(p.location[0] - 0.010, p.location[2] - 0.010) < 1e-9
    cv = b.data["part"]["convergence"]
    assert cv["range_coarse"] == pytest.approx(pa(300.0 * (1 / 1.5) ** 0.45), rel=1e-6) and cv["grows"]
    # the zones: 3 element sizes of 2 mm; the edge nodes are 1 mm apart on a quadratic mesh
    zones = {z["kind"]: z for z in b.data["singular_zones"]["zones"]}
    assert set(zones) == {"edge", "face"} and check(out, "SINGULAR-ZONES").passed is True
    assert zones["edge"]["element_size"] == pytest.approx(0.002, rel=0.15)
    assert zones["edge"]["distance"] == pytest.approx(0.006, rel=0.15)
    assert zones["face"]["distance"] == pytest.approx(0.006, rel=0.35)
    # the smooth result: no node in a zone, and it is the largest damage of the nodes out of the zones
    d_edge, d_face = zones["edge"]["distance"], zones["face"]["distance"]
    r_s = math.hypot(s.location[0] - 0.010, s.location[2] - 0.010)
    assert r_s > d_edge and 0.020 - s.location[0] > d_face
    x = xs[2]
    r = np.hypot(x[:, 0] - corner[0], x[:, 2] - corner[1])
    stress = 300.0 * (1e-3 / np.maximum(r, 0.001)) ** 0.45

    def largest(out_of_edge, out_of_face):          # by hand, with the exact distances of the geometry
        keep = (r > out_of_edge) & (0.020 - x[:, 0] > out_of_face)
        return damage(100, goodman(stress[keep].max() / 2, stress[keep].max() / 2))

    # the engine measures the distance to the nearest node on the edge or face: it is the exact distance or a little more
    assert largest(d_edge, d_face) <= s.value * (1 + 1e-9) and s.value <= largest(0.9 * d_edge, 0.9 * d_face)
    assert s.value < 0.05 * p.value and "rim of the singular zone" in s.note
    assert s.status in ("converged", "not_converged") and "information only" in s.note


# =========================================================================== inputs that do not fit


def test_different_node_sets_fail_clearly(bench):
    bench.state("LC-PROOF", uniaxial(200.0, 100.0))
    bench.state("LC-LEAK", uniaxial(100.0, 50.0, 20.0))                          # three nodes against two
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    r = out.record
    assert r.status == "failed" and r.id == "AN-FATIGUE" and not r.results
    assert "LC-PROOF" in r.message and "LC-LEAK" in r.message and "2 nodes and 3 nodes" in r.message
    assert "same mesh" in r.message

    bench.state("LC-LEAK", uniaxial(100.0, 50.0), x=[[0.0, 0, 0], [0.01, 0.0005, 0]])   # the same count, one node moved
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    assert out.record.status == "failed" and "differ by up to 0.5 mm" in out.record.message

    bench.state("LC-LEAK", {2: uniaxial(100.0, 50.0)})                           # no mesh level in common
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    assert out.record.status == "failed" and "no mesh level in common" in out.record.message


def test_a_damaged_field_file_fails_clearly(bench):
    bench.state("LC-PROOF", uniaxial(200.0))
    d = bench.project.analysis_dir("LC-PROOF")
    np.savez(d / "fields_L1.npz", x=np.zeros((1, 3)), u=np.zeros((1, 3)))            # no stress
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert out.record.status == "failed" and "no array stress" in out.record.message
    np.savez(d / "fields_L1.npz", x=np.zeros((2, 3)), stress=np.zeros((2, 3)))       # wrong shape
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert out.record.status == "failed" and "(n, 6)" in out.record.message
    np.savez(d / "fields_L1.npz", x=np.zeros((1, 3)), stress=np.full((1, 6), np.nan))
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert out.record.status == "failed" and "not finite" in out.record.message
    (d / "fields_L1.npz").unlink()
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert out.record.status == "failed" and "is not in the folder" in out.record.message


def test_missing_fatigue_data_is_not_supported(tmp_path):
    b = Bench(tmp_path, fatigue_data=False)
    b.state("LC-PROOF", uniaxial(200.0))
    out = b.run([("DC-PROOF", 2, ["LC-PROOF"])])
    r = out.record
    assert r.status == "not_supported" and not r.results and r.load_case == "LC-FATIGUE"
    assert "MAT-316L has no fatigue data" in r.message and "materials.yaml" in r.message
    assert not out.journal.decisions and not out.journal.assumptions

    b = Bench(tmp_path / "b", b=0.1)                                               # a Basquin line that rises
    b.state("LC-PROOF", uniaxial(200.0))
    out = b.run([("DC-PROOF", 2, ["LC-PROOF"])])
    assert out.record.status == "not_supported" and "exponent b below zero" in out.record.message


def test_missing_states_are_not_supported(bench):
    bench.state("LC-PROOF", uniaxial(200.0))
    out = bench.run([])
    assert out.record.status == "not_supported" and "no duty cycles" in out.record.message
    out = bench.run([("DC-LEAK", 200, ["LC-LEAK"])])                               # not in the plan
    assert out.record.status == "not_supported" and "no load case LC-LEAK" in out.record.message
    bench.state("LC-LEAK", record=False)                                           # planned, not analysed
    out = bench.run([("DC-LEAK", 200, ["LC-LEAK"])])
    assert out.record.status == "not_supported" and "LC-LEAK has no analysis record" in out.record.message
    bench.state("LC-LEAK", uniaxial(100.0), status="failed")                       # analysed, without success
    out = bench.run([("DC-LEAK", 200, ["LC-LEAK"])])
    assert out.record.status == "not_supported" and "AN-LEAK" in out.record.message and "status failed" in out.record.message
    out = bench.run([("DC-PROOF", 0, ["LC-PROOF"])])                               # no repetitions
    assert out.record.status == "not_supported" and "no stress state with a repetition" in out.record.message


# =========================================================================== the record


def test_repetitions_are_checked_against_the_system_context(bench):
    """The context gives 2 proof cycles and 200 leak checks. The engine uses the count of the load case;
    a count with a life factor in it gets a failed check, because the compliance stage applies the factor."""
    bench.state("LC-PROOF", uniaxial(200.0))
    bench.state("LC-LEAK", uniaxial(100.0))
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    assert check(out, "REPETITIONS").passed is True
    assert part(out).value == pytest.approx(damage(2, 120.0) + damage(200, 600 / 11))
    out = bench.run([("DC-PROOF", 8, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    c = check(out, "REPETITIONS")
    assert c.passed is False and "DC-PROOF: 8 in the load case and 2 in the system context" in c.statement
    assert part(out).value == pytest.approx(damage(8, 120.0) + damage(200, 600 / 11))


def test_the_record_and_its_journal(bench):
    bench.state("LC-PROOF", {1: uniaxial(200.0, 50.0), 2: uniaxial(200.0, 50.0)},
                loads=[model.LoadItem(effect="EF-IN-T-FIRE", kind="fluid_temperature", si=90.0, unit="K")])
    bench.state("LC-LEAK", scaled_from="LC-PROOF", scale=0.5)
    out = bench.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])])
    r, j = out.record, out.journal
    assert (r.id, r.load_case, r.analysis_type, r.status) == ("AN-FATIGUE", "LC-FATIGUE", "fatigue", "done")
    assert r.origin.by == "engine:fatigue" and r.runtime_s is not None and "Basquin" in r.solver
    assert set(r.results) == {"damage.total@part", "damage.total@smooth"}
    assert all(v.key == k for k, v in r.results.items())
    assert r.material["sigma_f"] == pytest.approx(pa(800)) and r.material["b"] == -0.2
    for a in r.artifacts:                                            # the files exist, relative to out/
        assert a.kind == "data" and (bench.project.out / a.path).is_file() and a.path.startswith("analyses/LC-FATIGUE/")
    text = (bench.project.out / r.artifacts[0].path).read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["part"]["damage"] == part(out).value and data["load_case"] == "LC-FATIGUE"
    assert {d.id for d in j.decisions} == {"DEC-LC-FATIGUE-METHOD", "DEC-LC-FATIGUE-STRESS-RANGE",
                                           "DEC-LC-FATIGUE-MEAN-STRESS", "DEC-LC-FATIGUE-CYCLE-COUNT",
                                           "DEC-LC-FATIGUE-CONVERGENCE"}
    assert {a.id for a in j.assumptions} == {"ASM-LC-FATIGUE-ORDER", "ASM-LC-FATIGUE-STATES",
                                             "ASM-LC-FATIGUE-TEMPERATURE", "ASM-LC-FATIGUE-FATIGUE-STRENGTH",
                                             "ASM-LC-FATIGUE-UNLOADED-STATE", "ASM-LC-FATIGUE-MINER"}
    by = {a.id: a for a in j.assumptions}
    assert by["ASM-LC-FATIGUE-ORDER"].conservative is True and by["ASM-LC-FATIGUE-STATES"].conservative is False
    assert by["ASM-LC-FATIGUE-FATIGUE-STRENGTH"].conservative is False and "weld" in by["ASM-LC-FATIGUE-FATIGUE-STRENGTH"].impact
    assert "from 90 K to 90 K" in by["ASM-LC-FATIGUE-TEMPERATURE"].statement and "293.15 K" in by["ASM-LC-FATIGUE-TEMPERATURE"].statement
    assert all(d.origin.stage == "solve" and d.subject == "LC-FATIGUE" and d.alternatives for d in j.decisions)
    # the record goes through the store and comes back
    bench.project.write_stage("analyze", bench.project.analysis_dir("LC-FATIGUE") / "analysis.json", analyses=[r],
                              decisions=j.decisions, assumptions=j.assumptions)
    back = next(a for a in bench.project.collection("analyses") if a.id == "AN-FATIGUE")
    assert back.results["damage.total@part"].value == part(out).value


def test_the_pipeline_finds_the_engine(bench):
    bench.state("LC-PROOF", uniaxial(200.0))
    bench.run([("DC-PROOF", 1000, ["LC-PROOF"])])
    out = pipeline.analyze_one(bench.project, bench.ctx, bench.geo, bench.lc, Options(quick=True, log=lambda _m: None))
    assert out.record.status == "done" and out.record.results["damage.total@part"].value == pytest.approx(0.151875)


# ---- the text rules of CONTRACTS.md (ASD-STE100), as far as a pattern can check them

_BANNED = re.compile(r"\b(should|may|might|could|would|has been|have been|had been|etc|e\.g|i\.e|and/or)\b|n't\b", re.I)
_PASSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+(not\s+)?(\w+ed|given|taken|known|found|shown|left|made|done|"
                      r"seen|set)\b", re.I)
_ING = re.compile(r"\b[a-z]+ing\b", re.I)
_ING_OK = {"during", "nothing", "thing", "ring", "string"}


def _ste_problems(text: str, limit: int = 25) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", para.strip()) if s]
        if len(sentences) > 6:
            out.append(f"paragraph of {len(sentences)} sentences: {para[:60]}...")
        for s in sentences:
            if len(s.split()) > limit:
                out.append(f"{len(s.split())} words: {s}")
            for rx, what in ((_BANNED, "word"), (_PASSIVE, "passive")):
                hit = rx.search(s)
                if hit:
                    out.append(f"{what} {hit.group(0)!r}: {s}")
            bad = [w for w in _ING.findall(s) if w.lower() not in _ING_OK]
            if bad:
                out.append(f"-ing form {bad}: {s}")
    return out


def _texts(out) -> list[tuple[str, int]]:
    r, j = out.record, out.journal
    texts = [(r.statement, 25), (r.message, 25), (r.title, 25)]
    texts += [(v.note, 25) for v in r.results.values()] + [(c.statement, 25) for c in r.checks]
    texts += [(a.caption, 25) for a in r.artifacts]
    for d in j.decisions:
        texts += [(d.title, 25), (d.choice, 25), (d.rationale, 25)]
        texts += [(t, 25) for alt in d.alternatives for t in (alt.option, alt.why_not)]
    for a in j.assumptions:
        texts += [(a.title, 25), (a.statement, 25), (a.rationale, 25), (a.impact, 25), (a.verify_by, 20)]
    return texts


def test_record_text_follows_the_ste_rules(tmp_path, monkeypatch):
    outs = []
    b = Bench(tmp_path / "a")                                        # two duty cycles, a scaled state, one mesh
    b.state("LC-PROOF", uniaxial(300.0, 20.0))
    b.state("LC-LEAK", scaled_from="LC-PROOF", scale=-2 / 3)
    b.state("LC-CHILL", uniaxial(2000.0, 5.0))
    outs.append(b.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"]), ("DC-RUN", 65, ["LC-CHILL"])]))
    outs.append(b.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-RUN", 0, ["LC-CHILL"])]))
    b = Bench(tmp_path / "b", mean_stress="swt", endurance="105 MPa")   # zones, two meshes, endurance limit
    monkeypatch.setattr(fatigue, "EXCLUSION_ELEMENTS", 2.5)
    x = _hand_mesh(b.project.out / "meshes" / "hand_L2.msh")
    level = model.MeshLevel(level=2, h=1e-3, nodes=len(x), cells=0, order=1, path="meshes/hand_L2.msh")
    b.state("LC-PROOF", {1: _hand_stress(x, 0.8), 2: _hand_stress(x)}, x={1: x, 2: x}, mesh=[level],
            bcs=[model.BCItem(type="fixed", faces=["f3"]), model.BCItem(type="displacement", value={"points": []})])
    b.geo = {"edges": {"reentrant": [{"tag": 7, "faces": [1, 2], "length": 0.010, "midpoint": [0.0, 0.005, 0.0]},
                                     {"tag": 9, "faces": [1, 5]}]}, "faces": {"1": {"features": ["FT-PAD-TOP"]}}}
    outs.append(b.run([("DC-PROOF", 1000, ["LC-PROOF"])]))
    b = Bench(tmp_path / "c", mean_stress="none")                      # states that do not fit
    b.state("LC-PROOF", uniaxial(200.0, 100.0))
    b.state("LC-LEAK", uniaxial(100.0))
    outs.append(b.run([("DC-PROOF", 2, ["LC-PROOF"]), ("DC-LEAK", 200, ["LC-LEAK"])]))
    outs.append(b.run([("DC-PROOF", 2, ["LC-PROOF"])]))
    outs.append(Bench(tmp_path / "d", fatigue_data=False).run([("DC-PROOF", 2, ["LC-PROOF"])]))
    assert [o.record.status for o in outs] == ["done", "done", "done", "failed", "done", "not_supported"]
    problems = [p for o in outs for text, limit in _texts(o) for p in _ste_problems(text, limit)]
    assert not problems, "\n".join(sorted(set(problems)))
