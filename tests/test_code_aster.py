"""code_aster reference extraction (static parsing only; nothing is executed)."""

import json
import math

import pytest

from feabench.paths import CODE_ASTER_DIR
from feabench.suites import code_aster as ca

COMM = '''
from math import *
E = 2.0e5
P = 100.
A = P * 0.5 ** 2 / (1 - 0.25)
DEBUT()
MA = DEFI_MATERIAU(ELAS=_F(E=E, NU=0.3))
MO = AFFE_MODELE(MAILLAGE=M, AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE", MODELISATION="3D"))
RESU = MECA_STATIQUE(MODELE=MO)
TEST_RESU(RESU=(_F(RESULTAT=RESU, NUME_ORDRE=1, NOM_CHAM="DEPL", GROUP_NO="A", NOM_CMP="DX",
                   VALE_CALC=1.01e-3, VALE_REFE=A / E * sqrt(4.0), REFERENCE="ANALYTIQUE", PRECISION=0.02),
                _F(RESULTAT=RESU, NUME_ORDRE=1, NOM_CHAM="DEPL", GROUP_NO="B", NOM_CMP="DY", VALE_CALC=2.0)))
for i in range(2):
    TEST_TABLE(TABLE=T, NOM_PARA="V", VALE_CALC=float(i), VALE_REFE=0.0, REFERENCE="AUTRE_ASTER")
os.system("this is never executed")
'''


def test_parse_comm_static_extraction():
    out = ca.parse_comm(COMM)
    assert out["parse_error"] is None
    assert out["modelisations"] == ["3D"] and out["operators"] == ["MECA_STATIQUE"]
    assert out["materials"][0]["ELAS"]["E"] == 2.0e5
    e = [x for x in out["entries"] if x.get("GROUP_NO") == "A"][0]
    A = 100.0 * 0.25 / 0.75
    assert e["VALE_REFE"] == pytest.approx(A / 2.0e5 * 2.0)  # constants and math resolved safely
    assert ca.classify(e) == "independent" and e["PRECISION"] == 0.02
    b = [x for x in out["entries"] if x.get("GROUP_NO") == "B"][0]
    assert ca.classify(b) == "regression_only"
    loop = [x for x in out["entries"] if x["_command"] == "TEST_TABLE"][0]
    assert loop["_in_loop"] and ca.classify(loop) == "other_code_aster_model"
    assert isinstance(loop["VALE_CALC"], dict)  # unresolved expression recorded, not evaluated


def test_safe_eval_refuses_calls_outside_math():
    import ast

    with pytest.raises(ca.Unresolved):
        ca.safe_eval(ast.parse("__import__('os').getcwd()", mode="eval").body, {})
    assert ca.safe_eval(ast.parse("sqrt(2.0) * pi", mode="eval").body, {}) == pytest.approx(math.sqrt(2) * math.pi)


def test_vendored_manifest_and_mesh_groups():
    man = json.loads((CODE_ASTER_DIR / "manifest.json").read_text())
    t = {x["id"]: x for x in man["tests"]}["sslv04a"]
    assert t["vendored"] and t["ref_counts"]["independent"] > 50
    groups = ca.mesh_groups(CODE_ASTER_DIR / "astest" / "sslv04a.mmed")
    # point E of the hollow cylinder (inner radius 0.1 m, at 45 degrees)
    assert groups["E"]["n"] == 1
    assert math.hypot(*groups["E"]["xyz"][:2]) == pytest.approx(0.1, rel=1e-5)
    entries = [e for e in t["entries"] if e.get("GROUP_NO") == "E" and "location" in e]
    assert entries and entries[0]["location"]["xyz"] == pytest.approx(groups["E"]["xyz"])
