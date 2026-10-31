"""Tests for the gold benchmarks (feacore.problems.gold, scripts/gold.py).

Every committed set must load as a registered problem whose statement, references and reference case
agree, whose STEP re-imports with the part's face tags (the f<tag> names agents use), and whose
public/ carries nothing private. The importer runs on an archive rebuilt from a committed set.
Re-solving on a stored mesh needs the meshes, which are not in git: those tests skip without them.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from feacore.problems.gold import load_all
from feacore.problems.registry import all_problems, discover, get

ROOT = Path(__file__).resolve().parents[1]
SETS, ERRORS = load_all()
IDS = [p.id for p in SETS]


def _script():
    spec = importlib.util.spec_from_file_location("gold_script", ROOT / "scripts" / "gold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _script()


def test_sets_load_and_register():
    assert SETS, "no gold sets under benchmarks/gold"
    assert not ERRORS
    discover()
    assert all(get(p.id).id == p.id for p in SETS)
    assert {p.id for p in all_problems(tags=["gold"])} == set(IDS)


@pytest.mark.parametrize("p", SETS, ids=IDS)
def test_set_is_consistent(p):
    assert G.check_set(p.dir) == []
    names = {q.name for q in p.qois()}
    refs = {r.qoi for r in p.references()}
    assert refs and refs <= names
    assert set(p.expectations()) <= names and not refs & set(p.expectations())
    assert p.scored_qoi() in refs
    case = p.case("x.msh", order=1)
    assert case.mesh == "x.msh" and case.analysis.order == 1
    assert {q.name for q in case.qois} == names and not case.validate()
    statement = json.dumps(p.statement())
    assert all(f"{r.value:.4g}" not in statement for r in p.references())
    sizes = p.study_sizes()
    assert sizes == sorted(sizes, reverse=True) and len(sizes) >= 3
    assert p.private()["reference_case"]["mesh"] == "MESH.msh"


@pytest.mark.parametrize("part", sorted({p.family for p in SETS}))
def test_step_faces_are_the_parts(part):
    assert G.check_faces(next(p for p in SETS if p.family == part)) is None


def test_one_step_blob_per_part():
    steps: dict[str, set[bytes]] = {}
    for p in SETS:
        steps.setdefault(p.family, set()).add(p.step_path.read_bytes())
    assert all(len(v) == 1 for v in steps.values())


def test_step_timestamp_is_ignored():
    a = b"HEADER;\nFILE_NAME('Open CASCADE Shape Model','2026-10-31T13:39:04',('Author'),(\nDATA;"
    b = a.replace(b"13:39:04", b"13:39:09")
    assert a != b and G._unstamped(a) == G._unstamped(b)


def test_export_all_leaves_gold_files_alone(tmp_path):
    from feacore.problems.store import export_all

    assert export_all(tmp_path, problems=SETS) == ([], {})
    assert not any(tmp_path.glob("abc_*"))


def test_import_archive(tmp_path):
    src = SETS[0]
    shutil.copytree(src.dir, tmp_path / "arch" / "out" / src.id)
    tgz = tmp_path / "goldgen-test_0000.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(tmp_path / "arch" / "out", arcname="out")
    root = tmp_path / "gold"
    assert G.main(["--root", str(root), "import", str(tgz)]) == 0
    index = json.loads((root / "index.json").read_text())
    assert index["sets"][src.id]["archive"] == "goldgen-test_0000"
    for rel in G.SET_FILES:
        assert (root / src.id / rel).read_bytes() == (src.dir / rel).read_bytes()
    assert G.main(["--root", str(root), "import", str(tgz)]) == 0          # again: unchanged
    loaded, errors = load_all(root)
    assert [p.id for p in loaded] == [src.id] and not errors


def test_import_rejects_private_data_in_public(tmp_path, capsys):
    src = SETS[0]
    bad = tmp_path / "out" / src.id
    shutil.copytree(src.dir, bad)
    statement = json.loads((bad / "public" / "problem.json").read_text())
    statement["references"] = json.loads((bad / "private" / "reference.json").read_text())["references"]
    (bad / "public" / "problem.json").write_text(json.dumps(statement))
    root = tmp_path / "gold"
    assert G.main(["--root", str(root), "import", str(tmp_path / "out")]) == 1
    assert not (root / src.id).exists()
    assert "private data" in capsys.readouterr().out


@pytest.mark.parametrize("p", SETS, ids=IDS)
def test_reproduces_the_study_on_L0(p):
    mesh = p.mesh_file(0)
    if mesh is None:
        pytest.skip("L0 mesh not imported (meshes are not in git: scripts/gold.py import ARCHIVE)")
    from feacore.solver import solve

    rows = G._compare(p, 0, solve(p.case(mesh), backend="skfem").evaluate_all())
    assert rows and all(ok for *_, ok in rows), [r for r in rows if not r[3]]
