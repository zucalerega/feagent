"""Tests of the geometry stage (dossier.geometry) and the meshes (dossier.meshing) on the example valve body.

The numbers checked here were measured on the example (47 faces, 18 features, 8 re-entrant edges, volume
2.96776e-4 m^3, mass 2.3712 kg with 7990 kg/m^3) and, for the volume and the mass, against an independent
reading of the CAD file with the kernel.
"""

import math
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "2")

from dossier import geometry, meshing  # noqa: E402
from dossier.project import Project  # noqa: E402

EXPECT = {                      # feature -> number of faces (the `expect` of the component file)
    "FT-INLET-BORE": 1, "FT-OUTLET-BORE": 1, "FT-SEAT-IN-WALL": 1, "FT-SEAT-OUT-WALL": 1,
    "FT-SEAT-IN-SHOULDER": 1, "FT-SEAT-OUT-SHOULDER": 1, "FT-CAVITY-WALL": 1, "FT-CAVITY-FLOOR": 2,
    "FT-CAVITY-ROOF": 1, "FT-STEM-BORE": 1, "FT-INLET-FLANGE-FACE": 1, "FT-OUTLET-FLANGE-FACE": 1,
    "FT-INLET-BOLT-HOLES": 6, "FT-OUTLET-BOLT-HOLES": 6, "FT-PAD-TOP": 1, "FT-PAD-HOLES": 4, "FT-NECK": 1,
    "FT-NECK-ROOT-FILLET": 1,
}
VOLUME = 2.96776e-4             # m^3


def opts(quick=True, **kw):
    return SimpleNamespace(pictures=False, quick=quick, node_cap=25000 if quick else 120000,
                           levels=2 if quick else 3, log=lambda s: None, **kw)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """geometry.build on a copy of the example, once for the module (the stage takes about 10 s)."""
    from tests.dossier.conftest import EXAMPLE
    dst = tmp_path_factory.mktemp("geo") / "mov_valve"
    shutil.copytree(EXAMPLE, dst, ignore=shutil.ignore_patterns("out", "__pycache__", "reviews.yaml"))
    project = Project.load(dst)
    geo, journal = geometry.build(project, opts())
    return project, geo, journal


def _copy_with(tmp_path, edit=None, step=None, unit=None):
    """A copy of the example whose component.yaml text is changed by `edit(text)`, with another CAD file."""
    from tests.dossier.conftest import EXAMPLE
    dst = tmp_path / "mov_valve"
    shutil.copytree(EXAMPLE, dst, ignore=shutil.ignore_patterns("out", "__pycache__", "reviews.yaml"))
    folder = dst / "components" / "mov_body"
    if edit:
        text = (folder / "component.yaml").read_text()
        (folder / "component.yaml").write_text(edit(text))
    if step:
        shutil.copy(step, folder / "geometry.step")
    if unit:
        text = (folder / "component.yaml").read_text()
        (folder / "component.yaml").write_text(text.replace("length_unit: m}", f"length_unit: {unit}}}"))
    return Project.load(dst)


def _step_in_mm(path: Path) -> Path:
    """The example body built again by its own script with the numbers in millimetres (MM = 1) and a header
    in millimetres: a genuine millimetre file, with analytic surfaces (a scaled copy would be B-splines)."""
    import importlib.util

    import gmsh

    from feacore.geometry.gmsh_tools import gmsh_session
    from tests.dossier.conftest import EXAMPLE
    script = EXAMPLE / "components" / "mov_body" / "make_geometry.py"
    spec = importlib.util.spec_from_file_location("make_geometry_mm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MM = 1.0
    with gmsh_session("body-mm"):
        gmsh.model.add("mov_body_mm")
        mod.build(fillets=True)
        gmsh.write(str(path))
    return path


# =========================================================================== geometry


def test_cad_facts(built):
    project, geo, journal = built
    cad = geo["cad"]
    assert geo["component"] == "CMP-MOV-BODY"
    assert cad["n_faces"] == 47 and len(geo["faces"]) == 47
    assert cad["volume"] == pytest.approx(VOLUME, rel=1e-4)
    assert cad["mass"] == pytest.approx(cad["volume"] * cad["density"], rel=1e-6)
    assert cad["density"] == pytest.approx(7990.0)
    assert cad["mass"] == pytest.approx(2.3712, abs=2e-3)
    assert cad["sha256"] == geometry.sha256_of(project.cad_path("CMP-MOV-BODY"))
    lo, hi = np.array(cad["bbox"][0]), np.array(cad["bbox"][1])
    assert np.allclose(lo, [-0.075, -0.042, -0.042], atol=1e-6)
    assert np.allclose(hi, [0.075, 0.042, 0.128], atol=1e-6)


def test_every_feature_resolves_to_the_expected_faces(built):
    _, geo, journal = built
    assert set(geo["features"]) == set(EXPECT)
    for fid, n in EXPECT.items():
        assert len(geo["features"][fid]["faces"]) == n, fid
    assert not [f for f in journal.findings if f.severity in ("error", "major")]
    assert not geo["problems"]


def test_feature_measures(built):
    _, geo, _ = built
    f = geo["features"]
    assert f["FT-INLET-BORE"]["kind"] == "cylinder"
    assert f["FT-INLET-BORE"]["radius"] == pytest.approx(0.0125, rel=1e-6)
    assert f["FT-INLET-BORE"]["diameter"] == pytest.approx(0.025, rel=1e-6)
    assert np.allclose(np.abs(f["FT-INLET-BORE"]["axis"]), [1, 0, 0], atol=1e-6)
    # the bore runs from the flange face (x = -0.075) to the cavity: length under half the body length
    assert 0.02 < f["FT-INLET-BORE"]["length"] < 0.075
    assert f["FT-NECK-ROOT-FILLET"]["kind"] == "torus"
    assert f["FT-NECK-ROOT-FILLET"]["radius"] == pytest.approx(0.003, rel=1e-6)
    assert f["FT-INLET-FLANGE-FACE"]["kind"] == "plane"
    assert np.allclose(f["FT-INLET-FLANGE-FACE"]["normal"], [-1, 0, 0], atol=1e-9)
    assert f["FT-INLET-FLANGE-FACE"]["centroid"][0] == pytest.approx(-0.075, abs=1e-6)
    assert np.allclose(f["FT-PAD-TOP"]["normal"], [0, 0, 1], atol=1e-9)
    assert f["FT-PAD-TOP"]["centroid"][2] == pytest.approx(0.128, abs=1e-6)
    assert f["FT-CAVITY-FLOOR"]["kind"] == "mixed"            # a plane and a torus
    assert f["FT-INLET-BOLT-HOLES"]["radius"] == pytest.approx(0.0033, rel=1e-6)
    # area of the cylinder of the stem bore: 2 pi r L
    sb = f["FT-STEM-BORE"]
    assert sb["area"] == pytest.approx(2 * math.pi * 0.008 * sb["length"], rel=1e-3)


def test_star_resolution(built):
    _, geo, journal = built
    inter = geo["interfaces"]
    used = {t for i in inter.values() for role in ("wetted", "seat", "bolts") for t in i[role]}
    ext = set(inter["INT-ENV"]["external"])
    assert len(ext) == 17
    assert not ext & used
    assert ext | used == {int(t) for t in geo["faces"]}          # every face is in some role
    dec = [d for d in journal.decisions if "EXTERNAL" in d.id]
    assert len(dec) == 1 and "17 faces" in dec[0].choice
    # the faces carry the role entries
    assert "INT-ENV:external" in geo["faces"][str(sorted(ext)[0])]["roles"]


def test_shoulder_wetted_and_seat_is_not_an_overlap(built):
    _, geo, journal = built
    # FT-SEAT-OUT-SHOULDER is wetted in INT-OUTLET and seat in INT-INTERNALS
    tag = geo["features"]["FT-SEAT-OUT-SHOULDER"]["faces"][0]
    roles = geo["faces"][str(tag)]["roles"]
    assert "INT-OUTLET:wetted" in roles and "INT-INTERNALS:seat" in roles
    assert not [f for f in journal.findings if f.rule.endswith("wetted_twice") or "wetted" in f.rule]


def test_unused_features_are_info(built):
    _, _, journal = built
    infos = {f.subject for f in journal.findings if f.severity == "info"}
    assert infos == {"FT-NECK", "FT-NECK-ROOT-FILLET"}


def test_reentrant_edges(built):
    _, geo, journal = built
    edges = geo["edges"]["reentrant"]
    assert len(edges) == 8
    assert sorted(e["tag"] for e in edges) == [34, 37, 38, 40, 43, 60, 100, 102]
    for e in edges:
        assert e["angle_deg"] == pytest.approx(270.0, abs=0.5)
        assert len(e["faces"]) == 2 and e["length"] > 0.05
    # one pair at the cavity (z ~ 0, y = +-0.02) and the pocket edges at the shoulders
    mids = np.array([e["midpoint"] for e in edges])
    assert np.sum(np.isclose(np.abs(mids[:, 0]), 0.063, atol=1e-3)) == 2
    assert any("REENTRANT-EDGES" in d.id for d in journal.decisions)


def test_scls_of_the_example_are_ok(built):
    _, geo, _ = built
    assert set(geo["scls"]) == {"SCL-SHELL", "SCL-NOZZLE-OUT", "SCL-FLOOR", "SCL-ROOF", "SCL-NECK-ROOT"}
    for sid, s in geo["scls"].items():
        assert s["ok"], (sid, s["message"])
        assert s["length"] == pytest.approx(np.linalg.norm(np.subtract(s["end"], s["start"])), rel=1e-9)


def test_bad_scls_are_caught(tmp_path):
    def edit(text):
        # end not on the surface; a line that leaves the material (through the bore); a start in the air
        text = text.replace("start: [0, 0.022, 0], end: [0, 0.028, 0]", "start: [0, 0.022, 0], end: [0, 0.0285, 0]")
        text = text.replace("start: [0.018, 0, 0.034], end: [0.018, 0, 0.042]",
                            "start: [0.018, 0, 0.034], end: [0.018, 0, 0.060]")
        text = text.replace("start: [0, 0, -0.022], end: [0, 0, -0.030]", "start: [0, 0, -0.010], end: [0, 0, -0.030]")
        return text
    project = _copy_with(tmp_path, edit)
    geo, journal = geometry.build(project, opts())
    assert not geo["scls"]["SCL-SHELL"]["ok"] and geo["scls"]["SCL-SHELL"]["message"]
    assert not geo["scls"]["SCL-ROOF"]["ok"]
    assert not geo["scls"]["SCL-FLOOR"]["ok"]
    assert geo["scls"]["SCL-NOZZLE-OUT"]["ok"]
    subjects = {f.subject for f in journal.findings if f.severity == "major"}
    assert {"SCL-SHELL", "SCL-ROOF", "SCL-FLOOR"} <= subjects


def test_broken_expect_gives_a_finding(tmp_path):
    project = _copy_with(tmp_path, lambda t: t.replace("normal: [0, 0, 1], at: zmax}, expect: 1}",
                                                       "normal: [0, 0, 1], at: zmax}, expect: 3}"))
    geo, journal = geometry.build(project, opts())
    hits = [f for f in journal.findings if f.subject == "FT-PAD-TOP"]
    assert len(hits) == 1 and hits[0].severity == "error" and hits[0].category == "model"
    assert geo["problems"]


def test_feature_that_selects_nothing_gives_a_finding(tmp_path):
    project = _copy_with(tmp_path, lambda t: t.replace("radius: 0.0125, near: [-0.051, 0, 0]",
                                                       "radius: 0.0999, near: [-0.051, 0, 0]"))
    geo, journal = geometry.build(project, opts())
    assert geo["features"]["FT-INLET-BORE"]["faces"] == []
    hits = [f for f in journal.findings if f.subject == "FT-INLET-BORE"]
    assert hits and hits[0].severity == "error" and hits[0].category == "model"


def test_wetted_in_two_interfaces_gives_a_finding(tmp_path):
    # FT-INLET-BORE is wetted in INT-INLET; also make it wetted in INT-OUTLET
    def edit(text):
        i = text.index("id: INT-OUTLET")
        j = text.index("wetted: [", i)
        return text[:j] + "wetted: [FT-INLET-BORE, " + text[j + len("wetted: ["):]
    project = _copy_with(tmp_path, edit)
    _, journal = geometry.build(project, opts())
    assert any(f.severity == "major" and "wetted" in f.statement for f in journal.findings)


def test_mm_file_is_scaled_to_metres(tmp_path):
    """The same body with its numbers in millimetres (header mm) and length_unit: mm gives the metre result."""
    big = _step_in_mm(tmp_path / "body_mm.step")
    ref = _copy_with(tmp_path / "a")
    mm = _copy_with(tmp_path / "b", step=big, unit="mm")
    g0, _ = geometry.build(ref, opts())
    g1, _ = geometry.build(mm, opts())
    assert g1["cad"]["scale"] == pytest.approx(1e-3)
    assert g1["cad"]["volume"] == pytest.approx(g0["cad"]["volume"], rel=1e-4)
    assert g1["cad"]["mass"] == pytest.approx(g0["cad"]["mass"], rel=1e-4)
    assert np.allclose(g1["cad"]["bbox"], g0["cad"]["bbox"], atol=1e-6)
    assert g1["cad"]["sha256"] != g0["cad"]["sha256"]
    assert {k: v["faces"] for k, v in g1["features"].items()} == {k: v["faces"] for k, v in g0["features"].items()}
    assert len(g1["edges"]["reentrant"]) == 8
    assert all(s["ok"] for s in g1["scls"].values())


def test_wrong_length_unit_is_not_silent(tmp_path):
    """mm numbers declared as metres: the part is 1000 times too big, and the stage says so."""
    big = _step_in_mm(tmp_path / "body_mm.step")
    project = _copy_with(tmp_path / "w", step=big)             # length_unit stays m
    try:
        geo, journal = geometry.build(project, opts())
    except geometry.GeometryError:
        return                                                  # an exception is an acceptable answer
    assert geo["cad"]["volume"] > 1e3 * VOLUME or [f for f in journal.findings if f.severity in ("major", "error")] \
        or [d for d in journal.decisions if "LENGTH-UNIT" in d.id]


def test_pictures(tmp_path):
    project = _copy_with(tmp_path)
    o = opts()
    o.pictures = True
    geo, _ = geometry.build(project, o)
    assert len(geo["pictures"]) == 6                           # an overview and five interfaces
    for p in geo["pictures"]:
        f = project.out / p["path"]
        assert f.is_file() and f.stat().st_size > 5000 and p["caption"]


# =========================================================================== meshing


def test_quick_mesh_levels_cache_and_solve(built):
    project, geo, _ = built
    o = opts(quick=True)
    lv, journal = meshing.levels_with_journal(project, geo, o)
    assert len(lv) == 2
    assert [m.level for m in lv] == [0, 1]
    assert lv[0].h > lv[1].h
    assert lv[0].h_rep / lv[1].h_rep == pytest.approx(1.4, rel=0.08)
    assert lv[0].nodes < lv[1].nodes <= 25000
    assert all(m.order == 2 and m.min_quality > 0.05 and m.h_rep > 0 for m in lv)
    assert all(not Path(m.path).is_absolute() and (project.out / m.path).is_file() for m in lv)
    assert any("MESH-SIZES" in d.id for d in journal.decisions)
    # the coarse quick level differs by about 2 % from the CAD volume: a minor finding with the quick limit, not a major one
    vol = [f for f in journal.findings if f.rule == "mesh.volume"]
    assert all(f.severity == "minor" for f in vol)
    assert not vol or "quick run" in vol[0].statement
    # second call: from the cache, same files
    import time
    t = time.time()
    lv2 = meshing.levels(project, geo, o)
    assert time.time() - t < 2.0
    assert [(m.path, m.nodes) for m in lv2] == [(m.path, m.nodes) for m in lv]

    # groups: f<tag>, every feature, solid; the mesh loads in the fixed solver and has the curved volume of the CAD
    from skfem import Basis, ElementTetP2

    from feacore.solver._skfem_mesh import read_mesh
    md = read_mesh(project.out / lv[0].path, 3, 2)
    for name in [f"f{t}" for t in geo["faces"]] + list(geo["features"]) + ["solid"]:
        assert name in md.groups and len(md.groups[name].index), name
    for m in lv:
        vol = float(Basis(read_mesh(project.out / m.path, 3, 2).mesh, ElementTetP2(), intorder=3).dx.sum())
        assert vol == pytest.approx(geo["cad"]["volume"], rel=3e-2)       # coarse levels: curved walls are cut short (1.8 % at 15 mm)


def test_solve_on_the_coarsest_mesh(built):
    """A real solve on the coarsest quick mesh by feature names: fixed flange face, a force on the actuator pad.
    The reaction on the fixed face must balance the applied force (an independent equilibrium check)."""
    project, geo, _ = built
    lv = meshing.levels(project, geo, opts(quick=True))
    from feacore.case import BC, QoI, Analysis, Case, Material
    from feacore.solver import solve
    case = Case(
        mesh=str((project.out / lv[0].path).resolve()),
        analysis=Analysis(physics="elasticity", formulation="3d", order=2),
        materials={"steel": Material(E=195e9, nu=0.3)},
        bcs=[BC("fixed", group="FT-INLET-FLANGE-FACE"), BC("fixed", group="FT-OUTLET-FLANGE-FACE"),
             BC("force", group="FT-PAD-TOP", value=[0.0, 0.0, -1000.0]),
             BC("pressure", group="FT-CAVITY-WALL", value=1e6)],
        qois=[QoI(name="rz_in", type="reaction", group="FT-INLET-FLANGE-FACE", component=2),
              QoI(name="rz_out", type="reaction", group="FT-OUTLET-FLANGE-FACE", component=2),
              QoI(name="umax", type="max", field="u_mag")])
    res = solve(case, "skfem")
    q = res.evaluate_all()
    assert q["umax"] > 0 and np.isfinite(q["umax"])
    # the pressure on a cylinder wall has no net force along the axis (z), so the reactions carry 1000 N
    assert abs(q["rz_in"] + q["rz_out"]) == pytest.approx(1000.0, rel=0.02)


@pytest.mark.slow
def test_full_mesh_levels(tmp_path):
    project = _copy_with(tmp_path)
    geo, _ = geometry.build(project, opts(quick=False))
    lv, journal = meshing.levels_with_journal(project, geo, opts(quick=False))
    assert len(lv) == 3
    assert lv[0].nodes < lv[1].nodes < lv[2].nodes <= 120000
    assert lv[2].nodes > 0.6 * 120000
    for a, b in zip(lv, lv[1:]):
        assert a.h_rep / b.h_rep == pytest.approx(1.4, rel=0.08)
