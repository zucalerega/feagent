import json
import subprocess
import sys

import gmsh
import numpy as np
import pytest

from feagent.errors import CadLoadError
from feagent.gmsh_session import gmsh_session, load_cad
from feagent.meshdata import read_msh
from feagent.parts import PARTS, make_part
from feagent.types import (
    FaceInfo, Feature, Gate, GeometryModel, GroupResult, Hazard, MeshResult,
    VolumeInfo, EdgeInfo, overall_status,
)


def test_make_all_writes_step_files(parts_dir):
    for name in PARTS:
        p = parts_dir / f"{name}.step"
        assert p.exists() and p.stat().st_size > 0, name
    with pytest.raises(ValueError):
        make_part("nope", parts_dir / "x.step")


def test_load_cad_returns_volumes(parts_dir):
    with gmsh_session():
        assert len(load_cad(parts_dir / "bracket.step")) == 1
    with gmsh_session():
        vols = load_cad(parts_dir / "cantilever.step")
        assert gmsh.model.occ.getMass(3, vols[0][1]) == pytest.approx(10000, abs=1e-6)


def test_load_cad_missing_file(tmp_path):
    with gmsh_session():
        with pytest.raises(CadLoadError) as ei:
            load_cad(tmp_path / "missing.step")
    assert ei.value.hint
    with gmsh_session():
        with pytest.raises(CadLoadError) as ei:
            load_cad(tmp_path / "x.txt")
    assert ".step" in ei.value.hint


def test_load_cad_keeps_stdout_clean(parts_dir):
    code = (
        "from feagent.gmsh_session import *\n"
        f"with gmsh_session():\n    load_cad({str(parts_dir / 'bracket.step')!r})\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""


def test_types_roundtrip():
    bb = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    face = FaceInfo(tag=1, kind="plane", area=2.0, center=(0.0, 1.0, 1.5), bbox=bb,
                    normal=(0.0, 0.0, 1.0), radius=None, axis=None, axis_point=None,
                    concave=None, angle_span=None, volumes=[1], edges=[1, 2], neighbors=[2],
                    thickness=0.5)
    geo = GeometryModel(
        source="a.step", units="mm", bbox=bb, diag=3.7, ref_size=1.0,
        volumes=[VolumeInfo(1, 6.0, (0.5, 1.0, 1.5), bb, [1])],
        faces=[face], edges=[EdgeInfo(1, "line", 1.0, [1], bb)],
        features=[Feature("hole", [1], {"r": 1.0})],
        hazards=[Hazard("small_face", [(2, 1)], 0.1, 0.5, (0.0, 0.0, 0.0), "m", "s")],
    )
    gate = Gate("geometry", "Geometry", "pass", 1.0, None, "ok")
    r = MeshResult(
        ok=True, spec_path="s.yaml", out_dir="out", mesh_path="out/m.msh",
        exports={"msh": "out/m.msh"},
        groups={"fixed": GroupResult("fixed", 2, [1], 10, 2.0)},
        n_nodes=5, n_elements=4, order=1, size_scale=1.0, elapsed_s=0.1,
        errors=[], warnings=["w"], gates=[gate], geometry=geo,
    )
    r2 = MeshResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert r2 == r
    assert isinstance(r2.geometry.faces[0].center, tuple)
    assert overall_status([gate]) == "pass"
    assert overall_status([gate, Gate("g", "g", "fail", None, None, "x")]) == "fail"


def test_read_msh_groups(parts_dir, tmp_path):
    out = tmp_path / "c.msh"
    with gmsh_session():
        vols = load_cad(parts_dir / "cantilever.step")
        faces = []
        for dim, tag in gmsh.model.getEntities(2):
            if abs(gmsh.model.occ.getCenterOfMass(2, tag)[0]) < 1e-6:
                faces.append(tag)
        gmsh.model.addPhysicalGroup(2, faces, name="fixed")
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], name="domain")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out))
    m = read_msh(out)
    assert m.tets.shape[1] == 4
    assert "fixed" in m.tris
    assert np.allclose(m.points[m.tris["fixed"]][:, :, 0], 0, atol=1e-6)
    assert m.volume_groups["domain"]
    assert m.order == 1
    assert len(m.tet_tags) == len(m.tets) == len(m.tet_volume)


def test_numpy_values_serialize_to_json():
    bb = ((np.float64(0), np.float64(0), np.float64(0)), (np.float64(1), np.float64(2), np.float64(3)))
    face = FaceInfo(tag=np.int64(1), kind="plane", area=np.float64(2.0),
                    center=(np.float64(0.5), np.float64(1.0), np.float64(1.5)), bbox=bb,
                    normal=np.array([0.0, 0.0, 1.0]), radius=None, axis=None, axis_point=None,
                    concave=np.bool_(False), angle_span=None, volumes=[np.int64(1)], edges=[],
                    neighbors=[np.int64(2)], thickness=np.float64(0.5))
    geo = GeometryModel(source="a.step", units="mm", bbox=bb, diag=np.float64(3.7),
                        ref_size=np.float64(1.0), volumes=[], faces=[face], edges=[],
                        features=[Feature("hole", [np.int64(1)], {"r": np.float64(1.0)})],
                        hazards=[])
    d = json.loads(json.dumps(geo.to_dict()))
    g2 = GeometryModel.from_dict(d)
    assert g2.faces[0].tag == 1 and g2.faces[0].normal == (0.0, 0.0, 1.0)
    assert g2.faces[0].concave is False and g2.features[0].params == {"r": 1.0}
    assert g2.bbox == ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
