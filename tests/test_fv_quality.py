"""Finite-volume mesh quality (feacore.geometry.fv_quality): OpenFOAM's checkMesh formulas."""

import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from feacore.geometry import fv_quality


def grid(nx: int, ny: int, a: float = 1.0, b: float = 1.0, shear: float = 0.0):
    """nx x ny quads of size a x b, columns sheared by `shear` (x shifts by shear * y)."""
    xs, ys = np.meshgrid(np.arange(nx + 1) * a, np.arange(ny + 1) * b, indexing="ij")
    pts = np.column_stack([(xs + shear * ys).ravel(), ys.ravel()])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    quads = np.array([[idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
                      for i in range(nx) for j in range(ny)])
    bottom = np.array([[idx[i, 0], idx[i + 1, 0]] for i in range(nx)])
    return pts, quads, bottom


def test_rectangles_are_orthogonal_with_the_aspect_ratio_and_wall_distance():
    pts, quads, bottom = grid(6, 4, a=0.4, b=0.1)
    q = fv_quality.quality(pts, {"quad": quads}, 2, {"wall": bottom})
    assert q["cells"] == 24 and q["negative_volumes"] == 0
    assert q["non_orthogonality"]["max"] < 1e-6 and q["skewness"]["max"] < 1e-9
    assert q["aspect_ratio"]["max"] == pytest.approx(4.0)
    assert q["volume_ratio"]["max"] == pytest.approx(1.0)
    w = q["near_wall"]["wall"]
    assert w["faces"] == 6 and w["wall_distance"] == pytest.approx(0.05) and w["face_size"] == pytest.approx(0.4)
    assert fv_quality.passes(q) == (True, None)


def test_sheared_grid_has_the_shear_angle_as_non_orthogonality():
    pts, quads, _ = grid(5, 5, shear=math.tan(math.radians(30)))
    q = fv_quality.quality(pts, {"quad": quads}, 2)
    # the faces between rows stay horizontal, the line between cell centres leans by the shear angle
    assert q["non_orthogonality"]["max"] == pytest.approx(30.0, abs=1e-6)


def test_clockwise_cells_are_handled_and_the_gate_reports_why():
    pts, quads, _ = grid(3, 3)
    q = fv_quality.quality(pts, {"quad": quads[:, ::-1]}, 2)       # clockwise node order
    assert q["negative_volumes"] == 0 and q["non_orthogonality"]["max"] < 1e-6
    ok, why = fv_quality.passes({**q, "non_orthogonality": {"max": 75.0}})
    assert not ok and "non-orthogonality" in why
    ok, why = fv_quality.passes({**q, "skewness": {"max": 6.0}})
    assert not ok and "skewness" in why


def test_tets_with_inverted_node_order_are_reoriented():
    # the 6-tet split of a unit cube; every second tet listed inside out
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    tets = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    tets[::2] = tets[::2][:, [0, 2, 1, 3]]
    q = fv_quality.quality(c, {"tetra": tets}, 3)
    assert q["cells"] == 6 and q["negative_volumes"] == 0


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam
    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


def _checkmesh(path: Path, dim: int) -> dict:
    from feacore.solver import _foam_io as fio
    from feacore.solver._foam_env import require_openfoam, run
    from feacore.solver._foam_mesh import FRONT_BACK, convert

    fe = require_openfoam()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        convert(path, dim, 1.0, d / "mesh.msh")
        fio.write_control(d, 10)
        run(fe, ["gmshToFoam", "mesh.msh"], d)
        fio.write_schemes(d, 2, False, False)
        fio.write_solution(d, 1e-6, False, None, 0.9, None, 0)
        if dim == 2:
            poly = fio.read_poly(d)
            fio.set_patch_types(d, {p: ("empty" if p == FRONT_BACK else "wall") for p in poly["patches"]})
        out = run(fe, ["checkMesh"], d, check=False).stdout
    no = re.search(r"Mesh non-orthogonality Max: (\S+) average: (\S+)", out)
    sk = re.search(r"Max skewness = (\S+)", out)
    return {"non_ortho_max": float(no.group(1)), "non_ortho_mean": float(no.group(2)), "skew_max": float(sk.group(1))}


@pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found")
@pytest.mark.parametrize("pid,request_", [
    ("cylinder_channel_dfg_2d1", {"h": 0.03, "quads": True, "boundary_layer": {"groups": ["cylinder", "walls"],
                                                                                "first": 0.002, "layers": 3}}),
    ("square_duct_3d", {"h": 0.002}),
    ("pipe_poiseuille_3d", {"h": 0.0012, "quads": True, "sweep": {"source": "inlet", "layers": 6}}),
])
def test_matches_checkmesh(pid, request_, tmp_path):
    """Same max / mean non-orthogonality and max skewness as OpenFOAM's checkMesh on the same mesh."""
    from ace import fea

    p = fea.get_problem(pid)
    m = fea.mesh_from_request(p, tmp_path / "m.msh", request_)
    assert m["ok"], m.get("error")
    q, ref = m["fv_quality"], _checkmesh(Path(m["path"]), p.dim)
    assert q["non_orthogonality"]["max"] == pytest.approx(ref["non_ortho_max"], rel=1e-3)
    assert q["non_orthogonality"]["mean"] == pytest.approx(ref["non_ortho_mean"], rel=1e-3)
    assert q["skewness"]["max"] == pytest.approx(ref["skew_max"], rel=1e-3)
    assert fv_quality.from_msh(m["path"], p.dim)["non_orthogonality"] == q["non_orthogonality"]
