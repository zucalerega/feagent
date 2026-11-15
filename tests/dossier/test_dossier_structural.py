"""The structural analysis engine of dossier against closed-form solutions.

The parts are small STEP files made here with gmsh's OpenCASCADE kernel, meshed with the groups of the
mesh contract (f<tag>, feature ids, solid; quadratic tetrahedra), in a minimal project:

    (a) section stresses of a stress classification line on analytic fields (exact)
    (b) a thick-walled cylinder with internal pressure and closed-end forces, as a free part: load
        balance, 3-2-1 restraint with negligible reactions, hoop stress against Lame
    (c) a cantilever tube with a moment on its end face against beam theory, and the traction resultant
    (d) the support that follows the thermal contraction: a uniform temperature change gives no stress
    (e) the example project's proof test (slow; needs the geometry, mesh and plan engines)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dossier import model
from dossier.analysis import idealize, scaled, scl, structural
from dossier.pipeline import Options
from dossier.project import Project
from dossier.records import Journal

gmsh = pytest.importorskip("gmsh")

MPa = 1e6


# =========================================================================== (a) section stresses


def _lin(field, start, end, **kw):
    return scl.section(lambda p: np.array([field(x) for x in p]), start, end, **kw)


def test_scl_linear_field_is_exact():
    """A stress that is linear along the line: the membrane is the mid value, the bending the half difference."""
    a = np.array([100.0, -40.0, 20.0, 15.0, -5.0, 8.0]) * MPa        # at the start
    b = np.array([-20.0, 60.0, 50.0, -25.0, 10.0, 0.0]) * MPa        # at the end
    start, end = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.0])

    def field(x):
        s = np.linalg.norm(x - start) / np.linalg.norm(end - start)
        return a + s * (b - a)

    sec = _lin(field, start, end)
    assert sec.membrane == pytest.approx(0.5 * (a + b), rel=1e-12)
    assert sec.bending == pytest.approx(0.5 * (a - b), rel=1e-12)
    assert sec.at_start() == pytest.approx(a, rel=1e-12) and sec.at_end() == pytest.approx(b, rel=1e-12, abs=1e-6)
    assert sec.length == pytest.approx(np.linalg.norm(end - start))
    # the invariant of the TENSOR, not the mean of the invariant
    assert sec.value("von_mises", "membrane") == pytest.approx(float(scl.von_mises(0.5 * (a + b))), rel=1e-12)
    assert sec.value("von_mises", "membrane_bending") == pytest.approx(
        max(float(scl.von_mises(a)), float(scl.von_mises(b))), rel=1e-12)
    assert sec.value("max_principal", "membrane_bending") == pytest.approx(
        max(scl.principals(a)[0], scl.principals(b)[0]), rel=1e-12)
    assert sec.value("syy", "membrane") == pytest.approx(10.0 * MPa, rel=1e-12)
    assert sec.value("sxx", "membrane_bending") == pytest.approx(100.0 * MPa, rel=1e-12)   # the larger end, signed
    assert sec.governing_end("max_principal") in ("start", "end")


def test_scl_cubic_field_and_plate_bending():
    """A cubic stress: the rule is exact (4 Gauss points on each panel). Pure plate bending: no membrane."""
    t = 0.008

    def cubic(x):
        z = x[2] / t                                        # 0 .. 1 through the wall
        s = 5.0 + 3.0 * z - 7.0 * z ** 2 + 4.0 * z ** 3
        return np.array([s, 2 * s, 0, 0, 0, 0]) * MPa

    sec = _lin(cubic, [0, 0, 0], [0, 0, t], panels=3, order=4)
    mean = 5.0 + 3.0 / 2 - 7.0 / 3 + 4.0 / 4
    first = 6.0 * (5.0 * 0 + 3.0 * (1 / 4 - 1 / 3) - 7.0 * (1 / 6 - 1 / 4) + 4.0 * (1 / 8 - 1 / 5))
    assert sec.membrane[0] == pytest.approx(mean * MPa, rel=1e-12)
    assert sec.bending[0] == pytest.approx(first * MPa, rel=1e-12)
    assert sec.membrane[1] == pytest.approx(2 * mean * MPa, rel=1e-12)

    def bending(x):                                          # sigma_xx = sigma_yy = s0 (1 - 2 z / t)
        s = 80.0 * (1 - 2 * x[2] / t)
        return np.array([s, s, 0, 0, 0, 0]) * MPa

    sec = _lin(bending, [0, 0, 0], [0, 0, t])
    assert abs(sec.value("von_mises", "membrane")) < 1e-6 * MPa
    assert sec.value("von_mises", "membrane_bending") == pytest.approx(80.0 * MPa, rel=1e-12)
    assert sec.value("max_principal", "membrane_bending") == pytest.approx(80.0 * MPa, rel=1e-12)
    assert sec.peak_ratio() == pytest.approx(1.0, abs=0.02)   # the samples do not reach the surfaces


def test_scl_lame_cylinder_analytic():
    """The Lame field of a thick cylinder: mean hoop stress p r_i / t, mean radial stress -p r_i / (r_i + r_o)."""
    p, ri, ro = 10 * MPa, 0.02, 0.03
    A, B = p * ri ** 2 / (ro ** 2 - ri ** 2), p * ri ** 2 * ro ** 2 / (ro ** 2 - ri ** 2)

    def lame(x):                                             # a line along y at z = 0: radial = y, hoop = z
        r = x[1]
        return np.array([A, A - B / r ** 2, A + B / r ** 2, 0, 0, 0])   # closed ends: axial stress A

    sec = _lin(lame, [0.05, ri, 0], [0.05, ro, 0])
    assert sec.value("szz", "membrane") == pytest.approx(p * ri / (ro - ri), rel=1e-9)
    assert sec.value("syy", "membrane") == pytest.approx(-p * ri / (ri + ro), rel=1e-9)
    assert sec.value("max_principal", "membrane") == pytest.approx(p * ri / (ro - ri), rel=1e-9)
    # bending part of the hoop stress: 6 / t^2 * integral of (A + B / r^2) (t / 2 - s) ds, in closed form
    t = ro - ri
    exact = 6 / t ** 2 * B * ((t / 2 + ri) * (1 / ri - 1 / ro) - math.log(ro / ri))
    assert sec.bending[2] == pytest.approx(exact, rel=1e-8)
    with pytest.raises(KeyError):
        sec.value("von_mises", "peak")
    with pytest.raises(ValueError):
        scl.section(lambda q: np.zeros((len(q), 3)), [0, 0, 0], [1, 0, 0])


# =========================================================================== test parts


STEEL = dict(id="MAT-T", name="Test steel", E="200 GPa", nu=0.3, rho="7850 kg/m^3", alpha="12e-6 1/K",
             yield_strength="250 MPa", ultimate_strength="400 MPa")


class Part:
    """A test part: a STEP file, meshes with the groups of the mesh contract, a project and its `geo`.

    `features`: {feature id: (predicate on a face description, extra geo fields)}. A face description is
    {"tag", "kind", "area", "centroid", "bbox"} of the imported STEP."""

    def __init__(self, tmp: Path, build, features: dict, interfaces: list[dict], scls: list[dict] = (),
                 reentrant=None, material: dict | None = None):
        self.root = Path(tmp) / "proj"
        (self.root / "comp").mkdir(parents=True)
        (self.root / "out" / "meshes").mkdir(parents=True)
        self.step = self.root / "comp" / "part.step"
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(self.step))
        gmsh.finalize()
        self.faces = self._open()
        gmsh.finalize()
        self.feature_tags = {fid: [f["tag"] for f in self.faces if pred(f)] for fid, (pred, _) in features.items()}
        for fid, tags in self.feature_tags.items():
            assert tags, f"feature {fid} selects no face of {[(f['tag'], f['kind'], f['area']) for f in self.faces]}"
        self.geo = {
            "component": "CMP-T",
            "features": {fid: {"faces": self.feature_tags[fid], **extra} for fid, (_, extra) in features.items()},
            "interfaces": {i["id"]: {role: [t for f in feats for t in self.feature_tags[f]]
                                     for role, feats in i["features"].items()} for i in interfaces},
            "faces": {str(f["tag"]): {"features": [fid for fid, tags in self.feature_tags.items() if f["tag"] in tags]}
                      for f in self.faces},
            "edges": {"reentrant": []},
            "scls": {s["id"]: {"start": list(s["start"]), "end": list(s["end"]), "ok": True, "message": ""} for s in scls},
        }
        if reentrant:
            self.geo["edges"]["reentrant"] = reentrant(self)
        comp = model.Component(
            id="CMP-T", element="EL-T", name="Test part", cad={"file": "part.step"}, material="MAT-T",
            features={fid: {"description": fid, "faces": tags} for fid, tags in self.feature_tags.items()},
            interfaces=[model.Interface(id=i["id"], name=i["id"], connection="IF-T", joint=i.get("joint", "bolted_flange"),
                                        features=i["features"]) for i in interfaces],
            scls=[model.Scl(id=s["id"], description=s["id"], start=s["start"], end=s["end"]) for s in scls])
        system = model.System(id="SYS-T", name="Test system", phases=[{"id": "PH-T", "name": "Test"}],
                              elements=[{"id": "EL-T", "name": "Part", "component": "CMP-T"}])
        mat = model.Material(**(material or STEEL))
        self.project = Project(self.root, model.ProjectFile(id="PRJ-T", name="Test", components=["comp"], subject="CMP-T"),
                               system, {"CMP-T": comp}, {"CMP-T": self.root / "comp"}, {"CMP-T": []}, {"CMP-T": ""},
                               {mat.id: mat}, model.Policy(name="Test policy"), {})

    def _open(self) -> list[dict]:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(str(self.step))
        gmsh.model.occ.synchronize()
        out = []
        for _, t in gmsh.model.getEntities(2):
            out.append({"tag": t, "kind": gmsh.model.getType(2, t), "area": gmsh.model.occ.getMass(2, t),
                        "centroid": np.array(gmsh.model.occ.getCenterOfMass(2, t)),
                        "bbox": np.array(gmsh.model.getBoundingBox(2, t))})
        return out

    def levels(self, sizes: list[float], curvature: int = 12) -> list[model.MeshLevel]:
        """Meshes of quadratic tetrahedra, coarse to fine, with the groups f<tag>, the feature ids and solid."""
        out = []
        for k, h in enumerate(sizes):
            self._open()
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2 * h)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
            for _, t in gmsh.model.getEntities(2):
                gmsh.model.addPhysicalGroup(2, [t], name=f"f{t}")
            for fid, tags in self.feature_tags.items():
                gmsh.model.addPhysicalGroup(2, tags, name=fid)
            gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")
            gmsh.model.mesh.generate(3)
            path = self.root / "out" / "meshes" / f"L{k}.msh"
            gmsh.write(str(path))
            nodes = len(gmsh.model.mesh.getNodes()[0])
            cells = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
            gmsh.finalize()
            out.append(model.MeshLevel(level=k, h=h, nodes=nodes, cells=cells, path=f"meshes/L{k}.msh"))
        return out

    def load_case(self, lc_id: str, loads: list[dict], supports=(), analysis_type="static_structural", **kw):
        items = [model.LoadItem(effect=x.pop("effect", f"EF-{i}"), unit="", **x) for i, x in enumerate(loads)]
        return model.LoadCase(id=lc_id, title=lc_id, analysis_type=analysis_type, component="CMP-T", phase="PH-T",
                              loads=items, supports=list(supports), **kw)

    def run(self, lc, levels, quick=False, **kw):
        opts = Options(quick=quick, pictures=False, log=lambda s: None)
        return structural.run_levels(self.project, None, self.geo, lc, levels, self.project.analysis_dir(lc.id), opts, **kw)


def _checks(rec) -> dict:
    return {c.id.split(rec.id + "-", 1)[1]: c for c in rec.checks}


RI, RO, LEN = 0.020, 0.030, 0.120


def _cylinder(occ):
    a = occ.addCylinder(0, 0, 0, LEN, 0, 0, RO)
    b = occ.addCylinder(0, 0, 0, LEN, 0, 0, RI)
    occ.cut([(3, a)], [(3, b)])


def _cyl_features():
    cyl = lambda f: f["kind"] == "Cylinder"                                   # noqa: E731
    bore = 2 * math.pi * RI * LEN
    return {
        "FT-BORE": (lambda f: cyl(f) and abs(f["area"] - bore) < 0.01 * bore,
                    {"kind": "cylinder", "axis": [1, 0, 0], "radius": RI}),
        "FT-OUTER": (lambda f: cyl(f) and abs(f["area"] - bore) > 0.01 * bore,
                     {"kind": "cylinder", "axis": [1, 0, 0], "radius": RO}),
        "FT-END-A": (lambda f: f["kind"] == "Plane" and abs(f["centroid"][0]) < 1e-9, {"kind": "plane"}),
        "FT-END-B": (lambda f: f["kind"] == "Plane" and abs(f["centroid"][0] - LEN) < 1e-9, {"kind": "plane"}),
    }


CYL_INTERFACES = [{"id": "INT-A", "features": {"seat": ["FT-END-A"], "wetted": ["FT-BORE"]}},
                  {"id": "INT-B", "features": {"seat": ["FT-END-B"]}}]
CYL_SCLS = [{"id": "SCL-WALL", "start": (0.06, RI, 0.0), "end": (0.06, RO, 0.0)},
            {"id": "SCL-SKEW", "start": (0.045, 0.0, -RI), "end": (0.045, 0.0, -RO)}]


@pytest.fixture(scope="module")
def cylinder(tmp_path_factory):
    part = Part(tmp_path_factory.mktemp("cyl"), _cylinder, _cyl_features(), CYL_INTERFACES, CYL_SCLS)
    part.mesh3 = part.levels([0.0075, 0.0054, 0.0039])
    return part


def _pressure_case(part, p, lc_id="LC-PV", ends=(1.0, 1.0), keys=()):
    thrust = p * math.pi * RI ** 2
    loads = [dict(effect="EF-P", interface="INT-A", kind="pressure", role="wetted", features=["FT-BORE"], si=p),
             dict(effect="EF-FA", interface="INT-A", kind="force", role="seat", features=["FT-END-A"],
                  si=[-ends[0] * thrust, 0, 0]),
             dict(effect="EF-FB", interface="INT-B", kind="force", role="seat", features=["FT-END-B"],
                  si=[ends[1] * thrust, 0, 0])]
    return part.load_case(lc_id, loads, result_keys=list(keys))


# =========================================================================== (b) thick-walled cylinder, free


def test_free_cylinder_against_lame(cylinder):
    p = 10 * MPa
    lc = _pressure_case(cylinder, p, keys=["displacement.max@part", "radial_displacement.range@FT-BORE",
                                           "von_mises.peak@FT-OUTER", "temperature.max@part"])
    out = cylinder.run(lc, cylinder.mesh3)
    rec, j = out.record, out.journal
    assert rec.status == "done" and rec.id == "AN-PV" and rec.load_case == "LC-PV"
    assert [m.level for m in rec.mesh] == [0, 1, 2] and rec.mesh[0].h_rep > rec.mesh[1].h_rep > rec.mesh[2].h_rep
    ck = _checks(rec)
    # the loads of the free part balance; three nodes hold it and carry nothing
    assert ck["BALANCE"].passed and ck["BALANCE"].value < 1e-6
    assert ck["RESTRAINT"].passed and ck["RESTRAINT"].value < 1e-8
    assert ck["EQUILIBRIUM"].passed and ck["ASSEMBLY"].passed and ck["MESH-QUALITY"].passed
    assert ck["ERROR-ESTIMATE"].value is not None and ck["ERROR-ESTIMATE"].value < 0.1
    free = [b for b in rec.bcs if b.type == "displacement"]
    assert len(free) == 1 and [len(x["components"]) for x in free[0].value["points"]] == [3, 2, 1]
    assert any(d.id == "DEC-LC-PV-IDEALIZE-FREE" for d in j.decisions)
    # Lame: sigma_hoop(r) = A + B / r^2, sigma_r = A - B / r^2, axial A (closed ends)
    A = p * RI ** 2 / (RO ** 2 - RI ** 2)
    hoop_i, hoop_mean = A * (1 + RO ** 2 / RI ** 2), p * RI / (RO - RI)
    vm_i = math.sqrt(0.5 * ((hoop_i + p) ** 2 + (-p - A) ** 2 + (A - hoop_i) ** 2))
    r = rec.results
    for sid in ("SCL-WALL", "SCL-SKEW"):
        m = r[f"max_principal.membrane@{sid}"]
        assert m.value == pytest.approx(hoop_mean, rel=0.01) and m.status == "converged" and len(m.per_level) == 3
        assert r[f"sxx.membrane@{sid}"].value == pytest.approx(A, rel=0.02)
    assert r["szz.membrane@SCL-WALL"].value == pytest.approx(hoop_mean, rel=0.01)
    assert r["syy.membrane@SCL-WALL"].value == pytest.approx(-p * RI / (RI + RO), rel=0.03)
    assert r["syy.membrane@SCL-SKEW"].value == pytest.approx(hoop_mean, rel=0.01)       # the hoop direction there is y
    # membrane + bending: the linearized stress at the bore, from the closed-form integrals of the Lame field
    B = A * RO ** 2
    t = RO - RI
    b_hoop = 6 / t ** 2 * B * ((t / 2 + RI) * (1 / RI - 1 / RO) - math.log(RO / RI))
    lin_i = np.array([A, -p * RI / (RI + RO) - b_hoop, hoop_mean + b_hoop, 0, 0, 0])   # sxx, syy (radial), szz (hoop)
    assert r["von_mises.membrane_bending@SCL-WALL"].value == pytest.approx(float(scl.von_mises(lin_i)), rel=0.02)
    assert r["max_principal.membrane_bending@SCL-WALL"].value == pytest.approx(hoop_mean + b_hoop, rel=0.02)
    # the peak: the von Mises stress at the bore, within a few percent; the engine says where it is
    peak = r["von_mises.peak@part"]
    assert peak.value == pytest.approx(vm_i, rel=0.03)
    assert "FT-BORE" in peak.near and math.hypot(peak.location[1], peak.location[2]) == pytest.approx(RI, rel=0.02)
    assert peak.uncertainty is not None and abs(peak.value - vm_i) / vm_i <= peak.uncertainty + 0.01
    assert r["von_mises.peak@FT-OUTER"].value < peak.value
    agg = r["von_mises.membrane@scls"]
    assert agg.near in ("SCL-WALL", "SCL-SKEW") and agg.value == pytest.approx(
        max(r["von_mises.membrane@SCL-WALL"].value, r["von_mises.membrane@SCL-SKEW"].value))
    # a free part: displacement relative to the mean rigid-body motion; the bore stays round
    u_i = RI / 200e9 * (hoop_i - 0.3 * (-p + A))                              # radial displacement of the bore
    u_ax = 0.5 * LEN / 200e9 * (A - 0.3 * (hoop_i - p))                       # half the change of length (at the bore)
    assert r["displacement.max@part"].status == "converged"
    assert r["displacement.max@part"].value > max(u_i, abs(u_ax))
    assert r["displacement.max@part"].value < 1.05 * math.hypot(RO / 200e9 * 2 * A, u_ax * 1.1) + u_i
    assert r["radial_displacement.range@FT-BORE"].value < 0.02 * u_i
    assert any(d.id == "DEC-LC-PV-DISPLACEMENT-DATUM" for d in j.decisions)
    assert any(d.id == "DEC-LC-PV-ROUNDNESS" and d.needs_review for d in j.decisions)
    # a key of another engine is named, not invented
    assert "temperature.max@part" not in r and "temperature.max@part" in rec.message
    # the fields of the two finest meshes, for the fatigue engine
    fields = [a for a in rec.artifacts if a.kind == "field"]
    assert [Path(a.path).name for a in fields] == ["fields_L1.npz", "fields_L2.npz"]
    z = np.load(cylinder.project.out / fields[-1].path)
    assert z["x"].shape == (rec.mesh[-1].nodes, 3) and z["stress"].shape == (rec.mesh[-1].nodes, 6)
    assert z["u"].shape == (rec.mesh[-1].nodes, 3)
    hoop = z["stress"][:, 1] * 0                                              # hoop stress at the nodes of the bore
    rr = np.hypot(z["x"][:, 1], z["x"][:, 2])
    c, s = z["x"][:, 1] / rr, z["x"][:, 2] / rr
    hoop = z["stress"][:, 1] * s * s + z["stress"][:, 2] * c * c - 2 * z["stress"][:, 4] * s * c
    mid = (abs(z["x"][:, 0] - LEN / 2) < 0.3 * LEN)
    bore = mid & (rr < RI * 1.001)
    outer = mid & (rr > RO * 0.999)
    assert np.median(hoop[bore]) == pytest.approx(hoop_i, rel=0.02)
    assert np.median(hoop[outer]) == pytest.approx(2 * A, rel=0.02)
    assert np.abs(hoop[bore] / hoop_i - 1).max() < 0.06                        # nodal scatter of the recovered stress
    # text for people: the statement names the method; the model is JSON
    assert "finite element" in rec.statement and rec.material["E"] == pytest.approx(200e9)
    json.dumps(rec.model_dump(mode="json"))


def test_free_part_that_does_not_balance_fails(cylinder):
    lc = _pressure_case(cylinder, 10 * MPa, lc_id="LC-UNBAL", ends=(1.0, 0.0))
    out = cylinder.run(lc, cylinder.mesh3[:1])
    rec = out.record
    assert rec.status == "failed" and not rec.results
    assert "must balance" in rec.message and "net force" in rec.message
    assert rec.checks and rec.checks[0].passed is False and rec.checks[0].value == pytest.approx(1.0, rel=1e-3)


def test_small_imbalance_goes_to_inertia_relief(cylinder):
    """Loads that balance to 0.5 %: the residual becomes a rigid-body acceleration field, not a point reaction."""
    lc = _pressure_case(cylinder, 10 * MPa, lc_id="LC-NEAR", ends=(1.0, 1.005))
    out = cylinder.run(lc, cylinder.mesh3[:2])
    rec = out.record
    ck = _checks(rec)
    assert rec.status == "done"
    assert ck["BALANCE"].passed and ck["BALANCE"].value == pytest.approx(0.005, rel=0.02)
    assert ck["RESTRAINT"].passed and ck["RESTRAINT"].value < 1e-8
    assert any(b.type == "body_force" and "residual" in b.description for b in rec.bcs)
    assert rec.results["max_principal.membrane@SCL-WALL"].value == pytest.approx(10 * MPa * RI / (RO - RI), rel=0.02)
    assert all(v.status in ("converged", "not_converged", "singular") and len(v.per_level) == 2 for v in rec.results.values())


# =========================================================================== (c) cantilever tube with a moment


TRI, TRO, TLEN, XS = 0.012, 0.020, 0.200, 0.120


def _tube(occ):
    a = occ.addCylinder(0, 0, 0, TLEN, 0, 0, TRO)
    b = occ.addCylinder(0, 0, 0, TLEN, 0, 0, TRI)
    occ.cut([(3, a)], [(3, b)])


def _tube_features():
    bore = 2 * math.pi * TRI * TLEN
    return {
        "FT-IN": (lambda f: f["kind"] == "Cylinder" and abs(f["area"] - bore) < 0.01 * bore,
                  {"kind": "cylinder", "axis": [1, 0, 0], "radius": TRI}),
        "FT-OUT": (lambda f: f["kind"] == "Cylinder" and abs(f["area"] - bore) > 0.01 * bore,
                   {"kind": "cylinder", "axis": [1, 0, 0], "radius": TRO}),
        "FT-ROOT": (lambda f: f["kind"] == "Plane" and abs(f["centroid"][0]) < 1e-9, {"kind": "plane"}),
        "FT-TIP": (lambda f: f["kind"] == "Plane" and abs(f["centroid"][0] - TLEN) < 1e-9, {"kind": "plane"}),
    }


TUBE_INTERFACES = [{"id": "INT-ROOT", "features": {"seat": ["FT-ROOT"]}},
                   {"id": "INT-TIP", "features": {"seat": ["FT-TIP"]}},
                   {"id": "INT-SLIDE", "joint": "sliding", "features": {"seat": ["FT-ROOT"]}},
                   {"id": "INT-EMPTY", "features": {"wetted": ["FT-IN"]}}]
TUBE_SCLS = [{"id": "SCL-TOP", "start": (XS, 0.0, TRI), "end": (XS, 0.0, TRO)}]


@pytest.fixture(scope="module")
def tube(tmp_path_factory):
    part = Part(tmp_path_factory.mktemp("tube"), _tube, _tube_features(), TUBE_INTERFACES, TUBE_SCLS)
    part.mesh2 = part.levels([0.0062, 0.0044])
    return part


def test_moment_on_flange_face_against_beam_theory(tube):
    """Bending M_y and torque T on the free end face of a cantilever tube: sigma = M z / I, tau = T r / J."""
    T, My = 60.0, 150.0
    lc = tube.load_case("LC-MOM", [dict(effect="EF-M", interface="INT-TIP", kind="moment", role="seat",
                                        features=["FT-TIP"], si=[T, My, 0.0])],
                        supports=["INT-ROOT"], result_keys=["displacement.max@FT-TIP", "sxx.membrane_bending@SCL-TOP"])
    out = tube.run(lc, tube.mesh2, quick=True)
    rec, j = out.record, out.journal
    assert rec.status == "done"
    ck = _checks(rec)
    # the traction field is statically equivalent to the moment, and the support reactions balance it
    m = ck["MOMENT-EF-M"]
    assert m.passed and m.value < 1e-9 and m.limit == pytest.approx(0.01) and m.kind == "load_resultant"
    assert ck["EQUILIBRIUM"].passed and ck["ASSEMBLY"].passed
    bc = next(b for b in rec.bcs if b.type == "traction")
    assert bc.value["moment"] == pytest.approx([T, My, 0.0]) and len(bc.value["expressions"]) == 3
    assert bc.decision == "DEC-LC-MOM-IDEALIZE-MOMENT" and bc.effect == "EF-M"
    dec = next(d for d in j.decisions if d.id == bc.decision)
    assert not dec.needs_review and len(dec.alternatives) == 2            # a flat face: the classical field
    # the traction itself: sigma_xx = M_y z / I on the face, shear T r / J around it
    I = math.pi / 4 * (TRO ** 4 - TRI ** 4)
    J = 2 * I
    pts = np.array([[TLEN, 0.0, TRO], [TLEN, TRO, 0.0], [TLEN, 0.0, -0.016]])
    t = idealize.eval_exprs(bc.value["expressions"], pts)
    assert t[0] == pytest.approx([My * TRO / I, -T * TRO / J, 0.0], rel=1e-4, abs=1.0)
    assert t[1] == pytest.approx([0.0, 0.0, T * TRO / J], rel=1e-4, abs=1.0)
    assert t[2] == pytest.approx([-My * 0.016 / I, T * 0.016 / J, 0.0], rel=1e-4, abs=1.0)
    # section stresses three diameters from the held face
    rm = 0.5 * (TRI + TRO)
    sig, tau = My * rm / I, T * rm / J
    r = rec.results
    assert r["sxx.membrane@SCL-TOP"].value == pytest.approx(sig, rel=0.02)
    assert r["sxx.membrane_bending@SCL-TOP"].value == pytest.approx(My * TRO / I, rel=0.03)
    assert r["von_mises.membrane@SCL-TOP"].value == pytest.approx(math.hypot(sig, math.sqrt(3) * tau), rel=0.02)
    assert r["max_principal.membrane@SCL-TOP"].value == pytest.approx(sig / 2 + math.hypot(sig / 2, tau), rel=0.02)
    assert r["von_mises.membrane_bending@SCL-TOP"].value == pytest.approx(
        math.hypot(My * TRO / I, math.sqrt(3) * T * TRO / J), rel=0.03)
    assert all(len(v.per_level) == 2 and v.uncertainty is not None for k, v in r.items() if "@SCL-TOP" in k)
    # tip displacement: bending deflection M L^2 / (2 E I), section rotation M L / (E I), twist T L / (G J)
    E, G = 200e9, 200e9 / 2.6
    kap, phi = My / (E * I), T * TLEN / (G * J)
    th = np.linspace(0, 2 * math.pi, 721)
    y, z = TRO * np.cos(th), TRO * np.sin(th)
    u = np.sqrt((kap * TLEN * z) ** 2 + (phi * z) ** 2 + (-kap * TLEN ** 2 / 2 + phi * y) ** 2)
    assert r["displacement.max@FT-TIP"].value == pytest.approx(u.max(), rel=0.02)
    assert r["displacement.max@FT-TIP"].status == "converged"
    # the held face: fixed, a decision for review, a zone without peak evaluation
    held = next(b for b in rec.bcs if b.type == "fixed")
    assert held.interface == "INT-ROOT" and held.features == ["FT-ROOT"] and held.faces
    ids = {d.id: d for d in j.decisions}
    assert ids["DEC-LC-MOM-IDEALIZE-SUPPORT"].needs_review and ids["DEC-LC-MOM-PEAK-ZONE"].needs_review
    assert any(a.id == "ASM-LC-MOM-RIGID-NEIGHBOUR" for a in j.assumptions)
    peak = r["von_mises.peak@part"]
    zone = 2.0 * rec.mesh[0].h_rep
    assert peak.location[0] > zone and f"{1e3 * zone:.3g}"[:3] in ids["DEC-LC-MOM-PEAK-ZONE"].choice
    assert peak.value == pytest.approx(math.hypot(My * TRO / I, math.sqrt(3) * T * TRO / J), rel=0.08)


def test_force_with_a_point_of_application_gets_the_offset_moment(tube):
    """A transverse force 50 mm beyond the end face: the engine adds the moment of the offset."""
    Fz, e = 400.0, 0.050
    lc = tube.load_case("LC-OFF", [dict(effect="EF-F", interface="INT-TIP", kind="force", role="seat",
                                        features=["FT-TIP"], si=[0.0, 0.0, Fz], at_si=[TLEN + e, 0.0, 0.0])],
                        supports=["INT-ROOT"])
    out = tube.run(lc, tube.mesh2[:1])
    rec = out.record
    ck = _checks(rec)
    assert rec.status == "done" and ck["MOMENT-EF-F-OFFSET"].passed and ck["EQUILIBRIUM"].passed
    kinds = [b.type for b in rec.bcs]
    assert kinds.count("force") == 1 and kinds.count("traction") == 1
    off = next(b for b in rec.bcs if b.type == "traction")
    assert off.value["moment"] == pytest.approx([0.0, -e * Fz, 0.0], abs=1e-9)
    I = math.pi / 4 * (TRO ** 4 - TRI ** 4)
    My = -(TLEN + e - XS) * Fz                                    # bending moment at the section of the line
    assert rec.results["sxx.membrane@SCL-TOP"].value == pytest.approx(My * 0.5 * (TRI + TRO) / I, rel=0.03)
    assert rec.results["sxx.membrane@SCL-TOP"].status == "not_converged"          # one mesh: no study
    assert rec.results["sxx.membrane@SCL-TOP"].uncertainty is None


# =========================================================================== (d) thermal contraction of a held part


def _uniform_field(temperature: float):
    """A stand-in for dossier.analysis.thermal.solve_temperature: one temperature at every node."""
    from feacore.case import write_temperature_field

    def solve(project, ctx, geo, thermal_lc, mesh, opts):
        x = idealize.MeshGeom(project.out / mesh.path).x
        T = np.full(len(x), float(temperature))
        path = project.analysis_dir(thermal_lc.id) / f"T_L{mesh.level}.npz"
        write_temperature_field(path, x, T)
        return {"x": x, "T": T, "path": str(path)}
    return solve


def test_held_part_contracts_freely_with_the_neighbour(tube):
    """90 K against a reference of 293.15 K: a fixed face gives hundreds of MPa; the engine's support gives none."""
    from feacore.case import BC, Analysis, Case, Load, Material
    from feacore.solver import solve

    T_cold, T_ref, alpha, E, nu = 90.0, 293.15, 12e-6, 200e9, 0.3
    th = model.LoadCase(id="LC-TH-COLD", title="cold", analysis_type="thermal_steady", component="CMP-T", phase="PH-T")
    tube.project.write_stage("plan", load_cases=[th])
    lc = tube.load_case("LC-COLD", [dict(effect="EF-T", interface="INT-ROOT", kind="fluid_temperature", role="wetted",
                                         features=["FT-IN"], si=T_cold)],
                        supports=["INT-ROOT"], analysis_type="thermo_structural", uses=["LC-TH-COLD"],
                        result_keys=["displacement.max@FT-TIP"])
    out = tube.run(lc, tube.mesh2[:1], thermal_solver=_uniform_field(T_cold))
    rec, j = out.record, out.journal
    assert rec.status == "done", rec.message
    false_stress = E * alpha * (T_ref - T_cold) / (1 - nu)                  # of a fully fixed face: 696 MPa
    peak = rec.results["von_mises.peak@part"]
    assert peak.value < 1e-6 * false_stress
    # the part contracts about the centroid of the held face: the far rim of the tip moves most
    free = alpha * (T_ref - T_cold) * math.hypot(TLEN, TRO)
    assert rec.results["displacement.max@FT-TIP"].value == pytest.approx(free, rel=1e-6)
    sup = next(b for b in rec.bcs if b.type == "displacement")
    assert sup.value["mean_temperature"] == pytest.approx(T_cold) and sup.value["components"] == [0, 1, 2]
    assert sup.value["thermal_strain"] == pytest.approx(alpha * (T_cold - T_ref))
    assert sup.value["centroid"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-7)
    assert any(b.type == "temperature" and b.value["T_ref"] == pytest.approx(T_ref) for b in rec.bcs)
    dec = next(d for d in j.decisions if d.id == "DEC-LC-COLD-IDEALIZE-SUPPORT")
    assert dec.needs_review and len(dec.alternatives) == 2 and "fully fixed" in dec.alternatives[0].option
    asm = next(a for a in j.assumptions if a.id == "ASM-LC-COLD-NEIGHBOUR-EXPANSION")
    assert asm.conservative is False and "not in the results" in asm.impact
    assert any(lk.to == "AN-TH-COLD" for lk in rec.links)
    assert _checks(rec)["EQUILIBRIUM"].passed
    # the trap, for comparison: the same solve with the face fully fixed
    mesh = tube.project.out / tube.mesh2[0].path
    root = [f"f{t}" for t in tube.feature_tags["FT-ROOT"]]
    case = Case(mesh=str(mesh), analysis=Analysis(physics="elasticity", formulation="3d", order=2, T_ref=T_ref),
                materials={"m": Material(E=E, nu=nu, alpha=alpha)}, bcs=[BC(type="fixed", group=g) for g in root],
                loads=[Load(type="temperature", value=T_cold)])
    vm = solve(case, backend="skfem").nodal("von_mises")[1]
    assert vm.max() > 0.5 * false_stress


def test_thermo_structural_case_needs_its_thermal_case(tube):
    lc = tube.load_case("LC-NOTH", [], supports=["INT-ROOT"], analysis_type="thermo_structural", uses=["LC-TH-MISSING"])
    rec = tube.run(lc, tube.mesh2[:1]).record
    assert rec.status == "not_supported" and "thermal load case" in rec.message


# =========================================================================== loads without a model


def test_loads_the_engine_cannot_idealize_are_not_supported(tube):
    cases = {
        "LC-NOFEAT": (dict(effect="EF-X", interface="INT-EMPTY", kind="force", role="seat", features=[], si=[1.0, 0, 0]),
                      ["INT-ROOT"], "has no features"),
        "LC-DISP": (dict(effect="EF-D", interface="INT-TIP", kind="displacement", role="seat", features=["FT-TIP"],
                         si=[0, 0, 1e-4]), ["INT-ROOT"], "no rule for a load of the kind displacement"),
        "LC-SLIDE": (dict(effect="EF-P", interface="INT-TIP", kind="pressure", role="seat", features=["FT-TIP"], si=1e6),
                     ["INT-SLIDE"], "joint type sliding"),
        "LC-NOSEAT": (dict(effect="EF-P", interface="INT-TIP", kind="pressure", role="seat", features=["FT-TIP"], si=1e6),
                      ["INT-EMPTY"], "has no seat features"),
    }
    for lc_id, (load, supports, text) in cases.items():
        out = tube.run(tube.load_case(lc_id, [load], supports=supports), tube.mesh2[:1])
        assert out.record.status == "not_supported" and text in out.record.message, (lc_id, out.record.message)
        assert not out.record.results
        assert any(f.subject == lc_id and f.category == "model" for f in out.journal.findings)
    lc = tube.load_case("LC-FLOW", [], supports=["INT-ROOT"], analysis_type="flow")
    assert tube.run(lc, tube.mesh2[:1]).record.status == "not_supported"


# =========================================================================== the convergence rule


def test_convergence_rule_monotone_order_and_absolute_uncertainty():
    """Monotone with the theoretical order: Richardson value, the order, u = max(1.25 GCI, 3 x GCI(p_theory))."""
    st = structural.study([4.0, 2.0, 1.0], [100.0, 101.0, 101.25], 2.0, 0.02)
    assert st.kind == "monotone" and st.status == "converged"
    assert st.order == pytest.approx(2.0) and st.extrapolated == pytest.approx(101.25 + 0.25 / 3)
    assert st.absolute == pytest.approx(3.0 * 0.25 / 3.0)               # the conservative index governs
    assert st.uncertainty == pytest.approx(st.absolute / 101.25)


def test_convergence_rule_order_far_above_theory_is_no_order():
    """A change that stops abruptly gives a huge observed order: not asymptotic, so no order, no
    extrapolation, and u = 3 x the spread of the three values."""
    st = structural.study([4.0, 2.0, 1.0], [100.0, 110.0, 110.0001], 2.0, 0.02)
    assert st.kind == "monotone" and st.order is None and st.extrapolated is None
    assert st.absolute == pytest.approx(3.0 * 10.0001) and st.status == "not_converged"


def test_convergence_rule_oscillation_uses_the_spread_of_the_last_three_values():
    st = structural.study([4.0, 2.0, 1.0], [100.0, 103.0, 101.5], 2.0, 0.10)
    assert st.kind == "oscillatory" and st.order is None and st.estimated
    assert st.absolute == pytest.approx(3.0 * 3.0) and st.status == "converged"         # 3 x (103 - 100)
    assert structural.study([4.0, 2.0, 1.0], [100.0, 103.0, 101.5], 2.0, 0.05).status == "not_converged"
    # alternating with growing changes: the spread
    st = structural.study([4.0, 2.0, 1.0], [100.0, 101.0, 98.0], 2.0, 0.5)
    assert st.kind == "oscillatory_divergent" and st.absolute == pytest.approx(9.0)


def test_convergence_rule_divergence_two_levels_reference_and_noise():
    st = structural.study([4.0, 2.0, 1.0], [100.0, 101.0, 103.0], 2.0, 0.5)
    assert st.kind == "divergent" and st.absolute == pytest.approx(9.0) and st.grew
    st = structural.study([2.0, 1.0], [100.0, 101.0], 2.0, 0.05)       # r = 2: factor 1 / (r^2 - 1) < 1 is not used
    assert st.kind == "two_levels" and st.absolute == pytest.approx(3.0) and st.status == "converged"
    st = structural.study([1.2, 1.0], [100.0, 101.0], 2.0, 0.05)       # r = 1.2: 3 x 1 / 0.44
    assert st.absolute == pytest.approx(3.0 / (1.44 - 1.0)) and st.status == "not_converged"
    # a small component of a large field is judged against the field (the reference)
    st = structural.study([4.0, 2.0, 1.0], [0.5, 0.2, 0.1], 2.0, 0.02, reference=100.0)
    assert st.status == "converged" and st.uncertainty > 1.0 and st.absolute == pytest.approx(0.15)
    assert structural.study([4.0, 2.0, 1.0], [0.5, 0.2, 0.1], 2.0, 0.02).status == "not_converged"
    # the line noise is a floor of the uncertainty
    st = structural.study([4.0, 2.0, 1.0], [100.0, 101.0, 101.25], 2.0, 0.02, noise=1.0)
    assert st.absolute == pytest.approx(3.0) and st.status == "not_converged"
    assert structural.study([1.0], [5.0], 2.0, 0.02).status == "not_converged"


def test_aggregate_covers_every_line_with_its_absolute_uncertainty():
    def rv(v, u, status="converged"):
        return model.ResultValue(key="k", value=v, unit="Pa", uncertainty=None if u is None else u / v,
                                 uncertainty_abs=u, status=status, per_level=[v])
    agg = structural._aggregate("von_mises.membrane@scls", {"A": rv(100.0, 1.0), "B": rv(95.0, 10.0, "not_converged")}, [])
    assert agg.value == 100.0 and agg.near == "A"
    assert agg.uncertainty_abs == pytest.approx(5.0) and agg.uncertainty == pytest.approx(0.05)
    assert agg.status == "not_converged"                     # B can reach the worst value
    agg = structural._aggregate("von_mises.membrane@scls", {"A": rv(100.0, 1.0), "B": rv(50.0, 10.0, "not_converged")}, [])
    assert agg.uncertainty_abs == pytest.approx(1.0) and agg.status == "converged"


def test_scaled_result_scales_the_absolute_uncertainty():
    r = model.ResultValue(key="von_mises.peak@part", value=100.0, unit="Pa", uncertainty=0.02, uncertainty_abs=2.0,
                          status="converged", per_level=[99.0, 100.0])
    s = scaled.scale_result(r, 1.5, "AN-BASE")
    assert s.value == 150.0 and s.uncertainty_abs == pytest.approx(3.0) and s.uncertainty == 0.02


# =========================================================================== (e) the example's proof test


@pytest.mark.slow
def test_example_proof_test_quick(workproject):
    """LC-PROOF of the example (a free part under the proof pressure) with Options(quick=True): two levels,
    a done record with every SCL result, the peak at a re-entrant edge never converged, the information-only
    error estimate with passed=None, and no failed check. About 40 s."""
    from dossier import context as context_mod
    from dossier import pipeline
    opts = Options(quick=True, log=lambda s: None)
    pipeline.run(workproject, "context", "plan", opts)
    ctx = context_mod.load(workproject)
    geo = pipeline.load_geometry(workproject)
    lc = next(x for x in workproject.collection("load_cases") if x.id == "LC-PROOF")
    rec = pipeline.analyze_one(workproject, ctx, geo, lc, opts).record
    assert rec.status == "done", rec.message
    assert len(rec.mesh) == 2
    peak = rec.results["von_mises.peak@part"]
    assert peak.status in ("singular", "not_converged") and peak.uncertainty is None and peak.order is None
    for key in ("von_mises.membrane@scls", "von_mises.membrane_bending@scls"):
        r = rec.results[key]
        assert r.uncertainty_abs is not None and 1e6 < r.value < peak.value
    for r in rec.results.values():
        if r.uncertainty is not None and r.uncertainty_abs is not None and r.value:
            assert r.uncertainty_abs == pytest.approx(r.uncertainty * abs(r.value))
    assert not [c for c in rec.checks if c.passed is False]
    zz = [c for c in rec.checks if c.kind == "error_estimate"]
    assert zz and all(c.passed is None and c.limit is None for c in zz)


# =========================================================================== review fixes: honest uncertainty

H3 = [0.0032234, 0.0022627, 0.0016858]          # the h_rep of the levels of the first full run


def _level_data(key, values, h=H3, where=None, info=None):
    out = []
    for k, (hk, v) in enumerate(zip(h, values)):
        d = structural.LevelData(level=model.MeshLevel(level=k, h=hk, nodes=10, cells=10, path=f"meshes/L{k}.msh"), h=hk)
        d.values[key] = v
        if where is not None:
            d.where[key] = {"location": [0.0, 0.0, 0.0], "near": "FT-X", "edge": False, "support": False, "h": hk,
                            **where}
        if info is not None:
            d.info[key] = dict(info)
        out.append(d)
    return out


def _lc(keys=()):
    return model.LoadCase(id="LC-T", title="LC-T", analysis_type="static_structural", component="CMP-T",
                          result_keys=list(keys))


def test_error_estimated_only_for_a_sequence_that_converges():
    est = {name: structural.study(h, v, 2.0, 0.05) for name, h, v in [
        ("monotone", H3, [105.0, 102.464, 101.368]),
        ("oscillatory", H3, [100.0, 120.0, 110.0]),
        ("divergent", H3, [100.0, 110.0, 121.0]),
        ("indeterminate", H3, [100.0, 100.0, 130.0]),
        ("two_levels", H3[1:], [100.0, 100.5]),
        ("one_level", H3[2:], [100.0])]}
    assert {n: s.kind for n, s in est.items()} == {n: n for n in est}
    assert {n: s.estimated for n, s in est.items()} == {"monotone": True, "oscillatory": True, "divergent": False,
                                                        "indeterminate": False, "two_levels": True, "one_level": False}
    # a spread of values that do not converge is not an error estimate, so the result is never converged
    st = structural.study([4.0, 2.0, 1.0], [100.0, 100.4, 101.0], 2.0, 0.05)       # 3 x spread = 3 %: under 5 %
    assert st.kind == "divergent" and not st.estimated and st.status == "not_converged"
    # no trend, but each change is inside the scatter of independent meshes (under 0.2 %): an estimate
    st = structural.study(H3, [2.82849, 2.83090, 2.82827], 3.0, 0.05)
    assert st.kind == "oscillatory_divergent" and st.estimated and st.status == "converged"


def test_oscillation_uses_three_times_the_spread_of_the_last_three_values():
    # the checker's case: 80, 100, 99.9. The last change (0.1) gave u = 0.3 = 0.3 %, converged. The spread is 20.
    st = structural.study(H3, [80.0, 100.0, 99.9], 2.0, 0.05)
    assert st.kind == "oscillatory" and st.estimated
    assert st.absolute == pytest.approx(60.0) and st.status == "not_converged"


def test_a_growing_peak_away_from_every_listed_edge_is_singular():
    """The checker's case: a peak that grows like h^-0.3 (200, 222.4, 242.9 MPa) at a place that the list of
    re-entrant edges misses. It was not_converged with u = 3 x spread = 128.8 MPa, and the compliance stage passed
    it against 600 MPa with k = 2. Now it is singular with no uncertainty."""
    from dossier import compliance as C

    key = "von_mises.peak@part"
    v = [200e6 * (H3[0] / h) ** 0.3 for h in H3]
    out, missing = structural._results(_level_data(key, v, where={}), [key], {}, 0.05, _lc([key]))
    rv = out[key]
    assert not missing and rv.status == "singular" and not rv.error_estimated
    assert rv.uncertainty is None and rv.uncertainty_abs is None and rv.order is None
    assert "The peak grew at each refinement and the values do not converge." in rv.note

    class Cr:
        comparator = "<="
    e = C._evaluate(rv, key, Cr, C._Limit(600e6, "Pa", "test", "limit"), C._Factor(1.0, "test"), "AN-T", 0.05, k=2.0)
    assert e.verdict == "inconclusive"
    # the old record (not converged with the spread as its uncertainty) is inconclusive too: no error estimate
    old = rv.model_copy(update={"status": "not_converged", "uncertainty_abs": 128.8e6, "uncertainty": 128.8 / 242.9})
    e = C._evaluate(old, key, Cr, C._Limit(600e6, "Pa", "test", "limit"), C._Factor(1.0, "test"), "AN-T", 0.05, k=2.0)
    assert e.verdict == "inconclusive" and e.k is None and e.cause == "the result has no error estimate"


def test_a_result_that_a_criterion_uses_is_judged_against_its_own_value():
    """Roundness: a range of 1.00 um from two levels (1.02 um, 1.00 um) has u = 3 x 0.02 = 0.06 um = 6 %. Against
    the reference of the feature (20 um of rigid-body motion and contraction) it read as converged (0.3 %)."""
    key = "radial_displacement.range@FT-SEAT"
    data = _level_data(key, [1.02e-6, 1.00e-6], h=[0.002, 0.001], info={"ref": 20e-6})
    judged, _ = structural._results(data, [key], {}, 0.05, _lc([key]))
    assert judged[key].status == "not_converged" and judged[key].uncertainty == pytest.approx(0.06)
    assert "not with the size of its field" in judged[key].note
    free, _ = structural._results(data, [key], {}, 0.05, _lc([]))                  # nothing judges it
    assert free[key].status == "converged"
    # a line of an @scls criterion is judged too
    line = "sxx.membrane@SCL-A"
    data = _level_data(line, [1.02e6, 1.00e6], h=[0.002, 0.001], info={"ref": 100e6})
    out, _ = structural._results(data, [line], {"SCL-A": {}}, 0.05, _lc(["sxx.membrane@scls"]))
    assert out[line].status == "not_converged"


def test_scaled_result_carries_the_error_estimated_flag_of_its_base():
    for flag in (True, False):
        r = model.ResultValue(key="von_mises.peak@part", value=100.0, unit="Pa", uncertainty=0.02, uncertainty_abs=2.0,
                              status="not_converged", per_level=[99.0, 100.0], error_estimated=flag)
        assert scaled.scale_result(r, 1.5, "AN-BASE").error_estimated is flag


def test_acceleration_sign_is_an_assumption(tube):
    lc = tube.load_case("LC-ACC", [dict(effect="EF-G", interface=None, kind="acceleration", role="", features=[],
                                        si=[0.0, 0.0, -9.81 * 20])], supports=["INT-ROOT"])
    j = Journal("solve", "engine:test")
    idealize.Idealizer(tube.project, tube.geo, lc, j)
    asm = [a for a in j.assumptions if a.id == "ASM-LC-ACC-ACCELERATION-SIGN"]
    assert len(asm) == 1
    assert "opposite sign" in asm[0].verify_by and "tension side" in asm[0].impact
    assert any(l.to == "EF-G" for l in asm[0].links)
    dec = next(d for d in j.decisions if d.id == "DEC-LC-ACC-IDEALIZE-ACCELERATION")
    assert "direction of the inertial load" not in dec.rationale and "one sign" in dec.rationale
    # no acceleration: no such assumption
    j2 = Journal("solve", "engine:test")
    idealize.Idealizer(tube.project, tube.geo, tube.load_case("LC-P", [dict(
        effect="EF-P", interface="INT-TIP", kind="pressure", role="seat", features=["FT-TIP"], si=1e6)],
        supports=["INT-ROOT"]), j2)
    assert not [a for a in j2.assumptions if a.id.endswith("ACCELERATION-SIGN")]
