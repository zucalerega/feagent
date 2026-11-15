"""The thermal analysis engine of dossier (dossier/analysis/thermal.py).

Verification of the solver function against closed forms (meshes made here with gmsh)
    slab with convection on both faces     the linear profile and the heat flow, exact to round-off
    hollow cylinder, convection inside and outside     the logarithmic profile; error 0.07 K of a 203 K
                                           range on 1545 nodes, 0.017 K on 13805 nodes; heat flow 2e-5
    hollow fin with a convective lumped tip     against scipy solve_bvp of the fin equation (which agrees
                                           with the cosh / sinh form to 4e-12). The three-dimensional
                                           result is 0.030 K below the fin equation at the tip on every
                                           mesh (the fin equation ignores the temperature change across the
                                           wall), and the root heat flow is 0.06 % lower.
    heat balance                           the heat flows sum to zero (1e-13 of the heat flow) in each case
Then the engine on a small project (the same fin as a component with a system context): the rules of the
idealization, the record, the convergence rule, the file of each mesh, and the pictures. The last tests run
the example project; they need the geometry and mesh engines.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_bvp

gmsh = pytest.importorskip("gmsh")

from dossier import context as context_mod  # noqa: E402
from dossier.analysis import thermal  # noqa: E402
from dossier.analysis.thermal import ThermalBC, flows, solve_heat, uncertainty  # noqa: E402
from dossier.model import AnalysisRecord, CriterionRef, LoadCase, MeshLevel  # noqa: E402
from dossier.pipeline import Options, StageError  # noqa: E402
from dossier.project import Project  # noqa: E402

K = 12.0                                    # W/(m K), as the example's steel


# --------------------------------------------------------------------------- meshes


def occ_model(build):
    """Run build(occ) in a fresh gmsh model; returns {face tag: (centre, area, type)} (the model stays open)."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("part")
    build(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    return {t: (gmsh.model.occ.getCenterOfMass(2, t), gmsh.model.occ.getMass(2, t), gmsh.model.getType(2, t))
            for _, t in gmsh.model.getEntities(2)}


def write_mesh(path, h, features=None):
    """Mesh the open model with quadratic tetrahedra of size h. The groups are those of the mesh engine: f<tag>
    for every face, the feature ids, and solid (named once for a model; every size has the same)."""
    gmsh.model.mesh.clear()
    if not gmsh.model.getPhysicalGroups():
        for _, t in gmsh.model.getEntities(2):
            gmsh.model.addPhysicalGroup(2, [t], name=f"f{t}")
        for name, tags in (features or {}).items():
            gmsh.model.addPhysicalGroup(2, list(tags), name=name)
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)
    gmsh.write(str(path))
    return len(gmsh.model.mesh.getNodes()[0]), len(gmsh.model.mesh.getElementsByType(11)[0])


def one_mesh(path, build, h):
    try:
        faces = occ_model(build)
        write_mesh(path, h)
        return faces
    finally:
        gmsh.finalize()


def pick(faces, pred):
    return [t for t, (c, area, kind) in faces.items() if pred(c, area, kind)]


# --------------------------------------------------------------------------- the solver function: closed forms


def test_slab_with_convection_on_both_faces(tmp_path):
    """q = (T1 - T2) / (1/h1 + L/k + 1/h2); the wall temperatures are T1 - q/h1 and T2 + q/h2, linear between.
    Quadratic elements hold the linear profile, so the comparison is exact (observed 8e-13 K)."""
    L, W = 0.05, 0.04
    h1, T1, h2, T2 = 400.0, 90.0, 12.0, 293.15
    faces = one_mesh(tmp_path / "slab.msh", lambda occ: occ.addBox(0, 0, 0, L, W, W), 0.012)
    bcs = [ThermalBC("convection", pick(faces, lambda c, a, k: abs(c[0]) < 1e-9), T1, h1),
           ThermalBC("convection", pick(faces, lambda c, a, k: abs(c[0] - L) < 1e-9), T2, h2)]
    x, T, s = solve_heat(tmp_path / "slab.msh", K, bcs)
    q = (T1 - T2) / (1 / h1 + L / K + 1 / h2)                       # W/m^2 in the direction +x (negative: T1 < T2)
    assert np.abs(T - (T1 - q / h1 - q * x[:, 0] / K)).max() < 1e-9
    Q = flows(bcs, s)                                                # heat INTO the part through each condition
    assert Q[0] == pytest.approx(q * W * W, rel=1e-10) and Q[1] == pytest.approx(-q * W * W, rel=1e-10)
    assert Q[0] < 0 < Q[1]                                           # the cold fluid takes heat out, the warm one gives it
    assert abs(sum(Q)) < 1e-9 * abs(Q[0])                            # heat balance
    # the evaluation of the mesh: extremes with their place, the mean over the volume, the faces
    p = s["part"]
    assert p["min"] == pytest.approx(T1 - q / h1, abs=1e-9) and p["xmin"][0] == pytest.approx(0.0, abs=1e-12)
    assert p["max"] == pytest.approx(T2 + q / h2, abs=1e-9) and p["xmax"][0] == pytest.approx(L, abs=1e-12)
    assert p["mean"] == pytest.approx(0.5 * (p["min"] + p["max"]), abs=1e-9) and p["volume"] == pytest.approx(L * W * W)
    assert sorted(s["faces"]) == sorted(str(t) for t in faces)
    assert sum(f["area"] for f in s["faces"].values()) == pytest.approx(2 * W * W + 4 * L * W, rel=1e-12)
    assert s["n_nodes"] == len(T) and s["order"] == 2 and 0.05 < s["min_quality"] <= 1 and s["warnings"] == []


def test_hollow_cylinder_with_convection_inside_and_outside(tmp_path):
    """Per length: q' = 2 pi (Ti - To) / (1/(hi a) + ln(b/a)/k + 1/(ho b)), T(r) = T(a) - q' ln(r/a) / (2 pi k).
    Observed: max |T - exact| 0.072 K at h = 12 mm (1545 nodes) and 0.043 K at 8 mm, heat flow 2e-5 and 1e-5."""
    a, b, H = 0.02, 0.05, 0.03
    hi, Ti, ho, To = 300.0, 90.0, 40.0, 293.15

    def build(occ):
        occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, H, b))], [(3, occ.addCylinder(0, 0, 0, 0, 0, H, a))])

    qp = 2 * math.pi * (Ti - To) / (1 / (hi * a) + math.log(b / a) / K + 1 / (ho * b))     # W/m, outward
    Ta, Tb = Ti - qp / (2 * math.pi * a * hi), To + qp / (2 * math.pi * b * ho)
    assert Ti < Ta < Tb < To
    err = {}
    for h in (0.012, 0.008):
        faces = one_mesh(tmp_path / "cyl.msh", build, h)
        inner = pick(faces, lambda c, area, k: k == "Cylinder" and abs(area - 2 * math.pi * a * H) < 1e-6)
        outer = pick(faces, lambda c, area, k: k == "Cylinder" and abs(area - 2 * math.pi * b * H) < 1e-6)
        bcs = [ThermalBC("convection", inner, Ti, hi), ThermalBC("convection", outer, To, ho)]
        x, T, s = solve_heat(tmp_path / "cyl.msh", K, bcs)
        exact = Ta - qp * np.log(np.hypot(x[:, 0], x[:, 1]) / a) / (2 * math.pi * K)
        err[h] = float(np.abs(T - exact).max())
        Q = flows(bcs, s)
        assert Q[0] == pytest.approx(qp * H, rel=1e-4) and abs(sum(Q)) < 1e-9 * abs(Q[0])
        assert s["faces"][str(inner[0])]["integral"] / s["faces"][str(inner[0])]["area"] == pytest.approx(Ta, abs=0.02)
        assert s["faces"][str(outer[0])]["integral"] / s["faces"][str(outer[0])]["area"] == pytest.approx(Tb, abs=0.02)
    assert err[0.012] < 0.15 and err[0.008] < 0.09 and err[0.008] < err[0.012]


# The fin of the example's hand calculation HC-ACT-TEMP: the neck as a tube between the cold shell and the pad.
R_I, R_O, L_FIN = 0.008, 0.012, 0.078
H_AIR, T_AIR, T_BASE, A_TIP = 12.0, 293.15, 90.0, 4476e-6
A_C, PERIMETER = math.pi * (R_O**2 - R_I**2), 2 * math.pi * R_O
H_TIP = H_AIR * A_TIP / A_C                 # the lumped tip as a coefficient on the end face of the tube


def fin_equation(k=K):
    """k A T'' = h P (T - T_air), T(0) = T_base, -k T'(L) = h_tip (T(L) - T_air), with scipy solve_bvp."""
    m2 = H_AIR * PERIMETER / (k * A_C)
    z = np.linspace(0, L_FIN, 200)
    sol = solve_bvp(lambda z, y: np.vstack([y[1], m2 * (y[0] - T_AIR)]),
                    lambda ya, yb: np.array([ya[0] - T_BASE, k * yb[1] + H_TIP * (yb[0] - T_AIR)]),
                    z, np.vstack([np.full_like(z, T_BASE), np.zeros_like(z)]), tol=1e-10, max_nodes=100000)
    assert sol.success
    return sol.sol


def build_fin(occ):
    occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, L_FIN, R_O))], [(3, occ.addCylinder(0, 0, 0, 0, 0, L_FIN, R_I))])


def fin_faces(faces):
    return {"base": pick(faces, lambda c, a, k: k == "Plane" and abs(c[2]) < 1e-9),
            "tip": pick(faces, lambda c, a, k: k == "Plane" and abs(c[2] - L_FIN) < 1e-9),
            "outer": pick(faces, lambda c, a, k: k == "Cylinder" and abs(a - PERIMETER * L_FIN) < 1e-7),
            "bore": pick(faces, lambda c, a, k: k == "Cylinder" and abs(a - 2 * math.pi * R_I * L_FIN) < 1e-7)}


def test_fin_equation_reference():
    """The numerical solution of the fin equation against its closed form (cosh / sinh)."""
    sol = fin_equation()
    m = math.sqrt(H_AIR * PERIMETER / (K * A_C))
    beta = H_TIP / (m * K)
    z = np.linspace(0, L_FIN, 60)
    closed = T_AIR + (T_BASE - T_AIR) * (np.cosh(m * (L_FIN - z)) + beta * np.sinh(m * (L_FIN - z))) \
        / (math.cosh(m * L_FIN) + beta * math.sinh(m * L_FIN))
    assert np.abs(sol(z)[0] - closed).max() < 1e-9
    assert sol(L_FIN)[0] == pytest.approx(241.2217, abs=1e-3)       # the 1D estimate of the mounting face


def test_hollow_fin_with_convective_lumped_tip(tmp_path):
    """The tube with a prescribed root temperature, convection outside, an adiabatic bore, and the lumped tip
    as convection on the end face. Observed (4658 nodes): tip mean 241.191 K against 241.222 K of the fin
    equation (-0.030 K, the same on finer meshes), largest nodal difference 0.25 K, root heat flow -5.7e-4."""
    faces = one_mesh(tmp_path / "fin.msh", build_fin, 0.004)
    f = fin_faces(faces)
    bcs = [ThermalBC("temperature", f["base"], T_BASE), ThermalBC("convection", f["outer"], T_AIR, H_AIR),
           ThermalBC("convection", f["tip"], T_AIR, H_TIP)]
    x, T, s = solve_heat(tmp_path / "fin.msh", K, bcs)
    sol = fin_equation()
    tip = s["faces"][str(f["tip"][0])]
    assert tip["integral"] / tip["area"] == pytest.approx(sol(L_FIN)[0], abs=0.06)
    assert np.abs(T - sol(x[:, 2])[0]).max() < 0.4                   # the whole profile, of a 203 K range
    Q = flows(bcs, s)
    q_root = -K * A_C * sol(0.0)[1]                                  # heat into the fin at the root (negative)
    assert q_root < 0 and Q[0] == pytest.approx(q_root, rel=2e-3)
    assert Q[1] > 0 and Q[2] > 0                                     # the air gives heat to the cold fin
    assert abs(sum(Q)) < 1e-9 * abs(Q[0])                            # heat balance, with a prescribed temperature
    bore = s["faces"][str(f["bore"][0])]
    assert T_BASE <= bore["min"] < bore["max"] < T_AIR              # the adiabatic bore follows the wall


def test_heat_flux_into_the_part_and_shared_nodes(tmp_path):
    """A heat flux q into the end of a bar with a prescribed temperature at the other end and adiabatic
    sides: T = T0 + q z / k, exact. The reaction at the prescribed end is -q A. Two prescribed-temperature
    conditions that share an edge count each node once."""
    faces = one_mesh(tmp_path / "bar.msh", lambda occ: occ.addBox(0, 0, 0, 0.01, 0.01, 0.05), 0.006)
    z0 = pick(faces, lambda c, a, k: abs(c[2]) < 1e-9)
    z1 = pick(faces, lambda c, a, k: abs(c[2] - 0.05) < 1e-9)
    x0 = pick(faces, lambda c, a, k: abs(c[0]) < 1e-9)
    q, T0 = 2500.0, 100.0
    bcs = [ThermalBC("temperature", z0, T0), ThermalBC("flux", z1, q)]
    x, T, s = solve_heat(tmp_path / "bar.msh", K, bcs)
    assert np.abs(T - (T0 + q * x[:, 2] / K)).max() < 1e-9          # a positive flux warms the part
    Q = flows(bcs, s)
    assert Q[1] == pytest.approx(q * 1e-4, rel=1e-10) and Q[0] == pytest.approx(-q * 1e-4, rel=1e-9)
    # two faces at prescribed temperatures with a common edge, and convection on the far end
    bcs = [ThermalBC("temperature", z0, T0), ThermalBC("temperature", x0, T0), ThermalBC("convection", z1, 300.0, 50.0)]
    _, _, s = solve_heat(tmp_path / "bar.msh", K, bcs)
    Q = flows(bcs, s)
    assert Q[2] > 0 and abs(sum(Q)) < 1e-9 * Q[2]                    # no node of the common edge counts twice


# --------------------------------------------------------------------------- the convergence rule


def test_uncertainty_rule():
    hs = [4.0, 2.0, 1.0]
    # one level: nothing; two levels: 3 |d| max(1, 1 / (r^2 - 1))
    assert uncertainty([1.0], [5.0], 3)["u"] is None
    assert uncertainty([2.0, 1.0], [10.0, 10.4], 3)["u"] == pytest.approx(3 * 0.4)             # r = 2: factor 1
    assert uncertainty([1.4, 1.0], [10.0, 10.4], 3)["u"] == pytest.approx(3 * 0.4 / (1.4**2 - 1))
    assert uncertainty([1.0, 1.0], [10.0, 10.4], 3)["u"] == math.inf
    # three levels, v = 7 + 0.5 h^2 (order 2, below the formal order 3): the extrapolation is exact
    c = uncertainty(hs, [7 + 0.5 * h**2 for h in hs], 3)
    assert c["kind"] == "monotone" and c["order"] == pytest.approx(2.0) and c["extrapolated"] == pytest.approx(7.0)
    d = 0.5 * (4 - 1)                                               # change between the two finest levels
    assert c["u"] == pytest.approx(max(1.25 * d / 3, 3 * d / 3))    # r^p - 1 = 3 for both factors
    assert c["u"] >= abs(7.5 - 7.0)                                 # it covers the true error
    # an observed order above the formal one is not used with the large factor
    c = uncertainty(hs, [7 + 0.01 * h**5 for h in hs], 3)
    assert c["order"] == pytest.approx(5.0) and c["u"] == pytest.approx(3 * 0.31 / (2**3 - 1))
    # values that change direction, values that move apart, values that are equal
    c = uncertainty(hs, [10.0, 10.6, 10.2], 3)
    assert c["kind"] == "oscillatory" and c["u"] >= 0.6
    c = uncertainty(hs, [10.0, 10.1, 10.5], 3)
    assert c["kind"] == "divergent" and c["u"] == pytest.approx(3 * 0.5)
    assert uncertainty(hs, [10.0, 10.0, 10.0], 3)["u"] == 0.0
    # more than three levels: the three finest
    assert uncertainty([8.0] + hs, [99.0] + [7 + 0.5 * h**2 for h in hs], 3)["extrapolated"] == pytest.approx(7.0)


# --------------------------------------------------------------------------- a small project: the fin as a component


SYSTEM = """
id: SYS-NECK
name: A neck between a cold body and a pad
parameters:
  t_base: {{value: {t_base} K, title: Temperature of the cold body}}
  t_air: {{value: {t_air} K, title: Ambient temperature}}
  h_air: {{value: "{h_air} W/(m^2*K)", title: Heat transfer coefficient to the air}}
  h_tip: {{value: "{h_tip} W/(m^2*K)", title: Coefficient of the lumped pad on the end face}}
phases:
  - {{id: PH-COLD, name: Cold operation, duration: 400 s}}
elements:
  - {{id: EL-BODY, name: Cold body}}
  - {{id: EL-NECK, name: Neck, component: CMP-NECK}}
  - {{id: EL-PAD, name: Pad}}
  - {{id: EL-STEM, name: Stem}}
  - {{id: EL-ENV, name: Environment, kind: environment}}
connections:
  - id: IF-BODY-NECK
    name: Root of the neck
    from: EL-BODY
    to: EL-NECK
    kind: thermal
    effects:
      - {{id: EF-BASE-T, phase: PH-COLD, kind: temperature, role: seat, value: {{expr: t_base}}}}
      - {{id: EF-BASE-HOLD, phase: PH-COLD, kind: support}}
  - id: IF-PAD-NECK
    name: Pad on the neck
    from: EL-PAD
    to: EL-NECK
    kind: thermal
    effects:
      - {{id: EF-TIP-T, phase: PH-COLD, kind: fluid_temperature, role: seat, value: {{expr: t_air}}, h: {{expr: h_tip}}}}
  - {{id: IF-STEM-NECK, name: Stem in the bore, from: EL-STEM, to: EL-NECK, kind: mechanical}}
  - id: IF-ENV-NECK
    name: Air around the neck
    from: EL-ENV
    to: EL-NECK
    kind: environment
    effects:
      - {{id: EF-AIR-T, phase: PH-COLD, kind: fluid_temperature, role: external, value: {{expr: t_air}}, h: {{expr: h_air}}}}
"""

COMPONENT = """
id: CMP-NECK
element: EL-NECK
name: Neck
cad: {{file: geometry.step, length_unit: m}}
material: MAT-STEEL
features:
  FT-BASE: {{description: "Root face of the neck.", faces: {base}}}
  FT-TIP: {{description: "End face of the neck.", faces: {tip}}}
  FT-BORE: {{description: "Bore of the neck.", faces: {bore}}}
interfaces:
  - {{id: INT-BODY, name: Root, connection: IF-BODY-NECK, features: {{seat: [FT-BASE]}}}}
  - {{id: INT-PAD, name: End face, connection: IF-PAD-NECK, features: {{seat: [FT-TIP]}}}}
  - {{id: INT-STEM, name: Bore, connection: IF-STEM-NECK, features: {{wetted: [FT-BORE]}}}}
  - {{id: INT-ENV, name: Outer surface, connection: IF-ENV-NECK, features: {{external: ["*"]}}}}
"""

REQUIREMENTS = """
document: NECK-SPEC
requirements:
  - id: REQ-NECK-001
    title: Temperature of the end face
    text: The temperature of the end face shall not be lower than 233 K.
    kind: thermal
    parameters: {t_end_min: {value: 233 K}}
    applies: {phases: [PH-COLD]}
    criteria:
      - {quantity: temperature, measure: min, location: INT-PAD, comparator: ">=", limit: {expr: t_end_min}}
"""

MATERIALS = """
materials:
  - id: MAT-STEEL
    name: Steel
    note: Thermal properties are mean values between 90 K and 293 K.
    E: 193 GPa
    nu: 0.29
    rho: 7990 kg/m^3
    alpha: 13.0e-6 1/K
    k: {k} W/(m*K)
    cp: 400 J/(kg*K)
    yield_strength: 170 MPa
    ultimate_strength: 485 MPa
"""

PROJECT = """
id: NECK-VERIF
name: Neck, thermal verification
components: [components/neck]
subject: CMP-NECK
"""

FIN_SIZES = (0.006, 0.0042, 0.003)          # three levels, each about 1.4 times finer


class Fin:
    """The small project on disk, its geometry data and its mesh levels."""

    def __init__(self, root: Path):
        cdir = root / "components" / "neck"
        cdir.mkdir(parents=True)
        (root / "out" / "meshes").mkdir(parents=True)
        try:
            faces = occ_model(build_fin)
            self.tags = fin_faces(faces)
            gmsh.write(str(cdir / "geometry.step"))
            feats = {"FT-BASE": self.tags["base"], "FT-TIP": self.tags["tip"], "FT-BORE": self.tags["bore"]}
            self.levels = []
            for i, h in enumerate(FIN_SIZES):
                nodes, cells = write_mesh(root / "out" / "meshes" / f"L{i}.msh", h, feats)
                self.levels.append(MeshLevel(level=i, h=h, nodes=nodes, cells=cells, order=2, path=f"meshes/L{i}.msh"))
        finally:
            gmsh.finalize()
        (root / "project.yaml").write_text(PROJECT)
        (root / "system.yaml").write_text(SYSTEM.format(t_base=T_BASE, t_air=T_AIR, h_air=H_AIR, h_tip=H_TIP))
        (root / "materials.yaml").write_text(MATERIALS.format(k=K))
        (root / "policy.yaml").write_text("name: Test policy\n")
        (cdir / "component.yaml").write_text(COMPONENT.format(**{k: v for k, v in self.tags.items()}))
        (cdir / "requirements.yaml").write_text(REQUIREMENTS)
        self.project = Project.load(root)
        self.ctx = context_mod.build(self.project)
        role = {"base": ("FT-BASE", "INT-BODY:seat"), "tip": ("FT-TIP", "INT-PAD:seat"),
                "bore": ("FT-BORE", "INT-STEM:wetted"), "outer": (None, "INT-ENV:external")}
        gfaces = {}
        for name, tags in self.tags.items():
            for t in tags:
                c, area, kind = faces[t]
                gfaces[str(t)] = {"kind": kind.lower(), "area": area, "centroid": list(c),
                                  "features": [role[name][0]] if role[name][0] else [], "roles": [role[name][1]]}
        self.geo = {
            "component": "CMP-NECK",
            "cad": {"file": "geometry.step", "volume": A_C * L_FIN, "bbox": [[-R_O, -R_O, 0.0], [R_O, R_O, L_FIN]],
                    "n_faces": len(faces)},
            "features": {f: {"faces": list(tags), "description": ""} for f, tags in feats.items()},
            "interfaces": {"INT-BODY": {"seat": list(self.tags["base"])}, "INT-PAD": {"seat": list(self.tags["tip"])},
                           "INT-STEM": {"wetted": list(self.tags["bore"])}, "INT-ENV": {"external": list(self.tags["outer"])}},
            "faces": gfaces, "edges": {"reentrant": []}, "scls": {}, "pictures": [], "problems": []}

    def load_case(self, result_keys=("temperature.min@INT-PAD", "temperature.mean@FT-TIP"), **changes) -> LoadCase:
        """The thermal load case of the phase, as the planner makes it: the thermal loads of the context."""
        assert set(changes) <= {"EF-BASE-T", "EF-TIP-T", "EF-AIR-T"}
        loads = []
        for x in self.ctx.loads(self.project, None, "PH-COLD"):
            if x.kind not in context_mod.THERMAL or (x.effect in changes and changes[x.effect] is None):
                continue                    # changes: {effect: fields to replace | None to leave the load out}
            loads.append(x.model_copy(update=changes.get(x.effect, {})))
        return LoadCase(id="LC-TH-COLD", title="Cold operation: steady temperatures", analysis_type="thermal_steady",
                        component="CMP-NECK", phase="PH-COLD", loads=loads, result_keys=list(result_keys),
                        criteria=[CriterionRef(requirement="REQ-NECK-001", criterion=0)])

    def run(self, lc=None, levels=None, **opts):
        lc = lc or self.load_case()
        o = Options(quick=True, log=lambda s: None)
        for k, v in opts.items():
            setattr(o, k, v)
        return thermal.run_levels(self.project, self.ctx, self.geo, lc, levels if levels is not None else self.levels,
                                  self.project.analysis_dir(lc.id), o)


@pytest.fixture(scope="module")
def fin(tmp_path_factory):
    return Fin(tmp_path_factory.mktemp("fin_project"))


@pytest.fixture(scope="module")
def fin_run(fin):
    return fin.run()


def test_engine_results_against_the_fin_equation(fin, fin_run):
    rec = fin_run.record
    assert rec.id == "AN-TH-COLD" and rec.load_case == "LC-TH-COLD" and rec.status == "done" and rec.message == ""
    assert rec.analysis_type == "thermal_steady" and rec.origin.by == "engine:thermal"
    sol = fin_equation()
    r = rec.results
    assert list(r) == ["temperature.min@part", "temperature.max@part", "temperature.min@INT-PAD",
                       "temperature.mean@FT-TIP", "heat_flow.total@INT-BODY", "heat_flow.total@INT-PAD",
                       "heat_flow.total@INT-ENV"]
    # temperatures: the fin equation, to the difference between one and three dimensions
    tip = r["temperature.mean@FT-TIP"]
    assert tip.value == pytest.approx(sol(L_FIN)[0], abs=0.06) and tip.unit == "K"
    assert r["temperature.min@INT-PAD"].value == pytest.approx(sol(L_FIN)[0], abs=0.12)
    assert r["temperature.min@INT-PAD"].value < tip.value < r["temperature.max@part"].value
    assert r["temperature.min@part"].value == pytest.approx(T_BASE, abs=1e-9)
    # where the extremes are, and at which feature
    lo, hi = r["temperature.min@part"], r["temperature.max@part"]
    assert lo.location[2] == pytest.approx(0.0, abs=1e-9) and lo.near == "FT-BASE"
    assert hi.location[2] == pytest.approx(L_FIN, abs=1e-9) and math.hypot(*hi.location[:2]) == pytest.approx(R_O, rel=1e-6)
    assert r["temperature.min@INT-PAD"].near == "FT-TIP" and r["temperature.min@INT-PAD"].location[2] == pytest.approx(L_FIN)
    # heat flows into the part: the root takes the heat out, the air gives it; they balance
    q = {k.split("@")[1]: v.value for k, v in r.items() if k.startswith("heat_flow")}
    assert q["INT-BODY"] == pytest.approx(-K * A_C * sol(0.0)[1], rel=2e-3) and q["INT-BODY"] < 0
    assert q["INT-ENV"] > 0 and q["INT-PAD"] > 0 and abs(sum(q.values())) < 1e-9 * abs(q["INT-BODY"])
    assert all(v.unit == "W" for k, v in r.items() if k.startswith("heat_flow"))


def test_engine_convergence_of_the_results(fin, fin_run):
    rec = fin_run.record
    t_range = T_AIR - T_BASE
    for key, v in rec.results.items():
        assert len(v.per_level) == 3 and v.per_level[-1] == v.value and v.method == "fe", key
        assert v.status == "converged" and v.uncertainty_abs is not None and v.uncertainty is not None, key
        if key.startswith("temperature"):
            assert v.uncertainty == pytest.approx(v.uncertainty_abs / t_range)       # relative to the range of the case
            assert v.uncertainty_abs < 0.05, key                                      # K
            assert "range" not in v.note                                              # the rule is not in the note
        else:
            assert v.uncertainty == pytest.approx(v.uncertainty_abs / abs(v.value)) and v.uncertainty < 0.01
    # the uncertainty covers the change between the two finest meshes
    tip = rec.results["temperature.mean@FT-TIP"]
    assert tip.uncertainty_abs >= abs(tip.per_level[-1] - tip.per_level[-2])
    # the meshes of the record
    assert [m.level for m in rec.mesh] == [0, 1, 2] and [m.path for m in rec.mesh] == [f"meshes/L{i}.msh" for i in range(3)]
    assert rec.mesh[0].nodes < rec.mesh[1].nodes < rec.mesh[2].nodes and all(m.dofs == m.nodes for m in rec.mesh)
    assert rec.mesh[0].h_rep > rec.mesh[1].h_rep > rec.mesh[2].h_rep and all(0.05 < m.min_quality <= 1 for m in rec.mesh)
    # two levels and one level
    two = fin.run(levels=fin.levels[:2]).record
    v2 = two.results["temperature.mean@FT-TIP"]
    d, r = abs(v2.per_level[1] - v2.per_level[0]), two.mesh[0].h_rep / two.mesh[1].h_rep
    assert v2.uncertainty_abs == pytest.approx(3 * d * max(1.0, 1 / (r**2 - 1))) and v2.status == "converged"
    one = fin.run(levels=fin.levels[:1]).record
    v1 = one.results["temperature.mean@FT-TIP"]
    assert v1.status == "not_converged" and v1.uncertainty is None and v1.uncertainty_abs is None
    assert next(c for c in one.checks if c.kind == "error_estimate").passed is None


def test_engine_checks(fin_run):
    checks = {c.kind: c for c in fin_run.record.checks}
    assert set(checks) == {"heat_balance", "mesh_quality", "error_estimate"}
    assert [c.id for c in fin_run.record.checks] == ["CHK-AN-TH-COLD-HEAT-BALANCE", "CHK-AN-TH-COLD-MESH-QUALITY",
                                                     "CHK-AN-TH-COLD-ERROR-ESTIMATE"]
    b = checks["heat_balance"]
    assert b.passed is True and b.value < 1e-9 and b.limit == 1e-6
    q = checks["mesh_quality"]
    assert q.passed is True and q.value > 0.05 and q.limit == 0.01 and "negative Jacobian" in q.statement
    e = checks["error_estimate"]
    assert e.passed is True and e.value < 0.01 and e.limit == 0.05
    worst = max((v for v in fin_run.record.results.values()), key=lambda v: v.uncertainty)
    assert e.value == worst.uncertainty and f"({worst.key})" in e.statement and "The limit of the policy is 5 %." in e.statement


def test_engine_idealization_records(fin, fin_run):
    rec, j = fin_run.record, fin_run.journal
    assert [b.type for b in rec.bcs] == ["temperature", "convection", "convection", "adiabatic"]
    base, tip, air, adiabatic = rec.bcs
    assert base.effect == "EF-BASE-T" and base.features == ["FT-BASE"] and base.value == {"T": T_BASE}
    assert base.faces == [f"f{t}" for t in fin.tags["base"]] and base.interface == "INT-BODY"
    assert tip.effect == "EF-TIP-T" and tip.value == {"h": pytest.approx(H_TIP), "T_fluid": T_AIR}
    assert air.effect == "EF-AIR-T" and air.features == [] and air.faces == [f"f{t}" for t in fin.tags["outer"]]
    assert adiabatic.features == ["FT-BORE"] and adiabatic.faces == [f"f{t}" for t in fin.tags["bore"]]
    # every boundary condition points to its decision; the adiabatic faces to their assumption
    decisions = {d.id: d for d in j.decisions}
    assert [d.id for d in j.decisions] == ["DEC-LC-TH-COLD-BC-EF-BASE-T", "DEC-LC-TH-COLD-BC-EF-TIP-T",
                                           "DEC-LC-TH-COLD-BC-EF-AIR-T", "DEC-LC-TH-COLD-CONVERGENCE"]
    assert [b.decision for b in rec.bcs[:3]] == list(decisions)[:3]
    assert all(d.subject == "LC-TH-COLD" and d.alternatives and d.origin.by == "engine:thermal" for d in j.decisions)
    assert all(d.stage == "idealize" for d in j.decisions[:3]) and j.decisions[3].stage == "solve"
    assert "prescribes a temperature of 90 K on the feature FT-BASE" in decisions[base.decision].choice
    assert "213.7 W/(m²·K)" in decisions[tip.decision].choice and "293.15 K" in decisions[tip.decision].choice
    assert "applies convection to the 1 face of the role external of INT-ENV" in decisions[air.decision].choice
    assert "no other role uses" in decisions[air.decision].rationale
    assert "203.15 K" in decisions["DEC-LC-TH-COLD-CONVERGENCE"].choice and "5 %" in decisions["DEC-LC-TH-COLD-CONVERGENCE"].choice
    # assumptions: the adiabatic bore (which way it errs) and the one conductivity (with its sensitivity)
    asm = {a.id: a for a in j.assumptions}
    assert list(asm) == ["ASM-LC-TH-COLD-ADIABATIC", "ASM-LC-TH-COLD-CONDUCTIVITY"]
    assert adiabatic.decision == "ASM-LC-TH-COLD-ADIABATIC"
    a = asm["ASM-LC-TH-COLD-ADIABATIC"]
    assert a.statement == ("The faces with no thermal load are adiabatic. The interface INT-STEM has the adiabatic "
                           "feature FT-BORE.") and a.conservative is None
    assert "colder than the face makes the part colder" in a.impact and "warmer than the face makes the part warmer" in a.impact
    assert "REQ-NECK-001 gives a lower limit of a temperature, thus a colder mating part is the unsafe case." in a.impact
    assert re.search(r"The analysis gives 90 K to 241\.\d+ K on the faces of INT-STEM\. The mating element is EL-STEM\.", a.impact)
    assert ("about", "INT-STEM") in [(x.rel, x.to) for x in a.links]
    c = asm["ASM-LC-TH-COLD-CONDUCTIVITY"]
    assert "MAT-STEEL is 12 W/(m·K) at every temperature" in c.statement
    assert '"Thermal properties are mean values between 90 K and 293 K."' in c.rationale
    m = re.search(r"20 % lower changes temperature\.min@INT-PAD by ([+-])([\d.]+) K, and one that is 20 % higher "
                  r"changes it by ([+-])([\d.]+) K\.", c.impact)
    assert m, c.impact
    # a lower conductivity lets less cold reach the end face: the fin equation gives the same changes
    low, high = (fin_equation(K * f)(L_FIN)[0] - fin_equation()(L_FIN)[0] for f in (0.8, 1.2))
    assert m.group(1) == "+" and float(m.group(2)) == pytest.approx(low, abs=0.05)
    assert m.group(3) == "-" and float(m.group(4)) == pytest.approx(-high, abs=0.05)
    # the record links to what it rests on
    links = {(x.rel, x.to) for x in rec.links}
    assert {("uses", "MAT-STEEL"), ("uses", "ASM-LC-TH-COLD-ADIABATIC"), ("uses", "ASM-LC-TH-COLD-CONDUCTIVITY"),
            ("uses", "DEC-LC-TH-COLD-CONVERGENCE"), ("uses", base.decision)} <= links
    assert rec.material == {"id": "MAT-STEEL", "k": K, "k_unit": "W/(m*K)"}
    assert "scikit-fem" in rec.solver and "steady heat conduction" in rec.solver


def test_engine_record_text_and_files(fin, fin_run):
    rec = fin_run.record
    assert rec.statement.startswith("The analysis gives the steady temperatures of the component CMP-NECK in the phase PH-COLD.")
    assert "on 3 meshes of quadratic tetrahedra" in rec.statement and "The other face is adiabatic." in rec.statement
    assert "1 prescribed temperature" in rec.statement and "2 convection" in rec.statement
    # the fields of the two finest levels
    fields = [a for a in rec.artifacts if a.kind == "field"]
    assert [a.path for a in fields] == ["analyses/LC-TH-COLD/fields_L1.npz", "analyses/LC-TH-COLD/fields_L2.npz"]
    for a, m in zip(fields, rec.mesh[1:]):
        with np.load(fin.project.out / a.path) as z:
            assert z["x"].shape == (m.nodes, 3) and z["T"].shape == (m.nodes,)
            assert T_BASE - 1e-9 <= z["T"].min() and z["T"].max() <= T_AIR
    assert not [a for a in rec.artifacts if a.kind == "picture"]     # quick: no pictures
    # the record and its journal go through the stage file of the pipeline
    path = fin.project.write_stage("analyze", fin.project.analysis_dir("LC-TH-COLD") / "analysis.json", analyses=[rec],
                                   decisions=fin_run.journal.decisions, assumptions=fin_run.journal.assumptions,
                                   findings=fin_run.journal.findings)
    doc = json.loads(path.read_text())
    again = AnalysisRecord.model_validate(doc["analyses"][0])
    assert again == rec and len(doc["decisions"]) == 4 and len(doc["assumptions"]) == 2 and doc["findings"] == []


STE_BANNED = re.compile(r"\b(should|may|might|could|would|has been|have been|etc|e\.g|i\.e)\b|and/or|n't\b", re.I)


def _sentences(text: str) -> list[str]:
    text = re.sub(r'"[^"]*"', "QUOTE", text)                        # text that an engineer wrote is quoted as it is
    text = re.sub(r"\([^)]*\)", "PAREN", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def test_engine_text_follows_the_writing_rules(fin, fin_run):
    """Every text of the records: sentences of 25 words or fewer, six sentences or fewer in a paragraph,
    none of the words the rules forbid. A number with its unit is one word, as is an identifier."""
    rec, j = fin_run.record, fin_run.journal
    texts = {"statement": rec.statement}
    texts.update({f"{c.id}": c.statement for c in rec.checks})
    texts.update({f"bc {i}": b.description for i, b in enumerate(rec.bcs)})
    texts.update({f"{k} note": v.note for k, v in rec.results.items()})
    for d in j.decisions:
        texts.update({f"{d.id} {f}": getattr(d, f) for f in ("title", "choice", "rationale")})
        texts.update({f"{d.id} alt {i}": f"{a.option} {a.why_not}" for i, a in enumerate(d.alternatives)})
    for a in j.assumptions:
        texts.update({f"{a.id} {f}": getattr(a, f) for f in ("title", "statement", "rationale", "impact", "verify_by")})
    for name, text in texts.items():
        assert text.strip(), name
        assert not STE_BANNED.search(re.sub(r'"[^"]*"', "", text)), (name, text)
        for para in text.split("\n\n"):
            sentences = _sentences(para)
            assert len(sentences) <= 6, (name, para)
            for s in sentences:
                words = re.sub(r"(\d[\d.e+-]*) (K|W|%|mm|m|s|W/\S+)\b", r"\1\2", s).split()
                assert len(words) <= 25, (name, len(words), s)
    ste = pytest.importorskip("dossier.ste")
    if hasattr(ste, "check"):
        problems = {n: [str(i) for i in ste.check(t)] for n, t in texts.items() if "title" not in n}
        assert not {n: p for n, p in problems.items() if p}


def test_not_supported_cases(fin):
    """A load the engine has no model for gives a record with the exact reason and no result."""
    def message(lc):
        out = fin.run(lc)
        rec = out.record
        assert rec.status == "not_supported" and rec.id == "AN-TH-COLD" and not rec.results and not rec.bcs
        assert not out.journal.decisions and not out.journal.assumptions
        return rec.message

    m = message(fin.load_case(**{"EF-TIP-T": {"h_si": None}}))
    assert m == ("The effect EF-TIP-T gives a fluid temperature of 293.15 K on the role seat of the interface INT-PAD "
                 "without a heat transfer coefficient. The analysis does not guess a coefficient. Give the effect a "
                 "heat transfer coefficient (h) in the system context.")
    assert "more than zero" in message(fin.load_case(**{"EF-TIP-T": {"h_si": 0.0}}))
    # heat flux loads only: the steady temperature is not defined
    m = message(fin.load_case(**{"EF-BASE-T": {"kind": "heat_flux", "si": 10.0}, "EF-TIP-T": {"kind": "heat_flux", "si": -10.0},
                                 "EF-AIR-T": None}))
    assert "has heat flux loads only" in m
    # two thermal loads on one face
    m = message(fin.load_case(**{"EF-TIP-T": {"features": ["FT-BASE"]}}))
    assert "EF-BASE-T and EF-TIP-T act on the same face" in m
    # no thermal load, a feature that the geometry does not have, a temperature with a load factor, another type
    lc = fin.load_case()
    assert "has no thermal load" in message(lc.model_copy(update={"loads": []}))
    assert "The geometry stage has no such feature" in message(fin.load_case(**{"EF-TIP-T": {"features": ["FT-NONE"]}}))
    assert "cannot multiply a temperature by a factor" in message(fin.load_case(**{"EF-BASE-T": {"factor": 1.5}}))
    assert "does not analyse the type static_structural" in message(lc.model_copy(update={"analysis_type": "static_structural"}))
    # the same reason reaches the structural engine as a StageError
    with pytest.raises(StageError, match="LC-TH-COLD is not supported: The effect EF-TIP-T gives a fluid temperature"):
        thermal.solve_temperature(fin.project, fin.ctx, fin.geo, fin.load_case(**{"EF-TIP-T": {"h_si": None}}),
                                  fin.levels[0], Options(quick=True))


def test_result_keys_the_engine_cannot_give(fin):
    keys = ["temperature.range@part", "temperature.max@FT-BORE", "temperature.mean@part", "temperature.peak@part",
            "von_mises.peak@part", "temperature.min@SCL-1", "heat_flow.total@INT-STEM", "heat_flow.total@FT-TIP",
            "temperature"]
    rec = fin.run(fin.load_case(result_keys=keys), levels=fin.levels[:2]).record
    r = rec.results
    assert rec.status == "done" and set(keys) <= set(r)
    assert r["temperature.range@part"].value == pytest.approx(r["temperature.max@part"].value - T_BASE)
    assert T_BASE < r["temperature.mean@part"].value < r["temperature.max@FT-BORE"].value < r["temperature.max@part"].value
    for key, why in (("temperature.peak@part", "does not give the measure peak"), ("von_mises.peak@part", "does not give von_mises"),
                     ("temperature.min@SCL-1", "is not the part, a feature or an interface"),
                     ("heat_flow.total@INT-STEM", "has no thermal load in this load case"),
                     ("heat_flow.total@FT-TIP", "is not an interface"), ("temperature", "does not have the form")):
        assert r[key].value is None and r[key].status == "not_run" and why in r[key].note, key
    # a result without a value does not spoil the error estimate
    assert next(c for c in rec.checks if c.kind == "error_estimate").passed is True


def test_prescribed_temperature_and_heat_flux_rules(fin):
    """The other two rules, with an exact solution: a prescribed root temperature, a heat flux into the end
    face, no air: T = T_base + q z / k along the neck. The mesh is exact only for a cylinder of its own cross
    section, so the end face is 3e-4 K above 99.75 K on the second mesh (1e-4 K on the third)."""
    q = 1500.0
    lc = fin.load_case(result_keys=["temperature.mean@FT-TIP", "temperature.max@INT-PAD"],
                       **{"EF-TIP-T": {"kind": "heat_flux", "si": q, "unit": "W/m^2", "h_si": None}, "EF-AIR-T": None})
    out = fin.run(lc, levels=fin.levels[:2])
    rec = out.record
    assert rec.status == "done"
    exact = T_BASE + q * L_FIN / K
    for key in ("temperature.mean@FT-TIP", "temperature.max@INT-PAD"):
        v = rec.results[key]
        assert v.value == pytest.approx(exact, abs=2e-3) and v.status == "converged"
        assert abs(v.value - exact) <= v.uncertainty_abs < 0.01     # the uncertainty covers the true error
    assert rec.results["heat_flow.total@INT-PAD"].value == pytest.approx(q * A_C, rel=2e-4)
    assert rec.results["heat_flow.total@INT-BODY"].value == pytest.approx(-q * A_C, rel=2e-4)
    assert [b.type for b in rec.bcs] == ["temperature", "flux", "adiabatic"] and rec.bcs[1].value == {"q": q}
    d = next(d for d in out.journal.decisions if d.id.endswith("BC-EF-TIP-T"))
    assert "applies a heat flux of 1500 W/m² to the feature FT-TIP. A positive value is a heat flow into the part." in d.choice
    a = next(a for a in out.journal.assumptions if a.id.endswith("ADIABATIC"))
    assert "The interface INT-STEM has the adiabatic feature FT-BORE." in a.statement
    assert f"The interface INT-ENV has the adiabatic face f{fin.tags['outer'][0]}." in a.statement


def test_solve_temperature_keeps_one_field_per_mesh(fin, monkeypatch, tmp_path):
    """The structural engine asks for the temperature of each mesh level: the field of a mesh is solved once,
    and its file is a load of the fixed solver on the same mesh."""
    lc = fin.load_case()
    opts = Options(quick=True, log=lambda s: None)
    calls = []
    real = thermal.solve_heat
    monkeypatch.setattr(thermal, "solve_heat", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    thermal._MEMORY.clear()
    a = thermal.solve_temperature(fin.project, fin.ctx, fin.geo, lc, fin.levels[0], opts)
    n_first = len(calls)                                            # 0 when the run of another test left the file
    b = thermal.solve_temperature(fin.project, fin.ctx, fin.geo, lc, fin.levels[0], opts)
    assert len(calls) == n_first and b["cached"] and b["path"] == a["path"] and np.array_equal(a["T"], b["T"])
    thermal._MEMORY.clear()                                         # a later process: the file is read, nothing is solved
    c = thermal.solve_temperature(fin.project, fin.ctx, fin.geo, lc, fin.levels[0], opts)
    assert len(calls) == n_first and c["cached"] and np.array_equal(a["T"], c["T"]) and np.array_equal(a["x"], c["x"])
    path = Path(a["path"])
    assert path.parent == fin.project.analysis_dir("LC-TH-COLD") and path.name.startswith("T_L0_") and path.suffix == ".npz"
    assert a["x"].shape == (fin.levels[0].nodes, 3) and a["T"].min() == pytest.approx(T_BASE)
    # another mesh, or another load on the same mesh, is another field
    other = thermal.solve_temperature(fin.project, fin.ctx, fin.geo, lc, fin.levels[1], opts)
    assert other["path"] != a["path"] and len(other["T"]) > len(a["T"])
    n = len(calls)
    warm = thermal.solve_temperature(fin.project, fin.ctx, fin.geo, fin.load_case(**{"EF-BASE-T": {"si": 150.0}}),
                                     fin.levels[0], opts)
    assert len(calls) == n + 1 and warm["path"] != a["path"] and warm["T"].min() == pytest.approx(150.0)
    # the file as the thermal load of an elasticity case on the same mesh. The root keeps its plane and is free
    # in it, so the neck contracts freely: its end moves by alpha x the integral of (T - T_ref) along it
    # (observed ratio to the fin equation: 1.0003).
    from feacore.case import BC, Analysis, Case, CaseError, Load, Material
    from feacore.solver import solve
    mesh = fin.project.out / fin.levels[0].path
    alpha = 13e-6

    def elastic(load):
        return solve(Case(mesh=str(mesh), analysis=Analysis(physics="elasticity", formulation="3d", order=2, T_ref=T_AIR),
                          materials={"m": Material(E=193e9, nu=0.29, alpha=alpha)}, loads=[load],
                          bcs=[BC(type="displacement", group="FT-BASE", components=[2], value=0.0),
                               BC(type="displacement", point=[R_O, 0, 0], components=[1], value=0.0),
                               BC(type="displacement", point=[-R_O, 0, 0], components=[1], value=0.0),
                               BC(type="displacement", point=[0, R_O, 0], components=[0], value=0.0)]))

    res = elastic(Load(type="temperature", value={"field": a["path"]}))
    X, uz = res.nodal("uz")
    top = np.abs(X[:, 2] - L_FIN) < 1e-9
    sol = fin_equation()
    z = np.linspace(0, L_FIN, 2001)
    shrink = alpha * float(np.sum((sol(z)[0][1:] + sol(z)[0][:-1]) / 2 - T_AIR) * (z[1] - z[0]))
    assert uz[top].mean() == pytest.approx(shrink, rel=5e-3) and shrink < 0     # the cold neck gets shorter
    assert any(w.startswith("info: loads[0] temperature field") and "largest distance 0.0e+00" in w for w in res.info.warnings)
    with pytest.raises(CaseError, match="does not belong to this mesh"):
        solve(Case(mesh=str(fin.project.out / fin.levels[1].path), analysis=Analysis(physics="elasticity", formulation="3d"),
                   materials={"m": Material(E=193e9, nu=0.29, alpha=alpha)}, bcs=[BC(type="fixed", group="FT-BASE")],
                   loads=[Load(type="temperature", value={"field": a["path"]})]))


def test_pictures(fin):
    """With opts.pictures the engine renders the temperature of the finest mesh in a subprocess."""
    if shutil.which("xvfb-run") is None:
        pytest.skip("no xvfb-run: no display for the renderer")
    rec = fin.run(levels=fin.levels[:2], pictures=True).record
    pics = [a for a in rec.artifacts if a.kind == "picture"]
    assert [a.path for a in pics] == ["analyses/LC-TH-COLD/temperature.png", "analyses/LC-TH-COLD/temperature_section.png"]
    assert not [a for a in rec.artifacts if a.kind == "log"]
    for a in pics:
        f = fin.project.out / a.path
        assert f.stat().st_size > 10_000 and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert re.search(r"The lowest temperature is 90 K and the highest is 241\.\d+ K\.", a.caption)
