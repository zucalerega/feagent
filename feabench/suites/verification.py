"""Verification suite: CAD-based tasks with trustworthy references.

Reference types (strongest first)
  analytical   exact 3D elasticity solution of exactly the stated problem (Lame cylinder and
               sphere, Saint-Venant torsion and flexure stresses, self-weight column)
  published    community benchmark target (NAFEMS LE1, LE10)
  numerical    no closed form: value extrapolated from a mesh-convergence study of the reference
               pipeline (Richardson, p fitted from the three finest meshes); the reported
               uncertainty is the Roache GCI (safety factor 1.25).  Closed-form beam-theory values
               are recorded next to it as a sanity check.

Every case is also *demonstrated*: ``convergence_study`` runs the reference pipeline (gmsh
quadratic tets -> CalculiX) on the task's own STEP file at several mesh sizes and scores each
run with the same evaluator the agents face, so we know each tolerance is attainable and at
what cost.

Units in every case: mm, N, MPa (N/mm^2), tonne, s  (density tonne/mm^3, acceleration mm/s^2).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..paths import VERIFICATION_DIR

UNITS = {"length": "mm", "force": "N", "stress": "MPa", "mass": "tonne", "time": "s",
         "density": "tonne/mm^3", "acceleration": "mm/s^2"}
STEEL = {"name": "steel", "E": 210000.0, "nu": 0.3, "density": 7.85e-9}


@dataclass
class Case:
    id: str
    title: str
    description: str
    geometry: Callable  # geometry(occ) builds exactly one solid in the current gmsh model
    analysis: dict
    materials: list
    boundary_conditions: list
    loads: list
    qois: list
    levels: list  # global mesh size per refinement level
    fields: Callable | None = None  # fields(h) -> list of local refinement fields
    curvature: int = 0
    notes: list = field(default_factory=list)
    volume_tolerance: float = 0.02
    tags: list = field(default_factory=list)


CASES: dict[str, Case] = {}


def _register(c: Case) -> Case:
    CASES[c.id] = c
    return c


def _q(**kw) -> dict:
    kw.setdefault("weight", 1.0)
    kw.setdefault("gate", False)
    return kw


def _cyl_frame():
    return {"type": "cylindrical", "point": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}


def _plane(point, normal, label=None):
    r = {"type": "plane", "point": list(map(float, point)), "normal": list(map(float, normal))}
    if label:
        r["label"] = label
    return r


# ============================================================================ 1. Lame cylinder
def _lame_cylinder():
    a, b, L, p = 50.0, 100.0, 20.0, 100.0
    E, nu = STEEL["E"], STEEL["nu"]
    A = p * a**2 / (b**2 - a**2)
    ur = lambda r: (1 + nu) / E * ((1 - 2 * nu) * A * r + A * b**2 / r)
    st = lambda r: A * (1 + b**2 / r**2)
    c45 = math.cos(math.pi / 4)
    pt = lambda r: [r * c45, r * c45, L / 2]

    def geom(occ):
        o = occ.addCylinder(0, 0, 0, 0, 0, L, b, angle=math.pi / 2)
        i = occ.addCylinder(0, 0, 0, 0, 0, L, a, angle=math.pi / 2)
        occ.cut([(3, o)], [(3, i)])

    src = "Lame thick-walled cylinder, plane strain; e.g. Timoshenko & Goodier, Theory of Elasticity, 3rd ed., sec. 33"
    return _register(Case(
        id="lame_cylinder",
        title="Thick-walled cylinder under internal pressure (plane strain)",
        description=(
            f"A quarter (90 degree) segment of a thick-walled steel tube, inner radius {a:g} mm, outer radius "
            f"{b:g} mm, axial length {L:g} mm (axis = z). The planes x=0 and y=0 are planes of symmetry of the full "
            f"tube. Both end faces (z=0 and z={L:g}) are held between frictionless rigid platens: no axial "
            f"displacement, free in-plane motion (plane strain). An internal pressure of {p:g} MPa acts on the "
            f"bore."),
        geometry=geom,
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[
            {"id": "sym_x", "type": "symmetry", "region": _plane([0, 0, 0], [1, 0, 0], "face x=0"),
             "description": "symmetry plane x=0: ux = 0"},
            {"id": "sym_y", "type": "symmetry", "region": _plane([0, 0, 0], [0, 1, 0], "face y=0"),
             "description": "symmetry plane y=0: uy = 0"},
            {"id": "end_z0", "type": "symmetry", "region": _plane([0, 0, 0], [0, 0, 1], "end face z=0"),
             "description": "frictionless platen: uz = 0"},
            {"id": "end_zL", "type": "symmetry", "region": _plane([0, 0, L], [0, 0, 1], f"end face z={L:g}"),
             "description": "frictionless platen: uz = 0"},
        ],
        loads=[{"id": "p_inner", "type": "pressure", "value": p,
                "region": {"type": "cylinder", "point": [0, 0, 0], "axis": [0, 0, 1], "radius": a,
                           "label": f"bore (inner cylindrical face r={a:g})"},
                "description": f"internal pressure {p:g} MPa"}],
        qois=[
            _q(id="ur_inner", kind="point", field="U", component="r", frame=_cyl_frame(), point=pt(a),
               reference=ur(a), tolerance={"rel": 0.005}, reference_type="analytical", provenance=src,
               description=f"radial displacement of the bore at 45 deg, mid-length {pt(a)}"),
            _q(id="ur_outer", kind="point", field="U", component="r", frame=_cyl_frame(), point=pt(b),
               reference=ur(b), tolerance={"rel": 0.005}, reference_type="analytical", provenance=src,
               description=f"radial displacement of the outer surface at {pt(b)}"),
            _q(id="hoop_inner", kind="point", field="S", component="tt", frame=_cyl_frame(), point=pt(a),
               reference=st(a), tolerance={"rel": 0.015}, reference_type="analytical", provenance=src,
               description=f"hoop stress at the bore {pt(a)}"),
            _q(id="hoop_outer", kind="point", field="S", component="tt", frame=_cyl_frame(), point=pt(b),
               reference=st(b), tolerance={"rel": 0.015}, reference_type="analytical", provenance=src,
               description=f"hoop stress at the outer surface {pt(b)}"),
            _q(id="axial_mid", kind="point", field="S", component="zz", point=pt((a + b) / 2),
               reference=2 * nu * A, tolerance={"rel": 0.015}, reference_type="analytical", provenance=src,
               description=f"axial (plane-strain) stress mid-wall {pt((a + b) / 2)}"),
            _q(id="reaction_x", kind="reaction", region=_plane([0, 0, 0], [1, 0, 0]), component="x",
               reference=-p * a * L, tolerance={"rel": 0.005}, gate=True, reference_type="analytical",
               provenance="statics: pressure resultant on the quarter bore", description="sum of x-reactions on x=0"),
        ],
        levels=[25.0, 12.0, 7.0, 4.0, 2.5],
        tags=["exact", "pressure", "symmetry", "plane_strain"],
    ))


# ============================================================================ 2. Lame sphere
def _lame_sphere():
    a, b, p = 50.0, 100.0, 100.0
    E, nu = STEEL["E"], STEEL["nu"]
    k = (b / a) ** 3
    ur = lambda r: p * a**3 * r / (E * (b**3 - a**3)) * ((1 - 2 * nu) + (1 + nu) * b**3 / (2 * r**3))
    st = lambda r: p * (b**3 / (2 * r**3) + 1) / (k - 1)
    d = 1 / math.sqrt(3)
    pt = lambda r: [r * d, r * d, r * d]
    sph = {"type": "spherical", "center": [0.0, 0.0, 0.0]}

    def geom(occ):
        o = occ.addSphere(0, 0, 0, b)
        i = occ.addSphere(0, 0, 0, a)
        shell = occ.cut([(3, o)], [(3, i)])[0]
        box = occ.addBox(0, 0, 0, 2 * b, 2 * b, 2 * b)
        occ.intersect(shell, [(3, box)])

    src = "Lame thick-walled sphere; e.g. Timoshenko & Goodier, Theory of Elasticity, 3rd ed., sec. 141"
    return _register(Case(
        id="lame_sphere",
        title="Thick-walled sphere under internal pressure",
        description=(
            f"One octant (x, y, z >= 0) of a thick-walled steel sphere, inner radius {a:g} mm, outer radius "
            f"{b:g} mm, centred at the origin. The three coordinate planes are planes of symmetry of the full "
            f"sphere. An internal pressure of {p:g} MPa acts on the inner spherical surface."),
        geometry=geom,
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[
            {"id": f"sym_{ax}", "type": "symmetry", "region": _plane([0, 0, 0], n, f"face {ax}=0"),
             "description": f"symmetry plane {ax}=0: u{ax} = 0"}
            for ax, n in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]))
        ],
        loads=[{"id": "p_inner", "type": "pressure", "value": p,
                "region": {"type": "sphere", "center": [0, 0, 0], "radius": a, "label": f"inner sphere r={a:g}"},
                "description": f"internal pressure {p:g} MPa"}],
        qois=[
            _q(id="ur_inner", kind="point", field="U", component="r", frame=sph, point=pt(a),
               reference=ur(a), tolerance={"rel": 0.005}, reference_type="analytical", provenance=src,
               description=f"radial displacement of the inner surface at {[round(x, 3) for x in pt(a)]}"),
            _q(id="ur_outer", kind="point", field="U", component="r", frame=sph, point=pt(b),
               reference=ur(b), tolerance={"rel": 0.005}, reference_type="analytical", provenance=src,
               description=f"radial displacement of the outer surface at {[round(x, 3) for x in pt(b)]}"),
            _q(id="hoop_inner", kind="point", field="S", component="tt", frame=sph, point=pt(a),
               reference=st(a), tolerance={"rel": 0.025}, reference_type="analytical", provenance=src,
               note="2.5 %: the hoop stress falls off as 1/r^3 at the bore (~3.4 MPa/mm), so nodal-averaged "
                    "surface stress on reasonable meshes scatters by ~1-2 % (convergence study: 0.5 %, 1.7 %, 0.2 %"
                    " on 13k/43k/112k DOF)",
               description="tangential (hoop) stress at the inner surface"),
            _q(id="hoop_outer", kind="point", field="S", component="tt", frame=sph, point=pt(b),
               reference=st(b), tolerance={"rel": 0.015}, reference_type="analytical", provenance=src,
               description="tangential (hoop) stress at the outer surface"),
            _q(id="reaction_x", kind="reaction", region=_plane([0, 0, 0], [1, 0, 0]), component="x",
               reference=-p * math.pi * a**2 / 4, tolerance={"rel": 0.005}, gate=True,
               reference_type="analytical", provenance="statics", description="sum of x-reactions on x=0"),
        ],
        levels=[30.0, 16.0, 10.0, 6.5, 4.5],
        tags=["exact", "pressure", "symmetry"],
    ))


# ============================================================================ 3. torsion
def _torsion_shaft():
    R, L, T = 10.0, 100.0, 1.0e5
    E, nu = STEEL["E"], STEEL["nu"]
    G = E / (2 * (1 + nu))
    J = math.pi * R**4 / 2
    twist = T / (G * J)
    src = "Saint-Venant torsion of a circular shaft (exact; fixed end consistent with the exact field)"
    return _register(Case(
        id="torsion_shaft",
        title="Solid circular shaft in torsion",
        description=(
            f"A solid steel shaft of radius {R:g} mm and length {L:g} mm along z. The face z=0 is fully fixed. "
            f"A pure torque of {T:g} N*mm about the shaft axis (+z, right-hand rule) is applied to the free end "
            f"face z={L:g}, distributed as the Saint-Venant shear traction (proportional to the distance from "
            f"the axis); the face is free to deform otherwise."),
        geometry=lambda occ: occ.addCylinder(0, 0, 0, 0, 0, L, R),
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[{"id": "root", "type": "fixed", "region": _plane([0, 0, 0], [0, 0, 1], "face z=0"),
                              "description": "fully fixed (all displacement components zero)"}],
        loads=[{"id": "torque", "type": "surface_torque", "torque": T, "point": [0, 0, 0], "axis": [0, 0, 1],
                "region": _plane([0, 0, L], [0, 0, 1], f"end face z={L:g}"),
                "description": f"torque {T:g} N*mm about +z (Saint-Venant shear traction)"}],
        qois=[
            _q(id="ut_mid", kind="point", field="U", component="t", frame=_cyl_frame(), point=[R, 0, L / 2],
               reference=twist * (L / 2) * R, tolerance={"rel": 0.01}, reference_type="analytical", provenance=src,
               description=f"tangential displacement of the surface point {[R, 0, L / 2]}"),
            _q(id="ut_3q", kind="point", field="U", component="t", frame=_cyl_frame(), point=[0, R, 0.75 * L],
               reference=twist * 0.75 * L * R, tolerance={"rel": 0.01}, reference_type="analytical",
               provenance=src, description=f"tangential displacement of the surface point {[0, R, 0.75 * L]}"),
            _q(id="mises_mid", kind="point", field="S", component="mises", point=[R, 0, L / 2],
               reference=math.sqrt(3) * T * R / J, tolerance={"rel": 0.015}, reference_type="analytical",
               provenance=src + "; pure shear tau = T R / J, von Mises = sqrt(3) tau",
               description=f"von Mises stress at the surface point {[R, 0, L / 2]}"),
            _q(id="reaction_torque", kind="reaction", region=_plane([0, 0, 0], [0, 0, 1]),
               moment={"point": [0, 0, 0], "axis": [0, 0, 1]}, reference=-T, tolerance={"rel": 0.005},
               gate=True, reference_type="analytical", provenance="statics",
               description="reaction moment about z at the fixed face"),
        ],
        levels=[6.0, 4.0, 2.8, 2.0, 1.5],
        tags=["exact", "torque", "fixed"],
    ))


# ============================================================================ 4. self-weight column
def _gravity_column():
    w, H = 50.0, 500.0
    g = 1.0e6  # mm/s^2 (1000 m/s^2, ~102 g)
    E, rho = STEEL["E"], STEEL["density"]
    src = "uniaxial self-weight solution sigma_zz = -rho g (H - z) (exact with a frictionless base)"
    corner = [-w / 2, -w / 2, 0.0]
    corner2 = [w / 2, -w / 2, 0.0]
    return _register(Case(
        id="gravity_column",
        title="Prismatic column under body acceleration (self-weight)",
        description=(
            f"A steel column with a {w:g} x {w:g} mm square cross-section, height {H:g} mm along z, base at z=0 "
            f"(centred on the z axis). It stands on a frictionless rigid floor (no vertical displacement of the "
            f"base, free horizontal motion) and is loaded only by a uniform body acceleration of {g:g} mm/s^2 "
            f"in -z (an inertial load of ~102 g). Rigid-body motion in the horizontal plane must be suppressed "
            f"without restraining the free lateral (Poisson) deformation, e.g. ux=uy=0 at base corner "
            f"{corner} and uy=0 at base corner {corner2}."),
        geometry=lambda occ: occ.addBox(-w / 2, -w / 2, 0, w, w, H),
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[
            {"id": "floor", "type": "symmetry", "region": _plane([0, 0, 0], [0, 0, 1], "base z=0"),
             "description": "frictionless floor: uz = 0"},
            {"id": "pin_a", "type": "displacement", "region": {"type": "point", "xyz": corner},
             "values": {"x": 0.0, "y": 0.0}, "description": "ux = uy = 0"},
            {"id": "pin_b", "type": "displacement", "region": {"type": "point", "xyz": corner2},
             "values": {"y": 0.0}, "description": "uy = 0"},
        ],
        loads=[{"id": "accel", "type": "gravity", "acceleration": [0.0, 0.0, -g],
                "description": f"uniform body acceleration {g:g} mm/s^2 in -z"}],
        qois=[
            _q(id="uz_top", kind="point", field="U", component="z", point=[0, 0, H],
               reference=-rho * g * H**2 / (2 * E), tolerance={"rel": 0.005}, reference_type="analytical",
               provenance=src,
               note="exact uniaxial solution lets the base warp by nu rho g r^2 / 2E; the uz=0 floor suppresses "
                    "that, shifting the top displacement by ~0.05 % (seen identically at every mesh size), well "
                    "inside the 0.5 % tolerance",
               description=f"vertical displacement of the top centre {[0, 0, H]}"),
            _q(id="szz_mid", kind="point", field="S", component="zz", point=[w / 2, 0, H / 2],
               reference=-rho * g * (H - H / 2), tolerance={"rel": 0.01}, reference_type="analytical",
               provenance=src, description=f"axial stress at mid-height on a side face {[w / 2, 0, H / 2]}"),
            _q(id="reaction_z", kind="reaction", region=_plane([0, 0, 0], [0, 0, 1]), component="z",
               reference=rho * w * w * H * g, tolerance={"rel": 0.02}, gate=True, reference_type="analytical",
               provenance="weight = rho V g; ccx RF at base nodes also contains their O(h) share of the body "
                          "force, hence the 2 % tolerance", description="total vertical reaction at the base"),
        ],
        levels=[50.0, 25.0, 15.0, 10.0, 7.0],
        tags=["exact", "body_force", "units", "rigid_body_modes"],
    ))


# ============================================================================ 5. cantilever
def _cantilever_static():
    L, b, h, P = 200.0, 10.0, 20.0, 1000.0
    E, nu = STEEL["E"], STEEL["nu"]
    G = E / (2 * (1 + nu))
    I = b * h**3 / 12
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    eb = P * L**3 / (3 * E * I)
    timo = eb + P * L / (kappa * G * b * h)
    return _register(Case(
        id="cantilever_tip_load",
        title="Cantilever with an end shear load",
        description=(
            f"A steel bar {L:g} mm long (x), cross-section {b:g} mm (y) x {h:g} mm (z), centred on the x axis. "
            f"The face x=0 is fully fixed. A total force of {P:g} N in -z is applied to the free end face "
            f"x={L:g} as a uniformly distributed traction."),
        geometry=lambda occ: occ.addBox(0, -b / 2, -h / 2, L, b, h),
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[{"id": "root", "type": "fixed", "region": _plane([0, 0, 0], [1, 0, 0], "face x=0"),
                              "description": "fully fixed"}],
        loads=[{"id": "tip", "type": "surface_force", "force": [0.0, 0.0, -P],
                "region": _plane([L, 0, 0], [1, 0, 0], f"end face x={L:g}"),
                "description": f"{P:g} N in -z as a uniform traction"}],
        qois=[
            _q(id="tip_deflection", kind="region", field="U", component="z", stat="mean",
               region=_plane([L, 0, 0], [1, 0, 0]), reference=None, tolerance={"rel": 0.01},
               reference_type="numerical", provenance="Richardson extrapolation of the reference pipeline",
               closed_form={"euler_bernoulli": -eb, "timoshenko_cowper": -timo},
               description="mean vertical displacement of the end face"),
            _q(id="bending_stress_mid", kind="point", field="S", component="xx", point=[L / 2, 0, h / 2],
               reference=P * (L / 2) * (h / 2) / I, tolerance={"rel": 0.01}, reference_type="analytical",
               provenance="Saint-Venant flexure: sigma_xx = P (L-x) z / I (exact away from the ends)",
               description=f"bending stress at the top surface, mid-span {[L / 2, 0, h / 2]}"),
            _q(id="reaction_z", kind="reaction", region=_plane([0, 0, 0], [1, 0, 0]), component="z",
               reference=P, tolerance={"rel": 0.005}, gate=True, reference_type="analytical", provenance="statics",
               description="vertical reaction at the root"),
        ],
        levels=[10.0, 6.0, 4.0, 3.0, 2.2],
        tags=["numerical", "beam", "traction"],
    ))


# ============================================================================ 6/7. NAFEMS LE10 / LE1
def _quarter_elliptic_annulus(occ, z0, t):
    outer = occ.addDisk(0, 0, 0, 3250.0, 2750.0)
    inner = occ.addDisk(0, 0, 0, 2000.0, 1000.0)
    ring = occ.cut([(2, outer)], [(2, inner)])[0]
    quad = occ.addRectangle(0, 0, 0, 4000.0, 4000.0)
    face = occ.intersect(ring, [(2, quad)])[0]
    ext = occ.extrude(face, 0, 0, t)
    vol = [e for e in ext if e[0] == 3]
    occ.translate(vol, 0, 0, z0)
    return vol


def _nafems_le10():
    p = 1.0

    def geom(occ):
        vol = _quarter_elliptic_annulus(occ, -300.0, 600.0)
        arc = occ.addEllipse(0, 0, 0, 3250.0, 2750.0, angle1=0.0, angle2=math.pi / 2)
        occ.fragment(vol, [(1, arc)])  # imprint line EE' (outer face at mid-plane) for the uz constraint

    outer = {"type": "elliptic_cylinder", "center": [0, 0], "semi_axes": [3250.0, 2750.0],
             "label": "outer elliptic face BCB'C'"}
    area = math.pi / 4 * (3250.0 * 2750.0 - 2000.0 * 1000.0)
    return _register(Case(
        id="nafems_le10",
        title="NAFEMS LE10: thick elliptic plate under pressure",
        description=(
            "NAFEMS standard benchmark LE10 (in mm). Quarter of an elliptic annular plate, 600 mm thick "
            "(z from -300 to +300). Inner boundary: ellipse (x/2000)^2 + (y/1000)^2 = 1; outer boundary: ellipse "
            "(x/3250)^2 + (y/2750)^2 = 1. Uniform normal pressure of 1 MPa on the upper face z=+300. "
            "Boundary conditions: uy=0 on the face y=0 (DCD'C'); ux=0 on the face x=0 (ABA'B'); ux=uy=0 on the "
            "outer elliptic face (BCB'C'); uz=0 on the line EE' where the outer face meets the mid-plane z=0 "
            "(the CAD face is split along that line)."),
        geometry=geom,
        analysis={"type": "static"},
        materials=[{"name": "steel", "E": 210000.0, "nu": 0.3, "applies_to": "all"}],
        boundary_conditions=[
            {"id": "sym_y", "type": "symmetry", "region": _plane([0, 0, 0], [0, 1, 0], "face y=0"),
             "description": "uy = 0"},
            {"id": "sym_x", "type": "symmetry", "region": _plane([0, 0, 0], [1, 0, 0], "face x=0"),
             "description": "ux = 0"},
            {"id": "outer", "type": "displacement", "region": outer, "values": {"x": 0.0, "y": 0.0},
             "description": "ux = uy = 0"},
            {"id": "line_EE", "type": "displacement",
             "region": {"type": "and", "regions": [outer, _plane([0, 0, 0], [0, 0, 1])],
                        "label": "line EE' (outer face at z=0)"},
             "values": {"z": 0.0}, "description": "uz = 0"},
        ],
        loads=[{"id": "pressure", "type": "pressure", "value": p,
                "region": _plane([0, 0, 300.0], [0, 0, 1], "upper face z=+300"),
                "description": "1 MPa normal pressure (pushing down)"}],
        qois=[
            _q(id="syy_D", kind="point", field="S", component="yy", point=[2000.0, 0.0, 300.0],
               reference=-5.38, tolerance={"rel": 0.02}, reference_type="published",
               provenance="NAFEMS LE10 target sigma_yy(D) = -5.38 MPa (The Standard NAFEMS Benchmarks, TNSB Rev. 3, 1990)",
               description="sigma_yy at point D = (2000, 0, 300)"),
            _q(id="reaction_z", kind="reaction",
               region={"type": "and", "regions": [outer, _plane([0, 0, 0], [0, 0, 1])]}, component="z",
               reference=p * area, tolerance={"rel": 0.005}, gate=True, reference_type="analytical",
               provenance="statics: total pressure load carried by the uz constraint on line EE'",
               description="sum of z-reactions on line EE'"),
        ],
        levels=[600.0, 350.0, 220.0, 150.0, 110.0],
        fields=lambda h: [{"type": "ball", "center": [2000.0, 0.0, 300.0], "radius": 400.0,
                           "size": h / 4, "thickness": 600.0}],
        tags=["published", "nafems", "pressure", "mixed_bcs"],
    ))


def _nafems_le1():
    p = 10.0
    outer = {"type": "elliptic_cylinder", "center": [0, 0], "semi_axes": [3250.0, 2750.0],
             "label": "outer elliptic face"}
    return _register(Case(
        id="nafems_le1",
        title="NAFEMS LE1: elliptic membrane (3D thin plate)",
        description=(
            "NAFEMS benchmark LE1 (plane stress) modelled as a 3D plate, in mm. Quarter of an elliptic annular "
            "plate of total thickness 100 mm; only the half-thickness z in [0, 50] is modelled and z=0 is the "
            "plate's mid-plane of symmetry. Inner boundary (x/2000)^2 + (y/1000)^2 = 1, outer boundary "
            "(x/3250)^2 + (y/2750)^2 = 1. A uniform outward normal traction (tension) of 10 MPa acts on the outer "
            "elliptic face. Symmetry: ux=0 on x=0, uy=0 on y=0, uz=0 on z=0."),
        geometry=lambda occ: _quarter_elliptic_annulus(occ, 0.0, 50.0),
        analysis={"type": "static"},
        materials=[{"name": "steel", "E": 210000.0, "nu": 0.3, "applies_to": "all"}],
        boundary_conditions=[
            {"id": "sym_x", "type": "symmetry", "region": _plane([0, 0, 0], [1, 0, 0], "face x=0"),
             "description": "ux = 0"},
            {"id": "sym_y", "type": "symmetry", "region": _plane([0, 0, 0], [0, 1, 0], "face y=0"),
             "description": "uy = 0"},
            {"id": "mid", "type": "symmetry", "region": _plane([0, 0, 0], [0, 0, 1], "mid-plane z=0"),
             "description": "uz = 0"},
        ],
        loads=[{"id": "tension", "type": "pressure", "value": -p, "region": outer,
                "description": "10 MPa outward normal traction (i.e. a pressure of -10 MPa)"}],
        qois=[
            _q(id="syy_D", kind="line", field="S", component="yy", start=[2000.0, 0.0, 0.0],
               end=[2000.0, 0.0, 50.0], n=21, reference=92.7, tolerance={"rel": 0.02}, reference_type="published",
               provenance="NAFEMS LE1 target sigma_yy(D) = 92.7 MPa (plane stress; TNSB Rev. 3, 1990)",
               note="plane stress = thickness average, so the 3D QoI averages sigma_yy along the hole edge "
                    "through D. Sensitive to element shape at the notch: the level-3 pipeline mesh has "
                    "sliver tets there (radius ratio 0.001) and is 14 % low, the others are within 0.6 %",
               description="sigma_yy averaged through the thickness along the hole edge at D (x=2000, y=0, "
                           "0<=z<=50)"),
            _q(id="reaction_x", kind="reaction", region=_plane([0, 0, 0], [1, 0, 0]), component="x",
               reference=-p * 2750.0 * 50.0, tolerance={"rel": 0.005}, gate=True, reference_type="analytical",
               provenance="statics: x-resultant of the outward traction", description="sum of x-reactions on x=0"),
        ],
        levels=[500.0, 300.0, 180.0, 120.0, 80.0],
        fields=lambda h: [{"type": "ball", "center": [2000.0, 0.0, 0.0], "radius": 500.0,
                           "size": max(h / 4, 10.0), "thickness": 800.0}],
        tags=["published", "nafems", "membrane", "thin"],
    ))


# ============================================================================ 8. modal
def _cantilever_modal():
    L, b, h = 200.0, 10.0, 20.0
    E, rho = STEEL["E"], STEEL["density"]
    A = b * h
    beta1 = 1.875104068711961
    f_eb = lambda I: beta1**2 / (2 * math.pi) * math.sqrt(E * I / (rho * A * L**4))
    return _register(Case(
        id="cantilever_modal",
        title="Natural frequencies of a cantilever bar",
        description=(
            f"The steel bar of the cantilever case ({L:g} x {b:g} x {h:g} mm, x along the length, {b:g} mm in "
            f"y, {h:g} mm in z), fully fixed at x=0. Compute the lowest natural frequencies (Hz)."),
        geometry=lambda occ: occ.addBox(0, -b / 2, -h / 2, L, b, h),
        analysis={"type": "frequency", "n_modes": 6},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[{"id": "root", "type": "fixed", "region": _plane([0, 0, 0], [1, 0, 0], "face x=0"),
                              "description": "fully fixed"}],
        loads=[],
        qois=[
            _q(id="f1", kind="frequency", mode=1, reference=None, tolerance={"rel": 0.01},
               reference_type="numerical", provenance="Richardson extrapolation of the reference pipeline",
               closed_form={"euler_bernoulli": f_eb(h * b**3 / 12)},
               description="first natural frequency [Hz] (bending in y)"),
            _q(id="f2", kind="frequency", mode=2, reference=None, tolerance={"rel": 0.01},
               reference_type="numerical", provenance="Richardson extrapolation of the reference pipeline",
               closed_form={"euler_bernoulli": f_eb(b * h**3 / 12)},
               description="second natural frequency [Hz] (bending in z)"),
        ],
        levels=[10.0, 6.0, 4.0, 3.0, 2.2],
        tags=["numerical", "modal", "density_units"],
    ))


# ============================================================================ 9. buckling
def _euler_buckling():
    a, L, P = 10.0, 400.0, 1000.0
    E = STEEL["E"]
    I = a**4 / 12
    pcr = math.pi**2 * E * I / (4 * L**2)
    return _register(Case(
        id="euler_buckling",
        title="Linear buckling of a slender column",
        description=(
            f"A steel column with a {a:g} x {a:g} mm square cross-section and length {L:g} mm along z, base "
            f"z=0 fully fixed, top free. A reference load of {P:g} N in -z (compression) is applied to the top "
            f"face as a uniform traction. Compute the linear (eigenvalue) buckling load factor of the lowest mode."),
        geometry=lambda occ: occ.addBox(-a / 2, -a / 2, 0, a, a, L),
        analysis={"type": "buckle", "n_modes": 2},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[{"id": "base", "type": "fixed", "region": _plane([0, 0, 0], [0, 0, 1], "face z=0"),
                              "description": "fully fixed"}],
        loads=[{"id": "top", "type": "surface_force", "force": [0.0, 0.0, -P],
                "region": _plane([0, 0, L], [0, 0, 1], f"top face z={L:g}"),
                "description": f"reference load {P:g} N compression"}],
        qois=[
            _q(id="lambda1", kind="buckling", mode=1, reference=None, tolerance={"rel": 0.01},
               reference_type="numerical", provenance="Richardson extrapolation of the reference pipeline",
               closed_form={"euler": pcr / P}, description="lowest buckling load factor"),
        ],
        levels=[10.0, 6.0, 4.0, 3.0, 2.2],
        tags=["numerical", "buckling"],
    ))


# ============================================================================ 10. bracket
def _bracket():
    F = 3000.0

    def geom(occ):
        import gmsh

        base = occ.addBox(0, 0, 0, 100, 80, 10)
        wall = occ.addBox(0, 0, 0, 100, 10, 90)
        body = occ.fuse([(3, base)], [(3, wall)])[0]
        occ.synchronize()
        edges = gmsh.model.getEntitiesInBoundingBox(-1, 9.9, 9.9, 101, 10.1, 10.1, dim=1)
        body = occ.fillet([body[0][1]], [e[1] for e in edges], [8.0])
        holes = [occ.addCylinder(x, 55, -1, 0, 0, 12, 5.5) for x in (25.0, 75.0)]
        pin = occ.addCylinder(50, -1, 65, 0, 12, 0, 8.0)
        occ.cut(body, [(3, t) for t in holes + [pin]])

    bolt = lambda x: {"type": "cylinder", "point": [x, 55.0, 0.0], "axis": [0, 0, 1], "radius": 5.5,
                      "range": [0.0, 10.0]}
    pin = {"type": "cylinder", "point": [50.0, 0.0, 65.0], "axis": [0, 1, 0], "radius": 8.0, "range": [0.0, 10.0],
           "label": "pin hole (diameter 16 mm, axis along y through (50, *, 65))"}
    return _register(Case(
        id="angle_bracket",
        title="Filleted angle bracket loaded through a pin hole",
        description=(
            "A steel angle bracket: base plate 100 x 80 x 10 mm (x, y, z) and an upright wall 100 x 10 x 90 mm "
            "along the edge y in [0, 10], joined by an 8 mm fillet on the inside corner. Two 11 mm bolt holes "
            "through the base at (25, 55) and (75, 55); a 16 mm pin hole through the wall at x=50, z=65 (axis "
            "along y). The bolt-hole surfaces are fully fixed (idealised rigid bolts). A force of 3000 N in -y "
            "(pulling the wall away from the base, opening the angle) is applied through the pin hole as a "
            "uniform traction on its cylindrical surface."),
        geometry=geom,
        analysis={"type": "static"},
        materials=[STEEL | {"applies_to": "all"}],
        boundary_conditions=[{"id": "bolts", "type": "fixed",
                              "region": {"type": "or", "regions": [bolt(25.0), bolt(75.0)],
                                         "label": "both bolt-hole surfaces"},
                              "description": "fully fixed"}],
        loads=[{"id": "pin_load", "type": "surface_force", "force": [0.0, -F, 0.0], "region": pin,
                "description": f"{F:g} N in -y as a uniform traction"}],
        qois=[
            _q(id="pin_disp_y", kind="region", field="U", component="y", stat="mean", region=pin,
               reference=None, tolerance={"rel": 0.01}, reference_type="numerical",
               provenance="Richardson extrapolation of the reference pipeline",
               description="mean y-displacement of the pin-hole surface"),
            _q(id="fillet_mises_max", kind="max", field="S", component="mises",
               within={"type": "box", "min": [10.0, 9.0, 9.0], "max": [90.0, 22.0, 22.0]},
               reference=None, tolerance={"rel": 0.035}, reference_type="numerical",
               provenance="Richardson extrapolation of the reference pipeline",
               note="3.5 %: the peak converges erratically under local refinement (finest value used, "
                    "conservative GCI 1.1 %), and the tolerance must stay >= 3x the reference uncertainty",
               description="peak von Mises stress in the fillet, 10 <= x <= 90 (away from the side faces)"),
            _q(id="reaction_y", kind="reaction",
               within={"type": "or", "regions": [{"type": "cylinder_solid", "point": [x, 55.0, 0.0],
                                                  "axis": [0, 0, 1], "radius": 5.6} for x in (25.0, 75.0)]},
               component="y", reference=F, tolerance={"rel": 0.005}, gate=True, reference_type="analytical",
               provenance="statics", description="total y-reaction at the bolts"),
        ],
        levels=[12.0, 8.0, 5.5, 4.0, 3.0],
        fields=lambda h: [{"type": "box", "min": [0.0, 6.0, 6.0], "max": [100.0, 22.0, 22.0],
                           "size": h / 3, "thickness": 10.0}],
        tags=["numerical", "fillet", "stress_concentration", "complex_geometry"],
    ))


for _f in (_lame_cylinder, _lame_sphere, _torsion_shaft, _gravity_column, _cantilever_static, _nafems_le10,
           _nafems_le1, _cantilever_modal, _euler_buckling, _bracket):
    _f()


# ============================================================================ building case files

@contextlib.contextmanager
def _quiet_stdout():
    """OpenCASCADE's STEP writer prints to the C-level stdout; silence it."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as dn:
        os.dup2(dn.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def _gmsh_init():
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    return gmsh


def _task_md(c: Case) -> str:
    from ..fem.regions import describe

    lines = [f"# {c.title}", "", c.description, "",
             "## Units", "", "mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)", "",
             "## Geometry", "", "`geometry.step` (one solid, millimetres).", "", "## Material", ""]
    for m in c.materials:
        props = ", ".join(f"{k} = {v:g}" for k, v in m.items() if k in ("E", "nu", "density"))
        lines.append(f"- {m['name']}: {props} (linear elastic, isotropic)")
    lines += ["", "## Boundary conditions", ""]
    for bc in c.boundary_conditions:
        lines.append(f"- **{bc['id']}**: {bc['description']} on {describe(bc['region'], 'mm')}")
    if c.loads:
        lines += ["", "## Loads", ""]
        for ld in c.loads:
            where = f" on {describe(ld['region'], 'mm')}" if "region" in ld else ""
            lines.append(f"- **{ld['id']}**: {ld['description']}{where}")
    a = c.analysis
    lines += ["", "## Analysis", "", {"static": "Linear static.",
                                      "frequency": f"Natural frequencies (at least {a.get('n_modes', 6)} modes).",
                                      "buckle": f"Linear buckling (at least {a.get('n_modes', 2)} modes)."}[a["type"]]]
    lines += ["", "## Quantities that will be evaluated", ""]
    for q in c.qois:
        lines.append(f"- `{q['id']}`: {q['description']}" + (" (load-path check)" if q.get("gate") else ""))
    lines += ["", "## What to submit", "",
              "A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. "
              "The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the "
              "quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.",
              ""]
    return "\n".join(lines)


def _public_task(c: Case) -> dict:
    return {
        "id": f"verification/{c.id}", "title": c.title, "description": c.description, "units": UNITS,
        "geometry": {"file": "geometry.step", "format": "step", "length_unit": "mm"},
        "analysis": c.analysis, "materials": c.materials, "boundary_conditions": c.boundary_conditions,
        "loads": c.loads, "solver": {"name": "calculix", "version": "2.21", "deck": "model.inp"},
        "evaluated_quantities": [{"id": q["id"], "kind": q["kind"], "description": q["description"],
                                  **{k: q[k] for k in ("point", "field", "component", "frame", "region", "mode", "stat",
                                                       "within", "moment", "start", "end") if k in q}}
                                 for q in c.qois],
        "tags": c.tags,
    }


def build_case(c: Case, root: Path = VERIFICATION_DIR) -> Path:
    d = root / c.id
    (d / "task").mkdir(parents=True, exist_ok=True)
    (d / "reference").mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_init()
    gmsh.model.add(c.id)
    c.geometry(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        raise RuntimeError(f"{c.id}: geometry has {len(vols)} volumes")
    volume = gmsh.model.occ.getMass(3, vols[0][1])
    bbox = gmsh.model.getBoundingBox(-1, -1)
    n_faces = len(gmsh.model.getEntities(2))
    with _quiet_stdout():
        gmsh.write(str(d / "task" / "geometry.step"))
    gmsh.finalize()
    (d / "task" / "task.json").write_text(json.dumps(_public_task(c), indent=1))
    (d / "task" / "task.md").write_text(_task_md(c))
    ref_path = d / "reference" / "reference.json"
    old = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    old_q = {q["id"]: q for q in old.get("qois", [])}
    qois = []
    for q in c.qois:
        q = dict(q)
        if q.get("reference") is None and q["id"] in old_q and old_q[q["id"]].get("reference") is not None:
            for k in ("reference", "uncertainty", "richardson"):
                if k in old_q[q["id"]]:
                    q[k] = old_q[q["id"]][k]
        qois.append(q)
    ref = {"id": f"verification/{c.id}", "volume": volume, "volume_tolerance": c.volume_tolerance,
           "bbox": list(bbox), "n_cad_faces": n_faces, "qois": qois, "notes": c.notes,
           "budget": old.get("budget", {})}
    ref_path.write_text(json.dumps(ref, indent=1))
    return d


def build_all(only: list[str] | None = None) -> list[Path]:
    return [build_case(c) for k, c in CASES.items() if not only or k in only]


# ============================================================================ reference pipeline

def mesh_task_geometry(case_id: str, h: float, root: Path = VERIFICATION_DIR, timeout: float = 3600.0):
    """Mesh the task's own STEP file (exactly what an agent receives) in a crash-isolated subprocess."""
    from ..fem.meshworker import mesh_step

    c = CASES[case_id]
    return mesh_step(root / case_id / "task" / "geometry.step", h, order=2, curvature=c.curvature,
                     fields=c.fields(h) if c.fields else None, timeout=timeout)


def run_reference(case_id: str, level: int, out: str | Path, threads: int = 1,
                  root: Path = VERIFICATION_DIR) -> dict:
    """Mesh the task STEP at refinement ``level`` (1 = coarsest), write model.inp, evaluate it."""
    from ..evaluate import evaluate_submission
    from ..fem.build import write_ccx_deck

    case_id = case_id.split("/")[-1]
    c = CASES[case_id]
    h = c.levels[level - 1]
    sub = Path(out) / f"{case_id}_L{level}"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    mesh = mesh_task_geometry(case_id, h, root)
    task = json.loads((root / case_id / "task" / "task.json").read_text())
    info = write_ccx_deck(sub / "model.inp", mesh, task)
    rep = evaluate_submission(root / case_id, sub, out=sub / "evaluation.json", threads=threads)
    rep["summary"]["level"] = level
    rep["summary"]["h"] = h
    rep["deck_info"] = info
    return rep


def richardson(h: list[float], q: list[float], safety: float = 1.25, formal_order: float = 2.0) -> dict:
    """Reference value + uncertainty from the three finest levels (h decreasing, h ~ n_dof^(-1/3)).

    Roache's grid convergence index: if the observed order p is plausible (0.5 <= p <= 4) the
    Richardson-extrapolated value is used with GCI = 1.25 |q3 - q2| / (r^p - 1).  Otherwise
    (oscillatory or erratic convergence) the finest value is used with the conservative
    GCI = 3 |q3 - q2| / (r^p_formal - 1).
    """
    (h1, h2, h3), (q1, q2, q3) = h[-3:], q[-3:]
    e12, e23 = q1 - q2, q2 - q3
    r32 = h2 / h3
    out = {"q_fine": q3, "monotone": bool(e12 * e23 > 0)}

    def fallback(reason):
        gci = 3.0 * abs(q3 - q2) / (r32**formal_order - 1)
        out.update(value=q3, order=None, uncertainty=float(gci), method=f"finest value ({reason}); GCI Fs=3")
        return out

    if abs(e23) < 1e-12 * max(1.0, abs(q3)):
        out.update(value=q3, order=None, uncertainty=0.0, method="converged to round-off")
        return out
    if e12 * e23 <= 0:
        return fallback("oscillatory")

    def f(p):  # observed order: (q1-q2)/(q2-q3) = (h1^p - h2^p)/(h2^p - h3^p)
        return (h1**p - h2**p) / (h2**p - h3**p) - e12 / e23

    lo, hi = 0.2, 8.0
    if f(lo) * f(hi) > 0:
        return fallback("no observed order")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    if not 0.5 <= p <= 4.0:
        return fallback(f"implausible order {p:.2f}")
    qx = q3 + (q3 - q2) / (r32**p - 1)
    gci = safety * abs(q3 - q2) / (r32**p - 1)
    out.update(value=float(qx), order=float(p), uncertainty=float(gci), method="richardson")
    return out


def _rescore(conv: dict, ref: dict) -> bool:
    """Re-score every recorded level against the current references and tolerances (no re-solve);
    sets ref["budget"] to the cheapest level that passes everything."""
    from ..scoring import ScoreConfig, case_score, score_qoi

    cfg = ScoreConfig()
    runs = conv["runs"]
    for r in runs:
        scored = [score_qoi(q, {"value": r["qois"][q["id"]]["value"]} if r["qois"].get(q["id"], {}).get("value")
                            is not None else None, cfg) for q in ref["qois"]]
        cs = case_score(scored, r["gates"], None, None, cfg)
        r["accuracy"] = cs["accuracy"]
        r["all_within_tolerance"] = cs["all_qois_within_tolerance"] and cs["gates_passed"]
        for q in scored:
            r["qois"].setdefault(q["id"], {}).update(error=q.get("error"), passed=q.get("passed"))
    conv["summary"] = {"levels": [r["level"] for r in runs], "n_dof": [r["n_dof"] for r in runs],
                       "accuracy": [round(r["accuracy"], 4) for r in runs],
                       "all_within_tolerance": [r["all_within_tolerance"] for r in runs],
                       "cpu_s": [round(r["solver_cpu_s"], 2) for r in runs]}
    passing = [r for r in runs if r["all_within_tolerance"] and all(r["gates"].values())]
    if passing:
        best = min(passing, key=lambda r: r["solver_cpu_s"])
        ref["budget"] = {"reference_level": best["level"], "reference_n_dof": best["n_dof"],
                         "reference_solver_cpu_s": best["solver_cpu_s"]}
        return True
    return False


def rescore_all(root: Path = VERIFICATION_DIR) -> None:
    """Refresh convergence evidence after tolerance/reference edits (cheap, no solver runs)."""
    for cid in CASES:
        ref_p, conv_p = root / cid / "reference" / "reference.json", root / cid / "reference" / "convergence.json"
        if not (ref_p.exists() and conv_p.exists()):
            continue
        ref, conv = json.loads(ref_p.read_text()), json.loads(conv_p.read_text())
        _rescore(conv, ref)
        ref_p.write_text(json.dumps(ref, indent=1))
        conv_p.write_text(json.dumps(conv, indent=1, default=float))


def convergence_study(case_id: str, levels: list[int] | None = None, workdir: str | Path = "/tmp/feabench-conv",
                      threads: int = 1, write: bool = True, root: Path = VERIFICATION_DIR) -> dict:
    case_id = case_id.split("/")[-1]
    c = CASES[case_id]
    levels = levels or list(range(1, len(c.levels) + 1))
    runs = []
    for lv in levels:
        rep = run_reference(case_id, lv, workdir, threads=threads, root=root)
        runs.append({
            "level": lv, "h": c.levels[lv - 1], "n_nodes": rep["mesh"].get("n_nodes"),
            "n_dof": 3 * (rep["mesh"].get("n_nodes") or 0), "solver_cpu_s": rep["summary"]["solver_cpu_s"],
            "solver_wall_s": rep["summary"]["solver_wall_s"], "gates": rep["gates"],
            "qois": {q["id"]: {"value": q.get("value"), "error": q.get("error"), "passed": q.get("passed"),
                               "failure": q.get("failure")} for q in rep["qois"]},
            "accuracy": rep["summary"]["accuracy"], "all_within_tolerance": rep["summary"]["all_qois_within_tolerance"],
        })
    out = {"case": f"verification/{case_id}", "pipeline": "gmsh 2nd-order tets (C3D10) -> ccx", "runs": runs}
    # numerical references
    ref_path = root / case_id / "reference" / "reference.json"
    ref = json.loads(ref_path.read_text())
    extrap = {}
    for q in ref["qois"]:
        vals = [r["qois"][q["id"]]["value"] for r in runs]
        hs = [r["n_dof"] ** (-1.0 / 3.0) if r["n_dof"] else None for r in runs]
        if len(runs) >= 3 and all(v is not None for v in vals[-3:]):
            formal = 3.0 if q.get("field") == "U" or q["kind"] in ("frequency", "buckling") else 2.0
            extrap[q["id"]] = richardson(hs, vals, formal_order=formal)
    out["richardson"] = extrap
    changed = False
    for q in ref["qois"]:
        if q.get("reference_type") == "numerical" and q["id"] in extrap:
            ex = extrap[q["id"]]
            q["reference"] = ex["value"]
            q["uncertainty"] = ex["uncertainty"]
            q["richardson"] = {k: v for k, v in ex.items() if k != "value"}
            rel_unc = abs(ex["uncertainty"] / ex["value"]) if ex["value"] else float("inf")
            q["uncertainty_rel"] = rel_unc
            # a tolerance tighter than 3x the reference uncertainty would be unfair
            if rel_unc * 3 > q["tolerance"].get("rel", 1):
                q.setdefault("warnings", []).append(f"tolerance < 3x reference uncertainty ({rel_unc:.2e})")
            changed = True
    changed |= _rescore(out, ref)
    if write:
        if changed:
            ref_path.write_text(json.dumps(ref, indent=1))
        (root / case_id / "reference" / "convergence.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ============================================================================ README table

def summary_table(root: Path = VERIFICATION_DIR) -> str:
    """Markdown table of every case: QoIs, reference type, tolerance, convergence evidence."""
    rows = ["| Case | QoI | Reference | Type | Tol. | Pipeline error at coarsest / finest mesh |",
            "|---|---|---|---|---|---|"]
    cost = ["| Case | Cheapest passing mesh (DOF) | Solver CPU there | Finest mesh (DOF) | CPU |", "|---|---|---|---|---|"]
    for cid in CASES:
        ref_p = root / cid / "reference" / "reference.json"
        conv_p = root / cid / "reference" / "convergence.json"
        if not ref_p.exists():
            continue
        ref = json.loads(ref_p.read_text())
        conv = json.loads(conv_p.read_text()) if conv_p.exists() else None
        for q in ref["qois"]:
            tol = q["tolerance"]
            tol_s = f"{tol['rel'] * 100:g} %" if "rel" in tol else f"{tol['abs']:g}"
            ev = ""
            if conv:
                errs = [r["qois"][q["id"]]["error"] for r in conv["runs"]]
                if errs[0] is not None and errs[-1] is not None:
                    ev = f"{errs[0] * 100:.2f} % / {errs[-1] * 100:.3f} %"
            refv = q.get("reference")
            unc = f" ± {q['uncertainty_rel'] * 100:.2g} %" if q.get("uncertainty_rel") else ""
            gate = " (gate)" if q.get("gate") else ""
            rows.append(f"| {cid} | `{q['id']}`{gate} | {refv:.6g}{unc} | {q['reference_type']} | {tol_s} | {ev} |")
        if conv:
            runs = conv["runs"]
            passing = [r for r in runs if r["all_within_tolerance"]]
            first = min(passing, key=lambda r: r["n_dof"]) if passing else None
            cost.append(f"| {cid} | {first['n_dof'] if first else '-'} | "
                        f"{first['solver_cpu_s']:.2f} s | {runs[-1]['n_dof']} | {runs[-1]['solver_cpu_s']:.1f} s |"
                        if first else f"| {cid} | none passing | - | {runs[-1]['n_dof']} | {runs[-1]['solver_cpu_s']:.1f} s |")
    return "\n".join(rows) + "\n\n" + "\n".join(cost) + "\n"


README_HEAD = """# Verification suite: CAD tasks with trustworthy references

Ten agent tasks. Each gives a STEP file (mm), a structured spec (`task/task.json`) and a brief
(`task/task.md`). The hidden reference in `reference/` holds QoI specs and reference values,
plus `convergence.json`: the evidence that each tolerance is attainable.

| Case | What it tests |
|---|---|
| `lame_cylinder` | pressure on a curved face, symmetry planes, plane strain (exact Lame solution) |
| `lame_sphere` | 3 symmetry planes, curved-surface stress with a steep 1/r^3 gradient (exact) |
| `torsion_shaft` | torque application (traction or distributing coupling), surface shear (exact Saint-Venant) |
| `gravity_column` | body force, density/acceleration units, rigid-body modes without over-constraint (exact) |
| `cantilever_tip_load` | distributed end load, root clamp; bending stress exact, deflection converged |
| `nafems_le10` | NAFEMS thick plate: mixed displacement constraints incl. a line constraint, 3D stress at a corner point (published) |
| `nafems_le1` | NAFEMS elliptic membrane as a 3D plate: outward traction, stress concentration at a notch (published, plane-stress definition) |
| `cantilever_modal` | natural frequencies, mass/density units (converged; Euler-Bernoulli shown) |
| `euler_buckling` | linear buckling with a reference load (converged; Euler shown) |
| `angle_bracket` | filleted part, bolt-hole fixation, pin-hole load, fillet peak stress (converged) |

**How references were established.** Closed-form solutions where an exact 3D solution of
exactly the stated problem exists. NAFEMS published targets for LE1/LE10. Otherwise, a
mesh-convergence study of the reference pipeline: gmsh quadratic tets on the task's own STEP,
fed to ccx 2.21 and scored by the same evaluator agents face, at five mesh sizes, with
Richardson extrapolation and Roache GCI (see `richardson()` in `feabench/suites/verification.py`).
Analytical and published references are *also* run through the convergence study, which proves
the tolerance is attainable and shows the cost of attaining it.

Regenerate: `feabench build-verification` (case files), `feabench convergence <case>`
(evidence and numerical references). Pipeline runs use 2 threads; CPU times are user+sys.

"""


def write_readme(root: Path = VERIFICATION_DIR) -> Path:
    p = root / "README.md"
    p.write_text(README_HEAD + "## References, tolerances and convergence evidence\n\n" + summary_table(root))
    return p
