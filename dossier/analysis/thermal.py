"""The thermal analysis engine: steady heat conduction in the component (analysis type `thermal_steady`).

    run(project, ctx, geo, lc, workdir, opts) -> pipeline.Analysis         the engine of the pipeline
    run_levels(project, ctx, geo, lc, levels, workdir, opts)                the same on given meshes
    solve_temperature(project, ctx, geo, thermal_lc, mesh, opts) -> dict    the temperature on ONE mesh, for
                                                                            the structural engine
    solve_heat(mesh_path, k, bcs, order) -> (x, T, summary)                 the solver function

The solver is the fixed solver of the repository (feacore, scikit-fem): quadratic tetrahedra, a direct
linear solve. This module makes its boundary conditions, evaluates the results and writes the record.

Idealization (each rule is one Decision of the load case; the BCItems of the record point to it)
    fluid_temperature with a heat transfer coefficient h
                       convection, k dT/dn = h (T_fluid - T), on every face of the features of its role. The
                       role `external` has no feature list: its faces are geo["interfaces"][id]["external"].
    temperature        a prescribed surface temperature.
    heat_flux          a heat flux. A positive value is a heat flow into the part.
    fluid_temperature without h      the engine does not guess a coefficient: status `not_supported`.
    a face with no thermal load      adiabatic. One Assumption names these features, gives their
                       temperatures in the analysis, and says in which direction an adjacent part changes them.
    one conductivity for all temperatures     an Assumption, with the change of the requested temperatures
                       for a conductivity that is 20 % lower and 20 % higher (two more solves on the
                       coarsest mesh).
    steady state       the planner records this assumption for every thermal load case (ASM-<load case>-
                       STEADY-STATE); the engine does not record it again.
Other reasons for `not_supported`: no thermal load at all, heat flux loads only (the steady temperature
is then not defined), two thermal loads on one face, a load on a role with no face, a material without k.

Results (keys `<quantity>.<measure>@<location>`, SI)
    temperature.min | max | mean | range @ part | <feature> | <interface>
                       `temperature.min@part` and `temperature.max@part` always, and every such key of
                       `lc.result_keys`. An interface id means every face of every role of the interface.
                       min and max are nodal values; `location` and `near` say where the extreme is. The mean
                       of faces is weighted by area, the mean of the part by volume.
    heat_flow.total@<interface>      the heat that enters the part through the loaded faces of the
                       interface, in W (negative: the heat leaves). For every interface with a thermal load.
A key of `lc.result_keys` that the engine cannot give is in the results with the status `not_run` and the
reason in its note.

Convergence (the rule of the structural engine, in absolute terms)
    h is the solver's element size (volume / cells)^(1/3) of each level, v the values, coarse to fine.
    one level       no study: no uncertainty, status not_converged.
    two levels      u = 3 |v2 - v1| x max(1, 1 / (r^2 - 1)), r = h1 / h2.
    three or more   feacore.verification.gci3 on the three finest levels, with the formal order p_t (3 for
                    a temperature, 2 for a heat flow):
                      monotone, observed order p    u = max(1.25 |d| / (r^p - 1), 3 |d| / (r^q - 1)), with
                                                    d the change between the two finest levels and
                                                    q = p limited to [0.5, p_t]
                      oscillatory                   u = max(3 |d| / (r^q - 1), spread of the three values)
                      no order                      u = 3 x spread
                      changes below 1e-6 of the value   u = 1.25 x the larger change
    `uncertainty_abs` is u, in the unit of the value; the compliance stage uses it for a temperature.
    `uncertainty` is u / |value| (CONTRACTS.md: relative to the value). The status of a temperature comes
    from u divided by the temperature range of the load case (the largest difference between the
    temperatures of its boundary conditions and of the part), and the note gives that ratio: heat
    conduction is driven by temperature differences, and u divided by the absolute temperature is small for
    every error. For a heat flow the status comes from u / |value|. A result is `converged` when that
    ratio is at most policy.max_uncertainty. The error-estimate check uses the same ratios: passed=False
    only when one is over the limit, passed=None when a result has no uncertainty (one mesh). The engine
    never gives `singular`: a temperature is bounded, and the heat flow through a whole face is finite also
    where the flux at an edge of the face is not. A Decision of the load case states the rule.

Checks
    heat balance    the sum of the heat flows through all loaded faces is zero. The flows come from the boundary
                    conditions and the FE temperature with the solver's own quadrature, so the sum is zero
                    to the accuracy of the linear solve on every mesh: the check finds a solve that failed
                    or a boundary condition that the evaluation lost. It does not measure the mesh error.
    mesh quality    feacore's corner mean-ratio quality (limit 0.01) and the solver's Jacobian check.
    error estimate  the largest relative uncertainty of the results against policy.max_uncertainty.

Files in the work folder (the paths in `artifacts` are relative to project.out)
    T_<mesh>_<key>.npz        x (n, 3), T (n,) in K and the evaluation of one mesh. `solve_temperature`
                              returns this path; it is a field file of the solver
                              (`Load(type="temperature", value={"field": path})`). The key is a hash of the
                              mesh file and of the boundary conditions, so a level is solved once: the next
                              call, in this process or a later one, reads the file. A changed mesh or load
                              gives a new file.
    fields_L<k>.npz           x and T of the two finest levels (Artifact kind `field`).
    temperature.png, temperature_section.png     when opts.pictures: the surface of the finest mesh and a
                              cut through the part. Rendered in a subprocess through feacore.ui.offscreen
                              (xvfb-run when there is no display). A failure of the renderer leaves
                              render.log and does not fail the analysis.

Verified (tests/dossier/test_dossier_thermal.py; numbers in the test docstrings)
    a slab with convection on both faces, a hollow cylinder with convection inside and outside (the
    logarithmic profile), a hollow fin with a convective lumped tip against a numerical solution of the fin
    equation, and the heat balance; the idealization rules and the record on a small project.
Not verified, not done
    * No radiation, no contact resistance, no heat source in the volume, no conductivity that depends on
      the temperature, no transient (the steady state does not bound the thermal stress of a chill-down).
    * A heat transfer coefficient is uniform on the faces of its role, as the system context gives it.
    * The direction in which an adiabatic face errs depends on the temperature of the adjacent part, which the
      system context does not give: the Assumption states the condition, it does not decide it.
    * The solves are not written to the run store of the dashboard (opts.record_runs is not used).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dossier import units
from dossier.model import AnalysisRecord, Artifact, BCItem, Check, Link, LoadCase, MeshLevel, Origin, ResultValue
from dossier.project import make_id
from dossier.records import Journal

THERMAL_KINDS = ("fluid_temperature", "temperature", "heat_flux")
ENGINE = "engine:thermal"
CACHE_VERSION = 1               # part of the cache key: raise it when the evaluation of a level changes
ORDER_TEMPERATURE = 3.0         # formal order of a nodal temperature, quadratic elements, smooth solution
ORDER_HEAT_FLOW = 2.0           # formal order of a heat flow
BALANCE_TOL = 1e-6              # heat balance: |sum of the heat flows| / heat flow through the part
QUALITY_LIMIT = 0.01            # feacore's "fail" threshold of the corner mean-ratio quality
K_PROBE = 0.2                   # the conductivity changes by this fraction in the sensitivity solves
MAX_NAMED = 8                   # feature ids in one sentence

# Sequential one-hue ramp for the pictures, light to dark (the reference palette of the dataviz guide).
RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b")
SURFACE, INK = "#fcfcfb", "#0b0b0b"


class NotSupported(Exception):
    """The engine has no model for this load case. The message is the exact reason, in STE."""


# --------------------------------------------------------------------------- text helpers


def fmt(value: float, unit: str) -> str:
    """An SI number for people: fmt(241.3, "K") -> "241.3 K"."""
    return units.display(units.from_si(float(value), unit))


def _join(items) -> str:
    """"A", "A and B", "A, B and C"."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _count(n: int, noun: str) -> str:
    """"1 face", "3 faces"."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def analysis_id(lc: LoadCase) -> str:
    return "AN-" + (lc.id[3:] if lc.id.startswith("LC-") else lc.id)


def _rel(project, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(project.out).resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- idealization


@dataclass
class ThermalBC:
    """One boundary condition of the heat conduction problem, on CAD faces (mesh groups `f<tag>`)."""

    kind: str                       # convection | temperature | flux
    faces: list[int]
    value: float                    # fluid temperature (K) | surface temperature (K) | heat flux into the part (W/m^2)
    h: float | None = None          # convection: heat transfer coefficient, W/(m^2 K)
    features: list[str] = field(default_factory=list)
    effect: str | None = None
    interface: str | None = None
    role: str = ""

    def spec(self) -> dict:
        """What the solution depends on (the cache key)."""
        return {"kind": self.kind, "faces": sorted(int(t) for t in self.faces), "value": float(self.value),
                "h": None if self.h is None else float(self.h)}


@dataclass
class Idealization:
    bcs: list[ThermalBC]
    adiabatic: list[int]            # faces with no thermal load
    k: float                        # W/(m K)
    material: str                   # material id


def _item_faces(geo: dict, item) -> list[int]:
    if item.role == "external" or list(item.features) == ["*"]:
        tags = geo.get("interfaces", {}).get(item.interface or "", {}).get("external", [])
    elif item.features:
        tags = []
        for f in item.features:
            feat = geo.get("features", {}).get(f)
            if feat is None:
                raise NotSupported(f"The effect {item.effect} acts on the feature {f}. The geometry stage has no "
                                   f"such feature.")
            tags += list(feat.get("faces", []))
    else:
        tags = geo.get("interfaces", {}).get(item.interface or "", {}).get(item.role, [])
    return sorted({int(t) for t in tags})


def idealize(project, ctx, geo: dict, lc: LoadCase) -> Idealization:
    """The thermal loads of a load case as boundary conditions. Raises NotSupported with the reason."""
    mat = project.material_of(lc.component)
    if mat.k is None:
        raise NotSupported(f"The material {mat.id} has no thermal conductivity. Give the material a value of k.")
    k = float(units.si(units.to(mat.k.quantity(ctx.names() if ctx is not None else None), "W/(m*K)")))
    if not k > 0:
        raise NotSupported(f"The thermal conductivity of the material {mat.id} is {fmt(k, 'W/(m*K)')}. It must be "
                           f"more than zero.")
    bcs: list[ThermalBC] = []
    for item in lc.loads:
        if item.kind not in THERMAL_KINDS:
            continue                        # a mechanical load is not a part of a heat conduction problem
        where = f"the role {item.role} of the interface {item.interface}" if item.interface else "the part"
        if item.si is None or isinstance(item.si, list):
            raise NotSupported(f"The effect {item.effect} ({item.kind}) on {where} has no value.")
        value = float(item.si)
        if item.kind != "heat_flux" and abs(item.factor - 1.0) > 1e-12:
            raise NotSupported(f"The effect {item.effect} gives a temperature with the load factor {item.factor:g}. "
                               f"The analysis cannot multiply a temperature by a factor.")
        faces = _item_faces(geo, item)
        if not faces:
            raise NotSupported(f"The effect {item.effect} ({item.kind}) acts on {where}. The geometry stage gives no "
                               f"face for it.")
        common = dict(faces=faces, features=[f for f in item.features if f != "*"], effect=item.effect,
                      interface=item.interface, role=item.role)
        if item.kind == "fluid_temperature":
            if item.h_si is None:
                raise NotSupported(
                    f"The effect {item.effect} gives a fluid temperature of {fmt(value, 'K')} on {where} without a "
                    f"heat transfer coefficient. The analysis does not guess a coefficient. Give the effect a heat "
                    f"transfer coefficient (h) in the system context.")
            if not item.h_si > 0:
                raise NotSupported(f"The effect {item.effect} gives a heat transfer coefficient of "
                                   f"{fmt(item.h_si, 'W/(m^2*K)')} on {where}. It must be more than zero.")
            bcs.append(ThermalBC(kind="convection", value=value, h=float(item.h_si), **common))
        elif item.kind == "temperature":
            bcs.append(ThermalBC(kind="temperature", value=value, **common))
        else:
            bcs.append(ThermalBC(kind="flux", value=value * item.factor, **common))
    if not bcs:
        raise NotSupported(f"The load case {lc.id} has no thermal load. A thermal analysis needs a fluid temperature, "
                           f"a surface temperature or a heat flux.")
    if all(b.kind == "flux" for b in bcs):
        raise NotSupported(f"The load case {lc.id} has heat flux loads only. A steady temperature needs a convection "
                           f"or a prescribed temperature.")
    owner: dict[int, str] = {}
    for b in bcs:
        for t in b.faces:
            if t in owner and owner[t] != b.effect:
                raise NotSupported(f"The effects {owner[t]} and {b.effect} act on the same face f{t}. The analysis "
                                   f"cannot apply two thermal loads to one face.")
            owner[t] = b.effect or ""
    every = sorted(int(t) for t in geo.get("faces", {}))
    return Idealization(bcs=bcs, adiabatic=[t for t in every if t not in owner], k=k, material=mat.id)


# --------------------------------------------------------------------------- the solver function


def heat_case(mesh_path, k: float, bcs: list[ThermalBC], order: int = 2):
    """The feacore case of the steady heat conduction problem: one boundary condition per face group."""
    from feacore.case import BC, Analysis, Case, Material

    out = []
    for b in bcs:
        for tag in b.faces:
            g = f"f{int(tag)}"
            if b.kind == "convection":
                out.append(BC(type="convection", group=g, h=float(b.h), T_inf=float(b.value)))
            elif b.kind == "temperature":
                out.append(BC(type="temperature", group=g, value=float(b.value)))
            elif b.kind == "flux":
                out.append(BC(type="flux", group=g, value=float(b.value)))
            else:
                raise ValueError(f"unknown thermal boundary condition {b.kind!r}")
    return Case(mesh=str(mesh_path), analysis=Analysis(physics="heat", formulation="3d", order=int(order)),
                materials={"material": Material(k=float(k))}, bcs=out)


def solve_heat(mesh_path, k: float, bcs: list[ThermalBC], order: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    """Steady heat conduction on one mesh with the fixed solver.

    The mesh has a facet group `f<tag>` for every CAD face. Returns the node coordinates x (n, 3), the
    temperatures T (n,) and the evaluation of the mesh (JSON data):
        faces      {tag: area, integral of T, min, max and where}, for every face group of the mesh
        part       min, max (with where), mean over the volume, volume
        reactions  for every boundary condition of kind temperature: the heat flow into the part, W
        n_nodes, n_cells, h_rep, min_quality, solver, warnings
    Areas and integrals use the quadrature of the solver, so `flows` balances to the accuracy of the solve."""
    from feacore.diagnostics.checks import mesh_quality
    from feacore.solver import solve

    res = solve(heat_case(mesh_path, k, bcs, order), backend="skfem")
    x, T = res.nodal("T")
    md = res.model
    faces: dict[str, dict] = {}
    for name, g in md.md.groups.items():
        if g.kind != "facets" or not (name.startswith("f") and name[1:].isdigit()):
            continue
        nodes = md.sdofs_of(name)
        integral, area = res._integral("T", name)         # the solver's quadrature: sum of T w, sum of w
        lo, hi = nodes[int(np.argmin(T[nodes]))], nodes[int(np.argmax(T[nodes]))]
        faces[name[1:]] = {"area": float(area), "integral": float(integral), "min": float(T[lo]), "max": float(T[hi]),
                           "xmin": x[lo].tolist(), "xmax": x[hi].tolist(), "nodes": int(len(nodes))}
    integral, volume = res._integral("T", None)
    lo, hi = int(np.argmin(T)), int(np.argmax(T))
    part = {"min": float(T[lo]), "max": float(T[hi]), "xmin": x[lo].tolist(), "xmax": x[hi].tolist(),
            "mean": float(integral / volume), "volume": float(volume)}
    # prescribed temperatures: the heat that the constraint gives to the part (each node counts once)
    reactions, taken = [], np.zeros(len(T), dtype=bool)
    fixed = np.zeros(len(T), dtype=bool)
    fixed[res.dirichlet_dofs] = True
    for b in bcs:
        if b.kind != "temperature":
            reactions.append(0.0)
            continue
        nodes = np.unique(np.concatenate([md.sdofs_of(f"f{int(t)}") for t in b.faces]))
        nodes = nodes[fixed[nodes] & ~taken[nodes]]
        taken[nodes] = True
        reactions.append(float(res.reaction_vector[nodes].sum()))
    info = res.info
    summary = {
        "faces": faces, "part": part, "reactions": reactions,
        "n_nodes": int(info.n_nodes), "n_cells": int(info.n_cells), "n_dofs": int(info.n_dofs),
        "order": int(info.order), "h_rep": float(info.h_rep), "wall_time": float(info.wall_time),
        "min_quality": float(mesh_quality(md.mesh).min()),
        "solver": next((w[6:].split(", n=")[0] for w in info.warnings if w.startswith("info: linear_solver")), ""),
        "warnings": [w for w in info.warnings if not w.startswith("info:")],
    }
    return np.ascontiguousarray(x), np.asarray(T, dtype=float), summary


def flows(bcs: list[ThermalBC], summary: dict) -> list[float]:
    """The heat that enters the part through each boundary condition, W (negative: it leaves)."""
    out = []
    for i, b in enumerate(bcs):
        f = [summary["faces"][str(int(t))] for t in b.faces]
        area, integral = sum(x["area"] for x in f), sum(x["integral"] for x in f)
        if b.kind == "convection":
            out.append(b.h * (b.value * area - integral))
        elif b.kind == "flux":
            out.append(b.value * area)
        else:
            out.append(summary["reactions"][i])
    return out


# --------------------------------------------------------------------------- one mesh level, solved once


@dataclass
class LevelSolution:
    x: np.ndarray
    T: np.ndarray
    summary: dict
    path: Path
    cached: bool = False


_MEMORY: dict[str, LevelSolution] = {}


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _solution(workdir: Path, mesh_path: Path, ideal: Idealization, order: int) -> LevelSolution:
    """The solution of one mesh, from the memory of this process, from its file, or from a new solve."""
    spec = {"v": CACHE_VERSION, "mesh": _file_digest(mesh_path), "k": ideal.k, "order": int(order),
            "bcs": [b.spec() for b in ideal.bcs]}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(workdir) / f"T_{Path(mesh_path).stem}_{key}.npz"
    hit = _MEMORY.get(str(path))
    if hit is not None and path.is_file():
        return LevelSolution(hit.x, hit.T, hit.summary, path, cached=True)
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    sol = LevelSolution(np.array(z["x"]), np.array(z["T"]), json.loads(str(z["summary"])), path, True)
                    _MEMORY[str(path)] = sol
                    return sol
        except Exception:  # noqa: BLE001  a file that cannot be read is solved again
            pass
    x, T, summary = solve_heat(mesh_path, ideal.k, ideal.bcs, order)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, x=x, T=T, key=np.array(key), summary=np.array(json.dumps(summary)))
    os.replace(tmp, path)
    sol = LevelSolution(x, T, summary, path, False)
    _MEMORY[str(path)] = sol
    return sol


def _mesh_path(project, mesh: MeshLevel) -> Path:
    p = Path(mesh.path)
    return p if p.is_absolute() else Path(project.out) / p


def solve_temperature(project, ctx, geo: dict, thermal_lc: LoadCase, mesh: MeshLevel, opts) -> dict:
    """The steady temperature of a thermal load case on one mesh: {"x": (n, 3), "T": (n,) in K, "path": str}.

    `path` is an .npz with x and T in the work folder of the thermal load case; the solver takes it as
    `Load(type="temperature", value={"field": path})` for an elasticity case on the SAME mesh and order. The
    field is kept per mesh: a second call for the same mesh file and the same loads reads the file.
    Raises StageError when the engine has no model for the load case."""
    from dossier.pipeline import StageError

    try:
        ideal = idealize(project, ctx, geo, thermal_lc)
    except NotSupported as e:
        raise StageError("analyze", f"the thermal load case {thermal_lc.id} is not supported: {e}") from e
    sol = _solution(project.analysis_dir(thermal_lc.id), _mesh_path(project, mesh), ideal, mesh.order or 2)
    return {"x": sol.x, "T": sol.T, "path": str(sol.path), "cached": sol.cached}


# --------------------------------------------------------------------------- results


class CannotServe(Exception):
    """The engine does not give this result key. The message is the reason, in STE."""


def _location_faces(geo: dict, loc: str) -> list[int] | None:
    """The faces of a result location; None for the whole part."""
    if loc == "part":
        return None
    if loc in geo.get("features", {}):
        return sorted(int(t) for t in geo["features"][loc].get("faces", []))
    if loc in geo.get("interfaces", {}):
        return sorted({int(t) for tags in geo["interfaces"][loc].values() for t in tags})
    raise CannotServe(f"The location {loc} is not the part, a feature or an interface of the component.")


def _near(geo: dict, tag: int) -> str:
    feats = geo.get("faces", {}).get(str(tag), {}).get("features") or []
    return feats[0] if feats else f"f{tag}"


def evaluate(key: str, ideal: Idealization, geo: dict, summary: dict) -> tuple[float, list[float] | None, str]:
    """One result on one mesh: (value, where the extreme is | None, the feature it is at | "")."""
    try:
        head, loc = key.split("@", 1)
        quantity, measure = head.split(".", 1)
    except ValueError:
        raise CannotServe(f"The key {key} does not have the form quantity.measure@location.") from None
    if quantity == "heat_flow":
        if measure != "total":
            raise CannotServe("The thermal analysis gives the total heat flow only (heat_flow.total).")
        if loc not in geo.get("interfaces", {}):
            raise CannotServe(f"The location {loc} is not an interface. A heat flow is the flow through an interface.")
        q = [v for b, v in zip(ideal.bcs, flows(ideal.bcs, summary)) if b.interface == loc]
        if not q:
            raise CannotServe(f"The interface {loc} has no thermal load in this load case, thus no heat flows through it.")
        return float(sum(q)), None, ""
    if quantity != "temperature":
        raise CannotServe(f"The thermal analysis gives temperatures and heat flows. It does not give {quantity}.")
    if measure not in ("min", "max", "mean", "range"):
        raise CannotServe(f"The thermal analysis gives the min, max, mean and range of a temperature. It does not "
                          f"give the measure {measure}.")
    tags = _location_faces(geo, loc)
    every = {int(t): x for t, x in summary["faces"].items()}
    f = every if tags is None else {t: every[t] for t in tags if t in every}
    if not f:
        raise CannotServe(f"The location {loc} has no face in the mesh.")
    # Faces that meet at an edge share its nodes, so several faces can hold the same extreme (a face at a
    # prescribed temperature and its neighbours). The face whose mean is nearest the extreme is the place.
    lo = min(f, key=lambda t: (f[t]["min"], f[t]["integral"] / f[t]["area"]))
    hi = max(f, key=lambda t: (f[t]["max"], f[t]["integral"] / f[t]["area"]))
    if tags is None:
        p = summary["part"]
        if measure == "mean":
            return p["mean"], None, ""
        if measure == "range":
            return p["max"] - p["min"], None, ""
        t = lo if measure == "min" else hi
        on_face = f[t][measure] == p[measure]           # else the extreme is at a node inside the part
        return p[measure], p["x" + measure], _near(geo, t) if on_face else ""
    if measure == "mean":
        return sum(x["integral"] for x in f.values()) / sum(x["area"] for x in f.values()), None, ""
    if measure == "range":
        return f[hi]["max"] - f[lo]["min"], None, ""
    t = lo if measure == "min" else hi
    return f[t][measure], f[t]["x" + measure], _near(geo, t)


def uncertainty(h: list[float], v: list[float], p_theory: float) -> dict:
    """The discretization uncertainty of the value on the finest mesh, in the unit of the value (module
    docstring, "Convergence"). h and v are coarse to fine. Returns u (None with one level), kind, and for
    a monotone sequence the observed order and the extrapolated value."""
    from feacore.verification import gci3

    out = {"u": None, "kind": "one_level", "order": None, "extrapolated": None}
    n = len(v)
    if n < 2:
        return out
    if n == 2 or len({round(float(x), 15) for x in h[-3:]}) < 3:
        d, r = abs(v[-1] - v[-2]), h[-2] / h[-1]
        out["kind"] = "two_levels"
        out["u"] = 3.0 * d * max(1.0, 1.0 / (r**2 - 1.0)) if r > 1.0 else (0.0 if d == 0 else math.inf)
        return out
    h3, v3 = [float(x) for x in h[-3:]], [float(x) for x in v[-3:]]
    g = gci3(h3, v3, p_theory=p_theory)
    d = abs(v3[2] - v3[1])
    spread = max(v3) - min(v3)
    r21 = g.r21

    def gci(factor: float, p: float) -> float:
        x = p * math.log(r21)
        return factor * d / math.expm1(x) if x < 700 else 0.0

    out["kind"] = g.convergence
    if g.convergence in ("exact", "converged"):
        out["u"] = 1.25 * max(abs(v3[2] - v3[1]), abs(v3[1] - v3[0]))
    elif g.convergence in ("monotone", "oscillatory") and g.p is not None:
        q = min(max(g.p, 0.5), p_theory)
        if g.convergence == "monotone":
            out["u"] = max(gci(1.25, g.p), gci(3.0, q))
            out["order"], out["extrapolated"] = float(g.p), (float(g.phi_ext) if g.phi_ext is not None else None)
        else:
            out["u"] = max(gci(3.0, q), spread)
    else:
        out["u"] = 3.0 * spread
    return out


_KIND_NOTE = {
    "one_level": "The analysis has one mesh. An uncertainty needs two meshes or more.",
    "two_levels": "The uncertainty comes from the change between two meshes.",
    "monotone": "The values of the three finest meshes converge in one direction.",
    "oscillatory": "The values of the three finest meshes go up and down.",
    "exact": "The values of the three finest meshes are equal.",
    "converged": "The values of the three finest meshes are equal to six digits.",
}


def _ratio(u: float | None, ref: float | None) -> float | None:
    """u / ref; None when u is not known."""
    if u is None or not math.isfinite(u):
        return None
    return (u / ref) if ref and ref > 0 else (0.0 if u == 0 else None)


def _status_ratio(r: ResultValue, t_range: float) -> float | None:
    """The relative uncertainty that sets the status of a result: a temperature in relation to the range."""
    if r.key.startswith("temperature."):
        return _ratio(r.uncertainty_abs, t_range)
    return r.uncertainty


def _result(key: str, values: list[float], hs: list[float], scale: float | None, limit: float,
            where: list[float] | None, near: str) -> ResultValue:
    """The ResultValue of one key from its values on the levels. `scale` is the reference of the relative
    uncertainty of a temperature (the temperature range of the load case); None: the value itself."""
    is_temp = key.startswith("temperature.")
    c = uncertainty(hs, values, ORDER_TEMPERATURE if is_temp else ORDER_HEAT_FLOW)
    u = c["u"]
    rel = _ratio(u, scale if is_temp else abs(values[-1]))      # the ratio that sets the status
    status = "converged" if rel is not None and rel <= limit else "not_converged"
    note = _KIND_NOTE.get(c["kind"], "The changes between the three finest meshes do not decrease.")
    if is_temp and rel is not None:
        note += (f" The uncertainty is {fmt(u, 'K')}. This is {100 * rel:.2g} % of the temperature range of "
                 f"{fmt(scale, 'K')} of the load case, and this ratio sets the status.")
    # `uncertainty` is relative to the value itself (CONTRACTS.md). For a temperature the compliance stage uses
    # `uncertainty_abs` in K; the ratio to the range goes in the note only.
    return ResultValue(key=key, value=float(values[-1]), unit="K" if is_temp else "W",
                       uncertainty=_ratio(u, abs(values[-1])) if is_temp else rel,
                       uncertainty_abs=u if u is not None and math.isfinite(u) else None,
                       status=status, location=where, near=near, per_level=[float(x) for x in values],
                       extrapolated=c["extrapolated"], order=c["order"], method="fe", note=note)


# --------------------------------------------------------------------------- the engine


def run(project, ctx, geo: dict, lc: LoadCase, workdir: Path, opts):
    """Analyse a thermal load case on the mesh levels of dossier.meshing (the engine of the pipeline)."""
    from dossier.pipeline import Analysis

    try:
        from dossier import meshing
    except ModuleNotFoundError as e:
        if e.name != "dossier.meshing":
            raise
        return Analysis(_stub(lc, "not_supported", "The mesh engine dossier.meshing is not in this version."),
                        Journal("solve", ENGINE))
    return run_levels(project, ctx, geo, lc, meshing.levels(project, geo, opts), workdir, opts)


def _stub(lc: LoadCase, status: str, message: str) -> AnalysisRecord:
    return AnalysisRecord(id=analysis_id(lc), title=lc.title, load_case=lc.id, analysis_type=lc.analysis_type,
                          status=status, message=message, origin=Origin(by=ENGINE, stage="analyze"))


def run_levels(project, ctx, geo: dict, lc: LoadCase, levels: list[MeshLevel], workdir: Path, opts):
    """The analysis on given mesh levels, coarse to fine: (AnalysisRecord, Journal)."""
    from dossier.pipeline import Analysis

    t0 = time.time()
    j = Journal("solve", ENGINE)
    log = getattr(opts, "log", None) or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if lc.analysis_type != "thermal_steady":
        return Analysis(_stub(lc, "not_supported", f"The thermal engine does not analyse the type {lc.analysis_type}."), j)
    try:
        ideal = idealize(project, ctx, geo, lc)
    except NotSupported as e:
        return Analysis(_stub(lc, "not_supported", str(e)), j)
    if not levels:
        return Analysis(_stub(lc, "failed", "The mesh engine gave no mesh level."), j)

    # ---- one solve per level (kept per mesh: the structural engine asks for the same fields)
    sols = []
    for lev in levels:
        sol = _solution(workdir, _mesh_path(project, lev), ideal, lev.order or 2)
        s = sol.summary
        log(f"  thermal L{lev.level}: {s['n_nodes']} nodes, T {s['part']['min']:.2f} to {s['part']['max']:.2f} K"
            + (" (from the file of an earlier solve)" if sol.cached else f", {s['wall_time']:.1f} s"))
        sols.append(sol)
    fine = sols[-1].summary
    hs = [s.summary["h_rep"] for s in sols]
    missing = sorted({t for b in ideal.bcs for t in b.faces} - {int(t) for t in fine["faces"]})
    if missing:     # the solver stops on an unknown group, so this is a group that is not a face group
        raise RuntimeError(f"the mesh has no face group for the faces {missing}")

    # ---- results
    bc_T = [b.value for b in ideal.bcs if b.kind != "flux"]
    t_range = max(bc_T + [fine["part"]["max"]]) - min(bc_T + [fine["part"]["min"]])
    limit = float(project.policy.max_uncertainty)
    keys = ["temperature.min@part", "temperature.max@part"]
    keys += [k for k in lc.result_keys if k not in keys]
    for iface in dict.fromkeys(b.interface for b in ideal.bcs if b.interface):
        if f"heat_flow.total@{iface}" not in keys:
            keys.append(f"heat_flow.total@{iface}")
    results: dict[str, ResultValue] = {}
    for key in keys:
        try:
            per = [evaluate(key, ideal, geo, s.summary) for s in sols]
        except CannotServe as e:
            results[key] = ResultValue(key=key, value=None, status="not_run", method="fe",
                                       note=f"The thermal analysis does not give this result. {e}")
            continue
        results[key] = _result(key, [p[0] for p in per], hs, t_range, limit, per[-1][1], per[-1][2])

    # ---- records of the idealization, the checks, the files
    bc_items = _bc_records(j, lc, geo, ideal, fine)
    asm_ids = _assumptions(j, project, ctx, geo, lc, ideal, levels, sols, results)
    conv = _convergence_decision(j, lc, len(levels), t_range, limit, project.policy.id)
    checks = _checks(lc, ideal, sols, results, t_range, limit)
    mesh = [lev.model_copy(update={"nodes": s.summary["n_nodes"], "cells": s.summary["n_cells"],
                                   "dofs": s.summary["n_dofs"], "order": s.summary["order"],
                                   "h_rep": s.summary["h_rep"],
                                   "min_quality": lev.min_quality if lev.min_quality is not None
                                   else s.summary["min_quality"]})
            for lev, s in zip(levels, sols)]
    artifacts = []
    for lev, sol in list(zip(levels, sols))[-2:]:
        f = workdir / f"fields_L{lev.level}.npz"
        with open(f, "wb") as fh:
            np.savez(fh, x=sol.x, T=sol.T)
        artifacts.append(Artifact(path=_rel(project, f), kind="field",
                                  caption=f"Node coordinates x in m and temperatures T in K of mesh level {lev.level}."))
    if getattr(opts, "pictures", False):
        artifacts += _pictures(project, workdir, _mesh_path(project, levels[-1]), sols[-1], lc, log)

    n_faces = len({t for b in ideal.bcs for t in b.faces})
    counts = {k: sum(1 for b in ideal.bcs if b.kind == k) for k in ("convection", "temperature", "flux")}
    loads = _join([f"{n} {name}" for k, name in (("convection", "convection"), ("temperature", "prescribed temperature"),
                                                 ("flux", "heat flux")) if (n := counts[k])])
    comp = project.components[lc.component]
    statement = (
        f"The analysis gives the steady temperatures of the component {comp.id}"
        + (f" in the phase {lc.phase}. " if lc.phase else ". ")
        + f"The fixed solver solves the heat conduction on {len(levels)} "
        + ("mesh" if len(levels) == 1 else "meshes") + f" of quadratic tetrahedra. The finest mesh has "
        f"{fine['n_nodes']} nodes. The thermal loads are {loads} boundary conditions on {_count(n_faces, 'face')}. "
        + ("The other face is adiabatic. " if len(ideal.adiabatic) == 1
           else f"The other {len(ideal.adiabatic)} faces are adiabatic. " if ideal.adiabatic
           else "Every face has a thermal load. ")
        + f"The lowest temperature of the part is {fmt(fine['part']['min'], 'K')} and the highest is "
        f"{fmt(fine['part']['max'], 'K')}.")
    import skfem
    record = AnalysisRecord(
        id=analysis_id(lc), title=lc.title or f"Steady temperatures of {lc.phase or lc.component}", load_case=lc.id,
        analysis_type=lc.analysis_type, status="done", statement=statement,
        solver=f"feacore fixed solver, backend skfem (scikit-fem {skfem.__version__}), steady heat conduction"
               + (f"; {fine['solver']}" if fine.get("solver") else ""),
        material={"id": ideal.material, "k": ideal.k, "k_unit": "W/(m*K)"},
        bcs=bc_items, mesh=mesh, results=results, checks=checks, artifacts=artifacts,
        links=[Link(rel="uses", to=i) for i in dict.fromkeys(
            [ideal.material, *[b.decision for b in bc_items if b.decision], *asm_ids, conv])],
        origin=Origin(by=ENGINE, stage="analyze"), runtime_s=round(time.time() - t0, 1))
    return Analysis(record, j)


# --------------------------------------------------------------------------- records


def _targets(b: ThermalBC) -> str:
    if b.role == "external" or not b.features:
        return f"the {_count(len(b.faces), 'face')} of the role {b.role or 'external'} of {b.interface}"
    if len(b.features) > MAX_NAMED:
        return f"the {len(b.features)} features of the role {b.role} of {b.interface}"
    return ("the feature " if len(b.features) == 1 else "the features ") + _join(b.features)


def _bc_records(j: Journal, lc: LoadCase, geo: dict, ideal: Idealization, fine: dict) -> list[BCItem]:
    """One Decision and one BCItem for each thermal load, and the BCItem of the adiabatic faces."""
    items = []
    for b in ideal.bcs:
        where = f"the role {b.role} of {b.interface}" if b.interface else "the part"
        T = fmt(b.value, "K")
        if b.kind == "convection":
            h = fmt(b.h, "W/(m^2*K)")
            title = f"Convection on {where}"
            choice = (f"The analysis applies convection to {_targets(b)}. The fluid temperature is {T} and the heat "
                      f"transfer coefficient is {h}.")
            why = f"The effect {b.effect} gives this fluid temperature and this heat transfer coefficient for {where}."
            alts = [("Prescribe the fluid temperature on the faces.",
                     "A prescribed temperature ignores the thermal resistance between the fluid and the wall."),
                    ("Use a heat transfer coefficient that changes along the faces.",
                     "The system context gives one value for the role.")]
            value = {"h": b.h, "T_fluid": b.value}
            text = (f"The analysis applies convection with a heat transfer coefficient of {h} and a fluid temperature of {T} "
                    f"to {_count(len(b.faces), 'face')}.")
        elif b.kind == "temperature":
            title = f"Prescribed temperature on {where}"
            choice = f"The analysis prescribes a temperature of {T} on {_targets(b)}."
            why = f"The effect {b.effect} gives this surface temperature for {where}."
            alts = [("Apply convection with a heat transfer coefficient.",
                     "The effect gives a surface temperature. It gives no fluid and no coefficient.")]
            value = {"T": b.value}
            text = f"The analysis prescribes a temperature of {T} on {_count(len(b.faces), 'face')}."
        else:
            q = fmt(b.value, "W/m^2")
            title = f"Heat flux on {where}"
            choice = (f"The analysis applies a heat flux of {q} to {_targets(b)}. A positive value is a heat flow "
                      f"into the part.")
            why = f"The effect {b.effect} gives this heat flux for {where}."
            alts = [("Prescribe a temperature on the faces.", "The effect gives a heat flux. It gives no temperature.")]
            value = {"q": b.value}
            text = f"The analysis applies a heat flux of {q} into the part to {_count(len(b.faces), 'face')}."
        if b.role == "external":
            why += " The geometry stage gives the faces that no other role uses."
        d = j.decide(subject=lc.id, key=f"bc-{b.effect}", title=title, choice=choice, rationale=why, alternatives=alts,
                     basis="rule", stage="idealize", links=[("uses", b.effect)] + ([("about", b.interface)] if b.interface else []))
        items.append(BCItem(type=b.kind, features=list(b.features), faces=[f"f{t}" for t in b.faces], value=value,
                            effect=b.effect, interface=b.interface, decision=d.id, description=text))
    if ideal.adiabatic:
        feats = list(dict.fromkeys(f for t in ideal.adiabatic for f in geo["faces"].get(str(t), {}).get("features", [])))
        items.append(BCItem(type="adiabatic", features=feats, faces=[f"f{t}" for t in ideal.adiabatic], value=None,
                            decision=make_id("ASM", lc.id, "adiabatic"),
                            description=f"Heat does not flow through the {_count(len(ideal.adiabatic), 'face')} with no thermal load."))
    return items


def _adiabatic_groups(project, geo: dict, lc: LoadCase, ideal: Idealization, fine: dict) -> list[dict]:
    """The adiabatic faces by interface: features, the element on the other side, the temperatures."""
    comp = project.components[lc.component]
    by_iface: dict[str, list[int]] = {}
    for t in ideal.adiabatic:
        roles = geo["faces"].get(str(t), {}).get("roles") or []
        ifaces = list(dict.fromkeys(r.split(":", 1)[0] for r in roles)) or [""]
        by_iface.setdefault(ifaces[0], []).append(t)
    conns = {c.id: c for c in project.system.connections}
    elements = {e.id: e for e in project.system.elements}
    out = []
    for iface, tags in by_iface.items():
        feats = list(dict.fromkeys(f for t in tags for f in geo["faces"].get(str(t), {}).get("features", [])))
        loose = [f"f{t}" for t in tags if not geo["faces"].get(str(t), {}).get("features")]
        it = next((i for i in comp.interfaces if i.id == iface), None)
        other = ""
        if it is not None and it.connection in conns:
            c = conns[it.connection]
            oid = c.to if c.from_ == comp.element else c.from_
            other = oid if oid in elements else ""
        f = [fine["faces"][str(t)] for t in tags if str(t) in fine["faces"]]
        out.append({"interface": iface, "features": feats, "faces": loose, "other": other,
                    "lo": min(x["min"] for x in f), "hi": max(x["max"] for x in f)})
    return out


def _named(iface: str, what: str, names: list[str]) -> list[str]:
    """Sentences that name the adiabatic features (or faces) of an interface, at most MAX_NAMED in each."""
    out = []
    for i in range(0, len(names), MAX_NAMED):
        part = names[i:i + MAX_NAMED]
        noun = what if len(part) == 1 else what + "s"
        if iface:
            out.append(f"{'The interface ' + iface + ' has' if not out else 'It also has'} the adiabatic {noun} "
                       f"{_join(part)}.")
        else:
            out.append(f"The adiabatic {noun} {_join(part)} " + ("is" if len(part) == 1 else "are")
                       + " in no interface.")
    return out


def _lower_limits(project, lc: LoadCase) -> tuple[list[str], list[str]]:
    """Requirements of the load case with a lower limit and with an upper limit of a temperature."""
    lower, upper = [], []
    for ref in lc.criteria:
        try:
            crit = project.requirement(ref.requirement).criteria[ref.criterion]
        except (KeyError, IndexError):
            continue
        if crit.quantity == "temperature":
            (lower if crit.comparator == ">=" else upper).append(ref.requirement)
    return list(dict.fromkeys(lower)), list(dict.fromkeys(upper))


def _assumptions(j: Journal, project, ctx, geo: dict, lc: LoadCase, ideal: Idealization, levels: list[MeshLevel],
                 sols: list[LevelSolution], results: dict[str, ResultValue]) -> list[str]:
    fine = sols[-1].summary
    ids = []
    lower, upper = _lower_limits(project, lc)

    # ---- faces with no thermal load
    if ideal.adiabatic:
        groups = _adiabatic_groups(project, geo, lc, ideal, fine)
        st = ["The faces with no thermal load are adiabatic."]
        for g in groups:
            st += _named(g["interface"], "feature", g["features"]) + _named(g["interface"], "face", g["faces"])
        impact = ["No heat flows through an adiabatic face. In the hardware, heat flows between the adjacent part and the face.",
                  "An adjacent part that is colder than the face makes the part colder than the analysis gives.",
                  "An adjacent part that is warmer than the face makes the part warmer than the analysis gives."]
        if lower and not upper:
            impact.append(f"{_join(lower)} gives a lower limit of a temperature, thus a colder adjacent part is the "
                          f"unsafe case.")
        elif upper and not lower:
            impact.append(f"{_join(upper)} gives an upper limit of a temperature, thus a warmer adjacent part is the "
                          f"unsafe case.")
        temps = []
        for g in groups:
            who = f"the faces of {g['interface']}" if g["interface"] else "the faces of no interface"
            mate = f" The adjacent element is {g['other']}." if g["other"] else ""
            temps.append(f"The analysis gives {fmt(g['lo'], 'K')} to {fmt(g['hi'], 'K')} on {who}.{mate}")
        a = j.assume(
            subject=lc.id, key="adiabatic", title="Faces with no thermal load are adiabatic", stage="idealize",
            statement=" ".join(st),
            rationale=(f"The system context gives no thermal effect for these faces"
                       + (f" in the phase {lc.phase}." if lc.phase else ".")
                       + " The analysis does not guess a heat transfer coefficient or a temperature of an adjacent part."),
            impact=" ".join(impact) + "\n\n" + "\n\n".join(temps),
            verify_by="Compare the temperature of each adjacent part with the temperatures that the analysis gives on "
                      "its faces. If the difference is important, add a thermal effect for the interface to the "
                      "system context.",
            conservative=None, links=[("about", g["interface"]) for g in groups if g["interface"]])
        ids.append(a.id)

    # ---- one conductivity
    mat = project.materials[ideal.material]
    k_text = fmt(ideal.k, "W/(m*K)")
    lo_T, hi_T = fine["part"]["min"], fine["part"]["max"]
    impact = (f"The temperatures in the part depend on the conductivity. The analysis gives temperatures from "
              f"{fmt(lo_T, 'K')} to {fmt(hi_T, 'K')}.")
    probes = []
    probe = [k for k in lc.result_keys if k.startswith("temperature.") and k in results and results[k].value is not None][:3]
    probe = probe or ["temperature.mean@part"]
    try:
        mesh0 = _mesh_path(project, levels[0])
        base = sols[0].summary
        low = solve_heat(mesh0, ideal.k * (1 - K_PROBE), ideal.bcs, levels[0].order or 2)[2]
        high = solve_heat(mesh0, ideal.k * (1 + K_PROBE), ideal.bcs, levels[0].order or 2)[2]
        pc = f"{100 * K_PROBE:.0f} %"
        for key in probe:
            v0, v1, v2 = (evaluate(key, ideal, geo, s)[0] for s in (base, low, high))
            probes.append(f"A conductivity that is {pc} lower changes {key} by {_signed(v1 - v0)}, and one that is "
                          f"{pc} higher changes it by {_signed(v2 - v0)}.")
        probes.append("These changes come from two more solves on the coarsest mesh.")
    except CannotServe:
        probes = []
    a = j.assume(
        subject=lc.id, key="conductivity", title="One thermal conductivity for all temperatures", stage="idealize",
        statement=f"The thermal conductivity of the material {mat.id} is {k_text} at every temperature of the load case.",
        rationale="The material data give one value of the conductivity."
                  + (f' The note of the material says: "{" ".join(mat.note.split())}"' if mat.note.strip() else ""),
        impact=impact + ("\n\n" + " ".join(probes) if probes else ""),
        verify_by=f"Compare {k_text} with the conductivity of the material specification at {fmt(lo_T, 'K')} and at "
                  f"{fmt(hi_T, 'K')}.",
        conservative=None, links=[("uses", mat.id)])
    ids.append(a.id)

    return ids


def _signed(dT: float) -> str:
    """A temperature change with its sign: "+4.21 K", "-3.50 K"."""
    return f"{dT:+.2f} K"


def _convergence_decision(j: Journal, lc: LoadCase, n: int, t_range: float, limit: float, policy_id: str) -> str:
    pc = f"{100 * limit:g} %"
    if n >= 3:
        how = ("The uncertainty of a result comes from the Grid Convergence Index of the three finest meshes, with "
               "a safety factor of 1.25 to 3.")
    elif n == 2:
        how = ("The uncertainty of a result is three times the change of its value between the two meshes. "
               "The uncertainty is more if the meshes are almost equal in size.")
    else:
        how = "One mesh gives no uncertainty, thus no result is converged."
    d = j.decide(
        subject=lc.id, key="convergence", title="Mesh convergence of the thermal results", stage="solve",
        choice=(f"The analysis solves the same model on {n} " + ("mesh" if n == 1 else "meshes") + f". {how} "
                f"For a temperature, the analysis divides the uncertainty by the temperature range of the load case, "
                f"{fmt(t_range, 'K')}. For a heat flow, it divides the uncertainty by the value. "
                f"A result is converged if this ratio is {pc} or less."),
        rationale=(f"Temperature differences cause the heat conduction, thus an error of a temperature is important "
                   f"in relation to the temperature range. The policy gives the limit of {pc}."),
        alternatives=[("Divide the uncertainty of a temperature by the absolute temperature.",
                       "This ratio is small for every error. It hides an error that is large in relation to the "
                       "temperature differences of the load case.")],
        basis="rule", links=[("uses", policy_id)])
    return d.id


def _checks(lc: LoadCase, ideal: Idealization, sols: list[LevelSolution], results: dict[str, ResultValue],
            t_range: float, limit: float) -> list[Check]:
    an = analysis_id(lc)
    out = []
    # heat balance, on every level; the worst one is reported
    worst, total, through = 0.0, 0.0, 0.0
    for s in sols:
        q = flows(ideal.bcs, s.summary)
        thr = 0.5 * sum(abs(x) for x in q)
        ratio = abs(sum(q)) / thr if thr > 0 else 0.0
        if ratio >= worst:
            worst, total, through = ratio, sum(q), thr
    out.append(Check(
        id=make_id("CHK", an, "heat-balance"), kind="heat_balance", passed=bool(worst <= BALANCE_TOL),
        value=float(worst), limit=BALANCE_TOL,
        statement=(f"The sum of the heat flows through all faces with a thermal load is {fmt(total, 'W')}. The heat flow "
                   f"through the part is {fmt(through, 'W')}. The ratio is {worst:.1e} and the limit is {BALANCE_TOL:g}.")))
    # mesh quality
    q = min(s.summary["min_quality"] for s in sols)
    bad = [w for s in sols for w in s.summary["warnings"] if "Jacobian" in w or "degenerate" in w]
    out.append(Check(
        id=make_id("CHK", an, "mesh-quality"), kind="mesh_quality", passed=bool(q >= QUALITY_LIMIT and not bad),
        value=float(q), limit=QUALITY_LIMIT,
        statement=(f"The smallest element quality of the meshes is {q:.2f} and the limit is {QUALITY_LIMIT:g}. "
                   f"A regular element has the quality 1. "
                   + ("The solver reports: " + "; ".join(dict.fromkeys(bad)) + "." if bad
                      else "No element has a negative Jacobian."))))
    # error estimate
    rated = {k: r for k, r in results.items() if r.value is not None}
    if len(sols) < 2:
        out.append(Check(id=make_id("CHK", an, "error-estimate"), kind="error_estimate", passed=None, limit=limit,
                         statement="The analysis has one mesh, thus it has no estimate of the discretization error."))
    else:
        ratios = {k: _status_ratio(r, t_range) for k, r in rated.items()}
        unknown = [k for k, x in ratios.items() if x is None]
        known = {k: x for k, x in ratios.items() if x is not None}
        k_worst = max(known, key=known.get) if known else None
        text = []
        if k_worst is not None:
            r = rated[k_worst]
            ref = (f"the temperature range of {fmt(t_range, 'K')}" if k_worst.startswith("temperature.")
                   else "its value")
            text.append(f"The largest relative uncertainty of a result is {100 * known[k_worst]:.2g} % ({k_worst}). "
                        f"This is {fmt(r.uncertainty_abs, r.unit)} in relation to {ref}.")
        if unknown:
            text.append(f"The analysis has no uncertainty for {_join(unknown)}.")
        text.append(f"The limit of the policy is {100 * limit:g} %.")
        # False only when a value is over the limit; a result with no uncertainty leaves the check open (None)
        over = bool(known) and max(known.values()) > limit
        ok = False if over else (True if known and not unknown else None)
        out.append(Check(id=make_id("CHK", an, "error-estimate"), kind="error_estimate", passed=ok,
                         value=float(max(known.values())) if known else None, limit=limit, statement=" ".join(text)))
    return out


# --------------------------------------------------------------------------- pictures


def _pictures(project, workdir: Path, mesh_path: Path, sol: LevelSolution, lc: LoadCase, log) -> list[Artifact]:
    """Render the temperature of the finest mesh in a subprocess (VTK needs an OpenGL context, and it stops
    the process when it has none). A failure leaves render.log and no picture."""
    surface, section = workdir / "temperature.png", workdir / "temperature_section.png"
    for p in (surface, section):
        p.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "feacore.ui.offscreen", str(Path(__file__).resolve()), "render",
           str(mesh_path), str(sol.path), str(surface), str(section)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        text, ok = (p.stdout or "") + (p.stderr or ""), p.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as e:
        text, ok = f"{type(e).__name__}: {e}", False
    s = sol.summary["part"]
    span = f"The lowest temperature is {fmt(s['min'], 'K')} and the highest is {fmt(s['max'], 'K')}."
    out = []
    if surface.is_file():
        out.append(Artifact(path=_rel(project, surface), kind="picture",
                            caption=f"Temperature on the surface of the part, finest mesh. {span}"))
    if section.is_file():
        out.append(Artifact(path=_rel(project, section), kind="picture",
                            caption=f"Temperature on a cut through the part, finest mesh. {span}"))
    if not ok or not out:
        (workdir / "render.log").write_text(text)
        log(f"  thermal: the picture renderer failed (see {_rel(project, workdir / 'render.log')})")
        out.append(Artifact(path=_rel(project, workdir / "render.log"), kind="log",
                            caption="The picture renderer failed. This file holds its messages."))
    return out


def render(mesh_path: str, field_path: str, surface_png: str, section_png: str) -> None:
    """Two pictures of a temperature field (runs in the render subprocess): the surface of the part, and the
    part cut by the plane through the middle of its bounding box, normal to its shortest side.

    One sequential hue: the lightest step is the temperature nearest the ambient end of the scale and the
    darkest step is the temperature farthest from the mean of the surface, so the region that the thermal
    loads change most is the dark one. The scale bar gives the numbers."""
    import meshio
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import cKDTree

    m = meshio.read(mesh_path)
    pts = np.asarray(m.points, dtype=float)
    cells = m.cells_dict
    if "tetra10" in cells:
        grid = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: np.asarray(cells["tetra10"])}, pts)
    else:
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: np.asarray(cells["tetra"])}, pts)
    with np.load(field_path, allow_pickle=False) as z:
        x, T = np.asarray(z["x"], dtype=float), np.asarray(z["T"], dtype=float)
    grid.point_data["T"] = T[cKDTree(x).query(pts)[1]]
    lo, hi = float(T.min()), float(T.max())
    # most of the surface near the cold end: the warm end is the exception and gets the dark steps, and the reverse
    dark_is_low = float(np.median(T)) > 0.5 * (lo + hi)
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP[::-1] if dark_is_low else RAMP)
    bar = dict(title="Temperature (K)", color=INK, title_font_size=22, label_font_size=18, n_labels=5, fmt="%.0f",
               vertical=True, position_x=0.88, position_y=0.2, height=0.6, width=0.05)
    box = np.asarray(grid.bounds, dtype=float).reshape(3, 2)
    normal = np.zeros(3)
    normal[int(np.argmin(box[:, 1] - box[:, 0]))] = 1.0
    views = ((surface_png, grid.extract_surface(), None),
             (section_png, grid.clip(normal=normal.tolist(), origin=box.mean(axis=1).tolist()).extract_surface(), normal))
    for path, surf, n in views:
        pl = pv.Plotter(off_screen=True, window_size=(1500, 1100))
        pl.set_background(SURFACE)
        pl.add_mesh(surf, scalars="T", cmap=cmap, clim=(lo, hi), scalar_bar_args=bar, smooth_shading=False)
        if n is None:
            pl.view_isometric()
        else:
            pl.view_vector((-n).tolist(), viewup=[0.0, 0.0, 1.0] if n[2] == 0 else [0.0, 1.0, 0.0])
        pl.camera.zoom(1.15)
        pl.screenshot(path)
        pl.close()


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "render":
        render(*sys.argv[2:6])
    else:
        raise SystemExit("usage: thermal.py render MESH.msh FIELD.npz SURFACE.png SECTION.png")
