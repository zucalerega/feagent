"""Compressible internal flow on the ``rhoCentralFoam`` path (private helper of
feacore/solver/_foam_compressible.py): the reservoir inlet, the pressure outlet that switches between an
imposed back pressure and extrapolation, the reference state of a run without a freestream, and the mass
balance. docs/solver_transient.md, "Compressible internal flow".

Why these boundary conditions (the solver is density based: its boundary fluxes are the fluxes of the
boundary values of p, T and U, so every boundary value must be a physical state of the gas):

* ``total_inlet`` (a subsonic inflow has dim + 1 incoming characteristics and one outgoing): the total
  pressure, the total temperature and the flow direction are given; the speed comes from the interior.
  OpenFOAM: U ``pressureInletOutletVelocity`` (the normal velocity extrapolated, no tangential velocity where
  the gas enters), p ``totalPressure`` and T ``totalTemperature`` with the compressibility ``thermo:psi`` and
  gamma, which are the isentropic relations p = p0 (1 + (gamma - 1) / 2 M^2)^(-gamma / (gamma - 1)) and
  T = T0 / (1 + (gamma - 1) / 2 M^2) with the boundary's own speed. ``pressureInletVelocity`` (the velocity
  from the flux) is circular here: the solver's boundary flux is itself made from the boundary velocity.
* ``pressure_outlet``: a subsonic outflow has one incoming characteristic (the back pressure is given),
  a supersonic one none (everything is extrapolated). OpenFOAM: p ``mixed`` with refValue = the back
  pressure and a value fraction per face of 1 (imposed) or 0 (zero gradient), U
  ``pressureInletOutletVelocity`` and T ``inletOutlet`` (zero gradient where the gas leaves). The value
  fraction is set here (``OutletSwitch``), at the end of every chunk of the run. v1912 has no stock
  condition that switches on the Mach number; ``waveTransmissive`` does not: it relaxes the pressure towards
  the far value in both regimes, so a subsonic outlet does not sit at the back pressure and a supersonic one
  still feels it.

The switch (per face). A fixed pressure on a face that the gas leaves supersonically is not a state the gas
can have: the Rankine-Hugoniot relations between the arriving state and (p_back, U, T) of the boundary hold
for two pressures only, the arriving one and the one behind a normal shock, so the last cell never settles.
The rule therefore looks at a state the boundary has not disturbed: the owner cell of a face that
extrapolates, the cell behind the owner (away from the boundary) of a face that imposes. With M_n the Mach
number of that state normal to the face and p its pressure:

    M_n < 1                                  impose the back pressure (subsonic outflow)
    M_n >= 1 and p_back >= p_ns(M_n, p)      impose it: the back pressure exceeds what a normal shock in
                                             the outlet plane reaches, p_ns = p (1 + 2 gamma / (gamma + 1)
                                             (M_n^2 - 1)); a shock forms and moves upstream
    M_n >= 1 and p_back <  p_ns              extrapolate: supersonic outflow, the back pressure cannot enter
                                             (an over- or under-expanded jet adjusts outside the domain)

A run counts as steady only when no face switched at the end of its last chunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from feacore.case import Case, CaseError, as_field
from feacore.solver import _foam_io as fio

INLET, OUTLET = "total_inlet", "pressure_outlet"
PATCH_TYPE = {INLET: "patch", OUTLET: "patch"}
LOW_MACH = 0.1        # below this (largest Mach number of the result) the density-based solver is out of range


def is_internal(case: Case) -> bool:
    """A reservoir drives the flow: the reference state comes from the first total_inlet, not a freestream."""
    return any(b.type == INLET for b in case.bcs)


def has_outlets(case: Case) -> bool:
    return any(b.type == OUTLET for b in case.bcs)


def choked_flux(p0: float, T0: float, gamma: float, R: float) -> float:
    """The largest mass flux a perfect gas from the reservoir (p0, T0) can reach, rho* a* (kg/(m^2 s))."""
    return p0 * math.sqrt(gamma / (R * T0)) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


@dataclass
class Reservoir:
    p0: float
    T0: float
    a0: float           # speed of sound of the gas at rest
    a_star: float       # critical speed of sound sqrt(2 gamma R T0 / (gamma + 1)): the velocity scale
    rho0: float
    flux: float         # choked mass flux rho* a*


def reservoir(case: Case, gamma: float, R: float, warnings: list) -> Reservoir:
    """The first total_inlet: the gas at rest in it is the initial field, and it scales the stopping rule."""
    inlets = [b for b in case.bcs if b.type == INLET]
    b = inlets[0]
    p0, T0 = float(b.p0), float(b.T0)
    if any(float(o.p0) != p0 or float(o.T0) != T0 for o in inlets[1:]):
        warnings.append(f"several total_inlet BCs with different reservoirs: {b.group!r}'s gives the initial "
                        "field and the scales of the stopping rule")
    if any(o.type == "freestream" for o in case.bcs):
        raise CaseError("a case with a total_inlet takes no freestream BC: the reservoir of the total_inlet is the "
                        "reference state (a supersonic inflow with a back pressure is freestream + pressure_outlet)")
    if not has_outlets(case):
        raise CaseError("a total_inlet needs a pressure_outlet: the run starts from the gas at rest in the "
                        "reservoir state, and only a back pressure below p0 sets it in motion (an 'outlet' "
                        "extrapolates everything; a pressure_outlet does the same wherever the gas leaves "
                        "supersonically)")
    backs = [o.value for o in case.bcs if o.type == OUTLET and not isinstance(o.value, str)]
    if backs and min(float(v) for v in backs) >= p0:
        warnings.append(f"no pressure_outlet has a back pressure below the total pressure {p0:g} Pa: nothing drives "
                        "a flow from the total_inlet to an outlet")
    return Reservoir(p0, T0, math.sqrt(gamma * R * T0), math.sqrt(2.0 * gamma * R * T0 / (gamma + 1.0)),
                     p0 / (R * T0), choked_flux(p0, T0, gamma, R))


# --------------------------------------------------------------------------- boundary dictionaries


def _uniform(v: float):
    """``uniform <v>`` for a field entry whose keyword foamlib does not know as a field (p0, T0, valueFraction)."""
    return ("uniform", float(v))


def _field(a: np.ndarray):
    a = np.asarray(a, dtype=float)
    if len(a) and np.all(a == a[0]):
        return _uniform(a[0])
    return ("nonuniform", "List<scalar>", a.tolist())


def inlet_fields(bc, gamma: float) -> tuple[dict, dict, dict]:
    """(U, p, T) boundary dictionaries of a total_inlet."""
    p0, T0 = float(bc.p0), float(bc.T0)
    U = {"type": "pressureInletOutletVelocity", "phi": "phi", "value": [0.0, 0.0, 0.0]}
    P = {"type": "totalPressure", "p0": _uniform(p0), "gamma": float(gamma), "psi": "thermo:psi", "U": "U",
         "phi": "phi", "value": p0}
    T = {"type": "totalTemperature", "T0": _uniform(T0), "gamma": float(gamma), "psi": "thermo:psi", "U": "U",
         "phi": "phi", "value": T0}
    return U, P, T


def back_pressure(bc, Cf: np.ndarray, dim: int) -> np.ndarray:
    """The back pressure of a pressure_outlet at its face centres (a number or an expression in x, y, z)."""
    v = bc.value
    if isinstance(v, str):
        pb = np.asarray(as_field(v, Cf[:, 0], Cf[:, 1], Cf[:, 2] if dim == 3 else None), dtype=float)
        pb = np.broadcast_to(pb, (len(Cf),)).copy()
    else:
        pb = np.full(len(Cf), float(v))
    if not np.all(pb > 0):
        raise CaseError(f"pressure_outlet {bc.group!r}: the back pressure is an absolute static pressure, > 0 "
                        f"(got {float(pb.min()):g} Pa)")
    return pb


def outlet_fields(p_back: np.ndarray, frac: np.ndarray, T_back: float) -> tuple[dict, dict, dict]:
    """(U, p, T) boundary dictionaries of a pressure_outlet; ``T_back`` is the temperature of gas that
    re-enters through it (the reservoir's total temperature, or the freestream's)."""
    U = {"type": "pressureInletOutletVelocity", "phi": "phi", "value": [0.0, 0.0, 0.0]}
    P = {"type": "mixed", "refValue": np.asarray(p_back, dtype=float) if np.ptp(p_back) > 0 else float(p_back[0]),
         "refGradient": 0.0, "valueFraction": _field(frac),
         "value": np.asarray(p_back, dtype=float) if np.ptp(p_back) > 0 else float(p_back[0])}
    T = {"type": "inletOutlet", "phi": "phi", "inletValue": float(T_back), "value": float(T_back)}
    return U, P, T


# --------------------------------------------------------------------------- the outlet switch


def normal_shock_pressure(mach: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """Static pressure behind a normal shock in a flow of Mach number ``mach`` (>= 1) and pressure ``p``."""
    return p * (1.0 + 2.0 * gamma / (gamma + 1.0) * (mach * mach - 1.0))


@dataclass
class _Patch:
    name: str
    group: str
    sl: slice
    own: np.ndarray         # owner cell of every face
    back: np.ndarray        # the cell behind the owner, away from the boundary (the owner where there is none)
    n: np.ndarray           # unit outward normals
    p_back: np.ndarray
    frac: np.ndarray        # 1 = the back pressure is imposed, 0 = extrapolated
    history: list = field(default_factory=list)     # faces switched at every update


class OutletSwitch:
    """The value fraction of every pressure_outlet face (module docstring, "The switch")."""

    def __init__(self, mesh, patches: dict, gamma: float, R: float):
        """``patches``: {patch name: (group, back pressure per face)}; every face starts imposed."""
        self.gamma, self.R = gamma, R
        self.patches: list[_Patch] = []
        n_int = mesh.n_int
        own_i, nei_i = mesh.owner[:n_int], mesh.neighbour
        for name, (group, p_back) in patches.items():
            sl = mesh.patch_slice(name)
            own = mesh.owner[sl]
            nrm = mesh.Sf[sl] / mesh.magSf[sl][:, None]
            back = own.copy()
            for k, c in enumerate(own):
                nb = np.concatenate([nei_i[own_i == c], own_i[nei_i == c]])
                if not len(nb):
                    continue
                d = mesh.C[nb] - mesh.C[c]
                cos = -(d @ nrm[k]) / np.maximum(np.linalg.norm(d, axis=1), 1e-300)
                j = int(np.argmax(cos))
                if cos[j] > 0.5:
                    back[k] = nb[j]
            self.patches.append(_Patch(name, group, sl, own, back, nrm, np.asarray(p_back, dtype=float),
                                       np.ones(len(own))))

    def start(self, p: np.ndarray, T: np.ndarray, U: np.ndarray) -> None:
        """The state of every face for a given initial field (``init_from``): the field is taken as
        undisturbed up to the boundary, so the owner cells decide."""
        for q in self.patches:
            q.frac = np.zeros(len(q.own))
        self.update(p, T, U)
        for q in self.patches:
            q.history.clear()

    def _state(self, cells: np.ndarray, nrm: np.ndarray, p, T, U) -> tuple[np.ndarray, np.ndarray]:
        c = np.sqrt(self.gamma * self.R * T[cells])
        return np.einsum("ij,ij->i", U[cells], nrm) / c, p[cells]

    def update(self, p: np.ndarray, T: np.ndarray, U: np.ndarray) -> int:
        """Set every face's value fraction from the cell values; returns the number of faces that switched."""
        switched = 0
        for q in self.patches:
            imposed = q.frac > 0.5
            cells = np.where(imposed, q.back, q.own)
            mn, ps = self._state(cells, q.n, p, T, U)
            hold = q.p_back >= normal_shock_pressure(np.maximum(mn, 1.0), ps, self.gamma)
            new = np.where((mn < 1.0) | hold, 1.0, 0.0)
            k = int(np.sum(new != q.frac))
            q.history.append(k)
            q.frac = new
            switched += k
        return switched

    def write(self, tdir) -> None:
        """Put the value fractions into the pressure field of the time directory ``tdir`` (the restart state)."""
        f = fio.FoamFieldFile(tdir / "p")
        for q in self.patches:
            f.boundary_field[q.name]["valueFraction"] = _field(q.frac)

    def describe(self) -> list[str]:
        out = []
        for q in self.patches:
            n, k = len(q.frac), int(np.sum(q.frac > 0.5))
            pb = f"{q.p_back[0]:g} Pa" if np.ptp(q.p_back) == 0 else f"{q.p_back.min():g} .. {q.p_back.max():g} Pa"
            if k == n:
                what = "imposed on every face (subsonic outflow)"
            elif k == 0:
                what = "not imposed: the gas leaves supersonically through every face (extrapolated)"
            else:
                what = f"imposed on {k} of {n} faces (subsonic), extrapolated on {n - k} (supersonic)"
            out.append(f"info: pressure_outlet {q.group!r}: back pressure {pb} {what}")
        return out


# --------------------------------------------------------------------------- mass balance


def mass_balance(res) -> dict:
    """Mass flow in and out through the boundary of a ``CompressibleResult`` (kg/s, both >= 0) and the
    imbalance |in - out| / max(in, out): the rate of change of the mass in the domain, zero in a steady flow."""
    m_in = m_out = 0.0
    for d in res.patch.values():
        m = res._mass_flux(d)
        m_in -= float(m[m < 0].sum())
        m_out += float(m[m > 0].sum())
    top = max(m_in, m_out)
    return {"in": m_in, "out": m_out, "imbalance": abs(m_in - m_out) / top if top > 0 else 0.0}
