"""Solution verification: Richardson extrapolation and the Grid Convergence Index.

Three-mesh procedure of Celik et al. (2008), "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. 130(7) 078001,
which follows Roache (1994/1998). Non-uniform refinement ratios are handled by the
fixed-point iteration for the observed order p. For four or more meshes a least-squares
fit phi = phi0 + a h^p is also reported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class GCIResult:
    h: list[float]                 # fine, medium, coarse (h1 < h2 < h3)
    phi: list[float]               # matching solutions
    r21: float
    r32: float
    R: float | None                # convergence ratio eps21 / eps32
    convergence: str               # monotone | oscillatory | divergent | oscillatory_divergent | exact | converged | indeterminate
    p: float | None                # observed order
    p_theory: float | None
    phi_ext: float | None          # Richardson-extrapolated value
    e_a21: float                   # relative change fine vs medium
    e_ext21: float | None          # relative difference fine vs extrapolated
    gci_fine: float | None         # relative uncertainty of the fine solution (fraction), Fs = 1.25
    gci_medium: float | None
    gci_fine_conservative: float | None  # Fs = 3 with p limited to [0.5, p_theory] (Roache)
    asymptotic_ratio: float | None  # gci_medium / (r21^p gci_fine), ~1 in the asymptotic range
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _observed_order(e21: float, e32: float, r21: float, r32: float, s: float,
                    max_iter: int = 500, tol: float = 1e-12) -> float | None:
    ratio = abs(e32 / e21)
    p = abs(math.log(ratio)) / math.log(r21)
    for _ in range(max_iter):
        num, den = r21**p - s, r32**p - s
        if num <= 0 or den <= 0:
            return None
        p_new = abs(math.log(ratio) + math.log(num / den)) / math.log(r21)
        if abs(p_new - p) < tol * max(1.0, p):
            return p_new
        p = 0.5 * (p + p_new) if p_new > 20 else p_new
    return p


def gci3(h, phi, p_theory: float | None = None, safety: float = 1.25,
         exact_tol: float = 1e-12, noise_floor: float = 1e-6) -> GCIResult:
    """Celik/Roache three-mesh analysis. ``h`` and ``phi`` in any order (sorted by h).

    ``noise_floor``: if both mesh-to-mesh changes are below this fraction of |phi|, the QoI is
    reported as ``converged`` (uncertainty = the change itself) instead of computing an order from
    round-off or stress-recovery noise, which would produce spurious divergent / oscillatory flags."""
    h = np.asarray(h, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if h.shape != (3,) or phi.shape != (3,):
        raise ValueError("gci3 needs exactly three meshes")
    idx = np.argsort(h)
    (h1, h2, h3), (f1, f2, f3) = h[idx], phi[idx]
    if not (h1 < h2 < h3):
        raise ValueError(f"mesh sizes must be distinct, got {h[idx]}")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    scale = max(abs(f1), abs(f2), abs(f3), 1e-300)
    flags: list[str] = []
    if min(r21, r32) < 1.3:
        flags.append("refinement_ratio_below_1.3")
    if abs(f1) < 1e-8 * scale:
        flags.append("phi_near_zero")
    e_a21 = abs(e21 / f1) if f1 != 0 else math.inf

    def result(**kw):
        base = dict(h=[h1, h2, h3], phi=[f1, f2, f3], r21=r21, r32=r32, p_theory=p_theory,
                    e_a21=e_a21, flags=flags, R=None, p=None, phi_ext=None, e_ext21=None,
                    gci_fine=None, gci_medium=None, gci_fine_conservative=None,
                    asymptotic_ratio=None)
        base.update(kw)
        return GCIResult(**base)

    if abs(e21) <= exact_tol * scale and abs(e32) <= exact_tol * scale:
        return result(R=0.0, convergence="exact", phi_ext=f1, e_ext21=0.0, gci_fine=0.0,
                      gci_medium=0.0, gci_fine_conservative=0.0)
    if max(abs(e21), abs(e32)) <= noise_floor * scale:
        flags.append("converged_within_noise")
        unc = safety * max(abs(e21), abs(e32)) / max(abs(f1), 1e-300)
        return result(R=e21 / e32 if e32 != 0 else 0.0, convergence="converged", phi_ext=f1,
                      e_ext21=0.0, gci_fine=unc, gci_medium=unc, gci_fine_conservative=unc)
    if abs(e32) <= exact_tol * scale or abs(e21) <= exact_tol * scale:
        flags.append("two_solutions_identical")
        return result(convergence="indeterminate")

    R = e21 / e32
    if 0 < R < 1:
        convergence = "monotone"
    elif -1 < R < 0:
        convergence = "oscillatory"
        flags.append("oscillatory")
    elif R >= 1:
        convergence = "divergent"
        flags.append("divergent")
    else:
        convergence = "oscillatory_divergent"
        flags.append("divergent")
    if convergence in ("divergent", "oscillatory_divergent"):
        return result(R=R, convergence=convergence)

    s = 1.0 if R > 0 else -1.0
    p = _observed_order(e21, e32, r21, r32, s)
    if p is None or p <= 0:
        flags.append("order_undefined")
        return result(R=R, convergence=convergence)

    phi_ext = (r21**p * f1 - f2) / (r21**p - 1)
    e_ext21 = abs((phi_ext - f1) / phi_ext) if phi_ext != 0 else math.inf
    gci_fine = safety * e_a21 / (r21**p - 1)
    e_a32 = abs(e32 / f2) if f2 != 0 else math.inf
    gci_medium = safety * e_a32 / (r32**p - 1)
    asym = gci_medium / (r21**p * gci_fine) if gci_fine > 0 else None

    p_lim = p
    if p_theory is not None:
        p_lim = min(max(p, 0.5), p_theory)
        if p < 0.5 * p_theory:
            flags.append("p_far_below_theory")
        elif p < 0.8 * p_theory:
            flags.append("p_below_theory")
        elif p > 1.5 * p_theory + 0.5:
            flags.append("p_above_theory")
    gci_cons = 3.0 * e_a21 / (r21**p_lim - 1)
    if asym is not None and abs(asym - 1.0) > 0.2:
        flags.append("not_asymptotic")
    return result(R=R, convergence=convergence, p=p, phi_ext=phi_ext, e_ext21=e_ext21,
                  gci_fine=gci_fine, gci_medium=gci_medium, gci_fine_conservative=gci_cons,
                  asymptotic_ratio=asym)


def fit_power_law(h, phi, p_bounds=(0.05, 8.0)) -> dict | None:
    """Least-squares fit phi = phi0 + a h^p over >= 4 meshes (Eca & Hoekstra style)."""
    from scipy.optimize import least_squares

    h = np.asarray(h, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if h.size < 4:
        return None
    hs = h / h.min()
    sc = max(np.abs(phi).max(), 1e-300)

    def resid(x):
        phi0, a, p = x
        return (phi0 + a * hs**p - phi) / sc

    best = None
    for p0 in (0.5, 1.0, 2.0, 3.0):
        A = np.vstack([np.ones_like(hs), hs**p0]).T
        (phi0, a), *_ = np.linalg.lstsq(A, phi, rcond=None)
        sol = least_squares(resid, [phi0, a, p0],
                            bounds=([-np.inf, -np.inf, p_bounds[0]], [np.inf, np.inf, p_bounds[1]]))
        if best is None or sol.cost < best.cost:
            best = sol
    phi0, a, p = best.x
    res = resid(best.x) * sc
    return {"phi0": float(phi0), "a": float(a / h.min() ** p), "p": float(p),
            "rms_residual": float(np.sqrt(np.mean(res**2))), "n": int(h.size)}


def analyze(h, phi, p_theory: float | None = None) -> dict:
    """GCI on the three finest meshes plus a power-law fit when >= 4 meshes are given."""
    h = np.asarray(h, dtype=float)
    phi = np.asarray(phi, dtype=float)
    idx = np.argsort(h)
    out = {"gci": gci3(h[idx[:3]], phi[idx[:3]], p_theory=p_theory).to_dict()}
    out["fit"] = fit_power_law(h, phi)
    return out
