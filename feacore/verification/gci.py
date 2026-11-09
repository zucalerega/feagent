"""Solution verification: Richardson extrapolation and the Grid Convergence Index.

Three-mesh procedure of Celik et al. (2008), "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. 130(7) 078001,
which follows Roache (1994/1998). For four or more meshes a least-squares fit
phi = phi0 + a h^p is also reported.

Non-uniform refinement ratios (r21 != r32, the norm for independently remeshed sequences) are
handled by Celik's equation for the observed order p with both ratios, solved by bisection
(``_observed_order``). The convergence class follows from whether a positive p exists, not from
the convergence ratio R = e21 / e32 alone: same-sign changes converge iff R < ln r21 / ln r32,
which is the familiar R < 1 only when r21 = r32.
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
    R: float | None                # convergence ratio eps21 / eps32 (monotone iff 0 < R < ln r21 / ln r32)
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


_EQUAL_RATIOS = 1e-9  # r21 and r32 closer than this (relative) count as equal: closed-form p


def _log_expm1(x: float) -> float:
    """ln(e^x - 1) for x > 0, without overflow or cancellation."""
    return x + math.log1p(-math.exp(-x)) if x > 1.0 else math.log(math.expm1(x))


def _log1p_exp(x: float) -> float:
    """ln(1 + e^x), without overflow."""
    return x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))


def _pow_m1(r: float, p: float) -> float:
    """r^p - 1, inf where r^p overflows (an absurd order, e.g. from two nearly coincident meshes)."""
    x = p * math.log(r)
    return math.expm1(x) if x < 709.0 else math.inf


def _observed_order(e21: float, e32: float, r21: float, r32: float, s: float,
                    max_iter: int = 200, tol: float = 1e-12) -> float | None:
    """Observed order p > 0 from Celik et al. (2008) eqs. (3)-(5), or None when no positive p fits.

    Without Celik's outer absolute value the equations say

        |e32 / e21| = r21^p (r32^p - s) / (r21^p - s),

    the ratio of successive changes of phi = phi_ext + C h^p (s = 1, same-sign changes) or of an
    error that alternates in sign (s = -1). The right side increases strictly with p and tends to
    ln r32 / ln r21 (s = 1) or to 1 (s = -1) as p -> 0+. So a positive root exists iff |e32 / e21|
    exceeds that limit; it is then bracketed and found by bisection. With r21 = r32 this reduces to
    p = ln|e32 / e21| / ln r21, positive iff |e21 / e32| < 1.

    Celik's fixed-point form p = |ln|e32 / e21| + q(p)| / ln r21 is not used: its absolute value
    turns the negative root of a diverging sequence into a spurious positive order, and for some
    unequal ratios the iteration itself diverges (overflow, then a math domain error).
    """
    target = math.log(abs(e32 / e21))
    if not math.isfinite(target):
        return None
    a, b = math.log(r21), math.log(r32)
    if math.isclose(r21, r32, rel_tol=_EQUAL_RATIOS):
        p = target / a
        return p if p > 0 else None
    if s > 0:
        def model(p):  # ln[r21^p (r32^p - 1) / (r21^p - 1)]
            return p * a + _log_expm1(p * b) - _log_expm1(p * a)
        at_zero = math.log(b / a)
        hi = _log1p_exp(target) / b       # model(p) > ln(r32^p - 1), which equals target here
    else:
        def model(p):  # ln[r21^p (r32^p + 1) / (r21^p + 1)]
            return p * a + _log1p_exp(p * b) - _log1p_exp(p * a)
        at_zero = 0.0
        hi = (math.log(2.0) + target) / b  # model(p) > ln((r32^p + 1) / 2) > target here
    if target <= at_zero:
        return None
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if model(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    return 0.5 * (lo + hi)


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

    # Converging iff a positive order fits both changes with both ratios: 0 < R < ln r21 / ln r32
    # for same-sign changes, -1 < R < 0 for alternating ones. With unequal ratios, R < 1 alone
    # calls some converging sequences divergent (r21 > r32) and some stalling ones monotone.
    R = e21 / e32
    s = 1.0 if R > 0 else -1.0
    p = _observed_order(e21, e32, r21, r32, s)
    if p is None:
        flags.append("divergent")
        return result(R=R, convergence="divergent" if s > 0 else "oscillatory_divergent")
    convergence = "monotone" if s > 0 else "oscillatory"
    if s < 0:
        flags.append("oscillatory")

    d21 = _pow_m1(r21, p)
    phi_ext = f1 - e21 / d21  # = (r21^p f1 - f2) / (r21^p - 1)
    e_ext21 = abs((phi_ext - f1) / phi_ext) if phi_ext != 0 else math.inf
    gci_fine = safety * e_a21 / d21
    e_a32 = abs(e32 / f2) if f2 != 0 else math.inf
    gci_medium = safety * e_a32 / _pow_m1(r32, p)
    asym = gci_medium / ((d21 + 1.0) * gci_fine) if gci_fine > 0 else None

    p_lim = p
    if p_theory is not None:
        p_lim = min(max(p, 0.5), p_theory)
        if p < 0.5 * p_theory:
            flags.append("p_far_below_theory")
        elif p < 0.8 * p_theory:
            flags.append("p_below_theory")
        elif p > 1.5 * p_theory + 0.5:
            flags.append("p_above_theory")
    gci_cons = 3.0 * e_a21 / _pow_m1(r21, p_lim)
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
