"""Zienkiewicz-Zhu (1987) a-posteriori error estimate in the energy norm, from one solve.

The raw element stresses sigma_h (discontinuous between cells) are compared with the recovered,
continuous field sigma* (the backend's L2 projection). Their difference measured in the
complementary energy norm estimates the true energy-norm error:

    eta^2 = sum_e int_e (sigma* - sigma_h) : C^-1 : (sigma* - sigma_h) dx

Relative error: eta_rel = sqrt(eta^2 / (||sigma_h||^2 + eta^2)). For a Galerkin solution the
relative error in strain energy equals the squared relative energy-norm error, so
eta_rel^2 estimates |U_exact - U_h| / U_exact without any reference.
"""

from __future__ import annotations

import numpy as np

_VOIGT = ("sxx", "syy", "szz", "sxy", "syz", "sxz")


def _compliance_density(s: dict, E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """sigma : C^-1 : sigma for isotropic material, full 3D tensor (plane strain szz included)."""
    xx, yy, zz, xy, yz, xz = (s[k] for k in _VOIGT)
    return ((xx**2 + yy**2 + zz**2) - 2 * nu * (xx * yy + yy * zz + xx * zz)
            + 2 * (1 + nu) * (xy**2 + yz**2 + xz**2)) / E


def zz_estimate(res) -> dict:
    """Global and per-cell ZZ estimate for a solved ``SkfemResult``."""
    nqp = res.dx.shape[1]
    E = np.repeat(res.cell_Enu[:, :1], nqp, axis=1)
    nu = np.repeat(res.cell_Enu[:, 1:], nqp, axis=1)
    star = {k: np.asarray(res.sbasis.interpolate(res.S[:, i])) for i, k in enumerate(_VOIGT)}
    diff = {k: star[k] - res.sig_qp[k] for k in _VOIGT}
    eta2_e = (_compliance_density(diff, E, nu) * res.dx).sum(axis=1)
    en2_e = (_compliance_density(res.sig_qp, E, nu) * res.dx).sum(axis=1)
    eta2, en2 = float(eta2_e.sum()), float(en2_e.sum())
    rel = float(np.sqrt(eta2 / (en2 + eta2))) if en2 + eta2 > 0 else 0.0
    local = np.sqrt(eta2_e / np.maximum(en2_e + eta2_e, 1e-300))
    order = np.argsort(eta2_e)[::-1]
    top = max(1, len(order) // 100)
    centroids = res.md.mesh.p[:, res.md.mesh.t].mean(axis=1).T
    return {
        "eta_rel": rel,                              # relative energy-norm error estimate
        "energy_error_est": rel**2,                  # relative strain-energy error estimate
        "eta2_cell": eta2_e,
        "local_rel": local,                          # per-cell relative error
        "concentration_top1pct": float(eta2_e[order[:top]].sum() / eta2) if eta2 > 0 else 0.0,
        "worst_cell": int(order[0]),
        "worst_cell_at": centroids[order[0]].tolist(),
    }
