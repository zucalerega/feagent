"""Closed-form series solutions of one-dimensional transient conduction, shared by the benchmark problems
and the tests (``tests/test_transient_heat.py``).

Sources: H.S. Carslaw & J.C. Jaeger, "Conduction of Heat in Solids", 2nd ed., Oxford (1959): the slab with
both faces held at a temperature (sec. 3.3), the slab cooled by convection (sec. 3.11, roots of
z tan z = Bi), the solid cylinder cooled by convection (sec. 7.6, roots of z J1(z) = Bi J0(z)), and the slab
with a face temperature that varies in time (Duhamel's theorem, sec. 1.14 / 3.5).
"""

from __future__ import annotations

import numpy as np


def biot_roots(bi: float, n: int = 80) -> np.ndarray:
    """The first n positive roots of z tan z = Bi (one in each interval (j pi, j pi + pi/2))."""
    from scipy.optimize import brentq

    eps = 1e-14
    return np.array([brentq(lambda z: z * np.sin(z) - bi * np.cos(z), j * np.pi + eps, j * np.pi + np.pi / 2 - eps,
                            xtol=1e-15, rtol=1e-15) for j in range(n)])


def bessel_roots(bi: float, n: int = 60) -> np.ndarray:
    """The first n positive roots of z J1(z) = Bi J0(z) (one between consecutive zeros of J1 ... J0)."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1, jn_zeros

    f = lambda z: z * j1(z) - bi * j0(z)  # noqa: E731
    a = np.concatenate([[1e-12], jn_zeros(1, n)])        # f(a_j) = -Bi J0(a_j) and f changes sign once on each span
    b = jn_zeros(0, n + 1)
    out = []
    for j in range(n):
        lo, hi = a[j], b[j]
        out.append(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15))
    return np.array(out)


def slab_quench(x, t, L: float, alpha: float, Ti: float, Ts: float, n: int = 400) -> np.ndarray:
    """Slab 0 <= x <= L at Ti, both faces held at Ts from t = 0 (t > 0)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    j = np.arange(n).reshape((-1,) + (1,) * x.ndim)
    lam = (2 * j + 1) * np.pi / L
    s = np.sum(4.0 / ((2 * j + 1) * np.pi) * np.sin(lam * x) * np.exp(-alpha * lam ** 2 * t), axis=0)
    return Ts + (Ti - Ts) * s


def slab_convection(x, t, L: float, alpha: float, bi: float, Ti: float, Tf: float, n: int = 80) -> np.ndarray:
    """Slab of half-thickness L (x = 0 the insulated mid-plane, x = L the cooled face, Bi = h L / k) at Ti,
    the fluid at Tf from t = 0."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    z = biot_roots(bi, n).reshape((-1,) + (1,) * x.ndim)
    C = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    return Tf + (Ti - Tf) * np.sum(C * np.exp(-z ** 2 * alpha * t / L ** 2) * np.cos(z * x / L), axis=0)


def cylinder_convection(r, t, R: float, alpha: float, bi: float, Ti: float, Tf: float, n: int = 60) -> np.ndarray:
    """Long solid cylinder of radius R (Bi = h R / k) at Ti, the fluid at Tf from t = 0."""
    from scipy.special import j0, j1

    r, t = np.broadcast_arrays(np.asarray(r, float), np.asarray(t, float))
    z = bessel_roots(bi, n).reshape((-1,) + (1,) * r.ndim)
    C = 2.0 * j1(z) / (z * (j0(z) ** 2 + j1(z) ** 2))
    return Tf + (Ti - Tf) * np.sum(C * np.exp(-z ** 2 * alpha * t / R ** 2) * j0(z * r / R), axis=0)


def slab_face_history(x: float, t: float, L: float, alpha: float, g, dg, n: int = 300) -> float:
    """Slab 0 <= x <= L at 0, T(0, t) = 0, T(L, t) = g(t) with g(0) = 0 (dg = g'): T = x/L g(t) + sum_n
    b_n(t) sin(n pi x / L), b_n(t) = -c_n int_0^t exp(-alpha l_n^2 (t - s)) g'(s) ds, c_n = 2 (-1)^(n+1)/(n pi)."""
    from scipy.integrate import quad

    s = x / L * g(t)
    for k in range(1, n + 1):
        lam = k * np.pi / L
        c = 2.0 * (-1) ** (k + 1) / (k * np.pi)
        b = -c * quad(lambda u: np.exp(-alpha * lam * lam * (t - u)) * dg(u), 0.0, t, limit=400)[0]
        s += b * np.sin(lam * x)
    return float(s)
