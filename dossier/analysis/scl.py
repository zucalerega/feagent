"""Section stresses along a stress classification line (SCL).

An SCL is a straight line through a wall, from one surface to the other. The stress along it is
replaced by its statically equivalent linear distribution:

    membrane tensor   sigma_m = (1 / t) * integral over the line of sigma ds          (the mean)
    bending tensor    sigma_b = (6 / t^2) * integral of sigma * (t / 2 - s) ds        (the linear part)
    linear stress     sigma_lin(s) = sigma_m + sigma_b * (1 - 2 s / t)

`sigma_b` is the bending tensor at the start of the line (s = 0). At the end (s = t) it is `-sigma_b`.
All six components are treated the same way, then the invariant of the TENSOR is taken (the contract
of dossier/CONTRACTS.md):

    <invariant>.membrane           the invariant of sigma_m
    <invariant>.membrane_bending   the larger invariant of sigma_m + sigma_b and sigma_m - sigma_b
    sxx | syy | szz.membrane       the component of sigma_m (global axes, with its sign)

Integration: the line is cut into `panels` equal parts with a Gauss-Legendre rule of `order` points on
each part (default 16 x 4 = 64 points). The rule is exact for a polynomial of degree 2 * order - 1 on
each part, so a linear, quadratic or cubic stress field gives the exact tensors. The sample points are
inside the line; no point is on a surface.

Limits (say them in a report that uses these numbers):
  * Every component is linearized. ASME VIII-2 Annex 5-A recommends the bending part only for the
    stress components in the plane of the wall (hoop and meridional); the through-thickness and shear
    components have no bending part there. The two methods give different membrane-plus-bending
    invariants where the through-thickness or the shear stress changes along the line (a thick wall
    under pressure). This module follows the dossier contract; it is not an ASME linearization.
  * The line must stay in the material and be normal to the wall mid-surface. This module does not
    check that; the structural engine checks that the sample points are in the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

COMPONENTS = ("sxx", "syy", "szz", "sxy", "syz", "sxz")
INVARIANTS = ("von_mises", "max_principal", "min_principal", "tresca")
PANELS = 16
ORDER = 4


def sample_points(start, end, panels: int = PANELS, order: int = ORDER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (n, 3), s (n,) from 0 at `start` to 1 at `end`, weights (n,) that sum to 1)."""
    if panels < 1 or order < 1:
        raise ValueError("an SCL needs at least one panel and one point")
    g, wg = np.polynomial.legendre.leggauss(order)          # on [-1, 1]
    edges = np.linspace(0.0, 1.0, panels + 1)
    s = (edges[:-1, None] + (g[None, :] + 1.0) * 0.5 * np.diff(edges)[:, None]).ravel()
    w = (0.5 * wg[None, :] * np.diff(edges)[:, None]).ravel()
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return a[None, :] + s[:, None] * (b - a)[None, :], s, w


def tensor(v) -> np.ndarray:
    """(..., 6) stress components sxx, syy, szz, sxy, syz, sxz -> (..., 3, 3) symmetric tensors."""
    v = np.asarray(v, dtype=float)
    xx, yy, zz, xy, yz, xz = (v[..., i] for i in range(6))
    return np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, zz], -1)], -2)


def von_mises(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    xx, yy, zz, xy, yz, xz = (v[..., i] for i in range(6))
    return np.sqrt(np.maximum(0.0, 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                              + 3.0 * (xy ** 2 + yz ** 2 + xz ** 2)))


def principals(v) -> np.ndarray:
    """(..., 3) principal stresses, the largest first."""
    return np.linalg.eigvalsh(tensor(v))[..., ::-1]


def invariant(name: str, v) -> np.ndarray:
    """One invariant of stress states (..., 6): von_mises, max_principal, min_principal, tresca."""
    if name == "von_mises":
        return von_mises(v)
    p = principals(v)
    if name == "max_principal":
        return p[..., 0]
    if name == "min_principal":
        return p[..., 2]
    if name == "tresca":
        return p[..., 0] - p[..., 2]
    raise KeyError(f"unknown invariant {name!r}; invariants: {', '.join(INVARIANTS)}")


def linearize(s, w, sigma) -> tuple[np.ndarray, np.ndarray]:
    """(membrane (6,), bending at the start (6,)) of stresses `sigma` (n, 6) at the normalized positions
    `s` (n,) with the weights `w` (n,) of `sample_points`."""
    s, w, sigma = np.asarray(s, float), np.asarray(w, float), np.asarray(sigma, float)
    if sigma.shape != (len(s), 6) or w.shape != s.shape:
        raise ValueError(f"need n positions, n weights and (n, 6) stresses; got {s.shape}, {w.shape}, {sigma.shape}")
    membrane = w @ sigma
    bending = 6.0 * ((w * (0.5 - s)) @ sigma)
    return membrane, bending


@dataclass
class Section:
    """The linearized stress of one line. Tensors are (6,) = sxx, syy, szz, sxy, syz, sxz."""

    start: np.ndarray
    end: np.ndarray
    membrane: np.ndarray
    bending: np.ndarray                 # at the start; the bending tensor at the end is -bending
    s: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    sigma: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 6)))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def at_start(self) -> np.ndarray:
        return self.membrane + self.bending

    def at_end(self) -> np.ndarray:
        return self.membrane - self.bending

    def value(self, quantity: str, measure: str) -> float:
        """One result of the section. quantity: an invariant or a component (sxx ... sxz); measure:
        membrane or membrane_bending. A component with membrane_bending gives the end value that is
        larger in magnitude, with its sign."""
        if measure not in ("membrane", "membrane_bending"):
            raise KeyError(f"an SCL has the measures membrane and membrane_bending, not {measure!r}")
        if quantity in COMPONENTS:
            i = COMPONENTS.index(quantity)
            if measure == "membrane":
                return float(self.membrane[i])
            a, b = float(self.at_start()[i]), float(self.at_end()[i])
            return a if abs(a) >= abs(b) else b
        if measure == "membrane":
            return float(invariant(quantity, self.membrane))
        return float(max(invariant(quantity, self.at_start()), invariant(quantity, self.at_end())))

    def governing_end(self, quantity: str) -> str:
        """"start" or "end": where the membrane-plus-bending invariant is larger."""
        return "start" if invariant(quantity, self.at_start()) >= invariant(quantity, self.at_end()) else "end"

    def peak_ratio(self) -> float:
        """The largest von Mises stress of the samples divided by the larger linearized von Mises stress at
        the ends: above 1 the line holds a local peak that the linear distribution leaves out."""
        lin = max(float(von_mises(self.at_start())), float(von_mises(self.at_end())), 1e-300)
        return float(von_mises(self.sigma).max() / lin) if len(self.sigma) else float("nan")


def section(sample: Callable[[np.ndarray], np.ndarray], start, end, panels: int = PANELS,
            order: int = ORDER) -> Section:
    """Linearize the stress along the line from `start` to `end`. `sample(points (n, 3))` returns the six
    stress components (n, 6) at the points."""
    pts, s, w = sample_points(start, end, panels, order)
    sigma = np.asarray(sample(pts), dtype=float)
    if sigma.shape != (len(pts), 6):
        raise ValueError(f"the sampler must return (n, 6) stresses for (n, 3) points; got {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("the sampler returned a stress that is not a number")
    m, b = linearize(s, w, sigma)
    return Section(np.asarray(start, float), np.asarray(end, float), m, b, s, sigma)
