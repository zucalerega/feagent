"""Backend-independent result interface. Every solver backend returns a ``Result``."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from feacore.case import Case, QoI


@dataclass
class SolveInfo:
    backend: str             # "skfem" | "ccx"
    physics: str
    formulation: str
    order: int               # FE interpolation order actually used
    dim: int                 # mesh dimension (2 or 3)
    cell_type: str           # mesh cell type, e.g. "triangle6", "tetra10"
    n_nodes: int             # mesh nodes
    n_cells: int             # mesh cells of dimension ``dim``
    n_dofs: int              # unknowns before Dirichlet elimination
    measure: float           # area (2D) / volume (3D) of the mesh, no thickness / 2*pi factors
    h_rep: float             # representative cell size (measure / n_cells) ** (1 / dim)
    wall_time: float         # seconds for assembly + solve + recovery
    warnings: list[str] = field(default_factory=list)


class Result(ABC):
    """Solution of one case. Field names are listed in ``feacore.case``."""

    info: SolveInfo
    case: Case

    @abstractmethod
    def probe(self, field: str, points) -> np.ndarray:
        """Values of ``field`` at points (n, dim) or (n, 3). Stresses come from the recovered
        continuous field. Points marginally outside the discrete domain (curved boundaries)
        are evaluated in the nearest cell."""

    @abstractmethod
    def nodal(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """(coordinates (n, dim), values (n,)) of ``field`` at the nodes of the solution space."""

    @abstractmethod
    def evaluate(self, qoi: QoI) -> float:
        """Evaluate one quantity of interest (see the QoI table in ``feacore.case``)."""

    def evaluate_all(self) -> dict[str, float]:
        return {q.name: float(self.evaluate_qoi(q)) for q in self.case.qois}

    def evaluate_qoi(self, qoi: QoI) -> float:
        """``evaluate`` plus the backend-independent QoI types (``point_difference``, computed
        from ``probe``). Prefer this over ``evaluate`` in callers."""
        if qoi.type == "point_difference":
            d = self.info.dim
            v = self.probe(qoi.field, np.array([qoi.point[:d], qoi.point2[:d]], dtype=float))
            return float(v[0] - v[1])
        return self.evaluate(qoi)

    def write_vtu(self, path: str | Path) -> Path:  # optional for backends
        raise NotImplementedError

    def summary(self) -> dict:
        qois = self.evaluate_all()  # first: QoI evaluation may append to info.warnings
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta}

    def save_summary(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))
        return path


# --------------------------------------------------------------------------- time-dependent and modal results


def dominant_frequency(times, values) -> float:
    """The dominant frequency (Hz) of a sampled, nearly periodic signal: the mean is removed and the
    frequency is one over the mean time between upward zero crossings (crossing times are interpolated
    linearly between samples). Needs at least two full periods; raises ValueError otherwise. For a signal
    with several strong frequencies this is the mean crossing rate, not a spectral peak."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size < 8 or t.size != y.size:
        raise ValueError("dominant_frequency needs at least 8 samples")
    y = y - np.trapezoid(y, t) / (t[-1] - t[0])
    up = np.nonzero((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    if up.size < 3:
        raise ValueError(f"dominant_frequency needs at least two full periods in the window "
                         f"({max(up.size - 1, 0)} found)")
    tc = t[up] - y[up] * (t[up + 1] - t[up]) / (y[up + 1] - y[up])
    return float((tc.size - 1) / (tc[-1] - tc[0]))


def reduce_series(times, values, how=None, window=None) -> float:
    """One number from a time series, as ``QoI.time`` asks: None = the last value; a number = the value at
    that time (linear between steps); "max", "min", "max_abs"; "mean" and "rms" (time-weighted, of the value
    itself); "frequency" (``dominant_frequency``). ``window`` = [t0, t1] restricts the series first."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.size == 0:
        raise ValueError("reduce_series: times and values must be 1D arrays of the same, non-zero length")
    if window is not None:
        eps = 1e-12 * max(abs(float(t[-1])), 1.0)
        keep = (t >= window[0] - eps) & (t <= window[1] + eps)
        if not keep.any():
            raise ValueError(f"reduce_series: no time step in the window {list(window)} "
                             f"(steps from {t[0]:g} to {t[-1]:g})")
        t, y = t[keep], y[keep]
    if how is None:
        return float(y[-1])
    if isinstance(how, (int, float)) and not isinstance(how, bool):
        if not t[0] - 1e-12 <= how <= t[-1] * (1 + 1e-12) + 1e-12:
            raise ValueError(f"reduce_series: time {how:g} is outside the series ({t[0]:g} to {t[-1]:g})")
        return float(np.interp(float(how), t, y))
    if how == "max":
        return float(y.max())
    if how == "min":
        return float(y.min())
    if how == "max_abs":
        return float(np.abs(y).max())
    if how in ("mean", "rms"):
        if t.size < 2:
            return float(y[0]) if how == "mean" else float(abs(y[0]))
        span = t[-1] - t[0]
        return float(np.trapezoid(y, t) / span) if how == "mean" else float(np.sqrt(np.trapezoid(y * y, t) / span))
    if how == "frequency":
        return dominant_frequency(t, y)
    raise ValueError(f"reduce_series: unknown reduction {how!r}")


class TransientResult(Result):
    """A time-dependent solution (``analysis.options["transient"]``).

    ``times``   every step, from 0 to t_end
    ``kept``    the indices into ``times`` of the steps whose full state is kept
    Subclasses give ``series(qoi)`` (the QoI at every step, evaluated while the solver runs) and
    ``state(t)`` (the kept state nearest to t, a ``Result`` with probe / nodal / evaluate).
    ``evaluate`` reduces the series as ``QoI.time`` and ``QoI.window`` ask; ``probe`` and ``nodal``
    are those of the final state."""

    times: np.ndarray
    kept: np.ndarray

    @abstractmethod
    def series(self, qoi: QoI) -> np.ndarray:
        """The QoI at every step (the same length as ``times``), ignoring ``qoi.time`` and ``qoi.window``."""

    @abstractmethod
    def state(self, t: float | None = None) -> Result:
        """The kept state nearest to time t (default: the last one)."""

    def history(self, qoi: QoI) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.times, dtype=float), np.asarray(self.series(qoi), dtype=float)

    def evaluate(self, qoi: QoI) -> float:
        t, y = self.history(qoi)
        return reduce_series(t, y, qoi.time, qoi.window)

    def evaluate_qoi(self, qoi: QoI) -> float:
        return self.evaluate(qoi)

    def probe(self, field: str, points) -> np.ndarray:
        return self.state().probe(field, points)

    def nodal(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        return self.state().nodal(field)


class ModalResult(Result):
    """Natural frequencies and mode shapes (``analysis.options["modal"]``).

    ``frequencies``     Hz, the elastic modes in ascending order (rigid-body modes are not counted)
    ``rigid_modes``     how many rigid-body modes the model has (0 for a restrained model)
    ``effective_mass``  (n_modes, dim): the share of the total mass each mode moves along each axis
    Subclasses give ``shape(mode)``: (coordinates (n, dim), displacements (n, dim)) of a mass-normalised
    mode. ``nodal`` and ``probe`` are those of mode 1 unless the subclass takes a mode."""

    frequencies: np.ndarray
    rigid_modes: int
    effective_mass: np.ndarray

    @abstractmethod
    def shape(self, mode: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(coordinates (n, dim), displacements (n, dim)) of mode ``mode`` (1 = lowest elastic mode)."""

    def evaluate(self, qoi: QoI) -> float:
        n = len(self.frequencies)
        if qoi.type not in ("natural_frequency", "effective_mass"):
            raise ValueError(f"a modal result gives natural_frequency and effective_mass, not {qoi.type!r}")
        if qoi.mode is None or not 1 <= qoi.mode <= n:
            raise ValueError(f"QoI {qoi.name!r}: mode {qoi.mode} is not one of the {n} modes that were computed")
        if qoi.type == "natural_frequency":
            return float(self.frequencies[qoi.mode - 1])
        return float(self.effective_mass[qoi.mode - 1, qoi.component])
