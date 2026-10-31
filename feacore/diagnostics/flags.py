"""Flag / report types and every threshold the detectors use (one place to tune them).

Severity:
  fail  the solution must not be used (numerically broken, unconstrained, invalid setup)
  warn  the solution is usable but a number in it is suspect (singular peak, under-resolved, locking)
  info  worth telling the agent / improver, no effect on validity
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

THRESHOLDS = {
    # solve integrity
    "residual_fail": 1e-6,           # ||K u - f||_free / ||f||
    "equilibrium_fail": 1e-6,        # |sum applied + sum reactions| / sum |applied|
    "energy_balance_fail": 1e-6,     # |0.5 u.K.u - 0.5 (f + r).u| / 0.5 u.K.u
    # linear-theory validity
    "disp_ratio_fail": 0.05,         # max |u| / bounding-box diagonal
    "strain_fail": 0.05,             # max equivalent strain (units mismatch / mechanism)
    "strain_warn": 0.01,
    # mesh
    "quality_warn": 0.10,            # min mean-ratio shape quality (1 = regular simplex)
    "quality_fail": 0.01,
    "jacobian_fail": 0.0,            # min det J at quadrature points
    # discretisation error (ZZ estimator, relative energy norm)
    "zz_warn": 0.10,
    "zz_fail": 0.30,
    "zz_local_hotspot": 0.30,        # element-local relative error marking a hotspot
    # stress peaks
    "peak_ratio_warn": 2.0,          # nodal max / 99th percentile of von Mises
    "peak_near_bc_h": 2.0,           # peak within this many local h of a support / load edge
    # setup
    "locking_nu": 0.45,              # P1 with nu above this locks volumetrically
    "p_sensitivity_warn": 0.05,      # |U(P2) - U(P1)| / U(P2) on the same mesh
    "overconstraint_ratio": 1.1,     # sum over groups, components |R| / sum over components |F| (1 = none)
    # accuracy scoring
    "over_resolved_factor": 0.1,     # est. error below tol * this -> wasted DOFs
}


@dataclass
class Flag:
    id: str                          # stable id, see docs/failure-modes.md
    category: str                    # setup | solve | validity | mesh | discretization | singularity | design | cost
    severity: str                    # fail | warn | info
    message: str
    value: float | None = None
    threshold: float | None = None
    where: list[float] | None = None  # location (coordinates) if the flag is local

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    status: str = "ok"                       # ok | warn | fail
    flags: list[Flag] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add(self, flag: Flag) -> None:
        self.flags.append(flag)
        rank = {"ok": 0, "info": 0, "warn": 1, "fail": 2}
        if rank[flag.severity] > rank[self.status]:
            self.status = flag.severity

    def ids(self, severity: str | None = None) -> list[str]:
        return [f.id for f in self.flags if severity is None or f.severity == severity]

    def to_dict(self) -> dict:
        return {"status": self.status, "flags": [f.to_dict() for f in self.flags], "metrics": self.metrics}
