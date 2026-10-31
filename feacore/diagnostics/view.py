"""Backend-facing adapter: what the detectors read from the fixed (skfem) solver.

``SolutionView.setup(case)`` assembles without solving (prechecks run on setups the solver would
reject); ``SolutionView.from_result(res)`` wraps a solved ``SkfemResult``. Vector dofs follow the
backend's interleaved numbering, dof = ncomp * scalar_dof + component.
"""

from __future__ import annotations

import numpy as np

from feacore.case import Case

_VOIGT = ("sxx", "syy", "szz", "sxy", "syz", "sxz")


class SolutionView:
    def __init__(self, case: Case, model, K, f, D, xD, result=None):
        self.case, self.model, self.result = case, model, result
        self.md = model.md
        self.dim = model.dim
        self.axi = model.axi
        self.sbasis = model.sb
        self.K, self.f, self.D, self.xD = K, f, D, xD
        nc = model.ncomp
        self.N = model.N
        self.comp = list(model.split)
        self.comp_of = np.arange(self.N) % nc
        self.doflocs = np.repeat(model.sb.doflocs, nc, axis=1)
        sdofs = model.sb.element_dofs
        self.element_dofs = (sdofs[None] * nc + np.arange(nc)[:, None, None]).reshape(-1, sdofs.shape[1])
        self.cell_Enu = np.column_stack([model.E, model.nu])
        self.detJ = np.asarray(model.sb.mapping.detDF(model.sb.X))
        self.cell_measure = np.abs(model.sb.dx).sum(axis=1)          # geometric, no thickness / 2 pi r
        self.dx = model.sb.dx * model.wfac(model.xq)                 # physical weights
        if result is not None:
            self._from_solution(result)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def setup(cls, case: Case, **options) -> "SolutionView":
        from feacore.solver import skfem_backend as sk

        case.check()
        opts = {**(case.analysis.options or {}), **options}
        md = sk._Model(case, opts)
        if md.physics != "elasticity":
            raise NotImplementedError("diagnostics cover elasticity only")
        K, f, cons = sk._assemble_elasticity(md)
        D, xD = cons.finalize(md.warnings)
        return cls(case, md, K, f, D, xD)

    @classmethod
    def from_result(cls, res) -> "SolutionView":
        md = res.model
        D = res.dirichlet_dofs
        return cls(res.case, md, res.K, res.f, D, res.x[D], result=res)

    # ---------------------------------------------------------------- solved state
    def _from_solution(self, res):
        from feacore.solver.skfem_backend import qp_stress_elasticity

        self.info = res.info
        self.u = res.x
        self.R = res.reaction_vector
        sig, eps = qp_stress_elasticity(self.model, self.u)
        zero = np.zeros_like(next(iter(sig.values())))
        if self.axi:                                  # hoop stress takes the out-of-plane slot
            sig = {**sig, "szz": sig["stt"]}
        self.sig_qp = {k: sig.get(k, zero) for k in _VOIGT}
        self.eps_qp = eps
        fields = res.fields
        self.S = np.column_stack([fields["stt"] if (self.axi and k == "szz") else fields[k] for k in _VOIGT])
        free = np.setdiff1d(np.arange(self.N), self.D)
        rhs = (self.f - self.K[:, self.D] @ self.xD)[free]
        r = self.K[free][:, free] @ self.u[free] - rhs
        self.residual = float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1e-300))

    def nodal(self, field: str):
        return self.result.nodal(field)

    def strain_energy(self) -> float:
        return float(0.5 * self.u @ (self.K @ self.u))

    def group_dofs(self, group: str) -> np.ndarray:
        s = self.model.sdofs_of(group)
        nc = self.model.ncomp
        return (s[:, None] * nc + np.arange(nc)).ravel()
