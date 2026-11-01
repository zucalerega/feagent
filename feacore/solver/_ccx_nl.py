"""Opt-in elastoplastic, load-stepped static analysis with CalculiX (``analysis.options.nonlinear``).

The deck (``ccx_backend._Deck``) writes ``*PLASTIC`` (J2, isotropic hardening), a ``*STATIC, DIRECT``
step with fixed equal increments (``*STEP, NLGEOM`` on request) and every load / prescribed value
x ``load_factor_max``, ramped linearly over the unit step time; ``U, RF`` / ``S, PEEQ`` are written to
the .frd and the integration-point PEEQ plus the reaction totals to the .dat at every increment.
This module parses every increment and gives:

* ``CcxNonlinearResult``: a ``CcxResult`` positioned on the last converged increment (so every QoI
  of ``feacore.case`` works), with ``increments`` (one ``Increment`` per output increment),
  ``use_increment(i)`` to evaluate QoIs at another one, ``increment_qois(i)`` (max von Mises,
  max PEEQ, plastic volume fraction, reactions vs applied load) and ``write_series(dir)``
  (one .vtu per increment + a .pvd series + ``summary.json``).
* ``converged`` / ``errors``: a fixed-increment run that diverges keeps its converged increments.

See ``docs/solver_ccx_nonlinear.md``.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feacore.solver._ccx_io import CcxError, parse_dat_series, parse_frd_series, parse_sta, run_ccx
from feacore.solver.ccx_backend import CcxResult, axi_radial_factor

# equivalent plastic strain above which an integration point / node counts as plastic
PEEQ_TOL = 1e-9

_VTK_CELL = {(3, 2): "tetra10", (3, 1): "tetra", (2, 2): "triangle6", (2, 1): "triangle"}


@dataclass
class Increment:
    index: int               # 1-based output increment
    time: float              # step time in (0, 1]
    lam: float               # load factor = time x load_factor_max
    U: np.ndarray            # (n, 3) displacements
    S: np.ndarray            # (n, 7) schema stress columns (see CcxResult._map_stress)
    RF: np.ndarray | None    # (n, 3) nodal internal forces
    PEEQ: np.ndarray         # (n,) nodal (extrapolated, averaged) equivalent plastic strain
    totals: dict             # .dat totals of this increment
    peeq_ip: np.ndarray | None = None   # (n_elems, n_ip) integration-point PEEQ (.dat)
    iterations: int | None = None       # Newton iterations of the increment (.sta)


def solve_nonlinear(case, model, deck, inp: str, options: dict, t0: float):
    keep = bool(options.get("keep", False)) or options.get("workdir") is not None
    run = run_ccx(inp, job="job", workdir=options.get("workdir"), ccx=options.get("ccx"),
                  threads=options.get("threads"), timeout=options.get("timeout"), check=False)
    try:
        series = parse_frd_series(run.frd)
        dat = parse_dat_series(run.dat) if run.dat.exists() else []
    except Exception as e:  # pragma: no cover - keep the directory for debugging
        raise CcxError(f"could not parse ccx results in {run.workdir}: {e}") from e
    series = [(t, b) for t, b in series if "DISP" in b and "STRESS" in b]
    if not series:
        tail = "\n".join(run.stdout.splitlines()[-25:])
        raise CcxError(f"ccx produced no converged increment in {run.workdir}:\n"
                       + ("\n".join(run.errors[:10]) or tail))
    res = CcxNonlinearResult(case, model, deck, series, dat, run)
    res.info.wall_time = time.perf_counter() - t0
    if not keep and res.converged:
        shutil.rmtree(run.workdir, ignore_errors=True)
        res.workdir = None
    return res


class CcxNonlinearResult(CcxResult):
    """Elastoplastic ccx solution; positioned on the last converged increment."""

    def __init__(self, case, model, deck, series, dat, run):
        t_last, last = series[-1]
        dat_by_t = {round(d["time"], 5): d for d in dat}
        super().__init__(case, model, deck, last, dat_by_t.get(round(t_last, 5), {}).get("totals", {}), run)
        self.errors = list(run.errors)
        sta = parse_sta(run.workdir / f"{run.job}.sta")
        self.n_requested = int(case.analysis.options.get("increments", 20))
        self.converged = not self.errors and abs(t_last - 1.0) < 1e-5
        self.increments = [self._increment(k + 1, t, b, dat_by_t.get(round(t, 5)), sta)
                           for k, (t, b) in enumerate(series)]
        if not self.converged:
            self.info.warnings.append(
                f"nonlinear run stopped at load factor {self.increments[-1].lam:.4g} "
                f"(step time {t_last:.4g} of 1): " + "; ".join(self.errors[:3]))
        self.use_increment(len(self.increments) - 1)

    def _increment(self, k, t, blocks, dat, sta) -> Increment:
        nn = self.model.n_nodes

        def nodal(name, ncols):
            if name not in blocks:
                return None
            b = blocks[name]
            out = np.zeros((nn, ncols))
            idx = b.nodes - 1
            ok = (idx >= 0) & (idx < nn)
            out[idx[ok]] = b.values[ok][:, :ncols]
            return out

        U = nodal("DISP", 3)
        if self.case.analysis.formulation == "axisymmetric":
            U[:, 0] /= axi_radial_factor(self.model)
        pe = nodal("PE", 1)
        its = [s["iterations"] for s in sta if abs(s["time"] - t) < 1e-5]
        return Increment(index=k, time=t, lam=t * self.deck.lf, U=U, S=self._map_stress(nodal("STRESS", 6)),
                         RF=nodal("FORC", 3), PEEQ=np.zeros(nn) if pe is None else pe[:, 0],
                         totals=(dat or {}).get("totals", {}), peeq_ip=self._ip_table(dat),
                         iterations=its[-1] if its else None)

    def _ip_table(self, dat) -> np.ndarray | None:
        if not dat or dat.get("peeq") is None:
            return None
        el, ip, v = dat["peeq"]
        ok = (el >= 1) & (el <= self.model.n_elems)
        nip = int(ip.max())
        tab = np.zeros((self.model.n_elems, nip))
        tab[el[ok] - 1, ip[ok] - 1] = v[ok]
        return tab

    # ---------------------------------------------------------------- increments
    def use_increment(self, i: int):
        """Position the result (fields, QoIs, reactions) on increment ``i`` (0-based, -1 = last)."""
        inc = self.increments[i]
        self.current = inc
        self.U, self.S, self.RF, self.totals = inc.U, inc.S, inc.RF, inc.totals
        self.load_scale = inc.lam
        self._fields = {}
        return self

    def strain_energy(self) -> float:
        """Total internal energy (CalculiX ELSE: elastic energy + plastic work) at the increment."""
        tot = self.totals.get(("internal energy", "EALL"))
        if tot is None:
            raise CcxError("no internal energy total in the .dat")
        return float(tot[0]) * self.deck.out_factor

    def cell_measures(self) -> np.ndarray:
        """Physical measure of every cell (thickness / 2 pi r included)."""
        _, _, x, w = self.deck.cell_quad(np.arange(self.model.n_elems), degree=4)
        return (w * self.deck.weight(x)).sum(axis=1)

    def balance(self) -> dict:
        """Reactions at the constrained nodes vs the applied load at the current increment."""
        d = self.deck
        dim = self.model.dim
        cons = np.array(sorted({n for n, _ in d.dirichlet}), dtype=np.int64)
        applied = self.load_scale * d.fext[:, :dim].sum(axis=0)
        if self.RF is None:
            return {"applied": applied.tolist()}
        # .frd RF: section thickness in plane stress (x out_factor) but full 360 degrees for CAX
        # elements (unlike the 2-degree .dat totals)
        rf_factor = 1.0 if d.axi else d.out_factor
        react = (self.RF[cons, :dim].sum(axis=0) * rf_factor
                 - self.load_scale * d.fext[cons, :dim].sum(axis=0))
        groups = {g: [self.reaction(g, c) for c in range(dim)] for g in dict.fromkeys(d.dirichlet_groups)}
        # axisymmetric: only the axial resultant is an equilibrium equation (radial forces are hoop)
        k = [1] if d.axi else list(range(dim))
        scale = max(float(np.linalg.norm(applied[k])),
                    max((float(np.linalg.norm(np.asarray(v)[k])) for v in groups.values()), default=0.0), 1e-300)
        return {"applied": applied.tolist(), "reaction_total": react.tolist(), "reactions": groups,
                "imbalance_rel": float(np.linalg.norm((react + applied)[k]) / scale)}

    def increment_qois(self, i: int, peeq_tol: float = PEEQ_TOL) -> dict:
        """Per-increment scalars: max von Mises, max PEEQ (nodal and integration point), plastic
        volume fraction (integration points with PEEQ > peeq_tol; nodal fallback), reaction balance,
        and the case QoIs."""
        self.use_increment(i)
        inc = self.current
        vm = self.nodal("von_mises")[1]
        out = {"increment": inc.index, "time": inc.time, "lambda": inc.lam, "iterations": inc.iterations,
               "max_von_mises": float(vm.max()), "max_peeq": float(inc.PEEQ.max()),
               "plastic_node_fraction": float(np.mean(inc.PEEQ > peeq_tol))}
        meas = self._measures if hasattr(self, "_measures") else self.cell_measures()
        self._measures = meas
        if inc.peeq_ip is not None:
            frac = np.mean(inc.peeq_ip > peeq_tol, axis=1)
            out["max_peeq_ip"] = float(inc.peeq_ip.max())
            out["plastic_volume_fraction"] = float(np.dot(frac, meas) / meas.sum())
            out["plastic_volume"] = float(np.dot(frac, meas))
        out.update(self.balance())
        out["qois"] = {q.name: float(self.evaluate_qoi(q)) for q in self.case.qois}
        return out

    # ---------------------------------------------------------------- output
    def write_increment_vtu(self, i: int, path, point_data: dict | None = None) -> Path:
        """One .vtu of increment ``i`` (fields above + ``point_data``: extra nodal arrays)."""
        import meshio

        self.use_increment(i)
        m, inc = self.model, self.current
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd = {"displacement": inc.U.copy(), "PEEQ": inc.PEEQ.copy(),
              "plastic": (inc.PEEQ > PEEQ_TOL).astype(np.float64)}
        for f in ("von_mises", "sxx", "syy", "szz", "sxy", "syz", "sxz", "s1", "s3", "u_mag"):
            pd[f] = self.nodal(f)[1]
        pd.update(point_data or {})
        cd = {}
        if inc.peeq_ip is not None:
            cd["PEEQ_ip_max"] = [inc.peeq_ip.max(axis=1)]
            cd["plastic_ip_fraction"] = [np.mean(inc.peeq_ip > PEEQ_TOL, axis=1)]
        cells = [(_VTK_CELL[(m.dim, m.order)], m.conn.astype(np.int64))]
        meshio.Mesh(m.X.copy(), cells, point_data=pd, cell_data=cd,
                    field_data={"lambda": np.array([inc.lam])}).write(str(path))
        return path

    def write_series(self, out_dir, stem: str = "inc", extra: dict | None = None, point_data=None) -> dict:
        """inc_000.vtu (lambda = 0, undeformed) .. inc_NNN.vtu, <stem>.pvd and summary.json.
        ``point_data(result, increment) -> {name: nodal array}`` adds fields to every .vtu (zeros at
        lambda = 0); ``extra`` is merged into summary.json."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows, files, keys = [], [], {}
        for i in range(len(self.increments)):
            rows.append(self.increment_qois(i))
            pd = point_data(self, self.current) if point_data else {}
            keys = {k: np.zeros_like(v) for k, v in pd.items()}
            files.append((self.increments[i].lam, self.write_increment_vtu(i, out / f"{stem}_{i + 1:03d}.vtu", pd)))
        files.insert(0, (0.0, self._write_zero(out / f"{stem}_000.vtu", keys)))
        ds = "\n".join(f'    <DataSet timestep="{lam:.6g}" part="0" file="{p.name}"/>' for lam, p in files)
        (out / f"{stem}.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                                         f"  <Collection>\n{ds}\n  </Collection>\n</VTKFile>\n")
        summary = self.series_summary(rows, extra)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        self.use_increment(-1)
        return summary

    def _write_zero(self, path, extra: dict | None = None) -> Path:
        import meshio

        m = self.model
        z = np.zeros(m.n_nodes)
        pd = {"displacement": np.zeros((m.n_nodes, 3)), "PEEQ": z, "plastic": z}
        pd.update({f: z for f in ("von_mises", "sxx", "syy", "szz", "sxy", "syz", "sxz", "s1", "s3", "u_mag")})
        pd.update(extra or {})
        cells = [(_VTK_CELL[(m.dim, m.order)], m.conn.astype(np.int64))]
        meshio.Mesh(m.X.copy(), cells, point_data=pd, field_data={"lambda": np.array([0.0])}).write(str(path))
        return path

    def series_summary(self, rows: list[dict], extra: dict | None = None) -> dict:
        o = self.case.analysis.options
        first = next((r for r in rows if r.get("plastic_volume_fraction", r["plastic_node_fraction"]) > 0), None)
        return {"backend": "ccx", "nonlinear": {k: o.get(k) for k in ("increments", "load_factor_max", "nlgeom",
                                                                       "adaptive")},
                "materials": {n: {"E": m.E, "nu": m.nu, "plasticity": m.plasticity}
                              for n, m in self.case.materials.items()},
                "converged": self.converged, "errors": self.errors[:10], "n_increments": len(rows),
                "lambda_reached": rows[-1]["lambda"] if rows else 0.0,
                "first_plastic_increment_lambda": first["lambda"] if first else None,
                "info": {k: getattr(self.info, k) for k in ("cell_type", "n_nodes", "n_cells", "n_dofs",
                                                             "wall_time", "warnings")},
                **(extra or {}), "increments": rows}
