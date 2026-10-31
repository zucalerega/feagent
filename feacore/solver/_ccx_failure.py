"""Progressive failure by element erosion for the opt-in nonlinear ccx path
(``analysis.options.failure = {"criterion": "peeq", "limit": ...}``).

CalculiX has no criterion-driven element deletion, so the load ramp is run in *restart blocks*:

    block 0: model + step (t 0 -> t1) + *RESTART, WRITE
    block k: *RESTART, READ (previous .rout) + step (t_k -> t_k+1) with
             *MODEL CHANGE, TYPE=ELEMENT, REMOVE (cells that failed at the end of block k-1) + *RESTART, WRITE

Every block is a ccx static step with automatic incrementation bounded by ``dt`` (``*STATIC`` dt, period,
dt / 1000, dt). Boundary values are written as totals at the block end; ccx ramps them from the previous
step's values, and the restart carries the full material state (plastic strains, stresses, NLGEOM
configuration), so a run split into blocks reproduces the single run exactly (tested). After every
block the cells whose integration-point PEEQ (mean over the cell's points, or max) reaches ``limit`` are
removed; cells in fragments that touch no constrained node are removed with them ("detached"). The run
stops when no remaining fragment connects all ``path_groups`` (default: every Dirichlet group), when the
force on the driven groups falls below ``stop_force_ratio`` x its peak after the first failure, or at
the end of the ramp.

Options (``failure`` dict; all but criterion / limit optional):
  limit             PEEQ at which a cell fails
  aggregate         "mean" (default) or "max" over the cell's integration points
  peeq_step         every block is one ccx step whose length is chosen from the last block's growth rate
                    of the max cell PEEQ so that it grows by about peeq_step x limit (default 0.2), clipped to
                    [1 / fine_increments, 1 / increments] of the ramp (fine_increments default 5 x increments)
  near_increments   once a cell has PEEQ >= refine_at x limit (or has failed) the block is at most
                    1 / near_increments of the ramp (default fine_increments / 2); refine_at default 0.5
  max_increment     bound of ccx's automatic increments inside a block (ramp fraction; default
                    1 / increments); max_increment_near the same once failure is near (default
                    1 / near_increments). Every converged ccx increment is an output increment.
  max_remove        cells removed per block at most (8), see below
  stop_force_ratio  0.01
  path_groups       groups that must stay connected (default: the Dirichlet groups)
  max_blocks        5000
Solver options: ``series_dir`` writes result_NNNN.vtu per converged increment while running,
``on_increment(record)`` is called with every increment's record, ``keep_files`` keeps every block's
.frd / .dat (default: deleted once parsed; a 400k-dof increment is ~50 MB of .frd).

Limits: no contact, so eroded faces / fragments may interpenetrate; removal releases the cell's
stress at once (no gradual softening), so the result depends on the mesh size near the crack
(regularise with the mesh, not here); loads must be displacement-driven (force / pressure / body /
thermal loads are refused with failure, as they would act on eroded faces).
"""

from __future__ import annotations

import os
import shutil
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from feacore.solver._ccx_io import CcxError, fnum, id_lines, parse_dat_series, parse_frd_series, run_ccx
from feacore.solver._ccx_nl import CcxNonlinearResult

ALIVE, FAILED, DETACHED = 0, 1, 2


def _split_deck(inp: str):
    i = inp.index("\n*STEP")
    model, step = inp[: i + 1], inp[i + 1:].splitlines()
    for kw in ("*DLOAD", "*CLOAD", "*TEMPERATURE"):
        if any(ln.upper().startswith(kw) for ln in step):
            raise NotImplementedError(
                f"ccx failure (erosion): {kw[1:]} loads are not supported; drive the model by prescribed "
                "displacements (the eroded faces would otherwise keep their loads)")
    return model, step


def _block_step(step: list[str], t1: float, period: float, dt: float, remove: np.ndarray,
                dt_max: float | None = None) -> str:
    """Step text of one block: automatic incrementation starting at dt, bounded by dt_max (default dt),
    Dirichlet totals at t1, removals."""
    dt_max = dt if dt_max is None else dt_max
    out, in_bc, k = [], False, 0
    while k < len(step):
        ln = step[k]
        if ln.startswith("*STATIC"):
            out += ["*STATIC", f"{fnum(min(dt, period))}, {fnum(period)}, {fnum(dt_max * 1e-3)}, "
                                f"{fnum(min(dt_max, period))}"]
            if len(remove):
                out += ["*MODEL CHANGE, TYPE=ELEMENT, REMOVE", id_lines(remove + 1)]
            k += 2
            continue
        if ln.startswith("*"):
            in_bc = ln.startswith("*BOUNDARY")
            if ln.startswith("*END STEP"):
                out.append("*RESTART, WRITE")
        elif in_bc:
            p = ln.split(",")
            ln = f"{p[0]},{p[1]},{p[2]}, {fnum(float(p[3]) * t1)}"
        out.append(ln)
        k += 1
    return "\n".join(out) + "\n"


class _Topology:
    """Fragments of the intact cells by shared faces (cells hanging on a node or an edge are mechanisms,
    not load paths). A fragment is supported when it holds >= 3 constrained corner nodes; it joins two
    groups when it holds nodes of both."""

    def __init__(self, model, deck, path_groups):
        from scipy.sparse import coo_matrix, csr_matrix
        self.m = model
        ne, nc = model.n_elems, model.dim + 1
        corners = model.conn[:, :nc]
        loc = list(combinations(range(nc), nc - 1))
        faces = np.sort(np.concatenate([corners[:, list(c)] for c in loc]), axis=1)
        owner = np.tile(np.arange(ne), len(loc))
        _, inv = np.unique(faces, axis=0, return_inverse=True)
        f2e = coo_matrix((np.ones(len(owner)), (inv.ravel(), owner))).tocsr()
        self.adj = (f2e.T @ f2e).tocsr()                  # cells sharing a face (and the diagonal)
        rows = np.repeat(np.arange(ne), nc)
        self.inc = csr_matrix((np.ones(ne * nc), (rows, corners.ravel())), shape=(ne, model.n_nodes))
        self.constrained = np.zeros(model.n_nodes, bool)
        self.constrained[[n for n, _ in deck.dirichlet]] = True
        self.paths = [np.isin(np.arange(model.n_nodes), model.nodes_of(g)) for g in path_groups]

    def _fragments(self, idx):
        from scipy.sparse.csgraph import connected_components
        n, lab = connected_components(self.adj[idx][:, idx], directed=False)
        A = self.inc[idx].tocoo()
        return n, lab, A.row, A.col

    def analyse(self, alive: np.ndarray):
        """(cells to remove, True if a supported fragment joins all path groups). Removed: cells at hinge
        contacts (a node shared by two face-connected fragments: the smaller fragments' cells at that node
        go, repeatedly, so fragments touch only through faces) and unsupported fragments."""
        alive = alive.copy()
        removed = []
        for _ in range(100):
            idx = np.nonzero(alive)[0]
            if len(idx) == 0:
                return np.concatenate(removed or [idx]), False
            n, lab, row, col = self._fragments(idx)
            size = np.bincount(lab, minlength=n)
            pairs = np.unique(np.c_[col, lab[row]], axis=0)                # (node, fragment)
            cnt = np.bincount(pairs[:, 0], minlength=self.m.n_nodes)
            if not np.any(cnt >= 2):
                break
            best = np.zeros(self.m.n_nodes)
            np.maximum.at(best, pairs[:, 0], size[pairs[:, 1]] + 1.0 / (2.0 + pairs[:, 1]))
            hinge = cnt[col] >= 2
            score = size[lab[row]] + 1.0 / (2.0 + lab[row])
            cut = np.unique(idx[row[hinge & (score < best[col])]])
            alive[cut] = False
            removed.append(cut)
        comp = np.zeros((n, self.m.n_nodes), bool)
        comp[lab[row], col] = True
        supported = comp[:, self.constrained].sum(axis=1) >= 3
        joined = np.all([comp[:, p].any(axis=1) for p in self.paths], axis=0) if self.paths else np.ones(n, bool)
        removed.append(idx[~supported[lab]])
        return np.unique(np.concatenate(removed)).astype(np.int64), bool(np.any(joined & supported))


def _attempts(max_remove: int):
    """(removal cap, increment scale) per attempt: full size, then smaller increments, then the cap halved
    down to one cell (a staggered release converges where removing many stressed cells at once does not)."""
    out, cap = [(max_remove, 1.0), (max_remove, 0.25)], max_remove // 2
    while cap >= 1:
        out.append((cap, 0.25))
        cap //= 2
    return out


def _plan(state, agg_peeq, limit, cap, topo):
    """Cells to remove at the start of the next block: the (up to ``cap``) cells over the limit with the
    highest PEEQ, then hinge contacts / unsupported fragments. -> (new state, removal, joined, deferred)."""
    st = state.copy()
    cand = np.nonzero((st == ALIVE) & (agg_peeq >= limit))[0]
    cand = cand[np.argsort(-agg_peeq[cand], kind="stable")]
    take = cand[:cap]
    st[take] = FAILED
    if len(take):
        det, joined = topo.analyse(st == ALIVE)
        st[det] = DETACHED
    else:
        det, joined = np.zeros(0, np.int64), True
    return st, np.concatenate([take, det]).astype(np.int64), joined, int(len(cand) - len(take))


def solve_failure(case, model, deck, inp: str, options: dict, t0: float):
    a = case.analysis
    fo = dict(a.options["failure"])
    limit, agg = float(fo["limit"]), fo.get("aggregate", "mean")
    n = int(a.options.get("increments", 20))
    n_fine = int(fo.get("fine_increments", 5 * n))
    n_near = int(fo.get("near_increments", max(n, n_fine // 2)))
    peeq_step, refine_at = float(fo.get("peeq_step", 0.2)), float(fo.get("refine_at", 0.5))
    max_remove = int(fo.get("max_remove", 8))
    inc_max = float(fo.get("max_increment", 1.0 / n))                 # ccx increment bound (ramp fraction)
    inc_near = float(fo.get("max_increment_near", 1.0 / n_near))
    stop_ratio, max_blocks = float(fo.get("stop_force_ratio", 0.01)), int(fo.get("max_blocks", 5000))
    paths = fo.get("path_groups") or list(dict.fromkeys(deck.dirichlet_groups))
    moved = np.zeros(model.n_nodes, bool)
    moved[[nd for (nd, _), v in deck.dirichlet.items() if v != 0]] = True
    driven = [g for g in dict.fromkeys(deck.dirichlet_groups) if moved[model.nodes_of(g)].any()] or paths
    model_txt, step = _split_deck(inp)
    wd = Path(options.get("workdir") or Path(os.environ.get("FEACORE_CCX_TMP", "/tmp")) / "ccx" /
              f"fail_{os.getpid()}_{int(time.time())}")
    wd.mkdir(parents=True, exist_ok=True)
    series_dir = options.get("series_dir")
    topo = _Topology(model, deck, paths)
    state = np.zeros(model.n_elems, np.int8)            # ALIVE / FAILED / DETACHED
    removed_at = np.full(model.n_elems, -1)
    agg_peeq = np.zeros(model.n_elems)
    frozen_U = np.zeros((model.n_nodes, 3))
    history = []
    t, k, peak, res, stop, rate, cut = 0.0, 0, 0.0, None, None, 0.0, None
    while t < 1.0 - 1e-12 and k < max_blocks:
        pk = agg_peeq[state == ALIVE].max(initial=0.0)
        near = bool(state.any() or pk >= refine_at * limit)
        hi = 1.0 / (n_near if near else n)
        inc = inc_near if near else inc_max
        dt = float(np.clip(peeq_step * limit / rate, 1.0 / n_fine, hi)) if rate > 0 else hi
        run, errors, plan, tried = None, [], None, set()
        for attempt, (cap, scale) in enumerate(_attempts(max_remove)):
            plan = _plan(state, agg_peeq, limit, cap, topo)
            key = (tuple(plan[1].tolist()), scale)
            if key in tried:                            # same removal and increment as an earlier attempt
                continue
            tried.add(key)
            ddt = dt if plan[3] == 0 else 0.1 / n_fine     # deferred failures: let the crack run at ~fixed travel
            t1 = min(1.0, t + ddt)
            job = f"b{k:04d}" + ("" if attempt == 0 else f"r{attempt}")
            if k > 0:
                rin = wd / f"{job}.rin"
                rin.unlink(missing_ok=True)
                rin.symlink_to(wd / f"{prev_job}.rout")
            head = model_txt if k == 0 else "*RESTART, READ\n"
            m = max(1, int(np.ceil((t1 - t) / inc - 1e-6)))     # equal increments <= the bound
            d_inc = (t1 - t) / m * scale
            txt = head + _block_step(step, t1, t1 - t, d_inc, plan[1], d_inc)
            try:
                run = run_ccx(txt, job=job, workdir=wd, ccx=options.get("ccx"), threads=options.get("threads"),
                              timeout=options.get("timeout"), check=False)
                errors = run.errors
            except CcxError as e:          # no .frd at all
                run, errors = None, [str(e).splitlines()[0]]
            if run is not None and not errors and (wd / f"{job}.rout").exists():
                break
            run = None
        if run is None:
            stop = f"no convergence in block {k} (t {t:.6g}): " + "; ".join(errors[:2])
            break
        st, removal, joined, deferred = plan
        removed_at[removal] = (history[-1]["step"] + 1) if history else 1
        state[:] = st
        series = [(tt, b) for tt, b in parse_frd_series(run.frd) if "DISP" in b and "STRESS" in b]
        dat = parse_dat_series(run.dat)
        if not series:
            stop = f"block {k}: no converged increment"
            break
        res = CcxNonlinearResult(case, model, deck, series, dat, run)
        res.converged = True
        res.info.warnings = [w for w in res.info.warnings if not w.startswith("nonlinear run stopped")]
        for i, inc in enumerate(res.increments):
            rec = _record(res, i, inc, k, attempt, state, agg, limit, driven, frozen_U, history)
            rec["removed_in_block"], rec["deferred"] = int(len(removal)), int(deferred)
            if series_dir:
                _write_vtu(res, i, Path(series_dir) / f"result_{rec['step']:04d}.vtu", state, rec, frozen_U, agg)
            peak = max(peak, rec["force"])
            history.append(rec)
            if options.get("on_increment"):
                options["on_increment"](rec)
        last = res.increments[-1]
        pk_before = agg_peeq[state == ALIVE].max(initial=0.0)
        if last.peeq_ip is not None:
            agg_peeq = last.peeq_ip.mean(axis=1) if agg == "mean" else last.peeq_ip.max(axis=1)
        rate = max(0.0, (agg_peeq[state == ALIVE].max(initial=0.0) - pk_before) / max(last.time - t, 1e-300))
        if not options.get("keep_files"):
            for suf in (".frd", ".dat"):
                (wd / f"{job}{suf}").unlink(missing_ok=True)
            if k > 0:
                (wd / f"{prev_job}.rout").unlink(missing_ok=True)
        prev_job, t, k = job, float(last.time), k + 1
        if not joined:              # this block was solved with the cut applied: its force is the residual
            stop = cut = "separated: no remaining fragment connects " + ", ".join(paths)
        elif state.any() and history[-1]["force"] < stop_ratio * peak:
            stop = f"force {history[-1]['force']:.4g} < {stop_ratio:g} x peak {peak:.4g}"
        if stop:
            break
    if res is None:
        raise CcxError(f"ccx failure run produced no converged increment in {wd}: {stop}")
    res.history, res.state, res.removed_at, res.stop_reason = history, state, removed_at, stop or "end of ramp"
    res.separated = bool(cut)
    res.converged = not (stop or "").startswith("no convergence")
    res.info.wall_time = time.perf_counter() - t0
    res.workdir = wd
    if not options.get("keep") and options.get("workdir") is None and res.converged:
        shutil.rmtree(wd, ignore_errors=True)
        res.workdir = None
    return res


def _record(res, i, inc, block, attempt, state, agg, limit, driven, frozen_U, history) -> dict:
    res.use_increment(i)
    bal = res.balance()
    F = np.sum([bal["reactions"].get(g, [0.0, 0.0, 0.0]) for g in driven], axis=0)
    alive = state == ALIVE
    pk = (inc.peeq_ip.mean(axis=1) if agg == "mean" else inc.peeq_ip.max(axis=1)) if inc.peeq_ip is not None \
        else np.zeros(len(state))
    nodes_alive = np.zeros(res.model.n_nodes, bool)
    nodes_alive[res.model.conn[alive].ravel()] = True
    frozen_U[nodes_alive] = inc.U[nodes_alive]
    return {"step": (history[-1]["step"] + 1) if history else 1, "block": block, "retry": attempt,
            "time": inc.time, "lambda": inc.lam, "iterations": inc.iterations,
            "reaction": [float(v) for v in F], "force": float(np.linalg.norm(F)), "driven_groups": driven,
            "n_failed": int((state == FAILED).sum()), "n_detached": int((state == DETACHED).sum()),
            "max_peeq_alive": float(pk[alive].max(initial=0.0)), "peeq_limit": limit,
            "imbalance_rel": bal.get("imbalance_rel")}


def _write_vtu(res, i, path, state, rec, frozen_U, agg, peeq=None):
    import meshio
    from feacore.solver._ccx_nl import _VTK_CELL

    m = res.model
    path.parent.mkdir(parents=True, exist_ok=True)
    res.use_increment(i)
    inc = res.current
    if peeq is None:
        peeq = (inc.peeq_ip.mean(axis=1) if agg == "mean" else inc.peeq_ip.max(axis=1)) \
            if inc.peeq_ip is not None else np.zeros(m.n_elems)
    pd = {"displacement": frozen_U.copy(), "von_mises": res.nodal("von_mises")[1], "PEEQ": inc.PEEQ.copy()}
    cd = {"PEEQ": [peeq], "failed": [(state != ALIVE).astype(np.int32)], "state": [state.astype(np.int32)]}
    meshio.Mesh(m.X.copy(), [(_VTK_CELL[(m.dim, m.order)], m.conn.astype(np.int64))], point_data=pd,
                cell_data=cd, field_data={"lambda": np.array([inc.lam]), "step": np.array([rec["step"]])}
                ).write(str(path))
