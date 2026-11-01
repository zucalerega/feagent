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
  criterion         which cells fail (default behaviour: "peeq"):
                    "peeq"          PEEQ >= limit, whatever the stress state;
                    "peeq_tension"  tensile PEEQ >= limit: the PEEQ a cell gains while it is in tension
                                    (stress triaxiality sigma_m / sigma_vm > triaxiality_min, default 0),
                                    summed over the output increments, so plastic flow under compression
                                    (crushing) does not count towards a tensile break; or the total PEEQ >=
                                    compression_factor x limit (a brittle resin breaks in tension, crushed
                                    material in compression keeps carrying load);
                    "principal"     max principal stress >= stress_limit (Pa), or PEEQ >= compression_factor x
                                    limit in any state.
                    The stress of a cell is the mean of the nodal (extrapolated, averaged) Cauchy stress at its
                    corner nodes (at every output increment for peeq_tension, at the block end for
                    principal). compression_factor default 5 (null: never).
                    Candidates are released in order of their damage index (PEEQ / its threshold, or
                    sigma_1 / stress_limit), highest first.
  limit             PEEQ at which a cell fails
  aggregate         "mean" (default) or "max" over the cell's integration points
  peeq_step         every block is one ccx step whose length is chosen from the last block's growth rate
                    of the max cell PEEQ so that it grows by about peeq_step x limit (default 0.2), clipped to
                    [1 / fine_increments, 1 / increments] of the ramp (fine_increments default 5 x increments)
  near_increments   once a cell has PEEQ >= refine_at x limit (or has failed) the block is at most
                    1 / near_increments of the ramp (default fine_increments / 2); refine_at default 0.5.
                    For the other criteria the damage index stands for PEEQ / limit here and in peeq_step
                    (peeq_tension: the tensile PEEQ, so crushing far from a tensile failure keeps the coarse
                    bounds)
  max_increment     bound of ccx's automatic increments inside a block (ramp fraction; default
                    1 / increments); max_increment_near the same once failure is near (default
                    1 / near_increments). Every converged ccx increment is an output increment.
  max_remove        cells removed per block at most (8), see below. A cell whose removal alone does not
                    converge is kept ("stuck") for STUCK_BLOCKS blocks while other candidates go; if no
                    removal converges the block runs without removal.
  adaptive_release  false (default): every block removes at most max_remove cells. true: max_remove is the upper
                    bound of an adaptive cap that starts there, doubles after every converged block that removed
                    failed cells at its first try, stays after a block that needed a smaller cap, and halves on
                    non-convergence (the retry ladder), down to 1; a block that removes cells fails fast (ccx may
                    cut its increment back once, not ~5 times: the released stress acts at once, so smaller
                    increments do not rescue a removal; converged removals needed at most one cut-back). Blocks that leave
                    failed cells waiting advance the platen by only 1 / (10 fine_increments), so the cap changes how
                    the removals are split up, not the travel at which cells fail.
  release_travel    ramp fraction a block advances while it leaves failed cells waiting (default 0.1 / fine_increments,
                    1e-3 / fine_increments with adaptive_release: the platen is practically held while a backlog
                    of failed cells is released, so the travel at which cells fail does not depend on the cap)
  stop_force_ratio  0.01
  stop_force_count  consecutive increments (after the first failure) with force < stop_force_ratio x peak
                    that stop the run (default 1)
  tail_force_ratio  once a cell has failed and the force at a block end is below this x peak, the crack is
                    established ("tail", latched): ccx increments are then bounded by max_increment_tail
                    (default max_increment) and blocks by 1 / tail_increments (default increments) instead
                    of the near-failure bounds. Default off.
  end_fraction      stop at this fraction of the ramp (default 1)
  path_groups       groups that must stay connected (default: the Dirichlet groups)
  force_groups      groups whose summed reaction is the recorded force (default: the Dirichlet groups with a
                    nonzero prescribed displacement, else path_groups); e.g. the fixed base when the moving
                    platen is a rigid contact surface added to the deck (crush.py --platen friction)
  max_blocks        5000
Solver options: ``checkpoint_dir`` writes checkpoint.json / checkpoint.npz after every block (the
driver state plus the path of that block's .rout, which is kept until the next block's checkpoint
is written); ``resume`` (a checkpoint dir or the dict of ``load_checkpoint``) continues such a run
from its last block (same case; a new workdir; the old workdir is only read).
``series_dir`` writes result_NNNN.vtu per converged increment while running,
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
from pathlib import Path

import numpy as np

from feacore.solver._ccx_io import CcxError, fnum, id_lines, parse_dat_series, parse_frd_series, run_ccx
from feacore.solver._ccx_nl import CcxNonlinearResult

ALIVE, FAILED, DETACHED = 0, 1, 2
CRITERIA = ("peeq", "peeq_tension", "principal")
CRIT_KEYS = ("criterion", "compression_factor", "triaxiality_min", "stress_limit")
STUCK_TRIES, STUCK_BLOCKS = 5, 5    # single-cell alternatives tried; blocks a stuck cell waits


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
                dt_max: float | None = None, dt_min_ratio: float = 1e-3) -> str:
    """Step text of one block: automatic incrementation starting at dt, bounded by dt_max (default dt) and
    dt_min_ratio x dt_max, Dirichlet totals at t1, removals."""
    dt_max = dt if dt_max is None else dt_max
    out, in_bc, k = [], False, 0
    while k < len(step):
        ln = step[k]
        if ln.startswith("*STATIC"):
            out += ["*STATIC", f"{fnum(min(dt, period))}, {fnum(period)}, {fnum(dt_max * dt_min_ratio)}, "
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
        ne, nc = model.n_elems, model.n_corners
        corners = model.conn[:, :nc]
        loc = model.faces_local[:, :model.n_face_corners]       # cell faces (corner nodes)
        faces = np.sort(np.concatenate([model.conn[:, list(c)] for c in loc]), axis=1)
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


def _attempts_adaptive(cap: int):
    """Adaptive release ladder: the cap halved down to one cell at the full increment (a smaller increment does
    not help a removal: the released stress acts at once), other single cells, then no removal."""
    out = []
    while cap >= 1:
        out.append((cap, 1.0))
        cap //= 2
    return out + [(1, 1.0)] * STUCK_TRIES + [(0, 1.0), (0, 0.25)]


def _attempts(max_remove: int):
    """(removal cap, increment scale) per attempt: full size, then smaller increments, then the cap halved
    down to one cell (a staggered release converges where removing many stressed cells at once does not)."""
    out, cap = [(max_remove, 1.0), (max_remove, 0.25)], max_remove // 2
    while cap >= 1:
        out.append((cap, 0.25))
        cap //= 2
    return out + [(1, 0.25)] * STUCK_TRIES + [(0, 1.0)]    # other single cells, then no removal


def _plan(state, agg_peeq, limit, cap, topo, exclude=()):
    """Cells to remove at the start of the next block: the (up to ``cap``) cells over the limit with the
    highest PEEQ, except ``exclude`` (stuck cells), then hinge contacts / unsupported fragments.
    -> (new state, removal, joined, deferred = candidates left for later blocks)."""
    st = state.copy()
    cand = np.nonzero((st == ALIVE) & (agg_peeq >= limit))[0]
    cand = cand[~np.isin(cand, list(exclude))]
    cand = cand[np.argsort(-agg_peeq[cand], kind="stable")]
    take = cand[:cap]
    st[take] = FAILED
    if len(take):
        det, joined = topo.analyse(st == ALIVE)
        st[det] = DETACHED
    else:
        det, joined = np.zeros(0, np.int64), True
    return st, np.concatenate([take, det]).astype(np.int64), joined, int(len(cand) - len(take))


def cell_stress_measures(model, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(triaxiality sigma_m / sigma_vm, max principal stress) per cell from the nodal schema stress (n, 7)
    averaged over the cell's corner nodes."""
    s = S[model.conn[:, :model.n_corners]].mean(axis=1)
    T = np.zeros((len(s), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
    sm = np.trace(T, axis1=1, axis2=2) / 3.0
    dev = T - sm[:, None, None] * np.eye(3)
    vm = np.sqrt(1.5 * np.einsum("cij,cij->c", dev, dev))
    return sm / np.maximum(vm, 1e-30 + 1e-12 * np.abs(sm)), np.linalg.eigvalsh(T)[:, -1]


def damage_index(fo: dict, agg_peeq: np.ndarray, model=None, S: np.ndarray | None = None,
                 peeq_t: np.ndarray | None = None) -> np.ndarray:
    """Per cell: >= 1 means failed under ``fo["criterion"]`` (see the module doc). peeq_tension: ``peeq_t`` is
    the tensile PEEQ (accumulated by the driver); without it the current stress state decides (PEEQ / limit
    where in tension)."""
    crit, limit = fo.get("criterion", "peeq"), float(fo["limit"])
    if crit == "peeq":
        return agg_peeq / limit
    kc = fo.get("compression_factor", 5.0)
    d_any = agg_peeq / (float(kc) * limit) if kc is not None else np.zeros_like(agg_peeq)
    if crit == "peeq_tension" and peeq_t is not None:
        return np.maximum(peeq_t / limit, d_any)
    if S is None:
        return d_any
    eta, s1 = cell_stress_measures(model, S)
    if crit == "peeq_tension":
        return np.where(eta > float(fo.get("triaxiality_min", 0.0)), agg_peeq / limit, d_any)
    if crit == "principal":
        return np.maximum(s1 / float(fo["stress_limit"]), d_any)
    raise ValueError(f"failure criterion {crit!r}: one of {CRITERIA}")


CKPT_ARRAYS = ("state", "removed_at", "agg_peeq", "frozen_U")
CKPT_OPTIONAL = ("damage", "peeq_t")


def save_checkpoint(d, ck: dict) -> None:
    """Driver state after a block (atomic: written to temporaries, then renamed)."""
    import json
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "checkpoint.tmp.npz", **{a: ck[a] for a in CKPT_ARRAYS + CKPT_OPTIONAL if a in ck})
    meta = {k: v for k, v in ck.items() if k not in CKPT_ARRAYS + CKPT_OPTIONAL}
    (d / "checkpoint.tmp.json").write_text(json.dumps(meta, indent=1, default=float))
    os.replace(d / "checkpoint.tmp.npz", d / "checkpoint.npz")
    os.replace(d / "checkpoint.tmp.json", d / "checkpoint.json")


def load_checkpoint(d) -> dict:
    import json
    d = Path(d)
    ck = json.loads((d / "checkpoint.json").read_text())
    with np.load(d / "checkpoint.npz") as z:
        ck.update({a: z[a] for a in CKPT_ARRAYS})
        ck.update({a: z[a] for a in CKPT_OPTIONAL if a in z.files})
    return ck


def solve_failure(case, model, deck, inp: str, options: dict, t0: float):
    a = case.analysis
    fo = dict(a.options["failure"])
    limit, agg = float(fo["limit"]), fo.get("aggregate", "mean")
    crit = fo.get("criterion", "peeq")
    if crit not in CRITERIA:
        raise ValueError(f"failure criterion {crit!r}: one of {CRITERIA}")
    if crit == "principal" and not float(fo.get("stress_limit", 0)) > 0:
        raise ValueError("failure criterion 'principal' needs stress_limit > 0 (Pa)")
    n = int(a.options.get("increments", 20))
    n_fine = int(fo.get("fine_increments", 5 * n))
    n_near = int(fo.get("near_increments", max(n, n_fine // 2)))
    peeq_step, refine_at = float(fo.get("peeq_step", 0.2)), float(fo.get("refine_at", 0.5))
    max_remove = int(fo.get("max_remove", 8))
    adaptive = bool(fo.get("adaptive_release", False))
    release_dt = float(fo.get("release_travel", (1e-3 if adaptive else 0.1) / n_fine))
    inc_max = float(fo.get("max_increment", 1.0 / n))                 # ccx increment bound (ramp fraction)
    inc_near = float(fo.get("max_increment_near", 1.0 / n_near))
    stop_ratio, max_blocks = float(fo.get("stop_force_ratio", 0.01)), int(fo.get("max_blocks", 5000))
    stop_count = int(fo.get("stop_force_count", 1))
    tail_ratio = fo.get("tail_force_ratio")
    inc_tail = float(fo.get("max_increment_tail", inc_max))
    n_tail = int(fo.get("tail_increments", n))
    t_end = min(1.0, float(fo.get("end_fraction", 1.0)))
    paths = fo.get("path_groups") or list(dict.fromkeys(deck.dirichlet_groups))
    moved = np.zeros(model.n_nodes, bool)
    moved[[nd for (nd, _), v in deck.dirichlet.items() if v != 0]] = True
    driven = fo.get("force_groups") or \
        [g for g in dict.fromkeys(deck.dirichlet_groups) if moved[model.nodes_of(g)].any()] or paths
    model_txt, step = _split_deck(inp)
    wd = Path(options.get("workdir") or Path(os.environ.get("FEACORE_CCX_TMP", "/tmp")) / "ccx" /
              f"fail_{os.getpid()}_{int(time.time())}")
    wd.mkdir(parents=True, exist_ok=True)
    series_dir = options.get("series_dir")
    topo = _Topology(model, deck, paths)
    state = np.zeros(model.n_elems, np.int8)            # ALIVE / FAILED / DETACHED
    removed_at = np.full(model.n_elems, -1)
    agg_peeq = np.zeros(model.n_elems)
    damage = np.zeros(model.n_elems)                # criterion's damage index (crit != "peeq")
    peeq_t = np.zeros(model.n_elems)                # tensile PEEQ (peeq_tension)
    eta_min = float(fo.get("triaxiality_min", 0.0))
    frozen_U = np.zeros((model.n_nodes, 3))
    history, stuck, prev_cap = [], {}, None
    t, k, peak, res, stop, rate, cut = 0.0, 0, 0.0, None, None, 0.0, None
    step0, prev_rout, tail, low = 0, None, False, 0
    rel_cap = max_remove                                # current release cap (adaptive_release)
    ckpt_dir = options.get("checkpoint_dir")
    resume = options.get("resume")
    if resume is not None:
        ck = load_checkpoint(resume) if isinstance(resume, (str, Path)) else resume
        state[:], removed_at[:] = ck["state"], ck["removed_at"]
        agg_peeq, frozen_U = np.array(ck["agg_peeq"], float), np.array(ck["frozen_U"], float)
        crit_now = {c: fo.get(c) for c in CRIT_KEYS}
        if "peeq_t" in ck:                      # tensile PEEQ is history: kept when parameters change
            peeq_t = np.array(ck["peeq_t"], float)
        if "damage" in ck and ck.get("criterion_options") == crit_now:
            damage = np.array(ck["damage"], float)
        elif crit == "peeq_tension" and "peeq_t" in ck:
            damage = damage_index(fo, agg_peeq, peeq_t=peeq_t)
        else:       # criterion changed on resume: no stress at hand; the next block end recomputes it
            damage = damage_index(fo, agg_peeq) if crit == "peeq" else np.zeros(model.n_elems)
        t, k, peak, rate = float(ck["t"]), int(ck["block"]), float(ck["peak"]), float(ck.get("rate", 0.0))
        if ck.get("rate_unit") != "damage":     # older checkpoints: max PEEQ growth per ramp
            rate /= limit
        step0, prev_rout = int(ck["step"]), Path(ck["rout"])
        tail, low = bool(ck.get("tail", False)), int(ck.get("low", 0))
        stuck = {int(c): int(u) for c, u in (ck.get("stuck") or {}).items()}
        rel_cap = min(max_remove, int(ck.get("release_cap", max_remove)))
        if not prev_rout.exists():
            raise CcxError(f"resume: restart file {prev_rout} is missing")
    while t < t_end - 1e-12 and k < max_blocks:
        # how close the next failure is: max cell PEEQ / limit, or for the other criteria their damage index
        pk = agg_peeq[state == ALIVE].max(initial=0.0) / limit if crit == "peeq" else \
            damage[state == ALIVE].max(initial=0.0)
        near = bool(state.any() or pk >= refine_at)
        hi = 1.0 / (n_near if near else n)
        inc = inc_near if near else inc_max
        if tail:
            hi, inc = 1.0 / n_tail, inc_tail
        dt = float(np.clip(peeq_step / rate, 1.0 / n_fine, hi)) if rate > 0 else hi
        run, errors, plan, tried = None, [], None, set()
        for attempt, (cap, scale) in enumerate(_attempts_adaptive(rel_cap) if adaptive else _attempts(max_remove)):
            if run is None and attempt and cap == 1 and len(plan[1]) and prev_cap == 1:
                stuck[int(plan[1][0])] = k + STUCK_BLOCKS     # its removal does not converge: keep it for now
            prev_cap = cap
            plan = _plan(state, *((agg_peeq, limit) if crit == "peeq" else (damage, 1.0)), cap, topo,
                         {c for c, u in stuck.items() if u > k})
            key = (tuple(plan[1].tolist()), scale)
            if key in tried:                            # same removal and increment as an earlier attempt
                continue
            tried.add(key)
            ddt = dt if plan[3] == 0 else release_dt       # deferred failures: let the crack run at ~fixed travel
            t1 = min(t_end, t + ddt)
            job = f"b{k:04d}" + ("" if attempt == 0 else f"r{attempt}")
            if k > 0:
                rin = wd / f"{job}.rin"
                rin.unlink(missing_ok=True)
                rin.symlink_to(prev_rout.resolve())
            head = model_txt if k == 0 else "*RESTART, READ\n"
            m = max(1, int(np.ceil((t1 - t) / inc - 1e-6)))     # equal increments <= the bound
            d_inc = (t1 - t) / m * scale
            fast = adaptive and len(plan[1]) > 0    # fail fast: 1 cutback (x 1/4) when cells are removed
            txt = head + _block_step(step, t1, t1 - t, d_inc, plan[1], d_inc, 0.2 if fast else 1e-3)
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
        n_take = int(((st == FAILED) & (state != FAILED)).sum())
        if adaptive and (n_take or cap == 0):     # cap 0: every removal failed, the block ran without one
            # double after a block whose first try converged; after a halving, start the next block at the cap
            # that converged (re-trying the failed size every block costs a diverging ccx run each time)
            rel_cap = (min(max_remove, 2 * cap) if attempt == 0 else cap) if cap else 1
        removed_at[removal] = (history[-1]["step"] if history else step0) + 1
        state[:] = st
        series = [(tt, b) for tt, b in parse_frd_series(run.frd) if "DISP" in b and "STRESS" in b]
        dat = parse_dat_series(run.dat)
        if not series:
            stop = f"block {k}: no converged increment"
            break
        res = CcxNonlinearResult(case, model, deck, series, dat, run)
        res.converged = True
        res.info.warnings = [w for w in res.info.warnings if not w.startswith("nonlinear run stopped")]
        agg_prev = agg_peeq
        for i, inc in enumerate(res.increments):
            if crit == "peeq_tension" and inc.peeq_ip is not None:
                agg_i = inc.peeq_ip.mean(axis=1) if agg == "mean" else inc.peeq_ip.max(axis=1)
                tens = cell_stress_measures(model, inc.S)[0] > eta_min
                peeq_t = peeq_t + np.maximum(agg_i - agg_prev, 0.0) * tens
                agg_prev = agg_i
            rec = _record(res, i, inc, k, attempt, state, agg, limit, driven, frozen_U,
                          (history[-1]["step"] if history else step0) + 1)
            rec["removed_in_block"], rec["deferred"] = int(len(removal)), int(deferred)
            rec["release_cap"] = int(cap)
            rec["stuck"] = int(sum(1 for c, u in stuck.items() if u > k and state[c] == ALIVE))
            if series_dir:
                _write_vtu(res, i, Path(series_dir) / f"result_{rec['step']:04d}.vtu", state, rec, frozen_U, agg,
                           fo=fo if crit != "peeq" else None, peeq_t=peeq_t if crit == "peeq_tension" else None)
            peak = max(peak, rec["force"])
            low = low + 1 if state.any() and rec["force"] < stop_ratio * peak else 0
            rec["tail"] = tail
            history.append(rec)
            if options.get("on_increment"):
                options["on_increment"](rec)
        last = res.increments[-1]
        pk_before = agg_peeq[state == ALIVE].max(initial=0.0) / limit if crit == "peeq" else \
            damage[state == ALIVE].max(initial=0.0)
        if last.peeq_ip is not None:
            agg_peeq = last.peeq_ip.mean(axis=1) if agg == "mean" else last.peeq_ip.max(axis=1)
        if crit != "peeq":
            damage = damage_index(fo, agg_peeq, model, last.S, peeq_t if crit == "peeq_tension" else None)
        pk_after = agg_peeq[state == ALIVE].max(initial=0.0) / limit if crit == "peeq" else \
            damage[state == ALIVE].max(initial=0.0)
        rate = max(0.0, (pk_after - pk_before) / max(last.time - t, 1e-300))    # damage (PEEQ / limit) per ramp
        if tail_ratio is not None and state.any() and history[-1]["force"] < float(tail_ratio) * peak:
            tail = True             # crack established: the tail bounds apply from the next block on
        old_rout, prev_rout, t, k = prev_rout, wd / f"{job}.rout", float(last.time), k + 1
        if ckpt_dir:
            save_checkpoint(ckpt_dir, {
                "t": t, "block": k, "step": history[-1]["step"], "peak": peak, "rate": rate, "rate_unit": "damage", "tail": tail,
                "low": low, "stuck": {str(c): u for c, u in stuck.items()}, "release_cap": rel_cap, "rout": str(prev_rout.resolve()),
                "state": state, "removed_at": removed_at, "agg_peeq": agg_peeq, "frozen_U": frozen_U}
                | ({"damage": damage, "peeq_t": peeq_t, "criterion_options": {c: fo.get(c) for c in CRIT_KEYS}}
                   if crit != "peeq" else {}))
        if not options.get("keep_files"):
            for suf in (".frd", ".dat"):
                (wd / f"{job}{suf}").unlink(missing_ok=True)
            if old_rout is not None and old_rout.resolve().parent == wd.resolve():   # never another run's
                old_rout.unlink(missing_ok=True)
        if not joined:              # this block was solved with the cut applied: its force is the residual
            stop = cut = "separated: no remaining fragment connects " + ", ".join(paths)
        elif low >= stop_count:
            stop = (f"force {history[-1]['force']:.4g} < {stop_ratio:g} x peak {peak:.4g}"
                    + (f" over {low} increments" if stop_count > 1 else ""))
        if stop:
            break
    if res is None:
        raise CcxError(f"ccx failure run produced no converged increment in {wd}: {stop}")
    res.history, res.state, res.removed_at = history, state, removed_at
    res.stop_reason = stop or ("end of ramp" if t_end >= 1.0 else f"end of ramp (end_fraction {t_end:g})")
    res.separated = bool(cut)
    res.converged = not (stop or "").startswith("no convergence")
    res.info.wall_time = time.perf_counter() - t0
    res.workdir = wd
    if not options.get("keep") and options.get("workdir") is None and res.converged:
        shutil.rmtree(wd, ignore_errors=True)
        res.workdir = None
    return res


def _record(res, i, inc, block, attempt, state, agg, limit, driven, frozen_U, step) -> dict:
    res.use_increment(i)
    bal = res.balance()
    F = np.sum([bal["reactions"].get(g, [0.0, 0.0, 0.0]) for g in driven], axis=0)
    alive = state == ALIVE
    pk = (inc.peeq_ip.mean(axis=1) if agg == "mean" else inc.peeq_ip.max(axis=1)) if inc.peeq_ip is not None \
        else np.zeros(len(state))
    nodes_alive = np.zeros(res.model.n_nodes, bool)
    nodes_alive[res.model.conn[alive].ravel()] = True
    frozen_U[nodes_alive] = inc.U[nodes_alive]
    return {"step": step, "block": block, "retry": attempt,
            "time": inc.time, "lambda": inc.lam, "iterations": inc.iterations,
            "reaction": [float(v) for v in F], "force": float(np.linalg.norm(F)), "driven_groups": driven,
            "n_failed": int((state == FAILED).sum()), "n_detached": int((state == DETACHED).sum()),
            "max_peeq_alive": float(pk[alive].max(initial=0.0)), "peeq_limit": limit,
            "imbalance_rel": bal.get("imbalance_rel")}


def _write_vtu(res, i, path, state, rec, frozen_U, agg, peeq=None, fo=None, peeq_t=None):
    import meshio

    m = res.model
    path.parent.mkdir(parents=True, exist_ok=True)
    res.use_increment(i)
    inc = res.current
    if peeq is None:
        peeq = (inc.peeq_ip.mean(axis=1) if agg == "mean" else inc.peeq_ip.max(axis=1)) \
            if inc.peeq_ip is not None else np.zeros(m.n_elems)
    pd = {"displacement": frozen_U.copy(), "von_mises": res.nodal("von_mises")[1], "PEEQ": inc.PEEQ.copy()}
    cd = {"PEEQ": [peeq], "failed": [(state != ALIVE).astype(np.int32)], "state": [state.astype(np.int32)]}
    if fo is not None:              # criterion fields of this increment (not only at block ends)
        eta, s1 = cell_stress_measures(m, inc.S)
        cd |= {"triaxiality": [eta], "s1_cell": [s1], "damage": [damage_index(fo, peeq, m, inc.S, peeq_t)]}
        if peeq_t is not None:
            cd["PEEQ_tension"] = [peeq_t.copy()]
    meshio.Mesh(m.X.copy(), [(m.cell_type, m.conn.astype(np.int64))], point_data=pd,
                cell_data=cd, field_data={"lambda": np.array([inc.lam]), "step": np.array([rec["step"]])}
                ).write(str(path))
