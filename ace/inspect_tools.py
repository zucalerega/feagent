"""The teacher's inspection tools (field mode): a detailed look at a run instead of a rerun.

error_map: where the run's stress error is. The run's scored setup is re-solved on its own mesh
(the solver is deterministic, so this is the solution the agent got) and compared with the fine
reference at the reference integration points. Each point i carries the error energy

    e_i = w_i * d_i : C^-1 : d_i,      d = sigma_run - sigma_ref      (as in field_error.energy_sq)

so sum_i e_i = ||d||_E^2 and the e_i say where the field error lives. The points carrying the top
80% of it are grouped into hotspots (balls of radius 5% of the model's diagonal).

mesh_layout: where the run put its elements: the local element size near each named group of its
mesh, against the mesh's median size.

Only meaningful in field mode (ACE_SCORE=field), where the reference field exists.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from ace import checkpoint as ckpt
from ace import config
from ace import counterfactual as cf
from ace import fea
from ace import field_error as fe
from ace.checks import reference_setup

TOP_SHARE = 0.8          # hotspots cover the points carrying this share of the error energy
RADIUS = 0.05            # hotspot radius, as a fraction of the model's diagonal
MAX_HOTSPOTS = 6


def _sig(x: float, n: int = 3) -> float:
    return float(f"{float(x):.{n}g}")


def _not_field() -> dict | None:
    return None if config.SCORE == "field" else {"error": "only available in field mode (ACE_SCORE=field)"}


def _scored_mesh(run: dict, workdir: Path) -> tuple[dict, dict, list]:
    """The run's scored checkpoint and its mesh (the saved .msh, or rebuilt from its request)."""
    cp = ckpt.checkpoint(run)
    if cp is None:
        raise ValueError("the run has no successful solve")
    notes: list = []
    mesh = cf._checkpoint_mesh(fea.get_problem(run["problem_id"]), cp, Path(workdir), notes)
    return cp, mesh, notes


class _Mesh:
    """The run's mesh read back: its elements (corner nodes only) and its named groups."""

    def __init__(self, path: str, dim: int):
        m = meshio.read(path)
        self.points = m.points[:, :3]
        kind = "triangle" if dim == 2 else "tetra"
        cells = np.vstack([c.data[:, : dim + 1] for c in m.cells if c.type.startswith(kind)])
        X = self.points[cells]                                         # (ne, dim+1, 3)
        pairs = [(a, b) for a in range(dim + 1) for b in range(a + 1, dim + 1)]
        self.size = np.mean([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in pairs], axis=0)
        self.cells, self.centroids = cells, X.mean(axis=1)
        self.median = float(np.median(self.size))
        self.tree = cKDTree(self.centroids)
        # Named lower-dimensional groups (faces, edges, vertices) -> their node indices. The domain
        # group and gmsh's own bookkeeping sets are left out.
        self.groups = {}
        for name, blocks in m.cell_sets_dict.items():
            if name.startswith("gmsh:") or any(t.startswith(kind) for t in blocks):
                continue
            self.groups[name] = np.unique(np.concatenate([m.cells_dict[t][i].ravel() for t, i in blocks.items()]))
        self.group_trees = {g: cKDTree(self.points[n]) for g, n in self.groups.items()}

    def local_size(self, x: np.ndarray, r: float) -> float:
        """Mean size of the elements whose centroid is within r of x (the nearest 3 if none is)."""
        near = self.tree.query_ball_point(x, r)
        if not near:
            near = np.atleast_1d(self.tree.query(x, k=min(3, len(self.size)))[1])
        return float(np.mean(self.size[near]))

    def nearest_group(self, x: np.ndarray) -> tuple[str | None, float]:
        best = min(((t.query(x)[0], g) for g, t in self.group_trees.items()), default=(np.inf, None))
        return best[1], float(best[0])


def _energy_density(s: dict, w: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Per-point terms of field_error.energy_sq (their sum is energy_sq)."""
    ss = sum(s[k] ** 2 for k in fe.DIAG) + 2 * sum(s[k] ** 2 for k in fe.OFFDIAG)
    tr = sum(s[k] for k in fe.DIAG)
    return w * ((1 + nu) * ss - nu * tr ** 2) / E


def _hotspots(pts: np.ndarray, e: np.ndarray, r: float) -> list[np.ndarray]:
    """Greedy balls over the points carrying the top TOP_SHARE of the error energy: the worst
    unassigned point seeds a ball of radius r that takes every unassigned top point inside it."""
    order = np.argsort(-e)
    top = order[: int(np.searchsorted(np.cumsum(e[order]), TOP_SHARE * e.sum())) + 1]
    tree = cKDTree(pts[top])
    free = np.ones(len(top), bool)
    spots = []
    for j in range(len(top)):                    # top is sorted worst first
        if free[j]:
            members = [k for k in tree.query_ball_point(pts[top[j]], r) if free[k]]
            free[members] = False
            spots.append(top[members])
    return spots


def error_map(problem_id: str, run: dict, workdir: Path) -> dict:
    if (msg := _not_field()) is not None:
        return msg
    p = fea.get_problem(problem_id)
    cp, mesh, notes = _scored_mesh(run, workdir)
    res = fe._solve(p, mesh, cp["setup"], config.TASKS[problem_id]["qoi"])
    mat = reference_setup(p)["material"]
    ref = fe.reference(problem_id)
    pts, sref = ref["pts"], ref["sigma"]
    srun = fe.stresses(res, pts, p.dim)
    e = _energy_density({k: srun[k] - sref[k] for k in srun}, ref["w"], mat["E"], mat["nu"])
    total = float(e.sum())
    vm_ref, vm_run = fe.von_mises(sref), fe.von_mises(srun)

    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    r = RADIUS * diag
    msh = _Mesh(mesh["path"], p.dim)
    spots = sorted(_hotspots(pts, e, r), key=lambda m: -e[m].sum())
    rows = []
    for m in spots[:MAX_HOTSPOTS]:
        c = np.average(pts[m], axis=0, weights=e[m])
        group, dist = msh.nearest_group(c)
        rows.append({"centroid": [_sig(x) for x in c[: p.dim]], "share": _sig(e[m].sum() / total, 2),
                     "nearest_group": group, "distance": _sig(dist), "local_h": _sig(msh.local_size(c, r)),
                     "vm_ref": _sig(vm_ref[m].mean()),
                     "vm_rel_error": _sig(np.abs(vm_run[m] - vm_ref[m]).mean() / vm_ref[m].mean())})

    # Check points: the teacher is the evaluator, so it sees the hidden ones too.
    pp = fe.probe_points(problem_id)
    vm_p = fe.von_mises(fe.stresses(res, np.array(pp["points"]), p.dim))
    rel = (vm_p - np.array(pp["vm_ref"])) / np.array(pp["vm_ref"])
    checks = [{"i": i, "point": [_sig(x) for x in pt[: p.dim]], "hidden": i >= config.PROBE_DISCLOSED,
               "vm_rel_error": _sig(rel[i])} for i, pt in enumerate(pp["points"])]
    hidden = sorted((c for c in checks if c["hidden"]), key=lambda c: -abs(c["vm_rel_error"]))
    return {"run_id": run.get("run_id"),
            "field_error": _sig(np.sqrt(total / fe.energy_sq(sref, ref["w"], mat["E"], mat["nu"]))),
            "probe_error": _sig(np.sqrt(np.mean(rel ** 2))),
            "model_diagonal": _sig(diag), "hotspot_radius": _sig(r), "median_h": _sig(msh.median),
            "n_hotspots": len(spots), "hotspots": rows,
            "check_points": checks, "worst_hidden": [c["i"] for c in hidden[:3]], "notes": notes}


def mesh_layout(problem_id: str, run: dict, workdir: Path) -> dict:
    if (msg := _not_field()) is not None:
        return msg
    p = fea.get_problem(problem_id)
    cp, mesh, notes = _scored_mesh(run, workdir)
    msh = _Mesh(mesh["path"], p.dim)
    r = RADIUS * float(np.linalg.norm(np.ptp(msh.points, axis=0)))
    groups = {}
    for g, nodes in msh.groups.items():
        touching = np.isin(msh.cells, nodes).any(axis=1)      # elements with a corner node on the group
        near = touching.copy()                                  # ... and those with a centroid within r of it
        for x in msh.tree.query_ball_point(msh.points[nodes], r):
            near[x] = True
        groups[g] = {"elements_touching": int(touching.sum()), "h_touching": _sig(msh.size[touching].mean()),
                     "elements_near": int(near.sum()), "h_near": _sig(msh.size[near].mean()),
                     "h_near_vs_median": _sig(msh.size[near].mean() / msh.median, 2)}
    solve = run["solves"][cp["solve_index"]]
    return {"run_id": run.get("run_id"), "mesh_request": cp["request"], "order": mesh.get("order"),
            "nodes": mesh.get("nodes"), "dofs": (solve["result"].get("info") or {}).get("n_dofs"),
            "elements": len(msh.size), "median_h": _sig(msh.median),
            "h_range": [_sig(msh.size.min()), _sig(msh.size.max())], "near_radius": _sig(r),
            "groups": groups, "notes": notes}
