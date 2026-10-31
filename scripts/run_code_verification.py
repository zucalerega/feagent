#!/usr/bin/env python
"""Run the full code-verification matrix (MMS) and write docs/code_verification.md.

    python scripts/run_code_verification.py                 # everything (incl. 3D), skfem + ccx
    python scripts/run_code_verification.py --quick         # skip the 3D (slow) studies
    python scripts/run_code_verification.py --backends skfem --only E-pe-annulus-mixed
    python scripts/run_code_verification.py --from-json docs/img/code_verification/results.json

Outputs: docs/code_verification.md (method, order-of-accuracy table, patch tests, skips,
failures), log-log convergence plots docs/img/code_verification/<problem>.png and the raw
numbers in docs/img/code_verification/results.json. Scratch meshes / cases go to --workdir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from feacore.verification import mms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# reference palette (dataviz skill, slots 1-2) + neutral inks
C_P = {1: "#2a78d6", 2: "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARK = {"skfem": ("o", "-"), "ccx": ("s", "--")}
MEASURE = {"u": "displacement / T", "s": "recovered stress / flux", "W": "strain energy"}


def run_matrix(backends, quick, only, workdir, verbose):
    out = {"convergence": [], "patch": [], "control": None}
    cat = [c for c in mms.convergence_catalog() if (not only or c["id"] in only) and not (quick and c["slow"])]
    for c in cat:
        p = mms.convergence_problem(c["id"])
        for backend in backends:
            for order in (1, 2):
                t0 = time.perf_counter()
                entry = {"problem": c["id"], "backend": backend, "order": order, "slow": c["slow"],
                         "formulation": p.formulation, "domain": p.domain.name, "features": p.features,
                         "description": p.description}
                try:
                    r = mms.run_convergence(p, order, backend, workdir=workdir)
                    entry.update(r)
                    entry.update(mms.judge(r, p))
                    entry["status"] = "ran"
                except mms.FeatureMissing as e:
                    entry.update(status="skipped", reason=str(e))
                except Exception as e:  # noqa: BLE001
                    entry.update(status="error", reason=mms.error_report(e))
                entry["seconds"] = time.perf_counter() - t0
                out["convergence"].append(entry)
                if verbose:
                    print(f"[conv] {c['id']:22s} P{order} {backend:5s} {entry['status']:7s} "
                          f"{entry.get('verdict', '')} {fmt_obs(entry)} ({entry['seconds']:.1f}s)", flush=True)
    for c in mms.patch_catalog():
        if only and c["id"] not in only:
            continue
        p = mms.patch_problem(c["id"])
        for backend in backends:
            for order in (1, 2):
                tol_u, tol_s = mms.PATCH_TOL[backend]
                entry = {"problem": c["id"], "backend": backend, "order": order,
                         "formulation": p.formulation, "domain": p.domain.name, "features": p.features}
                try:
                    r = mms.run_patch(p, order, backend, h=c["h"], workdir=workdir, s_scale=c["s_scale"],
                                      tol_u=tol_u, tol_s=tol_s)
                    entry.update(r)
                    entry["status"] = "ran"
                except mms.FeatureMissing as e:
                    entry.update(status="skipped", reason=str(e))
                except Exception as e:  # noqa: BLE001
                    entry.update(status="error", reason=mms.error_report(e))
                out["patch"].append(entry)
                if verbose:
                    print(f"[patch] {c['id']:20s} P{order} {backend:5s} {entry['status']:7s} {entry.get('verdict', '')} "
                          f"u={entry.get('err_u_nodal', float('nan')):.1e} s={entry.get('err_s_probe', float('nan')):.1e}",
                          flush=True)
    if "skfem" in backends and (not only or "E-pe-annulus-lame" in only):
        p = mms.convergence_problem("E-pe-annulus-lame")
        try:
            r = mms.run_convergence(p, 2, "skfem", geom_order=1, workdir=workdir)
            out["control"] = {"orders": r["orders"], "levels": [{k: lv[k] for k in ("h_rep", "e_u", "e_s")}
                                                                  for lv in r["levels"]]}
        except Exception as e:  # noqa: BLE001
            out["control"] = {"error": mms.error_report(e)}
    return out


def fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def fmt_obs(e):
    o = e.get("observed") or {}
    return " ".join(f"{k}={fmt(o.get(k))}" for k in ("u", "s", "W") if k in o)


# --------------------------------------------------------------------------- plots


def plot_problem(pid, entries, img_dir: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ran = [e for e in entries if e["status"] == "ran"]
    if not ran:
        return None
    keys = ["u", "s"] + (["W"] if any(lv.get("e_W") for e in ran for lv in e["levels"]) else [])
    fig, axes = plt.subplots(1, len(keys), figsize=(4.1 * len(keys), 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for e in ran:
            h = np.array([lv["h_rep"] for lv in e["levels"]])
            err = np.array([lv.get(f"e_{key}") or np.nan for lv in e["levels"]])
            if not np.any(np.isfinite(err)):
                continue
            mk, ls = MARK[e["backend"]]
            p = (e.get("observed") or {}).get(key)
            ax.loglog(h, err, ls, marker=mk, ms=6, lw=2, color=C_P[e["order"]],
                      markeredgecolor="white", markeredgewidth=1.0,
                      label=f"P{e['order']} {e['backend']} (p={fmt(p)})")
            # reference slope through the coarsest point
            th = e["theory"][key]
            if e["backend"] == ran[0]["backend"]:
                ax.loglog(h, err[0] * (h / h[0]) ** th, ":", lw=1.2, color=INK2)
                ax.annotate(f"h^{th:g}" if th == int(th) else f"h^{th:.2f}", (h[-1], err[0] * (h[-1] / h[0]) ** th),
                            textcoords="offset points", xytext=(4, -10), fontsize=8, color=INK2)
        ax.set_title(MEASURE[key], fontsize=10, color=INK)
        ax.set_xlabel("h = (|cells| measure / n_cells)^(1/dim)", fontsize=8, color=INK2)
        ax.set_ylabel("relative error", fontsize=8, color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp_ in ax.spines.values():
            sp_.set_color(GRID)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{pid}: {ran[0]['description']}", fontsize=9, color=INK)
    path = img_dir / f"{pid}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- markdown


def write_markdown(res, path: Path, img_dir: Path, backends):
    conv, patch = res["convergence"], res["patch"]
    rel = os.path.relpath(img_dir, path.parent)
    L = []
    a = L.append
    a("# Code verification of the solver backends (MMS)\n")
    a(f"Generated by `scripts/run_code_verification.py` on {res['meta']['date']} "
      f"({res['meta']['python']}, sympy {res['meta']['sympy']}, scikit-fem {res['meta']['skfem']}, "
      f"gmsh {res['meta']['gmsh']}); backends: {', '.join(backends)}. Raw numbers: "
      f"[`results.json`]({rel}/results.json). Tests: `pytest tests/test_code_verification.py` "
      f"(add `FEACORE_SLOW=1` for 3D and the CalculiX studies).\n")
    a(METHOD)
    # ---- order table
    a("## Order of accuracy\n")
    a("Observed order = log(e_i / e_i+1) / log(h_i / h_i+1) on the finest pair of a nested study whose "
      "errors are above the backend's round-off floor (skfem 1e-10, ccx 3e-6: the `.frd` has ~6 digits). "
      "Theory: u = k+1, stress = k, energy = 2k (or k+1+lambda with a Dirichlet-traction corner, see above). "
      "Pass: observed >= theory - 0.3 for every measure.\n")
    a("| problem | formulation | domain | P | backend | u order (th.) | stress order (th.) | energy order (th.) | finest e_u | verdict |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for e in conv:
        if e["status"] != "ran":
            a(f"| {e['problem']} | {e['formulation']} | {e['domain']} | {e['order']} | {e['backend']} | "
              f"- | - | - | - | **{e['status']}** |")
            continue
        o, th = e["observed"], e["theory"]
        W = f"{fmt(o.get('W'))} ({fmt(th['W'])})" if "W" in o else ("info " + fmt(e["orders"].get("W", [None])[-1])
                                                                     if e["orders"].get("W") else "n/a")
        verdict = "pass" if e["verdict"] == "pass" else "**FAIL**"
        a(f"| {e['problem']} | {e['formulation']} | {e['domain']} | {e['order']} | {e['backend']} | "
          f"{fmt(o.get('u'))} ({th['u']:g}) | {fmt(o.get('s'))} ({th['s']:g}) | {W} | "
          f"{e['levels'][-1]['e_u']:.1e} | {verdict} |")
    a("")
    # ---- per-level orders
    a("<details><summary>All pairwise orders (coarse to fine)</summary>\n")
    a("| problem | P | backend | u | stress | energy |")
    a("|---|---|---|---|---|---|")
    for e in conv:
        if e["status"] == "ran":
            row = [", ".join(fmt(v) for v in e["orders"].get(k, [])) or "-" for k in ("u", "s", "W")]
            a(f"| {e['problem']} | {e['order']} | {e['backend']} | " + " | ".join(row) + " |")
    a("\n</details>\n")
    # ---- patch
    a("## Patch tests\n")
    a("Linear fields on distorted meshes (interior nodes moved by up to 0.3 h; P2 midside nodes moved "
      "off the chord, i.e. curved interior edges). Max relative error of nodal u, probed u and probed "
      "recovered stress / flux; tolerance skfem 1e-9 / 1e-8, ccx 5e-5 (output precision).\n")
    a("| patch | formulation | P | backend | err u (nodal) | err u (probe) | err stress | err energy | verdict |")
    a("|---|---|---|---|---|---|---|---|---|")
    for e in patch:
        if e["status"] != "ran":
            a(f"| {e['problem']} | {e['formulation']} | {e['order']} | {e['backend']} | - | - | - | - | **{e['status']}** |")
            continue
        v = "pass" if e["verdict"] == "pass" else "**FAIL**"
        a(f"| {e['problem']} | {e['formulation']} | {e['order']} | {e['backend']} | {e['err_u_nodal']:.1e} | "
          f"{e['err_u_probe']:.1e} | {e['err_s_probe']:.1e} | "
          f"{('%.1e' % e['err_W']) if 'err_W' in e else '-'} | {v} |")
    a("")
    # ---- control
    c = res.get("control")
    if c and "orders" in c:
        a("## Negative control (isoparametric geometry)\n")
        a("P2 on a *straight-sided* (tri3) mesh of the quarter annulus (Lame cylinder) must lose the optimal "
          f"order: observed u orders {', '.join(fmt(v) for v in c['orders']['u'])} (vs ~3 on the curved "
          "tri6 mesh). The curved-boundary cases are therefore sensitive to the geometry order.\n")
    # ---- skips / errors
    sk = [e for e in conv + patch if e["status"] in ("skipped", "error")]
    a("## Skipped / errored\n")
    if not sk:
        a("None: every case ran on every backend listed above.\n")
    else:
        a("| case | P | backend | status | reason |")
        a("|---|---|---|---|---|")
        for e in sk:
            a(f"| {e['problem']} | {e['order']} | {e['backend']} | {e['status']} | {e['reason'][:160].replace('|', '/')} |")
        a("")
    a(res.get("findings", FINDINGS))
    # ---- plots
    a("## Convergence plots\n")
    for pid in dict.fromkeys(e["problem"] for e in conv):
        f = img_dir / f"{pid}.png"
        if f.exists():
            a(f"![{pid}]({rel}/{pid}.png)\n")
    path.write_text("\n".join(L) + "\n")
    return path


METHOD = """## Method

* **Manufactured solutions** (`feacore/verification/mms.py`): smooth, non-polynomial exact fields
  (exp / sin / cos products without symmetries, so neither P1 nor P2 is exact and no error component
  vanishes by accident). sympy derives, in exact rational arithmetic, the body force (`-div sigma`,
  with the axisymmetric `(s_rr - s_tt)/r`, `s_rz/r` terms) or heat source (`-k lap T`), Dirichlet data,
  tractions `sigma.n`, pressures `-n.sigma.n`, fluxes `k dT/dn`, the thermal-strain load, exact
  stresses / fluxes and the exact strain energy `0.5 a(u, u)` (Gauss-Legendre on the exact domain).
  Expressions are printed in the `feacore.case.eval_expr` grammar and checked against `lambdify`.
* **Special BCs are satisfied exactly by construction**: the field is corrected by `phi * c` with
  `phi = 0, dphi/dn = 1` on the boundary (and `phi = dphi/dn = 0` on parallel special boundaries), which
  removes the tangential traction (pressure / symmetry / roller), the normal displacement (symmetry), or
  the Robin residual of a constant `T_inf` (convection), without changing the trace.
* **Independent self-check** (`mms.self_check`, `test_mms_self_check`): the case data *as the solver
  sees it* (strings evaluated by `eval_expr`) is checked against the exact fields with central finite
  differences and an independent numpy constitutive law (plane stress via the 3D law with s_zz = 0).
  All 44 problems: residuals <= 3e-7 (finite-difference accuracy).
* **Domains** (gmsh, named groups): unit square; quarter annulus 0.5 < r < 1 (curved arcs: tests the
  isoparametric P2 geometry and normals of curved facets); unit cube; quarter annular cylinder (3D
  curved faces); axisymmetric ring (r in [0.5, 1.5]) and solid (r in [0, 1], touching the axis; the
  exact fields are regular there: u_r odd, u_z and T even in r).
* **Refinement**: unstructured base mesh (h0 = 0.25 in 2D, 0.5 in 3D) uniformly refined by gmsh
  (new boundary nodes on the CAD curves / surfaces): nested, ratio exactly 2; 2D 4 levels, 3D P1 4 /
  P2 3 levels; P2 meshes are tri6 / tet10 (curved boundaries). Independently remeshed unstructured
  sequences were tried first: their pairwise orders scatter by +-0.4 (e.g. 3.02, 2.88, 3.22 for a P2
  temperature), which makes a finest-pair criterion meaningless, so the nested family is the default.
* **Error measures**: relative RMS error over 2000 (2D) / 2500 (3D) fixed scrambled-Halton points
  covering the exact domain (same points for every mesh; points just outside a discrete curved
  boundary are evaluated in the nearest cell by `Result.probe`): displacement (ux, uy[, uz]) or T;
  recovered stress (sxx, syy, sxy[, szz plane strain][, stt axisymmetric][, syz, sxz]) or flux.
  Energy: `Result.evaluate(strain_energy)` vs the exact `0.5 a(u, u)`.
* **Energy order with corners**: where a Dirichlet face meets a traction / pressure face at 90 degrees
  the *dual* problem has an r^lambda corner singularity (Williams: lambda = 0.711 plane strain /
  axisymmetric / 3D edges, 0.758 plane stress at nu = 0.3), which caps the energy order at
  k + 1 + lambda = 3.71 for P2 even though the exact solution is smooth. Observed: an axisymmetric case
  stalls at 3.67-3.70 over five levels, and the *same* field and BC layout in plane strain stalls at
  3.71, while all-Dirichlet on the same domain gives 4.00: a property of the problem, not the solver.
* **Patch tests**: linear displacement / temperature fields (constant data only, or expression
  Dirichlet data for the general variant; uniform temperature for thermal strain), all BC types incl.
  `force`, `symmetry`, `convection`.
* **Skips**: a backend raising `NotImplementedError` (or not importable) is skipped, not failed.
"""

FINDINGS = """## Findings

1. **skfem: no defect found.** All 2D elasticity / heat studies (plane stress, plane strain,
   thermal strain, axisymmetric away from and touching the axis, curved boundaries, every BC type with
   expression data) converge at the theoretical orders: u ~3.0 (P2) / ~2.0 (P1), recovered (L2-projected)
   stress / flux ~2.0 (P2) / ~1.5 (P1, superconvergent), energy ~4 (P2) / 2 (P1). Patch tests pass to
   1e-13 on distorted meshes with curved interior P2 edges. See the table for the 3D results.
2. **`feacore.case.eval_expr` raises `KeyError('__import__')`** instead of returning inf / nan whenever
   numpy would emit a RuntimeWarning (division by zero, sqrt of a negative, overflow), because the eval
   runs with empty builtins and the warnings machinery needs `__import__`. Repro:
   `eval_expr("1/x", [0.0], [0.0])`. Any expression evaluated on the axis (r = 0) or outside its domain
   crashes the solve with a confusing error. Fix: wrap the eval in `np.errstate(all="ignore")` (or
   report non-finite values explicitly). The MMS expressions here are regularised so they are finite on
   the axis.
3. **ccx axisymmetric is not exact on the patch test** (2e-4 relative for P1, 1e-3 / 6e-3 in u / stress for P2,
   independent of the BC type; skfem: 1e-14). The P1 magnitude matches CalculiX expanding axisymmetric
   elements into a thin 3D wedge (chord error 1 - cos(1 deg) = 1.5e-4). This floors every ccx
   axisymmetric study, so ccx is a weak cross-check for axisymmetric problems at tight tolerances.
4. **ccx P2 3D (C3D10) fails the patch test on curved tets** (6e-5 u, 1.4e-4 stress), but passes
   (2e-6, output precision) on the same mesh with straight edges: its integration rule is too weak
   for curved tet10 cells. The error vanishes with refinement, but it is a consistency error on the
   curved boundary cells of gmsh meshes.
5. **ccx output precision**: results come through the ASCII `.frd` (~6 significant digits), so ccx
   cannot be compared below ~1e-6 relative (patch tolerance 5e-5; order floor 3e-6).
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="skfem,ccx")
    ap.add_argument("--quick", action="store_true", help="skip the 3D (slow) convergence studies")
    ap.add_argument("--only", default="", help="comma-separated problem ids (convergence or patch)")
    ap.add_argument("--workdir", default=str(mms.DEFAULT_WORKDIR))
    ap.add_argument("--out", default=str(ROOT / "docs" / "code_verification.md"))
    ap.add_argument("--img", default=str(ROOT / "docs" / "img" / "code_verification"))
    ap.add_argument("--from-json", default=None, help="re-render the report from a saved results.json")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    backends = [b for b in args.backends.split(",") if b]
    img_dir = Path(args.img)
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        res = json.loads(Path(args.from_json).read_text())
    else:
        import gmsh
        import skfem
        import sympy

        t0 = time.time()
        res = run_matrix(backends, args.quick, set(filter(None, args.only.split(","))), args.workdir,
                         not args.quiet)
        res["meta"] = {"date": str(date.today()), "python": f"Python {platform.python_version()}",
                       "sympy": sympy.__version__, "skfem": getattr(skfem, "__version__", "?"),
                       "gmsh": gmsh.__version__, "seconds": time.time() - t0, "quick": args.quick}
        (img_dir / "results.json").write_text(json.dumps(res, indent=1, default=float))
    for pid in dict.fromkeys(e["problem"] for e in res["convergence"]):
        plot_problem(pid, [e for e in res["convergence"] if e["problem"] == pid], img_dir)
    out = write_markdown(res, Path(args.out), img_dir, backends)
    n_fail = sum(e.get("verdict") == "fail" for e in res["convergence"] + res["patch"])
    print(f"wrote {out} ({n_fail} failing entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
