"""Physical test data as evidence for the loop (e.g. Instron pulls of a printed gold part).

A test is a small JSON spec next to the raw file the testing machine exported (Bluehill CSV):

  {
    "test_id": "abc3-pull-01",
    "problem_id": "abc_00000003_pull",          the loop problem whose setup is this test
    "description": "Rigid 10K, Form 3, 100 um, UV cure 60 min at 70 C; V-jaw wedge grips over 40 mm; ...",
    "data": {"file": "abc3_pull_01.csv",          relative to the spec
             "load": "Load", "displacement": "Extension (Video)",   column names as in the header
             "load_unit": "N", "displacement_unit": "mm"},          optional: read from the header
             (or "strain": "<column>" with "gauge_length_mm": L0 instead of "displacement")
    "measurement": {"qoi": "ext_marks"},          a QoI of the problem (default: its scored QoI), or an
                                                  explicit one: {"type": "point_difference", "field": "ux",
                                                  "point": [...], "point2": [...]} in metres
    "fit": {"window": [0.1, 0.9]},                share of each loading's peak load used for the slope
    "uncertainty": {"material": 0.05, "geometry": 0.05, "instrument": 0.01},   relative, 1 sigma
    "material": {"E": 9.7e9, "source": "3 D638 dogbones, same batch"}          measured modulus
  }

What the loop does with it (ace.run_loop --evidence DIR, or files dropped into <out>/evidence/
while it runs, so data that arrives after the population has settled is still used):
  1. reads the raw data, splits it into load cycles and fits the compliance (measurement per
     newton) on each loading, with an uncertainty budget (scatter between cycles, fit, instrument,
     material, geometry) and checks of linearity and of what is left after unloading;
  2. turns every run's scored solve into a prediction of the same measurement (the scored value
     when the measurement is the scored QoI, else a re-solve of the run's mesh and setup), scaled
     to the measured modulus when the spec gives one, and compares it with the test;
  3. computes the population's consensus (median of the latest generation's predictions) and how
     far it is from the test, in percent and in standard deviations;
  4. shows the test and that comparison to the agents (task prompt, and a test_check in every solve
     result), and to the teachers (diagnoser, reflector, deep teacher), who explain the gap and
     write lessons;
  5. with --evidence-score, scores runs against the test instead of the problem's reference
     (tolerance at least twice the test's combined uncertainty), and re-scores earlier runs.
Check a test before feeding it to the loop: python -m ace.evidence summarize SPEC [--predict].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ace import config
from ace.config import TASKS

ACTIVE_ENV = "ACE_EVIDENCE_DIR"      # processed evidence that agents, scorers and teachers read
SCORE_ENV = "ACE_EVIDENCE_SCORE"     # "1": score runs against the test
LOAD_UNITS = {"n": 1.0, "kn": 1e3, "mn": 1e-3, "lbf": 4.4482216152605, "kgf": 9.80665}
DISP_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "micron": 1e-6, "in": 0.0254}
CONSISTENT_SIGMA = 2.0               # |prediction - test| within this many sigma counts as consistent


# --------------------------------------------------------------------------- reading the raw data


def _match(cell: str, name: str) -> bool:
    c, n = cell.strip().strip('"').lower(), name.strip().lower()
    return c == n or c.startswith(n + " ") or c.startswith(n + "(")


def _unit_in(text: str) -> str | None:
    if "(" in text and ")" in text:
        return text[text.rindex("(") + 1:text.rindex(")")].strip().lower()
    return None


def _unit_after(cell: str, name: str) -> str | None:
    """The unit written after a column's name in its header cell ("Load (N)"), not brackets that are
    part of the name itself ("Extension (Video)")."""
    c = cell.strip().strip('"')
    return _unit_in(c[len(name):] if c.lower().startswith(name.strip().lower()) else c)


def _number(s: str, decimal_comma: bool) -> float | None:
    s = s.strip().strip('"')
    if decimal_comma:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _delimiter(lines: list[str]) -> str:
    """The delimiter that splits the most lines into the same number (>= 2) of fields. Counting
    characters fails on exports with decimal commas and ';' between fields."""
    best, best_score = ",", -1
    for d in ("\t", ";", ","):
        counts = [len(r) for r in csv.reader(lines[:80], delimiter=d) if len(r) > 1]
        if counts:
            modal = max(set(counts), key=counts.count)
            if counts.count(modal) > best_score:
                best, best_score = d, counts.count(modal)
    return best


def read_table(path: Path, data: dict) -> dict:
    """Load (N) and measurement (m) columns from a machine export (Bluehill CSV or similar): finds
    the header row by the column names, takes units from the header or a units row below it."""
    lines = Path(path).read_text(errors="replace").splitlines()
    delim = _delimiter(lines)
    rows = list(csv.reader(lines, delimiter=delim))
    disp_name = data.get("displacement") or data.get("strain")
    if not data.get("load") or not disp_name:
        raise ValueError("data needs 'load' and 'displacement' (or 'strain') column names")
    for h, row in enumerate(rows):
        il = next((i for i, c in enumerate(row) if _match(c, data["load"])), None)
        idisp = next((i for i, c in enumerate(row) if _match(c, disp_name)), None)
        if il is not None and idisp is not None:
            break
    else:
        raise ValueError(f"{path}: no header row with columns {data['load']!r} and {disp_name!r}")
    load_unit, disp_unit = _unit_after(rows[h][il], data["load"]), _unit_after(rows[h][idisp], disp_name)
    start = h + 1
    if start < len(rows) and len(rows[start]) > max(il, idisp) and _number(rows[start][il], False) is None:
        load_unit = load_unit or _unit_in(rows[start][il]) or rows[start][il].strip("() \"").lower()
        disp_unit = disp_unit or _unit_in(rows[start][idisp]) or rows[start][idisp].strip("() \"").lower()
        start += 1
    load_unit = (data.get("load_unit") or load_unit or "n").lower()
    decimal_comma = delim == ";" and any("," in c for r in rows[start:start + 20] for c in r)
    load, meas = [], []
    for row in rows[start:]:
        if len(row) <= max(il, idisp):
            continue
        a, b = _number(row[il], decimal_comma), _number(row[idisp], decimal_comma)
        if a is not None and b is not None:
            load.append(a)
            meas.append(b)
    if len(load) < 10:
        raise ValueError(f"{path}: only {len(load)} numeric rows under the header")
    if load_unit not in LOAD_UNITS:
        raise ValueError(f"unknown load unit {load_unit!r}; give data.load_unit ({sorted(LOAD_UNITS)})")
    load = np.array(load) * LOAD_UNITS[load_unit]
    meas = np.array(meas)
    if data.get("strain"):
        scale = 0.01 if (data.get("strain_unit") or disp_unit or "").strip() == "%" else 1.0
        meas = meas * scale * float(data["gauge_length_mm"]) * 1e-3
        disp_unit = "strain"
    else:
        disp_unit = (data.get("displacement_unit") or disp_unit or "mm").lower()
        if disp_unit not in DISP_UNITS:
            raise ValueError(f"unknown displacement unit {disp_unit!r}; give data.displacement_unit")
        meas = meas * DISP_UNITS[disp_unit]
    return {"load": load, "meas": meas, "units": {"load": load_unit, "displacement": disp_unit}}


# --------------------------------------------------------------------------- fitting


def loading_segments(load: np.ndarray, hysteresis: float = 0.05) -> list[tuple[int, int]]:
    """(start, peak) index pairs of each loading, found with a hysteresis of `hysteresis` x the
    largest load, so noise does not split a loading in two."""
    thr = hysteresis * float(np.max(load))
    segs, i_min, i_max, rising = [], 0, 0, True
    for i, value in enumerate(load):
        if rising:
            if value >= load[i_max]:
                i_max = i
            elif value < load[i_max] - thr:
                segs.append((i_min, i_max))
                rising, i_min = False, i
        else:
            if value <= load[i_min]:
                i_min = i
            elif value > load[i_min] + thr:
                rising, i_max = True, i
    if rising and i_max > i_min:
        segs.append((i_min, i_max))
    peak = float(np.max(load))
    return [(a, b) for a, b in segs if b - a >= 8 and load[b] >= 0.3 * peak]


def _line(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (c, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    resid = y - (c * x + b)
    s2 = float(resid @ resid) / max(n - 2, 1)
    sxx = float(((x - x.mean()) ** 2).sum())
    stderr = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return float(c), float(b), stderr, r2


def fit_cycles(load: np.ndarray, meas: np.ndarray, window=(0.1, 0.9)) -> list[dict]:
    """Slope of measurement vs load on each loading, between window[0] and window[1] of its peak."""
    out = []
    segs = loading_segments(load)
    for k, (a, b) in enumerate(segs):
        L, d = load[a:b + 1], meas[a:b + 1]
        sel = (L >= window[0] * L[-1]) & (L <= window[1] * L[-1])
        if sel.sum() < 6:
            continue
        x, y = L[sel], d[sel]
        c, b0, stderr, r2 = _line(x, y)
        half = np.median(x)
        lo, hi = x <= half, x >= half
        nonlin = nonlin_t = None
        if lo.sum() >= 3 and hi.sum() >= 3 and c:
            (c_hi, _, se_hi, _), (c_lo, _, se_lo, _) = _line(x[hi], y[hi]), _line(x[lo], y[lo])
            nonlin = (c_hi - c_lo) / c                    # upper-half slope vs lower-half slope
            nonlin_t = (c_hi - c_lo) / math.hypot(se_hi, se_lo) if se_hi + se_lo > 0 else None
        nxt = segs[k + 1][0] if k + 1 < len(segs) else len(load) - 1
        residual = float(meas[nxt] - meas[a]) if load[nxt] <= 0.2 * L[-1] else None
        out.append({"cycle": k + 1, "peak_load": float(L[-1]), "compliance": c, "stderr": stderr, "r2": r2,
                    "nonlinearity": nonlin, "nonlinearity_t": nonlin_t, "residual_after_unload": residual,
                    "points": int(sel.sum())})
    return out


def _downsample(load, meas, n=60) -> list[list[float]]:
    idx = np.unique(np.linspace(0, len(load) - 1, min(n, len(load))).astype(int))
    return [[round(float(load[i]), 4), float(f"{meas[i]:.6g}")] for i in idx]


# --------------------------------------------------------------------------- one test


def read_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for key in ("problem_id", "data"):
        if key not in spec:
            raise ValueError(f"{path}: test spec needs {key!r}")
    spec.setdefault("test_id", Path(path).stem)
    spec["_dir"] = str(Path(path).resolve().parent)
    return spec


def nominal_load(problem, spec: dict) -> float:
    """The load the problem's statement applies (its 'force' parameter) or the spec's model_load."""
    F = spec.get("model_load") or problem.params.get("force")
    if not F:
        raise ValueError(f"{problem.id}: give model_load (N) in the test spec: the problem has no 'force'")
    return float(F)


def measurement_qoi(problem_id: str, spec: dict):
    from ace import fea
    from feacore.case import QoI
    m = spec.get("measurement") or {}
    problem = fea.get_problem(problem_id)
    name = m.get("qoi") or (TASKS[problem_id]["qoi"] if not m.get("type") else None)
    if name:
        return next(q for q in problem.qois() if q.name == name)
    return QoI(name="test_measurement", type=m["type"], field=m.get("field"), point=m.get("point"),
               point2=m.get("point2"), group=m.get("group"), component=m.get("component"))


def process_test(spec: dict) -> dict:
    """Raw data -> measured compliance with its uncertainty budget, cycles and a short curve."""
    from ace import fea
    pid = spec["problem_id"]
    problem = fea.get_problem(pid)
    data_path = Path(spec["_dir"]) / spec["data"]["file"]
    table = read_table(data_path, spec["data"])
    fit = spec.get("fit") or {}
    window = tuple(fit.get("window", (0.1, 0.9)))
    if fit.get("until") == "peak":      # a test run to failure: the loading up to the peak only
        k = int(np.argmax(table["load"])) + 1
        table = {**table, "load": table["load"][:k], "meas": table["meas"][:k]}
    cycles = fit_cycles(table["load"], table["meas"], window)
    if not cycles:
        raise ValueError(f"{data_path}: no loading with enough points in the fit window {window}")
    used, dropped = cycles, []
    if len(cycles) >= 3:        # a first loading that differs is usually seating in the grips
        rest = float(np.median([c["compliance"] for c in cycles[1:]]))
        if abs(cycles[0]["compliance"] - rest) > 0.02 * abs(rest):
            used, dropped = cycles[1:], [1]
    cs = np.array([c["compliance"] for c in used])
    c = float(cs.mean())
    u_stat = float(cs.std(ddof=1) / math.sqrt(len(cs))) if len(cs) > 1 else used[0]["stderr"]
    u_fit = float(np.mean([x["stderr"] for x in used]))
    unc = spec.get("uncertainty") or {}
    budget = {"between_cycles": abs(u_stat / c), "fit": abs(u_fit / c), "instrument": float(unc.get("instrument", 0.01)),
              # a measured modulus (dogbones) still has its own scatter; the datasheet value is looser
              "material": float(unc.get("material", 0.02 if (spec.get("material") or {}).get("E") else 0.05)),
              "geometry": float(unc.get("geometry", 0.05))}
    E_test = (spec.get("material") or {}).get("E")
    E_model = (problem.params or {}).get("E")
    F = nominal_load(problem, spec)
    q = measurement_qoi(pid, spec)
    raw = Path(data_path).read_bytes()
    return {
        "test_id": spec["test_id"], "problem_id": pid, "description": spec.get("description", ""),
        "file": str(data_path), "sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(table["load"])),
        "units": table["units"], "peak_load": float(np.max(table["load"])), "fit_window": list(window),
        "cycles": cycles, "cycles_used": [x["cycle"] for x in used], "cycles_dropped": dropped,
        "compliance": c, "budget": budget,
        "u_rel_measurement": math.sqrt(budget["between_cycles"] ** 2 + budget["fit"] ** 2 + budget["instrument"] ** 2),
        "u_rel_total": math.sqrt(sum(v ** 2 for v in budget.values())),
        "nonlinearity": max((abs(x["nonlinearity"]) for x in used if x["nonlinearity"] is not None), default=None),
        # significant: the two half-window slopes differ by more than 3 of their standard errors
        "nonlinearity_significant": any(abs(x["nonlinearity_t"]) > 3 for x in used if x["nonlinearity_t"] is not None),
        "residual_after_unload": max((abs(x["residual_after_unload"]) for x in used
                                      if x["residual_after_unload"] is not None), default=None),
        "nominal_load": F, "value_at_nominal_load": c * F,
        # the machine reads compression as positive, the FEA quantity is negative: sign -1
        "sign": float((spec.get("measurement") or {}).get("sign", 1.0)),
        "notes": spec.get("notes", ""),
        "e_scale": (E_model / E_test) if (E_test and E_model) else 1.0, "E_test": E_test, "E_model": E_model,
        "measurement": {"name": q.name, "type": q.type, "field": q.field, "point": q.point,
                        "point2": getattr(q, "point2", None), "group": q.group},
        "curve": _downsample(table["load"], table["meas"]),
    }


def combine(tests: list[dict]) -> dict:
    """Several tests of one problem: inverse-variance mean of the compliances; the material and
    geometry parts of the uncertainty are shared, so they are not averaged down."""
    w = np.array([1 / (t["u_rel_measurement"] * t["compliance"]) ** 2 for t in tests])
    c = float((w * np.array([t["compliance"] for t in tests])).sum() / w.sum())
    u_meas = float(1 / math.sqrt(w.sum())) / abs(c)
    shared = max(math.hypot(t["budget"]["material"], t["budget"]["geometry"]) for t in tests)
    F = tests[0]["nominal_load"]
    return {"compliance": c, "u_rel_measurement": u_meas, "u_rel_total": math.hypot(u_meas, shared),
            "nominal_load": F, "value_at_nominal_load": c * F, "e_scale": tests[0]["e_scale"],
            "sign": tests[0].get("sign", 1.0), "n_tests": len(tests), "measurement": tests[0]["measurement"]}


# --------------------------------------------------------------------------- predictions


def _resolve(problem_id: str, run: dict, q) -> float | None:
    """Evaluate measurement QoI q on the run's scored solve (rebuilding its mesh if the file is gone)."""
    from ace import fea
    from ace.checkpoint import checkpoint
    from ace.field_error import _solve
    cp = checkpoint(run)
    if cp is None:
        return None
    problem = fea.get_problem(problem_id)
    with tempfile.TemporaryDirectory() as d:
        mesh = dict(cp["mesh"])
        if not Path(mesh.get("path", "")).exists():
            if not cp.get("request"):
                return None
            mesh = fea.make_mesh(problem, Path(d) / "m.msh", **{k: v for k, v in cp["request"].items()
                                                                  if k in ("h", "order", "refinements")})
            if not mesh.get("ok"):
                return None
        res = _solve(problem, mesh, cp["setup"], TASKS[problem_id]["qoi"])
        return float(res.evaluate_qoi(q))


def compare_value(value: float, ev: dict) -> dict:
    """A run's predicted measurement at the nominal load vs the test (`ev`: combined evidence)."""
    pred = ev.get("sign", 1.0) * value * ev["e_scale"] / ev["nominal_load"]
    rel = (pred - ev["compliance"]) / ev["compliance"]
    sigma = rel / ev["u_rel_total"] if ev["u_rel_total"] else float("inf")
    return {"value": value, "predicted_compliance": pred, "rel_err": rel, "sigma": sigma,
            "consistent": abs(sigma) <= CONSISTENT_SIGMA}


def predict_run(problem_id: str, run: dict, ev: dict) -> dict | None:
    from ace import nonlinear
    from ace.checkpoint import scored_solve_index
    if nonlinear.is_nonlinear(problem_id):    # scored on the whole curve (ace.nonlinear), not a compliance
        return None
    i = scored_solve_index(run)
    if i is None:
        return None
    s = run["solves"][i]
    if s.get("test_check") is not None:
        value = s["test_check"]["value"]
    elif ev["measurement"]["name"] == TASKS[problem_id]["qoi"]:
        value = s["result"]["value"]
    else:
        value = _resolve(problem_id, run, measurement_qoi(problem_id, {"measurement": ev["measurement_spec"]}))
        if value is None:
            return None
    return compare_value(value, ev)


def consensus(problem_id: str, runs: list[dict], ev: dict) -> dict | None:
    """Median prediction of the latest generation that has runs with a prediction."""
    preds = {}
    for r in runs:
        if r.get("error"):
            continue
        p = (r.get("checks") or {}).get("evidence") or predict_run(problem_id, r, ev)
        if p:
            preds.setdefault(r["generation"], []).append(p["predicted_compliance"])
    if not preds:
        return None
    gen = max(preds)
    v = np.array(preds[gen])
    med = float(np.median(v))
    return {"generation": gen, "runs": int(len(v)), "median_compliance": med,
            "spread_rel": float(np.median(np.abs(v - med)) / abs(med)) if med else None,
            **{k: x for k, x in compare_value(med * ev["nominal_load"] / ev["e_scale"] / ev.get("sign", 1.0), ev).items()
               if k != "value"},
            "trend": {g: float(np.median(x)) for g, x in sorted(preds.items())}}


# --------------------------------------------------------------------------- the active store


def active_dir() -> Path | None:
    d = os.environ.get(ACTIVE_ENV)
    return Path(d) if d else None


def load_active(problem_id: str) -> dict | None:
    d = active_dir()
    if d is None or not (d / f"{problem_id}.json").exists():
        return None
    return json.loads((d / f"{problem_id}.json").read_text())


def scoring_on() -> bool:
    return os.environ.get(SCORE_ENV) == "1"


def has_reference(problem_id: str) -> bool:
    """Does the problem have a reference value for its scored quantity (a ground truth)?"""
    from ace import fea
    qoi = TASKS[problem_id]["qoi"]
    return any(r.qoi == qoi for r in fea.get_problem(problem_id).references())


def scored_by_test(problem_id: str) -> bool:
    """Is this problem scored against its physical test? With --evidence-score, and always when the
    problem has no reference value (no ground truth: the test is the only check)."""
    return load_active(problem_id) is not None and (scoring_on() or not has_reference(problem_id))


def missing_tests(problem_ids: list[str]) -> list[str]:
    """Problems with neither a reference value nor test data: nothing can score them."""
    from ace import nonlinear
    return [p for p in problem_ids if not has_reference(p) and load_active(p) is None and not nonlinear.is_nonlinear(p)]


def automesher_test_error(problem_id: str) -> float:
    """Field mode, scored against the test: the automesher's own error against the test (its solve
    of the stated setup), the baseline a run's gain is measured from."""
    from ace.field_error import automesher
    ev = load_active(problem_id)["combined"]
    return abs(compare_value(automesher(problem_id, need_value=True)["value"], ev)["rel_err"])


def test_tolerance(problem_id: str) -> float:
    """Accuracy target when scored against the test: no tighter than 2 sigma of the test. In field
    mode also the automesher's own error divided by the target gain (as for a reference)."""
    from ace.config import AUTO_GAIN_TARGET, SCORE
    ev = load_active(problem_id)["combined"]
    base = automesher_test_error(problem_id) / AUTO_GAIN_TARGET if SCORE == "field" else 0.02
    return max(base, 2 * ev["u_rel_total"])


def value_error(problem_id: str, value: float) -> float | None:
    """|error| against the test of a solve whose scored value is `value`, when the test measures the
    scored quantity (else None: it needs a re-solve, see predict_run)."""
    rec = load_active(problem_id)
    if rec is None or rec["combined"]["measurement"]["name"] != TASKS[problem_id]["qoi"]:
        return None
    return abs(compare_value(value, rec["combined"])["rel_err"])


def discover(paths: list[Path]) -> list[Path]:
    """Test specs: JSON files with problem_id and data, in the given files or directories."""
    out = []
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p] if p.exists() else []
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and "problem_id" in d and "data" in d:
                out.append(f)
    return out


def refresh(out_dir: Path, sources: list[Path], all_runs: dict[str, list[dict]], generation: int) -> list[str]:
    """Process every test spec in `sources` and <out_dir>/evidence/ for the loop's problems and write
    the active store (<out_dir>/evidence_active/<problem_id>.json) that agents, scorers and teachers
    read. Returns the problems whose evidence is new or changed."""
    active = Path(out_dir) / "evidence_active"
    active.mkdir(parents=True, exist_ok=True)
    by_problem: dict[str, list[dict]] = {}
    for f in discover([Path(out_dir) / "evidence", *sources]):
        spec = read_spec(f)
        if spec["problem_id"] not in all_runs:
            print(f"  evidence {f}: problem {spec['problem_id']} is not in this run; ignored")
            continue
        try:
            t = process_test(spec)
        except (ValueError, KeyError, OSError) as e:
            print(f"  evidence {f}: could not be read: {e}")
            continue
        t["measurement_spec"] = spec.get("measurement") or {}
        by_problem.setdefault(spec["problem_id"], []).append(t)
    changed = []
    for pid, tests in by_problem.items():
        ev = {**combine(tests), "measurement_spec": tests[0]["measurement_spec"]}
        prev = active / f"{pid}.json"
        key = hashlib.sha256(json.dumps([[t["sha256"], t["test_id"], t["compliance"]] for t in tests]).encode()).hexdigest()
        old = json.loads(prev.read_text()) if prev.exists() else {}
        cons = consensus(pid, all_runs[pid], ev)
        record = {"problem_id": pid, "key": key, "since_generation": old.get("since_generation", generation)
                  if old.get("key") == key else generation, "combined": ev, "tests": tests, "consensus": cons}
        prev.write_text(json.dumps(record, indent=1, default=str))
        if old.get("key") != key:
            changed.append(pid)
            msg = (f"  evidence for {pid}: {len(tests)} test(s), measured {ev['value_at_nominal_load']:.4g} at "
                   f"{ev['nominal_load']:g} N (+-{ev['u_rel_total']:.1%})")
            if cons:
                msg += (f"; the population's consensus (generation {cons['generation']}) is "
                        f"{cons['rel_err']:+.1%} from it ({cons['sigma']:+.1f} sigma)")
            print(msg)
    return changed


# --------------------------------------------------------------------------- what LLMs read


def _shown_id(problem_id: str) -> str:
    if config.STATEMENT == "requirements":
        from ace.requirements import label_for
        return label_for(problem_id)
    return problem_id


def _scrub(text: str, problem_id: str) -> str:
    return text.replace(problem_id, _shown_id(problem_id))


def agent_section(problem_id: str) -> str:
    """The test as an agent reads it (empty when the problem has no evidence)."""
    rec = load_active(problem_id)
    if not rec:
        return ""
    ev, cons = rec["combined"], rec.get("consensus")
    m = ev["measurement"]
    lines = ["PHYSICAL TEST DATA (measured on the real part; treat it as the truth within its uncertainty)"]
    for t in rec["tests"]:
        lines.append(f"- test {t['test_id']}: {_scrub(t['description'], problem_id)}".rstrip(": "))
        if t.get("notes"):
            lines.append(f"  lab notes: {_scrub(t['notes'], problem_id)}")
    from ace import nonlinear
    if nonlinear.is_nonlinear(problem_id):    # scored on the force-displacement curve, not a compliance
        nm = nonlinear.measured()
        lines.append("Measured force at the scoring checkpoints (crosshead travel): " + ", ".join(
            f"{u:g} mm: {f:.0f} N" for u, f in zip(nm["checkpoints_mm"], nm["force_N"])) + ". Runs are scored by "
            "one frozen elastoplastic ramp of the submitted mesh against these (mean relative force error).")
        return "\n".join(lines)
    where = (f"between {m['point2']} and {m['point']}" if m.get("point2") else
             f"at {m['point']}" if m.get("point") else f"over group {m.get('group')!r}" if m.get("group") else "")
    lines.append(f"Measured: {m['name']} ({m['type']} of {m['field']} {where}) = {ev['value_at_nominal_load']:.4g} at "
                 f"the statement's {ev['nominal_load']:g} N, i.e. {ev['compliance']:.4g} per newton, "
                 f"+-{ev['u_rel_total']:.1%} (1 sigma: scatter, fit, instrument, material, geometry).")
    t0 = rec["tests"][0]
    if t0.get("nonlinearity") is not None:
        lines.append(f"The load-measurement curve bends by {t0['nonlinearity']:.1%} over the fitted range (slope of "
                     "the upper half vs the lower half): " + ("significant, so the part is not behaving linearly."
                     if t0.get("nonlinearity_significant") else "within the noise, so linear."))
    if ev.get("e_scale", 1.0) != 1.0:
        lines.append(f"The printed material's measured modulus is {t0['E_test']:.4g} Pa; predictions are compared "
                     f"after scaling to it (factor {ev['e_scale']:.4g} on the statement's modulus).")
    if cons:
        lines.append(f"The population's consensus so far (generation {cons['generation']}, {cons['runs']} runs) predicts "
                     f"{cons['median_compliance'] * ev['nominal_load']:.4g}: {cons['rel_err']:+.1%} from the test "
                     f"({cons['sigma']:+.1f} sigma).")
    lines.append("If your setup's prediction differs from the test by more than its uncertainty, find the modelling "
                 "choice that explains it (how the part is held and loaded, material, geometry, mesh). Every solve "
                 "result reports test_check: your prediction against the test.")
    return "\n".join(lines)


def teacher_section(problem_id: str, runs: list[dict] | None = None) -> str:
    """The agent's view plus the cycles, the curve, the budget and every run's prediction."""
    rec = load_active(problem_id)
    if not rec:
        return ""
    ev = rec["combined"]
    out = [agent_section(problem_id), "", "Test details for the teacher:"]
    for t in rec["tests"]:
        out.append(f"* {t['test_id']}: {t['rows']} rows, peak {t['peak_load']:.4g} N, fit window {t['fit_window']}, "
                   f"cycles used {t['cycles_used']}" + (f" (dropped {t['cycles_dropped']}: seating)" if t['cycles_dropped'] else ""))
        for c in t["cycles"]:
            out.append(f"    cycle {c['cycle']}: peak {c['peak_load']:.4g} N, compliance {c['compliance']:.5g} (+-{c['stderr']:.2g}), "
                       f"R2 {c['r2']:.5f}" + (f", nonlinearity {c['nonlinearity']:+.2%}" if c['nonlinearity'] is not None else "")
                       + (f", left after unloading {c['residual_after_unload']:.3g}" if c['residual_after_unload'] is not None else ""))
        out.append("    uncertainty budget (relative, 1 sigma): " + ", ".join(f"{k} {v:.1%}" for k, v in t["budget"].items()))
        out.append("    curve (load N, measurement m): " + " ".join(f"{a:g}:{b:.4g}" for a, b in t["curve"][::3]))
    preds = []
    for r in runs or []:
        p = (r.get("checks") or {}).get("evidence")
        if p:
            preds.append(f"    {r['run_id']}: {p['rel_err']:+.1%} ({p['sigma']:+.1f} sigma)")
    if preds:
        out += ["Each run's prediction against the test:", *preds]
    out.append("Explain the gap between the runs and the test. Separate modelling causes (supports and grips, load "
               "introduction, material, geometry, mesh) from the test's own uncertainty, and write lessons that would "
               f"bring the prediction within {CONSISTENT_SIGMA:g} sigma of tests like this one.")
    return _scrub("\n".join(out), problem_id)


def solve_check(problem_id: str, feacore_result) -> dict | None:
    """test_check for one solve: the solve's prediction of the measurement against the test."""
    from ace import nonlinear
    rec = load_active(problem_id)
    if not rec or feacore_result is None or nonlinear.is_nonlinear(problem_id):   # nonlinear: no linear check
        return None
    ev = rec["combined"]
    q = measurement_qoi(problem_id, {"measurement": ev.get("measurement_spec") or {"qoi": ev["measurement"]["name"]}})
    value = float(feacore_result.evaluate_qoi(q))
    return compare_value(value, ev)


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="read a test spec and its raw data; print the fit")
    s.add_argument("spec", type=Path)
    s.add_argument("--predict", action="store_true",
                   help="also solve the problem's own setup on a fine mesh and compare it with the test")
    s.add_argument("--h", type=float, help="mesh size for --predict (default: the problem's h0 / 2)")
    args = ap.parse_args(argv)
    spec = read_spec(args.spec)
    t = process_test(spec)
    print(json.dumps({k: v for k, v in t.items() if k != "curve"}, indent=1, default=str))
    if args.predict:
        from ace import fea
        from ace.checks import reference_setup
        from ace.field_error import _solve
        p = fea.get_problem(spec["problem_id"])
        with tempfile.TemporaryDirectory() as d:
            mesh = fea.make_mesh(p, Path(d) / "m.msh", h=args.h or p.params["h0"] / 2, order=2)
            res = _solve(p, mesh, reference_setup(p), TASKS[spec["problem_id"]]["qoi"])
            c = compare_value(float(res.evaluate_qoi(measurement_qoi(spec["problem_id"], spec))), t)
        print(f"problem's own setup ({mesh['nodes']} nodes): {c['predicted_compliance']:.5g} per N vs test "
              f"{t['compliance']:.5g}: {c['rel_err']:+.2%} ({c['sigma']:+.2f} sigma)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
