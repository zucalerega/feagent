"""Physical test data for the dashboard (#/testdata): the committed Instron tests, the printable test parts, and
uploads of new machine exports that become evidence for the ACE loop.

A test is an evidence spec (ace/evidence.py) next to the machine's export:

  committed   physical_tests/<test>/<spec>.json + the export it names (abc_00000065_compression/cload_compression.json)
  uploaded    <runs>/testdata/<id>/test.json + the export + summary.json (the parse, for the list; it has no
              problem_id, so the loop does not take it for a spec)

An uploaded record is a ready ``--evidence`` directory:

    python -m ace.evidence summarize <runs>/testdata/<id>/test.json       the loop's own reading of it
    python -m ace.run_loop --problems <problem> --evidence <runs>/testdata/<id> ...

Exports: what ace.evidence.read_table reads, because the loop reads them with it. Comma, semicolon or tab
separated (decimal commas with semicolons), a title block above a header row of column names, units in the
header ("Load (N)") or in a row under it ("(N)"); load in N, kN, mN, lbf or kgf; displacement in m, mm, um or
in, or a strain column with a gauge length. Bluehill CSV / TXT exports look like this.

API:
  GET    /api/testdata                       tests (committed + uploaded), problems that take test data, parts
  GET    /api/testdata/item?id=              one test: its spec, the loop's processing, the curve (mm, N), files
  POST   /api/testdata/parse                 multipart ``file`` (+ the fields below): columns, units, curve preview
  POST   /api/testdata                       multipart ``file`` + fields: save a record
  DELETE /api/testdata/item?id=              remove an uploaded record
  GET    /api/testdata/file?ref=&download=   a file of a test or a part (CSV, PDF, STEP, STL, PNG), path-checked

Form fields (parse and save): load, displacement | strain (column names as in the header), load_unit,
displacement_unit, gauge_length_mm, strain_unit; problem_id, test_id, specimen, part, description, notes,
sign (+1 tension, -1 compression), until_peak, window_lo, window_hi, E_GPa, E_source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Request          # module level: FastAPI resolves the routes' string annotations here

from feacore.ui.media import REPO_ROOT, RefError, file_response, kind_of, to_ref

TESTS_ROOT = REPO_ROOT / "physical_tests"
PARTS_DIR = TESTS_ROOT / "parts"
EXPORT_SUFFIXES = (".csv", ".txt", ".tsv", ".dat")
MAX_BYTES = 20 * 1024 * 1024
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
LOAD_WORDS = ("load", "force", "kraft")
DISP_WORDS = ("extension", "displacement", "deflection", "position", "travel", "elongation", "stroke", "crosshead",
              "compressive", "weg")


class ParseError(ValueError):
    def __init__(self, message: str, info: dict | None = None):
        super().__init__(message)
        self.info = info or {}


# --------------------------------------------------------------------------- reading an export


def _ev():
    from ace import evidence

    return evidence


def _clean(cell: str) -> str:
    return cell.strip().strip('"').strip()


def _unit_of(text: str, units: dict) -> str | None:
    """A known unit in brackets at the end of a header cell, or a units-row cell: '(kN)', 'mm'."""
    t = _clean(text)
    m = re.search(r"[(\[]\s*([^()\[\]]+?)\s*[)\]]\s*$", t)
    u = (m.group(1) if m else t).strip().lower()
    return u if u in units else None


def _is_units_row(row: list[str], known: set[str]) -> bool:
    """A row of units under the header: every non-empty cell bracketed ('(N)', '[mm]') or a known unit."""
    cells = [_clean(c) for c in row if c.strip()]
    return bool(cells) and all(re.fullmatch(r"\(.*\)|\[.*\]", c) or c.lower() in known for c in cells)


def sniff(path: Path) -> dict:
    """Delimiter, header row, units row and columns of an export (what the loop's reader will see)."""
    ev = _ev()
    lines = Path(path).read_text(errors="replace").splitlines()
    if not lines:
        raise ParseError("the file is empty")
    delim = ev._delimiter(lines)                          # noqa: SLF001 - the loop's own delimiter rule
    rows = list(csv.reader(lines, delimiter=delim))
    tail = [r for r in rows[-60:] if r]
    dc = delim == ";" and any("," in c for r in tail for c in r)

    def numeric(r):
        return sum(ev._number(c, dc) is not None for c in r if c.strip())   # noqa: SLF001

    def skip_blank(i):
        while i >= 0 and not any(c.strip() for c in rows[i]):
            i -= 1
        return i

    start = next((i for i in range(len(rows))
                  if all(numeric(rows[j]) >= 2 for j in range(i, min(i + 3, len(rows))))), None)
    if start is None:
        raise ParseError("no block of numeric rows (at least two numbers per row) in the file")
    known = {*ev.LOAD_UNITS, *ev.DISP_UNITS, "s", "sec", "min", "%", "mm/mm", "mpa", "-"}
    units_row, h = None, skip_blank(start - 1)
    if h >= 0 and numeric(rows[h]) == 0 and _is_units_row(rows[h], known):
        units_row, h = rows[h], skip_blank(h - 1)
    if h < 0 or numeric(rows[h]) >= len([c for c in rows[h] if c.strip()]):
        raise ParseError("no header row of column names above the numbers: the loop finds columns by name")
    header = rows[h]
    n = max(len(header), len(rows[start]))
    cols = []
    for k in range(n):
        name = _clean(header[k]) if k < len(header) else ""
        unit_cell = units_row[k] if units_row and k < len(units_row) else ""
        cols.append({"index": k, "name": name,
                     "load_unit": _unit_of(name, ev.LOAD_UNITS) or _unit_of(unit_cell, ev.LOAD_UNITS),
                     "disp_unit": _unit_of(name, ev.DISP_UNITS) or _unit_of(unit_cell, ev.DISP_UNITS),
                     "unit_text": _clean(unit_cell) or None})
    return {"delimiter": {",": "comma", ";": "semicolon", "\t": "tab"}.get(delim, delim), "decimal_comma": dc,
            "header_row": h + 1, "units_row": h + 2 if units_row else None, "first_data_row": start + 1,
            "columns": cols, "lines": len(lines)}


def _guess(cols: list[dict], words, unit_key: str) -> dict | None:
    named = [c for c in cols if c["name"]]
    for w in words:
        for c in named:
            if w in c["name"].lower():
                return c
    return next((c for c in named if c[unit_key]), None)


def _loop_column(cols: list[dict], name: str) -> int | None:
    """The column the loop's reader takes for a name (the first header cell that matches)."""
    ev = _ev()
    return next((c["index"] for c in cols if c["name"] and ev._match(c["name"], name)), None)   # noqa: SLF001


def choose(sn: dict, form: dict) -> tuple[dict, list[str]]:
    """The spec's ``data`` block: the chosen (or guessed) columns with explicit units, checked against how the
    loop's reader picks columns. Returns (data, warnings)."""
    cols, warnings = sn["columns"], []
    names = [c["name"] for c in cols]
    load = form.get("load") or (_guess(cols, LOAD_WORDS, "load_unit") or {}).get("name")
    strain = form.get("strain") or None
    disp = None if strain else (form.get("displacement") or (_guess(
        [c for c in cols if c["name"] != load], DISP_WORDS, "disp_unit") or {}).get("name"))
    if not load or load not in names:
        raise ParseError("choose the load column", {"sniff": sn})
    meas = strain or disp
    if not meas or meas not in names:
        raise ParseError("choose the displacement (or strain) column", {"sniff": sn})
    if meas == load:
        raise ParseError("load and displacement must be different columns", {"sniff": sn})
    for nm in (load, meas):
        k = _loop_column(cols, nm)
        if k != names.index(nm):
            raise ParseError(f"the loop's reader would take column {k + 1} ({names[k]!r}) for {nm!r}: rename one of "
                             "the two in the export", {"sniff": sn})
    lc, mc = cols[names.index(load)], cols[names.index(meas)]
    data = {"load": load, "load_unit": (form.get("load_unit") or lc["load_unit"] or "n").lower()}
    if not (form.get("load_unit") or lc["load_unit"]):
        warnings.append(f"no unit found for {load!r}: newtons assumed")
    if strain:
        gl = form.get("gauge_length_mm")
        if gl in (None, ""):
            raise ParseError("a strain column needs the gauge length (mm)", {"sniff": sn})
        data.update(strain=strain, gauge_length_mm=float(gl), strain_unit=form.get("strain_unit") or "")
    else:
        data.update(displacement=disp, displacement_unit=(form.get("displacement_unit") or mc["disp_unit"] or "mm").lower())
        if not (form.get("displacement_unit") or mc["disp_unit"]):
            warnings.append(f"no unit found for {disp!r}: millimetres assumed")
    ev = _ev()
    if data["load_unit"] not in ev.LOAD_UNITS:
        raise ParseError(f"unknown load unit {data['load_unit']!r}; one of {sorted(ev.LOAD_UNITS)}", {"sniff": sn})
    if not strain and data["displacement_unit"] not in ev.DISP_UNITS:
        raise ParseError(f"unknown displacement unit {data['displacement_unit']!r}; one of {sorted(ev.DISP_UNITS)}",
                         {"sniff": sn})
    return data, warnings


def _thin_curve(x: np.ndarray, y: np.ndarray, n: int) -> list[list[float]]:
    if len(x) <= n:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.concatenate([np.linspace(0, len(x) - 1, n).astype(int), [int(np.argmax(y))]]))
    return [[round(float(x[i]), 5), round(float(y[i]), 3)] for i in idx]


def _window(form: dict) -> tuple[float, float]:
    lo = float(form.get("window_lo") or 0.1)
    hi = float(form.get("window_hi") or 0.9)
    if not 0 <= lo < hi <= 1:
        raise ParseError("the fit window must be 0 <= low < high <= 1 (shares of each loading's peak)")
    return lo, hi


def curve_stats(load: np.ndarray, meas_m: np.ndarray, window, until_peak: bool) -> dict:
    """What the loop sees in the curve: peak, loadings, the fitted slope of each loading."""
    ev = _ev()
    ip = int(np.argmax(load))
    out = {"rows": int(len(load)), "peak_N": float(load[ip]), "disp_at_peak_mm": float(meas_m[ip] * 1e3),
           "last_mm": float(meas_m[-1] * 1e3), "negative_loads": bool(np.mean(load < 0) > 0.5)}
    L, d = (load[: ip + 1], meas_m[: ip + 1]) if until_peak else (load, meas_m)
    cyc = ev.fit_cycles(L, d, tuple(window))
    out["loadings"] = len(ev.loading_segments(L))
    out["cycles"] = [{"cycle": c["cycle"], "peak_N": c["peak_load"], "points": c["points"], "r2": c["r2"],
                      "stiffness_N_per_mm": (1e-3 / c["compliance"]) if c["compliance"] else None,
                      "nonlinearity": c["nonlinearity"]} for c in cyc]
    return out


def loop_view(spec_path: Path) -> dict:
    """The loop's processing of a spec (ace.evidence.process_test), or why it cannot use it."""
    ev = _ev()
    try:
        t = ev.process_test(ev.read_spec(spec_path))
    except Exception as e:  # noqa: BLE001 - shown to the user as the loop would report it
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    keep = ("test_id", "problem_id", "rows", "units", "peak_load", "fit_window", "cycles_used", "cycles_dropped",
            "compliance", "budget", "u_rel_total", "nonlinearity", "nonlinearity_significant", "nominal_load",
            "value_at_nominal_load", "sign", "measurement", "e_scale", "E_test", "E_model")
    out = {"ok": True, **{k: t.get(k) for k in keep}}
    try:
        from ace import nonlinear

        if nonlinear.is_nonlinear(t["problem_id"]):
            out["nonlinear_note"] = ("This problem is scored on the force-displacement curve of the frozen test in "
                                     "nonlinear_spec.json; a new test of it adds its description and notes to what "
                                     "the agents and teachers read.")
    except Exception:  # noqa: BLE001
        pass
    return out


def parse_upload(path: Path, form: dict, problem_id: str | None = None) -> dict:
    """Everything the upload preview shows; raises ParseError with the sniff info when columns are missing."""
    ev = _ev()
    sn = sniff(path)
    data, warnings = choose(sn, form)
    try:
        table = ev.read_table(path, {"file": path.name, **data})
    except ValueError as e:
        raise ParseError(f"the loop's reader refuses it: {e}", {"sniff": sn}) from None
    load, meas = table["load"], table["meas"]
    window = _window(form)
    until_peak = str(form.get("until_peak", "")).lower() in ("1", "true", "on", "yes")
    stats = curve_stats(load, meas, window, until_peak)
    if stats["negative_loads"]:
        warnings.append("most loads are negative: the loop finds loadings by rising load, so export compression "
                        "as positive numbers (Bluehill does by default)")
    if not stats["cycles"]:
        warnings.append(f"no loading has enough points in the fit window {window}: the loop would reject it")
    out = {"sniff": sn, "data": data, "units": table["units"], "stats": stats, "warnings": warnings,
           "curve": _thin_curve(meas * 1e3, load, 1500), "window": list(window), "until_peak": until_peak}
    if problem_id:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(path, Path(tmp) / path.name)
            spec = build_spec(form, data, path.name, problem_id)
            (Path(tmp) / "test.json").write_text(json.dumps(spec))
            out["loop"] = loop_view(Path(tmp) / "test.json")
    return out


# --------------------------------------------------------------------------- records


def _text(form: dict, key: str, limit: int = 4000) -> str:
    v = str(form.get(key) or "").strip()
    if len(v) > limit or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", v):
        raise ParseError(f"{key}: too long or has control characters")
    return v


def build_spec(form: dict, data: dict, file_name: str, problem_id: str) -> dict:
    """The evidence spec of an upload (ace/evidence.py format) plus a 'lab' block the loop ignores."""
    spec = {"test_id": _text(form, "test_id", 100) or Path(file_name).stem, "problem_id": problem_id,
            "description": _text(form, "description"), "data": {"file": file_name, **data}}
    sign = str(form.get("sign") or "").strip()
    if sign in ("-1", "+1", "1"):
        spec["measurement"] = {"sign": float(sign)}
    lo, hi = _window(form)
    spec["fit"] = {"window": [lo, hi]}
    if str(form.get("until_peak", "")).lower() in ("1", "true", "on", "yes"):
        spec["fit"]["until"] = "peak"
    if form.get("E_GPa"):
        E = float(form["E_GPa"])
        if not 0 < E < 1e4:
            raise ParseError("E_GPa must be a positive modulus in GPa")
        spec["material"] = {"E": E * 1e9, "source": _text(form, "E_source", 400)}
    if _text(form, "notes"):
        spec["notes"] = _text(form, "notes")
    spec["lab"] = {"specimen": _text(form, "specimen", 200), "part": _text(form, "part", 200)}
    return spec


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-._")[:40] or "test"


def _safe_name(name: str) -> str:
    stem, suf = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or "export"
    out = f"{stem}{suf}"
    return "export" + suf if out in ("test.json", "summary.json") else out


class TestData:
    """Committed tests (physical_tests/) and uploaded records (<runs>/testdata/)."""

    def __init__(self, store_root: Path):
        self.store_root = Path(store_root)

    @property
    def root(self) -> Path:
        return self.store_root / "testdata"

    # ------------------------------------------------------------------ lookup
    def committed(self) -> dict[str, Path]:
        ev = _ev()
        out = {}
        for f in ev.discover([TESTS_ROOT]):
            try:
                tid = json.loads(f.read_text()).get("test_id") or f.stem
            except (OSError, ValueError):
                continue
            out.setdefault(str(tid), f)
        return out

    def uploaded(self) -> dict[str, Path]:
        if not self.root.is_dir():
            return {}
        return {d.name: d / "test.json" for d in sorted(self.root.iterdir())
                if d.is_dir() and ID_RE.match(d.name) and (d / "test.json").exists()}

    def spec_path(self, test_id: str) -> tuple[Path, bool]:
        up = self.uploaded()
        if test_id in up:
            return up[test_id], True
        com = self.committed()
        if test_id in com:
            return com[test_id], False
        raise KeyError(test_id)

    # ------------------------------------------------------------------ views
    def brief(self, test_id: str, spec_path: Path, uploaded: bool) -> dict:
        spec = json.loads(spec_path.read_text())
        d = spec_path.parent
        summ = {}
        if uploaded and (d / "summary.json").exists():
            summ = json.loads((d / "summary.json").read_text())
        else:
            try:
                summ = self._curve_summary(spec_path, spec, points=300)
            except Exception as e:  # noqa: BLE001
                summ = {"error": f"{type(e).__name__}: {e}"}
        lab = spec.get("lab") or {}
        return {"id": test_id, "test_id": spec.get("test_id"), "problem_id": spec.get("problem_id"),
                "description": spec.get("description", ""), "notes": spec.get("notes", ""),
                "specimen": lab.get("specimen"), "part": lab.get("part"), "uploaded": uploaded,
                "uploaded_at": lab.get("uploaded_at"), "original_filename": lab.get("original_filename"),
                "spec_ref": to_ref(spec_path, self.store_root), "dir_ref": to_ref(d, self.store_root),
                "dir_abs": str(d), "spec_abs": str(spec_path),
                "stats": summ.get("stats"), "curve": (summ.get("curve") or [])[:: max(1, len(summ.get("curve") or []) // 300)],
                "error": summ.get("error")}

    def _curve_summary(self, spec_path: Path, spec: dict, points: int = 3000) -> dict:
        ev = _ev()
        table = ev.read_table(spec_path.parent / spec["data"]["file"], spec["data"])
        fit = spec.get("fit") or {}
        stats = curve_stats(table["load"], table["meas"], tuple(fit.get("window", (0.1, 0.9))),
                            fit.get("until") == "peak")
        return {"stats": stats, "curve": _thin_curve(table["meas"] * 1e3, table["load"], points),
                "units": table["units"]}

    def item(self, test_id: str) -> dict:
        spec_path, uploaded = self.spec_path(test_id)
        spec = json.loads(spec_path.read_text())
        out = self.brief(test_id, spec_path, uploaded)
        full = self._curve_summary(spec_path, spec)
        out.update(spec={k: v for k, v in spec.items() if not k.startswith("_")}, curve=full["curve"],
                   stats=full["stats"], units=full["units"], loop=loop_view(spec_path),
                   files=self._files(spec_path.parent), extra_curves=self._extra_curves(spec_path, spec))
        d = spec_path.parent
        ev_arg = str(d) if uploaded else str(d.relative_to(REPO_ROOT))
        spec_arg = str(spec_path) if uploaded else str(spec_path.relative_to(REPO_ROOT))
        out["commands"] = {
            "summarize": f"python -m ace.evidence summarize {spec_arg}",
            "loop": f"python -m ace.run_loop --problems {spec.get('problem_id')} --evidence {ev_arg} ...",
            "score": f"python -m ace.run_loop --problems {spec.get('problem_id')} --evidence {ev_arg} "
                     "--evidence-score ...",
        }
        return out

    def _files(self, d: Path) -> list[dict]:
        out = []
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in (".csv", ".txt", ".tsv", ".dat", ".json", ".pdf", ".md", ".png",
                                                    ".jpg"):
                ref = to_ref(p, self.store_root)
                if ref:
                    out.append({"name": p.name, "ref": ref, "bytes": p.stat().st_size, "kind": kind_of(p)})
        return out

    def _extra_curves(self, spec_path: Path, spec: dict) -> list[dict]:
        """Other exports in a committed test's folder that read with the same columns (the dogbone test)."""
        ev = _ev()
        out = []
        main = spec["data"]["file"]
        for f in sorted(spec_path.parent.glob("*.csv")):
            if f.name == main:
                continue
            try:
                t = ev.read_table(f, spec["data"])
            except (ValueError, OSError):
                continue
            out.append({"name": f.name, "ref": to_ref(f, self.store_root),
                        "curve": _thin_curve(t["meas"] * 1e3, t["load"], 1500), "peak_N": float(np.max(t["load"]))})
        return out

    # ------------------------------------------------------------------ write
    def save(self, src: Path, original_name: str, form: dict, problem_id: str) -> dict:
        parsed = parse_upload(src, form, problem_id)
        test_id = _text(form, "test_id", 100) or Path(original_name).stem
        rid = f"{_slug(test_id)}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        d = self.root / rid
        k = 1
        while d.exists():
            k += 1
            d = self.root / f"{rid}-{k}"
        d.mkdir(parents=True)
        name = _safe_name(original_name)
        shutil.copyfile(src, d / name)
        raw = (d / name).read_bytes()
        spec = build_spec({**form, "test_id": test_id}, parsed["data"], name, problem_id)
        spec["lab"].update(uploaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           original_filename=original_name[:200], sha256=hashlib.sha256(raw).hexdigest(),
                           via="dashboard upload")
        (d / "test.json").write_text(json.dumps(spec, indent=1))
        summary = {"stats": parsed["stats"], "curve": parsed["curve"], "units": parsed["units"],
                   "warnings": parsed["warnings"], "loop": loop_view(d / "test.json")}
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        return self.item(d.name)

    def delete(self, test_id: str) -> None:
        if test_id not in self.uploaded():
            raise KeyError(test_id)
        shutil.rmtree(self.root / test_id)

    # ------------------------------------------------------------------ listing
    def list(self) -> list[dict]:
        out = []
        for tid, p in self.uploaded().items():
            try:
                out.append(self.brief(tid, p, True))
            except Exception as e:  # noqa: BLE001
                out.append({"id": tid, "uploaded": True, "error": f"{type(e).__name__}: {e}"})
        out.sort(key=lambda t: t.get("uploaded_at") or "", reverse=True)
        for tid, p in self.committed().items():
            try:
                out.append(self.brief(tid, p, False))
            except Exception as e:  # noqa: BLE001
                out.append({"id": tid, "uploaded": False, "error": f"{type(e).__name__}: {e}"})
        return out


def _md_table(text: str, first_header: str) -> dict[str, dict]:
    """Rows of the markdown table whose first header cell is ``first_header``, keyed by the first cell."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if ln.startswith("|") and cells and cells[0] == first_header:
            head, out = cells, {}
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                vals = [c.strip() for c in row.strip().strip("|").split("|")]
                out[vals[0].strip("`")] = dict(zip(head[1:], vals[1:]))
            return out
    return {}


def parts(store_root: Path) -> dict:
    """The printable test parts (physical_tests/parts): STL for printing, STEP for FEA, and the planned setup."""
    try:
        text = (TESTS_ROOT / "README.md").read_text()
    except OSError:
        text = ""
    info, setup = _md_table(text, "Part"), {}
    tables = [m.start() for m in re.finditer(r"^\| Part \|", text, re.M)]
    if len(tables) > 1:
        setup = _md_table(text[tables[1]:], "Part")
    from feacore.problems import registry

    registry.discover()
    out = []
    for stl in sorted(PARTS_DIR.glob("*.stl")):
        pid = stl.stem
        step = stl.with_suffix(".step")
        out.append({"id": pid, "info": info.get(pid, {}), "setup": setup.get(pid, {}),
                    "stl": to_ref(stl, store_root), "step": to_ref(step, store_root) if step.exists() else None,
                    "stl_bytes": stl.stat().st_size, "problem": pid if pid in registry._REGISTRY else None})   # noqa: SLF001
    img = PARTS_DIR / "parts.png"
    return {"parts": out, "image": to_ref(img, store_root) if img.exists() else None,
            "readme": "physical_tests/README.md"}


def problems_for_tests() -> list[dict]:
    """Problems an upload can belong to: the physical-test problems first, then the loop's other problems."""
    from feacore.problems import registry

    registry.discover()
    try:
        from ace.config import TASKS
    except Exception:  # noqa: BLE001
        TASKS = {}
    out = []
    for p in registry.all_problems():
        if "physical" in p.tags or p.id in TASKS:
            out.append({"id": p.id, "title": p.title, "physical": "physical" in p.tags, "in_loop": p.id in TASKS,
                        "compression": "compression_test" in p.tags})
    out.sort(key=lambda p: (not p["physical"], p["id"]))
    return out


# --------------------------------------------------------------------------- API


def router(store):
    from fastapi import APIRouter, HTTPException
    from starlette.concurrency import run_in_threadpool
    from starlette.datastructures import UploadFile

    from feacore.runstore import jsonable

    r = APIRouter()
    root = Path(store.root)
    td = TestData(root)

    async def _form(request: Request) -> tuple[dict, UploadFile]:
        form = await request.form(max_files=1, max_fields=60, max_part_size=MAX_BYTES)
        up = form.get("file")
        if not isinstance(up, UploadFile):
            raise HTTPException(422, "missing file field 'file' (the machine's CSV / TXT export)")
        if Path(up.filename or "").suffix.lower() not in EXPORT_SUFFIXES:
            raise HTTPException(422, f"export files are {', '.join(EXPORT_SUFFIXES)}")
        fields = {k: str(v) for k, v in form.items() if k != "file" and not isinstance(v, UploadFile)}
        return fields, up

    async def _to_tmp(up: UploadFile, tmp: Path) -> Path:
        dst = tmp / _safe_name(up.filename or "export.csv")
        up.file.seek(0)
        data = up.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"exports are limited to {MAX_BYTES // 2**20} MB")
        dst.write_bytes(data)
        return dst

    def _problem(fields: dict, required: bool) -> str | None:
        pid = fields.get("problem_id") or None
        if pid is None:
            if required:
                raise HTTPException(422, "problem_id is required: the loop uses a test for one problem")
            return None
        if pid not in {p["id"] for p in problems_for_tests()}:
            raise HTTPException(422, f"unknown problem {pid!r} (or not one the loop runs)")
        return pid

    @r.get("/api/testdata")
    async def listing():
        tests = await run_in_threadpool(td.list)
        ev = _ev()
        return jsonable({"tests": tests, "problems": await run_in_threadpool(problems_for_tests),
                         "parts": await run_in_threadpool(parts, root), "store": str(td.root.resolve()),
                         "units": {"load": sorted(ev.LOAD_UNITS), "displacement": sorted(ev.DISP_UNITS)}})

    @r.get("/api/testdata/item")
    async def item(id: str):    # noqa: A002 - the query parameter is called id
        if not ID_RE.match(id or ""):
            raise HTTPException(404, "invalid test id")
        try:
            return jsonable(await run_in_threadpool(td.item, id))
        except KeyError:
            raise HTTPException(404, f"no test {id!r}") from None

    @r.post("/api/testdata/parse")
    async def parse(request: Request):
        fields, up = await _form(request)
        pid = _problem(fields, required=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = await _to_tmp(up, Path(tmp))
            try:
                return jsonable(await run_in_threadpool(parse_upload, path, fields, pid))
            except ParseError as e:
                return _parse_error(e)
            except (ValueError, OSError) as e:
                return _parse_error(ParseError(str(e)))

    @r.post("/api/testdata", status_code=201)
    async def create(request: Request):
        fields, up = await _form(request)
        pid = _problem(fields, required=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = await _to_tmp(up, Path(tmp))
            try:
                return jsonable(await run_in_threadpool(td.save, path, up.filename or path.name, fields, pid))
            except ParseError as e:
                return _parse_error(e)
            except (ValueError, OSError) as e:
                return _parse_error(ParseError(str(e)))

    @r.delete("/api/testdata/item")
    def delete(id: str):    # noqa: A002
        if not ID_RE.match(id or ""):
            raise HTTPException(404, "invalid test id")
        try:
            td.delete(id)
        except KeyError:
            raise HTTPException(404, f"no uploaded test {id!r} (committed tests cannot be deleted here)") from None
        return {"deleted": id}

    @r.get("/api/testdata/file")
    def file(ref: str, download: bool = False):
        try:
            return file_response(ref, root, download)
        except RefError as e:
            raise HTTPException(404, str(e)) from None

    return r


def _parse_error(e: ParseError):
    from fastapi.responses import JSONResponse

    from feacore.runstore import jsonable

    return JSONResponse(jsonable({"detail": str(e), **e.info}), status_code=422)
