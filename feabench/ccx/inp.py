"""Lightweight CalculiX/Abaqus ``.inp`` keyword scanner.

Not a full parser: it walks keyword cards (following ``*INCLUDE``), counts data lines,
collects element types, analysis procedures and modelling features, which is what we need
to categorise decks and to normalise output requests before running a submission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PROCEDURES = {
    "STATIC": "static",
    "FREQUENCY": "frequency",
    "BUCKLE": "buckle",
    "DYNAMIC": "dynamic",
    "MODAL DYNAMIC": "modal_dynamic",
    "STEADY STATE DYNAMICS": "steady_state_dynamics",
    "COMPLEX FREQUENCY": "complex_frequency",
    "HEAT TRANSFER": "heat_transfer",
    "COUPLED TEMPERATURE-DISPLACEMENT": "coupled_temperature_displacement",
    "UNCOUPLED TEMPERATURE-DISPLACEMENT": "uncoupled_temperature_displacement",
    "VISCO": "visco",
    "CFD": "cfd",
    "ELECTROMAGNETICS": "electromagnetics",
    "SENSITIVITY": "sensitivity",
    "GREEN": "green",
    "SUBSTRUCTURE GENERATE": "substructure_generate",
    "CRACK PROPAGATION": "crack_propagation",
    "FEASIBLE DIRECTION": "feasible_direction",
    "ROBUST DESIGN": "robust_design",
    "SELECT CYCLIC SYMMETRY MODES": "cyclic_symmetry",
}

NONLINEAR_MATERIAL_KEYWORDS = {
    "PLASTIC",
    "CYCLIC HARDENING",
    "DEFORMATION PLASTICITY",
    "HYPERELASTIC",
    "HYPERFOAM",
    "CREEP",
    "USER MATERIAL",
    "DAMAGE INITIATION",
}
CONTACT_KEYWORDS = {"CONTACT PAIR", "CONTACT PRINT", "CONTACT FILE", "SURFACE INTERACTION", "GAP", "TIE"}


def element_family(etype: str) -> str:
    e = etype.upper()
    if e.startswith("C3D"):
        return "solid"
    if e.startswith(("CPS", "CPE", "CAX")):
        return "plane"
    if e.startswith(("S3", "S4", "S6", "S8", "M3D")):
        return "shell"
    if e.startswith(("B3", "B2", "T3", "T2")):
        return "beam_truss"
    if e.startswith(("SPRING", "DASHPOT", "MASS", "GAPUNI", "DCOUP")):
        return "connector"
    if e.startswith("D"):
        return "network"
    if e.startswith("U"):
        return "user"
    return "other"


@dataclass
class DeckInfo:
    path: str
    keywords: dict[str, int] = field(default_factory=dict)
    element_types: dict[str, int] = field(default_factory=dict)
    procedures: list[str] = field(default_factory=list)
    n_nodes: int = 0
    n_elements: int = 0
    nlgeom: bool = False
    includes: list[str] = field(default_factory=list)
    missing_includes: list[str] = field(default_factory=list)

    @property
    def families(self) -> set[str]:
        return {element_family(t) for t in self.element_types}

    @property
    def nonlinear_material(self) -> bool:
        return any(k in self.keywords for k in NONLINEAR_MATERIAL_KEYWORDS)

    @property
    def contact(self) -> bool:
        return any(k in self.keywords for k in CONTACT_KEYWORDS)

    def tags(self) -> list[str]:
        t = set(self.procedures)
        t |= {f"elem:{f}" for f in self.families}
        if self.nlgeom:
            t.add("nlgeom")
        if self.nonlinear_material:
            t.add("nonlinear_material")
        if self.contact:
            t.add("contact")
        if "EQUATION" in self.keywords or "MPC" in self.keywords:
            t.add("constraints")
        if "DISTRIBUTING COUPLING" in self.keywords or "COUPLING" in self.keywords:
            t.add("coupling")
        if "RIGID BODY" in self.keywords:
            t.add("rigid_body")
        if "CYCLIC SYMMETRY MODEL" in self.keywords:
            t.add("cyclic_symmetry")
        if "TEMPERATURE" in self.keywords or "INITIAL CONDITIONS" in self.keywords:
            t.add("thermal_load_or_ic")
        if self.is_linear_structural():
            t.add("linear_structural")
        if self.is_linear_structural() and self.families <= {"solid"}:
            t.add("linear_solid")
        return sorted(t)

    def is_linear_structural(self) -> bool:
        structural = {"static", "frequency", "buckle"}
        return (
            bool(self.procedures)
            and set(self.procedures) <= structural
            and not self.nlgeom
            and not self.nonlinear_material
            and not self.contact
            and "user" not in self.families
            and "network" not in self.families
        )

    def to_dict(self) -> dict:
        return {
            "keywords": self.keywords,
            "element_types": self.element_types,
            "procedures": self.procedures,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "nlgeom": self.nlgeom,
            "includes": self.includes,
            "missing_includes": self.missing_includes,
            "tags": self.tags(),
        }


_KW = re.compile(r"^\*([A-Za-z][A-Za-z \-]*)(.*)$")


def parse_params(rest: str) -> dict[str, str]:
    params = {}
    for part in rest.split(",")[1:] if rest.startswith(",") else rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip().upper()] = v.strip()
        else:
            params[part.upper()] = ""
    return params


def iter_cards(path: Path, info: DeckInfo | None = None, _depth: int = 0):
    """Yield (keyword, params, data_lines) for each card, following *INCLUDE."""
    if _depth > 10:
        return
    text = path.read_text(errors="replace").splitlines()
    kw, params, data = None, {}, []
    for raw in text:
        line = raw.strip()
        if line.startswith("**") or not line:
            continue
        if line.startswith("*"):
            if kw is not None:
                yield kw, params, data
            m = _KW.match(line)
            kw = m.group(1).strip().upper() if m else line[1:].upper()
            params = parse_params(m.group(2) if m else "")
            data = []
            if kw == "INCLUDE":
                inc = params.get("INPUT", "").strip('"')
                inc_path = (path.parent / inc) if inc else None
                if info is not None:
                    info.includes.append(inc)
                if inc_path and inc_path.exists():
                    yield kw, params, data
                    kw = None
                    yield from iter_cards(inc_path, info, _depth + 1)
                else:
                    if info is not None:
                        info.missing_includes.append(inc)
        else:
            data.append(line)
    if kw is not None:
        yield kw, params, data


def scan_deck(path: str | Path) -> DeckInfo:
    path = Path(path)
    info = DeckInfo(path=str(path))
    for kw, params, data in iter_cards(path, info):
        info.keywords[kw] = info.keywords.get(kw, 0) + 1
        if kw == "NODE" and "INPUT" not in params:
            info.n_nodes += len(data)
        elif kw == "ELEMENT":
            et = params.get("TYPE", "?").upper()
            # continuation lines end with ','; count element records approximately
            n = sum(1 for d in data if not d.rstrip().endswith(","))
            info.element_types[et] = info.element_types.get(et, 0) + n
            info.n_elements += n
        elif kw == "STEP":
            if "NLGEOM" in params and params["NLGEOM"].upper() not in ("NO",):
                info.nlgeom = True
        if kw in PROCEDURES:
            info.procedures.append(PROCEDURES[kw])
    return info


def read_deck_materials(path: str | Path) -> dict[str, dict]:
    """{material name: {"E", "nu", "density"}} from *MATERIAL/*ELASTIC/*DENSITY (first data line)."""
    mats: dict[str, dict] = {}
    cur = None
    for kw, params, data in iter_cards(Path(path)):
        if kw == "MATERIAL":
            cur = params.get("NAME", f"_unnamed{len(mats)}").upper()
            mats[cur] = {}
        elif cur is not None and kw == "ELASTIC" and data:
            t = [x for x in data[0].replace(",", " ").split()]
            try:
                mats[cur].update(E=float(t[0]), nu=float(t[1]), elastic_type=params.get("TYPE", "ISO").upper())
            except (IndexError, ValueError):
                mats[cur]["elastic_error"] = data[0]
        elif cur is not None and kw == "DENSITY" and data:
            try:
                mats[cur]["density"] = float(data[0].replace(",", " ").split()[0])
            except (IndexError, ValueError):
                mats[cur]["density_error"] = data[0]
        elif kw not in ("ELASTIC", "DENSITY", "EXPANSION", "CONDUCTIVITY", "SPECIFIC HEAT", "PLASTIC",
                        "DAMPING", "USER MATERIAL", "DEPVAR", "HYPERELASTIC"):
            cur = None if kw != "MATERIAL" else cur
    return mats


def read_deck_nodes(path: str | Path) -> dict[int, tuple[float, float, float]]:
    """Full-precision node coordinates from *NODE cards (the .frd only keeps 6 digits)."""
    nodes: dict[int, tuple[float, float, float]] = {}
    for kw, params, data in iter_cards(Path(path)):
        if kw != "NODE" or "INPUT" in params:
            continue
        for d in data:
            t = [x for x in d.replace(",", " ").split()]
            if len(t) >= 2:
                try:
                    xyz = [float(v) for v in t[1:4]] + [0.0] * (4 - len(t))
                    nodes[int(t[0])] = (xyz[0], xyz[1], xyz[2])
                except ValueError:
                    continue
    return nodes


_STEP_PROC = re.compile(r"^\*(STATIC|FREQUENCY|BUCKLE)\b", re.I)


def ensure_output_requests(text: str) -> str:
    """Append node/element output requests the evaluator needs before each ``*END STEP``.

    ccx accumulates multiple ``*NODE FILE`` cards within one step, so appending is safe:
    static steps get U and RF on nodes plus S and E on elements; frequency/buckle steps get
    U and S.  Steps of other procedures are left untouched.
    """
    out: list[str] = []
    proc = None
    for line in text.splitlines():
        s = line.strip()
        m = _STEP_PROC.match(s)
        if m:
            proc = m.group(1).upper()
        if re.match(r"^\*END\s*STEP\b", s, re.I):
            if proc == "STATIC":
                out += ["** feabench: injected output requests", "*NODE FILE", "U, RF", "*EL FILE", "S, E"]
            elif proc in ("FREQUENCY", "BUCKLE"):
                out += ["** feabench: injected output requests", "*NODE FILE", "U", "*EL FILE", "S"]
            proc = None
        out.append(line)
    return "\n".join(out) + "\n"
