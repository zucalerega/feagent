"""gmsh lifecycle and CAD import. Native output is routed to stderr."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import gmsh

from .errors import CadLoadError

SUPPORTED_EXTENSIONS = (".step", ".stp", ".iges", ".igs", ".brep")


@contextmanager
def gmsh_session(model_name: str = "feagent") -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1)
    os.dup2(2, 1)  # OCC writes to fd 1 directly
    try:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add(model_name)
            gmsh.logger.start()
            yield
        finally:
            try:
                gmsh.logger.stop()
            except Exception:
                pass
            gmsh.finalize()
    finally:
        try:
            sys.stdout.flush()
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def gmsh_messages(level: str = "Error") -> list[str]:
    try:
        return [m for m in gmsh.logger.get() if level in m]
    except Exception:
        return []


def load_cad(path: str | Path, heal: bool = False) -> list[tuple[int, int]]:
    p = Path(path)
    hint = "Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS)
    if not p.is_file():
        raise CadLoadError(f"CAD file not found: {p}", hint="Check the path. " + hint)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise CadLoadError(f"Unsupported CAD format '{p.suffix}': {p}", hint=hint)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    occ = gmsh.model.occ
    try:
        occ.importShapes(str(p), highestDimOnly=True)
        occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if heal:
            occ.healShapes()
            occ.synchronize()
            vols = gmsh.model.getEntities(3)
        if len(vols) > 1:
            occ.fragment(vols, [])
            occ.synchronize()
        if heal:
            occ.removeAllDuplicates()
        occ.synchronize()
    except Exception as e:
        errs = "; ".join(gmsh_messages("Error"))
        raise CadLoadError(f"Failed to import {p.name}: {e} {errs}".strip(),
                           hint="The file may be corrupt or empty. " + hint) from e
    return gmsh.model.getEntities(3)
