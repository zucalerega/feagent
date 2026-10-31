"""Mesh a STEP file with gmsh in a separate process.

gmsh's high-order optimiser can abort() the whole interpreter (an uncaught C++ exception), so
callers run this as ``python -m feabench.fem.meshworker <job.json>`` and read the resulting
``.npz``.  A plain subprocess (rather than multiprocessing) does not re-import the caller's
``__main__``, so scripts without a ``__name__`` guard are safe.

job.json: {"step": path, "out": path.npz, "h": float, "order": 2, "curvature": 0,
           "fields": [...] | null, "optimize": true, "options": {"Mesh.OptimizeNetgen": 1, ...}}
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_job(job: dict) -> None:
    import gmsh

    from .build import mesh_gmsh_model

    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    for name, value in (job.get("options") or {}).items():
        if isinstance(value, str):
            gmsh.option.setString(name, value)
        else:
            gmsh.option.setNumber(name, value)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(job["step"])
    gmsh.model.occ.synchronize()
    m = mesh_gmsh_model(job["h"], order=job.get("order", 2), curvature=job.get("curvature", 0),
                        fields=job.get("fields"), optimize_high_order=job.get("optimize", True))
    gmsh.finalize()
    np.savez(job["out"], coords=m.coords, tets=m.tets, tris=m.tris, tri_entity=m.tri_entity)


def mesh_step(step: str | Path, h: float, order: int = 2, curvature: int = 0, fields: list | None = None,
              timeout: float = 3600.0, options: dict | None = None):
    """Mesh in a child process; if the high-order optimiser crashes, retry without it."""
    from .build import SolidMesh

    errors = []
    with tempfile.TemporaryDirectory(prefix="feabench-mesh-") as tmp:
        for optimize in (True, False):
            out = Path(tmp) / f"mesh_{int(optimize)}.npz"
            job = {"step": str(step), "out": str(out), "h": h, "order": order, "curvature": curvature,
                   "fields": fields, "optimize": optimize, "options": options or {}}
            jpath = Path(tmp) / "job.json"
            jpath.write_text(json.dumps(job))
            try:
                proc = subprocess.run([sys.executable, "-m", "feabench.fem.meshworker", str(jpath)],
                                      capture_output=True, text=True, timeout=timeout)
                rc, err = proc.returncode, proc.stderr[-500:]
            except subprocess.TimeoutExpired:
                rc, err = None, "timeout"
            if rc == 0 and out.exists():
                with np.load(out) as d:
                    mesh = SolidMesh(coords=d["coords"], tets=d["tets"], tris=d["tris"], tri_entity=d["tri_entity"])
                mesh.meta["high_order_optimized"] = optimize
                return mesh
            errors.append(f"optimize={optimize}: rc={rc} {err.strip()[-200:]}")
    raise RuntimeError(f"gmsh failed on {step} at h={h}: {errors}")


if __name__ == "__main__":
    run_job(json.loads(Path(sys.argv[1]).read_text()))
