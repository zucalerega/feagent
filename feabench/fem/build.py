"""Scripted pre-processing: gmsh meshing of a CAD model + CalculiX deck writing.

This is the *reference pipeline* (and a scripted baseline "agent"): it interprets a task's
structured boundary conditions and loads exactly as the task text describes them.  Agents are
free to do anything else as long as their deck represents the same physical problem.

Load/BC vocabulary (all regions are geometric predicates, see :mod:`feabench.fem.regions`):

  BCs    {"type": "fixed", "region": R}                  all displacements zero
         {"type": "symmetry", "region": plane}           zero normal displacement (axis-aligned planes)
         {"type": "displacement", "region": R, "values": {"x": 0.0, "z": 0.0}}
  loads  {"type": "pressure", "region": R, "value": p}   normal pressure, positive = pushing on the face
         {"type": "surface_force", "region": R, "force": [Fx, Fy, Fz]}
                                                         uniform traction with the given resultant
         {"type": "surface_torque", "region": R, "point": c, "axis": a, "torque": T}
                                                         tangential traction ~ distance from axis
                                                         (Saint-Venant distribution), resultant moment T
         {"type": "gravity", "acceleration": [gx, gy, gz]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .regions import select_nodes

GMSH_TO_CCX_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
# Abaqus/ccx face numbering of tetrahedra (corner indices, outward normal by right-hand rule)
TET_FACES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}


@dataclass
class SolidMesh:
    coords: np.ndarray  # (N, 3)
    tets: np.ndarray  # (M, 10) or (M, 4) node rows, ccx node order
    tris: np.ndarray  # (K, 6) or (K, 3) boundary triangles (node rows)
    tri_entity: np.ndarray  # (K,) CAD surface tag of each boundary triangle
    meta: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return 2 if self.tets.shape[1] == 10 else 1

    @property
    def etype(self) -> str:
        return "tet10" if self.order == 2 else "tet4"

    @property
    def n_dof(self) -> int:
        return 3 * len(self.coords)

    def node_mask(self, region: dict) -> np.ndarray:
        return select_nodes(self.coords, region)

    def faces_in(self, region: dict) -> np.ndarray:
        """Indices of boundary triangles whose nodes all lie in ``region``."""
        m = self.node_mask(region)
        return np.where(m[self.tris].all(axis=1))[0]

    def face_lookup(self) -> dict:
        if "_faces" not in self.meta:
            lut = {}
            for e, conn in enumerate(self.tets[:, :4]):
                for fno, (a, b, c) in TET_FACES.items():
                    lut[tuple(sorted((conn[a], conn[b], conn[c])))] = (e, fno)
            self.meta["_faces"] = lut
        return self.meta["_faces"]


def mesh_gmsh_model(h: float, order: int = 2, curvature: int = 0, fields: list[dict] | None = None,
                    h_min: float | None = None, optimize_high_order: bool = True) -> SolidMesh:
    """Mesh the current gmsh model (geometry already built/imported and synchronised)."""
    import gmsh

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min if h_min is not None else h / 50.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    if order == 2 and optimize_high_order:
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
    if fields:
        ids = []
        for i, f in enumerate(fields, start=1):
            fid = 100 + i
            if f["type"] == "ball":
                gmsh.model.mesh.field.add("Ball", fid)
                for k, v in zip(("XCenter", "YCenter", "ZCenter"), f["center"]):
                    gmsh.model.mesh.field.setNumber(fid, k, v)
                gmsh.model.mesh.field.setNumber(fid, "Radius", f["radius"])
            elif f["type"] == "cylinder":
                gmsh.model.mesh.field.add("Cylinder", fid)
                for k, v in zip(("XCenter", "YCenter", "ZCenter"), f["center"]):
                    gmsh.model.mesh.field.setNumber(fid, k, v)
                for k, v in zip(("XAxis", "YAxis", "ZAxis"), f["axis"]):
                    gmsh.model.mesh.field.setNumber(fid, k, v)
                gmsh.model.mesh.field.setNumber(fid, "Radius", f["radius"])
            elif f["type"] == "box":
                gmsh.model.mesh.field.add("Box", fid)
                for k, v in zip(("XMin", "YMin", "ZMin"), f["min"]):
                    gmsh.model.mesh.field.setNumber(fid, k, v)
                for k, v in zip(("XMax", "YMax", "ZMax"), f["max"]):
                    gmsh.model.mesh.field.setNumber(fid, k, v)
            else:
                raise ValueError(f["type"])
            gmsh.model.mesh.field.setNumber(fid, "VIn", f["size"])
            gmsh.model.mesh.field.setNumber(fid, "VOut", h)
            if "thickness" in f and f["type"] in ("ball", "box"):
                gmsh.model.mesh.field.setNumber(fid, "Thickness", f["thickness"])
            ids.append(fid)
        gmsh.model.mesh.field.add("Min", 99)
        gmsh.model.mesh.field.setNumbers(99, "FieldsList", ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(99)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.model.mesh.generate(3)
    if order == 2 and optimize_high_order:
        gmsh.model.mesh.optimize("HighOrder")
    return extract_gmsh_mesh()


def extract_gmsh_mesh() -> SolidMesh:
    import gmsh

    tags, xyz, _ = gmsh.model.mesh.getNodes()
    coords = xyz.reshape(-1, 3)
    row = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    row[tags.astype(np.int64)] = np.arange(len(tags))
    tets = []
    for et, _, en in zip(*gmsh.model.mesh.getElements(3)):
        if et == 11:
            tets.append(row[en.astype(np.int64)].reshape(-1, 10)[:, GMSH_TO_CCX_TET10])
        elif et == 4:
            tets.append(row[en.astype(np.int64)].reshape(-1, 4))
        else:
            raise ValueError(f"unsupported gmsh 3D element type {et}")
    tets = np.vstack(tets)
    tris, ent = [], []
    for dim, tag in gmsh.model.getEntities(2):
        for et, _, en in zip(*gmsh.model.mesh.getElements(2, tag)):
            k = {9: 6, 2: 3}.get(et)
            if k is None:
                continue
            t = row[en.astype(np.int64)].reshape(-1, k)
            tris.append(t)
            ent.append(np.full(len(t), tag))
    tris = np.vstack(tris) if tris else np.zeros((0, 6), np.int64)
    ent = np.concatenate(ent) if ent else np.zeros(0, np.int64)
    # drop nodes not used by volume elements (e.g. geometry points) and renumber
    used = np.zeros(len(coords), bool)
    used[tets.ravel()] = True
    new = np.cumsum(used) - 1
    keep_tris = used[tris].all(axis=1)
    return SolidMesh(coords=coords[used], tets=new[tets], tris=new[tris[keep_tris]], tri_entity=ent[keep_tris])


# ----------------------------------------------------------------------------- surface integrals

_TRI_GAUSS = (  # 6-point rule, exact to degree 4 (Dunavant); weights sum to 1/2
    np.array([[0.445948490915965, 0.445948490915965], [0.445948490915965, 0.108103018168070],
              [0.108103018168070, 0.445948490915965], [0.091576213509771, 0.091576213509771],
              [0.091576213509771, 0.816847572980459], [0.816847572980459, 0.091576213509771]]),
    np.array([0.223381589678011, 0.223381589678011, 0.223381589678011,
              0.109951743655322, 0.109951743655322, 0.109951743655322]) / 2.0,
)


def _tri_shape(xi: np.ndarray, k: int):
    x, y = xi[:, 0], xi[:, 1]
    L = np.stack([1 - x - y, x, y], axis=1)
    if k == 3:
        N = L
        dN = np.array([[-1, 1, 0], [-1, 0, 1]], float)[None].repeat(len(xi), 0)
        return N, dN
    N = np.hstack([L * (2 * L - 1), np.stack([4 * L[:, 0] * L[:, 1], 4 * L[:, 1] * L[:, 2], 4 * L[:, 2] * L[:, 0]], 1)])
    dL = np.array([[-1, 1, 0], [-1, 0, 1]], float)  # d(L1,L2,L3)/d(x,y)
    dN = np.zeros((len(xi), 2, 6))
    for d in range(2):
        for i in range(3):
            dN[:, d, i] = (4 * L[:, i] - 1) * dL[d, i]
        for m, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
            dN[:, d, 3 + m] = 4 * (dL[d, a] * L[:, b] + L[:, a] * dL[d, b])
    return N, dN


def surface_quadrature(mesh: SolidMesh, faces: np.ndarray):
    """Quadrature points, weights*|J| (area elements), outward-agnostic normals, shape values."""
    tri = mesh.tris[faces]
    k = tri.shape[1]
    xi, w = _TRI_GAUSS
    N, dN = _tri_shape(xi, k)
    X = mesh.coords[tri]  # (F, k, 3)
    pts = np.einsum("qk,fkd->fqd", N, X)
    t1 = np.einsum("qk,fkd->fqd", dN[:, 0, :], X)
    t2 = np.einsum("qk,fkd->fqd", dN[:, 1, :], X)
    nrm = np.cross(t1, t2)
    dA = np.linalg.norm(nrm, axis=2) * w[None, :]
    return tri, N, pts, dA, nrm / np.linalg.norm(nrm, axis=2, keepdims=True)


def traction_nodal_forces(mesh: SolidMesh, faces: np.ndarray, traction) -> dict[int, np.ndarray]:
    """Consistent nodal forces f_i = int N_i t dA; ``traction(points (P,3)) -> (P,3)``."""
    tri, N, pts, dA, _ = surface_quadrature(mesh, faces)
    t = traction(pts.reshape(-1, 3)).reshape(pts.shape)
    f = np.einsum("qk,fq,fqd->fkd", N, dA, t)
    out: dict[int, np.ndarray] = {}
    for rows, forces in zip(tri, f):
        for r, v in zip(rows, forces):
            out[int(r)] = out.get(int(r), 0.0) + v
    return out


def face_area(mesh: SolidMesh, faces: np.ndarray) -> float:
    return float(surface_quadrature(mesh, faces)[3].sum())


# ----------------------------------------------------------------------------- deck writer

def _lines(ids, per=16):
    ids = [str(int(i)) for i in ids]
    return [", ".join(ids[i:i + per]) for i in range(0, len(ids), per)]


def write_ccx_deck(path: str | Path, mesh: SolidMesh, task: dict, analysis: dict | None = None) -> dict:
    """Write a CalculiX deck for ``task`` (materials, boundary_conditions, loads, analysis)."""
    analysis = analysis or task.get("analysis", {"type": "static"})
    atype = analysis["type"]
    path = Path(path)
    out: list[str] = ["*HEADING", f"feabench reference pipeline: {task.get('id', '')}"]
    out.append("*NODE, NSET=NALL")
    out += [f"{i + 1}, {x:.12g}, {y:.12g}, {z:.12g}" for i, (x, y, z) in enumerate(mesh.coords)]
    etype = "C3D10" if mesh.order == 2 else "C3D4"
    out.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    out += [f"{e + 1}, " + ", ".join(str(n + 1) for n in conn) for e, conn in enumerate(mesh.tets)]
    info: dict = {"n_nodes": len(mesh.coords), "n_elements": len(mesh.tets), "element": etype, "bcs": {}, "loads": {}}

    # boundary conditions
    bc_lines: list[str] = []
    for bc in task.get("boundary_conditions", []):
        mask = mesh.node_mask(bc["region"])
        n = int(mask.sum())
        info["bcs"][bc["id"]] = n
        if n == 0:
            raise ValueError(f"boundary condition {bc['id']} selects no nodes")
        name = f"N_{bc['id']}".upper()
        out.append(f"*NSET, NSET={name}")
        out += _lines(np.where(mask)[0] + 1)
        if bc["type"] == "fixed":
            bc_lines.append(f"{name}, 1, 3, 0.0")
        elif bc["type"] == "symmetry":
            nrm = np.abs(np.asarray(bc["region"]["normal"], float))
            d = int(np.argmax(nrm))
            if nrm[d] < 0.999999 * np.linalg.norm(nrm):
                raise ValueError("symmetry plane must be axis-aligned for *BOUNDARY")
            bc_lines.append(f"{name}, {d + 1}, {d + 1}, 0.0")
        elif bc["type"] == "displacement":
            for comp, val in bc["values"].items():
                d = "xyz".index(comp) + 1
                bc_lines.append(f"{name}, {d}, {d}, {float(val):.12g}")
        else:
            raise ValueError(bc["type"])
    out.append("*BOUNDARY")
    out += bc_lines

    # material
    mats = task["materials"]
    if len(mats) != 1:
        raise NotImplementedError("reference pipeline supports a single material")
    mat = mats[0]
    out += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{mat['E']:.12g}, {mat['nu']:.12g}"]
    if "density" in mat:
        out += ["*DENSITY", f"{mat['density']:.12g}"]
    out.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")

    # loads
    step_loads: list[str] = []
    cload: dict[int, np.ndarray] = {}
    lut = mesh.face_lookup()
    for ld in task.get("loads", []):
        lt = ld["type"]
        if lt == "gravity":
            g = np.asarray(ld["acceleration"], float)
            mag = float(np.linalg.norm(g))
            d = g / mag
            step_loads += ["*DLOAD", f"EALL, GRAV, {mag:.12g}, {d[0]:.12g}, {d[1]:.12g}, {d[2]:.12g}"]
            info["loads"][ld["id"]] = {"type": lt}
            continue
        faces = mesh.faces_in(ld["region"])
        if len(faces) == 0:
            raise ValueError(f"load {ld['id']} selects no boundary faces")
        area = face_area(mesh, faces)
        info["loads"][ld["id"]] = {"type": lt, "n_faces": int(len(faces)), "area": area}
        if lt == "pressure":
            step_loads.append("*DLOAD")
            for tri in mesh.tris[faces]:
                e, fno = lut[tuple(sorted(tri[:3]))]
                step_loads.append(f"{e + 1}, P{fno}, {float(ld['value']):.12g}")
        elif lt == "surface_force":
            F = np.asarray(ld["force"], float)
            t = F / area
            for r, v in traction_nodal_forces(mesh, faces, lambda p: np.tile(t, (len(p), 1))).items():
                cload[r] = cload.get(r, 0.0) + v
        elif lt == "surface_torque":
            a = np.asarray(ld["axis"], float)
            a = a / np.linalg.norm(a)
            c = np.asarray(ld["point"], float)

            def tang(p):
                d = p - c
                d = d - np.outer(d @ a, a)
                return np.cross(a, d)  # |.| = distance from axis

            _, _, pts, dA, _ = surface_quadrature(mesh, faces)
            J = float(np.sum(dA * np.sum(tang(pts.reshape(-1, 3)).reshape(pts.shape) ** 2, axis=2)))
            scale = float(ld["torque"]) / J
            for r, v in traction_nodal_forces(mesh, faces, lambda p: scale * tang(p)).items():
                cload[r] = cload.get(r, 0.0) + v
            info["loads"][ld["id"]]["polar_moment"] = J
        else:
            raise ValueError(lt)
    if cload:
        step_loads.append("*CLOAD")
        for r, v in sorted(cload.items()):
            for d in range(3):
                if abs(v[d]) > 0:
                    step_loads.append(f"{r + 1}, {d + 1}, {v[d]:.12g}")

    out.append("*STEP")
    if atype == "static":
        out.append("*STATIC")
    elif atype == "frequency":
        out += ["*FREQUENCY", str(int(analysis.get("n_modes", 6)))]
    elif atype == "buckle":
        out += ["*BUCKLE", str(int(analysis.get("n_modes", 4)))]
    else:
        raise ValueError(atype)
    out += step_loads
    if atype == "static":
        out += ["*NODE FILE", "U, RF", "*EL FILE", "S, E"]
    else:
        out += ["*NODE FILE", "U", "*EL FILE", "S"]
    out.append("*END STEP")
    path.write_text("\n".join(out) + "\n")
    info["n_dof"] = 3 * len(mesh.coords)
    return info
