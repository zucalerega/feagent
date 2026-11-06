"""Spike: FEA parts -> one OpenUSD stage (+ GLB + HTML viewer). Throwaway."""
import json, sys, time
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf, Vt, Kind

TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def load_volume(path):
    """-> points (n,3), tets (m,4) linear corners, point_data dict. .msh or .vtu."""
    import pyvista as pv
    if not str(path).endswith(".vtu"):
        import meshio
        m = meshio.read(str(path))
        tets = np.vstack([c.data[:, :4] for c in m.cells if c.type in ("tetra", "tetra10")])
        return np.asarray(m.points, float), tets, {}
    g = pv.read(str(path))
    tets = g.cells_dict[10][:, :4] if 10 in g.cells_dict else g.cells_dict[24][:, :4]
    data = {k: np.asarray(v) for k, v in g.point_data.items() if np.asarray(v).ndim == 1}
    return np.asarray(g.points, float), tets, data


def boundary_surface(points, tets):
    """Outward-oriented boundary triangles of a tet mesh, compacted -> (used node ids, tris)."""
    f = tets[:, TET_FACES].reshape(-1, 3)
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    mask = cnt[inv.ravel()] == 1
    tri = f[mask]
    owner = np.repeat(np.arange(len(tets)), 4)[mask]
    cen = points[tets[owner]].mean(1)
    n = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    flip = np.einsum("ij,ij->i", n, points[tri[:, 0]] - cen) < 0
    tri[flip] = tri[flip][:, ::-1]
    used, newidx = np.unique(tri, return_inverse=True)
    return used, newidx.reshape(-1, 3)


def vertex_normals(p, t):
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    vn = np.zeros_like(p)
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    return (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)).astype(np.float32)


def colormap(s, lo, hi):
    x = np.clip((s - lo) / max(hi - lo, 1e-30), 0, 1)
    try:
        from matplotlib import colormaps
        return colormaps["turbo"](x)[:, :3].astype(np.float32)
    except Exception:
        return np.stack([x, 0.2 * np.ones_like(x), 1 - x], 1).astype(np.float32)


def add_part(stage, name, p, t, scalars, offset, meta, color_by):
    xf = UsdGeom.Xform.Define(stage, f"/World/Assembly/{name}")
    xf.AddTranslateOp().Set(Gf.Vec3d(*offset))
    Usd.ModelAPI(xf).SetKind(Kind.Tokens.component)
    for k, v in meta.items():
        xf.GetPrim().SetCustomDataByKey(k, v)
    m = UsdGeom.Mesh.Define(stage, f"/World/Assembly/{name}/Surface")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(t), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(t.astype(np.int32).ravel()))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vertex_normals(p, t)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr("none")
    api = UsdGeom.PrimvarsAPI(m)
    for k, s in scalars.items():
        pr = api.CreatePrimvar(k, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        pr.Set(Vt.FloatArray.FromNumpy(s.astype(np.float32)))
        m.GetPrim().SetCustomDataByKey(f"range_{k}", Vt.DoubleArray([float(s.min()), float(s.max())]))
    s = scalars[color_by]
    col = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    col.Set(Vt.Vec3fArray.FromNumpy(colormap(s, s.min(), s.max())))
    m.GetPrim().SetCustomDataByKey("colored_by", color_by)


def build_stage(parts, out, spacing=1.3):
    """parts: list of dict(name, mesh, [vtu], [report], [result])."""
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.Xform.Define(stage, "/World")
    Usd.ModelAPI(UsdGeom.Xform.Define(stage, "/World/Assembly").GetPrim()).SetKind(Kind.Tokens.assembly)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    x = 0.0
    for spec in parts:
        pts, tets, data = load_volume(spec.get("vtu") or spec["mesh"])
        used, tri = boundary_surface(pts, tets)
        p = pts[used]
        scal = {k: v[used] for k, v in data.items() if k in ("u_mag", "von_mises")}
        mock = not scal
        if mock:  # unsolved part: mock scalar so the pipeline is exercised (NOT a result)
            scal = {"mock_height": p[:, 2] - p[:, 2].min()}
        color_by = "von_mises" if "von_mises" in scal else list(scal)[0]
        rep = json.load(open(spec["report"])) if spec.get("report") else {}
        res = json.load(open(spec["result"])) if spec.get("result") else {}
        meta = {"part_id": spec["name"], "source_mesh": str(spec["mesh"]),
                "gate_status": rep.get("status", "unknown"),
                "n_dofs": int(res.get("info", {}).get("n_dofs", 0)),
                "n_volume_nodes": int(len(pts)), "n_volume_tets": int(len(tets)),
                "n_surface_tris": int(len(tri)), "scalar_is_mock": mock}
        if res:
            meta.update({f"qoi_{k}": float(v) for k, v in res["qois"].items() if "reaction" not in k})
        add_part(stage, spec["name"], p - p.min(0), tri, scal, (x, 0, 0), meta, color_by)
        x += (p[:, 0].max() - p[:, 0].min()) * spacing + 5
    stage.GetRootLayer().Save()
    return stage


def stage_to_glb(stage_path, glb_path):
    """Traverse the USD stage (not the source meshes) -> GLB with vertex colors."""
    import trimesh
    stage = Usd.Stage.Open(str(stage_path))
    sc = trimesh.Scene()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        p = np.array(m.GetPointsAttr().Get(), np.float32)
        t = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        c = np.array(UsdGeom.PrimvarsAPI(m).GetPrimvar("displayColor").Get(), np.float32)
        rgba = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1)
        tm = trimesh.Trimesh(p, t, vertex_colors=(rgba * 255).astype(np.uint8), process=False)
        sc.add_geometry(tm, node_name=str(prim.GetParent().GetName()), geom_name=str(prim.GetPath()),
                        transform=xf)
    sc.export(str(glb_path))


HTML = """<!doctype html><meta charset=utf-8><title>USD spike viewer</title>
<style>html,body{margin:0;height:100%%;background:#1b1d22;color:#ddd;font:13px sans-serif}
#i{position:absolute;top:8px;left:8px;pointer-events:none}</style>
<div id=i>%(title)s - drag: orbit, wheel: zoom, right-drag: pan</div>
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const r=new THREE.WebGLRenderer({antialias:true});r.setSize(innerWidth,innerHeight);document.body.appendChild(r.domElement);
const s=new THREE.Scene();s.background=new THREE.Color(0x1b1d22);
const cam=new THREE.PerspectiveCamera(40,innerWidth/innerHeight,0.1,1e5);
s.add(new THREE.HemisphereLight(0xffffff,0x444455,1.6));
const d=new THREE.DirectionalLight(0xffffff,1.2);d.position.set(1,-1,2);s.add(d);
const b64="%(b64)s";
const buf=Uint8Array.from(atob(b64),c=>c.charCodeAt(0)).buffer;
const ctl=new OrbitControls(cam,r.domElement);
new GLTFLoader().parse(buf,'',g=>{
 g.scene.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.6,metalness:0,side:THREE.DoubleSide});});
 s.add(g.scene);const bb=new THREE.Box3().setFromObject(g.scene),c=bb.getCenter(new THREE.Vector3()),z=bb.getSize(new THREE.Vector3()).length();
 cam.up.set(0,0,1);cam.position.copy(c).add(new THREE.Vector3(0.3,-1.2,0.7).multiplyScalar(z));ctl.target.copy(c);ctl.update();
 window.READY=true;});
(function a(){requestAnimationFrame(a);r.render(s,cam);})();
</script>"""


def write_html(glb_path, html_path, title):
    import base64
    b64 = base64.b64encode(Path(glb_path).read_bytes()).decode()
    Path(html_path).write_text(HTML % {"title": title, "b64": b64})


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
    for f in out.glob("assembly.*"):
        f.unlink()
    t0 = time.time()
    build_stage(cfg, out / "assembly.usda")
    t1 = time.time()
    build_stage(cfg, out / "assembly.usdc")
    t2 = time.time()
    stage_to_glb(out / "assembly.usdc", out / "assembly.glb")
    t3 = time.time()
    write_html(out / "assembly.glb", out / "viewer.html", "FEA assembly (USD -> GLB)")
    print(f"usda {t1-t0:.2f}s usdc {t2-t1:.2f}s glb {t3-t2:.2f}s")
