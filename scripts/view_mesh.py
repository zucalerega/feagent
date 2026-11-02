"""Interactive, Ansys-style mesh viewer: writes a self-contained HTML page (three.js) for a tet mesh.

    python scripts/view_mesh.py MESH.msh [--out mesh.html] [--title NAME] [--open]

In the browser: orbit (left drag), pan (right drag), zoom (wheel); a live section plane (axis,
position, flip) exposing the cut tetrahedra; element edges on/off; colour by named face group;
hover a face to read its group name. Same look as scripts/render_mesh.py. Mesh data is embedded;
three.js is loaded from unpkg.
"""

from __future__ import annotations

import argparse
import base64
import json
import webbrowser
from pathlib import Path

import meshio
import numpy as np

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#10233f; --panel:rgba(255,255,255,.92); --line:#6b7a90; }
  html,body { margin:0; height:100%; overflow:hidden; font:13px/1.35 "Segoe UI",Roboto,Arial,sans-serif; color:var(--ink);
    background:linear-gradient(#5c85c2 0%, #a9bfdf 55%, #f7f9ff 100%); }
  #c { position:fixed; inset:0; }
  #title { position:fixed; left:14px; top:10px; pointer-events:none; }
  #title b { font-size:17px; display:block; } #title span { font-size:13px; }
  #panel { position:fixed; right:12px; top:12px; width:250px; background:var(--panel); border:1px solid var(--line);
    padding:10px 12px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
  #panel h4 { margin:2px 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#3a4d6b; }
  #panel label { display:flex; align-items:center; gap:6px; margin:4px 0; }
  #panel input[type=range] { width:100%; }
  .row { display:flex; gap:4px; margin:4px 0 8px; }
  .row button, select { flex:1; padding:4px 0; border:1px solid var(--line); background:#f4f6fa; cursor:pointer; font:inherit; }
  .row button.on { background:#2c5aa0; color:#fff; border-color:#2c5aa0; }
  #stats { position:fixed; right:12px; bottom:12px; background:#fff; border:1px solid var(--line); padding:8px 12px;
    font:12px/1.5 ui-monospace,Menlo,Consolas,monospace; white-space:pre; }
  #tip { position:fixed; pointer-events:none; background:#10233f; color:#fff; padding:3px 7px; font-size:12px;
    border-radius:2px; display:none; }
  #triad { position:fixed; left:8px; bottom:8px; width:130px; height:130px; pointer-events:none; }
  hr { border:0; border-top:1px solid #d5dbe6; margin:8px 0; }
</style></head>
<body>
<canvas id="c"></canvas>
<div id="title"><b>__TITLE__</b><span>Mesh</span></div>
<div id="panel">
  <h4>Section plane</h4>
  <label><input type="checkbox" id="secOn"> Enable section</label>
  <div class="row"><button data-ax="0">X</button><button data-ax="1" class="on">Y</button><button data-ax="2">Z</button></div>
  <input type="range" id="secPos" min="0" max="1000" value="500">
  <div class="row"><button id="flip">Flip side</button></div>
  <hr><h4>Display</h4>
  <label><input type="checkbox" id="edges" checked> Element edges</label>
  <label><input type="checkbox" id="groups"> Colour by face group</label>
  <hr><h4>View</h4>
  <div class="row"><button data-view="iso">Iso</button><button data-view="front">Front</button><button data-view="top">Top</button><button data-view="right">Right</button></div>
</div>
<div id="stats"></div>
<div id="tip"></div>
<canvas id="triad"></canvas>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __DATA__;
const b64 = (s, T) => new T(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);
const P = b64(DATA.points, Float32Array);            // (n*3) mm, centred
const T = b64(DATA.tets, Uint32Array);               // (m*npe) tet nodes (4 or 10)
const SF = b64(DATA.surf, Uint32Array);              // (k*4) surface tris + group index
const BODY = new THREE.Color(0.60, 0.72, 0.74), CUT = new THREE.Color(0.93, 0.80, 0.42);

const groupOf = new Map();
for (let i = 0; i < SF.length; i += 4) groupOf.set(key(SF[i], SF[i+1], SF[i+2]), SF[i+3]);
function key(a, b, c) { if (a > b) [a, b] = [b, a]; if (b > c) [b, c] = [c, b]; if (a > b) [a, b] = [b, a]; return a + "," + b + "," + c; }
const palette = DATA.groups.map((_, i) => new THREE.Color().setHSL(((i * 0.618034) % 1), 0.55, 0.62));

const NPE = DATA.npe || 4, nt = T.length / NPE;
// faces: corners (outward) + the midside nodes of edges ab, bc, ca. Node numbering is meshio's
// (VTK): 4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3); gmsh's own files swap 8 and 9.
const FACES = [[0,2,1,6,5,4],[0,1,3,4,8,7],[0,3,2,7,9,6],[1,2,3,5,9,8]];
const SUB = NPE === 10 ? 4 : 1;
const cent = new Float32Array(nt * 3);
for (let e = 0; e < nt; e++) for (let d = 0; d < 3; d++)
  cent[3*e+d] = (P[3*T[NPE*e]+d] + P[3*T[NPE*e+1]+d] + P[3*T[NPE*e+2]+d] + P[3*T[NPE*e+3]+d]) / 4;
// boundary faces (used by exactly one kept tet): [a, b, c, mab, mbc, mca] (mids = -1 for tet4)
function boundary(keep) {
  const seen = new Map();
  for (let e = 0; e < nt; e++) { if (!keep(e)) continue;
    for (const f of FACES) {
      const a = T[NPE*e+f[0]], b = T[NPE*e+f[1]], c = T[NPE*e+f[2]], k = key(a, b, c);
      const m = NPE === 10 ? [T[NPE*e+f[3]], T[NPE*e+f[4]], T[NPE*e+f[5]]] : [-1, -1, -1];
      if (seen.has(k)) seen.set(k, null); else seen.set(k, [a, b, c, ...m]);
    } }
  return [...seen.values()].filter(Boolean);
}
// a face as drawn: 4 sub-triangles through the midside nodes (curved), or the flat triangle
const subTris = f => SUB === 4 ? [[f[0],f[3],f[5]],[f[3],f[1],f[4]],[f[5],f[4],f[2]],[f[3],f[4],f[5]]] : [[f[0],f[1],f[2]]];
// element edges as drawn: a-mab-b, b-mbc-c, c-mca-a (6 segments) or the 3 straight edges
const subEdges = f => SUB === 4 ? [[f[0],f[3]],[f[3],f[1]],[f[1],f[4]],[f[4],f[2]],[f[2],f[5]],[f[5],f[0]]] : [[f[0],f[1]],[f[1],f[2]],[f[2],f[0]]];

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById("c"), antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const R = DATA.radius;
const camera = new THREE.PerspectiveCamera(30, 1, R / 100, R * 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.9));
const key1 = new THREE.DirectionalLight(0xffffff, 1.6); scene.add(key1);
const fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(-1, -0.5, -1); scene.add(fill);

const mat = new THREE.MeshPhongMaterial({ vertexColors: true, flatShading: true, shininess: 40, specular: 0x333333,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, side: THREE.DoubleSide });
const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.55 });
let mesh = null, lines = null, faceList = [];

const st = { sec: false, ax: 1, pos: 0.5, flip: false, edges: true, groups: false };
const lo = DATA.lo, hi = DATA.hi;

function rebuild() {
  const plane = lo[st.ax] + st.pos * (hi[st.ax] - lo[st.ax]);
  const keep = st.sec ? (e => (st.flip ? cent[3*e+st.ax] <= plane : cent[3*e+st.ax] >= plane)) : (() => true);
  faceList = boundary(keep);
  const n = faceList.length, nt3 = SUB * 9, ns = SUB === 4 ? 6 : 3;
  const pos = new Float32Array(n * nt3), col = new Float32Array(n * nt3), seg = new Float32Array(n * ns * 6);
  const XP = v => [P[3*v], P[3*v+1], P[3*v+2]];
  faceList.forEach((f, i) => {
    const g = groupOf.get(key(f[0], f[1], f[2]));
    const color = g === undefined ? CUT : (st.groups ? palette[g] : BODY);
    subTris(f).forEach((t, q) => t.forEach((v, j) => { pos.set(XP(v), i*nt3 + 9*q + 3*j); col.set([color.r, color.g, color.b], i*nt3 + 9*q + 3*j); }));
    subEdges(f).forEach(([u, v], j) => seg.set([...XP(u), ...XP(v)], (i*ns + j) * 6));
  });
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); scene.remove(lines); lines.geometry.dispose(); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  mesh = new THREE.Mesh(g, mat); scene.add(mesh);
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(seg, 3));
  lines = new THREE.LineSegments(lg, lineMat); lines.visible = st.edges; scene.add(lines);
  document.getElementById("title").querySelector("span").textContent =
    st.sec ? `Section plane: ${"XYZ"[st.ax]} = ${(plane + DATA.offset[st.ax]).toFixed(2)} mm` : "Mesh";
}

function setView(v) {
  const d = R * 4.2, dirs = { iso: [0.8, -1.1, 0.75], front: [0, -1, 0], top: [0, 0, 1], right: [1, 0, 0] };
  const dir = new THREE.Vector3(...dirs[v]).normalize();
  camera.up.set(0, 0, v === "top" ? 0 : 1); if (v === "top") camera.up.set(0, 1, 0);
  camera.position.copy(dir.multiplyScalar(d)); controls.target.set(0, 0, 0); controls.update();
}

// triad (synced to the camera rotation)
const tr = new THREE.WebGLRenderer({ canvas: document.getElementById("triad"), antialias: true, alpha: true });
tr.setPixelRatio(window.devicePixelRatio); tr.setSize(130, 130, false);
const tScene = new THREE.Scene(), tCam = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
[[0xd62728, [1,0,0], "X"], [0x2ca02c, [0,1,0], "Y"], [0x1f4fd6, [0,0,1], "Z"]].forEach(([c, d, l]) => {
  tScene.add(new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(), 1, c, 0.3, 0.16));
  const cv = document.createElement("canvas"); cv.width = cv.height = 64; const x = cv.getContext("2d");
  x.fillStyle = "#" + c.toString(16).padStart(6, "0"); x.font = "bold 44px Arial"; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText(l, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv) })); s.position.set(...d.map(v => v * 1.35)); s.scale.set(0.45, 0.45, 1); tScene.add(s);
});

// hover: face group name
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = document.getElementById("tip");
renderer.domElement.addEventListener("pointermove", ev => {
  mouse.set(ev.clientX / innerWidth * 2 - 1, -ev.clientY / innerHeight * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = mesh && ray.intersectObject(mesh)[0];
  if (!hit) { tip.style.display = "none"; return; }
  const [a, b, c] = faceList[Math.floor(hit.faceIndex / SUB)], g = groupOf.get(key(a, b, c));
  tip.textContent = g === undefined ? "section (cut tetrahedra)" : DATA.groups[g];
  tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 10 + "px"; tip.style.display = "block";
});

// controls
const $ = id => document.getElementById(id);
$("secOn").onchange = e => { st.sec = e.target.checked; rebuild(); };
document.querySelectorAll("[data-ax]").forEach(b => b.onclick = () => {
  st.ax = +b.dataset.ax; document.querySelectorAll("[data-ax]").forEach(x => x.classList.toggle("on", x === b)); if (st.sec) rebuild(); });
let pending = 0;
$("secPos").oninput = e => { st.pos = e.target.value / 1000; if (!st.sec) return; cancelAnimationFrame(pending); pending = requestAnimationFrame(rebuild); };
$("flip").onclick = () => { st.flip = !st.flip; if (st.sec) rebuild(); };
$("edges").onchange = e => { st.edges = e.target.checked; lines.visible = st.edges; };
$("groups").onchange = e => { st.groups = e.target.checked; rebuild(); };
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => setView(b.dataset.view));
$("stats").textContent = `Statistics\n  Nodes     ${DATA.nodes.toLocaleString().padStart(10)}\n  Elements  ${nt.toLocaleString().padStart(10)}\n  Type      ${DATA.type.padStart(10)}\n  Faces     ${String(DATA.groups.length).padStart(10)}`;

function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize);
resize(); setView("iso"); rebuild();
(function loop() {
  requestAnimationFrame(loop); controls.update();
  key1.position.copy(camera.position).add(new THREE.Vector3(R, R * 0.6, R * 2));
  renderer.render(scene, camera);
  tCam.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(4); tCam.up.copy(camera.up); tCam.lookAt(0, 0, 0);
  tr.render(tScene, tCam);
})();
</script></body></html>
"""


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def load(mesh_path: str) -> tuple[dict, np.ndarray]:
    """Viewer data for a tet mesh, and the node coordinates in the file's units (m), in the same
    order as the data's points (so fields can be probed at them). Tet10 meshes keep their midside
    nodes, so curved faces and element edges are drawn curved."""
    m = meshio.read(mesh_path)
    ctype = "tetra10" if "tetra10" in m.cells_dict else "tetra"
    tets = m.cells_dict[ctype]                          # all nodes: midside nodes draw curved faces
    used = np.unique(tets)
    renum = np.full(len(m.points), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    X = m.points[used]
    P = X * 1000.0                                      # m -> mm
    lo, hi = P.min(0), P.max(0)
    centre = (lo + hi) / 2
    P = P - centre
    names = {int(v[0]): k for k, v in m.field_data.items() if int(v[1]) == 2}
    gnames = sorted(set(names.values()), key=lambda s: (len(s), s))
    gindex = {n: i for i, n in enumerate(gnames)}
    surf = []
    phys = m.cell_data.get("gmsh:physical")
    if phys is not None:
        for blk, tags in zip(m.cells, phys):
            if blk.type in ("triangle", "triangle6"):
                for tri, t in zip(blk.data[:, :3], tags):
                    if int(t) in names and np.all(renum[tri] >= 0):
                        surf.append([*renum[tri], gindex[names[int(t)]]])
    data = {
        "points": _b64(P.astype(np.float32)), "tets": _b64(renum[tets].astype(np.uint32)),
        "surf": _b64(np.asarray(surf, dtype=np.uint32).reshape(-1, 4)), "groups": gnames,
        "lo": (lo - centre).tolist(), "hi": (hi - centre).tolist(), "offset": centre.tolist(),
        "radius": float(np.linalg.norm(hi - lo) / 2), "nodes": int(len(used)), "npe": int(tets.shape[1]),
        "type": "Tet10" if ctype == "tetra10" else "Tet4",
    }
    return data, X


def build(mesh_path: str, title: str) -> str:
    data, _ = load(mesh_path)
    return PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.mesh).with_suffix(".html"))
    out.write_text(build(a.mesh, a.title or Path(a.mesh).stem))
    print(out)
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
