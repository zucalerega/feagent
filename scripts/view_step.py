"""Interactive view of a STEP file as CAD (not a mesh): smooth faces with exact B-rep normals,
black B-rep edges, colour by face or by face type, hover for face tag / type / area.

    python scripts/view_step.py PART.step [--out part.html] [--title NAME] [--open]

The STEP is tessellated with gmsh only for display, in the file's own units. Face tags are the
OCC tags the gold-benchmark pipeline names f<tag>.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import webbrowser
from pathlib import Path

import gmsh
import numpy as np

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html,body { margin:0; height:100%; overflow:hidden; font:13px/1.35 "Segoe UI",Roboto,Arial,sans-serif; color:#10233f;
    background:linear-gradient(#5c85c2 0%, #a9bfdf 55%, #f7f9ff 100%); }
  #c { position:fixed; inset:0; }
  #title { position:fixed; left:14px; top:10px; pointer-events:none; } #title b { font-size:17px; display:block; }
  #panel { position:fixed; right:12px; top:12px; width:300px; background:rgba(255,255,255,.93); border:1px solid #6b7a90; padding:10px 12px; }
  #panel h4 { margin:2px 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#3a4d6b; }
  .row { display:flex; gap:4px; margin:4px 0 8px; } .row button { flex:1; padding:4px 0; border:1px solid #6b7a90; background:#f4f6fa; cursor:pointer; font:inherit; }
  .row button.on { background:#2c5aa0; color:#fff; border-color:#2c5aa0; }
  label { display:flex; align-items:center; gap:6px; margin:4px 0; }
  table { border-collapse:collapse; width:100%; font-size:12px; } td { padding:2px 4px; border-bottom:1px solid #e3e8f0; vertical-align:top; }
  td:last-child { text-align:right; font-family:ui-monospace,Menlo,Consolas,monospace; word-break:break-all; }
  .sw { display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:-2px; border:1px solid #0003; }
  #tip { position:fixed; pointer-events:none; background:#10233f; color:#fff; padding:3px 7px; font-size:12px; border-radius:2px; display:none; }
  #triad { position:fixed; left:8px; bottom:8px; width:120px; height:120px; pointer-events:none; }
</style></head><body>
<canvas id="c"></canvas>
<div id="title"><b>__TITLE__</b><span>Geometry (original STEP)</span></div>
<div id="panel">
  <h4>Display</h4>
  <div class="row"><button data-m="plain" class="on">Plain</button><button data-m="type">By face type</button><button data-m="face">By face</button></div>
  <label><input type="checkbox" id="edges" checked> B-rep edges</label>
  <label><input type="checkbox" id="xray"> Transparent</label>
  <div id="typeLegend"></div>
  <h4 style="margin-top:10px">File</h4>
  <table id="info"></table>
  <div class="row" style="margin-top:8px"><button data-view="iso">Iso</button><button data-view="front">Front</button><button data-view="top">Top</button><button data-view="right">Right</button></div>
</div>
<div id="tip"></div><canvas id="triad"></canvas>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const b64 = (s, T) => new T(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById("c"), antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene(), R = D.radius;
const camera = new THREE.PerspectiveCamera(30, 1, R / 100, R * 100);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 1.5); scene.add(key);
const TYPE_COL = { Plane: 0x9fb8bd, Cylinder: 0xd9a441, Cone: 0x8bc34a, Sphere: 0xba68c8, Torus: 0x4fc3f7 };
const OTHER = 0xe57373, PLAIN = new THREE.Color(0.66, 0.74, 0.78);
const faces = [];
D.faces.forEach((f, i) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(b64(f.x, Float32Array), 3));
  g.setAttribute("normal", new THREE.BufferAttribute(b64(f.n, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(b64(f.t, Uint32Array), 1));
  const m = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ color: PLAIN, shininess: 60, specular: 0x444444, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
  m.userData = f; scene.add(m); faces.push(m);
});
const edges = new THREE.Group(); scene.add(edges);
for (const e of D.edges) { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(b64(e, Float32Array), 3));
  edges.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x000000 }))); }
function colour(mode) {
  faces.forEach((m, i) => { const t = m.userData.type;
    m.material.color.set(mode === "plain" ? PLAIN : mode === "type" ? (TYPE_COL[t] ?? OTHER) : new THREE.Color().setHSL((i * 0.618034) % 1, 0.55, 0.6)); });
  const used = [...new Set(D.faces.map(f => f.type))];
  document.getElementById("typeLegend").innerHTML = mode !== "type" ? "" : used.map(t =>
    `<div><span class="sw" style="background:#${(TYPE_COL[t] ?? OTHER).toString(16).padStart(6, "0")}"></span>${t === "BSplineSurface" || !(t in TYPE_COL) ? t + " (freeform)" : t}: ${D.faces.filter(f => f.type === t).length}</div>`).join("");
}
document.querySelectorAll("[data-m]").forEach(b => b.onclick = () => { colour(b.dataset.m);
  document.querySelectorAll("[data-m]").forEach(x => x.classList.toggle("on", x === b)); });
document.getElementById("edges").onchange = e => edges.visible = e.target.checked;
document.getElementById("xray").onchange = e => faces.forEach(m => { m.material.transparent = e.target.checked; m.material.opacity = e.target.checked ? 0.35 : 1; m.material.depthWrite = !e.target.checked; });
document.getElementById("info").innerHTML = D.info.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
function setView(v) { const dirs = { iso: [0.8, -1.1, 0.75], front: [0, -1, 0], top: [0, 0.0001, 1], right: [1, 0, 0] };
  camera.up.set(0, 0, 1); camera.position.copy(new THREE.Vector3(...dirs[v]).normalize().multiplyScalar(R * 4.2)); controls.target.set(0, 0, 0); controls.update(); }
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => setView(b.dataset.view));
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = document.getElementById("tip");
renderer.domElement.addEventListener("pointermove", ev => {
  mouse.set(ev.clientX / innerWidth * 2 - 1, -ev.clientY / innerHeight * 2 + 1); ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(faces)[0]; if (!hit) { tip.style.display = "none"; return; }
  const f = hit.object.userData; tip.textContent = `f${f.tag}  ${f.type}  area ${f.area.toPrecision(4)} ${D.unit}²`;
  tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 10 + "px"; tip.style.display = "block"; });
const tr = new THREE.WebGLRenderer({ canvas: document.getElementById("triad"), antialias: true, alpha: true });
tr.setPixelRatio(devicePixelRatio); tr.setSize(120, 120, false);
const tScene = new THREE.Scene(), tCam = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
[[0xd62728,[1,0,0],"X"],[0x2ca02c,[0,1,0],"Y"],[0x1f4fd6,[0,0,1],"Z"]].forEach(([c, d, l]) => {
  tScene.add(new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(), 1, c, 0.3, 0.16));
  const cv = document.createElement("canvas"); cv.width = cv.height = 64; const x = cv.getContext("2d");
  x.fillStyle = "#" + c.toString(16).padStart(6, "0"); x.font = "bold 44px Arial"; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText(l, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv) })); s.position.set(...d.map(v => v * 1.35)); s.scale.set(0.45, 0.45, 1); tScene.add(s); });
function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize(); setView("iso"); colour("plain");
(function loop() { requestAnimationFrame(loop); controls.update(); key.position.copy(camera.position).add(new THREE.Vector3(R, R * 0.6, R * 2));
  renderer.render(scene, camera); tCam.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(4);
  tCam.up.copy(camera.up); tCam.lookAt(0, 0, 0); tr.render(tScene, tCam); })();
</script></body></html>
"""


def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _header(path: Path) -> list[list[str]]:
    head = path.read_text(errors="replace")[:20000]
    out = [["File", path.name], ["Size", f"{path.stat().st_size / 1024:.0f} KB"]]
    m = re.search(r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'", head)
    if m:
        out.append(["Written", m.group(2)])
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", head)
    if m:
        out.append(["Schema", m.group(1).split("{")[0].strip()])
    return out


def build(step: str, title: str) -> str:
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("v")
        ents = gmsh.model.occ.importShapes(step)
        gmsh.model.occ.synchronize()
        unit = "mm"   # gmsh/OCC report STEP geometry in millimetres
        b = np.asarray(gmsh.model.getBoundingBox(-1, -1))
        lo, hi = b[:3], b[3:]
        centre = (lo + hi) / 2
        diag = float(np.linalg.norm(hi - lo))
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 60)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        gmsh.model.mesh.generate(2)
        faces = []
        types = {}
        for _, tag in gmsh.model.getEntities(2):
            ntags, x, uv = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True, returnParametricCoord=True)
            if len(ntags) == 0:
                continue
            idx = {int(t): i for i, t in enumerate(ntags)}
            et, _, enodes = gmsh.model.mesh.getElements(2, tag)
            tri = [idx[int(n)] for n in enodes[0]] if len(et) else []
            n = np.asarray(gmsh.model.getNormal(tag, uv)).reshape(-1, 3)
            typ = gmsh.model.getType(2, tag)
            types[typ] = types.get(typ, 0) + 1
            faces.append({"tag": tag, "type": typ, "area": float(gmsh.model.occ.getMass(2, tag)),
                          "x": _b64(np.asarray(x).reshape(-1, 3) - centre, np.float32), "n": _b64(n, np.float32),
                          "t": _b64(tri, np.uint32)})
        edges = []
        for _, tag in gmsh.model.getEntities(1):
            _, x, par = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True, returnParametricCoord=True)
            pts = np.asarray(x).reshape(-1, 3) - centre
            if len(pts) >= 2:   # order along the curve by its parameter
                order = np.argsort(np.asarray(par)) if len(par) == len(pts) else np.arange(len(pts))
                edges.append(_b64(pts[order], np.float32))
        vols = [t for d, t in ents if d == 3]
        info = _header(Path(step)) + [
            ["Solids", str(len(vols))], ["Faces", str(len(faces))],
            ["Face types", ", ".join(f"{k} {v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1]))],
            ["Edges", str(len(gmsh.model.getEntities(1)))],
            ["Bounding box", " x ".join(f"{v:.2f}" for v in hi - lo) + f" {unit}"],
            ["Volume", f"{sum(gmsh.model.occ.getMass(3, t) for t in vols):.4g} {unit}³"],
        ]
    finally:
        gmsh.finalize()
    data = {"faces": faces, "edges": edges, "radius": diag / 2, "unit": unit, "info": info}
    return PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.step).with_suffix(".html"))
    out.write_text(build(a.step, a.title or Path(a.step).stem))
    print(out)
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
