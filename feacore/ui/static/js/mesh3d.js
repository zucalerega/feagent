// three.js viewer for (a) FE fields on a boundary triangulation and (b) CAD geometry previews
// coloured by physical group. Renders on demand (no animation loop), so 100k+ triangles stay cheap.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { lut, NAN_RGB, GROUP_COLORS, OTHER_GROUP } from './colormap.js';

export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}

const ISO = new THREE.Vector3(1, -1.35, 0.95).normalize();
const VIEWS = { iso: ISO, x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, -1, 0), z: new THREE.Vector3(0, 0, 1) };

function linearRGB(hex) {
  const c = new THREE.Color(hex);  // sRGB hex -> linear working space
  return [c.r, c.g, c.b];
}

export class MeshView {
  constructor(container, { dim = 3, padRight = 0, padTop = 0, margin = 1.06 } = {}) {
    this.container = container;
    this.margin = margin;
    this.dim = dim;
    this.padRight = padRight;
    this.padTop = padTop;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0);
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    if (dim === 2) {
      this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e9, 1e9);
      this.camera.position.set(0, 0, 1);
    } else {
      this.camera = new THREE.PerspectiveCamera(30, 1, 1e-5, 1e5);
      this.camera.up.set(0, 0, 1);
    }
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b6ae, 2.2));
    const head = new THREE.DirectionalLight(0xffffff, 1.4);
    head.position.set(0.25, 0.5, 1);
    this.camera.add(head);
    this.scene.add(this.camera);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.zoomSpeed = 1.2;
    if (dim === 2) {
      this.controls.enableRotate = false;
      this.controls.screenSpacePanning = true;
      this.controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      this.controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
    }
    this.controls.addEventListener('change', () => this.render());
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.bbox = null;
    this._frame = null;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    (window.__feacoreViews ||= new Set()).add(this);  // debugging / headless tests
  }

  // ---------------------------------------------------------------- basics
  size() {
    const r = this.container.getBoundingClientRect();
    return [Math.max(50, r.width), Math.max(50, r.height)];
  }

  resize() {
    const [w, h] = this.size();
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    if (this.dim === 2) {
      const halfH = (this.camera.top - this.camera.bottom) / 2 || 1;
      const cx = (this.camera.left + this.camera.right) / 2;
      const halfW = halfH * w / h;
      this.camera.left = cx - halfW; this.camera.right = cx + halfW;
    } else {
      this.camera.aspect = w / h;
    }
    if (this.padRight || this.padTop) this.camera.setViewOffset(w, h, this.padRight / 2, -this.padTop / 2, w, h);
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      this.renderer.render(this.scene, this.camera);
    });
  }

  renderNow() { this.renderer.render(this.scene, this.camera); }

  clear() {
    for (const o of [...this.group.children]) {
      this.group.remove(o);
      o.traverse?.(c => { c.geometry?.dispose?.(); if (c.material) [].concat(c.material).forEach(m => { m.map?.dispose?.(); m.dispose(); }); });
    }
  }

  dispose() {
    window.__feacoreViews?.delete(this);
    this.ro.disconnect();
    if (this._frame) cancelAnimationFrame(this._frame);
    this.clear();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss?.();
    this.renderer.domElement.remove();
  }

  fit(view = 'iso') {
    if (!this.bbox) return;
    const c = this.bbox.getCenter(new THREE.Vector3());
    const size = this.bbox.getSize(new THREE.Vector3());
    const [w, h] = this.size();
    const effW = Math.max(50, w - this.padRight), effH = Math.max(50, h - this.padTop);
    if (this.dim === 2) {
      let hw = Math.max(size.x, 1e-12) / 2 * this.margin * 1.04, hh = Math.max(size.y, 1e-12) / 2 * this.margin * 1.04;
      if (hw / hh > effW / effH) hh = hw * effH / effW; else hw = hh * effW / effH;
      hh = hh * h / effH;                       // frustum covers the full canvas
      hw = hh * w / h;
      Object.assign(this.camera, { left: -hw, right: hw, top: hh, bottom: -hh, zoom: 1 });
      this.camera.position.set(c.x, c.y, c.z + Math.max(size.x, size.y) * 10 + 1);
      this.controls.target.set(c.x, c.y, c.z);
    } else {
      // centre on the projected extent of the model's points, then back off until all fit
      const dir = (VIEWS[view] || ISO).clone();
      const up = view === 'z' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
      const right = new THREE.Vector3().crossVectors(dir.clone().negate(), up).normalize();
      const upv = new THREE.Vector3().crossVectors(right, dir.clone().negate()).normalize();
      const ty = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) * effH / h;
      const tx = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) * effW / h;
      const P = this.fitPoints;
      const n = P ? P.length / 3 : 0;
      const step = Math.max(1, Math.floor(n / 6000));
      const pts = [];
      if (n) for (let i = 0; i < n; i += step) pts.push(new THREE.Vector3(P[3 * i] - c.x, P[3 * i + 1] - c.y, P[3 * i + 2] - c.z));
      else { const b = this.bbox; for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) pts.push(new THREE.Vector3(x, y, z).sub(c)); }
      const A = pts.map(d => d.dot(right)), B = pts.map(d => d.dot(upv)), Z = pts.map(d => d.dot(dir));
      // start from the orthographic centring, then iterate in perspective (near parts look bigger)
      let da = (Math.min(...A) + Math.max(...A)) / 2, db = (Math.min(...B) + Math.max(...B)) / 2, dist = 0;
      const distFor = () => { let D = 0; for (let i = 0; i < A.length; i++) D = Math.max(D, Z[i] + Math.abs(A[i] - da) / tx, Z[i] + Math.abs(B[i] - db) / ty); return D * this.margin; };
      for (let it = 0; it < 6; it++) {
        dist = distFor();
        let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity, zm = 0;
        for (let i = 0; i < A.length; i++) {
          const depth = dist - Z[i];
          const u = (A[i] - da) / depth, v = (B[i] - db) / depth;
          u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v); zm += depth;
        }
        zm /= A.length;
        da += (u0 + u1) / 2 * zm;
        db += (v0 + v1) / 2 * zm;
      }
      dist = Math.max(distFor(), 1e-9);
      c.addScaledVector(right, da).addScaledVector(upv, db);
      this.camera.position.copy(c).addScaledVector(dir, dist);
      this.camera.up.copy(up);
      this.camera.near = dist / 500;
      this.camera.far = dist * 50;
      this.controls.target.copy(c);
    }
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }

  // ---------------------------------------------------------------- field mode
  /** points Float32Array(3n), triangles Uint32Array(3m), edges Uint32Array(2k), displacement Float32Array(3n)|null */
  setField({ points, triangles, edges, displacement }) {
    this.clear();
    this.base = points;
    this.disp = displacement || null;
    const geom = new THREE.BufferGeometry();
    this.pos = new THREE.BufferAttribute(new Float32Array(points), 3);
    geom.setAttribute('position', this.pos);
    geom.setIndex(new THREE.BufferAttribute(triangles, 1));
    this.uv = new THREE.BufferAttribute(new Float32Array((points.length / 3) * 2), 2);
    geom.setAttribute('uv', this.uv);
    const common = { side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
    this.material = this.dim === 2 ? new THREE.MeshBasicMaterial(common)
      : new THREE.MeshLambertMaterial({ ...common, flatShading: true });
    this.mesh = new THREE.Mesh(geom, this.material);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', this.pos);
    eg.setIndex(new THREE.BufferAttribute(edges, 1));
    this.edges = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x1f1f1d, transparent: true, opacity: this.dim === 2 ? 0.22 : 0.28 }));
    this.group.add(this.mesh, this.edges);
    geom.computeBoundingBox();
    this.bbox = geom.boundingBox.clone();
    this.baseBox = this.bbox.clone();
    this.fitPoints = points;
  }

  /** Colour by values (Float32Array n) in [lo, hi]. */
  setValues(values, lo, hi, cmap = 'viridis', bands = 0) {
    const uv = this.uv.array;
    const span = hi - lo || 1;
    const a = 0.5 / 256, b = 1 - 1 / 256;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (Number.isFinite(v)) {
        uv[2 * i] = a + b * Math.min(1, Math.max(0, (v - lo) / span));
        uv[2 * i + 1] = 0.25;
      } else { uv[2 * i] = 0.5; uv[2 * i + 1] = 0.75; }
    }
    this.uv.needsUpdate = true;
    const rows = lut(cmap, 256, bands);
    const data = new Uint8Array(256 * 2 * 4);
    rows.forEach((c, i) => { data.set([c[0], c[1], c[2], 255], i * 4); data.set([...NAN_RGB, 255], (256 + i) * 4); });
    const tex = new THREE.DataTexture(data, 256, 2, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = bands ? THREE.NearestFilter : THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.flipY = false;
    tex.needsUpdate = true;
    if (this.material.map) this.material.map.dispose();
    this.material.map = tex;
    this.material.needsUpdate = true;
    this.render();
  }

  setEdgesVisible(on) { if (this.edges) { this.edges.visible = on; this.render(); } }

  /** Deformed shape: x = x0 + scale * u. */
  setDeformation(scale) {
    if (!this.disp || !this.pos) return;
    const p = this.pos.array, b = this.base, d = this.disp;
    for (let i = 0; i < p.length; i++) p[i] = b[i] + scale * d[i];
    this.pos.needsUpdate = true;
    this.mesh.geometry.computeBoundingSphere();
    this.edges.geometry.computeBoundingSphere();
    this.render();
  }

  /** Scale that makes the largest displacement ~``frac`` of the model size. */
  autoScale(frac = 0.08) {
    if (!this.disp || !this.baseBox) return 1;
    let m = 0;
    const d = this.disp;
    for (let i = 0; i < d.length; i += 3) m = Math.max(m, Math.hypot(d[i], d[i + 1], d[i + 2]));
    const L = this.baseBox.getSize(new THREE.Vector3()).length();
    return m > 0 ? frac * L / m : 1;
  }

  // ---------------------------------------------------------------- geometry preview mode
  /**
   * g: {dim, points, triangles, triEntity, lines, lineEntity, groups: [{name, dim, entities, points?}],
   *     entityGroups: {dim: {tag: [group idx]}}}. Returns the legend [{index, name, dim, color}].
   */
  setGeometry(g) {
    this.clear();
    this.geo = g;
    const dim = g.dim;
    // groups that get a colour: boundary-condition-like groups (lower dimension than the domain)
    const colored = g.groups.map((gr, i) => ({ ...gr, index: i }))
      .filter(gr => gr.dim < dim || (dim === 2 && gr.dim === 2 && g.groups.filter(x => x.dim === 2).length > 1));
    colored.sort((a, b) => b.dim - a.dim || a.index - b.index);
    this.legend = colored.map((gr, k) => ({ index: gr.index, name: gr.name, dim: gr.dim, color: k < GROUP_COLORS.length ? GROUP_COLORS[k] : OTHER_GROUP }));
    this.colorOf = new Map(this.legend.map(l => [l.index, l.color]));
    this.highlight = null;
    this._buildGeometry();
    const P = g.points;
    const box = new THREE.Box3();
    for (let i = 0; i < P.length; i += 3) box.expandByPoint(new THREE.Vector3(P[i], P[i + 1], P[i + 2]));
    this.bbox = box;
    this.fitPoints = P;
    return this.legend;
  }

  setHighlight(index) {
    this.highlight = index;
    this.clear();
    this._buildGeometry();
    this.render();
  }

  _groupColor(dimKey, tag, fallback) {
    const gl = (this.geo.entityGroups[dimKey] || {})[tag];
    if (!gl) return fallback;
    for (const gi of gl) {
      if (this.colorOf.has(gi)) {
        if (this.highlight !== null && this.highlight !== gi) return '#cfcdc6';
        return this.colorOf.get(gi);
      }
    }
    return fallback;
  }

  _buildGeometry() {
    const g = this.geo, dim = g.dim, P = g.points, T = g.triangles, TE = g.triEntity;
    const size = this.bbox ? this.bbox.getSize(new THREE.Vector3()).length() : (() => {
      const b = new THREE.Box3();
      for (let i = 0; i < P.length; i += 3) b.expandByPoint(new THREE.Vector3(P[i], P[i + 1], P[i + 2]));
      return b.getSize(new THREE.Vector3()).length();
    })();
    const base = dim === 2 ? '#dfe5ee' : '#d6d4cc';
    const nt = T.length / 3;
    const pos = new Float32Array(nt * 9), col = new Float32Array(nt * 9);
    const cache = new Map();
    for (let t = 0; t < nt; t++) {
      const hex = this._groupColor('2', TE[t], base);
      let rgb = cache.get(hex);
      if (!rgb) { rgb = linearRGB(hex); cache.set(hex, rgb); }
      for (let k = 0; k < 3; k++) {
        const v = T[3 * t + k];
        pos.set([P[3 * v], P[3 * v + 1], P[3 * v + 2]], 9 * t + 3 * k);
        col.set(rgb, 9 * t + 3 * k);
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = dim === 2
      ? new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })
      : new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, flatShading: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    this.group.add(new THREE.Mesh(geom, mat));

    // curves: thin dark feature lines; coloured groups drawn as ribbons (2D) or coloured lines (3D)
    const L = g.lines, LE = g.lineEntity, nl = L.length / 2;
    const plain = [], ribbons = [], ribbonCol = [];
    const w = size / 170;
    for (let i = 0; i < nl; i++) {
      const a = L[2 * i], b = L[2 * i + 1];
      const hex = this._groupColor('1', LE[i], null);
      const pa = [P[3 * a], P[3 * a + 1], P[3 * a + 2]], pb = [P[3 * b], P[3 * b + 1], P[3 * b + 2]];
      if (hex && dim === 2) {
        const dx = pb[0] - pa[0], dy = pb[1] - pa[1], len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len * w / 2, ny = dx / len * w / 2, z = pa[2];
        const ext = [dx / len * w / 2, dy / len * w / 2];
        const q = [[pa[0] - ext[0] + nx, pa[1] - ext[1] + ny], [pa[0] - ext[0] - nx, pa[1] - ext[1] - ny],
                   [pb[0] + ext[0] - nx, pb[1] + ext[1] - ny], [pb[0] + ext[0] + nx, pb[1] + ext[1] + ny]];
        for (const k of [0, 1, 2, 0, 2, 3]) ribbons.push(q[k][0], q[k][1], z);
        const rgb = linearRGB(hex);
        for (let k = 0; k < 6; k++) ribbonCol.push(...rgb);
      } else if (hex) {
        ribbons.push(...pa, ...pb);
        const rgb = linearRGB(hex);
        ribbonCol.push(...rgb, ...rgb);
      } else {
        plain.push(...pa, ...pb);
      }
    }
    if (plain.length) {
      const lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(plain), 3));
      this.group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x3a3936, transparent: true, opacity: 0.75 })));
    }
    if (ribbons.length) {
      const rg = new THREE.BufferGeometry();
      rg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ribbons), 3));
      rg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(ribbonCol), 3));
      const obj = dim === 2
        ? new THREE.Mesh(rg, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, depthTest: false }))
        : new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
      obj.renderOrder = 5;
      this.group.add(obj);
    }
    // point groups: small markers
    const rad = size / 110;
    for (const l of this.legend.filter(l => l.dim === 0)) {
      const gr = g.groups[l.index];
      const color = this.highlight !== null && this.highlight !== l.index ? '#cfcdc6' : l.color;
      for (const p of gr.points || []) {
        const m = new THREE.Mesh(
          dim === 2 ? new THREE.CircleGeometry(rad, 24) : new THREE.SphereGeometry(rad, 16, 12),
          new THREE.MeshBasicMaterial({ color, depthTest: dim !== 2 }));
        m.position.set(p[0], p[1], p[2] || 0);
        m.renderOrder = 6;
        this.group.add(m);
      }
    }
  }
}
