// Small dependency-free SVG charts: convergence (QoI vs log h), order (log-log error vs h),
// and a forest plot (one row per submission: value +- GCI band) for the consensus view.
import { fmt, sup, showTip, hideTip, esc } from './api.js';

const NS = 'http://www.w3.org/2000/svg';
const C = {
  s1: '#2a78d6', s2: '#eb6834', s7: '#6250d6', ink: '#0b0b0b', ink2: '#52514e', muted: '#898781',
  grid: '#e1e0d9', axis: '#c3c2b7', surface: '#fcfcfb', wash1: 'rgba(42,120,214,0.13)',
  washRef: 'rgba(11,11,11,0.07)', critical: '#d03b3b',
};

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

// ------------------------------------------------------------------ scales and ticks
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);
const logS = (d0, d1, r0, r1) => { const a = Math.log10(d0), b = Math.log10(d1); return v => r0 + (Math.log10(v) - a) / (b - a || 1) * (r1 - r0); };

export function niceTicks(min, max, count = 5) {
  if (!(isFinite(min) && isFinite(max))) return [];
  if (min === max) { const d = Math.abs(min) * 0.05 || 1; min -= d; max += d; }
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => st >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function logTicks(min, max, maxTicks = 7) {
  const out = [];
  const k0 = Math.floor(Math.log10(min)), k1 = Math.ceil(Math.log10(max));
  const decades = Math.log10(max / min);
  const mant = decades > 3 ? [1] : decades > 1.3 ? [1, 2, 5] : [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9];
  for (let k = k0; k <= k1; k++) for (const m of mant) {
    const v = m * Math.pow(10, k);
    if (v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9)) out.push(v);
  }
  if (out.length <= maxTicks) return out;
  // thin to roughly even spacing in log space
  const res = [];
  const step = (out.length - 1) / (maxTicks - 1);
  for (let i = 0; i < maxTicks; i++) res.push(out[Math.round(i * step)]);
  return [...new Set(res)];
}

/** Tick labels sharing one power of ten: returns {labels, exp} (exp = 0 when not needed). */
function axisLabels(ticks) {
  const ax = Math.max(...ticks.map(Math.abs));
  if (!(ax > 0)) return { labels: ticks.map(() => '0'), exp: 0 };
  let exp = Math.floor(Math.log10(ax));
  if (exp >= -2 && exp < 4) exp = 0;
  const scaled = ticks.map(t => t / Math.pow(10, exp));
  const step = ticks.length > 1 ? Math.abs(scaled[1] - scaled[0]) : 1;
  let dec = 0;
  while (dec < 8 && Math.abs(step * 10 ** dec - Math.round(step * 10 ** dec)) > 1e-6 * Math.max(1, step * 10 ** dec)) dec++;
  return { labels: scaled.map(v => (Math.abs(v) < step * 1e-9 ? 0 : v).toFixed(dec)), exp };
}

const tickLabel = v => {
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return String(+v.toPrecision(3));
  const e = Math.floor(Math.log10(a));
  const m = +(v / Math.pow(10, e)).toPrecision(2);
  return m === 1 ? `10${sup(e)}` : `${m}·10${sup(e)}`;
};

function frame(W, H) {
  return s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img' });
}

function hit(el, html) {
  el.addEventListener('mousemove', e => showTip(e, html));
  el.addEventListener('mouseleave', hideTip);
  return el;
}

// ------------------------------------------------------------------ convergence chart
/**
 * opts: points [{h, value, tip}], phiExt, p, band [lo, hi] (GCI of the fine value), hFine,
 *       reference {value, tol}, yTitle, xTitle
 */
export function convergenceChart(opts) {
  const W = 660, H = 360, m = { l: 86, r: 26, t: 20, b: 54 };
  const pts = (opts.points || []).filter(p => isFinite(p.h) && p.h > 0 && isFinite(p.value)).sort((a, b) => a.h - b.h);
  const svg = frame(W, H);
  if (!pts.length) return svg;
  const hmin = pts[0].h, hmax = pts[pts.length - 1].h;
  const showExt = isFinite(opts.phiExt);
  const x0 = hmin / (showExt ? 2.4 : 1.3), x1 = hmax * 1.25;
  const X = logS(x0, x1, m.l, W - m.r);

  const ys = pts.map(p => p.value);
  if (showExt) ys.push(opts.phiExt);
  if (opts.band) ys.push(...opts.band);
  const dataLo = Math.min(...ys), dataHi = Math.max(...ys);
  const span = Math.max(dataHi - dataLo, Math.abs(dataHi) * 1e-6, 1e-300);
  const ref = opts.reference && isFinite(opts.reference.value) ? opts.reference : null;
  let refOff = 0;
  if (ref) {
    const rlo = ref.value * (1 - (ref.tol || 0) * Math.sign(ref.value)), rhi = ref.value * (1 + (ref.tol || 0) * Math.sign(ref.value));
    if (ref.value < dataLo - 6 * span) refOff = -1;
    else if (ref.value > dataHi + 6 * span) refOff = 1;
    else ys.push(Math.min(rlo, rhi), Math.max(rlo, rhi));
  }
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0 || Math.abs(y1) * 0.01 || 1) * 0.1;
  y0 -= pad; y1 += pad;
  const yt = niceTicks(y0, y1, 5);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const Y = lin(y0, y1, H - m.b, m.t);
  const g = s('g');
  svg.append(g);

  // grid + y axis
  const { labels, exp } = axisLabels(yt);
  yt.forEach((t, i) => {
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: C.grid, 'stroke-width': 1 }));
    g.append(s('text', { x: m.l - 10, y: Y(t) + 4.5, 'text-anchor': 'end', class: 'tick' }, labels[i]));
  });
  const xt = logTicks(x0, x1, 7);
  xt.forEach(t => {
    g.append(s('line', { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: C.grid, 'stroke-width': 1 }));
    g.append(s('text', { x: X(t), y: H - m.b + 20, 'text-anchor': 'middle', class: 'tick' }, tickLabel(t)));
  });
  g.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: C.axis, 'stroke-width': 1 }));
  g.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 10, 'text-anchor': 'middle', class: 'axis-title' }, opts.xTitle || 'mesh size h  (log scale, finer ←)'));
  g.append(s('text', { x: 16, y: (m.t + H - m.b) / 2, transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})`, 'text-anchor': 'middle', class: 'axis-title' },
    (opts.yTitle || 'QoI') + (exp ? `  (× 10${sup(exp)})` : '')));

  const clipY = v => Math.max(m.t, Math.min(H - m.b, v));
  // reference band + line
  if (ref && !refOff) {
    const tol = ref.tol || 0;
    const a = Y(ref.value * (1 - tol)), b = Y(ref.value * (1 + tol));
    g.append(s('rect', { x: m.l, width: W - m.r - m.l, y: Math.min(a, b), height: Math.max(1, Math.abs(a - b)), fill: C.washRef }));
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(ref.value), y2: Y(ref.value), stroke: C.ink, 'stroke-width': 1.5 }));
    g.append(s('text', { x: W - m.r - 4, y: Y(ref.value) - 7, 'text-anchor': 'end', class: 'label-strong' }, `reference ${fmt(ref.value, 5)}`));
  } else if (ref && refOff) {
    const yy = refOff > 0 ? m.t + 12 : H - m.b - 8;
    g.append(s('text', { x: W - m.r - 4, y: yy, 'text-anchor': 'end', class: 'label-strong' }, `${refOff > 0 ? '↑' : '↓'} reference ${fmt(ref.value, 5)} (off scale)`));
  }
  // GCI band of the fine solution (from h -> 0 up to the fine mesh)
  if (opts.band && opts.band.every(isFinite)) {
    const [lo, hi] = opts.band;
    const xf = X(opts.hFine || hmin);
    g.append(s('rect', { x: m.l, width: xf - m.l, y: clipY(Y(hi)), height: Math.max(1.5, clipY(Y(lo)) - clipY(Y(hi))), fill: C.wash1 }));
  }
  // Richardson extrapolation + model curve
  if (showExt) {
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(opts.phiExt), y2: Y(opts.phiExt), stroke: C.s1, 'stroke-width': 1.5, 'stroke-dasharray': '6 5' }));
    g.append(s('text', { x: m.l + 6, y: Y(opts.phiExt) + (opts.phiExt >= pts[0].value ? -8 : 17), class: 'label-strong' }, `Richardson h→0: ${fmt(opts.phiExt, 6)}`));
    if (isFinite(opts.p) && opts.p > 0) {
      const f1 = pts[0].value, h1 = pts[0].h;
      const d = [];
      const n = 60;
      for (let i = 0; i <= n; i++) {
        const hh = Math.pow(10, Math.log10(x0) + (Math.log10(hmax) - Math.log10(x0)) * i / n);
        const v = opts.phiExt + (f1 - opts.phiExt) * Math.pow(hh / h1, opts.p);
        d.push(`${i ? 'L' : 'M'}${X(hh).toFixed(1)},${clipY(Y(v)).toFixed(1)}`);
      }
      g.append(s('path', { d: d.join(''), fill: 'none', stroke: C.s1, 'stroke-width': 1.2, 'stroke-dasharray': '1.5 4', 'stroke-linecap': 'round', opacity: 0.9 }));
    }
  }
  // data
  g.append(s('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.h)},${Y(p.value)}`).join(''), fill: 'none', stroke: C.s1, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  // error bar on the fine point
  if (opts.band && opts.band.every(isFinite)) {
    const xf = X(opts.hFine || hmin);
    const [lo, hi] = opts.band;
    g.append(s('line', { x1: xf, x2: xf, y1: clipY(Y(lo)), y2: clipY(Y(hi)), stroke: C.s1, 'stroke-width': 2 }));
    for (const v of [lo, hi]) g.append(s('line', { x1: xf - 6, x2: xf + 6, y1: clipY(Y(v)), y2: clipY(Y(v)), stroke: C.s1, 'stroke-width': 2 }));
  }
  pts.forEach(p => {
    g.append(s('circle', { cx: X(p.h), cy: Y(p.value), r: 5.5, fill: C.s1, stroke: C.surface, 'stroke-width': 2 }));
    g.append(hit(s('circle', { cx: X(p.h), cy: Y(p.value), r: 14, fill: 'transparent', style: 'cursor:default' }), p.tip || `h = ${fmt(p.h, 4)}<br>${fmt(p.value, 7)}`));
  });
  return svg;
}

// ------------------------------------------------------------------ order (log-log) chart
/** opts: points [{h, err}] (err > 0), pTheory, pObs, yTitle */
export function orderChart(opts) {
  const W = 660, H = 250, m = { l: 86, r: 26, t: 16, b: 50 };
  const pts = (opts.points || []).filter(p => p.h > 0 && p.err > 0 && isFinite(p.err)).sort((a, b) => a.h - b.h);
  const svg = frame(W, H);
  if (pts.length < 2) return null;
  const hmin = pts[0].h, hmax = pts[pts.length - 1].h;
  const x0 = hmin / 1.25, x1 = hmax * 1.25;
  const X = logS(x0, x1, m.l, W - m.r);
  const errs = pts.map(p => p.err);
  const th = isFinite(opts.pTheory) ? (hh => pts[0].err * Math.pow(hh / hmin, opts.pTheory)) : null;
  if (th) errs.push(th(hmax));
  let e0 = Math.min(...errs) / 1.6, e1 = Math.max(...errs) * 1.6;
  const Y = logS(e0, e1, H - m.b, m.t);
  const g = s('g');
  svg.append(g);
  logTicks(e0, e1, 5).forEach(t => {
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: C.grid }));
    g.append(s('text', { x: m.l - 10, y: Y(t) + 4.5, 'text-anchor': 'end', class: 'tick' }, tickLabel(t)));
  });
  logTicks(x0, x1, 7).forEach(t => {
    g.append(s('line', { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: C.grid }));
    g.append(s('text', { x: X(t), y: H - m.b + 20, 'text-anchor': 'middle', class: 'tick' }, tickLabel(t)));
  });
  g.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: C.axis }));
  g.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, 'mesh size h  (log scale)'));
  g.append(s('text', { x: 16, y: (m.t + H - m.b) / 2, transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})`, 'text-anchor': 'middle', class: 'axis-title' }, opts.yTitle || '|error|'));
  if (th) {
    g.append(s('line', { x1: X(hmin), y1: Y(th(hmin)), x2: X(hmax), y2: Y(th(hmax)), stroke: C.ink2, 'stroke-width': 1.5, 'stroke-dasharray': '6 5' }));
    g.append(s('text', { x: X(hmax) - 4, y: Y(th(hmax)) - 8, 'text-anchor': 'end', class: 'label-strong' }, `theory: slope ${fmt(opts.pTheory, 2)}`));
  }
  g.append(s('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.h)},${Y(p.err)}`).join(''), fill: 'none', stroke: C.s1, 'stroke-width': 2 }));
  pts.forEach(p => {
    g.append(s('circle', { cx: X(p.h), cy: Y(p.err), r: 5.5, fill: C.s1, stroke: C.surface, 'stroke-width': 2 }));
    g.append(hit(s('circle', { cx: X(p.h), cy: Y(p.err), r: 14, fill: 'transparent' }), `h = ${fmt(p.h, 4)}<br>|error| = ${fmt(p.err, 4)}`));
  });
  if (isFinite(opts.pObs)) {
    g.append(s('text', { x: X(pts[0].h) + 8, y: Y(pts[0].err) + 20, class: 'label-strong' }, `observed: ${fmt(opts.pObs, 3)}`));
  }
  return svg;
}

// ------------------------------------------------------------------ forest plot
/**
 * rows: [{label, sub, value, lo, hi, ext, status, tip, kind: 'submission'|'study'}]
 * reference {value, tol}, consensus {value, lo, hi}, xTitle
 */
export function forestChart(opts) {
  const rows = opts.rows || [];
  const rowH = 38, m = { l: 210, r: 34, t: 30, b: 54 };
  const W = 980, H = m.t + m.b + Math.max(1, rows.length) * rowH;
  const svg = frame(W, H);
  const vals = [];
  rows.forEach(r => [r.value, r.lo, r.hi, r.ext].forEach(v => isFinite(v) && vals.push(v)));
  const ref = opts.reference && isFinite(opts.reference.value) ? opts.reference : null;
  const cons = opts.consensus && isFinite(opts.consensus.value) ? opts.consensus : null;
  if (!vals.length && !ref) return svg;
  // robust domain: centre on the consensus / reference, clip far outliers to the edges
  const centre = cons ? cons.value : ref ? ref.value : vals.sort((a, b) => a - b)[Math.floor(vals.length / 2)];
  const core = vals.filter(v => Math.abs(v - centre) <= Math.max(Math.abs(centre) * 0.5, 1e-300));
  const dom = [...(core.length ? core : vals)];
  if (ref) { dom.push(ref.value * (1 - (ref.tol || 0)), ref.value * (1 + (ref.tol || 0))); }
  if (cons) dom.push(cons.lo ?? cons.value, cons.hi ?? cons.value);
  let x0 = Math.min(...dom), x1 = Math.max(...dom);
  const pad = (x1 - x0 || Math.abs(x1) * 0.02 || 1) * 0.12;
  x0 -= pad; x1 += pad;
  const xt = niceTicks(x0, x1, 6);
  x0 = Math.min(x0, xt[0]); x1 = Math.max(x1, xt[xt.length - 1]);
  const X = lin(x0, x1, m.l, W - m.r);
  const clampX = v => Math.max(m.l, Math.min(W - m.r, X(v)));
  const off = v => X(v) < m.l ? -1 : X(v) > W - m.r ? 1 : 0;
  const g = s('g');
  svg.append(g);
  const { labels, exp } = axisLabels(xt);
  xt.forEach((t, i) => {
    g.append(s('line', { x1: X(t), x2: X(t), y1: m.t - 8, y2: H - m.b, stroke: C.grid }));
    g.append(s('text', { x: X(t), y: H - m.b + 20, 'text-anchor': 'middle', class: 'tick' }, labels[i]));
  });
  g.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: C.axis }));
  g.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 10, 'text-anchor': 'middle', class: 'axis-title' }, (opts.xTitle || 'value') + (exp ? `  (× 10${sup(exp)})` : '')));
  if (ref) {
    const tol = ref.tol || 0;
    const a = X(ref.value * (1 - tol)), b = X(ref.value * (1 + tol));
    g.append(s('rect', { x: Math.min(a, b), width: Math.max(1, Math.abs(b - a)), y: m.t - 8, height: H - m.b - m.t + 8, fill: C.washRef }));
    g.append(s('line', { x1: X(ref.value), x2: X(ref.value), y1: m.t - 8, y2: H - m.b, stroke: C.ink, 'stroke-width': 1.5 }));
    g.append(s('text', { x: X(ref.value), y: m.t - 14, 'text-anchor': 'middle', class: 'label-strong' }, `reference ${fmt(ref.value, 5)} ±${(100 * tol).toPrecision(2)}%`));
  }
  if (cons) {
    g.append(s('line', { x1: X(cons.value), x2: X(cons.value), y1: m.t - 8, y2: H - m.b, stroke: C.s7, 'stroke-width': 1.5, 'stroke-dasharray': '6 5' }));
    if (!ref || Math.abs(X(cons.value) - X(ref.value)) > 150)
      g.append(s('text', { x: X(cons.value), y: m.t - 14, 'text-anchor': 'middle', class: 'label-strong' }, `consensus ${fmt(cons.value, 5)}`));
  }
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2;
    if (i % 2 === 0) g.append(s('rect', { x: 0, width: W, y: y - rowH / 2, height: rowH, fill: 'rgba(11,11,11,0.022)' }));
    const lab = s('text', { x: 12, y: y + 5, class: r.kind === 'study' ? 'label-strong' : '' });
    if (r.status === 'critical' || r.status === 'serious') lab.append(s('tspan', { fill: r.status === 'critical' ? C.critical : '#ec835a', 'font-weight': 800 }, r.status === 'critical' ? '✕ ' : '▲ '));
    lab.append(s('tspan', { fill: C.ink, 'font-weight': r.kind === 'study' ? 650 : 500 }, (r.label || '').length > 24 ? r.label.slice(0, 23) + '…' : r.label));
    g.append(lab);
    const col = r.kind === 'study' ? C.ink : C.s1;
    if (isFinite(r.lo) && isFinite(r.hi)) {
      g.append(s('line', { x1: clampX(r.lo), x2: clampX(r.hi), y1: y, y2: y, stroke: col, 'stroke-width': 2 }));
      for (const v of [r.lo, r.hi]) if (!off(v)) g.append(s('line', { x1: X(v), x2: X(v), y1: y - 6, y2: y + 6, stroke: col, 'stroke-width': 2 }));
    }
    if (isFinite(r.ext) && !off(r.ext)) {
      g.append(s('rect', { x: X(r.ext) - 5, y: y - 5, width: 10, height: 10, transform: `rotate(45 ${X(r.ext)} ${y})`, fill: C.surface, stroke: C.s2, 'stroke-width': 2 }));
    }
    if (isFinite(r.value)) {
      const o = off(r.value);
      if (o) {
        const xe = o < 0 ? m.l + 2 : W - m.r - 2;
        g.append(s('path', { d: o < 0 ? `M${xe},${y} l10,-7 v14 z` : `M${xe},${y} l-10,-7 v14 z`, fill: col }));
        g.append(s('text', { x: o < 0 ? xe + 14 : xe - 14, y: y - 9, 'text-anchor': o < 0 ? 'start' : 'end', class: 'tick' }, `${fmt(r.value, 4)} (off scale)`));
      } else {
        g.append(s('circle', { cx: X(r.value), cy: y, r: r.kind === 'study' ? 6 : 5.5, fill: col, stroke: C.surface, 'stroke-width': 2 }));
      }
    }
    g.append(hit(s('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }), r.tip || esc(r.label)));
  });
  return svg;
}
