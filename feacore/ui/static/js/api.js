// Shared helpers: API calls, typed-array decoding, DOM building, number formatting, status chips.

export async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text }; }
  if (!res.ok) {
    const d = body && body.detail;
    const msg = typeof d === 'string' ? d : (d ? JSON.stringify(d) : `${res.status} ${res.statusText}`);
    const err = new Error(msg);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

export const postJSON = (path, data) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

const DTYPES = { float32: Float32Array, float64: Float64Array, uint32: Uint32Array, int32: Int32Array,
                 uint16: Uint16Array, int16: Int16Array, uint8: Uint8Array };

/** {dtype, shape, b64 | data} -> typed array (little-endian). */
export function decode(arr) {
  if (!arr) return null;
  const T = DTYPES[arr.dtype];
  if (!T) throw new Error(`unsupported dtype ${arr.dtype}`);
  if (arr.data) return T.from(arr.data.flat(Infinity).map(v => (v === null ? NaN : v)));
  const bin = atob(arr.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new T(bytes.buffer);
}

// ------------------------------------------------------------------ DOM
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ------------------------------------------------------------------ numbers
const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
              '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
export const sup = n => String(n).split('').map(c => SUP[c] ?? c).join('');

function trimZeros(s) {
  return s.includes('.') && !s.includes('e') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

/** Engineering-friendly formatting: 0.012345, 1234.5, 1.4999 × 10⁸. */
export function fmt(x, sig = 5) {
  if (x === null || x === undefined || typeof x !== 'number' || !isFinite(x)) return '–';
  if (x === 0) return '0';
  const ax = Math.abs(x);
  if (ax >= 1e-3 && ax < 1e5) {
    const s = x.toPrecision(sig);
    if (!s.includes('e')) return trimZeros(s);
  }
  let e = Math.floor(Math.log10(ax));
  let m = (x / Math.pow(10, e)).toPrecision(sig);
  if (Math.abs(parseFloat(m)) >= 10) { e += 1; m = (x / Math.pow(10, e)).toPrecision(sig); }
  return `${trimZeros(m)} × 10${sup(e)}`;
}

export function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || !isFinite(x)) return '–';
  const v = 100 * x;
  const a = Math.abs(v);
  if (a !== 0 && a < 0.001) return `${v.toExponential(1)}%`;
  return `${v.toPrecision(digits)}%`.replace(/(\.\d*?)0+%$/, '$1%').replace(/\.%$/, '%');
}

export function fmtSigned(x, digits = 2) {
  if (x === null || x === undefined || !isFinite(x)) return '–';
  const s = fmtPct(Math.abs(x), digits);
  return (x > 0 ? '+' : x < 0 ? '−' : '') + s;
}

export function fmtTime(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '–';
  if (sec < 1) return `${(sec * 1000).toFixed(0)} ms`;
  if (sec < 60) return `${sec.toFixed(1)} s`;
  return `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
}

export function fmtInt(n) {
  if (n === null || n === undefined || !isFinite(n)) return '–';
  return Math.round(n).toLocaleString('en-US');
}

export function timeAgo(iso) {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (!isFinite(t)) return iso;
  const s = (Date.now() - t) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}

// ------------------------------------------------------------------ status chips
const ICON = { good: '✓', warning: '!', serious: '▲', critical: '✕', info: 'i', running: '…' };

export function statusChip(status, label, big = false) {
  return h('span', { class: `status ${status || 'info'}${big ? ' big' : ''}` },
    h('span', { class: 'ico' }, ICON[status] || 'i'), label);
}

const RUN_STATUS = {
  succeeded: ['good', 'Succeeded'], failed: ['critical', 'Failed'], running: ['running', 'Running'],
  queued: ['info', 'Queued'], interrupted: ['warning', 'Interrupted'],
};
export function runStatusChip(s) {
  const [st, label] = RUN_STATUS[s] || ['info', s || 'unknown'];
  return statusChip(st, label);
}

export function flagList(explanations) {
  if (!explanations || !explanations.length) return null;
  return h('div', { class: 'flag-list' }, explanations.map(e =>
    h('div', { class: `flag ${e.severity}` }, h('span', { class: 'ico' }, ICON[e.severity] || 'i'),
      h('div', {}, h('b', {}, e.title), h('span', {}, e.text)))));
}

// ------------------------------------------------------------------ tooltip / toast
const tipEl = () => document.getElementById('tooltip');
export function showTip(evt, html) {
  const t = tipEl();
  t.innerHTML = html;
  t.hidden = false;
  const pad = 14;
  const r = t.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export function hideTip() { tipEl().hidden = true; }

export function toast(msg, ms = 2600) {
  const t = h('div', { class: 'toast' }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

export function loading(msg = 'Loading…') {
  return h('div', { class: 'empty' }, h('span', { class: 'spinner' }), ' ', msg);
}

export function errorBox(err) {
  return h('div', { class: 'error-box' }, String(err && err.message ? err.message : err));
}

export function qs() {
  const hash = location.hash || '';
  const i = hash.indexOf('?');
  return new URLSearchParams(i >= 0 ? hash.slice(i + 1) : '');
}

export function qoiDesc(q) {
  if (!q) return '';
  const t = q.type;
  if (t === 'point') return `${q.field} at (${(q.point || []).map(v => fmt(v, 4)).join(', ')})`;
  if (['max', 'min', 'max_abs'].includes(t)) return `${t === 'max_abs' ? 'max |' + q.field + '|' : t + ' of ' + q.field}${q.group ? ' on ' + q.group : ''}`;
  if (t === 'reaction') return `reaction ${'xyz'[q.component] || q.component} on ${q.group}`;
  if (t === 'integral' || t === 'mean') return `${t} of ${q.field}${q.group ? ' over ' + q.group : ''}`;
  if (t === 'strain_energy') return 'total strain energy';
  return t || '';
}
