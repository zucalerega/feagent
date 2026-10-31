// Hash router + header health pill. Pages are ES modules loaded on demand (no build step).
import { api, h, clear, errorBox, getToken, setToken } from './api.js';

const ROUTES = {
  benchmarks: () => import('./pages/benchmarks.js'),
  study: () => import('./pages/study.js'),
  batch: () => import('./pages/batch.js'),
  consensus: () => import('./pages/consensus.js'),
  viewer: () => import('./pages/viewer.js'),
  solve: () => import('./pages/solve.js'),
  health: () => import('./pages/health.js'),
  connect: () => import('./pages/connect.js'),
};

let cleanup = null;
let renderToken = 0;

function parse() {
  const raw = (location.hash || '#/benchmarks').slice(2);
  const [path] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { page: parts[0] || 'benchmarks', args: parts.slice(1) };
}

async function render() {
  const token = ++renderToken;
  if (cleanup) { try { cleanup(); } catch (e) { console.warn(e); } cleanup = null; }
  const { page, args } = parse();
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.route === page));
  const main = document.getElementById('app');
  main.className = page === 'viewer' && args.length ? 'full' : '';
  clear(main);
  const loader = ROUTES[page];
  if (!loader) { main.append(errorBox(`Unknown page "${page}"`)); return; }
  try {
    const mod = await loader();
    if (token !== renderToken) return;
    const c = await mod.render(main, args);
    if (token !== renderToken) { if (typeof c === 'function') c(); return; }
    cleanup = typeof c === 'function' ? c : null;
  } catch (e) {
    console.error(e);
    if (token === renderToken) main.append(errorBox(e));
  }
}

async function health() {
  const el = document.getElementById('health');
  try {
    const hl = await api('/api/health');
    const b = hl.backends || {};
    clear(el);
    const items = ['skfem', 'ccx', 'nn'].filter(k => b[k]).map(k =>
      h('span', { class: 'pill', title: b[k].ok ? (b[k].doc || 'available') : (b[k].error || 'unavailable') },
        h('span', { class: `dot ${b[k].ok ? 'good' : 'critical'}` }), k));
    if (!hl.backends) items.push(h('span', { class: 'pill' }, h('span', { class: 'spinner', style: { width: '12px', height: '12px' } }), 'checking backends'));
    const gpus = (hl.gpu && hl.gpu.gpus) || [];
    const hw = gpus.length ? `${gpus[0].name}` : `${hl.cpu.count} CPU · no GPU`;
    items.push(h('a', { class: 'pill', href: '#/health', title: 'System health' }, hw));
    el.append(...items);
    if (!hl.backends || hl.backends_checking) setTimeout(health, 3000);
  } catch (e) {
    clear(el).append(h('span', { class: 'pill' }, h('span', { class: 'dot critical' }), 'server unreachable'));
    setTimeout(health, 5000);
  }
}

// ?token=... (in the query or the hash) is stored once and removed from the address bar
function captureToken() {
  const url = new URL(location.href);
  let t = url.searchParams.get('token');
  url.searchParams.delete('token');
  let hash = url.hash;
  const i = hash.indexOf('?');
  if (i >= 0) {
    const hp = new URLSearchParams(hash.slice(i + 1));
    if (hp.get('token')) { t = t || hp.get('token'); hp.delete('token'); hash = hash.slice(0, i) + (hp.toString() ? `?${hp}` : ''); }
  }
  if (t) {
    setToken(t);
    history.replaceState(null, '', url.pathname + url.search + hash);
  }
}

let prompted = false;
window.addEventListener('feacore-unauthorized', () => {
  if (prompted) return;
  prompted = true;
  const input = h('input', { type: 'text', placeholder: 'paste the access token', style: { width: '100%' } });
  const box = h('div', { style: { position: 'fixed', inset: 0, background: 'rgba(11,11,11,0.35)', zIndex: 70, display: 'flex', alignItems: 'center', justifyContent: 'center' } },
    h('div', { class: 'card stack', style: { width: '480px' } },
      h('h2', {}, 'Access token required'),
      h('p', { class: 'secondary' }, getToken() ? 'The stored token was rejected by this server.' : 'This server is shared with a token. Ask the person who shared the link, or open the link that ends in ?token=\u2026'),
      input,
      h('div', { class: 'row' },
        h('button', { class: 'btn primary', onclick: () => { setToken(input.value.trim()); location.reload(); } }, 'Save and reload'),
        h('button', { class: 'btn', onclick: () => { box.remove(); } }, 'Close'))));
  document.body.append(box);
  input.focus();
});

captureToken();
window.addEventListener('hashchange', render);
if (!location.hash) history.replaceState(null, '', '#/benchmarks');
render();
health();
setInterval(health, 60000);
