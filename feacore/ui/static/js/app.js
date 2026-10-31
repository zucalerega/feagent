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
  runs: () => import('./pages/runs.js'),
  swarm: () => import('./pages/swarm.js'),
};

// top-level sections; deeper pages highlight their parent
const SECTION = { benchmarks: 'benchmarks', batch: 'benchmarks', runs: 'runs', study: 'runs', viewer: 'runs',
                  consensus: 'runs', swarm: 'swarm', solve: 'solve', connect: 'connect', health: 'connect' };

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
  // old index pages now live under Runs
  if (!args.length && (page === 'viewer' || page === 'consensus')) { location.replace(`#/runs?tab=${page === 'viewer' ? 'solves' : 'consensus'}`); return; }
  const section = SECTION[page] || page;
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.route === section));
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
    const b = (hl.backends || {}).skfem;
    const gpus = (hl.gpu && hl.gpu.gpus) || [];
    const hw = gpus.length ? gpus[0].name : `${hl.cpu.count} CPU`;
    const state = !hl.backends ? ['', 'checking solver\u2026'] : b && b.ok ? ['good', 'solver ready'] : ['critical', 'solver unavailable'];
    const jobs = hl.active_jobs ? ` \u00b7 ${hl.active_jobs} job${hl.active_jobs === 1 ? '' : 's'} running` : '';
    clear(el).append(h('a', { class: 'pill', href: '#/health', title: 'system health' }, h('span', { class: `dot ${state[0]}` }), `${state[1]} \u00b7 ${hw}${jobs}`));
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
setInterval(health, 20000);
