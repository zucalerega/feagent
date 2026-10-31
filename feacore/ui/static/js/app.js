// Hash router + header health pill. Pages are ES modules loaded on demand (no build step).
import { api, h, clear, errorBox } from './api.js';

const ROUTES = {
  benchmarks: () => import('./pages/benchmarks.js'),
  study: () => import('./pages/study.js'),
  consensus: () => import('./pages/consensus.js'),
  viewer: () => import('./pages/viewer.js'),
  solve: () => import('./pages/solve.js'),
  health: () => import('./pages/health.js'),
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

window.addEventListener('hashchange', render);
if (!location.hash) history.replaceState(null, '', '#/benchmarks');
render();
health();
setInterval(health, 60000);
