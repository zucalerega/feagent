// Gold benchmarks: parts -> BC sets (references, singular QoIs, DOFs and time per level) and the nested
// study meshes with their status on this machine; regenerate, verify, render, import and nonlinear run as
// tasks. Server: feacore/ui/gold_api.py (/api/gold...), tools in feacore/ui/tools/gold.py.
import { api, postJSON, authFetch, withToken, h, clear, fmt, fmtInt, fmtPct, fmtTime, loading, errorBox, statusChip,
  timeAgo, toast } from '../api.js';
import { toolForm, taskTable } from '../tasks.js';

const MESH_STATUS = {
  ok: ['good', 'here, sha256 ok'], mismatch: ['critical', 'here, sha256 differs'],
  missing: ['info', 'not on this machine'], unlisted: ['warning', 'here, not in index.json'],
};
const RUN_CHIP = { succeeded: 'good', failed: 'critical', running: 'running', queued: 'info', interrupted: 'warning' };

export async function render(el) {
  el.append(loading('Loading gold benchmarks…'));
  let data;
  try { data = await api('/api/gold'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Gold benchmarks'),
      h('p', { class: 'lede' }, 'Real CAD parts (ABC dataset) with generated supports and loads, and reference values certified by a nested-mesh convergence study on the fixed solver. The study meshes are not in git: they come with the generator’s archives, or can be regenerated here.')),
    h('div', { class: 'row' },
      h('a', { class: 'btn', href: '#/benchmarks' }, 'All benchmarks'),
      h('button', { class: 'btn', type: 'button', onclick: () => refresh(true) }, 'Refresh'))));
  const tiles = h('div', { class: 'tiles' });
  const partsBox = h('div', { class: 'stack' });
  el.append(tiles, h('div', { style: { height: '24px' } }), partsBox);
  el.append(h('div', { style: { height: '24px' } }), importCard(), h('div', { style: { height: '24px' } }), nonlinearCard());
  const tasks = taskTable({ group: 'gold' }, { empty: 'No gold task has run on this server yet.', limit: 30 });
  el.append(h('div', { style: { height: '24px' } }), h('div', { class: 'card flush' },
    h('div', { class: 'card-head', style: { padding: '16px 22px 0' } }, h('h2', {}, 'Gold tasks')), tasks.el));

  const cards = new Map();     // part -> {meshes, sets} elements, redrawn on refresh (tool panels stay open)
  let alive = true, timer = null;
  const draw = d => {
    drawTiles(tiles, d);
    if (!d.parts.length) {
      clear(partsBox).append(h('div', { class: 'card empty-state' }, h('h2', {}, 'No gold sets here'),
        h('p', {}, `The gold root ${d.root} has no BC set. Import a generator archive below.`)));
      return;
    }
    for (const p of d.parts) {
      if (!cards.has(p.part)) {
        const c = partCard(p, d, () => refresh(true));
        cards.set(p.part, c);
        partsBox.append(c.el);
      }
      cards.get(p.part).update(p, d);
    }
  };
  async function refresh(now = false) {
    clearTimeout(timer);
    if (now) {
      try { data = await api('/api/gold'); } catch (e) { toast(String(e.message || e)); }
    }
    if (!alive) return;
    draw(data);
    const busy = data.parts.some(p => p.sets.some(s => s.runs.some(r => r.status === 'queued' || r.status === 'running')));
    let tasksBusy = false;
    try { tasksBusy = (await api('/api/tasks?group=gold&limit=20')).tasks.some(t => t.state === 'queued' || t.state === 'running'); } catch { /* the table shows errors */ }
    if (alive) timer = setTimeout(() => refresh(true), busy || tasksBusy ? 3000 : 20000);
  }
  refresh(false);
  return () => { alive = false; clearTimeout(timer); tasks.stop(); };
}

function drawTiles(el, d) {
  const tile = (label, value, sub) => h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null);
  const regen = d.parts.reduce((n, p) => n + p.meshes.filter(m => m.regenerated && m.regenerated.available).length, 0);
  clear(el).append(
    tile('BC sets', fmtInt(d.n_sets), `${d.n_parts} part${d.n_parts === 1 ? '' : 's'}`),
    tile('Study meshes here', `${d.n_present} / ${d.n_meshes}`, `${d.n_ok} with the stored sha256`),
    tile('Regenerated here', fmtInt(regen), 'in this run store'),
    tile('Gold root', h('span', { class: 'mono', style: { fontSize: '14px', wordBreak: 'break-all' } }, d.root), d.index_found ? 'index.json found' : 'no index.json'));
}

// ------------------------------------------------------------------ one part
function partCard(p0, d0, onChange) {
  const meshesBox = h('div');
  const setsBox = h('div');
  const panel = h('div');
  let openTool = null;
  const toggle = (name, opts) => {
    clear(panel);
    if (openTool === name) { openTool = null; return; }
    openTool = name;
    panel.append(h('div', { class: 'card tight', style: { marginTop: '16px', background: 'var(--surface-2)' } },
      toolForm(name, { ...opts, onStarted: () => { toast('Started: see Gold tasks below'); onChange(); } })));
  };
  const head = h('div', { class: 'card-head' });
  const el = h('div', { class: 'card' }, head, h('div', { class: 'section-title' }, 'Study meshes'), meshesBox, panel,
    h('div', { class: 'section-title', style: { marginTop: '22px' } }, 'BC sets'), setsBox);
  const update = (p, d) => {
    clear(head).append(
      h('div', {}, h('h2', { style: { margin: 0 } }, p.part),
        h('div', { class: 'meta-line' }, p.source ? `from ${p.source} · ` : '', p.faces ? `${p.faces} faces · ` : '', `${p.sets.length} BC set${p.sets.length === 1 ? '' : 's'}`,
          p.regenerated ? ` · regenerated here from the ${p.regenerated.input}` : '')),
      h('div', { class: 'row', style: { gap: '8px' } },
        h('button', { class: 'btn small', type: 'button', title: 'mesh the levels again from the STEP with the generator’s settings', onclick: () => toggle('gold.regenerate', { defaults: { part: p.part }, hide: ['part'], submitLabel: 'Regenerate' }) }, 'Regenerate meshes'),
        h('button', { class: 'btn small', type: 'button', title: 'scripts/gold.py verify', onclick: () => toggle('gold.verify', { defaults: { ids: p.part }, submitLabel: 'Verify' }) }, 'Verify'),
        h('button', { class: 'btn small', type: 'button', title: 'scripts/gold.py render: runs for the viewer', onclick: () => toggle('gold.render', { defaults: { ids: p.part }, submitLabel: 'Render' }) }, 'Render')));
    clear(meshesBox).append(meshTable(p, d));
    clear(setsBox).append(setTable(p, d, onChange));
  };
  update(p0, d0);
  return { el, update };
}

function meshTable(p, d) {
  if (!p.meshes.length) return h('p', { class: 'muted' }, 'index.json lists no study mesh for this part.');
  const dofs = k => Math.max(0, ...p.sets.map(s => (s.dofs || [])[k] || 0)) || null;
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['level', 'h', 'cells', 'DOFs', 'size', 'sha256 (index.json)', 'on this machine', 'regenerated here', ''].map(x => h('th', { class: ['h', 'cells', 'DOFs', 'size'].includes(x) ? 'num' : '' }, x)))),
    h('tbody', {}, p.meshes.map(m => {
      const [st, label] = MESH_STATUS[m.status] || ['info', m.status];
      const rg = m.regenerated;
      return h('tr', {},
        h('td', { class: 'mono' }, m.level, m.scheme ? h('div', { class: 'small muted' }, m.scheme) : null),
        h('td', { class: 'num' }, m.h ? `${fmt(m.h * 1000, 4)} mm` : '–'),
        h('td', { class: 'num' }, fmtInt(m.cells)),
        h('td', { class: 'num' }, fmtInt(dofs(m.k))),
        h('td', { class: 'num' }, m.bytes ? `${(m.bytes / 1e6).toFixed(1)} MB` : '–'),
        h('td', { class: 'mono small', title: m.sha256 || '' }, m.sha256 ? `${m.sha256.slice(0, 12)}…` : '–'),
        h('td', {}, statusChip(st, label), m.status === 'mismatch' ? h('div', { class: 'small muted mono', title: m.local_sha256 }, `here ${m.local_sha256.slice(0, 12)}…`) : null),
        h('td', {}, rg ? regenCell(rg) : h('span', { class: 'muted small' }, '–')),
        h('td', { class: 'small' },
          m.status !== 'missing' ? h('div', {}, h('a', { href: withToken(`/api/gold/meshes/${encodeURIComponent(p.part)}/${m.level}`) }, 'download')) : null,
          rg && rg.available ? h('div', {}, h('a', { href: withToken(`/api/gold/regenerated/${encodeURIComponent(p.part)}/${m.level}`) }, 'download regenerated')) : null));
    })));
}

function regenCell(rg) {
  if (rg.error) return h('div', {}, statusChip('critical', 'failed'), h('div', { class: 'small muted' }, rg.error.slice(0, 160)));
  const chip = rg.match ? statusChip('good', 'sha256 matches')
    : rg.cells_match ? statusChip('warning', 'same cells, other bytes')
      : statusChip('warning', `${fmtInt(rg.cells)} cells (stored ${fmtInt(rg.stored_cells)})`);
  return h('div', {}, chip, h('div', { class: 'small muted', title: rg.sha256 }, `${rg.scheme || ''} · ${timeAgo(rg.created_at)}${rg.installed ? ' · installed' : ''}`));
}

// ------------------------------------------------------------------ BC sets of a part
function setTable(p, d, onChange) {
  if (!p.sets.length) return h('p', { class: 'muted' }, 'No BC set of this part is imported.');
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['set', 'supports and loads', 'references (evaluator only)', 'singular', 'DOFs per level', 'solve time per level', 'solutions'].map(x => h('th', {}, x)))),
    h('tbody', {}, p.sets.map(s => h('tr', {},
      h('td', {}, h('a', { href: `#/benchmarks/${encodeURIComponent(s.id)}`, class: 'mono' }, s.id), h('div', { class: 'small muted' }, `scored: ${s.scored_qoi || '–'}`)),
      h('td', { class: 'small' }, s.bcs.map(b => h('div', {}, b))),
      h('td', { class: 'small' }, s.references.length ? s.references.map(r => h('div', { title: `${r.source}; ${r.notes || ''}` },
        h('span', { class: 'mono' }, r.qoi), ` ${fmt(r.value, 5)} ± ${fmtPct(r.tolerance, 2)}`)) : h('span', { class: 'muted' }, 'none')),
      h('td', { class: 'small mono' }, s.singular.length ? s.singular.map(q => h('div', {}, q)) : h('span', { class: 'muted' }, '–')),
      h('td', { class: 'small num' }, (s.dofs || []).map((n, k) => h('div', {}, `L${k} ${fmtInt(n)}`))),
      h('td', { class: 'small num' }, (s.seconds || []).map((t, k) => h('div', {}, `L${k} ${fmtTime(t)}`))),
      h('td', { class: 'small' }, solutions(p, s, d, onChange))))));
}

/** Solve buttons for the levels with a mesh here (stored, else regenerated) and links to the solves. */
function solutions(p, s, d, onChange) {
  const box = h('div', { class: 'stack', style: { minWidth: '210px' } });
  const buttons = h('div', { class: 'row', style: { gap: '6px' } });
  for (const m of p.meshes) {
    const stored = m.status === 'ok' || m.status === 'mismatch' || m.status === 'unlisted';
    const regen = m.regenerated && m.regenerated.available;
    if (!stored && !regen) continue;
    const source = stored ? 'stored' : 'regenerated';
    const est = stored ? ((s.dofs || [])[m.k] || 4.6 * (m.cells || 0)) : 4.6 * (m.regenerated.cells || 0);
    const tooBig = est > d.view_max_dofs;
    buttons.append(h('button', {
      class: 'btn small', type: 'button', disabled: tooBig ? true : null,
      title: tooBig ? `about ${fmtInt(est)} DOFs: more than this machine solves for the viewer (${fmtInt(d.view_max_dofs)})` : `solve ${s.id} on the ${source} ${m.level} mesh (about ${fmtInt(est)} DOFs) for the field viewer`,
      onclick: async e => {
        e.target.disabled = true;
        try { await postJSON('/api/gold/solve', { set_id: s.id, level: m.k, source }); toast(`Solving ${s.id} on ${source} ${m.level}`); onChange(); }
        catch (err) { toast(err.message); e.target.disabled = false; }
      } }, `Solve ${m.level}${stored ? '' : ' (regen.)'}`));
  }
  if (buttons.children.length) box.append(buttons);
  else box.append(h('span', { class: 'muted' }, 'no mesh here: import or regenerate one'));
  for (const r of s.runs.slice(0, 6)) {
    const ref = (s.levels || []).find(l => l.level === r.level);
    box.append(h('div', {},
      h('span', { class: `dot ${RUN_CHIP[r.status] || ''}`, style: { marginRight: '6px' }, title: r.status }),
      r.fields ? h('a', { href: `#/viewer/${r.run_id}` }, `L${r.level ?? '?'} ${r.mesh_source === 'regenerated' ? 'regenerated ' : ''}solution`) : `L${r.level ?? '?'} ${r.status}`,
      h('span', { class: 'muted' }, ` · ${timeAgo(r.created_at)}`),
      r.status === 'succeeded' && ref ? h('div', { class: 'muted' }, agreement(r.qois, ref.qois)) : null,
      r.error ? h('div', { class: 'muted' }, r.error.slice(0, 120)) : null));
  }
  return box;
}

/** Largest relative difference between a solve's QoIs and the generator's on that level. */
function agreement(mine, gold) {
  if (!mine || !gold) return '';
  let worst = 0, name = null;
  const scaleR = Math.max(0, ...Object.entries(gold).filter(([k]) => /^R[xyz]_/.test(k)).map(([, v]) => Math.abs(v)));
  for (const [k, g] of Object.entries(gold)) {
    if (!(k in mine) || g === null) continue;
    const scale = /^R[xyz]_/.test(k) ? scaleR : Math.abs(g);
    const e = scale > 0 ? Math.abs(mine[k] - g) / scale : 0;
    if (e > worst) { worst = e; name = k; }
  }
  return worst < 1e-9 ? 'reproduces the study (< 1e-9)' : `vs the study: up to ${fmtPct(worst, 2)} (${name})`;
}

// ------------------------------------------------------------------ import + nonlinear
function importCard() {
  const list = h('div', { class: 'small' });
  const file = h('input', { type: 'file', accept: '.tgz,.tar.gz,.tar' });
  const status = h('span', { class: 'small muted' });
  const formBox = h('div');
  const showForm = path => clear(formBox).append(h('div', { class: 'card tight', style: { marginTop: '14px', background: 'var(--surface-2)' } },
    toolForm('gold.import', { defaults: { archive: path }, submitLabel: 'Import', onStarted: () => toast('Import started: see Gold tasks') })));
  const loadList = async () => {
    try {
      const d = await api('/api/gold/archives');
      clear(list).append(d.archives.length ? d.archives.map(a => h('div', { class: 'row', style: { gap: '10px', alignItems: 'center' } },
        h('span', { class: 'mono' }, a.name), h('span', { class: 'muted' }, `${(a.bytes / 1e6).toFixed(1)} MB`),
        h('button', { class: 'btn small', type: 'button', onclick: () => showForm(a.path) }, 'Import…'))) : h('span', { class: 'muted' }, 'No archive uploaded to this run store yet.'));
    } catch (e) { clear(list).append(errorBox(e)); }
  };
  const upload = h('button', { class: 'btn', type: 'button', onclick: async () => {
    const f = file.files && file.files[0];
    if (!f) { toast('Choose an archive first'); return; }
    upload.disabled = true;
    status.textContent = `uploading ${(f.size / 1e6).toFixed(1)} MB…`;
    try {
      const res = await authFetch(`/api/gold/archives/${encodeURIComponent(f.name)}`, { method: 'PUT', body: f });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status} ${res.statusText}`);
      status.textContent = `uploaded ${body.name}`;
      loadList();
      showForm(body.path);
    } catch (e) { status.textContent = String(e.message || e); }
    upload.disabled = false;
  } }, 'Upload');
  loadList();
  return h('div', { class: 'card' },
    h('h2', {}, 'Import a generator archive'),
    h('p', { class: 'secondary' }, 'The generator (python -m feacore.goldgen run, scripts/goldgen_h100.sh) packs finished BC sets with their meshes into numbered archives (goldgen-l2_0000.tgz, ...); scripts/pull_gold_batches.sh copies them off the GPU nodes (it needs the autor CLI). Upload one here, or give a path in the repository or the run store; the import checks every set before adding it (scripts/gold.py import).'),
    h('div', { class: 'row', style: { alignItems: 'center', gap: '12px' } }, file, upload, status),
    h('div', { class: 'section-title', style: { marginTop: '18px' } }, 'Archives in this run store'), list, formBox);
}

function nonlinearCard() {
  return h('div', { class: 'card' },
    h('h2', {}, 'Nonlinear gold sets'),
    h('p', { class: 'secondary' }, 'scripts/gold_nonlinear.py turns linear gold sets into J2-plastic CalculiX benchmarks: S355 steel with linear hardening, loads scaled past first yield, a load ramp on every mesh level, and the gold certification over the levels. CalculiX runs on the CPU; certifying needs three levels, and L2 needs a large machine. Results go to the task’s folder in the run store.'),
    toolForm('gold.nonlinear', { submitLabel: 'Start' }));
}
