// Your CAD: upload a STEP file, pick supports and loads on its faces in 3D (or generate BC sets), and get
// registered feacore problems: the Benchmarks page, convergence studies, the solver and the agents use them.
// #/cad lists the uploads, #/cad/<upload> is one part. Server: feacore/ui/cad_api.py (/api/cad...),
// feacore/problems/custom.py (storage, discovery, references), tools in feacore/ui/tools/cad.py.
import { api, postJSON, authFetch, withToken, h, clear, fmt, fmtInt, fmtPct, loading, errorBox, statusChip, timeAgo,
  toast, decode, showTip, hideTip } from '../api.js';
import { toolForm, taskTable } from '../tasks.js';

const ROLE_COLOR = { fixed: '#2b2a27', roller: '#2a78d6', load: '#eb6834', selected: '#1baf7a' };
const AXES = ['x', 'y', 'z'];

export async function render(el, args) {
  if (args && args[0]) return detail(el, args[0]);
  return list(el);
}

// ------------------------------------------------------------------ list + upload
async function list(el) {
  el.append(loading('Loading your CAD…'));
  let d;
  try { d = await api('/api/cad'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Your CAD'),
    h('p', { class: 'lede' }, 'Bring your own STEP file: it becomes a registered problem that the Benchmarks page, convergence studies, the fixed solver and the agents work on, like a gold part.'))));
  el.append(uploadCard(d));
  el.append(h('div', { style: { height: '24px' } }));
  if (!d.uploads.length) {
    el.append(h('div', { class: 'card empty-state' }, h('h2', {}, 'No upload yet'),
      h('p', {}, 'Upload a STEP file above. The server reads it with the CAD kernel, checks that it is one closed solid and names every face f<tag>, the names supports, loads and agents use.')));
  } else {
    el.append(h('div', { class: 'cards' }, d.uploads.map(u => h('a', { class: 'card pcard', href: `#/cad/${u.id}` },
      h('div', { class: 'title' }, u.name), h('div', { class: 'id' }, u.id),
      h('div', { class: 'meta-line' }, `${(u.size_mm || []).map(v => fmt(v, 3)).join(' × ')} mm · ${u.n_faces} faces`),
      h('div', { class: 'foot' },
        h('span', {}, `${u.sets.length} problem${u.sets.length === 1 ? '' : 's'}`),
        h('span', {}, `${u.sets.filter(s => s.references.length).length} with a reference`),
        h('span', {}, timeAgo(u.created_at)))))));
  }
  el.append(h('div', { style: { height: '24px' } }), usageCard(d, null));
  if (Object.keys(d.errors || {}).length) el.append(h('details', { class: 'card tight', style: { marginTop: '20px' } },
    h('summary', {}, `${Object.keys(d.errors).length} stored problem(s) could not be read`),
    h('pre', { class: 'code' }, Object.entries(d.errors).map(([k, v]) => `${k}: ${v}`).join('\n'))));
}

function uploadCard(d) {
  const file = h('input', { type: 'file', accept: '.step,.stp,.STEP,.STP' });
  const units = h('select', {}, Object.entries(d.units).map(([k, v]) => h('option', { value: k, title: v }, `${k}: ${v}`)));
  const err = h('div');
  const btn = h('button', { class: 'btn primary', type: 'button' }, 'Upload and check');
  btn.onclick = async () => {
    clear(err);
    const f = file.files && file.files[0];
    if (!f) { err.append(errorBox('Choose a STEP file first.')); return; }
    if (f.size > d.max_bytes) { err.append(errorBox(`The file is ${(f.size / 1e6).toFixed(1)} MB; the limit is ${(d.max_bytes / 1e6).toFixed(0)} MB.`)); return; }
    btn.disabled = true;
    btn.textContent = 'Reading the CAD…';
    const fd = new FormData();
    fd.append('file', f);
    fd.append('units', units.value);
    try {
      const res = await authFetch('/api/cad/uploads', { method: 'POST', body: fd });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status} ${res.statusText}`);
      toast(body.existing ? 'This file was uploaded before: opening it' : `Uploaded ${body.name}`);
      location.hash = `#/cad/${body.id}`;
    } catch (e) {
      err.append(errorBox(`Upload refused: ${e.message}`));
      btn.disabled = false;
      btn.textContent = 'Upload and check';
    }
  };
  return h('div', { class: 'card' },
    h('h2', {}, 'Upload a STEP file'),
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '18px' } },
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'STEP file'), file),
      h('label', { class: 'field', style: { maxWidth: '520px' } }, h('span', { class: 'section-title' }, 'Units'), units),
      btn),
    h('p', { class: 'muted small', style: { marginTop: '12px' } }, `One closed solid per file, up to ${(d.max_bytes / 1e6).toFixed(0)} MB. The file is read on the server in its own process (a few seconds); a file that is not STEP, has no solid or several solids is refused with the reason.`),
    err);
}

/** How the problems are stored and found, how the agents use them, and how they are scored. */
function usageCard(d, problem) {
  const id = problem ? problem.id : '<problem id>';
  const upload = problem ? problem.id.replace(/_(bc|m)\d+$/, '') : '<upload id>';
  const multi = d.swarms.filter(s => (s.agents || []).length > 1).sort((a, b) => a.agents.length - b.agents.length);
  const swarm = (multi[0] || d.swarms[0] || { path: 'results/trained_swarms/div_s2' }).path;   // the smallest swarm: cheapest
  const hasRef = problem && problem.references.length;
  const lines = [
    `export FEACORE_RUNS_DIR=${d.runs_dir}    # this server's run store: its cad/ holds the uploads`,
    `python -m ace.infer --swarm ${swarm} --problem ${id} --mock    # free end-to-end check (canned LLM)`,
    `python -m ace.infer --swarm ${swarm} --problem ${id}           # trained agents (ANTHROPIC_API_KEY)`,
    `python -m feacore.problems.custom reference ${upload} --sets ${id}    # reference study on the CPU`,
    `python -m ace.run_loop --problems ${id} --holdout "" --generations 3 --mock    # training: needs a reference`,
  ];
  return h('div', { class: 'card' },
    h('h2', {}, 'How the agents and the solver use it'),
    h('p', { class: 'secondary' }, 'Every BC set is a problem in the gold layout under ', h('code', {}, d.root),
      ' (', h('code', {}, '<upload>/<problem>/public|private'), '), tag ', h('code', {}, 'custom'),
      '. Any process whose FEACORE_RUNS_DIR is this run store (or whose FEACORE_CAD_DIR lists the folder) registers it: this server, its studies and tasks, and command-line runs. Agents get the public statement and the STEP; the references stay private.'),
    h('p', { class: 'secondary' }, hasRef
      ? `Scoring: against the certified reference (${problem.scored_qoi}), as for a gold set; the training loops can use it.`
      : 'Scoring without a reference value is reference-free: ace.infer reports each run’s own ZZ error estimate (relative energy-norm error) and the agreement between agents. The training loops (ace.run_loop, ace.single_loop) need a reference: run a reference study, and the quantities that converge within the gold tolerance (1 % energy, 2 % displacement and reactions, 3 % stress) become references.'),
    h('pre', { class: 'code' }, lines.join('\n')),
    h('pre', { class: 'code' }, `from feacore.problems import registry\np = registry.get("${id}")    # p.case(mesh), p.mesh(path, h), p.references(), p.statement()`));
}

// ------------------------------------------------------------------ one upload
async function detail(el, uid) {
  el.append(loading('Loading the part…'));
  let up, overview;
  try { [up, overview] = await Promise.all([api(`/api/cad/uploads/${encodeURIComponent(uid)}`), api('/api/cad')]); }
  catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  const state = { roles: new Map(), selected: null, viewer: null, alive: true, timer: null };
  const faces = new Map(up.faces.map(f => [f.tag, f]));
  el.append(h('div', { class: 'page-head' },
    h('div', {},
      h('div', { class: 'secondary small' }, h('a', { href: '#/cad' }, '← Your CAD')),
      h('h1', {}, up.name),
      h('div', { class: 'meta-line' }, h('span', { class: 'mono' }, up.id), ` · ${(up.size_mm || []).map(v => fmt(v, 4)).join(' × ')} mm · ${fmt(up.volume_cm3, 4)} cm³ · ${up.n_faces} faces · units ${up.units}`)),
    h('div', { class: 'row', style: { gap: '8px' } },
      h('a', { class: 'btn', href: withToken(`/api/cad/uploads/${uid}/step`) }, 'STEP in metres'),
      h('button', { class: 'btn', type: 'button', onclick: async () => {
        if (!confirm(`Delete ${up.name} and its ${up.sets.length} problem(s)? Runs and studies already recorded stay.`)) return;
        try { await api(`/api/cad/uploads/${uid}`, { method: 'DELETE' }); location.hash = '#/cad'; } catch (e) { toast(e.message); }
      } }, 'Delete'))));
  for (const w of up.warnings || []) el.append(h('div', { class: 'error-box', style: { marginBottom: '16px' } }, w));

  const vp = h('div', { class: 'viewport', style: { height: '460px' } });
  const msg = h('div', { class: 'center-msg' }, h('span', {}, h('span', { class: 'spinner' }), ' loading the preview…'));
  const legend = h('div', { class: 'chips', style: { marginTop: '10px' } },
    [['fixed', 'clamped'], ['roller', 'roller'], ['load', 'load'], ['selected', 'selected']].map(([k, label]) =>
      h('span', { class: 'legend-item', style: { cursor: 'default' } }, h('span', { class: 'sw', style: { background: ROLE_COLOR[k] } }), label)));
  const left = h('div', { class: 'stack' }, h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('div', { class: 'section-title', style: { margin: 0 } }, 'Geometry: click a face to pick it'),
      h('span', { class: 'small muted' }, 'drag: rotate · right-drag: pan · wheel: zoom')),
    h('div', { class: 'viewport-wrap' }, vp, msg), legend));
  const builder = h('div');
  const right = h('div', { class: 'stack' }, h('div', { class: 'card' }, builder));
  el.append(h('div', { class: 'grid-2' }, left, right));
  const faceList = h('div', { style: { maxHeight: '340px', overflow: 'auto' } });
  left.append(h('div', { class: 'card' }, h('div', { class: 'section-title' }, `Faces (${up.faces.length})`), faceList));
  const problemsBox = h('div');
  const usageBox = h('div');
  el.append(h('div', { style: { height: '24px' } }), h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Problems from this part'),
      h('button', { class: 'btn small', type: 'button', onclick: () => reload() }, 'Refresh')), problemsBox));
  el.append(h('div', { style: { height: '24px' } }), usageBox);
  const tasks = liveTasks({ group: 'cad' }, { empty: 'No reference study or agent run started from this page yet.', limit: 20 });
  el.append(h('div', { style: { height: '24px' } }), h('div', { class: 'card flush' },
    h('div', { class: 'card-head', style: { padding: '16px 22px 0' } }, h('h2', {}, 'Tasks')), tasks.el));

  // ---- selection and roles
  const select = tag => { state.selected = tag; paint(); drawBuilder(); drawFaces(); };
  const paint = () => { if (state.viewer) state.viewer.paint(state.roles, state.selected); };
  const drawFaces = () => {
    clear(faceList).append(h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['face', 'kind', 'area', 'centroid (mm)', 'role'].map(x => h('th', {}, x)))),
      h('tbody', {}, up.faces.map(f => {
        const r = state.roles.get(f.tag);
        return h('tr', { class: 'click', style: f.tag === state.selected ? { background: 'var(--accent-wash)' } : null, onclick: () => select(f.tag) },
          h('td', { class: 'mono' }, `f${f.tag}`), h('td', {}, f.kind),
          h('td', { class: 'num' }, `${fmt(f.area * 1e6, 4)} mm²`),
          h('td', { class: 'small mono' }, f.centroid.map(mm).join(', ')),
          h('td', { class: 'small' }, r ? roleText(r) : ''));
      }))));
  };
  const builderState = { tab: 'define', count: 3, E: overview.defaults.E / 1e9, nu: overview.defaults.nu, cap: overview.defaults.node_cap, label: '' };
  const drawBuilder = () => drawBuilderInto(builder, up, faces, state, builderState, { select, paint, drawFaces, reload });

  // ---- problems of this upload
  let shown = '';
  async function reload(force = true) {
    let fresh;
    try { [fresh, overview] = await Promise.all([api(`/api/cad/uploads/${encodeURIComponent(uid)}`), api('/api/cad')]); }
    catch (e) { toast(e.message); return; }
    up = { ...up, sets: fresh.sets, meshes: fresh.meshes };
    const key = JSON.stringify(up.sets);
    if (force || key !== shown) drawProblems();     // polling leaves an open form alone unless something changed
  }
  let openPanel = null;
  const drawProblems = () => {
    shown = JSON.stringify(up.sets);
    clear(problemsBox).append(problemsTable(up, overview, { reload, onTask: () => tasks.restart(), isOpen: k => openPanel === k, toggle: k => { openPanel = openPanel === k ? null : k; drawProblems(); } }));
    const chosen = up.sets.find(s => s.id === openPanel?.split(':')[1]) || up.sets[0] || null;
    clear(usageBox).append(usageCard(overview, chosen));
  };
  drawBuilder();
  drawFaces();
  drawProblems();
  let wasBusy = false;
  const poll = async () => {
    if (!state.alive) return;
    let busy = false;
    try {
      const t = await api('/api/tasks?group=cad&limit=20');
      busy = t.tasks.some(x => x.state === 'queued' || x.state === 'running');
    } catch { /* shown by the task table */ }
    busy = busy || up.sets.some(s => s.studies.some(x => x.status === 'queued' || x.status === 'running'));
    if (busy || wasBusy) await reload(false);      // once more after the last task ends: its result
    wasBusy = busy;
    if (state.alive) state.timer = setTimeout(poll, busy ? 4000 : 15000);
  };
  state.timer = setTimeout(poll, 4000);

  faceViewer(vp, msg, uid, select).then(v => { state.viewer = v; paint(); }).catch(e => { msg.textContent = `Preview unavailable: ${e.message}`; });
  return () => { state.alive = false; clearTimeout(state.timer); tasks.stop(); if (state.viewer) state.viewer.dispose(); hideTip(); };
}

/** A coordinate in metres as mm; the CAD kernel's round-off (1e-15) shows as 0. */
const mm = v => (Math.abs(v) < 1e-9 ? '0' : fmt(v * 1000, 4));

/** taskTable whose list can be restarted (it polls slowly while nothing runs). */
function liveTasks(filter, opts) {
  const box = h('div');
  let t = taskTable(filter, opts);
  box.append(t.el);
  return { el: box, restart: () => { t.stop(); t = taskTable(filter, opts); clear(box).append(t.el); }, stop: () => t.stop() };
}

function roleText(r) {
  if (r.role === 'fixed') return 'clamped';
  if (r.role === 'roller') return `roller ${AXES[r.component]} = 0`;
  if (r.role === 'pressure') return `pressure ${fmt(r.mpa, 4)} MPa`;
  return `force (${r.force.map(v => fmt(v, 4)).join(', ')}) N`;
}

// ------------------------------------------------------------------ BC builder (define / generate)
function drawBuilderInto(box, up, faces, state, bs, fx) {
  const tab = (k, label) => h('button', { type: 'button', class: bs.tab === k ? 'on' : '', onclick: () => { bs.tab = k; fx.select(state.selected); } }, label);
  const err = h('div');
  const num = (key, opts = {}) => h('input', { type: 'number', value: bs[key], step: 'any', style: { width: opts.w || '120px' }, oninput: e => { bs[key] = e.target.value; } });
  const common = h('div', { class: 'row', style: { gap: '16px', marginTop: '14px' } },
    h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'E (GPa)'), num('E', { w: '100px' })),
    h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Poisson’s ratio'), num('nu', { w: '90px' })),
    h('label', { class: 'field', title: 'the agents’ mesh limit (a safety limit, not a target)' }, h('span', { class: 'section-title' }, 'Node cap'), num('cap', { w: '110px' })));
  const material = () => ({ material: { E: Number(bs.E) * 1e9, nu: Number(bs.nu) }, node_cap: Number(bs.cap) });
  const created = async ids => { toast(`Created ${ids.join(', ')}`); await fx.reload(); };
  clear(box).append(h('div', { class: 'card-head' }, h('h2', { style: { margin: 0 } }, 'Supports and loads'),
    h('div', { class: 'seg' }, tab('define', 'Define'), tab('generate', 'Generate'))));
  if (bs.tab === 'generate') {
    const btn = h('button', { class: 'btn primary', type: 'button', onclick: async () => {
      clear(err);
      btn.disabled = true;
      try { const r = await postJSON(`/api/cad/uploads/${up.id}/sets`, { mode: 'auto', count: Number(bs.count), ...material() }); await created(r.problem_ids); }
      catch (e) { err.append(errorBox(e)); }
      btn.disabled = false;
    } }, 'Generate BC sets');
    box.append(h('p', { class: 'secondary' }, 'The gold generator’s rules (feacore.goldgen.bcgen): one or two clamped faces or a clamp and a roller, then a pressure, a normal or shear force, or a bearing load on faces away from the supports, 1-50 kN or 1-50 MPa. Seeded by the upload, so the same part gets the same sets. Each set becomes a problem.'),
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'How many'), h('input', { type: 'number', min: 1, max: 20, value: bs.count, style: { width: '100px' }, oninput: e => { bs.count = e.target.value; } })),
      common, h('div', { style: { marginTop: '16px' } }, btn), err);
    return;
  }
  const f = state.selected != null ? faces.get(state.selected) : null;
  const sel = h('div', { class: 'card tight', style: { background: 'var(--surface-2)' } });
  if (!f) sel.append(h('span', { class: 'muted' }, 'Click a face in the 3D view or in the face list.'));
  else {
    const cur = state.roles.get(f.tag);
    const assign = r => { state.roles.set(f.tag, r); fx.paint(); fx.drawFaces(); fx.select(f.tag); };
    const forceIn = (cur && cur.role === 'force' ? cur.force : defaultForce(f)).map(v => h('input', { type: 'number', step: 'any', value: v, style: { width: '96px' } }));
    const pIn = h('input', { type: 'number', step: 'any', value: cur && cur.role === 'pressure' ? cur.mpa : 1, style: { width: '96px' } });
    const rollerSel = h('select', {}, AXES.map((a, k) => h('option', { value: k, selected: cur && cur.role === 'roller' && cur.component === k ? true : null }, `${a} displacement = 0`)));
    sel.append(
      h('div', { class: 'strong' }, `f${f.tag} · ${f.kind} · ${fmt(f.area * 1e6, 4)} mm²`,
        cur ? h('span', { class: 'secondary' }, ` · now: ${roleText(cur)}`) : null),
      h('div', { class: 'small muted' }, `centroid (${f.centroid.map(mm).join(', ')}) mm`,
        f.normal ? ` · outward normal (${f.normal.map(v => fmt(v, 3)).join(', ')})` : '', f.axis ? ` · axis (${f.axis.map(v => fmt(v, 3)).join(', ')}), radius ${fmt((f.radius || 0) * 1000, 4)} mm` : ''),
      h('div', { class: 'row', style: { gap: '8px', marginTop: '12px' } },
        h('button', { class: 'btn small', type: 'button', onclick: () => assign({ role: 'fixed' }) }, 'Clamp'),
        h('span', { class: 'row', style: { gap: '6px' } }, rollerSel, h('button', { class: 'btn small', type: 'button', onclick: () => assign({ role: 'roller', component: Number(rollerSel.value) }) }, 'Roller')),
        cur ? h('button', { class: 'btn small ghost', type: 'button', onclick: () => { state.roles.delete(f.tag); fx.paint(); fx.drawFaces(); fx.select(f.tag); } }, 'Free') : null),
      h('div', { class: 'row', style: { gap: '8px', marginTop: '10px', alignItems: 'center' } },
        h('span', { class: 'small secondary' }, 'Force (N)'), ...forceIn,
        h('button', { class: 'btn small', type: 'button', onclick: () => {
          const v = forceIn.map(i => Number(i.value));
          if (!v.some(x => x)) { toast('The force is zero'); return; }
          assign({ role: 'force', force: v });
        } }, 'Load: force')),
      h('div', { class: 'row', style: { gap: '8px', marginTop: '8px', alignItems: 'center' } },
        h('span', { class: 'small secondary' }, 'Pressure (MPa, pushes in)'), pIn,
        h('button', { class: 'btn small', type: 'button', onclick: () => {
          const p = Number(pIn.value);
          if (!p) { toast('The pressure is zero'); return; }
          assign({ role: 'pressure', mpa: p });
        } }, 'Load: pressure')));
  }
  const assigned = [...state.roles.entries()];
  const label = h('input', { type: 'text', value: bs.label, placeholder: 'optional, e.g. bolt preload', style: { width: '260px' }, oninput: e => { bs.label = e.target.value; } });
  const create = h('button', { class: 'btn primary', type: 'button', onclick: async () => {
    clear(err);
    const spec = { mode: 'manual', fixed: [], rollers: [], loads: [], label: bs.label, ...material() };
    for (const [tag, r] of state.roles) {
      const g = `f${tag}`;
      if (r.role === 'fixed') spec.fixed.push(g);
      else if (r.role === 'roller') spec.rollers.push({ group: g, component: r.component });
      else if (r.role === 'pressure') spec.loads.push({ group: g, type: 'pressure', value: r.mpa * 1e6 });
      else spec.loads.push({ group: g, type: 'force', value: r.force });
    }
    create.disabled = true;
    try {
      const r = await postJSON(`/api/cad/uploads/${up.id}/sets`, spec);
      state.roles.clear();
      state.selected = null;
      fx.paint();
      fx.drawFaces();
      await created(r.problem_ids);
      fx.select(null);
    } catch (e) { err.append(errorBox(e)); create.disabled = false; }
  } }, 'Create problem');
  box.append(sel,
    h('div', { class: 'section-title', style: { marginTop: '16px' } }, 'Your BC set'),
    assigned.length ? h('table', { class: 'data' }, h('tbody', {}, assigned.map(([tag, r]) => h('tr', {},
      h('td', { class: 'mono' }, h('a', { href: 'javascript:void 0', onclick: () => fx.select(tag) }, `f${tag}`)), h('td', {}, roleText(r)),
      h('td', {}, h('button', { class: 'btn small ghost', type: 'button', title: 'remove', onclick: () => { state.roles.delete(tag); fx.paint(); fx.drawFaces(); fx.select(state.selected); } }, '✕'))))))
      : h('p', { class: 'muted small' }, 'Nothing yet: clamp at least one face and load another (one support or load per face).'),
    common,
    h('div', { class: 'row', style: { gap: '12px', marginTop: '14px', alignItems: 'flex-end' } },
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Label'), label), create),
    err);
}

/** 'clamp f18', 'roller x f7', 'force 2 kN f27', 'pressure 1 MPa f1' (as the Gold page). */
function bcShort(b) {
  if (b.type === 'fixed') return `clamp ${b.group}`;
  if (b.type === 'displacement') return `roller ${(b.components || []).map(k => AXES[k]).join('')} ${b.group}`;
  if (b.type === 'pressure') return `pressure ${fmt(b.value / 1e6, 3)} MPa ${b.group}`;
  if (b.type === 'force') return `force ${fmt(Math.hypot(...b.value) / 1e3, 3)} kN ${b.group}`;
  return `${b.type} ${b.group || ''}`;
}

function defaultForce(f) {
  const n = f.normal || (f.axis ? perpendicular(f.axis) : [0, 0, 1]);
  return n.map(v => Math.round(-1000 * v * 10) / 10 + 0);
}

function perpendicular(a) {
  const e = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const p = [a[1] * e[2] - a[2] * e[1], a[2] * e[0] - a[0] * e[2], a[0] * e[1] - a[1] * e[0]];
  const n = Math.hypot(...p) || 1;
  return p.map(v => v / n);
}

// ------------------------------------------------------------------ problems table
function problemsTable(up, d, ui) {
  if (!up.sets.length) return h('p', { class: 'muted' }, 'No problem yet: define supports and loads above, or generate BC sets.');
  const rows = [];
  for (const s of up.sets) {
    const done = s.studies.filter(x => x.status === 'succeeded' && x.backend === 'skfem' && x.order === 2 && x.n_sizes >= 3);
    const running = s.studies.filter(x => x.status === 'queued' || x.status === 'running');
    const ref = s.references.length
      ? h('div', {}, statusChip('good', `${s.references.length} reference${s.references.length === 1 ? '' : 's'}`),
        s.references.map(r => h('div', { class: 'small', title: `${r.source}; ${r.notes || ''}` }, h('span', { class: 'mono' }, r.qoi), ` ${fmt(r.value, 5)} ± ${fmtPct(r.tolerance, 2)}`)),
        s.singular.length ? h('div', { class: 'small muted' }, `singular: ${s.singular.join(', ')}`) : null)
      : h('div', {}, statusChip('info', 'no reference'), h('div', { class: 'small muted' }, s.levels.length ? 'the study certified nothing' : 'scored reference-free'));
    const act = (k, label, title) => h('button', { class: `btn small${ui.isOpen(`${k}:${s.id}`) ? ' primary' : ''}`, type: 'button', title, onclick: () => ui.toggle(`${k}:${s.id}`) }, label);
    rows.push(h('tr', {},
      h('td', {}, h('div', { class: 'strong' }, s.title), h('a', { class: 'mono small', href: `#/benchmarks/${encodeURIComponent(s.id)}` }, s.id),
        h('div', { class: 'small muted' }, `${s.kind || ''} · ${timeAgo(s.created_at)}`)),
      h('td', { class: 'small', title: s.words.join('\n') }, s.bcs.map(b => h('div', {}, bcShort(b)))),
      h('td', { class: 'small' }, `E ${fmt(s.material.E / 1e9, 4)} GPa, ν ${fmt(s.material.nu, 3)}`, h('div', { class: 'muted' }, `node cap ${fmtInt(s.node_cap)}`), h('div', { class: 'muted' }, `scored: ${s.scored_qoi}`)),
      h('td', {}, ref),
      h('td', { class: 'small' }, s.studies.length ? s.studies.slice(0, 4).map(x => h('div', {}, h('a', { href: `#/study/${x.study_id}` }, x.status), ` · ${x.nested ? 'nested' : 'remeshed'}, ${x.n_sizes} meshes`)) : h('span', { class: 'muted' }, '–'),
        running.length ? h('div', { class: 'muted' }, 'running…') : null),
      h('td', {}, h('div', { class: 'row', style: { gap: '6px' } },
        act('ref', 'Reference study', 'gold-style nested study on the CPU; certifies the QoIs that converge'),
        h('button', { class: 'btn small', type: 'button', title: 'a convergence study with this problem’s default sizes (skfem, P2)', onclick: async e => {
          e.target.disabled = true;
          try { const st = await postJSON('/api/studies', { problem_id: s.id, backend: 'skfem', order: 2 }); toast(`Study ${st.study_id} queued`); await ui.reload(); }
          catch (err) { toast(err.message); e.target.disabled = false; }
        } }, 'Convergence study'),
        done.length ? act('adopt', 'Use a study', 'certify a finished convergence study as the reference') : null,
        act('agents', 'Agents', 'run trained agents on this problem (ace.infer)'),
        h('button', { class: 'btn small ghost', type: 'button', onclick: async () => {
          if (!confirm(`Delete ${s.id}?`)) return;
          try { await api(`/api/cad/problems/${s.id}`, { method: 'DELETE' }); await ui.reload(); } catch (err) { toast(err.message); }
        } }, 'Delete')))));
    const open = ['ref', 'adopt', 'agents'].find(k => ui.isOpen(`${k}:${s.id}`));
    if (open) rows.push(h('tr', {}, h('td', { colspan: 6, style: { background: 'var(--surface-2)' } }, panel(open, s, up, done, ui))));
  }
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['problem', 'supports and loads', 'material', 'reference (evaluator only)', 'studies', ''].map(x => h('th', {}, x)))),
    h('tbody', {}, rows));
}

function panel(kind, s, up, done, ui) {
  if (kind === 'ref') {
    return h('div', {}, h('p', { class: 'small secondary' }, 'Meshes the part with the gold generator’s nested pipeline (the finest level capped), solves this problem on every level, and certifies each QoI with the gold rules. Each solve is recorded for the field viewer. Small parts only: the finest level is solved on this machine’s CPU.'),
      toolForm('cad.reference', { defaults: { upload: up.id, sets: s.id }, hide: ['upload'], submitLabel: 'Start the reference study', onStarted: () => { ui.onTask(); ui.reload(); } }));
  }
  if (kind === 'agents') {
    return toolForm('cad.infer', { defaults: { problem: s.id }, hide: ['problem'], submitLabel: 'Run the agents', onStarted: () => { ui.onTask(); ui.reload(); } });
  }
  const pick = h('select', {}, done.map(x => h('option', { value: x.study_id }, `${x.study_id} (${x.nested ? 'nested' : 'remeshed'}, ${x.n_sizes} meshes, ${timeAgo(x.created_at)})`)));
  const out = h('div');
  const go = h('button', { class: 'btn primary small', type: 'button', onclick: async () => {
    go.disabled = true;
    clear(out);
    try {
      const r = await postJSON(`/api/cad/problems/${s.id}/adopt`, { study_id: pick.value });
      toast(r.references.length ? `Certified ${r.references.map(x => x.qoi).join(', ')}`
        : 'No QoI converged within the gold tolerance on this study: still no reference', 6000);
      ui.toggle(`adopt:${s.id}`);       // closes the panel; the table shows the outcome
      await ui.reload();
    } catch (e) { out.append(errorBox(e)); go.disabled = false; }
  } }, 'Certify as the reference');
  return h('div', {}, h('p', { class: 'small secondary' }, 'The study’s levels go through the gold certification: a QoI becomes a reference when it converges and its fine-grid GCI is within the gold tolerance. Independently remeshed studies scatter the observed order; nested ones certify more.'),
    h('div', { class: 'row', style: { gap: '10px', alignItems: 'center' } }, pick, go), out);
}

// ------------------------------------------------------------------ 3D view with face picking
async function faceViewer(vp, msg, uid, onPick) {
  const g = await api(`/api/cad/uploads/${encodeURIComponent(uid)}/geometry`);
  const [{ MeshView, webglAvailable }, THREE] = await Promise.all([import('../mesh3d.js'), import('three')]);
  if (!webglAvailable()) throw new Error('WebGL is not available in this browser');
  msg.remove();
  const view = new MeshView(vp, { dim: 3, margin: 1.1 });
  const base = { dim: 3, points: decode(g.points), triangles: decode(g.triangles), triEntity: decode(g.tri_entity),
    lines: decode(g.lines), lineEntity: decode(g.line_entity) };
  const order = ['fixed', 'roller', 'load', 'selected'];
  const paint = (roles, selected) => {
    const groups = order.map(name => ({ name, dim: 2, entities: [] }));
    const entityGroups = { 0: {}, 1: {}, 2: {}, 3: {} };
    for (const [tag, r] of roles || []) {
      const gi = r.role === 'fixed' ? 0 : r.role === 'roller' ? 1 : 2;
      groups[gi].entities.push(tag);
      entityGroups[2][tag] = [gi];
    }
    if (selected !== null && selected !== undefined) { groups[3].entities.push(selected); entityGroups[2][selected] = [3, ...(entityGroups[2][selected] || [])]; }
    view.setGeometry({ ...base, groups, entityGroups });
    view.colorOf = new Map(order.map((k, i) => [i, ROLE_COLOR[k]]));
    view.setHighlight(null);
  };
  paint(null, null);
  view.fit('iso');
  const ray = new THREE.Raycaster();
  const canvas = view.renderer.domElement;
  const pick = e => {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), view.camera);
    const mesh = view.group.children.find(o => o.isMesh);
    const hit = mesh ? ray.intersectObject(mesh, false)[0] : null;
    return hit ? base.triEntity[hit.faceIndex] : null;
  };
  let down = null;
  canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', e => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) { const tag = pick(e); if (tag !== null) onPick(tag); }
    down = null;
  });
  canvas.addEventListener('pointermove', e => {
    if (e.buttons) return;
    const tag = pick(e);
    if (tag === null) { hideTip(); canvas.style.cursor = ''; return; }
    canvas.style.cursor = 'pointer';
    showTip(e, `<b>f${tag}</b> · click to pick`);
  });
  canvas.addEventListener('pointerleave', () => hideTip());
  return { paint, dispose: () => view.dispose() };
}
