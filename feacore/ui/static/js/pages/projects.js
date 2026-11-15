// Projects: verification of one component by analysis, in the context of its system (dossier/).
//   #/projects                      the projects, and "New project from example"
//   #/projects/<folder>[?tab=...]   one project: status, run buttons with the live log, and the tabs
//                                   report | inputs | numbers | loadcases | reviews
// Server: feacore/ui/dossier_api.py (GET/POST /api/dossier...), tools in feacore/ui/tools/dossier.py (the
// pipeline runs on the task lane). The report is dossier's own page (out/report/index.html) in a frame; this
// page links into it by record id. Text from the project files is put on the page as text, never as HTML.
import { api, postJSON, authFetch, withToken, h, clear, loading, errorBox, timeAgo, statusChip, toast, qs } from '../api.js';
import { taskLog, taskChip, cancelButtons } from '../tasks.js';

const enc = encodeURIComponent;
const TABS = [['report', 'Report'], ['inputs', 'Inputs'], ['numbers', 'Numbers'], ['loadcases', 'Load cases'], ['reviews', 'Reviews']];
const STAGE_WORD = { done: 'done', stale: 'stale', not_run: 'not run', partial: 'in part', running: 'running', failed: 'failed' };
const VERDICT = { pass: ['good', 'pass'], fail: ['critical', 'fail'], inconclusive: ['warning', 'inconclusive'], not_assessed: ['', 'not assessed'] };
const REVIEW = { open: ['info', 'Not reviewed'], accepted: ['good', 'Accepted'], rejected: ['critical', 'Rejected'],
                 changes_requested: ['warning', 'Changes requested'], stale: ['serious', 'Changed after the review'] };
const SEVERITY = { error: 'critical', major: 'serious', minor: 'warning', info: '' };
const NAME_KEY = 'feacore_reviewer';
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const words = s => String(s || '').replace(/_/g, ' ');
const kB = n => (n === null || n === undefined) ? '' : n < 1000 ? `${n} B` : n < 1e6 ? `${(n / 1000).toFixed(n < 1e4 ? 1 : 0)} kB` : `${(n / 1e6).toFixed(1)} MB`;
// empty an element and append the parts (the DOM's append would write null as the text "null")
const put = (el, ...kids) => { clear(el).append(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };
const dot = cls => h('span', { class: `dot ${cls || ''}` });
const store = { get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
                set(k, v) { try { v === null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch { /* private mode */ } } };

export async function render(el, args) {
  if (args && args[0]) return project(el, args[0]);
  return list(el);
}

// ------------------------------------------------------------------ shared pieces
function stageTitle(s) {
  const parts = [`${s.stage}: ${STAGE_WORD[s.state] || s.state}`];
  if (s.error) parts.push(s.error);
  if (s.summary) parts.push(s.summary);
  if (s.written) parts.push(`written ${timeAgo(s.written)}`);
  return parts.join('\n');
}

/** The seven stages of the pipeline, each with its state. `big` adds the state in words. */
function stageStrip(stages, big = false) {
  return h('div', { class: `pj-strip${big ? ' big' : ''}` }, (stages || []).map(s =>
    h('div', { class: `pj-stage ${s.state}`, title: stageTitle(s) },
      h('span', { class: 'bar' }), h('span', { class: 'name' }, s.stage),
      big ? h('span', { class: 'word' }, STAGE_WORD[s.state] || s.state) : null)));
}

function verdictCounts(p) {
  const v = p.verdicts || {};
  const shown = Object.keys(VERDICT).filter(k => v[k]);
  if (!shown.length) return h('span', { class: 'muted' }, 'No verdicts yet');
  return h('span', { class: 'pj-counts' }, h('span', { class: 'muted' }, 'Verdicts'), shown.map(k => h('span', { class: 'gate' }, dot(VERDICT[k][0]), `${v[k]} ${VERDICT[k][1]}`)));
}

function findingCounts(p) {
  const f = p.findings || {};
  const shown = Object.keys(SEVERITY).filter(k => f[k]);
  if (!shown.length) return h('span', { class: 'muted' }, 'No findings');
  return h('span', { class: 'pj-counts' }, h('span', { class: 'muted' }, 'Findings'), shown.map(k => h('span', { class: 'gate' }, dot(SEVERITY[k]), `${f[k]} ${k}`)));
}

function reviewCounts(p) {
  const r = p.reviews || {};
  if (!r.total) return h('span', { class: 'muted' }, 'Nothing to review yet');
  const done = r.accepted + r.rejected + r.changes_requested;
  return h('span', { class: 'pj-counts' }, h('span', { class: 'muted' }, 'Reviews'), `${done} of ${r.total}`, r.stale ? `${r.stale} changed after the review` : null);
}

// ------------------------------------------------------------------ the list
async function list(el) {
  const fresh = h('div');
  const cards = h('div', {}, loading('Loading the projects…'));
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Projects'),
    h('p', { class: 'lede' }, 'A project verifies one component by analysis, in the context of its system. It checks the requirements against the system, selects the load cases, analyses them and reports the margins. An engineer reviews the result and signs it.'))),
  h('div', { class: 'stack' }, cards, fresh));
  let alive = true, timer = null, shown = null;
  const tick = async () => {
    let d;
    try { d = await api('/api/dossier'); } catch (e) { if (alive && shown === null) put(cards, errorBox(e)); return; }
    if (!alive) return;
    const key = JSON.stringify(d.projects);
    if (key !== shown) {
      if (shown === null) put(fresh, newProject(d));
      shown = key;
      put(cards, d.projects.length ? h('div', { class: 'pj-cards' }, d.projects.map(card))
        : h('div', { class: 'card empty-state' }, h('h2', {}, 'No project yet'),
          h('p', {}, 'Make the first project from the example below. It is a copy that you can change: the system, the requirements and the CAD.')),
      d.ignored.length ? h('p', { class: 'muted small', style: { marginTop: '12px' } }, d.ignored.map(x => `${x.folder}: ${x.reason}`).join(' ')) : null);
    }
    timer = setTimeout(tick, d.projects.some(p => p.active_task) ? 4000 : 20000);
  };
  tick();
  return () => { alive = false; clearTimeout(timer); };
}

function card(p) {
  if (p.error) return h('div', { class: 'card pcard' }, h('div', { class: 'title' }, p.folder), errorBox(p.error));
  const run = p.last_run;
  return h('a', { class: 'card pcard', href: `#/projects/${enc(p.folder)}` },
    h('div', {}, h('div', { class: 'title' }, p.name), h('div', { class: 'id' }, [p.folder, p.id].filter(Boolean).join(' · '))),
    h('div', { class: 'meta-line' }, p.subject.name || p.subject.id || 'no component', p.subject.material ? ` · ${p.subject.material}` : ''),
    stageStrip(p.stages),
    !p.ok ? h('div', { class: 'gate small' }, dot('critical'), `${plural(p.n_problems, 'problem')} in the inputs`) : null,
    p.active_task ? h('div', { class: 'gate small' }, h('span', { class: 'spinner' }), `${run.title}${run.stage ? `: ${run.stage}` : ''}`) : null,
    h('div', { class: 'foot' }, verdictCounts(p), findingCounts(p), reviewCounts(p)));
}

function newProject(d) {
  if (!d.examples.length) return h('div', { class: 'card' }, h('h2', {}, 'New project from example'), h('p', { class: 'muted' }, 'This server has no example project.'));
  const taken = new Set([...d.projects.map(p => p.folder), ...d.ignored.map(x => x.folder)]);
  const free = base => { let name = base, k = 2; while (taken.has(name)) name = `${base}_${k++}`; return name; };
  const example = h('select', { style: { height: '40px' } }, d.examples.map(e => h('option', { value: e.name }, `${e.name}: ${e.title}`)));
  const folder = h('input', { type: 'text', value: free(d.examples[0].name), style: { width: '260px', height: '40px' }, spellcheck: 'false' });
  const about = h('p', { class: 'secondary small', style: { margin: '12px 0 0', maxWidth: '100ch' } }, d.examples[0].description);
  const err = h('div');
  let touched = false;
  folder.addEventListener('input', () => { touched = true; });
  example.addEventListener('change', () => {
    const e = d.examples.find(x => x.name === example.value);
    about.textContent = e.description;
    if (!touched) folder.value = free(e.name);
  });
  const btn = h('button', { class: 'btn primary', type: 'button' }, 'Create project');
  const go = async () => {
    clear(err);
    btn.disabled = true;
    try {
      const p = await postJSON('/api/dossier', { example: example.value, folder: folder.value.trim() });
      toast(`Created ${p.folder}`);
      location.hash = `#/projects/${enc(p.folder)}`;
    } catch (e) { err.append(errorBox(e)); btn.disabled = false; }
  };
  btn.onclick = go;
  folder.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  return h('div', { class: 'card' },
    h('h2', {}, 'New project from example'),
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '18px' } },
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Example'), example),
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Folder name'), folder),
      btn),
    about,
    h('p', { class: 'muted small', style: { margin: '8px 0 0' } }, 'The project is a copy of the example in ', h('code', {}, d.root), '. The example itself does not change.'),
    err);
}

// ------------------------------------------------------------------ one project
async function project(el, folder) {
  const API = `/api/dossier/${enc(folder)}`;
  el.append(loading('Loading the project…'));
  let st;
  try { st = await api(API); } catch (e) { put(el, h('p', { class: 'secondary small' }, h('a', { href: '#/projects' }, '← Projects')), errorBox(e)); return; }

  const S = { alive: true, timer: null, key: '', tab: TABS.some(([k]) => k === qs().get('tab')) ? qs().get('tab') : 'report',
              shownTask: null, hiddenTask: null, log: null, logOpen: null, wasActive: st.active_task, check: null };
  const head = h('div');
  const runBox = h('div', { class: 'pj-run' });
  const tabBar = h('div', { class: 'seg' });
  const panes = h('div');
  // a part of the pipeline: from one stage to another, some load cases only
  const stageNames = st.stages.map(s => s.stage);
  const pick = value => h('select', { style: { height: '40px' } }, stageNames.map(s => h('option', { value: s, selected: s === value ? true : null }, s)));
  const part = { from: pick(stageNames[0]), to: pick(stageNames[stageNames.length - 1]),
                 only: h('input', { type: 'text', placeholder: 'all', style: { width: '220px', height: '40px' }, spellcheck: 'false' }),
                 quick: h('input', { type: 'checkbox', checked: true }), force: h('input', { type: 'checkbox' }),
                 go: h('button', { class: 'btn', type: 'button' }, 'Run') };
  part.go.onclick = () => startTask('dossier.run', { from: part.from.value, to: part.to.value, quick: part.quick.checked, force: part.force.checked,
                                                    ...(part.only.value.trim() ? { only: part.only.value.trim() } : {}) }, part.go);
  const partBox = h('details', { class: 'pj-part' }, h('summary', {}, 'Run a part of the pipeline'),
    h('div', { class: 'row', style: { gap: '16px', alignItems: 'flex-end', marginTop: '10px' } },
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'From stage'), part.from),
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'To stage'), part.to),
      h('label', { class: 'field', title: 'Load case ids, for example LC-FIRE LC-PROOF' }, h('span', { class: 'section-title' }, 'Only these load cases'), part.only),
      h('label', { class: 'check', title: 'Coarse meshes, two mesh levels, no pictures.' }, part.quick, 'quick'),
      h('label', { class: 'check', title: 'Analyse a load case again also when its analysis is current.' }, part.force, 'analyse again'),
      part.go));
  put(el, head, partBox, runBox, h('div', { class: 'toolbar', style: { marginTop: '22px' } }, tabBar), panes);

  // ---- starting work
  const startTask = async (tool, args, btn) => {
    if (btn) btn.disabled = true;
    try {
      const t = await postJSON(`${API}/tasks`, { tool, args });
      S.hiddenTask = null;
      toast(`Started: ${t.title}`);
      await refresh();
    } catch (e) {
      const errs = e.body && e.body.detail && e.body.detail.errors;
      toast(errs ? errs.join(' ') : e.message, 6000);
      if (btn) btn.disabled = false;
    }
  };
  const checkInputs = async btn => {
    btn.disabled = true;
    try { S.check = await api(`${API}/check`, { method: 'POST' }); } catch (e) { toast(e.message, 6000); btn.disabled = false; return; }
    btn.disabled = false;
    tabs.inputs.problems(S.check, false);
    drawRun();
    if (!S.check.ok) setTab('inputs');
  };

  // ---- header: name, status, buttons
  const drawHead = () => {
    const busy = !!st.active_task;
    const run = st.last_run;
    part.go.disabled = busy;
    const button = (label, tool, args, cls = '', title = '') => h('button', { class: `btn ${cls}`, type: 'button', disabled: busy || null, title: busy ? 'A run of this project is in progress.' : title,
      onclick: e => startTask(tool, args, e.currentTarget) }, label);
    const check = h('button', { class: 'btn', type: 'button', title: 'Read the input files and evaluate every number. It takes a second.', onclick: e => checkInputs(e.currentTarget) }, 'Check inputs');
    let last = null;
    if (run && run.state !== 'done' && !busy) {
      last = h('span', { class: 'gate' }, dot('critical'), `Last run ${run.state}${run.stage ? ` at ${run.stage}` : ''} ${timeAgo(run.finished_at)}`,
        run.message ? `: ${run.message}` : '', ' ', h('a', { href: 'javascript:void 0', onclick: () => { S.hiddenTask = null; drawRun(true); } }, 'log'));
    } else if (run && !busy) {
      last = h('span', { class: 'muted' }, `Last run ${timeAgo(run.finished_at)}: ${run.done.length ? `${run.done.join(', ')} done` : run.title}`);
    }
    put(head,
      h('div', { class: 'page-head', style: { marginBottom: '18px' } },
        h('div', {},
          h('div', { class: 'secondary small' }, h('a', { href: '#/projects' }, '← Projects')),
          h('h1', {}, st.name),
          h('div', { class: 'meta-line' }, h('span', { class: 'mono' }, folder), st.id ? ` · ${st.id}` : '',
            st.subject.name ? ` · ${st.subject.name}` : '', st.subject.id ? ` (${st.subject.id})` : '', st.subject.material ? ` · ${st.subject.material}` : '')),
        h('div', { class: 'row', style: { gap: '8px' } }, check,
          button('Run verification (quick)', 'dossier.run', { quick: true }, 'primary', 'Every stage with coarse meshes, two mesh levels and no pictures.'),
          button('Run verification (full)', 'dossier.run', { quick: false }, '', 'Every stage with the full mesh studies and the pictures. It can take a long time.'),
          button('Build report', 'dossier.report', {}, '', 'Write the report again from the stage files that exist.'))),
      st.ok ? null : h('div', { class: 'flag-list', style: { margin: '0 0 14px' } }, h('div', { class: 'flag critical' }, h('span', { class: 'ico' }, '✕'),
        h('div', {}, h('b', {}, `The inputs have ${plural(st.n_problems, 'problem')}`), h('span', {}, 'The pipeline cannot run until they are right. '),
          h('a', { href: 'javascript:void 0', onclick: () => setTab('inputs') }, 'Open the inputs')))),
      h('div', { class: 'card tight pj-status' },
        stageStrip(st.stages, true),
        h('div', { class: 'pj-facts small' },
          verdictCounts(st), findingCounts(st), reviewCounts(st), last)));
  };

  // ---- the task that runs (or the last one) with its live log, and the result of "Check inputs"
  const drawRun = (force = false) => {
    const t = (st.tasks || [])[0];
    const parts = [];
    if (S.check) {
      parts.push(h('div', { class: 'card tight pj-check' }, S.check.ok
        ? h('span', { class: 'gate' }, statusChip('good', 'The inputs are right'), h('span', { class: 'secondary' }, S.check.summary))
        : h('span', { class: 'gate' }, statusChip('critical', `${plural(S.check.problems.length, 'problem')} in the inputs`),
          h('a', { href: 'javascript:void 0', onclick: () => setTab('inputs') }, 'Show them')),
      h('button', { class: 'btn small ghost', type: 'button', onclick: () => { S.check = null; drawRun(); } }, 'Hide')));
    }
    const show = t && S.hiddenTask !== t.id && (force || st.active_task === t.id || S.shownTask === t.id);
    if (show) {
      const live = t.state === 'queued' || t.state === 'running';
      if (S.shownTask !== t.id) { S.shownTask = t.id; S.logOpen = null; if (S.log) { S.log.stop(); S.log = null; } }
      if (force) S.logOpen = true;
      const open = S.logOpen !== null ? S.logOpen : (live || t.state !== 'done');     // a run that ended well folds its log
      if (open && !S.log) S.log = taskLog(t.id, { height: '240px' });
      if (!open && S.log) { S.log.stop(); S.log = null; }
      parts.push(h('div', { class: 'card tight' },
        h('div', { class: 'card-head', style: { marginBottom: open ? '8px' : 0 } },
          h('div', { class: 'gate' }, h('b', {}, t.title), taskChip(t.state),
            h('span', { class: 'muted small' }, `${t.args.quick ? 'quick · ' : ''}${t.args.from && t.args.to ? `${t.args.from} to ${t.args.to} · ` : ''}started ${timeAgo(t.started_at || t.created_at)}`)),
          h('div', { class: 'row', style: { gap: '6px', alignItems: 'center' } },
            live ? cancelButtons({ ...t, out: null }, refresh) : null,
            h('button', { class: 'btn small ghost', type: 'button', onclick: () => { S.logOpen = !open; drawRun(); } }, open ? 'Hide the log' : 'Show the log'),
            h('a', { class: 'btn small ghost', href: `#/tasks/${t.id}` }, 'Task page'),
            live ? null : h('button', { class: 'btn small ghost', type: 'button', title: 'Close', onclick: () => { S.hiddenTask = t.id; drawRun(); } }, '✕'))),
        open ? S.log.el : null));
    } else {
      if (S.log) { S.log.stop(); S.log = null; }
      S.shownTask = null;
    }
    put(runBox, parts);
    tabs.report.fit();
  };

  // ---- tabs
  const openInReport = id => { setTab('report'); tabs.report.goto(id); };
  const ctx = { API, folder, status: () => st, refresh: () => refresh(), openInReport, setTab: k => setTab(k), startTask,
                checked: res => { if (S.check) { S.check = res; drawRun(); } } };
  const tabs = { report: reportTab(ctx), inputs: inputsTab(ctx), numbers: numbersTab(ctx), loadcases: loadCasesTab(ctx), reviews: reviewsTab(ctx) };
  for (const [k] of TABS) { tabs[k].el.hidden = true; panes.append(tabs[k].el); }
  // the tab names, with what waits there: records without a review, problems of the inputs
  const drawTabs = () => put(tabBar, TABS.map(([key, label]) => h('button', { type: 'button', class: key === S.tab ? 'on' : '', onclick: () => setTab(key) }, label,
    key === 'reviews' && st.reviews.open + st.reviews.stale ? h('span', { class: 'pj-n', title: 'records without a current review' }, st.reviews.open + st.reviews.stale) : null,
    key === 'inputs' && !st.ok ? h('span', { class: 'pj-n bad', title: 'problems of the inputs' }, st.n_problems) : null)));
  function setTab(k) {
    S.tab = k;
    drawTabs();
    for (const [key] of TABS) tabs[key].el.hidden = key !== k;
    el.classList.toggle('pj-fill', k === 'report');
    history.replaceState(null, '', `#/projects/${enc(folder)}?tab=${k}`);
    tabs[k].show(st);
  }

  // ---- keeping the page current
  async function refresh() {
    let next;
    try { next = await api(API); } catch { return; }
    if (!S.alive) return;
    const key = JSON.stringify(next);
    const ended = S.wasActive && !next.active_task;
    S.wasActive = next.active_task;
    if (key === S.key) return;
    S.key = key;
    st = next;
    if (S.check && (S.check.ok !== st.ok || S.check.problems.length !== st.n_problems)) S.check = null;   // the inputs changed
    drawHead();
    drawRun();
    drawTabs();
    for (const [k] of TABS) tabs[k].update(st, ended);
    if (ended && st.last_run) toast(st.last_run.state === 'done' ? 'The run is done.' : `The run ${st.last_run.state}.`, 4000);
  }
  const poll = async () => {
    if (!S.alive) return;
    await refresh();
    if (S.alive) S.timer = setTimeout(poll, st.active_task ? 2500 : 12000);
  };

  S.key = JSON.stringify(st);
  drawHead();
  drawRun();
  setTab(S.tab);
  S.timer = setTimeout(poll, st.active_task ? 1500 : 12000);
  const onResize = () => tabs.report.fit();
  window.addEventListener('resize', onResize);
  return () => {
    S.alive = false;
    clearTimeout(S.timer);
    window.removeEventListener('resize', onResize);
    if (S.log) S.log.stop();
    for (const [k] of TABS) if (tabs[k].stop) tabs[k].stop();
  };
}

// ------------------------------------------------------------------ Report: dossier's own page in a frame
function reportTab(ctx) {
  const el = h('div', { class: 'pj-report' });
  const note = h('span', { class: 'small secondary' });
  let frame = null;
  const url = () => withToken(`${ctx.API}/report/index.html`);
  const fit = () => {
    if (!frame || el.hidden) return;
    const top = frame.getBoundingClientRect().top + window.scrollY;
    frame.style.height = `${Math.max(480, window.innerHeight - top - 26)}px`;
  };
  const reload = () => {
    if (!frame) return;
    try { frame.contentWindow.location.reload(); } catch { frame.src = url(); }
  };
  const draw = st => {
    const stage = st.stages.find(s => s.stage === 'report') || {};
    if (!st.report.built) {
      frame = null;
      const busy = !!st.active_task;
      put(el, h('div', { class: 'card empty-state' },
        h('h2', {}, 'The report is not built yet'),
        stage.state === 'failed' ? h('p', {}, `The last run stopped at the report stage: ${stage.error || 'see the log'}`)
          : h('p', {}, 'The report is the last stage of the pipeline. Run the verification, or build the report from the stage files that exist.'),
        h('div', { class: 'row', style: { justifyContent: 'center', gap: '10px' } },
          h('button', { class: 'btn primary', type: 'button', disabled: busy || null, onclick: e => ctx.startTask('dossier.run', { quick: true }, e.currentTarget) }, 'Run verification (quick)'),
          h('button', { class: 'btn', type: 'button', disabled: busy || null, onclick: e => ctx.startTask('dossier.report', {}, e.currentTarget) }, 'Build report'))));
      return;
    }
    if (!frame) {
      frame = h('iframe', { class: 'pj-frame', src: url(), title: 'Verification report' });
      frame.addEventListener('load', fit);
      put(el, h('div', { class: 'pj-bar' }, note,
        h('span', { class: 'row', style: { gap: '6px' } },
          h('button', { class: 'btn small ghost', type: 'button', onclick: reload }, 'Reload'),
          h('a', { class: 'btn small ghost', href: url(), target: '_blank', rel: 'noopener' }, 'Open in a new tab'))),
      frame);
    }
    put(note, `Written ${timeAgo(st.report.written)}.`,
      stage.state === 'stale' ? [' ', h('span', { class: 'gate' }, dot('warning'), 'The stage files changed after that. Build the report again.')] : null,
      stage.state === 'failed' ? [' ', h('span', { class: 'gate' }, dot('critical'), `The last build stopped: ${stage.error || 'see the log'}`)] : null);
    fit();
  };
  return {
    el, fit,
    show: st => { draw(st); fit(); },
    update: (st, ended) => { const had = !!frame; draw(st); if (ended && had) reload(); },
    /** Scroll the report to the record with this id (its anchor). */
    goto: id => {
      if (!frame) return;
      try {
        const target = frame.contentDocument && frame.contentDocument.getElementById(id);
        if (target) { frame.contentWindow.location.hash = `#${id}`; target.scrollIntoView({ block: 'start' }); return; }
      } catch { /* the page is not there yet */ }
      frame.src = `${url()}#${enc(id)}`;
    },
  };
}

// ------------------------------------------------------------------ Inputs: the files, a text editor, the problems
function inputsTab(ctx) {
  const filesBox = h('div');
  const problemsBox = h('div');
  const editorBox = h('div', { class: 'card pj-edit' }, h('p', { class: 'muted' }, 'Choose a file.'));
  const el = h('div', { class: 'pj-inputs' }, h('div', { class: 'stack' }, h('div', { class: 'card pj-side' }, problemsBox), h('div', { class: 'card flush' }, filesBox)), editorBox);
  const draftKey = path => `feacore_dossier_draft:${ctx.folder}:${path}`;
  const F = { files: [], components: [], open: null, text: '', saved: '', sha: null, check: null, loaded: false, wantLine: null, maxCad: 0 };
  let area = null, gutter = null, stateEl = null, saveBtn = null, lineCount = 0;

  const sizeCell = f => h('td', { class: 'num small' }, f.exists ? kB(f.bytes) : 'missing', h('div', { class: 'muted' }, timeAgo(f.modified)));
  const problemsOf = path => ((F.check && F.check.problems) || []).filter(p => p.file === path);
  const dirty = () => F.open !== null && F.text !== F.saved;

  const drawFiles = () => {
    put(filesBox, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Input file'), h('th', { class: 'num' }, 'Size, changed'))),
      h('tbody', {}, F.files.map(f => {
        const n = problemsOf(f.path).length;
        const draft = f.editable && (f.path === F.open ? dirty() : store.get(draftKey(f.path)) !== null);
        const name = h('div', {}, h('span', { class: 'strong' }, f.label), n ? h('span', { class: 'pj-n bad', title: `${plural(n, 'problem')} in this file` }, n) : null,
          draft ? h('span', { class: 'badge', style: { marginLeft: '8px' } }, 'not saved') : null,
          h('div', { class: 'mono small muted' }, f.path));
        if (f.kind === 'cad') {
          const pick = h('input', { type: 'file', accept: '.step,.stp,.STEP,.STP', hidden: true });
          pick.addEventListener('change', () => { if (pick.files[0]) replaceCad(f, pick.files[0]); pick.value = ''; });
          return h('tr', {}, h('td', {}, name, pick, f.inside ? h('div', { class: 'row', style: { gap: '6px', marginTop: '6px' } },
            f.exists ? h('a', { class: 'btn small', href: withToken(`${ctx.API}/files/${f.path}`) }, 'Download') : null,
            f.role === 'cad' ? h('button', { class: 'btn small', type: 'button', onclick: () => pick.click() }, 'Replace…') : null) : null),
          sizeCell(f));
        }
        return h('tr', { class: f.editable ? 'click' : 'dim', style: f.path === F.open ? { background: 'var(--accent-wash)' } : null, onclick: f.editable ? () => open(f.path) : null },
          h('td', {}, name, f.inside ? null : h('div', { class: 'small muted' }, 'Outside the project folder: not shown here.')),
          sizeCell(f));
      }))));
  };

  const drawProblems = () => {
    const c = F.check;
    if (!c) { put(problemsBox, h('h3', {}, 'Problems'), h('p', { class: 'muted small' }, 'Save a file or use “Check inputs” to check the project.')); return; }
    if (c.ok) {
      put(problemsBox, h('h3', {}, 'Problems'), statusChip('good', 'The inputs are right'), c.summary ? h('p', { class: 'secondary small', style: { margin: '8px 0 0' } }, c.summary) : null);
      return;
    }
    put(problemsBox, h('h3', {}, `${plural(c.problems.length, 'problem')}`),
      h('p', { class: 'secondary small' }, c.stage === 'context' ? 'The files read, but the numbers do not evaluate.' : 'The project does not load.'),
      h('div', { class: 'pj-problems' }, c.problems.map(p => {
        const can = p.file && F.files.some(f => f.path === p.file && f.editable);
        return h(can ? 'button' : 'div', { class: 'pj-problem', type: can ? 'button' : null, onclick: can ? () => open(p.file, p.line) : null },
          h('span', { class: 'where' }, p.file ? `${p.file}${p.line ? `, line ${p.line}` : ''}` : 'project'),
          p.field ? h('span', { class: 'what' }, p.field) : null,
          h('span', { class: 'why' }, p.message));
      })));
  };

  const setState = () => {
    if (!stateEl) return;
    const f = F.files.find(x => x.path === F.open);
    stateEl.textContent = dirty() ? 'Not saved' : (f && !f.exists ? 'The file does not exist yet' : 'Saved');
    stateEl.className = `small ${dirty() ? 'pj-dirty' : 'muted'}`;
    if (saveBtn) saveBtn.disabled = !dirty();
  };

  const numbers = () => {
    const n = F.text.split('\n').length;
    const bad = new Map(problemsOf(F.open).filter(p => p.line).map(p => [p.line, p.message]));
    if (n === lineCount && !gutter.dataset.stale) return;
    lineCount = n;
    delete gutter.dataset.stale;
    const frag = document.createDocumentFragment();
    for (let i = 1; i <= n; i++) frag.append(h('div', { class: bad.has(i) ? 'bad' : null, title: bad.get(i) || null }, i));
    clear(gutter).append(frag);
  };

  const gotoLine = line => {
    if (!area || !line) return;
    const lines = area.value.split('\n');
    const n = Math.min(line, lines.length);
    let start = 0;
    for (let i = 0; i < n - 1; i++) start += lines[i].length + 1;
    area.focus();
    area.setSelectionRange(start, start + lines[n - 1].length);
    area.scrollTop = Math.max(0, (n - 1) * 20 - area.clientHeight / 3);
    area.scrollLeft = 0;
  };

  const drawEditor = f => {
    area = h('textarea', { class: 'pj-text', spellcheck: 'false', wrap: 'off', autocomplete: 'off', autocapitalize: 'off', 'aria-label': `${f.path}, YAML text` });
    area.value = F.text;
    gutter = h('div', { class: 'pj-gutter', 'aria-hidden': 'true' });
    stateEl = h('span', {});
    saveBtn = h('button', { class: 'btn primary small', type: 'button', onclick: () => save() }, 'Save');
    area.addEventListener('scroll', () => { gutter.scrollTop = area.scrollTop; });
    area.addEventListener('input', () => {
      F.text = area.value;
      store.set(draftKey(F.open), dirty() ? F.text : null);
      numbers();
      setState();
      drawFiles();
    });
    area.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); save(); }
      if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) { e.preventDefault(); area.setRangeText('  ', area.selectionStart, area.selectionEnd, 'end'); area.dispatchEvent(new Event('input')); }
    });
    lineCount = 0;
    put(editorBox,
      h('div', { class: 'card-head' },
        h('div', {}, h('h2', { style: { margin: 0 } }, f.label), h('div', { class: 'mono small muted' }, f.path)),
        h('div', { class: 'row', style: { gap: '8px', alignItems: 'center' } }, stateEl,
          h('button', { class: 'btn small', type: 'button', title: 'Load the saved file again. Your changes go away.', onclick: () => { store.set(draftKey(F.open), null); open(F.open, null, true); } }, 'Revert'),
          saveBtn)),
      h('div', { class: 'pj-editor' }, gutter, area),
      h('p', { class: 'muted small', style: { margin: '10px 0 0' } }, 'Save writes the file and checks the project. A file with a problem is saved as it is, and the problem shows on the left. Ctrl+S saves.'));
    numbers();
    setState();
    fit();
  };

  const fit = () => {
    if (!area || el.hidden) return;
    const top = area.getBoundingClientRect().top + window.scrollY;
    const hgt = Math.max(360, window.innerHeight - top - 92);
    area.style.height = `${hgt}px`;
    gutter.style.height = `${hgt}px`;
  };

  async function open(path, line = null, force = false) {
    if (path === F.open && !force) { gotoLine(line); return; }
    let d;
    try { d = await api(`${ctx.API}/files/${path}`); }
    catch (e) {
      const f = F.files.find(x => x.path === path);
      if (e.status !== 404 || !f || !f.editable) { toast(e.message, 6000); return; }
      d = { ...f, text: '', sha256: null };           // a file the project names that does not exist yet
    }
    const draft = store.get(draftKey(path));
    F.open = path;
    F.saved = d.text;
    F.sha = d.sha256;
    F.text = draft !== null ? draft : d.text;
    drawEditor(d);
    drawFiles();
    gotoLine(line);
    if (draft !== null && draft !== d.text) toast('Your changes that were not saved are back.', 4000);
  }

  async function save(overwrite = false) {
    if (F.open === null) return;
    const path = F.open, text = F.text;
    saveBtn.disabled = true;
    let res;
    try {
      res = await api(`${ctx.API}/files/${path}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(overwrite || F.sha === null ? { text } : { text, expect: F.sha }) });
    } catch (e) {
      setState();
      if (e.status === 409 && confirm(`${e.message}\n\nSave your text anyway? The other change goes away.`)) return save(true);
      toast(e.message, 8000);
      return;
    }
    F.saved = text;
    F.sha = res.saved.sha256;
    store.set(draftKey(path), null);
    problems(res);
    toast(res.ok ? 'Saved. The inputs are right.' : `Saved. ${plural(res.problems.length, 'problem')} in the inputs.`, 4000);
    await loadFiles();
    setState();
    ctx.refresh();
  }

  async function replaceCad(f, file) {
    if (file.size > F.maxCad) { toast(`The file is ${(file.size / 1e6).toFixed(1)} MB. The limit is ${(F.maxCad / 1e6).toFixed(0)} MB.`, 6000); return; }
    if (!confirm(`Replace ${f.path} (${kB(f.bytes)}) with ${file.name} (${kB(file.size)})?\n\nThe file that is there now goes away, and every stage must run again.`)) return;
    const fd = new FormData();
    fd.append('file', file);
    const res = await authFetch(`${ctx.API}/cad/${enc(f.component || f.path.split('/').slice(-2)[0])}`, { method: 'POST', body: fd });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) { toast(`Not replaced: ${typeof body.detail === 'string' ? body.detail : res.statusText}`, 8000); return; }
    problems(body);
    toast(`Replaced ${f.path}`);
    await loadFiles();
    ctx.refresh();
  }

  function problems(check, mine = true) {
    F.check = check;
    if (mine) ctx.checked(check);
    if (gutter) { gutter.dataset.stale = '1'; numbers(); }
    drawProblems();
    drawFiles();
  }

  async function loadFiles() {
    try {
      const d = await api(`${ctx.API}/files`);
      F.files = d.files;
      F.components = d.components;
      F.maxCad = d.max_cad_bytes;
    } catch (e) { put(filesBox, errorBox(e)); return; }
    drawFiles();
  }

  return {
    el, problems,
    async show(st) {
      if (!F.loaded) {
        F.loaded = true;
        if (!F.check) F.check = { ok: st.ok, problems: st.problems || [], stage: (st.problems || []).some(p => !p.file) ? 'context' : 'inputs', summary: '' };
        drawProblems();
        await loadFiles();
        const want = qs().get('file');
        const first = F.files.find(f => f.editable && f.path === want) || F.files.find(f => f.editable && problemsOf(f.path).length) || F.files.find(f => f.role === 'system' && f.editable) || F.files.find(f => f.editable);
        if (first) await open(first.path, (problemsOf(first.path)[0] || {}).line || null);
      }
      fit();
    },
    update(st) {
      if (!F.loaded) return;
      const now = { ok: st.ok, problems: st.problems || [], stage: F.check ? F.check.stage : 'inputs', summary: F.check && F.check.ok && st.ok ? F.check.summary : '' };
      if (JSON.stringify(now.problems) !== JSON.stringify((F.check || {}).problems) || now.ok !== (F.check || {}).ok) problems(now, false);
    },
  };
}

// ------------------------------------------------------------------ Numbers: the named numbers of the context
function numbersTab(ctx) {
  const el = h('div');
  const N = { params: null, error: null, text: '', basis: '', okWas: null };
  const filter = h('input', { type: 'search', placeholder: 'Filter: name, title, source, origin', style: { width: '360px' } });
  const table = h('div');
  const chips = h('div', { class: 'chips' });
  filter.addEventListener('input', () => { N.text = filter.value.trim().toLowerCase(); rows(); });

  const rows = () => {
    const all = N.params || [];
    const hit = p => (!N.basis || p.basis === N.basis) && (!N.text || [p.name, p.title, p.source, p.origin, p.expr, p.display, p.note].some(x => String(x || '').toLowerCase().includes(N.text)));
    const shown = all.filter(hit);
    const counts = {};
    for (const p of all) counts[p.basis] = (counts[p.basis] || 0) + 1;
    put(chips, [['', all.length], ...Object.entries(counts).sort()].map(([k, n]) =>
      h('button', { class: `chip${N.basis === k ? ' on' : ''}`, type: 'button', title: k ? `numbers that are ${k}` : null, onclick: () => { N.basis = k; rows(); } },
        k || 'all', h('span', { class: 'count' }, n))));
    put(table, shown.length ? h('table', { class: 'data pj-numbers' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Name'), h('th', { class: 'num' }, 'Value'), h('th', {}, 'Expression'), h('th', {}, 'Source'), h('th', {}, 'Origin'))),
      h('tbody', {}, shown.map(p => h('tr', {},
        h('td', {}, h('span', { class: 'mono strong' }, p.name), p.title ? h('div', { class: 'small secondary' }, p.title) : null),
        h('td', { class: 'num strong' }, p.display),
        h('td', { class: 'small' }, p.expr ? h('code', {}, p.expr) : h('span', { class: 'muted' }, '–'),
          p.used_by.length ? h('div', { class: 'small muted', title: p.used_by.join(', ') }, `used by ${p.used_by.length}`) : null),
        h('td', { class: 'small' }, p.source || h('span', { class: 'muted' }, '–'), p.note ? h('div', { class: 'muted' }, p.note) : null),
        h('td', { class: 'small nowrap' }, h('span', { class: 'mono' }, p.origin), h('div', { class: 'muted' }, p.basis))))))
      : h('p', { class: 'muted' }, 'No number matches the filter.'));
  };

  const load = async () => {
    put(el, loading('Evaluating the numbers…'));
    try {
      const d = await api(`${ctx.API}/params`);
      N.params = d.params;
      N.error = null;
      put(el, h('div', { class: 'toolbar' }, filter, chips, h('span', { class: 'muted small' }, `${d.params.length} numbers, ${d.effects} effects`)),
        h('div', { class: 'card flush' }, table));
      rows();
    } catch (e) {
      N.params = null;
      const list = (e.body && e.body.problems) || [];
      put(el, h('div', { class: 'card empty-state' }, h('h2', {}, 'The numbers do not evaluate'),
        h('p', {}, list.length ? `The inputs have ${plural(list.length, 'problem')}. Correct them, then the numbers show here.` : e.message),
        list.length ? h('button', { class: 'btn primary', type: 'button', onclick: () => ctx.setTab('inputs') }, 'Open the inputs') : null));
    }
  };
  return {
    el,
    show(st) { if (N.okWas !== `${st.ok}:${st.n_problems}:${(st.stale || []).length}` || N.params === null) { N.okWas = `${st.ok}:${st.n_problems}:${(st.stale || []).length}`; load(); } },
    update(st) { if (!el.hidden) this.show(st); },
  };
}

// ------------------------------------------------------------------ Load cases: the plan and its analyses
function loadCasesTab(ctx) {
  const el = h('div');
  const ANALYSIS = { done: ['good', 'done'], failed: ['critical', 'failed'], not_supported: ['warning', 'not supported'], skipped: ['info', 'skipped'] };
  const link = (id, text) => h('a', { href: 'javascript:void 0', title: 'Show it in the report', onclick: () => ctx.openInReport(id) }, text || id);
  const draw = st => {
    const cases = st.load_cases || [];
    const plan = st.stages.find(s => s.stage === 'plan') || {};
    const analyze = st.stages.find(s => s.stage === 'analyze') || {};
    if (!cases.length) {
      put(el, h('div', { class: 'card empty-state' }, h('h2', {}, 'No load cases yet'),
        plan.state === 'failed' ? h('p', {}, `The last run stopped at the plan stage: ${plan.error || 'see the log'}`)
          : h('p', {}, 'The plan stage selects the load cases: one for each phase that a requirement covers, at limit loads. Run the verification to see them.')));
      return;
    }
    const built = st.report.built;
    put(el,
      h('p', { class: 'secondary' }, plan.summary, ' ', analyze.summary, analyze.state === 'failed' ? ` The analysis stage stopped: ${analyze.error}` : ''),
      h('div', { class: 'card flush' }, h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ['Load case', 'Analysis type', 'Serves', 'Analysis', 'Results', ''].map(x => h('th', {}, x)))),
        h('tbody', {}, cases.map(c => {
          const a = c.analysis;
          const [cls, word] = a ? (ANALYSIS[a.status] || ['info', a.status]) : ['info', 'not analysed'];
          const res = a ? Object.entries(a.results).map(([k, n]) => `${n} ${words(k)}`).join(', ') : '';
          const checks = a ? a.checks : {};
          const nChecks = (checks.passed || 0) + (checks.failed || 0) + (checks.not_evaluated || 0);
          return h('tr', {},
            h('td', {}, h('div', { class: 'strong' }, c.title || c.id), h('div', { class: 'mono small' }, built ? link(c.id) : c.id),
              c.phase ? h('div', { class: 'small muted' }, `phase ${c.phase}`) : null),
            h('td', { class: 'small' }, words(c.analysis_type),
              c.scaled_from ? h('div', { class: 'muted' }, `scaled from ${c.scaled_from} × ${Number(c.scale).toFixed(3).replace(/\.?0+$/, '')}`) : null,
              c.status === 'not_supported' ? h('div', { class: 'muted' }, c.note || 'not supported') : null,
              (c.uses || []).length ? h('div', { class: 'muted' }, `uses ${c.uses.join(', ')}`) : null),
            h('td', { class: 'small' }, c.requirements.length ? c.requirements.map(r => h('div', { class: 'mono' }, built ? link(r) : r)) : h('span', { class: 'muted' }, '–'),
              (c.hand_calcs || []).length ? h('div', { class: 'muted' }, plural(c.hand_calcs.length, 'hand calculation')) : null),
            h('td', {}, statusChip(cls, word),
              a && a.message ? h('div', { class: 'small secondary pj-msg' }, a.message) : null,
              a ? h('div', { class: 'small muted' }, [a.quick ? 'quick run' : null, a.stale ? 'stale: the inputs changed' : null, a.runtime_s !== null && a.runtime_s !== undefined ? `${a.runtime_s} s` : null,
                (a.nodes || []).length ? `${a.nodes.filter(Boolean).map(n => n.toLocaleString('en-US')).join(' / ')} nodes` : null].filter(Boolean).join(' · ')) : null),
            h('td', { class: 'small' }, res || h('span', { class: 'muted' }, '–'),
              nChecks ? h('div', { class: checks.failed ? '' : 'muted' }, checks.failed ? h('span', { class: 'gate' }, dot('critical'), `${checks.failed} of ${nChecks} checks failed`) : `${checks.passed || 0} of ${nChecks} checks passed`) : null),
            h('td', { class: 'small nowrap' }, built && a ? link(a.id, 'In the report →') : null));
        })))));
  };
  return { el, show: draw, update: draw };
}

// ------------------------------------------------------------------ Reviews: decisions, assumptions and findings
function reviewsTab(ctx) {
  const el = h('div');
  const R = { records: null, reviews: {}, state: '', kind: '', key: null, comments: new Map(), loading: false };
  const who = h('input', { type: 'text', placeholder: 'Your name', style: { width: '220px' }, value: (() => { try { return localStorage.getItem(NAME_KEY) || ''; } catch { return ''; } })() });
  who.addEventListener('change', () => { try { localStorage.setItem(NAME_KEY, who.value.trim()); } catch { /* private mode */ } });
  const bar = h('div', { class: 'toolbar' });
  const listBox = h('div');
  const KIND = { findings: 'finding', decisions: 'decision', assumptions: 'assumption' };
  const stateOf = id => { const rv = R.reviews[id]; return !rv ? 'open' : rv.stale ? 'stale' : rv.state; };

  const send = async (rec, state) => {
    const by = who.value.trim();
    const comment = (R.comments.get(rec.id) || '').trim();
    if (state !== 'open' && !by) { toast('Give your name first: a review has a name.', 5000); who.focus(); return; }
    if ((state === 'rejected' || state === 'changes_requested') && !comment) { toast('Give the reason in the comment.', 5000); return; }
    try {
      R.reviews[rec.id] = await postJSON(`${ctx.API}/reviews`, { record: rec.id, state, by, comment });
      R.comments.delete(rec.id);
      draw();
      ctx.refresh();
    } catch (e) { toast(e.message, 8000); }
  };

  const text = rec => {
    const more = [];
    const line = (label, value) => { if (value) more.push(h('p', { class: 'small' }, h('span', { class: 'muted' }, `${label}: `), value)); };
    let lead;
    if (rec.kind === 'finding') { lead = rec.statement; line('Evidence', rec.rationale); line('Proposed correction', rec.suggestion); }
    else if (rec.kind === 'decision') {
      lead = rec.choice; line('Why', rec.rationale);
      for (const a of rec.alternatives || []) line('Not chosen', `${a.option} ${a.why_not}`);
    } else { lead = rec.statement; line('Why it is acceptable', rec.rationale); line('If it is wrong', rec.impact); line('How to check it', rec.verify_by); }
    return [h('p', { class: 'pj-lead' }, lead), more.length ? h('details', { class: 'pj-more' }, h('summary', {}, 'Why'), more) : null];
  };

  const row = rec => {
    const state = stateOf(rec.id);
    const rv = R.reviews[rec.id];
    const [cls, label] = REVIEW[state];
    const comment = h('input', { type: 'text', placeholder: 'Comment', value: R.comments.get(rec.id) || '', class: 'pj-comment' });
    comment.addEventListener('input', () => R.comments.set(rec.id, comment.value));
    const btn = (labelText, to, title) => h('button', { class: 'btn small', type: 'button', title, onclick: () => send(rec, to) }, labelText);
    const agent = String((rec.origin || {}).by || '').startsWith('agent:');
    return h('div', { class: 'pj-rec' },
      h('div', { class: 'pj-rec-head' },
        h('div', {},
          h('span', { class: 'badge' }, rec.kind === 'finding' ? [dot(SEVERITY[rec.severity]), ` ${rec.severity} finding`] : rec.kind),
          agent ? h('span', { class: 'badge', title: (rec.origin || {}).by }, 'proposed by an agent') : null,
          rec.kind === 'decision' && rec.needs_review ? h('span', { class: 'badge' }, 'asks for a review') : null,
          h('span', { class: 'strong pj-title' }, rec.title || rec.id)),
        h('div', { class: 'pj-state' }, statusChip(cls, label),
          rv && (rv.by || rv.date) ? h('div', { class: 'small muted' }, [rv.by, rv.date].filter(Boolean).join(', ')) : null)),
      h('div', { class: 'small muted' }, h('span', { class: 'mono' }, ctx.status().report.built ? h('a', { href: 'javascript:void 0', title: 'Show it in the report', onclick: () => ctx.openInReport(rec.id) }, rec.id) : rec.id),
        ` · ${rec.stage || (rec.origin || {}).stage || ''} · ${(rec.origin || {}).by || ''}`, rec.subject ? ` · about ${rec.subject}` : ''),
      text(rec),
      rv && rv.comment ? h('p', { class: 'small pj-said' }, `“${rv.comment}”`) : null,
      h('div', { class: 'pj-actions' }, btn('Accept', 'accepted', 'The record is right.'), btn('Reject', 'rejected', 'The record is wrong. Give the reason.'),
        btn('Request changes', 'changes_requested', 'The record must change. Say what.'),
        state !== 'open' ? h('button', { class: 'btn small ghost', type: 'button', title: 'Take the review back.', onclick: () => send(rec, 'open') }, 'Reopen') : null,
        comment));
  };

  const draw = () => {
    if (!R.records) return;
    const all = R.records;
    const counts = { '': all.length, open: 0, accepted: 0, rejected: 0, changes_requested: 0, stale: 0 };
    for (const r of all) counts[stateOf(r.id)] += 1;
    const kinds = { '': all.length };
    for (const r of all) kinds[r.kind] = (kinds[r.kind] || 0) + 1;
    const chip = (group, k, label, n) => h('button', { class: `chip${R[group] === k ? ' on' : ''}`, type: 'button', onclick: () => { R[group] = k; draw(); } }, label, h('span', { class: 'count' }, n));
    put(bar,
      h('div', { class: 'chips' }, chip('state', '', 'All', counts['']), Object.keys(REVIEW).filter(k => counts[k] || k === 'open').map(k => chip('state', k, REVIEW[k][1], counts[k]))),
      h('div', { class: 'chips' }, Object.keys(kinds).length > 2 ? Object.entries(kinds).map(([k, n]) => chip('kind', k, k ? `${k}s` : 'every kind', n)) : null),
      h('label', { class: 'gate small', style: { marginLeft: 'auto' } }, h('span', { class: 'muted' }, 'Reviewer'), who));
    const shown = all.filter(r => (!R.state || stateOf(r.id) === R.state) && (!R.kind || r.kind === R.kind));
    put(listBox, shown.length ? h('div', { class: 'card pj-recs' }, shown.map(row)) : h('p', { class: 'muted' }, 'No record matches the filter.'));
  };

  const load = async () => {
    if (R.loading) return;
    R.loading = true;
    try {
      const [recs, revs] = await Promise.all([api(`${ctx.API}/records`), api(`${ctx.API}/reviews`)]);
      const order = { error: 0, major: 1, minor: 2, info: 3 };
      R.records = Object.keys(KIND).flatMap(k => (recs.collections[k] || []).map(r => ({ ...r, kind: r.kind || KIND[k] })));
      R.records.sort((a, b) => (a.kind === 'finding' ? order[a.severity] ?? 4 : a.kind === 'decision' ? (a.needs_review ? 5 : 6) : 7) - (b.kind === 'finding' ? order[b.severity] ?? 4 : b.kind === 'decision' ? (b.needs_review ? 5 : 6) : 7));
      R.reviews = revs.reviews;
      if (!R.records.length) {
        put(el, h('div', { class: 'card empty-state' }, h('h2', {}, 'Nothing to review yet'),
          h('p', {}, 'Each stage writes its decisions, its assumptions and the findings about the inputs. Run the verification, then accept or reject them here or in the report.')));
      } else {
        put(el, bar, listBox, recs.warnings.length ? h('details', { class: 'more' }, h('summary', {}, `${plural(recs.warnings.length, 'record')} could not be read as the data model says`), h('pre', { class: 'code' }, recs.warnings.join('\n'))) : null);
        draw();
      }
    } catch (e) { put(el, errorBox(e)); }
    R.loading = false;
  };
  const keyOf = st => JSON.stringify([st.reviews, st.findings, st.stages.map(s => [s.state, s.written])]);
  return {
    el,
    show(st) { if (R.key !== keyOf(st) || !R.records) { R.key = keyOf(st); load(); } },
    update(st) { if (!el.hidden) this.show(st); },
  };
}
