// Shared task-lane widgets: a form for any registered tool, a live task table and a live log.
// Server side: feacore/ui/tasks.py (GET /api/tools, POST /api/tasks, GET /api/tasks/{id}/log, ...).
import { api, postJSON, authFetch, h, clear, errorBox, timeAgo, statusChip, toast } from './api.js';

let toolsCache = null;
export async function tools(force = false) {
  if (!toolsCache || force) toolsCache = api('/api/tools').catch(e => { toolsCache = null; throw e; });
  return toolsCache;
}
export async function tool(name) {
  const t = (await tools()).tools.find(x => x.name === name);
  if (!t) throw new Error(`the server has no tool "${name}"`);
  return t;
}

const STATE_CHIP = { queued: 'pending', running: 'running', done: 'succeeded', failed: 'failed', cancelled: 'cancelled', interrupted: 'failed' };
export function taskChip(state) { return statusChip(STATE_CHIP[state] || state, state); }

/** A form for tool `name`. `defaults` pre-fills values; `hide` lists params not shown (sent from defaults).
 *  `onStarted(task)` runs after a successful POST /api/tasks. Returns the form element. */
export function toolForm(name, { defaults = {}, hide = [], submitLabel = 'Start', onStarted = null, problems = null } = {}) {
  const box = h('div', { class: 'stack' }, h('span', { class: 'spinner' }));
  (async () => {
    let t;
    try { t = await tool(name); } catch (e) { clear(box).append(errorBox(e)); return; }
    const inputs = {};
    const field = p => {
      const v = defaults[p.name] ?? p.default;
      let input;
      if (p.type === 'bool') input = h('input', { type: 'checkbox', checked: v ? true : null });
      else if (p.type === 'choice') input = h('select', {}, (p.choices || []).map(c => h('option', { value: c, selected: String(c) === String(v) ? true : null }, c)));
      else if (p.type === 'problem' && problems) input = h('select', {}, h('option', { value: '' }, '(choose a problem)'), problems.map(id => h('option', { value: id, selected: id === v ? true : null }, id)));
      else if (p.type === 'text') input = h('textarea', { rows: 4, style: { width: '100%' } }, v ?? '');
      else input = h('input', { type: p.type === 'int' || p.type === 'float' ? 'number' : 'text', value: v ?? '', step: p.type === 'float' ? 'any' : null, min: p.min ?? null, max: p.max ?? null, placeholder: p.type === 'problems' ? 'comma-separated problem ids' : (p.name === t.out_param ? '(automatic)' : '') });
      inputs[p.name] = { p, input };
      return h('label', { class: 'field', title: p.help || null },
        h('span', { class: 'section-title' }, p.label, p.required ? ' *' : ''), input,
        p.help ? h('span', { class: 'muted small' }, p.help) : null);
    };
    const shown = t.params.filter(p => !hide.includes(p.name));
    const basic = shown.filter(p => !p.advanced), adv = shown.filter(p => p.advanced);
    const err = h('div');
    const btn = h('button', { class: 'btn primary', type: 'button' }, submitLabel);
    const envNote = t.missing_env && t.missing_env.length
      ? h('p', { class: 'muted small' }, `The server has no ${t.missing_env.join(', ')}${t.unless_param ? `: only "${t.unless_param}" runs work` : ''}. Set it in the server's environment for real runs.`) : null;
    btn.onclick = async () => {
      clear(err);
      const args = {};
      for (const [k, v] of Object.entries(defaults)) if (hide.includes(k)) args[k] = v;
      for (const [k, { p, input }] of Object.entries(inputs)) {
        const val = p.type === 'bool' ? input.checked : input.value.trim();
        if (val !== '' && val !== null) args[k] = val;
      }
      btn.disabled = true;
      try {
        const task = await postJSON('/api/tasks', { tool: name, args });
        toast(`Started ${t.title}`);
        if (onStarted) onStarted(task);
      } catch (e) {
        const errs = e.body && e.body.detail && e.body.detail.errors;
        err.append(errorBox(errs ? errs.join('\n') : e));
      }
      btn.disabled = false;
    };
    clear(box).append(
      t.description ? h('p', { class: 'muted' }, t.description) : null,
      h('div', { class: 'row wrap', style: { gap: '18px' } }, basic.map(field)),
      adv.length ? h('details', {}, h('summary', {}, 'More options'), h('div', { class: 'row wrap', style: { gap: '18px' } }, adv.map(field))) : null,
      envNote,
      h('div', { class: 'row', style: { alignItems: 'center', gap: '12px' } }, btn, h('code', { class: 'muted small' }, t.command)),
      err);
  })();
  return box;
}

/** Live table of tasks (filter: {group, tool}). `link(task)` -> optional href per row. Returns {el, stop}. */
export function taskTable(filter = {}, { link = null, empty = 'No tasks yet.', limit = 50 } = {}) {
  const el = h('div');
  let alive = true, timer = null;
  const q = new URLSearchParams({ limit });
  if (filter.group) q.set('group', filter.group);
  if (filter.tool) q.set('tool', filter.tool);
  const tick = async () => {
    let d;
    try { d = await api(`/api/tasks?${q}`); } catch (e) { clear(el).append(errorBox(e)); return; }
    if (!alive) return;
    clear(el).append(d.tasks.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['Task', 'State', 'Started', 'Output', ''].map(x => h('th', {}, x)))),
      h('tbody', {}, d.tasks.map(t => h('tr', {},
        h('td', {}, h('a', { href: `#/tasks/${t.id}` }, t.title), h('div', { class: 'muted small' }, t.id)),
        h('td', {}, taskChip(t.state), t.error ? h('div', { class: 'muted small' }, t.error) : null),
        h('td', { class: 'small' }, timeAgo(t.started_at || t.created_at)),
        h('td', { class: 'small' }, link && link(t) ? h('a', { href: link(t) }, 'open') : (t.out ? h('code', {}, t.out) : '')),
        h('td', {}, cancelButtons(t, tick)))))) : h('p', { class: 'muted' }, empty));
    const busy = d.tasks.some(t => t.state === 'queued' || t.state === 'running');
    timer = setTimeout(tick, busy ? 3000 : 15000);
  };
  tick();
  return { el, stop: () => { alive = false; clearTimeout(timer); } };
}

export function cancelButtons(t, after) {
  if (t.state !== 'queued' && t.state !== 'running') return null;
  const go = async mode => {
    try { await postJSON(`/api/tasks/${t.id}/cancel?mode=${mode}`, {}); toast(mode === 'stop' ? 'Asked to stop after the current step' : 'Stopping'); }
    catch (e) { toast(String(e)); }
    if (after) after();
  };
  return h('span', { class: 'row', style: { gap: '6px' } },
    t.can_stop !== false && t.out ? h('button', { class: 'btn small', type: 'button', title: 'graceful: the tool finishes its current step (ACE: the generation)', onclick: () => go('stop') }, 'Stop') : null,
    h('button', { class: 'btn small', type: 'button', title: 'terminate the process now', onclick: () => go('kill') }, 'Kill'));
}

/** Live log of task `id` (polls while it runs). Returns {el, stop}. */
export function taskLog(id, { height = '420px' } = {}) {
  const pre = h('pre', { class: 'log', style: { maxHeight: height, overflow: 'auto', whiteSpace: 'pre-wrap', fontSize: '13px' } }, '');
  let alive = true, timer = null;
  const tick = async () => {
    try {
      const [t, res] = await Promise.all([api(`/api/tasks/${id}`), authFetch(`/api/tasks/${id}/log`)]);
      if (!alive) return;
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
      pre.textContent = await res.text();
      if (atBottom) pre.scrollTop = pre.scrollHeight;
      if (t.state === 'queued' || t.state === 'running') timer = setTimeout(tick, 2000);
    } catch (e) { pre.textContent = String(e); }
  };
  tick();
  return { el: pre, stop: () => { alive = false; clearTimeout(timer); } };
}
