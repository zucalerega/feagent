// Connect: how teammates' scripts and agent harnesses reach this server (URL, token, copy-paste examples).
import { api, h, clear, loading, errorBox, getToken, setToken, toast } from '../api.js';

export async function render(el) {
  el.append(loading());
  let auth;
  try { auth = await api('/api/auth'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  const url = location.origin;
  const tok = getToken();
  const needs = auth.required;
  const H = needs ? '-H "Authorization: Bearer $TOKEN" ' : '';
  const PH = needs ? 'H = {"Authorization": f"Bearer {TOKEN}"}\n' : 'H = {}\n';

  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Connect a script or an agent'),
    h('p', { class: 'lede' }, 'Everything the UI does goes through the HTTP API below, so a harness, a notebook or a teammate’s laptop can solve cases and submit runs with nothing installed but curl or Python.'))));

  // ---- URL + token
  const masked = h('code', { class: 'mono', style: { cursor: 'pointer', fontSize: '15px' }, title: 'click to show / hide' }, tok ? '•'.repeat(12) : '(none stored)');
  let shown = false;
  masked.addEventListener('click', () => { if (!tok) return; shown = !shown; masked.textContent = shown ? tok : '•'.repeat(12); });
  const tokInput = h('input', { type: 'text', placeholder: 'paste a token', style: { width: '260px' } });
  el.append(h('div', { class: 'grid-2' },
    h('div', { class: 'card stack' },
      h('h2', {}, 'This server'),
      h('table', { class: 'kv' }, h('tbody', {},
        h('tr', {}, h('th', {}, 'API base URL'), h('td', {}, h('code', { class: 'mono', style: { fontSize: '15px' } }, url), ' ', copyBtn(url))),
        h('tr', {}, h('th', {}, 'authentication'), h('td', {}, needs ? 'shared token required on every /api/* call' : 'none (open server)')),
        needs ? h('tr', {}, h('th', {}, 'token in this browser'), h('td', {}, masked, ' ', tok ? copyBtn(tok, 'copy token') : null,
          tok ? h('span', { class: `status ${auth.valid ? 'good' : 'critical'}`, style: { marginLeft: '8px' } }, h('span', { class: 'ico' }, auth.valid ? '✓' : '✕'), auth.valid ? 'valid' : 'rejected') : null)) : null)),
      needs ? h('div', { class: 'row', style: { alignItems: 'center' } }, tokInput,
        h('button', { class: 'btn small', onclick: () => { setToken(tokInput.value.trim()); location.reload(); } }, 'Use this token'),
        tok ? h('button', { class: 'btn small', onclick: () => { setToken(''); location.reload(); } }, 'Forget token') : null) : null,
      h('p', { class: 'small secondary', style: { margin: 0 } }, needs
        ? 'Send the token as "Authorization: Bearer <token>", an "X-Feacore-Token" header or "?token=". Share the UI as <URL>/?token=<token>: the browser stores it and removes it from the address bar.'
        : 'Start the server with --token (or FEACORE_UI_TOKEN) before sharing it on a public tunnel.')),
    h('div', { class: 'card stack' },
      h('h2', {}, 'Set up a shell'),
      code(`export URL=${url}\nexport TOKEN=<the shared token>   # ${needs ? 'required' : 'not needed on this server'}`),
      h('p', { class: 'small secondary', style: { margin: 0 } }, 'Brev tunnels that ask for a browser login cannot be used by scripts. Make the tunnel public and protect it with the token, or use `brev port-forward <instance> --port 8000:8000` and URL=http://localhost:8000.'))));

  // ---- examples
  el.append(h('div', { class: 'card stack', style: { marginTop: '20px' } },
    h('h2', {}, 'Solve a case (fixed solver)'),
    code(`curl ${H}\\\n     -F case=@case.json -F mesh=@mesh.msh \\\n     -F agent_id=agent-07 -F problem_id=nafems_le1 \\\n     $URL/api/solve`),
    h('div', { class: 'grid-2' },
      h('div', {}, h('div', { class: 'section-title' }, 'Python · httpx'),
        code(`import os, httpx\nURL, TOKEN = os.environ["URL"], os.environ.get("TOKEN", "")\n${PH}files = {"case": open("case.json", "rb"), "mesh": open("mesh.msh", "rb")}\nr = httpx.post(f"{URL}/api/solve", headers=H, files=files,\n               data={"agent_id": "agent-07", "problem_id": "nafems_le1"},\n               timeout=None)\nrun = r.json()   # run_id, status, qois, info, error`)),
      h('div', {}, h('div', { class: 'section-title' }, 'Python · requests'),
        code(`import os, requests\nURL, TOKEN = os.environ["URL"], os.environ.get("TOKEN", "")\n${PH}files = {"case": open("case.json", "rb"), "mesh": open("mesh.msh", "rb")}\nr = requests.post(f"{URL}/api/solve", headers=H, files=files,\n                  data={"agent_id": "agent-07", "problem_id": "nafems_le1"})\nrun = r.json()`))),
    h('p', { class: 'small secondary', style: { margin: 0 } }, '200: solved (run record) · 422: invalid case or the solver rejected the setup (the record carries the error) · add -F wait=false to get 202 at once and poll GET /api/runs/<run_id>.')));

  el.append(h('div', { class: 'card stack', style: { marginTop: '20px' } },
    h('h2', {}, 'Submit an agent’s answer (runs on 3+ meshes)'),
    h('p', { class: 'secondary', style: { margin: 0 } }, 'A submission groups one agent’s runs on its own mesh sequence; the verification layer then checks convergence (Richardson / GCI) and the consensus page compares agents.'),
    h('div', { class: 'grid-2' },
      h('div', {}, h('div', { class: 'section-title' }, 'Runs solved here (/api/solve returned run_ids)'),
        code(`curl ${H}-H "Content-Type: application/json" \\\n     -X POST $URL/api/submissions -d '{\n  "problem_id": "nafems_le1", "agent_id": "agent-07",\n  "run_ids": ["run-...", "run-...", "run-..."]}'`)),
      h('div', {}, h('div', { class: 'section-title' }, 'Runs solved elsewhere'),
        code(`# register each run, then submit its run_ids as on the left ...\ncurl ${H}-H "Content-Type: application/json" \\\n     -X POST $URL/api/runs -d '{\n  "problem_id": "nafems_le1", "agent_id": "agent-07",\n  "backend": "skfem", "qois": {"syy_D": 9.26e7},\n  "info": {"h_rep": 0.05, "order": 2, "n_dofs": 31244}}'\n# ... or submit the levels directly:\n#   POST $URL/api/submissions  {"problem_id": ..., "agent_id": ...,\n#     "levels": [{"h": 0.1, "qois": {"syy_D": 9.25e7}, "order": 2}, ...]}`))),
    h('p', { class: 'small secondary', style: { margin: 0 } }, 'Results: GET /api/consensus/<problem_id> (all agents), GET /api/submissions/<id>. Full reference: docs/ui.md and the interactive API docs at ', h('a', { href: '/docs', target: '_blank' }, '/docs'), '.')));
}

function code(text) {
  return h('div', { style: { position: 'relative' } }, h('pre', { class: 'code' }, text),
    h('button', { class: 'btn small', style: { position: 'absolute', top: '8px', right: '8px', background: '#f0efec' }, onclick: () => copy(text) }, 'copy'));
}

function copyBtn(text, label = 'copy') {
  return h('button', { class: 'btn small', onclick: () => copy(text) }, label);
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied'); }
  catch { toast('Copy failed: select the text instead'); }
}
