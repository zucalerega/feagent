// Nonlinear: placeholder page (filled in by the dashboard consolidation).
import { h } from '../api.js';

export async function render(el) {
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Nonlinear'),
    h('p', { class: 'lede' }, 'Elastoplastic CalculiX ramps and crush-to-fracture runs, with force-displacement curves and videos.'))));
}
