// Pictures and videos of the Nonlinear and Test data pages: URLs that carry the access token, a thumbnail grid
// and a lightbox. Files come from /api/nonlinear/file?ref=... (feacore/ui/media.py checks every path).
import { h, withToken } from './api.js';

export const fileUrl = (ref, download = false) =>
  withToken(`/api/nonlinear/file?ref=${encodeURIComponent(ref)}${download ? '&download=true' : ''}`);

const CSS = `
.nl-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.nl-tile { display: flex; flex-direction: column; gap: 6px; background: var(--surface); border: 1px solid var(--grid);
  border-radius: 10px; padding: 8px; cursor: pointer; text-align: left; font: inherit; color: inherit; }
.nl-tile:hover { border-color: var(--accent); }
.nl-thumb { position: relative; width: 100%; aspect-ratio: 16 / 10; background: #eceae4; border-radius: 6px; overflow: hidden;
  display: flex; align-items: center; justify-content: center; }
.nl-thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
.nl-play { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 44px; height: 44px; border-radius: 50%;
  background: rgba(11, 11, 11, 0.62); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.nl-tile .t { font-size: 14px; font-weight: 600; line-height: 1.3; }
.nl-tile .c { font-size: 13px; color: var(--ink-2); line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.nl-box { position: fixed; inset: 0; background: rgba(11, 11, 11, 0.72); z-index: 80; display: flex; align-items: center;
  justify-content: center; padding: 24px; }
.nl-box-inner { background: var(--surface); border-radius: 12px; max-width: min(1280px, 96vw); max-height: 94vh; overflow: auto;
  padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }
.nl-box-inner img, .nl-box-inner video { max-width: 100%; max-height: 74vh; display: block; margin: 0 auto; background: #111; }
.nl-box-inner img { background: #fff; }
.nl-cap { font-size: 14.5px; color: var(--ink-2); max-width: 100ch; }
.nl-facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }
.nl-facts .tile .value { font-size: 22px; }
.nl-tools { display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }
.nl-pre { background: var(--surface-2); border-radius: 8px; padding: 10px 12px; font-family: var(--mono); font-size: 13px;
  white-space: pre-wrap; word-break: break-all; margin: 0; }
.nl-controls { display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: flex-end; margin-bottom: 10px; }
.nl-controls label { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--ink-2); font-weight: 600; }
.nl-controls input[type=number] { width: 120px; }
.nl-form select, .nl-form input[type=text], .nl-form input[type=number] { width: 100%; }
.nl-badge-rtx { background: #2b2a35; color: #fff; }
`;

export function ensureStyles() {
  if (document.getElementById('nl-styles')) return;
  const st = document.createElement('style');
  st.id = 'nl-styles';
  st.textContent = CSS;
  document.head.append(st);
}

/** A full-screen view of one picture or video with its caption. */
export function lightbox(item) {
  ensureStyles();
  const close = () => { box.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = e => { if (e.key === 'Escape') close(); };
  const media = item.kind === 'video'
    ? h('video', { src: fileUrl(item.ref), controls: true, autoplay: true, loop: true, muted: true, playsinline: true,
      poster: item.poster ? fileUrl(item.poster) : null })
    : h('img', { src: fileUrl(item.ref), alt: item.title || '' });
  const box = h('div', { class: 'nl-box', onclick: e => { if (e.target === box) close(); } },
    h('div', { class: 'nl-box-inner' },
      h('div', { class: 'row', style: { justifyContent: 'space-between', alignItems: 'center' } },
        h('b', {}, item.title || item.name || ''),
        h('span', { class: 'row', style: { gap: '8px' } },
          item.rtx ? h('span', { class: 'badge nl-badge-rtx', title: 'rendered with Omniverse RTX on the DGX Spark; not reproducible on this server' }, 'Omniverse RTX') : null,
          h('a', { class: 'btn small', href: fileUrl(item.ref, true) }, 'Download'),
          h('button', { class: 'btn small', type: 'button', onclick: close }, 'Close'))),
      media,
      item.caption ? h('div', { class: 'nl-cap' }, item.caption) : null,
      h('code', { class: 'muted small' }, item.ref)));
  document.body.append(box);
  document.addEventListener('keydown', onKey);
}

/** Thumbnails of pictures and videos; a click opens the lightbox. items: [{ref, kind, title, caption, poster}] */
export function mediaGrid(items, { empty = 'Nothing here yet.' } = {}) {
  ensureStyles();
  if (!items || !items.length) return h('p', { class: 'muted' }, empty);
  return h('div', { class: 'nl-grid' }, items.map(it => {
    const thumbRef = it.kind === 'video' ? it.poster : it.ref;
    return h('button', { class: 'nl-tile', type: 'button', title: it.caption || it.title || it.name || '', onclick: () => lightbox(it) },
      h('div', { class: 'nl-thumb' },
        thumbRef ? h('img', { src: fileUrl(thumbRef), alt: it.title || '', loading: 'lazy' }) : h('span', { class: 'muted small' }, it.kind === 'pdf' ? 'PDF' : 'video'),
        it.kind === 'video' ? h('span', { class: 'nl-play' }, '▶') : null),
      h('div', { class: 't' }, it.title || it.name, it.rtx ? h('span', { class: 'badge nl-badge-rtx', style: { marginLeft: '6px' } }, 'RTX') : null),
      it.caption ? h('div', { class: 'c' }, it.caption) : null);
  }));
}

/** Links to files (CSV, JSON, logs, meshes) with their sizes. */
export function fileList(files) {
  if (!files || !files.length) return h('p', { class: 'muted' }, 'No files.');
  const size = b => b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b > 1e3 ? `${(b / 1e3).toFixed(0)} kB` : `${b} B`;
  return h('div', { class: 'chips' }, files.map(f => h('a', { class: 'chip', href: fileUrl(f.ref, true), title: f.ref },
    f.name, h('span', { class: 'count' }, ` ${size(f.bytes || 0)}`))));
}
