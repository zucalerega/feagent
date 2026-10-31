// Colour maps for field plots. Sequential fields (von Mises, |u|, T) use a perceptually uniform
// map; signed fields (stress / displacement components) a two-hue diverging map with a neutral
// grey midpoint. The categorical order is used for named groups (never cycled; extras are grey).

export const COLORMAPS = {
  viridis: {
    label: 'Viridis (sequential)',
    stops: ['#440154', '#48186a', '#472d7b', '#424086', '#3b528b', '#33638d', '#2c728e', '#26828e', '#21918c',
            '#1fa088', '#28ae80', '#3fbc73', '#5ec962', '#84d44b', '#addc30', '#d8e219', '#fde725'],
  },
  blues: {
    label: 'Blue (sequential)',
    stops: ['#e7f0fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
  },
  diverging: {
    label: 'Blue – red (diverging)',
    stops: ['#0d366b', '#2a78d6', '#86b6ef', '#f0efec', '#f2a99f', '#e34948', '#8f2221'],
  },
};

export const NAN_RGB = [185, 184, 178];
export const GROUP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const OTHER_GROUP = '#9a9890';

export function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** n x [r, g, b] (0-255), linear interpolation between stops; ``bands`` > 0 gives discrete bands. */
export function lut(name, n = 256, bands = 0) {
  const stops = (COLORMAPS[name] || COLORMAPS.viridis).stops.map(hexToRgb);
  const at = t => {
    const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
  };
  const out = [];
  for (let j = 0; j < n; j++) {
    let t = j / (n - 1);
    if (bands > 1) t = Math.min(bands - 1, Math.floor(t * bands)) / (bands - 1);
    out.push(at(t));
  }
  return out;
}

export function cssGradient(name, bands = 0, vertical = true) {
  const n = bands > 1 ? bands : 24;
  const cols = lut(name, bands > 1 ? bands : n, 0);
  const dir = vertical ? 'to top' : 'to right';
  if (bands > 1) {
    const parts = cols.map((c, i) => `rgb(${c}) ${(100 * i / n).toFixed(2)}% ${(100 * (i + 1) / n).toFixed(2)}%`);
    return `linear-gradient(${dir}, ${parts.join(', ')})`;
  }
  return `linear-gradient(${dir}, ${cols.map((c, i) => `rgb(${c}) ${(100 * i / (n - 1)).toFixed(1)}%`).join(', ')})`;
}

/** Diverging for signed fields that straddle zero, sequential otherwise. */
export function defaultMap(field, range) {
  const signed = /^(ux|uy|uz|sxx|syy|szz|sxy|syz|sxz|stt|s1|s2|s3|qx|qy|qz)$/.test(field);
  if (signed && range && range[0] < 0 && range[1] > 0) return 'diverging';
  return 'viridis';
}
