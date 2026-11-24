import { useMemo } from 'react';
import { usePlot } from './plot';

type RGB = [number, number, number];

const lerp = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function ramp(stops: RGB[], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return lerp(stops[i], stops[i + 1], x - i);
}

/** Low = deep blue-violet, high = warm yellow. Used for loss landscapes (perceptually ordered). */
export const sequential = (t: number) =>
  ramp(
    [
      [38, 30, 110],
      [52, 88, 170],
      [40, 150, 170],
      [110, 196, 120],
      [235, 220, 100],
    ],
    t,
  );

/** For classifiers: t in [0,1], 0 → class B (orange), 0.5 → neutral, 1 → class A (blue). */
export const diverging = (t: number) =>
  ramp(
    [
      [232, 104, 47],
      [246, 196, 168],
      [238, 238, 238],
      [170, 196, 246],
      [47, 111, 228],
    ],
    t,
  );

/**
 * Renders f(x, y) over the plot area as an image, mapped through a colour map.
 * Optionally draws contour lines where the value crosses evenly spaced levels.
 */
export function Heatmap({
  f,
  range,
  colormap = sequential,
  resolution = 240,
  opacity = 0.85,
  contours = 0,
  transform,
}: {
  f: (x: number, y: number) => number;
  /** [min, max] of f used for colouring; values outside are clamped. */
  range: [number, number];
  colormap?: (t: number) => RGB;
  resolution?: number;
  opacity?: number;
  /** Number of contour levels (0 = none). */
  contours?: number;
  /** Optional transform applied to the normalised value before colouring, e.g. Math.sqrt to show low regions in more detail. */
  transform?: (t: number) => number;
}) {
  const { x, y } = usePlot();
  const [x0, x1] = x.domain;
  const [y0, y1] = y.domain;
  const url = useMemo(() => {
    const n = resolution;
    const canvas = document.createElement('canvas');
    canvas.width = n;
    canvas.height = n;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    const img = ctx.createImageData(n, n);
    const vals = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
      const yv = y1 - ((j + 0.5) / n) * (y1 - y0);
      for (let i = 0; i < n; i++) {
        const xv = x0 + ((i + 0.5) / n) * (x1 - x0);
        vals[j * n + i] = f(xv, yv);
      }
    }
    const norm = (v: number) => {
      let t = (v - range[0]) / (range[1] - range[0]);
      t = Math.min(1, Math.max(0, t));
      return transform ? transform(t) : t;
    };
    const band = (v: number) => Math.floor(norm(v) * contours);
    for (let k = 0; k < n * n; k++) {
      const t = norm(vals[k]);
      let [r, g, b] = colormap(t);
      if (contours) {
        const i = k % n;
        const j = Math.floor(k / n);
        const bk = band(vals[k]);
        const edge = (i < n - 1 && band(vals[k + 1]) !== bk) || (j < n - 1 && band(vals[k + n]) !== bk);
        if (edge) {
          r *= 0.72;
          g *= 0.72;
          b *= 0.72;
        }
      }
      img.data[4 * k] = r;
      img.data[4 * k + 1] = g;
      img.data[4 * k + 2] = b;
      img.data[4 * k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [f, range[0], range[1], colormap, resolution, contours, x0, x1, y0, y1, transform]);
  if (!url) return null;
  return <image href={url} x={x(x0)} y={y(y1)} width={x(x1) - x(x0)} height={y(y0) - y(y1)} preserveAspectRatio="none" opacity={opacity} />;
}

/**
 * Render f over a rectangle to a PNG data URL (no SVG needed) — used for small "what this neuron
 * responds to" thumbnails. Rows run from top (y1) to bottom (y0).
 */
export function fieldImage(
  f: (x: number, y: number) => number,
  [x0, x1, y0, y1]: [number, number, number, number],
  res: number,
  colour: (v: number) => RGB,
): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = res;
  canvas.height = res;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(res, res);
  for (let j = 0; j < res; j++) {
    const yv = y1 - ((j + 0.5) / res) * (y1 - y0);
    for (let i = 0; i < res; i++) {
      const xv = x0 + ((i + 0.5) / res) * (x1 - x0);
      const [r, g, b] = colour(f(xv, yv));
      const k = 4 * (j * res + i);
      img.data[k] = r;
      img.data[k + 1] = g;
      img.data[k + 2] = b;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}
