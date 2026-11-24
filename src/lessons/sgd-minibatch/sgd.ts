/**
 * Mini-batch gradient descent on a tiny regression problem: predict a delivery's minutes from a
 * traffic index x ∈ [−1, 1] with a line ŷ = w·x + b and squared error. Two parameters, so the whole
 * loss landscape can be drawn, but the data set is big enough (1,000 deliveries) for batching to matter.
 */
import { createRng } from '../../lib/rng';

export type Data = { x: Float64Array; y: Float64Array };
export const TRUE_W = 12;
export const TRUE_B = 30;
export const NOISE = 5;

export function makeData(n = 1000, seed = 7): Data {
  const r = createRng(seed);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = r.uniform(-1, 1);
    y[i] = TRUE_B + TRUE_W * x[i] + r.normal(0, NOISE);
  }
  return { x, y };
}

/** Mean squared error of the line (w, b) over the whole data set. */
export function loss(d: Data, w: number, b: number) {
  let s = 0;
  for (let i = 0; i < d.x.length; i++) s += (w * d.x[i] + b - d.y[i]) ** 2;
  return s / d.x.length;
}

/** Gradient of the mean squared error over the given example indices (all of them if omitted). */
export function grad(d: Data, w: number, b: number, idx?: ArrayLike<number>): [number, number] {
  let gw = 0;
  let gb = 0;
  const n = idx ? idx.length : d.x.length;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const e = w * d.x[i] + b - d.y[i];
    gw += 2 * e * d.x[i];
    gb += 2 * e;
  }
  return [gw / n, gb / n];
}

/** The least-squares solution (the bottom of the bowl). */
export function best(d: Data): { w: number; b: number; loss: number } {
  const n = d.x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += d.x[i];
    my += d.y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (d.x[i] - mx) * (d.y[i] - my);
    sxx += (d.x[i] - mx) ** 2;
  }
  const w = sxy / sxx;
  const b = my - w * mx;
  return { w, b, loss: loss(d, w, b) };
}

/** Indices of a random batch of size B, drawn without replacement. */
export function sampleBatch(n: number, B: number, r: ReturnType<typeof createRng>): number[] {
  if (B >= n) return Array.from({ length: n }, (_, i) => i);
  const chosen = new Set<number>();
  while (chosen.size < B) chosen.add(r.int(n));
  return [...chosen];
}

/** K independent mini-batch gradients at (w, b). */
export function batchGradients(d: Data, w: number, b: number, B: number, K: number, seed: number): [number, number][] {
  const r = createRng(seed);
  return Array.from({ length: K }, () => grad(d, w, b, sampleBatch(d.x.length, B, r)));
}

/** Root-mean-square distance of the estimates from the true gradient. */
export function spread(est: [number, number][], truth: [number, number]) {
  return Math.sqrt(est.reduce((s, [a, c]) => s + (a - truth[0]) ** 2 + (c - truth[1]) ** 2, 0) / est.length);
}

export type RunPoint = { seen: number; w: number; b: number; loss: number };

/**
 * Mini-batch gradient descent with a budget measured in examples processed (gradient evaluations).
 * Each epoch shuffles the data and walks through it in batches of B (B = n is full-batch GD).
 */
export function run(d: Data, B: number, lr: number, budget: number, start: [number, number], seed: number, recordEvery = 1): { points: RunPoint[]; steps: number } {
  const n = d.x.length;
  const r = createRng(seed);
  let [w, b] = start;
  let seen = 0;
  let steps = 0;
  const points: RunPoint[] = [{ seen: 0, w, b, loss: loss(d, w, b) }];
  const order = Array.from({ length: n }, (_, i) => i);
  let pos = n; // force a shuffle before the first batch
  while (seen + Math.min(B, n) <= budget) {
    if (pos + B > n) {
      for (let i = n - 1; i > 0; i--) {
        const j = r.int(i + 1);
        [order[i], order[j]] = [order[j], order[i]];
      }
      pos = 0;
    }
    const idx = order.slice(pos, pos + B);
    pos += B;
    const [gw, gb] = grad(d, w, b, idx);
    w -= lr * gw;
    b -= lr * gb;
    seen += idx.length;
    steps++;
    if (steps % recordEvery === 0 || seen + B > budget) points.push({ seen, w, b, loss: loss(d, w, b) });
    if (!Number.isFinite(w) || Math.abs(w) > 1e6) break;
  }
  return { points, steps };
}

/** Sufficient statistics, so the loss at any (w, b) costs O(1) — used to draw the landscape. */
export function stats(d: Data) {
  const n = d.x.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sx += d.x[i];
    sy += d.y[i];
    sxx += d.x[i] * d.x[i];
    sxy += d.x[i] * d.y[i];
    syy += d.y[i] * d.y[i];
  }
  return { ex: sx / n, ey: sy / n, exx: sxx / n, exy: sxy / n, eyy: syy / n };
}
export type Stats = ReturnType<typeof stats>;
/** E[(w·x + b − y)²] expanded: identical to loss(), but constant time. */
export const lossFast = (s: Stats, w: number, b: number) => w * w * s.exx + b * b + s.eyy + 2 * w * b * s.ex - 2 * w * s.exy - 2 * b * s.ey;
