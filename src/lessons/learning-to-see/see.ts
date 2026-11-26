// authored-by: Claude Opus 5.5
/**
 * Measurements for "learning to see" (V1, B1): every "moment" in the lesson is one of these
 * statistics crossing a threshold that is itself measured (on random filters, or on raw pixels).
 */
import { createRng } from '../../lib/rng';
import { irwinHall } from '../../lib/compute/detmath';
import { Cnn, type CnnConfig, type Trace } from './cnn';
import type { ImageSet } from './fashion';

export const CONFIG: CnnConfig = { size: 28, c1: 8, k1: 5, c2: 16, k2: 5, hidden: 32, classes: 10 };

// ------------------------------------------------------------------ oriented-edge templates

/** Odd (edge) and even (bar) Gabor templates at 8 orientations, zero-mean and unit-norm, K × K. */
export function gaborBank(K: number): { name: string; angle: number; phase: 'edge' | 'bar'; t: Float64Array }[] {
  const out: { name: string; angle: number; phase: 'edge' | 'bar'; t: Float64Array }[] = [];
  const c = (K - 1) / 2;
  const sigma = K / 4;
  const freq = 1 / K;
  for (let a = 0; a < 8; a++) {
    const th = (a * Math.PI) / 8;
    for (const phase of ['edge', 'bar'] as const) {
      const t = new Float64Array(K * K);
      for (let y = 0; y < K; y++)
        for (let x = 0; x < K; x++) {
          const u = (x - c) * Math.cos(th) + (y - c) * Math.sin(th);
          const env = Math.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma * sigma));
          t[y * K + x] = env * (phase === 'edge' ? Math.sin(2 * Math.PI * freq * u) : Math.cos(2 * Math.PI * freq * u));
        }
      normalise(t);
      out.push({ name: `${phase} at ${(a * 22.5).toFixed(1)}°`, angle: a * 22.5, phase, t });
    }
  }
  return out;
}

function normalise(t: Float64Array) {
  let m = 0;
  for (const v of t) m += v;
  m /= t.length;
  let n = 0;
  for (let i = 0; i < t.length; i++) n += (t[i] -= m) ** 2;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < t.length; i++) t[i] /= n;
}

/** Best |cosine| between a (mean-removed) filter and any template, and which template. */
export function templateMatch(filter: ArrayLike<number>, bank: ReturnType<typeof gaborBank>) {
  const f = Float64Array.from(filter);
  normalise(f);
  let best = 0;
  let which = 0;
  bank.forEach((b, i) => {
    let s = 0;
    for (let j = 0; j < f.length; j++) s += f[j] * b.t[j];
    if (Math.abs(s) > best) [best, which] = [Math.abs(s), i];
  });
  return { score: best, template: bank[which] };
}

/** The p-quantile of the best-template score of random (normal) filters: the "chance" bar. */
export function randomFilterBar(K: number, n: number, p: number, seed = 7): number {
  const bank = gaborBank(K);
  const r = createRng(seed);
  const s = Array.from({ length: n }, () => templateMatch(Array.from({ length: K * K }, () => irwinHall(r.next)), bank).score).sort((a, b) => a - b);
  return s[Math.floor(p * (n - 1))];
}

// ------------------------------------------------------------------ what it looks at

/**
 * Occlusion map: cover a patch × patch square with mid-grey (0.5) at every stride step and record
 * how much the probability of class `cls` drops. Exact: one forward pass per position.
 */
export function occlusion(net: Cnn, img: Float32Array, cls: number, patch = 6, stride = 2) {
  const S = net.cfg.size;
  const tr = net.newTrace();
  const base = net.forward(img, tr)[cls];
  const n = Math.floor((S - patch) / stride) + 1;
  const map = new Float64Array(n * n);
  const work = Float32Array.from(img);
  for (let gy = 0; gy < n; gy++)
    for (let gx = 0; gx < n; gx++) {
      work.set(img);
      for (let y = gy * stride; y < gy * stride + patch; y++) for (let x = gx * stride; x < gx * stride + patch; x++) work[y * S + x] = 0.5;
      map[gy * n + gx] = base - net.forward(work, tr)[cls];
    }
  return { map, n, base };
}

/** Saliency: ∂(score of class cls)/∂pixel, by backpropagation to the input. */
export function saliency(net: Cnn, img: Float32Array, cls: number): Float64Array {
  const tr = net.newTrace();
  net.forward(img, tr);
  const dl = new Float64Array(net.cfg.classes);
  dl[cls] = 1;
  const dx = new Float64Array(net.cfg.size * net.cfg.size);
  const keep = Float64Array.from(net.g);
  net.backward(tr, dl, dx);
  net.g.set(keep);
  return dx;
}

// ------------------------------------------------------------------ the internal space

export function penultimate(net: Cnn, imgs: Float32Array[], tr: Trace = net.newTrace()): Float64Array[] {
  return imgs.map((im) => {
    net.forward(im, tr);
    return Float64Array.from(tr.h);
  });
}

/** Accuracy of classifying `test` by the nearest class mean (Euclidean) computed on `train`. */
export function nearestCentroidAccuracy(train: ArrayLike<number>[], trainY: ArrayLike<number>, test: ArrayLike<number>[], testY: ArrayLike<number>, classes = 10): number {
  const D = train[0].length;
  const mean = Array.from({ length: classes }, () => new Float64Array(D));
  const cnt = new Array(classes).fill(0);
  train.forEach((v, i) => {
    cnt[trainY[i]]++;
    for (let j = 0; j < D; j++) mean[trainY[i]][j] += v[j];
  });
  mean.forEach((m, k) => m.forEach((_, j) => (m[j] /= Math.max(1, cnt[k]))));
  let ok = 0;
  test.forEach((v, i) => {
    let best = 0;
    let bd = Infinity;
    for (let k = 0; k < classes; k++) {
      let d = 0;
      for (let j = 0; j < D; j++) d += (v[j] - mean[k][j]) ** 2;
      if (d < bd) [bd, best] = [d, k];
    }
    if (best === testY[i]) ok++;
  });
  return ok / test.length;
}

/** Top-2 principal components of row vectors (power iteration with deflation, deterministic start). */
export function pca2(rows: ArrayLike<number>[]): { proj: [number, number][]; mean: Float64Array; axes: Float64Array[] } {
  const n = rows.length;
  const D = rows[0].length;
  const mean = new Float64Array(D);
  rows.forEach((r) => {
    for (let j = 0; j < D; j++) mean[j] += r[j] / n;
  });
  const X = rows.map((r) => Float64Array.from({ length: D }, (_, j) => r[j] - mean[j]));
  const axes: Float64Array[] = [];
  for (let c = 0; c < 2; c++) {
    let v = Float64Array.from({ length: D }, (_, j) => 1 + j / D);
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(D);
      for (const x of X) {
        let s = 0;
        for (let j = 0; j < D; j++) s += x[j] * v[j];
        for (let j = 0; j < D; j++) w[j] += s * x[j];
      }
      for (const a of axes) {
        let s = 0;
        for (let j = 0; j < D; j++) s += w[j] * a[j];
        for (let j = 0; j < D; j++) w[j] -= s * a[j];
      }
      const norm = Math.hypot(...w) || 1;
      v = w.map((x) => x / norm);
    }
    // fix the sign so the picture doesn't flip between checkpoints for no reason
    let big = 0;
    for (let j = 1; j < D; j++) if (Math.abs(v[j]) > Math.abs(v[big])) big = j;
    if (v[big] < 0) v = v.map((x) => -x);
    axes.push(v);
  }
  const proj = X.map((x) => axes.map((a) => x.reduce((s, xj, j) => s + xj * a[j], 0)) as [number, number]);
  return { proj, mean, axes };
}

// ------------------------------------------------------------------ training

export type SeeSpec = { cfg: CnnConfig; batch: number; steps: number; lr: number; seed: number; dataSeed: number };

export function seeCheckpoints(steps: number): number[] {
  const out = new Set<number>([0]);
  for (let k = 0; ; k++) {
    const v = Math.round(10 ** (k / 8));
    if (v >= steps) break;
    out.add(v);
  }
  out.add(steps);
  return [...out].sort((a, b) => a - b);
}

/** Train with Adam on seeded mini-batches; calls back at each checkpoint with the parameters. */
export function trainSee(spec: SeeSpec, data: ImageSet, onCheckpoint: (step: number, params: Float32Array, net: Cnn) => boolean | void) {
  const net = new Cnn(spec.cfg);
  net.init(spec.seed);
  const r = createRng(spec.dataSeed);
  const tr = net.newTrace();
  const marks = new Set(seeCheckpoints(spec.steps));
  if (onCheckpoint(0, Float32Array.from(net.p), net) === false) return net;
  for (let s = 1; s <= spec.steps; s++) {
    const idx = Array.from({ length: spec.batch }, () => r.int(data.train.length));
    net.batchGrad(
      idx.map((i) => data.train[i]),
      idx.map((i) => data.trainY[i]),
      tr,
    );
    net.adam(spec.lr);
    if (marks.has(s) && onCheckpoint(s, Float32Array.from(net.p), net) === false) break;
  }
  return net;
}

export function accuracy(net: Cnn, imgs: Float32Array[], ys: ArrayLike<number>): number {
  const tr = net.newTrace();
  let ok = 0;
  imgs.forEach((im, i) => {
    const p = net.forward(im, tr);
    let best = 0;
    for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
    if (best === ys[i]) ok++;
  });
  return ok / imgs.length;
}
