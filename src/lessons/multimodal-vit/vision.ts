// authored-by: Claude Opus 5.5
/**
 * Real, tiny vision models with hand-written backprop (finite-difference checked in `vit.test.ts`).
 *
 * The task: 8×8 grey images, each holding one 3×3 sprite (a plus or an X) at a random position, plus
 * a little pixel noise. Class = which sprite. Position is irrelevant, so a model that already knows
 * "the same pattern means the same thing wherever it appears" has a head start.
 *
 * - CNN: K 3×3 filters slid over the image (with ReLU), averaged over all positions, then one linear
 *   output. Translation is built in.
 * - ViT: the image cut into P×P patches; each patch flattened and linearly embedded into d numbers,
 *   optionally plus a learned position embedding; one single-head self-attention layer with a
 *   residual connection; mean over tokens; one linear output. Nothing tells it that a pattern in one
 *   patch relates to the same pattern in another: it has to learn that from data.
 */
import { createRng, type Rng } from '../../lib/rng';

export const IMG = 8;
export type Sample = { x: Float64Array; y: number };

const SPRITES = [
  [0, 1, 0, 1, 1, 1, 0, 1, 0], // plus
  [1, 0, 1, 0, 1, 0, 1, 0, 1], // X
];

export function drawSprite(cls: number, ox: number, oy: number, rng?: Rng, noise = 0) {
  const x = new Float64Array(IMG * IMG);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) x[(oy + i) * IMG + ox + j] = SPRITES[cls][i * 3 + j];
  if (rng && noise) for (let k = 0; k < x.length; k++) if (rng.next() < noise) x[k] = 1 - x[k];
  return x;
}

export function makeSet(n: number, seed: number, noise = 0.04): Sample[] {
  const r = createRng(seed);
  return Array.from({ length: n }, (_, i) => {
    const y = i % 2;
    return { x: drawSprite(y, r.int(IMG - 2), r.int(IMG - 2), r, noise), y };
  });
}

export const TEST_SET = makeSet(400, 999);

/** Shift an image by (dx, dy) with zero fill: the augmentation. */
export function shift(x: Float64Array, dx: number, dy: number) {
  const out = new Float64Array(IMG * IMG);
  for (let i = 0; i < IMG; i++)
    for (let j = 0; j < IMG; j++) {
      const si = i - dy;
      const sj = j - dx;
      if (si >= 0 && si < IMG && sj >= 0 && sj < IMG) out[i * IMG + j] = x[si * IMG + sj];
    }
  return out;
}

const sig = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export interface Model {
  params: Float64Array;
  /** Logit for "X" and the gradient of the log-loss w.r.t. every parameter (accumulated into g). */
  lossGrad(x: Float64Array, y: number, g?: Float64Array): number;
  logit(x: Float64Array): number;
}

// ------------------------------------------------------------------ CNN

export class Cnn implements Model {
  params: Float64Array;
  readonly K: number;
  constructor(K: number, seed: number) {
    this.K = K;
    const r = createRng(seed * 131 + 1);
    // per filter: 9 weights + bias; then K output weights + bias
    this.params = new Float64Array(K * 10 + K + 1);
    for (let k = 0; k < K; k++) for (let i = 0; i < 9; i++) this.params[k * 10 + i] = r.normal(0, Math.sqrt(2 / 9));
    for (let k = 0; k < K; k++) this.params[K * 10 + k] = r.normal(0, Math.sqrt(1 / K));
  }
  private forward(x: Float64Array) {
    const { K, params: p } = this;
    const S = IMG - 2;
    const pooled = new Float64Array(K);
    const pre = new Float64Array(K * S * S);
    for (let k = 0; k < K; k++)
      for (let i = 0; i < S; i++)
        for (let j = 0; j < S; j++) {
          let z = p[k * 10 + 9];
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) z += p[k * 10 + a * 3 + b] * x[(i + a) * IMG + j + b];
          pre[(k * S + i) * S + j] = z;
          if (z > 0) pooled[k] += z / (S * S);
        }
    let out = p[K * 11];
    for (let k = 0; k < K; k++) out += p[K * 10 + k] * pooled[k];
    return { pre, pooled, out };
  }
  logit(x: Float64Array) {
    return this.forward(x).out;
  }
  lossGrad(x: Float64Array, y: number, g?: Float64Array) {
    const { K, params: p } = this;
    const S = IMG - 2;
    const f = this.forward(x);
    const q = sig(f.out);
    const loss = -(y ? Math.log(Math.max(q, 1e-12)) : Math.log(Math.max(1 - q, 1e-12)));
    if (g) {
      const d = q - y;
      g[K * 11] += d;
      for (let k = 0; k < K; k++) {
        g[K * 10 + k] += d * f.pooled[k];
        const dp = (d * p[K * 10 + k]) / (S * S);
        for (let i = 0; i < S; i++)
          for (let j = 0; j < S; j++) {
            if (f.pre[(k * S + i) * S + j] <= 0) continue;
            g[k * 10 + 9] += dp;
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) g[k * 10 + a * 3 + b] += dp * x[(i + a) * IMG + j + b];
          }
      }
    }
    return loss;
  }
}

// ------------------------------------------------------------------ ViT

/** Cut an image into P×P patches, row by row; each patch flattened row by row. */
export function patchify(x: Float64Array, P: number) {
  const n = IMG / P;
  const out: number[][] = [];
  for (let pi = 0; pi < n; pi++)
    for (let pj = 0; pj < n; pj++) {
      const v: number[] = [];
      for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) v.push(x[(pi * P + a) * IMG + pj * P + b]);
      out.push(v);
    }
  return out;
}

export class Vit implements Model {
  params: Float64Array;
  readonly N: number;
  readonly D: number;
  private o: Record<string, number>;
  readonly P: number;
  readonly d: number;
  readonly pos: boolean;
  constructor(P: number, d: number, pos: boolean, seed: number) {
    this.P = P;
    this.d = d;
    this.pos = pos;
    this.N = (IMG / P) ** 2;
    this.D = P * P;
    const { N, D } = this;
    const r = createRng(seed * 977 + 5);
    // embedding E (d×D) and bias (d); position embedding (N×d); Wq, Wk, Wv (d×d each); output w (d) and bias
    const sizes = { E: d * D, eb: d, pe: N * d, q: d * d, k: d * d, v: d * d, w: d, wb: 1 };
    this.o = {};
    let off = 0;
    for (const [k, s] of Object.entries(sizes)) {
      this.o[k] = off;
      off += s;
    }
    this.params = new Float64Array(off);
    const p = this.params;
    for (let i = 0; i < d * D; i++) p[this.o.E + i] = r.normal(0, Math.sqrt(1 / D));
    for (let i = 0; i < N * d; i++) p[this.o.pe + i] = pos ? r.normal(0, 0.5) : 0;
    for (const m of ['q', 'k', 'v']) for (let i = 0; i < d * d; i++) p[this.o[m] + i] = r.normal(0, Math.sqrt(1 / d));
    for (let i = 0; i < d; i++) p[this.o.w + i] = r.normal(0, Math.sqrt(1 / d));
  }
  /** Patch embeddings (with positions if on) for an image. */
  tokens(x: Float64Array, order?: number[]) {
    const { N, D, d, params: p, o } = this;
    const pt = patchify(x, this.P);
    const h = Array.from({ length: N }, () => new Float64Array(d));
    for (let t = 0; t < N; t++) {
      const src = order ? pt[order[t]] : pt[t];
      for (let i = 0; i < d; i++) {
        let s = p[o.eb + i] + (this.pos ? p[o.pe + t * d + i] : 0);
        for (let j = 0; j < D; j++) s += p[o.E + i * D + j] * src[j];
        h[t][i] = s;
      }
    }
    return { h, pt };
  }
  private forward(x: Float64Array) {
    const { N, d, params: p, o } = this;
    const { h, pt } = this.tokens(x);
    const mat = (m: string, v: Float64Array) => {
      const out = new Float64Array(d);
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += p[o[m] + i * d + j] * v[j];
        out[i] = s;
      }
      return out;
    };
    const Q = h.map((v) => mat('q', v));
    const K = h.map((v) => mat('k', v));
    const V = h.map((v) => mat('v', v));
    const sc = 1 / Math.sqrt(d);
    const A = Q.map((q) => {
      const s = K.map((k) => q.reduce((a, qi, i) => a + qi * k[i], 0) * sc);
      const m = Math.max(...s);
      const e = s.map((v) => Math.exp(v - m));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
    const out = h.map((ht, t) => {
      const u = new Float64Array(d);
      for (let i = 0; i < d; i++) {
        let s = ht[i];
        for (let s2 = 0; s2 < N; s2++) s += A[t][s2] * V[s2][i];
        u[i] = s;
      }
      return u;
    });
    const pooled = new Float64Array(d);
    out.forEach((u) => u.forEach((v, i) => (pooled[i] += v / N)));
    let logit = p[o.wb];
    for (let i = 0; i < d; i++) logit += p[o.w + i] * pooled[i];
    return { pt, h, Q, K, V, A, pooled, logit };
  }
  logit(x: Float64Array) {
    return this.forward(x).logit;
  }
  /** Mean-pooled output vector (used to show what positions change). */
  pooledFrom(x: Float64Array, order?: number[]) {
    const { N, d, params: p, o } = this;
    const { h } = this.tokens(x, order);
    const mat = (m: string, v: Float64Array) => Array.from({ length: d }, (_, i) => v.reduce((a, vj, j) => a + p[o[m] + i * d + j] * vj, 0));
    const Q = h.map((v) => mat('q', v));
    const K = h.map((v) => mat('k', v));
    const V = h.map((v) => mat('v', v));
    const pooled = new Array(d).fill(0);
    for (let t = 0; t < N; t++) {
      const s = K.map((k) => Q[t].reduce((a, qi, i) => a + qi * k[i], 0) / Math.sqrt(d));
      const m = Math.max(...s);
      const e = s.map((v) => Math.exp(v - m));
      const z = e.reduce((a, b) => a + b, 0);
      for (let i = 0; i < d; i++) pooled[i] += (h[t][i] + e.reduce((a, ev, s2) => a + (ev / z) * V[s2][i], 0)) / N;
    }
    return pooled;
  }
  lossGrad(x: Float64Array, y: number, g?: Float64Array) {
    const { N, D, d, params: p, o } = this;
    const f = this.forward(x);
    const q = sig(f.logit);
    const loss = -(y ? Math.log(Math.max(q, 1e-12)) : Math.log(Math.max(1 - q, 1e-12)));
    if (!g) return loss;
    const dl = q - y;
    g[o.wb] += dl;
    const dPool = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      g[o.w + i] += dl * f.pooled[i];
      dPool[i] = dl * p[o.w + i];
    }
    // out_t = h_t + Σ_s A_ts V_s ; d out_t = dPool / N
    const dOut = dPool.map((v) => v / N);
    const dH = Array.from({ length: N }, () => Float64Array.from(dOut));
    const dV = Array.from({ length: N }, () => new Float64Array(d));
    const dQ = Array.from({ length: N }, () => new Float64Array(d));
    const dK = Array.from({ length: N }, () => new Float64Array(d));
    const sc = 1 / Math.sqrt(d);
    for (let t = 0; t < N; t++) {
      // dA_ts = dOut · V_s
      const dA = f.V.map((v) => v.reduce((a, vi, i) => a + vi * dOut[i], 0));
      for (let s = 0; s < N; s++) for (let i = 0; i < d; i++) dV[s][i] += f.A[t][s] * dOut[i];
      const dot = f.A[t].reduce((a, as, s) => a + as * dA[s], 0);
      for (let s = 0; s < N; s++) {
        const dS = f.A[t][s] * (dA[s] - dot) * sc;
        for (let i = 0; i < d; i++) {
          dQ[t][i] += dS * f.K[s][i];
          dK[s][i] += dS * f.Q[t][i];
        }
      }
    }
    // Q = Wq h etc.
    const back = (m: string, src: Float64Array[], dY: Float64Array[]) => {
      for (let t = 0; t < N; t++)
        for (let i = 0; i < d; i++) {
          if (!dY[t][i]) continue;
          for (let j = 0; j < d; j++) {
            g[o[m] + i * d + j] += dY[t][i] * src[t][j];
            dH[t][j] += dY[t][i] * p[o[m] + i * d + j];
          }
        }
    };
    back('q', f.h, dQ);
    back('k', f.h, dK);
    back('v', f.h, dV);
    for (let t = 0; t < N; t++)
      for (let i = 0; i < d; i++) {
        g[o.eb + i] += dH[t][i];
        if (this.pos) g[o.pe + t * d + i] += dH[t][i];
        for (let j = 0; j < D; j++) g[o.E + i * D + j] += dH[t][i] * f.pt[t][j];
      }
    return loss;
  }
}

// ------------------------------------------------------------------ training

export type TrainOpts = { epochs: number; lr: number; augment: boolean; batch: number };
export const TRAIN_OPTS: TrainOpts = { epochs: 60, lr: 0.02, augment: false, batch: 8 };

/** Adam on mini-batches; with augment, every example is randomly shifted by up to 2 pixels each time it is seen (zero fill). */
export function trainModel(m: Model, data: Sample[], seed: number, opts: TrainOpts = TRAIN_OPTS, onEpoch?: (e: number) => void) {
  const r = createRng(seed * 7 + 3);
  const P = m.params.length;
  const g = new Float64Array(P);
  const mm = new Float64Array(P);
  const vv = new Float64Array(P);
  let t = 0;
  const idx = data.map((_, i) => i);
  for (let e = 0; e < opts.epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = r.int(i + 1);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let b = 0; b < idx.length; b += opts.batch) {
      g.fill(0);
      const rows = idx.slice(b, b + opts.batch);
      for (const i of rows) {
        let x = data[i].x;
        if (opts.augment) x = shift(x, r.int(5) - 2, r.int(5) - 2);
        m.lossGrad(x, data[i].y, g);
      }
      t++;
      const c1 = 1 - 0.9 ** t;
      const c2 = 1 - 0.999 ** t;
      for (let j = 0; j < P; j++) {
        const gj = g[j] / rows.length;
        mm[j] = 0.9 * mm[j] + 0.1 * gj;
        vv[j] = 0.999 * vv[j] + 0.001 * gj * gj;
        m.params[j] -= (opts.lr * (mm[j] / c1)) / (Math.sqrt(vv[j] / c2) + 1e-8);
      }
    }
    onEpoch?.(e);
  }
}

export function accuracy(m: Model, data: Sample[] = TEST_SET) {
  let c = 0;
  for (const s of data) if ((m.logit(s.x) > 0 ? 1 : 0) === s.y) c++;
  return c / data.length;
}
