// authored-by: Claude Opus 5.5
/**
 * A small convolutional network for "learning to see" (V1, B1), written out by hand:
 *
 *   image (S × S, grey) → conv1: C1 filters K1 × K1 (valid), ReLU → 2 × 2 max-pool
 *     → conv2: C2 filters K2 × K2 × C1 (valid), ReLU → 2 × 2 max-pool
 *     → flatten → dense to H units, ReLU (the "penultimate" layer we map) → dense to 10 scores
 *   loss: softmax cross-entropy; optimiser: Adam.
 *
 * Every backward pass is checked against central finite differences (cnn.test.ts). Transcendentals
 * go through `detmath.ts`, so a float32 run is bit-identical in Node and in any browser.
 */
import { createRng } from '../../lib/rng';
import { dexp, irwinHall } from '../../lib/compute/detmath';

export type CnnConfig = {
  size: number; c1: number; k1: number; c2: number; k2: number; hidden: number; classes: number;
  /** Standard deviation of the first layer's starting weights (default: He, √(2 / K1²)). */
  firstStd?: number;
};

type Arr = Float32Array | Float64Array;

export function cnnShapes(c: CnnConfig) {
  const s1 = c.size - c.k1 + 1; // conv1 output side
  const p1 = s1 >> 1; // after pooling
  const s2 = p1 - c.k2 + 1;
  const p2 = s2 >> 1;
  const flat = c.c2 * p2 * p2;
  return { s1, p1, s2, p2, flat };
}

export function cnnLayout(c: CnnConfig) {
  const { flat } = cnnShapes(c);
  let o = 0;
  const take = (n: number) => ((o += n), o - n);
  const w1 = take(c.c1 * c.k1 * c.k1);
  const b1 = take(c.c1);
  const w2 = take(c.c2 * c.c1 * c.k2 * c.k2);
  const b2 = take(c.c2);
  const w3 = take(c.hidden * flat);
  const b3 = take(c.hidden);
  const w4 = take(c.classes * c.hidden);
  const b4 = take(c.classes);
  return { w1, b1, w2, b2, w3, b3, w4, b4, total: o };
}

/** What the forward pass keeps for one image (and for the widgets: feature maps, the penultimate vector). */
export type Trace = {
  x: Arr; // S×S
  z1: Arr; a1: Arr; // C1 × s1 × s1 (pre- and post-ReLU)
  m1: Arr; arg1: Int32Array; // C1 × p1 × p1 pooled, and the argmax index into a1
  z2: Arr; a2: Arr; // C2 × s2 × s2
  m2: Arr; arg2: Int32Array; // C2 × p2 × p2
  z3: Arr; h: Arr; // hidden (penultimate)
  logits: Arr; probs: Float64Array;
};

export class Cnn {
  readonly cfg: CnnConfig;
  readonly lay: ReturnType<typeof cnnLayout>;
  readonly sh: ReturnType<typeof cnnShapes>;
  readonly precision: 'f32' | 'f64';
  p: Arr;
  g: Arr;
  m: Arr;
  v: Arr;
  t = 0;
  private b1t = 1;
  private b2t = 1;

  constructor(cfg: CnnConfig, precision: 'f32' | 'f64' = 'f32') {
    this.cfg = cfg;
    this.precision = precision;
    this.lay = cnnLayout(cfg);
    this.sh = cnnShapes(cfg);
    const n = this.lay.total;
    this.p = this.arr(n);
    this.g = this.arr(n);
    this.m = this.arr(n);
    this.v = this.arr(n);
  }
  arr(n: number): Arr {
    return this.precision === 'f32' ? new Float32Array(n) : new Float64Array(n);
  }

  /** He initialisation (Irwin–Hall normal approximation), zero biases; resets Adam. */
  init(seed: number) {
    const r = createRng(seed);
    const { p, lay, cfg } = this;
    p.fill(0);
    const fill = (o: number, n: number, fanIn: number) => {
      const std = Math.sqrt(2 / fanIn);
      for (let i = 0; i < n; i++) p[o + i] = std * irwinHall(r.next);
    };
    if (cfg.firstStd !== undefined) for (let i = 0; i < cfg.c1 * cfg.k1 * cfg.k1; i++) p[lay.w1 + i] = cfg.firstStd * irwinHall(r.next);
    else fill(lay.w1, cfg.c1 * cfg.k1 * cfg.k1, cfg.k1 * cfg.k1);
    fill(lay.w2, cfg.c2 * cfg.c1 * cfg.k2 * cfg.k2, cfg.c1 * cfg.k2 * cfg.k2);
    fill(lay.w3, cfg.hidden * this.sh.flat, this.sh.flat);
    fill(lay.w4, cfg.classes * cfg.hidden, cfg.hidden);
    this.m.fill(0);
    this.v.fill(0);
    this.t = 0;
    this.b1t = 1;
    this.b2t = 1;
  }

  newTrace(): Trace {
    const { cfg, sh } = this;
    const A = (n: number) => this.arr(n);
    return {
      x: A(cfg.size * cfg.size),
      z1: A(cfg.c1 * sh.s1 * sh.s1), a1: A(cfg.c1 * sh.s1 * sh.s1),
      m1: A(cfg.c1 * sh.p1 * sh.p1), arg1: new Int32Array(cfg.c1 * sh.p1 * sh.p1),
      z2: A(cfg.c2 * sh.s2 * sh.s2), a2: A(cfg.c2 * sh.s2 * sh.s2),
      m2: A(cfg.c2 * sh.p2 * sh.p2), arg2: new Int32Array(cfg.c2 * sh.p2 * sh.p2),
      z3: A(cfg.hidden), h: A(cfg.hidden),
      logits: A(cfg.classes), probs: new Float64Array(cfg.classes),
    };
  }

  /** Forward one image (values in [0, 1], row-major S×S) into `tr`. Returns the class probabilities. */
  forward(img: ArrayLike<number>, tr: Trace): Float64Array {
    const { p, lay, cfg, sh } = this;
    const S = cfg.size;
    for (let i = 0; i < S * S; i++) tr.x[i] = img[i];
    // conv1 (single input channel)
    const { s1, p1, s2, p2 } = sh;
    const K1 = cfg.k1;
    for (let f = 0; f < cfg.c1; f++)
      for (let y = 0; y < s1; y++)
        for (let x = 0; x < s1; x++) {
          let s = p[lay.b1 + f];
          const wo = lay.w1 + f * K1 * K1;
          for (let ky = 0; ky < K1; ky++) {
            const ro = (y + ky) * S + x;
            for (let kx = 0; kx < K1; kx++) s += p[wo + ky * K1 + kx] * tr.x[ro + kx];
          }
          const o = f * s1 * s1 + y * s1 + x;
          tr.z1[o] = s;
          tr.a1[o] = s > 0 ? s : 0;
        }
    pool(tr.a1, cfg.c1, s1, p1, tr.m1, tr.arg1);
    // conv2
    const K2 = cfg.k2;
    for (let f = 0; f < cfg.c2; f++)
      for (let y = 0; y < s2; y++)
        for (let x = 0; x < s2; x++) {
          let s = p[lay.b2 + f];
          for (let c = 0; c < cfg.c1; c++) {
            const wo = lay.w2 + (f * cfg.c1 + c) * K2 * K2;
            const io = c * p1 * p1;
            for (let ky = 0; ky < K2; ky++) {
              const ro = io + (y + ky) * p1 + x;
              for (let kx = 0; kx < K2; kx++) s += p[wo + ky * K2 + kx] * tr.m1[ro + kx];
            }
          }
          const o = f * s2 * s2 + y * s2 + x;
          tr.z2[o] = s;
          tr.a2[o] = s > 0 ? s : 0;
        }
    pool(tr.a2, cfg.c2, s2, p2, tr.m2, tr.arg2);
    // dense layers
    const flat = sh.flat;
    for (let j = 0; j < cfg.hidden; j++) {
      let s = p[lay.b3 + j];
      const wo = lay.w3 + j * flat;
      for (let i = 0; i < flat; i++) s += p[wo + i] * tr.m2[i];
      tr.z3[j] = s;
      tr.h[j] = s > 0 ? s : 0;
    }
    let mx = -Infinity;
    for (let k = 0; k < cfg.classes; k++) {
      let s = p[lay.b4 + k];
      const wo = lay.w4 + k * cfg.hidden;
      for (let j = 0; j < cfg.hidden; j++) s += p[wo + j] * tr.h[j];
      tr.logits[k] = s;
      if (tr.logits[k] > mx) mx = tr.logits[k];
    }
    let sum = 0;
    for (let k = 0; k < cfg.classes; k++) sum += tr.probs[k] = dexp(tr.logits[k] - mx);
    for (let k = 0; k < cfg.classes; k++) tr.probs[k] /= sum;
    return tr.probs;
  }

  /**
   * Backward for one image after `forward`, from dL/dlogits = `dlogits`; ADDS parameter gradients
   * into `g`. If `dx` is given, writes ∂L/∂pixel into it (for saliency maps).
   */
  backward(tr: Trace, dlogits: ArrayLike<number>, dx?: Arr) {
    const { p, g, lay, cfg, sh } = this;
    const { s1, p1, s2, flat } = sh;
    const S = cfg.size;
    const dh = new Float64Array(cfg.hidden);
    for (let k = 0; k < cfg.classes; k++) {
      const d = dlogits[k];
      g[lay.b4 + k] += d;
      const wo = lay.w4 + k * cfg.hidden;
      for (let j = 0; j < cfg.hidden; j++) {
        g[wo + j] += d * tr.h[j];
        dh[j] += d * p[wo + j];
      }
    }
    const dm2 = new Float64Array(flat);
    for (let j = 0; j < cfg.hidden; j++) {
      if (tr.z3[j] <= 0) continue;
      const d = dh[j];
      g[lay.b3 + j] += d;
      const wo = lay.w3 + j * flat;
      for (let i = 0; i < flat; i++) {
        g[wo + i] += d * tr.m2[i];
        dm2[i] += d * p[wo + i];
      }
    }
    // unpool 2 → relu → conv2
    const dz2 = new Float64Array(cfg.c2 * s2 * s2);
    for (let i = 0; i < dm2.length; i++) {
      const a = tr.arg2[i];
      if (tr.z2[a] > 0) dz2[a] += dm2[i];
    }
    const dm1 = new Float64Array(cfg.c1 * p1 * p1);
    const K2 = cfg.k2;
    for (let f = 0; f < cfg.c2; f++)
      for (let y = 0; y < s2; y++)
        for (let x = 0; x < s2; x++) {
          const d = dz2[f * s2 * s2 + y * s2 + x];
          if (d === 0) continue;
          g[lay.b2 + f] += d;
          for (let c = 0; c < cfg.c1; c++) {
            const wo = lay.w2 + (f * cfg.c1 + c) * K2 * K2;
            const io = c * p1 * p1;
            for (let ky = 0; ky < K2; ky++) {
              const ro = io + (y + ky) * p1 + x;
              for (let kx = 0; kx < K2; kx++) {
                g[wo + ky * K2 + kx] += d * tr.m1[ro + kx];
                dm1[ro + kx] += d * p[wo + ky * K2 + kx];
              }
            }
          }
        }
    const dz1 = new Float64Array(cfg.c1 * s1 * s1);
    for (let i = 0; i < dm1.length; i++) {
      const a = tr.arg1[i];
      if (tr.z1[a] > 0) dz1[a] += dm1[i];
    }
    const K1 = cfg.k1;
    if (dx) dx.fill(0);
    for (let f = 0; f < cfg.c1; f++)
      for (let y = 0; y < s1; y++)
        for (let x = 0; x < s1; x++) {
          const d = dz1[f * s1 * s1 + y * s1 + x];
          if (d === 0) continue;
          g[lay.b1 + f] += d;
          const wo = lay.w1 + f * K1 * K1;
          for (let ky = 0; ky < K1; ky++) {
            const ro = (y + ky) * S + x;
            for (let kx = 0; kx < K1; kx++) {
              g[wo + ky * K1 + kx] += d * tr.x[ro + kx];
              if (dx) dx[ro + kx] += d * p[wo + ky * K1 + kx];
            }
          }
        }
  }

  /** Mean cross-entropy over a batch of images; fills `g` with its gradient. */
  batchGrad(images: ArrayLike<number>[], labels: ArrayLike<number>, tr: Trace): number {
    this.g.fill(0);
    let loss = 0;
    const n = images.length;
    const dl = new Float64Array(this.cfg.classes);
    for (let i = 0; i < n; i++) {
      const pr = this.forward(images[i], tr);
      const y = labels[i];
      loss -= Math.log(Math.max(pr[y], 1e-30));
      for (let k = 0; k < this.cfg.classes; k++) dl[k] = (pr[k] - (k === y ? 1 : 0)) / n;
      this.backward(tr, dl);
    }
    return loss / n;
  }

  /** One Adam step (PyTorch form, no weight decay); bias corrections kept by multiplication. */
  adam(lr: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.t++;
    this.b1t *= b1;
    this.b2t *= b2;
    const step = lr / (1 - this.b1t);
    const c2 = Math.sqrt(1 - this.b2t);
    const { p, g, m, v } = this;
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (step * m[i]) / (Math.sqrt(v[i]) / c2 + eps);
    }
  }

  /** First-layer filter f as a K1 × K1 array. */
  filter(f: number): Float64Array {
    const K = this.cfg.k1;
    return Float64Array.from(this.p.subarray(this.lay.w1 + f * K * K, this.lay.w1 + (f + 1) * K * K));
  }
}

/** 2 × 2 max-pool (stride 2) of C maps of side s into side q = ⌊s/2⌋, recording argmax indices. */
function pool(a: Arr, C: number, s: number, q: number, out: Arr, arg: Int32Array) {
  for (let c = 0; c < C; c++)
    for (let y = 0; y < q; y++)
      for (let x = 0; x < q; x++) {
        let best = -Infinity;
        let bi = 0;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const i = c * s * s + (2 * y + dy) * s + 2 * x + dx;
            if (a[i] > best) {
              best = a[i];
              bi = i;
            }
          }
        const o = c * q * q + y * q + x;
        out[o] = best;
        arg[o] = bi;
      }
}
