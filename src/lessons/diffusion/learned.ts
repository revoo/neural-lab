// authored-by: Claude Opus 5.5
/**
 * A learned denoiser, to compare with the exact one. A small network reads a noisy point and the
 * noise level and predicts the noise that was added (the standard DDPM objective):
 *   input  [x, y, sin/cos features of the step t]  →  64 tanh  →  64 tanh  →  ε̂ (2 numbers)
 * Hand-written backprop, checked against central finite differences in learned.test.ts.
 */
import { createRng, type Rng } from '../../lib/rng';
import { T, abarAt, bestGuess, impliedNoise, sampleData, type Denoiser, type Mixture, type Pt } from './diffusion';

export const FEATS = 6; // x, y, and 4 time features
export const timeFeatures = (t: number): number[] => {
  const u = t / T;
  return [Math.sin(Math.PI * u), Math.cos(Math.PI * u), Math.sin(2 * Math.PI * u), Math.cos(2 * Math.PI * u)];
};

/** Step whose signal share is closest to a (the network is trained on whole steps of DDPM's schedule). */
export function stepFor(a: number): number {
  let lo = 0;
  let hi = T;
  // LINEAR_ABAR decreases with t
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (abarAt(mid) > a) lo = mid;
    else hi = mid;
  }
  return Math.abs(abarAt(lo) - a) < Math.abs(abarAt(hi) - a) ? lo : hi;
}

export class NoiseNet {
  readonly H: number;
  p: Float64Array;
  g: Float64Array;
  private m: Float64Array;
  private v: Float64Array;
  private t = 0;
  readonly off: { W1: number; b1: number; W2: number; b2: number; W3: number; b3: number };

  constructor(H: number, seed: number) {
    this.H = H;
    const sizes = { W1: H * FEATS, b1: H, W2: H * H, b2: H, W3: 2 * H, b3: 2 };
    let o = 0;
    const off = {} as NoiseNet['off'];
    for (const k of Object.keys(sizes) as (keyof typeof sizes)[]) {
      off[k] = o;
      o += sizes[k];
    }
    this.off = off;
    this.p = new Float64Array(o);
    this.g = new Float64Array(o);
    this.m = new Float64Array(o);
    this.v = new Float64Array(o);
    const rng = createRng(seed);
    const init = (start: number, n: number, fin: number, fout: number) => {
      const sd = Math.sqrt(2 / (fin + fout));
      for (let i = 0; i < n; i++) this.p[start + i] = rng.normal(0, sd);
    };
    init(off.W1, H * FEATS, FEATS, H);
    init(off.W2, H * H, H, H);
    init(off.W3, 2 * H, H, 2);
  }

  get size() {
    return this.p.length;
  }

  forward(inp: number[]) {
    const { H, p, off } = this;
    const h1 = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let s = p[off.b1 + j];
      for (let i = 0; i < FEATS; i++) s += p[off.W1 + j * FEATS + i] * inp[i];
      h1[j] = Math.tanh(s);
    }
    const h2 = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let s = p[off.b2 + j];
      const row = off.W2 + j * H;
      for (let i = 0; i < H; i++) s += p[row + i] * h1[i];
      h2[j] = Math.tanh(s);
    }
    const out: Pt = [p[off.b3], p[off.b3 + 1]];
    for (let k = 0; k < 2; k++) for (let j = 0; j < H; j++) out[k] += p[off.W3 + k * H + j] * h2[j];
    return { h1, h2, out };
  }

  predict(x: Pt, t: number): Pt {
    return this.forward([x[0], x[1], ...timeFeatures(t)]).out;
  }

  /** Squared error between predicted and true noise for one example; adds scale × gradient into g. */
  lossOne(x: Pt, t: number, eps: Pt, accumulate = false, scale = 1) {
    const { H, p, g, off } = this;
    const inp = [x[0], x[1], ...timeFeatures(t)];
    const f = this.forward(inp);
    const d0 = f.out[0] - eps[0];
    const d1 = f.out[1] - eps[1];
    const loss = d0 * d0 + d1 * d1;
    if (!accumulate) return loss;
    const dout = [2 * d0 * scale, 2 * d1 * scale];
    const dh2 = new Float64Array(H);
    for (let k = 0; k < 2; k++) {
      g[off.b3 + k] += dout[k];
      for (let j = 0; j < H; j++) {
        g[off.W3 + k * H + j] += dout[k] * f.h2[j];
        dh2[j] += dout[k] * p[off.W3 + k * H + j];
      }
    }
    const dh1 = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      const da = dh2[j] * (1 - f.h2[j] * f.h2[j]);
      g[off.b2 + j] += da;
      const row = off.W2 + j * H;
      for (let i = 0; i < H; i++) {
        g[row + i] += da * f.h1[i];
        dh1[i] += da * p[row + i];
      }
    }
    for (let j = 0; j < H; j++) {
      const da = dh1[j] * (1 - f.h1[j] * f.h1[j]);
      g[off.b1 + j] += da;
      for (let i = 0; i < FEATS; i++) g[off.W1 + j * FEATS + i] += da * inp[i];
    }
    return loss;
  }

  adam(lr: number) {
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    const { p, g, m, v } = this;
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
    }
  }

  /** As a denoiser: predict the noise, then turn it into a guess of the clean point. */
  denoiser(): Denoiser {
    return (x, a) => {
      const t = stepFor(a);
      const at = abarAt(t);
      const e = this.predict(x, t);
      const ra = Math.sqrt(Math.max(at, 1e-12));
      return [(x[0] - Math.sqrt(1 - at) * e[0]) / ra, (x[1] - Math.sqrt(1 - at) * e[1]) / ra];
    };
  }
}

export type LearnConfig = { H: number; steps: number; batch: number; lr: number; seed: number };
export const LEARN: LearnConfig = { H: 64, steps: 4000, batch: 64, lr: 0.002, seed: 1 };

/** One training example: a clean point, a random step, fresh noise, and the noisy point. */
function example(mix: Mixture, rng: Rng) {
  const x0 = sampleData(mix, 1, rng)[0];
  const t = 1 + rng.int(T);
  const a = abarAt(t);
  const eps: Pt = [rng.normal(), rng.normal()];
  const x: Pt = [Math.sqrt(a) * x0[0] + Math.sqrt(1 - a) * eps[0], Math.sqrt(a) * x0[1] + Math.sqrt(1 - a) * eps[1]];
  return { x, t, eps };
}

/** A resumable training run (so a browser can animate it). */
export function noiseTrainer(mix: Mixture, cfg: LearnConfig = LEARN) {
  const net = new NoiseNet(cfg.H, cfg.seed);
  const rng = createRng(cfg.seed * 104729 + 7);
  const history: { step: number; loss: number }[] = [];
  let done = 0;
  let acc = 0;
  return {
    net,
    history,
    get done() {
      return done;
    },
    advance(n: number) {
      const end = Math.min(cfg.steps, done + n);
      while (done < end) {
        net.g.fill(0);
        let l = 0;
        for (let b = 0; b < cfg.batch; b++) {
          const e = example(mix, rng);
          l += net.lossOne(e.x, e.t, e.eps, true, 1 / cfg.batch) / cfg.batch;
        }
        net.adam(cfg.lr);
        done++;
        acc += l;
        if (done % 100 === 0) {
          history.push({ step: done, loss: acc / 100 });
          acc = 0;
        }
      }
      return done >= cfg.steps;
    },
  };
}

/**
 * The best loss any network could reach: even the perfect denoiser can't know the exact noise, only
 * its average given the noisy point. Estimated on a fixed set of examples, along with a network's
 * loss on the same examples.
 */
export function lossFloor(mix: Mixture, net: NoiseNet | null, n = 4000, seed = 31) {
  const rng = createRng(seed);
  let exact = 0;
  let learned = 0;
  for (let i = 0; i < n; i++) {
    const e = example(mix, rng);
    const a = abarAt(e.t);
    const x0 = bestGuess(mix, a, e.x).x0;
    const best = impliedNoise(e.x, x0, a);
    exact += ((best[0] - e.eps[0]) ** 2 + (best[1] - e.eps[1]) ** 2) / n;
    if (net) learned += net.lossOne(e.x, e.t, e.eps) / n;
  }
  return { exact, learned };
}
