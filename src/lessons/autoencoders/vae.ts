// authored-by: Claude Opus 5.5
/**
 * Autoencoders and VAEs, small enough to train in a browser tab.
 *
 * Data: 12×12 cartoon faces drawn from exactly two knobs, smile (−1 frown … +1 grin) and eye
 * openness (0 shut … 1 wide). So the 144 pixels secretly have two degrees of freedom, and we know
 * the truth for every image — which lets us *measure* whether a generated face is a real face.
 *
 * Network (hand-written backprop, checked against finite differences in vae.test.ts):
 *   encoder  144 → H (tanh) → 2 means μ and 2 log-variances (linear)
 *   decoder    2 → H (tanh) → 144 pixels (sigmoid)
 * Plain autoencoder: z = μ, loss = reconstruction only.
 * VAE: z = μ + σ·ε with ε ~ N(0, 1), loss = reconstruction + β · KL(N(μ, σ²) ‖ N(0, 1)).
 * Reconstruction loss is binary cross-entropy summed over the 144 pixels.
 */
import { createRng, type Rng } from '../../lib/rng';

export const SIDE = 12;
export const PIX = SIDE * SIDE;

// ------------------------------------------------------------------ the faces

export type Knobs = { smile: number; eyes: number };

/** Points along the strokes of a face: the mouth (a parabola) and two eyes (ellipses). */
function strokes(smile: number, eyes: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let k = 0; k <= 40; k++) {
    const u = -1 + (2 * k) / 40;
    pts.push([6 + 3.2 * u, 8.1 - 1.6 * smile * (u * u - 0.5)]);
  }
  const ry = 0.15 + 1.2 * eyes;
  for (const cx of [3.5, 8.5]) {
    for (let k = 0; k < 40; k++) {
      const t = (2 * Math.PI * k) / 40;
      pts.push([cx + Math.cos(t), 4.2 + ry * Math.sin(t)]);
    }
  }
  return pts;
}

/** Render a face: each pixel's brightness is a soft-edged distance to the nearest stroke (0 … 1). */
export function face(smile: number, eyes: number): Float64Array {
  const img = new Float64Array(PIX);
  const pts = strokes(smile, eyes);
  for (let r = 0; r < SIDE; r++)
    for (let c = 0; c < SIDE; c++) {
      const x = c + 0.5;
      const y = r + 0.5;
      let m = 0;
      for (const [px, py] of pts) {
        const d2 = (x - px) ** 2 + (y - py) ** 2;
        const v = Math.exp(-d2 / (2 * 0.45 * 0.45));
        if (v > m) m = v;
      }
      img[r * SIDE + c] = m;
    }
  return img;
}

export const rmse = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
};

export const mix = (a: ArrayLike<number>, b: ArrayLike<number>, t: number) => {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (1 - t) * a[i] + t * b[i];
  return out;
};

/** Every real face on a fine grid of knob settings, for "how close is this to a real face?" */
export const ATLAS_S = 41; // smile −1 … 1 in steps of 0.05
export const ATLAS_E = 41; // eyes 0 … 1 in steps of 0.025
let atlasCache: { knobs: Knobs[]; imgs: Float64Array[] } | null = null;
export function atlas() {
  if (!atlasCache) {
    const knobs: Knobs[] = [];
    const imgs: Float64Array[] = [];
    for (let i = 0; i < ATLAS_S; i++)
      for (let j = 0; j < ATLAS_E; j++) {
        const k = { smile: -1 + (2 * i) / (ATLAS_S - 1), eyes: j / (ATLAS_E - 1) };
        knobs.push(k);
        imgs.push(face(k.smile, k.eyes));
      }
    atlasCache = { knobs, imgs };
  }
  return atlasCache;
}

/** The nearest real face to an image (over the atlas), and how far away it is (pixel RMSE). */
export function nearestFace(img: ArrayLike<number>): { knobs: Knobs; dist: number } {
  const { knobs, imgs } = atlas();
  let best = Infinity;
  let bi = 0;
  for (let i = 0; i < imgs.length; i++) {
    const im = imgs[i];
    let s = 0;
    for (let p = 0; p < PIX; p++) {
      const d = img[p] - im[p];
      s += d * d;
      if (s >= best) break;
    }
    if (s < best) {
      best = s;
      bi = i;
    }
  }
  return { knobs: knobs[bi], dist: Math.sqrt(best / PIX) };
}

/** A decoded image counts as a "real-looking face" when it is this close (pixel RMSE) to some real face. */
export const REAL_TOL = 0.06;

/** Independent pixels: each pixel drawn on its own from its average brightness over the data. */
export function pixelMeans(data: Float64Array[]): Float64Array {
  const m = new Float64Array(PIX);
  for (const d of data) for (let p = 0; p < PIX; p++) m[p] += d[p] / data.length;
  return m;
}

export function independentPixelSample(means: Float64Array, rng: Rng): Float64Array {
  // each pixel is on (1) with probability equal to its average brightness
  const out = new Float64Array(PIX);
  for (let p = 0; p < PIX; p++) out[p] = rng.next() < means[p] ? 1 : 0;
  return out;
}

export type Dataset = { knobs: Knobs[]; imgs: Float64Array[] };
export function makeDataset(n: number, seed: number): Dataset {
  const rng = createRng(seed);
  const knobs: Knobs[] = [];
  const imgs: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const k = { smile: rng.uniform(-1, 1), eyes: rng.uniform(0, 1) };
    knobs.push(k);
    imgs.push(face(k.smile, k.eyes));
  }
  return { knobs, imgs };
}

// ------------------------------------------------------------------ the network

export type Mode = 'ae' | 'vae';

/** Parameter layout inside one flat Float64Array. */
export function layout(H: number, LATENT: number) {
  const sizes = {
    eW1: H * PIX,
    eb1: H,
    eW2: 2 * LATENT * H, // rows 0..L-1 → μ, rows L..2L-1 → log σ²
    eb2: 2 * LATENT,
    dW1: H * LATENT,
    db1: H,
    dW2: PIX * H,
    db2: PIX,
  };
  const off: Record<keyof typeof sizes, number> = {} as never;
  let o = 0;
  for (const k of Object.keys(sizes) as (keyof typeof sizes)[]) {
    off[k] = o;
    o += sizes[k];
  }
  return { off, total: o };
}

export class VAE {
  readonly H: number;
  readonly L: number;
  readonly mode: Mode;
  beta: number;
  p: Float64Array;
  g: Float64Array;
  private off: ReturnType<typeof layout>['off'];
  // Adam state
  private m: Float64Array;
  private v: Float64Array;
  private t = 0;

  constructor(H: number, mode: Mode, beta: number, seed: number, latent = 2) {
    const LATENT = latent;
    this.H = H;
    this.L = latent;
    this.mode = mode;
    this.beta = beta;
    const L = layout(H, LATENT);
    this.off = L.off;
    this.p = new Float64Array(L.total);
    this.g = new Float64Array(L.total);
    this.m = new Float64Array(L.total);
    this.v = new Float64Array(L.total);
    const rng = createRng(seed);
    const init = (o: number, n: number, fanIn: number, fanOut: number) => {
      const sd = Math.sqrt(2 / (fanIn + fanOut));
      for (let i = 0; i < n; i++) this.p[o + i] = rng.normal(0, sd);
    };
    init(L.off.eW1, H * PIX, PIX, H);
    init(L.off.eW2, 2 * LATENT * H, H, 2 * LATENT);
    init(L.off.dW1, H * LATENT, LATENT, H);
    init(L.off.dW2, PIX * H, H, PIX);
    // start the log-variances small so a fresh VAE isn't drowning in noise
    for (let j = 0; j < LATENT; j++) this.p[L.off.eb2 + LATENT + j] = -2;
  }

  /** Encoder: image → (μ, log σ²). */
  encode(x: ArrayLike<number>) {
    const { H, p, off, L: LATENT } = this;
    const h = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let s = p[off.eb1 + j];
      const row = off.eW1 + j * PIX;
      for (let i = 0; i < PIX; i++) s += p[row + i] * x[i];
      h[j] = Math.tanh(s);
    }
    const mu = new Float64Array(LATENT);
    const logvar = new Float64Array(LATENT);
    for (let k = 0; k < 2 * LATENT; k++) {
      let s = p[off.eb2 + k];
      const row = off.eW2 + k * H;
      for (let j = 0; j < H; j++) s += p[row + j] * h[j];
      if (k < LATENT) mu[k] = s;
      else logvar[k - LATENT] = s;
    }
    return { h, mu, logvar };
  }

  /** Decoder: latent code → image (pixel brightnesses in 0 … 1). */
  decode(z: ArrayLike<number>) {
    const { H, p, off, L: LATENT } = this;
    const h = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let s = p[off.db1 + j];
      for (let k = 0; k < LATENT; k++) s += p[off.dW1 + j * LATENT + k] * z[k];
      h[j] = Math.tanh(s);
    }
    const out = new Float64Array(PIX);
    for (let i = 0; i < PIX; i++) {
      let s = p[off.db2 + i];
      const row = off.dW2 + i * H;
      for (let j = 0; j < H; j++) s += p[row + j] * h[j];
      out[i] = 1 / (1 + Math.exp(-s));
    }
    return { h, out };
  }

  /** The code used for an image: the encoder's mean (no noise). */
  code(x: ArrayLike<number>): number[] {
    return [...this.encode(x).mu];
  }

  reconstruct(x: ArrayLike<number>) {
    return this.decode(this.code(x)).out;
  }

  /**
   * Loss for one image, with the VAE's noise ε supplied explicitly (so gradients can be checked).
   * Adds the gradient (times `scale`) into this.g when `accumulate` is set.
   */
  lossOne(x: ArrayLike<number>, eps: ArrayLike<number>, accumulate = false, scale = 1) {
    const { H, p, g, off, L: LATENT } = this;
    const enc = this.encode(x);
    const vae = this.mode === 'vae';
    const z = new Float64Array(LATENT);
    const sd = new Float64Array(LATENT);
    for (let k = 0; k < LATENT; k++) {
      sd[k] = vae ? Math.exp(0.5 * enc.logvar[k]) : 0;
      z[k] = enc.mu[k] + sd[k] * (vae ? eps[k] : 0);
    }
    const dec = this.decode(z);
    let recon = 0;
    for (let i = 0; i < PIX; i++) {
      const q = Math.min(Math.max(dec.out[i], 1e-12), 1 - 1e-12);
      recon -= x[i] * Math.log(q) + (1 - x[i]) * Math.log(1 - q);
    }
    let kl = 0;
    if (vae) for (let k = 0; k < LATENT; k++) kl += 0.5 * (enc.mu[k] ** 2 + Math.exp(enc.logvar[k]) - 1 - enc.logvar[k]);
    const loss = recon + (vae ? this.beta * kl : 0);
    if (!accumulate) return { loss, recon, kl };

    // ---- backward. BCE through a sigmoid: dLoss/d(pre-activation) = output − target.
    const dh2 = new Float64Array(H);
    for (let i = 0; i < PIX; i++) {
      const d = (dec.out[i] - x[i]) * scale;
      g[off.db2 + i] += d;
      const row = off.dW2 + i * H;
      for (let j = 0; j < H; j++) {
        g[row + j] += d * dec.h[j];
        dh2[j] += d * p[row + j];
      }
    }
    const dz = new Float64Array(LATENT);
    for (let j = 0; j < H; j++) {
      const da = dh2[j] * (1 - dec.h[j] * dec.h[j]);
      g[off.db1 + j] += da;
      for (let k = 0; k < LATENT; k++) {
        g[off.dW1 + j * LATENT + k] += da * z[k];
        dz[k] += da * p[off.dW1 + j * LATENT + k];
      }
    }
    // z = μ + σ ε,  σ = exp(½ log σ²)  →  dz/dμ = 1, dz/d(log σ²) = ½ σ ε
    const dout = new Float64Array(2 * LATENT);
    for (let k = 0; k < LATENT; k++) {
      dout[k] = dz[k] + (vae ? this.beta * enc.mu[k] * scale : 0);
      dout[LATENT + k] = vae ? dz[k] * 0.5 * sd[k] * eps[k] + this.beta * 0.5 * (Math.exp(enc.logvar[k]) - 1) * scale : 0;
    }
    const dh1 = new Float64Array(H);
    for (let k = 0; k < 2 * LATENT; k++) {
      if (dout[k] === 0) continue;
      g[off.eb2 + k] += dout[k];
      const row = off.eW2 + k * H;
      for (let j = 0; j < H; j++) {
        g[row + j] += dout[k] * enc.h[j];
        dh1[j] += dout[k] * p[row + j];
      }
    }
    for (let j = 0; j < H; j++) {
      const da = dh1[j] * (1 - enc.h[j] * enc.h[j]);
      if (da === 0) continue;
      g[off.eb1 + j] += da;
      const row = off.eW1 + j * PIX;
      for (let i = 0; i < PIX; i++) g[row + i] += da * x[i];
    }
    return { loss, recon, kl };
  }

  /** One Adam step on a mini-batch. Returns the batch's average loss terms. */
  step(batch: ArrayLike<number>[], rng: Rng, lr = 0.003) {
    this.g.fill(0);
    let loss = 0;
    let recon = 0;
    let kl = 0;
    const eps = new Float64Array(this.L);
    for (const x of batch) {
      for (let k = 0; k < this.L; k++) eps[k] = rng.normal();
      const r = this.lossOne(x, eps, true, 1 / batch.length);
      loss += r.loss / batch.length;
      recon += r.recon / batch.length;
      kl += r.kl / batch.length;
    }
    // Adam (β₁ = 0.9, β₂ = 0.999)
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
    return { loss, recon, kl };
  }
}

// ------------------------------------------------------------------ training + evaluation

export type TrainConfig = { mode: Mode; beta: number; latent: number; H: number; steps: number; batch: number; lr: number; seed: number };
export const DEFAULT_TRAIN: Omit<TrainConfig, 'mode' | 'beta'> = { latent: 2, H: 32, steps: 3000, batch: 32, lr: 0.003, seed: 1 };
export const TRAIN_SIZE = 400;
export const DATA_SEED = 7;

/** A training run that can be advanced in chunks (so a browser can animate it). */
export function trainer(cfg: TrainConfig, data: Dataset = makeDataset(TRAIN_SIZE, DATA_SEED)) {
  const net = new VAE(cfg.H, cfg.mode, cfg.beta, cfg.seed, cfg.latent);
  const rng = createRng(cfg.seed * 7919 + 13);
  let done = 0;
  const history: { step: number; loss: number; recon: number; kl: number }[] = [];
  return {
    net,
    data,
    get done() {
      return done;
    },
    history,
    advance(n: number) {
      const end = Math.min(cfg.steps, done + n);
      while (done < end) {
        const batch: Float64Array[] = [];
        for (let b = 0; b < cfg.batch; b++) batch.push(data.imgs[rng.int(data.imgs.length)]);
        const r = this.net.step(batch, rng, cfg.lr);
        done++;
        if (done % 50 === 0 || done === cfg.steps) history.push({ step: done, ...r });
      }
      return done >= cfg.steps;
    },
  };
}

export function train(cfg: TrainConfig, data?: Dataset) {
  const t = trainer(cfg, data);
  t.advance(cfg.steps);
  return t;
}

/** Mean and covariance of a cloud of codes (any dimension), and a sampler from the bell fitted to them. */
export function fitBell(codes: number[][]) {
  const n = codes.length;
  const D = codes[0].length;
  const mean = new Array(D).fill(0);
  for (const c of codes) for (let i = 0; i < D; i++) mean[i] += c[i] / n;
  const cov = Array.from({ length: D }, () => new Array(D).fill(0));
  for (const c of codes)
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++) cov[i][j] += ((c[i] - mean[i]) * (c[j] - mean[j])) / n;
  // Cholesky factor: cov = Lc Lcᵀ, so mean + Lc·(standard normals) has exactly this mean and covariance
  const Lc = Array.from({ length: D }, () => new Array(D).fill(0));
  for (let i = 0; i < D; i++)
    for (let j = 0; j <= i; j++) {
      let s = cov[i][j];
      for (let k = 0; k < j; k++) s -= Lc[i][k] * Lc[j][k];
      Lc[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / Lc[j][j];
    }
  return {
    mean,
    cov,
    sample: (rng: Rng): number[] => {
      const e = Array.from({ length: D }, () => rng.normal());
      return mean.map((m, i) => {
        let v = m;
        for (let k = 0; k <= i; k++) v += Lc[i][k] * e[k];
        return v;
      });
    },
  };
}

export type Report = {
  /** average pixel RMSE between a held-out face and its reconstruction */
  reconRmse: number;
  /** share of generated faces within REAL_TOL of some real face */
  realShare: number;
  /** share of the 4×4 grid of knob cells (smile × eyes) that at least one real-looking sample lands in */
  coverage: number;
  /** average KL per image (VAE only, else 0) */
  kl: number;
};

export const EVAL_SEED = 99;
export const N_SAMPLES = 300;

/**
 * How we sample new faces: the VAE draws codes from the standard bell N(0, 1) it was trained to
 * fill. A plain autoencoder never promised anything about its codes, so — to be generous — we
 * fit a bell to its own training codes and sample from that.
 */
export function sampler(net: VAE, data: Dataset) {
  if (net.mode === 'vae') return (rng: Rng): number[] => Array.from({ length: net.L }, () => rng.normal());
  return fitBell(data.imgs.map((x) => net.code(x))).sample;
}

export function evaluate(net: VAE, data: Dataset, nSamples = N_SAMPLES, seed = EVAL_SEED): Report {
  const test = makeDataset(100, 12345);
  let rec = 0;
  let kl = 0;
  for (const x of test.imgs) {
    rec += rmse(x, net.reconstruct(x)) / test.imgs.length;
    const { mu, logvar } = net.encode(x);
    if (net.mode === 'vae') for (let k = 0; k < net.L; k++) kl += (0.5 * (mu[k] ** 2 + Math.exp(logvar[k]) - 1 - logvar[k])) / test.imgs.length;
  }
  const rng = createRng(seed);
  const draw = sampler(net, data);
  let real = 0;
  const cells = new Set<number>();
  for (let i = 0; i < nSamples; i++) {
    const img = net.decode(draw(rng)).out;
    const nf = nearestFace(img);
    if (nf.dist <= REAL_TOL) {
      real++;
      const a = Math.min(3, Math.floor(((nf.knobs.smile + 1) / 2) * 4));
      const b = Math.min(3, Math.floor(nf.knobs.eyes * 4));
      cells.add(a * 4 + b);
    }
  }
  return { reconRmse: rec, realShare: real / nSamples, coverage: cells.size / 16, kl };
}
