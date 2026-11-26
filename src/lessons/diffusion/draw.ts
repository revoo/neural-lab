// authored-by: Claude Opus 5.5
/**
 * Learning to draw (V1, B3): a diffusion model on 8 × 8 pictures of clothes, trained from scratch.
 *
 * The denoiser is an ordinary MLP from the shared compute library (so it runs on the CPU tier or on
 * WebGPU): input = the noisy picture's 64 pixels plus a 16-number code for the noise level,
 * output = its guess of the noise that was added. DDPM training (Ho et al. 2020's simple loss) with 100
 * noise levels; sampling runs the 100 steps backwards, clipping each step's estimate of the clean
 * picture to the pixel range.
 *
 * "Looks like a real picture" is measured, not judged: a sample's distance to the nearest training
 * picture, compared with how far genuinely real held-out pictures are from their nearest training
 * picture.
 */
import type { MlpEngine, MlpSpec, Weights } from '../../lib/compute/types';
import { createRng } from '../../lib/rng';

export const N_LEVELS = 100;
export const DRAW_TRAIN = 1500;
export const DRAW_HELD = 150;
export const DRAW_CLASSES = ['Trouser', 'Sneaker', 'Bag'];

export const BETAS = Array.from({ length: N_LEVELS }, (_, t) => 1e-4 + ((0.02 - 1e-4) * t) / (N_LEVELS - 1));
export const ALPHA_BAR = (() => {
  const out: number[] = [];
  let a = 1;
  for (const b of BETAS) out.push((a *= 1 - b));
  return out;
})();

/** Sixteen numbers telling the network which noise level it is looking at (sines and cosines of t). */
export function levelCode(t: number): Float32Array {
  const f = new Float32Array(16);
  for (let k = 0; k < 8; k++) {
    const w = Math.pow(100, -k / 8);
    f[2 * k] = Math.sin(t * w);
    f[2 * k + 1] = Math.cos(t * w);
  }
  return f;
}

export type DrawSet = { train: Float32Array[]; held: Float32Array[] };
/** Bytes of fashion8.bin → pixels scaled to [−1, 1]. */
export function decodeDraw(bytes: Uint8Array): DrawSet {
  const img = (i: number) => Float32Array.from(bytes.subarray(i * 64, (i + 1) * 64), (v) => (v / 255) * 2 - 1);
  return { train: Array.from({ length: DRAW_TRAIN }, (_, i) => img(i)), held: Array.from({ length: DRAW_HELD }, (_, i) => img(DRAW_TRAIN + i)) };
}

export type DrawSpec = { hidden: number; steps: number; batch: number; lr: number; seed: number };
export const DRAW_RUN: DrawSpec = { hidden: 256, steps: 10000, batch: 64, lr: 0.002, seed: 1 };
export const drawMlp = (s: DrawSpec): MlpSpec => ({ sizes: [80, s.hidden, s.hidden, 64], acts: ['relu', 'relu', 'linear'], loss: 'mse', opt: { kind: 'adam' } });

export function initialDrawWeights(s: DrawSpec): Weights {
  const r = createRng(s.seed);
  const sz = drawMlp(s).sizes;
  return {
    W: sz.slice(1).map((o, l) => Float32Array.from({ length: o * sz[l] }, () => r.normal(0, Math.sqrt(2 / sz[l])))),
    b: sz.slice(1).map((o) => new Float32Array(o)),
  };
}

/** Distance from x to the nearest training picture (Euclidean, in the [−1, 1] pixel scale). */
export function nearestTrain(x: ArrayLike<number>, train: Float32Array[]): number {
  let best = Infinity;
  for (const im of train) {
    let d = 0;
    for (let j = 0; j < 64; j++) d += (x[j] - im[j]) ** 2;
    if (d < best) best = d;
  }
  return Math.sqrt(best);
}

/** The "real range": the 95th percentile of held-out real pictures' nearest-training distances. */
export function realRange(d: DrawSet): number {
  const s = d.held.map((h) => nearestTrain(h, d.train)).sort((a, b) => a - b);
  return s[Math.floor(0.95 * (s.length - 1))];
}

/**
 * Draw `n` pictures with the current denoiser: start from pure noise (fixed seed) and run the 100
 * noise levels backwards. If `keep` is given, it receives the first picture after every step.
 */
export async function sampleDraw(engine: MlpEngine, n: number, seed: number, keep?: (t: number, x: Float32Array) => void): Promise<Float32Array[]> {
  const g = createRng(seed);
  const xs = Array.from({ length: n }, () => Float32Array.from({ length: 64 }, () => g.normal()));
  for (let t = N_LEVELS - 1; t >= 0; t--) {
    const inp = new Float32Array(n * 80);
    const code = levelCode(t);
    xs.forEach((x, i) => {
      inp.set(x, i * 80);
      inp.set(code, i * 80 + 64);
    });
    const eps = await engine.predict(inp, n);
    const ab = ALPHA_BAR[t];
    const abPrev = t > 0 ? ALPHA_BAR[t - 1] : 1;
    const b = BETAS[t];
    xs.forEach((x, i) => {
      for (let j = 0; j < 64; j++) {
        const x0 = Math.max(-1, Math.min(1, (x[j] - Math.sqrt(1 - ab) * eps[i * 64 + j]) / Math.sqrt(ab)));
        x[j] = ((Math.sqrt(abPrev) * b) / (1 - ab)) * x0 + ((Math.sqrt(1 - b) * (1 - abPrev)) / (1 - ab)) * x[j];
        if (t > 0) x[j] += Math.sqrt(b) * g.normal();
      }
    });
    keep?.(t, Float32Array.from(xs[0]));
  }
  return xs;
}

export function drawCheckpoints(steps: number): number[] {
  const out = new Set<number>([0]);
  for (let k = 0; ; k++) {
    const v = Math.round(10 ** (k / 5));
    if (v >= steps) break;
    if (v >= 10) out.add(v);
  }
  out.add(steps);
  return [...out].sort((a, b) => a - b);
}

/**
 * Train the denoiser with the DDPM loss: pick a picture and a noise level, add that much noise, and
 * ask for the noise back (mean squared error). Calls back at each checkpoint; return false to stop.
 */
export async function trainDraw(engine: MlpEngine, d: DrawSet, s: DrawSpec, onCheckpoint: (step: number) => boolean | void | Promise<boolean | void>, chunk = 50) {
  engine.setWeights(0, initialDrawWeights(s));
  const r = createRng(s.seed + 1000);
  const marks = drawCheckpoints(s.steps);
  if ((await onCheckpoint(0)) === false) return;
  let step = 0;
  let mi = 1;
  const B = s.batch;
  while (step < s.steps) {
    const K = Math.min(chunk, marks[mi] - step);
    const X = new Float32Array(K * B * 80);
    const Y = new Float32Array(K * B * 64);
    for (let k = 0; k < K * B; k++) {
      const im = d.train[r.int(d.train.length)];
      const t = r.int(N_LEVELS);
      const sa = Math.sqrt(ALPHA_BAR[t]);
      const sn = Math.sqrt(1 - ALPHA_BAR[t]);
      for (let j = 0; j < 64; j++) {
        const e = r.normal();
        X[k * 80 + j] = sa * im[j] + sn * e;
        Y[k * 64 + j] = e;
      }
      X.set(levelCode(t), k * 80 + 64);
    }
    engine.setData(X, Y, K * B);
    const idx = Int32Array.from({ length: K * B }, (_, i) => i);
    await engine.train({ steps: K, batch: B, idx, lr: new Float32Array(K).fill(s.lr) });
    step += K;
    if (step === marks[mi]) {
      mi++;
      if ((await onCheckpoint(step)) === false) return;
    }
  }
}

export type DrawMoment = { id: 'first' | 'half'; label: string; step: number };
/** Moments that hold "for good": the first sample inside the real range, and half the samples. */
export function drawMoments(hist: readonly { step: number; nn: readonly number[] }[], bar: number): DrawMoment[] {
  const tests: [DrawMoment['id'], string, (nn: readonly number[]) => boolean][] = [
    ['first', 'its first picture within the real-picture distance', (nn) => nn.some((v) => v <= bar)],
    ['half', 'half its pictures within the real-picture distance', (nn) => nn.filter((v) => v <= bar).length >= nn.length / 2],
  ];
  const out: DrawMoment[] = [];
  for (const [id, label, t] of tests) {
    let from: number | null = null;
    for (const h of hist) {
      if (t(h.nn)) from ??= h.step;
      else from = null;
    }
    if (from !== null) out.push({ id, label, step: from });
  }
  return out;
}
