// authored-by: Claude Opus 5.5
/**
 * V2-WP3: a live grokking demo on modular addition, (a + b) mod 17.
 *
 * The model is the minimal one from Gromov (2023), "Grokking modular arithmetic" (arXiv 2301.02679):
 * one-hot a and b side by side, one hidden layer with a quadratic activation, MSE loss on a one-hot
 * answer, full-batch training. We use AdamW, which that paper reports makes grokking happen sooner.
 * Settings were chosen by a small search (recorded in `grokResults.ts`, produced by
 * `grok.run.ts`, re-verified in `grokking.test.ts`).
 *
 * Four seeds train side by side as four replicas of one engine; each has its own split of the
 * 289 sums into training and validation halves, passed as per-replica batch indices.
 */
import type { MlpEngine, MlpSpec, Weights } from '../../lib/compute/types';
import { createRng } from '../../lib/rng';

export const GROK = {
  p: 17,
  hidden: 64,
  trainFrac: 0.5,
  lr: 0.005,
  /** AdamW decoupled weight decay when "on". */
  weightDecay: 2,
  beta2: 0.98,
  steps: 6000,
  every: 100,
  seeds: [1, 2, 3, 4],
} as const;

export const grokSpec = (weightDecay: number, replicas: number): MlpSpec => ({
  sizes: [2 * GROK.p, GROK.hidden, GROK.p],
  acts: ['square', 'linear'],
  loss: 'mse',
  opt: { kind: 'adamw', weightDecay, beta2: GROK.beta2 },
  replicas,
});

/** All p² pairs, in order (a, b) = (0,0), (0,1), …: X is one-hot a ‖ one-hot b, Y is one-hot (a + b) mod p. */
export function grokData() {
  const { p } = GROK;
  const n = p * p;
  const X = new Float32Array(n * 2 * p);
  const Y = new Float32Array(n * p);
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      const i = a * p + b;
      X[i * 2 * p + a] = 1;
      X[i * 2 * p + p + b] = 1;
      Y[i * p + ((a + b) % p)] = 1;
    }
  return { X, Y, n };
}

/** Seeded shuffle of the p² rows into a training half and a validation half. */
export function grokSplit(seed: number) {
  const n = GROK.p * GROK.p;
  const rows = Array.from({ length: n }, (_, i) => i);
  const r = createRng(seed);
  for (let i = n - 1; i > 0; i--) {
    const j = r.int(i + 1);
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  const k = Math.round(n * GROK.trainFrac);
  return { train: Int32Array.from(rows.slice(0, k)), val: Int32Array.from(rows.slice(k)) };
}

/** Gaussian start, scaled by 1/√(fan-in). */
export function grokInit(seed: number): Weights {
  const { p, hidden: H } = GROK;
  const r = createRng(seed + 99);
  return {
    W: [Float32Array.from({ length: 2 * p * H }, () => r.normal(0, 1 / Math.sqrt(2 * p))), Float32Array.from({ length: H * p }, () => r.normal(0, 1 / Math.sqrt(H)))],
    b: [new Float32Array(H), new Float32Array(p)],
  };
}

/** Share of rows whose largest output is the right answer. `out` is one replica's [p² × p] outputs. */
export function accuracy(out: Float32Array, rows: Int32Array) {
  const { p } = GROK;
  let c = 0;
  for (const i of rows) {
    let am = 0;
    for (let j = 1; j < p; j++) if (out[i * p + j] > out[i * p + am]) am = j;
    if (am === (Math.floor(i / p) + (i % p)) % p) c++;
  }
  return c / rows.length;
}

export type GrokCheckpoint = { step: number; loss: number[]; trainAcc: number[]; valAcc: number[] };

/**
 * Train every seed for `GROK.steps` full-batch steps, reporting a checkpoint every `GROK.every`
 * steps. `onCheckpoint` returns false to stop early. Works with either compute tier.
 */
export async function runGrokking(engine: MlpEngine, seeds: readonly number[], onCheckpoint: (c: GrokCheckpoint) => boolean | Promise<boolean>, steps: number = GROK.steps) {
  const R = seeds.length;
  const data = grokData();
  engine.setData(data.X, data.Y, data.n);
  const splits = seeds.map(grokSplit);
  seeds.forEach((s, r) => engine.setWeights(r, grokInit(s)));
  const B = splits[0].train.length;
  const K = GROK.every;
  const idx = new Int32Array(K * R * B);
  for (let k = 0; k < K; k++) splits.forEach((sp, r) => idx.set(sp.train, (k * R + r) * B));
  const lr = new Float32Array(K * R).fill(GROK.lr);
  for (let step = K; step <= steps; step += K) {
    const losses = await engine.train({ steps: K, batch: B, idx, lr });
    const out = await engine.predict(data.X, data.n);
    const per = data.n * GROK.p;
    const c: GrokCheckpoint = {
      step,
      loss: seeds.map((_, r) => losses[(K - 1) * R + r]),
      trainAcc: splits.map((sp, r) => accuracy(out.subarray(r * per, (r + 1) * per), sp.train)),
      valAcc: splits.map((sp, r) => accuracy(out.subarray(r * per, (r + 1) * per), sp.val)),
    };
    if (!(await onCheckpoint(c))) return;
  }
}

/** First checkpoint step at which accuracy reached `level` for each seed (null if never). */
export function firstReach(history: GrokCheckpoint[], key: 'trainAcc' | 'valAcc', level: number, seeds: number) {
  return Array.from({ length: seeds }, (_, r) => history.find((h) => h[key][r] >= level)?.step ?? null);
}
