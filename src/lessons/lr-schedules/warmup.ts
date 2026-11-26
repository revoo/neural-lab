// authored-by: Claude Opus 5.5
/**
 * V2-WP3: the live warmup experiment.
 *
 * A deep, narrow-ish ReLU network (8 hidden layers of 64, He initialisation) learns a two-arm
 * spiral with Adam, mini-batches of 32, for 1,000 steps. Every seed is trained twice from the same
 * start and on the same batches: once at a constant step size, once after a linear warmup to that
 * same step size over the first 300 steps. Seeds × {no warmup, warmup} run side by side as replicas
 * of one engine (GPU when available, else the CPU Float32 tier).
 *
 * What our searches found (recorded in `warmupResults.ts` by `warmup.run.ts`, re-verified in
 * `warmup.test.ts`): warmup shrinks the loss spike at the start of training, but it does not change
 * how many runs end up solving the spiral. The lesson says exactly that.
 */
import type { MlpEngine, MlpSpec, Weights } from '../../lib/compute/types';
import { createRng } from '../../lib/rng';

export const WARM = {
  points: 400,
  hidden: [64, 64, 64, 64, 64, 64, 64, 64],
  batch: 32,
  steps: 1000,
  warmup: 300,
  every: 20,
  peaks: [0.01, 0.02, 0.03] as const,
  /** Seeds the widget trains live. The recorded results use more (see `WARM_SEEDS_RECORDED`). */
  seeds: [1, 2, 3, 4, 5, 6],
} as const;

export const WARM_SEEDS_RECORDED = Array.from({ length: 20 }, (_, i) => i + 1);

export const warmSpec = (replicas: number): MlpSpec => ({
  sizes: [2, ...WARM.hidden, 1],
  acts: [...WARM.hidden.map(() => 'relu' as const), 'sigmoid'],
  loss: 'bce',
  opt: { kind: 'adam' },
  replicas,
});

/** Two interleaved spiral arms, 200 points each, with a little noise (fixed: the data never changes). */
export function spiral() {
  const n = WARM.points;
  const r = createRng(2024);
  const X = new Float32Array(n * 2);
  const Y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const t = (Math.floor(i / 2) / (n / 2)) * 3 * Math.PI + 0.3;
    const rad = t / (3 * Math.PI + 0.3);
    const a = t + cls * Math.PI;
    X[2 * i] = rad * Math.cos(a) + r.normal(0, 0.02);
    X[2 * i + 1] = rad * Math.sin(a) + r.normal(0, 0.02);
    Y[i] = cls;
  }
  return { X, Y, n };
}

/** He initialisation (normal, std √(2 / fan-in)), zero biases. */
export function heInit(seed: number): Weights {
  const sizes = [2, ...WARM.hidden, 1];
  const r = createRng(seed * 7919 + 13);
  return {
    W: sizes.slice(1).map((o, l) => Float32Array.from({ length: o * sizes[l] }, () => r.normal(0, Math.sqrt(2 / sizes[l])))),
    b: sizes.slice(1).map((o) => new Float32Array(o)),
  };
}

/** The step size at step t (0-based): linear ramp to `peak` over `warm` steps, then constant. */
export const lrAt = (t: number, peak: number, warm: number) => (warm > 0 && t < warm ? (peak * (t + 1)) / warm : peak);

/** Mini-batch row indices for one seed: the same for both arms of that seed, so only warmup differs. */
export function batches(seed: number, steps: number) {
  const r = createRng(seed * 31 + 5);
  return Int32Array.from({ length: steps * WARM.batch }, () => r.int(WARM.points));
}

export function accuracy(out: Float32Array, Y: Float32Array) {
  let c = 0;
  for (let i = 0; i < Y.length; i++) if (out[i] > 0.5 === Y[i] > 0.5) c++;
  return c / Y.length;
}

export type WarmCheckpoint = {
  step: number;
  /** Mini-batch loss averaged over the last `every` steps, [seed][arm] with arm 0 = no warmup, 1 = warmup. */
  loss: number[][];
  /** Worst single mini-batch loss in the last `every` steps, [seed][arm]. */
  worst: number[][];
  /** Accuracy on all 400 points, [seed][arm]. */
  acc: number[][];
};

export type WarmRun = {
  /** Worst single mini-batch loss in the first `WARM.warmup` steps, [seed][arm]. */
  earlyMax: number[][];
  history: WarmCheckpoint[];
};

/** Replica layout: seed s, arm a → replica 2s + a. */
export async function runWarmup(engine: MlpEngine, seeds: readonly number[], peak: number, onCheckpoint?: (c: WarmCheckpoint, run: WarmRun) => boolean | Promise<boolean>, steps: number = WARM.steps): Promise<WarmRun> {
  const S = seeds.length;
  const R = 2 * S;
  const B = WARM.batch;
  const K = WARM.every;
  const d = spiral();
  engine.setData(d.X, d.Y, d.n);
  seeds.forEach((s, i) => {
    const w = heInit(s);
    engine.setWeights(2 * i, w);
    engine.setWeights(2 * i + 1, w);
  });
  const rows = seeds.map((s) => batches(s, steps));
  const run: WarmRun = { earlyMax: seeds.map(() => [0, 0]), history: [] };
  for (let t0 = 0; t0 < steps; t0 += K) {
    const idx = new Int32Array(K * R * B);
    const lr = new Float32Array(K * R);
    for (let k = 0; k < K; k++)
      for (let i = 0; i < S; i++)
        for (let a = 0; a < 2; a++) {
          const r = 2 * i + a;
          idx.set(rows[i].subarray((t0 + k) * B, (t0 + k + 1) * B), (k * R + r) * B);
          lr[k * R + r] = lrAt(t0 + k, peak, a ? WARM.warmup : 0);
        }
    const losses = await engine.train({ steps: K, batch: B, idx, lr });
    const loss = seeds.map(() => [0, 0]);
    const worst = seeds.map(() => [0, 0]);
    for (let k = 0; k < K; k++)
      for (let i = 0; i < S; i++)
        for (let a = 0; a < 2; a++) {
          const v = losses[k * R + 2 * i + a];
          loss[i][a] += v / K;
          if (v > worst[i][a]) worst[i][a] = v;
          if (t0 + k < WARM.warmup && v > run.earlyMax[i][a]) run.earlyMax[i][a] = v;
        }
    const out = await engine.predict(d.X, d.n);
    const acc = seeds.map((_, i) => [0, 1].map((a) => accuracy(out.subarray((2 * i + a) * d.n, (2 * i + a + 1) * d.n), d.Y)));
    const c = { step: t0 + K, loss, worst, acc };
    run.history.push(c);
    if (onCheckpoint && !(await onCheckpoint(c, run))) break;
  }
  return run;
}

/** A run "solved" the spiral if it classifies at least 99% of the points right; it "died" if it ends near guessing. */
export const solved = (acc: number) => acc >= 0.99;
export const died = (acc: number) => acc <= 0.6;

export const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Counts quoted in the lesson, for one arm of a recorded run. */
export function summarise(earlyMax: number[], finalAcc: number[]) {
  return {
    n: finalAcc.length,
    medianEarly: +median(earlyMax).toFixed(2),
    worstEarly: +Math.max(...earlyMax).toFixed(2),
    solved: finalAcc.filter(solved).length,
    died: finalAcc.filter(died).length,
  };
}
