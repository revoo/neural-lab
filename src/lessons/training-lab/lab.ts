/**
 * The training laboratory's engine: datasets, input features, and a deterministic trainer built on
 * the shared MLP and optimizers. Everything is seeded, so a configuration plus a seed always
 * produces exactly the same run — which is what lets challenge results be checked and reproduced.
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng, type Rng } from '../../lib/rng';
import type { ActivationName } from '../../lib/activations';

export const DOMAIN = 1.6; // points live in [−DOMAIN, DOMAIN]²

export type DatasetName = 'circle' | 'xor' | 'clusters' | 'spiral' | 'moons' | 'draw';
export const DATASET_LABELS: Record<DatasetName, string> = {
  circle: 'Circle',
  xor: 'XOR',
  clusters: 'Clusters',
  spiral: 'Spiral',
  moons: 'Moons',
  draw: 'Draw your own',
};

export type Pt = { x: number; y: number; label: 0 | 1 };

/** n points, half of each class, with Gaussian noise of size `noise` added to every coordinate. */
export function generate(name: Exclude<DatasetName, 'draw'>, n: number, noise: number, seed: number): Pt[] {
  const r = createRng(seed * 7919 + n);
  const pts: Pt[] = [];
  for (let k = 0; k < n; k++) {
    const label = (k % 2) as 0 | 1;
    let x = 0;
    let y = 0;
    if (name === 'circle') {
      const rad = label ? r.uniform(0.8, 1.3) : r.uniform(0, 0.5);
      const t = r.uniform(0, 2 * Math.PI);
      x = rad * Math.cos(t);
      y = rad * Math.sin(t);
    } else if (name === 'xor') {
      const sx = r.next() < 0.5 ? -1 : 1;
      const sy = label ? -sx : sx;
      x = sx * r.uniform(0.1, 1.3);
      y = sy * r.uniform(0.1, 1.3);
    } else if (name === 'clusters') {
      const c = label ? 0.6 : -0.6;
      x = r.normal(c, 0.3);
      y = r.normal(c, 0.3);
    } else if (name === 'spiral') {
      const t = r.uniform(0.05, 1);
      const a = t * 3.2 * Math.PI + label * Math.PI;
      x = 1.35 * t * Math.cos(a);
      y = 1.35 * t * Math.sin(a);
    } else {
      const t = r.uniform(0, Math.PI);
      if (label) {
        x = 1 - Math.cos(t) - 0.5;
        y = -Math.sin(t) + 0.25;
      } else {
        x = Math.cos(t) - 0.5;
        y = Math.sin(t) - 0.25;
      }
    }
    x += r.normal(0, noise);
    y += r.normal(0, noise);
    pts.push({ x: clamp(x), y: clamp(y), label });
  }
  return pts;
}
const clamp = (v: number) => Math.max(-DOMAIN, Math.min(DOMAIN, v));

/** Shuffle (seeded) and split into train and test sets. */
export function split(pts: Pt[], trainFrac: number, seed: number): { train: Pt[]; test: Pt[] } {
  const r = createRng(seed * 104729 + 17);
  const a = pts.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = r.int(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  const k = Math.max(1, Math.min(a.length - 1, Math.round(a.length * trainFrac)));
  return { train: a.slice(0, k), test: a.slice(k) };
}

// ---- Input features: what the network is allowed to see.

export type FeatureName = 'x' | 'y' | 'x2' | 'y2' | 'xy' | 'sinx' | 'siny';
export const FEATURES: { id: FeatureName; label: string; f: (x: number, y: number) => number }[] = [
  { id: 'x', label: 'x', f: (x) => x },
  { id: 'y', label: 'y', f: (_, y) => y },
  { id: 'x2', label: 'x²', f: (x) => x * x },
  { id: 'y2', label: 'y²', f: (_, y) => y * y },
  { id: 'xy', label: 'x·y', f: (x, y) => x * y },
  { id: 'sinx', label: 'sin 3x', f: (x) => Math.sin(3 * x) },
  { id: 'siny', label: 'sin 3y', f: (_, y) => Math.sin(3 * y) },
];
const FEATURE_FN = Object.fromEntries(FEATURES.map((f) => [f.id, f.f])) as Record<FeatureName, (x: number, y: number) => number>;
export const featurize = (features: FeatureName[], x: number, y: number) => features.map((f) => FEATURE_FN[f](x, y));

// ---- Configuration and training.

export type Act = 'tanh' | 'relu' | 'sigmoid' | 'linear';
export type OptName = 'sgd' | 'momentum' | 'adam';

export type Config = {
  features: FeatureName[];
  hidden: number[];
  activation: Act;
  optimizer: OptName;
  lr: number;
  batch: number;
  l2: number;
  seed: number;
};

export const DEFAULT_CONFIG: Config = { features: ['x', 'y'], hidden: [4, 4], activation: 'tanh', optimizer: 'adam', lr: 0.03, batch: 10, l2: 0, seed: 1 };

export const sizesOf = (cfg: Config) => [cfg.features.length, ...cfg.hidden, 1];
export const paramCount = (cfg: Config) => {
  const s = sizesOf(cfg);
  return s.slice(1).reduce((t, n, l) => t + s[l] * n + n, 0);
};

export type Metrics = { trainLoss: number; testLoss: number; trainAcc: number; testAcc: number };

export class Trainer {
  readonly cfg: Config;
  readonly net: MLP;
  private opt: Optimizer;
  private rng: Rng;
  private X: number[][];
  private Y: number[][];
  private Xt: number[][];
  private Yt: number[][];
  steps = 0;

  constructor(cfg: Config, train: Pt[], test: Pt[]) {
    this.cfg = cfg;
    const acts = [...cfg.hidden.map(() => cfg.activation as ActivationName), 'sigmoid' as ActivationName];
    this.net = new MLP(sizesOf(cfg), acts, createRng(cfg.seed), cfg.activation === 'relu' ? 'he' : 'xavier');
    this.opt = createOptimizer({ name: cfg.optimizer, lr: cfg.lr, beta1: 0.9 });
    this.rng = createRng(cfg.seed * 31 + 7);
    this.X = train.map((p) => featurize(cfg.features, p.x, p.y));
    this.Y = train.map((p) => [p.label]);
    this.Xt = test.map((p) => featurize(cfg.features, p.x, p.y));
    this.Yt = test.map((p) => [p.label]);
  }

  /** k mini-batch steps. Batches are drawn with a seeded RNG, so runs are reproducible. */
  step(k = 1) {
    const n = this.X.length;
    if (!n) return;
    const B = Math.min(this.cfg.batch, n);
    for (let s = 0; s < k; s++) {
      const bx: number[][] = [];
      const by: number[][] = [];
      for (let i = 0; i < B; i++) {
        const j = this.rng.int(n);
        bx.push(this.X[j]);
        by.push(this.Y[j]);
      }
      this.net.backprop(bx, by, 'bce', this.cfg.l2);
      this.opt.step(this.net.params(), this.net.grads());
      this.steps++;
    }
  }

  /**
   * V2-WP3: the rows `step(k)` would train on, drawn from the same seeded RNG and in the same order,
   * laid out [step][row]. The WebGPU path trains on exactly these batches, then hands the weights back
   * with `adopt`, so a run's batches don't depend on which tier computed it.
   */
  drawBatches(k: number): Int32Array {
    const n = this.X.length;
    const B = Math.min(this.cfg.batch, n);
    const out = new Int32Array(k * B);
    for (let i = 0; i < out.length; i++) out[i] = this.rng.int(n);
    return out;
  }

  /** The featurised training rows, flat and row-major (for uploading to a compute engine). */
  trainingData(): { X: Float32Array; Y: Float32Array; n: number } {
    return { X: Float32Array.from(this.X.flat()), Y: Float32Array.from(this.Y.flat()), n: this.X.length };
  }

  /** Take weights trained elsewhere (W[l] row-major [out × in]) and count the steps they took. */
  adopt(W: Float32Array[], b: Float32Array[], steps: number) {
    W.forEach((Wl, l) => this.net.W[l].forEach((row, j) => row.forEach((_, i) => (row[i] = Wl[j * row.length + i]))));
    b.forEach((bl, l) => this.net.b[l].forEach((_, j) => (this.net.b[l][j] = bl[j])));
    this.steps += steps;
  }

  get epochs() {
    return (this.steps * Math.min(this.cfg.batch, this.X.length)) / Math.max(1, this.X.length);
  }

  predictXY(x: number, y: number) {
    return this.net.predict(featurize(this.cfg.features, x, y))[0];
  }

  metrics(): Metrics {
    const ev = (X: number[][], Y: number[][]) => {
      if (!X.length) return { loss: NaN, acc: NaN };
      let loss = 0;
      let correct = 0;
      for (let i = 0; i < X.length; i++) {
        const p = Math.min(Math.max(this.net.predict(X[i])[0], 1e-7), 1 - 1e-7);
        loss += -(Y[i][0] ? Math.log(p) : Math.log(1 - p));
        if ((p > 0.5 ? 1 : 0) === Y[i][0]) correct++;
      }
      return { loss: loss / X.length, acc: correct / X.length };
    };
    const a = ev(this.X, this.Y);
    const b = ev(this.Xt, this.Yt);
    return { trainLoss: a.loss, testLoss: b.loss, trainAcc: a.acc, testAcc: b.acc };
  }

  get diverged() {
    return this.net.W.some((L) => L.some((r) => r.some((w) => !Number.isFinite(w))));
  }
}

/**
 * Train for up to `maxSteps`, checking every `every` steps. Returns the first checkpoint at which test
 * accuracy reached `target` (null if never) and the best test accuracy seen. This is exactly what a
 * learner experiences in the lab when they pause as soon as the target is hit.
 */
export function trainUntil(cfg: Config, train: Pt[], test: Pt[], target: number, maxSteps = 4000, every = 100) {
  const t = new Trainer(cfg, train, test);
  let best = 0;
  for (let s = every; s <= maxSteps; s += every) {
    t.step(every);
    if (t.diverged) return { reachedAt: null as number | null, best };
    const m = t.metrics();
    best = Math.max(best, m.testAcc);
    if (m.testAcc >= target) return { reachedAt: s, best };
  }
  return { reachedAt: null as number | null, best };
}
