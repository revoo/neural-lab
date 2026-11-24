/**
 * The model clinic's engine: a small, fully seeded classification trainer (adapted from the
 * training lab, duplicated so lessons stay independent) with the knobs needed to break training in
 * realistic ways — label noise, unscaled inputs, unshuffled data, a test set that leaks training
 * points, bad initialisation — and the instruments a practitioner would use to find out what went
 * wrong.
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';

export const DOMAIN = 1.6;

export type DatasetName = 'circle' | 'moons' | 'xor';
export type Pt = { x: number; y: number; label: 0 | 1 };

/** n points, half of each class, with Gaussian noise. Same generators as the training lab. */
export function generate(name: DatasetName, n: number, noise: number, seed: number): Pt[] {
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

// ---------------------------------------------------------------- configuration

export type Act = 'tanh' | 'relu';
export type OptName = 'sgd' | 'momentum' | 'adam';
export type Init = 'standard' | 'large' | 'negative-bias';

export type Config = {
  // the data pipeline
  dataset: DatasetName;
  /** total points collected (half train, half test) */
  n: number;
  noise: number;
  /** fraction of labels (train and test alike) replaced by coin flips */
  labelNoise: number;
  /** the x coordinate arrives multiplied by this (e.g. recorded in different units) */
  xScale: number;
  /** standardise each input using the training set's mean and spread */
  normalise: boolean;
  /** false: the training file is sorted by class and batches are taken in order */
  shuffle: boolean;
  /** fraction of the test set that is really copies of training points */
  leak: number;
  // the model
  hidden: number[];
  activation: Act;
  init: Init;
  // the optimisation
  optimizer: OptName;
  lr: number;
  batch: number;
  l2: number;
  steps: number;
  seed: number;
};

export const HEALTHY: Config = {
  dataset: 'moons',
  n: 400,
  noise: 0.15,
  labelNoise: 0,
  xScale: 1,
  normalise: false,
  shuffle: true,
  leak: 0,
  hidden: [8, 8],
  activation: 'tanh',
  init: 'standard',
  optimizer: 'adam',
  lr: 0.01,
  batch: 10,
  l2: 0,
  steps: 2000,
  seed: 1,
};

export const paramCount = (cfg: Config) => {
  const s = [2, ...cfg.hidden, 1];
  return s.slice(1).reduce((t, n, l) => t + s[l] * n + n, 0);
};

// ---------------------------------------------------------------- data, as the pipeline delivers it

export type Data = {
  train: Pt[];
  test: Pt[];
  /** fresh, clean data from the real source, collected later ("production") */
  fresh: Pt[];
  /** how the model sees a point: after unit mix-ups and (optional) normalisation */
  encode: (x: number, y: number) => number[];
};

export function buildData(cfg: Config): Data {
  const r = createRng(cfg.seed * 1_000_003 + 11);
  const pts = generate(cfg.dataset, cfg.n, cfg.noise, cfg.seed).map((p) => ({
    ...p,
    label: (r.next() < cfg.labelNoise ? (r.next() < 0.5 ? 0 : 1) : p.label) as 0 | 1,
  }));
  // seeded shuffle, then split in half
  for (let i = pts.length - 1; i > 0; i--) {
    const j = r.int(i + 1);
    [pts[i], pts[j]] = [pts[j], pts[i]];
  }
  const half = Math.floor(pts.length / 2);
  let train = pts.slice(0, half);
  let test = pts.slice(half);
  if (cfg.leak > 0) {
    const k = Math.round(test.length * cfg.leak);
    test = [...train.slice(0, k), ...test.slice(k)];
  }
  if (!cfg.shuffle) train = [...train].sort((a, b) => a.label - b.label);
  const fresh = generate(cfg.dataset, 600, cfg.noise, cfg.seed + 7777);
  // inputs as recorded: x possibly in other units
  const raw = (x: number, y: number) => [x * cfg.xScale, y];
  let mean = [0, 0];
  let sd = [1, 1];
  if (cfg.normalise) {
    const R = train.map((p) => raw(p.x, p.y));
    mean = [0, 1].map((i) => R.reduce((t, v) => t + v[i], 0) / R.length);
    sd = [0, 1].map((i) => Math.sqrt(R.reduce((t, v) => t + (v[i] - mean[i]) ** 2, 0) / R.length) || 1);
  }
  const encode = (x: number, y: number) => raw(x, y).map((v, i) => (v - mean[i]) / sd[i]);
  return { train, test, fresh, encode };
}

// ---------------------------------------------------------------- training, with a record of everything

export type Snapshot = {
  step: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
  freshAcc: number;
};

export type LayerStat = {
  /** neurons in this hidden layer that output exactly 0 for every training point (ReLU) */
  dead: number;
  /** neurons that sit at ±1 (|a| > 0.97) for at least 90% of training points (tanh) */
  saturated: number;
  size: number;
  /** root-mean-square weight coming into this layer */
  weightRms: number;
  /** root-mean-square gradient of those weights, on the last batch */
  gradRms: number;
};

export class Clinic {
  readonly cfg: Config;
  readonly data: Data;
  readonly net: MLP;
  private opt: ReturnType<typeof createOptimizer>;
  private rng;
  private X: number[][];
  private Y: number[][];
  private cursor = 0;
  steps = 0;
  history: Snapshot[] = [];
  /** fraction of class-1 examples in each recent batch */
  batchMix: number[] = [];
  /** fraction of hidden neurons dead (ReLU) or saturated (tanh) at step 0 */
  readonly initialTrouble: number;

  constructor(cfg: Config) {
    this.cfg = cfg;
    this.data = buildData(cfg);
    const acts = [...cfg.hidden.map(() => cfg.activation), 'sigmoid' as const];
    this.net = new MLP([2, ...cfg.hidden, 1], acts, createRng(cfg.seed), cfg.activation === 'relu' ? 'he' : 'xavier');
    if (cfg.init === 'large') for (const L of this.net.W.slice(0, -1)) for (const row of L) for (let i = 0; i < row.length; i++) row[i] *= 8;
    if (cfg.init === 'negative-bias') for (const b of this.net.b.slice(0, -1)) b.fill(-3);
    this.opt = createOptimizer({ name: cfg.optimizer, lr: cfg.lr, beta1: 0.9 });
    this.rng = createRng(cfg.seed * 31 + 7);
    this.X = this.data.train.map((p) => this.data.encode(p.x, p.y));
    this.Y = this.data.train.map((p) => [p.label]);
    this.initialTrouble = this.troubleFraction();
    this.record();
  }

  step(k = 1) {
    const n = this.X.length;
    const B = Math.min(this.cfg.batch, n);
    for (let s = 0; s < k; s++) {
      const bx: number[][] = [];
      const by: number[][] = [];
      for (let i = 0; i < B; i++) {
        let j: number;
        if (this.cfg.shuffle) j = this.rng.int(n);
        else {
          j = this.cursor;
          this.cursor = (this.cursor + 1) % n;
        }
        bx.push(this.X[j]);
        by.push(this.Y[j]);
      }
      this.net.backprop(bx, by, 'bce', this.cfg.l2);
      this.opt.step(this.net.params(), this.net.grads());
      this.steps++;
      this.batchMix.push(by.reduce((t, v) => t + v[0], 0) / B);
      if (this.batchMix.length > 120) this.batchMix.shift();
    }
  }

  /** Train to cfg.steps, recording a snapshot every `every` steps. */
  run(every = 50) {
    while (this.steps < this.cfg.steps) {
      this.step(Math.min(every, this.cfg.steps - this.steps));
      this.record();
      if (this.diverged) break;
    }
    return this;
  }

  get diverged() {
    return this.net.W.some((L) => L.some((r) => r.some((w) => !Number.isFinite(w))));
  }

  predict(x: number, y: number) {
    return this.net.predict(this.data.encode(x, y))[0];
  }

  evaluate(pts: Pt[]) {
    if (!pts.length) return { loss: NaN, acc: NaN };
    let loss = 0;
    let correct = 0;
    for (const p of pts) {
      const raw = this.predict(p.x, p.y);
      const q = Number.isFinite(raw) ? Math.min(Math.max(raw, 1e-7), 1 - 1e-7) : 0.5;
      loss += -(p.label ? Math.log(q) : Math.log(1 - q));
      if (Number.isFinite(raw) && (raw > 0.5 ? 1 : 0) === p.label) correct++;
    }
    return { loss: this.diverged ? NaN : loss / pts.length, acc: correct / pts.length };
  }

  record() {
    const a = this.evaluate(this.data.train);
    const b = this.evaluate(this.data.test);
    const c = this.evaluate(this.data.fresh);
    this.history.push({ step: this.steps, trainLoss: a.loss, testLoss: b.loss, trainAcc: a.acc, testAcc: b.acc, freshAcc: c.acc });
  }

  get last() {
    return this.history[this.history.length - 1];
  }

  /** Per hidden layer: dead and saturated neurons, weight and gradient sizes. */
  layerStats(): LayerStat[] {
    const outs = this.X.map((x) => this.net.forward(x).a);
    return this.cfg.hidden.map((size, l) => {
      let dead = 0;
      let saturated = 0;
      for (let j = 0; j < size; j++) {
        const vals = outs.map((a) => a[l + 1][j]);
        if (this.cfg.activation === 'relu' && vals.every((v) => v <= 1e-12)) dead++;
        if (this.cfg.activation === 'tanh' && vals.filter((v) => Math.abs(v) > 0.97).length >= 0.9 * vals.length) saturated++;
      }
      const W = this.net.W[l].flat();
      const G = this.net.gW[l].flat();
      const rms = (v: number[]) => Math.sqrt(v.reduce((t, x) => t + x * x, 0) / v.length);
      return { dead, saturated, size, weightRms: rms(W), gradRms: rms(G) };
    });
  }

  troubleFraction() {
    const s = this.layerStats();
    const total = s.reduce((t, l) => t + l.size, 0);
    return total ? s.reduce((t, l) => t + l.dead + l.saturated, 0) / total : 0;
  }
}

// ---------------------------------------------------------------- data checks (the "lab tests")

/** Range of each input, as the model receives it. */
export function inputRanges(c: Clinic) {
  const X = c.data.train.map((p) => c.data.encode(p.x, p.y));
  return [0, 1].map((i) => ({ min: Math.min(...X.map((v) => v[i])), max: Math.max(...X.map((v) => v[i])) }));
}

/** Fraction of test points that are exact copies of a training point. */
export function duplicateFraction(c: Clinic) {
  const keys = new Set(c.data.train.map((p) => `${p.x},${p.y}`));
  return c.data.test.filter((p) => keys.has(`${p.x},${p.y}`)).length / c.data.test.length;
}

/**
 * Label consistency: for each training point, does its label agree with the majority of its 5
 * nearest training neighbours? Clean data on these shapes scores well above 90%; random labels
 * score about 50%.
 */
export function neighbourAgreement(c: Clinic, k = 5) {
  const T = c.data.train;
  let agree = 0;
  for (const p of T) {
    const near = T.filter((q) => q !== p)
      .map((q) => ({ d: (q.x - p.x) ** 2 + (q.y - p.y) ** 2, l: q.label }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    const ones = near.filter((q) => q.l === 1).length;
    if ((ones > k / 2 ? 1 : 0) === p.label) agree++;
  }
  return agree / T.length;
}

/** Fraction of class 1 in the training set. */
export const classBalance = (c: Clinic) => c.data.train.filter((p) => p.label === 1).length / c.data.train.length;

/** Train a configuration to the end and report its final snapshot. */
export const runToEnd = (cfg: Config) => new Clinic(cfg).run();
