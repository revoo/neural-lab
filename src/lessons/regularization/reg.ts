/**
 * The regularisation lesson's engine. The data is the overfitting lesson's 12-point curve problem
 * (duplicated here so lessons stay independent), plus a validation set: every cure has a strength
 * to choose, and choosing it must not touch the test set.
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';

// ---------------------------------------------------------------- data (same source as the overfitting lesson)

export const truth = (x: number) => Math.sin(2.6 * x) + 0.4 * x;
export const NOISE = 0.3;
export type Sample = { x: number[]; y: number[] };

/** n points spread over [−1, 1] (one per equal slice) with y = truth(x) + noise. */
export function sample(n: number, seed: number, noise = NOISE): Sample {
  const r = createRng(seed * 7717 + n * 31 + 1);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = -1 + (2 * (i + r.next())) / n;
    x.push(xi);
    y.push(truth(xi) + r.normal(0, noise));
  }
  return { x, y };
}

export const TEST_SET = sample(2000, 999_983);
/** Training set k (k = 1, 2, …): 12 points. Seed 1 is the overfitting lesson's default sample. */
export const trainSet = (k: number) => sample(12, k);
/** Validation set k: 12 further points from the same source, never trained on. */
export const valSet = (k: number) => sample(12, 50_000 + k);

// ---------------------------------------------------------------- polynomial ridge regression (exact)

export function legendre(x: number, d: number): number[] {
  const p = [1];
  if (d >= 1) p.push(x);
  for (let k = 2; k <= d; k++) p.push(((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k);
  return p;
}

/**
 * Minimise mean squared error + λ·Σ c² (all coefficients but the constant): solve the normal
 * equations (AᵀA/n + λD) c = Aᵀy/n by Gaussian elimination. λ = 0 is ordinary least squares.
 */
export function ridgeFit(s: Sample, degree: number, lambda: number): number[] {
  const n = s.x.length;
  const k = degree + 1;
  const A = s.x.map((x) => legendre(x, degree));
  const M = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => A.reduce((t, r) => t + r[i] * r[j], 0) / n + (i === j && i > 0 ? lambda : 0)));
  const b = Array.from({ length: k }, (_, i) => A.reduce((t, r, q) => t + r[i] * s.y[q], 0) / n);
  // Gaussian elimination with partial pivoting
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < k; r++) {
      const f = M[r][c] / M[c][c];
      for (let t = c; t < k; t++) M[r][t] -= f * M[c][t];
      b[r] -= f * b[c];
    }
  }
  const c = new Array(k).fill(0);
  for (let i = k - 1; i >= 0; i--) {
    let t = b[i];
    for (let j = i + 1; j < k; j++) t -= M[i][j] * c[j];
    c[i] = t / M[i][i];
  }
  return c;
}

export const evalCoef = (c: number[], x: number) => legendre(x, c.length - 1).reduce((t, v, i) => t + v * c[i], 0);
export const coefMse = (c: number[], s: Sample) => s.x.reduce((t, x, i) => t + (evalCoef(c, x) - s.y[i]) ** 2, 0) / s.x.length;

// ---------------------------------------------------------------- the network and its cures

export const SIZES = [1, 32, 32, 1];
export const LR = 0.01;
export const MAX_STEPS = 4000;

export type Cure = { l2: number; dropout: number };
export const NO_CURE: Cure = { l2: 0, dropout: 0 };

const mseOn = (net: MLP, s: Sample) => s.x.reduce((t, x, i) => t + (net.predict([x])[0] - s.y[i]) ** 2, 0) / s.x.length;

/** A network being trained on one training set, with optional L2 and dropout. Full-batch Adam. */
export class Run {
  readonly net: MLP;
  private opt: ReturnType<typeof createOptimizer>;
  private X: number[][];
  private Y: number[][];
  steps = 0;
  readonly train: Sample;
  readonly cure: Cure;
  constructor(train: Sample, cure: Cure = NO_CURE, seed = 1) {
    this.train = train;
    this.cure = cure;
    this.net = new MLP(SIZES, ['tanh', 'tanh', 'linear'], createRng(seed), 'xavier');
    if (cure.dropout > 0) this.net.setDropout(cure.dropout, createRng(seed * 977 + 3));
    this.opt = createOptimizer({ name: 'adam', lr: LR });
    this.X = train.x.map((x) => [x]);
    this.Y = train.y.map((y) => [y]);
  }
  step(k = 1) {
    for (let i = 0; i < k; i++) {
      this.net.backprop(this.X, this.Y, 'mse', this.cure.l2);
      this.opt.step(this.net.params(), this.net.grads());
      this.steps++;
    }
  }
  predict = (x: number) => this.net.predict([x])[0];
  errorOn = (s: Sample) => mseOn(this.net, s);
  /** Sum of squared weights (biases excluded): what L2 penalises. */
  weightNorm2() {
    let t = 0;
    for (const L of this.net.W) for (const r of L) for (const w of r) t += w * w;
    return t;
  }
  snapshot() {
    return { W: this.net.W.map((L) => L.map((r) => r.slice())), b: this.net.b.map((r) => r.slice()) };
  }
}

export type Outcome = { val: number; test: number; stoppedAt: number };

/**
 * Train with a cure for MAX_STEPS and report the final model's validation and test error.
 * With `patience`, do early stopping instead: check the validation error every `every` steps,
 * remember the weights with the best validation error, and stop once it hasn't improved for
 * `patience` steps; the remembered weights are the result.
 */
export function trainAndScore(k: number, cure: Cure, patience: number | null = null, every = 10, seed = k): Outcome {
  const tr = trainSet(k);
  const va = valSet(k);
  const r = new Run(tr, cure, seed);
  if (patience === null) {
    r.step(MAX_STEPS);
    return { val: r.errorOn(va), test: r.errorOn(TEST_SET), stoppedAt: MAX_STEPS };
  }
  let best = r.errorOn(va);
  let bestAt = 0;
  let snap = r.snapshot();
  while (r.steps < MAX_STEPS) {
    r.step(every);
    const v = r.errorOn(va);
    if (v < best) {
      best = v;
      bestAt = r.steps;
      snap = r.snapshot();
    } else if (r.steps - bestAt >= patience) break;
  }
  const stoppedAt = r.steps;
  r.net.W = snap.W;
  r.net.b = snap.b;
  return { val: best, test: r.errorOn(TEST_SET), stoppedAt };
}
