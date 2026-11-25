// authored-by: Gemini Antigravity
// Pure TypeScript engine for Module 11 Lesson 1: Recurrent Neural Networks & BPTT.
// Zero external ML dependencies — exact Elman RNN forward pass, exact Backpropagation
// Through Time (BPTT) with optional truncation horizon K and global gradient-norm
// clipping, exact Jacobian spectral-radius flow, Pascanu-style exploding cliff, and
// exhaustive Lab: Tame BPTT.

import { createRng } from '../../lib/rng';

export type RnnAct = 'tanh' | 'linear' | 'relu';

export type RnnParams = {
  /** Hidden dimension H */
  H: number;
  /** Input-to-hidden weights (H × 1, flat length H) */
  Wx: number[];
  /** Hidden-to-hidden recurrent weights (H × H, row-major flat length H*H) */
  Wh: number[];
  /** Hidden bias (length H) */
  bh: number[];
  /** Readout weights (1 × H, flat length H) */
  Wy: number[];
  /** Readout bias (scalar) */
  by: number;
};

export type RnnGrads = {
  dWx: number[];
  dWh: number[];
  dbh: number[];
  dWy: number[];
  dby: number;
  /** Exact upstream gradient dL/dh_t for each t = 0..T-1 (length T, each H-vector) */
  dhTrace: number[][];
  /** Global L2 norm of all parameter gradients before clipping */
  gradNorm: number;
};

export type RnnForwardTrace = {
  /** Input sequence x_0 .. x_{T-1} */
  xs: number[];
  /** Pre-activations a_t = Wx*x_t + Wh*h_{t-1} + bh (length T, each H-vector) */
  as: number[][];
  /** Hidden states h_t = act(a_t) (length T, each H-vector); h_{-1} is zeros */
  hs: number[][];
  /** Readout logit z_t = Wy·h_t + by at every step t */
  logits: number[];
  /** Readout probability p_t = sigmoid(z_t) at every step t */
  probs: number[];
};

export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function applyAct(a: number, act: RnnAct): number {
  if (act === 'tanh') return Math.tanh(a);
  if (act === 'relu') return a > 0 ? a : 0;
  return a;
}

function applyActDeriv(a: number, h: number, act: RnnAct): number {
  if (act === 'tanh') return 1 - h * h;
  if (act === 'relu') return a > 0 ? 1 : 0;
  return 1;
}

/** Forward pass of a scalar-input, scalar-binary-output Elman RNN over sequence xs. */
export function rnnForward(params: RnnParams, xs: number[], act: RnnAct = 'tanh'): RnnForwardTrace {
  const { H, Wx, Wh, bh, Wy, by } = params;
  const T = xs.length;
  const as: number[][] = [];
  const hs: number[][] = [];
  const logits: number[] = [];
  const probs: number[] = [];

  let hPrev = new Array<number>(H).fill(0);
  for (let t = 0; t < T; t++) {
    const a = new Array<number>(H);
    const h = new Array<number>(H);
    const xt = xs[t];
    for (let i = 0; i < H; i++) {
      let s = Wx[i] * xt + bh[i];
      const rowOff = i * H;
      for (let j = 0; j < H; j++) {
        s += Wh[rowOff + j] * hPrev[j];
      }
      a[i] = s;
      h[i] = applyAct(s, act);
    }
    let logit = by;
    for (let i = 0; i < H; i++) logit += Wy[i] * h[i];
    as.push(a);
    hs.push(h);
    logits.push(logit);
    probs.push(sigmoid(logit));
    hPrev = h;
  }
  return { xs, as, hs, logits, probs };
}

/**
 * Exact Backpropagation Through Time (BPTT) for binary cross-entropy loss at the
 * final time step T-1: L = -[y log p + (1-y) log(1-p)].
 * If truncateK is provided (1 <= truncateK < T), gradient flow stops after
 * propagating through the last truncateK time steps (steps t < T - truncateK get 0).
 */
export function rnnLossAndGrad(
  params: RnnParams,
  xs: number[],
  yTarget: 0 | 1,
  opts: { act?: RnnAct; truncateK?: number } = {},
): { loss: number; trace: RnnForwardTrace; grads: RnnGrads } {
  const act = opts.act ?? 'tanh';
  const trace = rnnForward(params, xs, act);
  const { H, Wh, Wy } = params;
  const T = xs.length;
  const lastLogit = trace.logits[T - 1];
  const lastProb = trace.probs[T - 1];

  // Numerically stable BCE: max(z, 0) - z*y + log(1 + exp(-|z|))
  const loss = Math.max(lastLogit, 0) - lastLogit * yTarget + Math.log1p(Math.exp(-Math.abs(lastLogit)));

  const dWx = new Array<number>(H).fill(0);
  const dWh = new Array<number>(H * H).fill(0);
  const dbh = new Array<number>(H).fill(0);
  const dWy = new Array<number>(H).fill(0);
  const dhTrace: number[][] = Array.from({ length: T }, () => new Array<number>(H).fill(0));

  const dz = lastProb - yTarget;
  const dby = dz;
  const hLast = trace.hs[T - 1];
  for (let i = 0; i < H; i++) {
    dWy[i] = dz * hLast[i];
  }

  let dhNext = new Array<number>(H);
  for (let i = 0; i < H; i++) dhNext[i] = dz * Wy[i];

  const minT = opts.truncateK !== undefined ? Math.max(0, T - opts.truncateK) : 0;

  for (let t = T - 1; t >= minT; t--) {
    dhTrace[t] = dhNext.slice();
    const at = trace.as[t];
    const ht = trace.hs[t];
    const hPrev = t > 0 ? trace.hs[t - 1] : null;
    const da = new Array<number>(H);
    for (let i = 0; i < H; i++) {
      const deriv = applyActDeriv(at[i], ht[i], act);
      da[i] = dhNext[i] * deriv;
      dWx[i] += da[i] * xs[t];
      dbh[i] += da[i];
    }
    if (hPrev) {
      for (let i = 0; i < H; i++) {
        const rowOff = i * H;
        for (let j = 0; j < H; j++) {
          dWh[rowOff + j] += da[i] * hPrev[j];
        }
      }
    }
    const dhPrev = new Array<number>(H).fill(0);
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let i = 0; i < H; i++) {
        s += Wh[i * H + j] * da[i];
      }
      dhPrev[j] = s;
    }
    dhNext = dhPrev;
  }

  let sq = dby * dby;
  for (let i = 0; i < H; i++) sq += dWx[i] * dWx[i] + dbh[i] * dbh[i] + dWy[i] * dWy[i];
  for (let i = 0; i < H * H; i++) sq += dWh[i] * dWh[i];

  return {
    loss,
    trace,
    grads: { dWx, dWh, dbh, dWy, dby, dhTrace, gradNorm: Math.sqrt(sq) },
  };
}

// ------------------------------------------------------------------ 1. Unrolled Stepper Demo

export type StepperPreset = 'leaky' | 'long';

export function getStepperParams(preset: StepperPreset): RnnParams {
  if (preset === 'leaky') {
    // Fast-forgetting recurrent weight Wh = 0.55: early pulses fade out quickly
    return {
      H: 1,
      Wx: [0.9],
      Wh: [0.55],
      bh: [0.0],
      Wy: [2.5],
      by: 0.0,
    };
  }
  // Long-memory recurrent weight Wh = 0.95: keeps a running signed tally across all 6 steps
  return {
    H: 1,
    Wx: [0.45],
    Wh: [0.95],
    bh: [0.0],
    Wy: [3.0],
    by: 0.0,
  };
}

export const DEFAULT_STEPPER_SEQ: number[] = [1, 0, 0, -1, 0, 0];

// ------------------------------------------------------------------ 2. Exact Jacobian Spectral-Radius Flow

export type GradientFlowPoint = {
  /** Time step index t = 1..T (1 is earliest input, T is final output) */
  step: number;
  /** Steps back from final step (T - step) */
  stepsBack: number;
  /** Hidden state norm ||h_t|| */
  hNorm: number;
  /** Exact upstream Jacobian norm ||dh_T / dh_t|| (1.0 at t = T) */
  gradRatio: number;
};

/**
 * Measure exact BPTT Jacobian sensitivity ||d(v^T h_T) / dh_t|| for a unit probe vector
 * v = [1, 0] at step T, across a chain of length T using a 2D orthogonal rotation matrix
 * scaled by exact spectral radius `rho` (every singular value of Wh equals `rho`).
 * For linear activation this equals rho^(T - t) exactly; for tanh each step multiplies by
 * diag(1 - h_t^2) Wh^T, showing BOTH saturation-driven vanishing and small-signal explosion.
 */
export function spectralGradientFlow(
  rho: number,
  T: number,
  act: 'tanh' | 'linear',
  inputScale = 0.2,
): { points: GradientFlowPoint[]; step1Ratio: number; maxRatio: number; minRatio: number } {
  const theta = 0.65;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const params: RnnParams = {
    H: 2,
    Wx: [inputScale, 0.5 * inputScale],
    Wh: [rho * c, -rho * s, rho * s, rho * c],
    bh: [0, 0],
    Wy: [1, 0],
    by: 0,
  };
  const rng = createRng(17);
  const xs = Array.from({ length: T }, () => rng.next() * 2 - 1);
  const trace = rnnForward(params, xs, act);

  // Backprop unit probe vector v = [1, 0] directly from h_{T-1} (no sigmoid readout saturation)
  let dh = [1, 0];
  const dhNorms = new Array<number>(T);
  for (let t = T - 1; t >= 0; t--) {
    dhNorms[t] = Math.hypot(dh[0], dh[1]);
    if (t > 0) {
      const ht = trace.hs[t];
      const at = trace.as[t];
      const da0 = dh[0] * applyActDeriv(at[0], ht[0], act);
      const da1 = dh[1] * applyActDeriv(at[1], ht[1], act);
      const next0 = params.Wh[0] * da0 + params.Wh[2] * da1;
      const next1 = params.Wh[1] * da0 + params.Wh[3] * da1;
      dh = [next0, next1];
    }
  }

  const points: GradientFlowPoint[] = [];
  let maxRatio = 0;
  let minRatio = Infinity;
  for (let t = 0; t < T; t++) {
    const r = dhNorms[t];
    if (r > maxRatio) maxRatio = r;
    if (r < minRatio) minRatio = r;
    points.push({
      step: t + 1,
      stepsBack: T - 1 - t,
      hNorm: Math.hypot(trace.hs[t][0], trace.hs[t][1]),
      gradRatio: r,
    });
  }
  return { points, step1Ratio: points[0].gradRatio, maxRatio, minRatio };
}

// ------------------------------------------------------------------ 3. Delayed-Recall Dataset & Real BPTT Trainer

export type SeqExample = {
  xs: number[];
  y: 0 | 1;
};

/**
 * Delayed-recall benchmark:
 * x_0 in {-1, +1} encodes target y = (x_0 > 0 ? 1 : 0).
 * Followed by `lag` distractor steps drawn uniformly in [-noiseAmp, +noiseAmp].
 * Total sequence length T = lag + 1.
 */
export function makeDelayedRecallDataset(
  lag: number,
  count: number,
  seed: number,
  noiseAmp = 0.65,
): SeqExample[] {
  const rng = createRng(seed);
  const out: SeqExample[] = [];
  for (let n = 0; n < count; n++) {
    const y: 0 | 1 = n % 2 === 0 ? 1 : 0;
    const xs = new Array<number>(lag + 1);
    xs[0] = y === 1 ? 1.0 : -1.0;
    for (let t = 1; t <= lag; t++) {
      xs[t] = (rng.next() * 2 - 1) * noiseAmp;
    }
    out.push({ xs, y });
  }
  return out;
}

/** Initialise an H-unit Elman RNN with block-orthogonal Wh of exact spectral radius rho. */
export function initRnn(H: number, rho: number, seed: number, wxScale = 0.4): RnnParams {
  const rng = createRng(seed);
  const Wx = Array.from({ length: H }, () => rng.normal(0, wxScale));
  const bh = new Array<number>(H).fill(0);
  const Wy = Array.from({ length: H }, () => rng.normal(0, 0.4));
  const by = 0;

  const Wh = new Array<number>(H * H).fill(0);
  for (let b = 0; b + 1 < H; b += 2) {
    const ang = rng.next() * Math.PI * 2;
    const ca = Math.cos(ang) * rho;
    const sa = Math.sin(ang) * rho;
    Wh[b * H + b] = ca;
    Wh[b * H + (b + 1)] = -sa;
    Wh[(b + 1) * H + b] = sa;
    Wh[(b + 1) * H + (b + 1)] = ca;
  }
  if (H % 2 === 1) {
    Wh[(H - 1) * H + (H - 1)] = rho;
  }
  return { H, Wx, Wh, bh, Wy, by };
}

export type TrainRnnOpts = {
  H: number;
  rho: number;
  lr: number;
  epochs: number;
  clipNorm: number | null;
  truncateK?: number;
  act?: RnnAct;
  optimizer?: 'sgd' | 'adam';
  noiseAmp?: number;
  wxScale?: number;
  seed?: number;
};

export type TrainRnnResult = {
  params: RnnParams;
  lossCurve: number[];
  gradNormCurve: number[];
  trainAcc: number;
  testAcc: number;
  finalLoss: number;
  exploded: boolean;
  maxGradNorm: number;
};

function flattenParams(p: RnnParams): number[] {
  return [...p.Wx, ...p.Wh, ...p.bh, ...p.Wy, p.by];
}

function unflattenParams(H: number, vec: number[]): RnnParams {
  let k = 0;
  const Wx = vec.slice(k, k + H); k += H;
  const Wh = vec.slice(k, k + H * H); k += H * H;
  const bh = vec.slice(k, k + H); k += H;
  const Wy = vec.slice(k, k + H); k += H;
  const by = vec[k];
  return { H, Wx, Wh, bh, Wy, by };
}

export function evalRnnAcc(params: RnnParams, data: SeqExample[], act: RnnAct = 'tanh'): number {
  let ok = 0;
  for (const ex of data) {
    const tr = rnnForward(params, ex.xs, act);
    const pred = tr.probs[ex.xs.length - 1] >= 0.5 ? 1 : 0;
    if (pred === ex.y) ok++;
  }
  return ok / data.length;
}

/** Train an Elman RNN with exact BPTT on a delayed-recall dataset. */
export function trainDelayedRecall(
  lag: number,
  opts: TrainRnnOpts,
): TrainRnnResult {
  const seed = opts.seed ?? 42;
  const act = opts.act ?? 'tanh';
  const optimizer = opts.optimizer ?? 'adam';
  const noiseAmp = opts.noiseAmp ?? 0.65;
  const trainData = makeDelayedRecallDataset(lag, 40, seed + 101, noiseAmp);
  const testData = makeDelayedRecallDataset(lag, 40, seed + 909, noiseAmp);

  let params = initRnn(opts.H, opts.rho, seed, opts.wxScale ?? 0.4);
  let theta = flattenParams(params);
  const P = theta.length;
  const m = new Array<number>(P).fill(0);
  const v = new Array<number>(P).fill(0);

  const lossCurve: number[] = [];
  const gradNormCurve: number[] = [];
  let exploded = false;
  let maxGradNorm = 0;

  for (let ep = 1; ep <= opts.epochs; ep++) {
    const gSum = new Array<number>(P).fill(0);
    let lossSum = 0;
    for (const ex of trainData) {
      const { loss, grads } = rnnLossAndGrad(params, ex.xs, ex.y, {
        act,
        truncateK: opts.truncateK,
      });
      lossSum += loss;
      let k = 0;
      for (let i = 0; i < opts.H; i++) gSum[k++] += grads.dWx[i];
      for (let i = 0; i < opts.H * opts.H; i++) gSum[k++] += grads.dWh[i];
      for (let i = 0; i < opts.H; i++) gSum[k++] += grads.dbh[i];
      for (let i = 0; i < opts.H; i++) gSum[k++] += grads.dWy[i];
      gSum[k++] += grads.dby;
    }

    const meanLoss = lossSum / trainData.length;
    const invN = 1 / trainData.length;
    let sq = 0;
    for (let i = 0; i < P; i++) {
      gSum[i] *= invN;
      sq += gSum[i] * gSum[i];
    }
    const gNorm = Math.sqrt(sq);
    if (!Number.isFinite(meanLoss) || !Number.isFinite(gNorm) || meanLoss > 1e3 || gNorm > 1e6) {
      exploded = true;
      lossCurve.push(10);
      gradNormCurve.push(1e6);
      break;
    }
    if (gNorm > maxGradNorm) maxGradNorm = gNorm;
    lossCurve.push(meanLoss);
    gradNormCurve.push(gNorm);

    // Optional global L2 gradient-norm clipping
    if (opts.clipNorm !== null && gNorm > opts.clipNorm) {
      const scale = opts.clipNorm / gNorm;
      for (let i = 0; i < P; i++) gSum[i] *= scale;
    }

    if (optimizer === 'sgd') {
      for (let i = 0; i < P; i++) theta[i] -= opts.lr * gSum[i];
    } else {
      const b1 = 0.9;
      const b2 = 0.999;
      const eps = 1e-8;
      const b1c = 1 - b1 ** ep;
      const b2c = 1 - b2 ** ep;
      for (let i = 0; i < P; i++) {
        m[i] = b1 * m[i] + (1 - b1) * gSum[i];
        v[i] = b2 * v[i] + (1 - b2) * gSum[i] * gSum[i];
        const mHat = m[i] / b1c;
        const vHat = v[i] / b2c;
        theta[i] -= opts.lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
    params = unflattenParams(opts.H, theta);
  }

  const finalLoss = exploded ? 10 : lossCurve[lossCurve.length - 1];
  const trainAcc = exploded ? 0.5 : evalRnnAcc(params, trainData, act);
  const testAcc = exploded ? 0.5 : evalRnnAcc(params, testData, act);
  return { params, lossCurve, gradNormCurve, trainAcc, testAcc, finalLoss, exploded, maxGradNorm };
}

// ------------------------------------------------------------------ 4. Pascanu Exploding-Cliff Showcase

/**
 * Exact 1D recurrent exploding-cliff demonstration (Pascanu, Mikolov & Bengio 2013):
 * A 1-unit linear/ReLU-like recurrent cell predicting y = x_0 after T = 16 steps:
 *   h_t = w * h_{t-1} + 0.05 * x_t   (h_0 = 0),   loss = 0.5 * (h_T - target)^2
 * Starting at w = 1.25 (just above 1.0), h_T = 0.05 * w^15 ≈ 1.42, while the slope
 * dL/dw = (h_T - target) * 15 * 0.05 * w^14 is HUGE (~16–100+)!
 * Unclipped SGD catapults w off the cliff to overflow; gradient-norm clipping caps the
 * step and glides smoothly to the exact target w*!
 */
export type CliffPoint = {
  step: number;
  w: number;
  loss: number;
  rawGrad: number;
  clippedGrad: number;
};

export function runPascanuCliff(
  lr: number,
  clipNorm: number | null,
  steps = 40,
  wInit = 1.28,
): {
  trace: CliffPoint[];
  finalLoss: number;
  finalW: number;
  maxRawGrad: number;
  exploded: boolean;
} {
  const T = 16; // exponent is T-1 = 15
  const x0 = 0.05;
  const target = 0.65; // exact root w* = (0.65 / 0.05)^(1/15) = 13^(1/15) ≈ 1.1865

  let w = wInit;
  const trace: CliffPoint[] = [];
  let maxRawGrad = 0;
  let exploded = false;

  for (let s = 0; s < steps; s++) {
    const pow14 = w ** (T - 2);
    const pow15 = pow14 * w;
    const hT = x0 * pow15;
    const err = hT - target;
    const loss = 0.5 * err * err;
    const rawGrad = err * x0 * (T - 1) * pow14;

    if (!Number.isFinite(loss) || !Number.isFinite(rawGrad) || Math.abs(w) > 1e4 || loss > 1e6) {
      exploded = true;
      trace.push({ step: s, w: Math.sign(w) * 1e4, loss: 1e6, rawGrad: 1e6, clippedGrad: 1e6 });
      break;
    }

    const absG = Math.abs(rawGrad);
    if (absG > maxRawGrad) maxRawGrad = absG;
    const clippedGrad = clipNorm !== null && absG > clipNorm ? Math.sign(rawGrad) * clipNorm : rawGrad;

    trace.push({ step: s, w, loss, rawGrad, clippedGrad });
    w -= lr * clippedGrad;
  }

  const last = trace[trace.length - 1];
  return {
    trace,
    finalLoss: exploded ? 1e6 : last.loss,
    finalW: last.w,
    maxRawGrad,
    exploded,
  };
}

// ------------------------------------------------------------------ 5. Lag-Degradation Benchmark (Multi-Seed)

export const LAG_BENCH_LAGS = [1, 3, 6, 10, 15] as const;

export function runLagBenchmark(lag: number, seeds = [11, 23, 37]) {
  let accSum = 0;
  let lossSum = 0;
  for (const s of seeds) {
    const res = trainDelayedRecall(lag, {
      H: 4,
      rho: 0.65,
      lr: 0.08,
      epochs: 120,
      clipNorm: 1.0,
      noiseAmp: 1.0,
      optimizer: 'sgd',
      seed: s,
    });
    accSum += res.testAcc;
    lossSum += res.finalLoss;
  }
  return {
    lag,
    meanAcc: accSum / seeds.length,
    meanLoss: lossSum / seeds.length,
  };
}

// ------------------------------------------------------------------ 6. Lab: Tame BPTT (Exhaustive)

export type RnnBriefId = 'flow' | 'clip' | 'tbptt';

export const LAB_FLOW_RHOS = [0.5, 0.8, 1.0, 1.25, 1.6] as const;
export const LAB_FLOW_ACTS = ['tanh', 'linear'] as const;

export function evalRnnBrief1(rho: number, act: 'tanh' | 'linear') {
  const res = spectralGradientFlow(rho, 20, act, 0.2);
  const ratio = res.step1Ratio;
  const pass = ratio >= 0.1 && ratio <= 10.0;
  const logDist = Math.abs(Math.log10(Math.max(1e-30, ratio)));
  return { rho, act, ratio, logDist, pass };
}

export const LAB_CLIP_LRS = [0.015, 0.025, 0.05] as const;
export const LAB_CLIP_NORMS: (number | null)[] = [null, 0.5, 1.0, 5.0];

export function evalRnnBrief2(lr: number, clipNorm: number | null) {
  const res = runPascanuCliff(lr, clipNorm, 45, 1.32);
  const pass = !res.exploded && res.finalLoss <= 1e-4;
  return {
    lr,
    clipNorm,
    finalLoss: res.finalLoss,
    finalW: res.finalW,
    maxRawGrad: res.maxRawGrad,
    exploded: res.exploded,
    pass,
  };
}

export const LAB_TBPTT_KS = [2, 4, 6, 7, 10] as const;
export const LAB_TBPTT_HS = [2, 4] as const;

export function evalRnnBrief3(truncateK: number, H: number) {
  // Lag = 6 -> sequence length T = 7 (x_0 at t=0, readout at t=6).
  // Truncated BPTT backprops through the last K steps; K < 7 never reaches t = 0!
  const res = trainDelayedRecall(6, {
    H,
    rho: 0.60,
    lr: 0.12,
    epochs: 140,
    clipNorm: 1.0,
    truncateK,
    noiseAmp: 0.9,
    optimizer: 'sgd',
    seed: 19,
  });
  const memSlots = truncateK * H;
  const reachesOrigin = truncateK >= 7;
  const pass = res.testAcc >= 0.95 && res.finalLoss <= 0.10 && memSlots <= 20;
  return {
    truncateK,
    H,
    memSlots,
    reachesOrigin,
    testAcc: res.testAcc,
    finalLoss: res.finalLoss,
    pass,
  };
}

export function rnnLabExhaustive(brief: RnnBriefId) {
  if (brief === 'flow') {
    const all: ReturnType<typeof evalRnnBrief1>[] = [];
    for (const act of LAB_FLOW_ACTS) {
      for (const rho of LAB_FLOW_RHOS) {
        all.push(evalRnnBrief1(rho, act));
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => a.logDist - b.logDist);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0] ? `${passing[0].act}, ρ=${passing[0].rho} (ratio ${passing[0].ratio.toFixed(3)})` : 'none',
    };
  }
  if (brief === 'clip') {
    const all: ReturnType<typeof evalRnnBrief2>[] = [];
    for (const lr of LAB_CLIP_LRS) {
      for (const clip of LAB_CLIP_NORMS) {
        all.push(evalRnnBrief2(lr, clip));
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => a.finalLoss - b.finalLoss);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0]
        ? `lr=${passing[0].lr}, clip=${passing[0].clipNorm ?? 'none'} (loss ${passing[0].finalLoss.toExponential(1)})`
        : 'none',
    };
  }
  const all: ReturnType<typeof evalRnnBrief3>[] = [];
  for (const K of LAB_TBPTT_KS) {
    for (const H of LAB_TBPTT_HS) {
      all.push(evalRnnBrief3(K, H));
    }
  }
  const passing = all.filter((r) => r.pass).sort((a, b) => a.memSlots - b.memSlots || a.finalLoss - b.finalLoss);
  return {
    total: all.length,
    passingCount: passing.length,
    bestLabel: passing[0]
      ? `K=${passing[0].truncateK}, H=${passing[0].H} (${passing[0].memSlots} slots, ${(100 * passing[0].testAcc).toFixed(0)}%)`
      : 'none',
  };
}

export function rnnLabHint(
  brief: RnnBriefId,
  state: {
    b1Act: 'tanh' | 'linear';
    b1Rho: number;
    b2Lr: number;
    b2Clip: number | null;
    b3K: number;
    b3H: number;
  },
): string {
  if (brief === 'flow') {
    const r = evalRnnBrief1(state.b1Rho, state.b1Act);
    if (r.pass) {
      return `Step-1 Jacobian ratio is ${r.ratio.toExponential(2)} — comfortably inside [0.10, 10.0] across all 20 steps.`;
    }
    if (r.ratio < 0.1) {
      return `The gradient shrinks to ${r.ratio.toExponential(2)} after 19 multiplications because each backward step contracts by less than 1. Watch how the log curve bends when the recurrent scaling is closer to unity.`;
    }
    return `The gradient amplifies to ${r.ratio.toExponential(2)} because repeated multiplications above 1 compound exponentially without saturation damping them.`;
  }
  if (brief === 'clip') {
    const r = evalRnnBrief2(state.b2Lr, state.b2Clip);
    if (r.pass) {
      return `Loss converged to ${r.finalLoss.toExponential(1)} at w = ${r.finalW.toFixed(4)} without jumping off the cliff.`;
    }
    if (r.exploded) {
      return `Raw cliff slope hits ${r.maxRawGrad.toFixed(1)}, so the first step overshoots to overflow. Notice how capping the gradient vector norm limits the jump length without changing its direction.`;
    }
    if (state.b2Clip === null) {
      return `Raw cliff slope (${r.maxRawGrad.toFixed(1)}) catapults the weight across zero into the flat desert (w = ${r.finalW.toFixed(3)}, loss ${r.finalLoss.toFixed(3)}), where the slope nearly vanishes. Look at what happens when the step norm is bounded.`;
    }
    return `Current settings stall at loss ${r.finalLoss.toExponential(2)} (w = ${r.finalW.toFixed(4)}) after 45 steps — either the effective step is too small to reach the valley in time, or it still overshoots the cliff wall.`;
  }
  const r = evalRnnBrief3(state.b3K, state.b3H);
  if (r.pass) {
    return `All 7 steps reach t = 0 (${(100 * r.testAcc).toFixed(0)}% acc, loss ${r.finalLoss.toFixed(3)}) within ${r.memSlots} / 20 activation-RAM slots.`;
  }
  if (!r.reachesOrigin) {
    return `Check the tape: the cue arrives at t = 0 and the readout is at t = 6 (7 steps total), so truncating earlier cuts the backward wire before it can ever deliver credit to t = 0 (loss ${r.finalLoss.toFixed(3)}).`;
  }
  if (r.memSlots > 20) {
    return `The backward pass reaches t = 0 (${(100 * r.testAcc).toFixed(0)}% acc), but storing every state uses ${r.memSlots} slots — over the 20-slot RAM cap. Remember activation memory scales as horizon × width.`;
  }
  return `Test accuracy is ${(100 * r.testAcc).toFixed(0)}% and loss is ${r.finalLoss.toFixed(3)}; check both whether the gradient reaches t = 0 and whether activation memory fits within 20 slots.`;
}

