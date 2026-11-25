// authored-by: Gemini Antigravity
/**
 * Minimum-norm random-features regression for the Epilogue ("What Nobody Understands Yet").
 *
 * Fits y ≈ Σ_{j=1}^p c_j · √(2/p) cos(w_j x + b_j) with frozen random frequencies
 * w_j ~ Normal(0, 20) and phases b_j ~ Uniform(0, 2π) — the random Fourier features
 * model used to study double descent (Belkin et al., 2019).
 *
 * For p ≤ n we solve least squares by Householder QR; for p > n we compute the exact
 * minimum-L2-norm interpolator c = Φᵀ (Φ Φᵀ)⁻¹ y using `src/lib/linalg.ts`.
 */
import { dot, matMul, matVec, transpose, type Mat, type Vec } from '../../lib/linalg';
import { createRng } from '../../lib/rng';

export const N_TRAIN = 24;
export const DEFAULT_NOISE = 0.25;
export const FREQ_STD = 20;
export const P_VALUES = [2, 4, 8, 12, 16, 20, 22, 23, 24, 25, 26, 30, 40, 60, 90, 120, 160] as const;

/** Smooth 1D ground-truth function on [-1, 1]. */
export const targetFn = (x: number): number => Math.sin(2.4 * x) + 0.35 * x;

export interface Dataset {
  x: number[];
  y: number[];
}

export interface RandomFeature {
  freq: number;
  phase: number;
}

export interface FitResult {
  p: number;
  features: RandomFeature[];
  coef: Vec;
  trainMse: number;
  testMse: number;
  weightNorm: number;
}

export interface CurvePoint {
  p: number;
  trainMse: number;
  testMse: number;
  weightNorm: number;
}

/** Generate a deterministic stratified training set on [-1, 1]. */
export function makeTrainingSet(noise = DEFAULT_NOISE, seed = 101, n = N_TRAIN): Dataset {
  const rng = createRng(seed);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = -0.96 + (1.92 * (i + 0.15 + 0.7 * rng.next())) / n;
    x.push(xi);
    y.push(targetFn(xi) + (noise > 0 ? rng.normal(0, noise) : 0));
  }
  return { x, y };
}

/** Dense evaluation grid on [-0.95, 0.95] for clean noiseless test MSE (excess risk). */
export const EVAL_GRID: number[] = Array.from({ length: 191 }, (_, i) => -0.95 + (1.9 * i) / 190);
export const EVAL_TRUE: number[] = EVAL_GRID.map(targetFn);

/** Generate p nested random Fourier features (first p of a fixed seeded bank so moving p adds features). */
export function drawFeatures(p: number, featureSeed: number): RandomFeature[] {
  const rng = createRng(featureSeed * 9973 + 37);
  const feats: RandomFeature[] = [];
  for (let j = 0; j < p; j++) {
    const freq = rng.normal(0, FREQ_STD);
    const phase = rng.next() * 2 * Math.PI;
    feats.push({ freq, phase });
  }
  return feats;
}

/** Design matrix Φ (n × p) where Φ[i][j] = √(2/p) cos(w_j x_i + b_j). */
export function designMatrix(xs: number[], features: RandomFeature[]): Mat {
  const scale = Math.sqrt(2 / features.length);
  return xs.map((x) => features.map((f) => Math.cos(f.freq * x + f.phase) * scale));
}

/** Solve symmetric positive (semi)definite G a = y via Cholesky with tiny jitter. */
function solveCholesky(G: Mat, y: Vec, jitter = 1e-13): Vec {
  const n = G.length;
  const L: Mat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = G[i][j] + (i === j ? jitter : 0);
      for (let t = 0; t < j; t++) s -= L[i][t] * L[j][t];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
  }
  const z = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = y[i];
    for (let t = 0; t < i; t++) s -= L[i][t] * z[t];
    z[i] = s / L[i][i];
  }
  const a = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i];
    for (let t = i + 1; t < n; t++) s -= L[t][i] * a[t];
    a[i] = s / L[i][i];
  }
  return a;
}

/**
 * Minimum-norm least squares for A c ≈ y:
 * - Over-determined or square (m ≥ p): Householder QR (minimises ‖A c − y‖²).
 * - Under-determined (m < p): minimum-L2-norm interpolator c = Aᵀ (A Aᵀ)⁻¹ y.
 */
export function minNormLstsq(A: Mat, y: Vec): Vec {
  const m = A.length;
  const p = A[0].length;
  if (m < p) {
    const At = transpose(A);
    const G = matMul(A, At);
    const alpha = solveCholesky(G, y, 1e-13);
    return matVec(At, alpha);
  }
  const R = A.map((row) => row.slice());
  const b = y.slice();
  for (let j = 0; j < p; j++) {
    let normSq = 0;
    for (let i = j; i < m; i++) normSq += R[i][j] * R[i][j];
    const colNorm = Math.sqrt(normSq);
    if (colNorm < 1e-14) continue;
    const alpha = R[j][j] >= 0 ? -colNorm : colNorm;
    const v = new Array(m).fill(0);
    for (let i = j; i < m; i++) v[i] = R[i][j];
    v[j] -= alpha;
    let vv = 0;
    for (let i = j; i < m; i++) vv += v[i] * v[i];
    if (vv < 1e-30) continue;
    for (let col = j; col < p; col++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i] * R[i][col];
      const f = (2 * s) / vv;
      for (let i = j; i < m; i++) R[i][col] -= f * v[i];
    }
    let sb = 0;
    for (let i = j; i < m; i++) sb += v[i] * b[i];
    const fb = (2 * sb) / vv;
    for (let i = j; i < m; i++) b[i] -= fb * v[i];
  }
  const c = new Array(p).fill(0);
  for (let j = p - 1; j >= 0; j--) {
    let s = b[j];
    for (let t = j + 1; t < p; t++) s -= R[j][t] * c[t];
    c[j] = Math.abs(R[j][j]) < 1e-13 ? 0 : s / R[j][j];
  }
  return c;
}

export function predictSingle(features: RandomFeature[], coef: Vec, x: number): number {
  const scale = Math.sqrt(2 / features.length);
  let sum = 0;
  for (let j = 0; j < features.length; j++) {
    sum += coef[j] * Math.cos(features[j].freq * x + features[j].phase) * scale;
  }
  return sum;
}

export function mse(preds: number[], targets: number[]): number {
  let s = 0;
  for (let i = 0; i < preds.length; i++) {
    const d = preds[i] - targets[i];
    s += d * d;
  }
  return s / preds.length;
}

/** Fit a single minimum-norm random-features model of width p. */
export function fitRandomFeatures(data: Dataset, p: number, featureSeed = 1): FitResult {
  const features = drawFeatures(p, featureSeed);
  const Phi = designMatrix(data.x, features);
  const coef = minNormLstsq(Phi, data.y);
  const trainPreds = matVec(Phi, coef);
  const testPreds = EVAL_GRID.map((x) => predictSingle(features, coef, x));
  return {
    p,
    features,
    coef,
    trainMse: mse(trainPreds, data.y),
    testMse: mse(testPreds, EVAL_TRUE),
    weightNorm: Math.sqrt(dot(coef, coef)),
  };
}

/** Median helper for robust aggregation across random draws (since near p = n a single near-singular draw can explode). */
function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

/**
 * Compute the double-descent curve across P_VALUES, taking the median over `trials`
 * independent dataset + random-feature draws.
 */
export function doubleDescentCurve(noise = DEFAULT_NOISE, trials = 41): CurvePoint[] {
  return P_VALUES.map((p) => {
    const trainArr: number[] = [];
    const testArr: number[] = [];
    const normArr: number[] = [];
    for (let t = 1; t <= trials; t++) {
      const data = makeTrainingSet(noise, 100 + t);
      const res = fitRandomFeatures(data, p, 500 + t);
      trainArr.push(res.trainMse);
      testArr.push(res.testMse);
      normArr.push(res.weightNorm);
    }
    return {
      p,
      trainMse: median(trainArr),
      testMse: median(testArr),
      weightNorm: median(normArr),
    };
  });
}

/** Single fixed showcase dataset + feature draw for the live curve plot (matches median behaviour). */
export const SHOWCASE_NOISY_DATA = makeTrainingSet(DEFAULT_NOISE, 110);
export const SHOWCASE_CLEAN_DATA = makeTrainingSet(0, 110);
export const SHOWCASE_FEATURE_SEED = 510;

export const CURVE_NOISY = doubleDescentCurve(DEFAULT_NOISE, 41);
export const CURVE_CLEAN = doubleDescentCurve(0, 41);
