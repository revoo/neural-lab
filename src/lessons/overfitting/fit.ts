/**
 * Curve fitting for the overfitting lesson: noisy samples of a smooth "true" function, polynomial
 * least-squares fits of any degree, train/test error, and the bias–variance decomposition measured
 * over many independently drawn training sets. Everything is seeded and exact (no training loop):
 * a polynomial fit is a single least-squares solve.
 */
import { createRng } from '../../lib/rng';

/** The hidden pattern the data comes from. Smooth, with enough curvature that a straight line can't follow it. */
export const truth = (x: number) => Math.sin(2.6 * x) + 0.4 * x;
/** Standard deviation of the noise added to each observation. */
export const NOISE = 0.3;
export const X_RANGE: [number, number] = [-1, 1];

export type Sample = { x: number[]; y: number[] };

/**
 * n points spread over [−1, 1] (one at a random position inside each of n equal slices, so the data
 * covers the whole range) with y = truth(x) + noise.
 */
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

/** A large fresh sample from the same source, standing in for "all future data". */
export const TEST_SET = sample(2000, 999_983);
/** A few fresh points to draw as "test data" on the plots (the errors use TEST_SET). */
export const SHOWN_TEST = sample(30, 424_242);

/** Legendre polynomials P0..Pd at x: a well-conditioned basis for polynomials on [−1, 1]. */
export function legendre(x: number, d: number): number[] {
  const p = [1];
  if (d >= 1) p.push(x);
  for (let k = 2; k <= d; k++) p.push(((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k);
  return p;
}

/**
 * Least squares: minimise |A c − y|². Householder QR when there are at least as many rows as
 * columns; otherwise the minimum-norm exact solution c = Aᵀ (A Aᵀ)⁻¹ y.
 */
export function lstsq(A: number[][], y: number[]): number[] {
  const m = A.length;
  const k = A[0].length;
  if (m < k) {
    // minimum-norm solution via the m×m Gram matrix
    const G = A.map((ri) => A.map((rj) => ri.reduce((s, v, t) => s + v * rj[t], 0)));
    const a = solveSym(G, y);
    return Array.from({ length: k }, (_, j) => A.reduce((s, row, i) => s + row[j] * a[i], 0));
  }
  const R = A.map((r) => r.slice());
  const b = y.slice();
  for (let j = 0; j < k; j++) {
    let norm = 0;
    for (let i = j; i < m; i++) norm += R[i][j] ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[j][j] > 0 ? -norm : norm;
    const v = new Array(m).fill(0);
    for (let i = j; i < m; i++) v[i] = R[i][j];
    v[j] -= alpha;
    let vv = 0;
    for (let i = j; i < m; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let c = j; c < k; c++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i] * R[i][c];
      s = (2 * s) / vv;
      for (let i = j; i < m; i++) R[i][c] -= s * v[i];
    }
    let s = 0;
    for (let i = j; i < m; i++) s += v[i] * b[i];
    s = (2 * s) / vv;
    for (let i = j; i < m; i++) b[i] -= s * v[i];
  }
  const c = new Array(k).fill(0);
  for (let j = k - 1; j >= 0; j--) {
    let s = b[j];
    for (let t = j + 1; t < k; t++) s -= R[j][t] * c[t];
    c[j] = Math.abs(R[j][j]) < 1e-14 ? 0 : s / R[j][j];
  }
  return c;
}

/** Solve a symmetric positive (semi)definite system by Cholesky, with a tiny jitter for safety. */
function solveSym(G: number[][], y: number[]): number[] {
  const n = G.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = G[i][j] + (i === j ? 1e-10 : 0);
      for (let t = 0; t < j; t++) s -= L[i][t] * L[j][t];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
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

export type Poly = { degree: number; coef: number[] };

export function fitPoly(s: Sample, degree: number): Poly {
  const A = s.x.map((x) => legendre(x, degree));
  return { degree, coef: lstsq(A, s.y) };
}

export function evalPoly(p: Poly, x: number): number {
  const b = legendre(x, p.degree);
  let s = 0;
  for (let i = 0; i < b.length; i++) s += b[i] * p.coef[i];
  return s;
}

export function mse(p: Poly, s: Sample): number {
  let t = 0;
  for (let i = 0; i < s.x.length; i++) t += (evalPoly(p, s.x[i]) - s.y[i]) ** 2;
  return t / s.x.length;
}

export const trainError = (s: Sample, degree: number) => mse(fitPoly(s, degree), s);
export const testError = (s: Sample, degree: number) => mse(fitPoly(s, degree), TEST_SET);

/** Train and test error of every degree from 0 to maxDegree, on one training sample. */
export function errorCurve(s: Sample, maxDegree: number) {
  return Array.from({ length: maxDegree + 1 }, (_, d) => {
    const p = fitPoly(s, d);
    return { degree: d, train: mse(p, s), test: mse(p, TEST_SET) };
  });
}

/** Points on a uniform grid over [−1, 1], used to average bias and variance over the input range. */
export const GRID = Array.from({ length: 101 }, (_, i) => -1 + (2 * i) / 100);

/**
 * Bias–variance decomposition for polynomials of one degree, measured over `universes`
 * independently drawn training sets of size n (seeds 1..universes).
 *
 * At each grid point x: the average prediction across universes is compared with the truth
 * (bias² = (avg − truth)²) and the spread of predictions around their own average is the variance.
 * Averaged over the grid, bias² + variance + noise² is the expected error on a fresh point.
 */
export function biasVariance(n: number, degree: number, universes = 200) {
  const preds: number[][] = [];
  for (let u = 1; u <= universes; u++) {
    const p = fitPoly(sample(n, u), degree);
    preds.push(GRID.map((x) => evalPoly(p, x)));
  }
  let bias2 = 0;
  let variance = 0;
  const mean = GRID.map((_, g) => preds.reduce((s, row) => s + row[g], 0) / universes);
  GRID.forEach((x, g) => {
    bias2 += (mean[g] - truth(x)) ** 2;
    let v = 0;
    for (const row of preds) v += (row[g] - mean[g]) ** 2;
    variance += v / universes;
  });
  bias2 /= GRID.length;
  variance /= GRID.length;
  // the directly measured expected error on fresh data at the same grid points
  let direct = 0;
  for (const row of preds) GRID.forEach((x, g) => (direct += (row[g] - truth(x)) ** 2));
  direct = direct / (universes * GRID.length) + NOISE * NOISE;
  return { bias2, variance, noise: NOISE * NOISE, total: bias2 + variance + NOISE * NOISE, direct, mean, preds };
}

/** Average test error over `seeds` independent training sets: the fair way to compare degrees. */
export function avgTestError(n: number, degree: number, seeds = 50) {
  let t = 0;
  for (let s = 1; s <= seeds; s++) t += testError(sample(n, s), degree);
  return t / seeds;
}

/**
 * Expected squared error on a fresh point, computed exactly from the known truth: the fit's squared
 * distance from the true curve, averaged over the input range, plus the noise every prediction must
 * suffer. (TEST_SET estimates the same quantity by sampling; this version has no sampling noise.)
 */
export function expectedError(p: Poly): number {
  let t = 0;
  for (const x of GRID) t += (evalPoly(p, x) - truth(x)) ** 2;
  return t / GRID.length + NOISE * NOISE;
}

/**
 * Average expected error of degree-d fits over `seeds` training sets of n points. Degrees at or above
 * n have more coefficients than data points: infinitely many polynomials pass through every point,
 * and lstsq picks the one with the smallest coefficients (minimum norm).
 */
export function avgExpectedError(n: number, degree: number, seeds = 60) {
  let t = 0;
  for (let s = 1; s <= seeds; s++) t += expectedError(fitPoly(sample(n, s), degree));
  return t / seeds;
}
