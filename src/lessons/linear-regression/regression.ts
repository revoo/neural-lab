// authored-by: Gemini Antigravity
/**
 * Classical Linear & Logistic Regression, computed from scratch:
 *  - Exact OLS & Ridge via normal equations (X^T X + lambda I) w = X^T y (Gaussian elimination with pivoting)
 *  - Gradient descent on MSE for direct iteration-count comparison under collinearity
 *  - Exact convex Logistic Regression via Newton-Raphson / IRLS with L2 regularisation (no intercept penalty)
 *  - Exact 1D outlier breakdown demo (OLS vs Logistic Regression)
 *  - Imbalanced fraud dataset with exact confusion matrix, Precision/Recall/F1, ROC curve & AUC
 *  - Lab: Feature Engineering + Regularised Logistic Regression across 3 exhaustive briefs
 */
import { createRng } from '../../lib/rng';

// ------------------------------------------------------------------ linear algebra helpers

/** Solve A x = b (d x d) using Gaussian elimination with partial pivoting. Returns null if singular. */
export function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) return null;
    if (pivot !== col) [M[col], M[pivot]] = [M[pivot], M[col]];
    const diag = M[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / diag;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/** Exact Ridge / OLS fit: X has shape N x d (column 0 is usually 1 for intercept; intercept is unpenalised if penaliseFirst=false). */
export function fitRidge(X: number[][], y: number[], lambda = 0, penaliseFirst = false): number[] {
  const N = X.length;
  const d = X[0].length;
  const XtX = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const Xty = new Array<number>(d).fill(0);
  for (let i = 0; i < N; i++) {
    const row = X[i];
    for (let j = 0; j < d; j++) {
      Xty[j] += row[j] * y[i];
      for (let k = j; k < d; k++) XtX[j][k] += row[j] * row[k];
    }
  }
  for (let j = 0; j < d; j++) {
    for (let k = 0; k < j; k++) XtX[j][k] = XtX[k][j];
    if (j > 0 || penaliseFirst) XtX[j][j] += lambda;
  }
  return solveLinear(XtX, Xty) ?? new Array<number>(d).fill(0);
}

/** Exact 2x2 symmetric eigenvalues and condition number for the two-feature Gram matrix. */
export function conditionNumber2x2(a: number, b: number, c: number): number {
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr - 4 * det));
  const lMax = (tr + disc) / 2;
  const lMin = Math.max(1e-12, (tr - disc) / 2);
  return lMax / lMin;
}

// ------------------------------------------------------------------ 1. Collinearity & Normal Equations vs GD

export type CollinearResult = {
  rho: number;
  lambda: number;
  condNumber: number;
  wOls: [number, number];
  wRidge: [number, number];
  trainMseOls: number;
  testMseOls: number;
  trainMseRidge: number;
  testMseRidge: number;
  gdStepsToTol: number;
  gdFinalDist: number;
};

/**
 * Generate a 2-feature linear regression dataset (zero-mean) where x2 = rho * x1 + sqrt(1 - rho^2) * z,
 * true y = 1.5 * x1 + 1.5 * x2 + noise. On test data, the collinearity slightly loosens (a realistic
 * domain shift), exposing OLS's huge cancelling weights while Ridge stays rock-solid.
 */
export function collinearExperiment(rho: number, lambda: number): CollinearResult {
  const rng = createRng(17);
  const N = 40;
  const Xtr: number[][] = [];
  const ytr: number[] = [];
  const scale = Math.sqrt(Math.max(1e-8, 1 - rho * rho));
  for (let i = 0; i < N; i++) {
    const x1 = rng.normal();
    const x2 = rho * x1 + scale * rng.normal();
    const y = 1.5 * x1 + 1.5 * x2 + 0.45 * rng.normal();
    Xtr.push([x1, x2]);
    ytr.push(y);
  }
  // Test set: 200 fresh points where the two features decouple slightly (correlation 0.85 * rho)
  const Xte: number[][] = [];
  const yte: number[] = [];
  const rhoTe = 0.85 * rho;
  const scaleTe = Math.sqrt(Math.max(1e-8, 1 - rhoTe * rhoTe));
  for (let i = 0; i < 200; i++) {
    const x1 = rng.normal();
    const x2 = rhoTe * x1 + scaleTe * rng.normal();
    const y = 1.5 * x1 + 1.5 * x2 + 0.45 * rng.normal();
    Xte.push([x1, x2]);
    yte.push(y);
  }

  // Gram matrix for condition number
  let g00 = 0;
  let g01 = 0;
  let g11 = 0;
  for (const [x1, x2] of Xtr) {
    g00 += x1 * x1;
    g01 += x1 * x2;
    g11 += x2 * x2;
  }
  const condNumber = conditionNumber2x2(g00, g01, g11);

  const wOlsArr = fitRidge(Xtr, ytr, 0, true);
  const wRidgeArr = fitRidge(Xtr, ytr, lambda, true);
  const wOls: [number, number] = [wOlsArr[0], wOlsArr[1]];
  const wRidge: [number, number] = [wRidgeArr[0], wRidgeArr[1]];

  const mse = (X: number[][], y: number[], w: [number, number]) => {
    let s = 0;
    for (let i = 0; i < X.length; i++) {
      const e = w[0] * X[i][0] + w[1] * X[i][1] - y[i];
      s += e * e;
    }
    return s / X.length;
  };

  // Run batch gradient descent on MSE with the theoretically optimal safe step size 1 / L_max
  const lMax = ((g00 + g11) + Math.sqrt((g00 - g11) ** 2 + 4 * g01 * g01)) / 2;
  const lr = N / (2 * lMax); // exact 1/L for (1/N) ||Xw - y||^2 whose Hessian is (2/N) X^T X
  let wg0 = 0;
  let wg1 = 0;
  let gdStepsToTol = 500;
  for (let step = 1; step <= 500; step++) {
    let d0 = 0;
    let d1 = 0;
    for (let i = 0; i < N; i++) {
      const r = wg0 * Xtr[i][0] + wg1 * Xtr[i][1] - ytr[i];
      d0 += (2 / N) * r * Xtr[i][0];
      d1 += (2 / N) * r * Xtr[i][1];
    }
    wg0 -= lr * d0;
    wg1 -= lr * d1;
    const dist = Math.hypot(wg0 - wOls[0], wg1 - wOls[1]);
    if (dist < 0.05 && gdStepsToTol === 500) {
      gdStepsToTol = step;
      break;
    }
  }
  const gdFinalDist = Math.hypot(wg0 - wOls[0], wg1 - wOls[1]);

  return {
    rho,
    lambda,
    condNumber,
    wOls,
    wRidge,
    trainMseOls: mse(Xtr, ytr, wOls),
    testMseOls: mse(Xte, yte, wOls),
    trainMseRidge: mse(Xtr, ytr, wRidge),
    testMseRidge: mse(Xte, yte, wRidge),
    gdStepsToTol,
    gdFinalDist,
  };
}

// ------------------------------------------------------------------ 2. Logistic Regression (Newton-Raphson / IRLS)

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

/** Mean binary cross-entropy + (lambda / (2N)) ||w_1..||^2 (intercept w[0] is not penalised). */
export function logisticLossAndGrad(X: number[][], y: number[], w: number[], lambda = 0) {
  const N = X.length;
  const d = w.length;
  let loss = 0;
  const grad = new Array<number>(d).fill(0);
  for (let i = 0; i < N; i++) {
    let z = 0;
    for (let j = 0; j < d; j++) z += X[i][j] * w[j];
    // numerically stable log(1 + exp(z)) - y*z
    const softplus = z > 0 ? z + Math.log(1 + Math.exp(-z)) : Math.log(1 + Math.exp(z));
    loss += softplus - y[i] * z;
    const diff = sigmoid(z) - y[i];
    for (let j = 0; j < d; j++) grad[j] += diff * X[i][j];
  }
  loss /= N;
  for (let j = 0; j < d; j++) grad[j] /= N;
  for (let j = 1; j < d; j++) {
    loss += (lambda / (2 * N)) * w[j] * w[j];
    grad[j] += (lambda / N) * w[j];
  }
  return { loss, grad };
}

/** Fit L2-regularised logistic regression via damped Newton-Raphson / IRLS. */
export function fitLogistic(X: number[][], y: number[], lambda = 1e-4, maxIter = 40): number[] {
  const N = X.length;
  const d = X[0].length;
  let w = new Array<number>(d).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    const { loss, grad } = logisticLossAndGrad(X, y, w, lambda);
    const gNorm = Math.hypot(...grad);
    if (gNorm < 1e-9) break;

    // Exact Hessian H = (1/N) X^T diag(p(1-p)) X + (lambda/N) diag(0, 1, ..., 1)
    const H = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    for (let i = 0; i < N; i++) {
      let z = 0;
      for (let j = 0; j < d; j++) z += X[i][j] * w[j];
      const p = sigmoid(z);
      const wt = Math.max(1e-9, p * (1 - p));
      const row = X[i];
      for (let j = 0; j < d; j++) {
        const wj = wt * row[j];
        for (let k = j; k < d; k++) H[j][k] += wj * row[k];
      }
    }
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < j; k++) H[j][k] = H[k][j];
      H[j][j] = H[j][j] / N + (j > 0 ? lambda / N : 0) + 1e-8;
      for (let k = j + 1; k < d; k++) {
        H[j][k] /= N;
        H[k][j] = H[j][k];
      }
    }

    const step = solveLinear(H, grad);
    if (!step) break;

    // Backtracking line search to guarantee monotone descent
    let alpha = 1.0;
    let nextW = w;
    for (let ls = 0; ls < 12; ls++) {
      const cand = w.map((v, j) => v - alpha * step[j]);
      if (logisticLossAndGrad(X, y, cand, lambda).loss <= loss + 1e-12) {
        nextW = cand;
        break;
      }
      alpha *= 0.5;
    }
    const delta = Math.hypot(...nextW.map((v, j) => v - w[j]));
    w = nextW;
    if (delta < 1e-9) break;
  }
  return w;
}

export function predictLogistic(X: number[][], w: number[]): number[] {
  return X.map((row) => {
    let z = 0;
    for (let j = 0; j < w.length; j++) z += row[j] * w[j];
    return sigmoid(z);
  });
}

// ------------------------------------------------------------------ 2b. 1D Linear vs Logistic outlier breakdown

export const BASE_1D_POINTS: { x: number; y: 0 | 1 }[] = [
  { x: 1.0, y: 0 },
  { x: 1.6, y: 0 },
  { x: 2.2, y: 0 },
  { x: 2.8, y: 0 },
  { x: 3.3, y: 0 },
  { x: 3.7, y: 0 },
  { x: 4.5, y: 1 },
  { x: 5.0, y: 1 },
  { x: 5.6, y: 1 },
  { x: 6.2, y: 1 },
  { x: 6.8, y: 1 },
];

export function linearVsLogistic1D(outlierX: number) {
  const pts = [...BASE_1D_POINTS, { x: outlierX, y: 1 as const }];
  const X = pts.map((p) => [1, p.x]);
  const y = pts.map((p) => p.y);

  const wLin = fitRidge(X, y, 0, false);
  const wLog = fitLogistic(X, y, 0.05);

  // Decision boundaries where prediction = 0.5
  const boundLin = Math.abs(wLin[1]) > 1e-9 ? (0.5 - wLin[0]) / wLin[1] : Infinity;
  const boundLog = Math.abs(wLog[1]) > 1e-9 ? -wLog[0] / wLog[1] : Infinity;

  const linErrors = pts.filter((p) => (wLin[0] + wLin[1] * p.x >= 0.5 ? 1 : 0) !== p.y).length;
  const logErrors = pts.filter((p) => (sigmoid(wLog[0] + wLog[1] * p.x) >= 0.5 ? 1 : 0) !== p.y).length;

  return {
    pts,
    wLin: [wLin[0], wLin[1]] as [number, number],
    wLog: [wLog[0], wLog[1]] as [number, number],
    boundLin,
    boundLog,
    linErrors,
    logErrors,
    linAcc: (pts.length - linErrors) / pts.length,
    logAcc: (pts.length - logErrors) / pts.length,
  };
}

// ------------------------------------------------------------------ 3. Threshold, Precision/Recall & ROC

export type FraudEval = {
  probs: number[];
  labels: (0 | 1)[];
  roc: { thresh: number; fpr: number; tpr: number }[];
  auc: number;
};

let CACHED_FRAUD: FraudEval | null = null;

export function getFraudDataset(): FraudEval {
  if (CACHED_FRAUD) return CACHED_FRAUD;
  const rng = createRng(101);
  // 200 training, 200 test transactions, 5% fraud (10 fraud, 190 legit)
  const make = (nLegit: number, nFraud: number) => {
    const X: number[][] = [];
    const y: (0 | 1)[] = [];
    for (let i = 0; i < nLegit; i++) {
      const amt = rng.normal();
      const vel = rng.normal();
      X.push([1, amt, vel]);
      y.push(0);
    }
    for (let i = 0; i < nFraud; i++) {
      const amt = 1.65 + 0.9 * rng.normal();
      const vel = 1.45 + 0.9 * rng.normal();
      X.push([1, amt, vel]);
      y.push(1);
    }
    return { X, y };
  };
  const tr = make(380, 20);
  const te = make(190, 10);
  const w = fitLogistic(tr.X, tr.y, 0.5);
  const probs = predictLogistic(te.X, w);

  // Build exact ROC curve across all unique thresholds
  const unique = Array.from(new Set([1.01, ...probs, -0.01])).sort((a, b) => b - a);
  const roc = unique.map((thresh) => {
    let tp = 0;
    let fp = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] >= thresh) {
        if (te.y[i] === 1) tp++;
        else fp++;
      }
    }
    return { thresh, fpr: fp / 190, tpr: tp / 10 };
  });
  let auc = 0;
  for (let i = 1; i < roc.length; i++) {
    auc += ((roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr)) / 2;
  }

  CACHED_FRAUD = { probs, labels: te.y, roc, auc };
  return CACHED_FRAUD;
}

export function confusionAt(probs: number[], labels: (0 | 1)[], thresh: number) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < probs.length; i++) {
    const pred = probs[i] >= thresh ? 1 : 0;
    if (pred === 1 && labels[i] === 1) tp++;
    else if (pred === 1 && labels[i] === 0) fp++;
    else if (pred === 0 && labels[i] === 1) fn++;
    else tn++;
  }
  const n = probs.length;
  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
  /** Example cost: €100 per missed fraud (FN) + €5 per false-alarm analyst review (FP). */
  const costEur = fn * 100 + fp * 5;
  return { tp, fp, fn, tn, accuracy, precision, recall, f1, fpr, costEur };
}

// ------------------------------------------------------------------ 4. Lab: Feature Engineering + Regularised Logistic Regression

export type FeatureKey = 'x1' | 'x2' | 'x1sq' | 'x2sq' | 'x1x2' | 'abs';

export const LAB_FEATURES: { key: FeatureKey; label: string; fn: (x1: number, x2: number) => number }[] = [
  { key: 'x1', label: 'x₁', fn: (x1) => x1 },
  { key: 'x2', label: 'x₂', fn: (_, x2) => x2 },
  { key: 'x1sq', label: 'x₁²', fn: (x1) => x1 * x1 },
  { key: 'x2sq', label: 'x₂²', fn: (_, x2) => x2 * x2 },
  { key: 'x1x2', label: 'x₁·x₂', fn: (x1, x2) => x1 * x2 },
  { key: 'abs', label: '|x₁|+|x₂|', fn: (x1, x2) => Math.abs(x1) + Math.abs(x2) },
];

export const LAB_LAMBDAS = [0.001, 0.1, 1, 10] as const;
export const LAB_THRESHOLDS = [0.2, 0.35, 0.5, 0.65] as const;

export type LabBrief = {
  id: 'ring' | 'xor-risk' | 'imbalanced';
  title: string;
  story: string;
  minAcc?: number;
  minRecall?: number;
  minPrecision?: number;
  fixedThresh?: number;
};

export const LAB_BRIEFS: LabBrief[] = [
  {
    id: 'ring',
    title: 'Brief 1 · Beat a Linear Boundary on a Ring',
    story: 'Normal operating points sit inside a circle; faults sit outside it. Raw (x₁, x₂) logistic regression scores near chance. Pick engineered features and L2 λ to reach ≥ 92% held-out accuracy using as few parameters as possible.',
    minAcc: 0.92,
    fixedThresh: 0.5,
  },
  {
    id: 'xor-risk',
    title: 'Brief 2 · Interaction Risk (Same-Sign Faults)',
    story: 'A device fails when x₁ and x₂ are both high or both low (quadrants I and III — a continuous XOR). Reach ≥ 92% held-out accuracy with as few parameters as possible.',
    minAcc: 0.92,
    fixedThresh: 0.5,
  },
  {
    id: 'imbalanced',
    title: 'Brief 3 · Rare-Fault Alerting (~14% Positives)',
    story: 'Only ~14% of units fail, mostly in the curved upper-right risk zone. Tune features, L2 λ and the decision threshold to achieve Recall ≥ 85% AND Precision ≥ 72% on held-out data, with as few parameters as possible.',
    minRecall: 0.85,
    minPrecision: 0.72,
  },
];

export type Point2D = { x1: number; x2: number; y: 0 | 1 };

const LAB_DATA_CACHE = new Map<LabBrief['id'], { train: Point2D[]; test: Point2D[] }>();

export function getLabData(id: LabBrief['id']): { train: Point2D[]; test: Point2D[] } {
  const hit = LAB_DATA_CACHE.get(id);
  if (hit) return hit;
  const rng = createRng(id === 'ring' ? 31 : id === 'xor-risk' ? 47 : 73);
  const gen = (n: number): Point2D[] => {
    const out: Point2D[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = rng.normal();
      const x2 = rng.normal();
      let logit = 0;
      if (id === 'ring') {
        const r2 = x1 * x1 + x2 * x2;
        logit = 5.5 * (r2 - 1.45);
      } else if (id === 'xor-risk') {
        logit = 7.5 * (x1 * x2);
      } else {
        // Imbalanced ~14% positive: outer upper-right curved corner
        logit = 2.8 * (x1 + x2) + 1.4 * (x1 * x2) - 4.5;
      }
      const y: 0 | 1 = rng.next() < sigmoid(logit) ? 1 : 0;
      out.push({ x1, x2, y });
    }
    return out;
  };
  const data = { train: gen(220), test: gen(250) };
  LAB_DATA_CACHE.set(id, data);
  return data;
}

export function buildDesignMatrix(pts: Point2D[], keys: FeatureKey[]): number[][] {
  const fns = keys.map((k) => LAB_FEATURES.find((f) => f.key === k)!.fn);
  return pts.map((p) => [1, ...fns.map((fn) => fn(p.x1, p.x2))]);
}

export function evaluateLab(id: LabBrief['id'], keys: FeatureKey[], lambda: number, thresh: number) {
  const data = getLabData(id);
  const Xtr = buildDesignMatrix(data.train, keys);
  const ytr = data.train.map((p) => p.y);
  const Xte = buildDesignMatrix(data.test, keys);
  const yte = data.test.map((p) => p.y);

  const w = fitLogistic(Xtr, ytr, lambda);
  const probsTr = predictLogistic(Xtr, w);
  const probsTe = predictLogistic(Xte, w);
  const tr = confusionAt(probsTr, ytr, thresh);
  const te = confusionAt(probsTe, yte, thresh);

  return {
    w,
    params: 1 + keys.length, // intercept + selected features
    trainAcc: tr.accuracy,
    testAcc: te.accuracy,
    testPrecision: te.precision,
    testRecall: te.recall,
    testF1: te.f1,
    predictXY: (x1: number, x2: number) => {
      let z = w[0];
      for (let i = 0; i < keys.length; i++) {
        const fn = LAB_FEATURES.find((f) => f.key === keys[i])!.fn;
        z += w[i + 1] * fn(x1, x2);
      }
      return sigmoid(z);
    },
  };
}

export function briefPasses(b: LabBrief, r: ReturnType<typeof evaluateLab>): boolean {
  if (b.minAcc !== undefined && r.testAcc < b.minAcc - 1e-9) return false;
  if (b.minRecall !== undefined && r.testRecall < b.minRecall - 1e-9) return false;
  if (b.minPrecision !== undefined && r.testPrecision < b.minPrecision - 1e-9) return false;
  return true;
}

const LAB_EX_CACHE = new Map<LabBrief['id'], ReturnType<typeof computeLabExhaustive>>();

function computeLabExhaustive(b: LabBrief) {
  const allKeys = LAB_FEATURES.map((f) => f.key);
  const threshList = b.fixedThresh !== undefined ? [b.fixedThresh] : [...LAB_THRESHOLDS];
  let total = 0;
  let passing = 0;
  let bestParams = Infinity;
  let bestAccAtMin = -1;
  let best: { keys: FeatureKey[]; lambda: number; thresh: number; acc: number; recall: number; precision: number; params: number }[] = [];

  for (let mask = 1; mask < 1 << allKeys.length; mask++) {
    const keys = allKeys.filter((_, i) => (mask & (1 << i)) !== 0);
    for (const lambda of LAB_LAMBDAS) {
      for (const thresh of threshList) {
        total++;
        const r = evaluateLab(b.id, keys, lambda, thresh);
        if (!briefPasses(b, r)) continue;
        passing++;
        const params = r.params;
        const row = { keys, lambda, thresh, acc: r.testAcc, recall: r.testRecall, precision: r.testPrecision, params };
        if (params < bestParams) {
          bestParams = params;
          bestAccAtMin = r.testAcc;
          best = [row];
        } else if (params === bestParams) {
          if (r.testAcc > bestAccAtMin + 1e-9) {
            bestAccAtMin = r.testAcc;
            best = [row];
          } else if (Math.abs(r.testAcc - bestAccAtMin) <= 1e-9) {
            best.push(row);
          }
        }
      }
    }
  }
  return { total, passing, bestParams, bestAccAtMin, best };
}

export function labExhaustive(b: LabBrief) {
  const hit = LAB_EX_CACHE.get(b.id);
  if (hit) return hit;
  const res = computeLabExhaustive(b);
  LAB_EX_CACHE.set(b.id, res);
  return res;
}
