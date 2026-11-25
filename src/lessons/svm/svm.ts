// authored-by: Gemini Antigravity
/**
 * Support Vector Machine engine:
 * - Exact exhaustive 2D hard-margin solver (enumerating all 2-SV pairs and 3-SV triples)
 * - Dual SMO soft-margin solver in arbitrary feature dimension
 * - Primal subgradient hinge-loss solver (for independent cross-check & NumPy parity)
 * - Datasets, feature-bin kernel transforms, and exhaustive lab solvers for "Widen the Street"
 */
import { createRng } from '../../lib/rng';

export type Pt2 = { id: string; x: number; y: number; label: 1 | -1 };

export type LinearModel = {
  w: number[];
  b: number;
};

export type HardMarginResult = {
  separable: true;
  w: [number, number];
  b: number;
  /** Half-margin 1/||w|| (distance from centre line to either gutter). */
  halfMargin: number;
  /** Full street width 2/||w|| (distance between the two gutters). */
  streetWidth: number;
  /** Indices of points lying on the gutters (y_i (w·x_i + b) ≈ 1). */
  supportIndices: number[];
} | {
  separable: false;
};

export type SoftMarginResult = {
  w: number[];
  b: number;
  alphas: number[];
  /** Street width 2/||w|| (0 if w = 0). */
  streetWidth: number;
  /** Hinge slack ξ_i = max(0, 1 - y_i (w·x_i + b)) per training point. */
  slacks: number[];
  /** Indices with α_i > 1e-4 (margin or violating support vectors). */
  supportIndices: number[];
  trainAcc: number;
  hingeObj: number;
};

const dotN = (a: number[], b: number[]): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

const normN = (a: number[]): number => Math.sqrt(dotN(a, a));

export function predictScore(model: LinearModel, feat: number[]): number {
  return dotN(model.w, feat) + model.b;
}

export function accuracyOn(model: LinearModel, X: number[][], y: (1 | -1)[]): number {
  let ok = 0;
  for (let i = 0; i < X.length; i++) {
    const s = predictScore(model, X[i]);
    if ((s >= 0 ? 1 : -1) === y[i]) ok++;
  }
  return ok / X.length;
}

/**
 * Exact exhaustive 2D hard-margin SVM solver.
 * By KKT / convex-hull geometry in R^2, the maximum-margin boundary is determined by either:
 *   - 2 support vectors (one +1, one -1): perpendicular bisector of the pair; or
 *   - 3 support vectors (two of one class, one of the other): parallel to the same-class pair.
 * Checks feasibility on every point and returns the feasible candidate with smallest ||w||.
 */
export function solveHardMargin2D(pts: Pt2[]): HardMarginResult {
  const n = pts.length;
  const pos = pts.map((p, i) => ({ p, i })).filter((o) => o.p.label === 1);
  const neg = pts.map((p, i) => ({ p, i })).filter((o) => o.p.label === -1);
  if (pos.length === 0 || neg.length === 0) return { separable: false };

  let bestW: [number, number] | null = null;
  let bestB = 0;
  let bestNormSq = Infinity;

  const tryCandidate = (wx: number, wy: number, b: number) => {
    const n2 = wx * wx + wy * wy;
    if (!Number.isFinite(n2) || n2 < 1e-18 || n2 >= bestNormSq - 1e-12) return;
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      if (p.label * (wx * p.x + wy * p.y + b) < 1 - 1e-7) return;
    }
    bestNormSq = n2;
    bestW = [wx, wy];
    bestB = b;
  };

  // Case A: 2 support vectors (one +, one -)
  for (const a of pos) {
    for (const c of neg) {
      const dx = a.p.x - c.p.x;
      const dy = a.p.y - c.p.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < 1e-14) continue;
      const wx = (2 * dx) / d2;
      const wy = (2 * dy) / d2;
      const b = -(wx * (a.p.x + c.p.x) + wy * (a.p.y + c.p.y)) / 2;
      tryCandidate(wx, wy, b);
    }
  }

  // Case B: 3 support vectors (2 of class s, 1 of class -s)
  const tryTriples = (same: typeof pos, other: typeof pos, s: 1 | -1) => {
    for (let i = 0; i < same.length; i++) {
      for (let j = i + 1; j < same.length; j++) {
        const a = same[i].p;
        const bPt = same[j].p;
        // Normal perpendicular to (a - bPt)
        const ux = -(a.y - bPt.y);
        const uy = a.x - bPt.x;
        if (ux * ux + uy * uy < 1e-14) continue;
        for (const cObj of other) {
          const c = cObj.p;
          const denom = ux * (a.x - c.x) + uy * (a.y - c.y);
          if (Math.abs(denom) < 1e-12) continue;
          const scale = (2 * s) / denom;
          const wx = scale * ux;
          const wy = scale * uy;
          const b = s - (wx * a.x + wy * a.y);
          tryCandidate(wx, wy, b);
        }
      }
    }
  };
  tryTriples(pos, neg, 1);
  tryTriples(neg, pos, -1);

  if (!bestW) return { separable: false };
  const w: [number, number] = bestW;
  const wNorm = Math.hypot(w[0], w[1]);
  const supportIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const margin = p.label * (w[0] * p.x + w[1] * p.y + bestB);
    if (Math.abs(margin - 1) < 1e-4) supportIndices.push(i);
  }
  return {
    separable: true,
    w,
    b: bestB,
    halfMargin: 1 / wNorm,
    streetWidth: 2 / wNorm,
    supportIndices,
  };
}

/**
 * Evaluate a user-drawn straight line given by two distinct points A and B on a 2D dataset.
 * Automatically picks the orientation (which side is +1 vs -1) that maximises accuracy.
 * If separable, returns the exact street width (twice the perpendicular distance to the
 * nearest point, centred between the two closest opposing points if the line is off-centre,
 * or 2 * min signed distance for the line itself!).
 */
export function evaluateLine2D(
  pts: Pt2[],
  a: [number, number],
  b: [number, number],
): {
  valid: boolean;
  separates: boolean;
  accuracy: number;
  correct: number;
  /** Perpendicular distance from the drawn line to the nearest point (signed min >= 0 if separating). */
  minSignedDist: number;
  /** Street width symmetric around the drawn line: 2 * minSignedDist (0 if not separating). */
  streetWidth: number;
  /** Unit normal pointing to the +1 side. */
  normal: [number, number];
  /** Offset c such that normal·x + c = 0 is the drawn line. */
  offset: number;
  closestPosIdx: number;
  closestNegIdx: number;
} {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) {
    return {
      valid: false,
      separates: false,
      accuracy: 0,
      correct: 0,
      minSignedDist: -1,
      streetWidth: 0,
      normal: [0, 1],
      offset: 0,
      closestPosIdx: -1,
      closestNegIdx: -1,
    };
  }
  let nx = -dy / len;
  let ny = dx / len;
  let off = -(nx * a[0] + ny * a[1]);

  const countSide = (sign: number) => {
    let ok = 0;
    for (const p of pts) {
      const v = sign * (nx * p.x + ny * p.y + off);
      if (p.label * v > 0) ok++;
    }
    return ok;
  };
  if (countSide(-1) > countSide(1)) {
    nx = -nx;
    ny = -ny;
    off = -off;
  }

  let correct = 0;
  let minSigned = Infinity;
  let minPos = Infinity;
  let minNeg = Infinity;
  let closestPosIdx = -1;
  let closestNegIdx = -1;

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const signed = p.label * (nx * p.x + ny * p.y + off);
    if (signed > 0) correct++;
    if (signed < minSigned) minSigned = signed;
    if (p.label === 1 && signed < minPos) {
      minPos = signed;
      closestPosIdx = i;
    }
    if (p.label === -1 && signed < minNeg) {
      minNeg = signed;
      closestNegIdx = i;
    }
  }

  const separates = minSigned > 1e-6;
  return {
    valid: true,
    separates,
    accuracy: correct / pts.length,
    correct,
    minSignedDist: minSigned,
    streetWidth: separates ? 2 * minSigned : 0,
    normal: [nx, ny],
    offset: off,
    closestPosIdx,
    closestNegIdx,
  };
}

/**
 * Deterministic dual SMO solver for linear soft-margin SVM:
 *   min_{w, b} 0.5 ||w||^2 + C Σ max(0, 1 - y_i (w·x_i + b))
 */
export function solveSoftMarginSMO(X: number[][], y: (1 | -1)[], C: number, maxPasses = 80, tol = 1e-6): SoftMarginResult {
  const n = X.length;
  const d = X[0].length;
  const K: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => dotN(X[i], X[j])),
  );

  const alphas = new Array<number>(n).fill(0);
  let b = 0;

  const f = (i: number) => {
    let s = b;
    for (let j = 0; j < n; j++) if (alphas[j] > 0) s += alphas[j] * y[j] * K[j][i];
    return s;
  };

  let passes = 0;
  let iter = 0;
  const maxIter = 4000;

  while (passes < maxPasses && iter < maxIter) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      const r = y[i] * Ei;
      if ((r < -tol && alphas[i] < C) || (r > tol && alphas[i] > 0)) {
        // Choose partner j maximising |Ei - Ej| (deterministic heuristic, fallback scan)
        let bestJ = -1;
        let bestDelta = -1;
        for (let cand = 0; cand < n; cand++) {
          if (cand === i) continue;
          const delta = Math.abs(Ei - (f(cand) - y[cand]));
          if (delta > bestDelta) {
            bestDelta = delta;
            bestJ = cand;
          }
        }
        const candidates = [bestJ];
        for (let k = 0; k < n; k++) {
          const j = (i + 1 + k) % n;
          if (j !== i && j !== bestJ) candidates.push(j);
        }

        for (const j of candidates) {
          if (j < 0) continue;
          const Ej = f(j) - y[j];
          const aiOld = alphas[i];
          const ajOld = alphas[j];

          let L: number;
          let H: number;
          if (y[i] !== y[j]) {
            L = Math.max(0, ajOld - aiOld);
            H = Math.min(C, C + ajOld - aiOld);
          } else {
            L = Math.max(0, aiOld + ajOld - C);
            H = Math.min(C, aiOld + ajOld);
          }
          if (L >= H - 1e-12) continue;

          const eta = 2 * K[i][j] - K[i][i] - K[j][j];
          if (eta >= -1e-12) continue;

          let ajNew = ajOld - (y[j] * (Ei - Ej)) / eta;
          if (ajNew > H) ajNew = H;
          else if (ajNew < L) ajNew = L;

          if (Math.abs(ajNew - ajOld) < 1e-7) continue;

          const aiNew = aiOld + y[i] * y[j] * (ajOld - ajNew);
          const b1 = b - Ei - y[i] * (aiNew - aiOld) * K[i][i] - y[j] * (ajNew - ajOld) * K[i][j];
          const b2 = b - Ej - y[i] * (aiNew - aiOld) * K[i][j] - y[j] * (ajNew - ajOld) * K[j][j];

          if (aiNew > 1e-7 && aiNew < C - 1e-7) b = b1;
          else if (ajNew > 1e-7 && ajNew < C - 1e-7) b = b2;
          else b = 0.5 * (b1 + b2);

          alphas[i] = aiNew;
          alphas[j] = ajNew;
          changed++;
          break;
        }
      }
    }
    iter++;
    if (changed === 0) passes++;
    else passes = 0;
  }

  const w = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    if (alphas[i] > 0) {
      const coeff = alphas[i] * y[i];
      for (let k = 0; k < d; k++) w[k] += coeff * X[i][k];
    }
  }

  // Refine b from free support vectors (0 < α < C) for maximum precision
  const freeBs: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alphas[i] > 1e-5 * C && alphas[i] < (1 - 1e-5) * C) {
      freeBs.push(y[i] - dotN(w, X[i]));
    }
  }
  if (freeBs.length > 0) {
    b = freeBs.reduce((s, v) => s + v, 0) / freeBs.length;
  }

  const wNorm = normN(w);
  const slacks = X.map((xi, i) => Math.max(0, 1 - y[i] * (dotN(w, xi) + b)));
  const supportIndices = alphas
    .map((a, i) => (a > 1e-4 ? i : -1))
    .filter((i) => i >= 0);
  const hingeObj = 0.5 * wNorm * wNorm + C * slacks.reduce((s, v) => s + v, 0);
  return {
    w,
    b,
    alphas,
    streetWidth: wNorm > 1e-9 ? 2 / wNorm : 0,
    slacks,
    supportIndices,
    trainAcc: accuracyOn({ w, b }, X, y),
    hingeObj,
  };
}

/**
 * Independent primal subgradient hinge-loss solver (used to cross-check exact hard-margin
 * and dual SMO solvers in svm.test.ts).
 */
export function solveHingeSubgradient(
  X: number[][],
  y: (1 | -1)[],
  C: number,
  steps = 8000,
  lr0 = 0.25,
): LinearModel & { hingeObj: number } {
  const n = X.length;
  const d = X[0].length;
  const w = new Array<number>(d).fill(0);
  let b = 0;
  let bestObj = Infinity;
  let bestW = w.slice();
  let bestB = 0;

  for (let t = 1; t <= steps; t++) {
    const gw = w.slice();
    let gb = 0;
    let hingeSum = 0;
    for (let i = 0; i < n; i++) {
      const m = y[i] * (dotN(w, X[i]) + b);
      if (m < 1) {
        hingeSum += 1 - m;
        for (let k = 0; k < d; k++) gw[k] -= C * y[i] * X[i][k];
        gb -= C * y[i];
      }
    }
    const obj = 0.5 * dotN(w, w) + C * hingeSum;
    if (obj < bestObj) {
      bestObj = obj;
      bestW = w.slice();
      bestB = b;
    }
    const step = lr0 / (1 + 0.008 * t);
    const gNorm = Math.sqrt(dotN(gw, gw) + gb * gb) + 1e-12;
    const scale = step / Math.max(1, gNorm / 10);
    for (let k = 0; k < d; k++) w[k] -= scale * gw[k];
    b -= scale * gb;
  }

  return { w: bestW, b: bestB, hingeObj: bestObj };
}

// ------------------------------------------------------------------ Datasets

/**
 * 10 cleanly separable 2D points for MarginExplorer and Lab Briefs 1–2.
 * Diagonal widest street with 3 exact support vectors: P1 (-1.0, 1.1), P2 (1.0, 0.1), N1 (-0.6, -1.1).
 */
export const SEPARABLE_POINTS: Pt2[] = [
  { id: 'P1', x: -1.0, y: 1.1, label: 1 },
  { id: 'P2', x: 1.0, y: 0.1, label: 1 },
  { id: 'P3', x: -1.9, y: 2.3, label: 1 },
  { id: 'P4', x: 0.1, y: 2.1, label: 1 },
  { id: 'P5', x: 2.0, y: 1.5, label: 1 },
  { id: 'N1', x: -0.6, y: -1.1, label: -1 },
  { id: 'N2', x: 1.4, y: -2.3, label: -1 },
  { id: 'N3', x: -2.2, y: -1.8, label: -1 },
  { id: 'N4', x: -0.2, y: -2.4, label: -1 },
  { id: 'N5', x: 2.2, y: -2.7, label: -1 },
];

/** Three rival separating lines on SEPARABLE_POINTS for the opening Predict (all 100% on training data). */
export const RIVAL_LINES: { id: string; label: string; a: [number, number]; b: [number, number] }[] = [
  { id: 'tight-blue', label: 'Line A (hugs class +1)', a: [-2.6, 1.75], b: [2.4, -0.75] },
  { id: 'tight-orange', label: 'Line B (hugs class −1)', a: [-2.6, 0.1], b: [2.4, -2.0] },
  { id: 'widest', label: 'Line C (widest street)', a: [-2.6, 0.9], b: [2.4, -1.6] },
];

/** Held-out test cloud around the same two clusters, for comparing the rival lines honestly. */
export function generateSeparableTestCloud(seed = 2026): Pt2[] {
  const rng = createRng(seed);
  const out: Pt2[] = [];
  const opt = solveHardMargin2D(SEPARABLE_POINTS);
  if (!opt.separable) return out;
  const wn = Math.hypot(opt.w[0], opt.w[1]);
  const nx = opt.w[0] / wn;
  const ny = opt.w[1] / wn;
  const off = opt.b / wn;
  for (let i = 0; i < 200; i++) {
    const label: 1 | -1 = i < 100 ? 1 : -1;
    const along = rng.normal(0, 1.2);
    const perp = label * (opt.halfMargin + 0.25 + rng.normal(0, 0.55));
    const cx = -off * nx;
    const cy = -off * ny;
    const x = cx + perp * nx + along * -ny;
    const y = cy + perp * ny + along * nx;
    out.push({ id: `T${i}`, x, y, label });
  }
  return out;
}

/**
 * Overlapping dataset for the Soft Margin (C slider) widget:
 * Two clean clusters plus two mislabelled/outlier points intruding across the gap.
 */
export const OVERLAP_POINTS: Pt2[] = [
  // Clean +1 cluster (top-left)
  { id: 'A1', x: -2.2, y: 1.9, label: 1 },
  { id: 'A2', x: -1.5, y: 2.3, label: 1 },
  { id: 'A3', x: -1.1, y: 1.3, label: 1 },
  { id: 'A4', x: -0.4, y: 2.1, label: 1 },
  { id: 'A5', x: 0.3, y: 1.6, label: 1 },
  { id: 'A6', x: -2.0, y: 0.8, label: 1 },
  { id: 'A7', x: -0.6, y: 0.9, label: 1 },
  { id: 'A8', x: 0.9, y: 2.2, label: 1 },
  // Clean -1 cluster (bottom-right)
  { id: 'B1', x: 2.1, y: -1.8, label: -1 },
  { id: 'B2', x: 1.4, y: -2.2, label: -1 },
  { id: 'B3', x: 0.8, y: -1.3, label: -1 },
  { id: 'B4', x: 0.1, y: -2.0, label: -1 },
  { id: 'B5', x: -0.7, y: -1.6, label: -1 },
  { id: 'B6', x: 1.9, y: -0.7, label: -1 },
  { id: 'B7', x: 0.5, y: -0.9, label: -1 },
  { id: 'B8', x: -1.3, y: -2.2, label: -1 },
  // Two outliers crossing into the opposite cluster
  { id: 'O1', x: 0.45, y: -0.25, label: 1 },
  { id: 'O2', x: -0.35, y: 0.35, label: -1 },
];

/** Concentric rings for KernelLiftWidget and Lab Brief 3 (24 points). */
export const RING_POINTS: Pt2[] = (() => {
  const rng = createRng(1992);
  const pts: Pt2[] = [];
  for (let i = 0; i < 12; i++) {
    const ang = (2 * Math.PI * i) / 12 + (rng.next() - 0.5) * 0.25;
    const r = 0.35 + rng.next() * 0.6; // [0.35, 0.95]
    pts.push({ id: `I${i}`, x: +(r * Math.cos(ang)).toFixed(3), y: +(r * Math.sin(ang)).toFixed(3), label: 1 });
  }
  for (let i = 0; i < 12; i++) {
    const ang = (2 * Math.PI * i) / 12 + 0.15 + (rng.next() - 0.5) * 0.2;
    const r = 1.65 + rng.next() * 0.55; // [1.65, 2.20]
    pts.push({ id: `O${i}`, x: +(r * Math.cos(ang)).toFixed(3), y: +(r * Math.sin(ang)).toFixed(3), label: -1 });
  }
  return pts;
})();

export type FeatureId = 'x1' | 'x2' | 'x1sq' | 'x2sq' | 'x1x2' | 'r2' | 'l1';

export type FeatureDef = {
  id: FeatureId;
  label: string;
  formula: string;
  fn: (p: Pt2) => number;
};

export const FEATURE_BIN: FeatureDef[] = [
  { id: 'x1', label: 'x₁', formula: 'x₁', fn: (p) => p.x },
  { id: 'x2', label: 'x₂', formula: 'x₂', fn: (p) => p.y },
  { id: 'x1sq', label: 'x₁²', formula: 'x₁²', fn: (p) => p.x * p.x },
  { id: 'x2sq', label: 'x₂²', formula: 'x₂²', fn: (p) => p.y * p.y },
  { id: 'x1x2', label: 'x₁·x₂', formula: 'x₁·x₂', fn: (p) => p.x * p.y },
  { id: 'r2', label: 'x₁² + x₂²', formula: 'x₁² + x₂²', fn: (p) => p.x * p.x + p.y * p.y },
  { id: 'l1', label: '|x₁| + |x₂|', formula: '|x₁| + |x₂|', fn: (p) => Math.abs(p.x) + Math.abs(p.y) },
];

export function evalFeatureSubset(pts: Pt2[], ids: FeatureId[]): {
  acc: number;
  separates: boolean;
  res: SoftMarginResult | null;
} {
  if (ids.length === 0) return { acc: 0.5, separates: false, res: null };
  const defs = ids.map((id) => FEATURE_BIN.find((f) => f.id === id)!);
  const X = pts.map((p) => defs.map((f) => f.fn(p)));
  const y = pts.map((p) => p.label);
  const res = solveSoftMarginSMO(X, y, 50, 100);
  const separates = res.trainAcc === 1;
  return { acc: res.trainAcc, separates, res };
}

/** Exhaustive search over all 2^7 - 1 = 127 non-empty feature subsets for Brief 3. */
export function searchFeatureSubsets(pts: Pt2[] = RING_POINTS): {
  total: number;
  passing: FeatureId[][];
  minCount: number;
  minimalPassing: FeatureId[][];
} {
  const passing: FeatureId[][] = [];
  const total = (1 << FEATURE_BIN.length) - 1;
  for (let mask = 1; mask <= total; mask++) {
    const ids: FeatureId[] = [];
    for (let bit = 0; bit < FEATURE_BIN.length; bit++) {
      if (mask & (1 << bit)) ids.push(FEATURE_BIN[bit].id);
    }
    if (evalFeatureSubset(pts, ids).separates) passing.push(ids);
  }
  const minCount = Math.min(...passing.map((s) => s.length));
  const minimalPassing = passing.filter((s) => s.length === minCount);
  return { total, passing, minCount, minimalPassing };
}

/**
 * Brief 4: noisy training set + clean held-out validation set.
 * True boundary is horizontal (y = 0, class +1 above, class -1 below),
 * with two rogue outliers on the flanks that stay separable if you tilt sharply!
 * So small C keeps a relaxed boundary, moderate C aligns with the true horizontal split,
 * and huge C tilts diagonally to hit 100% training accuracy while losing validation accuracy!
 */
export const C_GRID = [0.005, 0.02, 0.05, 0.1, 0.3, 1, 3, 10, 50] as const;

export const BRIEF4_DATA: { train: Pt2[]; val: Pt2[] } = (() => {
  const rng = createRng(1995);
  const train: Pt2[] = [];
  const val: Pt2[] = [];

  // 30 clean training points near y = 0 (true boundary y = 0, x in [-1.5, 1.5])
  for (let i = 0; i < 30; i++) {
    const label: 1 | -1 = i < 15 ? 1 : -1;
    const x = (rng.next() - 0.5) * 2.8;
    const y = label * (0.45 + Math.abs(rng.normal(0, 0.75)));
    train.push({ id: `TR${i}`, x: +x.toFixed(3), y: +y.toFixed(3), label });
  }
  // Two leverage outliers on opposite flanks that a steep diagonal CAN separate at high C!
  train.push({ id: 'OUT1', x: -2.2, y: -0.65, label: 1 });
  train.push({ id: 'OUT2', x: 2.2, y: 0.65, label: -1 });

  // 60 clean held-out validation points across [-2.4, 2.4]
  for (let i = 0; i < 60; i++) {
    const label: 1 | -1 = i < 30 ? 1 : -1;
    const x = (rng.next() - 0.5) * 4.6;
    const y = label * (0.15 + Math.abs(rng.normal(0, 0.8)));
    val.push({ id: `VA${i}`, x: +x.toFixed(3), y: +y.toFixed(3), label });
  }
  return { train, val };
})();

export type CGridRow = {
  C: number;
  trainAcc: number;
  valAcc: number;
  streetWidth: number;
  supportCount: number;
  res: SoftMarginResult;
};

export function searchCGrid(
  train: Pt2[] = BRIEF4_DATA.train,
  val: Pt2[] = BRIEF4_DATA.val,
): { rows: CGridRow[]; bestValAcc: number; bestCs: number[] } {
  const Xtr = train.map((p) => [p.x, p.y]);
  const ytr = train.map((p) => p.label);
  const Xva = val.map((p) => [p.x, p.y]);
  const yva = val.map((p) => p.label);

  const rows: CGridRow[] = C_GRID.map((C) => {
    const res = solveSoftMarginSMO(Xtr, ytr, C, 120);
    const valAcc = accuracyOn(res, Xva, yva);
    return {
      C,
      trainAcc: res.trainAcc,
      valAcc,
      streetWidth: res.streetWidth,
      supportCount: res.supportIndices.length,
      res,
    };
  });
  const bestValAcc = Math.max(...rows.map((r) => r.valAcc));
  const bestCs = rows.filter((r) => Math.abs(r.valAcc - bestValAcc) < 1e-9).map((r) => r.C);
  return { rows, bestValAcc, bestCs };
}

// ------------------------------------------------------------------ Evidence-only hints

export const LAB_BRIEFS = [
  { id: 'separate', tab: '1 · Separate' },
  { id: 'widen', tab: '2 · Widen' },
  { id: 'lift', tab: '3 · Lift rings' },
  { id: 'soft', tab: '4 · Soft C' },
] as const;

export type SvmBriefId = (typeof LAB_BRIEFS)[number]['id'];

export function svmLabHint(
  brief: SvmBriefId,
  state: {
    separates?: boolean;
    marginPct?: number;
    featureCount?: number;
    ringAcc?: number;
    valAcc?: number;
    trainAcc?: number;
    streetWidth?: number;
  },
): string {
  if (brief === 'separate') {
    if (!state.separates) {
      return 'Look at which coloured points sit on the wrong side of your line. Drag the two handles so every blue point is on one side and every orange point is on the other.';
    }
    return 'All ten points are cleanly on their own side — this brief is already solved.';
  }
  if (brief === 'widen') {
    if (!state.separates) {
      return 'First separate the two colours completely; the street width stays zero while any point sits on the wrong side.';
    }
    return `Your street is at ${Math.round(state.marginPct ?? 0)}% of the widest possible gap. Notice which single point is closest to your line — tilt and shift the line away from that bottleneck until the nearest points on both sides sit at equal distance.`;
  }
  if (brief === 'lift') {
    if ((state.featureCount ?? 0) === 0) {
      return 'Pick at least one feature from the parts bin so the classifier has coordinates to cut across.';
    }
    if ((state.ringAcc ?? 0) < 1) {
      return 'Plain linear coordinates only draw straight cuts across the plane, which always slice through the outer ring. Look for a feature that measures how far a point sits from the centre.';
    }
    return 'Your chosen features separate the rings completely. Check whether any single measurement from the bin already separates inner points from outer ones by distance from the origin.';
  }
  // soft
  if ((state.trainAcc ?? 0) > (state.valAcc ?? 0) + 0.06) {
    return 'Training accuracy is noticeably higher than validation accuracy, and the street has narrowed to bend around a few noisy points. Check what happens to validation accuracy as you allow more margin violations.';
  }
  if ((state.streetWidth ?? 0) > 2.0) {
    return 'The street is very wide and relaxed, so it is ignoring real structure along with the noise. Watch how validation accuracy responds as you tighten the penalty.';
  }
  return 'Compare the training curve against validation accuracy across penalty settings: watch where validation peaks before the boundary starts tilting to chase the few training outliers.';
}
