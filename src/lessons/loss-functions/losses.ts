/** Loss functions for the loss-functions lesson, plus exact best fits under each. */

export const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
export function median(v: number[]) {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Mean squared error and mean absolute error of a constant guess c. */
export const mseConst = (ys: number[], c: number) => mean(ys.map((y) => (y - c) ** 2));
export const maeConst = (ys: number[], c: number) => mean(ys.map((y) => Math.abs(y - c)));
/** Huber loss with threshold δ: squared near zero, absolute far away (both scaled to join smoothly). */
export const huber = (r: number, delta = 1) => (Math.abs(r) <= delta ? 0.5 * r * r : delta * (Math.abs(r) - 0.5 * delta));

export type Pt = [number, number];
export type Line = { w: number; b: number };

export const lineMSE = (pts: Pt[], { w, b }: Line) => mean(pts.map(([x, y]) => (y - (w * x + b)) ** 2));
export const lineMAE = (pts: Pt[], { w, b }: Line) => mean(pts.map(([x, y]) => Math.abs(y - (w * x + b))));

/** Least-squares line (closed form). */
export function fitMSE(pts: Pt[]): Line {
  const mx = mean(pts.map((p) => p[0]));
  const my = mean(pts.map((p) => p[1]));
  let sxy = 0;
  let sxx = 0;
  for (const [x, y] of pts) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
  }
  const w = sxx ? sxy / sxx : 0;
  return { w, b: my - w * mx };
}

/**
 * Least-absolute-deviations line, exactly: an optimal L1 line always passes through (at least) two
 * of the data points, so try every pair. O(n³) — fine for the dozen or so points in the widget.
 */
export function fitMAE(pts: Pt[]): Line {
  let best: Line = { w: 0, b: median(pts.map((p) => p[1])) };
  let bestLoss = lineMAE(pts, best);
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[j];
      if (Math.abs(x2 - x1) < 1e-12) continue;
      const w = (y2 - y1) / (x2 - x1);
      const cand = { w, b: y1 - w * x1 };
      const l = lineMAE(pts, cand);
      if (l < bestLoss - 1e-12) {
        best = cand;
        bestLoss = l;
      }
    }
  return best;
}

// ---- Classification: one sigmoid output, true label 1.

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

/** Squared error on the probability, (p − y)², and its slope with respect to the score z. */
export const sqLoss = (z: number, y = 1) => (sigmoid(z) - y) ** 2;
export const sqGrad = (z: number, y = 1) => {
  const p = sigmoid(z);
  return 2 * (p - y) * p * (1 - p);
};
/** Binary cross-entropy, −[y ln p + (1 − y) ln(1 − p)], computed stably from z, and its slope p − y. */
export const bceLoss = (z: number, y = 1) => (y ? Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
export const bceGrad = (z: number, y = 1) => sigmoid(z) - y;

/**
 * Gradient descent on the score z alone, starting confidently wrong, until the model gives the true
 * class more than 50% (z > 0). Returns the path and the number of steps (capped).
 */
export function escape(loss: 'mse' | 'ce', z0: number, lr: number, cap = 5000) {
  let z = z0;
  const path = [z];
  for (let s = 1; s <= cap; s++) {
    z -= lr * (loss === 'mse' ? sqGrad(z) : bceGrad(z));
    path.push(z);
    if (z > 0) return { steps: s, path, escaped: true };
  }
  return { steps: cap, path, escaped: false };
}

/** Gradient descent on the score for a fixed number of steps; returns every score along the way. */
export function trajectory(loss: 'mse' | 'ce', z0: number, lr: number, steps: number) {
  let z = z0;
  const path = [z];
  for (let s = 1; s <= steps; s++) {
    z -= lr * (loss === 'mse' ? sqGrad(z) : bceGrad(z));
    path.push(z);
  }
  return path;
}
