import { describe, expect, it } from 'vitest';
import { bceGrad, bceLoss, escape, fitMAE, fitMSE, huber, lineMAE, lineMSE, maeConst, mean, median, mseConst, sigmoid, sqGrad, sqLoss, type Pt } from './losses';

const argminScan = (f: (c: number) => number, lo: number, hi: number, n = 20001) => {
  let best = lo;
  for (let k = 0; k <= n; k++) {
    const c = lo + ((hi - lo) * k) / n;
    if (f(c) < f(best)) best = c;
  }
  return best;
};

describe('best constant guess', () => {
  const ys = [22, 25, 27, 30, 31, 34, 180];
  it('MSE is minimised by the mean, MAE by the median', () => {
    expect(argminScan((c) => mseConst(ys, c), 0, 200)).toBeCloseTo(mean(ys), 1);
    expect(argminScan((c) => maeConst(ys, c), 0, 200)).toBeCloseTo(median(ys), 1);
  });
  it('median of an even-length list is the midpoint', () => {
    expect(median([1, 5, 3, 9])).toBe(4);
  });
});

describe('line fits', () => {
  const pts: Pt[] = [[0, 1], [1, 2.1], [2, 2.9], [3, 4.2], [4, 5], [5, 6.1], [2.5, 12]];
  it('least squares beats every nearby line on MSE', () => {
    const l = fitMSE(pts);
    for (const dw of [-0.05, 0, 0.05]) for (const db of [-0.05, 0, 0.05]) expect(lineMSE(pts, l)).toBeLessThanOrEqual(lineMSE(pts, { w: l.w + dw, b: l.b + db }) + 1e-12);
  });
  it('the pairwise L1 fit is the true minimum (checked by grid search)', () => {
    const l = fitMAE(pts);
    let best = Infinity;
    for (let w = -1; w <= 3; w += 0.005) for (let b = -2; b <= 4; b += 0.005) best = Math.min(best, lineMAE(pts, { w, b }));
    expect(lineMAE(pts, l)).toBeLessThanOrEqual(best + 1e-9);
  });
  it('the outlier drags the MSE line much more than the MAE line', () => {
    const clean = pts.slice(0, 6);
    const dMse = Math.abs(fitMSE(pts).b - fitMSE(clean).b);
    const dMae = Math.abs(fitMAE(pts).b - fitMAE(clean).b);
    expect(dMse).toBeGreaterThan(5 * dMae);
  });
});

describe('classification losses', () => {
  it('gradients match finite differences', () => {
    for (const z of [-6, -2, 0, 1.5]) {
      const h = 1e-6;
      expect(sqGrad(z)).toBeCloseTo((sqLoss(z + h) - sqLoss(z - h)) / (2 * h), 6);
      expect(bceGrad(z)).toBeCloseTo((bceLoss(z + h) - bceLoss(z - h)) / (2 * h), 6);
      expect(bceGrad(z, 0)).toBeCloseTo((bceLoss(z + h, 0) - bceLoss(z - h, 0)) / (2 * h), 6);
    }
  });
  it('BCE equals −ln p for the true class', () => {
    expect(bceLoss(-2)).toBeCloseTo(-Math.log(sigmoid(-2)), 12);
    expect(bceLoss(3, 0)).toBeCloseTo(-Math.log(1 - sigmoid(3)), 12);
  });
  it('confidently wrong: squared error barely pushes, cross-entropy pushes hard', () => {
    expect(Math.abs(bceGrad(-6)) / Math.abs(sqGrad(-6))).toBeGreaterThan(100);
    const ce = escape('ce', -6, 1);
    const sq = escape('mse', -6, 1);
    expect(ce.escaped && sq.escaped).toBe(true);
    expect(sq.steps).toBeGreaterThan(20 * ce.steps);
  });
  it('huber joins smoothly at δ', () => {
    expect(huber(1, 1)).toBeCloseTo(0.5, 12);
    expect(huber(1 + 1e-9, 1)).toBeCloseTo(0.5, 8);
    expect(huber(3, 1)).toBeCloseTo(2.5, 12);
  });
});
