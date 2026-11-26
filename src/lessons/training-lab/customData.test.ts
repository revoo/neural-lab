// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, Trainer, type Pt } from './lab';
import {
  addPaintedPoint,
  clampPoint,
  hasBothClasses,
  makeStarterCloud,
  seedWithOutlier,
  splitCustomTrainVal,
} from './customData';

describe('Training Lab custom paint helpers (customData.ts)', () => {
  it('clamps coordinates to the specified bound', () => {
    expect(clampPoint(2.5, -3.1, 1.0)).toEqual({ x: 1.0, y: -1.0 });
    expect(clampPoint(-0.4, 0.7, 1.0)).toEqual({ x: -0.4, y: 0.7 });
    expect(clampPoint(4.0, -4.0)).toEqual({ x: 1.6, y: -1.6 });
  });

  it('deduplicates points closer than minSpacing and caps at 160 points', () => {
    let pts: Pt[] = [];
    pts = addPaintedPoint(pts, 0.2, 0.2, 0, 0.06, 160);
    expect(pts.length).toBe(1);
    // Closer than 0.06 -> ignored
    pts = addPaintedPoint(pts, 0.22, 0.21, 0, 0.06, 160);
    expect(pts.length).toBe(1);
    // Further than 0.06 -> accepted
    pts = addPaintedPoint(pts, 0.30, 0.20, 1, 0.06, 160);
    expect(pts.length).toBe(2);

    // Fill up to 160 points on a well-spaced grid and verify cap
    let many: Pt[] = [];
    for (let r = 0; r < 15; r++) {
      for (let c = 0; c < 15; c++) {
        const x = -1.4 + c * 0.2;
        const y = -1.4 + r * 0.2;
        many = addPaintedPoint(many, x, y, ((r + c) % 2) as 0 | 1, 0.06, 160);
      }
    }
    expect(many.length).toBe(160);
  });

  it('requires at least 2 points of each class in hasBothClasses', () => {
    expect(hasBothClasses([])).toBe(false);
    expect(hasBothClasses([{ x: 0, y: 0, label: 0 }, { x: 0.2, y: 0, label: 0 }, { x: 0.5, y: 0, label: 1 }])).toBe(false);
    expect(
      hasBothClasses([
        { x: 0, y: 0, label: 0 },
        { x: 0.2, y: 0, label: 0 },
        { x: 0.5, y: 0, label: 1 },
        { x: 0.7, y: 0, label: 1 },
      ]),
    ).toBe(true);
  });

  it('splits custom painted data deterministically with stratified class balance', () => {
    const starter = makeStarterCloud();
    expect(starter.length).toBe(12);
    expect(hasBothClasses(starter)).toBe(true);

    const { train, test } = splitCustomTrainVal(starter, 0.8);
    expect(train.length + test.length).toBe(12);
    // 6 per class * 0.8 = 5 train + 1 test per class -> 10 train, 2 test
    expect(train.filter((p) => p.label === 0).length).toBe(5);
    expect(train.filter((p) => p.label === 1).length).toBe(5);
    expect(test.filter((p) => p.label === 0).length).toBe(1);
    expect(test.filter((p) => p.label === 1).length).toBe(1);

    // Dropping an outlier appends cleanly without mutating the base array
    const withOutlier = seedWithOutlier(starter, -0.7, -0.4, 1);
    expect(starter.length).toBe(12);
    expect(withOutlier.length).toBe(13);
    expect(withOutlier[12]).toEqual({ x: -0.7, y: -0.4, label: 1 });
  });

  it('real Trainer converges to >= 95% accuracy on a painted 2-cluster dataset', () => {
    const cloud = makeStarterCloud();
    // Add 4 more points per cluster for a rich 20-point painted set
    let pts = cloud;
    for (const [x, y] of [[-0.6, -0.45], [-0.8, -0.5], [-0.5, -0.8], [-0.95, -0.7]]) {
      pts = addPaintedPoint(pts, x, y, 0);
    }
    for (const [x, y] of [[0.6, 0.45], [0.8, 0.5], [0.5, 0.8], [0.95, 0.7]]) {
      pts = addPaintedPoint(pts, x, y, 1);
    }
    const { train, test } = splitCustomTrainVal(pts, 0.8);
    const trainer = new Trainer({ ...DEFAULT_CONFIG, hidden: [4, 4], lr: 0.03, seed: 1 }, train, test);
    trainer.step(400);
    const m = trainer.metrics();
    expect(m.trainAcc).toBeGreaterThanOrEqual(0.95);
    expect(m.testAcc).toBeGreaterThanOrEqual(0.95);
  });
});
