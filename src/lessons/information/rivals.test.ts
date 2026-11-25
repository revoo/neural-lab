// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { LOSSES, SCORE_SLOPE, measuredSlope, rivals } from './rivals';

describe('information: why cross-entropy, and not something else', () => {
  it('the hand-derived slopes match nudging the score', () => {
    for (const id of ['squared', 'crossEntropy'] as const)
      for (const p of [0.0001, 0.01, 0.3, 0.6, 0.95]) expect(SCORE_SLOPE[id](p)).toBeCloseTo(measuredSlope(id, p), 6);
  });

  it('accuracy is flat', () => {
    expect([0.6, 0.95].map(LOSSES.accuracy)).toEqual([0, 0]);
    expect(LOSSES.accuracy(0.4)).toBe(1);
    expect(measuredSlope('accuracy', 0.6)).toBe(0);
  });

  it('squared error barely separates 1% from 0.01%, and its push fades', () => {
    expect(LOSSES.squared(0.01).toFixed(2)).toBe('0.98');
    expect(LOSSES.squared(0.0001).toFixed(2)).toBe('1.00');
    expect(SCORE_SLOPE.squared(0.0001).toPrecision(1)).toBe('-0.0002');
    // the Predict: squared error pushes about 5,000 times more weakly than cross-entropy
    expect(Math.round(SCORE_SLOPE.crossEntropy(0.0001) / SCORE_SLOPE.squared(0.0001) / 100) * 100).toBe(5000);
  });

  it('cross-entropy punishes without limit and pushes hardest when confidently wrong', () => {
    expect(LOSSES.crossEntropy(0.01).toFixed(2)).toBe('4.61');
    expect(LOSSES.crossEntropy(0.0001).toFixed(2)).toBe('9.21');
    expect(SCORE_SLOPE.crossEntropy(0.0001).toFixed(2)).toBe('-1.00');
  });

  it('only cross-entropy passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['Cross-entropy']);
    expect(rs[1].result).toContain('only −0.0002');
    expect(rs[2].result).toContain('−1.00 for that');
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
  });
});
