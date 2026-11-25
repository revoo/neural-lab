// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { rivals, ruleErrors, bestSingleRule, squashCollisions, squashGap, nearestAgree, pairDistance } from './rivals';

describe('vectors: why a vector, and not something else', () => {
  it('the hand-written rule, and the best single-measurement rule, both make mistakes', () => {
    expect(ruleErrors(3)).toBe(2);
    const best = bestSingleRule();
    expect(best).toMatchObject({ key: 'bangs', t: 1, errors: 1 });
    expect(best.wrong).toHaveLength(1);
    expect(best.wrong[0]).toMatch(/^Happy birthday/);
  });

  it('squashing into a total separates this data but merges 7 of 45 pairs', () => {
    expect(squashGap()).toBeGreaterThan(0);
    expect(squashCollisions()).toEqual({ pairs: 7, total: 45 });
    expect(pairDistance().toFixed(2)).toBe('4.24');
  });

  it('the vector keeps every neighbour honest', () => {
    expect(nearestAgree()).toBe(10);
  });

  it('only the vector passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['A list of numbers: a vector']);
    expect(rs.every((r) => !r.result.includes('NaN') && !r.result.includes('undefined'))).toBe(true);
  });
});
