// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { gridLog10, randomSearchLog10, rivals, simulateOneKnob, slopeRuns } from './rivals';

describe('derivatives: why slopes, and not search', () => {
  it('search costs grow as a power of ten per knob; slopes grow linearly', () => {
    expect([1, 10, 1e6].map(randomSearchLog10)).toEqual([1, 10, 1e6]);
    expect([1, 10, 1e6].map(gridLog10)).toEqual([1, 10, 1e6]);
    expect([1, 10, 1e6].map(slopeRuns)).toEqual([2, 20, 2e6]);
  });

  it('a simulation agrees: one knob takes about 10 random guesses', () => {
    for (const seed of [1, 2, 3]) expect(Math.abs(simulateOneKnob(20000, seed) - 10)).toBeLessThan(0.4);
  });

  it('the table states the counts', () => {
    const rs = rivals();
    expect(rs[0].result).toContain('10, 10^10, 10^1,000,000 runs');
    expect(rs[1].result).toContain('10, 10^10, 10^1,000,000 runs');
    expect(rs[2].result).toContain('2, 20, 2,000,000 runs');
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['The slope, by backpropagation']);
  });
});
