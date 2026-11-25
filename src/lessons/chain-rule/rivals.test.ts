// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { forward } from './ChainExplorer';
import { backwardRuns, chainSlope, forwardWalks, nudgeRuns, nudgedSlope, rivals } from './rivals';

describe('chain-rule: why walk backwards, and not something else', () => {
  it('counts runs for 5, 1,000 and 1,000,000 weights', () => {
    expect([5, 1000, 1e6].map(nudgeRuns)).toEqual([6, 1001, 1000001]);
    expect([5, 1000, 1e6].map(forwardWalks)).toEqual([5, 1000, 1e6]);
    expect([5, 1000, 1e6].map(backwardRuns)).toEqual([3, 3, 3]);
  });

  it('the chain-rule slope matches a central difference; a 0.01 nudge is visibly off', () => {
    const h = 1e-5;
    expect(chainSlope()).toBeCloseTo((forward(0.4 + h).L - forward(0.4 - h).L) / (2 * h), 8);
    expect(chainSlope().toFixed(4)).toBe('-0.2280');
    expect(nudgedSlope(0.01).toFixed(4)).toBe('-0.2189');
    expect(chainSlope().toFixed(4)).not.toBe(nudgedSlope(0.01).toFixed(4));
  });

  it('the cards', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['The chain rule, walked backwards']);
    expect(rs[0].result).toContain('6, 1,001, 1,000,001 runs');
    expect(rs[0].result).toContain('measures −0.2189 where the true slope is −0.2280');
  });
});
