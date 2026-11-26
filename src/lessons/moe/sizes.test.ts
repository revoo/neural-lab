// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { countParams, flopsPerToken, MIXTRAL, weightGB } from './sizes';

describe('MoE size calculator', () => {
  it('reproduces Mixtral’s published 47B total and 13B active parameters', () => {
    const c = countParams(MIXTRAL);
    expect(+(c.total / 1e9).toFixed(1)).toBe(46.7);
    expect(+(c.active / 1e9).toFixed(1)).toBe(12.9);
    expect(Math.round(c.total / 1e9)).toBe(47);
    expect(Math.round(c.active / 1e9)).toBe(13);
  });
  it('numbers quoted in the lesson', () => {
    const c = countParams(MIXTRAL);
    expect(+weightGB(c.total).toFixed(1)).toBe(93.4);
    expect(+(flopsPerToken(c.active) / 1e9).toFixed(1)).toBe(25.8);
    // top-1 instead of top-2: same memory, fewer active weights
    expect(+(countParams({ ...MIXTRAL, topK: 1 }).active / 1e9).toFixed(1)).toBe(7.2);
    // doubling the experts doubles the expert weights but leaves the active count alone
    const d = countParams({ ...MIXTRAL, experts: 16 });
    expect(+(d.total / 1e9).toFixed(1)).toBe(91.8);
    expect(d.active).toBe(c.active + 32 * 4096 * 8); // only the router grows
    // one expert holds 176M weights; the eight of them are 97% of the model
    expect(+(c.expert / 1e6).toFixed(1)).toBe(176.2);
    expect(+((32 * 8 * c.expert) / c.total).toFixed(2)).toBe(0.97);
  });
});
