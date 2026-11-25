// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { BRIEFS, EMAILS, allConfigs, score, solve } from './featureLab';

describe('Feature Workbench data', () => {
  it('has 12 emails, 6 spam and 6 not, with integer measurements', () => {
    expect(EMAILS.length).toBe(12);
    expect(EMAILS.filter((e) => e.spam).length).toBe(6);
    for (const e of EMAILS) for (const v of Object.values(e.f)) expect(Number.isInteger(v)).toBe(true);
  });
});

describe('Feature Workbench solver (exhaustive: 15 pairs × 5 × 5 units)', () => {
  const s = solve();
  it('scores all 375 configurations', () => {
    expect(allConfigs().length).toBe(375);
    expect(s.all.length).toBe(375);
  });
  it('pins the counts quoted in the lab', () => {
    expect(s.perfect.length).toBe(22);
    expect(s.atLeast10.length).toBe(111);
    expect(new Set(s.perfect.map((r) => r.c.x + '/' + r.c.y))).toEqual(new Set(['links/length', 'excl/name', 'caps/name']));
  });
  it('is solvable but not trivial: most settings fail every brief', () => {
    expect(s.perfect.length).toBeGreaterThanOrEqual(3); // several valid answers
    expect(s.perfect.length / 375).toBeLessThan(0.3); // a random setting usually fails
    expect(s.atLeast10.length / 375).toBeLessThan(0.5);
  });
  it('brief 3: raw units score at most 8, and exactly two unit-only fixes exist', () => {
    expect(score({ x: 'links', y: 'length', ux: '1', uy: '1' })).toBe(7);
    const fixes = s.byBrief[2].map((r) => `${r.c.ux},${r.c.uy}`).sort();
    expect(fixes).toEqual(['1,std', 'std,std']);
  });
  it('brief 4 (no links) has 20 solutions', () => {
    expect(s.byBrief[3].length).toBe(20);
    for (const r of s.byBrief[3]) expect([r.c.x, r.c.y]).not.toContain('links');
  });
  it('every brief has more than one solution', () => {
    BRIEFS.forEach((_, i) => expect(s.byBrief[i].length).toBeGreaterThan(1));
  });
});
