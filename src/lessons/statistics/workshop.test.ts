// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import { recipeChangeLine, recipeDist, recipeMoments, rollRecipe, type Recipe } from './stats';

const R = (counts: Partial<Recipe['counts']>, plus = 0): Recipe => ({ counts: { d2: 0, d4: 0, d6: 0, d8: 0, d12: 0, d20: 0, ...counts }, plus });

describe('Dice Workshop: roll and log', () => {
  it('the plan\'s example log line: adding a d6 to 3d6', () => {
    expect(recipeChangeLine(R({ d6: 3 }), R({ d6: 4 }))).toBe('+d6: mean 10.50 → 14.00 · variance 8.75 → 11.67 · SD 2.96 → 3.42');
    expect(recipeChangeLine(R({ d8: 4 }), R({ d8: 4 }, 12))).toMatch(/^constant \+12: mean 18\.00 → 30\.00/);
  });
  it('1,000 seeded rolls land close to the exact distribution', () => {
    const r = R({ d6: 3 });
    const totals = rollRecipe(r, 1000, createRng(7));
    expect(Math.min(...totals)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...totals)).toBeLessThanOrEqual(18);
    const m = totals.reduce((a, b) => a + b, 0) / totals.length;
    // the sample mean of 1,000 rolls wobbles by about SD/√1000 ≈ 0.09 around 10.5
    expect(Math.abs(m - recipeMoments(r).mean)).toBeLessThan(0.3);
    const d = recipeDist(r);
    const p10 = totals.filter((t) => t === 10).length / 1000;
    expect(Math.abs(p10 - d.p[10 - d.min])).toBeLessThan(0.04);
  });
});
