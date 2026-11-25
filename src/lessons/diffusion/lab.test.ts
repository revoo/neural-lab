// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { labRun } from './diffusion';
import { BRIEFS, exhaustive, plansWithSteps } from './lab';
import { LAB_RESULTS } from './labResults';

describe('Lab: Spend the Step Budget', () => {
  for (const b of BRIEFS)
    it(`${b.id}: the recorded exhaustive search is reproduced`, () => {
      const r = exhaustive(b);
      const rec = LAB_RESULTS[b.id];
      expect(r.fewest).toBe(rec.fewest);
      r.perSteps.forEach((p, i) => {
        const q = rec.perSteps[i];
        expect([p.steps, p.total, p.passing, p.ddim, p.ddpm]).toEqual([q.steps, q.total, q.passing, q.ddim, q.ddpm]);
        expect(p.best).toBeCloseTo(q.best, 4);
      });
    }, 60_000);

  it('fewest steps: ring 4, smiley 5, rare blob 4, and the same with two other sets of starting noise', () => {
    expect([LAB_RESULTS.ring.fewest, LAB_RESULTS.smiley.fewest, LAB_RESULTS.lopsided.fewest]).toEqual([4, 5, 4]);
    for (const id of ['ring', 'smiley', 'lopsided'] as const) expect(LAB_RESULTS[id].fewestWithOtherSeeds).toEqual([LAB_RESULTS[id].fewest, LAB_RESULTS[id].fewest]);
    // re-check the fast briefs with the other seeds
    for (const b of [BRIEFS[0], BRIEFS[2]]) for (const seed of [6, 7]) expect(exhaustive(b, seed).fewest).toBe(LAB_RESULTS[b.id].fewest);
  }, 60_000);

  it('debrief counts and patterns', () => {
    const ring = LAB_RESULTS.ring.perSteps;
    expect(ring[2].best).toBeCloseTo(0.16, 2);
    expect([ring[3].passing, ring[3].total, ring[3].ddim, ring[3].ddpm]).toEqual([55, 728, 20, 35]);
    const pass = plansWithSteps('ring', 4).filter((p) => p.tv <= 0.1);
    for (const p of pass) {
      expect(p.levels[0]).toBeGreaterThanOrEqual(0.2);
      expect(p.levels[0]).toBeLessThanOrEqual(0.9);
      expect(p.levels[1]).toBeGreaterThanOrEqual(0.8);
      expect(p.levels[2]).toBeGreaterThanOrEqual(0.98);
    }
    const sm = LAB_RESULTS.smiley.perSteps[4];
    expect([sm.passing, sm.total, sm.ddpm]).toEqual([63, 2002, 0]);
    const lop = LAB_RESULTS.lopsided.perSteps[3];
    expect([lop.passing, lop.total]).toEqual([6, 728]);
    expect(labRun('lopsided', [0.95, 0.98, 0.99], 0).judge.counts[1]).toBe(0);
    const lpass = plansWithSteps('lopsided', 4).filter((p) => p.tv <= 0.06);
    for (const p of lpass) {
      expect(p.levels[0]).toBeGreaterThanOrEqual(0.2);
      expect(p.levels[0]).toBeLessThanOrEqual(0.4);
      expect([0.6, 0.7]).toContain(p.levels[1]);
      expect([0.9, 0.95]).toContain(p.levels[2]);
    }
  }, 60_000);
});
