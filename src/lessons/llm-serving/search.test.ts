// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { DESK_BRIEFS, SPEC_BRIEFS, evaluateDesk, specBriefSummary } from './lab';
import { BRIEF1_STATIC_SWEEP, DESK_RESULTS } from './results';
import { brief1StaticWaitSweep, searchBrief } from './search.run';
import type { DeskConfig } from './realCode';

describe('llm-serving S3: Lab 1 exhaustive 300-config search & Lab 2 exhaustive 48-cell summary', () => {
  it('every Lab 1 default starting config fails its brief', () => {
    for (const brief of DESK_BRIEFS) {
      const { passes } = evaluateDesk(brief, brief.defaultConfig);
      expect(passes, `${brief.id} default should fail`).toBe(false);
    }
  });

  it('re-verifies recorded optima, failing routes, and sampled configs for all 3 Lab 1 briefs', () => {
    for (const brief of DESK_BRIEFS) {
      const rec = DESK_RESULTS[brief.id];
      expect(rec.totalConfigs).toBe(300);
      expect(rec.passingCount).toBeGreaterThanOrEqual(18);

      // Re-run recorded optimum directly
      const bestCfg = rec.bestConfig as unknown as DeskConfig;
      const bestEval = evaluateDesk(brief, bestCfg);
      expect(bestEval.passes).toBe(true);
      expect(bestEval.sim.metrics.goodputReqPerSec).toBeCloseTo(rec.bestGoodput, 3);
      expect(bestEval.sim.metrics.preemptions).toBe(rec.bestStats.preemptions);

      // Re-run full search on Brief 1/2/3 (takes ~2s each) and check counts match exactly
      const fresh = searchBrief(brief);
      expect(fresh.passingCount).toBe(rec.passingCount);
      expect(fresh.bestGoodput).toBe(rec.bestGoodput);
      expect(fresh.bestKeys).toEqual(rec.bestKeys);
      expect(fresh.failingCheck).toEqual(rec.failingCheck);
      expect(fresh.paretoKeys).toEqual(rec.paretoKeys);
    }

    // Pin exact counts quoted in prose & debrief
    expect(DESK_RESULTS.chat.passingCount).toBe(90);
    expect(DESK_RESULTS.chat.failingCheck.staticPassing).toBe(0);
    expect(DESK_RESULTS.chat.bestGoodput).toBeCloseTo(4.17, 2);

    expect(DESK_RESULTS.longdoc.passingCount).toBe(32);
    expect(DESK_RESULTS.longdoc.failingCheck.staticPassing).toBe(0);
    expect(DESK_RESULTS.longdoc.failingCheck.noChunkPassing).toBe(0);
    expect(DESK_RESULTS.longdoc.failingCheck.reserveMaxPassing).toBe(0);
    expect(DESK_RESULTS.longdoc.bestGoodput).toBeCloseTo(3.72, 2);

    expect(DESK_RESULTS.crowd.passingCount).toBe(18);
    expect(DESK_RESULTS.crowd.failingCheck.staticPassing).toBe(0);
    expect(DESK_RESULTS.crowd.failingCheck.reserveMaxPassing).toBe(0);
    expect(DESK_RESULTS.crowd.failingCheck.pagedNoPrefixPassing).toBe(0);
    expect(DESK_RESULTS.crowd.bestGoodput).toBeCloseTo(5.72, 2);
    expect(DESK_RESULTS.crowd.bestStats.preemptions).toBe(0);
  }, 20_000);

  it('re-verifies Brief 1 static-batching 56-config wait sweep (never exceeds 25% SLO)', () => {
    const fresh = brief1StaticWaitSweep();
    expect(fresh).toEqual(BRIEF1_STATIC_SWEEP);
    expect(fresh.gridSize).toBe(56);
    expect(fresh.bestAttainment).toBe(0.25);
  });

  it('pins Lab 2 (Tune the draft) exhaustive counts across all 3 briefs', () => {
    const s1 = specBriefSummary(SPEC_BRIEFS[0]);
    const s2 = specBriefSummary(SPEC_BRIEFS[1]);
    const s3 = specBriefSummary(SPEC_BRIEFS[2]);

    // Brief 1 (greedy, >= 1.95x): multiple passing configs across both drafters; best = bigram k=5 (2.30x)
    expect(s1.total).toBe(16);
    expect(s1.passingCount).toBe(6);
    expect(s1.bestSpeedup).toBeCloseTo(2.30, 2);
    expect(s1.bestCells[0].drafter).toBe('bigram');
    expect(s1.bestCells[0].k).toBe(5);

    // Brief 2 (T = 1.0, >= 2.00x): multiple passing configs across both drafters; best = bigram k=4 (2.28x)
    expect(s2.total).toBe(16);
    expect(s2.passingCount).toBe(8);
    expect(s2.bestSpeedup).toBeCloseTo(2.28, 2);
    expect(s2.bestCells[0].drafter).toBe('bigram');
    expect(s2.bestCells[0].k).toBe(4);

    // Brief 3 (CPU wall-clock > 1.0x): impossible across all 48 recorded cells!
    expect(s3.total).toBe(48);
    expect(s3.passingCount).toBe(0);
    expect(s3.bestSpeedup).toBeLessThan(1.0);
  });
});
