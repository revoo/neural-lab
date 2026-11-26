// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { SPEC_BRIEFS, specBriefSummary, specHint } from './lab';
import { SPEC_DRAFTERS, SPEC_KS } from './spec.run';

const BANNED_SPEC_PHRASES = [
  'bigram',
  'small gpt',
  'small-gpt',
  'k =',
  'k=',
  'try ',
  'switch to',
  'pick ',
  'select ',
  'choose ',
];

describe('speculative-decoding lab: exhaustive summary and evidence-only hints', () => {
  it('pins exhaustive brief counts across all 48 cells', () => {
    const s1 = specBriefSummary(SPEC_BRIEFS[0]);
    const s2 = specBriefSummary(SPEC_BRIEFS[1]);
    const s3 = specBriefSummary(SPEC_BRIEFS[2]);

    expect(s1.total).toBe(16);
    expect(s1.passingCount).toBe(6);
    expect(s1.bestSpeedup).toBeCloseTo(2.30, 2);
    expect(s1.bestCells[0].drafter).toBe('bigram');
    expect(s1.bestCells[0].k).toBe(5);

    expect(s2.total).toBe(16);
    expect(s2.passingCount).toBe(8);
    expect(s2.bestSpeedup).toBeCloseTo(2.28, 2);
    expect(s2.bestCells[0].drafter).toBe('bigram');
    expect(s2.bestCells[0].k).toBe(4);

    expect(s3.total).toBe(48);
    expect(s3.passingCount).toBe(0);
    expect(s3.bestSpeedup).toBeLessThan(1.0);
  });

  it('hints never name a drafter or k value across all 48 cells', () => {
    for (const brief of SPEC_BRIEFS) {
      for (const d of SPEC_DRAFTERS) {
        for (const k of SPEC_KS) {
          const h = specHint(brief, d, k).toLowerCase();
          expect(h.length).toBeGreaterThan(20);
          for (const banned of BANNED_SPEC_PHRASES) {
            expect(h, `${brief.id} (${d}, ${k}) leaked "${banned}": ${h}`).not.toContain(banned);
          }
        }
      }
    }
  });
});
