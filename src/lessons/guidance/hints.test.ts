// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { GUIDE_BRIEFS, LAB_STEPS, LAB_WS, guideLabHint, labTry } from './guide';

const LEAK_PATTERNS = [
  /w\s*=\s*(1\.25|1\.75|3(\.0)?)\b/i,
  /needs?\s+[256]\s+steps/i,
  /set\s+steps/i,
  /set\s+w/i,
];

describe('guidance lab evidence-only hints', () => {
  it('never names winning settings across sampled settings for all 3 briefs', () => {
    for (const b of GUIDE_BRIEFS) {
      for (const steps of [LAB_STEPS[0], LAB_STEPS[2], LAB_STEPS[5]]) {
        for (const w of [LAB_WS[0], LAB_WS[4], LAB_WS[8], LAB_WS[16]]) {
          const h = guideLabHint(b, labTry(steps, w));
          expect(h.length).toBeGreaterThan(20);
          for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
        }
      }
    }
  });
});
