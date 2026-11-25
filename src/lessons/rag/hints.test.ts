// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, LAB_KS, LAB_LAMBDAS, LAB_SIZES, ragLabHint } from './lab';

const LEAK_PATTERNS = [
  /82\.3/,
  /109\.8/,
  /set\s+λ\s*=\s*0\.(25|5)/i,
  /top\s+[234]\b/i,
  /two-sentence/i,
  /one-sentence/i,
];

describe('rag lab evidence-only hints', () => {
  it('never names winning settings and never asks to change locked lambda on Brief 3', () => {
    for (const b of LAB_BRIEFS) {
      for (const size of [LAB_SIZES[0], LAB_SIZES[2], LAB_SIZES[4]]) {
        for (const lambda of b.fixedLambda !== undefined ? [b.fixedLambda] : [LAB_LAMBDAS[0], LAB_LAMBDAS[2], LAB_LAMBDAS[4]]) {
          for (const k of [LAB_KS[0], LAB_KS[2], LAB_KS[4]]) {
            const h = ragLabHint(b, size, lambda, k);
            expect(h.length).toBeGreaterThan(20);
            for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
            if (b.fixedLambda !== undefined) {
              expect(h).not.toMatch(/blending keyword|hybrid weight/i);
            }
          }
        }
      }
    }
  });
});
