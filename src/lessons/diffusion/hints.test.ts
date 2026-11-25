// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { BRIEFS, stepBudgetHint } from './lab';

const LEAK_PATTERNS = [
  /0\.98/,
  /0\.99/,
  /0\.95/,
  /needs?\s+[45]\s+steps/i,
  /pick\s+ddim/i,
  /pick\s+ddpm/i,
];

describe('diffusion lab evidence-only hints', () => {
  it('never names winning stops or step counts across all 3 briefs and failure modes', () => {
    for (const b of BRIEFS) {
      for (const levels of [[], [0.02, 0.3], [0.3, 0.6], [0.95, 0.98, 0.99]]) {
        for (const offShape of [0.05, 0.85]) {
          const h = stepBudgetHint(b.id, levels, offShape, b.id === 'lopsided' ? 0.01 : null);
          expect(h.length).toBeGreaterThan(20);
          for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
        }
      }
    }
  });
});
