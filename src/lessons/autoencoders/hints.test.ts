// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { BRIEFS, allDesigns, designKey, facePrinterHint } from './lab';
import { LAB_RESULTS } from './labResults';

const LEAK_PATTERNS = [
  /β\s*=\s*0\.[13]\b/,
  /beta\s*=\s*0\.[13]\b/i,
  /width\s+16\b/i,
  /4,884/,
  /set\s+latent\s*=\s*2/i,
];

describe('autoencoders lab evidence-only hints', () => {
  it('never names winning settings across all 54 designs for both briefs', () => {
    for (const b of BRIEFS) {
      for (const d of allDesigns()) {
        const before = facePrinterHint(b.id, d, null);
        const after = facePrinterHint(b.id, d, LAB_RESULTS[designKey(d)].seeds);
        expect(before.length).toBeGreaterThan(20);
        expect(after.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) {
          expect(before).not.toMatch(pat);
          if (!(pat.source === '4,884' && d.latent === 2 && d.hidden === 16 && d.beta > 0)) {
            expect(after).not.toMatch(pat);
          }
        }
      }
    }
  });
});
