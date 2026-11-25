// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { GAN_LAB_BRIEFS, ganLabHint } from './gan';

const LEAK_PATTERNS = [
  /lrRatio\s*=\s*0\.[25]/i,
  /ratio\s*0\.5/i,
  /dSteps\s*=\s*1/i,
  /noiseDim\s*=\s*2/i,
  /set\s+noise\s+to\s+2/i,
  /pick\s+1\s+D-step/i,
];

describe('GAN lab evidence-only hints', () => {
  it('never leaks winning LR ratio, D-steps or noise dim across briefs', () => {
    for (const b of GAN_LAB_BRIEFS) {
      for (const state of [
        { meanModes: 1.2, meanW1: 1.6, updatesToCover3: null, meanOffMode: 0.4 },
        { meanModes: 2.6, meanW1: 0.55, updatesToCover3: null, meanOffMode: 0.2 },
        { meanModes: 3.0, meanW1: 0.36, updatesToCover3: 120, meanOffMode: 0.16 },
      ]) {
        const h = ganLabHint(b.id, state);
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
  });
});
