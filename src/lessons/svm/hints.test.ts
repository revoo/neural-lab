// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, svmLabHint } from './svm';

const LEAK_PATTERNS = [
  /x₁²\s*\+\s*x₂²/,
  /\|x₁\|\s*\+\s*\|x₂\|/,
  /C\s*=\s*0\.[13]/i,
  /C\s*=\s*[13]\b/i,
  /set\s+C\s+to/i,
  /select\s+r2/i,
];

describe('SVM lab evidence-only hints', () => {
  it('never names winning features or exact C settings across all briefs', () => {
    for (const b of LAB_BRIEFS) {
      for (const state of [
        { separates: false, marginPct: 0, featureCount: 0, ringAcc: 0.5, valAcc: 0.7, trainAcc: 0.69, streetWidth: 15 },
        { separates: true, marginPct: 72, featureCount: 2, ringAcc: 0.58, valAcc: 0.917, trainAcc: 1.0, streetWidth: 0.19 },
        { separates: true, marginPct: 97, featureCount: 2, ringAcc: 1.0, valAcc: 0.95, trainAcc: 0.94, streetWidth: 1.4 },
      ]) {
        const hint = svmLabHint(b.id, state);
        expect(hint.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(hint).not.toMatch(pat);
      }
    }
  });
});
