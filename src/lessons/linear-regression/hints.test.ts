// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, evaluateLab, labHint, type FeatureKey } from './regression';

describe('linear-regression lab hints', () => {
  const sampleSets: FeatureKey[][] = [
    ['x1', 'x2'],
    ['x1sq', 'x2sq'],
    ['x1', 'x2', 'x1sq', 'x2sq'],
    ['x1x2'],
    ['x1', 'x2', 'abs'],
  ];

  it('never mentions threshold on briefs where threshold is locked', () => {
    for (const brief of LAB_BRIEFS) {
      if (brief.fixedThresh === undefined) continue;
      for (const keys of sampleSets) {
        const res = evaluateLab(brief.id, keys, 0.1, brief.fixedThresh);
        const h = labHint(brief, res);
        expect(h, `${brief.id} [${keys.join(',')}]`).not.toMatch(/threshold/i);
      }
    }
  });

  it('points at evidence and never names specific engineered feature buttons', () => {
    for (const brief of LAB_BRIEFS) {
      for (const keys of sampleSets) {
        const t = brief.fixedThresh ?? 0.5;
        const res = evaluateLab(brief.id, keys, 0.1, t);
        const h = labHint(brief, res);
        expect(h).not.toMatch(/x₁²|x₂²|x₁·x₂|\|x₁\|/);
      }
    }
  });
});
