// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, evaluateKnnLab, knnLabHint, type KnnLabConfig } from './knn';

describe('knn lab hints', () => {
  it('never names locked controls (scaling or column toggles) on briefs where they are absent', () => {
    const cfgs: Record<(typeof LAB_BRIEFS)[number]['id'], KnnLabConfig[]> = {
      unscaled: [
        { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1] },
        { scaled: true, metric: 'l2', weight: 'uniform', k: 1, cols: [0, 1] },
      ],
      'noisy-cols': [
        { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1, 2, 3, 4, 5] },
        { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1, 2] },
      ],
      pockets: [
        { scaled: false, metric: 'l2', weight: 'uniform', k: 25, cols: [0, 1] },
        { scaled: false, metric: 'l2', weight: 'distance', k: 15, cols: [0, 1] },
      ],
    };

    for (const brief of LAB_BRIEFS) {
      for (const cfg of cfgs[brief.id]) {
        const res = evaluateKnnLab(brief, cfg);
        const h = knnLabHint(brief, cfg, res);
        if (!brief.allowScalingToggle) {
          expect(h, `${brief.id}`).not.toMatch(/standardis|z-score|scale/i);
        }
        if (!brief.featureLabels) {
          expect(h, `${brief.id}`).not.toMatch(/column/i);
        }
        // Never spoil exact answer columns or weighting button name
        expect(h).not.toMatch(/Col A|Col B|Col C|1\/d|distance weighting/i);
      }
    }
  });
});
