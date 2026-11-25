// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, evaluateTreeLab, treeLabHint, type TreeLabConfig } from './trees';

describe('decision-trees lab hints', () => {
  const cfgs: TreeLabConfig[] = [
    { method: 'tree', maxDepth: 1, nTrees: 1 },
    { method: 'tree', maxDepth: 5, nTrees: 1 },
    { method: 'rf', maxDepth: 5, nTrees: 15 },
    { method: 'gbt', maxDepth: 3, nTrees: 30 },
  ];

  it('never names the winning family directly as an answer', () => {
    for (const brief of LAB_BRIEFS) {
      for (const cfg of cfgs) {
        const res = evaluateTreeLab(brief.id, cfg);
        const h = treeLabHint(brief, cfg, res);
        expect(h, `${brief.id} ${cfg.method}`).not.toMatch(/Random Forest|Gradient Boosting|Single Tree/i);
      }
    }
  });
});
