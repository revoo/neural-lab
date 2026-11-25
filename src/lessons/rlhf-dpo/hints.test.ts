// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { ALIGNMENT_BRIEFS, LAB_BETAS, LAB_METHODS, LAB_STEPS, alignmentHint, evaluateAlignmentBrief } from './dpo';

describe('alignment-lab hints never name a locked method', () => {
  it('Brief 1 (proxy RM only) never suggests DPO or other methods', () => {
    const brief = ALIGNMENT_BRIEFS.find((b) => !b.allowDpo)!;
    for (const method of LAB_METHODS)
      for (const beta of LAB_BETAS) {
        const cfg = { method, beta, steps: 50 };
        const ev = evaluateAlignmentBrief(brief, cfg);
        if (ev.passes) continue;
        const hint = alignmentHint(brief, cfg, ev);
        expect(hint).not.toMatch(/DPO|SFT|method|steps/);
      }
  });
  it('briefs with DPO unlocked give a hint for every failing setting', () => {
    for (const brief of ALIGNMENT_BRIEFS.filter((b) => b.allowDpo))
      for (const method of LAB_METHODS)
        for (const beta of [0.1, 1.0])
          for (const steps of [LAB_STEPS[0], LAB_STEPS[4]]) {
            const cfg = { method, beta, steps };
            const ev = evaluateAlignmentBrief(brief, cfg);
            if (!ev.passes) expect(alignmentHint(brief, cfg, ev).length).toBeGreaterThan(20);
          }
  });
});
