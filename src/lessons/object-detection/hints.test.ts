// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { DET_LAB_BRIEFS, LAB_CONFS, LAB_NMS_IOUS, detLabHint, type NmsMode } from './detect';

describe('object-detection: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact thresholds before passing', () => {
    for (const b of DET_LAB_BRIEFS) {
      for (const mode of ['hard', 'soft'] as NmsMode[]) {
        const ious = mode === 'hard' ? LAB_NMS_IOUS : ([0.5] as const);
        for (const conf of LAB_CONFS) {
          for (const nmsIou of ious) {
            const h = detLabHint(b.id, conf, nmsIou, mode);
            expect(h.length).toBeGreaterThan(20);
            expect(h).not.toMatch(/set conf to|pick soft-nms|select 0\.30|choose 0\.65/i);
          }
        }
      }
    }
  });
});
