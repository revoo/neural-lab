// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { detectorRealCode } from './realCode';

describe('object-detection: real torchvision code generator', () => {
  it('emits batched_nms for Hard NMS and Gaussian decay for Soft-NMS', () => {
    const hard = detectorRealCode('hard', 0.30, 0.50);
    expect(hard).toContain('scores >= 0.30');
    expect(hard).toContain('torchvision.ops.batched_nms');
    expect(hard).toContain('iou_threshold=0.50');

    const soft = detectorRealCode('soft', 0.35, 0.50);
    expect(soft).toContain('torch.exp(-(ious ** 2) / 0.45)');
    expect(soft).toContain('scores >= 0.35');
  });
});
