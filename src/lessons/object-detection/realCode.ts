// authored-by: Gemini Antigravity
import type { NmsMode } from './detect';

export function detectorRealCode(mode: NmsMode, conf: number, nmsIou: number): string {
  const c = conf.toFixed(2);
  if (mode === 'hard') {
    return [
      `keep = scores >= ${c}`,
      `idx = torchvision.ops.batched_nms(`,
      `    boxes[keep], scores[keep], labels[keep], iou_threshold=${nmsIou.toFixed(2)}`,
      `)`,
    ].join('\n');
  }
  return [
    `# Gaussian Soft-NMS (Bodla et al., 2017): decay overlapping same-class scores`,
    `scores[same_cls] *= torch.exp(-(ious ** 2) / 0.45)`,
    `kept = boxes[scores >= ${c}]`,
  ].join('\n');
}
