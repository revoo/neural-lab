// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { visionCodeLines } from './realCode';

describe('images-as-tensors realCode', () => {
  const base = { rep: 'hsv-hue' as const, alias: 'box-avg' as const, layout: 'NCHW' as const, dtype: 'fp16' as const, norm: 'zscore' as const };
  it('emits accurate Python/PyTorch snippets for each brief', () => {
    expect(visionCodeLines('lighting', base)).toContain('COLOR_RGB2HSV');
    expect(visionCodeLines('aliasing', base)).toContain('F.avg_pool2d');
    expect(visionCodeLines('vram', base)).toContain('permute(0, 3, 1, 2)');
    expect(visionCodeLines('vram', base)).toContain('torch.float16');
  });
});
