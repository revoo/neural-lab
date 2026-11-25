// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { convCodeLines } from './realCode';

describe('convolution realCode', () => {
  it('emits F.conv2d tensor for kernel briefs and nn.Conv2d Sequential for stack brief', () => {
    const base = { vertW: [-1, 0, 1, -1, 0, 1, 0, 0, 0], diagW: [0, 0, 0, 0, 0, 0, 0, 0, 0], k: 3 as const, pad: 1 as const, stride: 2 as const };
    expect(convCodeLines('vertical', base)).toContain('F.conv2d(img, kernel');
    expect(convCodeLines('stack-shape', base)).toContain('nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)');
  });
});
