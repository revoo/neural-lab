// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { visionLabHint } from './vision';

describe('images-as-tensors lab hints', () => {
  it('points at visual evidence without spoiling the exact menu option name', () => {
    const base = { rep: 'raw-rgb' as const, alias: 'stride-0' as const, layout: 'NHWC' as const, dtype: 'fp32' as const, norm: 'none' as const };
    expect(visionLabHint('lighting', base)).not.toMatch(/HSV Hue|Chromaticity/i);
    expect(visionLabHint('aliasing', base)).not.toMatch(/4×4 Box Average/i);
    expect(visionLabHint('vram', base)).not.toMatch(/FP16/i);
  });
});
