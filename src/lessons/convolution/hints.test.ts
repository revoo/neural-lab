// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { convLabHint } from './conv';

describe('convolution lab hints', () => {
  it('gives evidence-based feedback without printing raw answer arrays', () => {
    const base = { vertW: [0, 0, 0, -1, 0, 1, 0, 0, 0], diagW: [0, 0, 0, 0, 0, 0, 0, 0, 0], k: 3 as const, pad: 0 as const, stride: 2 as const };
    expect(convLabHint('vertical', base)).not.toMatch(/\[-1/);
    expect(convLabHint('diagonal', base)).not.toMatch(/\[-1/);
    expect(convLabHint('stack-shape', base)).not.toMatch(/K=3, P=1/);
  });
});
