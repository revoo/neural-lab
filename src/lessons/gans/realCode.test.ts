// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { buildGanCode } from './realCode';

describe('GAN live code builder', () => {
  it('reflects lr ratio, D steps and noise dimension in PyTorch training loop', () => {
    const code = buildGanCode({ noiseDim: 2, dSteps: 1, lrRatio: 0.5 });
    expect(code).toContain('lr=0.0075');
    expect(code).toContain('range(1)');
    expect(code).toContain('torch.randn(batch, 2)');
  });
});
