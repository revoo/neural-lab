// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { rnnLabRealCode } from './realCode';

describe('rnns lab live PyTorch code', () => {
  it('emits orthogonal init, clip_grad_norm_, and detach TBPTT code matching controls', () => {
    const c1 = rnnLabRealCode('flow', { b1Act: 'tanh', b1Rho: 1.25, b2Lr: 0.025, b2Clip: null, b3K: 2, b3H: 4 });
    expect(c1).toContain('nn.init.orthogonal_');
    expect(c1).toContain('gain=1.25');

    const c2None = rnnLabRealCode('clip', { b1Act: 'tanh', b1Rho: 0.5, b2Lr: 0.05, b2Clip: null, b3K: 2, b3H: 4 });
    expect(c2None).toContain('No gradient clipping');
    expect(c2None).toContain('lr=0.05');

    const c2Clip = rnnLabRealCode('clip', { b1Act: 'tanh', b1Rho: 0.5, b2Lr: 0.015, b2Clip: 1.0, b3K: 2, b3H: 4 });
    expect(c2Clip).toContain('clip_grad_norm_');
    expect(c2Clip).toContain('max_norm=1');

    const c3 = rnnLabRealCode('tbptt', { b1Act: 'tanh', b1Rho: 0.5, b2Lr: 0.025, b2Clip: null, b3K: 7, b3H: 2 });
    expect(c3).toContain('hidden_size=2');
    expect(c3).toContain('x.split(7, dim=1)');
    expect(c3).toContain('h = h.detach()');
  });
});
