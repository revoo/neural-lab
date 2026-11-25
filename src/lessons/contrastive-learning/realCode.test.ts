// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { contrastiveRealCode } from './realCode';

describe('contrastive-learning: real PyTorch InfoNCE / MoCo code generator', () => {
  it('emits accurate PyTorch contrastive loss code across all three briefs', () => {
    const base = { b1Negs: true, b1Tau: 0.15, b2Tau: 0.05, b3Opt: 'moco-64-q512' as const };
    expect(contrastiveRealCode('circle', { ...base, b1Negs: false })).toContain('Alignment-only');
    expect(contrastiveRealCode('circle', base)).toContain('/ 0.15');
    expect(contrastiveRealCode('circle', base)).toContain('F.cross_entropy');

    expect(contrastiveRealCode('temp', base)).toContain('logit_scale = 20.0');

    expect(contrastiveRealCode('batch', { ...base, b3Opt: 'simclr-512' })).toContain('B=512 active');
    expect(contrastiveRealCode('batch', base)).toContain('queue.clone().detach()');
    expect(contrastiveRealCode('batch', base)).toContain('Q=512');
  });
});
