// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { seqLabRealCode } from './realCode';

describe('seq2seq-attention lab live PyTorch code', () => {
  it('emits matching PyTorch code for bottleneck, Luong/Bahdanau, and Transformer', () => {
    const c1 = seqLabRealCode('bottleneck', { b1Arch: 'cross-attn', b1D: 8, b2Kind: 'luong', b2Tau: 0.15, b3Arch: 'transformer' });
    expect(c1).toContain('MultiheadAttention(8');
    expect(c1).toContain('160 floats');

    const c2 = seqLabRealCode('align', { b1Arch: 'fixed', b1D: 16, b2Kind: 'luong', b2Tau: 0.15, b3Arch: 'transformer' });
    expect(c2).toContain('enc_states.transpose(-1, -2)) / 0.15');

    const c3 = seqLabRealCode('latency', { b1Arch: 'fixed', b1D: 16, b2Kind: 'luong', b2Tau: 0.15, b3Arch: 'transformer' });
    expect(c3).toContain('nn.TransformerEncoderLayer');
  });
});
