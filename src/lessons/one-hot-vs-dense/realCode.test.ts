// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { embedRealCode } from './realCode';

describe('one-hot-vs-dense: real PyTorch code generator', () => {
  it('emits accurate PyTorch nn.Embedding / F.one_hot calls for all three briefs', () => {
    const base = { rep: 'dense-2' as const, b2Dim: 128, b2Bytes: 1 as const, sku: 'emb-16' as const };
    expect(embedRealCode('transfer', { ...base, rep: 'onehot' })).toContain('F.one_hot(token_ids, num_classes=14)');
    expect(embedRealCode('transfer', base)).toContain('nn.Linear(2, 1)');

    expect(embedRealCode('edge-ram', base)).toContain('nn.Embedding(num_embeddings=50_000, embedding_dim=128)');
    expect(embedRealCode('edge-ram', base)).toContain('torch.qint8');
    expect(embedRealCode('edge-ram', { ...base, b2Bytes: 2 })).toContain('torch.float16');

    expect(embedRealCode('sku-mlp', { ...base, sku: 'onehot' })).toContain('nn.Linear(100_000, 128)');
    expect(embedRealCode('sku-mlp', base)).toContain('nn.Embedding(num_embeddings=100_000, embedding_dim=16)');
    expect(embedRealCode('sku-mlp', base)).toContain('nn.Linear(16, 128)');
  });
});
