// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { word2vecRealCode } from './realCode';

describe('word-embeddings: real Gensim Word2Vec code generator', () => {
  it('emits accurate Gensim Word2Vec and Full Softmax code across all three briefs', () => {
    const base = { b1K: 5, b2Dim: 3, b2Win: 3, b3Mode: 'sgns-5' as const };
    expect(word2vecRealCode('collapse', base)).toContain('negative=5');
    expect(word2vecRealCode('collapse', base)).toContain('ns_exponent=0.75');

    const c2 = word2vecRealCode('analogies', base);
    expect(c2).toContain('vector_size=3, window=3');

    expect(word2vecRealCode('scale', { ...base, b3Mode: 'full-softmax' })).toContain('F.cross_entropy');
    expect(word2vecRealCode('scale', base)).toContain('negative=5');
  });
});
