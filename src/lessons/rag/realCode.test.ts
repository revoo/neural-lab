// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { ragLabRealCode } from './realCode';

describe('rag lab live retrieval code', () => {
  it('emits lexical, vector-only, and hybrid search code matching controls', () => {
    expect(ragLabRealCode(1, 0, 2)).toContain('bm25_index.search(query, top_k=2)');
    expect(ragLabRealCode(2, 1, 4)).toContain('vector_db.query(embed(query), top_k=4)');
    const hyb = ragLabRealCode(2, 0.25, 3);
    expect(hyb).toContain('sentences_per_chunk=2');
    expect(hyb).toContain('alpha=0.25');
    expect(hyb).toContain('top_k=3');
  });
});
