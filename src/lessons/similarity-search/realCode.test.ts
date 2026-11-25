// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { faissRealCode } from './realCode';

describe('similarity-search: real FAISS code generator', () => {
  it('emits matching FAISS index code for IVF, LSH, and quantization formats', () => {
    const base = { nlist: 32, nprobe: 6, bits: 6, tables: 16, fmt: 'int8' as const };
    const c1 = faissRealCode('ivf', base);
    expect(c1).toContain('faiss.IndexIVFFlat(quantizer, 16, 32)');
    expect(c1).toContain('index.nprobe = 6');

    expect(faissRealCode('lsh', base)).toContain('L=16 tables × 6 bits');

    expect(faissRealCode('ram', { ...base, fmt: 'fp32' })).toContain('IndexFlatIP(1536)');
    expect(faissRealCode('ram', base)).toContain('QT_8bit');
    expect(faissRealCode('ram', { ...base, fmt: 'pq64' })).toContain('IndexPQ(1536, 64, 8)');
    expect(faissRealCode('ram', { ...base, fmt: 'binary' })).toContain('IndexBinaryFlat(1536)');
  });
});
