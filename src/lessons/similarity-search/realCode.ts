// authored-by: Gemini Antigravity
import type { AnnBriefId, CompFormat } from './ann';

export function faissRealCode(
  briefId: AnnBriefId,
  st: { nlist: number; nprobe: number; bits: number; tables: number; fmt: CompFormat },
): string {
  if (briefId === 'ivf') {
    return [
      `import faiss`,
      `quantizer = faiss.IndexFlatL2(16)`,
      `index = faiss.IndexIVFFlat(quantizer, 16, ${st.nlist})`,
      `index.train(X); index.add(X)`,
      `index.nprobe = ${st.nprobe}`,
      `D, I = index.search(Q, k=5)`,
    ].join('\n');
  }

  if (briefId === 'lsh') {
    return [
      `# Multi-table Random-Hyperplane LSH (L=${st.tables} tables × ${st.bits} bits each)`,
      `planes = rng.normal(size=(${st.tables}, ${st.bits}, 16))`,
      `codes = (X @ planes.transpose(0, 2, 1)) >= 0  # L hash keys per vector`,
    ].join('\n');
  }

  if (st.fmt === 'fp32') return `index = faiss.IndexFlatIP(1536)  # 4 B/dim -> 57.22 GiB for 10M vecs`;
  if (st.fmt === 'fp16') return `index = faiss.IndexScalarQuantizer(1536, faiss.ScalarQuantizer.QT_fp16)`;
  if (st.fmt === 'int8') return `index = faiss.IndexScalarQuantizer(1536, faiss.ScalarQuantizer.QT_8bit)`;
  if (st.fmt === 'pq64') return `index = faiss.IndexPQ(1536, 64, 8)  # 64 sub-vectors × 8-bit centroid ID`;
  return `index = faiss.IndexBinaryFlat(1536)  # 1 bit/dim -> 192 B/vector`;
}
