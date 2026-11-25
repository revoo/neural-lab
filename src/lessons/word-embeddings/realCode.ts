// authored-by: Gemini Antigravity
import type { ScaleMode, W2vBriefId } from './word2vec';

export function word2vecRealCode(
  briefId: W2vBriefId,
  st: { b1K: number; b2Dim: number; b2Win: number; b3Mode: ScaleMode },
): string {
  if (briefId === 'collapse') {
    return [
      `from gensim.models import Word2Vec`,
      `model = Word2Vec(`,
      `    sentences, vector_size=6, window=2, sg=1, hs=0,`,
      `    negative=${st.b1K}, ns_exponent=0.75, epochs=140,`,
      `)`,
    ].join('\n');
  }
  if (briefId === 'analogies') {
    return [
      `from gensim.models import Word2Vec`,
      `model = Word2Vec(`,
      `    sentences, vector_size=${st.b2Dim}, window=${st.b2Win}, sg=1, hs=0,`,
      `    negative=5, ns_exponent=0.75, epochs=140,`,
      `)`,
    ].join('\n');
  }
  if (st.b3Mode === 'full-softmax') {
    return [
      `# Full V=100,000 Softmax: 100,000 dot products per (centre, context) pair`,
      `logits = W_out @ v_center          # shape (100_000,)`,
      `loss = F.cross_entropy(logits, target_id)`,
    ].join('\n');
  }
  const k = Number(st.b3Mode.split('-')[1]);
  return [
    `from gensim.models import Word2Vec`,
    `model = Word2Vec(`,
    `    sentences, vector_size=300, window=5, sg=1, hs=0,`,
    `    negative=${k}, ns_exponent=0.75,  # ${1 + k} dot products / pair`,
    `)`,
  ].join('\n');
}
