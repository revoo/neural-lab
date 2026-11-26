// authored-by: Claude Opus 5.5
/**
 * Parameter counting for a transformer whose feed-forward blocks are mixtures of experts, with the
 * architecture of Mixtral 8x7B as the starting point (Table 1 of arXiv 2401.04088): width 4096,
 * 32 layers, head size 128, 32 query heads and 8 key/value heads, SwiGLU experts of hidden size
 * 14,336, vocabulary 32,000, 8 experts with 2 active. Counting this way gives 46.7B parameters in
 * all and 12.9B active per token; the paper rounds them to 47B and 13B (tested in `sizes.test.ts`).
 * Layer norms (a few hundred thousand weights) are left out.
 */
export const MIXTRAL = { dim: 4096, layers: 32, headDim: 128, heads: 32, kvHeads: 8, hidden: 14336, vocab: 32000, experts: 8, topK: 2 };
export type Arch = typeof MIXTRAL;

export function countParams(a: Arch) {
  const attn = a.dim * a.heads * a.headDim * 2 + a.dim * a.kvHeads * a.headDim * 2; // Wq, Wo; Wk, Wv
  const expert = 3 * a.dim * a.hidden; // SwiGLU: gate, up, down
  const router = a.dim * a.experts;
  const embed = 2 * a.vocab * a.dim; // input table and output layer (not shared in Mixtral)
  const total = a.layers * (attn + a.experts * expert + router) + embed;
  const active = a.layers * (attn + a.topK * expert + router) + embed;
  return { attn, expert, router, embed, total, active };
}

/** Memory to hold the weights at 2 bytes each (bf16), in GB (10⁹ bytes). */
export const weightGB = (params: number) => (2 * params) / 1e9;
/** Forward-pass compute per token: about 2 FLOPs per active weight (one multiply, one add). */
export const flopsPerToken = (active: number) => 2 * active;
