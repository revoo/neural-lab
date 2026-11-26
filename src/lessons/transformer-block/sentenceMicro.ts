// authored-by: Gemini Antigravity
/**
 * 4-token Transformer Sentence Microscope (d_model = 4, n_heads = 2, d_head = 2, d_ff = 8).
 *
 * Runs a real 1-block pre-norm causal Transformer on ["The", "cat", "sat", "on"] over a
 * 6-word vocabulary ["The", "cat", "sat", "on", "mat", "dog"] with target "mat" at position 3.
 *
 * Every matrix multiplication, LayerNorm, causal attention head, ReLU feed-forward layer,
 * unembedding projection, softmax probability, and cross-entropy logit gradient (p - y) is
 * computed directly so learners can inspect the 4-number residual stream channel-by-channel
 * and ablate Attention, Residual skips, or Positional encodings live.
 */

export const VOCAB = ['The', 'cat', 'sat', 'on', 'mat', 'dog'] as const;
export type VocabWord = (typeof VOCAB)[number];

export const PROMPT_IDS = [0, 1, 2, 3] as const; // ["The", "cat", "sat", "on"]
export const TARGET_AT_LAST = 4; // "mat"
export const NEXT_TARGETS = [1, 2, 3, 4] as const; // expected next token at positions 0..3

export const STREAM_LABELS = [
  'c₀ · subject',
  'c₁ · verb',
  'c₂ · prep / end',
  'c₃ · position',
] as const;

export type MicroOptions = {
  useAttention: boolean;
  useResidual: boolean;
  usePositions: boolean;
  /** Optional custom token sequence (defaults to ["The", "cat", "sat", "on"]) */
  tokens?: readonly number[];
};

export type TokenTrace = {
  pos: number;
  tokenId: number;
  word: VocabWord;
  targetId: number;
  targetWord: VocabWord;
  emb: number[];
  posVec: number[];
  x0: number[];       // emb + pos (if enabled)
  ln1: number[];      // LayerNorm(x0)
  attnOut: number[];  // MultiHeadAttention(ln1) (0 if ablated)
  x1: number[];       // (useResidual ? x0 : 0) + attnOut
  ln2: number[];      // LayerNorm(x1)
  mlpHidden: number[]; // ReLU(W1 * ln2 + b1) (length 8)
  mlpOut: number[];   // W2 * mlpHidden + b2 (length 4)
  x2: number[];       // (useResidual ? x1 : 0) + mlpOut
  logits: number[];   // W_U * x2 + b_U (length 6)
  probs: number[];    // softmax(logits) (length 6)
  dLogits: number[];  // exact d(CE)/d(logits) = p - y (length 6)
  loss: number;       // -log(probs[targetId])
};

export type SentenceTrace = {
  options: Required<Pick<MicroOptions, 'useAttention' | 'useResidual' | 'usePositions'>>;
  tokens: TokenTrace[];
  /** attnWeights[head][q][k]: strictly causal (0 for k > q), rows sum to 1 */
  attnWeights: [number[][], number[][]];
};

// ---- Clean, interpretable 1-block Transformer weights (d = 4, H = 2, d_head = 2, d_ff = 8)

// Token embeddings E (6 x 4): [subject, verb, prep/function, reserved-for-pos]
const EMB: readonly (readonly number[])[] = [
  [0.0, 0.0, -1.1, 0.05],  // 0: The
  [1.2, -0.2, 0.0, 0.05],  // 1: cat  (strong +subject)
  [-0.1, 1.2, 0.0, -0.05], // 2: sat  (strong +verb)
  [0.0, 0.0, 1.2, 0.05],   // 3: on   (strong +preposition)
  [0.9, 0.9, 0.9, 0.0],    // 4: mat
  [-1.2, -0.2, 0.0, 0.05], // 5: dog  (strong -subject)
];

// Positional embeddings P (4 x 4): injects distinct order ramp into c3
const POS: readonly (readonly number[])[] = [
  [-0.08, -0.08, -0.15, -1.35], // pos 0 ("The")
  [0.08, -0.08, 0.0, -0.45],    // pos 1 ("cat")
  [-0.08, 0.08, 0.08, 0.45],    // pos 2 ("sat")
  [0.0, 0.0, 0.2, 1.35],        // pos 3 ("on")
];

// Head 1 (Subject head): "on" (high c2, high c3) queries for "cat" (high c0, early-mid c3)
const WQ1: readonly (readonly number[])[] = [
  [0.0, 0.0, 1.0, 1.15],
  [0.0, 0.0, 0.2, 1.2],
];
const WK1: readonly (readonly number[])[] = [
  [1.4, -0.2, -0.35, -0.2],
  [0.6, -0.15, -0.3, -0.25],
];
const WV1: readonly (readonly number[])[] = [
  [1.35, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.35],
];
const WO1: readonly (readonly number[])[] = [
  [1.15, 0.0],  // writes fetched subject into c0
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.45],
];

// Head 2 (Verb / Phrase head): "on" (high c2, high c3) queries for "sat" (high c1, positive c3)
const WQ2: readonly (readonly number[])[] = [
  [0.0, 0.0, 0.95, 1.2],
  [0.0, 0.0, 0.2, 1.3],
];
const WK2: readonly (readonly number[])[] = [
  [-0.2, 1.35, -0.35, 0.45],
  [-0.15, 0.7, -0.25, 0.55],
];
const WV2: readonly (readonly number[])[] = [
  [0.0, 1.35, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.35],
];
const WO2: readonly (readonly number[])[] = [
  [0.0, 0.0],
  [1.15, 0.0],  // writes fetched verb into c1
  [0.0, 0.0],
  [0.0, 0.45],
];

// Feed-Forward Network: 4 -> 8 (ReLU) -> 4
// Neuron 0: 4-way conjunction (subject + verb + prep + late position) -> fires strongly ONLY when
//           attention gathered "cat" & "sat", residual preserved "on", AND position confirmed end-of-phrase!
const W1: readonly (readonly number[])[] = [
  [0.95, 0.95, 1.05, 1.45], // n0: cat + sat + on + late pos -> "mat" trigger!
  [0.0, 0.0, -1.4, -0.7],   // n1: "The" at start -> predicts "cat"
  [1.3, -0.4, -0.3, 0.0],   // n2: "cat" -> predicts "sat"
  [-0.3, 1.3, -0.3, 0.2],   // n3: "sat" -> predicts "on"
  [0.0, 0.0, 1.2, -0.3],    // n4: bare "on" without context
  [-1.1, 0.9, 0.9, 0.6],    // n5: dog + sat + on
  [0.6, 0.6, -0.8, 0.4],    // n6: context without residual "on"
  [0.0, 0.0, 0.3, 0.9],     // n7: generic late-position prior
];
const B1: readonly number[] = [-1.45, -0.55, -0.45, -0.45, -0.55, -1.4, -0.85, -0.65];

const W2: readonly (readonly number[])[] = [
  [1.15, 1.1, -1.1, -0.2, -0.2, -0.8, 0.15, 0.0],   // c0 output
  [1.15, -0.2, 2.1, -0.25, 0.1, 0.55, 0.15, 0.0],   // c1 output
  [1.20, -0.25, -0.2, 1.2, 0.25, 0.6, -0.45, 0.15], // c2 output
  [0.95, -0.2, 0.0, 0.15, -0.15, 0.35, 0.1, 0.25],  // c3 output
];
const B2: readonly number[] = [0.0, 0.0, 0.0, 0.0];

// Unembedding matrix W_U (6 x 4) and bias b_U (6)
const WU: readonly (readonly number[])[] = [
  [-0.35, -0.35, -0.75, -0.3], // 0: The
  [0.85, -0.45, -0.45, -0.15], // 1: cat
  [-0.3, 0.9, -0.4, 0.1],      // 2: sat
  [-0.3, -0.25, 0.65, 0.15],   // 3: on
  [0.72, 0.72, 0.78, 0.82],    // 4: mat (requires strong +c0, +c1, +c2, +c3 together!)
  [-0.75, -0.3, -0.35, 0.1],   // 5: dog
];
const BU: readonly number[] = [-0.25, 0.0, 0.0, 0.15, -2.55, -0.15];

// ---- Exact linear algebra primitives

export function layerNorm(v: readonly number[], eps = 1e-5): number[] {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const variance = v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length;
  const invStd = 1 / Math.sqrt(variance + eps);
  return v.map((x) => (x - mean) * invStd);
}

function matVec(M: readonly (readonly number[])[], v: readonly number[]): number[] {
  return M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
}

function addVec(a: readonly number[], b: readonly number[]): number[] {
  return a.map((v, i) => v + b[i]);
}

export function softmax(logits: readonly number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((z) => (Number.isFinite(z) ? Math.exp(z - m) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function crossEntropyLoss(logits: readonly number[], targetId: number): number {
  const m = Math.max(...logits);
  const logSumExp = m + Math.log(logits.reduce((s, z) => s + Math.exp(z - m), 0));
  return logSumExp - logits[targetId];
}

function runCausalHead(
  ln1s: readonly (readonly number[])[],
  WQ: readonly (readonly number[])[],
  WK: readonly (readonly number[])[],
  WV: readonly (readonly number[])[],
  WO: readonly (readonly number[])[],
): { weights: number[][]; out: number[][] } {
  const T = ln1s.length;
  const Q = ln1s.map((x) => matVec(WQ, x));
  const K = ln1s.map((x) => matVec(WK, x));
  const V = ln1s.map((x) => matVec(WV, x));
  const scale = 1 / Math.SQRT2; // 1 / sqrt(d_head = 2)

  const weights: number[][] = [];
  const out: number[][] = [];
  for (let q = 0; q < T; q++) {
    const scores = Array.from({ length: T }, (_, k) =>
      k > q ? -Infinity : (Q[q][0] * K[k][0] + Q[q][1] * K[k][1]) * scale,
    );
    const w = softmax(scores);
    weights.push(w);
    const ctx = [0, 0];
    for (let k = 0; k <= q; k++) {
      ctx[0] += w[k] * V[k][0];
      ctx[1] += w[k] * V[k][1];
    }
    out.push(matVec(WO, ctx));
  }
  return { weights, out };
}

/** Run the complete 4-token causal Transformer block and return every intermediate stream vector. */
export function traceSentence(opts: Partial<MicroOptions> = {}): SentenceTrace {
  const useAttention = opts.useAttention ?? true;
  const useResidual = opts.useResidual ?? true;
  const usePositions = opts.usePositions ?? true;
  const tokenIds = opts.tokens ? [...opts.tokens] : [...PROMPT_IDS];
  const T = tokenIds.length;

  const embs = tokenIds.map((id) => [...EMB[id]]);
  const posVecs = tokenIds.map((_, t) => (usePositions ? [...POS[Math.min(t, POS.length - 1)]] : [0, 0, 0, 0]));
  const x0s = embs.map((e, t) => addVec(e, posVecs[t]));
  const ln1s = x0s.map((x) => layerNorm(x));

  const h1 = runCausalHead(ln1s, WQ1, WK1, WV1, WO1);
  const h2 = runCausalHead(ln1s, WQ2, WK2, WV2, WO2);

  const tokens: TokenTrace[] = [];
  for (let t = 0; t < T; t++) {
    const attnOut = useAttention ? addVec(h1.out[t], h2.out[t]) : [0, 0, 0, 0];
    const x1 = useResidual ? addVec(x0s[t], attnOut) : [...attnOut];
    const ln2 = layerNorm(x1);
    const mlpHidden = matVec(W1, ln2).map((z, i) => Math.max(0, z + B1[i]));
    const mlpOut = addVec(matVec(W2, mlpHidden), B2);
    const x2 = useResidual ? addVec(x1, mlpOut) : [...mlpOut];
    const logits = addVec(matVec(WU, x2), BU);
    const probs = softmax(logits);
    const targetId = NEXT_TARGETS[Math.min(t, NEXT_TARGETS.length - 1)];
    const dLogits = probs.map((p, i) => p - (i === targetId ? 1 : 0));
    const loss = crossEntropyLoss(logits, targetId);

    tokens.push({
      pos: t,
      tokenId: tokenIds[t],
      word: VOCAB[tokenIds[t]],
      targetId,
      targetWord: VOCAB[targetId],
      emb: embs[t],
      posVec: posVecs[t],
      x0: x0s[t],
      ln1: ln1s[t],
      attnOut,
      x1,
      ln2,
      mlpHidden,
      mlpOut,
      x2,
      logits,
      probs,
      dLogits,
      loss,
    });
  }

  return {
    options: { useAttention, useResidual, usePositions },
    tokens,
    attnWeights: [h1.weights, h2.weights],
  };
}
