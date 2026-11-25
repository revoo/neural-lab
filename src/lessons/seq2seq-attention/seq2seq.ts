// authored-by: Gemini Antigravity
// Pure TypeScript engine for Module 11 Lesson 3: Encoder–Decoder and Early Attention.
// Zero external ML dependencies — exact fixed-vector superposition bottleneck vs
// cross-attention memory benchmark, Bahdanau (additive) vs Luong (dot-product) alignment
// matrix on an English->French adjective-noun reordering sentence, sequential-depth vs
// parallelism model bridging directly to Transformers, and exhaustive Lab.

import { createRng } from '../../lib/rng';

export function softmax(xs: number[]): number[] {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const exps = xs.map((v) => Math.exp(v - m));
  let sum = 0;
  for (let i = 0; i < exps.length; i++) sum += exps[i];
  return exps.map((e) => e / sum);
}

// ------------------------------------------------------------------ 1. Fixed-Vector Bottleneck vs Cross-Attention

export type BottleneckRow = {
  length: number;
  fixedAcc: number;
  attnAcc: number;
  /** Mean squared reconstruction error of the fixed bottleneck */
  fixedMse: number;
};

/**
 * Exact capacity benchmark on variable-length token-copying sequences of length L
 * drawn from a V = 8 token vocabulary (8 orthogonal unit basis vectors):
 * - Fixed-Vector Encoder-Decoder packs all L tokens into ONE single d-dimensional vector
 *   c = sum_{s=1..L} P_s e_{x_s} where P_s in R^{d x V} are unit-column position-tagged
 *   projection matrices (exact superposition in R^d). Decoding position s projects back
 *   via score_v = (P_s^T c)_v and picks argmax_v. As L grows past d, cross-talk between
 *   superposed tokens causes accuracy to collapse!
 * - Cross-Attention stores all L encoder states h_1..h_L in R^d and retrieves exact
 *   context c_t = sum_s alpha_{t,s} h_s via peaked position-content lookup -> 100% at every L!
 */
export function evaluateBottleneck(
  d: number,
  lengths: readonly number[] = [2, 4, 8, 12, 16, 20, 24],
  trials = 40,
  seed = 31,
): BottleneckRow[] {
  const V = 8;
  const maxL = Math.max(...lengths);
  const rng = createRng(seed);

  // Build unit-norm random projection columns P[s][v] in R^d for each position s and token v
  const P: number[][][] = [];
  for (let s = 0; s < maxL; s++) {
    const posMat: number[][] = [];
    for (let v = 0; v < V; v++) {
      const vec = Array.from({ length: d }, () => rng.normal(0, 1));
      let norm = 0;
      for (let i = 0; i < d; i++) norm += vec[i] * vec[i];
      norm = Math.sqrt(norm);
      posMat.push(vec.map((x) => x / norm));
    }
    P.push(posMat);
  }

  // Orthogonal-ish unit vocabulary embeddings in R^d for cross-attention states
  const E: number[][] = [];
  for (let v = 0; v < V; v++) {
    const vec = new Array<number>(d).fill(0);
    if (v < d) {
      vec[v] = 1.0;
    } else {
      for (let i = 0; i < d; i++) vec[i] = rng.normal(0, 1);
      let norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
      for (let i = 0; i < d; i++) vec[i] /= norm;
    }
    E.push(vec);
  }

  const rows: BottleneckRow[] = [];
  for (const L of lengths) {
    let fixedOk = 0;
    let attnOk = 0;
    let totalTokens = 0;
    let sqErr = 0;

    for (let tr = 0; tr < trials; tr++) {
      const seq = Array.from({ length: L }, () => Math.floor(rng.next() * V));

      // Fixed-vector bottleneck: single d-dim context vector c
      const c = new Array<number>(d).fill(0);
      for (let s = 0; s < L; s++) {
        const col = P[s][seq[s]];
        for (let i = 0; i < d; i++) c[i] += col[i];
      }

      // Cross-attention encoder keeps all L hidden states h_s = E[seq[s]]
      const H = seq.map((tok) => E[tok]);

      for (let s = 0; s < L; s++) {
        totalTokens++;
        // Decode from fixed vector c
        let bestV = 0;
        let bestScore = -Infinity;
        for (let v = 0; v < V; v++) {
          const col = P[s][v];
          let dot = 0;
          for (let i = 0; i < d; i++) dot += col[i] * c[i];
          if (dot > bestScore) {
            bestScore = dot;
            bestV = v;
          }
          const target = v === seq[s] ? 1 : 0;
          sqErr += (dot - target) ** 2;
        }
        if (bestV === seq[s]) fixedOk++;

        // Decode with cross-attention (peaked alignment focused on position s, tau = 0.2)
        const logits = new Array<number>(L);
        for (let j = 0; j < L; j++) {
          logits[j] = j === s ? 6.0 : 0.0;
        }
        const alpha = softmax(logits);
        const ctx = new Array<number>(d).fill(0);
        for (let j = 0; j < L; j++) {
          const a = alpha[j];
          const hj = H[j];
          for (let i = 0; i < d; i++) ctx[i] += a * hj[i];
        }
        let bestAttnV = 0;
        let bestAttnScore = -Infinity;
        for (let v = 0; v < V; v++) {
          let dot = 0;
          for (let i = 0; i < d; i++) dot += E[v][i] * ctx[i];
          if (dot > bestAttnScore) {
            bestAttnScore = dot;
            bestAttnV = v;
          }
        }
        if (bestAttnV === seq[s]) attnOk++;
      }
    }

    rows.push({
      length: L,
      fixedAcc: fixedOk / totalTokens,
      attnAcc: attnOk / totalTokens,
      fixedMse: sqErr / (totalTokens * V),
    });
  }

  return rows;
}

// ------------------------------------------------------------------ 2. Bahdanau vs Luong Cross-Attention Alignment

export type ScoreKind = 'luong' | 'bahdanau';

export const SOURCE_TOKENS = ['the', 'black', 'cat', 'saw', 'the', 'red', 'bird'] as const;
export const TARGET_TOKENS = ['le', 'chat', 'noir', 'vit', 'l’oiseau', 'rouge'] as const;

/**
 * True gold alignment index in SOURCE_TOKENS for each TARGET_TOKENS step:
 * le -> the(0), chat -> cat(2), noir -> black(1), vit -> saw(3), l'oiseau -> bird(6), rouge -> red(5).
 * Notice TWO clean adjective-noun crossovers: (black cat -> chat noir) and (red bird -> oiseau rouge)!
 */
export const GOLD_ALIGNMENT: readonly number[] = [0, 2, 1, 3, 6, 5];

// 6D semantic+role encoder hidden states h_s: [det, animal1, color1, verb, animal2, color2]
const ENC_STATES: number[][] = [
  [1.0, 0.1, 0.0, 0.0, 0.0, 0.0], // the (0)
  [0.0, 0.2, 1.0, 0.0, 0.0, 0.1], // black (1)
  [0.1, 1.0, 0.2, 0.1, 0.1, 0.0], // cat (2)
  [0.0, 0.1, 0.0, 1.0, 0.1, 0.0], // saw (3)
  [0.8, 0.0, 0.0, 0.0, 0.2, 0.0], // the (4)
  [0.0, 0.0, 0.1, 0.0, 0.2, 1.0], // red (5)
  [0.0, 0.1, 0.0, 0.1, 1.0, 0.2], // bird (6)
];

// 6D decoder queries s_t at each French target step
const DEC_QUERIES: number[][] = [
  [1.0, 0.0, 0.0, 0.0, -0.2, 0.0], // le -> the(0)
  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // chat -> cat(2)
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  // noir -> black(1)
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  // vit -> saw(3)
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  // l'oiseau -> bird(6)
  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  // rouge -> red(5)
];

export type AlignmentResult = {
  /** Raw pre-softmax alignment scores e_{t,s} (scaled by 1/tau) */
  rawScores: number[][];
  /** Softmax attention weights alpha_{t,s}, rows sum to 1 */
  weights: number[][];
  /** Mean weight placed on the true gold source token across all 6 target words */
  meanGoldWeight: number;
  /** Fraction of target words whose argmax source token matches gold alignment */
  argmaxAcc: number;
  /** Max absolute pre-softmax score |e_{t,s}| (checks numerical stability) */
  maxAbsLogit: number;
  /** Context vectors c_t = sum_s alpha_{t,s} h_s */
  contexts: number[][];
};

/**
 * Compute exact cross-attention alignment matrix for either:
 * - Luong (dot-product): e_{t,s} = (s_t · h_s) / tau
 * - Bahdanau (additive): e_{t,s} = (v^T tanh(W_s s_t + W_h h_s)) / tau
 *   with clean matched-channel alignment weights so both produce real interpretable alignments!
 */
export function computeCrossAttention(kind: ScoreKind, tau: number): AlignmentResult {
  const T = TARGET_TOKENS.length;
  const S = SOURCE_TOKENS.length;
  const D = 6;
  const rawScores: number[][] = [];
  const weights: number[][] = [];
  const contexts: number[][] = [];

  let goldSum = 0;
  let argmaxOk = 0;
  let maxAbsLogit = 0;

  for (let t = 0; t < T; t++) {
    const q = DEC_QUERIES[t];
    const rowScores = new Array<number>(S);
    for (let s = 0; s < S; s++) {
      const h = ENC_STATES[s];
      let score = 0;
      if (kind === 'luong') {
        for (let k = 0; k < D; k++) score += q[k] * h[k];
      } else {
        // Genuine Bahdanau (2015) additive MLP: v^T tanh(W_s s_t + W_h h_s + b)
        // With coincidence threshold b = -1.2, a hidden neuron fires strongly (+0.83)
        // only when BOTH query q_k and encoder state h_k are active in channel k!
        for (let k = 0; k < D; k++) {
          score += 0.55 * (Math.tanh(1.5 * (q[k] + h[k] - 1.2)) + 0.9468);
        }
      }
      const scaled = score / tau;
      rowScores[s] = scaled;
      maxAbsLogit = Math.max(maxAbsLogit, Math.abs(scaled));
    }

    const alpha = softmax(rowScores);
    rawScores.push(rowScores);
    weights.push(alpha);

    goldSum += alpha[GOLD_ALIGNMENT[t]];
    let bestS = 0;
    for (let s = 1; s < S; s++) if (alpha[s] > alpha[bestS]) bestS = s;
    if (bestS === GOLD_ALIGNMENT[t]) argmaxOk++;

    const ctx = new Array<number>(D).fill(0);
    for (let s = 0; s < S; s++) {
      for (let k = 0; k < D; k++) ctx[k] += alpha[s] * ENC_STATES[s][k];
    }
    contexts.push(ctx);
  }

  return {
    rawScores,
    weights,
    meanGoldWeight: goldSum / T,
    argmaxAcc: argmaxOk / T,
    maxAbsLogit,
    contexts,
  };
}

// ------------------------------------------------------------------ 3. Sequential Depth vs Parallelism Calculator

export type SeqArchId = 'fixed-lstm' | 'lstm-attn' | 'transformer';

export type ArchMetrics = {
  arch: SeqArchId;
  label: string;
  seqLen: number;
  /** Sequential dependent steps during one forward training pass */
  seqDepth: number;
  /** Maximum dependency path length between any two tokens */
  maxPathLen: number;
  /** Simulated GPU training forward-step latency (ms) combining sequential depth + O(L^2) parallel attention */
  trainLatencyMs: number;
  /** Relative attention matrix memory entries (0 for fixed LSTM, L^2 for attention models) */
  attnMatrixEntries: number;
};

/**
 * Clean first-principles hardware timing model for training forward pass on sequence length L:
 * - Each sequential recurrent step takes ~0.08 ms of kernel launch + dependency serialisation.
 * - Cross-attention adds L^2 parallel dot products (~0.000012 ms * L^2 on a parallel GPU).
 * - Transformer replaces the O(L) recurrent loop with 6 parallel layers (~1.2 ms base + 0.000025 ms * L^2).
 */
export function computeArchMetrics(arch: SeqArchId, L: number): ArchMetrics {
  if (arch === 'fixed-lstm') {
    return {
      arch,
      label: 'Fixed-Vector LSTM',
      seqLen: L,
      seqDepth: 2 * L,
      maxPathLen: L,
      trainLatencyMs: Number((0.08 * (2 * L)).toFixed(2)),
      attnMatrixEntries: 0,
    };
  }
  if (arch === 'lstm-attn') {
    return {
      arch,
      label: 'LSTM + Cross-Attention',
      seqLen: L,
      seqDepth: 2 * L,
      maxPathLen: 1,
      trainLatencyMs: Number((0.08 * (2 * L) + 1.2e-5 * L * L).toFixed(2)),
      attnMatrixEntries: L * L,
    };
  }
  return {
    arch,
    label: 'Transformer (All Attention)',
    seqLen: L,
    seqDepth: 1,
    maxPathLen: 1,
    trainLatencyMs: Number((1.2 + 2.5e-5 * L * L).toFixed(2)),
    attnMatrixEntries: L * L,
  };
}

// ------------------------------------------------------------------ 4. Lab: Break the Bottleneck (Exhaustive)

export type SeqBriefId = 'bottleneck' | 'align' | 'latency';

export const LAB_BN_ARCHS = ['fixed', 'cross-attn'] as const;
export const LAB_BN_DS = [8, 16, 32] as const;

export function evalSeqBrief1(arch: 'fixed' | 'cross-attn', d: number) {
  const row = evaluateBottleneck(d).find((r) => r.length === 20)!;
  const acc = arch === 'fixed' ? row.fixedAcc : row.attnAcc;
  const pass = acc >= 0.95;
  return { arch, d, acc, pass };
}

export const LAB_ALIGN_KINDS: ScoreKind[] = ['luong', 'bahdanau'];
export const LAB_ALIGN_TAUS = [0.05, 0.15, 0.40, 1.0] as const;

export function evalSeqBrief2(kind: ScoreKind, tau: number) {
  const res = computeCrossAttention(kind, tau);
  // Require sharp gold alignment >= 0.85, 100% argmax match, AND safe logits <= 15 (no overflow!)
  const pass = res.meanGoldWeight >= 0.85 && res.argmaxAcc >= 0.99 && res.maxAbsLogit <= 15.0;
  return {
    kind,
    tau,
    meanGoldWeight: res.meanGoldWeight,
    argmaxAcc: res.argmaxAcc,
    maxAbsLogit: res.maxAbsLogit,
    pass,
  };
}

export const LAB_LAT_ARCHS: SeqArchId[] = ['fixed-lstm', 'lstm-attn', 'transformer'];
export const LAB_LAT_LS = [64, 512, 2048] as const;

export function evalSeqBrief3(arch: SeqArchId) {
  // SLA on L = 512 training documents: <= 20 ms/step AND O(1) direct token path!
  const m = computeArchMetrics(arch, 512);
  const pass = m.trainLatencyMs <= 20.0 && m.maxPathLen === 1;
  return { ...m, pass };
}

export function seqLabExhaustive(brief: SeqBriefId) {
  if (brief === 'bottleneck') {
    const all: ReturnType<typeof evalSeqBrief1>[] = [];
    for (const a of LAB_BN_ARCHS) {
      for (const d of LAB_BN_DS) {
        all.push(evalSeqBrief1(a, d));
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => a.d - b.d);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0]
        ? `${passing[0].arch} d=${passing[0].d} (${(100 * passing[0].acc).toFixed(1)}%)`
        : 'none',
    };
  }
  if (brief === 'align') {
    const all: ReturnType<typeof evalSeqBrief2>[] = [];
    for (const k of LAB_ALIGN_KINDS) {
      for (const tau of LAB_ALIGN_TAUS) {
        all.push(evalSeqBrief2(k, tau));
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => b.meanGoldWeight - a.meanGoldWeight);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0]
        ? `${passing[0].kind} τ=${passing[0].tau} (gold ${(100 * passing[0].meanGoldWeight).toFixed(1)}%, max|e|=${passing[0].maxAbsLogit.toFixed(1)})`
        : 'none',
    };
  }
  const all = LAB_LAT_ARCHS.map((a) => evalSeqBrief3(a));
  const passing = all.filter((r) => r.pass).sort((a, b) => a.trainLatencyMs - b.trainLatencyMs);
  return {
    total: all.length,
    passingCount: passing.length,
    bestLabel: passing[0]
      ? `${passing[0].label} (${passing[0].trainLatencyMs.toFixed(2)} ms, depth=${passing[0].seqDepth})`
      : 'none',
  };
}
