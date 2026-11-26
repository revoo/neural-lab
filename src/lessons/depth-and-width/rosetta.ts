// authored-by: Gemini Antigravity
/**
 * "It's All Still W*x + b" — Architecture Rosetta Stone.
 *
 * Represents five canonical neural building blocks on the same 6-element input vector x
 * as an exact 6 x 6 effective linear map y = W_eff(x) * x:
 *   1. Dense MLP:        36 independent weights.
 *   2. 1D Convolution:   Toeplitz band matrix with kernel [-1, 0, +1] (3 shared weights).
 *   3. Unrolled RNN:     3 time steps of width 2, block-lower-triangular (8 shared weights).
 *   4. Self-Attention:   row-stochastic W(x) = softmax(Q(x) K(x)^T / sqrt(2)) (8 Q/K weights, dynamic!).
 *   5. LoRA (rank 1):    frozen W_0 + outer(a, b) (12 trainable weights vs 36).
 */

export type ArchId = 'dense' | 'conv' | 'rnn' | 'attention' | 'lora';

export type InputPresetId = 'edge' | 'spike' | 'wave';

export const INPUT_PRESETS: Record<InputPresetId, { label: string; x: readonly [number, number, number, number, number, number] }> = {
  edge: { label: 'Step edge', x: [0.2, 0.9, 1.0, 0.3, -0.4, -0.8] },
  spike: { label: 'Single spike', x: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] },
  wave: { label: 'Alternating', x: [0.8, -0.8, 0.8, -0.8, 0.8, -0.8] },
};

export type RosettaView = {
  arch: ArchId;
  title: string;
  shortRule: string;
  /** 6 x 6 effective linear matrix W_eff */
  W: number[][];
  /** Optional cell tags (e.g. shared kernel group id or block label) */
  tags: (string | null)[][];
  /** Optional LoRA factors */
  lora?: { W0: number[][]; a: number[]; b: number[]; dW: number[][] };
  /** Output vector y = W * x */
  y: number[];
  /** Number of free/trainable parameters defining W_eff (or the trainable adapter for LoRA) */
  freeParams: number;
  /** Total cells in a 6 x 6 matrix */
  denseParams: number;
  /** Non-zero multiply-adds in W * x */
  nonZeroMults: number;
  /** Whether W_eff itself depends on the input vector x */
  inputDependent: boolean;
};

const DENSE_W: number[][] = [
  [0.65, -0.30, 0.45, -0.20, 0.15, -0.50],
  [-0.25, 0.80, 0.35, -0.40, 0.20, 0.10],
  [0.40, -0.55, 0.70, 0.25, -0.35, 0.30],
  [-0.30, 0.20, -0.45, 0.85, 0.50, -0.15],
  [0.15, 0.35, -0.25, 0.40, 0.75, -0.60],
  [-0.45, 0.10, 0.30, -0.35, 0.55, 0.65],
];

export const CONV_KERNEL: readonly [number, number, number] = [-1, 0, 1];

// 2 x 2 RNN input and recurrence blocks (3 steps of width 2 -> 6-dim stacked state)
const RNN_WX: [[number, number], [number, number]] = [
  [0.8, -0.2],
  [0.3, 0.7],
];
const RNN_WH: [[number, number], [number, number]] = [
  [0.6, 0.25],
  [-0.2, 0.55],
];

// Self-attention Q and K projections from 2D token feature u_i = [x_i, pos_i]
const ATTN_WQ: [[number, number], [number, number]] = [
  [2.2, 0.35],
  [0.0, 0.65],
];
const ATTN_WK: [[number, number], [number, number]] = [
  [2.2, -0.25],
  [0.0, 0.65],
];

// LoRA rank-1 adapter vectors: W_eff = DENSE_W + a * b^T
export const LORA_A: readonly number[] = [0.6, -0.4, 0.8, 0.2, -0.5, 0.3];
export const LORA_B: readonly number[] = [0.5, 0.7, -0.3, 0.4, -0.6, 0.2];

export function matVec(W: readonly (readonly number[])[], x: readonly number[]): number[] {
  return W.map((row) => row.reduce((acc, w, j) => acc + w * x[j], 0));
}

function mat2Mul(A: readonly (readonly number[])[], B: readonly (readonly number[])[]): [[number, number], [number, number]] {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
  ];
}

/** Direct 1D same-padded convolution with kernel [-1, 0, +1] (zero outside [0, 5]). */
export function directConv1D(x: readonly number[]): number[] {
  const n = x.length;
  return Array.from({ length: n }, (_, i) => {
    const left = i > 0 ? x[i - 1] : 0;
    const mid = x[i];
    const right = i + 1 < n ? x[i + 1] : 0;
    return CONV_KERNEL[0] * left + CONV_KERNEL[1] * mid + CONV_KERNEL[2] * right;
  });
}

/** Direct step-by-step unrolled linear RNN: h[t] = W_x x[t] + W_h h[t-1] for t = 0, 1, 2. */
export function directRnnUnroll(x: readonly number[]): number[] {
  let h0 = 0;
  let h1 = 0;
  const out: number[] = [];
  for (let t = 0; t < 3; t++) {
    const x0 = x[2 * t];
    const x1 = x[2 * t + 1];
    const n0 = RNN_WX[0][0] * x0 + RNN_WX[0][1] * x1 + RNN_WH[0][0] * h0 + RNN_WH[0][1] * h1;
    const n1 = RNN_WX[1][0] * x0 + RNN_WX[1][1] * x1 + RNN_WH[1][0] * h0 + RNN_WH[1][1] * h1;
    h0 = n0;
    h1 = n1;
    out.push(h0, h1);
  }
  return out;
}

function buildConvMatrix(): { W: number[][]; tags: (string | null)[][] } {
  const W = Array.from({ length: 6 }, () => new Array(6).fill(0));
  const tags: (string | null)[][] = Array.from({ length: 6 }, () => new Array(6).fill(null));
  for (let i = 0; i < 6; i++) {
    if (i > 0) {
      W[i][i - 1] = CONV_KERNEL[0];
      tags[i][i - 1] = 'k₋';
    }
    W[i][i] = CONV_KERNEL[1];
    tags[i][i] = 'k₀';
    if (i < 5) {
      W[i][i + 1] = CONV_KERNEL[2];
      tags[i][i + 1] = 'k₊';
    }
  }
  return { W, tags };
}

function buildRnnMatrix(): { W: number[][]; tags: (string | null)[][] } {
  const W = Array.from({ length: 6 }, () => new Array(6).fill(0));
  const tags: (string | null)[][] = Array.from({ length: 6 }, () => new Array(6).fill(null));
  const whWx = mat2Mul(RNN_WH, RNN_WX);
  const wh2Wx = mat2Mul(RNN_WH, whWx);
  const place = (br: number, bc: number, block: readonly (readonly number[])[], tag: string) => {
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        W[2 * br + r][2 * bc + c] = block[r][c];
        tags[2 * br + r][2 * bc + c] = tag;
      }
    }
  };
  place(0, 0, RNN_WX, 'Wₓ');
  place(1, 0, whWx, 'WₕWₓ');
  place(1, 1, RNN_WX, 'Wₓ');
  place(2, 0, wh2Wx, 'Wₕ²Wₓ');
  place(2, 1, whWx, 'WₕWₓ');
  place(2, 2, RNN_WX, 'Wₓ');
  return { W, tags };
}

function buildAttentionMatrix(x: readonly number[]): { W: number[][]; tags: (string | null)[][] } {
  const pos = [0, 1, 2, 3, 4, 5].map((i) => (i - 2.5) / 2.5);
  const Q = x.map((xi, i) => [
    ATTN_WQ[0][0] * xi + ATTN_WQ[0][1] * pos[i],
    ATTN_WQ[1][0] * xi + ATTN_WQ[1][1] * pos[i],
  ]);
  const K = x.map((xi, i) => [
    ATTN_WK[0][0] * xi + ATTN_WK[0][1] * pos[i],
    ATTN_WK[1][0] * xi + ATTN_WK[1][1] * pos[i],
  ]);
  const invSqrtD = 1 / Math.SQRT2;
  const W = Q.map((q) => {
    const scores = K.map((k) => (q[0] * k[0] + q[1] * k[1]) * invSqrtD);
    const maxS = Math.max(...scores);
    const exps = scores.map((s) => Math.exp(s - maxS));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  });
  const tags: (string | null)[][] = W.map((row) => row.map((v) => (v >= 0.25 ? '★' : null)));
  return { W, tags };
}

function buildLoraMatrix(): { W: number[][]; dW: number[][]; tags: (string | null)[][] } {
  const dW = LORA_A.map((ai) => LORA_B.map((bj) => ai * bj));
  const W = DENSE_W.map((row, i) => row.map((w, j) => w + dW[i][j]));
  const tags: (string | null)[][] = dW.map((row) => row.map((v) => (Math.abs(v) >= 0.25 ? 'Δ' : null)));
  return { W, dW, tags };
}

/** Build the complete 6 x 6 Rosetta Stone snapshot for a given architecture and 6-vector x. */
export function computeRosetta(arch: ArchId, x: readonly number[]): RosettaView {
  if (arch === 'dense') {
    const W = DENSE_W.map((r) => [...r]);
    const tags: (string | null)[][] = W.map((r) => r.map(() => null));
    return {
      arch,
      title: 'Dense MLP',
      shortRule: 'Every cell in the 6 × 6 matrix is an independent weight.',
      W,
      tags,
      y: matVec(W, x),
      freeParams: 36,
      denseParams: 36,
      nonZeroMults: 36,
      inputDependent: false,
    };
  }
  if (arch === 'conv') {
    const { W, tags } = buildConvMatrix();
    const nonZeroMults = W.flat().filter((v) => Math.abs(v) > 1e-12).length;
    return {
      arch,
      title: '1D Convolution',
      shortRule: 'Same 3-number kernel [−1, 0, +1] slid down the diagonals; off-band entries locked to 0.',
      W,
      tags,
      y: matVec(W, x),
      freeParams: 3,
      denseParams: 36,
      nonZeroMults,
      inputDependent: false,
    };
  }
  if (arch === 'rnn') {
    const { W, tags } = buildRnnMatrix();
    const nonZeroMults = W.flat().filter((v) => Math.abs(v) > 1e-12).length;
    return {
      arch,
      title: 'Unrolled RNN',
      shortRule: 'Causal lower-triangular blocks: the same 2 × 2 input (Wₓ) and memory (Wₕ) weights repeat across time.',
      W,
      tags,
      y: matVec(W, x),
      freeParams: 8,
      denseParams: 36,
      nonZeroMults,
      inputDependent: false,
    };
  }
  if (arch === 'attention') {
    const { W, tags } = buildAttentionMatrix(x);
    return {
      arch,
      title: 'Self-Attention',
      shortRule: 'Each row of W(x) is a softmax over query–key dot products computed live from the input x.',
      W,
      tags,
      y: matVec(W, x),
      freeParams: 8, // 2x2 W_Q + 2x2 W_K
      denseParams: 36,
      nonZeroMults: 36,
      inputDependent: true,
    };
  }
  const { W, dW, tags } = buildLoraMatrix();
  return {
    arch: 'lora',
    title: 'LoRA (rank 1)',
    shortRule: 'Freeze the 36-weight base matrix W₀ and train only a rank-1 outer product a · bᵀ (6 + 6 = 12 numbers).',
    W,
    tags,
    lora: { W0: DENSE_W.map((r) => [...r]), a: [...LORA_A], b: [...LORA_B], dW },
    y: matVec(W, x),
    freeParams: 12,
    denseParams: 36,
    nonZeroMults: 36,
    inputDependent: false,
  };
}

/** Max absolute cell difference between two 6 x 6 matrices. */
export function maxMatrixDiff(A: readonly (readonly number[])[], B: readonly (readonly number[])[]): number {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) {
      const d = Math.abs(A[i][j] - B[i][j]);
      if (d > m) m = d;
    }
  }
  return m;
}
