// authored-by: Neon High
import { createRng } from '../../lib/rng';

const GIB = 1024 ** 3;

// ------------------------------------------------------------------ 1. Low-Rank SVD Matrix Workshop (12x12)

export const SVD_DIM = 12;

export interface SvdResult {
  matrix: number[][];
  singularValues: number[]; // length 12, descending
  U: number[][]; // 12 x 12 (columns are u_k: U[k] is vector k)
  V: number[][]; // 12 x 12 (V[k] is vector k)
  totalEnergy: number; // sum of sigma_k^2
}

/** Exact SVD of a small d x d matrix via power iteration + deflation on M^T M. */
export function computeSvd(M: number[][]): SvdResult {
  const d = M.length;
  // Build A = M^T M (symmetric PSD)
  const A: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => {
      let s = 0;
      for (let r = 0; r < d; r++) s += M[r][i] * M[r][j];
      return s;
    })
  );

  const singularValues: number[] = [];
  const U: number[][] = [];
  const V: number[][] = [];

  // Work on a deflated copy of A
  const curA = A.map((row) => row.slice());
  const rng = createRng(101);

  for (let k = 0; k < d; k++) {
    // Power iteration for largest eigenvector of curA
    let v = Array.from({ length: d }, () => rng.normal(0, 1));
    // Orthogonalise against previous V vectors
    const orth = (vec: number[]) => {
      for (let p = 0; p < V.length; p++) {
        let dot = 0;
        for (let i = 0; i < d; i++) dot += vec[i] * V[p][i];
        for (let i = 0; i < d; i++) vec[i] -= dot * V[p][i];
      }
      const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
      for (let i = 0; i < d; i++) vec[i] /= norm;
    };
    orth(v);

    for (let iter = 0; iter < 80; iter++) {
      const next = new Array<number>(d).fill(0);
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += curA[i][j] * v[j];
        next[i] = s;
      }
      orth(next);
      v = next;
    }

    // Compute u = M v, sigma = ||u||
    const uRaw = new Array<number>(d).fill(0);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += M[i][j] * v[j];
      uRaw[i] = s;
    }
    const sigma = Math.sqrt(uRaw.reduce((s, x) => s + x * x, 0));
    const u = sigma > 1e-12 ? uRaw.map((x) => x / sigma) : Array.from({ length: d }, (_, i) => (i === k ? 1 : 0));

    singularValues.push(sigma);
    U.push(u);
    V.push(v);

    // Deflate curA -= sigma^2 v v^T
    const lam = sigma * sigma;
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        curA[i][j] -= lam * v[i] * v[j];
      }
    }
  }

  const totalEnergy = singularValues.reduce((s, sv) => s + sv * sv, 0);
  return { matrix: M, singularValues, U, V, totalEnergy };
}

export function reconstructRankR(svd: SvdResult, rank: number): {
  approx: number[][];
  relError: number;
  energyPct: number;
  loraParams: number;
  fullParams: number;
} {
  const d = svd.matrix.length;
  const r = Math.max(0, Math.min(d, rank));
  const approx: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  let keptEnergy = 0;

  for (let k = 0; k < r; k++) {
    const s = svd.singularValues[k];
    keptEnergy += s * s;
    const uk = svd.U[k];
    const vk = svd.V[k];
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        approx[i][j] += s * uk[i] * vk[j];
      }
    }
  }

  let errSq = 0;
  let totSq = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const diff = svd.matrix[i][j] - approx[i][j];
      errSq += diff * diff;
      totSq += svd.matrix[i][j] * svd.matrix[i][j];
    }
  }

  return {
    approx,
    relError: Math.sqrt(errSq / Math.max(1e-12, totSq)),
    energyPct: (keptEnergy / Math.max(1e-12, svd.totalEnergy)) * 100,
    loraParams: r * (d + d),
    fullParams: d * d,
  };
}

export function makeWorkshopMatrix(kind: 'task-shift' | 'random-noise'): number[][] {
  const rng = createRng(kind === 'task-shift' ? 2024 : 7777);
  const d = SVD_DIM;
  if (kind === 'random-noise') {
    return Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, 0.5)));
  }
  // Structured rank-2 core + subtle rank-3 component + tiny ambient noise (realistic fine-tuning ΔW)
  const M: number[][] = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, 0.04)));
  const strengths = [1.4, 0.9, 0.35];
  for (const s of strengths) {
    const u = Array.from({ length: d }, () => rng.normal(0, 1));
    const v = Array.from({ length: d }, () => rng.normal(0, 1));
    const un = Math.sqrt(u.reduce((a, b) => a + b * b, 0));
    const vn = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        M[i][j] += (s * u[i] * v[j]) / (un * vn) * 2.2;
      }
    }
  }
  return M;
}

// ------------------------------------------------------------------ 2. Exact LoRA / QLoRA Parameter & VRAM Memory Calculator

export interface LoraModelArch {
  id: 'llama3-8b' | 'llama3-70b';
  name: string;
  shortName: string;
  paramsB: number;
  layers: number;
  dModel: number;
  dKv: number; // kvHeads * dHead
  dFfn: number;
}

export const LORA_MODELS: LoraModelArch[] = [
  {
    id: 'llama3-8b',
    name: 'Llama-3 8B',
    shortName: 'Llama-3 8B',
    paramsB: 8.03,
    layers: 32,
    dModel: 4096,
    dKv: 1024,
    dFfn: 14336,
  },
  {
    id: 'llama3-70b',
    name: 'Llama-3 70B',
    shortName: 'Llama-3 70B',
    paramsB: 70.6,
    layers: 80,
    dModel: 8192,
    dKv: 1024,
    dFfn: 28672,
  },
];

export type TargetScope = 'qv' | 'attn' | 'all';
export type TuneMethod = 'full' | 'lora' | 'qlora';

export function countLoraParams(model: LoraModelArch, rank: number, scope: TargetScope): number {
  const { layers, dModel, dKv, dFfn } = model;
  // Per layer:
  // W_Q: dModel x dModel -> r*(dModel + dModel)
  // W_K: dKv x dModel    -> r*(dKv + dModel)
  // W_V: dKv x dModel    -> r*(dKv + dModel)
  // W_O: dModel x dModel -> r*(dModel + dModel)
  // Gate, Up, Down: each dFfn x dModel (or transpose) -> r*(dFfn + dModel)
  const q = rank * (dModel + dModel);
  const kv = rank * (dKv + dModel);
  const o = rank * (dModel + dModel);
  const mlp = 3 * rank * (dFfn + dModel);

  let perLayer = q + kv; // Q + V
  if (scope === 'attn') perLayer = q + 2 * kv + o;
  if (scope === 'all') perLayer = q + 2 * kv + o + mlp;
  return layers * perLayer;
}

export interface MemoryBreakdown {
  totalParams: number;
  trainableParams: number;
  trainablePct: number;
  baseWeightsGiB: number;
  adapterWeightsGiB: number; // BF16 adapter weights + FP32 master copy (6 B/trainable param)
  gradsGiB: number; // BF16 grads (2 B/trainable param)
  adamStatesGiB: number; // FP32 m + v (8 B/trainable param)
  totalGiB: number;
  checkpointMB: number; // saved adapter weights in BF16 (2 B/trainable param)
}

/**
 * Exact mixed-precision AdamW memory footprint (weights + master + grads + Adam m, v;
 * excludes batch activations). Standard mixed precision = 16 bytes per trainable param
 * (2 B BF16 + 4 B FP32 master + 2 B grad + 8 B Adam m,v).
 */
export function estimateFineTuneMemory(
  model: LoraModelArch,
  method: TuneMethod,
  rank: number,
  scope: TargetScope
): MemoryBreakdown {
  const totalParams = model.paramsB * 1e9;
  if (method === 'full') {
    const baseWeightsGiB = (totalParams * 2) / GIB; // BF16 active weights
    const adapterWeightsGiB = (totalParams * 4) / GIB; // FP32 master weights
    const gradsGiB = (totalParams * 2) / GIB;
    const adamStatesGiB = (totalParams * 8) / GIB;
    return {
      totalParams,
      trainableParams: totalParams,
      trainablePct: 100,
      baseWeightsGiB,
      adapterWeightsGiB,
      gradsGiB,
      adamStatesGiB,
      totalGiB: baseWeightsGiB + adapterWeightsGiB + gradsGiB + adamStatesGiB,
      checkpointMB: (totalParams * 2) / (1024 ** 2),
    };
  }

  const trainableParams = countLoraParams(model, rank, scope);
  const baseBytesPerParam = method === 'qlora' ? 0.5625 : 2.0; // NF4 + group scales (4.5 b) vs BF16 (16 b)
  const baseWeightsGiB = (totalParams * baseBytesPerParam) / GIB;
  const adapterWeightsGiB = (trainableParams * 6) / GIB; // BF16 + FP32 master
  const gradsGiB = (trainableParams * 2) / GIB;
  const adamStatesGiB = (trainableParams * 8) / GIB;

  return {
    totalParams,
    trainableParams,
    trainablePct: (trainableParams / totalParams) * 100,
    baseWeightsGiB,
    adapterWeightsGiB,
    gradsGiB,
    adamStatesGiB,
    totalGiB: baseWeightsGiB + adapterWeightsGiB + gradsGiB + adamStatesGiB,
    checkpointMB: (trainableParams * 2) / (1024 ** 2),
  };
}

// ------------------------------------------------------------------ 3. Live LoRA Training & Zero-Init Symmetry Experiment

export type LoraInitMode = 'lora-zero-b' | 'both-zero' | 'both-random';

export interface LoraTrainResult {
  initMode: LoraInitMode;
  rank: number;
  trainableParams: number;
  step0Loss: number;
  pretrainedLoss: number;
  finalLoss: number;
  step0GradANorm: number;
  step0GradBNorm: number;
  losses: number[]; // every 2 steps (0..80)
}

function buildAdaptationTask() {
  const rng = createRng(314);
  const d = 8;
  // Frozen pretrained weights W0 (8x8)
  const W0 = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, 0.45)));
  // True task shift ΔW* = exact rank-3 update (64 entries governed by 3 directions)
  const dWStar: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const strengths = [0.75, 0.5, 0.3];
  for (const s of strengths) {
    const u = Array.from({ length: d }, () => rng.normal(0, 1));
    const v = Array.from({ length: d }, () => rng.normal(0, 1));
    const un = Math.sqrt(u.reduce((a, b) => a + b * b, 0));
    const vn = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        dWStar[i][j] += (s * u[i] * v[j] * 2.2) / (un * vn);
      }
    }
  }
  const WTrue = W0.map((row, i) => row.map((v, j) => v + dWStar[i][j]));

  // 48 unit-Gaussian input examples X (48 x 8), targets Y = WTrue * X
  const N = 48;
  const X = Array.from({ length: N }, () => Array.from({ length: d }, () => rng.normal(0, 1)));
  const Y = X.map((x) => WTrue.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)));
  return { d, W0, WTrue, dWStar, X, Y };
}

const ADAPT_TASK = buildAdaptationTask();

/**
 * Exact loss and analytical gradients (dA, dB) for y = (W0 + scale * B * A) x,
 * exposed for central finite-difference testing in lora.test.ts!
 */
export function loraStepGrads(
  W0: number[][],
  A: number[][],
  B: number[][],
  scale: number,
  X: number[][],
  Y: number[][]
): { loss: number; gA: number[][]; gB: number[][] } {
  const d = W0.length;
  const r = A.length;
  const N = X.length;

  // Effective weight W = W0 + scale * B * A
  const W = W0.map((row) => row.slice());
  for (let i = 0; i < d; i++) {
    for (let k = 0; k < r; k++) {
      const bik = scale * B[i][k];
      for (let j = 0; j < d; j++) {
        W[i][j] += bik * A[k][j];
      }
    }
  }

  const gA: number[][] = Array.from({ length: r }, () => new Array<number>(d).fill(0));
  const gB: number[][] = Array.from({ length: d }, () => new Array<number>(r).fill(0));
  let totalSq = 0;

  for (let n = 0; n < N; n++) {
    const x = X[n];
    const y = Y[n];
    // Hidden u = A x (length r)
    const u = new Array<number>(r).fill(0);
    for (let k = 0; k < r; k++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += A[k][j] * x[j];
      u[k] = s;
    }
    // Residual e = W x - y (length d)
    const e = new Array<number>(d);
    for (let i = 0; i < d; i++) {
      let pred = 0;
      for (let j = 0; j < d; j++) pred += W[i][j] * x[j];
      e[i] = pred - y[i];
      totalSq += e[i] * e[i];
    }

    // dLoss/dPred = (2 / (N * d)) * e
    const coeff = (2 * scale) / (N * d);
    for (let i = 0; i < d; i++) {
      const gi = coeff * e[i];
      for (let k = 0; k < r; k++) {
        gB[i][k] += gi * u[k];
      }
    }
    // gU = B^T * (coeff * e)
    for (let k = 0; k < r; k++) {
      let guk = 0;
      for (let i = 0; i < d; i++) guk += B[i][k] * (coeff * e[i]);
      for (let j = 0; j < d; j++) gA[k][j] += guk * x[j];
    }
  }

  return { loss: totalSq / (N * d), gA, gB };
}

function frobNorm(M: number[][]): number {
  let s = 0;
  for (const row of M) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

/**
 * Trains LoRA (rank r) on the frozen 8x8 adaptation task (exact rank-3 shift)
 * for 80 steps of SGD, demonstrating zero-init symmetry breaking.
 */
export function runLoraExperiment(rank: number, initMode: LoraInitMode, steps = 80): LoraTrainResult {
  const { d, W0, X, Y } = ADAPT_TASK;
  const rng = createRng(99);
  const scale = rank > 0 ? 1.0 : 0; // alpha = r => alpha/r = 1

  // Pretrained loss before any adapter
  const pretrainedLoss = loraStepGrads(W0, [], [], 0, X, Y).loss;

  const A: number[][] = Array.from({ length: rank }, () =>
    Array.from({ length: d }, () => (initMode === 'both-zero' ? 0 : rng.normal(0, 0.35)))
  );
  const B: number[][] = Array.from({ length: d }, () =>
    Array.from({ length: rank }, () => (initMode === 'both-random' ? rng.normal(0, 0.5) : 0))
  );

  const lr = 0.25;
  const losses: number[] = [];
  let step0Loss = 0;
  let step0GradANorm = 0;
  let step0GradBNorm = 0;
  let finalLoss = 0;

  for (let s = 0; s <= steps; s++) {
    const { loss, gA, gB } = loraStepGrads(W0, A, B, scale, X, Y);
    if (s === 0) {
      step0Loss = loss;
      step0GradANorm = frobNorm(gA);
      step0GradBNorm = frobNorm(gB);
    }
    if (s % 2 === 0) losses.push(loss);
    finalLoss = loss;
    if (s === steps) break;

    for (let k = 0; k < rank; k++) {
      for (let j = 0; j < d; j++) A[k][j] -= lr * gA[k][j];
    }
    for (let i = 0; i < d; i++) {
      for (let k = 0; k < rank; k++) B[i][k] -= lr * gB[i][k];
    }
  }

  return {
    initMode,
    rank,
    trainableParams: rank * (d + d),
    step0Loss,
    pretrainedLoss,
    finalLoss,
    step0GradANorm,
    step0GradBNorm,
    losses,
  };
}

/** Full fine-tuning reference on the exact same 8x8 task (64 trainable params). */
export function runFullFtReference(steps = 80): { trainableParams: number; finalLoss: number; losses: number[] } {
  const { d, W0, X, Y } = ADAPT_TASK;
  const N = X.length;
  const W = W0.map((row) => row.slice());
  const lr = 0.25;
  const losses: number[] = [];
  let finalLoss = 0;

  for (let s = 0; s <= steps; s++) {
    const gW = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    let totalSq = 0;
    for (let n = 0; n < N; n++) {
      const x = X[n];
      const y = Y[n];
      for (let i = 0; i < d; i++) {
        let pred = 0;
        for (let j = 0; j < d; j++) pred += W[i][j] * x[j];
        const e = pred - y[i];
        totalSq += e * e;
        const coeff = (2 * e) / (N * d);
        for (let j = 0; j < d; j++) gW[i][j] += coeff * x[j];
      }
    }
    finalLoss = totalSq / (N * d);
    if (s % 2 === 0) losses.push(finalLoss);
    if (s === steps) break;
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) W[i][j] -= lr * gW[i][j];
    }
  }
  return { trainableParams: d * d, finalLoss, losses };
}

// ------------------------------------------------------------------ 4. Lab: Adapter Allocation Budget (3 layers, 125 configs)

export const LAB_RANKS = [0, 1, 2, 3, 4] as const;
export const LAB_PARAM_BUDGET = 80; // vs 3 * 64 = 192 for full FT!
export const LAB_ERR_GOAL = 0.005;

export interface LabLayerSpec {
  id: string;
  name: string;
  role: string;
  svd: SvdResult;
  /** Exact residual squared error at each rank r=0..4 (computed from exact SVD!) */
  mseByRank: number[];
}

function makeShiftBlock(seed: number, strengths: number[], noiseStd: number): SvdResult {
  const rng = createRng(seed);
  const d = 8;
  const M: number[][] = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, noiseStd)));
  for (const s of strengths) {
    const u = Array.from({ length: d }, () => rng.normal(0, 1));
    const v = Array.from({ length: d }, () => rng.normal(0, 1));
    const un = Math.sqrt(u.reduce((a, b) => a + b * b, 0));
    const vn = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        M[i][j] += (s * u[i] * v[j]) / (un * vn);
      }
    }
  }
  return computeSvd(M);
}

function buildLabLayers(): LabLayerSpec[] {
  // Three 8x8 layers with genuinely different shift ranks:
  // Layer 1 (Early features): almost unchanged (tiny noise only -> r=0 already ~0.002!)
  // Layer 2 (Middle attention/MLP): rich rank-3 domain shift -> needs r=3!
  // Layer 3 (Output head): rank-2 calibration shift -> needs r=2!
  const defs: Array<{ id: string; name: string; role: string; seed: number; strengths: number[]; noise: number }> = [
    { id: 'early', name: 'Layer 1 · Early features', role: 'Low-level syntax already transfers cleanly', seed: 401, strengths: [0.35], noise: 0.02 },
    { id: 'mid', name: 'Layer 2 · Middle reasoning', role: 'New domain concepts live here (richer shift)', seed: 402, strengths: [2.2, 1.5, 1.05], noise: 0.02 },
    { id: 'head', name: 'Layer 3 · Output formatting', role: 'Style & tone adjustment at the top', seed: 403, strengths: [1.6, 0.95], noise: 0.02 },
  ];

  return defs.map((def) => {
    const svd = makeShiftBlock(def.seed, def.strengths, def.noise);
    const d = 8;
    const mseByRank = LAB_RANKS.map((r) => {
      // Exact Eckart–Young residual MSE = (sum_{k>=r} sigma_k^2) / (d*d)
      let rem = 0;
      for (let k = r; k < d; k++) rem += svd.singularValues[k] ** 2;
      return rem / (d * d);
    });
    return { id: def.id, name: def.name, role: def.role, svd, mseByRank };
  });
}

export const LAB_LAYERS: LabLayerSpec[] = buildLabLayers();

export interface LoraLabEval {
  ranks: [number, number, number];
  totalParams: number;
  layerMses: [number, number, number];
  avgMse: number;
  fitsBudget: boolean;
  meetsErr: boolean;
  passes: boolean;
}

export function evaluateLoraLab(ranks: [number, number, number]): LoraLabEval {
  // Each 8x8 layer at rank r costs r*(8+8) = 16*r trainable params
  const totalParams = ranks.reduce((s, r) => s + r * 16, 0);
  const layerMses: [number, number, number] = [
    LAB_LAYERS[0].mseByRank[ranks[0]],
    LAB_LAYERS[1].mseByRank[ranks[1]],
    LAB_LAYERS[2].mseByRank[ranks[2]],
  ];
  const avgMse = (layerMses[0] + layerMses[1] + layerMses[2]) / 3;
  const fitsBudget = totalParams <= LAB_PARAM_BUDGET;
  const meetsErr = avgMse <= LAB_ERR_GOAL;
  return { ranks, totalParams, layerMses, avgMse, fitsBudget, meetsErr, passes: fitsBudget && meetsErr };
}

export function solveLoraLab(): {
  totalConfigs: number;
  passingCount: number;
  best: LoraLabEval;
  uniformR1: LoraLabEval;
  uniformR2: LoraLabEval;
} {
  let totalConfigs = 0;
  let passingCount = 0;
  let best: LoraLabEval | null = null;

  for (const r0 of LAB_RANKS) {
    for (const r1 of LAB_RANKS) {
      for (const r2 of LAB_RANKS) {
        totalConfigs++;
        const ev = evaluateLoraLab([r0, r1, r2]);
        if (ev.passes) {
          passingCount++;
          if (!best || ev.avgMse < best.avgMse - 1e-12) {
            best = ev;
          }
        }
      }
    }
  }

  return {
    totalConfigs,
    passingCount,
    best: best!,
    uniformR1: evaluateLoraLab([1, 1, 1]),
    uniformR2: evaluateLoraLab([2, 2, 2]),
  };
}
