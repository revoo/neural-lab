// authored-by: Neon High
import { createRng } from '../../lib/rng';

export const KIB = 1024;
export const GIB = 1024 ** 3;

// ------------------------------------------------------------------ 1. Exact KV cache equivalence & work counts

export const DEMO_TOKENS = ['The', 'quick', 'brown', 'fox', 'jumps', 'over'];

export interface StepWork {
  step: number; // 1-indexed sequence length n
  token: string;
  uncachedProjections: number; // token Q/K/V projections this step (= n)
  cachedProjections: number; // token Q/K/V projections this step (= 1)
  uncachedDotProducts: number; // causal attention pairs this step (= n(n+1)/2)
  cachedDotProducts: number; // causal attention pairs this step (= n)
  cumUncachedProjections: number;
  cumCachedProjections: number;
  cumUncachedDots: number;
  cumCachedDots: number;
  maxAbsDiff: number; // max |cached_out - uncached_out| at this step
}

function matVec(W: number[][], x: number[]): number[] {
  return W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));
}

function softmax(scores: number[]): number[] {
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Runs a real 4-dim causal self-attention layer across DEMO_TOKENS twice:
 * once recomputing every token from scratch at each decode step, and once
 * reusing cached K and V vectors. Proves numerically that both produce the
 * exact same output vector to floating-point precision (< 1e-14).
 */
export function runKvEquivalenceDemo(): { steps: StepWork[]; maxDiffOverall: number } {
  const rng = createRng(42);
  const d = 4;
  const randMat = (r: number, c: number) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => rng.uniform(-0.8, 0.8)));

  const X = randMat(DEMO_TOKENS.length, d);
  const Wq = randMat(d, d);
  const Wk = randMat(d, d);
  const Wv = randMat(d, d);
  const invSqrtD = 1 / Math.sqrt(d);

  const cacheK: number[][] = [];
  const cacheV: number[][] = [];

  const steps: StepWork[] = [];
  let cumUncProj = 0;
  let cumCacProj = 0;
  let cumUncDots = 0;
  let cumCacDots = 0;
  let maxDiffOverall = 0;

  for (let n = 1; n <= DEMO_TOKENS.length; n++) {
    // --- Uncached: project all n tokens and run full causal attention ---
    const Qall = X.slice(0, n).map((x) => matVec(Wq, x));
    const Kall = X.slice(0, n).map((x) => matVec(Wk, x));
    const Vall = X.slice(0, n).map((x) => matVec(Wv, x));

    // Only the last row (index n-1) is actually needed to predict the next token!
    const qLastUnc = Qall[n - 1];
    const scoresUnc = Kall.map((k) => qLastUnc.reduce((s, qj, j) => s + qj * k[j], 0) * invSqrtD);
    const weightsUnc = softmax(scoresUnc);
    const outUnc = Array.from({ length: d }, (_, j) =>
      weightsUnc.reduce((s, w, i) => s + w * Vall[i][j], 0)
    );

    // --- Cached: project ONLY the new token n-1 and append to KV cache ---
    const xNew = X[n - 1];
    const qNew = matVec(Wq, xNew);
    const kNew = matVec(Wk, xNew);
    const vNew = matVec(Wv, xNew);
    cacheK.push(kNew);
    cacheV.push(vNew);

    const scoresCac = cacheK.map((k) => qNew.reduce((s, qj, j) => s + qj * k[j], 0) * invSqrtD);
    const weightsCac = softmax(scoresCac);
    const outCac = Array.from({ length: d }, (_, j) =>
      weightsCac.reduce((s, w, i) => s + w * cacheV[i][j], 0)
    );

    let maxAbsDiff = 0;
    for (let j = 0; j < d; j++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(outUnc[j] - outCac[j]));
    }
    maxDiffOverall = Math.max(maxDiffOverall, maxAbsDiff);

    const uncProj = n;
    const cacProj = 1;
    const uncDots = (n * (n + 1)) / 2;
    const cacDots = n;
    cumUncProj += uncProj;
    cumCacProj += cacProj;
    cumUncDots += uncDots;
    cumCacDots += cacDots;

    steps.push({
      step: n,
      token: DEMO_TOKENS[n - 1],
      uncachedProjections: uncProj,
      cachedProjections: cacProj,
      uncachedDotProducts: uncDots,
      cachedDotProducts: cacDots,
      cumUncachedProjections: cumUncProj,
      cumCachedProjections: cumCacProj,
      cumUncachedDots: cumUncDots,
      cumCachedDots: cumCacDots,
      maxAbsDiff,
    });
  }

  return { steps, maxDiffOverall };
}

/**
 * Total token forward-passes (Q/K/V + FFN projections) and causal attention dot-product pairs
 * to prefill P prompt tokens and generate G new tokens.
 */
export function generationWorkTotals(promptLen: number, genLen: number) {
  // Prefill processes P tokens once; then G decode steps at lengths P+1 .. P+G-1 (or P .. P+G-1)
  // Step 0 processes P tokens (producing token P+1). Steps s=1..G-1 process lengths P+1 .. P+G-1.
  let uncachedTokens = 0;
  let uncachedPairs = 0;
  let cachedTokens = promptLen + Math.max(0, genLen - 1);
  let cachedPairs = (promptLen * (promptLen + 1)) / 2;

  for (let s = 0; s < genLen; s++) {
    const n = promptLen + s;
    uncachedTokens += n;
    uncachedPairs += (n * (n + 1)) / 2;
    if (s > 0) cachedPairs += n;
  }

  return {
    uncachedTokens,
    cachedTokens,
    tokenSpeedup: uncachedTokens / Math.max(1, cachedTokens),
    uncachedPairs,
    cachedPairs,
    pairSpeedup: uncachedPairs / Math.max(1, cachedPairs),
  };
}

// ------------------------------------------------------------------ 2. Real model architectures & KV cache memory

export interface ModelSpec {
  id: string;
  name: string;
  shortName: string;
  paramsB: number; // billions of parameters
  layers: number;
  qHeads: number;
  kvHeads: number;
  dHead: number;
  note: string;
}

export const MODEL_SPECS: ModelSpec[] = [
  {
    id: 'llama3-8b',
    name: 'Llama-3 8B (GQA 4:1)',
    shortName: 'Llama-3 8B',
    paramsB: 8.03,
    layers: 32,
    qHeads: 32,
    kvHeads: 8,
    dHead: 128,
    note: '4 query heads share each KV head — cuts KV cache 4× compared to standard MHA.',
  },
  {
    id: 'llama3-8b-mha',
    name: 'Llama-3 8B if MHA (1:1)',
    shortName: '8B (MHA)',
    paramsB: 8.03,
    layers: 32,
    qHeads: 32,
    kvHeads: 32,
    dHead: 128,
    note: 'Hypothetical 8B model with 32 KV heads (standard Multi-Head Attention): 4× larger cache.',
  },
  {
    id: 'llama3-70b',
    name: 'Llama-3 70B (GQA 8:1)',
    shortName: 'Llama-3 70B',
    paramsB: 70.6,
    layers: 80,
    qHeads: 64,
    kvHeads: 8,
    dHead: 128,
    note: '8 query heads share each KV head — cuts KV cache 8× so 128k context fits in 40 GiB instead of 320 GiB.',
  },
  {
    id: 'llama3-70b-mha',
    name: 'Llama-3 70B if MHA (1:1)',
    shortName: '70B (MHA)',
    paramsB: 70.6,
    layers: 80,
    qHeads: 64,
    kvHeads: 64,
    dHead: 128,
    note: 'Hypothetical 70B model without GQA: 2.5 MiB per token — 320 GiB at 128k context for a single user.',
  },
];

export const MODEL_BY_ID = Object.fromEntries(MODEL_SPECS.map((m) => [m.id, m]));

export interface PrecisionFormat {
  id: 'fp16' | 'int8' | 'fp8' | 'int4';
  label: string;
  bytesPerElem: number;
  bitsLabel: string;
}

export const WEIGHT_FORMATS: PrecisionFormat[] = [
  { id: 'fp16', label: 'FP16 / BF16', bytesPerElem: 2.0, bitsLabel: '16 b' },
  { id: 'int8', label: 'INT8 (g=64)', bytesPerElem: 1.03125, bitsLabel: '8.25 b' },
  { id: 'int4', label: 'INT4 (g=32)', bytesPerElem: 0.5625, bitsLabel: '4.5 b' },
];

export const KV_FORMATS: PrecisionFormat[] = [
  { id: 'fp16', label: 'FP16 / BF16', bytesPerElem: 2.0, bitsLabel: '16 b' },
  { id: 'fp8', label: 'FP8 / INT8', bytesPerElem: 1.0, bitsLabel: '8 b' },
  { id: 'int4', label: 'INT4 (g=32)', bytesPerElem: 0.5625, bitsLabel: '4.5 b' },
];

/** Exact KV cache bytes for a single token in a single sequence. */
export function kvBytesPerToken(model: ModelSpec, kvBytesPerElem: number): number {
  return 2 * model.layers * model.kvHeads * model.dHead * kvBytesPerElem;
}

/** Exact KV cache size in GiB (1 GiB = 1024^3 bytes). */
export function kvCacheGiB(model: ModelSpec, ctxLen: number, batchSize: number, kvBytesPerElem: number): number {
  return (kvBytesPerToken(model, kvBytesPerElem) * ctxLen * batchSize) / GIB;
}

/** Model weight footprint in GiB (1 GiB = 1024^3 bytes). */
export function weightsGiB(model: ModelSpec, weightBytesPerElem: number): number {
  return (model.paramsB * 1e9 * weightBytesPerElem) / GIB;
}

// ------------------------------------------------------------------ 3. GPU hardware specs & Roofline model

export interface GpuSpec {
  id: string;
  name: string;
  shortName: string;
  vramGiB: number;
  bwTBps: number; // memory bandwidth in TB/s (10^12 bytes/s)
  peakTflops: number; // dense FP16/BF16 Tensor Core peak (10^12 FLOP/s)
  overheadGiB: number; // typical CUDA context + activation scratchpad
  note: string;
}

export const GPU_SPECS: GpuSpec[] = [
  {
    id: 'm3-max',
    name: 'Apple M3 Max (128 GB unified)',
    shortName: 'M3 Max',
    vramGiB: 128,
    bwTBps: 0.4,
    peakTflops: 14.2,
    overheadGiB: 4.0,
    note: 'Huge unified memory pool, modest bandwidth (400 GB/s) — can hold 70B INT4 locally at ~8–10 tok/s.',
  },
  {
    id: 'rtx-4070',
    name: 'NVIDIA RTX 4070 (12 GiB)',
    shortName: 'RTX 4070 12G',
    vramGiB: 12,
    bwTBps: 0.504,
    peakTflops: 58.3,
    overheadGiB: 1.0,
    note: 'Consumer GPU: 12 GiB VRAM, 504 GB/s bandwidth, 58.3 TFLOP/s FP16 Tensor Core (FP32 acc).',
  },
  {
    id: 'rtx-4090',
    name: 'NVIDIA RTX 4090 (24 GiB)',
    shortName: 'RTX 4090 24G',
    vramGiB: 24,
    bwTBps: 1.008,
    peakTflops: 165.2,
    overheadGiB: 1.5,
    note: 'Desktop powerhouse: 1.0 TB/s bandwidth, 165 TFLOP/s FP16 Tensor Core (FP32 acc).',
  },
  {
    id: 'a100-80',
    name: 'NVIDIA A100 80GB SXM',
    shortName: 'A100 80G',
    vramGiB: 80,
    bwTBps: 2.039,
    peakTflops: 312,
    overheadGiB: 3.0,
    note: 'Datacentre workhorse: 2.04 TB/s HBM2e, 312 TFLOP/s FP16/BF16, ridge point 153 FLOPs/byte.',
  },
  {
    id: 'h100-80',
    name: 'NVIDIA H100 80GB SXM',
    shortName: 'H100 80G',
    vramGiB: 80,
    bwTBps: 3.35,
    peakTflops: 989,
    overheadGiB: 3.0,
    note: 'Frontier datacentre GPU: 3.35 TB/s HBM3, 989 TFLOP/s dense FP16/BF16 — ridge point 295 FLOPs/byte.',
  },
];

export const GPU_BY_ID = Object.fromEntries(GPU_SPECS.map((g) => [g.id, g]));

export function ridgePoint(gpu: GpuSpec): number {
  return gpu.peakTflops / gpu.bwTBps; // FLOPs / byte
}

export interface RooflineResult {
  flops: number; // total FLOPs for this step
  bytes: number; // total HBM bytes read for this step
  weightBytes: number;
  kvBytes: number;
  arithmeticIntensity: number; // FLOPs / byte
  ridge: number;
  bottleneck: 'memory' | 'compute';
  achievedTflops: number;
  computeUtilPct: number; // 0..100
  stepTimeMs: number;
  tokensPerSecPerUser: number;
  totalTokensPerSec: number;
}

/**
 * First-principles roofline model for one Transformer step:
 * - phase='decode': generate 1 token for each of B sequences at context length ctxLen
 * - phase='prefill': process a prompt of length ctxLen for B sequences simultaneously
 *
 * Uses the exact hardware roofline ceiling (time = max(FLOPs/peak, Bytes/bandwidth)).
 * Real serving engines (vLLM, TensorRT-LLM) typically reach ~70–85% of this bound.
 */
export function evaluateRoofline(params: {
  gpu: GpuSpec;
  model: ModelSpec;
  phase: 'prefill' | 'decode';
  batchSize: number;
  ctxLen: number;
  weightBytesPerElem: number;
  kvBytesPerElem: number;
}): RooflineResult {
  const { gpu, model, phase, batchSize, ctxLen, weightBytesPerElem, kvBytesPerElem } = params;
  const N = model.paramsB * 1e9;
  const weightBytes = N * weightBytesPerElem;
  const kvBytes = kvBytesPerToken(model, kvBytesPerElem) * ctxLen * batchSize;

  let flops: number;
  let bytes: number;
  let tokensProduced: number;

  if (phase === 'decode') {
    // 1 new token per sequence: linear layers = 2*N*B; causal attention against ctxLen keys/vals = 4*L*n_q*d_head*ctxLen*B
    const linearFlops = 2 * N * batchSize;
    const attnFlops = 4 * model.layers * model.qHeads * model.dHead * ctxLen * batchSize;
    flops = linearFlops + attnFlops;
    bytes = weightBytes + kvBytes;
    tokensProduced = batchSize;
  } else {
    // Prefill: process B * ctxLen prompt tokens at once. Weights read once; KV written once.
    const linearFlops = 2 * N * batchSize * ctxLen;
    const attnFlops = 2 * model.layers * model.qHeads * model.dHead * ctxLen * ctxLen * batchSize;
    flops = linearFlops + attnFlops;
    bytes = weightBytes + kvBytes;
    tokensProduced = batchSize * ctxLen;
  }

  const intensity = flops / bytes;
  const ridge = ridgePoint(gpu);
  const bottleneck: 'memory' | 'compute' = intensity < ridge ? 'memory' : 'compute';

  const memTimeSec = bytes / (gpu.bwTBps * 1e12);
  const compTimeSec = flops / (gpu.peakTflops * 1e12);
  const stepTimeSec = Math.max(memTimeSec, compTimeSec);
  const achievedTflops = flops / stepTimeSec / 1e12;
  const computeUtilPct = (achievedTflops / gpu.peakTflops) * 100;
  const totalTokensPerSec = tokensProduced / stepTimeSec;
  const tokensPerSecPerUser = phase === 'decode' ? 1 / stepTimeSec : totalTokensPerSec / batchSize;

  return {
    flops,
    bytes,
    weightBytes,
    kvBytes,
    arithmeticIntensity: intensity,
    ridge,
    bottleneck,
    achievedTflops,
    computeUtilPct,
    stepTimeMs: stepTimeSec * 1000,
    tokensPerSecPerUser,
    totalTokensPerSec,
  };
}

// ------------------------------------------------------------------ 4. Lab: VRAM Tetris (3 engineering briefs)

export const TETRIS_BATCHES = [1, 2, 4, 8, 16, 32, 64] as const;
export const TETRIS_CTXS = [2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;

export interface TetrisConfig {
  weightFmt: 'fp16' | 'int8' | 'int4';
  kvFmt: 'fp16' | 'fp8' | 'int4';
  batchSize: number;
  ctxLen: number;
}

export interface TetrisBrief {
  id: string;
  title: string;
  scenario: string;
  gpuId: string;
  modelId: string;
  minBatch: number;
  minCtx: number;
  minTokPerUser: number;
  minTotalTokPerSec: number;
  forbidDoubleInt4?: boolean; // quality guardrail: don't quantise BOTH weights and KV to INT4
  defaultConfig: TetrisConfig;
  optLabel: string;
  optMetric: 'totalTokPerSec' | 'tokPerUser' | 'ctxThenSpeed';
}

export const TETRIS_BRIEFS: TetrisBrief[] = [
  {
    id: 'local-coder',
    title: 'Brief 1 · Local coding assistant',
    scenario:
      'Run Llama-3 8B on a 12 GiB RTX 4070 for one developer. You need at least 32k context to hold a codebase and ≥ 55 tok/s roofline speed — without quantising BOTH weights and KV cache down to INT4.',
    gpuId: 'rtx-4070',
    modelId: 'llama3-8b',
    minBatch: 1,
    minCtx: 32768,
    minTokPerUser: 55,
    minTotalTokPerSec: 55,
    forbidDoubleInt4: true,
    defaultConfig: { weightFmt: 'fp16', kvFmt: 'fp16', batchSize: 1, ctxLen: 32768 },
    optLabel: 'highest per-user speed (at ≥ 32k ctx)',
    optMetric: 'tokPerUser',
  },
  {
    id: 'single-h100-70b',
    title: 'Brief 2 · Single-GPU 70B server',
    scenario:
      'Serve Llama-3 70B on ONE 80 GiB H100 (no tensor parallelism). Support at least 8 concurrent users at ≥ 16k context with ≥ 40 tok/s per user. Optional goal: maximise total throughput across all users.',
    gpuId: 'h100-80',
    modelId: 'llama3-70b',
    minBatch: 8,
    minCtx: 16384,
    minTokPerUser: 40,
    minTotalTokPerSec: 320,
    forbidDoubleInt4: false,
    defaultConfig: { weightFmt: 'fp16', kvFmt: 'fp16', batchSize: 8, ctxLen: 16384 },
    optLabel: 'maximum total throughput',
    optMetric: 'totalTokPerSec',
  },
  {
    id: 'api-throughput',
    title: 'Brief 3 · High-throughput 32k API',
    scenario:
      'Serve Llama-3 8B at ≥ 32k context on an 80 GiB H100. Keep per-user speed ≥ 40 tok/s while pushing total system throughput ≥ 1,400 tok/s (without double-INT4). Watch how KV cache precision unlocks bigger batches and higher bandwidth!',
    gpuId: 'h100-80',
    modelId: 'llama3-8b',
    minBatch: 1,
    minCtx: 32768,
    minTokPerUser: 40,
    minTotalTokPerSec: 1400,
    forbidDoubleInt4: true,
    defaultConfig: { weightFmt: 'fp16', kvFmt: 'fp16', batchSize: 8, ctxLen: 32768 },
    optLabel: 'maximum total throughput',
    optMetric: 'totalTokPerSec',
  },
];

export interface TetrisEval {
  wGiB: number;
  kvGiB: number;
  overheadGiB: number;
  totalGiB: number;
  vramLimitGiB: number;
  fitsVram: boolean;
  roofline: RooflineResult;
  checks: {
    fitsVram: boolean;
    meetsBatch: boolean;
    meetsCtx: boolean;
    meetsUserSpeed: boolean;
    meetsTotalSpeed: boolean;
    meetsQuality: boolean;
  };
  passes: boolean;
}

export function evaluateTetris(brief: TetrisBrief, cfg: TetrisConfig): TetrisEval {
  const gpu = GPU_BY_ID[brief.gpuId];
  const model = MODEL_BY_ID[brief.modelId];
  const wFmt = WEIGHT_FORMATS.find((f) => f.id === cfg.weightFmt)!;
  const kFmt = KV_FORMATS.find((f) => f.id === cfg.kvFmt)!;

  const wGiB = weightsGiB(model, wFmt.bytesPerElem);
  const kvGiB = kvCacheGiB(model, cfg.ctxLen, cfg.batchSize, kFmt.bytesPerElem);
  const overheadGiB = gpu.overheadGiB;
  const totalGiB = wGiB + kvGiB + overheadGiB;
  const fitsVram = totalGiB <= gpu.vramGiB;

  const roofline = evaluateRoofline({
    gpu,
    model,
    phase: 'decode',
    batchSize: cfg.batchSize,
    ctxLen: cfg.ctxLen,
    weightBytesPerElem: wFmt.bytesPerElem,
    kvBytesPerElem: kFmt.bytesPerElem,
  });

  const checks = {
    fitsVram,
    meetsBatch: cfg.batchSize >= brief.minBatch,
    meetsCtx: cfg.ctxLen >= brief.minCtx,
    meetsUserSpeed: roofline.tokensPerSecPerUser >= brief.minTokPerUser,
    meetsTotalSpeed: roofline.totalTokensPerSec >= brief.minTotalTokPerSec,
    meetsQuality: !(brief.forbidDoubleInt4 && cfg.weightFmt === 'int4' && cfg.kvFmt === 'int4'),
  };

  return {
    wGiB,
    kvGiB,
    overheadGiB,
    totalGiB,
    vramLimitGiB: gpu.vramGiB,
    fitsVram,
    roofline,
    checks,
    passes: Object.values(checks).every(Boolean),
  };
}

export interface TetrisSolution {
  totalConfigs: number;
  passingCount: number;
  bestConfig: TetrisConfig;
  bestEval: TetrisEval;
}

/** Exhaustive search over all 3 × 3 × 7 × 7 = 441 configurations for a given brief. */
export function solveTetrisBrief(brief: TetrisBrief): TetrisSolution {
  let totalConfigs = 0;
  let passingCount = 0;
  let bestConfig: TetrisConfig | null = null;
  let bestEval: TetrisEval | null = null;

  const score = (e: TetrisEval) =>
    brief.optMetric === 'tokPerUser' ? e.roofline.tokensPerSecPerUser : e.roofline.totalTokensPerSec;

  for (const w of WEIGHT_FORMATS) {
    for (const k of KV_FORMATS) {
      for (const batchSize of TETRIS_BATCHES) {
        for (const ctxLen of TETRIS_CTXS) {
          totalConfigs++;
          const cfg: TetrisConfig = {
            weightFmt: w.id as 'fp16' | 'int8' | 'int4',
            kvFmt: k.id as 'fp16' | 'fp8' | 'int4',
            batchSize,
            ctxLen,
          };
          const ev = evaluateTetris(brief, cfg);
          if (ev.passes) {
            passingCount++;
            if (!bestEval || score(ev) > score(bestEval) + 1e-9) {
              bestConfig = cfg;
              bestEval = ev;
            }
          }
        }
      }
    }
  }

  return {
    totalConfigs,
    passingCount,
    bestConfig: bestConfig!,
    bestEval: bestEval!,
  };
}
