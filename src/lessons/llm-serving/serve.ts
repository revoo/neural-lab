// authored-by: Gemini Antigravity
/**
 * Core serving engine for Module 16's `llm-serving` lesson:
 *   1. Hardware & model constants (duplicated from `kv-cache/kv.ts` so lessons stay independent;
 *      pinned by a consistency test reproducing 205.2 tok/s on H100).
 *   2. Unified roofline step-cost model for mixed prefill-chunk + decode batches.
 *   3. Three KV memory allocators: contiguous reserve-max, contiguous reserve-grow (with real
 *      external fragmentation holes), and paged KV blocks (with optional prefix sharing & refcounts).
 *   4. Deterministic discrete-event scheduler supporting static and continuous batching, chunked
 *      prefill, LIFO preemption with recomputation, and per-step / per-request telemetry.
 */

export const KIB = 1024;
export const GIB = 1024 ** 3;

export interface ModelSpec {
  id: string;
  name: string;
  shortName: string;
  paramsB: number; // billions of parameters
  layers: number;
  qHeads: number;
  kvHeads: number;
  dHead: number;
}

export interface GpuSpec {
  id: string;
  name: string;
  shortName: string;
  vramGiB: number;
  bwTBps: number; // HBM/GDDR bandwidth in TB/s (10^12 B/s)
  peakTflops: number; // dense FP16/BF16 Tensor Core peak (10^12 FLOP/s)
  overheadGiB: number; // CUDA context + activation scratchpad (matches kv-cache)
}

export const LLAMA3_8B: ModelSpec = {
  id: 'llama3-8b',
  name: 'Llama-3 8B (GQA 4:1)',
  shortName: 'Llama-3 8B',
  paramsB: 8.03,
  layers: 32,
  qHeads: 32,
  kvHeads: 8,
  dHead: 128,
};

export const LLAMA3_70B: ModelSpec = {
  id: 'llama3-70b',
  name: 'Llama-3 70B (GQA 8:1)',
  shortName: 'Llama-3 70B',
  paramsB: 70.6,
  layers: 80,
  qHeads: 64,
  kvHeads: 8,
  dHead: 128,
};

export const RTX_4090: GpuSpec = {
  id: 'rtx-4090',
  name: 'NVIDIA RTX 4090 (24 GiB)',
  shortName: 'RTX 4090 24G',
  vramGiB: 24,
  bwTBps: 1.008,
  peakTflops: 165.2,
  overheadGiB: 1.5,
};

export const H100_80G: GpuSpec = {
  id: 'h100-80',
  name: 'NVIDIA H100 80GB SXM',
  shortName: 'H100 80G',
  vramGiB: 80,
  bwTBps: 3.35,
  peakTflops: 989,
  overheadGiB: 3.0,
};

export const FP16_BYTES = 2.0;

/** Named per-step fixed overhead (ms). Defaults to 0 — no invented overheads. */
export const DEFAULT_FIXED_OVERHEAD_MS = 0;

/** Exact KV cache bytes per token for one sequence. Llama-3 8B FP16 = 131,072 B (128 KiB). */
export function kvBytesPerToken(model: ModelSpec = LLAMA3_8B, kvBytesPerElem = FP16_BYTES): number {
  return 2 * model.layers * model.kvHeads * model.dHead * kvBytesPerElem;
}

/** Model weight footprint in bytes. Llama-3 8B FP16 = 16.06e9 B (14.96 GiB). */
export function weightsBytes(model: ModelSpec = LLAMA3_8B, weightBytesPerElem = FP16_BYTES): number {
  return model.paramsB * 1e9 * weightBytesPerElem;
}

/** Available VRAM in bytes for KV cache after weights and runtime scratchpad. */
export function kvBudgetBytes(
  gpu: GpuSpec = RTX_4090,
  model: ModelSpec = LLAMA3_8B,
  weightBytesPerElem = FP16_BYTES,
): number {
  return Math.max(0, gpu.vramGiB * GIB - weightsBytes(model, weightBytesPerElem) - gpu.overheadGiB * GIB);
}

/** Maximum whole tokens of KV cache that fit in the GPU's KV budget. */
export function kvCapacityTokens(
  gpu: GpuSpec = RTX_4090,
  model: ModelSpec = LLAMA3_8B,
  weightBytesPerElem = FP16_BYTES,
  kvBytesPerElem = FP16_BYTES,
): number {
  return Math.floor(kvBudgetBytes(gpu, model, weightBytesPerElem) / kvBytesPerToken(model, kvBytesPerElem));
}

/** Ridge point in FLOPs/byte: peak compute divided by memory bandwidth. */
export function ridgePoint(gpu: GpuSpec): number {
  return gpu.peakTflops / gpu.bwTBps;
}

// ------------------------------------------------------------------ 1. Unified roofline step-cost model

export interface BatchWorkItem {
  kind: 'decode' | 'prefill';
  /** Tokens already present in this sequence's KV cache before this step. */
  cachedTokens: number;
  /** New tokens processed in this step (1 for decode; chunk size for prefill). */
  newTokens: number;
}

export interface StepCost {
  flops: number;
  linearFlops: number;
  attnFlops: number;
  bytes: number;
  weightBytes: number;
  kvBytes: number;
  arithmeticIntensity: number;
  ridge: number;
  bottleneck: 'memory' | 'compute';
  computeUtilPct: number;
  memTimeMs: number;
  compTimeMs: number;
  stepTimeMs: number;
  decodeTokens: number;
  prefillTokens: number;
  totalNewTokens: number;
}

/**
 * First-principles hardware roofline ceiling for one mixed Transformer iteration.
 *
 * Derivation for each sequence processing `m` new tokens on top of `p` cached tokens:
 *   - Linear projections (Q/K/V/O + MLP): 2 · N · m FLOPs.
 *   - Causal attention: the t-th new token (1 ≤ t ≤ m) attends to (p + t) keys/values.
 *     Each Q·K dot product costs 2·d_head FLOPs and each weighted V sum costs 2·d_head FLOPs,
 *     giving 4·L·n_q·d_head FLOPs per causal pair. Summing over t = 1..m yields
 *     pairs = m·p + m·(m + 1) / 2.
 *     Notice:
 *       * Decode (m = 1, p = ctx - 1) → pairs = ctx → 4·L·n_q·d_head·ctx (matches kv-cache).
 *       * Full prefill (m = P, p = 0) → pairs = P(P+1)/2 ≈ P²/2 → 2·L·n_q·d_head·P² (matches kv-cache).
 *   - HBM traffic: model weights are read ONCE per step (N · b_w bytes). Each sequence reads its
 *     `p` cached KV tokens and writes its `m` new KV tokens, for (p + m) tokens of KV traffic.
 */
export function stepTime(
  items: BatchWorkItem[],
  gpu: GpuSpec = RTX_4090,
  model: ModelSpec = LLAMA3_8B,
  opts?: { weightBytesPerElem?: number; kvBytesPerElem?: number; fixedOverheadMs?: number },
): StepCost {
  const wBytes = opts?.weightBytesPerElem ?? FP16_BYTES;
  const kBytes = opts?.kvBytesPerElem ?? FP16_BYTES;
  const overheadMs = opts?.fixedOverheadMs ?? DEFAULT_FIXED_OVERHEAD_MS;

  const N = model.paramsB * 1e9;
  const kvTokBytes = kvBytesPerToken(model, kBytes);
  const attnPerPair = 4 * model.layers * model.qHeads * model.dHead;

  let totalNewTokens = 0;
  let decodeTokens = 0;
  let prefillTokens = 0;
  let totalAttnPairs = 0;
  let totalKvTokens = 0;

  for (const it of items) {
    const m = it.newTokens;
    const p = it.cachedTokens;
    totalNewTokens += m;
    if (it.kind === 'decode') decodeTokens += m;
    else prefillTokens += m;
    totalAttnPairs += m * p + (m * (m + 1)) / 2;
    totalKvTokens += p + m;
  }

  const linearFlops = 2 * N * totalNewTokens;
  const attnFlops = attnPerPair * totalAttnPairs;
  const flops = linearFlops + attnFlops;

  const weightBytesVal = items.length > 0 ? N * wBytes : 0;
  const kvBytesVal = totalKvTokens * kvTokBytes;
  const bytes = weightBytesVal + kvBytesVal;

  const memTimeMs = (bytes / (gpu.bwTBps * 1e12)) * 1000;
  const compTimeMs = (flops / (gpu.peakTflops * 1e12)) * 1000;
  const rawStepMs = Math.max(memTimeMs, compTimeMs);
  const stepTimeMs = items.length > 0 ? rawStepMs + overheadMs : 0;

  const ridge = ridgePoint(gpu);
  const arithmeticIntensity = bytes > 0 ? flops / bytes : 0;
  const bottleneck: 'memory' | 'compute' = arithmeticIntensity < ridge ? 'memory' : 'compute';
  const achievedTflops = rawStepMs > 0 ? flops / (rawStepMs / 1000) / 1e12 : 0;
  const computeUtilPct = gpu.peakTflops > 0 ? (achievedTflops / gpu.peakTflops) * 100 : 0;

  return {
    flops,
    linearFlops,
    attnFlops,
    bytes,
    weightBytes: weightBytesVal,
    kvBytes: kvBytesVal,
    arithmeticIntensity,
    ridge,
    bottleneck,
    computeUtilPct,
    memTimeMs,
    compTimeMs,
    stepTimeMs,
    decodeTokens,
    prefillTokens,
    totalNewTokens,
  };
}

/**
 * Exact latency breakdown for one request behind `usersAhead` identical requests served
 * strictly one at a time (FIFO, no batching). Drives Widget A (LatencyAnatomy).
 */
export function singleRequestLatency(params: {
  promptLen: number;
  replyLen: number;
  usersAhead?: number;
  gpu?: GpuSpec;
  model?: ModelSpec;
}) {
  const gpu = params.gpu ?? H100_80G;
  const model = params.model ?? LLAMA3_8B;
  const promptLen = Math.max(1, Math.round(params.promptLen));
  const replyLen = Math.max(1, Math.round(params.replyLen));
  const usersAhead = Math.max(0, Math.round(params.usersAhead ?? 0));

  const prefill = stepTime([{ kind: 'prefill', cachedTokens: 0, newTokens: promptLen }], gpu, model);
  let decodeTotalMs = 0;
  let decodeUtilSum = 0;
  for (let i = 1; i < replyLen; i++) {
    const s = stepTime([{ kind: 'decode', cachedTokens: promptLen + i - 1, newTokens: 1 }], gpu, model);
    decodeTotalMs += s.stepTimeMs;
    decodeUtilSum += s.computeUtilPct;
  }
  const meanTpotMs = replyLen > 1 ? decodeTotalMs / (replyLen - 1) : 0;
  const meanDecodeUtilPct = replyLen > 1 ? decodeUtilSum / (replyLen - 1) : 0;
  const oneRequestServiceMs = prefill.stepTimeMs + decodeTotalMs;
  const queueWaitMs = usersAhead * oneRequestServiceMs;
  const ttftMs = queueWaitMs + prefill.stepTimeMs;
  const totalLatencyMs = ttftMs + decodeTotalMs;

  return {
    queueWaitMs,
    prefillMs: prefill.stepTimeMs,
    decodeTotalMs,
    ttftMs,
    meanTpotMs,
    totalLatencyMs,
    queueShareOfTtft: ttftMs > 0 ? queueWaitMs / ttftMs : 0,
    prefillUtilPct: prefill.computeUtilPct,
    meanDecodeUtilPct,
    prefillBottleneck: prefill.bottleneck,
    decodeTokPerSec: meanTpotMs > 0 ? 1000 / meanTpotMs : 0,
  };
}

// ------------------------------------------------------------------ 2. Trace & configuration types

export interface TraceRequest {
  id: number;
  arrivalMs: number;
  promptLen: number;
  outputLen: number;
  /** Shared prefix length (e.g. 1,024-token system prompt). 0 when prompts are unshared. */
  sharedPrefixLen: number;
  /** Short human label for tables/tooltips (e.g. "chat" or "16k doc"). */
  kind: 'chat' | 'doc';
}

export interface WorkloadTrace {
  id: string;
  name: string;
  description: string;
  maxModelLen: number;
  requests: TraceRequest[];
}

export type BatchingMode = 'static' | 'continuous';
export type AllocatorMode = 'reserve-max' | 'reserve-grow' | 'paged';
export type AdmissionPolicy = 'fcfs' | 'sjf';

export interface ServingConfig {
  batching: BatchingMode;
  /** Static batching only: maximum requests gathered into one fixed batch. */
  staticBatchSize?: number;
  /** Static batching only: max ms to wait for a batch to fill before launching. */
  staticMaxWaitMs?: number;
  /** Maximum concurrent running sequences (active slots). */
  maxRunning: number;
  /** Per-step token budget for chunked prefill; null = no chunking (prefill runs whole). */
  tokenBudget: number | null;
  allocator: AllocatorMode;
  /** Tokens per physical KV block (paged allocator only). */
  blockSize: 8 | 16 | 32;
  /** Reuse physical blocks for identical shared prefixes (paged allocator only). */
  prefixSharing: boolean;
  admission: AdmissionPolicy;
  gpu?: GpuSpec;
  model?: ModelSpec;
  slo?: { ttftMs: number; tpotMs: number };
  /** Record lightweight per-step KV maps (off during bulk grid searches for speed). */
  recordSnapshots?: boolean;
}

// ------------------------------------------------------------------ 3. KV memory allocator

interface GrowSlab {
  start: number;
  size: number;
}

export interface MemorySnapshot {
  allocator: AllocatorMode;
  totalTokensCap: number;
  usedTokens: number;
  reservedTokens: number;
  internalWasteTokens: number;
  reservedUnusedTokens: number;
  externalFragTokens: number;
  totalBlocks: number;
  usedBlocks: number;
  sharedBlocks: number;
  /** For paged mode: owner request id per block (-1 = free, -2 = shared prefix). Capped/binned in UI. */
  blockOwners?: Int32Array;
  /** For paged mode: block table per active request (reqId → physical block indices). */
  blockTables?: Record<number, number[]>;
  /** For contiguous modes: allocated intervals [start, size, reqId, usedTokens]. */
  slabs?: { reqId: number; start: number; size: number; used: number }[];
  prefixRefCount: number;
}

class KvAllocator {
  readonly mode: AllocatorMode;
  readonly capTokens: number;
  readonly maxModelLen: number;
  readonly blockSize: number;
  readonly prefixSharing: boolean;
  readonly totalBlocks: number;

  // Accounting (for conservation tests)
  blocksAllocated = 0;
  blocksFreed = 0;

  // Paged state
  private blockOwner: Int32Array; // -1 free, -2 shared prefix, >=0 reqId
  private freeStack: number[];
  private reqBlocks = new Map<number, number[]>();
  private sharedPrefixBlocks: number[] = [];
  private prefixRefCount = 0;
  private prefixCached = false;

  // Contiguous state (used by reserve-max and reserve-grow)
  private slabs = new Map<number, GrowSlab>();
  private usedByReq = new Map<number, number>();

  constructor(
    mode: AllocatorMode,
    capTokens: number,
    maxModelLen: number,
    blockSize: number,
    prefixSharing: boolean,
  ) {
    this.mode = mode;
    this.capTokens = capTokens;
    this.maxModelLen = maxModelLen;
    this.blockSize = blockSize;
    this.prefixSharing = mode === 'paged' && prefixSharing;
    this.totalBlocks = Math.floor(capTokens / blockSize);
    this.blockOwner = new Int32Array(this.totalBlocks).fill(-1);
    this.freeStack = [];
    for (let i = this.totalBlocks - 1; i >= 0; i--) this.freeStack.push(i);
  }

  hasCachedPrefix(): boolean {
    return this.prefixSharing && this.prefixCached && this.sharedPrefixBlocks.length > 0;
  }

  sharedPrefixTokenCount(reqSharedPrefixLen: number): number {
    if (!this.prefixSharing || reqSharedPrefixLen < this.blockSize) return 0;
    return Math.floor(reqSharedPrefixLen / this.blockSize) * this.blockSize;
  }

  private allocBlock(owner: number): number {
    if (this.freeStack.length === 0) return -1;
    const b = this.freeStack.pop()!;
    this.blockOwner[b] = owner;
    this.blocksAllocated++;
    return b;
  }

  private freeBlock(b: number) {
    this.blockOwner[b] = -1;
    this.freeStack.push(b);
    this.blocksFreed++;
  }

  /** Evict unreferenced cached prefix blocks if memory is tight. Returns count freed. */
  evictUnusedPrefix(): number {
    if (!this.prefixCached || this.prefixRefCount > 0) return 0;
    const n = this.sharedPrefixBlocks.length;
    for (const b of this.sharedPrefixBlocks) this.freeBlock(b);
    this.sharedPrefixBlocks = [];
    this.prefixCached = false;
    return n;
  }

  /** Release any lingering unreferenced prefix cache at the end of a simulation. */
  flushCacheAtEnd() {
    this.evictUnusedPrefix();
  }

  freeBlockCount(): number {
    return this.freeStack.length + (this.prefixCached && this.prefixRefCount === 0 ? this.sharedPrefixBlocks.length : 0);
  }

  /** Find first contiguous free interval of at least `need` tokens in [0, capTokens). */
  private findContiguousHole(need: number, ignoreReqId = -1): number {
    const intervals: { start: number; end: number }[] = [];
    for (const [rid, s] of this.slabs) {
      if (rid === ignoreReqId) continue;
      intervals.push({ start: s.start, end: s.start + s.size });
    }
    intervals.sort((a, b) => a.start - b.start);
    let cursor = 0;
    for (const iv of intervals) {
      if (iv.start - cursor >= need) return cursor;
      cursor = Math.max(cursor, iv.end);
    }
    if (this.capTokens - cursor >= need) return cursor;
    return -1;
  }

  /** Initial slab size for reserve-grow: prompt + 64 tokens headroom, rounded to 64. */
  private initialGrowSize(promptLen: number): number {
    const raw = Math.ceil((promptLen + 64) / 64) * 64;
    return Math.min(this.maxModelLen, Math.max(64, raw));
  }

  /** Check whether a new request can be admitted right now (without preempting others). */
  canAdmit(req: TraceRequest, initialTokens: number): boolean {
    if (this.mode === 'reserve-max') {
      return this.findContiguousHole(this.maxModelLen) >= 0;
    }
    if (this.mode === 'reserve-grow') {
      return this.findContiguousHole(this.initialGrowSize(req.promptLen)) >= 0;
    }
    // Paged: need blocks for initial tokens (+1 headroom block so we don't thrash immediately)
    const sharedTok = this.sharedPrefixTokenCount(req.sharedPrefixLen);
    const sharedNeedBlocks = Math.floor(sharedTok / this.blockSize);
    const totalNeededBlocks = Math.max(1, Math.ceil(initialTokens / this.blockSize));
    const privateNeeded = this.hasCachedPrefix() && sharedNeedBlocks > 0
      ? Math.max(0, totalNeededBlocks - sharedNeedBlocks)
      : totalNeededBlocks;
    // Also check if we'll be populating the shared prefix ourselves
    const need = privateNeeded + 1;
    return this.freeBlockCount() >= need;
  }

  /**
   * Admit a request into memory. Returns how many shared prefix tokens were already cached
   * (so the scheduler can skip prefilling them!).
   */
  admit(req: TraceRequest, initialTokens: number): { ok: boolean; prefixHitTokens: number } {
    if (this.mode === 'reserve-max') {
      const start = this.findContiguousHole(this.maxModelLen);
      if (start < 0) return { ok: false, prefixHitTokens: 0 };
      this.slabs.set(req.id, { start, size: this.maxModelLen });
      this.usedByReq.set(req.id, initialTokens);
      return { ok: true, prefixHitTokens: 0 };
    }
    if (this.mode === 'reserve-grow') {
      const sz = this.initialGrowSize(req.promptLen);
      const start = this.findContiguousHole(sz);
      if (start < 0) return { ok: false, prefixHitTokens: 0 };
      this.slabs.set(req.id, { start, size: sz });
      this.usedByReq.set(req.id, initialTokens);
      return { ok: true, prefixHitTokens: 0 };
    }

    // Paged mode
    const sharedTok = this.sharedPrefixTokenCount(req.sharedPrefixLen);
    const sharedNum = Math.floor(sharedTok / this.blockSize);
    const totalNeed = Math.max(1, Math.ceil(initialTokens / this.blockSize));

    let prefixHitTokens = 0;
    const table: number[] = [];

    if (this.prefixSharing && sharedNum > 0 && this.prefixCached && this.sharedPrefixBlocks.length === sharedNum) {
      // Reuse cached shared prefix blocks!
      for (const b of this.sharedPrefixBlocks) table.push(b);
      this.prefixRefCount++;
      prefixHitTokens = sharedTok;
    }

    const extraNeeded = Math.max(0, totalNeed - table.length);
    if (this.freeStack.length < extraNeeded) this.evictUnusedPrefix();
    if (this.freeStack.length < extraNeeded) {
      if (prefixHitTokens > 0) this.prefixRefCount--;
      return { ok: false, prefixHitTokens: 0 };
    }

    for (let i = 0; i < extraNeeded; i++) {
      const b = this.allocBlock(req.id);
      table.push(b);
    }

    this.reqBlocks.set(req.id, table);
    this.usedByReq.set(req.id, Math.max(initialTokens, prefixHitTokens));
    return { ok: true, prefixHitTokens };
  }

  /**
   * Ensure request `reqId` has capacity for `newTotalTokens`. Returns false if OOM (caller
   * should preempt another sequence and retry).
   */
  ensureCapacity(req: TraceRequest, newTotalTokens: number): boolean {
    if (this.mode === 'reserve-max') {
      const slab = this.slabs.get(req.id);
      if (!slab || newTotalTokens > slab.size) return false;
      this.usedByReq.set(req.id, newTotalTokens);
      return true;
    }
    if (this.mode === 'reserve-grow') {
      const slab = this.slabs.get(req.id);
      if (!slab) return false;
      if (newTotalTokens <= slab.size) {
        this.usedByReq.set(req.id, newTotalTokens);
        return true;
      }
      // Need to grow: double slab (capped at maxModelLen)
      let nextSize = slab.size;
      while (nextSize < newTotalTokens) nextSize = Math.min(this.maxModelLen, nextSize * 2);
      // Try growing in place first
      let blocked = slab.start + nextSize > this.capTokens;
      if (!blocked) {
        for (const [rid, other] of this.slabs) {
          if (rid === req.id) continue;
          if (other.start < slab.start + nextSize && other.start + other.size > slab.start) {
            blocked = true;
            break;
          }
        }
      }
      if (!blocked) {
        slab.size = nextSize;
        this.usedByReq.set(req.id, newTotalTokens);
        return true;
      }
      // Relocate to another contiguous hole (leaves old spot free -> external fragmentation!)
      const newStart = this.findContiguousHole(nextSize, req.id);
      if (newStart < 0) return false;
      slab.start = newStart;
      slab.size = nextSize;
      this.usedByReq.set(req.id, newTotalTokens);
      return true;
    }

    // Paged mode
    const table = this.reqBlocks.get(req.id);
    if (!table) return false;
    const needBlocks = Math.ceil(newTotalTokens / this.blockSize);
    const missing = needBlocks - table.length;
    if (missing > 0) {
      if (this.freeStack.length < missing) this.evictUnusedPrefix();
      if (this.freeStack.length < missing) return false;
      for (let i = 0; i < missing; i++) table.push(this.allocBlock(req.id));
    }
    this.usedByReq.set(req.id, newTotalTokens);

    // Promote shared prefix blocks into the prefix cache once this request finishes prefilling them
    const sharedTok = this.sharedPrefixTokenCount(req.sharedPrefixLen);
    const sharedNum = Math.floor(sharedTok / this.blockSize);
    if (this.prefixSharing && !this.prefixCached && sharedNum > 0 && newTotalTokens >= sharedTok && table.length >= sharedNum) {
      this.sharedPrefixBlocks = table.slice(0, sharedNum);
      for (const b of this.sharedPrefixBlocks) this.blockOwner[b] = -2;
      this.prefixCached = true;
      this.prefixRefCount = 1;
    }
    return true;
  }

  /** Free all memory owned by `reqId` (on completion or preemption). */
  release(reqId: number) {
    this.usedByReq.delete(reqId);
    if (this.mode !== 'paged') {
      this.slabs.delete(reqId);
      return;
    }
    const table = this.reqBlocks.get(reqId);
    if (!table) return;
    this.reqBlocks.delete(reqId);
    let touchedShared = false;
    for (const b of table) {
      if (this.blockOwner[b] === -2) {
        touchedShared = true;
      } else {
        this.freeBlock(b);
      }
    }
    if (touchedShared) {
      this.prefixRefCount = Math.max(0, this.prefixRefCount - 1);
    }
  }

  /** Compute exact used/reserved/waste stats and optional block/slab layout for widgets. */
  stats(includeLayout = false): MemorySnapshot {
    let usedTokens = 0;
    for (const u of this.usedByReq.values()) usedTokens += u;

    if (this.mode !== 'paged') {
      let reservedTokens = 0;
      const slabs: { reqId: number; start: number; size: number; used: number }[] = [];
      let maxEnd = 0;
      for (const [reqId, s] of this.slabs) {
        reservedTokens += s.size;
        maxEnd = Math.max(maxEnd, s.start + s.size);
        if (includeLayout) slabs.push({ reqId, start: s.start, size: s.size, used: this.usedByReq.get(reqId) ?? 0 });
      }
      if (includeLayout) slabs.sort((a, b) => a.start - b.start);
      // External fragmentation = free holes sitting below the highest allocated address
      const externalFragTokens = Math.max(0, maxEnd - reservedTokens);
      return {
        allocator: this.mode,
        totalTokensCap: this.capTokens,
        usedTokens,
        reservedTokens,
        internalWasteTokens: 0,
        reservedUnusedTokens: Math.max(0, reservedTokens - usedTokens),
        externalFragTokens,
        totalBlocks: this.totalBlocks,
        usedBlocks: Math.ceil(reservedTokens / this.blockSize),
        sharedBlocks: 0,
        slabs: includeLayout ? slabs : undefined,
        prefixRefCount: 0,
      };
    }

    const usedBlocks = this.totalBlocks - this.freeStack.length;
    const sharedBlocks = this.prefixCached ? this.sharedPrefixBlocks.length : 0;
    const reservedTokens = usedBlocks * this.blockSize;
    // Internal waste: unfilled slots in each sequence's last private block
    let internalWasteTokens = 0;
    for (const u of this.usedByReq.values()) {
      const rem = u % this.blockSize;
      if (rem > 0) internalWasteTokens += this.blockSize - rem;
    }
    // Note: when prefixSharing is on, logical usedTokens across sequences can exceed physical
    // reservedTokens because shared prefix tokens are stored only once!
    const blockTables: Record<number, number[]> | undefined = includeLayout
      ? Object.fromEntries([...this.reqBlocks.entries()].map(([k, v]) => [k, [...v]]))
      : undefined;

    return {
      allocator: 'paged',
      totalTokensCap: this.capTokens,
      usedTokens,
      reservedTokens,
      internalWasteTokens,
      reservedUnusedTokens: 0,
      externalFragTokens: 0,
      totalBlocks: this.totalBlocks,
      usedBlocks,
      sharedBlocks,
      blockOwners: includeLayout ? Int32Array.from(this.blockOwner) : undefined,
      blockTables,
      prefixRefCount: this.prefixRefCount,
    };
  }
}

// ------------------------------------------------------------------ 4. Simulator telemetry types & engine

export interface StepSlot {
  slotIndex: number;
  reqId: number | null; // null = idle/empty slot
  phase: 'prefill' | 'decode' | 'idle';
  tokensThisStep: number;
}

export interface SimStep {
  step: number;
  startMs: number;
  durationMs: number;
  endMs: number;
  slots: StepSlot[];
  activeSlots: number;
  slotCapacity: number;
  decodeTokens: number;
  prefillTokens: number;
  flops: number;
  bytes: number;
  bottleneck: 'memory' | 'compute';
  computeUtilPct: number;
  kvUsedTokens: number;
  kvReservedTokens: number;
  kvUsedBlocks: number;
  kvTotalBlocks: number;
  sharedBlocks: number;
  maxTpotMs: number;
  preemptionsThisStep: number;
  snapshot?: MemorySnapshot;
}

export interface RequestResult {
  id: number;
  kind: 'chat' | 'doc';
  arrivalMs: number;
  admitMs: number;
  firstTokenMs: number;
  finishMs: number;
  promptLen: number;
  outputLen: number;
  sharedPrefixLen: number;
  prefixHitTokens: number;
  ttftMs: number;
  meanTpotMs: number;
  maxTpotMs: number;
  totalLatencyMs: number;
  preemptions: number;
  meetsSlo: boolean;
}

export interface SimEvent {
  timeMs: number;
  step: number;
  type: 'admit' | 'retire' | 'preempt' | 'prefix-hit';
  reqId: number;
  message: string;
}

export interface SimMetrics {
  totalTimeMs: number;
  totalSteps: number;
  totalOutputTokens: number;
  expectedOutputTokens: number;
  throughputTokPerSec: number;
  requestsPerSec: number;
  ttftP50Ms: number;
  ttftP95Ms: number;
  tpotP50Ms: number;
  tpotP95Ms: number;
  worstTpotMs: number;
  sloAttainment: number; // fraction 0..1 of requests meeting both TTFT and TPOT SLOs
  goodputReqPerSec: number; // SLO-meeting requests / total wall time (sec)
  idleSlotShare: number; // fraction of (step × slotCapacity) left idle
  meanComputeUtilPct: number;
  peakKvUsedTokens: number;
  peakKvReservedTokens: number;
  peakKvWasteTokens: number;
  totalPrefixHitTokens: number;
  preemptions: number;
  blocksAllocated: number;
  blocksFreed: number;
}

export interface SimResult {
  steps: SimStep[];
  requests: RequestResult[];
  events: SimEvent[];
  metrics: SimMetrics;
}

interface ActiveSeq {
  req: TraceRequest;
  slotIndex: number;
  admitMs: number;
  prefilledTokens: number;
  generatedTokens: number;
  firstTokenMs: number | null;
  lastTokenMs: number | null;
  tpotSamples: number[];
  maxTpotMs: number;
  preemptions: number;
  prefixHitTokens: number;
}

/** Nearest-rank percentile (p in 0..100). */
export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

/**
 * Deterministic discrete-event serving simulator.
 * Drives Widgets A–D and Lab 1 (The Serving Desk).
 */
export function simulate(trace: WorkloadTrace, config: ServingConfig): SimResult {
  const gpu = config.gpu ?? RTX_4090;
  const model = config.model ?? LLAMA3_8B;
  const capTokens = kvCapacityTokens(gpu, model);
  const allocator = new KvAllocator(
    config.allocator,
    capTokens,
    trace.maxModelLen,
    config.blockSize,
    config.prefixSharing,
  );

  // Clone & sort incoming requests by arrival time, then id
  const pending = [...trace.requests].sort((a, b) => a.arrivalMs - b.arrivalMs || a.id - b.id);
  let nextPendingIdx = 0;

  const waiting: TraceRequest[] = [];
  const running: ActiveSeq[] = [];
  const freeSlotIds: number[] = [];
  const slotCap = config.batching === 'static'
    ? Math.max(1, config.staticBatchSize ?? config.maxRunning)
    : Math.max(1, config.maxRunning);
  for (let i = slotCap - 1; i >= 0; i--) freeSlotIds.push(i);

  const completed = new Map<number, RequestResult>();
  const preemptCounts = new Map<number, number>();
  const firstAdmitMs = new Map<number, number>();
  const steps: SimStep[] = [];
  const events: SimEvent[] = [];

  let nowMs = pending.length > 0 ? pending[0].arrivalMs : 0;
  const startMs = nowMs;
  let stepIdx = 0;

  let peakKvUsedTokens = 0;
  let peakKvReservedTokens = 0;
  let peakKvWasteTokens = 0;
  let totalPreemptions = 0;
  let totalPrefixHitTokens = 0;
  let totalSlotSteps = 0;
  let idleSlotSteps = 0;
  let utilWeightedSum = 0;
  let activeTimeSum = 0;

  // Static batching state: when a batch is running, no new requests join until all finish.
  let staticBatchSlots: (ActiveSeq | null)[] = [];
  let staticBatchSizeActual = 0;

  const enqueueArrivals = (upToMs: number) => {
    while (nextPendingIdx < pending.length && pending[nextPendingIdx].arrivalMs <= upToMs + 1e-9) {
      waiting.push(pending[nextPendingIdx++]);
    }
  };

  const sortWaiting = () => {
    if (config.admission === 'sjf') {
      waiting.sort((a, b) => a.promptLen - b.promptLen || a.arrivalMs - b.arrivalMs || a.id - b.id);
    }
  };

  const tryAdmitOne = (req: TraceRequest, slotIndex: number): ActiveSeq | null => {
    // Initial tokens needed on admission: if unchunked, full prompt; if chunked, first chunk (or shared prefix)
    const sharedTok = allocator.hasCachedPrefix() ? allocator.sharedPrefixTokenCount(req.sharedPrefixLen) : 0;
    const uncachedPrompt = Math.max(0, req.promptLen - sharedTok);
    const firstChunk = config.tokenBudget === null ? uncachedPrompt : Math.min(uncachedPrompt, config.tokenBudget);
    const initialNeed = Math.max(1, sharedTok + firstChunk);

    if (!allocator.canAdmit(req, initialNeed)) return null;
    const res = allocator.admit(req, initialNeed);
    if (!res.ok) return null;

    if (!firstAdmitMs.has(req.id)) firstAdmitMs.set(req.id, nowMs);
    totalPrefixHitTokens += res.prefixHitTokens;

    if (res.prefixHitTokens > 0) {
      events.push({
        timeMs: nowMs,
        step: stepIdx,
        type: 'prefix-hit',
        reqId: req.id,
        message: `Req #${req.id}: prefix cache hit (${res.prefixHitTokens} tokens reused, ${req.promptLen - res.prefixHitTokens} new)`,
      });
    } else {
      events.push({
        timeMs: nowMs,
        step: stepIdx,
        type: 'admit',
        reqId: req.id,
        message: `Req #${req.id} admitted (${req.promptLen} prompt, ${req.outputLen} output)`,
      });
    }

    return {
      req,
      slotIndex,
      admitMs: firstAdmitMs.get(req.id)!,
      prefilledTokens: res.prefixHitTokens,
      generatedTokens: 0,
      firstTokenMs: null,
      lastTokenMs: null,
      tpotSamples: [],
      maxTpotMs: 0,
      preemptions: preemptCounts.get(req.id) ?? 0,
      prefixHitTokens: res.prefixHitTokens,
    };
  };

  const preemptMostRecent = (protectReqId: number): boolean => {
    // Evict the most recently admitted running request (other than the protected one)
    let victimIdx = -1;
    for (let i = running.length - 1; i >= 0; i--) {
      if (running[i].req.id !== protectReqId) {
        victimIdx = i;
        break;
      }
    }
    if (victimIdx < 0) return false;
    const victim = running.splice(victimIdx, 1)[0];
    allocator.release(victim.req.id);
    freeSlotIds.push(victim.slotIndex);
    freeSlotIds.sort((a, b) => b - a); // pop lowest slot id first
    const cnt = (preemptCounts.get(victim.req.id) ?? 0) + 1;
    preemptCounts.set(victim.req.id, cnt);
    totalPreemptions++;
    // Re-queue at front so it resumes as soon as blocks free up (recompute on resume)
    waiting.unshift(victim.req);
    events.push({
      timeMs: nowMs,
      step: stepIdx,
      type: 'preempt',
      reqId: victim.req.id,
      message: `Preempted Req #${victim.req.id} (freed KV, queued for recompute; preemption #${cnt})`,
    });
    return true;
  };

  const retireSeq = (seq: ActiveSeq, finishMs: number) => {
    allocator.release(seq.req.id);
    const firstMs = seq.firstTokenMs ?? finishMs;
    const ttftMs = firstMs - seq.req.arrivalMs;
    const meanTpotMs = seq.tpotSamples.length > 0
      ? seq.tpotSamples.reduce((a, b) => a + b, 0) / seq.tpotSamples.length
      : 0;
    const slo = config.slo;
    const meetsSlo = slo
      ? ttftMs <= slo.ttftMs + 1e-9 && (seq.req.outputLen <= 1 || seq.maxTpotMs <= slo.tpotMs + 1e-9)
      : true;

    completed.set(seq.req.id, {
      id: seq.req.id,
      kind: seq.req.kind,
      arrivalMs: seq.req.arrivalMs,
      admitMs: seq.admitMs,
      firstTokenMs: firstMs,
      finishMs,
      promptLen: seq.req.promptLen,
      outputLen: seq.req.outputLen,
      sharedPrefixLen: seq.req.sharedPrefixLen,
      prefixHitTokens: seq.prefixHitTokens,
      ttftMs,
      meanTpotMs,
      maxTpotMs: seq.maxTpotMs,
      totalLatencyMs: finishMs - seq.req.arrivalMs,
      preemptions: preemptCounts.get(seq.req.id) ?? 0,
      meetsSlo,
    });
    events.push({
      timeMs: finishMs,
      step: stepIdx,
      type: 'retire',
      reqId: seq.req.id,
      message: `Req #${seq.req.id} finished (TTFT ${ttftMs.toFixed(0)} ms, worst TPOT ${seq.maxTpotMs.toFixed(1)} ms)`,
    });
  };

  const MAX_STEPS = 200_000;

  while (completed.size < trace.requests.length && stepIdx < MAX_STEPS) {
    enqueueArrivals(nowMs);

    if (config.batching === 'static') {
      // Static batching: only form a new batch when the current batch is completely empty
      if (running.length === 0) {
        if (waiting.length === 0) {
          if (nextPendingIdx < pending.length) {
            nowMs = pending[nextPendingIdx].arrivalMs;
            enqueueArrivals(nowMs);
          } else {
            break;
          }
        }
        // Optionally wait up to staticMaxWaitMs for the batch to fill
        const maxWait = config.staticMaxWaitMs ?? 0;
        const earliestArrival = waiting[0].arrivalMs;
        const deadline = earliestArrival + maxWait;
        while (
          waiting.length < slotCap &&
          nextPendingIdx < pending.length &&
          pending[nextPendingIdx].arrivalMs <= deadline + 1e-9
        ) {
          nowMs = Math.max(nowMs, pending[nextPendingIdx].arrivalMs);
          enqueueArrivals(nowMs);
        }
        sortWaiting();
        // Admit up to slotCap requests that fit in memory
        staticBatchSlots = [];
        let slotI = 0;
        while (slotI < slotCap && waiting.length > 0) {
          const req = waiting[0];
          const seq = tryAdmitOne(req, slotI);
          if (!seq) break;
          waiting.shift();
          running.push(seq);
          staticBatchSlots.push(seq);
          slotI++;
        }
        staticBatchSizeActual = staticBatchSlots.length;
        if (running.length === 0) {
          // Safety guard if even 1 request couldn't fit
          break;
        }
      }

      // Run one step for the static batch
      const workItems: BatchWorkItem[] = [];
      const activePlan: { seq: ActiveSeq; kind: 'prefill' | 'decode'; tokens: number }[] = [];

      // Check if any request in the batch still needs prefill
      const needsPrefill = running.filter((s) => s.prefilledTokens < s.req.promptLen);
      if (needsPrefill.length > 0) {
        let budgetLeft = config.tokenBudget ?? Infinity;
        for (const seq of needsPrefill) {
          const rem = seq.req.promptLen - seq.prefilledTokens;
          const take = Math.min(rem, budgetLeft);
          if (take <= 0) continue;
          // Ensure KV capacity for prefilledTokens + take
          if (!allocator.ensureCapacity(seq.req, seq.prefilledTokens + take)) {
            // In static batching, shouldn't happen unless reserve-grow overflows; skip/break
            continue;
          }
          workItems.push({ kind: 'prefill', cachedTokens: seq.prefilledTokens, newTokens: take });
          activePlan.push({ seq, kind: 'prefill', tokens: take });
          budgetLeft -= take;
        }
      } else {
        // All prefilled: decode 1 token for each unfinished request in the batch
        for (const seq of [...running]) {
          const nextTotal = seq.req.promptLen + seq.generatedTokens;
          while (!allocator.ensureCapacity(seq.req, nextTotal)) {
            if (!preemptMostRecent(seq.req.id)) break;
          }
          workItems.push({ kind: 'decode', cachedTokens: nextTotal - 1, newTokens: 1 });
          activePlan.push({ seq, kind: 'decode', tokens: 1 });
        }
      }

      const cost = stepTime(workItems, gpu, model);
      const stepEndMs = nowMs + cost.stepTimeMs;
      let stepMaxTpot = 0;

      // Record Gantt slots (length = staticBatchSizeActual, showing empty padded slots!)
      const planByReq = new Map(activePlan.map((p) => [p.seq.req.id, p]));
      const stepSlots: StepSlot[] = [];
      let activeCount = 0;
      for (let i = 0; i < staticBatchSizeActual; i++) {
        const s = staticBatchSlots[i];
        const plan = s ? planByReq.get(s.req.id) : undefined;
        if (s && plan) {
          activeCount++;
          stepSlots.push({ slotIndex: i, reqId: s.req.id, phase: plan.kind, tokensThisStep: plan.tokens });
        } else {
          stepSlots.push({ slotIndex: i, reqId: null, phase: 'idle', tokensThisStep: 0 });
        }
      }

      totalSlotSteps += staticBatchSizeActual;
      idleSlotSteps += staticBatchSizeActual - activeCount;

      // Apply token progress
      for (const p of activePlan) {
        const seq = p.seq;
        if (p.kind === 'prefill') {
          seq.prefilledTokens += p.tokens;
          if (seq.prefilledTokens >= seq.req.promptLen) {
            // Prefill completion emits output token #1
            seq.generatedTokens = 1;
            seq.firstTokenMs = stepEndMs;
            seq.lastTokenMs = stepEndMs;
          }
        } else {
          const gap = seq.lastTokenMs !== null ? stepEndMs - seq.lastTokenMs : cost.stepTimeMs;
          seq.tpotSamples.push(gap);
          seq.maxTpotMs = Math.max(seq.maxTpotMs, gap);
          stepMaxTpot = Math.max(stepMaxTpot, gap);
          seq.lastTokenMs = stepEndMs;
          seq.generatedTokens++;
        }
      }

      // Check memory stats BEFORE retiring so peak reflects live step memory
      const mem = allocator.stats(Boolean(config.recordSnapshots));
      peakKvUsedTokens = Math.max(peakKvUsedTokens, mem.usedTokens);
      peakKvReservedTokens = Math.max(peakKvReservedTokens, mem.reservedTokens);
      peakKvWasteTokens = Math.max(peakKvWasteTokens, mem.internalWasteTokens + mem.reservedUnusedTokens + mem.externalFragTokens);

      // Retire completed sequences (their static slot stays blocked until the whole batch finishes!)
      for (let i = running.length - 1; i >= 0; i--) {
        const seq = running[i];
        if (seq.generatedTokens >= seq.req.outputLen) {
          running.splice(i, 1);
          staticBatchSlots[seq.slotIndex] = null;
          retireSeq(seq, stepEndMs);
        }
      }

      utilWeightedSum += cost.computeUtilPct * cost.stepTimeMs;
      activeTimeSum += cost.stepTimeMs;

      steps.push({
        step: stepIdx,
        startMs: nowMs,
        durationMs: cost.stepTimeMs,
        endMs: stepEndMs,
        slots: stepSlots,
        activeSlots: activeCount,
        slotCapacity: staticBatchSizeActual,
        decodeTokens: cost.decodeTokens,
        prefillTokens: cost.prefillTokens,
        flops: cost.flops,
        bytes: cost.bytes,
        bottleneck: cost.bottleneck,
        computeUtilPct: cost.computeUtilPct,
        kvUsedTokens: mem.usedTokens,
        kvReservedTokens: mem.reservedTokens,
        kvUsedBlocks: mem.usedBlocks,
        kvTotalBlocks: mem.totalBlocks,
        sharedBlocks: mem.sharedBlocks,
        maxTpotMs: stepMaxTpot,
        preemptionsThisStep: 0,
        snapshot: config.recordSnapshots ? mem : undefined,
      });

      nowMs = stepEndMs;
      stepIdx++;
      continue;
    }

    // ---------------- Continuous (iteration-level) batching ----------------
    if (running.length === 0 && waiting.length === 0) {
      if (nextPendingIdx < pending.length) {
        nowMs = pending[nextPendingIdx].arrivalMs;
        enqueueArrivals(nowMs);
      } else {
        break;
      }
    }

    // 1. Admit waiting requests into free slots if memory permits
    sortWaiting();
    while (running.length < slotCap && waiting.length > 0 && freeSlotIds.length > 0) {
      const req = waiting[0];
      const slotIndex = freeSlotIds[freeSlotIds.length - 1];
      const seq = tryAdmitOne(req, slotIndex);
      if (!seq) break;
      freeSlotIds.pop();
      waiting.shift();
      running.push(seq);
    }

    if (running.length === 0) {
      // No request could be admitted (shouldn't happen when maxModelLen <= capTokens)
      break;
    }

    // 2. Build this iteration's batch: decode tokens first, then remaining token budget to prefills
    let preemptionsThisStep = 0;
    const decodeSeqs: ActiveSeq[] = [];
    const prefillSeqs: ActiveSeq[] = [];
    for (const s of running) {
      if (s.prefilledTokens >= s.req.promptLen) decodeSeqs.push(s);
      else prefillSeqs.push(s);
    }

    const workItems: BatchWorkItem[] = [];
    const activePlan: { seq: ActiveSeq; kind: 'prefill' | 'decode'; tokens: number }[] = [];

    // Ensure KV space for decoding sequences (LIFO preemption if out of blocks)
    for (let i = 0; i < decodeSeqs.length; i++) {
      const seq = decodeSeqs[i];
      if (!running.includes(seq)) continue; // may have been preempted by an earlier seq in this loop
      const nextTotal = seq.req.promptLen + seq.generatedTokens;
      let ok = allocator.ensureCapacity(seq.req, nextTotal);
      while (!ok) {
        if (!preemptMostRecent(seq.req.id)) break;
        preemptionsThisStep++;
        ok = allocator.ensureCapacity(seq.req, nextTotal);
      }
      if (ok && running.includes(seq)) {
        workItems.push({ kind: 'decode', cachedTokens: nextTotal - 1, newTokens: 1 });
        activePlan.push({ seq, kind: 'decode', tokens: 1 });
      }
    }

    // Assign remaining token budget to pending prefills
    let budgetLeft = config.tokenBudget === null
      ? Infinity
      : Math.max(0, config.tokenBudget - activePlan.length);

    for (const seq of prefillSeqs) {
      if (!running.includes(seq)) continue;
      if (budgetLeft <= 0) break;
      const rem = seq.req.promptLen - seq.prefilledTokens;
      const take = config.tokenBudget === null ? rem : Math.min(rem, budgetLeft);
      if (take <= 0) continue;
      const targetTotal = seq.prefilledTokens + take;
      let ok = allocator.ensureCapacity(seq.req, targetTotal);
      while (!ok) {
        if (!preemptMostRecent(seq.req.id)) break;
        preemptionsThisStep++;
        ok = allocator.ensureCapacity(seq.req, targetTotal);
      }
      if (ok && running.includes(seq)) {
        workItems.push({ kind: 'prefill', cachedTokens: seq.prefilledTokens, newTokens: take });
        activePlan.push({ seq, kind: 'prefill', tokens: take });
        budgetLeft -= take;
      }
    }

    // If budget was smaller than active decodes AND no decodes ran (can't happen since decodes always run),
    // guarantee at least one prefill chunk makes forward progress
    if (activePlan.length === 0 && running.length > 0) {
      const seq = running[0];
      const rem = seq.req.promptLen - seq.prefilledTokens;
      const take = config.tokenBudget === null ? rem : Math.min(rem, Math.max(1, config.tokenBudget));
      if (allocator.ensureCapacity(seq.req, seq.prefilledTokens + take)) {
        workItems.push({ kind: 'prefill', cachedTokens: seq.prefilledTokens, newTokens: take });
        activePlan.push({ seq, kind: 'prefill', tokens: take });
      }
    }

    const cost = stepTime(workItems, gpu, model);
    const stepEndMs = nowMs + cost.stepTimeMs;
    let stepMaxTpot = 0;

    // Build slot telemetry
    const planBySlot = new Map(activePlan.map((p) => [p.seq.slotIndex, p]));
    const stepSlots: StepSlot[] = [];
    for (let i = 0; i < slotCap; i++) {
      const p = planBySlot.get(i);
      if (p) {
        stepSlots.push({ slotIndex: i, reqId: p.seq.req.id, phase: p.kind, tokensThisStep: p.tokens });
      } else {
        stepSlots.push({ slotIndex: i, reqId: null, phase: 'idle', tokensThisStep: 0 });
      }
    }
    const effectiveCap = Math.min(slotCap, running.length + waiting.length);
    const activeCount = activePlan.length;
    totalSlotSteps += Math.max(activeCount, effectiveCap);
    idleSlotSteps += Math.max(0, effectiveCap - activeCount);

    // Apply progress
    for (const p of activePlan) {
      const seq = p.seq;
      if (p.kind === 'prefill') {
        seq.prefilledTokens += p.tokens;
        if (seq.prefilledTokens >= seq.req.promptLen) {
          seq.generatedTokens = 1;
          seq.firstTokenMs = stepEndMs;
          seq.lastTokenMs = stepEndMs;
        }
      } else {
        const gap = seq.lastTokenMs !== null ? stepEndMs - seq.lastTokenMs : cost.stepTimeMs;
        seq.tpotSamples.push(gap);
        seq.maxTpotMs = Math.max(seq.maxTpotMs, gap);
        stepMaxTpot = Math.max(stepMaxTpot, gap);
        seq.lastTokenMs = stepEndMs;
        seq.generatedTokens++;
      }
    }

    // Snapshot memory BEFORE retiring finished sequences
    const mem = allocator.stats(Boolean(config.recordSnapshots));
    peakKvUsedTokens = Math.max(peakKvUsedTokens, mem.usedTokens);
    peakKvReservedTokens = Math.max(peakKvReservedTokens, mem.reservedTokens);
    peakKvWasteTokens = Math.max(peakKvWasteTokens, mem.internalWasteTokens + mem.reservedUnusedTokens + mem.externalFragTokens);

    // Retire finished requests immediately (freeing their slot for the very next iteration!)
    for (let i = running.length - 1; i >= 0; i--) {
      const seq = running[i];
      if (seq.generatedTokens >= seq.req.outputLen) {
        running.splice(i, 1);
        freeSlotIds.push(seq.slotIndex);
        freeSlotIds.sort((a, b) => b - a);
        retireSeq(seq, stepEndMs);
      }
    }

    utilWeightedSum += cost.computeUtilPct * cost.stepTimeMs;
    activeTimeSum += cost.stepTimeMs;

    steps.push({
      step: stepIdx,
      startMs: nowMs,
      durationMs: cost.stepTimeMs,
      endMs: stepEndMs,
      slots: stepSlots,
      activeSlots: activeCount,
      slotCapacity: slotCap,
      decodeTokens: cost.decodeTokens,
      prefillTokens: cost.prefillTokens,
      flops: cost.flops,
      bytes: cost.bytes,
      bottleneck: cost.bottleneck,
      computeUtilPct: cost.computeUtilPct,
      kvUsedTokens: mem.usedTokens,
      kvReservedTokens: mem.reservedTokens,
      kvUsedBlocks: mem.usedBlocks,
      kvTotalBlocks: mem.totalBlocks,
      sharedBlocks: mem.sharedBlocks,
      maxTpotMs: stepMaxTpot,
      preemptionsThisStep,
      snapshot: config.recordSnapshots ? mem : undefined,
    });

    nowMs = stepEndMs;
    stepIdx++;
  }

  allocator.flushCacheAtEnd();

  const requests = trace.requests.map((r) => completed.get(r.id)!).filter(Boolean);
  const totalTimeMs = Math.max(1e-6, nowMs - startMs);
  const totalOutputTokens = requests.reduce((s, r) => s + r.outputLen, 0);
  const expectedOutputTokens = trace.requests.reduce((s, r) => s + r.outputLen, 0);
  const ttfts = requests.map((r) => r.ttftMs);
  const meanTpots = requests.filter((r) => r.outputLen > 1).map((r) => r.meanTpotMs);
  const maxTpots = requests.filter((r) => r.outputLen > 1).map((r) => r.maxTpotMs);
  const sloPasses = requests.filter((r) => r.meetsSlo).length;

  const metrics: SimMetrics = {
    totalTimeMs,
    totalSteps: steps.length,
    totalOutputTokens,
    expectedOutputTokens,
    throughputTokPerSec: totalOutputTokens / (totalTimeMs / 1000),
    requestsPerSec: requests.length / (totalTimeMs / 1000),
    ttftP50Ms: percentile(ttfts, 50),
    ttftP95Ms: percentile(ttfts, 95),
    tpotP50Ms: percentile(meanTpots, 50),
    tpotP95Ms: percentile(maxTpots, 95),
    worstTpotMs: maxTpots.length > 0 ? Math.max(...maxTpots) : 0,
    sloAttainment: requests.length > 0 ? sloPasses / requests.length : 0,
    goodputReqPerSec: sloPasses / (totalTimeMs / 1000),
    idleSlotShare: totalSlotSteps > 0 ? idleSlotSteps / totalSlotSteps : 0,
    meanComputeUtilPct: activeTimeSum > 0 ? utilWeightedSum / activeTimeSum : 0,
    peakKvUsedTokens,
    peakKvReservedTokens,
    peakKvWasteTokens,
    totalPrefixHitTokens,
    preemptions: totalPreemptions,
    blocksAllocated: allocator.blocksAllocated,
    blocksFreed: allocator.blocksFreed,
  };

  return { steps, requests, events, metrics };
}

// ------------------------------------------------------------------ 5. Standalone KV memory explorer snapshot (Widget D)

export interface MemoryMapReport {
  mode: AllocatorMode;
  prefixSharing: boolean;
  blockSize: 8 | 16 | 32;
  maxModelLen: number;
  kvCapTokens: number;
  kvBudgetGiB: number;
  sequencesFit: number;
  totalRequested: number;
  usedTokens: number;
  reservedTokens: number;
  internalWasteTokens: number;
  reservedUnusedTokens: number;
  externalFragTokens: number;
  totalWasteTokens: number;
  wastePct: number;
  sharedBlocks: number;
  usedBlocks: number;
  totalBlocks: number;
  prefixRefCount: number;
  /** Resulting decode roofline throughput (tok/s) at the batch size that fits! */
  decodeThroughputTokPerSec: number;
  decodePerUserTokPerSec: number;
  decodeComputeUtilPct: number;
  decodeBottleneck: 'memory' | 'compute';
  /** Per-sequence block table (paged mode) or slab (contiguous modes) for inspection. */
  fittedSeqs: {
    id: number;
    tokens: number;
    sharedPrefix: number;
    blocks?: number[];
    slab?: { start: number; size: number };
  }[];
  /** Compressed 256-cell visual map of physical VRAM occupancy for clean responsive rendering. */
  cells: { owner: number; kind: 'free' | 'used' | 'reserved-waste' | 'internal-waste' | 'frag' | 'shared' }[];
}

/**
 * Packs a list of live conversations into the GPU's KV pool under the chosen allocator mode,
 * returning exact capacity, waste breakdown, block tables, and the resulting decode throughput.
 * For `reserve-grow`, simulates a wave of arrivals, finishes and reallocations so external
 * fragmentation holes appear genuinely rather than being faked.
 */
export function evaluateKvMemoryMap(params: {
  sequences: { id: number; promptLen: number; currentLen: number; sharedPrefixLen: number }[];
  mode: AllocatorMode;
  blockSize: 8 | 16 | 32;
  prefixSharing: boolean;
  maxModelLen: number;
  gpu?: GpuSpec;
  model?: ModelSpec;
}): MemoryMapReport {
  const gpu = params.gpu ?? RTX_4090;
  const model = params.model ?? LLAMA3_8B;
  const capTokens = kvCapacityTokens(gpu, model);
  const budgetGiB = kvBudgetBytes(gpu, model) / GIB;
  const alloc = new KvAllocator(params.mode, capTokens, params.maxModelLen, params.blockSize, params.prefixSharing);

  const fitted: { id: number; tokens: number; sharedPrefix: number }[] = [];

  if (params.mode === 'reserve-grow') {
    // Realistic lifecycle: admit first wave, let every 3rd request finish to leave holes,
    // grow the remaining ones so some relocate, then try admitting the rest.
    const wave1 = params.sequences.slice(0, Math.min(20, params.sequences.length));
    const active = new Map<number, { id: number; promptLen: number; currentLen: number; sharedPrefixLen: number }>();
    for (const s of wave1) {
      const req: TraceRequest = { id: s.id, arrivalMs: 0, promptLen: s.promptLen, outputLen: 64, sharedPrefixLen: s.sharedPrefixLen, kind: 'chat' };
      if (alloc.admit(req, s.promptLen).ok) active.set(s.id, s);
    }
    // Every 3rd sequence finishes, freeing small contiguous slabs
    for (const s of wave1) {
      if (s.id % 3 === 2 && active.has(s.id)) {
        alloc.release(s.id);
        active.delete(s.id);
      }
    }
    // Survivors grow to their currentLen (causing real reallocations & fragmentation holes)
    for (const s of active.values()) {
      const req: TraceRequest = { id: s.id, arrivalMs: 0, promptLen: s.promptLen, outputLen: 64, sharedPrefixLen: s.sharedPrefixLen, kind: 'chat' };
      alloc.ensureCapacity(req, s.currentLen);
    }
    // Admit remaining sequences into whatever contiguous holes fit
    for (const s of params.sequences) {
      if (active.has(s.id)) continue;
      const req: TraceRequest = { id: s.id, arrivalMs: 0, promptLen: s.promptLen, outputLen: 64, sharedPrefixLen: s.sharedPrefixLen, kind: 'chat' };
      if (alloc.admit(req, s.promptLen).ok) {
        if (alloc.ensureCapacity(req, s.currentLen)) {
          active.set(s.id, s);
        } else {
          alloc.release(s.id);
        }
      }
    }
    for (const s of active.values()) fitted.push({ id: s.id, tokens: s.currentLen, sharedPrefix: s.sharedPrefixLen });
  } else {
    for (const s of params.sequences) {
      const req: TraceRequest = { id: s.id, arrivalMs: 0, promptLen: s.promptLen, outputLen: 64, sharedPrefixLen: s.sharedPrefixLen, kind: 'chat' };
      if (!alloc.admit(req, s.currentLen).ok) break;
      if (!alloc.ensureCapacity(req, s.currentLen)) {
        alloc.release(s.id);
        break;
      }
      fitted.push({ id: s.id, tokens: s.currentLen, sharedPrefix: s.sharedPrefixLen });
    }
  }

  const snap = alloc.stats(true);
  const totalWasteTokens = snap.internalWasteTokens + snap.reservedUnusedTokens + snap.externalFragTokens;
  const denom = Math.max(1, snap.reservedTokens + snap.externalFragTokens);
  const wastePct = (totalWasteTokens / denom) * 100;

  // Roofline decode throughput for the sequences that actually fit!
  const decodeBatch: BatchWorkItem[] = fitted.map((s) => ({
    kind: 'decode',
    cachedTokens: Math.max(0, s.tokens - 1),
    newTokens: 1,
  }));
  const cost = stepTime(decodeBatch, gpu, model);
  const decodePerUserTokPerSec = cost.stepTimeMs > 0 ? 1000 / cost.stepTimeMs : 0;
  const decodeThroughputTokPerSec = decodePerUserTokPerSec * fitted.length;

  // Build 256-cell proportional visual map of physical KV VRAM
  const CELLS = 256;
  const cells: MemoryMapReport['cells'] = [];
  if (params.mode !== 'paged' && snap.slabs) {
    const tokensPerCell = capTokens / CELLS;
    const maxEnd = snap.slabs.reduce((m, s) => Math.max(m, s.start + s.size), 0);
    for (let c = 0; c < CELLS; c++) {
      const mid = (c + 0.5) * tokensPerCell;
      const slab = snap.slabs.find((s) => mid >= s.start && mid < s.start + s.size);
      if (slab) {
        const offset = mid - slab.start;
        cells.push({ owner: slab.reqId, kind: offset < slab.used ? 'used' : 'reserved-waste' });
      } else if (mid < maxEnd) {
        cells.push({ owner: -1, kind: 'frag' });
      } else {
        cells.push({ owner: -1, kind: 'free' });
      }
    }
  } else if (snap.blockOwners) {
    const blocksPerCell = snap.totalBlocks / CELLS;
    for (let c = 0; c < CELLS; c++) {
      const b = Math.min(snap.totalBlocks - 1, Math.floor((c + 0.5) * blocksPerCell));
      const owner = snap.blockOwners[b];
      if (owner === -2) cells.push({ owner: -2, kind: 'shared' });
      else if (owner >= 0) cells.push({ owner, kind: 'used' });
      else cells.push({ owner: -1, kind: 'free' });
    }
  }

  const slabById = new Map((snap.slabs ?? []).map((s) => [s.reqId, { start: s.start, size: s.size }]));
  const fittedSeqs = fitted.map((s) => ({
    ...s,
    blocks: snap.blockTables?.[s.id],
    slab: slabById.get(s.id),
  }));

  return {
    mode: params.mode,
    prefixSharing: params.mode === 'paged' && params.prefixSharing,
    blockSize: params.blockSize,
    maxModelLen: params.maxModelLen,
    kvCapTokens: capTokens,
    kvBudgetGiB: budgetGiB,
    sequencesFit: fitted.length,
    totalRequested: params.sequences.length,
    usedTokens: snap.usedTokens,
    reservedTokens: snap.reservedTokens,
    internalWasteTokens: snap.internalWasteTokens,
    reservedUnusedTokens: snap.reservedUnusedTokens,
    externalFragTokens: snap.externalFragTokens,
    totalWasteTokens,
    wastePct,
    sharedBlocks: snap.sharedBlocks,
    usedBlocks: snap.usedBlocks,
    totalBlocks: snap.totalBlocks,
    prefixRefCount: snap.prefixRefCount,
    decodeThroughputTokPerSec,
    decodePerUserTokPerSec,
    decodeComputeUtilPct: cost.computeUtilPct,
    decodeBottleneck: cost.bottleneck,
    fittedSeqs,
    cells,
  };
}
