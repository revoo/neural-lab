// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  GIB,
  H100_80G,
  LLAMA3_8B,
  RTX_4090,
  evaluateKvMemoryMap,
  kvBudgetBytes,
  kvBytesPerToken,
  kvCapacityTokens,
  ridgePoint,
  simulate,
  singleRequestLatency,
  stepTime,
  weightsBytes,
  type ServingConfig,
} from './serve';
import {
  LAB_TRACE_CHAT,
  LAB_TRACE_CROWD,
  LAB_TRACE_LONGDOC,
  LAUNCH_BURST_TRACE,
  MEMORY_MAP_SEQUENCES,
  RACE_TRACE,
  TROLLEY_TRACE,
  summarizeTrace,
} from './trace';
import { runStaticRaceSearch } from './staticSearch.run';
import { STATIC_RACE_SEARCH } from './staticBest';

describe('llm-serving engine (S1): constants, roofline, conservation & fairness search', () => {
  it('reproduces kv-cache headline numbers exactly (Llama-3 8B FP16 on H100 & RTX 4090)', () => {
    expect(kvBytesPerToken(LLAMA3_8B, 2)).toBe(128 * 1024);
    expect(weightsBytes(LLAMA3_8B, 2) / GIB).toBeCloseTo(14.96, 2);
    expect(ridgePoint(H100_80G)).toBeCloseTo(295.2, 1);
    expect(ridgePoint(RTX_4090)).toBeCloseTo(163.9, 1);

    // H100 B = 1, 2k context decode roofline: 205.2 tok/s, 0.36% compute util, 1.05 FLOPs/byte
    const dec = stepTime([{ kind: 'decode', cachedTokens: 2047, newTokens: 1 }], H100_80G, LLAMA3_8B);
    expect(dec.bottleneck).toBe('memory');
    expect(1000 / dec.stepTimeMs).toBeCloseTo(205.2, 1);
    expect(dec.computeUtilPct).toBeCloseTo(0.36, 2);
    expect(dec.arithmeticIntensity).toBeCloseTo(1.05, 2);

    // H100 B = 1, 2k prefill: 34.4 ms, 100% compute-bound
    const pref = stepTime([{ kind: 'prefill', cachedTokens: 0, newTokens: 2048 }], H100_80G, LLAMA3_8B);
    expect(pref.bottleneck).toBe('compute');
    expect(pref.computeUtilPct).toBeCloseTo(100.0, 2);
    expect(pref.stepTimeMs).toBeCloseTo(34.37, 1);

    // RTX 4090 KV budget: 24 - 14.96 - 1.5 = 7.54 GiB -> 61,791 FP16 KV tokens
    expect(kvBudgetBytes(RTX_4090, LLAMA3_8B) / GIB).toBeCloseTo(7.54, 2);
    expect(kvCapacityTokens(RTX_4090, LLAMA3_8B)).toBe(61791);

    // Unified causal formula: 1-token prefill chunk at cachedTokens = c - 1 equals decode at ctx = c
    const asChunk = stepTime([{ kind: 'prefill', cachedTokens: 2047, newTokens: 1 }], H100_80G, LLAMA3_8B);
    expect(asChunk.flops).toBe(dec.flops);
    expect(asChunk.bytes).toBe(dec.bytes);
    expect(asChunk.stepTimeMs).toBe(dec.stepTimeMs);
  });

  it('launch-day 200-user burst on H100 (B=1) and single-request latency anatomy', () => {
    const burst = simulate(LAUNCH_BURST_TRACE, {
      batching: 'continuous',
      maxRunning: 1,
      tokenBudget: null,
      allocator: 'paged',
      blockSize: 16,
      prefixSharing: false,
      admission: 'fcfs',
      gpu: H100_80G,
      model: LLAMA3_8B,
    });
    const last = burst.requests[burst.requests.length - 1];
    expect(burst.requests).toHaveLength(200);
    expect(last.ttftMs / 1000).toBeCloseTo(169.5, 1);
    expect(burst.metrics.ttftP50Ms / 1000).toBeCloseTo(84.4, 1);
    expect(burst.metrics.totalTimeMs / 1000).toBeCloseTo(171.3, 1);

    // Widget A single-request anatomy identity: totalLatency = TTFT + (outputLen - 1) * meanTpot
    const a = singleRequestLatency({ promptLen: 512, replyLen: 128, usersAhead: 10, gpu: H100_80G, model: LLAMA3_8B });
    expect(a.ttftMs + 127 * a.meanTpotMs).toBeCloseTo(a.totalLatencyMs, 6);
    expect(a.queueShareOfTtft).toBeGreaterThan(0.95);
    expect(a.meanDecodeUtilPct).toBeLessThan(0.5);
  });

  it('conservation: every request finishes, Eq 1 holds, and paged blocks never leak', () => {
    const configs: Partial<ServingConfig>[] = [
      { batching: 'static', staticBatchSize: 8, staticMaxWaitMs: 100, maxRunning: 8, tokenBudget: null, allocator: 'reserve-max' },
      { batching: 'static', staticBatchSize: 8, staticMaxWaitMs: 50, maxRunning: 8, tokenBudget: null, allocator: 'paged' },
      { batching: 'continuous', maxRunning: 16, tokenBudget: null, allocator: 'reserve-max' },
      { batching: 'continuous', maxRunning: 16, tokenBudget: 512, allocator: 'reserve-grow' },
      { batching: 'continuous', maxRunning: 16, tokenBudget: 256, allocator: 'paged', blockSize: 8, prefixSharing: false },
      { batching: 'continuous', maxRunning: 48, tokenBudget: 512, allocator: 'paged', blockSize: 16, prefixSharing: false },
      { batching: 'continuous', maxRunning: 48, tokenBudget: 512, allocator: 'paged', blockSize: 16, prefixSharing: true },
      { batching: 'continuous', maxRunning: 32, tokenBudget: 512, allocator: 'paged', blockSize: 32, prefixSharing: true, admission: 'sjf' },
    ];

    for (const trace of [RACE_TRACE, LAB_TRACE_CROWD]) {
      for (const partial of configs) {
        const cfg: ServingConfig = {
          batching: partial.batching ?? 'continuous',
          staticBatchSize: partial.staticBatchSize,
          staticMaxWaitMs: partial.staticMaxWaitMs,
          maxRunning: partial.maxRunning ?? 16,
          tokenBudget: partial.tokenBudget ?? null,
          allocator: partial.allocator ?? 'paged',
          blockSize: partial.blockSize ?? 16,
          prefixSharing: partial.prefixSharing ?? false,
          admission: partial.admission ?? 'fcfs',
          gpu: RTX_4090,
          model: LLAMA3_8B,
        };
        const res = simulate(trace, cfg);
        expect(res.requests).toHaveLength(trace.requests.length);
        expect(res.metrics.totalOutputTokens).toBe(res.metrics.expectedOutputTokens);
        expect(res.metrics.blocksAllocated).toBe(res.metrics.blocksFreed);
        for (const r of res.requests) {
          const reconstructed = r.ttftMs + Math.max(0, r.outputLen - 1) * r.meanTpotMs;
          expect(reconstructed).toBeCloseTo(r.totalLatencyMs, 5);
        }
      }
    }
  });

  it('pins committed trace summary statistics so traces cannot silently drift', () => {
    expect(summarizeTrace(LAUNCH_BURST_TRACE)).toMatchObject({ count: 200, docs: 0, totalOutputTokens: 35548 });
    expect(summarizeTrace(RACE_TRACE)).toMatchObject({ count: 24, docs: 0, totalOutputTokens: 2101 });
    expect(summarizeTrace(TROLLEY_TRACE)).toMatchObject({ count: 11, docs: 1, maxPrompt: 16384, totalOutputTokens: 836 });

    const chatSum = summarizeTrace(LAB_TRACE_CHAT);
    const docSum = summarizeTrace(LAB_TRACE_LONGDOC);
    const crowdSum = summarizeTrace(LAB_TRACE_CROWD);
    expect(chatSum.count).toBe(40);
    expect(chatSum.docs).toBe(0);
    expect(docSum.count).toBe(36);
    expect(docSum.docs).toBeGreaterThanOrEqual(2);
    expect(crowdSum.count).toBe(64);
    expect(crowdSum.sharedPrefixLen).toBe(1024);

    // Every request must fit within its trace's maxModelLen
    for (const tr of [LAUNCH_BURST_TRACE, RACE_TRACE, TROLLEY_TRACE, LAB_TRACE_CHAT, LAB_TRACE_LONGDOC, LAB_TRACE_CROWD]) {
      for (const req of tr.requests) {
        expect(req.promptLen + req.outputLen).toBeLessThanOrEqual(tr.maxModelLen);
      }
    }
  });

  it('re-verifies the static-batching fairness search (Widget B)', () => {
    const fresh = runStaticRaceSearch();
    expect(fresh).toEqual(STATIC_RACE_SEARCH);
    expect(fresh.gridSize).toBe(36);
    expect(fresh.bestByThroughput.batchSize).toBe(16);
    expect(fresh.bestByThroughput.maxWaitMs).toBe(400);
    expect(fresh.bestByThroughput.throughputTokPerSec).toBeCloseTo(200.6, 1);
    expect(fresh.bestByThroughput.ttftP95Ms).toBeCloseTo(3969.2, 0);
    expect(fresh.bestByThroughput.idleSlotShare).toBeCloseTo(0.647, 2);

    expect(fresh.continuousAtSameSlots.throughputTokPerSec).toBeCloseTo(303.3, 1);
    expect(fresh.continuousAtSameSlots.ttftP95Ms).toBeCloseTo(52.0, 0);
    expect(fresh.continuousAtSameSlots.idleSlotShare).toBe(0);
  });

  it('pins Widget C chunked-prefill roofline knee and Widget D KV memory map numbers', () => {
    // Widget C: 16k doc unchunked vs 64 (below ridge) vs 256 (sweet spot) vs 512
    const runC = (budget: number | null) =>
      simulate(TROLLEY_TRACE, {
        batching: 'continuous',
        maxRunning: 16,
        tokenBudget: budget,
        allocator: 'paged',
        blockSize: 16,
        prefixSharing: false,
        admission: 'fcfs',
        gpu: RTX_4090,
        model: LLAMA3_8B,
      });

    const cOff = runC(null);
    const c64 = runC(64);
    const c128 = runC(128);
    const c256 = runC(256);
    const c512 = runC(512);

    expect(cOff.metrics.worstTpotMs).toBeCloseTo(2019.4, 0);
    expect(cOff.requests.find((r) => r.kind === 'doc')!.ttftMs).toBeCloseTo(2035, 0);

    // Below ridge (64): memory-bound, doc TTFT more than doubles!
    expect(c64.metrics.worstTpotMs).toBeCloseTo(18.2, 1);
    expect(c64.requests.find((r) => r.kind === 'doc')!.ttftMs).toBeCloseTo(4552, 0);

    // Right near ridge (128) and above ridge (256, 512)
    expect(c128.metrics.worstTpotMs).toBeCloseTo(19.0, 1);
    expect(c128.requests.find((r) => r.kind === 'doc')!.ttftMs).toBeCloseTo(2277, 0);
    expect(c256.metrics.worstTpotMs).toBeCloseTo(37.9, 1);
    expect(c256.requests.find((r) => r.kind === 'doc')!.ttftMs).toBeCloseTo(2072, 0);
    expect(c512.metrics.worstTpotMs).toBeCloseTo(75.4, 1);

    // Widget D: RTX 4090, maxModelLen = 8,192
    const dMax = evaluateKvMemoryMap({ sequences: MEMORY_MAP_SEQUENCES, mode: 'reserve-max', blockSize: 16, prefixSharing: false, maxModelLen: 8192 });
    const dGrow = evaluateKvMemoryMap({ sequences: MEMORY_MAP_SEQUENCES, mode: 'reserve-grow', blockSize: 16, prefixSharing: false, maxModelLen: 8192 });
    const dPaged16 = evaluateKvMemoryMap({ sequences: MEMORY_MAP_SEQUENCES, mode: 'paged', blockSize: 16, prefixSharing: false, maxModelLen: 8192 });
    const dPrefix = evaluateKvMemoryMap({ sequences: MEMORY_MAP_SEQUENCES, mode: 'paged', blockSize: 16, prefixSharing: true, maxModelLen: 8192 });

    expect(dMax.sequencesFit).toBe(7);
    expect(dMax.wastePct).toBeCloseTo(82.1, 1);
    expect(dMax.decodeThroughputTokPerSec).toBeCloseTo(405, 0);

    expect(dGrow.sequencesFit).toBe(27);
    expect(dGrow.externalFragTokens).toBeGreaterThan(0);

    expect(dPaged16.sequencesFit).toBe(42);
    expect(dPaged16.wastePct).toBeCloseTo(0.5, 1);
    expect(dPaged16.decodeThroughputTokPerSec).toBeCloseTo(1756, 0);

    expect(dPrefix.sequencesFit).toBe(64);
    expect(dPrefix.sharedBlocks).toBe(64); // 1024 / 16
    expect(dPrefix.prefixRefCount).toBe(64);
    expect(dPrefix.decodeThroughputTokPerSec).toBeCloseTo(2267, 0);
  });
});
