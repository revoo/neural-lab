// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  GPU_BY_ID,
  MODEL_BY_ID,
  TETRIS_BRIEFS,
  evaluateRoofline,
  evaluateTetris,
  generationWorkTotals,
  kvBytesPerToken,
  kvCacheGiB,
  ridgePoint,
  runKvEquivalenceDemo,
  solveTetrisBrief,
  weightsGiB,
} from './kv';

describe('KV caching & memory-bound decoding — numbers quoted in prose', () => {
  it('cached and uncached causal self-attention match to machine precision (< 1e-14)', () => {
    const { steps, maxDiffOverall } = runKvEquivalenceDemo();
    expect(steps).toHaveLength(6);
    expect(maxDiffOverall).toBeLessThan(1e-14);
    expect(steps[5].cumUncachedProjections).toBe(21);
    expect(steps[5].cumCachedProjections).toBe(6);
    expect(steps[5].cumUncachedDots).toBe(56);
    expect(steps[5].cumCachedDots).toBe(21);

    // 512-token prompt + 256 generated tokens
    const w512 = generationWorkTotals(512, 256);
    expect(w512.uncachedTokens).toBe(163712);
    expect(w512.cachedTokens).toBe(767);
    expect(w512.tokenSpeedup).toBeCloseTo(213.4, 1);

    // 2048-token prompt + 512 generated tokens
    const w2048 = generationWorkTotals(2048, 512);
    expect(w2048.uncachedTokens).toBe(1179392);
    expect(w2048.cachedTokens).toBe(2559);
    expect(w2048.tokenSpeedup).toBeCloseTo(460.9, 1);
  });

  it('exact KV cache footprints for Llama-3 8B and 70B (GQA vs MHA)', () => {
    const m8 = MODEL_BY_ID['llama3-8b'];
    const m8mha = MODEL_BY_ID['llama3-8b-mha'];
    const m70 = MODEL_BY_ID['llama3-70b'];
    const m70mha = MODEL_BY_ID['llama3-70b-mha'];

    expect(kvBytesPerToken(m8, 2)).toBe(128 * 1024);
    expect(kvCacheGiB(m8, 32768, 1, 2)).toBeCloseTo(4.0, 5);
    expect(kvCacheGiB(m8, 131072, 1, 2)).toBeCloseTo(16.0, 5);
    expect(weightsGiB(m8, 2)).toBeCloseTo(14.96, 2);

    // MHA on 8B is exactly 4x larger
    expect(kvBytesPerToken(m8mha, 2)).toBe(512 * 1024);
    expect(kvCacheGiB(m8mha, 131072, 1, 2)).toBeCloseTo(64.0, 5);

    // Llama-3 70B GQA (8:1) vs MHA
    expect(kvBytesPerToken(m70, 2)).toBe(320 * 1024);
    expect(kvCacheGiB(m70, 131072, 1, 2)).toBeCloseTo(40.0, 5);
    expect(kvCacheGiB(m70mha, 131072, 1, 2)).toBeCloseTo(320.0, 5);
  });

  it('H100 roofline: B=1 decode is 99.6% memory-bound while 2k prefill is 100% compute-bound', () => {
    const h100 = GPU_BY_ID['h100-80'];
    const m8 = MODEL_BY_ID['llama3-8b'];
    expect(ridgePoint(h100)).toBeCloseTo(295.2, 1);

    const decFp16 = evaluateRoofline({
      gpu: h100, model: m8, phase: 'decode', batchSize: 1, ctxLen: 2048, weightBytesPerElem: 2, kvBytesPerElem: 2,
    });
    expect(decFp16.bottleneck).toBe('memory');
    expect(decFp16.arithmeticIntensity).toBeCloseTo(1.05, 2);
    expect(decFp16.computeUtilPct).toBeCloseTo(0.36, 2);
    expect(decFp16.tokensPerSecPerUser).toBeCloseTo(205.2, 1);

    const decInt4 = evaluateRoofline({
      gpu: h100, model: m8, phase: 'decode', batchSize: 1, ctxLen: 2048, weightBytesPerElem: 0.5625, kvBytesPerElem: 2,
    });
    expect(decInt4.bottleneck).toBe('memory');
    expect(decInt4.tokensPerSecPerUser).toBeCloseTo(700.1, 1);

    // At 128k context, FP16 KV reads drag INT4 weights down to 154.4 tok/s; FP8 KV lifts it to 255.6
    const decLongFp16Kv = evaluateRoofline({
      gpu: h100, model: m8, phase: 'decode', batchSize: 1, ctxLen: 131072, weightBytesPerElem: 0.5625, kvBytesPerElem: 2,
    });
    const decLongFp8Kv = evaluateRoofline({
      gpu: h100, model: m8, phase: 'decode', batchSize: 1, ctxLen: 131072, weightBytesPerElem: 0.5625, kvBytesPerElem: 1,
    });
    expect(decLongFp16Kv.tokensPerSecPerUser).toBeCloseTo(154.4, 1);
    expect(decLongFp8Kv.tokensPerSecPerUser).toBeCloseTo(255.6, 1);

    // Batching B=64 multiplies total throughput 31x while per-user speed stays 100.8 tok/s
    const decB64 = evaluateRoofline({
      gpu: h100, model: m8, phase: 'decode', batchSize: 64, ctxLen: 2048, weightBytesPerElem: 2, kvBytesPerElem: 2,
    });
    expect(decB64.tokensPerSecPerUser).toBeCloseTo(100.8, 1);
    expect(decB64.totalTokensPerSec).toBeCloseTo(6450.1, 0);

    // Prefill of 2,048 tokens is 100% compute-bound and finishes in 34.4 ms
    const pref = evaluateRoofline({
      gpu: h100, model: m8, phase: 'prefill', batchSize: 1, ctxLen: 2048, weightBytesPerElem: 2, kvBytesPerElem: 2,
    });
    expect(pref.bottleneck).toBe('compute');
    expect(pref.computeUtilPct).toBeCloseTo(100.0, 2);
    expect(pref.stepTimeMs).toBeCloseTo(34.37, 1);
  });

  it('VRAM Tetris exhaustive solutions across all 441 configs per brief', () => {
    // Brief 1: Local coding assistant (RTX 4070 12G, Llama-3 8B)
    const s1 = solveTetrisBrief(TETRIS_BRIEFS[0]);
    expect(evaluateTetris(TETRIS_BRIEFS[0], TETRIS_BRIEFS[0].defaultConfig).passes).toBe(false);
    expect(s1.totalConfigs).toBe(441);
    expect(s1.passingCount).toBe(4);
    expect(s1.bestConfig).toEqual({ weightFmt: 'int4', kvFmt: 'fp8', batchSize: 1, ctxLen: 32768 });
    expect(s1.bestEval.roofline.tokensPerSecPerUser).toBeCloseTo(75.6, 1);
    expect(s1.bestEval.totalGiB).toBeCloseTo(7.21, 2);

    // Brief 2: Single-GPU 70B server (H100 80G, Llama-3 70B)
    const s2 = solveTetrisBrief(TETRIS_BRIEFS[1]);
    expect(evaluateTetris(TETRIS_BRIEFS[1], TETRIS_BRIEFS[1].defaultConfig).passes).toBe(false);
    expect(s2.passingCount).toBe(7);
    expect(s2.bestConfig).toEqual({ weightFmt: 'int4', kvFmt: 'int4', batchSize: 16, ctxLen: 16384 });
    expect(s2.bestEval.roofline.totalTokensPerSec).toBeCloseTo(839.2, 1);
    expect(s2.bestEval.totalGiB).toBeCloseTo(62.49, 2);

    // Brief 3: High-throughput 32k API (H100 80G, Llama-3 8B)
    const s3 = solveTetrisBrief(TETRIS_BRIEFS[2]);
    expect(evaluateTetris(TETRIS_BRIEFS[2], TETRIS_BRIEFS[2].defaultConfig).passes).toBe(false);
    expect(s3.passingCount).toBe(6);
    expect(s3.bestConfig).toEqual({ weightFmt: 'int8', kvFmt: 'int4', batchSize: 32, ctxLen: 32768 });
    expect(s3.bestEval.roofline.totalTokensPerSec).toBeCloseTo(2284.0, 0);
    expect(s3.bestEval.totalGiB).toBeCloseTo(46.71, 2);
  });
});
