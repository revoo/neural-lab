// authored-by: Gemini Antigravity
import { describe, it, expect } from 'vitest';
import {
  computeArchMetrics,
  computeCrossAttention,
  evalSeqBrief1,
  evalSeqBrief2,
  evalSeqBrief3,
  evaluateBottleneck,
  GOLD_ALIGNMENT,
  seqLabExhaustive,
} from './seq2seq';

describe('Module 11 Lesson 3: Encoder–Decoder and Early Attention', () => {
  it('pins the fixed-vector information bottleneck collapsing with sequence length vs cross-attention holding 100%', () => {
    const rows = evaluateBottleneck(16);
    const byL = Object.fromEntries(rows.map((r) => [r.length, r]));

    expect(byL[2].fixedAcc).toBeCloseTo(1.0, 3);
    expect(byL[4].fixedAcc).toBeCloseTo(0.788, 2);
    expect(byL[8].fixedAcc).toBeCloseTo(0.547, 2);
    expect(byL[16].fixedAcc).toBeCloseTo(0.384, 2);
    expect(byL[20].fixedAcc).toBeCloseTo(0.338, 2);
    expect(byL[24].fixedAcc).toBeCloseTo(0.313, 2);

    for (const r of rows) {
      expect(r.attnAcc).toBeCloseTo(1.0, 6);
    }

    const d32 = evaluateBottleneck(32, [20])[0];
    expect(d32.fixedAcc).toBeCloseTo(0.514, 2);
    expect(d32.attnAcc).toBeCloseTo(1.0, 6);
  });

  it('pins Bahdanau and Luong cross-attention alignments including French adjective-noun swaps', () => {
    const luong = computeCrossAttention('luong', 0.15);
    const bahdanau = computeCrossAttention('bahdanau', 0.15);

    expect(luong.argmaxAcc).toBe(1.0);
    expect(bahdanau.argmaxAcc).toBe(1.0);
    expect(luong.meanGoldWeight).toBeCloseTo(0.960, 3);
    expect(luong.maxAbsLogit).toBeCloseTo(6.67, 2);
    expect(bahdanau.meanGoldWeight).toBeCloseTo(0.415, 3);

    // Verify every softmax row sums to 1 and "noir" (target index 2) aligns to "black" (source index 1)
    for (const row of luong.weights) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 9);
    }
    expect(GOLD_ALIGNMENT[1]).toBe(2); // chat -> cat
    expect(GOLD_ALIGNMENT[2]).toBe(1); // noir -> black
    expect(luong.weights[2][1]).toBeCloseTo(0.988, 3);
  });

  it('pins sequential depth, max path length, and GPU training latency across architectures', () => {
    const fixed = computeArchMetrics('fixed-lstm', 512);
    const lstmAttn = computeArchMetrics('lstm-attn', 512);
    const trans = computeArchMetrics('transformer', 512);

    expect(fixed.seqDepth).toBe(1024);
    expect(fixed.maxPathLen).toBe(512);
    expect(fixed.trainLatencyMs).toBeCloseTo(81.92, 2);

    expect(lstmAttn.seqDepth).toBe(1024);
    expect(lstmAttn.maxPathLen).toBe(1);
    expect(lstmAttn.trainLatencyMs).toBeCloseTo(85.07, 2);

    expect(trans.seqDepth).toBe(1);
    expect(trans.maxPathLen).toBe(1);
    expect(trans.trainLatencyMs).toBeCloseTo(7.75, 2);
    expect(lstmAttn.trainLatencyMs / trans.trainLatencyMs).toBeGreaterThan(10);
  });

  it('verifies exhaustive Lab: Break the Bottleneck counts and unique winners', () => {
    const b1 = seqLabExhaustive('bottleneck');
    expect(b1.total).toBe(6);
    expect(b1.passingCount).toBe(3);
    expect(evalSeqBrief1('fixed', 32).pass).toBe(false);
    expect(evalSeqBrief1('cross-attn', 8).pass).toBe(true);

    const b2 = seqLabExhaustive('align');
    expect(b2.total).toBe(8);
    expect(b2.passingCount).toBe(1);
    expect(evalSeqBrief2('luong', 0.15).pass).toBe(true);
    expect(evalSeqBrief2('luong', 0.05).pass).toBe(false);

    const b3 = seqLabExhaustive('latency');
    expect(b3.total).toBe(3);
    expect(b3.passingCount).toBe(1);
    expect(evalSeqBrief3('fixed-lstm').pass).toBe(false);
    expect(evalSeqBrief3('lstm-attn').pass).toBe(false);
    expect(evalSeqBrief3('transformer').pass).toBe(true);
  });
});
